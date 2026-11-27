"""Signed single-use approvals for runs that have no terminal.

A write tool in an unattended run (cron, a script, a desktop launch with no TTY)
still needs a human decision behind it. The owner issues one token for one exact
action; the unattended process may consume it once.

This is deliberately *not* an "always approve" broker. A token names the tool,
every argument and (optionally) the target, it expires, and it is moved out of
``pending/`` before the action runs, so replaying it finds nothing to consume.
The signing key lives beside the database, readable only by its owner, and is
never printed.
"""

from __future__ import annotations

import hmac
import json
import os
import secrets as _secrets
import shlex
import time
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, Mapping

from claireai.approval import ApprovalDecision, ApprovalRequest

TOKEN_VERSION = 1
DEFAULT_TTL_SECONDS = 600
MAX_TTL_SECONDS = 24 * 3600
KEY_NAME = "approval.key"
PENDING = Path("approvals") / "pending"
CONSUMED = Path("approvals") / "consumed"


class ApprovalFileError(RuntimeError):
    """The token, the key or the request cannot be handled safely."""


def canonical_payload(value: Any) -> bytes:
    """Stable bytes for signing: same action always hashes the same way."""
    try:
        text = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    except (TypeError, ValueError) as exc:
        raise ApprovalFileError("approval arguments must be JSON-serialisable") from exc
    return text.encode("utf-8")


def _key(data_dir: Path, *, create: bool) -> bytes:
    path = Path(data_dir) / KEY_NAME
    if path.is_file():
        key = path.read_bytes()
        if len(key) < 32:
            raise ApprovalFileError(f"{path.name} is too short to sign approvals")
        return key
    if not create:
        raise ApprovalFileError(
            "no approval key exists yet; issue a token from a terminal first"
        )
    Path(data_dir).mkdir(parents=True, exist_ok=True)
    key = _secrets.token_bytes(32)
    try:
        # O_EXCL + 0600: never adopt a file someone else created, never leave the
        # key readable by another user even for an instant.
        handle = os.fdopen(
            os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "wb"
        )
    except FileExistsError:
        return _key(data_dir, create=False)
    with handle:
        handle.write(key)
    return key


def token_hint(request: ApprovalRequest) -> str:
    """The exact command that would authorise ``request``.

    Included in every refusal so the owner can copy it instead of guessing which
    arguments the model produced.
    """
    parts = ["claireai", "--preauthorize", request.tool]
    for key, value in sorted(dict(request.arguments).items()):
        parts += ["--arg", f"{key}={value}"]
    return " ".join(shlex.quote(part) for part in parts)


def issue(
    request: ApprovalRequest,
    *,
    data_dir: str | Path,
    ttl_seconds: int = DEFAULT_TTL_SECONDS,
) -> Path:
    """Sign one token for ``request`` and return where it was written."""
    if type(ttl_seconds) is not int or not 1 <= ttl_seconds <= MAX_TTL_SECONDS:
        raise ValueError(f"ttl_seconds must be an integer from 1 to {MAX_TTL_SECONDS}")
    if not request.tool:
        raise ApprovalFileError("an approval token needs a tool name")
    directory = Path(data_dir)
    key = _key(directory, create=True)
    nonce = _secrets.token_hex(8)
    now = int(time.time())
    payload: dict[str, Any] = {
        "v": TOKEN_VERSION,
        "nonce": nonce,
        "tool": request.tool,
        "arguments": dict(request.arguments),
        "target": request.target,
        "issued_at": now,
        "expires_at": now + ttl_seconds,
    }
    mac = hmac.new(key, canonical_payload(payload), sha256).hexdigest()
    pending = directory / PENDING
    pending.mkdir(parents=True, exist_ok=True)
    path = pending / f"{nonce}.json"
    handle = os.fdopen(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "w")
    with handle:
        json.dump({"payload": payload, "mac": mac}, handle, sort_keys=True, indent=2)
    return path


def _load_pending(path: Path, key: bytes) -> dict[str, Any] | None:
    """Return a verified, unexpired token payload, or None for a bad file."""
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(raw, dict):
        return None
    payload, mac = raw.get("payload"), raw.get("mac")
    if not isinstance(payload, dict) or not isinstance(mac, str):
        return None
    if payload.get("v") != TOKEN_VERSION:
        return None
    try:
        expected = hmac.new(key, canonical_payload(payload), sha256).hexdigest()
    except ApprovalFileError:
        return None
    if not hmac.compare_digest(mac, expected):
        return None
    expires_at = payload.get("expires_at")
    if not isinstance(expires_at, (int, float)) or expires_at < time.time():
        return None
    return payload


def pending_approvals(data_dir: str | Path) -> list[dict[str, Any]]:
    """Metadata for currently valid signed tokens, without approving anything."""
    directory = Path(data_dir)
    pending = directory / PENDING
    if not pending.is_dir():
        return []
    try:
        key = _key(directory, create=False)
    except ApprovalFileError:
        return []
    items: list[dict[str, Any]] = []
    for path in sorted(pending.glob("*.json")):
        payload = _load_pending(path, key)
        if payload is None:
            continue
        items.append(
            {
                "ref": str(payload.get("nonce") or path.stem),
                "tool": payload.get("tool"),
                "arguments": payload.get("arguments") or {},
                "target": payload.get("target"),
                "issued_at": payload.get("issued_at"),
                "expires_at": payload.get("expires_at"),
                "hint": token_hint(
                    ApprovalRequest(
                        ref=str(payload.get("nonce") or path.stem),
                        task_id="preauthorize",
                        tool=str(payload.get("tool") or ""),
                        risk="needs_approval",
                        arguments=payload.get("arguments") or {},
                        target=payload.get("target"),
                    )
                ),
            }
        )
    return items


def revoke_approval(data_dir: str | Path, ref: str) -> bool:
    """Remove one pending token. This is a deny, never an approval."""
    directory = Path(data_dir)
    pending = directory / PENDING
    if not pending.is_dir():
        return False
    try:
        key = _key(directory, create=False)
    except ApprovalFileError:
        return False
    for path in pending.glob("*.json"):
        payload = _load_pending(path, key)
        if payload is None:
            continue
        nonce = str(payload.get("nonce") or path.stem)
        if nonce != str(ref):
            continue
        revoked = directory / "approvals" / "revoked"
        revoked.mkdir(parents=True, exist_ok=True)
        try:
            os.replace(path, revoked / path.name)
        except FileNotFoundError:
            return False
        return True
    return False


def _matches(payload: Mapping[str, Any], request: ApprovalRequest) -> bool:
    if payload.get("tool") != request.tool:
        return False
    if payload.get("arguments") != dict(request.arguments):
        return False
    # A token without a target means "the arguments are the whole action"; a
    # token that names one must be used for that target only.
    target = payload.get("target")
    return target is None or target == request.target


def consume(request: ApprovalRequest, *, data_dir: str | Path) -> ApprovalDecision:
    """Approve ``request`` only if a valid, unexpired token matches it exactly."""
    directory = Path(data_dir)
    pending = directory / PENDING
    if not pending.is_dir():
        return ApprovalDecision(
            approved=False,
            reason=(
                "no signed approval token is waiting for this action; "
                f"issue one from a terminal with: {token_hint(request)}"
            ),
            decided_by="policy",
        )
    try:
        key = _key(directory, create=False)
    except ApprovalFileError as exc:
        return ApprovalDecision(approved=False, reason=str(exc), decided_by="policy")

    now = time.time()
    for path in sorted(pending.glob("*.json")):
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if not isinstance(raw, dict):
            continue
        payload, mac = raw.get("payload"), raw.get("mac")
        if not isinstance(payload, dict) or not isinstance(mac, str):
            continue
        if payload.get("v") != TOKEN_VERSION:
            continue
        try:
            expected = hmac.new(key, canonical_payload(payload), sha256).hexdigest()
        except ApprovalFileError:
            continue
        if not hmac.compare_digest(mac, expected):
            continue
        expires_at = payload.get("expires_at")
        if not isinstance(expires_at, (int, float)) or expires_at < now:
            continue
        if not _matches(payload, request):
            continue
        consumed = directory / CONSUMED
        consumed.mkdir(parents=True, exist_ok=True)
        try:
            # Take it before the action runs: a replay must find nothing, and a
            # crash mid-action cannot leave a reusable token behind.
            os.replace(path, consumed / path.name)
        except FileNotFoundError:
            continue  # another process consumed it first
        return ApprovalDecision(
            approved=True,
            reason=f"owner signed token {payload.get('nonce')}",
            decided_by="owner-file",
        )

    return ApprovalDecision(
        approved=False,
        reason=(
            "no signed approval token matches this action; "
            f"issue one from a terminal with: {token_hint(request)}"
        ),
        decided_by="policy",
    )


@dataclass
class FileApprovalBroker:
    """Approve writes only through one signed token per action."""

    data_dir: Path
    name: str = "approval-file"

    async def decide(self, request: ApprovalRequest) -> ApprovalDecision:
        return consume(request, data_dir=self.data_dir)
