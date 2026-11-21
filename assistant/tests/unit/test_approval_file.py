"""Signed single-use approvals for runs that have no terminal."""

from __future__ import annotations

import hmac
import json
import os
import stat
import time
from hashlib import sha256

import pytest

from claireai.approval import ApprovalRequest, new_ref
from claireai.approval_file import (
    FileApprovalBroker,
    canonical_payload,
    consume,
    issue,
    token_hint,
)


def request(**overrides):
    fields = {
        "ref": new_ref(),
        "task_id": "task-1",
        "tool": "create_folder",
        "risk": "needs_approval",
        "arguments": {"path": "~/Downloads/reports"},
        "target": "~/Downloads/reports",
    }
    fields.update(overrides)
    return ApprovalRequest(**fields)


def resign(path, mutate):
    """Rewrite a token's payload with a valid signature, for expiry tests."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    mutate(raw["payload"])
    key = (path.parents[2] / "approval.key").read_bytes()
    raw["mac"] = hmac.new(key, canonical_payload(raw["payload"]), sha256).hexdigest()
    path.write_text(json.dumps(raw), encoding="utf-8")


def test_issue_then_consume_approves_exactly_once(tmp_path):
    path = issue(request(), data_dir=tmp_path)
    assert path.is_file()

    decision = consume(request(), data_dir=tmp_path)
    assert decision.approved is True
    assert decision.decided_by == "owner-file"
    assert not path.exists()

    replay = consume(request(), data_dir=tmp_path)
    assert replay.approved is False
    assert "--preauthorize" in replay.reason


def test_a_different_action_is_not_approved(tmp_path):
    issue(request(), data_dir=tmp_path)
    other = request(arguments={"path": "~/Documents/reports"}, target="~/Documents/reports")
    assert consume(other, data_dir=tmp_path).approved is False


def test_expired_token_is_refused(tmp_path):
    path = issue(request(), data_dir=tmp_path, ttl_seconds=60)
    resign(path, lambda payload: payload.__setitem__("expires_at", int(time.time()) - 5))
    assert consume(request(), data_dir=tmp_path).approved is False


def test_tampered_arguments_are_refused(tmp_path):
    path = issue(request(), data_dir=tmp_path)
    raw = json.loads(path.read_text(encoding="utf-8"))
    raw["payload"]["arguments"]["path"] = "~/Documents/elsewhere"
    path.write_text(json.dumps(raw), encoding="utf-8")

    forged = request(arguments={"path": "~/Documents/elsewhere"}, target=None)
    assert consume(forged, data_dir=tmp_path).approved is False


@pytest.mark.skipif(
    os.name == "nt", reason="Windows ACLs do not map to POSIX st_mode"
)
def test_key_is_readable_by_its_owner_only(tmp_path):
    issue(request(), data_dir=tmp_path)
    mode = stat.S_IMODE((tmp_path / "approval.key").stat().st_mode)
    assert mode == 0o600


def test_consuming_without_any_key_explains_the_way_forward(tmp_path):
    decision = consume(request(), data_dir=tmp_path)
    assert decision.approved is False
    assert "--preauthorize" in decision.reason


def test_issue_rejects_a_nonsense_lifetime(tmp_path):
    with pytest.raises(ValueError):
        issue(request(), data_dir=tmp_path, ttl_seconds=0)
    with pytest.raises(ValueError):
        issue(request(), data_dir=tmp_path, ttl_seconds=24 * 3600 + 1)


@pytest.mark.asyncio
async def test_broker_consumes_one_token_per_action(tmp_path):
    issue(request(), data_dir=tmp_path)
    broker = FileApprovalBroker(tmp_path)
    assert (await broker.decide(request())).approved is True
    assert (await broker.decide(request())).approved is False


def test_hint_names_the_exact_action():
    hint = token_hint(request())
    assert hint.startswith("claireai --preauthorize create_folder")
    assert "path=~/Downloads/reports" in hint
