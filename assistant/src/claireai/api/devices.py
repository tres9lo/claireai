"""Local API device pairing.

The owner-facing bearer token is the administrator credential. A phone or
another local client can be paired with a short-lived one-time code; the raw
device token is returned once and only its SHA-256 hash is stored. Revocation is
a timestamp, so existing rows remain auditable.
"""

from __future__ import annotations

import hashlib
import secrets
import sqlite3
import time
import uuid
from dataclasses import dataclass
from typing import Any

from claireai.storage.database import transaction
from claireai.storage.tasks import utc_now

PAIRING_TTL_SECONDS = 300


def _hash(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class Device:
    id: str
    name: str
    created_at: str
    last_seen_at: str | None
    revoked_at: str | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "created_at": self.created_at,
            "last_seen_at": self.last_seen_at,
            "revoked": self.revoked_at is not None,
            "revoked_at": self.revoked_at,
        }


def create_pairing(
    conn: sqlite3.Connection, *, name: str, ttl_seconds: int = PAIRING_TTL_SECONDS
) -> dict[str, Any]:
    clean = " ".join(str(name or "").split()).strip()
    if not clean or len(clean) > 80:
        raise ValueError("device name must contain 1-80 characters")
    if type(ttl_seconds) is not int or not 30 <= ttl_seconds <= 900:
        raise ValueError("pairing lifetime must be from 30 to 900 seconds")
    code = secrets.token_hex(4).upper()
    expires_at = int(time.time()) + ttl_seconds
    with transaction(conn):
        conn.execute(
            "INSERT INTO api_pairings (code_hash, name, created_at, expires_at)"
            " VALUES (?, ?, ?, ?)",
            (_hash(code), clean, utc_now(), expires_at),
        )
    return {
        "code": code,
        "name": clean,
        "expires_at": expires_at,
        "expires_in": ttl_seconds,
    }


def claim_pairing(
    conn: sqlite3.Connection, *, code: str, name: str | None = None
) -> tuple[dict[str, Any], str]:
    supplied = "".join(str(code or "").split()).upper()
    if len(supplied) < 4:
        raise ValueError("pairing code is invalid or expired")
    row = conn.execute(
        "SELECT code_hash, name, expires_at, used_at FROM api_pairings WHERE code_hash = ?",
        (_hash(supplied),),
    ).fetchone()
    if row is None or row["used_at"] is not None or int(row["expires_at"]) < int(time.time()):
        raise ValueError("pairing code is invalid or expired")
    device_name = " ".join(str(name or row["name"]).split()).strip()
    if not device_name or len(device_name) > 80:
        raise ValueError("device name must contain 1-80 characters")
    token = secrets.token_urlsafe(32)
    device_id = uuid.uuid4().hex
    now = utc_now()
    with transaction(conn):
        updated = conn.execute(
            "UPDATE api_pairings SET used_at = ? WHERE code_hash = ? AND used_at IS NULL",
            (now, row["code_hash"]),
        )
        if updated.rowcount != 1:
            raise ValueError("pairing code is invalid or expired")
        conn.execute(
            "INSERT INTO api_devices (id, name, token_hash, created_at)"
            " VALUES (?, ?, ?, ?)",
            (device_id, device_name, _hash(token), now),
        )
    return (
        {
            "id": device_id,
            "name": device_name,
            "created_at": now,
            "token": token,
        },
        token,
    )


def list_devices(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    rows = conn.execute(
        "SELECT id, name, created_at, last_seen_at, revoked_at"
        " FROM api_devices ORDER BY created_at DESC"
    ).fetchall()
    return [
        Device(
            id=row["id"],
            name=row["name"],
            created_at=row["created_at"],
            last_seen_at=row["last_seen_at"],
            revoked_at=row["revoked_at"],
        ).as_dict()
        for row in rows
    ]


def revoke_device(conn: sqlite3.Connection, device_id: str) -> bool:
    with transaction(conn):
        cursor = conn.execute(
            "UPDATE api_devices SET revoked_at = COALESCE(revoked_at, ?) WHERE id = ?",
            (utc_now(), device_id),
        )
    return cursor.rowcount == 1


def device_for_token(conn: sqlite3.Connection, token: str) -> dict[str, Any] | None:
    if not token:
        return None
    row = conn.execute(
        "SELECT id, name, created_at, last_seen_at, revoked_at"
        " FROM api_devices WHERE token_hash = ?",
        (_hash(token),),
    ).fetchone()
    if row is None or row["revoked_at"] is not None:
        return None
    now = utc_now()
    conn.execute(
        "UPDATE api_devices SET last_seen_at = ? WHERE id = ?", (now, row["id"])
    )
    return {
        "id": row["id"],
        "name": row["name"],
        "created_at": row["created_at"],
        "last_seen_at": now,
        "revoked": False,
    }


__all__ = [
    "PAIRING_TTL_SECONDS",
    "claim_pairing",
    "create_pairing",
    "device_for_token",
    "list_devices",
    "revoke_device",
]
