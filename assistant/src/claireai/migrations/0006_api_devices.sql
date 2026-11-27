-- Local API devices. Tokens are stored only as SHA-256 hashes; the raw token is
-- returned once when a pairing code is claimed. Pairing codes are one-time and
-- short-lived. This is a local service, so there is still no remote TLS here.
CREATE TABLE api_devices (
    id           TEXT PRIMARY KEY,
    name         TEXT NOT NULL,
    token_hash   TEXT NOT NULL UNIQUE,
    created_at   TEXT NOT NULL,
    last_seen_at TEXT,
    revoked_at   TEXT
);

CREATE INDEX api_devices_active_idx
    ON api_devices (revoked_at, created_at);

CREATE TABLE api_pairings (
    code_hash  TEXT PRIMARY KEY,
    name       TEXT NOT NULL,
    created_at TEXT NOT NULL,
    expires_at INTEGER NOT NULL,
    used_at    TEXT
);
