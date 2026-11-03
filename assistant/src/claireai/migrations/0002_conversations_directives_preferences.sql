-- Locally retained conversation history and owner-managed customization.
CREATE TABLE conversations (
    id         TEXT PRIMARY KEY,
    title      TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE INDEX conversations_updated_at_idx ON conversations (updated_at);

CREATE TABLE conversation_messages (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id TEXT NOT NULL REFERENCES conversations (id) ON DELETE CASCADE,
    task_id         TEXT REFERENCES tasks (id) ON DELETE SET NULL,
    sequence        INTEGER NOT NULL,
    created_at      TEXT NOT NULL,
    role            TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
    content         TEXT NOT NULL,
    UNIQUE (conversation_id, sequence)
);

CREATE INDEX conversation_messages_history_idx
    ON conversation_messages (conversation_id, sequence);

CREATE TABLE user_directives (
    id         TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    title      TEXT NOT NULL,
    content    TEXT NOT NULL,
    enabled    INTEGER NOT NULL DEFAULT 1 CHECK (enabled IN (0, 1)),
    position   INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE user_preferences (
    key        TEXT PRIMARY KEY,
    value_json TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
