-- Local memory: notes the owner asks ClaireAI to keep and retrieved excerpts.
-- The embedding is a packed little-endian float32 vector. Text and vectors stay
-- in this SQLite file; nothing is uploaded.
CREATE TABLE memory_entries (
    id         TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,              -- UTC ISO-8601
    kind       TEXT NOT NULL CHECK (kind IN ('note', 'turn')),
    source_id  TEXT,                       -- task id for a turn, optional for a note
    text       TEXT NOT NULL,
    embedding  BLOB NOT NULL,
    dimensions INTEGER NOT NULL CHECK (dimensions > 0),
    model      TEXT NOT NULL               -- embedding model or hash scheme
);

CREATE UNIQUE INDEX memory_entries_source_idx
    ON memory_entries (kind, source_id)
    WHERE source_id IS NOT NULL;
CREATE INDEX memory_entries_created_idx ON memory_entries (created_at);
