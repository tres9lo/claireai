-- 0001_initial.sql — first schema for the ClaireAI assistant.
--
-- Week 1 scope: tasks, task_log (append-only audit), summaries, notes.
-- Status values mirror the task state machine in claireai.storage.tasks.

CREATE TABLE tasks (
    id           TEXT PRIMARY KEY,
    created_at   TEXT NOT NULL,              -- UTC ISO-8601
    updated_at   TEXT NOT NULL,              -- UTC ISO-8601
    finished_at  TEXT,
    device       TEXT NOT NULL,
    actor        TEXT NOT NULL,
    kind         TEXT NOT NULL,
    request      TEXT NOT NULL,              -- sanitized user request
    status       TEXT NOT NULL CHECK (status IN (
                     'queued', 'planning', 'awaiting_approval', 'running',
                     'succeeded', 'failed', 'cancelled', 'interrupted'
                 )),
    current_step INTEGER,
    error        TEXT
);

CREATE INDEX tasks_created_at_idx ON tasks (created_at);

CREATE TABLE task_log (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at   TEXT NOT NULL,              -- UTC ISO-8601
    device       TEXT NOT NULL,
    task_id      TEXT REFERENCES tasks (id),
    actor        TEXT NOT NULL,
    tool         TEXT,
    action       TEXT NOT NULL,
    target       TEXT,
    status       TEXT NOT NULL CHECK (status IN ('done', 'failed', 'skipped', 'pending')),
    approval_ref TEXT,
    detail       TEXT                        -- sanitized JSON, truncated
);

CREATE INDEX task_log_task_id_idx ON task_log (task_id);
CREATE INDEX task_log_created_at_idx ON task_log (created_at);

-- Append-only at the storage layer: the application never exposes an edit or
-- delete path for audit rows, and the triggers keep accidental SQL from
-- rewriting history.
CREATE TRIGGER task_log_no_update
BEFORE UPDATE ON task_log
BEGIN
    SELECT RAISE(ABORT, 'task_log is append-only');
END;

CREATE TRIGGER task_log_no_delete
BEFORE DELETE ON task_log
BEGIN
    SELECT RAISE(ABORT, 'task_log is append-only');
END;

CREATE TABLE summaries (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    summary_date TEXT NOT NULL UNIQUE,       -- YYYY-MM-DD, local day
    created_at   TEXT NOT NULL,
    content_md   TEXT NOT NULL
);

CREATE TABLE notes (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    title      TEXT,
    body       TEXT NOT NULL
);
