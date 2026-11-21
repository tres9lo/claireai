-- Owner-assigned tasks for a local day. This is a schedule, not a grant of
-- autonomy: the runner still goes through the normal tool registry, risk policy,
-- approvals and audit log.
CREATE TABLE day_plan_items (
    id           TEXT PRIMARY KEY,
    created_at   TEXT NOT NULL,              -- UTC ISO-8601
    updated_at   TEXT NOT NULL,              -- UTC ISO-8601
    finished_at  TEXT,
    day          TEXT NOT NULL,              -- YYYY-MM-DD, local
    starts_at    TEXT NOT NULL,              -- HH:MM, local
    request      TEXT NOT NULL,              -- sanitized owner request
    status       TEXT NOT NULL CHECK (status IN (
                     'planned', 'running', 'awaiting_approval',
                     'succeeded', 'failed', 'cancelled', 'interrupted'
                 )),
    task_id      TEXT REFERENCES tasks (id),
    approval_ref TEXT,
    error        TEXT
);

CREATE INDEX day_plan_items_day_idx ON day_plan_items (day, starts_at);
CREATE INDEX day_plan_items_status_idx ON day_plan_items (status);
