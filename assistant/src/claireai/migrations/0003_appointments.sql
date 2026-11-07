-- The owner's own calendar. A day is a local calendar date, because "what is on
-- tomorrow" means tomorrow where the owner is, not a UTC instant.
CREATE TABLE appointments (
    id         TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    day        TEXT NOT NULL,              -- YYYY-MM-DD, local
    starts_at  TEXT NOT NULL,              -- HH:MM, local
    ends_at    TEXT,                       -- HH:MM, local, optional
    title      TEXT NOT NULL,
    location   TEXT
);

CREATE INDEX appointments_day_idx ON appointments (day, starts_at);
