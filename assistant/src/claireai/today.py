"""The "today" page: a read-only view of the day's log and summary.

This exists so the daily summary has somewhere to be read without a terminal,
and so the desktop shell has a page to wrap when the Tauri toolchain arrives. It
is deliberately the least capable surface in the project:

* **read-only** — the server opens SQLite with ``mode=ro``, and the page has no
  JavaScript, no forms and no buttons, so it cannot act, approve, or change state;
* **no free HTML** — every string taken from storage is escaped; the summary is
  shown inside ``<pre>`` rather than rendered as HTML;
* **derived, not stored** — the page is built from the same
  :mod:`claireai.summary` logic the CLI prints, so the two can never disagree.

Approve/deny buttons are explicitly *not* here yet: granting a capability over
loopback HTTP needs a per-run token, POST-only routing and a nonce per request,
or any local process could approve a file write. That is its own design task, not
a page feature.
"""

from __future__ import annotations

import html
import sqlite3
from datetime import date, datetime
from pathlib import Path

from claireai.summary import build, stored

MAX_REQUESTS_SHOWN = 25


def _esc(value: str) -> str:
    return html.escape(value, quote=True)


def _clip(text: str, limit: int = 90) -> str:
    single = " ".join(text.split())
    return single if len(single) <= limit else single[: limit - 1] + "…"


STYLE = """
:root { color-scheme: dark; }
* { box-sizing: border-box; }
body {
  margin: 0; padding: 2rem 1.25rem 4rem;
  background: #0b0f14; color: #d7e3ee;
  font: 15px/1.55 system-ui, -apple-system, "Segoe UI", Roboto, sans-serif;
}
main { max-width: 60rem; margin: 0 auto; }
h1 { font-size: 1.35rem; font-weight: 600; margin: 0 0 .25rem; }
h2 { font-size: .8rem; letter-spacing: .12em; text-transform: uppercase;
     color: #7d93a8; margin: 2rem 0 .6rem; }
.sub { color: #7d93a8; margin: 0 0 1.5rem; font-size: .9rem; }
.counts { display: flex; gap: .5rem; flex-wrap: wrap; margin-bottom: .5rem; }
.pill { border: 1px solid #22303d; border-radius: 999px; padding: .15rem .6rem;
        font-size: .8rem; color: #9fb3c6; }
pre { background: #0f151c; border: 1px solid #1c2733; border-radius: 8px;
      padding: 1rem; overflow-x: auto; white-space: pre-wrap; word-break: break-word;
      font: 13px/1.6 ui-monospace, SFMono-Regular, Menlo, monospace; }
table { width: 100%; border-collapse: collapse; font-size: .9rem; }
th, td { text-align: left; padding: .45rem .5rem; border-bottom: 1px solid #1c2733;
         vertical-align: top; }
th { color: #7d93a8; font-weight: 500; font-size: .78rem; text-transform: uppercase;
     letter-spacing: .08em; }
td.time { color: #8ba3b8; white-space: nowrap; font-variant-numeric: tabular-nums; }
td.status { white-space: nowrap; }
.succeeded { color: #6fcf97; }
.failed { color: #eb5757; }
.running, .awaiting_approval, .planning { color: #f2c94c; }
.muted { color: #6b7f92; }
a { color: #6aa9e0; }
"""


def _summary_block(conn: sqlite3.Connection, day: date) -> str:
    content = stored(conn, day)
    if content is None:
        return (
            '<p class="muted">No summary stored for this day yet. '
            "Generate it with <code>claireai --summary</code>.</p>"
        )
    return f"<pre>{_esc(content)}</pre>"


def _tasks_table(lines) -> str:  # noqa: ANN001
    if not lines:
        return '<p class="muted">No tasks recorded for this day.</p>'
    rows = []
    for line in lines[:MAX_REQUESTS_SHOWN]:
        # Built outside the f-string on purpose: backslashes inside an f-string
        # expression are a SyntaxError on Python 3.11, which CI still runs.
        tool_note = f'<span class="muted"> — {_esc(", ".join(line.tools))}</span>' if line.tools else ""
        skip_note = f'<span class="muted"> · skipped: {_esc(", ".join(line.skipped))}</span>' if line.skipped else ""
        rows.append(
            "<tr>"
            f'<td class="time">{_esc(line.local_time)}</td>'
            f"<td>{_esc(_clip(line.request))}{tool_note}{skip_note}</td>"
            f'<td class="status {_esc(line.status)}">{_esc(line.status)}</td>'
            "</tr>"
        )
    shown = min(len(lines), MAX_REQUESTS_SHOWN)
    footer = (
        f'<p class="muted">Showing {shown} of {len(lines)} tasks.</p>'
        if len(lines) > shown
        else ""
    )
    return (
        "<table><thead><tr><th>Time</th><th>Request</th><th>Status</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>{footer}"
    )


def render_today(conn: sqlite3.Connection, *, day: date | None = None, tz=None) -> str:  # noqa: ANN001
    """Build the page. Every value that came from storage is escaped.

    ``tz`` is threaded through to :func:`claireai.summary.build` so tests can pin
    a timezone instead of depending on the machine's — the first version of the
    local-time test passed only because the development box was UTC+2.
    """
    day = day or datetime.now(tz).date()
    summary = build(conn, day=day, tz=tz)
    counts = (
        f'<span class="pill">{len(summary.done)} done</span>'
        f'<span class="pill">{len(summary.failed)} failed or skipped</span>'
        f'<span class="pill">{len(summary.pending)} pending</span>'
    )
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="color-scheme" content="dark">
<title>ClaireAI — {_esc(day.isoformat())}</title>
<style>{STYLE}</style>
</head>
<body>
<main>
<h1>ClaireAI — {_esc(day.isoformat())}</h1>
<p class="sub">Read-only view of the task log. The log is the source of truth;
this page cannot act, approve or change anything.</p>
<div class="counts">{counts}</div>

<h2>Daily summary</h2>
{_summary_block(conn, day)}

<h2>Tasks</h2>
{_tasks_table(summary.lines)}
</main>
</body>
</html>
"""


def render_unavailable(reason: str) -> str:
    """A page for "there is no database to show" — still escaped, still honest."""
    return (
        "<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\">"
        "<meta name=\"color-scheme\" content=\"dark\">"
        f"<title>ClaireAI — no data</title><style>{STYLE}</style></head>"
        f"<body><main><h1>No task log to show</h1><p class=\"sub\">{_esc(reason)}</p>"
        "</main></body></html>"
    )


def open_read_only(path: Path) -> sqlite3.Connection | None:
    """Open the database read-only, or ``None`` if there is no file yet.

    ``mode=ro`` is the guarantee that this page cannot write, whatever a handler
    bug might try.
    """
    path = Path(path)
    if not path.exists():
        return None
    conn = sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


__all__ = ["MAX_REQUESTS_SHOWN", "open_read_only", "render_today", "render_unavailable"]
