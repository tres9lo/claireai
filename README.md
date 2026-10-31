# ClaireAI

A personal assistant that runs on your own machine: local model inference, a
logged task trail, and (later) tools that act on files, the browser and the
desktop — with approvals, allow-lists and dry runs before anything mutates.

The project name and the assistant name are both **ClaireAI**. Package names,
application IDs and visible branding stay provisional until the owner confirms
them; nothing user-facing has been branded yet.

## Status: Month 1, Week 4 (daily summary; the desktop window is not built)

Implemented and tested:

- validated configuration with loopback-only model endpoints by default;
- SQLite storage with a versioned, checksummed migration runner;
- `tasks`, `task_log` (append-only audit), `summaries`, `notes` tables;
- provider-neutral `Brain` interface with an `OllamaBrain` adapter, including
  tool calls;
- CLI chat that records every turn as a task plus an audit entry;
- a tool registry with five **read-only** file tools (`list_folder`,
  `get_file_info`, `read_text_file`, `search_files`, `find_largest_files`), each
  with a name, description, JSON schema and risk level;
- six **approval-gated** write tools (`create_folder`, `create_text_file`,
  `write_to_file`, `rename_file`, `move_file`, `move_to_trash`). Nothing deletes
  permanently: `move_to_trash` relocates into ClaireAI's own recoverable bin;
- an approval flow: the exact tool, arguments and target are shown to the owner,
  the decision is logged with a reference, and a non-interactive run refuses
  instead of self-approving;
- an agent loop: the model proposes a call, the runtime validates the arguments,
  applies the risk policy and the folder allow-list, asks for approval when
  needed, runs it, feeds the result back, and repeats up to 8 steps — logging
  every step;
- **tool routing** (`claireai.routing`): each request gets only the tools its
  intent can reach — three schemas instead of eleven for a folder listing — which
  roughly halves the prompt tokens re-processed on every step. It is keyword
  based, so it costs no model call, and it only changes what the model is *shown*:
  validation, the allow-list and approvals still run against the full registry;
- the **daily summary** (`claireai.summary`): Done / Failed or skipped / Still
  pending, built from the task log by plain code and saved to the `summaries`
  table and `<data-dir>/summaries/YYYY-MM-DD.md`, on demand or on a schedule;
- a **presence renderer v0**: a canvas 2D HUD (no npm, no WebGL dependency) served
  over loopback by the standard library, driven by the `PresenceEvent` stream;
- `scripts/smoke.py`: an end-to-end check (database, model, tool loop, approval
  refusal, presence, append-only log) that runs unchanged on Linux and Windows;
- a model capability harness (`scripts/eval_model.py`) that scores the local
  model against fixed cases before features depend on it.

Not implemented yet (planned for later weeks): the FastAPI service, the Playwright
browser agent, the Tauri desktop shell (the HUD is a loopback preview, not the
packaged window), mobile clients, the from-scratch model, and machine access —
camera, screen, cursor and keyboard — which is specified below but has no code.

See `ROADMAP.md` for the full plan and `PROGRESS.md` for the current state,
verified results, and open blockers.

## Layout

```
assistant/        Python runtime (Track A): config, storage, audit, brain, CLI
brain-lab/        from-scratch tokenizer/model/training research (Track B), isolated
desktop/          Tauri v2 + React shell (not started)
mobile/           Expo/React Native client (not started)
docs/             setup, architecture, platform support, licenses
scripts/          diagnostics: doctor.py, eval_model.py, model_eval_cases.json
runtime-data/     local only, git-ignored (databases, logs, downloads, profiles)
```

## Quick start (Linux)

Requires Python 3.11+ and a local [Ollama](https://ollama.com/download) install.

```bash
# 1. Python environment (uses uv's seeding here because python3-venv was absent)
uv venv --seed --python 3.12 assistant/.venv
uv pip install --python assistant/.venv/bin/python -e "assistant[dev]"

# 2. Local model, downloaded once (no API key, works offline afterwards)
ollama pull qwen2.5:1.5b

# 3. Ask one question; the turn is logged to SQLite
CLAIREAI_DATA_DIR=./runtime-data assistant/.venv/bin/python -m claireai.cli --once "Hello"

# 4. Let the model use read-only file tools (deny-by-default allow-list)
CLAIREAI_DATA_DIR=./runtime-data \
CLAIREAI_ALLOWED_FOLDERS='["~/Downloads"]' \
assistant/.venv/bin/python -m claireai.cli --once "List the biggest files in my Downloads folder"
```

Details, Windows steps and troubleshooting: `docs/setup/linux.md`,
`docs/setup/windows.md`, `docs/platform-support.md`.

## Tools and the allow-list

`CLAIREAI_ALLOWED_FOLDERS` is a JSON array of the only folders file tools may
read. It is **empty by default**: with no entries, every file tool is refused.

```bash
CLAIREAI_ALLOWED_FOLDERS='["~/Downloads", "~/Documents"]'
```

The model chooses a tool and its arguments; the runtime does all of this:

- validates arguments against the tool's schema and refuses unexpected keys;
- resolves the path, follows symlinks, and refuses anything outside the
  allow-list (a link inside a trusted folder that points out is refused too);
- refuses any tool whose risk is above `safe` — this build has no approval flow,
  so "needs approval" means "refused", never "allowed silently";
- logs one `task_log` row per step (tool, arguments, target, status, result
  size) and one for the final answer;
- stops after `CLAIREAI_AGENT_MAX_STEPS` steps (default 8) and fails the task
  rather than looping forever.

Write tools ask before they act:

```
ClaireAI wants to run `create_folder` (needs_approval).
  target: /home/tresor/Downloads/reports
  arguments: path='/home/tresor/Downloads/reports', parents=True
  approval reference: 3f9c1e7a55b2
approve? [y/N]
```

Only an explicit `y`/`yes` approves. Piping input into the CLI cannot approve
anything — a non-interactive run refuses every write and says why. Each decision
is logged as `approval.decision`, and the tool row that follows carries the same
reference, so the audit trail names the exact human decision that allowed it.

Useful flags: `--no-tools` for plain chat, `--dry-run` to see which tools would
be called without running or reading anything, `--presence` to print state
changes, `--no-routing` to offer every tool instead of narrowing per request.

## Daily summary

```bash
# Today, printed and saved (no model needed — the report is derived from the log)
assistant/.venv/bin/python -m claireai.cli --summary

# A specific day
assistant/.venv/bin/python -m claireai.cli --summary 2026-10-31

# Every evening at 18:00 local time, in the foreground
assistant/.venv/bin/python -m claireai.cli --summary-at 18:00

# Any time, from the interactive prompt
/summary
```

The report has three fixed sections — **Done**, **Failed or skipped**, **Still
pending** — and each task appears exactly once: a task that finished but had a
write refused is listed under Failed or skipped as "completed, but skipped".
Nothing is summarised by the model, so a report cannot invent a task. Re-running a
day replaces that day's report in both places.

## Presence preview

```bash
assistant/.venv/bin/python scripts/presence_preview.py --demo
# open http://127.0.0.1:8765/ — cycles idle → listening → thinking → tool_running
#   → awaiting_approval → speaking, with mic/camera/screen/input indicators
```

Standard library only: the page is canvas 2D, capped at 30 fps, honours reduced
motion, and the event stream is the same `PresenceEvent` contract the real UI
will consume. The server is loopback-only, serves one static file plus read-only
state, and has no way to invoke a tool.

## Smoke check

```bash
assistant/.venv/bin/python scripts/smoke.py              # needs Ollama
assistant/.venv/bin/python scripts/smoke.py --skip-model # no model needed
```

Twelve checks in one run: settings, migrations, model reachability, a real chat
turn, the tool loop, routing narrowing the schema set, a refused write, the
presence stream, the daily summary, and that the audit log really rejects
UPDATE/DELETE. It uses `pathlib` and a temporary directory, so the same command
works on Windows.

## Planned: presence renderer and machine access

Two features are specified and scheduled but **not built yet**. Their contracts
are written down first on purpose, because both touch privacy:

- **Presence renderer** (`docs/architecture/presence-renderer.md`) — the
  assistant's face: a local, state-driven scene showing idle / listening /
  thinking / tool_running / awaiting_approval / speaking / error, plus indicators
  for microphone, camera, screen and input. It renders state from the same task
  log the audit trail uses; it holds no capability and never decides anything. A
  canvas 2D v0 already runs as a loopback preview; the WebGL scene and the Tauri
  window follow when the desktop toolchain is installed.
- **Machine access** (`docs/architecture/machine-access.md`) — camera snapshots,
  screen capture, clipboard, window listing, cursor and keyboard control, all
  denied by default and granted per task with a visible indicator. Perception
  arrives in Months 2–3; input control is `dangerous` and asks every time until
  a named action is trusted.

Reserved configuration (validation is live, the features are not):

```bash
CLAIREAI_PRESENCE_MODE=auto          # off | auto | 2d | webgl
CLAIREAI_PRESENCE_FPS_CAP=30         # 5-60, 30 recommended on integrated GPUs
CLAIREAI_PRESENCE_REDUCE_MOTION=false
CLAIREAI_ALLOWED_CAPABILITIES='["screen","camera"]'   # empty = refuse everything
CLAIREAI_CAMERA_DEVICE=0
```

`claireai.permissions.check_capability` already enforces the deny-by-default rule
(and has tests), so the first camera or cursor tool must ask that policy rather
than invent its own. `scripts/doctor.py` reports the configured model and paths
today; it does not yet print capability or renderer settings.

## Safety defaults

- The model runtime must be loopback unless explicitly overridden.
- No hosted model provider, no API keys.
- File tools are deny-by-default: nothing is readable until a folder is
  allow-listed, and every path is checked in code, not by the model.
- Tools that write, move or delete do not exist yet; when they arrive they are
  `needs_approval`/`dangerous` and refused until an approval flow exists.
- Audit rows are append-only at the database level (triggers reject UPDATE/DELETE).
- Secrets are redacted and long text truncated before anything is stored.
- `runtime-data/`, databases, model weights, browser profiles and `.env` files are git-ignored.
