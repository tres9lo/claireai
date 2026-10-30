# ClaireAI

A personal assistant that runs on your own machine: local model inference, a
logged task trail, and (later) tools that act on files, the browser and the
desktop — with approvals, allow-lists and dry runs before anything mutates.

The project name and the assistant name are both **ClaireAI**. Package names,
application IDs and visible branding stay provisional until the owner confirms
them; nothing user-facing has been branded yet.

## Status: Month 1, Week 2 (first tools and agent loop)

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
- an agent loop: the model proposes a call, the runtime validates the arguments,
  applies the risk policy and the folder allow-list, runs it, feeds the result
  back, and repeats up to 8 steps — logging every step;
- a model capability harness (`scripts/eval_model.py`) that scores the local
  model against fixed cases before features depend on it.

Not implemented yet (planned for later weeks): approval prompts and any tool that
mutates files (Week 3), the FastAPI service, the browser agent, the daily summary
scheduler, desktop and mobile clients, the from-scratch model, and the two
planned surfaces described below — the presence renderer and machine access.

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

Useful flags: `--no-tools` for plain chat, `--dry-run` to see which tools would
be called without reading anything.

## Planned: presence renderer and machine access

Two features are specified and scheduled but **not built yet**. Their contracts
are written down first on purpose, because both touch privacy:

- **Presence renderer** (`docs/architecture/presence-renderer.md`) — the
  assistant's face: a local, state-driven 2D/WebGL scene in the desktop shell
  showing idle / listening / thinking / tool_running / awaiting_approval /
  speaking / error, plus indicators for microphone, camera, screen and input.
  It renders state from the same task log the audit trail uses; it holds no
  capability and never decides anything. Built as v0 in Month 1 Week 4.
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
