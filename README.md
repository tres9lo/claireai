# ClaireAI

A personal assistant that runs on your own machine: local model inference, a
logged task trail, and (later) tools that act on files, the browser and the
desktop — with approvals, allow-lists and dry runs before anything mutates.

The project name and the assistant name are both **ClaireAI**. Package names,
application IDs and visible branding stay provisional until the owner confirms
them; nothing user-facing has been branded yet.

## Status: Month 1, Week 1 (foundations)

Implemented and tested:

- validated configuration with loopback-only model endpoints by default;
- SQLite storage with a versioned, checksummed migration runner;
- `tasks`, `task_log` (append-only audit), `summaries`, `notes` tables;
- provider-neutral `Brain` interface with an `OllamaBrain` adapter;
- CLI chat that records every turn as a task plus an audit entry;
- a model capability harness (`scripts/eval_model.py`) that scores the local
  model against fixed cases before features depend on it.

Not implemented yet (planned for later weeks): the FastAPI service, tool
registry, permission/approval layer, browser and file agents, the daily summary
scheduler, desktop and mobile clients, and the from-scratch model.

See `ROADMAP.md` for the full plan and `PROGRESS.md` for the current state,
verified results, and open blockers.

## Layout

```
assistant/        Python runtime (Track A): config, storage, audit, brain, CLI
brain-lab/        from-scratch tokenizer/model/training research (Track B), isolated
desktop/          Tauri v2 + React shell (not started)
mobile/           Expo/React Native client (not started)
docs/             setup, architecture, platform support, licenses
scripts/          setup and diagnostic scripts (not written yet)
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
```

Details, Windows steps and troubleshooting: `docs/setup/linux.md`,
`docs/setup/windows.md`, `docs/platform-support.md`.

## Safety defaults

- The model runtime must be loopback unless explicitly overridden.
- No hosted model provider, no API keys.
- Audit rows are append-only at the database level (triggers reject UPDATE/DELETE).
- Secrets are redacted and long text truncated before anything is stored.
- `runtime-data/`, databases, model weights, browser profiles and `.env` files are git-ignored.
