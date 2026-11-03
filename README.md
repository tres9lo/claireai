# ClaireAI

**A personal assistant that runs entirely on your own machine.** Local model inference, an append-only audit trail of every action, and file and browser tools that cannot act without your explicit approval.

No cloud provider. No API keys. Nothing leaves the laptop unless you allow it.

![ClaireAI presence HUD](docs/screenshots/presence-hud.png)

<p align="center">
  <a href="https://github.com/tres9lo/claireai/actions/workflows/python-tests.yml"><img src="https://github.com/tres9lo/claireai/actions/workflows/python-tests.yml/badge.svg" alt="Tests"></a>
  <img src="https://img.shields.io/badge/python-3.11%20%7C%203.12-blue" alt="Python 3.11 and 3.12">
  <img src="https://img.shields.io/badge/platform-Linux%20%7C%20Windows-lightgrey" alt="Linux and Windows">
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-green" alt="MIT license"></a>
</p>

---

## Why this exists

Most assistants are a chat box wired to someone else's server. They read your files through an API, remember what you said on infrastructure you don't own, and ask you to trust that the model will behave.

ClaireAI takes the opposite position: **the model is untrusted, the runtime is the authority.** The model may only propose an action. The runtime validates the tool name, checks the arguments against a schema, applies a folder allow-list, consults the risk policy, asks you when a write is involved, then executes — and writes an audit row that the database itself refuses to let you edit.

The result is an assistant you can actually point at real folders, because the blast radius is defined in code rather than in a prompt.

## What works today

| Capability | Status |
| --- | --- |
| From-scratch byte tokenizer and decoder-only transformer research model | ✅ foundation |
| SQLite storage with a versioned, checksummed migration runner | ✅ |
| Append-only audit log (`task_log`), enforced by database triggers | ✅ |
| Provider-neutral `Brain` interface + local scratch-checkpoint adapter | ✅ foundation |
| CLI chat that records every turn as a task plus audit entries | ✅ |
| 5 read-only file tools (list, info, read, search, largest) | ✅ |
| 6 approval-gated write tools (create, write, rename, move, trash) | ✅ |
| Interactive approval flow with per-decision audit references | ✅ |
| Agent loop with schema validation, risk policy and step cap | ✅ |
| Deterministic tool routing (roughly halves prompt tokens per step) | ✅ |
| Daily summary (Done / Failed / Pending) from the task log | ✅ |
| Read-only `/today` page served over loopback | ✅ |
| Retention/pruning for trash and summaries | ✅ |
| Presence HUD (canvas 2D, loopback preview) | ✅ |
| Browser tools, disabled until enabled in the local control panel | ✅ |
| Runtime stop switch and browser access control panel | ✅ first version |
| OS family detection (Linux/Windows first targets) | ✅ detection only |
| Licensed corpus, trained capable assistant, TTS voice controls | 🚧 not included yet |
| FastAPI service · Playwright engine · Tauri desktop shell | 🚧 planned |
| Camera, screen, cursor and keyboard control | 🚧 specified, not built |

The project is at **Month 1 of its roadmap** and has not had a public release. See [ROADMAP.md](ROADMAP.md) for the plan and [PROGRESS.md](PROGRESS.md) for verified results and open blockers.

## How a request flows

```
you (terminal)
  │  "List the biggest files in my Downloads folder"
  ▼
claireai.cli ──► claireai.agent.run_task
  │                 │
  │                 ├─ tasks.create()          → queued → running
  │                 ├─ presence.emit("thinking")
  │                 ▼
  │            loop (≤ 8 steps)
  │                 ├─ brain.generate(messages, tools) ──► locally trained PyTorch checkpoint
  │                 ├─ no tool call → answer, status = succeeded
  │                 └─ tool call →
  │                      1. registry lookup      unknown → refused
  │                      2. validate arguments   bad schema → refused
  │                      3. risk policy          above safe → approval
  │                      4. approval decision    CLI prompt or deny
  │                      5. sandboxed execution  allow-list enforced
  │                      6. audit row            append-only
  ▼
answer + full trail in SQLite
```

Every step is a row: the tool, its arguments, the resolved target, the status, and the approval reference that authorised it.

## The safety model

This is the part worth reviewing before you run it.

- **Deny by default.** `CLAIREAI_ALLOWED_FOLDERS` is empty out of the box. With no entries, every file tool is refused.
- **Paths are resolved in code.** Symlinks are followed, and a link inside a trusted folder that points outside it is refused.
- **Arguments are validated, not trusted.** Each tool declares a pydantic schema; unexpected keys are rejected rather than ignored.
- **Writes ask first.** The exact tool, arguments and target are printed, and only an explicit `y`/`yes` approves. Piping input into the CLI cannot approve anything — a non-interactive run refuses and says why.
- **The audit log is append-only at the database level.** Triggers reject `UPDATE` and `DELETE`; retention reports the log instead of pruning it.
- **Nothing is deleted permanently.** `move_to_trash` relocates into ClaireAI's own recoverable bin.
- **Secrets are redacted and long text truncated** before storage.
- **Inference is local.** ClaireAI loads its own checkpoint and does not call a hosted model API.
- **Capabilities are opt-in.** Browser access starts disabled. The local HUD can enable it; browser actions still use the existing approval path. The stop control blocks new model/tool steps after the current model call finishes.

Read [docs/architecture/threat-model.md](docs/architecture/threat-model.md) for the full boundary analysis.

## Quick start (Linux)

Requires **Python 3.11+**, PyTorch, and a licensed text corpus. The scratch model starts with random weights; it is a research prototype and will not behave like a pretrained assistant until trained and evaluated on substantial task-specific data.

```bash
git clone https://github.com/tres9lo/claireai.git
cd claireai

# 1. Python environment
uv venv --seed --python 3.12 assistant/.venv
uv pip install --python assistant/.venv/bin/python --index-url https://download.pytorch.org/whl/cpu torch
uv pip install --python assistant/.venv/bin/python -e "assistant[dev,model]"
uv pip install --python assistant/.venv/bin/python -e ./brain-lab

# 2. Add licensed .txt training data under brain-lab/data/raw/ (not committed)
# Then train a small CPU checkpoint. Increase --steps after confirming the first run.
export CLAIREAI_DATA_DIR=./runtime-data
assistant/.venv/bin/python brain-lab/train.py brain-lab/data/raw --steps 1000

# 3. Ask one question; the turn is logged to SQLite
CLAIREAI_DATA_DIR=./runtime-data \
  assistant/.venv/bin/python -m claireai.cli --once "Hello"

# 4. Let the model use read-only file tools (allow-list required)
CLAIREAI_DATA_DIR=./runtime-data \
CLAIREAI_ALLOWED_FOLDERS='["~/Downloads"]' \
  assistant/.venv/bin/python -m claireai.cli --once "List the biggest files in my Downloads folder"
```

Windows setup uses the same Python and CPU training path; see [docs/setup/windows.md](docs/setup/windows.md).

### Granting a folder

```bash
export CLAIREAI_ALLOWED_FOLDERS='["~/Downloads", "~/Documents"]'
```

Only those trees are visible to the file tools. Everything else is refused.

### Everyday commands

```bash
# Interactive session
assistant/.venv/bin/python -m claireai.cli

# Plain chat, no tools
assistant/.venv/bin/python -m claireai.cli --no-tools

# Show which tools would be called, without running them
assistant/.venv/bin/python -m claireai.cli --dry-run "clean up my downloads"

# Today's summary, printed and saved (no model needed)
assistant/.venv/bin/python -m claireai.cli --summary

# Daily summary at 18:00 local time
assistant/.venv/bin/python -m claireai.cli --summary-at 18:00

# Local HUD and runtime controls at http://127.0.0.1:8765/
assistant/.venv/bin/python scripts/presence_preview.py --data-dir ./runtime-data

# Self-improvement: enable it in the HUD, then train an isolated candidate
assistant/.venv/bin/python brain-lab/train.py brain-lab/data/raw --steps 200 --improvement

# Diagnostics
assistant/.venv/bin/python scripts/doctor.py
```

## Architecture

```
assistant/      Python runtime: config, storage, audit, brain, tools, CLI
desktop/        Presence HUD (canvas 2D) and the future Tauri shell
docs/           Setup, architecture, platform support, licenses
scripts/        doctor.py, eval_model.py, smoke.py, presence_preview.py
runtime-data/   Local only, git-ignored: databases, logs, downloads, trash
```

The runtime is deliberately modular: `brain` never imports a tool, `tools` never import the agent, and `presence` holds no capability. A full module-by-module map is in [docs/architecture/overview.md](docs/architecture/overview.md).

### Why tool routing

Re-sending every tool schema on every loop step is the dominant cost on a CPU-only machine. `claireai.routing` narrows the schemas a request can reach using deterministic keyword intent — no model call — and measured roughly **50–60% fewer prompt tokens** on three requests with no change in tool choice. Routing changes what the model *sees*, never what is *allowed*: validation, the allow-list and approvals still run against the full registry.

## Testing

```bash
cd assistant && python -m pytest        # unit + security tests
assistant/.venv/bin/python scripts/smoke.py --skip-model   # end-to-end, no model
assistant/.venv/bin/python scripts/smoke.py                # full run, needs a trained checkpoint
```

CI runs the portable unit suite on Ubuntu and Windows across Python 3.11 and 3.12. The 14-check smoke script verifies the real loop: migrations, a chat turn, tool execution, a refused write, the presence stream, the daily summary, the read-only page, retention, and that the audit log rejects mutation.

Model behaviour is measured, not assumed: `scripts/eval_model.py` scores the local model against fixed cases before features depend on it.

## Documentation

| Document | Contents |
| --- | --- |
| [docs/setup/linux.md](docs/setup/linux.md) | Install and first run on Linux |
| [docs/setup/windows.md](docs/setup/windows.md) | Install and first run on Windows |
| [docs/architecture/overview.md](docs/architecture/overview.md) | Module map and request lifecycle |
| [docs/architecture/threat-model.md](docs/architecture/threat-model.md) | Trust boundaries and abuse cases |
| [docs/architecture/approval-from-the-page.md](docs/architecture/approval-from-the-page.md) | Why approvals are not a web button |
| [docs/architecture/presence-renderer.md](docs/architecture/presence-renderer.md) | The assistant's visual state contract |
| [docs/architecture/machine-access.md](docs/architecture/machine-access.md) | Camera, screen and input policy |
| [docs/platform-support.md](docs/platform-support.md) | What is verified on which OS |
| [ROADMAP.md](ROADMAP.md) | Phased plan |
| [PROGRESS.md](PROGRESS.md) | Current state, measurements, blockers |
| [CHANGELOG.md](CHANGELOG.md) | Notable changes |

## Contributing

Issues and pull requests are welcome. Please read [CONTRIBUTING.md](CONTRIBUTING.md) and [SECURITY.md](SECURITY.md) first — changes that touch the permission layer, the audit log or the sandbox need tests that prove the refusal path still holds.

## License

[MIT](LICENSE). Bundled model and third-party data licences are listed in [docs/model-and-data-licenses.md](docs/model-and-data-licenses.md).
