# ClaireAI

**A personal assistant that runs entirely on your own machine.** Local model inference, an append-only audit trail of every action, and file and browser tools that cannot act without your explicit approval.

No cloud provider or API key is required by default. Network features and connectors are opt-in; nothing leaves the laptop unless you enable and approve it.

> **Runtime change (2026-11-25).** ClaireAI now uses **Ollama** as its brain: **Qwen3 1.7B** for normal turns and **Qwen2.5-Coder 1.5B** for code-heavy turns. Qwen3's hidden thinking is off by default because it cost 7.5 s for a 32-token probe versus 1.8 s with thinking off; enable it with `CLAIREAI_BRAIN_THINK=true` when a task needs deliberate reasoning. The from-scratch trainer in `brain-lab/` is kept as archived research, not as the required path. The reason is practical: this project is developed on a **budget PC (7 GB RAM, CPU only)**. Training a useful model from random initialization was slow and difficult, several candidates failed the gate, and the time cost did not produce a usable assistant. Serving a small pretrained model locally is the honest fit for that hardware.

### Why the model approach changed

Training from scratch asked a laptop-class CPU to do two jobs at once: learn a tokenizer and language from a corpus, then learn tool use and conversation on top. Each full run took hours, the v15 candidate failed its gate, and the next run had to be restarted. That is not a good use of the machine or the owner's time, and it delayed the parts that make the assistant useful: permissions, approvals, the audit trail, tools and a desktop surface.

Ollama changes the constraint rather than hiding it. The model is pretrained, so the CPU only has to run inference. It still stays local: the daemon listens on loopback, the weights are on this disk, and the audit row records which model answered. The runtime remains the authority; a better model does not get extra permissions.


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

## The plan, stated plainly

ClaireAI is built around one idea: **you should own the assistant and the data it touches.** Three things follow from that, in this order.

1. **A small pretrained model, served locally by Ollama.** Qwen3 1.7B is the main brain and Qwen2.5-Coder 1.5B handles code turns. Ollama runs on loopback, the weights stay on disk, and inference needs no API key. This is the right trade for a budget laptop: no multi-hour training run, no GPU, and a model that can actually hold a conversation.
2. **The runtime is still the authority.** The model may only propose a tool call. `claireai.agent` validates the name, validates the arguments with Pydantic, applies the folder allow-list and risk policy, asks you before a write, executes in the sandbox, and appends the audit row. A stronger model does not get a wider permission boundary.
3. **The trainer remains as research, not as a prerequisite.** `brain-lab/` still documents the from-scratch experiments and their failures honestly, but a fresh install no longer starts by asking you to train a checkpoint.

### On the model work, honestly

From-scratch training was measured, not hand-waved: the runs and failures are in [PROGRESS.md](PROGRESS.md). The conclusion on this hardware is that the effort was disproportionate. The supported runtime is now Ollama + Qwen; `brain-lab/` is kept for experiments and reproducibility, not for the normal install path.

## What works today

| Capability | Status |
| --- | --- |
| Ollama runtime with Qwen3 1.7B (main) and Qwen2.5-Coder 1.5B (code) | ✅ default |
| SQLite storage with a versioned, checksummed migration runner | ✅ |
| Append-only audit log (`task_log`), enforced by database triggers | ✅ |
| Provider-neutral `Brain` interface + Ollama / hosted adapters | ✅ |
| Archived from-scratch tokenizer and transformer research | 🧪 legacy research only |
| Opt-in Claude and ChatGPT connectors | ✅ optional |
| CLI chat that records every turn as a task plus audit entries | ✅ |
| Read-only tools (files, calendar, day plan, image identification; browser tools when enabled) | ✅ |
| 10 approval-gated writes (folder, file, overwrite/append, surgical replace, rename, move, trash, download, appointment, day plan) | ✅ |
| Interactive approval flow with per-decision audit references | ✅ |
| Agent loop with schema validation, risk policy and step cap | ✅ |
| Strict ToolACE/Hermes name mapping to real registry schemas | ✅ code path; full mapped sample pending |
| Tokenizer-backed corpus audit (budgets, calls, duplicates, held-out overlap) | ✅ |
| Legacy post-training gate for `brain-lab` candidates | 🧪 research only |
| Deterministic tool routing (roughly halves prompt tokens per step) | ✅ |
| Daily summary (Done / Failed or skipped / Still pending) from the task log | ✅ |
| Dates and times answered from the clock; `yesterday`/`last friday` accepted by the summary | ✅ deterministic, no model call |
| Workspace calendar: appointments in SQLite, `what's on tomorrow` answered from them | ✅ listing is read-only, adding needs approval |
| Retention/pruning for trash and summaries | ✅ |
| Browser tools, disabled until enabled in Settings | ✅ |
| Unknown public question → approved web search, cited answer, optional confirmed lesson | ✅ |
| Runtime stop switch and browser access control | ✅ first version |
| Native desktop window and terminal chat, no browser surface | ✅ |
| Stored conversation history, owner directives and preferences | ✅ first version |
| Offline speech output with pitch/rate controls | ✅ Piper neural voice, else eSpeak NG |
| Push-to-talk local speech transcription in the desktop Voice page | ✅ optional faster-whisper + local model |
| Local semantic memory (SQLite vector index; Ollama or explicit hash embeddings) | ✅ `remember_note` / `search_memory`; Memory page lists/deletes entries and can embed turns; embeddings stay local |
| Screen reading and CUA input driver | ✅ capability-gated; live screen capture, pointer movement and keyboard input verified on Linux; cursor/keyboard are `dangerous` and always approved |
| Wake-word engine | ✅ local speech phrase engine for **“Hey Claire”** using the installed faster-whisper model (`stt`), plus the openWakeWord adapter; synthetic spoken clip recognized, human-spoken acceptance still pending |
| Visual canvases | ✅ `create_canvas` turns structured sections/steps/sources into a modern local HTML page; `media` adds live news cards, YouTube/Vimeo embeds, images and links. Native **Canvas** page lists and opens them |
| App and browser launch | ✅ `list_apps` / `open_app` / `open_in_browser` behind the opt-in `apps` capability and per-action approval; pointer/keyboard control operates the window |
| Window arrangement | ✅ approved `list_windows` / `arrange_windows`: focus, move, resize, center, tile left/right, maximize, tile all, and browser tab reorder via keyboard |
| Presence-first desktop | ✅ animated local-neural **Presence** screen with a Presence/Chat switch; the permission card handles allow/deny in-window |
| Learning and self-improvement | ✅ opt-in journal + background learner + gated trainer; startup now starts both workers correctly and the Settings page reports candidate status |
| FastAPI backend/API service | ✅ loopback bearer auth, tasks/cancel, devices, approvals, audit, summaries, settings, memory and SSE; tool writes still need a signed approval |
| OS family detection (Linux/Windows first targets) | ✅ detection only |
| Reproducible MIT synthetic task corpus + split-aware text training | ✅ starter data |
| Random-init visual-grounding prototype using attributed online synthetic images | 🧪 research only; not integrated |
| Tauri + React desktop shell | 🚧 target; current desktop is the tested Tk shell (kept deliberately) |
| Playwright engine | 🚧 planned |
| Camera capture and clipboard reading | 🚧 not built; screen and input control are implemented |

The tool-data path is guarded now. `brain-lab/tool_mapping.py` maps a source
name only when it normalises to a real registry tool or matches a documented
semantic alias, and validates the transformed arguments against that tool's
schema. A record with any unmapped offered tool, unmapped call or invalid
argument is dropped rather than teaching a capability the runtime cannot
execute. `brain-lab/audit_corpus.py` audits a corpus without loading the model,
and `scripts/run_gate.py` runs the complete post-training gate and writes one
`gate.json`; it never installs a checkpoint. The gate and install rule are in
[docs/gate-runbook.md](docs/gate-runbook.md).

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
  │                 ├─ brain.generate(messages, tools) ──► Ollama /api/chat (Qwen3 1.7B)
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
- **Task and audit logs redact secrets.** Full user and assistant messages are retained verbatim in the local conversation database so ClaireAI can show history and continue chats. The owner can review or delete conversations from the UI; the database is owner-readable only on POSIX systems.
- **Inference is local by default.** ClaireAI talks to an Ollama daemon on loopback and runs Qwen on this machine. No hosted API is called unless the owner explicitly selects one, and the audit row records which model answered. A remote model can only *propose* a tool call — approvals and the sandbox still decide what runs.
- **Capabilities are opt-in.** Browser access starts disabled. The desktop Settings page can enable it; browser actions still use the existing approval path. Stop ends model generation between tokens, lets a current bounded tool action finish, prevents subsequent model/tool calls, and exits the interactive CLI even while it is waiting for input. The window remains open; after allowing the runtime again, start a new turn.

Read [docs/architecture/threat-model.md](docs/architecture/threat-model.md) for the full boundary analysis.

## Quick start (Linux and Windows)

Requires **Python 3.11+** and **Ollama**. No training run is needed.

```bash
# 1. Install Ollama: https://ollama.com/download (Linux and Windows installers)
ollama serve                    # Linux; the Windows installer starts the service

# 2. Pull the two Qwen models once. They are stored locally by Ollama.
ollama pull qwen3:1.7b
ollama pull qwen2.5-coder:1.5b

# 3. Install ClaireAI. The installer lists every download and asks first.
git clone https://github.com/tres9lo/claireai.git
cd claireai
./install.sh                     # Windows PowerShell: .\install.ps1

# 4. Start the app and ask a question.
./scripts/start.sh
assistant/.venv/bin/python -m claireai.cli --once "Hello"       # terminal
.\assistant\.venv\Scripts\python.exe -m claireai.cli --once "Hello"  # Windows
```

The installer picks a Python whose Tk can draw antialiased fonts on Linux. Do not
`pip install` the app into a conda base on Linux: conda's Tk may be built for
the old X11 bitmap font system, which makes the window pixelated. The installer
checks this and can use the system Python instead.

```bash
# Optional local speech output. Piper is a neural voice; eSpeak NG is fallback.
assistant/.venv/bin/python -m pip install piper-tts
sudo apt install espeak-ng pulseaudio-utils

# Voice models go under runtime-data/models/voice/piper/ (or CLAIREAI_PIPER_DIR).
# https://huggingface.co/rhasspy/piper-voices

# Give the model one read-only folder (writes still need explicit approval).
CLAIREAI_DATA_DIR=./runtime-data \
CLAIREAI_ALLOWED_FOLDERS='["~/Downloads"]' \
  assistant/.venv/bin/python -m claireai.cli --once "List the biggest files in my Downloads folder"
```

On Windows, Ollama installs as a background service and exposes the same
`127.0.0.1:11434` endpoint. The Python runtime, SQLite storage, allow-list and
approval flow are shared; the setup steps are in [docs/setup/windows.md](docs/setup/windows.md).

### The desktop app

The native window is built on Tk with customtkinter for its surfaces: no
browser, no loopback web server, no JavaScript. It opens on **Chat** and has
seven separate screens in the sidebar — Chat, Activity, Approvals, History,
Voice, Settings and About — each with its own job rather than one long page.
The terminal chat is the same runtime and database, without the window.

Design tokens live in `assistant/src/claireai/desktop/theme.py`: one accent colour, rounded surfaces,
and motion timed from the wall clock rather than a frame count, so a slow
machine drops frames instead of playing every animation in slow motion. The
selected sidebar pill eases between pages, the status dot breathes, the chat
shows a live thinking indicator, and each page fades its surface in.

```bash
assistant/.venv/bin/claireai              # the window
assistant/.venv/bin/claireai --cli        # the terminal, same data
```

`./install.sh` also adds **ClaireAI** to the applications menu on Linux, using
the venv Python it installed. `claireai --update` (or **Update app** in Settings)
rebuilds this checkout and reinstalls it into that environment; the checkout path
is remembered, so the button works even when the window was started from the
menu. The first update from a copy with no remembered path can be pointed at the
source once with `claireai --update --source /path/to/claireai/assistant`.
`claireai --setup --install` is the same consent flow for missing pieces,
including the local speech model.

If the window does not open, the command names the missing piece instead of
printing a traceback: the `desktop` extra
(`assistant/.venv/bin/python -m pip install -e "assistant[desktop]"`) or Python's
Tk bindings (`python3-tk` on Debian/Ubuntu). It exits 2.

- **Chat** runs real agent turns; write actions wait in **Approvals** until you
  press Approve, and the sidebar shows how many are waiting.
- **Activity** lists every task with its status, and the error text when one failed.
- **Voice** speaks in English or French (the language is detected from the text,
  or chosen by hand), and records from the microphone when a speech model is
  installed.
- **Settings** holds the same preferences as the terminal, including the
  emergency stop switch; both share one database, so changes apply immediately.
- **Setup and updates** are explicit: `claireai --setup` reports the host and
  what is missing, `--setup --install` lists each download (including the local
  speech model) and asks first, and `--update` rebuilds and reinstalls this
  checkout without touching your data.

The window can start with your session: `claireai --install-autostart` writes
`~/.config/autostart/claireai.desktop`, which is the only file it ever adds
there, and `--uninstall-autostart` removes exactly that file.

### Background learning (legacy training)

> **Training is no longer part of the supported install.** The from-scratch
> runs below are kept as documented research. They are slow on this budget CPU
> and did not produce a dependable assistant. The Ollama + Qwen path does not
> need a corpus, a gate or a promoted checkpoint.

**Background learning** in Settings remains an opt-in journal. It records
exchanges locally for review; it does not silently rewrite the Qwen weights. The
old training worker and gate code remain in `brain-lab/` for experiments, and
the v15/v16 failures stay in [PROGRESS.md](PROGRESS.md) rather than being erased.

Background learning no longer teaches the model its own unverified answers. An
ordinary exchange is journaled for the owner, but only an owner-confirmed,
source-bearing research lesson is kept as a reviewable note. It does not update
the Qwen weights. If the model says it does not know a public factual question,
the runtime can offer a `research_web` approval, search, read the top public
result, and ask a second time before saving that sourced excerpt. Private and
first-person requests are never sent to the web automatically.

The archived training worker used to run without the window through a legacy
**systemd user timer**:

```bash
claireai --install-training-service    # daily, no root, no display
claireai --training-service-status     # active / next run
claireai --uninstall-training-service
```

The timer and `brain-lab` code are retained for research only. They are not part
of the Ollama/Qwen install and are not needed for normal use. Any training code
still stays local, does not scrape, and does not log into or post to an email or
social account. Official routes and requirements are in
[docs/email-and-social.md](docs/email-and-social.md).

Plain arithmetic, percentages and unit conversions are answered by deterministic
code, not by the model, in `claireai/calculations.py`. Licensed data sources,
licences and access routes are listed in
[docs/training-data-sources.md](docs/training-data-sources.md) and the machine-
readable [manifest](brain-lab/data-sources.json).

**Country news** is fetched from public publisher feeds and Bing News country
RSS, not generated by the model. Rwanda uses local publishers such as KT Press
and Taarifa first, because Google can redirect this host's `RW` edition to
`en-US`. Every item keeps its publisher, link and summary when one is available.
Browser access must be on; if it is off, the assistant says so. Use either form:

```bash
claireai --news Kenya,France
claireai --once "latest headlines from Kenya and France"
```

Transient DNS, timeout and 5xx failures are retried three times, and a final
resolver failure is reported as a DNS problem with the affected country instead
of being folded into an opaque `[Errno -3]` line. Headlines and publisher
summaries are not verified facts; the publisher link is the source to check.
Google News is a last-resort fallback for a non-DNS Bing failure when no local
story was found.
The old `brain-lab` pipeline remains documented for anyone who wants to rerun
the experiment: it builds `chat-v16`, applies token budgets, deduplicates and
gates candidates. It is not invoked by the Ollama/Qwen runtime and does not need
to run on a fresh install.

### Granting a folder

In the native window, open **Settings → File access** and use **Add folder…**.
The same allow-list can be set before launch:

```bash
export CLAIREAI_ALLOWED_FOLDERS='["~/Downloads", "~/Documents"]'
```

Only those trees are visible to the file tools. **Allow whole PC…** is an
explicit, separately confirmed choice and is never enabled by default. Writes
always ask for approval, even inside an allowed folder.

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

# Daily briefing now, or every day at 07:30 local time
assistant/.venv/bin/python -m claireai.cli --briefing --countries Kenya,France
assistant/.venv/bin/python -m claireai.cli --briefing-at 07:30 --countries Kenya,France

# Guided email setup: verifies read-only IMAP, then stores the address and
# app password in the OS keyring. Use a Gmail App Password, not the account
# password. Environment variables still override the stored values.
assistant/.venv/bin/python -m claireai.cli --email-setup
assistant/.venv/bin/python -m claireai.cli --inbox 10

# Daily summary at 18:00 local time
assistant/.venv/bin/python -m claireai.cli --summary-at 18:00

# Assign a full day of tasks, then inspect the plan
assistant/.venv/bin/python -m claireai.cli \
  --plan-task "09:00 Read the inbox" \
  --plan-task "14:00 Review the weekly report"
assistant/.venv/bin/python -m claireai.cli --day-plan

# Run tasks whose local time has arrived. Safe actions run; writes pause.
assistant/.venv/bin/python -m claireai.cli --day-plan-run
```

`--day-plan-run` is safe by default: it pauses at the first approval and leaves
that item `awaiting_approval` instead of granting itself a write. Add
`--day-plan-date tomorrow` when assigning or running tomorrow's plan.

In the native window, **Day plan** lists a local day, adds timed requests, and
runs due work in the background. It uses the same approval boundary as the CLI:
a write is paused and shown as `awaiting_approval`, never inherited from the
plan.

```bash
# The whole app in one command (installs the venv on first run, then opens it)
./scripts/start.sh

# What this machine is missing, without downloading anything
assistant/.venv/bin/claireai --setup

# List missing parts and download them only after a yes
assistant/.venv/bin/claireai --setup --install

# Start ClaireAI with your desktop session (XDG autostart), and undo it
assistant/.venv/bin/claireai --install-autostart
assistant/.venv/bin/claireai --autostart-status
assistant/.venv/bin/claireai --uninstall-autostart

# Terminal chat instead of the window
assistant/.venv/bin/claireai --cli

# Self-improvement: enable it in Settings, then train an isolated candidate
assistant/.venv/bin/python brain-lab/train.py brain-lab/data/bootstrap-v1/text --steps 200 --improvement

# Diagnostics
assistant/.venv/bin/python scripts/doctor.py

# Optional visual prototype (downloaded image data; kept separate from chat)
assistant/.venv/bin/python -m pip install -e "brain-lab[vision]"
assistant/.venv/bin/python brain-lab/vision_train.py --epochs 20
```

## Architecture

```
assistant/      Python runtime: config, storage, audit, brain, tools, desktop, CLI
docs/           Setup, architecture, platform support, licenses
scripts/        bootstrap.py, doctor.py, eval_model.py, conversation_eval.py, smoke.py
runtime-data/   Local only, git-ignored: databases, logs, downloads, trash
```

The runtime is deliberately modular: `brain` never imports a tool, `tools` never import the agent, and `presence` holds no capability. A full module-by-module map is in [docs/architecture/overview.md](docs/architecture/overview.md).

### Why tool routing

Re-sending every tool schema on every loop step is the dominant cost on a CPU-only machine. `claireai.routing` narrows the schemas a request can reach using deterministic keyword intent — no model call — and measured roughly **50–60% fewer prompt tokens** on three requests with no change in tool choice. Routing changes what the model *sees*, never what is *allowed*: validation, the allow-list and approvals still run against the full registry.

## Testing

```bash
# The desktop installer does not install test-only packages. Add them once.
assistant/.venv/bin/python -m pip install -e "assistant[dev]"

cd assistant && .venv/bin/python -m pytest   # unit + security tests
assistant/.venv/bin/python scripts/smoke.py --skip-model   # end-to-end, no model
assistant/.venv/bin/python scripts/smoke.py                # full run, needs a trained checkpoint
```

CI runs the portable unit suite on Ubuntu and Windows across Python 3.11 and 3.12. The smoke script verifies the assembled runtime: migrations, a chat turn, tool execution, tool routing, a refused write, the presence stream, the daily summary, retention, and that the audit log rejects mutation.

Model behaviour is measured, not assumed: `scripts/eval_model.py` scores the
configured brain against fixed cases, `conversation_eval.py` scores multi-turn
answers, and `tool_call_eval.py` scores whether the configured brain proposes
tool calls the runtime can actually read. With Ollama running, these exercise
Qwen directly.

```bash
ollama serve
assistant/.venv/bin/python scripts/doctor.py
assistant/.venv/bin/python scripts/tool_call_eval.py --runs 3
```

The old checkpoint gate remains for archived `brain-lab` candidates and is no
longer part of a normal install.

## Documentation

| Document | Contents |
| --- | --- |
| [docs/setup/linux.md](docs/setup/linux.md) | Install and first run on Linux |
| [docs/setup/windows.md](docs/setup/windows.md) | Install and first run on Windows |
| [docs/setup/models.md](docs/setup/models.md) | Install Ollama, pull Qwen and embedding models, storage and troubleshooting |
| [docs/setup/extras.md](docs/setup/extras.md) | Optional CUA, wake-word, memory and API setup |
| [docs/api.md](docs/api.md) | Local FastAPI service, authentication and endpoints |
| [docs/architecture/overview.md](docs/architecture/overview.md) | Module map and request lifecycle |
| [docs/architecture/threat-model.md](docs/architecture/threat-model.md) | Trust boundaries and abuse cases |
| [docs/architecture/approval-from-the-page.md](docs/architecture/approval-from-the-page.md) | Why approvals are not a web button |
| [docs/architecture/presence-renderer.md](docs/architecture/presence-renderer.md) | The assistant's visual state contract |
| [docs/architecture/machine-access.md](docs/architecture/machine-access.md) | Camera, screen and input policy |
| [docs/connectors.md](docs/connectors.md) | Ollama/Qwen default and opt-in hosted brains |
| [docs/platform-support.md](docs/platform-support.md) | What is verified on which OS |
| [docs/gate-runbook.md](docs/gate-runbook.md) | Post-training gate and install rule |
| [ROADMAP.md](ROADMAP.md) | Phased plan |
| [PROGRESS.md](PROGRESS.md) | Current state, measurements, blockers |
| [CHANGELOG.md](CHANGELOG.md) | Notable changes |

## Contributing

Issues and pull requests are welcome. Please read [CONTRIBUTING.md](CONTRIBUTING.md) and [SECURITY.md](SECURITY.md) first — changes that touch the permission layer, the audit log or the sandbox need tests that prove the refusal path still holds.

## License

[MIT](LICENSE). Bundled model and third-party data licences are listed in [docs/model-and-data-licenses.md](docs/model-and-data-licenses.md).
