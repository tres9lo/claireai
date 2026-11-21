# ClaireAI

**A personal assistant that runs entirely on your own machine.** Local model inference, an append-only audit trail of every action, and file and browser tools that cannot act without your explicit approval.

No cloud provider. No API keys. Nothing leaves the laptop unless you allow it.


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

1. **A small local model I train myself.** The default brain is a from-scratch transformer trained on this machine, in `brain-lab/`. It is small on purpose: it fits in RAM on a laptop with no GPU, it runs with the network cable unplugged, and nothing it reads is sent anywhere. Today it holds short conversations, does simple arithmetic, routes intents, extracts slots, emits tool-call JSON and refuses the unsafe requests covered in this repository. It is not a general-purpose assistant, and this README will not pretend otherwise.
2. **Connectors, for when I want more brain than my laptop can hold.** The `Brain` interface is deliberately provider-neutral, and it now has three adapters: Claude, ChatGPT and a larger model served by Ollama. **They are opt-in and off by default.** The local checkpoint answers unless you set one, the audit row names the provider that answered, and the validation, folder allow-list, approval flow and audit log do not change, because the model was never the thing enforcing them. Setup and the honest limits are in [docs/connectors.md](docs/connectors.md).
3. **An assistant I do not have to worry about.** Deny-by-default folders, no permanent deletes, approvals a model cannot fake, an audit log the database will not let you edit, and inference that never leaves the machine. That is the property worth having, and it is where most of the effort in this project goes.

### On the model work, honestly

I am not a model-training expert. I am learning this in public: every training run in this repository is measured, and the runs that did not work are documented in [PROGRESS.md](PROGRESS.md) rather than quietly deleted. If you know more about training small transformers than I do — data mixing, curriculum, tokenizer design, evaluation that is more than loss going down — I would genuinely like your pointers. Open an issue with what you would change, or a pull request that makes the numbers better; both are the fastest way to move this project forward.

## What works today

| Capability | Status |
| --- | --- |
| From-scratch byte tokenizer and decoder-only transformer research model | ✅ foundation |
| SQLite storage with a versioned, checksummed migration runner | ✅ |
| Append-only audit log (`task_log`), enforced by database triggers | ✅ |
| Provider-neutral `Brain` interface + local scratch-checkpoint adapter | ✅ foundation |
| Opt-in Claude, ChatGPT and Ollama connectors (local model stays the default) | ✅ scratch remains default |
| CLI chat that records every turn as a task plus audit entries | ✅ |
| Read-only tools (files, calendar, day plan, image identification; browser tools when enabled) | ✅ |
| 9 approval-gated writes (create, write, rename, move, trash, download, add appointment, plan day) | ✅ |
| Interactive approval flow with per-decision audit references | ✅ |
| Agent loop with schema validation, risk policy and step cap | ✅ |
| Strict ToolACE/Hermes name mapping to real registry schemas | ✅ code path; full mapped sample pending |
| Tokenizer-backed corpus audit (budgets, calls, duplicates, held-out overlap) | ✅ |
| One-command post-training gate (model, conversation, standard + extended tools) | ✅ |
| Deterministic tool routing (roughly halves prompt tokens per step) | ✅ |
| Daily summary (Done / Failed / Pending) from the task log | ✅ |
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
| OS family detection (Linux/Windows first targets) | ✅ detection only |
| Reproducible MIT synthetic task corpus + split-aware text training | ✅ starter data |
| Random-init visual-grounding prototype using attributed online synthetic images | 🧪 research only; not integrated |
| Capable scratch-trained assistant; speech recognition and custom trained voice | 🚧 not included yet |
| Playwright engine · Tauri desktop shell | 🚧 planned |
| Camera, screen, cursor and keyboard control | 🚧 specified, not built |

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
- **Task and audit logs redact secrets.** Full user and assistant messages are retained verbatim in the local conversation database so ClaireAI can show history and continue chats. The owner can review or delete conversations from the UI; the database is owner-readable only on POSIX systems.
- **Inference is local by default.** ClaireAI loads its own checkpoint and, out of the box, calls no hosted model API. A connector is an explicit setting the owner writes (`CLAIREAI_BRAIN_PROVIDER`), it is opt-in per provider, and the audit row for the turn records which one answered. A remote model can only *propose* a tool call — approvals and the sandbox still decide what runs.
- **Capabilities are opt-in.** Browser access starts disabled. The desktop Settings page can enable it; browser actions still use the existing approval path. Stop ends model generation between tokens, lets a current bounded tool action finish, prevents subsequent model/tool calls, and exits the interactive CLI even while it is waiting for input. The window remains open; after allowing the runtime again, start a new turn.

Read [docs/architecture/threat-model.md](docs/architecture/threat-model.md) for the full boundary analysis.

## Quick start (Linux)

Requires **Python 3.11+**. The scratch model starts with random weights; it is a research prototype and will not behave like a capable assistant until trained and evaluated on substantial task-specific data.

```bash
git clone https://github.com/tres9lo/claireai.git
cd claireai

# 1. Install. This detects the OS, architecture, Python, Tk and display, prints
#    every download with an approximate size, and downloads nothing until you
#    answer yes. --dry-run shows the plan; --no-speech-model skips the voice
#    model. Re-running later only fills what is missing.
./install.sh

#    Manual equivalent, if you prefer to run each step:
#    python3 -m venv assistant/.venv
#    assistant/.venv/bin/python -m pip install -e "assistant[dev,desktop]"
#    assistant/.venv/bin/claireai --setup --install
```

The installer picks a Python whose Tk can draw antialiased fonts. Do not
`pip install` the app into a conda base on Linux: conda's Tk may be built for
the old X11 bitmap font system (`fontsystem x11`), which makes the whole window
look pixelated. `./install.sh` checks this and can use the system Python
instead; the applications-menu entry it writes runs that environment explicitly,
so it does not depend on which `claireai` is first on `PATH`.

```bash
# continue here for training; the app itself is already installed

# Speech output. Piper is a neural voice and sounds like a person; eSpeak NG is
# the fallback and always works. Neither one sends audio anywhere.
uv pip install --python assistant/.venv/bin/python piper-tts
sudo apt install pulseaudio-utils            # for paplay; espeak-ng as fallback

# Voice models (~61 MB each). Download the ones you want into:
#   ./runtime-data/models/voice/piper/  (or set CLAIREAI_PIPER_DIR)
#   https://huggingface.co/rhasspy/piper-voices  →  en/en_US/lessac/medium,
#   fr/fr_FR/siwis/medium. Each voice needs its .onnx and .onnx.json side by side.

# 2. Prepare the licensed bilingual corpus, train a candidate and evaluate it.
#    This one command fetches bounded Dolly/OASST1/GSM8K/FineWeb plus ToolACE and
#    Hermes tool-call samples, builds the authored English/French curriculum,
#    merges and trains. It never installs the candidate; promotion stays an
#    explicit owner decision after the gate.
export CLAIREAI_DATA_DIR=./runtime-data
assistant/.venv/bin/python brain-lab/train_claireai.py --profile full
# Smaller smoke run on this laptop:
# assistant/.venv/bin/python brain-lab/train_claireai.py --profile quick
# Start the app: chat, history, preferences, approvals, controls. One command.
./scripts/start.sh

# 3. Ask one question; the turn is logged to SQLite
CLAIREAI_DATA_DIR=./runtime-data \
  assistant/.venv/bin/python -m claireai.cli --once "Hello"

# 4. Let the model use read-only file tools (allow-list required)
CLAIREAI_DATA_DIR=./runtime-data \
CLAIREAI_ALLOWED_FOLDERS='["~/Downloads"]' \
  assistant/.venv/bin/python -m claireai.cli --once "List the biggest files in my Downloads folder"
```

Windows setup uses the same Python and CPU training path; see [docs/setup/windows.md](docs/setup/windows.md).

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
printing a traceback: the `desktop` extra (`pip install "claireai[desktop]"`)
or Python's Tk bindings (`python3-tk` on Debian/Ubuntu). It exits 2.

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

### Background learning and training

**Background learning** in Settings is opt-in and does two separate jobs. It
journals exchanges locally and compiles them into training material, and at most
once a day — only when the network check succeeds — one bounded CPU training run
turns that material plus an authored practice curriculum into a **candidate** in
`models/candidates/`. The curriculum covers conversation, guest service,
reports, accounting, arithmetic and unit calculations, small coding tasks,
routines, email and social-media drafting, research, and business margins and
sales. Every candidate is then scored against the installed checkpoint on the
same fixed tool and conversation suites. Only a strict tool-check win with no
conversation regression is installed; the previous checkpoint pair is archived
under `models/archive/`, and a rejected candidate is kept under
`models/candidates/`. The capability suite is recorded as a diagnostic.

The installed scratch checkpoint has a **512-token architecture** expanded from
the proven 288-token weights. Only the first 288 position rows were trained;
inference now fits prompts to that learned range instead of feeding untrained
position rows, which was producing word salad on ordinary greetings. A future
candidate trained on the full 512 positions can raise the limit. The nightly
worker continues from the installed checkpoint instead of creating an unrelated
small model.

The model code now also supports a **10,240-token RoPE architecture** with
memory-efficient causal attention. A converted candidate was measured at
**9,734 prompt tokens in 18.7 s and 461 MB RSS** on this 4-thread i5 with 7 GB
RAM; it still failed the long-context recall probe, so its larger window is
capacity, not demonstrated usefulness. The candidate was not installed. Old
learned-position checkpoints keep loading unchanged, and only the gate can
promote a new one.

Background learning no longer teaches the model its own unverified answers. An
ordinary exchange is journaled for the owner, but only an owner-confirmed,
source-bearing research lesson is compiled into the next corpus. If the model
says it does not know a public factual question, the runtime can offer a
`research_web` approval, search, read the top public result, and ask a second
time before saving that sourced excerpt as a lesson. Private and first-person
requests are never sent to the web automatically. The authored curriculum also
contains multi-turn clarification and intent examples.

The worker also runs without the window through a **systemd user timer**:

```bash
claireai --install-training-service    # daily, no root, no display
claireai --training-service-status     # active / next run
claireai --uninstall-training-service
```

The timer respects the same **Background learning** switch and emergency stop. It
runs at most once a day, retries a candidate whose review was unavailable, and
never starts a second training run just because a review failed. Training is
local, it does not scrape, and it does not log into or post to any email or
social account: those need separate connectors, credentials and explicit
approvals. The official routes and their requirements are in
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
The daily worker trains on `brain-lab/data/chat-v15` when present, then falls
back to the older `chat-v13-small` licensed mix. The one-command
`brain-lab/train_claireai.py --profile full` builds the larger bilingual
`chat-v15` corpus instead, with English/French OASST1 and French FineWeb-2.
`fetch_corpora.py` and `merge_corpora.py` still rebuild the small corpus; the
merge removes exact duplicates and train/validation overlap and records the
counts.

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

`--day-plan-run` is safe by default: it pauses at the first approval and leaves
that item `awaiting_approval` instead of granting itself a write. Add
`--day-plan-date tomorrow` when assigning or running tomorrow's plan.

In the native window, **Day plan** lists a local day, adds timed requests, and
runs due work in the background. It uses the same approval boundary as the CLI:
a write is paused and shown as `awaiting_approval`, never inherited from the
plan.

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
uv pip install --python assistant/.venv/bin/python -e "brain-lab[vision]"
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
cd assistant && python -m pytest        # unit + security tests
assistant/.venv/bin/python scripts/smoke.py --skip-model   # end-to-end, no model
assistant/.venv/bin/python scripts/smoke.py                # full run, needs a trained checkpoint
```

CI runs the portable unit suite on Ubuntu and Windows across Python 3.11 and 3.12. The 14-check smoke script verifies the real loop: migrations, a chat turn, tool execution, a refused write, the presence stream, the daily summary, the read-only page, retention, and that the audit log rejects mutation.

Model behaviour is measured, not assumed: `scripts/eval_model.py` scores the local model against fixed cases before features depend on it, `conversation_eval.py` scores multi-turn answers, and `tool_call_eval.py` scores whether the configured brain proposes tool calls the runtime can actually read. The gate wrapper runs all four suites against one checkpoint:

```bash
assistant/.venv/bin/python scripts/run_gate.py \
  --checkpoint runtime-data/models/candidates/claire-chat-v15-full.pt
```

A corpus can be checked without loading the model:

```bash
assistant/.venv/bin/python brain-lab/audit_corpus.py \
  --corpus brain-lab/data/chat-v16 --strict
```

Every gate stage must pass before a checkpoint is eligible for installation; a
rejected candidate remains a candidate.

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
| [docs/connectors.md](docs/connectors.md) | Opt-in Claude, ChatGPT and Ollama brains |
| [docs/platform-support.md](docs/platform-support.md) | What is verified on which OS |
| [docs/gate-runbook.md](docs/gate-runbook.md) | Post-training gate and install rule |
| [ROADMAP.md](ROADMAP.md) | Phased plan |
| [PROGRESS.md](PROGRESS.md) | Current state, measurements, blockers |
| [CHANGELOG.md](CHANGELOG.md) | Notable changes |

## Contributing

Issues and pull requests are welcome. Please read [CONTRIBUTING.md](CONTRIBUTING.md) and [SECURITY.md](SECURITY.md) first — changes that touch the permission layer, the audit log or the sandbox need tests that prove the refusal path still holds.

## License

[MIT](LICENSE). Bundled model and third-party data licences are listed in [docs/model-and-data-licenses.md](docs/model-and-data-licenses.md).
