# TresorAI / ClaireAI Master Roadmap (12 Months) — Updated Execution Edition (2026-10)

## Owner direction (2026-11-03; supersedes older runtime/model choices)

ClaireAI must use a model trained from random initialization in this repository.
Ollama and downloaded pretrained model weights are removed from the active
runtime path. The target product includes a clear assistant stop control,
owner-controlled browser/device access, OS-aware adapters, a desktop UI, and
local speech with configurable voice characteristics. These are staged goals;
they must not be described as complete until implemented and evaluated. The
original phase tables below contain historical Ollama-era plans and are not
current acceptance criteria where they conflict with this direction.

The owner also requires an always-available wake-word interface, locally
learned and reviewable routines, voice identity checks, spoken consent, and a
fully configurable UI. These are separate acceptance requirements: detecting
a wake word or matching a voice must never silently grant a computer capability.

> **Audience:** Freebuff or another coding agent working with Tresor. **How to use this file:** Read the whole roadmap and the handoff section before writing code. Work on one small task at a time, run tests, show real output, update `PROGRESS.md`, and stop for human confirmation at each phase gate. Do not claim a check passed unless it was actually run.
>
> **Naming decision required at project start:** the uploaded filename says TresorAI, while the original document calls the assistant ClaireAI. This edition provisionally treats **TresorAI** as the repository/project name and **ClaireAI** as the assistant's in-product name. Ask Tresor to confirm before renaming directories, package IDs, application IDs, or visible branding.

## 0. Agent Operating Rules

1. Read this whole file before writing any code.
2. Work in small steps. After each step, run the code and show the output.
3. Prefer simple, readable code over clever code. Add comments that explain why, not what.
4. Never run destructive commands (delete, format, uninstall) without explicit human approval.
5. Keep the core model, the agent layer, and the clients in separate folders with clean interfaces.
6. Write a short `PROGRESS.md` entry at the end of every working session: what was done, what is broken, what is next.
7. Commit to git after every working step with a clear message.
8. If a step is unclear, ask the human instead of guessing.

## 1. Vision

Build **ClaireAI**: a personal assistant that lives on Tresor's PC and Android phone and works across both. It should:

- Understand typed and spoken commands, including hands-free activation through a configurable wake word such as **“Hey CLaire.”**
- Direct small AI agents that perform real tasks: files and folders, documents, email, browser tasks, downloads, installing and uninstalling software.
- Keep a log of everything it does.
- Write a **daily summary** each evening: what happened, what did not happen, what is still pending.
- Talk with a natural voice.
- Show a **live visual presence**: a local rendering layer that mirrors what the assistant is doing — idle, listening, thinking, running a tool, speaking, waiting for approval — and can draw an avatar/HUD scene, in the spirit of Jarvis-style assistants.
- **Perceive and operate the computer when a task needs it**: camera snapshots, screen capture, cursor and keyboard control, clipboard, windows and apps. All of it approval-gated with a visible indicator; none of it a background capability.
- Train ClaireAI's model from scratch (own tokenizer, own transformer, randomly initialized weights, own training pipeline) and improve it on computer-use tasks using licensed data and fixed evaluations. The first CPU model will be limited; keep safety decisions in the application runtime.

## 2. Strategy: Two Tracks, One Product

The project has two tracks that run in parallel and share a common `Brain` interface.

- **Track A: The Assistant (agent layer).** Keep the tool runtime, approvals, audit log, UI and cross-platform adapters behind the `Brain` protocol. The configured brain is now the locally trained checkpoint; inference must remain offline.
- **Track B: The Brain (from-scratch model).** The `brain-lab/` transformer and tokenizer are the source of ClaireAI's model. Training remains an independent, measurable subsystem, but its checkpoint is intended to serve Track A. No pretrained weights or hosted model API are part of the normal runtime.

The locally trained model is the default brain. Application code talks to a `Brain` interface so model architecture changes do not rewrite agents. Cloud model providers are out of scope.

## 3. Constraints and Assumptions

- Tresor's first computer is modest, so the first locally trained assistant model must be small, and CPU training must be feasible for experiments.
- PyTorch and NumPy are allowed as tensor and autograd tools. The transformer architecture and training loop are written by hand.
- Desktop must support **Linux and Windows** as first-class platforms. Tresor develops on Linux, so Linux setup and packaging must be tested throughout—not added at the end. Android is the mobile target (Tecno Spark 20).
- Budget is limited. Prefer free, local, and open-source tools. The default experience must not require API keys, subscriptions, paid inference, or pretrained model downloads. Inference should work offline.
- Internet is not always reliable. Design for offline-first where possible: queue tasks, retry later.
- Tresor already knows: TypeScript, React Native/Expo, Next.js, Node/Express, MongoDB, Tauri with Rust, SQLite, Python, PyTorch, NumPy. Reuse these skills instead of introducing new stacks without a reason.

## 4. Recommended Tech Stack

### 4.1 Track A: Assistant

| Layer | Choice | Why |
| --- | --- | --- |
| Desktop shell | **Tauri v2** (Rust + web UI) | Cross-platform desktop shell for Linux and Windows; test both targets continuously |
| Desktop UI | React + TypeScript + Tailwind | Matches the mobile skills |
| Agent runtime | **Python 3.11+** sidecar process with FastAPI | Best ecosystem for agents, speech and ML. Tauri launches it as a sidecar |
| Local model runtime | ClaireAI's own PyTorch transformer checkpoint | Starts from random initialization, runs locally on CPU, and needs no model service |
| Local memory | **SQLite** (tasks, logs, summaries, notes) | Zero setup, one file, works offline |
| Browser control | **Playwright** (Python) with a dedicated automation profile | Reliable, scriptable, can run visible or headless |
| File and system tasks | Python `pathlib`, `shutil`, `subprocess`, plus OS-specific adapters (`apt`/`dnf`/`pacman` where appropriate and `winget` on Windows) | Keep common logic portable; never assume one package manager or shell |
| Email | IMAP/SMTP via Python, or Gmail API with OAuth | Start read-only, add sending later with approval |
| Calendar | Google Calendar API | Matches the daily summary |
| Speech to text | **faster-whisper** with a small model, with a pluggable backend | Local speech recognition; CPU-capable, though speed depends on hardware |
| Text to speech | **Piper** (offline) | Local speech output; select and document an openly licensed voice |
| Wake word | **openWakeWord** (local detection) behind a `WakeWordEngine` interface | Open-source, on-device detection; design for Linux and Windows, with push-to-talk fallback |
| Phone client | **Expo / React Native** with TypeScript | Android client |\n| Sync between devices | Small **FastAPI or Node/Express** service with authenticated API, or LAN sync first | Keeps PC and phone in step; sync is optional for local desktop use |
| Scheduler | APScheduler inside the Python sidecar | Runs the nightly summary |
| Presence renderer | **three.js** + **@react-three/fiber** (WebGL2) inside the Tauri webview, with a canvas 2D fallback | Local, GPU-light, no CDN; the 2D fallback keeps an integrated-GPU laptop usable |
| Screen and camera | `mss` for screenshots, `opencv-python` for camera frames, `xdg-desktop-portal` on Wayland | Local only; Wayland requires a user-granted portal session |
| Cursor, keyboard, windows | `pyautogui`/`SendInput` on Windows and X11, portal + `ydotool`/libei on Wayland | Powerful and OS-dependent; must degrade honestly instead of pretending to work |
| UI state transport | local FastAPI server + Tauri events (SSE) | The renderer is a read-only consumer of state, never a second source of truth |

### 4.2 Track B: Brain

| Layer | Choice |
| --- | --- |
| Language | Python |
| Tensor library | PyTorch (CPU first), NumPy for learning exercises |
| Tokenizer | Corpus-trained byte-pair encoding written by hand |
| Model | Decoder-only transformer written by hand (embeddings, causal self-attention, MLP, layer norm, residuals) |
| Training | Own training loop: AdamW, learning-rate schedule, gradient clipping, checkpoints |
| Data | Public-domain and permissively licensed text, plus Tresor's own writing. Record the source and license of every dataset |
| Evaluation | Validation loss, perplexity, and fixed prompt samples saved at every checkpoint |
| Later | Instruction tuning data, quantization (int8/int4) for phone inference, GPU rental for bigger runs |

### 4.3 ClaireAI model and runtime

- Initialize all model weights randomly and train the tokenizer and decoder-only
  transformer from the tracked training code. No pretrained language-model
  weights or external inference runtime are part of the product.
- Keep architecture, tokenizer, data manifests, checkpoints and fixed
  evaluations versioned or reproducibly recorded. Raw data and checkpoints
  remain outside Git.
- The first CPU model is a learning baseline, not a capable general assistant.
  Evaluate it on held-out computer-use instructions, tool-call validity,
  factual reporting, and refusal behavior before enabling any capability.
- The runtime reads a local checkpoint path. All proposed actions remain subject
  to tool schemas, path allow-lists, capability gates, approvals and audit logs.
- Voice, wake-word and browser assets need separate licensing records. Web
  access is an explicit owner-controlled feature and requires internet; local
  inference does not.

### 4.4 Wake word and voice behavior

- **Wake phrase:** begin with a configurable phrase, with **“Hey Claire”** as the desired phrase. Do not claim it works until a matching detector model has been trained or obtained and tested. Generic pretrained wake-word models do not automatically recognize arbitrary custom phrases.
- **Wake-word engine:** use `openWakeWord` behind a small `WakeWordEngine` interface. Keep the implementation replaceable because microphone APIs, model availability, supported runtimes, and packaging differ by OS.
- **Build in stages:** (1) microphone permission and audio capture, (2) push-to-talk, (3) local speech-to-text, (4) local speech output, (5) wake-word detection, (6) custom “Hey Claire” model and false-activation testing.
- **Privacy:** microphone capture must be visible in the UI. Wake-word detection should process audio locally and use a rolling buffer; do not save or upload continuous audio. Start with the microphone disabled until the user explicitly enables hands-free mode. Provide a clear mute/disable control.
- **Reliability:** add a cooldown after activation, an activation sound or visual indicator, configurable sensitivity, and a push-to-talk fallback. Test false activations from TV/audio and missed activations across different microphone levels and accents.
- **Voice identity:** provide explicit local enrollment and deletion for the owner's voice profile. Keep enrollment samples/embeddings local and encrypted where the OS provides a usable key store. Evaluate false accepts, false rejects, replayed recordings and synthetic/replayed speech before enabling voice-gated actions. Voice matching is a signal, not proof by itself.
- **Spoken consent:** vocal approval must be tied to the currently pending approval reference and the exact tool, arguments and target. ClaireAI reads a concise action summary, issues a fresh challenge, requires a bounded-time spoken confirmation, verifies the enrolled speaker, and logs the result without storing raw audio. A wake phrase, generic "yes", transcript match, or voice match alone never approves. Keep an accessible on-screen approval path. Until anti-replay and speaker verification pass the security gate, voice may confirm only low-risk actions; dangerous actions still require the existing explicit approval path.
- **Wake word is not authentication:** wake detection only starts a listening turn; it never grants a capability or counts as consent.
- **Routines:** learn a routine only from an owner-selected, completed task after showing the exact ordered steps, folders, applications and inputs for review. Store the reviewed routine locally. Replay must use registered tools, revalidate current paths and OS support, ask again for each approval-gated step, observe step/time limits, and stop immediately on owner request. Never capture passwords, hidden fields, arbitrary shell input or unreviewed free-form action sequences. Routines can be edited, disabled and deleted from the UI.
- **Wake-word and voice security gate:** test false activations from TV/audio and missed activations across microphone levels and accents; also test replay and spoof attempts. No microphone audio, speaker profile or routine leaves the device unless the owner separately enables an explicitly described sync feature.

### 4.5 Cross-platform engineering rules

- Use `pathlib` and platform-neutral abstractions for filesystem paths. Do not hard-code drive letters, `/home/...`, shell commands, or package-manager names into shared code.
- Keep OS-specific behavior in adapters, such as `WindowsSystemAdapter` and `LinuxSystemAdapter`; shared agents call the common interface.
- Use `sys.platform` or a maintained platform abstraction to choose behavior. Avoid assuming PowerShell, Bash, `winget`, `apt`, or `systemd` is available.
- Test both Linux and Windows on every phase that touches process launching, file paths, browser automation, audio, permissions, packaging, or system tasks. Android is a separate client and must not be assumed to run the desktop agent.
- CI should run portable unit tests on Linux and Windows. Where microphone devices or desktop integration cannot run in CI, document and perform manual acceptance tests on each OS.
- Do not promise identical system-control features on both operating systems. If a feature cannot be implemented safely or reliably on one OS, report it as unsupported instead of silently falling back to a risky command.
- The presence renderer is optional decoration over real state: if it is disabled, slow, or crashes, every task must still work. Never let rendering failure change task behaviour.
- Machine access (camera, screen, cursor, keyboard) differs per display server. On **Wayland** global input and screen capture go through the desktop portal and may be refused; on X11 they use XTest; on Windows they use `SendInput`. Detect and report, never assume.
- Accessibility is a requirement, not a later polish: respect the OS "reduce motion" setting, avoid flashing above 3 Hz, and keep every indicator readable without relying on colour alone.

## 5. Repository Layout

Create the repository root as `tresorai/`. Do not create every placeholder file immediately; create folders as the first task that uses them. Keep generated files, model weights, private databases, browser profiles, logs, and secrets out of Git.

```text
tresorai/
├── README.md
├── ROADMAP.md                         # this file, renamed by explicit approval
├── PROGRESS.md
├── CHANGELOG.md
├── CONTRIBUTING.md
├── SECURITY.md
├── PRIVACY.md
├── .gitignore
├── .editorconfig
├── .env.example                       # examples only; never real secrets
├── .pre-commit-config.yaml            # optional quality hooks
├── docs/
│   ├── architecture/
│   │   ├── overview.md
│   │   ├── decisions/                 # short ADRs for consequential choices
│   │   └── threat-model.md
│   ├── setup/
│   │   ├── linux.md
│   │   ├── windows.md
│   │   └── android.md
│   ├── api/
│   │   └── openapi-notes.md
│   ├── platform-support.md
│   ├── testing.md
│   ├── troubleshooting.md
│   └── model-and-data-licenses.md
├── assistant/                         # Track A: Python application/runtime
│   ├── pyproject.toml                 # dependencies and tool configuration
│   ├── requirements.txt               # only if the chosen workflow needs it
│   ├── requirements-dev.txt
│   ├── src/
│   │   └── claireai/
│   │       ├── __init__.py
│   │       ├── main.py                # app lifecycle / composition root
│   │       ├── config/                # validated settings, paths, defaults
│   │       ├── api/                   # FastAPI routes, schemas, dependencies
│   │       ├── brain/                 # provider-neutral interface/adapters
│   │       ├── agent_runtime/         # plan/execute/observe loop and limits
│   │       ├── tool_registry/         # schemas, validation, registration
│   │       ├── agents/                # file, browser, system, email, summary
│   │       ├── tools/                 # small deterministic operations
│   │       ├── permissions/           # policy, approvals, kill switch
│   │       ├── storage/               # SQLite connection and repositories
│   │       ├── migrations/            # versioned database migrations
│   │       ├── memory/                # notes and retrieval
│   │       ├── audit/                 # append-only event logging
│   │       ├── jobs/                  # queue, retries, scheduler
│   │       ├── sync/                  # device pairing and sync contracts
│   │       ├── integrations/          # Gmail/Calendar, optional adapters
│   │       ├── voice/                 # STT, TTS, audio abstraction
│   │       ├── wakeword/              # detector interface and tests
│   │       └── platform/              # shared interfaces + OS adapters
│   └── tests/
│       ├── unit/
│       ├── integration/
│       ├── security/
│       └── fixtures/
├── desktop/                           # Tauri v2 + React + TypeScript
│   ├── package.json
│   ├── src/
│   │   ├── app/
│   │   ├── components/
│   │   ├── features/                  # chat, tasks, approvals, settings
│   │   ├── lib/                       # API client, validation, utilities
│   │   └── styles/
│   ├── src-tauri/
│   │   ├── Cargo.toml
│   │   ├── tauri.conf.json
│   │   ├── capabilities/              # least-privilege frontend permissions
│   │   └── src/
│   └── tests/
├── mobile/                            # Expo / React Native / TypeScript
│   ├── package.json
│   ├── app.json
│   ├── app/                           # routes/screens
│   ├── src/
│   │   ├── components/
│   │   ├── features/                  # chat, approvals, logs, settings
│   │   ├── api/
│   │   ├── storage/
│   │   └── notifications/
│   └── tests/
├── brain-lab/                         # Track B: from-scratch research only
│   ├── pyproject.toml
│   ├── README.md
│   ├── configs/
│   ├── src/
│   │   └── brain_lab/
│   │       ├── tokenizer/
│   │       ├── data/
│   │       ├── model/
│   │       ├── training/
│   │       ├── inference/
│   │       └── evaluation/
│   ├── tests/
│   ├── experiments/                   # small configs and result metadata only
│   └── data/
│       ├── README.md                  # dataset instructions/licenses
│       ├── raw/                       # ignored by Git
│       ├── processed/                 # ignored by Git
│       └── manifests/                 # tracked source/license manifests
├── shared/
│   ├── api-contracts/                 # versioned JSON schemas / OpenAPI
│   └── examples/
├── scripts/
│   ├── setup_linux.sh
│   ├── setup_windows.ps1
│   ├── doctor.py                      # read-only prerequisite diagnostics
│   ├── dev_linux.sh
│   ├── dev_windows.ps1
│   └── backup.py
├── .github/
│   └── workflows/
│       ├── python-tests.yml
│       ├── desktop-checks.yml
│       └── mobile-checks.yml
└── runtime-data/                      # local-only; ignored by Git
    ├── db/
    ├── logs/
    ├── exports/
    ├── trash/
    ├── browser-profile/
    ├── downloads/
    └── models/                        # model weights are not committed
```

### 5.1 Repository rules

- Keep `brain-lab/` independent from the shipping assistant. It may implement a `Brain` adapter later, but must never be required for basic assistant startup.
- Keep Python package code inside `assistant/src/claireai/`; run it through the project environment, not by relying on the current working directory.
- `runtime-data/` is a development default only. Production data must use the OS-specific application-data directory, discovered through a maintained library or platform API.
- Do not commit databases, browser profiles, downloads, audio recordings, model weights, `.env` files, tokens, private datasets, or generated checkpoints.
- Keep shared API contracts versioned and validate them on both server and clients.
- The directory tree is a target architecture, not a requirement to create hundreds of empty files in the first commit.

---

## 6. Core Interfaces (build these early)

### 6.1 Brain interface

A provider-neutral interface with a main method such as `generate(messages, tools=None) -> response`. The active adapter loads ClaireAI's own local scratch checkpoint. The rest of the app must never import model implementation details.

### 6.2 Tool interface

Every tool is a plain function with: a name, a description, a JSON schema for its inputs, and a **risk level** (`safe`, `needs_approval`, `dangerous`). The agent runtime exposes the tool list to the brain.

### 6.3 Task log

Every action writes a row: timestamp, device, agent, tool, inputs, result, status (`done`, `failed`, `skipped`, `pending`), and approval info. The daily summary reads only from this log plus the calendar. Nothing else.

### 6.4 Presence renderer interface (visual state sync)

The renderer draws the assistant's state; it never decides anything. One contract, two consumers (the Tauri window and, later, the phone as a mirror):

```ts
// Server -> renderer. `seq` is monotonic per task stream; the renderer drops
// out-of-order or duplicate events instead of guessing.
type PresenceEvent = {
  seq: number;
  ts: string;              // ISO-8601 UTC
  taskId: string | null;
  state: "idle" | "listening" | "thinking" | "tool_running"
       | "awaiting_approval" | "speaking" | "error" | "offline";
  detail?: { tool?: string; label?: string; progress?: number };
  active: { microphone: boolean; camera: boolean; screen: boolean; input: boolean };
};
```

Rules:

- **Derived, not invented.** State comes from the same task/tool log the audit trail uses; if the log says `failed`, the renderer shows `error`, not a hopeful idle animation.
- **Read-only consumer.** The renderer has no database, filesystem, shell or model access, and makes no network requests at runtime. Assets ship with the app.
- **Indicators are mandatory.** Whenever `microphone`, `camera`, `screen` or `input` is true the UI shows it plainly: what is active and for which task.
- **Cheap by default.** Target 30 fps with a configurable cap, drop frames under load, ≤5% CPU and ≤10% of one core when idle, no per-frame allocations. WebGL scene with a canvas 2D fallback for integrated GPUs and remote sessions.
- **Failure is cosmetic.** A crashed or disabled renderer must never block, slow or alter a task.

### 6.5 Machine access interface (camera, screen, cursor, keyboard)

Machine access is a capability, not a tool with a hidden side door. Each capability declares: a name (`screen`, `camera`, `cursor`, `keyboard`, `clipboard`), the data it can read or the change it can make, a risk level, the platforms where it is supported, and the indicator text shown while it is active.

- Capabilities are **deny-by-default**: absent from `CLAIREAI_ALLOWED_CAPABILITIES`, the call is refused in code, exactly like a path outside the folder allow-list.
- A capability grant is **per task and expiring**, and names the exact device/display and the approval reference. A new task needs a new grant; nothing is granted for the whole session by accident.
- Read capabilities (`screen`, `camera`, `clipboard`) are `needs_approval` on first use per task; control capabilities (`cursor`, `keyboard`) are `dangerous` and ask every time until the owner marks a specific named pattern as trusted.
- Every call logs one task_log row: capability, target (display/device/window), what was read or changed, status, and approval reference. **Typed text is redacted** before it is stored; screenshot and camera frames are not persisted unless the task explicitly saves them with approval.

## 7. Safety and Permissions (non-negotiable)

1. **Risk levels:** `safe` actions (read files, list folders, open a web page) run freely. `needs_approval` actions (move, rename, create, send email, download) ask the human first until Tresor marks a pattern as trusted. `dangerous` actions (delete, install, uninstall, change system settings, spend money) always ask, every time.
2. **Sandbox folders:** file agents may only touch folders on an allow-list that Tresor configures.
3. **Dry run mode:** every file or system tool supports a dry run that prints what it *would* do.
4. **Recycle, don't delete:** deletions move files to a ClaireAI trash folder first.
5. **Browser profile:** the browser agent uses its own profile, not Tresor's personal logged-in browser, until specific logins are deliberately granted.
6. **Secrets:** API keys and passwords go in environment variables or the OS keychain, never in git.
7. **Kill switch:** a big visible button and a hotkey that stops all agents immediately.
8. **Audit trail:** the task log is append-only from the application's normal UI/API; sanitize secrets and private content.
9. **Prompt-injection boundary:** web pages, emails, files, model outputs, and tool results are untrusted input and cannot change permissions or system policy.
10. **Approval binding:** approval applies only to the displayed action and exact target/arguments; changes to the plan require a new approval.
11. **Recovery:** interrupted actions must be reconciled before retrying; never blindly replay non-idempotent or dangerous operations.
12. **Network exposure:** the local UI binds to loopback by default. LAN access requires explicit pairing/authentication and a documented threat review.
13. **Camera and screen are microphone-class permissions:** visible indicator whenever active, an explicit grant per task, never continuous capture by default, frames processed locally and never uploaded, and audio is never captured together with video unless the owner asks for both.
14. **Cursor and keyboard control are dangerous:** per-action approval until a named pattern is trusted, a visible indicator, a kill-switch hotkey that releases input immediately, a step cap, and a refusal to type into password fields or send credentials.
15. **Perception output is untrusted input:** text read from a screenshot, a camera frame, a clipboard or a window title cannot change permissions, policy or the tool set. The prompt-injection boundary (item 9) applies to pixels and OCR too.

## 7A. Architecture Contracts and Missing Implementation Requirements

### 7A.1 Process boundaries

- The desktop UI must not directly execute arbitrary shell commands or filesystem operations. It talks to the Python service through a narrow, authenticated local API or a narrowly scoped Tauri command.
- The Python service owns agent orchestration, tool validation, permissions, the task queue, database access, and audit records.
- The scratch model loads from a local checkpoint in the assistant process. Verify the checkpoint and PyTorch availability at startup and show a useful status if unavailable.
- The Android app is a client, not a remote shell. It sends typed, schema-validated requests to a paired desktop service and displays task state. Phone-local features use Android APIs only where the OS permits them.
- Track B's tokenizer/model/training code belongs in `brain-lab/`. Do not import experimental training modules into the production runtime.

### 7A.2 API and task lifecycle

Define and document versioned contracts before wiring the clients. At minimum, the API should support:

- health/readiness and app/runtime status;
- create task, get task, cancel task where safe, and list recent tasks;
- list pending approvals, approve, deny, and expire an approval;
- list audit events and retrieve a daily summary;
- read/update non-secret settings;
- pair/revoke a mobile device and sync queued client requests;
- stream task progress using Server-Sent Events or WebSockets only after the basic request/response API is stable.

Every task needs a stable ID, creation time, requesting device, status, current step, and timestamps. Use an explicit state machine such as `queued → planning → awaiting_approval → running → succeeded|failed|cancelled`; reject invalid transitions. A denied or expired approval must never execute later by accident.

### 7A.3 Agent loop limits

- Validate every model-proposed tool name and every argument against a strict schema. Reject unknown fields unless a schema explicitly permits them.
- Set maximum steps, wall-clock duration, output size, retries, and per-tool timeouts. Defaults must be conservative and configurable.
- Treat model output, website text, email content, downloaded files, and tool results as untrusted data. Instructions found in them must not override system policy or tool permissions.
- Separate planning from execution. For consequential actions, show a human-readable plan and exact targets before asking for approval.
- Re-check permissions immediately before execution, not only when the model proposes a tool call.
- Never let the model decide that its own approval is granted. Approval must come from a trusted UI/API action tied to the exact action payload.

### 7A.4 Database and audit

Use SQLite migrations from the first schema rather than relying on ad hoc schema creation. Enable foreign keys, transactions, and an appropriate busy timeout. Use WAL only after testing its behavior for the target storage location and backup plan.

Suggested entities: `tasks`, `task_steps`, `task_log`, `approvals`, `devices`, `notes`, `summaries`, `settings`, `schema_migrations`, and `sync_outbox`. Keep credentials out of ordinary settings and audit records.

Audit events should include event ID, UTC timestamp, device ID, task ID, actor, tool/action, target summary, status, approval reference, and sanitized result/error. Do not store raw secrets, full email bodies, or audio by default. Append-only means the app does not offer an edit/delete operation for audit events; it does not by itself prove tamper resistance against a machine administrator.

### 7A.5 Sync and pairing security

- Default to loopback-only desktop API access. LAN sync is opt-in and must show the network exposure clearly.
- Pair devices using a short-lived, one-time pairing code or QR token displayed by the desktop. Do not use a permanent shared secret embedded in the mobile app.
- Issue a revocable per-device credential; store it in Android Keystore-backed secure storage via an appropriate library such as Expo SecureStore. Never place tokens in logs or source control.
- Require authenticated requests, request-size limits, rate limits for pairing/auth endpoints, expiry, revocation, and replay protection for one-time pairing.
- Use TLS for remote access. Do not expose the Python service directly to the public internet. Remote access is out of scope until reviewed separately.
- LAN discovery is optional. If implemented, pairing and authentication remain mandatory; discovery itself is not authorization.
- Queue only idempotent or uniquely identified requests while offline. Before retrying a non-idempotent action, query task status to avoid duplicate execution.

### 7A.6 File safety

- Canonicalize and resolve candidate paths, then verify the final resolved path remains inside an approved root. Reject path traversal, symlink escapes, unsupported device paths, and ambiguous paths.
- Start with read-only tools. Write, move, rename, download, and trash operations require approval until explicitly configured otherwise.
- A dry run must use the same validation and planning path as real execution but perform no mutation.
- Use a dedicated application trash directory with metadata to support restoration. Do not promise safe restoration until it has been tested on each supported filesystem.
- Use atomic writes where possible: write a temporary file in the target directory, flush, then replace. Avoid overwriting an existing file without explicit confirmation.
- Place strict limits on file size, recursion depth, number of files, archive extraction, and time spent scanning.
- Do not open or execute a downloaded file automatically. Downloads go into quarantine/staging first and are logged.

### 7A.7 Browser and external content safety

- Keep the Playwright profile separate from the user's daily browser profile.
- Browser content is untrusted and can contain prompt-injection attempts. Web pages may provide information, but cannot grant permissions or override policy.
- Restrict navigation to user-approved destinations when practical; block local/private network destinations and file URLs by default to reduce SSRF and local-resource access risks.
- Ask for approval before submitting forms, sending messages, purchasing, posting, changing account settings, or downloading executable files.
- Provide a visible indicator when browser automation is active and a stop action that interrupts the current task.
- Do not store passwords in prompts, task logs, screenshots, or ordinary configuration.

### 7A.8 Email, calendar, and privacy

- Begin with read-only OAuth scopes or IMAP access. Request the narrowest scopes that support the feature.
- Draft replies without sending them. Sending, forwarding, deleting, or changing mailbox state requires a clear approval showing recipients and content.
- Calendar summaries should distinguish confirmed events from inferred status. Do not claim an event was attended or missed unless reliable data supports it.
- Use OAuth where available; document token refresh, revocation, and data retention. Keep tokens in the OS credential store or another protected store.
- Summaries should use only data sources enabled by the user. Provide controls to disable integrations and delete application data, with clear consequences.
- Do not record continuous microphone audio. Default to microphone off, use visible status, and avoid saving audio unless a specific feature requires it and the user explicitly enables it.

### 7A.9 Reliability and observability

- Use structured logs with severity, correlation/task IDs, and secret redaction. Provide actionable errors without dumping credentials or full private payloads.
- Add health/readiness checks for the API, database, and model runtime. Distinguish “application started” from “model ready.”
- Use bounded retries with exponential backoff and jitter for transient network errors; do not retry permission denials or invalid requests automatically.
- Store task queue state durably. On restart, recover queued tasks and mark interrupted in-flight tasks as `interrupted` until they are reconciled; never blindly replay a potentially destructive operation.
- Back up the SQLite database using a consistent SQLite backup mechanism. Document restore steps and test a restore before claiming backups work.
- Include a diagnostic command that reports versions and configuration status without printing secrets.

### 7A.10 Configuration and dependency management

- Validate configuration at startup and fail with a clear message for invalid settings.
- Provide `.env.example` only for non-secret examples. Actual secrets belong in OS credential storage or a local ignored secrets file with restrictive permissions.
- Pin or lock dependencies for reproducible installs. Update dependencies deliberately and run tests after each update.
- Record tested versions of Python, PyTorch, Node, Rust, Tauri, and Expo SDK in setup docs; verify current stable versions against official documentation when implementing.
- Avoid claiming that “latest” is a permanent version. Use supported release lines and a deliberate upgrade process.
- Add a dependency/license inventory before packaging. Check the license of each model, voice, wake-word asset, dataset, and bundled runtime.

### 7A.11 Presence renderer and machine access

- The renderer is a UI consumer: it receives `PresenceEvent` over the authenticated local API / Tauri events and holds no privileged capability. It must not be able to invoke tools directly.
- Camera, screen, cursor, keyboard and clipboard live in the Python sidecar behind the tool registry, the risk policy and the task log. The desktop UI never calls an OS API for them on its own.
- A capability grant is bound to one task id, one device/display, and one approval reference, and expires with the task. Interrupted tasks lose their grants.
- The renderer shows indicators from the same event stream that logs the action, so "what is on screen" and "what the audit log says" cannot disagree for long; any mismatch is a bug in the log, not something to hide in the UI.
- Wayland, X11 and Windows each get an explicit support status per capability with the measured result; a refused portal call is reported to the user, never retried in a loop.

## 7B. Updated Technology Decisions and Verification Rules

The original stack remains a reasonable baseline, with these qualifications:

- **Scratch model:** keep architecture and checkpoint formats versioned. Test training, held-out loss, response quality, tool-call validity, RAM use, and latency before raising capability limits. Never treat training loss as evidence of safe tool use.
- **Tauri v2:** use the official v2 documentation and least-privilege capabilities. Do not enable broad shell, filesystem, or updater permissions by default. Updater artifacts must be signed; protect the private signing key and never commit it. Reference: https://v2.tauri.app/.
- **Expo:** use the current stable Expo SDK and `npx expo install` for compatible native package versions. Test permissions and notifications on a real Android device. Background execution and always-on microphone behavior are OS-controlled and must not be promised. References: https://docs.expo.dev/ and https://docs.expo.dev/versions/latest/sdk/securestore/.
- **FastAPI:** use Pydantic request/response models, explicit authentication dependencies for non-loopback access, and generated OpenAPI contracts. Never assume CORS is authentication.
- **Wake word:** treat `openWakeWord` as a candidate backend, not proof that arbitrary “Hey Claire” detection is available. The custom phrase remains pending until a model is obtained or trained, license checked, and false-positive/false-negative tests pass.
- **Speech:** test faster-whisper model size and latency on the actual PC. Piper voice availability, packaging, and license must be checked per voice. Keep voice features optional so the app can still start if audio dependencies are unavailable.
- **Android model inference:** defer until the PC/client workflow is stable. A small model running on Android is an experiment; measure memory, latency, battery, heat, and supported device ABI before promising it.
- **Model training:** use small CPU runs to validate correctness, not to promise broad capability. Validation loss falling is necessary but insufficient; also test memorization, generation samples, repetition, data leakage, and fixed evaluation prompts.

### 12A. Presence renderer and machine access — honest expectations

- The renderer costs CPU/GPU and battery on a 4-thread laptop. Default to 2D, cap the frame rate, and measure before raising quality. A pretty avatar is never worth a slow assistant.
- Screen capture and the camera are the most sensitive features in the product. One accidentally stored or uploaded frame would break the privacy promise, so frames stay transient and the indicator is always visible.
- Cursor and keyboard automation can misclick into something destructive. Step caps, dry runs, per-action approval and the kill switch are mandatory; on Wayland the desktop portal may simply refuse, and that must be reported as unsupported rather than worked around.
- Jarvis-style expectations are easy to over-promise. The renderer is decoration over a small local model: it must never imply capabilities the assistant does not actually have.
- Lip-sync is approximate mouth-shape animation from audio, not phoneme-accurate visemes; document it as such.

---

## 8. Twelve-Month Roadmap

Each phase lists a goal, the steps, and **acceptance checks**. Time estimates assume part-time work of about 8 to 12 hours per week.

---

### Month 1: First Visible Progress (Phase 1)

**Goal:** By the end of month 1, Tresor can type or speak a command on his PC and watch the assistant do a real task, and receive a first daily summary.

**Week 1: Foundations**

- Create the repo and folder layout from section 5.
- Set up Python 3.11 virtual environment, install FastAPI, uvicorn, httpx, pydantic, APScheduler.
- Create the SQLite schema: `tasks`, `task_log`, `summaries`, `notes`.
- Implement the `Brain` interface and a local scratch-checkpoint adapter. Do not add hosted inference or downloaded pretrained weights.
- Build a command-line chat that talks to the brain.
- Install CPU PyTorch, train a tiny checkpoint from licensed text, then check the CLI loads it locally and logs a turn. Treat the first checkpoint as a wiring experiment, not a capable assistant.

**Week 2: First tools and agent loop**

- Write the tool registry (section 6.2).
- Implement safe file tools: `list_folder`, `read_text_file`, `search_files`, `get_file_info`.
- Implement an agent loop: brain proposes a tool call, runtime validates it, runs it, returns the result, loop until done or a step limit (default 8) is reached.
- Add the permission layer and the allow-list of folders.
- *Check:* "List the biggest files in my Downloads folder" works and every step is logged.

**Week 3: Doing things**

- Add approval-gated tools: `create_folder`, `move_file`, `rename_file`, `create_text_file`, `write_to_file`, `move_to_trash`.
- Add dry-run mode and a simple approval prompt in the CLI.
- Add a basic Playwright browser tool: `open_url`, `get_page_text`, `search_web`, `download_file` (approval required).
- Build the approval record as a **per-task grant** (action, target, reference, expiry) so the camera, screen and input capabilities in section 6.5 reuse it instead of inventing a second mechanism.
- *Check:* "Organize my Downloads folder into folders by file type" shows a plan, asks for approval, then executes and logs.

**Week 4: Daily summary and a face**

- Build `summary_agent`: reads the day's task log plus today's calendar events, produces a short report with three sections: **Done**, **Failed or skipped**, **Still pending**.
- Schedule it with APScheduler for a configurable time each evening, and save to the `summaries` table and a markdown file.
- Build a minimal Tauri window on both Linux and Windows: chat box, live task log, approve/deny buttons, kill switch, voice controls, wake-word status, and a "Today's summary" tab. Keep styling and behavior consistent, while respecting each OS's conventions.
- Build the **presence renderer v0** inside that window (section 6.4): a state HUD driven by `PresenceEvent` — idle, listening, thinking, tool_running, awaiting_approval, error — canvas 2D first, with a WebGL scene behind a flag. Include indicators for microphone, camera, screen and input control, honour the OS "reduce motion" setting, and allow switching the renderer off entirely.
- *Check:* every state the HUD shows matches the task log for the same task; with the renderer disabled the rest of the app behaves identically; the renderer stays inside its CPU/fps budget on this laptop. Then the daily-summary check: after a day of use, the summary correctly lists what was done and what was not. **This is the month-1 demo.**

**In parallel (Track B, light touch, about 2 hours per week):** read about tokenizers and attention, and write a toy character-level language model on a small text file using NumPy or PyTorch. No pressure to finish.

---

### Months 2 to 3: Make the Assistant Useful Daily (Phase 2)

**Goal:** Tresor uses the assistant every day for real tasks.

- Email agent: read inbox through IMAP or the Gmail API, summarize unread mail, draft replies. Sending always needs approval at first.
- Calendar integration in the daily summary (events attended, missed, upcoming).
- Notes and memory: the assistant saves facts and tasks Tresor tells it, in SQLite, and retrieves them with simple keyword and recency search first, embeddings later.
- Software agent: use an OS adapter for software installation and removal (`winget` on Windows and the supported package-manager flow on Linux). These operations always require dangerous-level approval and must be logged.
- Voice input: add faster-whisper with a push-to-talk hotkey first; this is the reliable fallback while wake-word detection is being built.
- Voice output: add Piper with a chosen, documented open-licensed voice. Never assume all voice packs share the same license.
- Improve reliability: retries, clearer error messages, a task queue so failed jobs can be retried later.
- **Machine perception** (new, approval-gated): `screen_capture`, `camera_snapshot`, `list_windows`, `read_clipboard`, each behind `CLAIREAI_ALLOWED_CAPABILITIES` and a per-task grant. Visible indicator while active, frames kept in memory unless the task saves them with approval, OCR/window text treated as untrusted input.
- **Machine control** (new, `dangerous`): `move_cursor`, `click`, `type_text`, `press_key`, `focus_window`, `launch_app`. Per-action approval, dry run, kill-switch hotkey, step cap, typed text redacted in the log. Wayland only through the portal where the environment allows it; otherwise report the capability as unsupported.
- **Presence sync:** `speaking` state and lip-sync driven by the Piper audio actually playing, plus the WebGL scene on machines that can hold it.
- *Acceptance:* ten different real tasks completed end to end across a week, with an accurate daily summary each night. At least one of those tasks needs the screen ("find the total in the open invoice and save it as text") and one needs the cursor, each with a visible indicator, a recorded approval, and log rows that match what happened on screen. Repeat the core workflow on Linux and Windows.

**Track B this period:** build the byte-level BPE tokenizer from scratch, train it on a small corpus, write tests (encode then decode returns the original text).

---

### Months 4 to 5: Phone Client and Sync (Phase 3)

**Goal:** the assistant exists on the phone and the two devices share one story.

- Build a small sync service (FastAPI or Node/Express) with token authentication. Start with LAN-only sync if hosting is a problem.
- Expo app with: chat, voice button, task log, daily summary, approvals inbox.
- Phone can send commands to the PC agent ("download this file on my PC") and receive results.
- Push notifications for approval requests and the daily summary.
- Phone-side tools through Android intents where possible: alarms, timers, calendar events, opening apps.
- Offline queue: commands typed offline are sent when the connection returns.
- *Acceptance:* send a task from the phone, approve it from the phone, see it execute on the PC, see it in the shared daily summary.

**Track B this period:** write the transformer by hand: embeddings, positional encoding, causal self-attention, MLP blocks, layer norm, residual connections. Verify shapes and run a tiny overfit test (the model must memorize one short sentence).

---

### Months 6 to 7: Train the First Real Tiny Model (Phase 4)

**Goal:** a from-scratch model that produces sensible text on CPU.

- Build the data pipeline: collect, clean, deduplicate, tokenize, and shard text. Record sources and licenses.
- Write the training loop: AdamW, warmup and cosine decay, gradient clipping, periodic evaluation, checkpoint saving and resuming.
- Train a model of roughly 1 to 10 million parameters on CPU. Log loss curves.
- Build inference code with sampling (temperature, top-k, top-p) and a KV cache.
- Save sample generations at each checkpoint so progress is visible.
- Presence renderer quality pass: WebGL scene with texture/mesh budgets, measured fps on integrated graphics, and the 2D fallback kept as the default on weak GPUs.
- *Acceptance:* validation loss steadily falls, and the model writes coherent short passages in the style of its training data. Be honest about quality. It will be small.

**Track A this period:** hardening, a settings screen, better approval rules, and packaging for non-developer installs (see section 9).

---

### Months 8 to 9: Make It Conversational (Phase 5)

**Goal:** ClaireAI can follow instructions and hold a simple conversation.

- Create an instruction dataset: write by hand and generate with a stronger model where licensing allows. Tag the source of every example.
- Add a chat format with special tokens for system, user, and assistant turns.
- Run supervised instruction tuning on the from-scratch model.
- Evaluate with a fixed test set of 50 prompts, scored by hand each time.
- Improve the **from-scratch ClaireAI model** with instruction data, held-out evaluation, and a settings-bound local checkpoint. Do not add cloud inference or pretrained weights as a fallback.
- *Acceptance:* ClaireAI handles simple, well-defined assistant tasks such as classifying a command, extracting a file name, or writing a short summary.

---

### Months 10 to 11: Tool Use and Bigger Training (Phase 6)

**Goal:** ClaireAI chooses tools and the model grows.

- Create tool-calling training examples from the real task log (with Tresor's approval, and with private content removed).
- Fine-tune ClaireAI on a strict tool-call format. Validate outputs against the tool schemas, and reject malformed calls.
- Rent GPU time for a larger run if the budget allows, with a clear spending cap. Consider 50 to 150 million parameters.
- Quantize the model (int8, then int4) and test inference speed on the Tecno Spark 20.
- *Acceptance:* ClaireAI completes a defined list of common tasks with its evaluated scratch checkpoint selected, and fails safely when unsure.

---

### Month 12: Ship and Review (Phase 7)

- Port inference to Android (for example via ONNX Runtime or a compiled C++ runtime) and measure battery and memory use.
- Final installers for Linux, Windows, and Android (section 9).
- Write user documentation, a troubleshooting guide, and a privacy statement.
- Run a one-month dogfood review: what the assistant did, what it got wrong, what to improve.
- Decide the next year: larger models, more agents, and possibly a product for East African users.
- *Acceptance:* Tresor uses ClaireAI as his daily assistant on both devices for 30 days.

## 8A. Initial Repository Setup: Exact Human/Agent Sequence

Follow this sequence before asking the coding agent to implement features. Commands below are templates; the agent must check the OS and official current instructions before running them. Do not run a script that silently installs system packages or changes machine-wide settings.

### Step 1 — Confirm project identity and machine baseline

1. Confirm whether the repo is named `tresorai` and the user-facing assistant is named `ClaireAI`.
2. Confirm the development OS and version, available RAM/disk, Python version, and CPU/PyTorch availability.
3. Confirm the target Android phone can be connected for testing.
4. Create a project directory in a location chosen by Tresor; do not assume a home path.

### Step 2 — Initialize Git and safety files

```bash
mkdir tresorai
cd tresorai
git init
```

Create `.gitignore` before installing tools or downloading assets. At minimum, ignore:

```gitignore
# Secrets and local configuration
.env
.env.*
!.env.example
*.pem
*.key
secrets/
credentials/

# Python
.venv/
__pycache__/
*.py[cod]
.pytest_cache/
.mypy_cache/
.ruff_cache/
.coverage
htmlcov/

# JavaScript / Rust build output
node_modules/
dist/
build/
target/
.expo/
.tauri/

# Local application data
runtime-data/
*.sqlite
*.sqlite3
*.db
logs/
downloads/
browser-profile/
trash/

# Model/data artifacts
*.gguf
*.safetensors
*.pt
*.pth
*.onnx
brain-lab/data/raw/
brain-lab/data/processed/
brain-lab/**/checkpoints/
```

Review the ignore rules before the first commit. Do not ignore source files, dataset license manifests, experiment configs, or small synthetic test fixtures.

### Step 3 — Create the minimal folders

Start with the root files and top-level folders from section 5. Create detailed subfolders only when implementing the corresponding phase. Add `README.md`, `PROGRESS.md`, `.gitignore`, and `docs/platform-support.md` first.

### Step 4 — Create the Python environment

Use Python 3.11+ unless a dependency's currently supported range requires a different tested version. From the repository root:

Linux:
```bash
python3 --version
python3 -m venv assistant/.venv
source assistant/.venv/bin/activate
python -m pip install --upgrade pip
```

Windows PowerShell:
```powershell
py -3 --version
py -3 -m venv assistant/.venv
.\assistant\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

If PowerShell blocks activation, do not weaken machine-wide execution policy automatically; use the documented safe activation alternative or invoke the environment's Python executable directly.

The agent should then create `assistant/pyproject.toml`, install only the initial dependencies needed for the first milestone, and record resolved versions. Do not install every future dependency at once.

### Step 5 — Create the local scratch-model path

1. Install the CPU PyTorch build in the project environment.
2. Add a licensed training corpus with a tracked source manifest; do not download or include pretrained model weights.
3. Train a tiny checkpoint from random initialization and save its architecture, tokenizer merges, data hashes, and training/validation losses.
4. Load that checkpoint through `ScratchBrain` and verify inference works offline.
5. Run held-out instruction and tool-call evaluations before relying on model output.

Record CPU/RAM use and actual output in `PROGRESS.md`. A smoke checkpoint proves wiring only; never describe it as a useful assistant.

### Step 6 — Build the database and CLI before the UI

Implement, in order:
1. configuration validation and platform-aware application-data paths;
2. SQLite connection and first migration;
3. task and audit-log repositories;
4. provider-neutral `Brain` protocol and local scratch-checkpoint adapter;
5. CLI chat with request IDs and sanitized logs;
6. read-only tools and allow-list tests;
7. approval service and dry-run mode;
8. FastAPI endpoints only after the core service functions can be tested without HTTP.

### Step 7 — Set up desktop and mobile clients

- Desktop: scaffold Tauri v2 with React + TypeScript; keep capabilities narrow. Do not enable arbitrary shell access for the UI.
- Mobile: scaffold Expo + TypeScript and test on the actual Android device. Store device credentials in secure storage, not AsyncStorage or source code.
- Do not connect mobile to the PC until authenticated pairing and API authorization tests pass.
- Keep the first mobile release a client for chat, task status, approvals, and summaries; add local Android actions one at a time.

### Step 8 — Configure testing and CI

Add Python unit tests and formatting/linting first. Add Linux CI early, then Windows CI for portable behavior. Desktop packaging and audio/microphone checks require OS-specific manual acceptance when CI cannot provide the required devices. Mobile builds should use a development build when native modules or permissions require it.

### Step 9 — First commit and handoff baseline

The initial commit should contain only the scaffold, project docs, ignore rules, dependency manifests, and passing minimal tests. It must not contain model weights, database files, real secrets, or generated user data.

### Step 10 — Required first-run report

Before proceeding, the agent must show:
- repository tree;
- OS and tool versions;
- dependency install output;
- test output;
- training corpus manifest, checkpoint metadata, held-out evaluation and inference output;
- the created database schema/migration result;
- known blockers and the next single task.

---

## 9. Shipping and Setup

### 9.1 Developer machine requirements

- Linux (Ubuntu/Zorin OS or another supported distribution) and Windows 10/11; 8 GB RAM minimum for early development, 16 GB recommended; reserve at least 10–20 GB for runtime, training data, voice assets, and build artifacts. Actual training speed depends on RAM, CPU, and GPU.
- Python 3.11+, Node.js LTS, Rust toolchain (`rustup`), Git, and Tauri prerequisites for the chosen OS. On Linux install the documented system libraries and WebKit/GTK build dependencies; on Windows install the required Microsoft C++ build tools. Keep OS-specific setup scripts separate.
- Android: Android Studio or Expo with a physical phone for testing.
- Accounts: none required for local chat, wake word, speech recognition, or speech output. Google Cloud OAuth is optional for Gmail/Calendar integrations; optional sync hosting is not required for local desktop use.

### 9.2 First-time setup commands (agent should verify each one works)

Create both `scripts/setup_linux.sh` and `scripts/setup_windows.ps1` (plus a shared Python setup module where practical). Each script should check prerequisites, create the virtual environment, install Python requirements and CPU PyTorch, create the SQLite database, and print a checklist. Do not silently install system packages or make destructive changes. Explain how to add licensed training data and download optional speech/wake-word assets.

### 9.3 How the desktop app ships

- Tauri builds a Linux package appropriate to the target distribution (for example `.deb` and/or AppImage where supported) and a Windows installer (`.msi` or `.exe`). Verify the chosen formats in CI or on real test machines.
- The Python agent runtime is bundled as a **sidecar executable** (built with PyInstaller or Nuitka) that Tauri starts and stops with the app.
- Playwright's browser is installed on first run into the app's data folder, so the installer stays small.
- The local LLM, speech-recognition model, Piper voice, and wake-word model are downloaded on first run or provided as optional model packs. Show model licenses and disk requirements before downloading; do not bundle third-party weights without checking redistribution terms.
- Renderer assets (meshes, textures, shaders, avatar art) ship inside the app bundle and must be licence-checked like any model; no CDN or remote fetch at runtime, because the assistant must work offline.
- Auto-update: use Tauri's updater with signed releases.
- Start on login is optional and off by default.

### 9.4 How it can use the browser and do tasks

- The browser agent launches Chromium through **Playwright** using a separate automation profile stored in the app data folder.
- Pages are read through the DOM and accessibility tree, with screenshots as a fallback. Actions are clicks, typing, and navigation, each logged.
- Logging into a site is done by Tresor himself once, inside the automation profile, so passwords are never handled by the model.
- Downloads go to a ClaireAI downloads folder first, then the file agent moves them after approval.
- OS tasks use a small allow-list of commands. Software installation/removal goes through the OS-specific adapter (`winget` on Windows and a supported package-manager flow on Linux) with dangerous-level approval.
- Optionally add a **browser extension** later, so the assistant can act inside Tresor's real browser with explicit per-site permission.

### 9.5 How the phone app ships

- Build with Expo EAS to produce an Android APK or AAB.
- Distribute the APK directly to Tresor's phone first. Use the Play Store only when ready.
- On the same wifi as the laptop the phone ai must be able to sync with the laptop if the actions Tresor requests require the app to run on the laptop or other pc but it must also work in the mobile doing actions like opening files, emails, browsing , and many other things.
- Request Android permissions one at a time, at the moment each feature is first used, with a clear reason.
- Keep the phone app a thin client at first. Heavy work happens on the PC or the sync server.

## 10. Cross-Platform Release Gate

Before calling any phase complete, maintain a simple matrix in `docs/platform-support.md`:

| Capability | Linux | Windows | Android |
| --- | --- | --- | --- |
| Text chat with local model | Required | Required | Client/sync phase |
| Local model management | Required | Required | Optional later |
| File tools and allow-list | Required | Required | Limited to Android APIs |
| Browser automation | Required, if Playwright supports target browser | Required | Not the desktop browser agent |
| Presence renderer (visual state sync) | Required | Required | Optional mirror in the client |
| Camera, screen capture, cursor and keyboard | Required where the OS permits (Wayland portal may refuse; report it honestly) | Required | Limited to Android APIs and explicit permissions |
| Push-to-talk voice | Required | Required | Optional/mobile implementation |
| Wake-word detection | Required, after microphone testing | Required, after microphone testing | Separate mobile implementation; do not assume background listening is permitted |
| Daily summary and local log | Required | Required | View through client after sync |
| Installer/package | Required | Required | APK/AAB |

A feature is not “cross-platform” just because it compiles. Record tested OS versions, setup steps, limitations, and actual test results. Keep `scripts/setup_linux.sh` and `scripts/setup_windows.ps1` maintained as part of the product.

---

## 11. Working With Coding Agents

- Give the agent **one phase at a time**, not the whole file, because small models lose track of long plans.
- Ask it to restate the current step's goal and acceptance check before it starts coding.
- Keep each task small enough to finish in one session. If a step touches more than about five files, split it.
- Require the agent to run tests and show real output. "It should work" is not an acceptable result.
- Ask it to update `PROGRESS.md` and commit after every step.
- If the agent gets stuck in a loop on the same error three times, stop and ask the human.
- When the agent is unsure about a library API, it should check the official documentation instead of guessing.

## 12. Risks and Honest Expectations

- A from-scratch model trained on a modest computer will be small and limited. It will not match large pretrained or commercial assistants. Improve it through legal task-specific data, code-owned safeguards, and repeatable evaluation; do not overstate its capability.
- There is no pretrained-model fallback. If the scratch model is not capable enough for a task, the runtime must refuse or report its limit while deterministic code and approval gates remain authoritative.
- Voice cloning of a specific commercial voice is out of scope. Use an open-licensed voice or Tresor's own.
- Automation that touches email, files, and installs is powerful. Keep approvals on, keep the audit log, and widen trust slowly.
- Money and time are limited. If a phase slips, shorten the scope. Do not skip the acceptance checks.

## 13. Month-by-Month Summary

| Month | Track A: Assistant | Track B: Brain |
| --- | --- | --- |
| 1 | CLI, local scratch checkpoint, tools, approvals, daily summary, Linux/Windows desktop UI, presence renderer v0 | Tokenizer, transformer, CPU training baseline |
| 2 to 3 | Email, calendar, memory, local voice, wake-word prototype, OS-specific installs, camera/screen/cursor access, lip-sync | Byte-level BPE tokenizer |
| 4 to 5 | Phone app, sync, notifications | Hand-written transformer |
| 6 to 7 | Hardening, packaging, presence renderer quality (WebGL) | First real CPU training run |
| 8 to 9 | ClaireAI adapter and switch | Instruction tuning |
| 10 to 11 | Tool-use fine-tuning | Bigger run, quantization |
| 12 | Linux + Windows installers, Android client/inference experiment, review | Evaluation and next plan |

## 14. First Prompt to Give the Coding Agent

> Read the owner direction at the top of `ROADMAP.md` first. ClaireAI must use a from-scratch model trained in this repository; do not add Ollama, pretrained language-model weights, or hosted inference. Keep capability grants explicit, the stop control visible, tool use approval-gated, and model limits accurate.


## Appendix A. First Setup Smoke Test

Run these checks after installation. The coding agent must adapt commands to the current OS and report actual output rather than claiming success without running them.

1. Confirm Python, Node.js, Git, Rust, Tauri prerequisites, and PyTorch CPU availability.
2. Confirm the training corpus has a tracked license/source manifest.
3. Train and load a small scratch checkpoint, then record held-out metrics.
4. Run one CLI question and verify that the task log records it.
5. Disconnect the network and repeat local chat to confirm inference is genuinely local.
6. Confirm microphone capture is off by default; enable push-to-talk and test transcription.
7. Test wake-word detection only after the chosen model is available. If “Hey Claire” has no tested model yet, mark custom wake-word detection as pending—not complete.
8. Run unit tests on Linux and Windows.
9. Test allow-list enforcement, approval prompts, dry-run mode, and the kill switch.
10. Verify that no API key is required for local chat and that prompts/audio are not sent to a remote service.
11. Presence renderer: force each state (idle, thinking, tool_running, error) and confirm the HUD matches the task log; disable the renderer and confirm tasks behave identically.
12. Machine access: confirm camera, screen and input indicators appear only while the matching capability is granted and active, and that a refused Wayland portal grant is reported instead of retried.
13. Kill switch: while a cursor action is running, trigger the hotkey and confirm input stops immediately and the task is logged as cancelled.

## Appendix B. Model and Runtime References

Check official documentation at implementation time because supported model tags, install instructions, runtime behavior, and licenses can change:

- openWakeWord project and platform notes: https://github.com/dscripka/openWakeWord
- Tauri prerequisites and platform build instructions: https://v2.tauri.app/start/prerequisites/

Do not bundle or redistribute a model, wake-word model, voice, or dataset until its license and redistribution terms have been checked.


---

## 15. Definition of Done and Phase Gates

A phase is complete only when all applicable items are true:

- [ ] Acceptance checks were run on the actual supported environment and their output is shown.
- [ ] Automated tests pass; any skipped tests are listed with a reason.
- [ ] Failure paths were tested, not just the happy path.
- [ ] Security/permission tests cover the new tools and API endpoints.
- [ ] Renderer and machine-access changes show the correct indicator, can be disabled, and disabling them changes no task result.
- [ ] Linux and Windows behavior is tested for shared desktop/runtime changes; Android is tested for mobile changes.
- [ ] `docs/platform-support.md` records tested versions, results, and limitations.
- [ ] `PROGRESS.md` records changes, tests, blockers, and next task.
- [ ] Documentation and `.env.example` are updated when configuration or setup changes.
- [ ] No secrets, personal data, databases, model weights, or generated runtime files entered Git.
- [ ] A small, understandable commit exists.
- [ ] Tresor reviewed the result and explicitly approved moving to the next phase.

A compile-only result is not an acceptance test. A mocked test is not proof that a real integration works. Label mocked, simulated, manual, and real integration results separately.

## 16. First-Day Folder and Setup Checklist

Use this checklist when starting the project:

- [ ] Confirm project/release naming: repository `tresorai`, assistant `ClaireAI` (provisional until confirmed).
- [ ] Create the repository and initialize Git.
- [ ] Add `.gitignore`, `README.md`, `PROGRESS.md`, and `docs/platform-support.md`.
- [ ] Create the minimal top-level folder structure in section 5.
- [ ] Check Python, Git, Node.js, Rust, Tauri prerequisites, and PyTorch CPU availability without changing system configuration.
- [ ] Create the Python virtual environment.
- [ ] Create `assistant/pyproject.toml` and the first test.
- [ ] Implement the first SQLite migration.
- [ ] Implement and evaluate the `Brain` interface and local scratch-checkpoint adapter.
- [ ] Run one CLI prompt and verify the task/audit record.
- [ ] Disable network and repeat the local chat smoke test.
- [ ] Commit the scaffold and show the commit hash.
- [ ] Stop and ask Tresor to confirm before beginning Month 1, Week 2.

## 17. Freebuff Agent Handoff Prompt

Copy this prompt to Freebuff together with this updated Markdown file:

> You are the implementation agent for the TresorAI project. Read the entire attached `TresorAI_Master_Roadmap_Updated_2026-10.md` before changing any files. The roadmap is the source of truth, but do not assume that a library, model tag, permission, or feature is currently available until you verify it against official documentation and test it in this environment.
>
> First, report the project naming ambiguity: the repository/project is provisionally `TresorAI` and the assistant UI name is provisionally `ClaireAI`. Ask for confirmation before changing public branding or package IDs.
>
> Start only with the “Initial Repository Setup” and Month 1, Week 1. Inspect the existing environment and repository before creating anything. Do not overwrite existing work. Do not run destructive commands or install system-wide packages without explicit approval. Create only the minimal folder structure and files needed for this step.
>
> Work in small, reviewable changes. Before coding, state the goal and acceptance checks. After coding, run the tests and show the real output. Do not claim success for tests that were not run. Keep local data, model weights, browser profiles, credentials, and secrets out of Git. Update `PROGRESS.md`, document blockers, make a clear commit if Git is configured, and stop for human confirmation before moving to the next step.
>
> Local inference is the default; do not add a hosted model/API dependency as a shortcut. Keep the experimental from-scratch model isolated in `brain-lab/`. Enforce allow-lists, approval gates, dry-run support, audit logging, and conservative tool limits before adding actions that mutate files or the operating system.

## 18. Official References to Verify at Implementation Time

The roadmap is a project plan, not a substitute for checking current upstream documentation. Re-check exact versions, install commands, platform support, API behavior, and licenses when implementing:

- Tauri v2 documentation: https://v2.tauri.app/
- Tauri updater and signing: https://v2.tauri.app/plugin/updater/
- Expo documentation: https://docs.expo.dev/
- Expo SecureStore: https://docs.expo.dev/versions/latest/sdk/securestore/
- Expo notifications: https://docs.expo.dev/versions/latest/sdk/notifications/
- FastAPI documentation: https://fastapi.tiangolo.com/
- PyTorch documentation: https://pytorch.org/docs/stable/index.html
- Playwright Python: https://playwright.dev/python/
- openWakeWord project: https://github.com/dscripka/openWakeWord
- three.js: https://threejs.org/docs/ and React Three Fiber: https://r3f.docs.pmnd.rs/
- `mss` screen capture: https://python-mss.readthedocs.io/ · `opencv-python`: https://docs.opencv.org/
- `pyautogui`: https://pyautogui.readthedocs.io/ · `ydotool` for Wayland input: https://github.com/ReimuNotMoe/ydotool
- xdg-desktop-portal (Wayland screen/input grants): https://flatpak.github.io/xdg-desktop-portal/

Before distributing the app, verify license obligations for every third-party model, tokenizer, voice, wake-word model, dataset, dependency, and bundled runtime. Record the checked version, license, source URL, date checked, and whether redistribution is allowed in `docs/model-and-data-licenses.md`.
