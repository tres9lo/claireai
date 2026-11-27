# Architecture overview

What exists in the repository today, how the pieces fit, and where the trust
boundaries are. This describes the code, not the plan; `ROADMAP.md` holds the
plan and `PROGRESS.md` holds what is proven and what is not.

## One request, end to end

```
 you (terminal)
   │  "List the biggest files in my Downloads folder"
   ▼
 claireai.cli ──────────────────────────────────────────────┐
   │  builds Settings, opens SQLite, checks brain health     │
   ▼                                                        │
 claireai.agent.run_task                                     │
   │                                                        │
   │  tasks.create(kind="agent") ──► tasks table (queued→running)
   │  presence.emit("thinking")                              │
   ▼                                                        │
 loop (≤ CLAIREAI_AGENT_MAX_STEPS, default 8)                │
   │                                                        │
   ├─► brain.generate(messages, tools=schemas)               │
   │      └─ claireai.brain.ollama ──► Qwen3 1.7B (loopback)   │
   │                                                         │
   ├─ no tool calls → record answer, status=succeeded, return │
   │                                                         │
   └─ tool calls → for each call:                            │
         1. registry.get(name)        unknown → refused      │
         2. tool.validate(args)       pydantic → refused     │
         3. check_risk(risk)          > safe → needs approval│
         4. approvals.decide(req)     CLI prompt or deny     │
         5. tool.handler(context)     sandbox applied        │
         6. audit.record(tool.call)   append-only row         │
            └─ result text back into messages, next step ────┘
```

Every tool-using loop iteration re-sends the conversation and its selected tool
schemas to the model. That is the dominant cost on a CPU-only machine (see
"Performance" below), which is why schema size and tool count are treated as
product constraints rather than cosmetics. Conversation and explanation routes
send no schemas and a compact system prompt, so a short reply does not pay that
cost.

## Modules

| Module | Responsibility | Does not |
| --- | --- | --- |
| `config/settings.py` | Validated `CLAIREAI_*` configuration, loopback enforcement, path resolution | Read the network, create files |
| `storage/database.py` | SQLite connection, `BEGIN IMMEDIATE` transactions, checksummed migration runner | Know about tasks or tools |
| `storage/user_data.py` | Local conversation history, owner directives and editable UI preferences | Send stored content over the network or grant tools |
| `storage/tasks.py` | Task rows and the explicit status state machine | Run anything |
| `audit/audit_log.py` | Append-only `task_log` writes, redacted detail | Update or delete (triggers reject it) |
| `audit/redaction.py` | Pattern-based secret redaction, length truncation | Guarantee that no secret is ever stored |
| `brain/base.py` | `Brain` protocol, `Message`/`ToolCall`/`BrainResponse` wire-neutral shapes | Import any concrete runtime |
| `brain/ollama.py` | Local Qwen chat, tool-call parsing and code-model routing | Train the model, decide permissions, or run tools |
| `brain/scratch.py` | Archived checkpoint loading for `brain-lab` research | Become the default install path again |
| `permissions.py` | `Sandbox` (folder allow-list, symlink-aware resolution), risk policy, capability policy | Prompt the user |
| `tools/base.py` | `Tool` (name, description, risk, pydantic args, handler), registry, schema stripping | Execute side effects |
| `tools/files.py` | Five read-only file tools | Write anything |
| `tools/files_write.py` | Six mutating tools, all `needs_approval` | Delete permanently, bypass the sandbox |
| `tools/canvas.py` | Structured local HTML canvases with news cards, video embeds, images and links | Let the model author scripts or upload anything |
| `tools/apps.py` | Launch installed desktop entries, open approved URLs, and arrange windows | Accept arbitrary shell commands |
| `windows.py` | X11/XWayland EWMH window enumeration, move/resize and tiling | Touch a native-Wayland window that XWayland cannot see |
| `approval.py` | `ApprovalRequest`, per-task grants, `DenyAll` default | Decide on the owner's behalf |
| `agent.py` | The loop: validate, authorize, run, log, feed back | Trust model output |
| `routing.py` | Narrow the tool set per request (deterministic, no model call) | Remove the fallback to all tools |
| `memory.py` | Local Ollama/hash embeddings and SQLite cosine search | Upload text or vectors |
| `cua.py` | Optional pyautogui/mss screen and input driver | Decide its own permission; the agent does that |
| `wakeword.py` | Local speech (“Hey Claire”) and openWakeWord microphone listeners with phrase pre-roll | Download a model, save audio, or grant a capability |
| `api/` | Loopback FastAPI service with bearer-token auth, reusing the same agent/approval path | Approve a write with only an HTTP token |
| `presence.py` | `PresenceEvent` stream (idle, thinking, tool_running, …) | Hold any capability |
| `desktop/` | Native Tk/customtkinter window: Presence, Chat, Canvas, Activity, Approvals, History, Voice, Memory, Settings, About | Open a network listener or run a tool without approval |
| `preflight.py` | OS/Python/Tk/display/package checks, a sized download plan and consent before `pip` runs | Download anything without an explicit yes |
| `training.py` | Opt-in daily practice curriculum and one bounded local training run; writes a candidate only | Promote a candidate, log into accounts, or post |
| `gate.py` | Scores candidate and installed checkpoints on fixed suites; archives the old pair before a strict-gate promotion | Promote on a tie, a regression, or a failed/missing suite |
| `service.py` | Installs the daily training worker as a systemd user timer; no root or display | Enable training behind the owner's back |
| `calculations.py` | Exact arithmetic, percentages and unit conversions before the model is called | Guess at a calculation |
| `retention.py` | Plans and applies removal of old trash batches and summaries | Delete inside the audit log, or outside the data dir |
| `summary.py` | Daily report from `tasks` + `task_log`, saved to `summaries` and markdown | Invent an event |
| `scheduler.py` | APScheduler job that builds the summary each evening | Need a running app |
| `evaluation.py` | Fixed cases, deterministic scoring, report rendering | Grade creativity |
| `doctor.py` | Read-only environment/schema/model diagnostics | Write anything |
| `cli.py` | Argument parsing, interactive loop, approval prompt | Contain business logic |

`scripts/` holds entry points that use the package: `bootstrap.py`, `doctor.py`,
`eval_model.py`, `smoke.py`.

## Trust boundaries

1. **The model is untrusted.** It produces text and proposed calls. Everything it
   proposes is validated by the runtime: name in the registry, arguments against
   a pydantic model (unexpected keys rejected), risk policy, folder allow-list,
   approval where required. The model cannot widen its own permissions.
2. **Tool output is untrusted when it re-enters the prompt.** File contents may
   contain instructions aimed at the model. The system prompt says so, tool
   results are clipped to 4000 characters, and no tool result can trigger
   approval — only the owner can.
3. **Paths are resolved in code.** `Sandbox` expands `~`, resolves symlinks and
   compares resolved parents against the allow-list, so a link inside a trusted
   folder cannot escape it. An empty allow-list refuses every file tool.
4. **Approval is a human channel.** `DenyAll` is the default broker; the CLI
   broker refuses when stdin is not a TTY, so a piped `y` can never approve a
   write. What is shown is exactly what runs; the decision and the tool row share
   an `approval_ref`.
5. **Storage is local and append-only for audit.** `task_log` triggers reject
   `UPDATE` and `DELETE`. Detail is redacted and truncated on the way in.
6. **Inference is local by default.** `OllamaBrain` calls a loopback daemon that serves Qwen on this machine. Hosted providers remain explicit opt-in settings. The archived `ScratchBrain` loads a checkpoint from disk.

Deeper: `docs/architecture/threat-model.md` (16 threats, controls and residual
risks, including the open camera/screen/cursor scope),
`docs/architecture/machine-access.md` and
`docs/architecture/presence-renderer.md`.

## Data model

`0001_initial.sql` creates the task/audit schema; `0002_conversations_directives_preferences.sql` adds user-managed local data.

- `tasks` — one row per request: `id`, timestamps, `device`, `actor`, `kind`,
  sanitized `request`, `status` (constrained), `current_step`, `error`.
  Transitions are enforced in `storage/tasks.py`, not in SQL.
- `task_log` — append-only audit: one row per tool call, approval decision,
  brain call and final answer. `approval_ref` links a decision to the call it
  allowed.
- `summaries` — `summary_date` (unique, `YYYY-MM-DD`), `content_md`.
- `notes` — reserved; no code writes it yet.
- `conversations`, `conversation_messages` — verbatim local user/assistant chat history; conversation messages are deletable from the UI and are not copied into the redacted audit log.
- `user_directives`, `user_preferences` — owner-authored directives and UI settings used in later CLI agent turns and local chat.
- `schema_migrations` — created by the runner: version, applied_at, checksum.

The daily summary is derived from `tasks` + `task_log` alone. That is deliberate:
the log is the single source of truth for what happened, so a summary cannot
claim something the audit trail does not contain.

## Performance

Previously measured on this machine (4-thread CPU, a downloaded 1.5B model): prompt processing ran at
~31 tokens/s, generation at ~9.7 tokens/s. This historical measurement does not predict scratch-model speed. Because every step re-processes the
whole prompt, cost scales with *schema bytes × steps*, not with answer length.

Three changes took one representative request from 126.8 s to 71.7 s:

| Change | Effect |
| --- | --- |
| `num_predict` capped (`CLAIREAI_MAX_OUTPUT_TOKENS`) | An unbounded reply cannot run for minutes |
| Short tool descriptions | Fewer prompt tokens |
| pydantic `title` keys stripped from schemas (−17%) | Fewer prompt tokens |

The remaining lever is sending fewer tools per call (`routing.py`): classify
locally by keyword, send only the reachable tools, and fall back to the full
registry whenever the router is unsure or the model asks for a filtered-out tool.
The 2026-11-27 measurements made this concrete: the same short answer took
118.05 s with 30 schemas, 37.8 s with the full tool policy and no schemas, and
**4.9 s** with the compact conversation prompt. The runtime now keeps Qwen
resident with a startup warmup and `keep_alive=30m`. A mechanism canvas now sends
one tool instead of three; a news canvas sends only the news fetch plus the
canvas/list pair.

Decoding temperature is chosen per phase rather than fixed: an assistant has two
different jobs in one turn. While no step has succeeded it is *deciding* what to
do — that wants repeatability, so it runs at `CLAIREAI_TEMPERATURE_TOOLS`
(default 0.2). Once a step has produced real data it is *phrasing* an answer —
that wants natural language, so it runs at `CLAIREAI_TEMPERATURE` (default 0.8).
Both are ordinary settings, and a caller can pass a temperature for one call
without the adapter's own options changing. Neither default is 0: measured, a
model pinned at 0 emitted a malformed tool call on every attempt, which also
disables the one bounded retry (identical input, identical reply).

## Deliberate non-decisions

- **The API is opt-in and loopback-first.** `claireai --api` / `claireai-api`
  exposes chat, background tasks, devices, approvals, audit, summaries,
  settings, memory and SSE. The same runtime, allow-list and approval policy
  starts an authenticated FastAPI service for other local clients. The Tk window
  still calls the runtime directly and remains the desktop UI. A bearer token
  authorizes API calls, not file writes; those still use signed one-action
  approvals.
- **Browser tools are HTTP, not Playwright.** `open_url`, `get_page_text`,
  `search_web` and `download_file` are implemented, approval-gated and tested,
  but they fetch markup instead of running a browser: no JavaScript, so
  client-rendered pages come back empty. Swapping in Playwright changes one
  module; the tool names and risks the model sees stay the same.
- **Screen and input control are optional extras.** `see_screen` needs the
  `screen` capability and approval; `computer_control` needs `cursor` or
  `keyboard` plus approval, and reports unsupported Wayland sessions honestly.
  Camera capture and clipboard reading are still not built.
- **The wake phrase uses local speech, not a custom model.** The default `stt`
  engine hears “Hey Claire” with the installed faster-whisper model, a hotword
  and pre-roll. openWakeWord is an alternative and still needs a matching model
  for an arbitrary phrase. A human-spoken acceptance run remains pending.
- **No Tauri shell yet.** The tested desktop is Tk/customtkinter and is kept
  deliberately for this phase. Tauri + React is a later target.
- **No hosted inference by default.** Ollama + Qwen is local; hosted providers
  remain opt-in. The archived scratch model is still available through the same
  `Brain` protocol for research.
