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
   │      └─ claireai.brain.ollama ──► POST /api/chat        │
   │                                  (127.0.0.1:11434)      │
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

Every loop iteration re-sends the conversation *and all tool schemas* to the
model. That is the dominant cost on a CPU-only machine (see "Performance"
below), which is why schema size and tool count are treated as product
constraints rather than cosmetics.

## Modules

| Module | Responsibility | Does not |
| --- | --- | --- |
| `config/settings.py` | Validated `CLAIREAI_*` configuration, loopback enforcement, path resolution | Read the network, create files |
| `storage/database.py` | SQLite connection, `BEGIN IMMEDIATE` transactions, checksummed migration runner | Know about tasks or tools |
| `storage/tasks.py` | Task rows and the explicit status state machine | Run anything |
| `audit/audit_log.py` | Append-only `task_log` writes, redacted detail | Update or delete (triggers reject it) |
| `audit/redaction.py` | Pattern-based secret redaction, length truncation | Guarantee that no secret is ever stored |
| `brain/base.py` | `Brain` protocol, `Message`/`ToolCall`/`BrainResponse` wire-neutral shapes | Import any concrete runtime |
| `brain/ollama.py` | `POST /api/chat`, `GET /api/tags`, tool-call parsing | Retry, decide, or run tools |
| `permissions.py` | `Sandbox` (folder allow-list, symlink-aware resolution), risk policy, capability policy | Prompt the user |
| `tools/base.py` | `Tool` (name, description, risk, pydantic args, handler), registry, schema stripping | Execute side effects |
| `tools/files.py` | Five read-only file tools | Write anything |
| `tools/files_write.py` | Six mutating tools, all `needs_approval` | Delete permanently, bypass the sandbox |
| `approval.py` | `ApprovalRequest`, per-task grants, `DenyAll` default | Decide on the owner's behalf |
| `agent.py` | The loop: validate, authorize, run, log, feed back | Trust model output |
| `routing.py` | Narrow the tool set per request (deterministic, no model call) | Remove the fallback to all tools |
| `presence.py` | `PresenceEvent` stream (idle, thinking, tool_running, …) | Hold any capability |
| `presence_server.py` | Loopback stdlib HTTP server: `/state`, `/events` (SSE), the static HUD and the generated `/today` page | Invoke a tool, accept a POST, or write anything |
| `today.py` | The read-only page: summary plus the day's tasks, fully escaped, `mode=ro` | Act, approve, or render stored strings as HTML |
| `retention.py` | Plans and applies removal of old trash batches and summaries | Delete inside the audit log, or outside the data dir |
| `summary.py` | Daily report from `tasks` + `task_log`, saved to `summaries` and markdown | Invent an event |
| `scheduler.py` | APScheduler job that builds the summary each evening | Need a running app |
| `evaluation.py` | Fixed cases, deterministic scoring, report rendering | Grade creativity |
| `doctor.py` | Read-only environment/schema/model diagnostics | Write anything |
| `cli.py` | Argument parsing, interactive loop, approval prompt | Contain business logic |

`scripts/` holds entry points that use the package: `doctor.py`, `eval_model.py`,
`smoke.py`, `presence_preview.py`.

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
6. **Inference is loopback.** `Settings` rejects a non-loopback model URL unless
   `CLAIREAI_ALLOW_NON_LOOPBACK_OLLAMA=true`; no hosted provider exists.

Deeper: `docs/architecture/threat-model.md` (16 threats, controls and residual
risks, including the open camera/screen/cursor scope),
`docs/architecture/machine-access.md` and
`docs/architecture/presence-renderer.md`.

## Data model

`0001_initial.sql` is the only migration so far.

- `tasks` — one row per request: `id`, timestamps, `device`, `actor`, `kind`,
  sanitized `request`, `status` (constrained), `current_step`, `error`.
  Transitions are enforced in `storage/tasks.py`, not in SQL.
- `task_log` — append-only audit: one row per tool call, approval decision,
  brain call and final answer. `approval_ref` links a decision to the call it
  allowed.
- `summaries` — `summary_date` (unique, `YYYY-MM-DD`), `content_md`.
- `notes` — reserved; no code writes it yet.
- `schema_migrations` — created by the runner: version, applied_at, checksum.

The daily summary is derived from `tasks` + `task_log` alone. That is deliberate:
the log is the single source of truth for what happened, so a summary cannot
claim something the audit trail does not contain.

## Performance

Measured on this machine (4-thread CPU, `qwen2.5:1.5b`): prompt processing runs at
~31 tokens/s, generation at ~9.7 tokens/s. Because every step re-processes the
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

## Deliberate non-decisions

- **No FastAPI service.** The dependency is installed; nothing serves HTTP except
  the read-only presence server. The desktop window will need an API surface, and
  that shape is not chosen yet.
- **No browser or machine tools.** Playwright is not installed and camera/screen/
  cursor have contracts but no code. Both are scheduled, both are privacy-heavy.
- **No packaging.** There is no Tauri shell, no installer, no signed binary.
- **No hosted inference, ever by default.** Track B's from-scratch model will be
  another `Brain` implementation behind the same protocol.
