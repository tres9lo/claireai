# Changelog

All notable changes are recorded here. Versions follow the roadmap's phases;
nothing has been released yet.

## [Unreleased]

### Added — Month 1, Week 4: read-only today page and retention

- `claireai.today` and a `/today` route on the existing loopback server: the
  daily summary plus the day's tasks, for reading without a terminal and as the
  page the desktop shell will wrap. SQLite is opened `mode=ro`, responses carry
  `default-src 'none'` CSP and `nosniff`, every stored string is escaped, and the
  page contains no JavaScript, form or button. Approve/deny controls are
  deliberately absent: granting a capability over loopback HTTP needs a per-run
  token and a nonce, not a page feature.
- `claireai.retention` plus `claireai --prune [--days N] [--apply]`: plans the
  removal of trash batches and summaries older than the window, then applies it
  and records `retention.prune`. Symlinked batches, non-timestamp folder names
  and paths outside the data directory are refused. The audit log is reported and
  left alone — pruning it would mean giving up the append-only triggers.
- `presence_preview.py --db PATH` enables the today page; the HUD footer links to it.
- Smoke check asserts the page renders without script/form and that retention
  removes only stale batches (now 14 checks).

### Added — Month 1, Week 4: tool routing and the daily summary

- `claireai.routing`: deterministic (keyword, no model call) intent routing that
  narrows the tool schemas sent on each step, with a fallback to the whole
  registry when nothing matches. Measured on this laptop: prompt tokens roughly
  halved (−50% / −46% / −60% on three requests) and no change in tool choice.
  Routing changes what the model *sees*, never what is *allowed*: a filtered-out
  tool that is called anyway still passes validation and approval.
- `CLAIREAI_TOOL_ROUTING=auto|off` and `--no-routing`; each task records an
  `agent.route` audit row with the intent, the tools shown and the schema bytes
  saved.
- `claireai.summary`: the daily report (Done / Failed or skipped / Still pending)
  built from `tasks` + `task_log` by plain code, upserted into `summaries` and
  written to `<data-dir>/summaries/YYYY-MM-DD.md`. Times are converted to the
  owner's local day and printed locally.
- `claireai.scheduler`: the APScheduler cron job behind `claireai --summary-at
  HH:MM`; `--summary [DATE]` and the `/summary` command print one day on demand,
  with no model call.
- Tool descriptions rewritten so each tool owns one job (`list_folder` no longer
  advertises itself for "biggest" requests — with fewer schemas competing, it was
  winning that request and answering non-recursively).

### Added — Month 1, Week 1 foundations

- `claireai` Python package (src layout) with validated settings, platform-aware
  application-data paths, and a loopback-only default for the model runtime.
- SQLite storage: connection settings, checksummed migration runner, and the
  first migration creating `tasks`, `task_log` (append-only), `summaries`,
  `notes`.
- Task repository with an explicit state machine; append-only audit log with
  credential redaction and truncation.
- Provider-neutral `Brain` interface and `OllamaBrain` adapter for local
  inference (no hosted provider, no API keys).
- CLI chat (`python -m claireai.cli`) that records every turn as a task plus an
  audit entry; exit codes distinguish configuration errors from runtime
  unavailability.
- Unit tests (49) covering settings validation, migrations and schema safety,
  state transitions, redaction, the Ollama adapter (mocked) and CLI logging.
- Docs: README, `.env.example`, Linux/Windows setup guides, platform support
  matrix, model licence record.

### Added — model evaluation harness

- `claireai.evaluation` with four deterministic check kinds and a repeat-each-case
  runner; no model grades another model.
- `scripts/eval_model.py` with a clearly-labelled stub mode and a real
  `temperature=0` run against the configured local model; results written outside
  git with exit codes for stable / failed / misconfigured / unavailable.
- `scripts/model_eval_cases.json`: 10 cases covering intent routing, slot
  extraction, JSON tool calls, the three-section summary and two safety cases.
- `OllamaBrain` accepts decoding `options`; `docs/testing.md` documents the
  suite and the harness.

### Added — Month 1, Week 3: approval-gated writes, presence v0, smoke check

- `claireai.approval`: approval requests carrying tool, risk, validated
  arguments, target and a reference; `DenyAll` as the default for programmatic
  runs (there is deliberately no auto-approve broker in the product).
- Six mutating file tools, all `needs_approval`: `create_folder`,
  `create_text_file`, `write_to_file`, `rename_file`, `move_file`,
  `move_to_trash`. Nothing deletes permanently — trash is a recoverable folder in
  the app data directory. Existing files are never silently overwritten.
- Agent loop: validates before asking, shows the exact target, logs
  `approval.decision` plus the `tool.call` row sharing one `approval_ref`, feeds a
  denial back to the model, and never runs a denied action.
- CLI approval prompt; refuses when stdin is not an interactive terminal.
- `claireai.presence` + `claireai.presence_server`: the `PresenceEvent` stream, a
  loopback-only standard-library server (`/state`, `/events` SSE, `/healthz`) and
  a canvas 2D HUD in `desktop/presence/index.html`; `scripts/presence_preview.py`
  serves it with `--demo`. No npm or WebGL dependency.
- `scripts/smoke.py`: ten end-to-end checks that run on Linux and Windows,
  including a refused write and an append-only audit assertion.
- Performance: `CLAIREAI_MAX_OUTPUT_TOKENS` (default 400), compacted tool schemas
  (pydantic `title` keys stripped, −17% parameter bytes), and shorter tool
  descriptions. The "biggest files" turn went from 126.8 s to 71.7 s on this
  laptop; the remaining cost is CPU prompt-processing, measured at ~31 tok/s.
- Model evaluation grew to 13 cases (three tool-selection cases added); the
  shipped model scores 12/13 stable, 36/39 checks.
- `docs/architecture/threat-model.md` written: trust boundaries, 16 threats with
  controls, assumptions, residual risks and a verification checklist.

### Planned and specified — presence renderer and machine access

Documentation and configuration only; no renderer, camera or input code exists
in this release.

- **Presence renderer** contract (`docs/architecture/presence-renderer.md`):
  `PresenceEvent` state stream (idle, listening, thinking, tool_running,
  awaiting_approval, speaking, error, offline) derived from the task log, a
  performance budget for integrated GPUs, 2D fallback, accessibility rules, and
  the rule that a renderer failure never changes task behaviour. Scheduled as v0
  in Month 1 Week 4, lip-sync in Months 2–3, WebGL quality in Months 6–7.
- **Machine access** contract (`docs/architecture/machine-access.md`): camera,
  screen capture, clipboard, windows, cursor and keyboard as deny-by-default
  capabilities with per-task expiring grants, mandatory indicators, 1-frame
  in-memory image handling, platform matrix (Wayland portal limits included), and
  redaction of typed text in the audit log. Scheduled for Months 2–3.
- Settings: `presence_mode`, `presence_fps_cap`, `presence_reduce_motion`,
  `allowed_capabilities`, `camera_device`. Unknown capability names are rejected
  at startup; an empty list refuses every machine-access call.
- `claireai.permissions.check_capability` plus 11 tests pinning the
  deny-by-default behaviour.
- Licences recorded from upstream package metadata: three.js MIT,
  @react-three/fiber MIT, `mss` MIT, `opencv-python` Apache-2.0, `pyautogui` BSD.
  `pynput` (LGPLv3) deliberately not adopted.

### Added — Month 1, Week 2: tool registry, allow-list and agent loop

- `claireai.tools`: registry of tools with name, description, JSON schema and
  risk level. Arguments are validated against each tool's pydantic model before
  a handler runs; unexpected keys are rejected.
- Five read-only file tools: `list_folder`, `get_file_info`, `read_text_file`,
  `search_files`, `find_largest_files`. All are `safe`; none writes.
- `claireai.permissions`: folder allow-list (`CLAIREAI_ALLOWED_FOLDERS`, empty by
  default = deny everything) with symlink-aware containment checks, plus a risk
  policy that refuses `needs_approval`/`dangerous` tools while no approval flow
  exists.
- `claireai.agent`: the agent loop — validate, authorize, execute, log, feed the
  result back, stop after `CLAIREAI_AGENT_MAX_STEPS` (default 8). Every step and
  the final answer are written to `task_log`.
- `BrainResponse`/`Message` now carry tool calls and tool results; `OllamaBrain`
  parses `message.tool_calls` and rejects malformed calls instead of guessing.
- CLI: tools on by default, `--no-tools` for plain chat, `--dry-run` to show
  calls without reading anything; exit code 1 when a task cannot finish.
- Tests: 155 passing, including a security suite for the allow-list (parent
  traversal, symlink escape, empty allow-list) and agent-loop tests for refusal,
  invalid arguments, dry run and the step limit.

### Verified — first real local inference (Linux, 2026-10-30)

No product code changed; this records the Week 1 acceptance result and the
measurements it produced.

- Ollama 0.40.2 + `qwen2.5:1.5b` answered through the CLI without an API key,
  online and inside a network namespace with no route (offline proof), with the
  task and `task_log` audit row written each time (0 orphans across 6 turns).
- Measured cost: ~1.12 GiB RSS for the model runner, 8.1 s cold, 1.2–2.5 s warm.
- First model evaluation against the real model: 9/10 cases stable, 27/30 checks.
  The one failure (`refuses-to-bypass-approval`) is recorded, not deleted: the
  model refuses safely but its wording does not mention approval. Approval and
  refusal text will therefore come from code, not model output.

### Added — CI and diagnostics

- `claireai.doctor` + `scripts/doctor.py`: read-only report of versions, paths,
  database/schema state and local model-runtime status; creates nothing and
  prints no secret values.
- `.github/workflows/python-tests.yml`: pytest on Ubuntu and Windows, Python
  3.11 and 3.12. Pushed on 2026-10-30, so it has triggered, but no run result
  has been read back; not evidence of Windows support.
