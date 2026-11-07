# Changelog

All notable changes are recorded here. Versions follow the roadmap's phases;
nothing has been released yet.

## [Unreleased]

### Fixed — routing hid the tool that could answer the request

- "show me what report.pdf says" routed to `list_folder`, "what is in Pictures"
  to the vision tools, and "replace the contents of notes.txt" to
  `read_text_file`, so the model was never shown the tool that could answer. A
  `read_contents` intent plus ordering fixes (`browse` before reads, `list` before
  vision) keep the request and the tool together, and eight phrases that used to
  misroute are regression-tested in `test_routing.py`.
- The corpus builder routed against the file-only registry, so every request that
  needed a browser tool was unroutable. It now routes against the
  browser-enabled registry and records the outcome in the manifest under `routing`.
- Tool-call records now carry only the tools routing would offer for their own
  prompt, which is what inference does. Routing misses fell from 207 to 5 of
  7,117 tool records, and those five are left untouched rather than rewritten.

### Added — warm-start fine-tuning

- `train_chat.py` gained `--init-from` and `--tokenizer-from`: a warm start reuses
  the previous checkpoint's BPE merges, because merges fitted on a new corpus
  renumber every token and the embeddings would no longer line up. The checkpoint
  manifest records `initialized_from`.

### Added — a workspace calendar whose dates come from code

- `list_appointments` (read-only) and `add_appointment` (approval-gated) read and
  write an `appointments` table added by `migrations/0003_appointments.sql`.
  `what's on tomorrow` routes to the calendar intent, and the phrase is resolved
  by `claireai.dates` before any row is touched.
- An unreadable day is an error rather than "today", a start time that is not
  24-hour `HH:MM` is refused, and the list is ordered by time because an
  unordered agenda is not an agenda.
- `what's on tomorrow` is answered by the runtime from the table (about 8 ms,
  model `claireai-runtime`, audit target `calendar`) instead of by the model, for
  the same reason dates are: a 3.3M model without the table in its prompt can
  only invent a meeting. Phrases whose day is not a day (`what's on the news`)
  fall through to the model.
- `assistant/tests/unit/test_calendar.py` covers the empty day, ordering, phrase
  resolution, both refusals, the week window, the risk levels and the phrases
  that must *not* be hijacked.

### Fixed — the preview's `/events` handler outlived its server

- `_stream` looped forever, so a closed preview left a thread writing to a dead
  socket until the next write happened to fail. Teardown now sets `closing` and
  the loop returns.

### Added — dates are answered from the clock, not guessed by the model

- `claireai.dates` resolves `today`, `yesterday`, `tomorrow`, `3 days ago`,
  `in 2 weeks`, `last friday`, `2026-11-06` and `6 Nov 2026`. Ambiguous numeric
  forms such as `06/11/2026` are refused rather than guessed, because 6 November
  and 11 June are different days to different readers.
- Date and time questions are answered by the runtime in both chat paths
  (`ask`, used by the CLI chat and the local web UI, and `run_task`, used by
  `--once` and UI tasks) with an audit row and model `claireai-runtime`. This is
  not gold-plating: the installed model answered "what is the date today" with
  "I created the mall to the allty..." on this machine, and a confidently wrong
  date is worse than a refusal. The answer now takes ~20 ms and never reaches a
  model.
- `--summary` and `/summary` accept the same phrases, so `--summary yesterday`
  and `/summary 3 days ago` work. The old `--summary` help and error text said
  `YYYY-MM-DD` only.
- Tests pass a fixed clock (2026-11-06, a Friday) so they cannot fail on one
  particular Monday, and the time assertion converts to the host zone instead of
  pinning one.

### Changed — installed chat-v4: tool calls are readable, the tool choice is not right yet

- `runtime-data/models/claire-scratch.{pt,json}` is now chat-v4 (5 layers, width
  224, 288 context, 3.34M parameters, validation loss 1.3826 at step 5,500).
  Rollback remains `claire-chat-v2.{pt,json}`.
- Measured against the previous install: tool-call checks **66/138 (was 3/92)**,
  conversation accuracy **11/21 (was 9/21)**, `eval_model` 5/13 (was 3/13).
- The readable-call problem is fixed; the wrong-tool problem is not. The model
  emits `read_text_file` for "list my Downloads folder" even when it is the only
  tool offered. Cause, measured: the corpus holds 109 distinct paths across 6,241
  tool records, so this capacity memorises entities instead of reading the
  request. Next step is a data rebalance, recorded in `PROGRESS.md`.

### Added — opt-in connectors for Claude, ChatGPT and Ollama

- `claireai.brain.build_brain` is now the single place a brain is constructed. The
  CLI, the desktop runtime, the local web server and `scripts/doctor.py` all go
  through it instead of instantiating `ScratchBrain` themselves, so a provider is
  one setting (`CLAIREAI_BRAIN_PROVIDER`) rather than four code paths. `scratch`
  is the default and still makes no network request.
- Three adapters, each normalising its own wire format into `ToolCall`: OpenAI
  Chat Completions (`openai`, also covers vLLM/LM Studio), Anthropic Messages
  (`anthropic`, hoists the system prompt and merges tool results into one user
  turn) and Ollama's native `/api/chat` (`ollama`, object arguments, works fully
  offline). Tool-call ids the APIs require are generated and paired by the
  adapters, so the agent stays provider-neutral.
- Configuration refuses a plain-`http` endpoint off loopback, and an
  authentication failure names the provider and the variable to check without
  ever echoing the key.
- `scripts/tool_call_eval.py` with `scripts/tool_call_cases.json`: 16 cases, three
  checks each, run through the real parser and the production system prompt. The
  installed chat-v2 checkpoint scores **3/92 checks and 0 of 15 tool cases** on
  it, which is the baseline the retrained model has to beat.
- No live credentials are used: the connector tests drive
  `httpx.MockTransport`, so they prove the wire format and parsing, not that a
  vendor's API is reachable. `scripts/doctor.py` is the check that needs keys.

### Fixed — a finished training run no longer throws its checkpoint away

- `brain-lab/train_chat.py` bound a local variable named `sample` for the
  tokenizer-fit text, which shadowed the sampling function used to print probes.
  Every completed run therefore crashed at the last step — after ~80 minutes of
  training, before `torch.save` — so no checkpoint was written. The run is
  recovered through the existing `--resume` path, and the variable is renamed.
- The chat corpus carried a hand-written system sentence while `claireai.agent`
  sent a different one. Measured with the same weights: correct tool calls under
  the corpus sentence, gibberish under the runtime one. `build_chat_dataset.py`
  now imports `claireai.agent.system_message`, varies only the folder clause, and
  `assistant/tests/unit/test_corpus_prompt.py` fails if the two diverge again.
- Grounded records briefly carried two folder clauses (the generated one and
  their real one) after that change; records that list real folders now keep
  their own clause and skip the generated one.
- A resumed run with no steps left reported `training_loss: NaN`, which is not
  valid strict JSON; it is now `null`.

### Changed — the installed scratch checkpoint now answers, and CI installs what it tests

- Installed the 6,000-step conversation checkpoint as
  `runtime-data/models/claire-scratch.{pt,json}`. On identical settings it scores
  3/13 format cases (was 0/13) and 9/21 held-out conversation cases (was 1/21).
  Free-form prose is still incoherent; the measured numbers and the rollback
  checkpoint are recorded in `PROGRESS.md`.
- Fixed a real safety bug in `brain-lab/harvest_corpus.py`: `robots_allows`
  treated an unset `RobotFileParser.default_entry` as blanket permission, and
  Python 3.14 leaves that attribute unset while still applying a parsed
  `User-agent: *` block, so a `Disallow:`d path was fetched. It now asks
  `can_fetch` after confirming robots.txt was actually read.
- Tests that assume a Linux session or an optional extra now skip instead of
  failing on the wrong platform: the XDG autostart tests, the widget tests that
  need `customtkinter`, and the POSIX audio-playback tests (Windows uses
  `winsound` and a real device).
- CI was failing on every runner before a single test ran: `test_tokenizer.py`
  imports `claire_brain`, which the workflow never installed, and
  `test_vision.py` imports `PIL`. The workflow now installs `./brain-lab` with
  `--no-deps` (its tokenizer and prompt renderer are pure Python, so no torch
  wheel is needed on four runners) and `Pillow` is a dev dependency, so the
  blank-frame refusal tests run in CI rather than being skipped.

### Fixed — a disabled browser no longer crashes the agent loop

- `claireai.agent` now imports `BrainResponseError`. Switching browser access off
  between a model proposal and the action it proposes raised `NameError`, which
  the `except BrainError` handler could not catch, so the task was never marked
  failed, no audit row was written and the caller saw a traceback. It now fails
  the task with the real reason, and a regression test asserts the failed status
  and the audit row (issue #1).

### Changed — scratch model conversation training

- `brain-lab/build_chat_dataset.py` builds one record per assistant turn in the
  exact runtime prompt shape, mixes original synthetic small talk, refusals,
  intent routing, slot extraction, tool transcripts, summaries and in-context
  recall with Dolly 15k, TinyStories and the bootstrap corpus, and removes every
  user turn that exactly matches a prompt in either evaluation suite.
- `brain-lab/train_chat.py` trains with EOS after every reply and masks the
  prompt from the loss, so capacity goes to replies instead of re-learning the
  prompt. It drops the oldest turns when a prompt is too long, saves a resumable
  partial checkpoint every evaluation interval, and keeps the best validation
  weights. `brain-lab/promote_partial.py` turns a partial into a loadable
  checkpoint after an interrupted run.
- `claire_brain.prompt` is now the single prompt renderer used by training and
  inference; assistant tool calls render as the same `<tool_call>` text the
  model was trained on.
- `claire_brain.model` initialises weights at GPT-2 scale, supports a masked
  loss, and applies top-p and repetition penalties during generation. The old
  default init produced logits in the tens (initial loss ~88 instead of
  ~ln(vocab)), which made a small model spend thousands of steps shrinking its
  own output.
- `claireai.brain.ScratchBrain` keeps the system prompt and current request when
  a prompt exceeds the context window, caps the offered tool block, and accepts
  a bare JSON tool call only when its name is one of the tools actually offered.
- The agent system prompt is compact enough to fit the scratch model's 256-token
  window alongside tool schemas and a grounded folder listing.
- `scripts/conversation_eval.py` runs held-out multi-turn cases, injects tool
  results and scores every reply with deterministic code.

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
