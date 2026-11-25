# Changelog

All notable changes are recorded here. Versions follow the roadmap's phases;
nothing has been released yet.

## [Unreleased]

### Added — CUA, wake word, local vector memory and the API service

- **CUA driver:** `claireai.cua` drives the pointer, clicks, scrolling, typing
  and keys through optional `pyautogui`; screenshots use `mss`. The
  `computer_control` tool is registered only when `cursor` or `keyboard` is in
  `CLAIREAI_ALLOWED_CAPABILITIES`, checks the capability for each action, is
  `dangerous`, always needs approval, and redacts typed text from the audit row.
  `see_screen` now enforces the `screen` capability. Camera and clipboard remain
  unimplemented.
- **Wake word:** `claireai.wakeword` wraps openWakeWord with bounded 16 kHz
  frame scoring, a cooldown and a microphone listener that saves no audio.
  `scripts/wakeword_check.py` is the explicit check. No model is bundled or
  downloaded; "Hey Claire" needs a matching trained model. The ONNX engine was
  live-checked with a synthesized "hey jarvis" clip (top score 0.9967), and the
  listener opened ALSA capture on this Linux host; a human-spoken acceptance
  test is still pending.
- **Local semantic memory:** migration `0005_memory` adds `memory_entries`;
  `claireai.memory` provides Ollama embeddings (`nomic-embed-text`), an explicit
  hash fallback, exact cosine search and the `remember_note`/`search_memory`
  tools. Text and vectors stay in SQLite; no background embedding of chat.
  Verified live with Ollama `nomic-embed-text`: 768-dimension vectors and the
  correct top semantic hit for a paraphrase with no shared keywords.
- **FastAPI backend:** `claireai --api` / `claireai-api` serves a loopback,
  bearer-token-authenticated API (`/health`, status/models, chat, conversations,
  memory search, `/docs`). It reuses `run_task` and the audit/sandbox policy;
  write tools still need a signed one-action approval and cannot be approved by
  the API token. The listener was exercised over a real loopback socket:
  `/health` returned 200, and `/v1/status` returned 401 without a token and 200
  with one.
- Docs: `docs/api.md`, `docs/setup/extras.md`; model storage now includes the
  optional embedding model.

### Changed — Ollama + Qwen replaces from-scratch training as the default

- The default brain is now **Qwen3 1.7B** served by a local Ollama daemon, with
  **Qwen2.5-Coder 1.5B** selected for code-heavy turns. No checkpoint has to be
  trained, gated or promoted before chatting.
- This is a deliberate response to the development hardware: a CPU-only budget
  PC with about 7 GB RAM. From-scratch runs took hours, several candidates
  failed the gate, and the result was not a dependable assistant. Local
  pretrained inference is the proportionate choice; `brain-lab/` remains as
  archived research.
- Setup, doctor, the CLI header, identity answers, the desktop model status and
  the evaluation/smoke harnesses now use the configured brain instead of
  hard-coding a scratch checkpoint. Setup explains `ollama pull` and no longer
  requires PyTorch for the desktop extra.
- Exact Linux and Windows model-install steps are in
  [docs/setup/models.md](docs/setup/models.md).

### Added — bounded ToolACE and Hermes fetchers

- `brain-lab/fetch_corpora.py --source toolace` and `--source hermes` fetch at
  most 500 rows per source, require one or two calls, validate call names against
  the row's source schema, normalise ToolACE's bracket syntax to the runtime's
  `<tool_call>` JSON format, remove held-out evaluation prompts and exact
  duplicates, and write a licence / retrieval-time / SHA-256 sidecar. Tool names
  remain metadata and must be mapped to registry tools before training.

### Added — replay buffer and retrieval closeout

- `LearningJournal.compile()` now refreshes `learning/replay.jsonl` with a
  deterministic, bounded sample of older confirmed, source-bearing lessons.
  `TrainingCurriculum` replays those examples so a new lesson does not crowd out
  greetings, refusals or tool calls. Unverified model answers are never replayed.
- The unknown-answer path is covered end to end: after an approved search, the
  model receives the cited `RESEARCH RESULT`, the final answer is generated from
  it, and no training corpus is written during the conversation. The playbook
  now documents the difference between the journal, the compiled corpus and the
  gated candidate.

### Added — bilingual corpus, one-command training, and trained-context fitting

- `brain-lab/train_claireai.py --profile full` is the single reproducible path:
  fetch the bounded licensed sources, build the authored English/French corpus,
  merge it, train a candidate and run both fixed evaluations. It never installs
  or promotes automatically; a pipeline log records each stage.
- The authored curriculum now teaches everyday English and French conversation,
  vocabulary, grammar, approximate pronunciation and French intent variants.
  OASST1 is read from its released archive as separate English and French
  splits, and FineWeb-2 French adds bounded prose.
- `train_chat.py` stores token ids in compact arrays, so the larger corpus no
  longer forces the 5k-record fallback, and uses the tokenizer's reserved pad id.
- Context-only expansions now record `trained_context`, and `ScratchBrain` fits
  prompts to the positions that were actually trained. Feeding an expanded
  checkpoint's untrained rows was producing gibberish on short greetings.
- `brain-lab/image_training.py` and an empty manifest reserve the future
  image/point/box/text schema; training remains unimplemented and the model is
  still text-only.

### Added — confirmed research, intent training, and a 10K-capable architecture

- `--email-setup` guides Gmail/IMAP setup, verifies a read-only login before
  storing anything, uses hidden input, and writes only to the OS keyring. It
  stores the address/host/port beside the password so `--inbox` works from the
  desktop menu; environment variables still override them. The `keyring`
  package is now a core dependency instead of an uninstalled optional import;
  the setup reports a missing backend instead of falling back to a file.
- News retrieval retries transient DNS/timeout/5xx failures three times and
  reports a resolver failure as a DNS problem with the affected country, instead
  of surfacing `[Errno -3] Temporary failure in name resolution`.
- The runtime can detect that the local model does not know a public factual
  answer, ask `research_web` approval, search, read the top public result, and
  ask a second time before saving a source-bearing lesson. Private requests are
  never searched automatically, page text is stripped of prompt-role markers,
  and only owner-confirmed lessons can enter training.
- Background learning no longer compiles every model answer. Ordinary answers
  remain journaled for inspection but are not training targets; confirmed
  lessons require an http(s) source. The nightly worker now merges authored
  intent/conversation records and confirmed lessons into the external corpus
  rather than silently choosing the external corpus and dropping both.
- The transformer supports an optional **10,240-token rotary-position**
  architecture with memory-efficient causal attention. A converted candidate ran
  at 9,734 prompt tokens in 18.7 s and 461 MB RSS on the reference i5; it failed
  the recall probe and was not installed. Old learned-position checkpoints keep
  loading unchanged.

### Added — daily briefing, read-only inbox, and corpus cleanup

- `claireai --briefing [--countries Kenya,France]` combines cited country news,
  today's appointments and pending tasks in one local report;
  `--briefing-at HH:MM` runs it daily in the foreground. The report is assembled
  by code, not the model, and omits news while browser access is off.
- `claireai --inbox [N]` reads unread message headers through IMAP over TLS in a
  read-only mailbox with `BODY.PEEK`, so nothing is marked read. There is no
  send, delete or move path. Credentials come from the OS keyring account
  `email_app_password`, with `CLAIREAI_EMAIL_ADDRESS` and
  `CLAIREAI_EMAIL_IMAP_HOST` selecting the mailbox.
- `fetch_corpora.py` now includes a bounded FineWeb-Edu reading sample (ODC-By
  1.0). `merge_corpora.py` removes exact duplicate pairs, drops training pairs
  that also appear in validation, and records the removed counts. A rebuilt
  `chat-v13-small` removed 948 duplicates/overlaps and 20 validation duplicates.
- A long-context probe (`scripts/long_context_eval.py`) tests fact retrieval at
  short, medium and near-limit prompt lengths. It is diagnostic, not a gate: the
  installed 512-token checkpoint did not retrieve the marker at any tested length.

### Added — 512-token context, public corpora, and country news

- The installed scratch checkpoint now uses **512 tokens**, expanded from the
  proven 288-token weights. A structural installer verifies every old tensor and
  the first 288 position rows before replacing the checkpoint, and archives the
  old pair. The merged-corpus pipeline is included, and the nightly worker now
  continues from the installed architecture.
- `fetch_corpora.py` downloads bounded, licensed samples (GSM8K MIT, OASST1
  Apache-2.0 best-effort) with sidecar licence/hash records. `merge_corpora.py`
  combines them with the curated corpus, removes evaluation prompts, and writes
  `chat-v13` / `chat-v13-small`.
- `claireai --news Kenya,France` and plain “latest news from Kenya” requests read
  public Google News country editions and return cited headlines. The model never
  invents a headline; an unavailable feed is reported as unavailable.

### Added — training without the window, and exact calculations

- `claireai --install-training-service` installs a systemd **user** timer for a
  daily headless `--training-run`. It needs no root or display, honours the
  Background learning switch and emergency stop, and can be inspected or removed
  with `--training-service-status` / `--uninstall-training-service`.
- A candidate whose review was unavailable is retried without training again; a
  failed gate no longer strands it until a new candidate is made.
- Plain arithmetic, percentages and common unit conversions are answered exactly
  by `claireai/calculations.py`, in both the CLI and the agent path. The model is
  no longer asked to calculate when code can.
- Added a licensed training-source catalog with explicit access states, plus the
  official Gmail, Microsoft Graph, WhatsApp, Instagram, Facebook, X and LinkedIn
  routes. Restricted sources are marked `do-not-use` rather than worked around.

### Added — a candidate must pass the gate before install

- Training candidates are now compared with the installed checkpoint on the
  same fixed tool-call, conversation and capability suites at the same run
  count. Promotion requires a strict tool-check win and no conversation
  regression; an unavailable or failed suite rejects the candidate instead of
  guessing. The old `.pt`/`.json` pair is archived in `models/archive/` before
  replacement, and rejected candidates stay in `models/candidates/`. Settings
  shows the gate reason. The capability score is diagnostic, not a promotion
  criterion.

### Added — background curriculum and bounded training

- With **Background learning** on in Settings, a second worker turns the journal
  plus an authored practice curriculum into a train/validation corpus covering
  conversation, hospitality, reports, accounting, arithmetic, unit calculations,
  coding, routines, email/social drafting, research and business/sales examples.
  At most once a day, when the network check succeeds, it runs one bounded CPU
  training job from the checkout and writes a **candidate** under
  `models/candidates/`. Nothing is promoted automatically; the evaluation gate
  still decides. Training never logs into or posts to email or social accounts.
- Settings shows the training state, last attempt and candidate name.

### Fixed — voice failed even after installing its pieces

- `paplay`/`pw-play` do not accept WAV on stdin, so the old `player -` pipe
  exited 1 and reported "local audio player could not play the generated
  speech". Playback now uses a mode-0600 temporary WAV and tries `pw-play`,
  `paplay` and `aplay` in turn, reporting the last real player error.
- `faster-whisper` 1.2.1 calls `av.open(..., metadata_errors=...)`, which av 19
  removed; the voice extra resolved av 19 and then failed on the first
  recording. The extras now pin `av>=11,<16`. Transcription was verified end to
  end on a generated WAV; the tiny model's accuracy on synthetic eSpeak audio is
  rough, so treat its transcripts as a draft and type when precision matters.
- The menu icon used an absolute path inside the virtual environment, which
  some docks do not load. It is now installed as the hicolor icon `claireai`
  and the entry sets `Icon=claireai`; the window uses the same mark.
- Setup reports the Tk font backend, so a bitmap-only Tk says to run
  `./install.sh` instead of letting the window look pixelated.
- The broken miniconda copy of `claireai` was removed; the applications-menu
  entry is the launcher and uses the Xft/Tk environment.

### Changed — the app is desktop and terminal only

- The browser surface is gone: its loopback HTTP server, canvas page, read-only
  `/today` page, `/api/*` routes, `claireai-ui` script and the
  `--port`/`--no-browser` flags were removed, along with the now-unused FastAPI
  and uvicorn dependencies. `claireai` opens the native window; `claireai --cli`
  is the terminal. The old flags exit with a message naming the replacement
  instead of being silently passed to the CLI.

### Fixed — an installed copy looked pixelated and lost its model

- The pixelated window was not the theme: conda's Tk 9 reports `fontsystem x11`,
  so every glyph came from a bitmap face. `./install.sh` now probes candidate
  interpreters and builds its environment with one whose Tk has Xft (on this
  machine `/usr/bin/python3`, not the conda base). The desktop launcher also
  refuses a bitmap-font Tk with that explanation, unless
  `CLAIREAI_ALLOW_BITMAP_FONTS=1` is set.
- An installed wheel lives in `site-packages`, so model discovery could not see
  `runtime-data/models` and showed "No checkpoint installed yet" even though the
  checkout had one. Discovery now follows the remembered/discovered checkout;
  the checkpoint is still copied, never moved or downloaded.
- The installer writes an applications-menu entry that runs its own venv Python,
  so starting ClaireAI from the menu cannot pick up a different `claireai` on
  PATH (or that interpreter's bitmap Tk). It also records the checkout path.

### Fixed — Update could not find the project after an installed update

- A wheel installed by Update into `/tmp` left pip's `direct_url.json` pointing
  at a file that no longer existed, so clicking Update again could not find a
  checkout. Update now tries `CLAIREAI_SOURCE`, the path remembered from the
  last install/update, pip's direct URL, its own location, the cwd, and a
  bounded search of the usual project folders. `--update --source PATH` sets and
  remembers it explicitly, and the Settings button works from any cwd after
  that.
- Tests now run against `src/` through pytest's `pythonpath`, so a stale wheel
  in the virtual environment cannot hide a source change.

### Added — installation says what it will download, then asks

- `./install.sh` (and `install.ps1`) detects the OS, architecture, Python, Tk
  and display first; refuses Python older than 3.11; prints each component with
  an approximate download size; and downloads nothing until the answer is yes.
  `--dry-run` prints the plan, and `--no-speech-model` skips the optional voice
  model. Streamed pip output shows every real file and size as it arrives.
- `claireai --setup` reports the same host/package/model/voice facts without
  downloading; `claireai --setup --install` offers the missing parts, including
  the local faster-whisper model, with the same consent. A declined prompt
  leaves the install untouched and says how to resume.
- The `desktop` extra is now the complete application: customtkinter, torch and
  faster-whisper. Installing it no longer produces a window whose Voice page
  cannot import its own runtime.
- `claireai --update` reconciles the desktop extra after replacing the code, so
  a dependency added by an update is installed rather than left for the next
  confusing failure.

### Fixed — the app page could not send a message

- (historical, browser surface removed) The composer sent only on Ctrl+Enter, and the answer was written into a
  history box far down the page, so a typed message looked ignored. Enter now
  sends, Shift+Enter adds a line, the message appears immediately in a
  transcript beside the composer, and any failure appears there as a bubble
  carrying the server's own text.
- An unexpected error in a local API request closed the socket with no reply,
  which the page could only report as "Failed to fetch". It now answers `500`
  with the exception type and prints the traceback to the terminal.

### Fixed — `claireai-desktop` failed before it could explain itself

- `claireai-desktop` is a core console script, but `claireai.desktop.app`
  imports `customtkinter` at module level, so a plain `pip install claireai`
  ended in a `ModuleNotFoundError` traceback with no way to run the check that
  was supposed to produce a message. The script now goes through a launcher
  that imports nothing optional: it reports the missing `desktop` extra
  (`pip install "claireai[desktop]"`) or missing Tk bindings (`python3-tk` on
  Debian/Ubuntu) as one line on stderr and exits 2. `claireai --desktop` and
  `python -m claireai.desktop` (the command the autostart entry runs) take the
  same path. A complete install still opens the window as before.

### Changed — a proper desktop-app surface

- The presence page is rebuilt around the layout these apps actually use: a
  sidebar with a primary New chat action, recent conversations and section
  navigation; a minimal top bar with the conversation title and a compact
  status chip; a centred, bottom-anchored transcript with plain assistant text
  beside a gradient avatar and a soft bubble for your own messages; and a
  floating composer card that grows as you type and sends with a circular
  button. The five panels reuse the same token set (one neutral ramp, one
  accent, two radii) instead of hand-picked colours per component.
- The empty state is a real one now: a short explanation and four concrete
  starting points that fill the composer, instead of a sentence stranded in
  the middle of the window. Long answers are capped at a comfortable reading
  measure, and scrollbars are styled for both Chromium and Firefox (Firefox
  drew a bright system scrollbar through the chat column).
- On a phone the sidebar becomes a brand row with the new-chat button on the
  right and a wrapping tab row beneath it, rather than an icon wedged between
  the name and the tabs.

### Fixed — the wheel shipped an app that could not start

- The page lived at `desktop/presence/index.html`, outside the Python package,
  so `pip install claireai` produced an app that resolved no page at all:
  `claireai --setup` reported "this build did not include the app page" and
  exited 2. The wheel build now copies the page to `claireai/web/index.html`,
  the location `app.web_page_path()` already checked first. Verified by
  installing the built wheel in a clean virtual environment and serving the app
  from `site-packages`.

### Changed — one page instead of one long scroll

- The presence page is an app shell now: a fixed rail switches Chat,
  Preferences, Directives, History and Controls, each panel scrolls on its own,
  and the chat transcript fills the window with the composer pinned underneath.
  The layout no longer stacks five cards down a 520 px column. Panels are
  deep-linkable (`/#history`), and on a phone the rail wraps instead of
  scrolling its last tabs out of sight.
- A transcript longer than the window stays scrollable: the old bottom-aligned
  layout pushed the overflow above the scrollport, so the first message could
  not be reached at all.

### Fixed — the vision extra installed without its runtime

- `pip install "claireai[vision]"` resolved cleanly, passed `pip check`, and then
  failed at the first image: the extra listed only `transformers`, while
  `vision.py` imports torch and Pillow. Both are part of the extra now.
- An earlier note here said transformers 5 required `timm` and therefore
  `torchvision`. That was wrong: transformers lists `timm` and `torchvision`
  only as extras, and the failing `pip check` in this environment came from a
  `timm` installed by hand that nothing here requires. CLIP was then run on both
  4.57.6 and 5.22.1 — identical rankings, identical scores — so the cap is
  `<6` (the next major), not `<5`.
- Dependencies carry upper bounds now, so installing ClaireAI cannot silently
  upgrade a package the owner already relies on, and `claireai[all]` installs
  every optional runtime in a single resolution instead of one extra at a time.
- `requirements.lock.txt` pinned the project with an absolute
  `-e file:///home/...` path and listed `pip` itself, so replaying it on another
  machine failed or downgraded the installer running it. It is portable, and it
  now says what it is a snapshot of and what it leaves out.

### Added — a signed approval for runs with no terminal

- Unattended runs still refuse writes, but can now consume one signed
  single-use token per action (`--preauthorize` to issue, `--approval-file` to
  use). The token names the tool, every argument and the target, expires, and is
  moved to `approvals/consumed/` before the action runs, so a replay finds
  nothing. The signing key is `approval.key`, mode `0600`; a refusal prints the
  exact command that would authorise the action the model proposed.
- `claireai --setup` prepares the data directory and database and reports the
  model, voice and page situation, so a fresh install can see why chat or speech
  would not work yet without running anything.

### Fixed — the app page was only found in a checkout

- `app.HTML_PATH` was hard-coded to `<repo>/desktop/presence/index.html`, which
  cannot exist in an installed wheel, so `pipx install claireai` would start and
  immediately fail with "the app page is missing". The page is now resolved from
  the packaged location first, then the checkout, with `CLAIREAI_WEB_DIR` as an
  explicit override.

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

### Changed — the tool-call harness covers the calendar

- `scripts/tool_call_cases.json` is now 19 cases / 165 checks, including
  `list_appointments` and `add_appointment`. A model that cannot call the calendar
  is a number instead of a surprise; chat-v4 measures 68/165 on the larger file.
  Case counts change the denominator, so compare check totals only within the same
  file.
- `build_chat_dataset.py` learned the calendar tools. They were missing from its
  `TOOL_LINES` table, so a calendar request routed to an empty tool list and the
  corpus taught a call with no tools shown. It also widened the smalltalk and
  summary pools, and the comment no longer claims that raising a weight adds
  records a pool does not have — a bucket smaller than its quota contributes all
  of itself, so the pool is the real limit.

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

### Added — calendar tools on the tool-call evaluation harness

- `scripts/tool_call_cases.json` now has 19 cases and 165 checks, including three
  calendar cases (`agenda-tomorrow`, `agenda-friday`, `add-appointment`) run
  through the real parser and the production system prompt. The harness was
  extended rather than editing existing cases so the calendar scores stay
  comparable to the earlier baselines.
- `brain-lab/build_chat_dataset.py` was fixed: the calendar tools already had
  `TOOL_LINES` and `TOOL_ARGS` entries, but the `TOOL_REQUESTS` table — the
  staged prompt table the builder uses — did not list them, so a calendar request
  routed to an empty tool list. That is repaired in the same build that
  introduces `brain-lab/data/chat-v9`.
- The calendar cases are generalisation tests, not memorisation tests: the corpus
  builder's new `calendar_tasks` uses day phrases and `HH:MM` times nowhere else
  in the corpus, so the training prompts deliberately differ from the eval case
  users. A new checkpoint is installed only if it passes the install gate.

### Changed — corpus builder: calendar rebuilt from scratch for chat-v9 and article inflection fixed

- `brain-lab/build_chat_dataset.py`: the calendar entries were removed from the
  staged prompt table (`TOOL_REQUESTS`) and rebuilt as a dedicated
  `calendar_tasks(rng)` generator with per-prompt routing, so the calendar has
  its own mass from the first training step instead of arriving late in a
  warm start. The generator adds `CALENDAR_DAYS` (word days, fixed dates and
  relative phrases such as `last friday` / `next monday`), `CALENDAR_LIST_WORDS`,
  `CALENDAR_TITLES`, `CALENDAR_TIMES` and `CALENDAR_ADD_WORDS`, with four
  rotating time slots per day/title pair so the model sees every day name with
  several titles and every title at several times instead of one memorised
  combination.
- The training prompts stay different from the eval cases: a new `"what's on
  {day}"` list-phrase was added to `CALENDAR_LIST_WORDS` precisely so the corpus
  does not duplicate the three eval users.
- Article inflection is now correct (`"a tooth"`, `"an interview"`) instead of
  always producing `"a an"`; the new calendar bucket is weighted 2 and the
  v9 build produces `train=13968`, `validation=736`, `calendar=882` with
  `routing_miss=2`, and no calendar training prompt equals any
  `tool_call_cases.json` user prompt.
- DATASET_VERSION is now `claireai-chat-v9` and `--output` defaults to
  `brain-lab/data/chat-v9`. The chat-v9 run itself is not in git: model files,
  the `.partial` checkpoint, eval logs and the corpus dir are gitignored or never
  staged. The run did not pass the install gate, so the installed scratch
  checkpoint is still chat-v4.

### Planned and specified — presence renderer and machine access

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
