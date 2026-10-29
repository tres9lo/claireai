# Progress log

## 2026-10-29 — Session 1: repository setup + Month 1, Week 1 foundations

**Scope taken:** Initial Repository Setup (steps 1–6, 9) and Month 1 Week 1 —
configuration, SQLite schema via migrations, task/audit repositories, `Brain`
interface, `OllamaBrain` adapter, CLI chat. No tools, permissions, approvals,
API server, desktop or mobile work (those are Weeks 2–4).

### Done

- Confirmed naming with the owner: **repo folder, Python package, project name
  and assistant name all stay `claireai`**. Nothing user-facing is branded yet;
  package/application IDs remain provisional.
- Checked the existing scaffold first: folder skeleton and empty `docs/*` files
  already existed, nothing was overwritten. Git repo had **no commits**.
- `.gitignore` extended with the roadmap list plus `*.pem`, `*.key`, `*.p12`,
  `secrets/`, `credentials/`, `.mypy_cache/`, coverage output, SQLite WAL/SHM
  files and `*.bin`.
- `assistant/pyproject.toml` (hatchling, src layout, console script `claireai`),
  dev extras, pytest config. Resolved versions frozen in
  `assistant/requirements.lock.txt`.
- `claireai.config.Settings`: `CLAIREAI_*` env/.env config, strict field set,
  timeout bounds, and a model endpoint that **must be loopback** unless
  `CLAIREAI_ALLOW_NON_LOOPBACK_OLLAMA=true`. Data directory comes from
  `platformdirs` (OS application-data dir), overridable for development.
- `claireai.storage`: `Database` (foreign keys, 5 s busy timeout, explicit
  transactions), a migration runner with per-migration SHA-256 checksums that
  rejects edits to already-applied migrations, and statement splitting that
  keeps trigger bodies intact.
- `claireai/migrations/0001_initial.sql`: `tasks`, `task_log` (append-only via
  UPDATE/DELETE-blocking triggers), `summaries`, `notes`, plus indexes.
- `claireai.storage.tasks.TaskRepository` with an explicit state machine
  (`queued → planning → awaiting_approval → running → succeeded|failed|cancelled`,
  `interrupted` reconcilable) and rejection of invalid transitions.
- `claireai.audit`: credential redaction + truncation, and an append-only
  `AuditLog` reader/writer.
- `claireai.brain`: provider-neutral `Brain` protocol, `Message`,
  `BrainResponse`, `BrainHealth`, error types, and `OllamaBrain` over Ollama's
  local `/api/chat` + `/api/tags` (checked against
  <https://docs.ollama.com/api/chat> on 2026-10-29). No hosted provider, no API
  key path.
- `claireai.cli`: one-shot (`--once`) and interactive chat. Every turn creates a
  task row and exactly one audit row; failures mark the task `failed` and audit
  the redacted error. Exit codes: 0 ok, 2 invalid configuration, 3 model runtime
  unavailable.
- Docs: `README.md`, `.env.example`, `docs/setup/linux.md`,
  `docs/setup/windows.md`, `docs/platform-support.md`,
  `docs/model-and-data-licenses.md` (Qwen2.5 1.5B = Apache-2.0, verified).

### Verified (real output, this machine)

Environment: Zorin OS 18.1, x86_64, 7.1 GiB RAM, Python 3.12.3, Git 2.43.0,
Node 24.20.0. Rust and Ollama were not installed at the time of these runs.

Unit tests (mocked Ollama via `httpx.MockTransport`, in-memory-free temp SQLite):

```
$ cd assistant && .venv/bin/python -m pytest
.................................................                        [100%]
49 passed in 0.88s
```

Migration + failure path (Ollama absent):

```
$ CLAIREAI_DATA_DIR=./runtime-data python -m claireai.cli --once "Hello, are you local?"
Applied migrations: 0001_initial
database: runtime-data/db/claireai.sqlite3
brain: ollama @ http://127.0.0.1:11434 model=qwen2.5:1.5b
ERROR: local model runtime unavailable — cannot reach Ollama at http://127.0.0.1:11434: ConnectError
Install/start Ollama, then retry. No hosted fallback is used.
EXIT=3
```

Second run applied no migrations (idempotent) and exited 3 again. Schema check:

```
['notes', 'schema_migrations', 'sqlite_sequence', 'summaries', 'task_log', 'tasks']
[('0001_initial', '2026-10-29T13:25:36')]
```

Invalid configuration path:

```
$ claireai --once "ping" --base-url http://192.168.1.50:11434
Invalid configuration:
  Value error, CLAIREAI_OLLAMA_BASE_URL points at a non-loopback host ('192.168.1.50'). ...
EXIT=2
```

Ollama model tag checked on the library page: `qwen2.5:1.5b` exists (Q4_K_M,
986 MB, Apache-2.0, lists `tools` support). It has **not** been pulled or run
here yet.

### Blockers / not verified

1. **Real local inference not yet proven on this machine.** Ollama was not
   installed during this session (the owner is installing it, with Rust/cargo).
   The Week 1 acceptance check "answers a question without an API key and logs
   it" is therefore **not yet passed**; everything around it is.
2. `python3.12-venv` (system package) is missing, so `python3 -m venv` produces a
   venv without pip. Worked around with `uv venv --seed`; the standard-library
   route is documented but untested here.
3. Windows: nothing has been run on Windows. Setup steps are written but
   unverified — no Windows claims are made.
4. Machine has 7.1 GiB RAM (~2.6 GiB available). Expect the 1.5B model to load
   slowly; a larger model may not fit comfortably.
5. Redaction is pattern-based best effort, not a guarantee.

### Next single task

Once Ollama is installed and the model pulled: run the real smoke test
(`ollama pull qwen2.5:1.5b`, then `claireai --once ...`), record RAM use,
latency and the task/audit rows in this file, then disconnect the network and
repeat to prove inference is local. Only after that, Month 1 Week 2 (tool
registry, read-only file tools, allow-list) with owner approval.
