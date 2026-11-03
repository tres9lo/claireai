# Testing

Two separate things get tested here, and they answer different questions:

1. **Our code** — `pytest` unit tests. Deterministic, no model needed.
2. **The model** — `scripts/eval_model.py`. Measures what the configured local
   model can actually do, scored by fixed rules.

Neither one is a substitute for the other. A passing suite does not prove the
model can drive a feature, and a good model score does not prove our validation,
permissions or logging work.

## Unit tests

```bash
cd assistant && .venv/bin/python -m pytest
```

Run it from `assistant/`, not the repository root: the pytest configuration
(including `asyncio_mode = "auto"`) lives in `assistant/pyproject.toml`, so from
the root the async tests fail with "async def functions are not natively
supported" and the run reports 23 failed / 52 passed. CI uses
`working-directory: assistant` for the same reason.

Covered today: settings validation and loopback enforcement, migration runner
(idempotency, checksum tampering, statement splitting with trigger bodies),
schema safety (foreign keys, append-only audit triggers), task state machine,
credential redaction, the scratch model adapter and checkpoint validation, CLI task/audit bookkeeping, the
evaluation checkers/runner, the tool registry, the five read-only file tools, the
agent loop (refusals, invalid arguments, empty replies, dry run, step limit), the
approval flow (brokers, the CLI prompt, non-TTY refusal, denial logging, shared
`approval_ref`), the six write tools, the presence stream and its HTTP server,
and a security suite for the folder allow-list in `tests/security/`.

The allow-list tests are the ones to keep strict: they assert that parent
traversal, absolute paths outside the list, and symlinks that resolve outside it
are all refused, and that an empty allow-list allows nothing.

Scratch-model tests cover checkpoint loading and response parsing. They do not
measure conversational capability; use the fixed evaluation harness for that.

No Windows results exist yet; see `docs/platform-support.md`.

## Smoke check

```bash
assistant/.venv/bin/python scripts/smoke.py              # uses a configured scratch checkpoint
assistant/.venv/bin/python scripts/smoke.py --skip-model # no model needed
```

Fourteen checks over the assembled system: settings, migrations, model
reachability, a real chat turn, the tool loop, tool routing narrowing the schema
set, a write refused without an approval channel, the presence stream, the daily
summary being built and saved, the read-only `/today` page rendering with no
script or form, retention removing only a stale trash batch, and an UPDATE on
`task_log` being rejected by the append-only triggers. It uses `pathlib` and a
temporary directory only, so it should run unchanged on Windows — that has not
been done yet, so treat Windows as unproven.

## Presence server

```bash
assistant/.venv/bin/python scripts/presence_preview.py --demo
curl -s http://127.0.0.1:8765/state
curl -s --max-time 2 -N http://127.0.0.1:8765/events
```

The HTTP tests in `tests/unit/test_presence_server.py` start the server on port 0
and assert `/`, `/state`, `/events`, `/healthz`, `/today`, a 404 for unknown
paths and the loopback-only refusal. If they ever look flaky, remember the stream
sends `retry:` before the first event — read until a data frame arrives.

The `/today` tests assert the page writes nothing (the database file's mtime is
unchanged after repeated requests) and that the CSP header is present. The local
control POST requires a loopback Host, same-origin header, and JSON body.
`tests/unit/test_today.py` adds that a request containing `<script>` comes back
escaped, and that `open_read_only` raises on a write attempt.

## Retention

`tests/unit/test_retention.py` builds dated trash batches and summary files and
asserts what is planned, what is kept, and that planning alone deletes nothing.
It covers the refusals (symlinked batch, folder name that is not a timestamp, a
window below one day) and that `task_log` is never named for removal.

## Model evaluation harness

```bash
# Wiring check: proves loading/running/scoring/reporting work. NOT a model result.
assistant/.venv/bin/python scripts/eval_model.py --stub

# Real measurement against the configured local model (temperature 0)
assistant/.venv/bin/python scripts/eval_model.py --runs 3
assistant/.venv/bin/python scripts/eval_model.py --model qwen2.5:1.5b --runs 5
```

Cases live in `scripts/model_eval_cases.json`; each one declares a category, a
prompt (with the output format it must follow) and an expectation. Scoring kinds:

| `expect.kind` | Meaning |
| --- | --- |
| `exact` | reply equals `value` after normalising case/punctuation |
| `json` | reply contains a JSON object matching `equals` / `contains` on dotted paths |
| `contains_all` | every phrase in `value` appears |
| `contains_any` | at least one phrase in `value` appears |
| `refuses` | a phrase in `value` appears (a refusal) **and** no phrase in `forbidden` does (no claimed approval) |

Current categories: `intent` (routing), `slots` (argument extraction),
`tool_call` (proposed JSON tool call), `summary` (three-section daily summary),
`safety` (refuses to bypass approval, does not invent private data).

Reading the output: `PASS` = all runs passed, `FLAKY` = some runs passed,
`FAIL` = none passed. The report ends with `cases stable: X/N   checks passed: A/B`.

Exit codes: `0` all stable, `1` at least one case failed or flaked, `2` bad
configuration or case file, `3` runtime unavailable / model not installed.

### Current model status

The current architecture is a small transformer trained from random
initialization. No task-relevant corpus or capable assistant checkpoint is
bundled, so there are no current model capability results to report. Historical
predecessor-model measurements are in `PROGRESS.md`; they do not describe the
active runtime. Evaluate every newly trained checkpoint with the fixed harness
before relying on it.

Two categories read differently from the rest:

- `summary` measures a *model's* three-section output, not the product. The
  shipped daily summary is built by `claireai.summary` from `tasks` and
  `task_log` with no model call at all, precisely because a model asked to
  summarise invents events. A failing `summary` case is evidence for that
  design, not a product defect.
- `refuses-to-bypass-approval` fails on both models: the refusal is in the right
  spirit but does not use the word the check looks for. That is a limitation of
  the check, not of the runtime.

Results are written to `<data-dir>/eval/eval-<timestamp>.json` (development
default `runtime-data/eval/`), which is git-ignored. Copy the numbers into
`PROGRESS.md` — do not commit the raw JSON.

Rules for the case file:

- Add a case **before** depending on a model behaviour in a feature.
- Never delete or loosen a failing case to make the run green; record the failure
  and design around it (deterministic code instead of free-form model output).
- Keep cases small and unambiguous; a flaky case usually means the prompt or the
  output contract is ambiguous, not that the model is random.
