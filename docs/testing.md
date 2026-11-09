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

### Tool-call accuracy

```bash
assistant/.venv/bin/python scripts/tool_call_eval.py --runs 3
```

A separate suite, because tool calls fail in a way answer quality does not
predict: a model can write a good sentence and still emit a call the runtime
cannot read. Cases live in `scripts/tool_call_cases.json` and carry the tool
names offered for that request. Each run is scored on three checks — `format`
(exactly one call, no leftover prose), `name` (the tool the request needs) and
`arguments` (the values the request names) — and a case only passes when every
run passes, so a model that lands a call one time in three does not look correct. Cases cover
the file tools, the browser tools and the calendar (`list_appointments`,
`add_appointment`). Adding a case changes the check total, so compare check
counts only between checkpoints measured on the same case file — the calendar
cases took it from 138 to 165.

It goes through `ScratchBrain` with the runtime's own system message, which is
what makes it a measurement of the assistant rather than of the model in
isolation. That distinction is not academic: the chat-v3 checkpoint emitted
correct calls when given the corpus's system sentence and gibberish when given
the runtime's, because the two had drifted apart. Fixing that is one import and
one regression test — `assistant/tests/unit/test_corpus_prompt.py`.

Rule: **the corpus never writes its own system prompt.** `build_chat_dataset.py`
imports `claireai.agent.system_message` and varies only the folder clause, so a
change to the assistant's instructions reaches training data automatically.

Rule: **a tool-call record offers only the tools routing would offer.** At
inference `claireai.routing` narrows the schemas per request, so training a
record on all nineteen tools and serving it four is the same drift in a new
place. The builder rewrites each record's `TOOLS:` block to its routed subset and
counts the exceptions in the manifest under `routing`. A record whose routed set
**omits the tool it answers with** is counted as `routing_miss` and left alone —
rewriting it would teach an answer with a tool that was never shown. chat-v6 is
at 5 misses out of 7,117 tool records; chat-v5 was at 207. A jump means routing
has started hiding a tool the corpus expects the model to call.

### Current model status

The current architecture is a small transformer trained from random
initialization on this repository's own corpus; no capable assistant checkpoint
is bundled with the source. Measured results for each installed checkpoint,
including the ones that were not installed, are in `PROGRESS.md`. Evaluate every
newly trained checkpoint with the fixed harnesses (`eval_model.py`,
`conversation_eval.py`, `tool_call_eval.py`) before relying on it, and compare
against the previous checkpoint rather than against zero.

Compare at the same run count. `conversation_eval.py` samples at temperature 0.2,
so a single run is a sample, not a result: chat-v4 measured 11/21 in one run and
14/21 at `--runs 3`. A one-case gap between checkpoints is noise; a four-check
swing on `tool_call_eval.py` is not.

The install gate is do not relax. A new checkpoint replaces
`runtime-data/models/claire-scratch.{pt,json}` only if it beats chat-v4 on tool
checks **and** does not lose conversation cases, both measured at the same run
count. A one-case conversation difference is noise; a four-check tool swing is
not. chat-v9 was measured at the same run counts as v4 (one eval run, three
conversation runs, three tool-call runs on the same 19-case / 165-check file)
and did not pass the gate: it scored 67/165 tool checks (v4: 68) and 11/21
conversation cases (v4: 14), so scratch stayed on chat-v4.

A from-scratch calendar run is not a pass on its own. chat-v9 included the
calendar records from the first step and still scored 0/3 stable on the three
calendar cases, with the right tool name appearing once per case but the
day-name and `starts_at` arguments garbled by overlapping date tokens and
rotated time tokens. That is expected at this vocabulary — the same conclusion
that the calendar cases gave for chat-v7 — and it is why the product answers
agenda questions in code rather than waiting on the model. Do not read a
passing calendar case in a later run as evidence the calendar is product-ready;
calendar answers stay deterministic in `answer_agenda`.

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

## Dataset and visual-grounding experiments

`brain-lab/build_dataset.py` creates the tracked MIT synthetic starter corpus.
Its train, validation and test files are separate; `train.py` learns tokenizer
merges only from train and never reads test. For external image experiments,
`brain-lab/fetch_gui_images.py` pins and verifies the synthetic-only licensed
subset, then `brain-lab/vision_train.py` trains a separate random-init CNN and
reports click-within-target metrics on held-out screenshots. That visual model
is not integrated into chat or screen control; its metrics are research results,
not a product capability claim.

## Tool-call baseline (2026-11-09)

`scripts/tool_call_eval.py` asks for one thing: does the model emit a call the
runtime can read, with the right tool and arguments. It reports three checks per
case and only counts a case as passing when all runs pass.

Run it with `--sandbox home` (the default) unless you specifically mean to test a
fresh install with nothing allow-listed. `--sandbox empty` reproduces the old
prompt, in which no file tool can run, so a correct answer there is not a call;
that prompt measured a state no working install is in and once made a rejected
checkpoint look far worse than it was.

| checkpoint | suite prompt | checks | stable cases |
| --- | --- | --- | --- |
| chat-v4 (installed) | `--sandbox home` | 75/165 | 0/19 |
| chat-v10 | `--sandbox home` | 74/165 | 6/19 |
| chat-v4 (installed) | `--sandbox empty` | 68/165 | 0/19 |
| chat-v10 | `--sandbox empty` | 30/165 | 0/19 |

Numbers are only comparable within a sandbox mode. A case counts as passing only
when every one of the three runs passes, so a single-run comparison is noise.
