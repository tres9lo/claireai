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
credential redaction, the Ollama adapter via `httpx.MockTransport` (including
tool-call parsing and malformed-call rejection), CLI task/audit bookkeeping, the
evaluation checkers/runner, the tool registry, the five read-only file tools, the
agent loop (refusals, invalid arguments, empty replies, dry run, step limit), the
approval flow (brokers, the CLI prompt, non-TTY refusal, denial logging, shared
`approval_ref`), the six write tools, the presence stream and its HTTP server,
and a security suite for the folder allow-list in `tests/security/`.

The allow-list tests are the ones to keep strict: they assert that parent
traversal, absolute paths outside the list, and symlinks that resolve outside it
are all refused, and that an empty allow-list allows nothing.

**Labelled honestly:** every Ollama test is **mocked** — it proves our request
shape and error mapping, not that a real runtime answers. The real proof is the
CLI smoke test in `docs/setup/linux.md`.

No Windows results exist yet; see `docs/platform-support.md`.

## Smoke check

```bash
assistant/.venv/bin/python scripts/smoke.py              # needs Ollama
assistant/.venv/bin/python scripts/smoke.py --skip-model # no model needed
```

Ten checks over the assembled system: settings, migrations, model reachability,
a real chat turn, the tool loop, a write refused without an approval channel, the
presence stream, and an UPDATE on `task_log` being rejected by the append-only
triggers. It uses `pathlib` and a temporary directory only, so it should run
unchanged on Windows — that has not been done yet, so treat Windows as unproven.

## Presence server

```bash
assistant/.venv/bin/python scripts/presence_preview.py --demo
curl -s http://127.0.0.1:8765/state
curl -s --max-time 2 -N http://127.0.0.1:8765/events
```

The HTTP tests in `tests/unit/test_presence_server.py` start the server on port 0
and assert `/`, `/state`, `/events`, `/healthz`, a 404 for unknown paths and the
loopback-only refusal. If they ever look flaky, remember the stream sends
`retry:` before the first event — read until a data frame arrives.

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

Current categories: `intent` (routing), `slots` (argument extraction),
`tool_call` (proposed JSON tool call), `summary` (three-section daily summary),
`safety` (refuses to bypass approval, does not invent private data).

Reading the output: `PASS` = all runs passed, `FLAKY` = some runs passed,
`FAIL` = none passed. The report ends with `cases stable: X/N   checks passed: A/B`.

Exit codes: `0` all stable, `1` at least one case failed or flaked, `2` bad
configuration or case file, `3` runtime unavailable / model not installed.

Results are written to `<data-dir>/eval/eval-<timestamp>.json` (development
default `runtime-data/eval/`), which is git-ignored. Copy the numbers into
`PROGRESS.md` — do not commit the raw JSON.

Rules for the case file:

- Add a case **before** depending on a model behaviour in a feature.
- Never delete or loosen a failing case to make the run green; record the failure
  and design around it (deterministic code instead of free-form model output).
- Keep cases small and unambiguous; a flaky case usually means the prompt or the
  output contract is ambiguous, not that the model is random.

## Proving inference is local (offline check)

The roadmap asks for a repeat with the network disconnected. If the owner cannot
drop their connection (it may be carrying the coding session), the same
condition can be created for the process alone: run the CLI inside a network
namespace that has no route anywhere, bridged to the local runtime over a Unix
socket. Nothing leaves the machine, and the model still answers.

```bash
# scratch relay, outside git: runtime-data/tmp/relay.py unix-out|tcp-out
python runtime-data/tmp/relay.py unix-out --socket "$PWD/runtime-data/tmp/ollama.sock" &
unshare -rn bash -c '
  ip link set lo up
  # prove it: no route, egress blocked, DNS broken
  # then bridge 127.0.0.1:11434 to the Unix socket and run the real CLI
  python -m claireai.cli --data-dir ./runtime-data --once "what does a tokenizer do?"'
```

Record it in `PROGRESS.md` with the egress/DNS lines, the answer and the audit
row. State the limit honestly: this proves the client process used no network
and that the runtime serves from local model files; it does not audit the
runtime's own syscalls.
