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

Covered today: settings validation and loopback enforcement, migration runner
(idempotency, checksum tampering, statement splitting with trigger bodies),
schema safety (foreign keys, append-only audit triggers), task state machine,
credential redaction, the Ollama adapter via `httpx.MockTransport`, CLI
task/audit bookkeeping, and the evaluation checkers/runner.

**Labelled honestly:** every Ollama test is **mocked** — it proves our request
shape and error mapping, not that a real runtime answers. The real proof is the
CLI smoke test in `docs/setup/linux.md`.

No Windows results exist yet; see `docs/platform-support.md`.

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
