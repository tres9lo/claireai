# Post-training gate runbook

This gate measures a candidate; it never installs one. Run it after training has
finished, not while another run is using the CPU.

## Preconditions

- The candidate checkpoint exists and is a single `.pt` file.
- The installed checkpoint is available for a same-run-count comparison.
- No training process is active.
- The tool-call suites pass their unit tests.

## One command

From the repository root:

```bash
assistant/.venv/bin/python scripts/run_gate.py \
  --checkpoint runtime-data/models/candidates/claire-chat-v15-full.pt
```

The wrapper runs, in order:

1. `scripts/eval_model.py` — one-shot formatting and safety cases.
2. `scripts/conversation_eval.py` — multi-turn conversation cases.
3. `scripts/tool_call_eval.py` with `scripts/tool_call_cases.json` — the stable
   standard tool gate.
4. `scripts/tool_call_eval.py` with `scripts/tool_call_cases_extended.json` —
   direct-answer, missing-argument and refusal probes.

Reports are written to `runtime-data/eval/gate-<timestamp>/`, including
`gate.json`. Use `--out-dir` for a fixed path, `--dry-run` to inspect the
commands, and `--fail-fast` to stop at the first failing stage.

## Pass rule

The candidate passes only when every stage exits `0`:

- model cases: every case stable;
- conversation cases: no regression against the installed checkpoint at the
  same `--runs` count;
- standard tool cases: every case stable;
- extended tool cases: every case stable, with the per-category breakdown
  inspected rather than hidden behind the total.

A runtime/config failure (`2` or `3`) is not a quality result. Fix the setup and
rerun the stage.

## Before installing

1. Compare both checkpoints at the same run count. Single-run differences are
   noise.
2. Check the extended tool breakdown: `format`, `name`, `arguments`, `no_call`
   and `final_answer` each need to be credible.
3. Confirm the conversation suite did not regress on greetings, refusals or
   tool-call follow-ups.
4. Install only after the owner explicitly promotes the candidate.

## Corpus check before the next run

Audit the next corpus without loading a model:

```bash
assistant/.venv/bin/python brain-lab/audit_corpus.py \
  --corpus brain-lab/data/chat-v16 \
  --out runtime-data/eval/corpus-audit.json \
  --strict
```

Install nothing if the audit reports over-context records, over-target records,
malformed or unknown tool calls, held-out overlap, duplicates or train/validation
overlap.
