# Contributing to ClaireAI

Thanks for taking the time to look. ClaireAI is a small, safety-focused project, so the rules below are stricter than a typical app.

## Before you start

- Open an issue describing the change, especially for anything that touches the permission layer, the audit log, the sandbox or network access.
- Small fixes (typos, docs, a failing test) can go straight to a pull request.

## Development setup

```bash
git clone https://github.com/tres9lo/claireai.git
cd claireai

uv venv --seed --python 3.12 assistant/.venv
uv pip install --python assistant/.venv/bin/python -e "assistant[dev]"

cd assistant
python -m pytest
```

The model smoke check is skipped unless a scratch checkpoint exists. To run the
dependency-free runtime checks:

```bash
assistant/.venv/bin/python scripts/smoke.py --skip-model
```

## What a pull request needs

1. **A failing test first** for behaviour changes — the test suite is the specification.
2. **A clean run** of `python -m pytest` from `assistant/`.
3. **A changelog entry** under `## [Unreleased]` in `CHANGELOG.md`, written in the existing voice: what changed and why it matters.
4. **Docs updated** when a setting, command, tool or boundary changes.

## Safety invariants

These are not preferences. A change that weakens one of them without an explicit design discussion will be declined.

- The model is untrusted. It proposes; the runtime validates, authorises and executes.
- Validation, the folder allow-list and the risk policy run against the **full** registry, never a filtered view.
- Deny-by-default stays the default. No feature may be enabled by the absence of configuration.
- `task_log` stays append-only at the database level. Do not add code that assumes it can update or delete audit rows.
- Nothing is deleted permanently; write tools move to the recoverable bin.
- Non-interactive runs refuse writes. They never self-approve.
- No hosted model provider and no API keys in the runtime path.
- New capabilities (camera, screen, input) must go through `claireai.permissions.check_capability` rather than implementing their own policy.

If a change requires breaking one of these to be useful, say so in the issue and describe the compensating control — for example the per-run token and nonce that approving over loopback HTTP would need.

## Model training: help wanted

The assistant's brain is a from-scratch model trained on CPU inside `brain-lab/`. It is the part of this project I know least well, and it is the current bottleneck: the model can hold a short conversation, do simple arithmetic and emit valid tool calls on easy prompts, but it is nowhere near a general assistant.

Useful contributions right now, roughly in order of value:

- **Better training recipes.** Optimizer settings, warmup/decay schedules, and width/depth/context trade-offs at a fixed CPU budget.
- **Better data mixes.** Which proportions of task data, dialogue, prose and synthetic examples produce a model that *answers* instead of rambling. Measured, not asserted.
- **Evaluation that predicts usefulness.** `scripts/eval_model.py` and `scripts/conversation_eval.py` are the bar nothing ships without. If a case is badly written or a metric is misleading, that is a real bug.
- **Cheaper inference.** Quantisation or batching that keeps replies correct on a 2-core laptop CPU.

If you would rather discuss than code, open an issue with what you would change and why. Comparing notes is welcome — I would rather learn the right way to train this than defend the current one.

## Code style

- Python 3.11+, standard library where it is sufficient.
- Type hints on public functions; `pydantic` for anything crossing a trust boundary.
- No new runtime dependency without a note in the pull request explaining what it replaces.
- Tests live under `assistant/tests/unit` or `assistant/tests/security`.

## Commits

Short imperative subject lines that describe the change, for example:

```
Add retention planning for trash batches
Refuse symlinks that escape the allow-list
```

## Reporting a security issue

Please do not open a public issue. See [SECURITY.md](SECURITY.md).
