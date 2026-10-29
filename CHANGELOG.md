# Changelog

All notable changes are recorded here. Versions follow the roadmap's phases;
nothing has been released yet.

## [Unreleased]

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
