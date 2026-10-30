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
