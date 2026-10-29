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
