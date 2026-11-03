# Privacy

ClaireAI is local-first by design. The short version: **your data stays on the machine you installed it on**, and the project collects nothing about you.

## What ClaireAI stores

Everything lives under `CLAIREAI_DATA_DIR` (default: the OS application-data directory for `claireai`):

| Data | Where | Why |
| --- | --- | --- |
| Tasks and their status | SQLite `tasks` | So a run can be followed and summarised |
| Append-only action log | SQLite `task_log` | So every tool call has a reviewable trail |
| Daily summaries | SQLite `summaries` + `summaries/YYYY-MM-DD.md` | So you can read what happened without a terminal |
| Files moved to trash | `trash/<timestamp>/` | So a write can be reversed |
| Downloads and browser profiles | `downloads/`, profiles directory | Where the browser tools put their output |

Nothing is uploaded anywhere by the application.

## What ClaireAI does not do

- No telemetry, analytics, crash reporting or usage counters.
- No account, sign-in or licence check.
- No hosted model provider and no API keys in the runtime path.
- No background sync, remote backup or cloud storage.

The only network traffic the runtime initiates is to the model endpoint, which must be **loopback** (`127.0.0.1` / `localhost`) unless you explicitly override it. Browser tools fetch URLs you ask them to fetch; those requests go directly from your machine to the site, with no proxy in between.

## What the model sees

Text you type, the content of files inside allow-listed folders that a tool reads, and tool results. That content is sent to your local model process, and it is recorded in the task log as part of the conversation trail. Keep that in mind before pointing a tool at a folder with material you would not want written to a local database.

Secrets are redacted by pattern (tokens, keys, passwords) and long strings are truncated before storage, but redaction is best-effort, not a guarantee. See [docs/architecture/threat-model.md](docs/architecture/threat-model.md).

## Retention and deletion

```bash
# Show what is older than 30 days — nothing is deleted
claireai --prune

# Delete it and record the run
claireai --prune --apply
```

Retention removes old trash batches and summaries. **The audit log is deliberately excluded**: it is append-only at the database level, and giving that up would cost more than the disk space it saves.

To delete everything, stop the assistant and remove the data directory. There is no remote copy to clean up.

## Third-party services

- **Ollama** runs locally. Its own logs and model files are governed by its configuration, not ClaireAI's.
- **Model weights** carry their own licences, listed in [docs/model-and-data-licenses.md](docs/model-and-data-licenses.md).
- **GitHub** hosts this repository. Reading it, opening issues or commenting is covered by GitHub's privacy policy and is unrelated to the application's runtime behaviour.

## Changes

This document describes the code as it exists on `main`. When a feature changes what is stored or where it goes — the planned FastAPI service, the desktop shell, camera or screen capture — this file is updated in the same pull request.
