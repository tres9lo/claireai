# Local API service

The API is an opt-in backend for local clients. It is not the desktop UI and it
does not replace the approval path.

```bash
pip install -e "assistant[api]"
claireai --api --api-host 127.0.0.1 --api-port 8765
# equivalent:
assistant/.venv/bin/claireai-api --host 127.0.0.1 --port 8765
```

`/health` and one-time device claiming are public. Every other route requires a
bearer token:

```bash
TOKEN=$(cat ~/.local/share/ClaireAI/api.token)
curl -s http://127.0.0.1:8765/v1/status \
  -H "Authorization: Bearer $TOKEN"
```

The administrator token is read from `CLAIREAI_API_TOKEN`, or created as a 0600
`api.token` file in the data directory. The data directory is OS-specific;
`claireai --setup` prints it. Do not put the token in a URL, a screenshot or a
repository.

## Routes

| Method | Path | Purpose |
| --- | --- | --- |
| GET | `/health` | Liveness only; no model or database work |
| GET | `/v1/status` | Provider, model, task/device/memory counts and capabilities |
| GET | `/v1/models` | Live Ollama health and model readiness |
| POST | `/v1/chat` | One synchronous chat turn |
| POST | `/v1/tasks` | Queue a background task and return its ID immediately |
| GET | `/v1/tasks` | Recent task rows |
| GET | `/v1/tasks/{id}` | One task plus its audit rows |
| POST | `/v1/tasks/{id}/cancel` | Cancel a queued or running task |
| GET | `/v1/conversations` | Recent local conversations |
| GET | `/v1/conversations/{id}` | Messages in one conversation |
| GET | `/v1/memory` | Newest local memory entries |
| POST | `/v1/memory/search` | Cosine search over local notes and turns |
| DELETE | `/v1/memory/{id}` | Delete one local memory entry |
| GET | `/v1/approvals` | Valid signed approval requests (administrator only) |
| POST | `/v1/approvals/{ref}/deny` | Revoke one pending signed request |
| POST | `/v1/approvals/{ref}/expire` | Expire one pending signed request |
| POST | `/v1/approvals/{ref}/approve` | Always 403; an API token cannot approve |
| GET | `/v1/audit` | Recent append-only audit rows (administrator only) |
| GET | `/v1/summary` | Stored or freshly built daily summary |
| POST | `/v1/summary` | Rebuild and save the daily summary |
| GET | `/v1/settings` | Non-secret effective preferences (administrator only) |
| PUT | `/v1/settings` | Update editable non-secret preferences |
| POST | `/v1/devices/pair` | Create a short-lived one-time pairing code |
| POST | `/v1/devices/claim` | Claim a code and receive a device token once |
| GET | `/v1/devices` | List local devices (administrator only) |
| DELETE | `/v1/devices/{id}` | Revoke a device token |
| GET | `/v1/events` | Server-Sent Events for task/chat progress |
| GET | `/docs` | Generated OpenAPI UI |

## Chat

```bash
curl -s http://127.0.0.1:8765/v1/chat \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"message":"Say hello","use_tools":false}'
```

```json
{
  "conversation_id": "…",
  "task_id": "…",
  "content": "Hello.",
  "model": "qwen3:1.7b",
  "duration_ms": 1234,
  "route": null
}
```

`use_tools` defaults to `true`. Safe tools run through the same registry as the
window. A tool marked `needs_approval` or `dangerous` goes through
`FileApprovalBroker`: the API caller cannot approve it with the bearer token.
Issue a signed token from a terminal for the exact action:

```bash
claireai --preauthorize create_folder --arg path=~/Documents/reports
```

If no matching token exists, the write is refused and the refusal is audited.

## Tasks

A task returns immediately with `status: "queued"`; the runtime updates it as it
moves through planning, running and a terminal state. Poll the task or subscribe
to SSE.

```bash
curl -s -X POST http://127.0.0.1:8765/v1/tasks \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"message":"List my Downloads folder"}'
```

Cancellation is cooperative for a running model call and immediately marks a
queued task cancelled. A tool that is already executing is allowed to finish its
bounded action; no new step starts.

## Devices

Pair a local client with a one-time code from the administrator token:

```bash
curl -s -X POST http://127.0.0.1:8765/v1/devices/pair \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"name":"phone"}'
```

The response contains a code valid for five minutes. Claiming it returns a raw
device token exactly once; the database stores only its SHA-256 hash. Revoke it
later with `DELETE /v1/devices/{id}`. Device tokens can use chat, task, memory
and status routes, but cannot change settings, list devices, read approvals or
approve writes.

## Server-Sent Events

```bash
curl -N http://127.0.0.1:8765/v1/events \
  -H "Authorization: Bearer $TOKEN"
```

Events include `task.queued`, `task.started`, `task.completed`, `task.failed`,
`task.cancelled`, `chat.started` and `chat.completed`. A keepalive comment is sent
every 15 seconds. The stream is in-process; history belongs in `/v1/tasks` and
`/v1/audit`.

## Memory

```bash
curl -s "http://127.0.0.1:8765/v1/memory?limit=20" \
  -H "Authorization: Bearer $TOKEN"

curl -s -X POST http://127.0.0.1:8765/v1/memory/search \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"query":"budget PC","limit":5}'
```

The embedding model must be installed (`ollama pull nomic-embed-text`) unless
`CLAIREAI_EMBEDDING_PROVIDER=hash` was chosen explicitly. Automatic embedding of
completed turns is opt-in with `CLAIREAI_MEMORY_AUTO_EMBED=true` or the Memory
page.

## Safety boundary

- Bind loopback by default. Changing `--host` is an explicit network exposure;
  the launcher prints a warning. There is no TLS in this service, so use an HTTPS
  reverse proxy before exposing it beyond a trusted network.
- There is no CORS middleware. A browser page cannot use a stored token to call
  the service from another origin.
- The API reuses `run_task`, `ask`, the folder allow-list, capability checks,
  audit log and approval broker. It does not have a second policy.
- A bearer token can inspect a pending signed approval, deny it or let it expire,
  but it cannot approve it. Approval remains a terminal-issued, single-use token.
- `/health` reveals only liveness. `/docs` is static schema documentation; it
  does not execute a turn.
- Conversation history and memory are local SQLite rows. The normal retention
  tools apply to the database; API chat does not upload text.

## Tests

The API is exercised through ASGI with `TestClient`; no port is opened in the
unit suite. The checks cover public health, missing/invalid bearer tokens, status,
memory search/list/delete, stubbed no-tool chat, task lifecycle, device pairing
and revocation, signed-approval listing/denial, settings, audit, summary and SSE
frame ordering.
