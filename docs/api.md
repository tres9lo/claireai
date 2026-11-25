# Local API service

The API is an opt-in backend for local clients. It is not the desktop UI and it
does not replace the approval path.

```bash
pip install -e "assistant[api]"
claireai --api --api-host 127.0.0.1 --api-port 8765
# equivalent:
assistant/.venv/bin/claireai-api --host 127.0.0.1 --port 8765
```

`/health` is public. Every other route requires a bearer token:

```bash
TOKEN=$(cat ~/.local/share/ClaireAI/api.token)
curl -s http://127.0.0.1:8765/v1/status \
  -H "Authorization: Bearer $TOKEN"
```

The token is read from `CLAIREAI_API_TOKEN`, or created as a 0600 `api.token`
file in the data directory. The data directory is OS-specific; `claireai --setup`
prints it. Do not put the token in a URL, a screenshot or a repository.

## Routes

| Method | Path | Purpose |
| --- | --- | --- |
| GET | `/health` | Liveness only; no model or database work |
| GET | `/v1/status` | Provider, model, memory count and enabled capabilities |
| GET | `/v1/models` | Live Ollama health and model readiness |
| POST | `/v1/chat` | One chat turn, with safe tools or plain chat |
| GET | `/v1/conversations` | Recent local conversations |
| GET | `/v1/conversations/{id}` | Messages in one conversation |
| POST | `/v1/memory/search` | Cosine search over local notes |
| GET | `/docs` | Generated OpenAPI UI |

### Chat

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

### Memory search

```bash
curl -s http://127.0.0.1:8765/v1/memory/search \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"query":"budget PC","limit":5}'
```

The response contains a readable `text` field and structured `results`. The
embedding model must be installed (`ollama pull nomic-embed-text`) unless
`CLAIREAI_EMBEDDING_PROVIDER=hash` was chosen explicitly.

## Safety boundary

- Bind loopback by default. Changing `--host` is an explicit network exposure;
  the launcher prints a warning. There is no TLS in this service, so use an HTTPS
  reverse proxy before exposing it beyond a trusted network.
- There is no CORS middleware. A browser page cannot use a stored token to call
  the service from another origin.
- The API reuses `run_task`, `ask`, the folder allow-list, capability checks,
  audit log and approval broker. It does not have a second policy.
- `/health` reveals only liveness. `/docs` is static schema documentation; it
  does not execute a turn.
- Conversation history and memory are local SQLite rows. The normal retention
  tools apply to the database; API chat does not upload text.

## Tests

The API is exercised through ASGI with `TestClient`; no port is opened in the
unit suite. The checks cover public health, missing/invalid bearer tokens, status,
memory search and a stubbed no-tool chat turn.
