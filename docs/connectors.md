# Connectors: the local model, Claude, ChatGPT and Ollama

ClaireAI's default brain is the checkpoint in `runtime-data/models/` — the model
trained from scratch in this repository. It is the only brain that makes no
network request, and it stays the default.

A connector is for the case where the owner decides the small model is not
enough for a particular stretch of work. Each connector is opt-in, configured by
the owner, and visible afterwards: the audit row for the turn names the provider
that answered, so a conversation is never silently answered by a service the
owner did not choose.

| Provider    | Value of `CLAIREAI_BRAIN_PROVIDER` | Where it runs                | Key needed |
| ----------- | ---------------------------------- | ---------------------------- | ---------- |
| Scratch     | `scratch` (default)                | this machine, on disk        | no         |
| Ollama      | `ollama`                           | this machine, local daemon   | no         |
| OpenAI      | `openai`                           | `api.openai.com`             | yes        |
| Anthropic   | `anthropic`                        | `api.anthropic.com`          | yes        |

## What a connector sends, and what it does not

Sent to the configured endpoint: the system prompt, the conversation so far, the
tool schemas the request can reach (the same filtered set the local model gets),
and the results of tools that already ran with the owner's approval.

Not sent: the SQLite database, the audit log, the folder allow-list itself, the
control-panel state, the API key of any other provider, or anything the tools
did not return. Tool side effects stay on this machine — a remote model can only
propose a tool call; the same approval gate and sandbox apply either way.

## Setup

Set the values in `.env` (loaded automatically) or the environment:

```bash
# Claude
CLAIREAI_BRAIN_PROVIDER=anthropic
CLAIREAI_BRAIN_MODEL=claude-sonnet-4-5
CLAIREAI_BRAIN_API_KEY=sk-ant-...

# ChatGPT
CLAIREAI_BRAIN_PROVIDER=openai
CLAIREAI_BRAIN_MODEL=gpt-4o-mini
CLAIREAI_BRAIN_API_KEY=sk-...

# A larger model on this machine, still local
CLAIREAI_BRAIN_PROVIDER=ollama
CLAIREAI_BRAIN_MODEL=qwen2.5:7b
```

Optional knobs, shared by all connectors:

| Variable                    | Default             | Meaning                              |
| --------------------------- | ------------------- | ------------------------------------ |
| `CLAIREAI_BRAIN_BASE_URL`   | per provider        | endpoint override, e.g. vLLM or LM Studio |
| `CLAIREAI_BRAIN_TIMEOUT_SECONDS` | `60`           | per-request timeout                  |
| `CLAIREAI_BRAIN_MAX_TOKENS` | `1024`              | reply budget                         |

Return to the local model by unsetting the provider or setting
`CLAIREAI_BRAIN_PROVIDER=scratch`.

### Endpoint rules

The endpoint must be `https`, unless it is loopback (`127.0.0.1`, `::1`,
`localhost`). A key sent over plain `http` to another host is a credential leak,
so configuration refuses to start rather than warn: the assistant fails loudly
or not at all, and this is one of the failures worth having.

`CLAIREAI_BRAIN_BASE_URL` also points the OpenAI adapter at an
OpenAI-compatible server (vLLM, LM Studio, llama.cpp), which is the way to keep
a connector-shaped interface while the weights stay on your network.

## Checking it works

```bash
assistant/.venv/bin/python scripts/doctor.py
```

The brain section prints the provider, the endpoint, whether the endpoint
answered, and whether the configured model is actually there. For Ollama, a
model that is not pulled is reported as such with the command that fixes it.

```bash
assistant/.venv/bin/python scripts/tool_call_eval.py --runs 3
```

This scores tool-call accuracy for whatever brain is configured, using the real
parser and the real system prompt. It is the number to compare providers on:
a connector that chats well but emits calls the runtime cannot read is not
usable for tasks.

## How it is built

- `claireai.brain.factory.build_brain` is the single construction point; the CLI,
  the desktop runtime, the local web server and `doctor` all call it.
- `OpenAIBrain` renders Chat Completions messages, generating the `tool_call_id`
  values the API requires and pairing each tool result with its call.
- `AnthropicBrain` hoists the system prompt to the top-level field the Messages
  API expects and merges consecutive tool results into one user turn.
- `OllamaBrain` speaks the native `/api/chat` shape, where arguments arrive as an
  object rather than a JSON string.
- `RemoteBrain` owns the timeout, the error mapping and the single shared HTTP
  client. Authentication failures, rate limits and transport errors are distinct
  `BrainError`s; none of them includes the key in its message.

## Honest limits

- **No live-endpoint testing in this repository.** No API keys are committed, and
  the tests use `httpx.MockTransport`, so they prove the wire format and the
  parsing, not that a vendor's API is up today. `scripts/doctor.py` is the
  check that needs credentials.
- **No streaming, no retries, no backoff.** A failed request is reported, not
  silently repeated: a retried tool-driving turn can double-execute a proposal.
- **No prompt caching or conversation trimming.** Long conversations are sent in
  full, and a long one can exceed a provider's context window; the error comes
  back as a `BrainResponseError` with the provider's message.
- **Tool-call format differs per provider.** The adapters normalise all three into
  `ToolCall`, and the runtime's own validation still decides what may run.
- **Cost is the owner's.** The local model has no marginal cost; a connector does,
  and the runtime does not meter it.
