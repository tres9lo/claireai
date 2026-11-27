# Brains: Ollama/Qwen by default

ClaireAI's brain is **Qwen served by Ollama on this machine**. The default main
model is `qwen3:1.7b`; turns whose main job is code use
`qwen2.5-coder:1.5b`. No checkpoint has to be trained, and no hosted API is
called.

The change from from-scratch training was deliberate. On the development PC
(CPU only, 7 GB RAM), a full training run took hours, several candidates failed
the gate, and the result was not a dependable assistant. A small pretrained
model is the honest fit for that hardware.

| Provider | `CLAIREAI_BRAIN_PROVIDER` | Where it runs | Key needed |
| --- | --- | --- | --- |
| Ollama + Qwen | `ollama` (default) | this machine, loopback | no |
| Archived scratch checkpoint | `scratch` | this machine, on disk | no |
| OpenAI | `openai` | `api.openai.com` | yes |
| Anthropic | `anthropic` | `api.anthropic.com` | yes |

## Default configuration

Nothing needs to be set. The defaults are:

```dotenv
CLAIREAI_BRAIN_PROVIDER=ollama
CLAIREAI_BRAIN_MODEL=qwen3:1.7b
CLAIREAI_BRAIN_CODING_MODEL=qwen2.5-coder:1.5b
CLAIREAI_BRAIN_THINK=false
CLAIREAI_BRAIN_BASE_URL=http://127.0.0.1:11434/api
```

Prepare the runtime once:

```bash
ollama serve                    # Linux; Windows starts the service
ollama pull qwen3:1.7b
ollama pull qwen2.5-coder:1.5b
```

Check it:

```bash
assistant/.venv/bin/python scripts/doctor.py
```

The brain section reports whether Ollama answered, whether both models are
pulled, and the exact `ollama pull` command for anything missing.

## How code routing works

`OllamaBrain` reads the user text of the turn. If it clearly concerns code
(Python, JavaScript, Rust, SQL, a function/class, a traceback, debugging,
refactoring, tests and similar), it sends that turn to
`CLAIREAI_BRAIN_CODING_MODEL`. Everything else uses
`CLAIREAI_BRAIN_MODEL`. This is a model choice only; the agent loop, tool schemas,
approvals and audit behaviour are identical.

Qwen3 can emit hidden reasoning before its answer. On this CPU that measured
7.5 s for a 32-token probe versus 1.8 s with thinking off, so
`CLAIREAI_BRAIN_THINK` defaults to `false`. Tool calls are already structured;
enable thinking only when a turn genuinely needs deeper reasoning. The flag is
sent only to Qwen3 tags, because other Ollama models do not declare it.

The default 180-second request timeout is a backstop for slow CPU prompt
processing, not a target. Routing narrows file-path requests to the relevant
schemas; if a request does time out, the task fails with a clear
`Ollama did not answer within 180s` error rather than hanging.

## Hosted providers (opt-in)

Sent to a hosted endpoint: the system prompt, the conversation so far, the tool
schemas the request can reach, and tool results already approved and executed.
Not sent: the SQLite database, the audit log, the folder allow-list itself,
control-panel state, or another provider's key.

```bash
# Claude
CLAIREAI_BRAIN_PROVIDER=anthropic
CLAIREAI_BRAIN_MODEL=claude-sonnet-4-5
CLAIREAI_BRAIN_API_KEY=sk-ant-...

# ChatGPT
CLAIREAI_BRAIN_PROVIDER=openai
CLAIREAI_BRAIN_MODEL=gpt-4o-mini
CLAIREAI_BRAIN_API_KEY=sk-...
```

| Variable | Default | Meaning |
| --- | --- | --- |
| `CLAIREAI_BRAIN_BASE_URL` | per provider | endpoint override, e.g. vLLM or LM Studio |
| `CLAIREAI_BRAIN_TIMEOUT_SECONDS` | `180` | per-request timeout; CPU tool calls can legitimately exceed a minute |
| `CLAIREAI_BRAIN_MAX_TOKENS` | `1024` | initial provider reply budget; the saved **Max answer tokens** preference becomes the per-turn cap |
| `CLAIREAI_BRAIN_CODING_MODEL` | `qwen2.5-coder:1.5b` | Ollama code model |
| `CLAIREAI_BRAIN_THINK` | `false` | Qwen3 hidden reasoning |

### Endpoint rules

The endpoint must be `https`, unless it is loopback (`127.0.0.1`, `::1`,
`localhost`). A key sent over plain `http` to another host is a credential leak,
so configuration refuses to start instead of warning.

## Measuring a brain

```bash
assistant/.venv/bin/python scripts/tool_call_eval.py --runs 3
assistant/.venv/bin/python scripts/conversation_eval.py --runs 3
```

These use the real parser and system prompt. A model that chats well but emits
calls the runtime cannot read is not usable for tasks.

## Honest limits

- **A 1.7B model is still small.** It can be wrong, invent a fact, or choose a
  poor tool. Verify facts that matter and keep writes approval-gated.
- **No streaming.** A request waits for the complete reply.
- **No automatic retry.** A failed request is reported; retrying a tool-driving
  turn can double-execute a proposal.
- **Long conversations are sent in full.** A very long chat can exceed the
  model's context window; the provider error is surfaced.
- **Hosted cost is the owner's.** The local Ollama path has no marginal API cost;
  a hosted connector does, and the runtime does not meter it.
- **The audit trail names the model.** The model never enforces permissions.
