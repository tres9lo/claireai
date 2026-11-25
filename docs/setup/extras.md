# Optional capabilities

The Tk desktop remains the normal UI. These extras are opt-in so a basic install
does not pull GUI automation, audio or HTTP-server packages.

## Local semantic memory

Memory uses Ollama embeddings and a SQLite cosine index. It is local and needs no
API key.

```bash
ollama pull nomic-embed-text
```

The defaults already enable it:

```dotenv
CLAIREAI_MEMORY_ENABLED=true
CLAIREAI_EMBEDDING_PROVIDER=ollama
CLAIREAI_EMBEDDING_MODEL=nomic-embed-text
```

Ask ClaireAI to "remember that ..."; it will request approval before writing the
note. "What do you remember about ..." uses `search_memory`. The API endpoint is
`POST /v1/memory/search`.

If the embedding model is missing, the tool fails with the exact `ollama pull`
command instead of falling back silently. For a deliberately offline lexical
fallback, set `CLAIREAI_EMBEDDING_PROVIDER=hash`. Hash vectors are not compatible
with Ollama vectors, so keep one provider for a given memory database.

The table is `memory_entries` in the normal SQLite file. The text is stored; the
embedding is a packed float32 vector. Deleting the database deletes both.

## CUA driver (screen and input)

Install the optional packages:

```bash
pip install -e "assistant[cua]"
```

Enable only what you intend to use, then restart:

```bash
export CLAIREAI_ALLOWED_CAPABILITIES='["screen","cursor","keyboard"]'
```

- `see_screen` uses the `screen` capability and the local vision model.
- `computer_control` moves, clicks, scrolls, types or presses a key. It is
  `dangerous`, always asks for approval, and never runs without the matching
  `cursor` or `keyboard` capability.

The agent redacts typed text from the audit row. Do not type passwords or API
keys into a chat request. PyAutoGUI's fail-safe is enabled: slamming the pointer
into a screen corner aborts the next action. On Wayland, input and capture may be
refused by the compositor; the tool reports that rather than retrying.

## Wake word

Install the optional runtime:

```bash
pip install -e "assistant[wakeword]"
```

Supply a matching openWakeWord model. The package's pretrained models can be
fetched with its downloader, or pass a model name/path. No model is downloaded by
this repository.

On Linux, `sounddevice` needs the PortAudio system library
(`sudo apt install libportaudio2`). If PortAudio is missing, the listener falls
back to ALSA `arecord` from `alsa-utils`, so a pip-only install can still capture
on a machine with a working `/dev/snd` device.

```bash
# Explicit 15-second check; uses the local microphone, saves no audio.
assistant/.venv/bin/python scripts/wakeword_check.py --seconds 15
```

`hey_jarvis` is the default example, not a custom "Hey Claire" model. A custom
phrase requires a matching trained model and false-activation testing. Detection
only starts a listening turn; it never grants a capability or counts as consent.
The real ONNX model, frame scoring and ALSA listener path were verified on
2026-11-25; a human-spoken microphone acceptance run is still required.

## FastAPI service

```bash
pip install -e "assistant[api]"
claireai --api                  # loopback: 127.0.0.1:8765
# or: assistant/.venv/bin/claireai-api --host 127.0.0.1 --port 8765
```

The service creates `api.token` (0600) in the data directory on first use, or
reads `CLAIREAI_API_TOKEN`. Every route except `/health` needs
`Authorization: Bearer <token>`. See [docs/api.md](../api.md).

A bearer token authorizes API access; it does **not** approve a file write. Tool
writes still use the signed one-action approval file, so an API caller cannot
self-approve a mutation.

## What stays unchanged

- The Tk desktop is still the UI; this does not introduce Tauri or React.
- Camera capture, clipboard reading and speaker identity are still not built.
- Nothing in these extras starts a microphone, downloads a model or opens a
  listener merely because the package is installed.
