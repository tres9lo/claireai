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
note. "What do you remember about ..." uses `search_memory`. The desktop **Memory**
page lists and deletes entries and can turn on automatic embedding of completed
turns. The API exposes `GET /v1/memory`, `POST /v1/memory/search` and
`DELETE /v1/memory/{id}`.

Automatic turn embedding is off by default (`CLAIREAI_MEMORY_AUTO_EMBED=false`)
because it adds an embedding call to every completed turn. Enable it in the
Memory page or with the environment variable; failures never turn a delivered
answer into a failed turn.

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

Enable only what you intend to use from **Settings → Capabilities** (Screen
access, Pointer control, Keyboard control), or set them explicitly:

```bash
export CLAIREAI_ALLOWED_CAPABILITIES='["screen","cursor","keyboard"]'
```

Enabling a capability only registers the tool. Each screen, pointer or keyboard
action still asks for approval.

- `see_screen` uses the `screen` capability and the local vision model.
- `computer_control` moves, clicks, scrolls, types or presses a key. It is
  `dangerous`, always asks for approval, and never runs without the matching
  `cursor` or `keyboard` capability.

The agent redacts typed text from the audit row. Do not type passwords or API
keys into a chat request. PyAutoGUI's fail-safe is enabled: slamming the pointer
into a screen corner aborts the next action. On this host Xwayland stores its
cookie under `hostname/unix:` while `DISPLAY=:0`; the driver copies that session
cookie into a private temporary authority file mapped to the active display, so
pointer and keyboard input work without modifying the owner's Xauthority file.
On a compositor that refuses input entirely, the tool reports that rather than
retrying.

## Wake word

Install the optional runtime:

```bash
pip install -e "assistant[wakeword]"
# Python 3.12: openWakeWord's tflite-runtime dependency has no wheel.
assistant/.venv/bin/claireai-install-wakeword
```

The first command installs the ONNX, NumPy and microphone packages. On Python
3.12 the second command installs openWakeWord itself with `--no-deps`; ClaireAI
only uses its ONNX backend and stubs the training-only scipy imports. On Python
3.11 pip can install the normal package.

For **“Hey Claire”**, select **Local speech (Hey Claire)** on the Voice page (or
set `CLAIREAI_WAKEWORD_ENGINE=stt`). This uses the already-installed local
faster-whisper model with a phrase hotword, so it hears the owner's own voice;
it is not an openWakeWord model and needs no custom ONNX training. The phrase is
`CLAIREAI_WAKEWORD_PHRASE`, default `hey claire`.

`openwakeword` remains available as an alternative engine. Supply a matching
openWakeWord model; its pretrained `hey_jarvis` is not a custom "Hey Claire"
model. No model is downloaded by this repository.

On Linux, `sounddevice` needs the PortAudio system library
(`sudo apt install libportaudio2`). If PortAudio is missing, the listener falls
back to ALSA `arecord` from `alsa-utils`, so a pip-only install can still capture
on a machine with a working `/dev/snd` device.

Open **Voice → Wake word** to start or stop listening. The choice is persisted;
the sidebar shows a microphone indicator and the status line says which phrase
is being listened for. The speech engine keeps a short pre-roll so the first
syllable is not cut off, transcribes the phrase and command locally, and sends
the command directly as the next chat turn. The openWakeWord engine records one
bounded command after detection. No audio is saved. An openWakeWord model can be
selected with `CLAIREAI_WAKEWORD_MODEL=/path/to/model.onnx` or a pretrained name.
The speech engine is exercised end-to-end with a synthesized “Hey Claire, open
the notes” clip; a human-spoken acceptance run is still required.

```bash
# Explicit 15-second check; uses the local microphone, saves no audio.
assistant/.venv/bin/python scripts/wakeword_check.py --seconds 15
```

Detection only starts a listening turn; it never grants a capability or counts
as consent. The real ONNX model, frame scoring, ALSA capture and the Tk
start/stop path were verified on 2026-11-25. The default speech engine was
verified end-to-end with a synthesized “Hey Claire, open the notes” clip on
2026-11-27; it matched the phrase, but the tiny ASR misheard the command as
“open the nose,” and a human-spoken acceptance run is still required.
openWakeWord still needs a trained model for a custom phrase.

## FastAPI service

```bash
pip install -e "assistant[api]"
claireai --api                  # loopback: 127.0.0.1:8765
# or: assistant/.venv/bin/claireai-api --host 127.0.0.1 --port 8765
```

The service creates `api.token` (0600) in the data directory on first use, or
reads `CLAIREAI_API_TOKEN`. Every route except `/health` and one-time device
claiming needs `Authorization: Bearer <token>`. See [docs/api.md](../api.md).

The API now covers background tasks and cancellation, device pairing and
revocation, signed-approval listing/denial, audit, summaries, editable non-secret
settings, memory list/delete and Server-Sent Events. A bearer token authorizes
API access; it does **not** approve a file write. Tool writes still use the
signed one-action approval file, so an API caller cannot self-approve a mutation.

## What stays unchanged

- The Tk desktop is still the UI; this does not introduce Tauri or React.
- Camera capture, clipboard reading and speaker identity are still not built.
- Nothing in these extras starts a microphone, downloads a model or opens a
  listener merely because the package is installed.
