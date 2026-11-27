# Install the Ollama models

ClaireAI needs two chat models, plus one small optional embedding model if you
use local memory:

| Model | Role | Approximate download |
| --- | --- | --- |
| `qwen3:1.7b` | normal conversation and tool calls | ~1.1 GB |
| `qwen2.5-coder:1.5b` | code-heavy turns | ~1.0 GB |
| `nomic-embed-text` | local semantic memory (optional but recommended) | ~275 MB |

Sizes vary with the Ollama build and quantisation. Plan for about **2.5 GB** for
the two chat models, or **2.8 GB** with memory, plus Ollama itself.

## 1. Install Ollama

### Linux

Use the official installer:

```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama --version
```

The installer normally registers a systemd service. Check it:

```bash
systemctl status ollama --no-pager
curl -s http://127.0.0.1:11434/api/version
```

If the service is not running:

```bash
sudo systemctl enable --now ollama
# or, for a foreground session:
ollama serve
```

If `ollama` is not found, open a new terminal. The installer puts it in
`/usr/local/bin/ollama` on most systems.

### Windows

1. Download **Ollama for Windows** from <https://ollama.com/download>.
2. Run `OllamaSetup.exe` and finish the installer.
3. Ollama starts in the notification area. Open a **new** PowerShell window.
4. Verify:

```powershell
ollama --version
Invoke-RestMethod http://127.0.0.1:11434/api/version
```

If the command is not found, sign out and back in, or add
`%LOCALAPPDATA%\Programs\Ollama` to `PATH`. The desktop app normally starts the
service automatically.

## 2. Pull the models

Run these once. They are downloads, not API calls, and they can be resumed.

```bash
ollama pull qwen3:1.7b
ollama pull qwen2.5-coder:1.5b
# Only needed for local semantic memory:
ollama pull nomic-embed-text
```

Windows PowerShell uses exactly the same commands. `nomic-embed-text` is only
used when `CLAIREAI_MEMORY_ENABLED=true` (the default); if you turn memory off
you can skip it.

Verify they are present:

```bash
ollama list
```

Expected names include `qwen3:1.7b`, `qwen2.5-coder:1.5b` and, when memory is on,
`nomic-embed-text`. Test the main model directly:

```bash
ollama run qwen3:1.7b "Reply with exactly: OK"
```

Exit the interactive prompt with `/bye`.

## 3. Where the weights are stored

Ollama owns the model store; ClaireAI does not copy or bundle the weights.

| OS | Default location |
| --- | --- |
| Linux, per-user service | `~/.ollama/models` |
| Linux, system service | often `/usr/share/ollama/.ollama/models` |
| Windows | `%USERPROFILE%\.ollama\models` |
| macOS | `~/.ollama/models` |

Override it before starting Ollama:

```bash
export OLLAMA_MODELS=/path/with/space/models
```

```powershell
$env:OLLAMA_MODELS = "D:\ollama-models"
```

The directory can be large; keep it on a disk with room. Do not commit it to the
repository, and do not point it at a synced cloud folder unless you intend the
weights to sync.

## 4. Point ClaireAI at Ollama

The defaults are already correct:

```dotenv
CLAIREAI_BRAIN_PROVIDER=ollama
CLAIREAI_BRAIN_MODEL=qwen3:1.7b
CLAIREAI_BRAIN_CODING_MODEL=qwen2.5-coder:1.5b
# Off is faster on CPU. Set true only when deeper reasoning is worth the wait.
CLAIREAI_BRAIN_THINK=false
CLAIREAI_BRAIN_BASE_URL=http://127.0.0.1:11434/api

# Local memory uses the same Ollama daemon.
CLAIREAI_MEMORY_ENABLED=true
CLAIREAI_EMBEDDING_PROVIDER=ollama
CLAIREAI_EMBEDDING_MODEL=nomic-embed-text
```

Check the whole install:

```bash
assistant/.venv/bin/python scripts/doctor.py
```

Windows:

```powershell
.\assistant\.venv\Scripts\python.exe scripts\doctor.py
```

A healthy report says the provider is `ollama`, the local model is available,
`qwen3:1.7b ready` is yes, and `Coding model ready` is yes. The setup report also
prints the memory model and API bind address.

If you do not want to pull the embedding model, set
`CLAIREAI_EMBEDDING_PROVIDER=hash` for a deterministic offline fallback. It is
lexical rather than semantic; the Ollama embedder is the recommended mode.

## 5. Memory on a budget PC

A 1.7B model is appropriate for a CPU-only machine with about 7 GB RAM. To keep
Ollama from keeping both models loaded at once:

```bash
export OLLAMA_MAX_LOADED_MODELS=1
export OLLAMA_KEEP_ALIVE=5m
```

Restart the Ollama service after changing these. ClaireAI sends code turns to
the coder model only when the request is clearly about code.

## 6. Troubleshooting

| Symptom | Fix |
| --- | --- |
| `connection refused` / `Ollama is unreachable` | Start the service: `ollama serve` or `sudo systemctl start ollama`; on Windows open the Ollama tray app |
| `model is not pulled` | `ollama pull <model>` for the exact name in the error |
| `ollama: command not found` | Open a new terminal; check `/usr/local/bin/ollama`; on Windows check `PATH` |
| Download stalls | Re-run the same `ollama pull`; it resumes |
| Low memory / swaps badly | Set `OLLAMA_MAX_LOADED_MODELS=1`; close other apps; do not run both models at once |
| Wrong endpoint | `CLAIREAI_BRAIN_BASE_URL` must be `http://127.0.0.1:11434/api` unless you intentionally run Ollama elsewhere on loopback |
| A non-loopback URL is refused | This is deliberate. Plain HTTP to another host could leak data; use HTTPS or keep inference local |

Ollama does not need an API key. Do not put one in `.env`.
