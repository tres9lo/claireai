# Platform support

Feature status is only marked ✅ after it was actually run on that OS. Everything
else says what is missing. A compile or a mocked test is not a platform result.

Last updated: 2026-11-25 (live wake-word ONNX, ALSA capture, Ollama memory, API listener and screen-capture checks).

| Capability | Linux | Windows | Android |
| --- | --- | --- | --- |
| Text chat with local model | ✅ Ollama adapter and Qwen routing unit-tested; a live Qwen run must be recorded separately | 🟨 unit tests only; Ollama on Windows unverified by a human | ⛔ not started (Phase 3) |
| Local model management | ✅ `ollama pull` instructions, health check and missing-model diagnostics | 🟨 unit tests only | ⛔ not planned |
| File tools, calendar and allow-list | ✅ tested 2026-10-30/31 on Linux: 5 read-only + 6 approval-gated write tools (now 6 + 8; the calendar tools added 2026-11-07 are unit-tested only); allow-list refusal and non-interactive approval refusal verified against the live model; every step logged | 🟨 unit tests pass on `windows-latest`; CLI, approval prompt and trash folder never run by a human there | ⛔ n/a |
| Browser tools | ✅ tested 2026-10-31 on Linux against live pages: `open_url`, `get_page_text`, `search_web` (10 results, redirects unwrapped), `download_file` into the sandbox. **HTTP only — no JavaScript**, so a client-rendered page comes back empty; the Playwright engine itself is ⛔ | 🟨 mocked-transport unit tests only | ⛔ n/a |
| Screen capture | ✅ live `mss` capture returned a 1280x720 frame on 2026-11-25; `see_screen`/CUA still enforce the `screen` capability | 🟨 unit tests only; no manual run | ⛔ Android APIs only |
| Cursor / keyboard control | 🟨 optional CUA driver, `dangerous`, always approved; live pointer input is blocked here by X authorization even with `XAUTHORITY`, and the failure is reported | 🟨 unit tests only; no manual run | ⛔ not planned |
| Local semantic memory | ✅ live Ollama `nomic-embed-text`: 768-dimension vectors and correct top semantic hit on 2026-11-25; SQLite cosine index, hash fallback, no network in tests | 🟨 mocked/unit tests only | ⛔ later |
| FastAPI backend | ✅ live loopback listener verified 2026-11-25: `/health` 200, `/v1/status` 401 without token and 200 with token | 🟨 unit tests only | ⛔ later |
| Push-to-talk voice | 🟨 UI, local transcription API and safety bounds implemented; ALSA capture exists, but push-to-talk with the speech model has not had a human acceptance run | 🟨 unit tests only; live microphone path unverified | ⛔ later |
| Wake-word detection | 🟨 real `hey_jarvis` ONNX inference verified (synthesized clip scored 0.9967; silence ignored) and ALSA capture opened; no human-spoken run and no custom "Hey Claire" model | 🟨 unit tests only | ⛔ separate implementation, may not be permitted in background |
| Daily summary and local log | ✅ log, `--summary`, `/summary` and `--summary-at` (no model needed) | ⛔ never run | ⛔ client phase |
| Installer / package | 🟨 `./install.sh` + `claireai --setup --install`: OS/Python/Tk/display detection, a sized consent list, streamed pip output and the local speech model; dry-run and editable install verified | 🟨 `install.ps1` written, never run on Windows | ⛔ not started |
| OS detection | 🟨 reported by setup/diagnostics; install paths differ per OS | 🟨 unit tests only | ⛔ not started |

## Tested environments

| OS | Version | Python | Result |
| --- | --- | --- | --- |
| Linux (Zorin OS) | 18.1, x86_64, Intel i5-6300U, 7.1 GiB RAM | 3.12.3 | 839 unit tests pass (2026-11-25); live ONNX wake-word scoring, ALSA capture, Ollama 768-dim embedding search, API listener and `mss` screen capture verified; pointer input remains blocked by X authorization. Previously verified real local inference online and with no network route, 5 live read-only tools, allow-list refusal, SQLite task/audit rows and 9/10 stable model eval cases |
| Ubuntu (GitHub runner) | `ubuntu-latest`, CI run 41144463051 | 3.11, 3.12 | 155 unit tests pass on both (no display) |
| Windows (GitHub runner) | `windows-latest`, CI run 41144463051 | 3.11, 3.12 | 155 unit tests pass on both; **no manual run** — no CLI, audio, installer or desktop evidence |
| Android | Tecno Spark 20 | — | not tested |

## Desktop window

The current window is Tk/customtkinter, runs natively on Linux (verified) and
has no browser or loopback server behind it. A future Tauri shell is still
blocked on the toolchain below and is not part of this install; two steps need
`sudo` or a network fetch the build agent cannot perform:

```bash
# 1. WebKit/GTK runtime the window links against (Ubuntu/Debian family)
sudo apt update
sudo apt install -y libwebkit2gtk-4.1-dev libgtk-3-dev libsoup-3.0-dev librsvg2-dev

# 2. Rust: the distro's rustc here is 1.75; Tauri 2 needs >= 1.90
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source "$HOME/.cargo/env"
cargo --version   # expect 1.9x or newer
```

Until both are done, the Tk window is the desktop UI and `claireai --cli` is the
terminal UI. Nothing waits on the Tauri shell.

## Known platform differences to keep in mind

- Data locations come from `platformdirs` (`~/.local/share/ClaireAI` on Linux,
  `%LOCALAPPDATA%\ClaireAI\ClaireAI` on Windows). Use `CLAIREAI_DATA_DIR` to
  override for development; never hard-code a home path in shared code.
- `python3 -m venv` needs the `python3-<version>-venv` system package on
  Debian/Ubuntu-family systems. `uv venv --seed` avoids that dependency.
- The default runtime calls Ollama on `127.0.0.1:11434`; it is local but it is
  a separate service that must be running. PyTorch is only needed for archived
  `brain-lab` research, not for normal chat.
- Audio, microphone and desktop integration will need manual acceptance testing
  per OS; CI cannot cover them.
- Tool results use POSIX separators (`sub/inner.txt`) on every OS; only the
  absolute path in the header follows the platform. CI caught the Windows
  variant of this in Week 2, which is why the tests assert on it.
- Creating symlinks on Windows needs Developer Mode or an elevated process. The
  allow-list tests that need one pass on the GitHub runners (elevated) but may
  need adjusting for a normal Windows user session.
- Display server matters for machine access. The development machine runs
  `XDG_SESSION_TYPE=wayland`, where screen capture needs `xdg-desktop-portal` and
  global cursor/keyboard control needs the portal or `ydotool`; none of
  `ydotool`, `xdotool` or `grim` is installed. Do not promise Linux input control
  until that path is tested, and report "unsupported" instead of falling back to
  a risky workaround.
- The Tk window is CPU-rendered, so no GPU is required; animations drop frames
  rather than slowing down. A future Tauri shell would be GPU-bound and must be
  measured on the target machine before raising quality.
