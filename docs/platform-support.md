# Platform support

Feature status is only marked ✅ after it was actually run on that OS. Everything
else says what is missing. A compile or a mocked test is not a platform result.

Last updated: 2026-11-19 (desktop-only surface and consent-based installer).

| Capability | Linux | Windows | Android |
| --- | --- | --- | --- |
| Text chat with local model | 🟨 scratch checkpoint wiring verified; no useful task model trained | 🟨 unit tests only | ⛔ not started (Phase 3) |
| Local model management | 🟨 opt-in candidate ledger and owner promotion in desktop Settings | 🟨 unit tests only | ⛔ not planned |
| File tools, calendar and allow-list | ✅ tested 2026-10-30/31 on Linux: 5 read-only + 6 approval-gated write tools (now 6 + 8; the calendar tools added 2026-11-07 are unit-tested only); allow-list refusal and non-interactive approval refusal verified against the live model; every step logged | 🟨 unit tests pass on `windows-latest`; CLI, approval prompt and trash folder never run by a human there | ⛔ n/a |
| Browser tools | ✅ tested 2026-10-31 on Linux against live pages: `open_url`, `get_page_text`, `search_web` (10 results, redirects unwrapped), `download_file` into the sandbox. **HTTP only — no JavaScript**, so a client-rendered page comes back empty; the Playwright engine itself is ⛔ | 🟨 mocked-transport unit tests only | ⛔ n/a |
| Camera / screen capture | ⛔ Phase 2, deny-by-default; this machine is **Wayland** and has `/dev/video0`–`/dev/video1` | ⛔ never run | ⛔ Android APIs only |
| Cursor / keyboard control | ⛔ Phase 2, `dangerous`; on Wayland needs the portal or `ydotool` (neither installed here) — may be unsupported | ⛔ never run | ⛔ not planned |
| Push-to-talk voice | 🟨 UI, local transcription API and safety bounds implemented; no microphone device on this host, so live capture is unverified | 🟨 unit tests only; live microphone path unverified | ⛔ later |
| Wake-word detection | ⛔ no "Hey Claire" model exists yet; treat as pending | ⛔ never run | ⛔ separate implementation, may not be permitted in background |
| Daily summary and local log | ✅ log, `--summary`, `/summary` and `--summary-at` (no model needed) | ⛔ never run | ⛔ client phase |
| Installer / package | 🟨 `./install.sh` + `claireai --setup --install`: OS/Python/Tk/display detection, a sized consent list, streamed pip output and the local speech model; dry-run and editable install verified | 🟨 `install.ps1` written, never run on Windows | ⛔ not started |
| OS detection | 🟨 reported by setup/diagnostics; install paths differ per OS | 🟨 unit tests only | ⛔ not started |

## Tested environments

| OS | Version | Python | Result |
| --- | --- | --- | --- |
| Linux (Zorin OS) | 18.1, x86_64, Intel i5-6300U, 7.1 GiB RAM | 3.12.3 | 155 unit tests pass; real local inference verified online and in a network namespace with no route; 5 read-only tools exercised against the live model; allow-list refusal verified with the live model; task/audit rows verified in SQLite; 9/10 model eval cases stable |
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
- Model training and inference use the locally installed CPU PyTorch package;
  no external model runtime is used.
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
