# Machine access: camera, screen, cursor, keyboard, clipboard

ClaireAI should be able to use what the computer offers *while a task needs it*:
look through the camera, read the screen, move the cursor, type, inspect windows.
This document fixes the rules before any of it exists, because these are the most
dangerous capabilities in the product.

Status: **planned**. Per-task grants and the approval flow start in Week 3;
perception arrives in Months 2–3, control in Months 2–3 with per-action approval.
Today the repository has none of these tools.

## Why this is separate from file tools

Reading a file is bounded by a folder allow-list that code can check. A camera
frame is whatever happened to be in front of the lens; a screenshot may contain
bank details; a cursor click can hit "delete" in an app ClaireAI does not model.
So machine access is denied by default, granted per task, and always visible.

## Capabilities

| Capability | What it does | Risk | Default | Indicator |
| --- | --- | --- | --- | --- |
| `screen` | screenshot the display / a window | `needs_approval` | off | screen light + "reading screen for task …" |
| `camera` | capture one frame from a camera | `needs_approval` | off | camera light + device name |
| `clipboard` | read clipboard text | `needs_approval` | off | clipboard notice |
| `cursor` | move, click, drag, scroll | `dangerous` | off | pointer badge + "controlling input" |
| `keyboard` | type text, press keys/hotkeys | `dangerous` | off | keyboard badge |

`CLAIREAI_ALLOWED_CAPABILITIES` is a JSON array of capability names. **Empty is
the default and means every machine-access tool is refused in code**, exactly
like an empty folder allow-list disables file tools:

```bash
CLAIREAI_ALLOWED_CAPABILITIES='["screen", "camera"]'
```

Being listed is permission for the capability to *exist*; each use still needs
its own approval while no named pattern is trusted.

## Grants

A grant is a record, not a boolean:

- bound to **one task id**, the **specific device/display/window**, and the
  **approval reference** shown to the user;
- expires when the task ends, is cancelled, or is interrupted — an interrupted
  task loses its grants and must be re-approved after reconciling;
- never inherited by a follow-up request, and never granted for a whole session
  by a config flag;
- for `cursor`/`keyboard`, granted per action until the owner explicitly trusts a
  named action ("click inside the PDF viewer"), never "all input for 10 minutes".

## Data handling

- Screenshots and camera frames are processed **in memory**, are never uploaded,
  and are not persisted unless the task explicitly saves a file and the owner
  approves that save.
- OCR / window-title / clipboard text is **untrusted input**: it cannot change
  permissions, tool availability, or system policy (prompt-injection boundary).
- Typed text is redacted before it reaches `task_log`; key presses are logged by
  name and count, not by the characters they produced into a field.
- The audit row records: capability, target (display/device/window/app), what was
  read or changed, status, approval reference, duration. Never the frame.
- Microphone stays a separate permission with its own indicator; camera video
  never captures audio.

## Platform matrix

| Capability | Linux X11 | Linux Wayland | Windows |
| --- | --- | --- | --- |
| Screen capture | `mss` / X screenshot | `xdg-desktop-portal` ScreenCast/Screenshot, user grant may be refused | `mss` (Win32 GDI) |
| Camera | `opencv-python` (V4L2) | same (V4L2) | `opencv-python` (MediaFoundation) |
| Cursor / keyboard | XTest via `pyautogui` | portal/libei or `ydotool`; often unavailable — report unsupported | `SendInput` via `pyautogui`/ctypes |
| Windows / apps | `wmctrl`/`xdotool` where installed | portal | `pygetwindow`/Win32 |

This development machine runs **Wayland** (`XDG_SESSION_TYPE=wayland`) with
`/dev/video0`–`/dev/video1` present and no `ydotool`, `xdotool` or `grim`
installed. That is recorded here so nobody promises Linux cursor control on this
laptop before testing the portal path.

## Dependencies and licences

Verified from package metadata on 2026-10-31; re-check versions at implementation
time and keep the detail in `docs/model-and-data-licenses.md`.

| Package | Version seen | Licence | Use |
| --- | --- | --- | --- |
| `mss` | 10.2.0 | MIT | screenshots |
| `opencv-python` | 5.0.0.93 | Apache-2.0 | camera frames |
| `pyautogui` | 0.9.54 | BSD | cursor/keyboard on X11 and Windows |
| `Pillow` | 13.0.0 | MIT-CMU (upstream LICENSE; PyPI has no classifier) | image handling |
| `pynput` | 1.8.2 | LGPLv3 | **avoid for now** — LGPL obligations are not worth it when `pyautogui` works |

System dependencies, not pip packages: `xdg-desktop-portal` (Wayland), `ydotool`
+ `uinput` permission (Wayland input), `wmctrl`/`xdotool` (X11 window control).

## Safety rules (non-negotiable)

1. Deny by default: not in `CLAIREAI_ALLOWED_CAPABILITIES`, no call.
2. Visible indicator whenever a capability is active, from the same event stream
   that writes the audit row.
3. Approval bound to the exact action and target; a changed plan needs new
   approval.
4. Kill-switch hotkey stops all agents and releases input immediately; the task
   is logged as `cancelled`.
5. Step cap for control loops; no unbounded click/type actions.
6. Refuse to type into password fields or send credentials; never read a password
   manager window.
7. Dry run prints the intended action ("click at 812,433 in window X") without
   moving the pointer.
8. Every refusal is logged as `skipped` with a reason the user can act on.

## Acceptance checks (when built)

1. Camera/screen tools refuse when the capability is absent from the allow-list,
   and log the refusal.
2. Indicator appears and disappears with the actual capture, not with the request.
3. Screenshot of a window is not written to disk unless a save is approved.
4. Cursor action: dry run moves nothing; live run needs approval; kill switch
   stops a multi-step sequence mid-way and logs `cancelled`.
5. A refused Wayland portal grant produces one clear error, not a retry loop.
6. Typed text containing something that looks like a secret is redacted in the
   task log while still being typed correctly.
