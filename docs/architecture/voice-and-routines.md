# Voice, consent and routines

This document records the owner requirements and the security boundary for the
hands-free assistant. A local “Hey Claire” speech listener and the optional
openWakeWord adapter exist now; voice identity, spoken approvals and routine
replay remain targets. Push-to-talk speech input is optional and local.

## Voice interaction

The voice pipeline is local and visible:

1. The owner explicitly enables the microphone and sees a persistent listening
   indicator plus a one-action mute/stop control.
2. A local phrase detector recognizes “Hey Claire” from a rolling in-memory
   audio window. The default `stt` engine uses the local faster-whisper model with
   a hotword and pre-roll; openWakeWord remains an alternative. It does not save
   or upload that window.
3. The system records a bounded command, transcribes it locally, and shows the
   recognized text before executing a consequential task. Push-to-talk remains
   available if wake detection is disabled or unavailable.
4. Speech output uses a separately licensed voice pack and the owner's UI
   controls. The speaking state follows actual playback.

Wake-word detection starts a turn only. It is not speaker identity or consent.

## Voice identity and approval

The owner enrolls a local voice profile explicitly and can inspect/delete it in
the UI. Enrollment and recognition use local processing. Raw recordings are
short-lived; only the minimum derived profile data is retained. The UI must
explain retention and provide deletion.

A spoken approval is valid only for a live, single-use approval request. The
assistant reads the action and target; the owner must confirm that exact action
within a short expiry window. The runtime binds the result to the pending
approval reference and invalidates it after one decision, timeout, task change or
stop. It also checks the enrolled speaker and requires a fresh challenge to make
simple playback replay less useful. No generic “yes,” wake phrase or voice match
alone approves an action. The approval audit record contains the decision,
request reference and verification outcome, never the audio or full transcript.

Speaker matching is not cryptographic identity: recordings, synthetic speech,
noise and a similar voice can fool a detector. Before any sensitive action uses
spoken approval, evaluation must include false-accept/false-reject rates and
replay/spoof attempts. Until that gate passes, spoken approval cannot approve
`dangerous` actions; the existing visible approval path remains available.

## Learning and replaying routines

A routine is a reviewed sequence of existing ClaireAI tools, not an arbitrary
script. The owner selects a completed task to propose as a routine. The UI shows
the exact steps, arguments, paths, apps and expected effects. The owner edits and
confirms the routine before it is saved locally.

Every replay rechecks the current folder allow-list, OS adapter support, current
file state, stop state, step/time budget and tool schemas. Each action that
normally requires approval still asks for fresh approval; a saved routine never
stores or reuses an approval. The routine can be edited, disabled or deleted in
the UI. Password fields, credential material, hidden input and unreviewed model
text are never captured into a routine.

## Current implementation evidence

- The desktop Voice page has opt-in push-to-talk capture and transcript review,
  and the wake listener uses the local Whisper model for “Hey Claire.” On
  2026-11-27 a synthesized “Hey Claire, open the notes” clip was detected as
  `('hey claire', 'open the nose')`: phrase detection works end-to-end, but the
  small ASR misheard one command and a human-spoken acceptance run is still
  pending. Start/stop returned `running False`, `state stopped`.
- A detection only starts a turn. It does not grant a capability, and this local
  chat path does not invoke computer tools without the normal approval.
- `claireai.wakeword` has two local listeners. The default `stt` engine uses the
  installed faster-whisper model with a `hey claire` hotword and 0.45 s pre-roll,
  so it does not need a custom trained model; the alternative wraps
  openWakeWord. `scripts/wakeword_check.py` is an explicit openWakeWord check.
  The Tk **Voice** page starts and stops hands-free mode, persists the choice,
  shows a sidebar microphone indicator and turns a detection into the next chat
  turn. Audio is held in memory; detection grants no capability. On 2026-11-27 a
  synthesized “Hey Claire, open the notes” clip was detected end-to-end, and
  start/stop returned `running False`, `state stopped`. On 2026-11-25 the real
  `hey_jarvis` ONNX model scored a synthesized clip up to 0.9967 and ignored
  silence. A human-spoken acceptance test remains pending; openWakeWord still
  needs a matching custom model. Speaker enrollment, vocal approval and routine
  replay are unimplemented. eSpeak NG output remains optional and separately
  installed.
- The active development host is a Wayland session. The session portal advertises
  ScreenCast and Screenshot, but no screenshot tool is implemented. The normal
  screen-observation path should use the user-granted ScreenCast stream so frames
  stay transient; the portal Screenshot method returns a file URI and is not an
  appropriate default for ephemeral observation.
- This host exposes an ALSA capture device (`/dev/snd/pcmC0D0c`). `sounddevice`
  is installed but PortAudio is absent, which is why the wake-word listener uses
  the `arecord` fallback here. Speech recognition and wake-word packages are
  installed; actual human speech still needs a manual acceptance run, because a
  synthesized clip and mocked tests are not full platform evidence.

The implementation sequence is: live microphone acceptance on Linux and
Windows; wake-word detection and false-activation evaluation; voice enrollment
and spoof testing; request-bound spoken consent; reviewed routine capture and
replay. See `ROADMAP.md` for the acceptance gate and licensing constraints.
