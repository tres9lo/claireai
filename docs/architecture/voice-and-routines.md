# Voice, consent and routines

This document records the owner requirements and the security boundary for the
hands-free assistant. It is a target design, not a claim that speech capture,
wake-word detection, voice identity, spoken approvals or routine replay are
implemented. Current speech support is optional local text-to-speech only.

## Voice interaction

The voice pipeline is local and visible:

1. The owner explicitly enables the microphone and sees a persistent listening
   indicator plus a one-action mute/stop control.
2. A local wake-word detector recognizes the configurable phrase (initial target:
   “Hey Claire”) from a rolling in-memory audio window. It does not save or upload
   that window.
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

- The local HUD now has opt-in push-to-talk capture and transcript review, and
  can install/use an optional local Whisper tiny speech-recognition model. The
  user still presses Send; this local chat path does not invoke computer tools.
- There is no wake-word engine, speaker enrollment, vocal approval or routine
  recorder/replayer yet. eSpeak NG output remains optional and separately
  installed.
- The active development host is a Wayland session. The session portal advertises
  ScreenCast and Screenshot, but no screenshot tool is implemented. The normal
  screen-observation path should use the user-granted ScreenCast stream so frames
  stay transient; the portal Screenshot method returns a file URI and is not an
  appropriate default for ephemeral observation.
- This host has no `/dev/snd` capture device, and the project environment currently
  has no speech-recognition or wake-word packages. UI/runtime audio behavior must
  be manually tested on a host with a microphone; mocked tests are not platform
  evidence.

The implementation sequence is: live microphone acceptance on Linux and
Windows; wake-word detection and false-activation evaluation; voice enrollment
and spoof testing; request-bound spoken consent; reviewed routine capture and
replay. See `ROADMAP.md` for the acceptance gate and licensing constraints.
