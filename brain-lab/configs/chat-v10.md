# Capacity experiment config: chat-v10

This directory holds the config for the fourth research training run, written as
its own tracked file so the exact shape of the failed run is recoverable without
reading a log. It is the experiment labelled **R2** in the private plan
(`NEXT_STEPS.md`): change capacity, not the corpus mix.

## Why this experiment, in one paragraph

Three fine-tunes (chat-v5, v6, v7) each bought tool accuracy with conversation
accuracy, and none held both against chat-v4. At 3.34M parameters with a 288-token
window the model cannot hold tool discrimination and conversation at once. The
remaining lever with evidence behind it is capacity — vocabulary and width — which
is what this run tests.

## What this run changes, relative to chat-v4

chat-v4: vocab 1152, width 224, layers 5, heads 7, context 288, LR 3e-4→3e-5,
warmup to 3e-4 at step 100, 6000 steps, 0.72s/step, validation loss 1.3826.

chat-v10 target: vocab 4096, width 288, layers 6, heads 8, context 512, LR
3e-4→3e-5, warmup to 3e-4 at step 100, steps selected on the resource budget
below, validation loss target ~1.7 or lower.

The comparison to chat-v4 is fair only if the run started from scratch. A warm
start from chat-v4 on a larger architecture is not a capacity test, it is just a
long run with a bigger window. If this run uses `--init-from`, it is no longer the
R2 capacity experiment and the run notes must say so.

## Resource budget and hard stop

Host: i5-6300U (2c/4t), 7.1 GB RAM, no CUDA, ~60GB spare on the root volume at
the time of writing. The run writes only into `runtime-data/`, which is gitignored.

Hard stop — both of the following, whichever hits first:

1. **Time budget:** 12 hours wall clock from the moment the supervised run starts.
   At chat-v4's 0.72s/step, 6000 steps was ~80 minutes; this run's steps per
   second will be lower because the model is bigger and the context longer, so the
   time budget is the binding constraint, not the step count. The run guard watches
   wall clock and stops the supervised training loop when the budget is spent; it
   does **not** kill the invoking shell or any other process.
2. **Disk budget:** do not let `runtime-data/` grow past 80% of the root volume.
   Before the run starts, the guard computes the predicted artifact footprint for
   this run (final `.pt` + `.json` manifest + periodic `.partial` checkpoints + log
   + eval logs if eval is run on schedule) and refuses to start if there is not
   enough headroom for that footprint plus the 80% safety margin. If the run would
   breach the disk budget mid-flight, the guard truncates the current log and writes
   a run-end note; it does not kill the invoking shell or anything else.

## Artifact retention profile

Default behavior for every run here:

- Keep the final checkpoint if one was written (best weights + manifest).
- Keep the `.partial` checkpoint only if the run was interrupted; promote or delete
  it on a clean finish, never leave both final and partial lying around unless the
  run was interrupted.
- Keep the run log and the run-end note.
- Never touch anything outside `runtime-data/` for the run.
- Never overwrite an installed checkpoint (`claire-scratch.{pt,json}`) unless the
  run passed the install gate, and the install gate is an explicit hand-off, not an
  automatic step.

For this experiment specifically, because chat-v10 is not an install candidate on size
alone and the comparison must be fair to chat-v4, the run must not write to
`claire-scratch.{pt,json}` even if it finishes. It writes to its own output name only:
`claire-chat-v10.{pt,json}` plus the `.partial` checkpoints along the way. The
run's `--output` is `runtime-data/models/claire-chat-v10.pt` and its manifest is
`runtime-data/models/claire-chat-v10.json`; nothing it writes is named
`claire-scratch`.

## Eval schedule (do NOT eval on every version write)

The failing pattern was eval-on-every-version-prompt mismatch. This run does not do
that. Eval only at:

- pre-set step boundaries (for example 3000, 6000, and the final step), and
- whenever a new best validation loss is recorded,

whichever is less frequent. A single eval run is a sample, not a result, so any
post-run measurement that is meant to be evidence uses `--runs 3` (temperature 0.2)
for conversation and the same run count for tool calls, compared against chat-v4 at
the same run count on the same case file.

## Corpus

Build the corpus with the same builder that produced chat-v9's calendar, with the
chat-v10 vocabulary and a holdout for the eval prompts.

- Corpus version: `claireai-chat-v10` (builder default now `brain-lab/build_chat_dataset.py`;
  manifest field `dataset` should read `claireai-chat-v10`).
- Builder: `brain-lab/build_chat_dataset.py` with `--output brain-lab/data/chat-v10`.
  The builder is vocabulary- and context-agnostic — it emits JSONL records, not tokens —
  so a larger vocab (4096) and longer context (512) are traits of the training run, not
  of the corpus. If a future builder ever needs to hold a prompt out of a
  larger-vocabulary corpus, add that logic to the builder.
- The eval prompts held out of the corpus stay held out: no training prompt equals any
  prompt in `scripts/tool_call_cases.json` (19 cases / 165 checks) or
  `scripts/model_eval_cases.json` (13 cases). Verified at build time (see next step's
  pre-launch check).
- Calendar: present from the first step, as a dedicated `calendar_tasks` bucket with
  per-prompt routing, so the calendar has its own mass and the model does not arrive at
  calendar args by accident late in a warm start.

## Fairness checklist before the run is called evidence

Before this run is used as evidence for anything, confirm:

1. started from scratch (no `--init-from`; `--tokenizer-from` only if starting from
   scratch with a pre-fit tokenizer, and if so, say which one and why);
2. measured at the same run count as chat-v4 (one eval run, three conversation runs,
   three tool-call runs on the same 19-case / 165-check file);
3. tool checks compared against chat-v4 68/165, conversation cases against chat-v4
   14/21;
4. the run did not install itself over chat-v4 by accident;
5. the run artifacts are still out of git (gitignored runtime-data), and only the
   config + run notes + the builder changes (if any) are in git.

## Open questions this config does not decide

- Exact step count: derived from the time budget at run start, not fixed in advance, because
  the bigger model (3.34M→~4.9M) and longer context (288→512) change seconds per step.
  For a first supervised run under this config, plan for about 12h of wall-clock and
  leave a step-count note in the run-end record for the post-run report.
- Whether a warm start from chat-v4 is worth trying in a separate experiment: no, for this
  run. If someone wants that, it is a different experiment with its own config.
- Whether a larger context (512) is worth the seconds-per-step cost: this run tests it once
  under the budget above; a repeat is a new run with its own notes.

## Run guard (deterministic hard stop)

The supervised run is wrapped by a guard script
(`brain-lab/scripts/run_chat_v10_guard.sh`). That guard:

- sets `CLAIREAI_DATA_DIR` to `$PWD/runtime-data` so all artifacts land under the
  gitignored tree;
- chooses `--steps` from the remaining 12h budget measured at launch, not from a fixed
  number;
- refuses to start if there is not enough root-volume headroom for the predicted
  footprint plus the 80% safety margin;
- tails the log while the run is alive;
- on either hard-stop trigger, stops only the supervised training loop, truncates the
  current log, writes a run-end note, and exits. It never kills the invoking shell or any
  other process.

The guard is not part of the model; it is just the supervised wrapper for this run. If you
run the training command by hand, you take over the guard's responsibilities.
