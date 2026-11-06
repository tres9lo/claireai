# ClaireAI scratch model lab

This package contains the tokenizer and decoder-only transformer used by the
assistant. The model has no pretrained weights: its parameters are initialized
randomly by PyTorch, and its byte-pair merges are learned only from the local
corpus supplied to a training run.

## Prepare a corpus

The repository includes `brain-lab/data/bootstrap-v1/`, a deterministic,
original synthetic dataset under the project's MIT license. It contains
computer-task text examples, structured image annotations and 80 generated SVG
screens. No third-party material or personal data is included. To regenerate
it, run `assistant/.venv/bin/python brain-lab/build_dataset.py --examples 320`.

For additional data, add only UTF-8 text you own or are licensed to use under
`brain-lab/data/raw/`. Record every source, license, retrieval date and
processing step in `docs/model-and-data-licenses.md` and a tracked manifest.
Raw corpus files and checkpoints are ignored by Git.

An optional downloader fetches the synthetic-only subset of GUI-Primitives
(CC BY 4.0) into git-ignored local storage. It pins the upstream revision,
records hashes and attribution, reserves clean human-verified examples for
evaluation, and excludes real application screenshots:

```bash
assistant/.venv/bin/python brain-lab/fetch_gui_images.py
```

Those image/annotation pairs are prepared for the visual model stage. The
current transformer is text-only, so do not merge them into its text corpus.

## Train a conversation model (recommended)

`train.py` treats the corpus as one long document. That is enough for prose, but
it teaches a chat model to continue the document instead of answering: replies
run into the next `SYSTEM:` block and the prompt itself is trained as a target.
`train_chat.py` fixes the things that mattered most:

- one example per assistant turn, rendered by `claire_brain.prompt` — the same
  code `ScratchBrain` uses at inference;
- EOS after every reply, so the model learns where a turn ends;
- the prompt is masked from the loss, so capacity goes to replies;
- long prompts drop the oldest turns, never the system prompt or current request.

Build the conversation corpus and train:

```bash
assistant/.venv/bin/python brain-lab/build_chat_dataset.py
assistant/.venv/bin/python brain-lab/train_chat.py \
  --corpus brain-lab/data/chat-v2 --steps 6000 \
  --output runtime-data/models/claire-chat-v2.pt
```

The builder mixes original synthetic templates (small talk, refusals, intent
routing, slot extraction, tool-call transcripts, summaries and in-context
recall), Dolly 15k, TinyStories and the repository's bootstrap examples. It
removes any user turn that exactly matches a prompt in either fixed suite
(`scripts/model_eval_cases.json` or `scripts/conversation_cases.json`), so both
evaluations stay held out.

Measure it with both suites:

```bash
assistant/.venv/bin/python scripts/eval_model.py \
  --model-checkpoint runtime-data/models/claire-chat-v2.pt --runs 1
assistant/.venv/bin/python scripts/conversation_eval.py \
  --model-checkpoint runtime-data/models/claire-chat-v2.pt
```

`eval_model.py` checks one-shot formatting and safety. `conversation_eval.py`
runs multi-turn conversations and injects tool results; both score with plain
code, never with another model.

## Train the visual-grounding prototype

The isolated CNN+text component is also initialized from random weights. It
trains on the synthetic PNGs and instructions, validates on a disjoint screen
set, then reports click-within-target accuracy once on the clean held-out set:

```bash
uv pip install --python assistant/.venv/bin/python -e "brain-lab[vision]"
assistant/.venv/bin/python brain-lab/vision_train.py --epochs 20
```

This is an experiment, not a screen-reading feature in the assistant. Its
current small dataset and untrained baseline are unlikely to produce useful
grounding accuracy. The checkpoint is kept separate from the chat model and is
not allowed to control the computer.

## Install and train

From the repository root, install the assistant, CPU PyTorch extra, and lab:

```bash
uv venv --seed --python 3.12 assistant/.venv
uv pip install --python assistant/.venv/bin/python -e "assistant[dev,model]"
uv pip install --python assistant/.venv/bin/python -e ./brain-lab
export CLAIREAI_DATA_DIR=./runtime-data
assistant/.venv/bin/python brain-lab/train.py brain-lab/data/bootstrap-v1/text --steps 1000
```

The default model is intentionally small (128-wide, 4 layers, 256-token
context, up to 512 BPE tokens). Keep the first run short on CPU. `--output`,
`--context`, `--vocab-size`, `--batch-size`, and `--learning-rate` are
configurable. The checkpoint and a JSON manifest are written under the
configured ClaireAI data directory. For split-aware corpora, the trainer learns
token merges on `train/`, measures against `validation/`, and ignores `test/`.

## Owner-reviewed improvement runs

The presence panel's **Enable self-improvement** switch is off by default. When
enabled, it permits an explicit training command to continue from the active
checkpoint and save a separate candidate. It does not let the model edit code,
change permissions, start training itself, or replace the active model.

```bash
assistant/.venv/bin/python brain-lab/train.py brain-lab/data/raw --steps 200 \
  --improvement
assistant/.venv/bin/python scripts/eval_model.py --runs 1 \
  --model-checkpoint runtime-data/models/candidates/<candidate>.pt
```

Candidates and evaluation outcomes appear in the panel's append-only
improvement log. Only a fully passing run of the repository's fixed evaluation
suite can be approved; custom case files are logged but cannot qualify. The
owner can reject any candidate or explicitly promote an approved one; promotion
archives the current checkpoint first. Training data and candidate files remain
local. Disabling the switch prevents new improvement training runs.

After a checkpoint exists, run `assistant/.venv/bin/python scripts/doctor.py`
and `assistant/.venv/bin/python scripts/eval_model.py --runs 1`. Start the
matching local control panel with
`assistant/.venv/bin/python scripts/presence_preview.py --data-dir ./runtime-data`
(adjust the data directory to match your runtime). Browser access is off by
default. The pause control is checked after in-flight model calls; this first
version does not terminate the CLI process itself.

## Limits

Training loss only measures next-token prediction on this corpus. It does not
prove that the model can converse, follow instructions, select tools, make
jokes, or report accurately. Do not enable writes based on training loss. Use
the fixed model evaluation and the existing runtime's schema validation,
allow-lists, approvals, and audit log. A useful general assistant requires a
large, diverse, lawful corpus, repeated evaluation, and much more compute than
the currently documented laptop provides.
