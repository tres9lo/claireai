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
The curated starting catalog is [`data-sources.json`](data-sources.json) with
licence/access notes in [`docs/training-data-sources.md`](../docs/training-data-sources.md).
Sources marked `do-not-use` are excluded rather than scraped or worked around.
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

## One-command bilingual training

For a reproducible run that fetches the bounded licensed sources, builds the
authored corpus, merges it, trains a fresh candidate and runs both fixed
evaluations:

```bash
assistant/.venv/bin/python brain-lab/train_claireai.py --profile full
```

The command writes only to `brain-lab/data/chat-v14`, `brain-lab/data/chat-v15`
and `runtime-data/models/candidates/`. It **never installs** the candidate. Only
after the tool and conversation gates pass should the owner promote it
explicitly. A `*.pipeline.json` file beside the candidate records the exact
stage exit codes and corpus counts.

| Profile | Authored quota | External sample per source | Steps | Model | Intended use |
| --- | ---: | ---: | ---: | --- | --- |
| `quick` | 8,000 | 500 | 1,500 | 224 wide, 5 layers, 512 context | pipeline smoke run on this laptop |
| `full` | 34,000 | 4,000 | 8,000 | 224 wide, 5 layers, 512 context | practical CPU quality run |
| `huge` | 60,000 | 8,000 | 15,000 | 320 wide, 6 layers, 768 context | larger-data run; many hours on CPU |

Even `huge` is a bounded sample, not a claim that the model has general
knowledge. The 10K window remains unproven: the RoPE candidate loads and runs,
but the recall probe failed, so a larger window is not the default.

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
  --corpus brain-lab/data/chat-v14 --steps 6000 \
  --output runtime-data/models/claire-chat-v14.pt
```

The builder mixes original synthetic templates (small talk, refusals, intent
routing, slot extraction, tool-call transcripts, summaries, in-context recall
and an authored English/French language curriculum), Dolly 15k, TinyStories and
the repository's bootstrap examples. It removes any user turn that exactly
matches a prompt in either fixed suite (`scripts/model_eval_cases.json` or
`scripts/conversation_cases.json`), so both evaluations stay held out.

`train_chat.py` stores token ids as compact arrays rather than boxed Python
integers. That keeps a 50k-record corpus within this host's RAM budget; the old
representation is what forced the earlier memory-bounded corpus.

Measure it with both suites:

```bash
assistant/.venv/bin/python scripts/eval_model.py \
  --model-checkpoint runtime-data/models/claire-chat-v15.pt --runs 1
assistant/.venv/bin/python scripts/conversation_eval.py \
  --model-checkpoint runtime-data/models/claire-chat-v15.pt
```

`eval_model.py` checks one-shot formatting and safety. `conversation_eval.py`
runs multi-turn conversations and injects tool results; both score with plain
code, never with another model. The rules behind this recipe — response-only
loss, masked tool results, held-out contamination checks and the install gate —
are in [`docs/training-playbook.md`](../docs/training-playbook.md).

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
and `assistant/.venv/bin/python scripts/eval_model.py --runs 1`. Open the
desktop window with `assistant/.venv/bin/claireai --data-dir ./runtime-data`
(adjust the data directory to match your runtime). Browser access is off by
default. The pause control is checked after in-flight model calls; this first
version does not terminate the CLI process itself.

## External corpora and context expansion

The catalog in `data-sources.json` is a starting point, not blanket permission.
`fetch_corpora.py` downloads bounded samples and records licence, URL, date and
SHA-256 beside each file. `merge_corpora.py` merges those records with `chat-v14`,
removes held-out evaluation prompts, exact duplicate pairs and pairs that also
appear in validation, then writes `chat-v15`. The manifest records how many rows
each cleanup removed:

```bash
assistant/.venv/bin/python brain-lab/fetch_corpora.py --source all --limit 800
assistant/.venv/bin/python brain-lab/merge_corpora.py \
  --base brain-lab/data/chat-v14 --output brain-lab/data/chat-v15
```

The current mix is GSM8K (MIT), FineWeb-Edu (ODC-By 1.0), French FineWeb-2
(ODC-By 1.0), OASST1 English/French (Apache-2.0), and the already-present
Dolly/TinyStories material. OASST1 is read from the released `ready` message
archive instead of the rate-limited rows API, so both languages are reproducible.
FineWeb answers are extractive leads, not model output. A failed fetch is
reported rather than written as an empty success.

## Bilingual curriculum

The authored `languages` bucket teaches everyday English and French turn-taking,
common vocabulary, grammar and approximate pronunciation, plus French variants
of the file/summary/chit-chat intents. It is deliberately not a dictionary: a
224-wide model should learn how to answer a language question, not memorise a
lexicon. The external OASST1 and FineWeb-2 French samples add human-written
French dialogue and prose. Language practice is balanced against the product's
tool/format records; it does not replace the deny-by-default behaviour.

## Image-training placeholder

`image_training.py` and `data/image-training/manifest.json` reserve the future
vision schema (image path, instruction, point/box/text target, split, source and
licence). The module validates the manifest and explicitly refuses to train;
the shipped chat transformer remains text-only. No images are bundled, and
`fetch_gui_images.py` remains the only optional image downloader.

To widen an existing checkpoint without changing its learned weights:

```bash
assistant/.venv/bin/python brain-lab/expand_context.py \
  --source ~/.local/share/ClaireAI/models/claire-scratch.pt \
  --context 512 \
  --output runtime-data/models/candidates/claire-context512-base.pt
assistant/.venv/bin/python brain-lab/install_context_upgrade.py \
  --candidate runtime-data/models/candidates/claire-context512-base.pt \
  --installed ~/.local/share/ClaireAI/models/claire-scratch.pt --yes
```

The installer refuses unless every old tensor and the old position rows match,
and it archives the previous checkpoint. The new manifest also records
`trained_context` (the old window), and `ScratchBrain` fits prompts to that
limit: copying rows is not the same as training them. This is a structural
window increase, not a substitute for the training-candidate gate.

For a long window, convert to rotary positions. This removes only the learned
position table; every other tensor is copied. The model uses
`scaled_dot_product_attention` so a 10K window does not allocate a 10K x 10K
mask per layer:

```bash
assistant/.venv/bin/python brain-lab/expand_context.py \
  --source ~/.local/share/ClaireAI/models/claire-scratch.pt \
  --context 10240 --position-encoding rope \
  --output runtime-data/models/candidates/claire-context10240-rope.pt
assistant/.venv/bin/python scripts/long_context_eval.py \
  --checkpoint runtime-data/models/candidates/claire-context10240-rope.pt \
  --lengths 512,2048,8192,10240 --new-tokens 4
```

Measured on the reference i5/7 GB host: 9,734 prompt tokens, 18.7 s for four
generated tokens, 461 MB peak RSS. The recall probe still failed at every length,
so the 10K window is infrastructure, not a proven capability. Train and gate a
candidate before installing it; the old learned-position checkpoint remains a
valid separate architecture.

## Limits

Training loss only measures next-token prediction on this corpus. It does not
prove that the model can converse, follow instructions, select tools, make
jokes, or report accurately. Do not enable writes based on training loss. Use
the fixed model evaluation and the existing runtime's schema validation,
allow-lists, approvals, and audit log. A useful general assistant requires a
large, diverse, lawful corpus, repeated evaluation, and much more compute than
the currently documented laptop provides.
