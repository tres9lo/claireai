# ClaireAI scratch model lab

This package contains the tokenizer and decoder-only transformer used by the
assistant. The model has no pretrained weights: its parameters are initialized
randomly by PyTorch, and its byte-pair merges are learned only from the local
corpus supplied to a training run.

## Prepare a corpus

Create `brain-lab/data/raw/` and add UTF-8 `.txt` files you own or are licensed
to use. Before training, record every source, license, retrieval date and
processing step in `docs/model-and-data-licenses.md` and a tracked manifest.
Raw corpus files and checkpoints are ignored by Git. No corpus is bundled yet.

## Install and train

From the repository root, install the assistant, CPU PyTorch extra, and lab:

```bash
uv venv --seed --python 3.12 assistant/.venv
uv pip install --python assistant/.venv/bin/python -e "assistant[dev,model]"
uv pip install --python assistant/.venv/bin/python -e ./brain-lab
export CLAIREAI_DATA_DIR=./runtime-data
assistant/.venv/bin/python brain-lab/train.py brain-lab/data/raw --steps 1000
```

The default model is intentionally small (128-wide, 4 layers, 256-token
context, up to 512 BPE tokens). Keep the first run short on CPU. `--output`,
`--context`, `--vocab-size`, `--batch-size`, and `--learning-rate` are
configurable. The checkpoint and a JSON manifest are written under the
configured ClaireAI data directory.

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
