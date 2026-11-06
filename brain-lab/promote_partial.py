#!/usr/bin/env python3
"""Promote the best weights from a ``train_chat.py`` partial checkpoint.

``train_chat.py`` writes ``<output>.partial`` every evaluation interval so a
long CPU run can be interrupted.  That file contains the current weights, the
best-so-far weights, and the step numbers, but not the tokenizer merges (they
are deterministic from the corpus).  This script rebuilds the tokenizer and
writes a normal checkpoint that ``claireai.brain.ScratchBrain`` can load.

    assistant/.venv/bin/python brain-lab/promote_partial.py \
        --partial runtime-data/models/claire-chat-v2.pt.partial \
        --output runtime-data/models/claire-chat-v2.pt

It never overwrites the partial, and it refuses to write a checkpoint whose
tokenizer or architecture does not match the saved weights.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from claire_brain.tokenizer import BytePairTokenizer
from train_chat import corpus_texts, load_records

ROOT = Path(__file__).resolve().parent


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--partial", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, default=ROOT / "data" / "chat-v2")
    parser.add_argument("--vocab-size", type=int, default=1024)
    parser.add_argument("--context", type=int, default=256)
    parser.add_argument("--width", type=int, default=192)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--heads", type=int, default=6)
    parser.add_argument("--name", default="claire-chat-v2")
    parser.add_argument("--use-current", action="store_true", help="use the latest weights, not the best")
    args = parser.parse_args()

    partial = args.partial.expanduser()
    if not partial.is_file():
        parser.error(f"partial checkpoint not found: {partial}")
    payload = torch.load(partial, map_location="cpu", weights_only=True)
    state = payload.get("state_dict") if args.use_current else payload.get("best_state")
    if state is None:
        state = payload.get("state_dict")
    if state is None:
        parser.error("partial has neither best_state nor state_dict")

    train_path = args.corpus / "train.jsonl"
    if not train_path.is_file():
        parser.error(f"training corpus not found: {train_path}")
    records = load_records(train_path)
    tokenizer = BytePairTokenizer.train(corpus_texts(records), vocab_size=args.vocab_size)

    # Shape checks catch the most expensive mistake: a checkpoint that loads
    # silently into the wrong architecture.
    expected = {
        "token_embedding.weight": (args.vocab_size, args.width),
        "position_embedding.weight": (args.context, args.width),
    }
    for key, shape in expected.items():
        actual = tuple(state[key].shape)
        if actual != shape:
            parser.error(f"{key} has shape {actual}, expected {shape}")

    output = args.output.expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "format": 1,
        "name": args.name,
        "architecture": {
            "context": args.context,
            "width": args.width,
            "layers": args.layers,
            "heads": args.heads,
            "vocab_size": tokenizer.vocab_size,
        },
        "tokenizer": "corpus-bpe-v1",
        "tokenizer_merges": tokenizer.merges,
        "steps": int(payload.get("step", 0)),
        "best_step": int(payload.get("best_step", 0)),
        "training_loss": None,
        "validation_loss": float(payload.get("best_loss", float("nan"))),
        "prompt_format": "SYSTEM / USER / ASSISTANT, TOOL <name> for results, EOS after replies",
        "training_corpus": [
            {"file": str(train_path), "sha256": hashlib.sha256(train_path.read_bytes()).hexdigest()}
        ],
        "promoted_from_partial": str(partial),
    }
    torch.save({"state_dict": state, "manifest": manifest}, output)
    output.with_suffix(".json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"promoted step {manifest['best_step']} (validation loss {manifest['validation_loss']:.4f}) to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
