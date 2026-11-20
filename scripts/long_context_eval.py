#!/usr/bin/env python3
"""Probe whether the installed checkpoint can retrieve a fact deeper in context.

This is a diagnostic, not a promotion gate. A pass means the model retrieved a
marker at a tested prompt length; a failure means the larger window is not yet
useful for that kind of recall.

    assistant/.venv/bin/python scripts/long_context_eval.py --lengths 128,320,480
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

from claire_brain.model import ScratchTransformer
from claire_brain.tokenizer import BytePairTokenizer


def _load(checkpoint: Path):
    payload = torch.load(checkpoint.expanduser(), map_location="cpu", weights_only=True)
    manifest = payload["manifest"]
    model = ScratchTransformer(**manifest["architecture"])
    model.load_state_dict(payload["state_dict"], strict=True)
    model.eval()
    tokenizer = BytePairTokenizer([tuple(pair) for pair in manifest["tokenizer_merges"]])
    return model, tokenizer, manifest


def _prompt(tokenizer: BytePairTokenizer, target: int) -> str:
    head = (
        "SYSTEM: You are ClaireAI. Answer the question using the conversation.\n"
        "USER: Remember this access code for later: ORANGE-742.\n"
        "ASSISTANT: Understood.\n"
        "USER: "
    )
    tail = "\nWhat was the access code I asked you to remember?\nASSISTANT:"
    filler = "The archive contains routine records and no action is needed. "
    # Estimate from one encoded filler first. Encoding the growing prompt on
    # every iteration is quadratic and makes a 10K probe spend minutes in the
    # tokenizer before the model is even called.
    filler_tokens = max(1, len(tokenizer.encode(filler, append_eos=False)))
    base_tokens = len(tokenizer.encode(head + tail, append_eos=False))
    copies = max(0, (target - base_tokens) // filler_tokens)
    text = head + filler * copies
    while len(tokenizer.encode(text + tail, append_eos=False)) > target and copies:
        copies -= 1
        text = head + filler * copies
    return text + tail


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path.home() / ".local/share/ClaireAI/models/claire-scratch.pt",
    )
    parser.add_argument("--lengths", default="128,320,480")
    parser.add_argument(
        "--new-tokens",
        type=int,
        default=12,
        help="answer tokens per probe; keep this small for a 10K CPU run",
    )
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    model, tokenizer, manifest = _load(args.checkpoint)
    length_cap = int(model.context)
    results = []
    for raw in args.lengths.split(","):
        try:
            wanted = int(raw.strip())
        except ValueError:
            continue
        prompt = _prompt(tokenizer, min(wanted, length_cap - 8))
        ids = tokenizer.encode(prompt, append_eos=False)[-length_cap:]
        started = time.perf_counter()
        with torch.inference_mode():
            output = model.generate(
                torch.tensor([ids], dtype=torch.long),
                max_new_tokens=max(1, args.new_tokens),
                temperature=0.0,
                repetition_penalty=1.15,
            )[0].tolist()[len(ids):]
        elapsed = time.perf_counter() - started
        reply = tokenizer.decode(output).strip()
        passed = "ORANGE-742" in reply or "ORANGE 742" in reply.upper()
        results.append(
            {
                "requested_tokens": wanted,
                "prompt_tokens": len(ids),
                "passed": passed,
                "reply": reply[:240],
                "seconds": round(elapsed, 2),
            }
        )
        print(
            f"{wanted:>5} requested / {len(ids):>5} actual: "
            f"{'PASS' if passed else 'FAIL'} {elapsed:6.2f}s {reply[:120]!r}"
        )
    print(f"context={length_cap} checkpoint={args.checkpoint} name={manifest.get('name')}")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    return 0 if all(item["passed"] for item in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
