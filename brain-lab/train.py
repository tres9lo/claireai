"""Train the scratch transformer on a UTF-8 text corpus (CPU by default).

Corpus format: plain .txt files, one or more. Only data you own or are licensed
to use should be included. The run records source file names and SHA-256 hashes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

import torch

from claire_brain.model import ScratchTransformer
from claire_brain.tokenizer import BytePairTokenizer


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("corpus", type=Path, help="directory containing licensed .txt files")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--context", type=int, default=256)
    parser.add_argument("--vocab-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, help="checkpoint path (default: ClaireAI data directory)")
    parser.add_argument("--improvement", action="store_true", help="continue the active checkpoint and log a candidate (requires UI opt-in)")
    parser.add_argument("--continue-from", type=Path, help="parent checkpoint to continue training from")
    args = parser.parse_args()
    if args.steps < 1 or args.batch_size < 1:
        parser.error("steps and batch-size must be positive")
    files = sorted(args.corpus.rglob("*.txt")) if args.corpus.is_dir() else []
    if not files:
        parser.error(f"no .txt training files found under {args.corpus}")
    docs = [(path, path.read_text(encoding="utf-8")) for path in files]
    settings = None
    if args.improvement:
        from claireai.config import Settings
        from claireai.runtime_control import RuntimeControl
        from claireai.improvements import ImprovementLedger

        settings = Settings()
        if not RuntimeControl(settings.control_state_path).read()["self_improvement"]:
            parser.error("self-improvement is disabled; enable it in the local presence UI first")
        parent = args.continue_from or settings.resolved_model_checkpoint
        try:
            parent_payload = torch.load(parent, map_location="cpu", weights_only=True)
            parent_manifest = parent_payload["manifest"]
            tokenizer = BytePairTokenizer(merges=[tuple(pair) for pair in parent_manifest["tokenizer_merges"]])
        except (OSError, KeyError, TypeError, ValueError) as exc:
            parser.error(f"cannot load parent scratch checkpoint: {exc}")
        if tokenizer.vocab_size > args.vocab_size:
            parser.error("--vocab-size must be at least the parent tokenizer vocabulary")
    else:
        parent = None
        parent_payload = None
        parent_manifest = None
        tokenizer = BytePairTokenizer.train([doc for _, doc in docs], vocab_size=args.vocab_size)
    tokens = torch.tensor([token for _, doc in docs for token in tokenizer.encode(doc)], dtype=torch.long)
    if len(tokens) < 2 * (args.context + 1):
        parser.error(f"corpus has {len(tokens)} tokens; need at least {2 * (args.context + 1)} for train/validation splits")
    split = int(len(tokens) * 0.9)
    train_tokens, validation_tokens = tokens[:split], tokens[split:]

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))
    model = ScratchTransformer(context=args.context, vocab_size=tokenizer.vocab_size)
    if parent_payload is not None:
        architecture = parent_manifest.get("architecture", {})
        if (architecture.get("context"), architecture.get("width"), architecture.get("layers"), architecture.get("heads"), architecture.get("vocab_size")) != (args.context, 128, 4, 4, tokenizer.vocab_size):
            parser.error("continued training requires the parent's architecture; match --context and tokenizer vocabulary")
        model.load_state_dict(parent_payload["state_dict"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    model.train()

    def batch_from(source):
        starts = torch.randint(0, len(source) - args.context - 1, (args.batch_size,))
        x = torch.stack([source[i:i + args.context] for i in starts])
        y = torch.stack([source[i + 1:i + args.context + 1] for i in starts])
        return x, y

    def validation_loss():
        model.eval()
        with torch.inference_mode():
            losses = [model(*batch_from(validation_tokens))[1].item() for _ in range(10)]
        model.train()
        return sum(losses) / len(losses)

    for step in range(1, args.steps + 1):
        x, y = batch_from(train_tokens)
        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step == 1 or step % 100 == 0 or step == args.steps:
            print(f"step {step}/{args.steps} train_loss={loss.item():.4f} validation_loss={validation_loss():.4f}")

    if settings is None:
        from claireai.config import Settings
        settings = Settings()
    if args.improvement:
        output = args.output.expanduser() if args.output else settings.resolved_data_dir / "models" / "candidates" / f"candidate-{__import__('uuid').uuid4().hex}.pt"
        try:
            output.resolve().relative_to((settings.resolved_data_dir / "models" / "candidates").resolve())
        except ValueError:
            parser.error("improvement candidates must be saved under <data-dir>/models/candidates")
    else:
        output = args.output.expanduser() if args.output else settings.resolved_model_checkpoint
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "format": 1,
        "architecture": {"context": args.context, "width": 128, "layers": 4, "heads": 4, "vocab_size": tokenizer.vocab_size},
        "tokenizer": "corpus-bpe-v1",
        "tokenizer_merges": tokenizer.merges,
        "seed": args.seed,
        "steps": args.steps,
        "parent_checkpoint": str(parent.resolve()) if parent else None,
        "training_loss": float(loss.item()),
        "validation_loss": validation_loss(),
        "corpus": [
            {
                "file": str(path.relative_to(args.corpus)),
                "sha256": hashlib.sha256(text.encode()).hexdigest(),
            }
            for path, text in docs
        ],
    }
    torch.save({"state_dict": model.state_dict(), "manifest": manifest}, output)
    output.with_suffix(".json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"checkpoint: {output}")
    if args.improvement:
        candidate_id = ImprovementLedger(settings.resolved_data_dir).record_candidate(checkpoint=output, parent=parent, manifest=manifest)
        print(f"improvement candidate: {candidate_id} (awaiting evaluation and owner review)")
    print("Note: training loss is not an evaluation of assistant or tool-use capability.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
