"""Train the scratch transformer on conversation records (CPU by default).

This is the chat-aware successor to ``train.py``.  The differences that matter:

- one training example per assistant turn, rendered with the same
  ``claire_brain.prompt`` code the runtime uses;
- EOS after every reply, so the model learns to stop instead of continuing into
  the next ``SYSTEM:`` block;
- the prompt is masked out of the loss, so capacity goes to producing replies;
- when a prompt is too long, the oldest turns are dropped, never the system or
  the current user request.

    assistant/.venv/bin/python brain-lab/build_chat_dataset.py
    assistant/.venv/bin/python brain-lab/train_chat.py \
        --corpus brain-lab/data/chat-v2 --steps 3000

The checkpoint is written in the same manifest format as ``train.py`` so
``claireai.brain.ScratchBrain`` can load it unchanged.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import time
from pathlib import Path
from typing import Any, Iterable, Sequence

import torch

from claire_brain.model import ScratchTransformer
from claire_brain.prompt import render_prompt
from claire_brain.tokenizer import BytePairTokenizer, EOS

ROOT = Path(__file__).resolve().parent


def load_records(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            record = json.loads(line)
            if record.get("messages"):
                records.append(record)
    return records


def corpus_texts(records: Iterable[dict[str, Any]]) -> list[str]:
    """Everything the tokenizer may learn from: prompts and replies."""
    texts: list[str] = []
    for record in records:
        system = str(record.get("system") or "")
        chunks = [system]
        for message in record.get("messages", []):
            chunks.append(str(message.get("content") or ""))
        texts.append("\n".join(chunks))
    return texts


def tokenizer_sample(texts: list[str], limit: int) -> list[str]:
    """A strided slice of the corpus for fitting merges.

    Tokenizer training holds every sequence in memory at once, so its footprint
    grows with the whole corpus. Merges are a corpus *statistic*, not a lookup
    table for specific examples, so a strided sample drawn from the already
    shuffled record order learns the same merges within a fraction of the RAM.
    The full corpus is still tokenised afterwards for the examples themselves.
    """
    total = sum(len(text) for text in texts)
    if limit <= 0 or total <= limit:
        return texts
    step = max(1, math.ceil(total / limit))
    return texts[::step]


def _message_role(message: dict[str, Any]) -> str:
    return str(message.get("role") or "user")


def _fit_history(
    system: str,
    history: Sequence[dict[str, Any]],
    tokenizer: BytePairTokenizer,
    limit: int,
) -> tuple[str, list[int]] | None:
    """Render a prompt no longer than ``limit`` tokens, dropping oldest turns.

    The system prompt and the most recent turn are always kept: they carry the
    task and the request.  Returns the prompt and its token ids, or ``None``
    when even that minimum does not fit.
    """
    kept = list(history)
    while True:
        prompt = render_prompt(system, kept)
        tokens = tokenizer.encode(prompt, append_eos=False)
        if len(tokens) <= limit:
            return prompt, tokens
        if len(kept) <= 1:
            return None
        # Drop the oldest turn.  A user message and the assistant reply that
        # follows it belong together, so drop the pair when it is a pair.
        drop = 2 if len(kept) >= 2 and _message_role(kept[0]) == "user" else 1
        kept = kept[drop:]
        if not kept:
            # Never throw away the current request just to make room.
            kept = [history[-1]]


def build_examples(
    records: Iterable[dict[str, Any]],
    tokenizer: BytePairTokenizer,
    *,
    context: int,
    max_target_tokens: int | None = None,
) -> list[tuple[list[int], int]]:
    """Return ``(token_ids, prompt_length)`` for every assistant turn."""
    examples: list[tuple[list[int], int]] = []
    target_cap = max_target_tokens or context
    for record in records:
        system = str(record.get("system") or "")
        messages = list(record.get("messages", []))
        for index, message in enumerate(messages):
            if _message_role(message) != "assistant":
                continue
            target = str(message.get("content") or "").strip()
            if not target:
                continue
            target_ids = tokenizer.encode(target, append_eos=True)
            if len(target_ids) > target_cap:
                target_ids = target_ids[:target_cap]
                target_ids[-1] = EOS
            history = messages[:index]
            budget = context + 1 - len(target_ids)
            if budget < 8:
                continue
            fitted = _fit_history(system, history, tokenizer, budget)
            if fitted is None:
                continue
            _, prompt_ids = fitted
            ids = prompt_ids + target_ids
            if len(ids) > context + 1:
                continue
            examples.append((ids, len(prompt_ids)))
    return examples


def collate(
    batch: Sequence[tuple[list[int], int]],
    *,
    pad: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    width = max(len(ids) for ids, _ in batch)
    x = torch.full((len(batch), width - 1), pad, dtype=torch.long)
    y = torch.zeros((len(batch), width - 1), dtype=torch.long)
    mask = torch.zeros((len(batch), width - 1), dtype=torch.float32)
    for row, (ids, prompt_len) in enumerate(batch):
        length = len(ids)
        x[row, : length - 1] = torch.tensor(ids[:-1], dtype=torch.long)
        y[row, : length - 1] = torch.tensor(ids[1:], dtype=torch.long)
        # Predictions start at the last prompt token and cover every reply token.
        mask[row, prompt_len - 1 : length - 1] = 1.0
    return x, y, mask


def evaluate(
    model: ScratchTransformer,
    examples: Sequence[tuple[list[int], int]],
    *,
    batch_size: int,
    pad: int,
) -> tuple[float, float]:
    model.eval()
    losses: list[float] = []
    with torch.inference_mode():
        for start in range(0, len(examples), batch_size):
            x, y, mask = collate(examples[start : start + batch_size], pad=pad)
            _, loss = model(x, y, loss_mask=mask)
            losses.append(float(loss))
    model.train()
    if not losses:
        return float("nan"), float("nan")
    mean = sum(losses) / len(losses)
    return mean, math.exp(min(mean, 20.0))


def sample(
    model: ScratchTransformer,
    tokenizer: BytePairTokenizer,
    prompt: str,
    *,
    max_new_tokens: int = 80,
) -> str:
    model.eval()
    tokens = tokenizer.encode(prompt, append_eos=False)[-model.context :]
    with torch.inference_mode():
        output = model.generate(
            torch.tensor([tokens], dtype=torch.long),
            max_new_tokens=max_new_tokens,
            temperature=0.0,
            repetition_penalty=1.15,
        )[0].tolist()[len(tokens) :]
    model.train()
    return tokenizer.decode(output).strip()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=ROOT / "data" / "chat-v6")
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--context", type=int, default=256)
    parser.add_argument(
        "--vocab-size", type=int, default=1024,
        help="vocabulary size for the tokenizer and the model embeddings",
    )
    parser.add_argument(
        "--width", type=int, default=192)
    parser.add_argument(
        "--layers", type=int, default=4)
    parser.add_argument(
        "--heads", type=int, default=6)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--min-learning-rate", type=float, default=3e-5)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--eval-interval", type=int, default=250)
    parser.add_argument("--eval-batches", type=int, default=8)
    parser.add_argument("--max-target-tokens", type=int, default=160)
    parser.add_argument(
        "--tokenizer-chars",
        type=int,
        default=3_000_000,
        help="fit BPE merges on at most this many characters (0 = use everything)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--name", default="claire-chat-v2")
    parser.add_argument("--resume", action="store_true", help="continue from <output>.partial if present")
    parser.add_argument(
        "--init-from",
        type=Path,
        default=None,
        help="warm-start the weights from a previous checkpoint of the same architecture",
    )
    parser.add_argument(
        "--tokenizer-from",
        type=Path,
        default=None,
        help="reuse the BPE merges in a checkpoint manifest; a warm start needs this," " because merges fitted on a different corpus renumber every token",
    )
    args = parser.parse_args()

    if args.width % args.heads:
        parser.error("--width must divide evenly by --heads")
    if min(args.steps, args.batch_size, args.context, args.vocab_size, args.layers, args.heads) < 1:
        parser.error("steps, batch-size, context, vocab-size, layers and heads must be positive")

    train_path = args.corpus / "train.jsonl"
    validation_path = args.corpus / "validation.jsonl"
    if not train_path.is_file() or not validation_path.is_file():
        parser.error(f"{args.corpus} must contain train.jsonl and validation.jsonl")
    train_records = load_records(train_path)
    validation_records = load_records(validation_path)
    if not train_records or not validation_records:
        parser.error("train.jsonl and validation.jsonl must both contain records")

    print(f"records: train={len(train_records)} validation={len(validation_records)}", flush=True)
    started = time.perf_counter()
    # Named for what it holds: an earlier version bound this to `sample`, which
    # shadowed the sampling function below and crashed every run at the very end,
    # after the training loop and before the checkpoint was written.
    if args.tokenizer_from is None and args.init_from is not None:
        sibling = args.init_from.expanduser().with_suffix(".json")
        if sibling.is_file():
            args.tokenizer_from = sibling
    if args.tokenizer_from is not None:
        manifest_path = args.tokenizer_from.expanduser()
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        merges = payload.get("tokenizer_merges")
        if not merges:
            parser.error(f"{manifest_path} carries no tokenizer_merges")
        tokenizer = BytePairTokenizer([tuple(pair) for pair in merges])
        if tokenizer.vocab_size != args.vocab_size:
            parser.error(
                f"{manifest_path} has vocab {tokenizer.vocab_size}, but --vocab-size is "
                f"{args.vocab_size}; reusing merges is what keeps the embeddings aligned"
            )
        print(
            f"tokenizer: reused {len(tokenizer.merges)} merges from {manifest_path}", flush=True
        )
    else:
        fit_texts = tokenizer_sample(corpus_texts(train_records), args.tokenizer_chars)
        tokenizer = BytePairTokenizer.train(fit_texts, vocab_size=args.vocab_size)
        sample_chars = sum(len(text) for text in fit_texts)
        print(
            f"tokenizer: vocab={tokenizer.vocab_size} merges={len(tokenizer.merges)} "
            f"on {sample_chars:,} sampled characters ({time.perf_counter()-started:.1f}s)",
            flush=True,
        )
        del fit_texts

    train_examples = build_examples(
        train_records, tokenizer, context=args.context, max_target_tokens=args.max_target_tokens
    )
    validation_examples = build_examples(
        validation_records, tokenizer, context=args.context, max_target_tokens=args.max_target_tokens
    )
    if not train_examples or not validation_examples:
        parser.error("no usable examples after context fitting")
    lengths = sorted(len(ids) for ids, _ in train_examples)
    print(
        f"examples: train={len(train_examples)} validation={len(validation_examples)} "
        f"median_tokens={lengths[len(lengths)//2]} max_tokens={lengths[-1]}",
        flush=True,
    )

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(max(1, min(4, torch.get_num_threads() * 2)))
    model = ScratchTransformer(
        context=args.context,
        width=args.width,
        layers=args.layers,
        heads=args.heads,
        vocab_size=tokenizer.vocab_size,
    )
    parameters = sum(weight.numel() for weight in model.parameters())
    print(
        f"model: width={args.width} layers={args.layers} heads={args.heads} "
        f"context={args.context} vocab={tokenizer.vocab_size} params={parameters:,}",
        flush=True,
    )
    if args.init_from is not None:
        warm = torch.load(args.init_from.expanduser(), map_location="cpu", weights_only=True)
        try:
            model.load_state_dict(warm["state_dict"])
        except RuntimeError as exc:
            parser.error(f"--init-from {args.init_from} does not fit this architecture: {exc}")
        print(f"warm start: weights from {args.init_from}", flush=True)
    if args.output is not None:
        output = args.output.expanduser()
    else:
        data_dir = args.data_dir.expanduser() if args.data_dir else None
        if data_dir is None:
            from claireai.config import Settings

            data_dir = Settings().resolved_data_dir
        output = data_dir / "models" / f"{args.name}.pt"
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_name(output.name + ".partial")

    best_loss = float("inf")
    best_state = None
    best_step = 0
    start_step = 1
    if args.resume and partial.is_file():
        payload = torch.load(partial, map_location="cpu", weights_only=True)
        model.load_state_dict(payload["state_dict"])
        best_loss = float(payload.get("best_loss", best_loss))
        best_state = payload.get("best_state")
        best_step = int(payload.get("best_step", 0))
        start_step = int(payload.get("step", 0)) + 1
        print(f"resumed from {partial} at step {start_step - 1}", flush=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.95), weight_decay=0.01)
    model.train()

    def lr_at(step: int) -> float:
        if step <= args.warmup:
            return args.learning_rate * step / max(1, args.warmup)
        progress = (step - args.warmup) / max(1, args.steps - args.warmup)
        cosine = 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
        return args.min_learning_rate + (args.learning_rate - args.min_learning_rate) * cosine

    last_loss = float("nan")
    train_started = time.perf_counter()
    for step in range(start_step, args.steps + 1):
        for group in optimizer.param_groups:
            group["lr"] = lr_at(step)
        batch = [random.choice(train_examples) for _ in range(args.batch_size)]
        x, y, mask = collate(batch, pad=257)
        _, loss = model(x, y, loss_mask=mask)
        last_loss = loss.item()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        done = step - start_step + 1
        if step == start_step or step % 50 == 0 or step == args.steps:
            elapsed = time.perf_counter() - train_started
            print(
                f"step {step}/{args.steps} loss={loss.item():.4f} lr={lr_at(step):.2e} "
                f"{elapsed/max(1,done):.2f}s/step",
                flush=True,
            )
        if step % args.eval_interval == 0 or step == args.steps:
            eval_set = validation_examples[: args.eval_batches * args.batch_size]
            mean, perplexity = evaluate(model, eval_set, batch_size=args.batch_size, pad=257)
            print(f"  validation loss={mean:.4f} perplexity={perplexity:.2f}", flush=True)
            if mean < best_loss:
                best_loss = mean
                best_step = step
                best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
            # A multi-hour CPU run must survive an interrupted shell.  Write the
            # current weights and the best-so-far weights, then keep going.
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "best_state": best_state,
                    "best_loss": best_loss,
                    "best_step": best_step,
                    "step": step,
                },
                partial,
            )

    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"restored best checkpoint from step {best_step} (validation loss {best_loss:.4f})", flush=True)

    # A quick, deterministic look at what the model actually says.  Training loss
    # alone has already fooled this project once; this is the cheapest check that
    # the reply is on-topic and stops.
    probes = [
        render_prompt("You are ClaireAI, a friendly local assistant.", [{"role": "user", "content": "hello"}], None),
        render_prompt("You are ClaireAI, a friendly local assistant.", [{"role": "user", "content": "what can you do"}], None),
        render_prompt("You are ClaireAI. Never claim an action succeeded without a tool result.", [{"role": "user", "content": "thanks, that is all for now"}], None),
    ]
    print("\nsamples:", flush=True)
    for prompt in probes:
        print(f"  {sample(model, tokenizer, prompt, max_new_tokens=60)!r}", flush=True)

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
        "seed": args.seed,
        "initialized_from": str(args.init_from) if args.init_from is not None else None,
        "steps": args.steps,
        "best_step": best_step,
        # A resumed run that had no steps left reports no training loss; NaN in
        # the manifest would serialise as invalid strict JSON.
        "training_loss": None if last_loss != last_loss else last_loss,
        "validation_loss": best_loss,
        "prompt_format": "SYSTEM / USER / ASSISTANT, TOOL <name> for results, EOS after replies",
        "training_corpus": [
            {"file": str(train_path), "sha256": hashlib.sha256(train_path.read_bytes()).hexdigest()}
        ],
        "validation_corpus": [
            {"file": str(validation_path), "sha256": hashlib.sha256(validation_path.read_bytes()).hexdigest()}
        ],
    }
    torch.save({"state_dict": model.state_dict(), "manifest": manifest}, output)
    output.with_suffix(".json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    partial.unlink(missing_ok=True)
    print(f"checkpoint: {output}")
    print("Note: training loss is not an evaluation of assistant capability; run scripts/eval_model.py next.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
