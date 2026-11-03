"""Train ClaireAI's small visual-grounding component from random weights."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any

import torch

from claire_brain.tokenizer import BytePairTokenizer
from claire_brain.vision import ScratchGrounder

PAD = 257
IMAGE_SIZE = (160, 100)


def _load_rows(dataset: Path, split: str, tokenizer: BytePairTokenizer) -> list[dict[str, Any]]:
    rows = []
    for line in (dataset / "examples.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("split") != split:
            continue
        width, height = row["image_size"]
        x, y = row["target_center"]
        box = row["target_bbox"]
        image_path = (dataset / row["image"]).resolve()
        if dataset.resolve() not in image_path.parents:
            raise ValueError(f"image path escapes dataset: {row['image']}")
        rows.append({
            "image": image_path,
            "tokens": tokenizer.encode(row["instruction"])[:64],
            "target": [x / width, y / height],
            "box": [box[0] / width, box[1] / height, box[2] / width, box[3] / height],
            "image_sha256": row.get("image_sha256"),
        })
    return rows


def _batch(rows: list[dict[str, Any]], indices: list[int], torch_module, Image, ImageOps):
    images, token_rows, targets, boxes = [], [], [], []
    for index in indices:
        row = rows[index]
        with Image.open(row["image"]) as opened:
            converted = ImageOps.fit(opened.convert("RGB"), IMAGE_SIZE, method=Image.Resampling.BILINEAR)
            raw = bytearray(converted.tobytes())
            tensor = torch_module.frombuffer(raw, dtype=torch_module.uint8)
            image_tensor = tensor.view(IMAGE_SIZE[1], IMAGE_SIZE[0], 3).permute(2, 0, 1).float() / 255.0
        images.append(image_tensor)
        token_rows.append(row["tokens"])
        targets.append(row["target"])
        boxes.append(row["box"])
    max_tokens = max(len(tokens) for tokens in token_rows)
    padded = [tokens + [PAD] * (max_tokens - len(tokens)) for tokens in token_rows]
    return (
        torch_module.stack(images),
        torch_module.tensor(padded, dtype=torch_module.long),
        torch_module.tensor(targets, dtype=torch_module.float32),
        torch_module.tensor(boxes, dtype=torch_module.float32),
    )


@torch.inference_mode()
def _measure(model, rows, torch_module, Image, ImageOps) -> dict[str, float]:
    model.eval()
    image, tokens, target, boxes = _batch(rows, list(range(len(rows))), torch_module, Image, ImageOps)
    predicted = model(image, tokens)
    loss = torch_module.nn.functional.smooth_l1_loss(predicted, target).item()
    inside = (
        (predicted[:, 0] >= boxes[:, 0]) & (predicted[:, 0] <= boxes[:, 2])
        & (predicted[:, 1] >= boxes[:, 1]) & (predicted[:, 1] <= boxes[:, 3])
    )
    return {"loss": loss, "clicks_inside_target": int(inside.sum().item()), "examples": len(rows), "accuracy": float(inside.float().mean().item())}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path(__file__).resolve().parent / "data" / "external" / "gui-primitives")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.threads) < 1:
        parser.error("epochs, batch-size, and threads must be positive")
    try:
        from PIL import Image, ImageOps
    except ImportError:
        parser.error("install image support with: uv pip install --python assistant/.venv/bin/python -e 'brain-lab[vision]'")
    manifest_path = args.dataset / "manifest.json"
    if not manifest_path.is_file():
        parser.error(f"dataset not found; run brain-lab/fetch_gui_images.py first: {args.dataset}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("license") != "CC-BY-4.0 (synthetic examples and screenshots only)":
        parser.error("dataset license/provenance is not the expected synthetic-only source")

    raw_rows = [json.loads(line) for line in (args.dataset / "examples.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    image_hashes = {item["path"]: item["sha256"] for item in manifest.get("images", [])}
    for relative, expected in image_hashes.items():
        path = (args.dataset / relative).resolve()
        if args.dataset.resolve() not in path.parents or not path.is_file():
            parser.error(f"dataset image is missing or escapes its root: {relative}")
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            parser.error(f"dataset image checksum mismatch: {relative}")
    for row in raw_rows:
        if row.get("image") not in image_hashes:
            parser.error(f"example references an image absent from manifest: {row.get('image')}")
        row["image_sha256"] = image_hashes[row["image"]]
    train_texts = [row["instruction"] for row in raw_rows if row.get("split") == "train"]
    if not train_texts:
        parser.error("dataset has no training rows")
    tokenizer = BytePairTokenizer.train(train_texts, vocab_size=512)
    train_rows = _load_rows(args.dataset, "train", tokenizer)
    validation_rows = _load_rows(args.dataset, "validation", tokenizer)
    test_rows = _load_rows(args.dataset, "test", tokenizer)
    if not validation_rows or not test_rows:
        parser.error("dataset must contain non-empty validation and held-out test splits")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    model = ScratchGrounder(vocab_size=tokenizer.vocab_size)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    best_loss = float("inf")
    best_state = None
    for epoch in range(1, args.epochs + 1):
        model.train()
        indices = list(range(len(train_rows)))
        random.shuffle(indices)
        losses = []
        for start in range(0, len(indices), args.batch_size):
            selected = indices[start:start + args.batch_size]
            image, tokens, target, _ = _batch(train_rows, selected, torch, Image, ImageOps)
            predicted = model(image, tokens)
            loss = torch.nn.functional.smooth_l1_loss(predicted, target)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.item()))
        validation = _measure(model, validation_rows, torch, Image, ImageOps)
        print(f"epoch {epoch}/{args.epochs} train_loss={sum(losses) / len(losses):.4f} validation_loss={validation['loss']:.4f} validation_hit={validation['accuracy']:.3f}")
        if validation["loss"] < best_loss:
            best_loss = validation["loss"]
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

    assert best_state is not None
    model.load_state_dict(best_state)
    test_metrics = _measure(model, test_rows, torch, Image, ImageOps)
    output = args.output.expanduser() if args.output else Path("runtime-data/models/claire-vision-grounder.pt")
    output.parent.mkdir(parents=True, exist_ok=True)
    train_fingerprint = hashlib.sha256("\n".join(sorted(row.get("image_sha256") or "" for row in raw_rows if row.get("split") == "train")).encode()).hexdigest()
    run_manifest = {
        "architecture": "ScratchGrounder-cnn-text-pool-v1",
        "random_initialization": True,
        "source_dataset": manifest["upstream"],
        "source_revision": manifest["upstream_revision"],
        "license": manifest["license"],
        "training_image_hashes_sha256": train_fingerprint,
        "train_examples": len(train_rows),
        "validation_examples": len(validation_rows),
        "test_examples": len(test_rows),
        "best_validation_loss": best_loss,
        "held_out_test": test_metrics,
        "seed": args.seed,
        "epochs": args.epochs,
    }
    torch.save({"state_dict": best_state, "tokenizer_merges": tokenizer.merges, "manifest": run_manifest}, output)
    output.with_suffix(".json").write_text(json.dumps(run_manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"checkpoint": str(output), **run_manifest}, indent=2))
    print("This is an isolated visual-grounding research component; it is not yet connected to ClaireAI chat or screen control.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
