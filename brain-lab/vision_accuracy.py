"""Measure the local vision model's accuracy on a standard, labelled benchmark.

The model ranks phrases against an image; this script asks a held-out question
with a known answer: given the ten CIFAR-10 classes, how often is the top-ranked
phrase the true class? That is a real accuracy number, not an impression.

    python brain-lab/vision_accuracy.py --limit 2000

Two deliberate choices, both learned the hard way on this machine:

* The data is read from the Hugging Face parquet mirror rather than the canonical
  ``cs.toronto.edu`` tarball, which redirects and then serves at a few hundred
  bytes per second here. The mirror is the same 10,000-image test split.
* Nothing here imports torchvision. The wheel that resolves on this machine does
  not register its operators against the installed torch, and a benchmark must not
  depend on a package that cannot import.

Chance accuracy is 10%, so a result at or below that means the pipeline is broken
rather than the model being weak. Images are 32x32 and are upscaled by the model's
preprocessing, so they are blurrier than a camera photo; that costs accuracy and is
the honest price of a benchmark that runs on a laptop.
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from claireai import vision  # noqa: E402

PARQUET_URL = (
    "https://huggingface.co/datasets/uoft-cs/cifar10/resolve/main/plain_text/test-00000-of-00001.parquet"
)

# The published CIFAR-10 ordering. Row 0 of the test split is a cat with this list,
# which is a useful sanity check that the mirror really is CIFAR-10.
CLASSES = ("airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")

# Prompt ensembling: averaging several phrasings is what CLIP's published
# zero-shot numbers use, so measuring with one phrasing would understate it.
TEMPLATES = (
    "a photo of a {}.",
    "a picture of a {}.",
    "a blurry photo of a {}.",
    "a low resolution photo of a {}.",
)


def load_test_set(root: Path):
    """The 10,000-image test split, downloading it once (~24 MB)."""
    import pyarrow.parquet as pq

    target = root / "cifar10-test.parquet"
    if not target.is_file():
        root.mkdir(parents=True, exist_ok=True)
        print(f"downloading {PARQUET_URL} (about 24 MB, once)")
        urllib.request.urlretrieve(PARQUET_URL, target)  # noqa: S310 - fixed https URL
    return pq.read_table(target)


def image_at(table, index: int):
    """One benchmark image as a PIL image."""
    from PIL import Image

    payload = table.column("img")[index].as_py()
    data = payload["bytes"] if isinstance(payload, dict) else payload
    return Image.open(io.BytesIO(data)).convert("RGB")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=2000, help="test images to score")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent / "data" / "cifar10")
    parser.add_argument("--json-out", type=Path, help="write the full result here")
    args = parser.parse_args(argv)

    table = load_test_set(args.root)
    labels = [int(value) for value in table.column("label").to_pylist()]
    limit = min(args.limit, len(labels))
    print(f"scoring {limit} of {len(labels)} CIFAR-10 test images against {len(CLASSES)} classes")

    texts = [template.format(name) for name in CLASSES for template in TEMPLATES]
    try:
        class_features = vision.text_features(texts)
    except vision.VisionUnavailable as error:
        print(f"vision unavailable: {error}")
        return 2

    # One row per class: average the phrasings, then re-normalise.
    class_features = class_features.view(len(CLASSES), len(TEMPLATES), -1).mean(dim=1)
    class_features = class_features / class_features.norm(dim=-1, keepdim=True)

    import torch

    correct = 0
    hits = {name: 0 for name in CLASSES}
    totals = {name: 0 for name in CLASSES}
    started = time.time()

    for start in range(0, limit, args.batch_size):
        stop = min(start + args.batch_size, limit)
        images = [image_at(table, index) for index in range(start, stop)]
        with torch.inference_mode():
            features = vision.image_features(images)
            predictions = (100.0 * features @ class_features.T).argmax(dim=-1).tolist()
        for offset, predicted in enumerate(predictions):
            truth = labels[start + offset]
            name = CLASSES[truth]
            totals[name] += 1
            if predicted == truth:
                correct += 1
                hits[name] += 1
        if stop % (args.batch_size * 10) == 0 or stop == limit:
            elapsed = time.time() - started
            print(f"  {stop}/{limit}  accuracy so far {correct / stop * 100:.1f}%  ({elapsed:.0f}s)")

    elapsed = time.time() - started
    accuracy = correct / limit
    result = {
        "model": f"{vision.MODEL_NAME}/{vision.PRETRAINED}",
        "dataset": "CIFAR-10 test (10,000 images)",
        "images": limit,
        "chance_accuracy": round(1 / len(CLASSES), 4),
        "top1_accuracy": round(accuracy, 4),
        "seconds": round(elapsed, 1),
        "milliseconds_per_image": round(elapsed / limit * 1000, 1),
        "per_class": {
            name: {
                "accuracy": round(hits[name] / totals[name], 4) if totals[name] else None,
                "images": totals[name],
            }
            for name in CLASSES
        },
    }

    print()
    print(f"top-1 accuracy: {accuracy * 100:.2f}%  ({correct}/{limit})")
    print(f"chance:         {100 / len(CLASSES):.2f}%")
    print(f"speed:          {result['milliseconds_per_image']} ms/image on CPU")
    print("per class:")
    for name in CLASSES:
        entry = result["per_class"][name]
        if entry["accuracy"] is not None:
            print(f"  {name:<12} {entry['accuracy'] * 100:5.1f}%  ({entry['images']} images)")

    if args.json_out:
        args.json_out.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"wrote {args.json_out}")
    return 0


if __name__ == "__main__":  # pragma: no cover - script entry point
    raise SystemExit(main())
