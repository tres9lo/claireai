"""Fetch the synthetic-only GUI-Primitives subset with attribution and hashes.

Real application screenshots are intentionally excluded. This script pins a
reviewed dataset revision and refuses paths or responses outside size limits.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path, PurePosixPath
from typing import Any

REPOSITORY = "kagnlp/gui-primitives"
REVISION = "a940226b327770dee78e821ff3fc3aa575a73be1"
BASE = f"https://huggingface.co/datasets/{REPOSITORY}/resolve/{REVISION}/"
ATTRIBUTION = (
    "GUI-Primitives by Md Abrar Jahin and Md Rizwan Parvez; synthetic annotations and images, CC BY 4.0. "
    "https://huggingface.co/datasets/kagnlp/gui-primitives"
)
MAX_METADATA_BYTES = 8_000_000
MAX_IMAGE_BYTES = 8_000_000
MAX_TOTAL_BYTES = 128_000_000
PINNED_METADATA_SHA256 = {
    "README.md": "4bbaaa0b2fd994ef5e03ea956dbbeb97ef93a04ee2dd554bf8a320e7300d3ba3",
    "LICENSE": "5c3c127b6398fdeb38c71e86278d27a388f0ea97bbe90dbd04cd1d26358bd507",
    "data/items.jsonl": "85b7e5a43d04ec5ab3cff1456e49cc45e9c122cbed3ccc124c5adc5d9fcac4f1",
}


def _safe_relative(value: str) -> Path:
    candidate = PurePosixPath(value)
    if candidate.is_absolute() or not candidate.parts or any(part in {".", ".."} for part in candidate.parts):
        raise ValueError(f"unsafe dataset path: {value!r}")
    return Path(*candidate.parts)


def _download(relative: str, destination: Path, *, limit: int, expected_sha256: str | None = None) -> bytes:
    if destination.is_file():
        existing = destination.read_bytes()
        valid = len(existing) <= limit
        if relative.lower().endswith(".png"):
            valid = valid and existing.startswith(b"\x89PNG\r\n\x1a\n")
        if expected_sha256:
            valid = valid and hashlib.sha256(existing).hexdigest() == expected_sha256
        if valid:
            return existing
    rel = _safe_relative(relative)
    url = BASE + urllib.parse.quote(rel.as_posix(), safe="/")
    request = urllib.request.Request(url, headers={"User-Agent": "ClaireAI-dataset-prep/1.0"})
    try:
        with urllib.request.urlopen(request, timeout=45) as response:
            final_host = urllib.parse.urlparse(response.geturl()).hostname or ""
            if not (final_host == "huggingface.co" or final_host.endswith(".hf.co")):
                raise ValueError(f"unexpected dataset download host: {final_host}")
            declared = response.headers.get("Content-Length")
            if declared and int(declared) > limit:
                raise ValueError(f"{relative} exceeds the {limit}-byte download limit")
            content = response.read(limit + 1)
    except (OSError, urllib.error.URLError) as exc:
        raise RuntimeError(f"could not download {relative}: {exc}") from exc
    if len(content) > limit:
        raise ValueError(f"{relative} exceeds the {limit}-byte download limit")
    if expected_sha256 and hashlib.sha256(content).hexdigest() != expected_sha256:
        raise ValueError(f"download checksum mismatch for {relative}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".part")
    temporary.write_bytes(content)
    os.replace(temporary, destination)
    return content


def _split(image_path: str) -> str:
    bucket = int(hashlib.sha256(image_path.encode()).hexdigest()[:8], 16) % 5
    return "validation" if bucket == 0 else "train"


def fetch(output: Path, *, max_images: int | None = None) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    previous_manifest_path = output / "manifest.json"
    try:
        previous_manifest = json.loads(previous_manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        previous_manifest = {}
    previous_hashes = {
        item.get("path"): item.get("sha256")
        for item in previous_manifest.get("images", [])
        if isinstance(item, dict)
    }
    metadata_paths = ["README.md", "LICENSE", "data/items.jsonl"]
    metadata: dict[str, bytes] = {}
    metadata_hashes: dict[str, str] = {}
    total = 0
    for relative in metadata_paths:
        destination = output / "upstream" / _safe_relative(relative)
        content = _download(
            relative, destination, limit=MAX_METADATA_BYTES,
            expected_sha256=PINNED_METADATA_SHA256[relative],
        )
        total += len(content)
        if total > MAX_TOTAL_BYTES:
            raise ValueError("download exceeds total size limit")
        metadata[relative] = content
        metadata_hashes[relative] = hashlib.sha256(content).hexdigest()

    upstream_lines = metadata["data/items.jsonl"].decode("utf-8").splitlines()
    synthetic = [json.loads(line) for line in upstream_lines if line.strip() and json.loads(line).get("source") == "synthetic"]
    image_paths = sorted({_safe_relative(row["image_path"]).as_posix() for row in synthetic})
    if max_images is not None:
        if max_images < 1:
            raise ValueError("--max-images must be positive")
        image_paths = image_paths[:max_images]
        allowed_images = set(image_paths)
        synthetic = [row for row in synthetic if _safe_relative(row["image_path"]).as_posix() in allowed_images]

    image_records = []
    local_image_paths: dict[str, str] = {}
    def fetch_image(relative: str) -> tuple[str, dict[str, Any]]:
        destination = output / "images" / _safe_relative(relative).relative_to("images")
        local_name = str(destination.relative_to(output))
        content = _download(
            relative, destination, limit=MAX_IMAGE_BYTES,
            expected_sha256=previous_hashes.get(local_name),
        )
        if not content.startswith(b"\x89PNG\r\n\x1a\n"):
            raise ValueError(f"upstream file is not a PNG image: {relative}")
        return relative, {"path": str(destination.relative_to(output)), "sha256": hashlib.sha256(content).hexdigest(), "bytes": len(content)}

    with ThreadPoolExecutor(max_workers=8) as pool:
        fetched_images = list(pool.map(fetch_image, image_paths))
    for relative, item in fetched_images:
        total += item["bytes"]
        if total > MAX_TOTAL_BYTES:
            raise ValueError("download exceeds total size limit")
        image_records.append(item)
        local_image_paths[relative] = item["path"]

    held_out_images = {
        _safe_relative(row["image_path"]).as_posix()
        for row in synthetic if row.get("human_verified_clean") is True
    }
    normalized = []
    counts_by_source = {"pool": 0, "human_verified_clean": 0, "excluded_for_image_leakage": 0}
    for row in synthetic:
        image_path = _safe_relative(row["image_path"]).as_posix()
        if row.get("human_verified_clean") is True:
            split = "test"
            counts_by_source["human_verified_clean"] += 1
        elif row.get("split") == "pool" and image_path in held_out_images:
            counts_by_source["excluded_for_image_leakage"] += 1
            continue
        elif row.get("split") == "pool":
            split = _split(image_path)
            counts_by_source["pool"] += 1
        else:
            continue
        box = [float(value) for value in row["target_bbox"]]
        x = round((box[0] + box[2]) / 2)
        y = round((box[1] + box[3]) / 2)
        record = {
            "id": f"gui-primitives-{row['item_id']}",
            "split": split,
            "family": "external_synthetic_gui_grounding",
            "license": "CC-BY-4.0",
            "attribution": ATTRIBUTION,
            "upstream_revision": REVISION,
            "image": local_image_paths[image_path],
            "image_path_upstream": image_path,
            "instruction": row["instruction"],
            "image_size": row["image_size"],
            "target_bbox": box,
            "target_center": [x, y],
            "pair_id": row["pair_id"],
            "primitive": row["primitive"],
            "source": "GUI-Primitives synthetic subset",
            "pixel_training_ready": False,
        }
        normalized.append(record)

    examples_path = output / "examples.jsonl"
    examples_path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in normalized), encoding="utf-8")
    counts = {split: sum(1 for row in normalized if row["split"] == split) for split in ("train", "validation", "test")}
    manifest = {
        "dataset": "gui-primitives-synthetic-import",
        "upstream": f"https://huggingface.co/datasets/{REPOSITORY}",
        "upstream_revision": REVISION,
        "attribution": ATTRIBUTION,
        "license": "CC-BY-4.0 (synthetic examples and screenshots only)",
        "excluded_content": "Real UI-Vision screenshots were not downloaded.",
        "counts_by_split": counts,
        "source_counts": counts_by_source,
        "image_count": len(image_records),
        "images": image_records,
        "upstream_files_sha256": metadata_hashes,
        "examples_sha256": hashlib.sha256(examples_path.read_bytes()).hexdigest(),
        "total_download_bytes": total,
        "limitations": [
            "The upstream card describes this as a diagnostic benchmark; only unverified synthetic pool items are supplemental training data. The human-verified clean synthetic items are held out for evaluation.",
            "Pool items sharing an image with the clean holdout are excluded to prevent screenshot leakage.",
            "These image/annotation pairs are not consumed by the current text-only trainer; add a vision encoder before using them for pixel-level training.",
        ],
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "data" / "external" / "gui-primitives")
    parser.add_argument("--max-images", type=int, help="download a small prefix for a network check; default downloads all synthetic images")
    args = parser.parse_args()
    print(json.dumps(fetch(args.output, max_images=args.max_images), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
