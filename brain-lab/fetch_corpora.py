#!/usr/bin/env python3
"""Download a bounded, licensed slice of public instruction data.

This is deliberately not a bulk scraper. It fetches a small sample from two
public datasets, normalises each item into the trainer's JSONL shape, and writes
a sidecar with the licence, URL, retrieval time and SHA-256. Review the licence
before increasing ``--limit``.

    assistant/.venv/bin/python brain-lab/fetch_corpora.py --source all --limit 1000

Sources:
  oasst1     OpenAssistant Conversations, Apache-2.0
  gsm8k      OpenAI grade-school math, MIT
  fineweb    FineWeb-Edu sample, ODC-By 1.0
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import re
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "data" / "external"
USER_AGENT = "ClaireAI-corpus-fetcher/0.1 (personal research; contact local owner)"
SYSTEM = "You are ClaireAI, a helpful local assistant. Be concise, correct and honest; never invent a fact or an action."

OASST_READY_URL = (
    "https://huggingface.co/datasets/OpenAssistant/oasst1/resolve/main/"
    "2023-04-12_oasst_ready.messages.jsonl.gz"
)
OASST_LANGS = {"en": "English", "fr": "French"}
FINEWEB_API = "https://datasets-server.huggingface.co/rows"
FINEWEB_DATASET = "HuggingFaceFW/fineweb-edu"
FINEWEB_FR_DATASET = "HuggingFaceFW/fineweb-2"
FINEWEB_FR_CONFIG = "fra_Latn"
GSM8K_URL = "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/train.jsonl"


class FetchError(Exception):
    pass


def _get(url: str, *, timeout: int = 60) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.read()
    except OSError as exc:
        raise FetchError(f"could not fetch {url}: {exc}") from exc


def _get_json(url: str) -> dict[str, Any]:
    last: FetchError | None = None
    for attempt in range(4):
        try:
            payload = json.loads(_get(url).decode("utf-8"))
            if not isinstance(payload, dict):
                raise FetchError(f"{url} returned an unexpected JSON shape")
            return payload
        except (UnicodeDecodeError, ValueError, FetchError) as exc:
            last = exc if isinstance(exc, FetchError) else FetchError(f"{url} did not return JSON")
            if attempt < 3:
                time.sleep(1.5 * (attempt + 1))
    assert last is not None
    raise last


def _clean(text: Any) -> str:
    return " ".join(str(text or "").replace("\x00", " ").split())


def _clip(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    window = text[:limit]
    for stop in (". ", "! ", "? ", "\n"):
        index = window.rfind(stop)
        if index > limit // 2:
            return window[: index + 1].strip()
    return window.rsplit(" ", 1)[0].strip()


def _record(user: str, assistant: str, *, source: str, license_name: str, url: str) -> dict[str, Any] | None:
    user, assistant = _clean(user), _clean(assistant)
    if len(user) < 8 or len(assistant) < 8:
        return None
    return {
        "system": SYSTEM,
        "messages": [
            {"role": "user", "content": user[:1200]},
            {"role": "assistant", "content": _clip(assistant, 700)},
        ],
        "source": source,
        "license": license_name,
        "url": url,
    }


def fetch_oasst(limit: int) -> dict[str, list[dict[str, Any]]]:
    """Download reviewed OASST1 prompt/reply pairs in English and French.

    The public rows endpoint rate-limits and returns a different field name for
    the tree id than the released file.  The ready-message archive is a single
    34 MB gzip, so fetching it once is both faster and reproducible; the SHA-256
    written beside each split is the actual bytes used.
    """
    raw = _get(OASST_READY_URL, timeout=180)
    try:
        text = gzip.decompress(raw).decode("utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise FetchError(f"OASST1 archive was not valid gzip text: {exc}") from exc

    rows: dict[str, dict[str, Any]] = {}
    for line in text.splitlines():
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except ValueError:
            continue
        if isinstance(item, dict) and item.get("message_id"):
            rows[str(item["message_id"])] = item

    records: dict[str, list[dict[str, Any]]] = {lang: [] for lang in OASST_LANGS}
    seen: dict[str, set[str]] = {lang: set() for lang in OASST_LANGS}
    for item in rows.values():
        lang = str(item.get("lang") or "")
        if lang not in OASST_LANGS or item.get("role") != "assistant":
            continue
        if len(records[lang]) >= limit:
            continue
        if item.get("deleted") or item.get("synthetic") or item.get("review_count", 0) < 1:
            continue
        if item.get("review_result") is False:
            continue
        # Walk up to the nearest human prompt in the same language.  A reply to
        # another assistant turn is skipped rather than paired with the wrong
        # question, which is how synthetic follow-ups ended up in chat data.
        parent = rows.get(str(item.get("parent_id") or ""))
        while parent is not None and parent.get("role") != "prompter":
            parent = rows.get(str(parent.get("parent_id") or ""))
        if parent is None or str(parent.get("lang") or "") != lang:
            continue
        prompt = _clean(parent.get("text"))
        answer = _clean(item.get("text"))
        digest = hashlib.sha256((prompt + "\x00" + answer).encode("utf-8")).hexdigest()
        if digest in seen[lang]:
            continue
        record = _record(
            prompt,
            answer,
            source=f"oasst1-{lang}",
            license_name="Apache-2.0",
            url="https://huggingface.co/datasets/OpenAssistant/oasst1",
        )
        if record:
            record["lang"] = lang
            seen[lang].add(digest)
            records[lang].append(record)
            if len(records[lang]) >= limit:
                # Keep scanning only if the other language still needs rows.
                if all(len(records[code]) >= limit for code in OASST_LANGS):
                    break
    return records


def fetch_fineweb(limit: int) -> list[dict[str, Any]]:
    """Bounded reading-comprehension sample from FineWeb-Edu.

    The answer is an extractive two-sentence summary, so the target is checked
    against the passage rather than invented by a model.
    """
    records: list[dict[str, Any]] = []
    page = 100
    offset = 0
    while len(records) < limit:
        query = urllib.parse.urlencode(
            {
                "dataset": FINEWEB_DATASET,
                "config": "default",
                "split": "train",
                "offset": offset,
                "length": page,
            }
        )
        payload = _get_json(f"{FINEWEB_API}?{query}")
        batch = payload.get("rows") or []
        if not batch:
            break
        for item in batch:
            row = item.get("row") or {}
            if str(row.get("language") or "en").lower() not in {"en", "eng"}:
                continue
            passage = _clean(row.get("text"))[:1600]
            if len(passage) < 240:
                continue
            summary = _clip(passage, 360)
            if len(summary.split()) < 12 or summary == passage:
                continue
            record = _record(
                "Read this passage and state its main point in two sentences:\n" + passage,
                summary,
                source="fineweb-edu",
                license_name="ODC-By 1.0",
                url=str(row.get("url") or "https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu"),
            )
            if record:
                records.append(record)
                if len(records) >= limit:
                    break
        offset += len(batch)
        if len(batch) < page:
            break
    return records


def fetch_fineweb_french(limit: int) -> list[dict[str, Any]]:
    """Bounded French reading sample from FineWeb-2 (ODC-By 1.0).

    The target is an extractive lead, not a model-written summary.  It teaches
    French orthography and sentence rhythm, not factual question answering.
    """
    records: list[dict[str, Any]] = []
    page = 100
    offset = 0
    while len(records) < limit:
        query = urllib.parse.urlencode(
            {
                "dataset": FINEWEB_FR_DATASET,
                "config": FINEWEB_FR_CONFIG,
                "split": "train",
                "offset": offset,
                "length": page,
            }
        )
        payload = _get_json(f"{FINEWEB_API}?{query}")
        batch = payload.get("rows") or []
        if not batch:
            break
        for item in batch:
            row = item.get("row") or {}
            if str(row.get("language") or "").lower() not in {"fra", "fr", "french"}:
                continue
            passage = _clean(row.get("text"))[:1600]
            if len(passage) < 240:
                continue
            summary = _clip(passage, 360)
            if len(summary.split()) < 12 or summary == passage:
                continue
            record = _record(
                "Lis ce passage et donne son idée principale en deux phrases :\n" + passage,
                summary,
                source="fineweb-2-fr",
                license_name="ODC-By 1.0",
                url=str(row.get("url") or "https://huggingface.co/datasets/HuggingFaceFW/fineweb-2"),
            )
            if record:
                records.append(record)
                if len(records) >= limit:
                    break
        offset += len(batch)
        if len(batch) < page:
            break
    return records


def fetch_gsm8k(limit: int) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line in _get(GSM8K_URL).decode("utf-8").splitlines():
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except ValueError:
            continue
        question = _clean(item.get("question"))
        answer = re.sub(r"<<[^>]*>>", "", str(item.get("answer") or ""))
        record = _record(
            question,
            answer,
            source="gsm8k",
            license_name="MIT",
            url="https://github.com/openai/grade-school-math",
        )
        if record:
            records.append(record)
            if len(records) >= limit:
                break
    return records


def _write(records: Iterable[dict[str, Any]], output: Path, *, source: str, license_name: str, url: str) -> dict[str, Any]:
    material = list(records)
    if not material:
        raise FetchError(f"{source} returned no usable records; refusing to write an empty corpus")
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in material)
    output.write_text(payload, encoding="utf-8")
    manifest = {
        "source": source,
        "license": license_name,
        "url": url,
        "retrieved": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "records": len(material),
        "sha256": hashlib.sha256(payload.encode("utf-8")).hexdigest(),
        "notes": "Bounded sample for a local from-scratch model; review the dataset card before increasing limits.",
    }
    output.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        choices=("oasst1", "gsm8k", "fineweb", "fineweb-fr", "all"),
        default="all",
    )
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    if args.limit < 1:
        parser.error("--limit must be positive")

    sources = (
        ("oasst1", "gsm8k", "fineweb", "fineweb-fr")
        if args.source == "all"
        else (args.source,)
    )
    failed = False
    for source in sources:
        try:
            if source == "oasst1":
                by_lang = fetch_oasst(args.limit)
                for lang, records in by_lang.items():
                    manifest = _write(
                        records,
                        args.output / f"oasst1-{lang}.jsonl",
                        source=f"oasst1-{lang}",
                        license_name="Apache-2.0",
                        url="https://huggingface.co/datasets/OpenAssistant/oasst1",
                    )
                    print(
                        f"oasst1-{lang}: {manifest['records']} records -> "
                        f"{args.output / f'oasst1-{lang}.jsonl'}"
                    )
            elif source == "gsm8k":
                records = fetch_gsm8k(args.limit)
                manifest = _write(
                    records,
                    args.output / "gsm8k.jsonl",
                    source="gsm8k",
                    license_name="MIT",
                    url="https://github.com/openai/grade-school-math",
                )
                print(f"gsm8k: {manifest['records']} records -> {args.output / 'gsm8k.jsonl'}")
            elif source == "fineweb-fr":
                records = fetch_fineweb_french(args.limit)
                manifest = _write(
                    records,
                    args.output / "fineweb-2-fr.jsonl",
                    source="fineweb-2-fr",
                    license_name="ODC-By 1.0",
                    url="https://huggingface.co/datasets/HuggingFaceFW/fineweb-2",
                )
                print(
                    f"fineweb-2-fr: {manifest['records']} records -> "
                    f"{args.output / 'fineweb-2-fr.jsonl'}"
                )
            else:
                records = fetch_fineweb(args.limit)
                manifest = _write(
                    records,
                    args.output / "fineweb-edu.jsonl",
                    source="fineweb-edu",
                    license_name="ODC-By 1.0",
                    url="https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu",
                )
                print(
                    f"fineweb-edu: {manifest['records']} records -> "
                    f"{args.output / 'fineweb-edu.jsonl'}"
                )
        except FetchError as exc:
            failed = True
            print(f"ERROR: {source}: {exc}")
            continue
    return 2 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
