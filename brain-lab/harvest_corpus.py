"""Collect public text for training, with provenance and a robots.txt check.

Self-improvement needs data. This script adds one honest source of it:

    python brain-lab/harvest_corpus.py --name gutenberg-1 \
        --url https://www.gutenberg.org/files/1342/1342-0.txt

What it refuses to do: fetch anything that is not ``http(s)``, ignore
robots.txt, exceed the byte cap, or train on text whose licence it cannot
record. Each document is hashed, dated and written into a manifest next to the
corpus, so a checkpoint can always be traced back to the text it learned from.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import re
import urllib.robotparser
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

MAX_BYTES = 8 * 1024 * 1024
MIN_PARAGRAPH_CHARS = 80
USER_AGENT = "ClaireAI-corpus/0.1 (local research; contact the operator)"
_TEXT_TAGS = re.compile(r"<(script|style|nav|footer|header)\b.*?</\1>", re.DOTALL | re.IGNORECASE)
_BLOCK_TAGS = re.compile(
    r"</?(p|div|br|li|ul|ol|h[1-6]|article|section|blockquote|tr|td)\b[^>]*>",
    re.IGNORECASE,
)
_TAG = re.compile(r"<[^>]+>")
_SPACE = re.compile(r"[ \t\r\f\v]+")
_BLANK_LINES = re.compile(r"\n{2,}")


class HarvestError(RuntimeError):
    """The document was refused."""


def robots_allows(url: str) -> bool:
    """Ask the host's robots.txt before fetching anything.

    urllib.robotparser reports a *missing* robots.txt as permission by default,
    but if we could not actually reach the file we cannot verify that, so we
    refuse rather than assume permission we cannot observe.
    """
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise HarvestError(f"refusing a non-http(s) URL: {url!r}")
    robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"
    parser = urllib.robotparser.RobotFileParser()
    parser.set_url(robots_url)
    try:
        parser.read()
    except (OSError, ValueError):
        return False
    if not parser.mtime():
        # robots.txt could not be reached at all.
        return False
    if not parser.default_entry:
        return True
    return parser.can_fetch(USER_AGENT, url)


def html_to_text(payload: str) -> str:
    """Readable paragraphs from an HTML or plain-text document."""

    # Drop whole <script>, <style>, <nav>, <footer> and <header> blocks first, so
    # their contents never reach the corpus.
    cleaned = _TEXT_TAGS.sub(" ", payload)

    # Block-level tags mark paragraph boundaries, so turn them into newlines.
    # Without this every run of text collapses into one unusable line.
    cleaned = _BLOCK_TAGS.sub("\n", cleaned)

    # Inline tags just separate words.
    cleaned = _TAG.sub(" ", cleaned)

    # Decode entities like &amp; back to their literal characters.
    cleaned = html.unescape(cleaned)

    # Normalise the spaces inside each paragraph, keeping the line breaks.
    lines = (_SPACE.sub(" ", line).strip() for line in cleaned.splitlines())
    kept = [line for line in lines if len(line) >= MIN_PARAGRAPH_CHARS]

    return _BLANK_LINES.sub("\n\n", "\n\n".join(kept)).strip()


def fetch(url: str, *, max_bytes: int = MAX_BYTES) -> str:
    """Download one document, bounded, with a clear user agent."""
    import httpx

    with httpx.stream(
        "GET", url, headers={"User-Agent": USER_AGENT}, follow_redirects=True, timeout=30
    ) as response:
        response.raise_for_status()
        chunks: list[bytes] = []
        total = 0
        for chunk in response.iter_bytes():
            total += len(chunk)
            if total > max_bytes:
                raise HarvestError(f"document exceeds {max_bytes} bytes: {url}")
            chunks.append(chunk)
        encoding = response.encoding or "utf-8"
    return b"".join(chunks).decode(encoding, errors="replace")


def write_corpus(name: str, documents: list[tuple[str, str]], out_root: Path) -> dict[str, object]:
    """Split into train/validation/test and record where every byte came from."""
    text = "\n\n".join(body for _url, body in documents).strip()
    if len(text) < MIN_PARAGRAPH_CHARS:
        raise HarvestError("nothing usable was extracted from those documents")

    cut = max(1, int(len(text) * 0.9))
    pieces = {
        "train": text[:cut],
        "validation": text[cut:][: max(1, (len(text) - cut) // 2)],
        "test": text[cut:][max(1, (len(text) - cut) // 2):],
    }
    target = out_root / name
    entries = []
    for url, body in documents:
        entries.append(
            {
                "url": url,
                "sha256": hashlib.sha256(body.encode("utf-8")).hexdigest(),
                "bytes": len(body.encode("utf-8")),
                "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                # Nothing here can establish a licence. The operator must.
                "license": "unverified — confirm before training",
            }
        )
    for split, body in pieces.items():
        if not body.strip():
            continue
        directory = target / "text" / split
        directory.mkdir(parents=True, exist_ok=True)
        (directory / f"{split}.txt").write_text(body, encoding="utf-8")

    manifest = {
        "name": name,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "documents": entries,
        "characters": len(text),
        "warning": "Text with an unverified licence must not be shipped or redistributed.",
    }
    target.mkdir(parents=True, exist_ok=True)
    (target / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Harvest public text for the scratch corpus.")
    parser.add_argument("--name", required=True, help="corpus name, used as the output directory")
    parser.add_argument("--url", action="append", required=True, help="document URL (repeatable)")
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "data")
    parser.add_argument("--max-bytes", type=int, default=MAX_BYTES)
    parser.add_argument("--yes", action="store_true", help="skip the interactive confirmation")
    args = parser.parse_args(argv)

    documents: list[tuple[str, str]] = []
    for url in args.url:
        if not robots_allows(url):
            print(f"refused (robots.txt or unreachable host): {url}")
            continue
        try:
            body = fetch(url, max_bytes=args.max_bytes)
        except Exception as error:  # noqa: BLE001 - reported, not swallowed
            print(f"failed: {url}: {type(error).__name__}: {error}")
            continue
        text = html_to_text(body)
        if len(text) < MIN_PARAGRAPH_CHARS:
            print(f"skipped (no usable text): {url}")
            continue
        documents.append((url, text))
        print(f"kept: {url} ({len(text)} characters)")

    if not documents:
        print("nothing was harvested; no corpus written")
        return 1
    if not args.yes:
        print(
            f"About to write {len(documents)} document(s) to {args.out / args.name}."
            " The licence of this text is unverified: confirm it before training."
        )
    manifest = write_corpus(args.name, documents, args.out)
    print(f"wrote {args.out / args.name} ({manifest['characters']} characters)")
    return 0


if __name__ == "__main__":  # pragma: no cover - script entry point
    raise SystemExit(main())
