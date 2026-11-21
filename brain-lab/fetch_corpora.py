#!/usr/bin/env python3
"""Download a bounded, licensed slice of public instruction data.

This is deliberately not a bulk scraper. It fetches a small sample from a
public dataset, normalises each item into the trainer's JSONL shape, and writes
a sidecar with the licence, URL, retrieval time and SHA-256. Review the licence
before increasing ``--limit``.

    assistant/.venv/bin/python brain-lab/fetch_corpora.py --source all --limit 1000

Sources:
  oasst1     OpenAssistant Conversations, Apache-2.0
  gsm8k      OpenAI grade-school math, MIT
  fineweb    FineWeb-Edu sample, ODC-By 1.0
  fineweb-fr French FineWeb-2 sample, ODC-By 1.0
  toolace    ToolACE bounded tool-call sample, Apache-2.0
  hermes     Hermes Function Calling v1 bounded sample, Apache-2.0
"""

from __future__ import annotations

import argparse
import ast
import gzip
import hashlib
import json
import os
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
ROWS_API = "https://datasets-server.huggingface.co/rows"
HERMES_DATASET = "NousResearch/hermes-function-calling-v1"
HERMES_CONFIG = "func_calling_singleturn"
HERMES_URL = "https://huggingface.co/datasets/NousResearch/hermes-function-calling-v1"
TOOLACE_DATASET = "Team-ACE/ToolACE"
TOOLACE_URL = "https://huggingface.co/datasets/Team-ACE/ToolACE"
TOOL_SOURCE_LIMIT = 500
CONTEXT_TOKENS = 512


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


def _find_tokenizer_manifest(explicit: Path | None = None) -> Path | None:
    if explicit is not None:
        return explicit.expanduser()
    candidates: list[Path] = []
    data_dir = os.environ.get("CLAIREAI_DATA_DIR")
    if data_dir:
        candidates.append(Path(data_dir).expanduser() / "models" / "claire-scratch.json")
    candidates.extend(
        [
            # Prefer the checkpoint the app actually runs; a stale development
            # copy can carry different tokenizer merges and a misleading budget.
            Path.home() / ".local" / "share" / "ClaireAI" / "models" / "claire-scratch.json",
            ROOT.parent / "runtime-data" / "models" / "claire-scratch.json",
        ]
    )
    return next((path for path in candidates if path.is_file()), None)


def _load_tokenizer(manifest: Path):
    try:
        payload = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise FetchError(f"could not read tokenizer manifest {manifest}: {exc}") from exc
    if not isinstance(payload, dict):
        raise FetchError(f"tokenizer manifest {manifest} is not an object")
    merges = payload.get("tokenizer_merges")
    if not merges:
        raise FetchError(f"tokenizer manifest {manifest} carries no tokenizer_merges")
    try:
        pairs = [tuple(int(value) for value in pair) for pair in merges]
    except (TypeError, ValueError) as exc:
        raise FetchError(f"tokenizer manifest {manifest} has malformed merges") from exc
    if any(len(pair) != 2 for pair in pairs):
        raise FetchError(f"tokenizer manifest {manifest} has malformed merges")
    try:
        from claire_brain.tokenizer import BytePairTokenizer
    except ImportError as exc:
        raise FetchError("claire_brain is not importable; install the brain-lab package") from exc
    return BytePairTokenizer(pairs)


def _tool_system(available_tools: list[str] | None) -> str:
    """Put the offered tool names in the prompt the trainer actually renders.

    Training records otherwise carry ``available_tools`` as metadata that the
    trainer never reads, so the model would learn a call format without ever
    seeing the tool list it receives at inference.
    """
    names = [str(name).strip() for name in available_tools or [] if str(name).strip()]
    if not names:
        return SYSTEM
    try:
        from claire_brain.prompt import tool_block
    except ImportError:
        return SYSTEM
    schemas = [
        {
            "type": "function",
            "function": {
                "name": name,
                "description": "Available tool",
                "parameters": {"type": "object", "properties": {}},
            },
        }
        for name in names
    ]
    block = tool_block(schemas)
    return f"{SYSTEM}\n{block}" if block else SYSTEM


def _fits_token_budget(
    tokenizer: Any, system: str, user: str, assistant: str, max_tokens: int
) -> bool:
    from claire_brain.prompt import render_prompt

    prompt = render_prompt(system, [{"role": "user", "content": user}])
    prompt_tokens = len(tokenizer.encode(prompt, append_eos=False))
    target_tokens = len(tokenizer.encode(assistant, append_eos=True))
    return prompt_tokens + target_tokens <= max_tokens


def _tool_record(
    user: str,
    assistant: str,
    *,
    source: str,
    license_name: str,
    url: str,
    available_tools: list[str],
    tokenizer: Any | None = None,
    max_tokens: int = CONTEXT_TOKENS,
) -> dict[str, Any] | None:
    """A tool-use record in the runtime's own wire format, bounded for 512 tokens."""
    user, assistant = _clean(user), _clean(assistant)
    if len(user) < 8 or len(assistant) < 8:
        return None
    system = _tool_system(available_tools)
    assistant = _clip(assistant, 900)
    if tokenizer is not None:
        if not _fits_token_budget(tokenizer, system, user, assistant, max_tokens):
            return None
    elif len(user) + len(assistant) + len(system) > 2048:
        # No tokenizer was available. Keep the conservative character bound so
        # a standalone fetch still cannot emit obviously oversized records.
        return None
    return {
        "system": system,
        "messages": [
            {"role": "user", "content": user[:1200]},
            {"role": "assistant", "content": assistant},
        ],
        "source": source,
        "license": license_name,
        "url": url,
        "available_tools": available_tools,
        "format": "runtime-tool-call-v1",
    }


def _json_list(text: Any) -> list[dict[str, Any]]:
    """Decode a JSON list embedded in a dataset field without guessing."""
    raw = str(text or "")
    start = raw.find("[")
    if start < 0:
        return []
    try:
        value, _ = json.JSONDecoder().raw_decode(raw[start:])
    except ValueError:
        return []
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def _source_tool_names(system: str) -> list[str]:
    marker = "Here is a list of functions in JSON format that you can invoke:"
    start = str(system or "").find(marker)
    if start < 0:
        return []
    names: list[str] = []
    for item in _json_list(str(system)[start + len(marker) :]):
        name = item.get("name")
        if not name and isinstance(item.get("function"), dict):
            name = item["function"].get("name")
        if isinstance(name, str) and name.strip():
            names.append(name.strip())
    return names


def _split_top_level(text: str) -> list[str]:
    """Split a call list on commas that are not inside quotes or parentheses."""
    parts: list[str] = []
    start = 0
    depth = 0
    quote: str | None = None
    escaped = False
    for index, char in enumerate(text):
        if quote is not None:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == quote:
                quote = None
            continue
        if char in "'\"":
            quote = char
        elif char in "([{":
            depth += 1
        elif char in ")]}":
            depth = max(0, depth - 1)
        elif char == "," and depth == 0:
            parts.append(text[start:index])
            start = index + 1
    parts.append(text[start:])
    return [part.strip() for part in parts if part.strip()]


def _parse_toolace_calls(value: str) -> list[dict[str, Any]] | None:
    """Parse ToolACE's ``[name(key=value), ...]`` call syntax.

    The source is not JSON, so this converts only calls whose names and argument
    literals are unambiguous. Anything malformed is dropped rather than taught.
    """
    text = str(value or "").strip()
    if not (text.startswith("[") and text.endswith("]")):
        return None
    inner = text[1:-1].strip()
    if not inner:
        return []
    calls: list[dict[str, Any]] = []
    for part in _split_top_level(inner):
        if "(" not in part or not part.endswith(")"):
            return None
        name, raw_arguments = part.split("(", 1)
        name = name.strip()
        if not name:
            return None
        arguments: dict[str, Any] = {}
        raw_arguments = raw_arguments[:-1].strip()
        if raw_arguments:
            for pair in _split_top_level(raw_arguments):
                if "=" not in pair:
                    return None
                key, raw_value = pair.split("=", 1)
                key = key.strip()
                if not key:
                    return None
                raw_value = raw_value.strip()
                try:
                    arguments[key] = ast.literal_eval(raw_value)
                except (SyntaxError, ValueError):
                    # Some source rows use bare ``true``/``false`` or an
                    # unquoted token. Keep the literal as text only when it is
                    # not a structural fragment.
                    if not raw_value or raw_value.startswith(("[", "{", "(")):
                        return None
                    arguments[key] = raw_value
        calls.append({"name": name, "arguments": arguments})
    return calls


def _parse_hermes_calls(value: str) -> list[dict[str, Any]] | None:
    # Take everything between the tags and let the JSON decoder handle nested
    # objects. A non-greedy ``\{.*?\}`` regex stops at the first closing brace
    # and silently drops valid calls whose arguments contain an object.
    blocks = re.findall(r"<tool_call>\s*(.*?)\s*</tool_call>", str(value or ""), re.DOTALL)
    if not blocks:
        return None
    calls: list[dict[str, Any]] = []
    for block in blocks:
        try:
            payload = json.loads(block)
        except ValueError:
            return None
        if not isinstance(payload, dict):
            return None
        name = payload.get("name")
        arguments = payload.get("arguments")
        if not isinstance(name, str) or not name.strip() or not isinstance(arguments, dict):
            return None
        calls.append({"name": name.strip(), "arguments": arguments})
    return calls


def _call_text(call: dict[str, Any]) -> str:
    return (
        "<tool_call>"
        + json.dumps(call, ensure_ascii=False, separators=(",", ":"))
        + "</tool_call>"
    )


def _held_out_prompts() -> set[str]:
    """Evaluation prompts that must never enter a training sample."""
    prompts: set[str] = set()
    for name in (
        "model_eval_cases.json",
        "conversation_cases.json",
        "tool_call_cases.json",
        "tool_call_cases_extended.json",
    ):
        path = ROOT.parent / "scripts" / name
        if not path.is_file():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except ValueError:
            continue
        if not isinstance(payload, dict) or not isinstance(payload.get("cases"), list):
            continue
        for case in payload["cases"]:
            if not isinstance(case, dict):
                continue
            for key in ("prompt", "user"):
                if case.get(key):
                    prompts.add(str(case[key]).strip())
            turns = case.get("turns")
            if not isinstance(turns, list):
                continue
            for turn in turns:
                if isinstance(turn, dict) and turn.get("user"):
                    prompts.add(str(turn["user"]).strip())
    return prompts


def _rows(dataset: str, config: str, *, offset: int, length: int = 100) -> list[dict[str, Any]]:
    query = urllib.parse.urlencode(
        {"dataset": dataset, "config": config, "split": "train", "offset": offset, "length": length}
    )
    payload = _get_json(f"{ROWS_API}?{query}")
    rows = payload.get("rows") or []
    return [item.get("row") or {} for item in rows if isinstance(item, dict)]


def fetch_toolace(
    limit: int,
    *,
    tokenizer: Any | None = None,
    max_tokens: int = CONTEXT_TOKENS,
) -> list[dict[str, Any]]:
    """Bounded ToolACE sample, normalised into the runtime call format."""
    limit = max(1, min(int(limit), TOOL_SOURCE_LIMIT))
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    held_out = _held_out_prompts()
    offset = 0
    while len(records) < limit:
        batch = _rows(TOOLACE_DATASET, "default", offset=offset)
        if not batch:
            break
        for row in batch:
            conversations = row.get("conversations") or []
            user = next(
                (str(item.get("value") or "") for item in conversations if item.get("from") == "user"),
                "",
            )
            assistant = next(
                (
                    str(item.get("value") or "")
                    for item in conversations
                    if item.get("from") == "assistant"
                ),
                "",
            )
            if not user or not assistant or user.strip() in held_out:
                continue
            available = _source_tool_names(str(row.get("system") or ""))
            calls = _parse_toolace_calls(assistant)
            if calls is not None:
                if not 1 <= len(calls) <= 2:
                    continue
                if not available or any(call["name"] not in available for call in calls):
                    continue
                target = "".join(_call_text(call) for call in calls)
            else:
                if assistant.lstrip().startswith("[") or "<tool_call>" in assistant:
                    continue
                target = assistant
            record = _tool_record(
                user,
                target,
                source="toolace",
                license_name="Apache-2.0",
                url=TOOLACE_URL,
                available_tools=available,
                tokenizer=tokenizer,
                max_tokens=max_tokens,
            )
            if record is None:
                continue
            digest = hashlib.sha256(
                (record["messages"][0]["content"] + "\x00" + target).encode("utf-8")
            ).hexdigest()
            if digest in seen:
                continue
            seen.add(digest)
            records.append(record)
            if len(records) >= limit:
                break
        offset += len(batch)
        if len(batch) < 100:
            break
    return records


def fetch_hermes(
    limit: int,
    *,
    tokenizer: Any | None = None,
    max_tokens: int = CONTEXT_TOKENS,
) -> list[dict[str, Any]]:
    """Bounded Hermes Function Calling v1 sample (single-turn config)."""
    limit = max(1, min(int(limit), TOOL_SOURCE_LIMIT))
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    held_out = _held_out_prompts()
    offset = 0
    while len(records) < limit:
        batch = _rows(HERMES_DATASET, HERMES_CONFIG, offset=offset)
        if not batch:
            break
        for row in batch:
            conversations = row.get("conversations") or []
            user = next(
                (
                    str(item.get("value") or "")
                    for item in conversations
                    if item.get("from") in {"human", "user"}
                ),
                "",
            )
            assistant = next(
                (
                    str(item.get("value") or "")
                    for item in conversations
                    if item.get("from") in {"gpt", "assistant"}
                ),
                "",
            )
            if not user or not assistant or user.strip() in held_out:
                continue
            available = [
                str(name)
                for item in _json_list(row.get("tools"))
                for name in [
                    item.get("function", {}).get("name")
                    if isinstance(item.get("function"), dict)
                    else item.get("name")
                ]
                if isinstance(name, str) and name.strip()
            ]
            calls = _parse_hermes_calls(assistant)
            if calls is not None:
                if not 1 <= len(calls) <= 2:
                    continue
                if not available or any(call["name"] not in available for call in calls):
                    continue
                target = "".join(_call_text(call) for call in calls)
            else:
                if "<tool_call>" in assistant:
                    continue
                target = assistant
            record = _tool_record(
                user,
                target,
                source="hermes-function-calling-v1",
                license_name="Apache-2.0",
                url=HERMES_URL,
                available_tools=available,
                tokenizer=tokenizer,
                max_tokens=max_tokens,
            )
            if record is None:
                continue
            digest = hashlib.sha256(
                (record["messages"][0]["content"] + "\x00" + target).encode("utf-8")
            ).hexdigest()
            if digest in seen:
                continue
            seen.add(digest)
            records.append(record)
            if len(records) >= limit:
                break
        offset += len(batch)
        if len(batch) < 100:
            break
    return records


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


def _write(
    records: Iterable[dict[str, Any]],
    output: Path,
    *,
    source: str,
    license_name: str,
    url: str,
    notes: str | None = None,
) -> dict[str, Any]:
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
        "notes": notes
        or "Bounded sample for a local from-scratch model; review the dataset card before increasing limits.",
    }
    output.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        choices=(
            "oasst1",
            "gsm8k",
            "fineweb",
            "fineweb-fr",
            "toolace",
            "hermes",
            "all",
        ),
        default="all",
    )
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument(
        "--tokenizer-manifest",
        type=Path,
        default=None,
        help=(
            "checkpoint sidecar whose tokenizer_merges enforce the 512-token tool-record "
            "budget (default: discover the installed claire-scratch.json)"
        ),
    )
    args = parser.parse_args()
    if args.limit < 1:
        parser.error("--limit must be positive")

    tokenizer = None
    tokenizer_manifest = _find_tokenizer_manifest(args.tokenizer_manifest)
    if tokenizer_manifest is not None:
        try:
            tokenizer = _load_tokenizer(tokenizer_manifest)
        except FetchError as exc:
            if args.tokenizer_manifest is not None:
                parser.error(str(exc))
            print(f"WARNING: {exc}; tool records will use the character bound")
    else:
        print("WARNING: no tokenizer manifest found; tool records will use the character bound")

    sources = (
        ("oasst1", "gsm8k", "fineweb", "fineweb-fr", "toolace", "hermes")
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
            elif source == "toolace":
                records = fetch_toolace(args.limit, tokenizer=tokenizer)
                manifest = _write(
                    records,
                    args.output / "toolace.jsonl",
                    source="toolace",
                    license_name="Apache-2.0",
                    url=TOOLACE_URL,
                    notes=(
                        "Bounded one/two-call ToolACE sample in runtime-tool-call-v1 format; "
                        "tool names are source metadata and must be mapped to registry tools before training."
                    ),
                )
                print(
                    f"toolace: {manifest['records']} records -> "
                    f"{args.output / 'toolace.jsonl'}"
                )
            elif source == "hermes":
                records = fetch_hermes(args.limit, tokenizer=tokenizer)
                manifest = _write(
                    records,
                    args.output / "hermes-function-calling.jsonl",
                    source="hermes-function-calling-v1",
                    license_name="Apache-2.0",
                    url=HERMES_URL,
                    notes=(
                        "Bounded one/two-call Hermes sample in runtime-tool-call-v1 format; "
                        "tool names are source metadata and must be mapped to registry tools before training."
                    ),
                )
                print(
                    f"hermes-function-calling-v1: {manifest['records']} records -> "
                    f"{args.output / 'hermes-function-calling.jsonl'}"
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
