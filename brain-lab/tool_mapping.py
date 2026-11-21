"""Deterministic mapping from public tool names to ClaireAI's registry.

The ToolACE and Hermes samples contain thousands of provider-specific APIs.
Only a call that resolves to a real registry tool with valid arguments may be
kept; everything else is dropped. There is no fuzzy matching, because a close
name with different semantics is worse than no training row.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from claireai.tools import ToolError, default_registry


@dataclass(frozen=True)
class Alias:
    """One explicit source name, with only unambiguous argument changes."""

    target: str
    rename: Mapping[str, str] = field(default_factory=dict)
    pattern_from: str | None = None


# Keys are normalised source names. Add an alias only when the source tool does
# the same thing as the local tool and its argument names can be translated
# without guessing. A name that merely sounds similar is not enough.
ALIASES: Mapping[str, Alias] = {
    "create_directory": Alias("create_folder"),
    "list_directory": Alias("list_folder"),
    "list_directory_contents": Alias("list_folder", {"all_files": "include_hidden"}),
    "list_files": Alias("list_folder"),
    "read_file": Alias("read_text_file"),
    "read_text": Alias("read_text_file"),
    "write_file": Alias("write_to_file"),
    "write_text_file": Alias("write_to_file"),
    "create_file": Alias("create_text_file"),
    "delete_file": Alias("move_to_trash"),
    "move_folder": Alias(
        "move_file",
        {"source_path": "source", "destination_path": "destination"},
    ),
    "move_files_to_new_folder": Alias(
        "move_file",
        {"source_folder_path": "source", "destination_folder_path": "destination"},
    ),
    "search_files_by_extension": Alias(
        "search_files",
        {"directory": "path"},
        pattern_from="extension",
    ),
    "get_file_metadata": Alias("get_file_info"),
    "file_info": Alias("get_file_info"),
    "open_web_page": Alias("open_url"),
    "web_search": Alias("search_web"),
    "take_screenshot": Alias("see_screen"),
    "capture_screen": Alias("see_screen"),
}

_REGISTRY = default_registry(browser_enabled=True)


def _normalise(name: Any) -> str:
    """Make ``GetFileInfo``, ``get-file-info`` and ``get_file_info`` agree."""
    text = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", str(name or ""))
    return re.sub(r"[^a-z0-9]+", "_", text.casefold()).strip("_")


_BY_NORMALISED = {_normalise(name): name for name in _REGISTRY.names()}


def map_tool_name(name: Any) -> str | None:
    """Return one exact registry name, or ``None`` when no safe match exists."""
    normalised = _normalise(name)
    if normalised in _BY_NORMALISED:
        return _BY_NORMALISED[normalised]
    alias = ALIASES.get(normalised)
    if alias is not None and alias.target in _REGISTRY:
        return alias.target
    return None


def map_available(names: Sequence[Any]) -> list[str]:
    """Resolve the offered tool list, preserving order and dropping unsupported APIs."""
    resolved: list[str] = []
    for name in names:
        target = map_tool_name(name)
        if target is not None and target not in resolved:
            resolved.append(target)
    return resolved


def map_call(call: Mapping[str, Any]) -> dict[str, Any] | None:
    """Resolve one call and validate its arguments against the local tool schema."""
    name = call.get("name")
    arguments = call.get("arguments")
    target = map_tool_name(name)
    if target is None or not isinstance(arguments, Mapping):
        return None

    alias = ALIASES.get(_normalise(name))
    transformed = {
        (alias.rename.get(str(key), str(key)) if alias is not None else str(key)): value
        for key, value in arguments.items()
    }
    if alias is not None and alias.pattern_from is not None:
        raw_extension = str(transformed.pop(alias.pattern_from, "")).strip().lstrip(".")
        if raw_extension:
            transformed["pattern"] = f"*.{raw_extension}"

    cleaned = {key: value for key, value in transformed.items() if value is not None}
    try:
        _REGISTRY.get(target).validate(cleaned)
    except ToolError:
        return None
    return {"name": target, "arguments": cleaned}
