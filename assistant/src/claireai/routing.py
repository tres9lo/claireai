"""Narrow the tool set per request — deterministically, with no model call.

Why this exists: on a CPU-only machine, prompt processing runs at roughly 31
tokens/s and *every* step re-sends every tool schema. Sending eleven schemas is
therefore the single largest avoidable cost per step (measured: 34.5 s of a
37.7 s model call). Showing the model three relevant tools instead of eleven
removes most of that.

The rule that makes this safe: **routing only changes what the model sees, never
what is allowed.** A call is still validated, risk-checked, sandboxed and (where
required) approved against the full registry. If a filtered-out tool is called
anyway, it runs normally. So a wrong route can cost accuracy, never permission.

A model-based classifier was rejected for now: an extra model call costs a
prompt evaluation of its own on this hardware, and it fails exactly when the
model is weak. Keyword routing is free, instant, and unit-testable. When the
schemas stop being the bottleneck, replace it — the interface is one function.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Pattern, Sequence

from claireai.tools.base import ToolRegistry

# An intent is a name plus two regex lists that must BOTH match: a verb phrase
# and an object noun. Requiring the pair is what stops "what did I write in my
# journal" from being routed to the write tools.
_VERB = 1
_OBJECT = 2

# A path-ish noun, including a bare extension ("rename report.pdf").
_FILE_NOUN = r"\b(file|files|folder|folders|document|documents|name|pdf|txt|doc)\b|\.\w{1,5}\b"

INTENTS: tuple[tuple[str, str, str, tuple[str, ...], str], ...] = (
    # Invariant, enforced by a test: **an intent that can write must also offer
    # a read tool.** A write needs a fact only a read can establish — does the
    # source exist, is there already a file there, what is really in the folder
    # — and without one the model has to invent a path. Measured: with `move`
    # offering only write tools, a small model answered with
    # `/Users/yourname/quarterly-report.pdf` and the sandbox refused it, twice.
    # Reads come after the tool that owns the intent, because order is a prior.
    (
        "trash",
        r"\b(delete|trash|discard|remove|get rid of|throw away|bin)\b",
        r"\b(file|files|folder|folders|item|items|everything|document|documents|it)\b",
        ("move_to_trash", "get_file_info"),
        "asking to remove something; the only removal that exists is the recoverable trash",
    ),
    (
        "rename",
        r"\b(rename|renaming|call it|change the name)\b",
        _FILE_NOUN + r"|\bit\b",
        ("rename_file", "get_file_info"),
        "asking to rename something",
    ),
    (
        "create_folder",
        r"\b(create|make|new|add)\b",
        r"\b(folder|folders|directory|directories)\b",
        ("create_folder", "list_folder"),
        "asking for a new folder, which needs the parent seen before it is made",
    ),
    (
        "organize",
        r"\b(organi[sz]e|tidy|sort|group|clean up|arrange)\b",
        r"\b(folder|folders|downloads|files|file|type|types|kind|extension)\b",
        ("move_file", "list_folder", "create_folder"),
        "asking to rearrange files, which needs listing, creating and moving",
    ),
    (
        "move",
        r"\b(move|put|transfer|shift|place)\b",
        r"\b(file|files|into|in my|to my|folder|document|documents|downloads)\b",
        ("move_file", "list_folder", "get_file_info", "create_folder"),
        "asking to move a file or folder, so both ends must be discoverable",
    ),
    (
        "write",
        r"\b(write|save|append|add|create|make|put)\b",
        r"\b(file|files|note|notes|text|txt|\.md|\.txt|saying|content)\b",
        ("create_text_file", "write_to_file", "get_file_info", "list_folder"),
        "asking to write text into a file, which needs the existing file checked first",
    ),
    (
        "largest",
        r"\b(biggest|largest|heaviest|taking up|takes up|take up|space|size|sizes)\b",
        r"\b(file|files|folder|downloads|disk|what|which)\b",
        ("find_largest_files",),
        "asking about size, which only a recursive scan can answer",
    ),
    (
        "search",
        r"\b(find|search|locate|look for|where is|where are|which files|every|all)\b",
        r"\b(file|files|pdf|doc|docs|document|documents|named|called|matching|pattern)\b",
        ("search_files",),
        "asking to locate files by name",
    ),
    (
        "browse",
        r"\b(open|visit|go to|browse|fetch|look up|google|search the web|on the web|download)\b",
        r"(\b(url|website|site|page|link|web|internet|online)\b|https?://)",
        ("open_url", "get_page_text", "search_web", "download_file"),
        "asking to use the web, which needs its own approval",
    ),
    (
        "see",
        r"\b(see|look|identify|recogni[sz]e|describe|what is|what's|whats|show me)\b",
        r"\b(image|images|picture|pictures|photo|photos|screenshot|screenshots|screen|display|monitor)\b",
        ("identify_image", "see_screen"),
        "asking what an image or the screen shows",
    ),
    (
        "read",
        r"\b(read|open|summar(y|ise|ize)|what does|contents|show me the text)\b",
        r"\b(file|files|text|content|contents|note|notes|document|it)\b",
        ("read_text_file", "get_file_info"),
        "asking to read a file's contents",
    ),
    (
        "info",
        r"\b(how big|how large|modified|when was|details|info|information|size of)\b",
        r"\b(file|files|folder|path|it|this)\b",
        ("get_file_info", "find_largest_files"),
        "asking about one path",
    ),
    (
        "list",
        r"\b(list|show|what is in|what's in|whats in|inside|contents|browse)\b",
        r"\b(folder|folders|downloads|documents|directory|desktop|inside|in my)\b",
        ("list_folder",),
        "asking to see what a folder holds",
    ),
)

_COMPILED: tuple[tuple[str, Pattern[str], Pattern[str], tuple[str, ...], str], ...] = tuple(
    (
        name,
        re.compile(verb, re.IGNORECASE),
        re.compile(obj, re.IGNORECASE),
        tools,
        reason,
    )
    for name, verb, obj, tools, reason in INTENTS
)


@dataclass(frozen=True)
class Route:
    """Which tools the model is shown for one request."""

    intent: str
    tools: tuple[str, ...]
    reason: str
    narrowed: bool

    def schemas(self, registry: ToolRegistry) -> list[dict]:
        """Wire schemas for exactly ``self.tools``, primary tool first.

        Order is a real, if soft, prior for a small model: the tool that owns the
        intent leads, supporting tools follow. Measured — with the schemas in
        plain registry order, `list_folder` was chosen for a "biggest files"
        request even though `find_largest_files` was offered.
        """
        return [registry.get(name).schema() for name in self.tools]


def _normalise(request: str) -> str:
    return " ".join(request.split())


def route(request: str, registry: ToolRegistry) -> Route:
    """Pick the tools for ``request``; never returns an empty or foreign set.

    Unrecognised requests get the whole registry: narrowing is an optimisation,
    so when there is no evidence it must not hide anything the owner needs.
    """
    text = _normalise(request)
    for name, verb, obj, tools, reason in _COMPILED:
        if not verb.search(text) or not obj.search(text):
            continue
        chosen = tuple(tool for tool in tools if tool in registry)
        if chosen:
            return Route(intent=name, tools=chosen, reason=reason, narrowed=len(chosen) < len(registry))

    return Route(
        intent="unrouted",
        tools=registry.names(),
        reason="no intent matched, so every tool is offered",
        narrowed=False,
    )


def describe(route: Route, registry: ToolRegistry) -> str:
    """One line for the CLI header and for measurement output."""
    shown = len(route.tools)
    return f"{route.intent} ({shown}/{len(registry)} tools)"


def schema_bytes(tools: Sequence[dict]) -> int:
    """Rough payload size, used to report the saving rather than claim it."""
    import json

    return len(json.dumps(list(tools)))


def reachable_tools() -> set[str]:
    """Every tool name any intent can offer — used to prove nothing is orphaned."""
    return {tool for _, _, _, tools, _ in INTENTS for tool in tools}


__all__: Iterable[str] = [
    "INTENTS",
    "Route",
    "describe",
    "reachable_tools",
    "route",
    "schema_bytes",
]
