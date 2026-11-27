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

# A path-ish noun, including a bare extension ("rename report.pdf") or an
# absolute path with no extension ("delete /tmp/scratch-dir"). Without the path
# form, a real trash request was unrouted and received all 25 schemas; one CPU
# call then took 112.8 s and hit the old 60 s timeout.
_FILE_NOUN = r"\b(file|files|folder|folders|document|documents|name|pdf|txt|doc)\b|\.\w{1,5}\b"
_PATH_NOUN = r"(?:^|\s)(?:~[/\\]|/|[A-Za-z]:[/\\])[^\s]+"

INTENTS: tuple[tuple[str, str, str, tuple[str, ...], str], ...] = (
    # Invariant, enforced by a test: **an intent that can write must also offer
    # a read tool.** A write needs a fact only a read can establish — does the
    # source exist, is there already a file there, what is really in the folder
    # — and without one the model has to invent a path. Measured: with `move`
    # offering only write tools, a small model answered with
    # `/Users/yourname/quarterly-report.pdf` and the sandbox refused it, twice.
    # Reads come after the tool that owns the intent, because order is a prior.
    (
        "conversation",
        r"^\s*(?:(?:hi|hello|hey)(?:[,.! ]+how are you)?|good (?:morning|afternoon|evening)|"
        r"how are you|who are you|what can you do|thanks|thank you|nice to meet you)"
        r"[\s,.!?]*$",
        r".*",
        ("search_memory",),
        "ordinary conversation; no tool schema is needed for a greeting",
    ),
    (
        "memory",
        r"\b(remember|memorize|memorise|learn|keep in mind|note that|save this)\b",
        r"\b(that|this|fact|facts|preference|preferences|note|notes|memory|remember|know)\b",
        ("remember_note", "search_memory"),
        "asking the assistant to keep or recall a durable local note",
    ),
    (
        "canvas_news",
        r"\b(canvas|visual|briefing|infographic|illustrate)\b",
        r"\b(news|headlines?|briefing|top stories|current events)\b",
        ("world_news", "create_canvas", "list_canvases"),
        "asking for a visual news briefing, which needs headlines before it is drawn",
    ),
    (
        "canvas",
        r"\b(canvas|diagram|infographic|flowchart|visuali[sz]e|illustrate|draw|sketch|chart)\b",
        r"\b(canvas|diagram|infographic|flowchart|visual|process|mechanism|system|how .* works|news|explainer)\b",
        ("create_canvas", "list_canvases"),
        "asking for a visual page; the list read lets the model reuse an earlier canvas",
    ),
    (
        "windows",
        r"\b(arrange|rearrange|tile|resize|move|maximize|minimize|snap|center|"
        r"side by side|layout)\b",
        r"\b(window|windows|desktop|screen|side by side|layout|tab|tabs|app|apps)\b",
        ("list_windows", "arrange_windows", "computer_control"),
        "asking to arrange or resize open application windows",
    ),
    (
        "apps",
        r"\b(open|launch|start|switch to|show)\b",
        r"\b(app|apps|application|applications|browser|firefox|chrome|chromium|edge|"
        r"libreoffice|spotify|calculator|terminal|notes|gedit|settings)\b",
        ("list_apps", "open_app", "open_in_browser", "computer_control"),
        "asking to open or operate a desktop app, not to fetch a page",
    ),
    (
        "calendar",
        r"\b(schedule|scheduled|appointment|appointments|meeting|meetings|agenda|calendar|what's on|what is on)\b",
        r"\b(today|tomorrow|yesterday|tonight|week|monday|tuesday|wednesday|thursday|friday|saturday|sunday|appointment|appointments|meeting|meetings|agenda|calendar)\b",
        ("list_appointments", "add_appointment"),
        "asking what is scheduled; the calendar answers from the local database",
    ),
    (
        "day_plan",
        r"\b(plan|plan out|assign|schedule|organi[sz]e|show|list|what(?:'s| is))\b",
        r"\b(day plan|day|today|tomorrow|morning|afternoon|evening|task|tasks|agenda)\b",
        ("plan_day", "list_day_plan", "list_appointments"),
        "assigning or reading tasks for a full local day",
    ),
    (
        "news",
        r"\b(news|headlines?|briefing|top stories|what(?:'s| is) happening)\b",
        r"\b(world|country|countries|from|in|about|latest)\b",
        ("world_news",),
        "asking for live headlines by country",
    ),
    # Before `read_contents`: "open https://example.com and tell me what it says"
    # is a web request, and the read rule would otherwise claim it.
    (
        "browse",
        r"\b(open|visit|go to|browse|fetch|look up|google|search the web|on the web|download|show me the text|page contents)\b",
        r"(\b(url|website|site|page|link|web|internet|online)\b|https?://)",
        ("open_url", "get_page_text", "search_web", "download_file"),
        "asking to use the web, which needs its own approval",
    ),
    # "show me what report.pdf says" is a read, but the words alone match the
    # list and vision intents first, which hid `read_text_file` from the model.
    (
        "read_contents",
        r"\b(show me what|what does|what do|tell me what)\b",
        r"\b(say|says|said|contain|contains|text|contents?)\b",
        ("read_text_file", "get_file_info"),
        "asking what a file says, which only a read can answer",
    ),
    (
        "trash",
        r"\b(delete|trash|discard|remove|get rid of|throw away|bin)\b",
        r"\b(file|files|folder|folders|item|items|everything|document|documents|it)\b|"
        + _FILE_NOUN + "|" + _PATH_NOUN,
        ("get_file_info", "move_to_trash"),
        "asking to remove something; read the exact path, then use recoverable trash",
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
        r"\b(file|files|into|in my|to my|folder|document|documents|downloads)\b|"
        + _FILE_NOUN + "|" + _PATH_NOUN,
        ("move_file", "list_folder", "get_file_info", "create_folder"),
        "asking to move a file or folder, so both ends must be discoverable",
    ),
    (
        "edit",
        r"\b(edit|modify|replace|change|update|insert|patch)\b",
        _FILE_NOUN,
        ("read_text_file", "replace_in_file", "get_file_info"),
        "asking for a surgical edit; read the current text before replacing it",
    ),
    (
        "write",
        r"\b(write|save|append|add|create|make|put|replace|overwrite|update)\b",
        r"\b(file|files|note|notes|text|txt|\.md|\.txt|saying|contents?)\b",
        ("create_text_file", "write_to_file", "replace_in_file", "get_file_info", "list_folder"),
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
        r"\b(file|files|pdf|doc|docs|document|documents|named|called|matching|pattern)\b|"
        + _PATH_NOUN,
        ("search_files",),
        "asking to locate files by name",
    ),
    # Before `see`, so "what is in Pictures" lists the folder instead of being
    # read as a request to identify an image.
    (
        "list",
        r"\b(list|show|what is in|what's in|whats in|inside|contents|browse)\b",
        r"\b(folder|folders|downloads|documents|directory|desktop|pictures|inside|in my)\b|"
        + _PATH_NOUN,
        ("list_folder",),
        "asking to see what a folder holds",
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
        r"\b(file|files|text|content|contents|note|notes|document|it)\b|"
        + _FILE_NOUN + "|" + _PATH_NOUN,
        ("read_text_file", "get_file_info"),
        "asking to read a file's contents",
    ),
    (
        "info",
        r"\b(how big|how large|modified|when was|details|info|information|size of)\b",
        r"\b(file|files|folder|path|it|this)\b|" + _FILE_NOUN + "|" + _PATH_NOUN,
        ("get_file_info", "find_largest_files"),
        "asking about one path",
    ),
    # After every action rule: a concept question gets one small schema instead
    # of the whole catalog. "How do I delete a file?" still matches trash first.
    (
        "explain",
        r"^\s*(what is|what are|who is|why is|why do|why does|how does|define|"
        r"meaning of|explain|tell me about)\b",
        r".*",
        ("search_memory",),
        "explaining a concept; uses a local memory lookup instead of every tool",
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
    """Pick the tools for ``request``; never returns a foreign set.

    Unrecognised requests get the whole registry: narrowing is an optimisation,
    so when there is no evidence it must not hide anything the owner needs. A
    narrow conversation/explanation intent may return no tools at all.
    """
    text = _normalise(request)
    for name, verb, obj, tools, reason in _COMPILED:
        if not verb.search(text) or not obj.search(text):
            continue
        # Conversation and explanation deliberately expose no tool at all. The
        # model should answer, not spend a CPU prompt evaluation on schemas it
        # cannot use. The patterns are anchored and every action rule is tried
        # first, so "how do I delete a file?" still reaches the trash tools.
        if name in {"conversation", "explain"}:
            return Route(intent=name, tools=(), reason=reason, narrowed=True)
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
