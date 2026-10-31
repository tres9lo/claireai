"""The deterministic tool router.

Two properties matter and are tested separately:

* it narrows the *useful* way, on the requests the product actually serves;
* it can never remove a tool from the sandbox, the risk policy or approval —
  that is enforced by the agent using the full registry for execution, and by
  the invariant that a route is always a subset of the registry.
"""

from __future__ import annotations

import json

import pytest

from claireai.routing import INTENTS, describe, reachable_tools, route, schema_bytes
from claireai.tools import default_registry


@pytest.fixture()
def registry():
    return default_registry()


@pytest.mark.parametrize(
    ("request_text", "expected_intent", "must_include"),
    [
        ("List the biggest files in my Downloads folder", "largest", "find_largest_files"),
        ("which files are the biggest inside my Downloads folder", "largest", "find_largest_files"),
        ("what is taking up space in my Downloads folder", "largest", "find_largest_files"),
        ("show me what is inside my Downloads folder", "list", "list_folder"),
        ("find every pdf inside my Documents folder", "search", "search_files"),
        ("put quarterly-report.pdf into my Documents folder", "move", "move_file"),
        ("move quarterly-report.pdf into my Documents folder", "move", "move_file"),
        ("rename report.pdf to final.pdf", "rename", "rename_file"),
        ("delete everything in my Documents folder", "trash", "move_to_trash"),
        ("write a file called todo.txt in my Documents folder saying buy milk", "write", "create_text_file"),
        ("create a folder called reports in Downloads", "create_folder", "create_folder"),
        ("organise my Downloads folder into folders by file type", "organize", "move_file"),
        ("read the file notes.txt and summarise it", "read", "read_text_file"),
    ],
)
def test_routes_real_requests_to_the_tool_that_matters(
    registry, request_text: str, expected_intent: str, must_include: str
) -> None:
    result = route(request_text, registry)
    assert result.intent == expected_intent
    assert must_include in result.tools


def test_organizing_needs_the_whole_plan_not_one_tool(registry) -> None:
    # The Week 3 acceptance request cannot be done by one tool: it lists,
    # creates folders and moves files. A route that showed only move_file would
    # make a correct plan impossible.
    result = route("organise my Downloads folder into folders by file type", registry)
    assert set(result.tools) >= {"list_folder", "create_folder", "move_file"}


def test_conversation_gets_every_tool(registry) -> None:
    # Nothing to go on means nothing may be hidden.
    for text in ("thanks, that is all for now", "what is a tokenizer?", ""):
        result = route(text, registry)
        assert result.intent == "unrouted"
        assert set(result.tools) == set(registry.names())
        assert result.narrowed is False


def test_writing_about_a_journal_is_not_a_write_request(registry) -> None:
    # The verb matches; the object does not. Requiring both is what keeps a
    # question from being starved of tools.
    result = route("what did I write in my journal on 12 March?", registry)
    assert result.intent == "unrouted"


def test_a_route_is_always_a_non_empty_subset(registry) -> None:
    requests = [
        "delete it",
        "rename it",
        "make a folder",
        "organise things",
        "move it",
        "save it",
        "biggest files",
        "search pdf",
        "open it",
        "how big is it",
        "list it",
        "!!!",
    ]
    known = set(registry.names())
    for text in requests:
        result = route(text, registry)
        assert result.tools, f"{text!r} routed to nothing"
        assert set(result.tools) <= known, f"{text!r} invented a tool"
        assert len(result.tools) == len(set(result.tools))


def test_routing_reduces_schema_payload(registry) -> None:
    narrowed = route("List the biggest files in my Downloads folder", registry)
    full = registry.schemas()
    reduced = narrowed.schemas(registry)
    assert len(reduced) < len(full)
    assert schema_bytes(reduced) < schema_bytes(full)
    # A real saving, not a rounding difference: the whole point was prompt bytes.
    assert schema_bytes(reduced) <= schema_bytes(full) * 0.75


def test_schemas_put_the_primary_tool_first(registry) -> None:
    result = route("List the biggest files in my Downloads folder", registry)
    names = [schema["function"]["name"] for schema in result.schemas(registry)]
    assert names == list(result.tools)
    assert names[0] == "find_largest_files"


def test_no_tool_is_orphaned(registry) -> None:
    # Every registered tool must be reachable from at least one intent, or it is
    # dead weight that only ever costs prompt tokens.
    assert reachable_tools() <= set(registry.names())
    assert set(registry.names()) <= reachable_tools()


def test_one_tool_intents_name_exactly_one_tool(registry) -> None:
    # Competition is what made the model choose list_folder for a "biggest files"
    # request once the tool count dropped. Where one tool owns the intent, it is
    # offered alone.
    assert route("List the biggest files in my Downloads folder", registry).tools == (
        "find_largest_files",
    )
    assert route("show me what is inside my Downloads folder", registry).tools == ("list_folder",)
    assert route("find every pdf inside my Documents folder", registry).tools == ("search_files",)


def test_every_intent_that_can_write_also_offers_a_read(registry) -> None:
    # A write needs a fact only a read can establish: does the source exist, is
    # there already a file there, what is really in the folder. Without one,
    # routing forces the model to invent a path — measured with qwen2.5:1.5b,
    # which named /Users/yourname/quarterly-report.pdf for a move in the real
    # loop and was refused by the sandbox, twice.
    reads = {name for name in registry.names() if registry.get(name).risk == "safe"}
    assert reads, "no read-only tool exists, so the invariant is meaningless"

    for name, verb, obj, tools, reason in INTENTS:
        offered = [tool for tool in tools if tool in registry]
        writes = [tool for tool in offered if registry.get(tool).risk != "safe"]
        if not writes:
            continue
        readable = [tool for tool in offered if tool in reads]
        assert readable, f"{name!r} can write via {writes} but offers no read tool"


def test_a_move_offers_discovery_and_leads_with_the_move(registry) -> None:
    # Both ends of a move have to be discovered, and the tool that owns the
    # intent must still come first — order is a prior for a small model.
    result = route("put quarterly-report.pdf into my Documents folder", registry)
    assert set(result.tools) >= {"move_file", "list_folder", "get_file_info"}
    assert result.tools[0] == "move_file"
    names = [schema["function"]["name"] for schema in result.schemas(registry)]
    assert names[0] == "move_file"


def test_a_write_offers_the_check_that_decides_create_or_overwrite(registry) -> None:
    # create_text_file refuses an existing file and write_to_file requires one,
    # so which of them to call is a fact only get_file_info can settle.
    result = route("write a file called todo.txt in my Documents folder saying buy milk", registry)
    assert set(result.tools) >= {"create_text_file", "write_to_file", "get_file_info"}
    assert result.tools[0] == "create_text_file"


def test_intent_metadata_is_well_formed() -> None:
    names = [name for name, *_ in INTENTS]
    assert len(names) == len(set(names))
    for name, verb, obj, tools, reason in INTENTS:
        assert name.strip() and reason.strip()
        assert verb and obj, f"{name} needs both a verb and an object pattern"
        assert tools, f"{name} offers no tools"


def test_describe_states_the_saving(registry) -> None:
    assert describe(route("find every pdf", registry), registry) == "search (1/11 tools)"


def test_route_is_json_serialisable_for_the_audit_detail(registry) -> None:
    result = route("find every pdf", registry)
    payload = json.dumps({"intent": result.intent, "tools": list(result.tools)})
    assert json.loads(payload)["intent"] == "search"
