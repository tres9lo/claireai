"""Source tool names may only become real runtime calls through explicit aliases."""

from __future__ import annotations

import sys
from pathlib import Path

from claireai.tools import default_registry

ROOT = Path(__file__).resolve().parents[3]
BRAIN_LAB = ROOT / "brain-lab"
if str(BRAIN_LAB) not in sys.path:
    sys.path.insert(0, str(BRAIN_LAB))

import tool_mapping  # noqa: E402


def test_every_alias_targets_a_real_registry_tool() -> None:
    registry = default_registry(browser_enabled=True)
    assert tool_mapping.ALIASES
    for alias in tool_mapping.ALIASES.values():
        assert alias.target in registry


def test_create_directory_maps_to_create_folder() -> None:
    call = tool_mapping.map_call(
        {"name": "create_directory", "arguments": {"path": "~/Documents/new"}}
    )
    assert call == {"name": "create_folder", "arguments": {"path": "~/Documents/new"}}


def test_list_directory_contents_maps_its_hidden_flag() -> None:
    call = tool_mapping.map_call(
        {
            "name": "list_directory_contents",
            "arguments": {"path": "~/Downloads", "all_files": True},
        }
    )
    assert call == {
        "name": "list_folder",
        "arguments": {"path": "~/Downloads", "include_hidden": True},
    }


def test_move_folder_maps_source_and_destination() -> None:
    call = tool_mapping.map_call(
        {
            "name": "move_folder",
            "arguments": {"source_path": "~/Downloads/a", "destination_path": "~/Documents/a"},
        }
    )
    assert call == {
        "name": "move_file",
        "arguments": {"source": "~/Downloads/a", "destination": "~/Documents/a"},
    }


def test_extension_search_builds_a_glob_pattern() -> None:
    call = tool_mapping.map_call(
        {
            "name": "search_files_by_extension",
            "arguments": {"directory": "~/Downloads", "extension": "txt"},
        }
    )
    assert call == {
        "name": "search_files",
        "arguments": {"path": "~/Downloads", "pattern": "*.txt"},
    }


def test_unknown_tools_and_invalid_arguments_are_dropped() -> None:
    assert tool_mapping.map_call({"name": "execute_trades", "arguments": {}}) is None
    assert tool_mapping.map_call({"name": "create_directory", "arguments": {"folder": "x"}}) is None
    assert tool_mapping.map_call({"name": "list_folder", "arguments": {}}) is None


def test_available_tools_drop_unsupported_names_and_keep_order() -> None:
    assert tool_mapping.map_available(
        ["create_directory", "execute_trades", "list_folder", "list_directory"]
    ) == ["create_folder", "list_folder"]
