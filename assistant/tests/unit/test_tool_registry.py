"""The registry is the model's menu: names, schemas, risk levels."""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from claireai.tools import default_registry
from claireai.tools.base import Tool, ToolContext, ToolError, ToolRegistry, ToolResult, UnknownTool


class NoArgs(BaseModel):
    pass


def _tool(name: str = "noop", risk: str = "safe") -> Tool:
    return Tool(
        name=name,
        description="does nothing",
        risk=risk,
        args_model=NoArgs,
        handler=lambda context, arguments: ToolResult(text="ok"),
    )


def test_default_registry_has_the_read_only_tools() -> None:
    registry = default_registry()
    assert registry.names() == (
        "find_largest_files",
        "get_file_info",
        "list_folder",
        "read_text_file",
        "search_files",
    )
    assert all(tool.risk == "safe" for tool in registry)


def test_schemas_are_function_shaped_for_ollama() -> None:
    schema = default_registry().get("list_folder").schema()
    assert schema["type"] == "function"
    function = schema["function"]
    assert function["name"] == "list_folder"
    assert function["parameters"]["type"] == "object"
    assert "path" in function["parameters"]["required"]
    assert "description" in function


def test_duplicate_names_are_rejected() -> None:
    with pytest.raises(ValueError, match="duplicate tool name"):
        ToolRegistry([_tool(), _tool()])


def test_unknown_risk_level_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown risk level"):
        _tool(risk="probably-fine")


def test_unknown_tool_error_lists_what_exists() -> None:
    registry = default_registry()
    with pytest.raises(UnknownTool) as excinfo:
        registry.get("format_disk")
    assert "unknown tool" in str(excinfo.value)
    assert "list_folder" in str(excinfo.value)


def test_arguments_must_match_the_tool_model() -> None:
    tool = default_registry().get("list_folder")
    with pytest.raises(ToolError, match="invalid arguments"):
        tool.validate({})  # path is required


def test_unexpected_argument_keys_are_rejected() -> None:
    tool = default_registry().get("list_folder")
    with pytest.raises(ToolError, match="invalid arguments"):
        tool.validate({"path": "/tmp", "recursive": True})


def test_null_optional_arguments_fall_back_to_defaults() -> None:
    # qwen2.5:1.5b really does send {"limit": null} instead of omitting it.
    tool = default_registry().get("list_folder")
    validated = tool.validate({"path": "/tmp", "limit": None, "sort": None})
    assert validated.limit == 200
    assert validated.sort == "name"


def test_a_null_required_argument_is_still_missing() -> None:
    tool = default_registry().get("list_folder")
    with pytest.raises(ToolError, match="invalid arguments"):
        tool.validate({"path": None})


def test_registry_reports_membership_and_length() -> None:
    registry = ToolRegistry([_tool()])
    assert "noop" in registry
    assert "other" not in registry
    assert len(registry) == 1
    assert registry.schemas()[0]["function"]["name"] == "noop"
    assert ToolContext.__dataclass_fields__["dry_run"].default is False
