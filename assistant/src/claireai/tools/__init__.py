"""The registered tool set.

Week 2 ships only read-only file tools. Adding a mutating tool means adding a
non-``safe`` risk level, and :func:`claireai.permissions.check_risk` will refuse
it until an approval flow exists — that refusal is the point, not a bug.
"""

from __future__ import annotations

from claireai.tools.base import Tool, ToolContext, ToolError, ToolRegistry, ToolResult, UnknownTool
from claireai.tools.files import (
    FileInfoArgs,
    LargestFilesArgs,
    ListFolderArgs,
    ReadTextFileArgs,
    SearchFilesArgs,
    find_largest_files,
    get_file_info,
    list_folder,
    read_text_file,
    search_files,
)

FILE_TOOLS: tuple[Tool, ...] = (
    Tool(
        name="list_folder",
        description=(
            "List the entries of one folder. Use sort='size' with descending=true "
            "to find the biggest files in a folder."
        ),
        risk="safe",
        args_model=ListFolderArgs,
        handler=list_folder,
    ),
    Tool(
        name="get_file_info",
        description="Report the type, size and modification time of one file or folder.",
        risk="safe",
        args_model=FileInfoArgs,
        handler=get_file_info,
    ),
    Tool(
        name="read_text_file",
        description="Read a UTF-8 text file and return its contents, truncated to a byte budget.",
        risk="safe",
        args_model=ReadTextFileArgs,
        handler=read_text_file,
    ),
    Tool(
        name="search_files",
        description="Search a folder tree for files whose name matches a glob pattern.",
        risk="safe",
        args_model=SearchFilesArgs,
        handler=search_files,
    ),
    Tool(
        name="find_largest_files",
        description=(
            "Find the biggest files inside a folder, including its subfolders, "
            "sorted biggest first with their real sizes. Use this for any request "
            "about the largest or biggest files; do not sort a listing yourself."
        ),
        risk="safe",
        args_model=LargestFilesArgs,
        handler=find_largest_files,
    ),
)


def default_registry() -> ToolRegistry:
    return ToolRegistry(FILE_TOOLS)


__all__ = [
    "FILE_TOOLS",
    "LargestFilesArgs",
    "Tool",
    "ToolContext",
    "ToolError",
    "ToolRegistry",
    "ToolResult",
    "UnknownTool",
    "default_registry",
]
