"""The registered tool set.

Read-only tools are ``safe`` and run freely. Every mutating tool is
``needs_approval``, so the agent asks the owner with the exact target and effect
before it runs (:mod:`claireai.approval`). A run without an approval channel
refuses them — that refusal is the point, not a bug.
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
from claireai.tools.files_write import (
    CreateFolderArgs,
    CreateTextFileArgs,
    MoveFileArgs,
    MoveToTrashArgs,
    RenameFileArgs,
    WriteToFileArgs,
    create_folder,
    create_text_file,
    move_file,
    move_to_trash,
    rename_file,
    write_to_file,
)

FILE_TOOLS: tuple[Tool, ...] = (
    Tool(
        name="list_folder",
        description="List one folder's entries with sizes. Not recursive.",
        risk="safe",
        args_model=ListFolderArgs,
        handler=list_folder,
    ),
    Tool(
        name="get_file_info",
        description="Type, size and modified time of one path.",
        risk="safe",
        args_model=FileInfoArgs,
        handler=get_file_info,
    ),
    Tool(
        name="read_text_file",
        description="Read a UTF-8 text file, truncated to a byte budget.",
        risk="safe",
        args_model=ReadTextFileArgs,
        handler=read_text_file,
    ),
    Tool(
        name="search_files",
        description="Find files by name glob under a folder.",
        risk="safe",
        args_model=SearchFilesArgs,
        handler=search_files,
    ),
    Tool(
        name="find_largest_files",
        description="The biggest files under a folder, recursive, with real sizes. Use whenever the owner asks what is biggest, largest, or taking up space.",
        risk="safe",
        args_model=LargestFilesArgs,
        handler=find_largest_files,
    ),
)


WRITE_TOOLS: tuple[Tool, ...] = (
    Tool(
        name="create_folder",
        description="Create a folder.",
        risk="needs_approval",
        args_model=CreateFolderArgs,
        handler=create_folder,
    ),
    Tool(
        name="create_text_file",
        description="Create a new UTF-8 text file.",
        risk="needs_approval",
        args_model=CreateTextFileArgs,
        handler=create_text_file,
    ),
    Tool(
        name="write_to_file",
        description="Overwrite or append to an existing text file.",
        risk="needs_approval",
        args_model=WriteToFileArgs,
        handler=write_to_file,
    ),
    Tool(
        name="rename_file",
        description="Rename a file or folder in place.",
        risk="needs_approval",
        args_model=RenameFileArgs,
        handler=rename_file,
    ),
    Tool(
        name="move_file",
        description="Move a file or folder; never overwrites.",
        risk="needs_approval",
        args_model=MoveFileArgs,
        handler=move_file,
    ),
    Tool(
        name="move_to_trash",
        description="Move to a recoverable trash folder; nothing is deleted permanently.",
        risk="needs_approval",
        args_model=MoveToTrashArgs,
        handler=move_to_trash,
    ),
)


def default_registry() -> ToolRegistry:
    return ToolRegistry((*FILE_TOOLS, *WRITE_TOOLS))


__all__ = [
    "FILE_TOOLS",
    "LargestFilesArgs",
    "WRITE_TOOLS",
    "Tool",
    "ToolContext",
    "ToolError",
    "ToolRegistry",
    "ToolResult",
    "UnknownTool",
    "default_registry",
]
