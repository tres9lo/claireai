"""The registered tool set.

Read-only tools are ``safe`` and run freely. Every mutating tool is
``needs_approval``, so the agent asks the owner with the exact target and effect
before it runs (:mod:`claireai.approval`). A run without an approval channel
refuses them — that refusal is the point, not a bug.
"""

from __future__ import annotations

from claireai.tools.base import Tool, ToolContext, ToolError, ToolRegistry, ToolResult, UnknownTool
from claireai.tools.browser import (
    DownloadFileArgs,
    OpenUrlArgs,
    PageTextArgs,
    SearchWebArgs,
    download_file,
    get_page_text,
    open_url,
    search_web,
)
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
from claireai.tools.calendar import (
    AddAppointmentArgs,
    ListAppointmentsArgs,
    add_appointment,
    list_appointments,
)
from claireai.tools.vision import (
    IdentifyImageArgs,
    SeeScreenArgs,
    identify_image,
    see_screen,
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
        mutates=True,
    ),
    Tool(
        name="create_text_file",
        description="Create a new UTF-8 text file.",
        risk="needs_approval",
        args_model=CreateTextFileArgs,
        handler=create_text_file,
        mutates=True,
    ),
    Tool(
        name="write_to_file",
        description="Overwrite or append to an existing text file.",
        risk="needs_approval",
        args_model=WriteToFileArgs,
        handler=write_to_file,
        mutates=True,
    ),
    Tool(
        name="rename_file",
        description="Rename a file or folder in place.",
        risk="needs_approval",
        args_model=RenameFileArgs,
        handler=rename_file,
        mutates=True,
    ),
    Tool(
        name="move_file",
        description="Move a file or folder; never overwrites.",
        risk="needs_approval",
        args_model=MoveFileArgs,
        handler=move_file,
        mutates=True,
    ),
    Tool(
        name="move_to_trash",
        description="Move to a recoverable trash folder; nothing is deleted permanently.",
        risk="needs_approval",
        args_model=MoveToTrashArgs,
        handler=move_to_trash,
        mutates=True,
    ),
)


BROWSER_TOOLS: tuple[Tool, ...] = (
    Tool(
        name="open_url",
        description="Open an http(s) page and return its readable text.",
        risk="needs_approval",
        args_model=OpenUrlArgs,
        handler=open_url,
    ),
    Tool(
        name="get_page_text",
        description="The text of the page this task last opened.",
        risk="safe",
        args_model=PageTextArgs,
        handler=get_page_text,
    ),
    Tool(
        name="search_web",
        description="Search the web and list result titles with their links.",
        risk="needs_approval",
        args_model=SearchWebArgs,
        handler=search_web,
    ),
    Tool(
        name="download_file",
        description="Download a URL to a new file inside an allowed folder.",
        risk="needs_approval",
        args_model=DownloadFileArgs,
        handler=download_file,
        mutates=True,
    ),
)


VISION_TOOLS: tuple[Tool, ...] = (
    Tool(
        name="identify_image",
        description="Look at an image file and say what it most likely shows.",
        risk="safe",
        args_model=IdentifyImageArgs,
        handler=identify_image,
    ),
    # Reading the whole screen exposes whatever else the owner has open, so it
    # is never a free action even though it changes nothing.
    Tool(
        name="see_screen",
        description="Screenshot the display and say what is on it.",
        risk="needs_approval",
        args_model=SeeScreenArgs,
        handler=see_screen,
    ),
)


CALENDAR_TOOLS: tuple[Tool, ...] = (
    Tool(
        name="list_appointments",
        description="What is scheduled on a day: 'today', 'tomorrow', '2026-11-07' or a weekday name.",
        risk="safe",
        args_model=ListAppointmentsArgs,
        handler=list_appointments,
    ),
    Tool(
        name="add_appointment",
        description="Add an appointment to the local calendar for a day and time. Needs approval.",
        risk="needs_approval",
        args_model=AddAppointmentArgs,
        handler=add_appointment,
    ),
)


def default_registry(*, browser_enabled: bool = False) -> ToolRegistry:
    """Build the active tools; browser networking stays off unless enabled."""
    tools = (
        *FILE_TOOLS,
        *WRITE_TOOLS,
        *CALENDAR_TOOLS,
        *VISION_TOOLS,
        *(BROWSER_TOOLS if browser_enabled else ()),
    )
    return ToolRegistry(tools)


__all__ = [
    "AddAppointmentArgs",
    "BROWSER_TOOLS",
    "CALENDAR_TOOLS",
    "FILE_TOOLS",
    "ListAppointmentsArgs",
    "LargestFilesArgs",
    "VISION_TOOLS",
    "WRITE_TOOLS",
    "Tool",
    "ToolContext",
    "ToolError",
    "ToolRegistry",
    "ToolResult",
    "UnknownTool",
    "default_registry",
]
