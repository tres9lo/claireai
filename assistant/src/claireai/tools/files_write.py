"""Approval-gated file tools (Month 1, Week 3).

Every tool here changes the filesystem, so every one is ``needs_approval``: the
agent asks the owner with the exact target and effect before it runs. Nothing in
this module deletes: ``move_to_trash`` relocates into the assistant's own trash
folder, because "recycle, don't delete" is a product rule (roadmap 7.4).

All paths go through the same :class:`claireai.permissions.Sandbox` as the
read-only tools, so a write can only ever land inside an allow-listed folder.
The trash folder is the single deliberate exception: it belongs to ClaireAI, and
the task log records the exact destination of every moved file.
"""

from __future__ import annotations

import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from pydantic import Field

from claireai.tools.base import ToolContext, ToolError, ToolResult
from claireai.tools.files import _Args, _human_size

MAX_WRITE_BYTES = 200_000


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _require_parent(target: Path) -> None:
    if not target.parent.exists():
        raise ToolError(f"{target.parent} does not exist; create the folder first")


def _refuse_if_exists(target: Path) -> None:
    if target.exists():
        raise ToolError(f"{target} already exists")


class CreateFolderArgs(_Args):
    path: str = Field(description="Absolute path of the folder to create.")
    parents: bool = Field(default=True, description="Create missing parent folders too.")


class CreateTextFileArgs(_Args):
    path: str = Field(description="Absolute path of the new file. It must not exist yet.")
    content: str = Field(description="UTF-8 text to write.")
    overwrite: bool = Field(default=False, description="Allow replacing an existing file.")


class WriteToFileArgs(_Args):
    path: str = Field(description="Absolute path of an existing file to modify.")
    content: str = Field(description="UTF-8 text to write.")
    mode: Literal["overwrite", "append"] = Field(
        default="overwrite", description="Replace the file or append to it."
    )


class RenameFileArgs(_Args):
    path: str = Field(description="Absolute path of the file or folder to rename.")
    new_name: str = Field(description="New name only, without folders. Stays in the same folder.")


class MoveFileArgs(_Args):
    source: str = Field(
        description="Absolute path of the item being moved: the file or folder that exists now."
    )
    destination: str = Field(
        description="Where it goes: an existing folder, or a full path for a new location. "
        "Never a file name. An existing target is never overwritten."
    )


class MoveToTrashArgs(_Args):
    path: str = Field(description="Absolute path of the file or folder to move to ClaireAI's trash.")


def _check_text(content: str) -> bytes:
    payload = content.encode("utf-8")
    if len(payload) > MAX_WRITE_BYTES:
        raise ToolError(
            f"content is {_human_size(len(payload))}; the limit is {_human_size(MAX_WRITE_BYTES)}"
        )
    return payload


def create_folder(context: ToolContext, arguments: CreateFolderArgs) -> ToolResult:
    target = context.sandbox.resolve(arguments.path)
    if target.exists():
        if target.is_dir():
            return ToolResult(
                text=f"{target} already exists (nothing to do)",
                detail={"path": str(target), "created": False},
            )
        raise ToolError(f"{target} exists and is not a folder")
    target.mkdir(parents=arguments.parents, exist_ok=True)
    return ToolResult(
        text=f"created folder {target}",
        detail={"path": str(target), "created": True, "parents": arguments.parents},
    )


def create_text_file(context: ToolContext, arguments: CreateTextFileArgs) -> ToolResult:
    target = context.sandbox.resolve(arguments.path)
    if target.exists() and not arguments.overwrite:
        raise ToolError(f"{target} already exists; use write_to_file to replace it")
    if target.exists() and target.is_dir():
        raise ToolError(f"{target} is a folder")
    _require_parent(target)
    payload = _check_text(arguments.content)
    target.write_bytes(payload)
    return ToolResult(
        text=f"wrote {_human_size(len(payload))} to {target}",
        detail={"path": str(target), "bytes": len(payload), "overwrote": arguments.overwrite},
    )


def write_to_file(context: ToolContext, arguments: WriteToFileArgs) -> ToolResult:
    target = context.sandbox.resolve(arguments.path)
    if not target.exists():
        raise ToolError(f"{target} does not exist; use create_text_file to create it")
    if not target.is_file():
        raise ToolError(f"{target} is not a regular file")
    payload = _check_text(arguments.content)

    if arguments.mode == "append":
        with target.open("ab") as handle:
            handle.write(payload)
    else:
        target.write_bytes(payload)
    return ToolResult(
        text=f"{arguments.mode} wrote {_human_size(len(payload))} to {target}",
        detail={
            "path": str(target),
            "bytes": len(payload),
            "mode": arguments.mode,
            "bytes_after": target.stat().st_size,
        },
    )


def rename_file(context: ToolContext, arguments: RenameFileArgs) -> ToolResult:
    source = context.sandbox.resolve(arguments.path)
    if not source.exists():
        raise ToolError(f"{source} does not exist")
    new_name = arguments.new_name.strip()
    if not new_name or new_name in {".", ".."} or "/" in new_name or "\\" in new_name:
        raise ToolError("new_name must be a plain file name without folders")

    destination = context.sandbox.resolve(source.parent / new_name)
    if destination == source:
        raise ToolError("new_name is the same as the current name")
    _refuse_if_exists(destination)

    source.rename(destination)
    return ToolResult(
        text=f"renamed {source.name} to {destination.name}",
        detail={"path": str(source), "destination": str(destination)},
    )


def _looks_like_file_name(name: str) -> bool:
    """True when ``name`` carries a file extension.

    Deliberately loose: a folder called ``v1.2`` would match, so callers must
    only apply it to a destination that does not already exist — an existing
    folder is never in question.
    """
    suffix = Path(name).suffix
    return bool(suffix) and suffix != "."


def move_file(context: ToolContext, arguments: MoveFileArgs) -> ToolResult:
    source = context.sandbox.resolve(arguments.source)
    if not source.exists():
        raise ToolError(f"{source} does not exist")

    destination_arg = context.sandbox.resolve(arguments.destination)
    destination_is_dir = destination_arg.is_dir()
    destination = destination_arg
    if destination_is_dir:
        destination = context.sandbox.resolve(destination_arg / source.name)
    if destination == source:
        raise ToolError("source and destination are the same path")

    # Guards against the arguments arriving reversed — observed for real with a
    # thin model that answered {source: "Documents",
    # destination: "quarterly-report.pdf"}. Reversed arguments here would move a
    # whole folder onto a file-shaped path; approval shows the arguments, but the
    # refusal before that gate is cheaper than asking the owner to catch it.
    if source.is_dir() and not destination_is_dir:
        if destination_arg.is_file():
            raise ToolError(
                f"destination {destination_arg} is a file, but source {source} is a "
                "folder; the arguments look reversed — pass a folder as the destination"
            )
        if not destination.exists() and _looks_like_file_name(destination_arg.name):
            raise ToolError(
                f"destination {destination_arg.name!r} is a file name, but source "
                f"{source.name!r} is a folder; the arguments look reversed — pass the "
                "folder as source and the folder it belongs in as destination"
            )

    _refuse_if_exists(destination)
    _require_parent(destination)

    shutil.move(str(source), str(destination))
    return ToolResult(
        text=f"moved {source.name} to {destination}",
        detail={"path": str(source), "destination": str(destination)},
    )


def move_to_trash(context: ToolContext, arguments: MoveToTrashArgs) -> ToolResult:
    source = context.sandbox.resolve(arguments.path)
    if not source.exists():
        raise ToolError(f"{source} does not exist")
    if context.data_dir is None:
        raise ToolError("no data directory is configured, so the trash folder is unknown")

    trash_root = Path(context.data_dir) / "trash" / _utc_stamp()
    trash_root.mkdir(parents=True, exist_ok=True)
    destination = trash_root / source.name
    counter = 1
    while destination.exists():
        destination = trash_root / f"{source.name}.{counter}"
        counter += 1

    shutil.move(str(source), str(destination))
    return ToolResult(
        text=f"moved {source.name} to ClaireAI's trash ({destination})",
        detail={
            "path": str(source),
            "trash": str(destination),
            "recoverable": True,
            "permanent": False,
        },
    )
