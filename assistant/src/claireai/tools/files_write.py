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

import os
import shutil
import stat
import tempfile
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
    # lexists also catches a broken symlink. exists() would return False for it,
    # and a later copy could then follow or replace the link unexpectedly.
    if os.path.lexists(target):
        raise ToolError(f"{target} already exists")


def _copy_no_follow(source: str | Path, destination: str | Path) -> str:
    """Copy a file without following a symlink swapped in after the check."""
    return shutil.copy2(source, destination, follow_symlinks=False)


def _refuse_symlink(target: Path) -> None:
    if target.is_symlink():
        raise ToolError(
            f"{target} is a symbolic link; rename, move and trash operate on "
            "real files and folders. Pass the real path instead."
        )


def _resolve_mutation(context: ToolContext, raw: str) -> Path:
    """Resolve a rename/move/trash path without following a final symlink.

    A symlink inside the allow-list can point at the link itself or at an
    outside target.  Following it would make these tools rename or trash the
    target, so the consistent rule is to refuse the link and ask for its real
    path instead.
    """
    target = context.sandbox.resolve(raw, follow_final_symlink=False)
    _refuse_symlink(target)
    return target


def _move_no_follow(source: Path, destination: Path) -> None:
    """Move a path, rechecking for a symlink at the operation boundary.

    The earlier resolve() check catches the normal case.  Rechecking here closes
    the practical gap where another process replaces the source between that
    check and the move.  Cross-device copies use copy2 with follow_symlinks=False
    so a swapped-in link is copied or unlinked as a link, never dereferenced.
    """
    _refuse_symlink(source)
    shutil.move(str(source), str(destination), copy_function=_copy_no_follow)


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


class ReplaceInFileArgs(_Args):
    path: str = Field(description="Absolute path of an existing UTF-8 text file.")
    old_text: str = Field(
        min_length=1,
        description=(
            "Exact text to find, including enough surrounding context to make it "
            "unique. A literal match, not a regular expression."
        ),
    )
    new_text: str = Field(description="Replacement text. Empty removes old_text.")
    replace_all: bool = Field(
        default=False,
        description="Replace every occurrence. The default refuses when old_text is not unique.",
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


def _read_editable_text(target: Path) -> str:
    """Read a bounded UTF-8 file for an in-place edit."""
    if not target.exists():
        raise ToolError(f"{target} does not exist")
    if not target.is_file():
        raise ToolError(f"{target} is not a regular file")
    size = target.stat().st_size
    if size > MAX_WRITE_BYTES:
        raise ToolError(
            f"{target} is {_human_size(size)}; in-place edits are limited to "
            f"{_human_size(MAX_WRITE_BYTES)}. Use a smaller file or a dedicated script."
        )
    raw = target.read_bytes()
    if b"\x00" in raw:
        raise ToolError(f"{target} looks like a binary file, not text")
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ToolError(f"{target} is not valid UTF-8 text (byte {exc.start})") from exc


def _atomic_write(target: Path, payload: bytes) -> None:
    """Replace a file without leaving a half-written edit after a crash."""
    mode = stat.S_IMODE(target.stat().st_mode)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".claireai-tmp", dir=target.parent
    )
    temporary_path = Path(temporary)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary_path, mode)
        os.replace(temporary_path, target)
    except BaseException:
        try:
            temporary_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise


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


def replace_in_file(context: ToolContext, arguments: ReplaceInFileArgs) -> ToolResult:
    """Make one surgical text replacement instead of rewriting a whole file."""
    target = context.sandbox.resolve(arguments.path)
    original = _read_editable_text(target)
    occurrences = original.count(arguments.old_text)
    if occurrences == 0:
        raise ToolError(
            f"old_text was not found in {target}; read the file and copy the exact "
            "text you want to replace"
        )
    if occurrences > 1 and not arguments.replace_all:
        raise ToolError(
            f"old_text appears {occurrences} times in {target}; include more "
            "surrounding context, or set replace_all=true if every occurrence should change"
        )

    replacements = occurrences if arguments.replace_all else 1
    updated = original.replace(
        arguments.old_text, arguments.new_text, -1 if arguments.replace_all else 1
    )
    payload = _check_text(updated)
    _atomic_write(target, payload)
    return ToolResult(
        text=(
            f"replaced {replacements} occurrence(s) in {target} "
            f"({_human_size(len(original.encode('utf-8')))} -> {_human_size(len(payload))})"
        ),
        detail={
            "path": str(target),
            "occurrences": replacements,
            "bytes_before": len(original.encode("utf-8")),
            "bytes_after": len(payload),
            "replace_all": arguments.replace_all,
        },
    )


def rename_file(context: ToolContext, arguments: RenameFileArgs) -> ToolResult:
    source = _resolve_mutation(context, arguments.path)
    if not source.exists():
        raise ToolError(f"{source} does not exist")
    new_name = arguments.new_name.strip()
    if not new_name or new_name in {".", ".."} or "/" in new_name or "\\" in new_name:
        raise ToolError("new_name must be a plain file name without folders")

    destination = _resolve_mutation(context, str(source.parent / new_name))
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
    source = _resolve_mutation(context, arguments.source)
    if not source.exists():
        raise ToolError(f"{source} does not exist")

    destination_arg = _resolve_mutation(context, arguments.destination)
    destination_is_dir = destination_arg.is_dir()
    destination = destination_arg
    if destination_is_dir:
        destination = _resolve_mutation(context, str(destination_arg / source.name))
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

    _move_no_follow(source, destination)
    return ToolResult(
        text=f"moved {source.name} to {destination}",
        detail={"path": str(source), "destination": str(destination)},
    )


def move_to_trash(context: ToolContext, arguments: MoveToTrashArgs) -> ToolResult:
    source = _resolve_mutation(context, arguments.path)
    if not source.exists():
        raise ToolError(f"{source} does not exist")
    if context.data_dir is None:
        raise ToolError("no data directory is configured, so the trash folder is unknown")

    trash_root = Path(context.data_dir) / "trash" / _utc_stamp()
    trash_root.mkdir(parents=True, exist_ok=True)
    destination = trash_root / source.name
    counter = 1
    while os.path.lexists(destination):
        destination = trash_root / f"{source.name}.{counter}"
        counter += 1

    _move_no_follow(source, destination)
    return ToolResult(
        text=f"moved {source.name} to ClaireAI's trash ({destination})",
        detail={
            "path": str(source),
            "trash": str(destination),
            "recoverable": True,
            "permanent": False,
        },
    )
