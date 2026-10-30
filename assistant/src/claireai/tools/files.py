"""Read-only file tools (Month 1, Week 2).

Every tool here is ``safe``: it reads and never modifies, so it runs without an
approval prompt. All paths go through :class:`claireai.permissions.Sandbox`
first, so an allow-list entry is the only way in.

Results are deliberately plain text with bounded sizes: they are fed back to a
1.5B model and stored in the audit log, and neither should receive a whole
directory tree or a large file.
"""

from __future__ import annotations

import fnmatch
import os
import stat
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from claireai.permissions import PermissionDenied, Sandbox
from claireai.tools.base import ToolContext, ToolError, ToolResult

MAX_READ_BYTES = 200_000
MAX_SCAN_ENTRIES = 20_000


def _human_size(size: int) -> str:
    value = float(size)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024 or unit == "TB":
            return f"{value:.0f} {unit}" if unit == "B" else f"{value:.1f} {unit}"
        value /= 1024
    return f"{value:.1f} TB"  # pragma: no cover - unreachable


def _format_time(timestamp: float) -> str:
    return datetime.fromtimestamp(timestamp, tz=timezone.utc).isoformat(timespec="seconds")


def _require_directory(path: Path) -> Path:
    if not path.exists():
        raise ToolError(f"{path} does not exist")
    if not path.is_dir():
        raise ToolError(f"{path} is not a folder")
    return path


def _require_file(path: Path) -> Path:
    if not path.exists():
        raise ToolError(f"{path} does not exist")
    if not path.is_file():
        raise ToolError(f"{path} is not a regular file")
    return path


class _Args(BaseModel):
    """Reject unknown keys: a tool must not be handed arguments it ignores."""

    model_config = ConfigDict(extra="forbid")


class ListFolderArgs(_Args):
    path: str = Field(description="Absolute path of the folder to list.")
    include_hidden: bool = Field(default=False, description="Include dotfiles.")
    sort: Literal["name", "size"] = Field(default="name", description="Sort by name or by size.")
    descending: bool = Field(
        default=False,
        description="Sort descending. Use with sort=size to get the biggest entries first.",
    )
    limit: int = Field(default=200, ge=1, le=1000, description="Maximum entries to return.")


class FileInfoArgs(_Args):
    path: str = Field(description="Absolute path of the file or folder to inspect.")


class ReadTextFileArgs(_Args):
    path: str = Field(description="Absolute path of the UTF-8 text file to read.")
    max_bytes: int = Field(default=20_000, ge=1, le=MAX_READ_BYTES, description="Byte budget.")


class SearchFilesArgs(_Args):
    path: str = Field(description="Absolute path of the folder to search inside.")
    pattern: str = Field(description="Glob matched against file names, e.g. '*.pdf' or 'invoice*'.")
    max_results: int = Field(default=100, ge=1, le=500, description="Maximum matches to return.")


class LargestFilesArgs(_Args):
    path: str = Field(description="Absolute path of the folder to search inside, recursively.")
    limit: int = Field(default=10, ge=1, le=100, description="How many files to return.")
    max_depth: int = Field(
        default=6,
        ge=1,
        le=20,
        description=(
            "How many folder levels to include, counting the given folder as "
            "level 1. max_depth=1 searches only that folder."
        ),
    )


def list_folder(context: ToolContext, arguments: ListFolderArgs) -> ToolResult:
    folder = _require_directory(context.sandbox.resolve(arguments.path))

    entries: list[tuple[str, bool, int, float]] = []
    try:
        with os.scandir(folder) as iterator:
            for entry in iterator:
                if not arguments.include_hidden and entry.name.startswith("."):
                    continue
                try:
                    # Do not follow links here: the link itself is what is listed.
                    info = entry.stat(follow_symlinks=False)
                except OSError:
                    continue
                entries.append((entry.name, entry.is_dir(follow_symlinks=False), info.st_size, info.st_mtime))
    except PermissionError as exc:
        raise ToolError(f"the operating system denied reading {folder}: {exc.strerror}") from exc

    if arguments.sort == "size":
        entries.sort(key=lambda item: (item[2], item[0].lower()), reverse=arguments.descending)
    else:
        entries.sort(key=lambda item: (not item[1], item[0].lower()), reverse=arguments.descending)

    shown = entries[: arguments.limit]
    lines = []
    for name, is_dir, size, modified in shown:
        if is_dir:
            lines.append(f"dir            {name}/")
        else:
            lines.append(f"file {_human_size(size):>9}  {name}")
    footer = f"{len(entries)} entr{'y' if len(entries) == 1 else 'ies'} in {folder}"
    if len(shown) < len(entries):
        footer += f", showing {len(shown)}"

    return ToolResult(
        text="\n".join([footer, *lines]) if lines else footer + " (empty)",
        detail={
            "path": str(folder),
            "entries": len(entries),
            "returned": len(shown),
            "sort": arguments.sort,
            "descending": arguments.descending,
        },
    )


def get_file_info(context: ToolContext, arguments: FileInfoArgs) -> ToolResult:
    target = context.sandbox.resolve(arguments.path)
    if not target.exists():
        raise ToolError(f"{target} does not exist")
    info = target.stat()
    kind = "folder" if stat.S_ISDIR(info.st_mode) else "file" if stat.S_ISREG(info.st_mode) else "other"
    fields = {
        "path": str(target),
        "name": target.name,
        "type": kind,
        "size_bytes": info.st_size,
        "modified_utc": _format_time(info.st_mtime),
    }
    text = "\n".join(f"{key}: {value}" for key, value in fields.items())
    return ToolResult(text=text, detail=fields)


def read_text_file(context: ToolContext, arguments: ReadTextFileArgs) -> ToolResult:
    target = _require_file(context.sandbox.resolve(arguments.path))
    size = target.stat().st_size

    try:
        with target.open("rb") as handle:
            raw = handle.read(arguments.max_bytes)
    except PermissionError as exc:
        raise ToolError(f"the operating system denied reading {target}: {exc.strerror}") from exc

    if raw[:4096].count(b"\x00"):
        raise ToolError(f"{target} looks like a binary file, not text")
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ToolError(f"{target} is not valid UTF-8 text (byte {exc.start})") from exc

    truncated = size > len(raw)
    note = f"\n[... truncated: showing {len(raw)} of {size} bytes]" if truncated else ""
    return ToolResult(
        text=text + note,
        detail={
            "path": str(target),
            "bytes_read": len(raw),
            "size_bytes": size,
            "truncated": truncated,
        },
    )


def find_largest_files(context: ToolContext, arguments: LargestFilesArgs) -> ToolResult:
    """The largest *files* under a folder, with their real sizes.

    Exists because a small model cannot sort a listing itself: measured on
    2026-10-30, ``qwen2.5:1.5b`` either listed folder names as "files" or
    invented sizes for them. Doing the measurement here means the model only has
    to choose a folder and relay numbers it was given.
    """
    root = _require_directory(context.sandbox.resolve(arguments.path))
    root_depth = len(root.parts)

    found: list[tuple[int, str]] = []  # (size, path relative to root)
    scanned = 0
    skipped_outside = 0
    truncated = False

    for current_root, dir_names, file_names in os.walk(root, followlinks=False):
        current = Path(current_root)
        # Level 1 is the given folder itself, so a depth budget of N includes
        # levels 1..N and stops descending after level N.
        if len(current.parts) - root_depth + 1 >= arguments.max_depth:
            dir_names[:] = []
        dir_names.sort()
        for name in sorted(file_names):
            scanned += 1
            if scanned > MAX_SCAN_ENTRIES:
                truncated = True
                break
            candidate = current / name
            try:
                resolved = context.sandbox.resolve(candidate)
            except PermissionDenied:
                skipped_outside += 1
                continue
            try:
                info = resolved.stat()
            except OSError:
                continue
            if not stat.S_ISREG(info.st_mode):
                continue
            found.append((info.st_size, str((current / name).relative_to(root))))
        if truncated:
            break

    found.sort(key=lambda item: (item[0], item[1]), reverse=True)
    top = found[: arguments.limit]

    lines = [f"{_human_size(size):>9}  {path}" for size, path in top]
    header = (
        f"{len(top)} largest file(s) under {root}, biggest first "
        f"(searched {min(scanned, MAX_SCAN_ENTRIES)} file(s))"
    )
    if not top:
        return ToolResult(
            text=header + ": no files found in this folder",
            detail={"path": str(root), "returned": 0, "scanned": scanned},
        )
    if truncated:
        header += f"; stopped after {MAX_SCAN_ENTRIES} files"
    return ToolResult(
        text="\n".join([header, *lines]),
        detail={
            "path": str(root),
            "returned": len(top),
            "scanned": scanned,
            "largest_bytes": top[0][0],
            "largest_path": top[0][1],
            "skipped_outside_allow_list": skipped_outside,
        },
    )


def search_files(context: ToolContext, arguments: SearchFilesArgs) -> ToolResult:
    root = _require_directory(context.sandbox.resolve(arguments.path))
    if not arguments.pattern.strip():
        raise ToolError("a glob pattern is required, for example '*.pdf'")

    matches: list[tuple[Path, int]] = []
    scanned = 0
    skipped_outside = 0
    denied = 0

    for current_root, dir_names, file_names in os.walk(root, followlinks=False):
        dir_names.sort()
        for name in sorted(file_names) + list(dir_names):
            scanned += 1
            if scanned > MAX_SCAN_ENTRIES:
                break
            if not fnmatch.fnmatch(name.lower(), arguments.pattern.lower()):
                continue
            candidate = Path(current_root) / name
            try:
                resolved = context.sandbox.resolve(candidate)
            except PermissionDenied:
                # A symlink pointing outside the allow-list: skip it, do not leak it.
                skipped_outside += 1
                continue
            try:
                size = resolved.stat().st_size
            except OSError:
                denied += 1
                continue
            matches.append((Path(current_root).relative_to(root) / name, size))
            if len(matches) >= arguments.max_results:
                break
        if scanned > MAX_SCAN_ENTRIES or len(matches) >= arguments.max_results:
            break

    lines = [f"{_human_size(size):>9}  {path}" for path, size in matches]
    header = f"{len(matches)} match(es) for {arguments.pattern!r} under {root}"
    if len(matches) >= arguments.max_results:
        header += " (stopped at the result limit)"
    return ToolResult(
        text="\n".join([header, *lines]) if lines else header + " (no matches)",
        detail={
            "path": str(root),
            "pattern": arguments.pattern,
            "matches": len(matches),
            "scanned": min(scanned, MAX_SCAN_ENTRIES),
            "skipped_outside_allow_list": skipped_outside,
            "unreadable": denied,
        },
    )
