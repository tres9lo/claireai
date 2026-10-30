"""Behaviour of the four read-only file tools."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from claireai.permissions import Sandbox
from claireai.tools.base import ToolContext, ToolError
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


@pytest.fixture()
def root(tmp_path: Path) -> Path:
    folder = tmp_path / "Downloads"
    folder.mkdir()
    (folder / "small.txt").write_text("x" * 5, encoding="utf-8")
    (folder / "big.bin").write_bytes(b"y" * 5000)
    (folder / "notes.md").write_text("# Notes", encoding="utf-8")
    (folder / ".hidden").write_text("secret", encoding="utf-8")
    nested = folder / "sub"
    nested.mkdir()
    (nested / "inner.txt").write_text("inner", encoding="utf-8")
    return folder


@pytest.fixture()
def context(root: Path) -> ToolContext:
    return ToolContext(sandbox=Sandbox(roots=(root.resolve(),)))


def test_list_folder_by_name(context: ToolContext, root: Path) -> None:
    result = list_folder(context, ListFolderArgs(path=str(root)))

    assert result.detail["entries"] == 4  # hidden excluded
    assert "small.txt" in result.text
    assert "sub/" in result.text
    assert ".hidden" not in result.text
    assert result.text.splitlines()[0].startswith("4 entries")


def test_list_folder_can_include_hidden(context: ToolContext, root: Path) -> None:
    result = list_folder(context, ListFolderArgs(path=str(root), include_hidden=True))
    assert result.detail["entries"] == 5
    assert ".hidden" in result.text


def test_list_folder_by_size_descending_puts_biggest_first(context: ToolContext, root: Path) -> None:
    result = list_folder(
        context, ListFolderArgs(path=str(root), sort="size", descending=True, limit=2)
    )

    lines = result.text.splitlines()[1:]
    assert "big.bin" in lines[0]
    assert result.detail["returned"] == 2
    assert result.detail["entries"] == 4


def test_list_folder_limit_keeps_the_largest_with_sort_size(context: ToolContext, root: Path) -> None:
    everything = list_folder(context, ListFolderArgs(path=str(root), sort="size", descending=True))
    biggest_name = everything.text.splitlines()[1].split()[-1]
    limited = list_folder(
        context, ListFolderArgs(path=str(root), sort="size", descending=True, limit=1)
    )
    assert biggest_name in limited.text
    assert "showing 1" in limited.text.splitlines()[0]


def test_list_folder_rejects_a_file(context: ToolContext, root: Path) -> None:
    with pytest.raises(ToolError, match="is not a folder"):
        list_folder(context, ListFolderArgs(path=str(root / "small.txt")))


def test_list_folder_missing_path(context: ToolContext, root: Path) -> None:
    with pytest.raises(ToolError, match="does not exist"):
        list_folder(context, ListFolderArgs(path=str(root / "nope")))


def test_list_folder_outside_allow_list_is_refused(context: ToolContext, tmp_path: Path) -> None:
    from claireai.permissions import PermissionDenied

    with pytest.raises(PermissionDenied):
        list_folder(context, ListFolderArgs(path=str(tmp_path)))


def test_get_file_info_reports_type_and_size(context: ToolContext, root: Path) -> None:
    result = get_file_info(context, FileInfoArgs(path=str(root / "small.txt")))
    assert result.detail["type"] == "file"
    assert result.detail["size_bytes"] == 5
    assert result.detail["name"] == "small.txt"
    assert "modified_utc" in result.detail

    folder = get_file_info(context, FileInfoArgs(path=str(root)))
    assert folder.detail["type"] == "folder"


def test_get_file_info_missing_file(context: ToolContext, root: Path) -> None:
    with pytest.raises(ToolError, match="does not exist"):
        get_file_info(context, FileInfoArgs(path=str(root / "ghost.txt")))


def test_read_text_file_returns_contents(context: ToolContext, root: Path) -> None:
    result = read_text_file(context, ReadTextFileArgs(path=str(root / "notes.md")))
    assert result.text == "# Notes"
    assert result.detail["truncated"] is False


def test_read_text_file_truncates_to_budget(context: ToolContext, root: Path) -> None:
    result = read_text_file(context, ReadTextFileArgs(path=str(root / "big.bin"), max_bytes=10))
    assert result.detail["bytes_read"] == 10
    assert result.detail["truncated"] is True
    assert "truncated" in result.text


def test_read_text_file_refuses_binary(context: ToolContext, root: Path) -> None:
    (root / "blob.dat").write_bytes(b"abc\x00def")
    with pytest.raises(ToolError, match="binary"):
        read_text_file(context, ReadTextFileArgs(path=str(root / "blob.dat")))


def test_read_text_file_refuses_invalid_utf8(context: ToolContext, root: Path) -> None:
    (root / "latin1.txt").write_bytes(b"caf\xe9")
    with pytest.raises(ToolError, match="not valid UTF-8"):
        read_text_file(context, ReadTextFileArgs(path=str(root / "latin1.txt")))


def test_read_text_file_refuses_a_directory(context: ToolContext, root: Path) -> None:
    with pytest.raises(ToolError, match="not a regular file"):
        read_text_file(context, ReadTextFileArgs(path=str(root)))


def test_search_files_finds_nested_matches(context: ToolContext, root: Path) -> None:
    result = search_files(context, SearchFilesArgs(path=str(root), pattern="*.txt"))
    assert result.detail["matches"] == 2
    assert "small.txt" in result.text
    assert "inner.txt" in result.text
    assert "notes.md" not in result.text


def test_search_files_is_case_insensitive(context: ToolContext, root: Path) -> None:
    result = search_files(context, SearchFilesArgs(path=str(root), pattern="*.MD"))
    assert result.detail["matches"] == 1


def test_search_files_respects_max_results(context: ToolContext, root: Path) -> None:
    result = search_files(context, SearchFilesArgs(path=str(root), pattern="*", max_results=1))
    assert result.detail["matches"] == 1
    assert "result limit" in result.text


def test_search_files_skips_symlinks_leaving_the_allow_list(
    context: ToolContext, root: Path, tmp_path: Path
) -> None:
    outside = tmp_path / "outside-secret.txt"
    outside.write_text("secret", encoding="utf-8")
    os.symlink(outside, root / "linked-report.txt")

    result = search_files(context, SearchFilesArgs(path=str(root), pattern="*report*"))
    assert result.detail["matches"] == 0
    assert result.detail["skipped_outside_allow_list"] == 1
    assert "secret" not in result.text.lower()


def test_search_files_requires_a_pattern(context: ToolContext, root: Path) -> None:
    with pytest.raises(ToolError, match="glob pattern is required"):
        search_files(context, SearchFilesArgs(path=str(root), pattern="   "))


def test_search_files_rejects_a_file_as_root(context: ToolContext, root: Path) -> None:
    with pytest.raises(ToolError, match="is not a folder"):
        search_files(context, SearchFilesArgs(path=str(root / "small.txt"), pattern="*"))


def test_find_largest_files_sorts_by_real_size(context: ToolContext, root: Path) -> None:
    result = find_largest_files(context, LargestFilesArgs(path=str(root), limit=2))

    lines = result.text.splitlines()[1:]
    assert result.detail["largest_path"] == "big.bin"
    assert result.detail["largest_bytes"] == 5000
    assert len(lines) == 2
    assert lines[0].endswith("big.bin")
    assert result.detail["returned"] == 2


def test_find_largest_files_searches_subfolders(context: ToolContext, root: Path) -> None:
    result = find_largest_files(context, LargestFilesArgs(path=str(root), limit=10))
    assert "sub/inner.txt" in result.text  # nested file was found
    # Folders are not reported as files, whatever their apparent size.
    assert "sub/" not in [line.split()[-1] for line in result.text.splitlines()[1:]]


def test_find_largest_files_respects_max_depth(context: ToolContext, root: Path) -> None:
    shallow = find_largest_files(context, LargestFilesArgs(path=str(root), max_depth=1))
    assert "sub/inner.txt" not in shallow.text


def test_find_largest_files_reports_an_empty_folder(context: ToolContext, tmp_path: Path) -> None:
    empty = tmp_path / "Downloads" / "empty"
    empty.mkdir()
    context = ToolContext(sandbox=Sandbox(roots=((tmp_path / "Downloads").resolve(),)))
    result = find_largest_files(context, LargestFilesArgs(path=str(empty)))
    assert result.detail["returned"] == 0
    assert "no files found" in result.text


def test_find_largest_files_refuses_outside_the_allow_list(context: ToolContext, tmp_path: Path) -> None:
    from claireai.permissions import PermissionDenied

    with pytest.raises(PermissionDenied):
        find_largest_files(context, LargestFilesArgs(path=str(tmp_path)))
