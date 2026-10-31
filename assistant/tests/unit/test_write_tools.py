"""The approval-gated mutating tools.

These call the handlers directly; the approval requirement itself is a
registry/risk property covered in `test_agent.py` and `test_tool_registry.py`.
What matters here: a write cannot leave the allow-list, nothing silently
overwrites, and nothing is ever deleted permanently.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from claireai.permissions import PermissionDenied, Sandbox
from claireai.tools.base import ToolContext, ToolError
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


@pytest.fixture()
def root(tmp_path: Path) -> Path:
    folder = tmp_path / "Downloads"
    folder.mkdir()
    (folder / "note.txt").write_text("original", encoding="utf-8")
    (folder / "sub").mkdir()
    return folder


@pytest.fixture()
def context(root: Path, tmp_path: Path) -> ToolContext:
    return ToolContext(
        sandbox=Sandbox(roots=(root.resolve(),)),
        data_dir=tmp_path / "data",
    )


def test_create_folder_makes_a_directory(context: ToolContext, root: Path) -> None:
    target = root / "reports"
    result = create_folder(context, CreateFolderArgs(path=str(target)))
    assert target.is_dir()
    assert result.detail["created"] is True


def test_create_folder_is_idempotent(context: ToolContext, root: Path) -> None:
    target = root / "reports"
    target.mkdir()
    result = create_folder(context, CreateFolderArgs(path=str(target)))
    assert result.detail["created"] is False
    assert "already exists" in result.text


def test_create_folder_refuses_when_a_file_has_the_name(context: ToolContext, root: Path) -> None:
    with pytest.raises(ToolError, match="not a folder"):
        create_folder(context, CreateFolderArgs(path=str(root / "note.txt")))


def test_create_folder_cannot_escape_the_allow_list(context: ToolContext, tmp_path: Path) -> None:
    with pytest.raises(PermissionDenied):
        create_folder(context, CreateFolderArgs(path=str(tmp_path / "outside")))


def test_create_text_file_writes_content(context: ToolContext, root: Path) -> None:
    target = root / "new.txt"
    result = create_text_file(context, CreateTextFileArgs(path=str(target), content="hello"))
    assert target.read_text(encoding="utf-8") == "hello"
    assert result.detail["bytes"] == 5


def test_create_text_file_refuses_to_clobber(context: ToolContext, root: Path) -> None:
    with pytest.raises(ToolError, match="already exists"):
        create_text_file(context, CreateTextFileArgs(path=str(root / "note.txt"), content="x"))
    assert (root / "note.txt").read_text(encoding="utf-8") == "original"


def test_create_text_file_can_overwrite_when_asked(context: ToolContext, root: Path) -> None:
    target = root / "note.txt"
    create_text_file(context, CreateTextFileArgs(path=str(target), content="new", overwrite=True))
    assert target.read_text(encoding="utf-8") == "new"


def test_create_text_file_needs_an_existing_parent(context: ToolContext, root: Path) -> None:
    with pytest.raises(ToolError, match="does not exist"):
        create_text_file(
            context, CreateTextFileArgs(path=str(root / "missing" / "a.txt"), content="x")
        )


def test_create_text_file_has_a_size_limit(context: ToolContext, root: Path) -> None:
    with pytest.raises(ToolError, match="limit"):
        create_text_file(
            context, CreateTextFileArgs(path=str(root / "big.txt"), content="x" * 200_001)
        )


def test_write_to_file_overwrites(context: ToolContext, root: Path) -> None:
    target = root / "note.txt"
    write_to_file(context, WriteToFileArgs(path=str(target), content="replaced"))
    assert target.read_text(encoding="utf-8") == "replaced"


def test_write_to_file_appends(context: ToolContext, root: Path) -> None:
    target = root / "note.txt"
    write_to_file(context, WriteToFileArgs(path=str(target), content=" +more", mode="append"))
    assert target.read_text(encoding="utf-8") == "original +more"


def test_write_to_file_requires_an_existing_file(context: ToolContext, root: Path) -> None:
    with pytest.raises(ToolError, match="does not exist"):
        write_to_file(context, WriteToFileArgs(path=str(root / "nope.txt"), content="x"))


def test_write_to_file_refuses_a_directory(context: ToolContext, root: Path) -> None:
    with pytest.raises(ToolError, match="not a regular file"):
        write_to_file(context, WriteToFileArgs(path=str(root / "sub"), content="x"))


def test_rename_file_stays_in_the_same_folder(context: ToolContext, root: Path) -> None:
    result = rename_file(context, RenameFileArgs(path=str(root / "note.txt"), new_name="renamed.txt"))
    assert not (root / "note.txt").exists()
    assert (root / "renamed.txt").read_text(encoding="utf-8") == "original"
    assert result.detail["destination"] == str(root / "renamed.txt")


def test_rename_file_refuses_paths_in_the_new_name(context: ToolContext, root: Path) -> None:
    for name in ["../escape.txt", "sub/moved.txt", "sub\\moved.txt", "", ".."]:
        with pytest.raises(ToolError, match="plain file name"):
            rename_file(context, RenameFileArgs(path=str(root / "note.txt"), new_name=name))


def test_rename_file_refuses_to_clobber(context: ToolContext, root: Path) -> None:
    (root / "taken.txt").write_text("keep", encoding="utf-8")
    with pytest.raises(ToolError, match="already exists"):
        rename_file(context, RenameFileArgs(path=str(root / "note.txt"), new_name="taken.txt"))


def test_rename_file_refuses_the_same_name(context: ToolContext, root: Path) -> None:
    with pytest.raises(ToolError, match="same as the current name"):
        rename_file(context, RenameFileArgs(path=str(root / "note.txt"), new_name="note.txt"))


def test_move_file_into_a_folder(context: ToolContext, root: Path) -> None:
    result = move_file(context, MoveFileArgs(source=str(root / "note.txt"), destination=str(root / "sub")))
    assert not (root / "note.txt").exists()
    assert (root / "sub" / "note.txt").read_text(encoding="utf-8") == "original"
    assert result.detail["destination"] == str(root / "sub" / "note.txt")


def test_move_file_to_a_full_path(context: ToolContext, root: Path) -> None:
    target = root / "sub" / "renamed.txt"
    move_file(context, MoveFileArgs(source=str(root / "note.txt"), destination=str(target)))
    assert target.exists()


def test_move_file_refuses_to_overwrite(context: ToolContext, root: Path) -> None:
    (root / "sub" / "note.txt").write_text("existing", encoding="utf-8")
    with pytest.raises(ToolError, match="already exists"):
        move_file(context, MoveFileArgs(source=str(root / "note.txt"), destination=str(root / "sub")))
    assert (root / "sub" / "note.txt").read_text(encoding="utf-8") == "existing"


def test_move_file_refuses_a_destination_outside_the_allow_list(
    context: ToolContext, root: Path, tmp_path: Path
) -> None:
    with pytest.raises(PermissionDenied):
        move_file(
            context,
            MoveFileArgs(source=str(root / "note.txt"), destination=str(tmp_path / "outside.txt")),
        )
    assert (root / "note.txt").exists()


def test_move_file_needs_an_existing_source(context: ToolContext, root: Path) -> None:
    with pytest.raises(ToolError, match="does not exist"):
        move_file(context, MoveFileArgs(source=str(root / "ghost.txt"), destination=str(root / "sub")))


def test_move_file_refuses_reversed_arguments(context: ToolContext, root: Path) -> None:
    # Observed for real with a thin model: {source: "sub",
    # destination: "report.pdf"}. Reversed, that would rename a whole folder to a
    # file-shaped path. The refusal must happen before approval, not after.
    with pytest.raises(ToolError, match="look reversed"):
        move_file(
            context,
            MoveFileArgs(source=str(root / "sub"), destination=str(root / "report.pdf")),
        )
    assert (root / "sub").is_dir()
    assert not (root / "report.pdf").exists()


def test_move_file_refuses_a_folder_onto_an_existing_file(
    context: ToolContext, root: Path
) -> None:
    target = root / "report.pdf"
    target.write_text("pdf", encoding="utf-8")

    with pytest.raises(ToolError, match="look reversed"):
        move_file(context, MoveFileArgs(source=str(root / "sub"), destination=str(target)))

    assert (root / "sub").is_dir()
    assert target.read_text(encoding="utf-8") == "pdf"


def test_the_reversal_guard_still_allows_a_real_folder_move(
    context: ToolContext, root: Path
) -> None:
    archive = root / "archive"
    archive.mkdir()
    move_file(context, MoveFileArgs(source=str(root / "sub"), destination=str(archive)))
    assert (archive / "sub").is_dir()
    assert not (root / "sub").exists()


def test_the_reversal_guard_still_allows_renaming_a_folder(
    context: ToolContext, root: Path
) -> None:
    # A folder moved to a new path that is not file-shaped is legitimate, even
    # though the destination does not exist yet.
    move_file(context, MoveFileArgs(source=str(root / "sub"), destination=str(root / "renamed")))
    assert (root / "renamed").is_dir()


def test_move_to_trash_keeps_the_file_recoverable(context: ToolContext, root: Path, tmp_path: Path) -> None:
    result = move_to_trash(context, MoveToTrashArgs(path=str(root / "note.txt")))

    assert not (root / "note.txt").exists()
    trash_path = Path(result.detail["trash"])
    assert trash_path.is_file()
    assert trash_path.read_text(encoding="utf-8") == "original"
    assert result.detail["permanent"] is False
    assert (tmp_path / "data" / "trash") in trash_path.parents


def test_move_to_trash_handles_a_name_collision(context: ToolContext, root: Path) -> None:
    first = move_to_trash(context, MoveToTrashArgs(path=str(root / "note.txt")))
    (root / "note.txt").write_text("second", encoding="utf-8")
    second = move_to_trash(context, MoveToTrashArgs(path=str(root / "note.txt")))

    assert first.detail["trash"] != second.detail["trash"]
    assert Path(second.detail["trash"]).read_text(encoding="utf-8") == "second"


def test_move_to_trash_needs_a_data_directory(root: Path) -> None:
    context = ToolContext(sandbox=Sandbox(roots=(root.resolve(),)), data_dir=None)
    with pytest.raises(ToolError, match="trash folder is unknown"):
        move_to_trash(context, MoveToTrashArgs(path=str(root / "note.txt")))


def test_move_to_trash_cannot_reach_outside_the_allow_list(context: ToolContext, tmp_path: Path) -> None:
    outside = tmp_path / "secret.txt"
    outside.write_text("private", encoding="utf-8")
    with pytest.raises(PermissionDenied):
        move_to_trash(context, MoveToTrashArgs(path=str(outside)))
    assert outside.exists()
