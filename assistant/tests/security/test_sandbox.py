"""The allow-list is the only thing standing between the model and the disk.

These tests are the security boundary: if they pass, a model-supplied string
cannot reach a file outside the folders the owner named.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from claireai.config import Settings
from claireai.permissions import PermissionDenied, Sandbox, check_risk


@pytest.fixture()
def downloads(tmp_path: Path) -> Path:
    folder = tmp_path / "Downloads"
    folder.mkdir()
    (folder / "report.pdf").write_text("pdf", encoding="utf-8")
    (folder / "sub").mkdir()
    return folder


@pytest.fixture()
def sandbox(downloads: Path) -> Sandbox:
    return Sandbox(roots=(downloads.resolve(),))


def test_path_inside_allow_list_is_allowed(sandbox: Sandbox, downloads: Path) -> None:
    assert sandbox.resolve(str(downloads / "report.pdf")) == (downloads / "report.pdf").resolve()


def test_root_itself_is_allowed(sandbox: Sandbox, downloads: Path) -> None:
    assert sandbox.resolve(str(downloads)) == downloads.resolve()


def test_empty_allow_list_allows_nothing(tmp_path: Path) -> None:
    sandbox = Sandbox()
    assert sandbox.is_empty
    with pytest.raises(PermissionDenied) as excinfo:
        sandbox.resolve(str(tmp_path))
    assert "CLAIREAI_ALLOWED_FOLDERS" in str(excinfo.value)


def test_absolute_path_outside_allow_list_is_refused(sandbox: Sandbox, tmp_path: Path) -> None:
    secret = tmp_path / "outside.txt"
    secret.write_text("secret", encoding="utf-8")
    with pytest.raises(PermissionDenied) as excinfo:
        sandbox.resolve(str(secret))
    assert "outside the allow-listed folders" in str(excinfo.value)


def test_parent_traversal_is_refused(sandbox: Sandbox, downloads: Path) -> None:
    with pytest.raises(PermissionDenied):
        sandbox.resolve(str(downloads / ".." / "outside.txt"))


def test_symlink_escaping_the_allow_list_is_refused(
    sandbox: Sandbox, downloads: Path, tmp_path: Path, make_symlink
) -> None:
    secret = tmp_path / "secret.txt"
    secret.write_text("secret", encoding="utf-8")
    link = downloads / "shortcut.txt"
    make_symlink(link, secret)

    assert link.exists()  # the link resolves on disk...
    with pytest.raises(PermissionDenied):  # ...but the sandbox refuses it
        sandbox.resolve(str(link))


def test_directory_symlink_escape_is_refused(
    sandbox: Sandbox, downloads: Path, tmp_path: Path, make_symlink
) -> None:
    outside = tmp_path / "outside-dir"
    outside.mkdir()
    (outside / "leak.txt").write_text("leak", encoding="utf-8")
    make_symlink(downloads / "escape", outside)

    with pytest.raises(PermissionDenied):
        sandbox.resolve(str(downloads / "escape" / "leak.txt"))


def test_symlink_inside_allow_list_is_followed(sandbox: Sandbox, downloads: Path, make_symlink) -> None:
    make_symlink(downloads / "alias.pdf", downloads / "report.pdf")
    assert sandbox.resolve(str(downloads / "alias.pdf")) == (downloads / "report.pdf").resolve()


def test_relative_name_matching_a_root_is_accepted(sandbox: Sandbox, downloads: Path) -> None:
    assert sandbox.resolve("Downloads") == downloads.resolve()


def test_relative_path_below_a_root_is_accepted(sandbox: Sandbox, downloads: Path) -> None:
    assert sandbox.resolve("Downloads/report.pdf") == (downloads / "report.pdf").resolve()


def test_relative_path_matching_nothing_is_refused(sandbox: Sandbox) -> None:
    with pytest.raises(PermissionDenied):
        sandbox.resolve("no-such-folder/whatever.txt")


def test_blank_path_is_refused(sandbox: Sandbox) -> None:
    with pytest.raises(PermissionDenied):
        sandbox.resolve("   ")


def test_settings_expand_the_allow_list(downloads: Path) -> None:
    settings = Settings(_env_file=None, allowed_folders=[downloads])
    assert Sandbox.from_settings(settings).roots == (downloads.resolve(),)


def test_settings_reject_empty_allow_list_entries() -> None:
    with pytest.raises(ValueError, match="must not contain empty entries"):
        Settings(_env_file=None, allowed_folders=[" "])


def test_safe_tools_run_but_risky_ones_are_refused() -> None:
    check_risk("safe")
    with pytest.raises(PermissionDenied, match="no approval flow"):
        check_risk("needs_approval")
    with pytest.raises(PermissionDenied, match="no approval flow"):
        check_risk("dangerous")
    # Week 3 flips this flag once approvals exist; the policy is already written.
    check_risk("needs_approval", approval_available=True)


def test_unknown_risk_level_is_refused() -> None:
    with pytest.raises(PermissionDenied, match="unknown risk level"):
        check_risk("probably-fine")
