"""Updating an installed copy is a rebuild from the checkout, not a download."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from claireai import update


def test_this_checkout_is_found_from_the_package() -> None:
    root = update.checkout_root()

    assert root is not None
    assert (root / "pyproject.toml").is_file()
    assert (root / "src" / "claireai").is_dir()


def test_the_source_override_wins(tmp_path, monkeypatch) -> None:
    fake = tmp_path / "assistant"
    (fake / "src" / "claireai").mkdir(parents=True)
    (fake / "pyproject.toml").write_text("[project]\nname='claireai'\n", encoding="utf-8")
    monkeypatch.setenv("CLAIREAI_SOURCE", str(fake))

    assert update.checkout_root() == fake.resolve()


def test_without_uv_the_commands_use_this_interpreter(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(update.shutil, "which", lambda name: None)

    build = update.build_command(tmp_path, tmp_path / "out")
    install = update.install_command(tmp_path / "claireai-0.1.0-py3-none-any.whl")

    assert build[:3] == [sys.executable, "-m", "pip"]
    assert install[:4] == [sys.executable, "-m", "pip", "install"]
    assert "--force-reinstall" in install and "--no-deps" in install


def test_with_uv_it_builds_and_reinstalls_into_this_python(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(update.shutil, "which", lambda name: "/usr/bin/uv")

    build = update.build_command(tmp_path, tmp_path / "out")
    install = update.install_command(tmp_path / "claireai-0.1.0-py3-none-any.whl")

    assert build[:3] == ["/usr/bin/uv", "build", "--wheel"]
    assert install[:3] == ["/usr/bin/uv", "pip", "install"]
    assert "--python" in install and sys.executable in install


def test_the_desktop_extra_includes_the_model_and_voice_runtimes() -> None:
    root = update.checkout_root()
    requirements = update.desktop_requirements(root)

    assert any(item.startswith("customtkinter") for item in requirements)
    assert any(item.startswith("torch") for item in requirements)
    assert any(item.startswith("faster-whisper") for item in requirements)


def test_a_remembered_checkout_is_found_from_another_directory(tmp_path, monkeypatch) -> None:
    fake = tmp_path / "assistant"
    (fake / "src" / "claireai").mkdir(parents=True)
    (fake / "pyproject.toml").write_text("[project]\nname='claireai'\n", encoding="utf-8")
    data = tmp_path / "data"
    data.mkdir()
    (data / update.SOURCE_MARKER).write_text(str(fake), encoding="utf-8")
    monkeypatch.delenv("CLAIREAI_SOURCE", raising=False)
    monkeypatch.chdir(tmp_path)

    assert update.checkout_root(data_dir=data) == fake.resolve()


def test_run_update_accepts_the_repository_root(tmp_path, monkeypatch) -> None:
    root = tmp_path / "repo"
    (root / "assistant" / "src" / "claireai").mkdir(parents=True)
    (root / "assistant" / "pyproject.toml").write_text("[project]\nname='claireai'\n", encoding="utf-8")
    calls: list[list[str]] = []

    def fake_run(command, cwd):  # noqa: ANN001, ANN202
        calls.append(list(command))
        if "build" in command or "wheel" in command:
            flag = "--out-dir" if "--out-dir" in command else "--wheel-dir"
            output = Path(command[command.index(flag) + 1])
            output.mkdir(parents=True, exist_ok=True)
            (output / "claireai-0.1.0-py3-none-any.whl").write_bytes(b"wheel")
        return ""

    monkeypatch.setattr(update, "_run", fake_run)
    monkeypatch.setattr(update, "desktop_requirements", lambda _root: [])
    monkeypatch.setattr(update, "remember_source", lambda _root: None)

    result = update.run_update(source=root)

    assert result["source"] == str((root / "assistant").resolve())
    assert calls


def test_run_update_reconciles_the_extra_after_replacing_the_code(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("CLAIREAI_DATA_DIR", str(tmp_path / "data"))
    root = tmp_path / "assistant"
    (root / "src" / "claireai").mkdir(parents=True)
    (root / "pyproject.toml").write_text(
        "[project]\nname='claireai'\n[project.optional-dependencies]\n"
        "desktop=['customtkinter>=5.2,<7', 'faster-whisper>=1.1,<2']\n",
        encoding="utf-8",
    )
    calls: list[list[str]] = []

    def fake_run(command, cwd):
        calls.append(list(command))
        if "build" in command:
            output = Path(command[command.index("--out-dir") + 1])
            (output / "claireai-0.1.0-py3-none-any.whl").write_bytes(b"wheel")
        return "ok"

    monkeypatch.setattr(update, "_run", fake_run)
    result = update.run_update(source=root)

    assert result["wheel"] == "claireai-0.1.0-py3-none-any.whl"
    assert result["extras"] == "desktop"
    assert result["warning"] == ""
    assert any("faster-whisper>=1.1,<2" in command for command in calls)
    assert (tmp_path / "data" / update.SOURCE_MARKER).read_text(encoding="utf-8") == str(root)


def test_no_checkout_is_an_explanation_not_a_traceback(tmp_path) -> None:
    with pytest.raises(update.UpdateError):
        update.run_update(source=tmp_path / "not-a-checkout")
