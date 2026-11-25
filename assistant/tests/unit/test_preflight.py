"""Preflight: detect first, list every download, and never download silently."""

from __future__ import annotations

from pathlib import Path

from claireai import preflight


def test_the_plan_names_each_component_and_its_size() -> None:
    text = preflight.format_plan(
        [
            preflight.Requirement(
                key="torch", label="local model runtime", module="torch",
                spec="torch>=2.2,<3", download="about 200 MB",
            ),
            preflight.Requirement(
                key="faster_whisper", label="voice input", module="faster_whisper",
                spec="faster-whisper>=1.1,<2", download="about 90 MB",
            ),
        ]
    )

    assert "local model runtime" in text
    assert "about 200 MB" in text
    assert "voice input" in text
    assert "about 90 MB" in text


def test_a_non_terminal_never_blocks_or_downloads(monkeypatch, capsys) -> None:
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)
    requirements = [preflight.REQUIREMENTS[0]]

    assert preflight.ask_to_install(requirements) is False
    out = capsys.readouterr().out
    assert "Nothing was downloaded" in out


def test_only_an_explicit_yes_accepts_the_plan(monkeypatch) -> None:
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    requirements = [preflight.REQUIREMENTS[0]]
    prompts: list[str] = []

    assert preflight.ask_to_install(
        requirements, input_fn=lambda prompt: prompts.append(prompt) or "no"
    ) is False
    assert preflight.ask_to_install(requirements, input_fn=lambda prompt: "yes") is True


def test_install_passes_the_requirement_and_the_cpu_index_to_pip() -> None:
    seen: list[list[str]] = []
    requirement = preflight.Requirement(
        key="torch", label="local model runtime", module="torch",
        spec="torch>=2.2,<3", download="about 200 MB",
        index_url="https://download.pytorch.org/whl/cpu",
    )

    failed = preflight.install([requirement], runner=lambda command: seen.append(list(command)) or 0)

    assert failed == []
    assert "--index-url" in seen[0]
    assert "https://download.pytorch.org/whl/cpu" in seen[0]
    assert seen[0][-1] == "torch>=2.2,<3"


def test_the_speech_model_is_listed_and_installed_like_anything_else(tmp_path, monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(
        preflight, "install_speech_model",
        lambda data_dir, **_kwargs: calls.append(Path(data_dir)) or True,
    )
    requirement = preflight.speech_requirement()

    failed = preflight.install([requirement], data_dir=tmp_path)

    assert failed == []
    assert calls == [tmp_path]
    assert requirement.download == preflight.SPEECH_MODEL_DOWNLOAD


def test_a_failed_install_is_reported_not_hidden() -> None:
    requirement = preflight.Requirement(
        key="customtkinter", label="desktop window", module="customtkinter",
        spec="customtkinter>=5.2,<7", download="about 2 MB",
    )
    failed = preflight.install([requirement], runner=lambda command: 1)

    assert failed == [requirement]


def test_cpu_wheels_are_selected_on_x86_linux_and_windows() -> None:
    linux = preflight.Environment(
        family="linux", system="Linux", release="6", machine="x86_64", python="3.12",
        python_ok=True, display=":0", tkinter=True, audio="none",
    )
    arm = preflight.Environment(
        family="linux", system="Linux", release="6", machine="aarch64", python="3.12",
        python_ok=True, display=":0", tkinter=True, audio="none",
    )

    assert preflight.cpu_index_url(linux).endswith("/whl/cpu")
    assert preflight.cpu_index_url(arm) == ""
