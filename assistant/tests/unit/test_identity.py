"""Self-description and conversational replies come from the runtime, not a guess."""

from __future__ import annotations

from pathlib import Path

from claireai.config import Settings
from claireai.identity import answer_self_question


def _settings(tmp_path: Path) -> Settings:
    return Settings(_env_file=None, data_dir=tmp_path / "data")


def test_how_are_you_is_answered_without_claiming_feelings(tmp_path) -> None:
    settings = _settings(tmp_path)
    answer = answer_self_question("How are you?", settings)

    assert answer is not None
    assert "do not have feelings" in answer
    assert "running well" in answer
    assert "How are you?" in answer


def test_the_origin_answer_names_ollama_and_pretrained_qwen(tmp_path) -> None:
    answer = answer_self_question("Who made you?", _settings(tmp_path))

    assert answer is not None
    assert "Ollama" in answer
    assert "qwen3:1.7b" in answer
    assert "do not" in answer and "from scratch" in answer


def test_capabilities_are_derived_from_the_real_configuration(tmp_path) -> None:
    settings = Settings(
        _env_file=None, data_dir=tmp_path / "data", allowed_folders=[tmp_path / "docs"]
    )
    answer = answer_self_question("What can you do?", settings)

    assert answer is not None
    assert "no folders are allowed" not in answer
    assert str(tmp_path / "docs") in answer
    assert "Web: browsing and downloads are off." in answer


def test_the_capability_list_names_the_local_ollama_models(tmp_path) -> None:
    answer = answer_self_question("what can you do", _settings(tmp_path))

    assert answer is not None
    assert "qwen3:1.7b" in answer
    assert "qwen2.5-coder:1.5b" in answer
    assert "stay on this machine" in answer


def test_a_normal_question_is_not_hijacked(tmp_path) -> None:
    assert answer_self_question("What is a tokenizer?", _settings(tmp_path)) is None
    assert answer_self_question("list the files in Downloads", _settings(tmp_path)) is None
