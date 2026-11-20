"""Self-description and conversational replies come from the runtime, not a guess."""

from __future__ import annotations

from pathlib import Path

from claireai.config import Settings
from claireai.identity import answer_self_question


def _settings(tmp_path: Path) -> Settings:
    return Settings(_env_file=None, data_dir=tmp_path / "data")


def test_how_are_you_is_answered_without_claiming_feelings(tmp_path) -> None:
    settings = _settings(tmp_path)
    (settings.resolved_data_dir / "models").mkdir(parents=True)
    (settings.resolved_data_dir / "models" / "claire-scratch.pt").write_bytes(b"x")
    answer = answer_self_question("How are you?", settings)

    assert answer is not None
    assert "do not have feelings" in answer
    assert "running well" in answer
    assert "How are you?" in answer


def test_the_origin_answer_says_from_scratch_and_no_pretrained_weights(tmp_path) -> None:
    answer = answer_self_question("Who made you?", _settings(tmp_path))

    assert answer is not None
    assert "from scratch" in answer
    assert "no pretrained weights" in answer


def test_capabilities_are_derived_from_the_real_configuration(tmp_path) -> None:
    settings = Settings(
        _env_file=None, data_dir=tmp_path / "data", allowed_folders=[tmp_path / "docs"]
    )
    answer = answer_self_question("What can you do?", settings)

    assert answer is not None
    assert "no folders are allowed" not in answer
    assert str(tmp_path / "docs") in answer
    assert "Web: browsing and downloads are off." in answer


def test_without_a_checkpoint_the_capability_list_says_so(tmp_path) -> None:
    answer = answer_self_question("what can you do", _settings(tmp_path))

    assert answer is not None
    assert "No local checkpoint is installed yet" in answer


def test_a_normal_question_is_not_hijacked(tmp_path) -> None:
    assert answer_self_question("What is a tokenizer?", _settings(tmp_path)) is None
    assert answer_self_question("list the files in Downloads", _settings(tmp_path)) is None
