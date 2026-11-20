"""Only locally trained, from-scratch checkpoints are adopted."""

from __future__ import annotations

from pathlib import Path

import pytest

from claireai.config import Settings
from claireai.models import (
    ModelInstallError,
    candidate_directories,
    describe_checkpoint,
    ensure_installed_checkpoint,
    find_local_checkpoint,
)

torch = pytest.importorskip("torch")


def make_checkpoint(path: Path, *, name: str, tokenizer: str = "corpus-bpe-v1") -> Path:
    torch.save(
        {
            "manifest": {
                "name": name,
                "tokenizer": tokenizer,
                "architecture": {"context": 128, "width": 64, "layers": 2, "vocab_size": 512},
            },
            "state_dict": {"weight": torch.zeros(10, 4)},
        },
        path,
    )
    return path


def test_a_checkpoint_from_this_lab_is_adopted_into_an_empty_data_dir(tmp_path, monkeypatch) -> None:
    source = tmp_path / "checkout" / "runtime-data" / "models"
    source.mkdir(parents=True)
    make_checkpoint(source / "claire-scratch.pt", name="claire-chat-v4")
    make_checkpoint(source / "claire-chat-v9.pt", name="claire-chat-v9")
    monkeypatch.setenv("CLAIREAI_MODEL_DIRS", str(source))

    settings = Settings(_env_file=None, data_dir=tmp_path / "data")
    installed = ensure_installed_checkpoint(settings)

    assert installed == tmp_path / "data" / "models" / "claire-scratch.pt"
    assert installed.is_file()
    assert describe_checkpoint(installed)["name"] == "claire-chat-v4"
    # The source is a copy source, never moved or modified.
    assert (source / "claire-scratch.pt").is_file()


def test_an_installed_copy_finds_the_remembered_checkouts_runtime_data(tmp_path, monkeypatch) -> None:
    checkout = tmp_path / "repo" / "assistant"
    (checkout / "src" / "claireai").mkdir(parents=True)
    (checkout / "pyproject.toml").write_text("[project]\nname='claireai'\n", encoding="utf-8")
    monkeypatch.setattr("claireai.update.checkout_root", lambda: checkout)
    monkeypatch.delenv("CLAIREAI_MODEL_DIRS", raising=False)

    directories = candidate_directories(Settings(_env_file=None, data_dir=tmp_path / "data"))

    assert (checkout.parent / "runtime-data" / "models").resolve() in directories


def test_a_pretrained_checkpoint_is_refused(tmp_path, monkeypatch) -> None:
    models = tmp_path / "models"
    models.mkdir()
    foreign = make_checkpoint(models / "downloaded.pt", name="gpt2", tokenizer="gpt2")
    assert describe_checkpoint(foreign) is None
    with pytest.raises(ModelInstallError):
        from claireai.models import _load_payload

        _load_payload(foreign)

    # Isolate discovery from this repository's own runtime-data/models, which
    # would otherwise legitimately provide a checkpoint to adopt.
    monkeypatch.setattr("claireai.models.candidate_directories", lambda settings: [models])
    settings = Settings(_env_file=None, data_dir=tmp_path / "empty")
    assert ensure_installed_checkpoint(settings) is None


def test_the_newest_run_is_found_when_no_promoted_checkpoint_exists(tmp_path, monkeypatch) -> None:
    import os

    models = tmp_path / "models"
    models.mkdir()
    first = make_checkpoint(models / "claire-chat-v2.pt", name="claire-chat-v2")
    second = make_checkpoint(models / "claire-chat-v10.pt", name="claire-chat-v10")
    os.utime(first, (1_000, 1_000))
    os.utime(second, (2_000, 2_000))
    # A partial download must never be chosen.
    (models / "claire-chat-v11.pt.partial").write_bytes(b"not finished")

    settings = Settings(_env_file=None, data_dir=tmp_path / "data")
    monkeypatch.setattr("claireai.models.candidate_directories", lambda settings: [models])

    found = find_local_checkpoint(settings)
    assert found is not None
    assert found.name == "claire-chat-v10.pt"


def test_candidate_directories_include_the_checkout_and_an_override(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("CLAIREAI_MODEL_DIRS", str(tmp_path / "extra"))
    settings = Settings(_env_file=None, data_dir=tmp_path / "data")

    directories = [str(path) for path in candidate_directories(settings)]

    assert str(tmp_path / "extra") in directories
    assert any(path.endswith("runtime-data/models") for path in directories)
