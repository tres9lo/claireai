"""Background training: opt-in, daily, bounded, and gated before install."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from claireai import training


def test_the_curriculum_covers_every_domain_with_correct_numbers() -> None:
    curriculum = training.TrainingCurriculum(Path("/tmp/does-not-exist"))
    records = curriculum.examples()
    domains = {str(record["source"]).split(":")[-1] for record in records}

    assert set(training.DOMAINS) <= domains
    arithmetic = next(
        record for record in records if record["source"] == "claireai-curriculum:arithmetic"
    )
    assert "=" in arithmetic["messages"][1]["content"]
    assert len(records) >= 40


def test_building_writes_a_train_and_validation_split(tmp_path) -> None:
    summary = training.TrainingCurriculum(tmp_path).build()

    assert (tmp_path / "learning" / "curriculum" / "train.jsonl").is_file()
    assert (tmp_path / "learning" / "curriculum" / "validation.jsonl").is_file()
    assert summary["train"] + summary["validation"] == summary["total"]
    assert summary["total"] >= 40


def test_an_offline_tick_writes_no_candidate(tmp_path) -> None:
    worker = training.TrainingWorker(
        tmp_path, enabled=lambda: True, online_check=lambda: False
    )

    assert worker.tick() is False
    assert worker.status()["status"] == "offline"
    assert not (tmp_path / "models" / "candidates").exists()


def test_a_disabled_tick_does_nothing(tmp_path) -> None:
    worker = training.TrainingWorker(
        tmp_path,
        enabled=lambda: False,
        online_check=lambda: True,
        runner=lambda *_args: SimpleNamespace(returncode=0),
    )

    assert worker.tick() is False
    assert not worker.status().get("status")


def test_a_bad_steps_override_falls_back_instead_of_crashing(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("CLAIREAI_TRAINING_STEPS", "not-a-number")
    worker = training.TrainingWorker(tmp_path, enabled=lambda: False)

    assert worker.steps == training.DEFAULT_STEPS


def test_a_bounded_run_writes_a_candidate_and_asks_the_gate(tmp_path, monkeypatch) -> None:
    source = tmp_path / "repo" / "assistant"
    brain_lab = tmp_path / "repo" / "brain-lab"
    (source / "src" / "claireai").mkdir(parents=True)
    (source / "pyproject.toml").write_text("[project]\nname='claireai'\n", encoding="utf-8")
    brain_lab.mkdir()
    (brain_lab / "train_chat.py").write_text("print('train')\n", encoding="utf-8")
    monkeypatch.setattr("claireai.update.checkout_root", lambda: source)
    commands: list[list[str]] = []
    reviewed: list[Path] = []

    def gate(candidate):  # noqa: ANN001, ANN202
        reviewed.append(candidate)
        return {
            "passed": False,
            "promoted": False,
            "reason": "tools did not improve",
        }

    def runner(command, cwd, timeout):  # noqa: ANN001, ANN202
        commands.append(command)
        output = Path(command[command.index("--output") + 1])
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(b"candidate")
        return SimpleNamespace(returncode=0)

    worker = training.TrainingWorker(
        tmp_path / "data",
        enabled=lambda: True,
        online_check=lambda: True,
        steps=40,
        runner=runner,
        gate=gate,
    )

    assert worker.tick() is True
    assert worker.due() is False
    assert worker.status()["status"] == "candidate rejected"
    assert worker.status()["gate"]["reason"] == "tools did not improve"
    assert Path(worker.status()["candidate"]).is_file()
    assert reviewed == [Path(worker.status()["candidate"])]
    assert "--steps" in commands[0] and "40" in commands[0]
    assert "--init-from" not in commands[0]
    # A rejected candidate is not the installed checkpoint.
    assert not (tmp_path / "data" / "models" / "claire-scratch.pt").exists()


def test_a_gate_pass_is_reported_as_installed(tmp_path, monkeypatch) -> None:
    source = tmp_path / "repo" / "assistant"
    (source / "src" / "claireai").mkdir(parents=True)
    (source / "pyproject.toml").write_text("[project]\nname='claireai'\n", encoding="utf-8")
    (tmp_path / "repo" / "brain-lab").mkdir()
    (tmp_path / "repo" / "brain-lab" / "train_chat.py").write_text("pass\n", encoding="utf-8")
    monkeypatch.setattr("claireai.update.checkout_root", lambda: source)

    def runner(command, cwd, timeout):  # noqa: ANN001, ANN202
        output = Path(command[command.index("--output") + 1])
        output.write_bytes(b"candidate")
        return SimpleNamespace(returncode=0)

    worker = training.TrainingWorker(
        tmp_path / "data",
        enabled=lambda: True,
        online_check=lambda: True,
        steps=40,
        runner=runner,
        gate=lambda _candidate: {"passed": True, "promoted": True, "reason": "passed"},
    )

    assert worker.tick() is True
    assert worker.status()["status"] == "installed"
    assert worker.status()["gate"]["promoted"] is True


def test_a_gate_error_keeps_the_candidate_for_review(tmp_path, monkeypatch) -> None:
    source = tmp_path / "repo" / "assistant"
    (source / "src" / "claireai").mkdir(parents=True)
    (source / "pyproject.toml").write_text("[project]\nname='claireai'\n", encoding="utf-8")
    (tmp_path / "repo" / "brain-lab").mkdir()
    (tmp_path / "repo" / "brain-lab" / "train_chat.py").write_text("pass\n", encoding="utf-8")
    monkeypatch.setattr("claireai.update.checkout_root", lambda: source)

    def runner(command, cwd, timeout):  # noqa: ANN001, ANN202
        output = Path(command[command.index("--output") + 1])
        output.write_bytes(b"candidate")
        return SimpleNamespace(returncode=0)

    def broken_gate(_candidate):  # noqa: ANN001
        raise RuntimeError("no evaluation scripts")

    worker = training.TrainingWorker(
        tmp_path / "data",
        enabled=lambda: True,
        online_check=lambda: True,
        steps=40,
        runner=runner,
        gate=broken_gate,
    )

    assert worker.tick() is True
    assert worker.status()["status"] == "candidate needs review"
    assert "no evaluation scripts" in worker.status()["gate"]["reason"]
    assert Path(worker.status()["candidate"]).is_file()
