"""Background training: opt-in, daily, bounded, and gated before install."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import torch

from claire_brain.prompt import render_prompt
from claire_brain.tokenizer import EOS, PAD, BytePairTokenizer
from claireai import training

TRAIN_CHAT = Path(__file__).resolve().parents[3] / "brain-lab" / "train_chat.py"


def _train_chat_module():
    spec = importlib.util.spec_from_file_location("train_chat_under_test", TRAIN_CHAT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_a_decoded_batch_keeps_the_prompt_masked_and_the_reply_learned() -> None:
    """Inspect the exact tensors the trainer feeds the model.

    This catches the boundary bugs that loss curves hide: a shifted target, a
    tool result leaking into the loss, a missing EOS, or padded labels counted as
    real targets.
    """
    module = _train_chat_module()
    tokenizer = BytePairTokenizer.train(
        [
            "You are ClaireAI. USER: list my downloads. "
            "ASSISTANT: <tool_call>{\"name\":\"list_folder\",\"arguments\":{}}"
            "</tool_call> TOOL list_folder: secret-tool-result-123 ASSISTANT: Two files."
        ],
        vocab_size=300,
    )
    tool_result = "secret-tool-result-123"
    messages = [
        {"role": "user", "content": "list my downloads"},
        {
            "role": "assistant",
            "content": (
                '<tool_call>{"name":"list_folder","arguments":{"path":"~/Downloads"}}'
                "</tool_call>"
            ),
        },
        {"role": "tool", "tool_name": "list_folder", "content": tool_result},
        {"role": "assistant", "content": "Two files."},
    ]
    record = {"system": "You are ClaireAI.", "messages": messages}
    examples = module.build_examples([record], tokenizer, context=512, max_target_tokens=160)
    assert len(examples) == 2
    assistant_indexes = [
        index for index, message in enumerate(messages) if message["role"] == "assistant"
    ]

    for (ids, prompt_len), assistant_index in zip(examples, assistant_indexes):
        ids_list = ids.tolist()
        prompt = tokenizer.decode(ids_list[:prompt_len])
        target = ids_list[prompt_len:]
        target_text = tokenizer.decode(target)

        # The prompt is byte-for-byte the renderer inference uses.
        assert prompt == render_prompt(record["system"], messages[:assistant_index])
        assert target_text == messages[assistant_index]["content"]
        assert target[-1] == EOS
        assert PAD not in target

        x, y, mask = module.collate([(ids, prompt_len)], pad=PAD)
        assert mask.shape == (1, len(ids_list) - 1)
        assert not mask[0, : prompt_len - 1].any()
        assert mask[0, prompt_len - 1 :].all()
        assert torch.equal(
            y[0, prompt_len - 1 :],
            torch.tensor(target, dtype=torch.long),
        )
        # The tool result is context, never a target token.
        if assistant_index == 3:
            assert tool_result in prompt
            assert tool_result not in target_text

    first_ids, first_prompt_len = examples[0]
    first_target = first_ids[first_prompt_len:].tolist()
    assert "<tool_call>" in tokenizer.decode(first_target)

    # Padding must not teach the model to predict pad ids.
    x, y, mask = module.collate(examples, pad=PAD)
    first_length = len(examples[0][0]) - 1
    assert first_length < x.shape[1]
    assert (x[0, first_length:] == PAD).all()
    assert (y[0, first_length:] == 0).all()
    assert (mask[0, first_length:] == 0).all()
    assert mask[0].sum().item() == len(examples[0][0]) - examples[0][1]
    assert mask[1].sum().item() == len(examples[1][0]) - examples[1][1]


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


def test_only_confirmed_lessons_enter_the_curriculum(tmp_path) -> None:
    from claireai.learning import LearningJournal

    journal = LearningJournal(tmp_path)
    journal.record("an unverified guess", "a model answer")
    journal.record_lesson(
        "a confirmed fact",
        "a sourced answer",
        sources=["https://example.test/fact"],
    )
    records = training.TrainingCurriculum(tmp_path, journal=journal).examples()
    questions = [record["messages"][0]["content"] for record in records]
    assert "a confirmed fact" in questions
    assert "an unverified guess" not in questions


def test_curriculum_replays_older_confirmed_lessons(tmp_path) -> None:
    from claireai.learning import LearningJournal

    journal = LearningJournal(tmp_path)
    for index in range(270):
        journal.record_lesson(
            f"older fact {index}",
            f"answer {index}",
            sources=[f"https://example.test/{index}"],
        )
    assert journal.compile() == 270

    records = training.TrainingCurriculum(tmp_path, journal=journal).examples()
    questions = [record["messages"][0]["content"] for record in records]
    assert "older fact 0" in questions
    assert "older fact 269" in questions
    assert len(questions) == len(set(questions))


def test_merge_keeps_authored_intent_and_confirmed_lessons(tmp_path) -> None:
    authored = tmp_path / "authored"
    external = tmp_path / "external"
    authored.mkdir()
    external.mkdir()
    record = lambda prompt, reply: {  # noqa: E731
        "messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": reply},
        ]
    }
    (authored / "train.jsonl").write_text(
        json.dumps(record("an intent example", "clarify")) + "\n", encoding="utf-8"
    )
    (authored / "validation.jsonl").write_text(
        json.dumps(record("a held out intent", "chat")) + "\n", encoding="utf-8"
    )
    (external / "train.jsonl").write_text(
        json.dumps(record("external", "answer")) + "\n", encoding="utf-8"
    )
    (external / "validation.jsonl").write_text(
        json.dumps(record("external held out", "answer")) + "\n", encoding="utf-8"
    )

    output = training.merge_training_corpus(authored, external, tmp_path / "merged")
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["train"] == 2
    assert manifest["validation"] == 2
    assert manifest["authored_train"] == 1
    train = (output / "train.jsonl").read_text(encoding="utf-8")
    assert "an intent example" in train and "external" in train


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


def test_a_pending_review_is_retried_without_retraining(tmp_path) -> None:
    data = tmp_path / "data"
    candidate = data / "models" / "candidates" / "candidate.pt"
    candidate.parent.mkdir(parents=True)
    candidate.write_bytes(b"candidate")
    (data / "learning").mkdir(parents=True)
    (data / "learning" / "training.json").write_text(
        json.dumps(
            {
                "status": "candidate needs review",
                "candidate": str(candidate),
                "last_attempt": "2026-11-20T00:00:00+00:00",
                "records": 58,
            }
        ),
        encoding="utf-8",
    )
    reviewed: list[Path] = []

    def training_must_not_run(*_args):  # noqa: ANN002, ANN202
        raise AssertionError("a pending review must not start another training run")

    def gate(path):  # noqa: ANN001, ANN202
        reviewed.append(path)
        return {"passed": False, "promoted": False, "reason": "tools did not improve"}

    worker = training.TrainingWorker(
        data,
        enabled=lambda: True,
        online_check=lambda: False,
        runner=training_must_not_run,
        gate=gate,
    )

    assert worker.tick() is True
    assert reviewed == [candidate]
    assert worker.status()["status"] == "candidate rejected"


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
