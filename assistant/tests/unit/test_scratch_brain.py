"""Owner stop checks during scratch-model autoregressive generation."""

from __future__ import annotations

from pathlib import Path

import pytest
from types import SimpleNamespace

from claireai.brain import BrainStopped, Message, ScratchBrain
from claireai.runtime_control import RuntimeControl


class TokenBatch:
    def __init__(self, tokens: list[list[int]]) -> None:
        self.tokens = tokens

    def tolist(self) -> list[list[int]]:
        return self.tokens

    def __getitem__(self, index: int) -> TokenRow:
        return TokenRow(self.tokens[index])


class TokenRow:
    def __init__(self, tokens: list[int]) -> None:
        self.tokens = tokens

    def tolist(self) -> list[int]:
        return self.tokens


class FakeTokenizer:
    def encode(self, text: str, *, append_eos: bool) -> list[int]:  # noqa: ARG002
        return [1, 2]

    def decode(self, tokens: list[int]) -> str:  # noqa: ARG002
        return "partial"


class FakeModel:
    context = 32

    def __init__(self, controls: RuntimeControl) -> None:
        self.controls = controls
        self.checked_after_stop = False

    def generate(self, input_ids: TokenBatch, *, should_stop, **kwargs) -> TokenBatch:  # noqa: ANN001
        assert should_stop() is False
        self.controls.update({"shutdown": True})
        self.checked_after_stop = should_stop() is True
        return input_ids


@pytest.mark.asyncio
async def test_stop_during_generation_discards_partial_answer(tmp_path: Path) -> None:
    control_path = tmp_path / "runtime-control.json"
    controls = RuntimeControl(control_path)
    model = FakeModel(controls)
    brain = ScratchBrain(checkpoint=tmp_path / "not-loaded.pt", control_path=control_path)
    brain._torch = SimpleNamespace(
        tensor=lambda tokens, **kwargs: TokenBatch(tokens), long=int
    )
    brain._model = model
    brain._tokenizer = FakeTokenizer()
    brain._manifest = {"name": "test"}

    with pytest.raises(BrainStopped, match="stopped from the local control panel"):
        await brain.generate([Message(role="user", content="hello")])

    assert model.checked_after_stop is True
