from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from claireai.config import Settings


def test_defaults_to_local_scratch_checkpoint() -> None:
    settings = Settings(_env_file=None)
    assert settings.model_checkpoint is None
    assert settings.resolved_model_checkpoint == settings.resolved_data_dir / "models" / "claire-scratch.pt"


def test_checkpoint_can_be_overridden(tmp_path: Path) -> None:
    checkpoint = tmp_path / "custom.pt"
    settings = Settings(_env_file=None, model_checkpoint=checkpoint)
    assert settings.resolved_model_checkpoint == checkpoint


def test_temperature_defaults_are_per_phase() -> None:
    settings = Settings(_env_file=None)
    assert settings.temperature == 0.8
    assert settings.temperature_tools == 0.2
    assert settings.tool_temperature == settings.temperature_tools


@pytest.mark.parametrize("value", [-0.1, 2.1])
def test_temperature_outside_range_is_rejected(value: float) -> None:
    with pytest.raises(ValidationError):
        Settings(_env_file=None, temperature=value)


def test_output_token_limit_is_bounded() -> None:
    with pytest.raises(ValidationError):
        Settings(_env_file=None, max_output_tokens=2048)


def test_data_dir_override_controls_paths(tmp_path: Path) -> None:
    settings = Settings(_env_file=None, data_dir=tmp_path / "runtime")
    assert settings.database_path == tmp_path / "runtime" / "db" / "claireai.sqlite3"
    assert settings.log_dir == tmp_path / "runtime" / "logs"
    assert settings.control_state_path == tmp_path / "runtime" / "runtime-control.json"


def test_unknown_capabilities_are_rejected() -> None:
    with pytest.raises(ValidationError, match="unknown capability"):
        Settings(_env_file=None, allowed_capabilities=["teleport"])


def test_unknown_setting_is_rejected() -> None:
    with pytest.raises(ValidationError):
        Settings(_env_file=None, unknown_setting=True)  # type: ignore[call-arg]
