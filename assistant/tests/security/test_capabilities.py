"""Configuration and policy for the planned presence renderer and machine access.

The tools do not exist yet. These tests pin the deny-by-default contract now, so
the first camera or cursor tool has to ask this policy instead of inventing one.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from claireai.config import Settings
from claireai.permissions import PermissionDenied, check_capability


def _settings(**overrides) -> Settings:  # noqa: ANN003
    return Settings(_env_file=None, **overrides)


def test_machine_access_is_denied_by_default() -> None:
    settings = _settings()
    assert settings.allowed_capabilities == []
    with pytest.raises(PermissionDenied, match="not enabled"):
        check_capability("screen", settings.allowed_capabilities)
    with pytest.raises(PermissionDenied, match="not enabled"):
        check_capability("camera", settings.allowed_capabilities)


def test_a_listed_capability_is_allowed() -> None:
    settings = _settings(allowed_capabilities=["screen"])
    check_capability("screen", settings.allowed_capabilities)


def test_an_unlisted_capability_is_still_refused() -> None:
    settings = _settings(allowed_capabilities=["screen"])
    with pytest.raises(PermissionDenied, match="not enabled"):
        check_capability("cursor", settings.allowed_capabilities)


def test_unknown_capability_names_are_refused() -> None:
    with pytest.raises(PermissionDenied, match="unknown capability"):
        check_capability("teleport", [])


def test_settings_reject_unknown_capabilities() -> None:
    with pytest.raises(ValidationError, match="unknown capability"):
        _settings(allowed_capabilities=["screen", "telekinesis"])


def test_capabilities_are_normalised_and_deduplicated() -> None:
    settings = _settings(allowed_capabilities=["Screen ", "screen", "CAMERA"])
    assert settings.allowed_capabilities == ["screen", "camera"]


def test_presence_defaults_are_cheap_and_safe() -> None:
    settings = _settings()
    assert settings.presence_mode == "auto"
    assert settings.presence_fps_cap == 30
    assert settings.presence_reduce_motion is False
    assert settings.camera_device == 0


def test_presence_mode_must_be_one_of_the_supported_values() -> None:
    assert _settings(presence_mode="off").presence_mode == "off"
    with pytest.raises(ValidationError):
        _settings(presence_mode="hologram")


@pytest.mark.parametrize("fps", [4, 61])
def test_presence_frame_cap_is_bounded(fps: int) -> None:
    with pytest.raises(ValidationError):
        _settings(presence_fps_cap=fps)


def test_camera_device_must_be_a_non_negative_index() -> None:
    with pytest.raises(ValidationError):
        _settings(camera_device=-1)
