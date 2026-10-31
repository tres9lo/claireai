"""The presence state stream the renderer consumes."""

from __future__ import annotations

import json

import pytest

from claireai.presence import ACTIVE_KEYS, STATES, PresenceStream, idle_event, sse_frame


def test_the_stream_starts_idle_and_inactive() -> None:
    event = PresenceStream().snapshot()
    assert event.state == "idle"
    assert event.seq == 0
    assert event.task_id is None
    assert set(event.active) == set(ACTIVE_KEYS)
    assert all(value is False for value in event.active.values())


def test_the_sequence_only_grows() -> None:
    stream = PresenceStream()
    seqs = [stream.emit("thinking").seq, stream.emit("tool_running").seq, stream.emit("idle").seq]
    assert seqs == [1, 2, 3]


def test_the_task_id_is_carried_until_it_is_replaced() -> None:
    stream = PresenceStream()
    stream.emit("thinking", task_id="abc")
    assert stream.emit("tool_running").task_id == "abc"
    assert stream.emit("idle", task_id=None).task_id == "abc"
    assert stream.emit("idle", task_id="other").task_id == "other"


def test_activity_flags_merge_and_persist() -> None:
    stream = PresenceStream()
    stream.emit("listening", active={"microphone": True})
    event = stream.emit("thinking", active={"camera": True})
    assert event.active["microphone"] is True
    assert event.active["camera"] is True
    assert event.active["screen"] is False


def test_an_unknown_state_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown presence state"):
        PresenceStream().emit("dancing")


def test_an_unknown_activity_flag_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown activity flag"):
        PresenceStream().emit("thinking", active={"telepathy": True})


def test_every_documented_state_is_accepted() -> None:
    stream = PresenceStream()
    for state in STATES:
        assert stream.emit(state).state == state


def test_the_wire_shape_is_camel_case_json() -> None:
    event = PresenceStream().emit("tool_running", task_id="t1", detail={"tool": "list_folder"})
    payload = event.as_dict()
    assert payload["taskId"] == "t1"
    assert payload["state"] == "tool_running"
    assert payload["detail"] == {"tool": "list_folder"}
    assert set(payload) == {"seq", "ts", "taskId", "state", "detail", "active"}


def test_the_sse_frame_is_a_valid_event_stream_message() -> None:
    frame = sse_frame(PresenceStream().emit("thinking"))
    assert frame.startswith("event: presence\ndata: ")
    assert frame.endswith("\n\n")
    payload = json.loads(frame.split("data: ", 1)[1].strip())
    assert payload["state"] == "thinking"


def test_reset_clears_activity_and_bumps_the_sequence() -> None:
    stream = PresenceStream()
    stream.emit("listening", task_id="t", active={"microphone": True})
    event = stream.reset()
    assert event.state == "idle"
    assert all(value is False for value in event.active.values())
    assert event.seq == 2


def test_the_reporter_sees_every_change() -> None:
    seen: list[str] = []
    stream = PresenceStream(reporter=lambda event: seen.append(event.state))
    stream.emit("thinking")
    stream.emit("idle")
    assert seen == ["thinking", "idle"]


def test_a_snapshot_is_immutable() -> None:
    event = idle_event()
    with pytest.raises(Exception):
        event.state = "thinking"  # type: ignore[misc]
