"""The daily worker is a user timer, not a hidden root service."""

from __future__ import annotations

from types import SimpleNamespace

from claireai import service


def test_units_run_the_headless_entry_and_are_persistent(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    text = service.render_service(
        python="/opt/My AI/python",
        data_dir=tmp_path / "data dir",
        working_directory=tmp_path / "repo",
    )
    timer = service.render_timer()

    assert '"/opt/My AI/python"' in text
    assert "--training-run" in text
    assert '"' + str(tmp_path / "data dir") + '"' in text
    assert f"WorkingDirectory={tmp_path / 'repo'}" in text
    assert "Type=oneshot" in text
    assert "NoNewPrivileges=true" in text
    assert "OnCalendar=*-*-* 03:17:00" in timer
    assert "Persistent=true" in timer
    assert f"Unit={service.SERVICE_NAME}" in timer


def test_install_writes_units_and_enables_the_timer(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    calls: list[list[str]] = []

    def runner(args):  # noqa: ANN001, ANN202
        calls.append(list(args))
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    paths = service.install(
        tmp_path / "data",
        python="/opt/venv/bin/python",
        working_directory=tmp_path / "repo",
        runner=runner,
    )

    assert paths["service"].is_file()
    assert paths["timer"].is_file()
    assert calls == [["daemon-reload"], ["enable", "--now", service.TIMER_NAME]]
    assert service.status()["installed"] is True


def test_uninstall_is_reversible(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    calls: list[list[str]] = []

    def runner(args):  # noqa: ANN001, ANN202
        calls.append(list(args))
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    service.install(tmp_path / "data", working_directory=tmp_path, runner=runner)
    assert service.uninstall(runner=runner) is True

    assert not service.service_path().exists()
    assert not service.timer_path().exists()
    assert calls[-1] == ["daemon-reload"]
    assert service.uninstall(runner=runner) is False
