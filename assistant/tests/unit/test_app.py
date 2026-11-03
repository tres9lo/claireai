"""The launcher: one command, and every old command still works."""

from __future__ import annotations

import socket
from pathlib import Path

import pytest

from claireai import app
from claireai.app import Running, banner, main, pick_port, prepare, wants_browser
from claireai.config import Settings


def _settings(tmp_path: Path) -> Settings:
    return Settings(_env_file=None, data_dir=tmp_path / "data")


def test_arguments_are_passed_to_the_cli_unchanged(monkeypatch) -> None:
    seen: list[list[str]] = []

    def fake_cli(argv):  # noqa: ANN001, ANN202
        seen.append(list(argv))
        return 7

    monkeypatch.setattr("claireai.cli.main", fake_cli)
    assert main(["--once", "hello"]) == 7
    assert seen == [["--once", "hello"]]


def test_cli_flag_opens_the_terminal_chat(monkeypatch) -> None:
    seen: list[list[str]] = []

    def fake_cli(argv):  # noqa: ANN001, ANN202
        seen.append(list(argv))
        return 0

    monkeypatch.setattr("claireai.cli.main", fake_cli)
    assert main(["--cli"]) == 0
    assert seen == [[]]


def test_the_app_starts_and_serves_by_default(monkeypatch, capsys, tmp_path) -> None:
    running = Running(
        url="http://127.0.0.1:8765/",
        host="127.0.0.1",
        port=8765,
        data_dir=tmp_path / "data",
        database_path=tmp_path / "data" / "db" / "claireai.sqlite3",
        browser_tools=False,
    )
    served: list[object] = []

    monkeypatch.setattr(app, "prepare", lambda settings, port: (object(), running))

    assert main([], serve_forever=served.append) == 0
    assert len(served) == 1

    out = capsys.readouterr().out
    assert "ClaireAI is running." in out
    assert "http://127.0.0.1:8765/" in out
    assert str(running.data_dir) in out
    assert "Ctrl+C" in out


def test_a_missing_page_is_a_config_error(monkeypatch, capsys, tmp_path) -> None:
    def explode(settings, port):  # noqa: ANN001, ANN202
        raise FileNotFoundError("the app page is missing: /nowhere/index.html")

    monkeypatch.setattr(app, "prepare", explode)

    assert main([], serve_forever=lambda server: None) == app.EXIT_CONFIG
    assert "ERROR" in capsys.readouterr().err


def test_a_taken_port_is_skipped(tmp_path) -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as held:
        held.bind(("127.0.0.1", 0))
        held.listen(1)
        busy = held.getsockname()[1]

        free = pick_port("127.0.0.1", busy)
        assert free != busy
        assert busy < free < busy + app.PORT_ATTEMPTS


def test_prepare_migrates_the_database_and_reports_the_url(tmp_path) -> None:
    settings = _settings(tmp_path)
    server, running = prepare(settings, port=0)

    try:
        assert running.url.startswith("http://127.0.0.1:")
        assert running.url.endswith("/")
        assert running.database_path.is_file()
        assert running.data_dir == settings.resolved_data_dir
        assert server.server_address[0] == "127.0.0.1"
    finally:
        server.server_close()


def test_prepare_refuses_a_missing_page(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        prepare(_settings(tmp_path), port=0, html_path=tmp_path / "nope.html")


def test_the_banner_says_where_it_is_and_how_to_stop(tmp_path) -> None:
    running = Running(
        url="http://127.0.0.1:9000/",
        host="127.0.0.1",
        port=9000,
        data_dir=tmp_path,
        database_path=tmp_path / "db.sqlite3",
        browser_tools=True,
    )
    text = banner(running)
    assert "http://127.0.0.1:9000/" in text
    assert "http://127.0.0.1:9000/today" in text
    assert "approvals still required" in text
    assert "Ctrl+C" in text


def test_the_banner_does_not_promise_a_browser_it_will_not_open(tmp_path) -> None:
    running = Running(
        url="http://127.0.0.1:9000/",
        host="127.0.0.1",
        port=9000,
        data_dir=tmp_path,
        database_path=tmp_path / "db.sqlite3",
        browser_tools=False,
    )
    assert "Open http://127.0.0.1:9000/ yourself" in banner(running, opening=False)
    assert "Opening it in your browser" in banner(running, opening=True)


def test_the_configured_data_directory_is_honoured(monkeypatch, tmp_path) -> None:
    seen: list[Path] = []

    def fake_prepare(settings, port):  # noqa: ANN001, ANN202
        seen.append(settings.resolved_data_dir)
        return object(), Running(
            url="http://127.0.0.1:1/", host="127.0.0.1", port=1,
            data_dir=settings.resolved_data_dir, database_path=tmp_path / "db.sqlite3",
            browser_tools=False,
        )

    monkeypatch.setenv("CLAIREAI_DATA_DIR", str(tmp_path / "from-env"))
    monkeypatch.setattr(app, "prepare", fake_prepare)

    assert main([], serve_forever=lambda server: None) == 0
    assert seen == [tmp_path / "from-env"]


class _Stream:
    def __init__(self, tty: bool) -> None:
        self.tty = tty

    def isatty(self) -> bool:
        return self.tty


def test_a_browser_opens_only_for_a_terminal(monkeypatch) -> None:
    monkeypatch.delenv("CLAIREAI_NO_BROWSER", raising=False)
    assert wants_browser(flag=True, stream=_Stream(True)) is True
    assert wants_browser(flag=True, stream=_Stream(False)) is False
    assert wants_browser(flag=False, stream=_Stream(True)) is False


def test_the_environment_can_disable_the_browser(monkeypatch) -> None:
    monkeypatch.setenv("CLAIREAI_NO_BROWSER", "1")
    assert wants_browser(flag=True, stream=_Stream(True)) is False
