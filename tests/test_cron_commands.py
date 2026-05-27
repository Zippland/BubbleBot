import json

from typer.testing import CliRunner

from bubbles.cli.commands import app
from bubbles.cron.service import CronService
from bubbles.cron.types import CronSchedule

runner = CliRunner()


def test_cron_add_rejects_invalid_timezone(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr("bubbles.config.loader.get_data_dir", lambda: tmp_path)

    result = runner.invoke(
        app,
        [
            "cron",
            "add",
            "--name",
            "demo",
            "--message",
            "hello",
            "--cron",
            "0 9 * * *",
            "--tz",
            "America/Vancovuer",
        ],
    )

    assert result.exit_code == 1
    assert "Error: unknown timezone 'America/Vancovuer'" in result.stdout
    assert not (tmp_path / "cron" / "jobs.json").exists()


def _seed_two_session_jobs(tmp_path):
    service = CronService(tmp_path / "cron" / "jobs.json")
    service.add_job(
        name="job-a",
        schedule=CronSchedule(kind="every", every_ms=60_000),
        message="a",
        session_key="telegram:me",
        channel="telegram",
    )
    service.add_job(
        name="job-b",
        schedule=CronSchedule(kind="every", every_ms=60_000),
        message="b",
        session_key="feishu:other",
        channel="feishu",
    )


def test_cron_list_filters_by_session(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr("bubbles.config.loader.get_data_dir", lambda: tmp_path)
    _seed_two_session_jobs(tmp_path)

    result = runner.invoke(app, ["cron", "list", "--session", "telegram:me"])

    assert result.exit_code == 0
    assert "job-a" in result.stdout
    assert "job-b" not in result.stdout


def test_cron_list_filters_by_channel(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr("bubbles.config.loader.get_data_dir", lambda: tmp_path)
    _seed_two_session_jobs(tmp_path)

    result = runner.invoke(app, ["cron", "list", "--channel", "feishu"])

    assert result.exit_code == 0
    assert "job-b" in result.stdout
    assert "job-a" not in result.stdout


def test_cron_list_json_output(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr("bubbles.config.loader.get_data_dir", lambda: tmp_path)
    _seed_two_session_jobs(tmp_path)

    result = runner.invoke(app, ["cron", "list", "--json"])

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert isinstance(payload, list)
    assert {j["name"] for j in payload} == {"job-a", "job-b"}
    # Every entry must carry the canonical fields the contract promises.
    for entry in payload:
        assert {"id", "name", "enabled", "schedule", "next_relative", "last_status", "session_key", "channel"} <= entry.keys()
