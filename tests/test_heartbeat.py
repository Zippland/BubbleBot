"""Tests for /heartbeat command + helpers."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from bubbles.agent.loop import AgentLoop, HEARTBEAT_TICK_MESSAGE
from bubbles.bus.events import InboundMessage
from bubbles.cron.service import CronService
from bubbles.session.manager import SessionManager


# ---------- Static helpers (pure functions) ----------

@pytest.mark.parametrize("s,expected", [
    ("30", 30),
    ("30m", 30),
    ("2h", 120),
    ("1h", 60),
    ("90s", 2),       # 90 / 60 with rounding → 2
    ("30s", 1),       # subminute floors to 1
    ("1", 1),
    ("  45m  ", 45),  # surrounding whitespace
    ("30 m", 30),     # inner space between number and unit is allowed
    ("60M", 60),      # case-insensitive
])
def test_parse_heartbeat_interval_valid(s: str, expected: int) -> None:
    assert AgentLoop._parse_heartbeat_interval(s) == expected


@pytest.mark.parametrize("s", [
    "", "abc", "30x", "1.5h", "-5", "h", "30min",
])
def test_parse_heartbeat_interval_invalid(s: str) -> None:
    assert AgentLoop._parse_heartbeat_interval(s) is None


def test_humanize_minutes_cn() -> None:
    assert AgentLoop._humanize_minutes_cn(30) == "30 分钟"
    assert AgentLoop._humanize_minutes_cn(60) == "1 小时"
    assert AgentLoop._humanize_minutes_cn(120) == "2 小时"
    assert AgentLoop._humanize_minutes_cn(90) == "90 分钟"  # not a whole hour


def test_humanize_minutes_en() -> None:
    assert AgentLoop._humanize_minutes_en(30) == "30 minutes"
    assert AgentLoop._humanize_minutes_en(1) == "1 minute"
    assert AgentLoop._humanize_minutes_en(60) == "1 hour"
    assert AgentLoop._humanize_minutes_en(120) == "2 hours"


def test_heartbeat_job_name() -> None:
    assert AgentLoop._heartbeat_job_name("cli:direct") == "heartbeat:cli:direct"
    assert AgentLoop._heartbeat_job_name("foo") == "heartbeat:foo"


# ---------- Command integration ----------

@pytest.fixture
def loop(tmp_path):
    """Build an AgentLoop with real cron+sessions, mocked bus+provider."""
    bus = MagicMock()
    provider = MagicMock()
    provider.get_default_model.return_value = "test-model"

    cron = CronService(tmp_path / "cron" / "jobs.json")

    sessions = SessionManager(sessions_dir=tmp_path / "sessions")

    return AgentLoop(
        bus=bus,
        provider=provider,
        max_tokens=4096,
        memory_window=20,
        context_limit=128_000,
        cron_service=cron,
        session_manager=sessions,
    )


def _make_msg(content: str) -> InboundMessage:
    return InboundMessage(channel="cli", sender_id="user", chat_id="direct", content=content)


def test_heartbeat_status_when_off(loop) -> None:
    session = loop.sessions.get_or_create("cli:direct")
    msg = _make_msg("/heartbeat")
    out = loop._handle_heartbeat_command(msg, session, "cli:direct", "")
    assert "心跳未开启" in out.content
    assert loop._build_heartbeat_info("cli:direct") is None


def test_heartbeat_enable_creates_job_and_template(loop) -> None:
    session = loop.sessions.get_or_create("cli:direct")
    assert session.directory is not None
    hb_path = session.directory / "HEARTBEATS.md"
    assert not hb_path.exists()

    msg = _make_msg("/heartbeat 30m")
    out = loop._handle_heartbeat_command(msg, session, "cli:direct", "30m")

    assert "心跳已开启" in out.content
    assert "30 分钟" in out.content
    assert hb_path.exists(), "HEARTBEATS.md template should be auto-written"
    assert "Heartbeats" in hb_path.read_text(encoding="utf-8")

    # Cron job registered with correct name + interval + tick message
    job = loop._get_heartbeat_job("cli:direct")
    assert job is not None
    assert job.schedule.kind == "every"
    assert job.schedule.every_ms == 30 * 60 * 1000
    assert job.payload.message == HEARTBEAT_TICK_MESSAGE
    assert job.payload.deliver is True


def test_heartbeat_info_block_present_when_on(loop) -> None:
    session = loop.sessions.get_or_create("cli:direct")
    msg = _make_msg("/heartbeat 30m")
    loop._handle_heartbeat_command(msg, session, "cli:direct", "30m")

    info = loop._build_heartbeat_info("cli:direct")
    assert info is not None
    assert "## Heartbeat: ON" in info
    assert "30 minutes" in info
    assert "stay_silent" in info


def test_heartbeat_replace_keeps_existing_template(loop) -> None:
    """Re-enabling should not overwrite a user-edited HEARTBEATS.md."""
    session = loop.sessions.get_or_create("cli:direct")
    msg = _make_msg("/heartbeat 30m")
    loop._handle_heartbeat_command(msg, session, "cli:direct", "30m")

    hb_path = session.directory / "HEARTBEATS.md"
    hb_path.write_text("# my own checklist\n- item A", encoding="utf-8")

    # Change interval — should replace cron job but keep file
    out = loop._handle_heartbeat_command(msg, session, "cli:direct", "1h")
    assert "已开启" in out.content
    assert "1 小时" in out.content
    assert hb_path.read_text(encoding="utf-8") == "# my own checklist\n- item A"
    assert "沿用现有" in out.content

    # Only one heartbeat job exists (old was replaced, not duplicated)
    matches = [
        j for j in loop.cron_service.list_jobs(include_disabled=True)
        if j.name == "heartbeat:cli:direct"
    ]
    assert len(matches) == 1
    assert matches[0].schedule.every_ms == 60 * 60 * 1000


def test_heartbeat_off(loop) -> None:
    session = loop.sessions.get_or_create("cli:direct")
    msg = _make_msg("/heartbeat 30m")
    loop._handle_heartbeat_command(msg, session, "cli:direct", "30m")
    assert loop._get_heartbeat_job("cli:direct") is not None

    out = loop._handle_heartbeat_command(_make_msg("/heartbeat off"), session, "cli:direct", "off")
    assert "已关闭" in out.content
    assert loop._get_heartbeat_job("cli:direct") is None
    assert loop._build_heartbeat_info("cli:direct") is None


def test_heartbeat_off_when_not_on(loop) -> None:
    session = loop.sessions.get_or_create("cli:direct")
    out = loop._handle_heartbeat_command(_make_msg("/heartbeat off"), session, "cli:direct", "off")
    assert "本来就没开" in out.content


def test_heartbeat_invalid_format(loop) -> None:
    session = loop.sessions.get_or_create("cli:direct")
    out = loop._handle_heartbeat_command(_make_msg("/heartbeat junk"), session, "cli:direct", "junk")
    assert "格式不对" in out.content
    assert loop._get_heartbeat_job("cli:direct") is None


def test_heartbeat_out_of_range(loop) -> None:
    session = loop.sessions.get_or_create("cli:direct")
    out = loop._handle_heartbeat_command(_make_msg("/heartbeat 0m"), session, "cli:direct", "0m")
    assert "间隔需在" in out.content
    out = loop._handle_heartbeat_command(_make_msg("/heartbeat 25h"), session, "cli:direct", "25h")
    assert "间隔需在" in out.content


def test_heartbeat_isolated_per_session(loop) -> None:
    """Two sessions should have independent heartbeat state."""
    s1 = loop.sessions.get_or_create("cli:alice")
    s2 = loop.sessions.get_or_create("cli:bob")
    loop._handle_heartbeat_command(_make_msg("/heartbeat 30m"), s1, "cli:alice", "30m")

    assert loop._get_heartbeat_job("cli:alice") is not None
    assert loop._get_heartbeat_job("cli:bob") is None
    assert loop._build_heartbeat_info("cli:alice") is not None
    assert loop._build_heartbeat_info("cli:bob") is None
