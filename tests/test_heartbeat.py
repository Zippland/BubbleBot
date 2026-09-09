"""Compatibility tests for removal of the former heartbeat feature."""

from unittest.mock import AsyncMock

import pytest

from bubbles.cli.status_cmd import _collect_session_overrides
from bubbles.cron.service import CronService
from bubbles.cron.types import CronSchedule, is_retired_heartbeat
from bubbles.session.manager import SessionManager


@pytest.mark.parametrize("name,key,retired", [
    ("heartbeat:shared", "shared", True),
    ("heartbeat:cli:direct", "cli:direct", True),
    ("heartbeat:shared", "other", False),
    ("heartbeat:shared", None, False),
    ("regular", "shared", False),
])
def test_retirement_only_matches_reserved_job_identity(name, key, retired):
    assert is_retired_heartbeat(name, key) is retired


@pytest.mark.asyncio
async def test_legacy_heartbeat_cannot_be_forced_even_before_service_start(tmp_path):
    path = tmp_path / "jobs.json"
    old = CronService(path)
    job = old.add_job("heartbeat:shared", CronSchedule(kind="every", every_ms=60_000),
        "old heartbeat", session_key="shared")
    callback = AsyncMock()
    service = CronService(path, on_job=callback)
    assert not await service.run_job(job.id, force=True)
    assert service.enable_job(job.id) is None
    callback.assert_not_awaited()


def test_status_omits_legacy_heartbeat_but_keeps_regular_jobs(tmp_path):
    sessions = tmp_path / "sessions"
    manager = SessionManager(sessions_dir=sessions)
    session = manager.get_or_create("shared")
    manager.save(session)
    service = CronService(tmp_path / "cron" / "jobs.json")
    service.add_job("heartbeat:shared", CronSchedule(kind="every", every_ms=60_000),
        "tick", session_key="shared")
    assert _collect_session_overrides(sessions, tmp_path, "test-model") == ([], 0)
    service.add_job("regular", CronSchedule(kind="every", every_ms=60_000),
        "run", session_key="shared")
    overrides, unassigned = _collect_session_overrides(sessions, tmp_path, "test-model")
    assert unassigned == 0
    assert len(overrides) == 1
    assert overrides[0]["cron"] == 1
    assert "heartbeat_ms" not in overrides[0]
