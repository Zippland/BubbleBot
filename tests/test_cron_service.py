import asyncio
import json

import pytest

from bubbles.cron.service import (
    BACKOFF_SCHEDULE_MS,
    MAX_TIMER_DELAY_MS,
    CronService,
    _backoff_delay_ms,
    _compute_next_run,
)
from bubbles.cron.types import CronSchedule


def test_add_job_rejects_unknown_timezone(tmp_path) -> None:
    service = CronService(tmp_path / "cron" / "jobs.json")

    with pytest.raises(ValueError, match="unknown timezone 'America/Vancovuer'"):
        service.add_job(
            name="tz typo",
            schedule=CronSchedule(kind="cron", expr="0 9 * * *", tz="America/Vancovuer"),
            message="hello",
        )

    assert service.list_jobs(include_disabled=True) == []


def test_add_job_accepts_valid_timezone(tmp_path) -> None:
    service = CronService(tmp_path / "cron" / "jobs.json")

    job = service.add_job(
        name="tz ok",
        schedule=CronSchedule(kind="cron", expr="0 9 * * *", tz="America/Vancouver"),
        message="hello",
    )

    assert job.schedule.tz == "America/Vancouver"
    assert job.state.next_run_at_ms is not None


def test_every_compute_next_strictly_future() -> None:
    """`every` must always return a strictly-future slot, even at exact tick boundary."""
    sched = CronSchedule(kind="every", every_ms=5 * 60_000, anchor_ms=10_000_000)
    # At exact tick (anchor + 5min): must skip to next slot, not return same time
    nxt = _compute_next_run(sched, now_ms=10_000_000 + 5 * 60_000)
    assert nxt == 10_000_000 + 10 * 60_000

    # Just past a tick: also next slot
    nxt = _compute_next_run(sched, now_ms=10_000_000 + 5 * 60_000 + 1)
    assert nxt == 10_000_000 + 10 * 60_000

    # Far past anchor (24 ticks elapsed): next is 25th tick
    nxt = _compute_next_run(sched, now_ms=10_000_000 + 120 * 60_000)
    assert nxt == 10_000_000 + 125 * 60_000

    # Before anchor: returns anchor itself
    nxt = _compute_next_run(sched, now_ms=10_000_000 - 1000)
    assert nxt == 10_000_000


def test_add_every_sets_anchor(tmp_path) -> None:
    service = CronService(tmp_path / "cron" / "jobs.json")
    job = service.add_job(
        name="poll",
        schedule=CronSchedule(kind="every", every_ms=60_000),
        message="ping",
    )
    assert job.schedule.anchor_ms is not None
    assert job.schedule.anchor_ms > 0


def test_anchor_survives_restart(tmp_path) -> None:
    """`every` interval must remain aligned to the original anchor across restarts.

    Without anchor persistence, a job added at T=0 with every=300s would, after
    a restart at T=400s, schedule next at T=700s (now + 300). With anchor
    persistence it schedules at T=600s (2nd tick of original alignment).
    """
    store_path = tmp_path / "cron" / "jobs.json"
    svc1 = CronService(store_path)
    job = svc1.add_job(
        name="poll",
        schedule=CronSchedule(kind="every", every_ms=300_000),
        message="ping",
    )
    original_anchor = job.schedule.anchor_ms

    # Simulate restart: fresh service, same store
    svc2 = CronService(store_path)
    reloaded = svc2.list_jobs(include_disabled=True)[0]
    assert reloaded.schedule.anchor_ms == original_anchor


def test_sanitize_clears_stale_running_at_ms(tmp_path) -> None:
    """A `running_at_ms` left over from a crashed process must be cleared on load."""
    store_path = tmp_path / "cron" / "jobs.json"
    svc1 = CronService(store_path)
    svc1.add_job(
        name="poll",
        schedule=CronSchedule(kind="every", every_ms=60_000),
        message="ping",
    )

    # Simulate crash: write running_at_ms directly into store file
    data = json.loads(store_path.read_text())
    data["jobs"][0]["state"]["runningAtMs"] = 12345
    store_path.write_text(json.dumps(data))

    # Boot a fresh service — sanitize must clear the flag
    svc2 = CronService(store_path)

    async def boot():
        await svc2.start()
        svc2.stop()

    asyncio.run(boot())
    cleaned = svc2.list_jobs(include_disabled=True)[0]
    assert cleaned.state.running_at_ms is None


def test_execute_pre_advances_next_run(tmp_path) -> None:
    """next_run_at_ms must be advanced before the on_job callback runs.

    Verifies crash safety: if the process dies inside on_job, the next slot is
    already past the current one so we don't re-fire on restart.
    """
    store_path = tmp_path / "cron" / "jobs.json"

    observed_next_run: list[int | None] = []

    async def callback(job):
        # Inspect the persisted state mid-run
        data = json.loads(store_path.read_text())
        observed_next_run.append(data["jobs"][0]["state"]["nextRunAtMs"])

    svc = CronService(store_path, on_job=callback)
    job = svc.add_job(
        name="poll",
        schedule=CronSchedule(kind="every", every_ms=60_000),
        message="ping",
    )

    asyncio.run(svc.run_job(job.id, force=True))

    assert len(observed_next_run) == 1
    # next_run during the callback must already be set to a future slot
    assert observed_next_run[0] is not None
    assert observed_next_run[0] > job.schedule.anchor_ms


def test_arm_timer_capped(tmp_path) -> None:
    """A far-future next_wake must not cause asyncio.sleep beyond MAX_TIMER_DELAY_MS."""
    svc = CronService(tmp_path / "cron" / "jobs.json")

    sleep_seen: list[float] = []
    real_sleep = asyncio.sleep

    async def fake_sleep(s):
        sleep_seen.append(s)
        await real_sleep(0)

    async def run():
        svc._running = True
        svc._load_store()
        # Job 1 hour in the future (must be added inside running loop)
        svc.add_job(
            name="far",
            schedule=CronSchedule(kind="every", every_ms=3600_000),
            message="future",
        )

        asyncio.sleep = fake_sleep  # type: ignore
        try:
            svc._arm_timer()
            await real_sleep(0.05)  # let timer task start
        finally:
            asyncio.sleep = real_sleep  # type: ignore
        svc.stop()

    asyncio.run(run())

    assert sleep_seen, "timer did not schedule any sleep"
    # Even though next_wake is ~3600s out, the requested delay must be capped
    assert sleep_seen[0] <= MAX_TIMER_DELAY_MS / 1000


def test_backoff_delay_curve() -> None:
    """1st→30s, 2nd→1m, 3rd→5m, 4th→15m, 5th+→60m."""
    assert _backoff_delay_ms(1) == 30_000
    assert _backoff_delay_ms(2) == 60_000
    assert _backoff_delay_ms(3) == 5 * 60_000
    assert _backoff_delay_ms(4) == 15 * 60_000
    assert _backoff_delay_ms(5) == 60 * 60_000
    # Beyond table → tail value
    assert _backoff_delay_ms(99) == BACKOFF_SCHEDULE_MS[-1]


def test_failure_sets_backoff(tmp_path) -> None:
    store_path = tmp_path / "cron" / "jobs.json"

    async def failing(job):
        raise RuntimeError("simulated")

    svc = CronService(store_path, on_job=failing)

    async def run():
        job = svc.add_job(
            name="poll",
            schedule=CronSchedule(kind="every", every_ms=60_000),
            message="ping",
        )
        await svc.run_job(job.id, force=True)
        return job

    job = asyncio.run(run())
    assert job.state.consecutive_errors == 1
    assert job.state.backoff_until_ms is not None
    assert job.state.last_status == "error"


def test_success_clears_backoff(tmp_path) -> None:
    store_path = tmp_path / "cron" / "jobs.json"
    call_count = {"n": 0}

    async def flaky(job):
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise RuntimeError("first call fails")
        # second call succeeds

    svc = CronService(store_path, on_job=flaky)

    async def run():
        job = svc.add_job(
            name="poll",
            schedule=CronSchedule(kind="every", every_ms=60_000),
            message="ping",
        )
        await svc.run_job(job.id, force=True)
        assert job.state.consecutive_errors == 1
        assert job.state.backoff_until_ms is not None
        # Force again — bypasses backoff window
        await svc.run_job(job.id, force=True)
        return job

    job = asyncio.run(run())
    assert job.state.consecutive_errors == 0
    assert job.state.backoff_until_ms is None
    assert job.state.last_status == "ok"


def test_backoff_blocks_due_job_in_on_timer(tmp_path) -> None:
    """A job in backoff window must not be picked up by _on_timer."""
    store_path = tmp_path / "cron" / "jobs.json"
    call_count = {"n": 0}

    async def failing(job):
        call_count["n"] += 1
        raise RuntimeError("nope")

    svc = CronService(store_path, on_job=failing)

    async def run():
        svc._running = True
        svc._load_store()
        job = svc.add_job(
            name="poll",
            schedule=CronSchedule(kind="every", every_ms=60_000),
            message="ping",
        )
        # Trigger failure → backoff window set
        await svc.run_job(job.id, force=True)
        assert call_count["n"] == 1

        # Make the job "due" by force-advancing next_run into the past
        job.state.next_run_at_ms = 0
        # _on_timer should still skip due to backoff
        await svc._on_timer()
        return call_count["n"]

    n = asyncio.run(run())
    assert n == 1, "Expected backoff to suppress re-fire"


def test_backoff_persists_through_restart(tmp_path) -> None:
    store_path = tmp_path / "cron" / "jobs.json"

    async def failing(job):
        raise RuntimeError("nope")

    svc1 = CronService(store_path, on_job=failing)

    async def first_run():
        job = svc1.add_job(
            name="poll",
            schedule=CronSchedule(kind="every", every_ms=60_000),
            message="ping",
        )
        await svc1.run_job(job.id, force=True)
        return job.id

    job_id = asyncio.run(first_run())

    svc2 = CronService(store_path, on_job=failing)
    reloaded = next(j for j in svc2.list_jobs(include_disabled=True) if j.id == job_id)
    assert reloaded.state.consecutive_errors == 1
    assert reloaded.state.backoff_until_ms is not None
