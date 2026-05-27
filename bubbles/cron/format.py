"""Human-readable formatting helpers for cron jobs.

Shared by the agent `cron` tool and the `bubbles cron` CLI so both surfaces
present the same vocabulary for schedule, next-run time, and status.
"""

from __future__ import annotations

import time
from datetime import datetime
from zoneinfo import ZoneInfo

from bubbles.cron.types import CronJob, CronJobState, CronSchedule


def format_schedule(schedule: CronSchedule) -> str:
    """Render a schedule as a short human phrase."""
    if schedule.kind == "every":
        return f"every {format_duration_short((schedule.every_ms or 0) // 1000)}"
    if schedule.kind == "cron":
        expr = schedule.expr or ""
        return f"{expr} ({schedule.tz})" if schedule.tz else expr
    if schedule.kind == "at" and schedule.at_ms:
        return f"once at {format_absolute(schedule.at_ms, schedule.tz)}"
    return schedule.kind


def format_duration_short(seconds: int) -> str:
    """Compact duration: `30s`, `5m`, `2h`, `1d 3h`. Used inside `every <X>`."""
    if seconds < 60:
        return f"{seconds}s"
    if seconds < 3600:
        return f"{seconds // 60}m"
    if seconds < 86400:
        h, rem = divmod(seconds, 3600)
        m = rem // 60
        return f"{h}h" if m == 0 else f"{h}h {m}m"
    d, rem = divmod(seconds, 86400)
    h = rem // 3600
    return f"{d}d" if h == 0 else f"{d}d {h}h"


def format_relative(target_ms: int, now_ms: int | None = None) -> str:
    """Render `target_ms` as `in 3h 12m` / `in 14m` / `in <1m` / `overdue`.

    Tiered precision (intentionally no seconds): the scheduler reasons about
    minutes-scale slots — second-precision in user-facing strings is noise.
    """
    now = now_ms if now_ms is not None else int(time.time() * 1000)
    delta_s = (target_ms - now) // 1000
    if delta_s < 0:
        return "overdue"
    if delta_s < 60:
        return "in <1m"
    if delta_s < 3600:
        return f"in {delta_s // 60}m"
    if delta_s < 86400:
        h, rem = divmod(delta_s, 3600)
        m = rem // 60
        return f"in {h}h" if m == 0 else f"in {h}h {m}m"
    d, rem = divmod(delta_s, 86400)
    h = rem // 3600
    return f"in {d}d" if h == 0 else f"in {d}d {h}h"


def format_absolute(ts_ms: int, tz: str | None = None) -> str:
    """Render `ts_ms` as `YYYY-MM-DD HH:MM` in the given tz (or local)."""
    try:
        zone = ZoneInfo(tz) if tz else None
        return datetime.fromtimestamp(ts_ms / 1000, zone).strftime("%Y-%m-%d %H:%M")
    except Exception:
        return time.strftime("%Y-%m-%d %H:%M", time.localtime(ts_ms / 1000))


def format_status(state: CronJobState, now_ms: int | None = None) -> str:
    """One-line status: `ok` / `error` / `never ran`, with backoff hint when relevant."""
    now = now_ms if now_ms is not None else int(time.time() * 1000)
    last = state.last_status
    if last is None:
        base = "never ran"
    elif last == "ok":
        base = "ok"
    elif last == "error":
        base = "error"
    else:
        base = last

    if state.backoff_until_ms and state.backoff_until_ms > now:
        return f"{base} · backoff until {format_relative(state.backoff_until_ms, now)}"
    return base


def format_job_block(
    job: CronJob,
    index: int,
    now_ms: int | None = None,
) -> str:
    """Multi-line block describing a single job. Used by the agent tool output."""
    now = now_ms if now_ms is not None else int(time.time() * 1000)
    name = job.name
    if not job.enabled:
        name = f"{name} [disabled]"

    lines = [f"[{index}] {name}  (id: {job.id})"]
    lines.append(f"    schedule: {format_schedule(job.schedule)}")

    next_ms = job.state.next_run_at_ms
    if next_ms:
        rel = format_relative(next_ms, now)
        if job.state.backoff_until_ms and job.state.backoff_until_ms > next_ms:
            gate_rel = format_relative(job.state.backoff_until_ms, now)
            lines.append(f"    next: {rel} (gated by backoff: {gate_rel})")
        else:
            lines.append(f"    next: {rel}")
    elif not job.enabled:
        lines.append("    next: —")
    else:
        lines.append("    next: —")

    lines.append(f"    status: {format_status(job.state, now)}")
    if job.state.last_status == "error" and job.state.last_error:
        lines.append(f"    last error: {job.state.last_error}")

    return "\n".join(lines)
