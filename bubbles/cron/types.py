"""Cron types."""

from dataclasses import dataclass, field
from typing import Literal


@dataclass
class CronSchedule:
    """Schedule definition for a cron job."""
    kind: Literal["at", "every", "cron"]
    # For "at": timestamp in ms
    at_ms: int | None = None
    # For "every": interval in ms
    every_ms: int | None = None
    # For "every": anchor timestamp (ms). Next-run slots are anchor + N*every_ms,
    # so interval alignment survives restarts. Set on first add_job.
    anchor_ms: int | None = None
    # For "cron": cron expression (e.g. "0 9 * * *")
    expr: str | None = None
    # Timezone for cron expressions
    tz: str | None = None


@dataclass
class CronPayload:
    """What to do when the job runs."""
    kind: Literal["system_event", "agent_turn"] = "agent_turn"
    message: str = ""
    # Deliver response to channel
    deliver: bool = False
    channel: str | None = None  # e.g. "whatsapp"
    to: str | None = None  # e.g. phone number
    # Session key to inject history from (if None, uses cron:{job_id})
    session_key: str | None = None


@dataclass
class CronJobState:
    """Runtime state of a job."""
    next_run_at_ms: int | None = None
    last_run_at_ms: int | None = None
    # Set when _execute_job begins, cleared when it returns. If set on startup
    # it means the previous process crashed mid-run; sanitize logic treats the
    # run as completed (next_run_at_ms is pre-advanced before execution).
    running_at_ms: int | None = None
    # Exponential backoff: number of consecutive failures (reset on success).
    consecutive_errors: int = 0
    # When set, the scheduler treats the job as not-due until now >= this.
    backoff_until_ms: int | None = None
    last_status: Literal["ok", "error", "skipped"] | None = None
    last_error: str | None = None


@dataclass
class CronJob:
    """A scheduled job."""
    id: str
    name: str
    enabled: bool = True
    schedule: CronSchedule = field(default_factory=lambda: CronSchedule(kind="every"))
    payload: CronPayload = field(default_factory=CronPayload)
    state: CronJobState = field(default_factory=CronJobState)
    created_at_ms: int = 0
    updated_at_ms: int = 0
    delete_after_run: bool = False


@dataclass
class CronStore:
    """Persistent store for cron jobs."""
    version: int = 1
    jobs: list[CronJob] = field(default_factory=list)
