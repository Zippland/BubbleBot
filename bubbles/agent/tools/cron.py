"""Cron tool for scheduling reminders and tasks."""

from typing import Any

from bubbles.agent.tools.base import Tool
from bubbles.cron.format import format_job_block
from bubbles.cron.service import CronService
from bubbles.cron.types import CronSchedule


class CronTool(Tool):
    """Tool to schedule reminders and recurring tasks."""

    def __init__(self, cron_service: CronService):
        self._cron = cron_service
        self._channel = ""
        self._chat_id = ""
        self._session_key = ""

    def set_context(self, channel: str, chat_id: str, session_key: str = "") -> None:
        """Set the current session context for delivery and history injection."""
        self._channel = channel
        self._chat_id = chat_id
        self._session_key = session_key or f"{channel}:{chat_id}"
    
    @property
    def name(self) -> str:
        return "cron"

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["add", "list", "remove"],
                    "description": "Action to perform"
                },
                "message": {
                    "type": "string",
                    "description": "Reminder message (for add)"
                },
                "every_seconds": {
                    "type": "integer",
                    "description": "Interval in seconds (for recurring tasks)"
                },
                "cron_expr": {
                    "type": "string",
                    "description": "Cron expression like '0 9 * * *' (for scheduled tasks)"
                },
                "tz": {
                    "type": "string",
                    "description": "IANA timezone for cron expressions (e.g. 'America/Vancouver')"
                },
                "at": {
                    "type": "string",
                    "description": "ISO datetime for one-time execution (e.g. '2026-02-12T10:30:00')"
                },
                "job_id": {
                    "type": "string",
                    "description": "Job ID (for remove)"
                },
                "include_disabled": {
                    "type": "boolean",
                    "description": "For list: include disabled jobs (default false)"
                }
            },
            "required": ["action"]
        }

    async def execute(
        self,
        action: str,
        message: str = "",
        every_seconds: int | None = None,
        cron_expr: str | None = None,
        tz: str | None = None,
        at: str | None = None,
        job_id: str | None = None,
        include_disabled: bool = False,
        **kwargs: Any
    ) -> str:
        if action == "add":
            return self._add_job(message, every_seconds, cron_expr, tz, at)
        elif action == "list":
            return self._list_jobs(include_disabled=include_disabled)
        elif action == "remove":
            return self._remove_job(job_id)
        return f"Unknown action: {action}"
    
    def _add_job(
        self,
        message: str,
        every_seconds: int | None,
        cron_expr: str | None,
        tz: str | None,
        at: str | None,
    ) -> str:
        if not message:
            return "Error: message is required for add"
        if not self._channel or not self._chat_id:
            return "Error: no session context (channel/chat_id)"
        if tz and not cron_expr:
            return "Error: tz can only be used with cron_expr"
        if tz:
            from zoneinfo import ZoneInfo
            try:
                ZoneInfo(tz)
            except (KeyError, Exception):
                return f"Error: unknown timezone '{tz}'"
        
        # Build schedule
        delete_after = False
        if every_seconds:
            schedule = CronSchedule(kind="every", every_ms=every_seconds * 1000)
        elif cron_expr:
            schedule = CronSchedule(kind="cron", expr=cron_expr, tz=tz)
        elif at:
            from datetime import datetime
            dt = datetime.fromisoformat(at)
            at_ms = int(dt.timestamp() * 1000)
            schedule = CronSchedule(kind="at", at_ms=at_ms)
            delete_after = True
        else:
            return "Error: either every_seconds, cron_expr, or at is required"
        
        job = self._cron.add_job(
            name=message[:30],
            schedule=schedule,
            message=message,
            deliver=True,
            channel=self._channel,
            to=self._chat_id,
            delete_after_run=delete_after,
            session_key=self._session_key,
        )
        return f"Created job '{job.name}' (id: {job.id})"
    
    def _list_jobs(self, include_disabled: bool = False) -> str:
        # Session isolation: agent only ever sees jobs created from this session.
        # Legacy jobs without session_key (e.g. created via CLI) are invisible here.
        if not self._session_key:
            return "No scheduled jobs in this session."

        all_jobs = self._cron.list_jobs(include_disabled=include_disabled)
        jobs = [j for j in all_jobs if j.payload.session_key == self._session_key]
        if not jobs:
            return "No scheduled jobs in this session."

        header = f"Scheduled jobs in this session ({len(jobs)}):\n"
        blocks = [format_job_block(j, i + 1) for i, j in enumerate(jobs)]
        return header + "\n\n".join(blocks)

    def _remove_job(self, job_id: str | None) -> str:
        if not job_id:
            return "Error: job_id is required for remove"

        # Session isolation: only allow removing jobs that belong to this session.
        # To avoid leaking the existence of jobs in other sessions, we return the
        # same "not found" message for (a) genuinely missing ids and (b) ids that
        # exist but belong to other sessions.
        for job in self._cron.list_jobs(include_disabled=True):
            if job.id != job_id:
                continue
            if job.payload.session_key != self._session_key:
                return f"Job {job_id} not found"
            if self._cron.remove_job(job_id):
                return f"Removed job {job_id}"
            return f"Job {job_id} not found"
        return f"Job {job_id} not found"
