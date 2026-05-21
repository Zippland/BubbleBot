"""Group-chat heartbeat lurking service (SPEC §5.2 item 3).

A single poll loop wakes every ``poll_seconds`` and asks the session manager which group
sessions opted in via ``/heartbeat on``. For each enabled session it checks the
per-session interval (or the global default) against the persisted ``last_heartbeat_at``
watermark; the ones whose time is up get a heartbeat turn dispatched to
``AgentLoop.run_heartbeat``. Concurrency is bounded by a semaphore.

The watermark itself lives in each session's metadata (persisted in session.jsonl), so a
gateway crash never silently drops messages that arrived between the last scan and the
crash. The in-memory ``_route`` map is allowed to reset on restart — the channel hook
repopulates it on the next inbound message.
"""

from __future__ import annotations

import asyncio
from datetime import datetime
from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    from bubbles.agent.loop import AgentLoop
    from bubbles.session.manager import SessionManager


# The poll loop wakes at this cadence (in seconds). Per-session intervals can be as short
# as one minute; polling every 60s ensures we never miss a due session by more than ~60s.
POLL_SECONDS = 60


class GroupHeartbeat:
    def __init__(
        self,
        *,
        agent_loop: "AgentLoop",
        session_manager: "SessionManager",
        default_interval_minutes: int,
        history_window: int,
        max_concurrent: int,
    ) -> None:
        self.agent_loop = agent_loop
        self.session_manager = session_manager
        self.default_interval_minutes = max(int(default_interval_minutes), 1)
        self.history_window = history_window
        self.max_concurrent = max(int(max_concurrent), 1)

        # Per-process state. The watermark itself is persisted in each session's
        # metadata (see AgentLoop.run_heartbeat); _route is fine to lose on restart —
        # the channel hook repopulates it on the next inbound message.
        self._route: dict[str, tuple[str, str]] = {}  # session_key -> (channel, chat_id)
        self._stop = asyncio.Event()
        self._task: asyncio.Task | None = None

    @property
    def interval_minutes(self) -> int:
        """Default interval shown in /heartbeat on confirmation when no per-session override."""
        return self.default_interval_minutes

    def note_outbound_route(self, session_key: str, channel: str, chat_id: str) -> None:
        """Called from the channel-layer hook every non-@ group message.

        Records where to send the heartbeat reply if the model decides to speak.
        """
        self._route[session_key] = (channel, chat_id)

    async def start(self) -> None:
        if self._task is not None:
            return
        self._stop.clear()
        self._task = asyncio.create_task(self._run(), name="group-heartbeat")
        logger.info(
            "GroupHeartbeat started (poll={}s, default_interval={}m, max_concurrent={}, history_window={})",
            POLL_SECONDS, self.default_interval_minutes, self.max_concurrent, self.history_window,
        )

    async def stop(self) -> None:
        self._stop.set()
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except (asyncio.CancelledError, Exception):
                pass
            self._task = None
        logger.info("GroupHeartbeat stopped")

    async def _run(self) -> None:
        while not self._stop.is_set():
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=POLL_SECONDS)
                break  # stop requested
            except asyncio.TimeoutError:
                pass
            try:
                await self._tick()
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("heartbeat tick crashed")

    async def _tick(self) -> None:
        enabled = self.session_manager.list_heartbeat_enabled_groups()
        if not enabled:
            logger.debug("heartbeat poll: 0 enabled groups")
            return

        now = datetime.now()
        candidates: list[str] = []
        for entry in enabled:
            key = entry["key"]
            if key not in self._route:
                continue
            interval_min = entry.get("interval_minutes") or self.default_interval_minutes
            last_iso = entry.get("last_heartbeat_at")
            if last_iso:
                try:
                    last = datetime.fromisoformat(last_iso)
                    if (now - last).total_seconds() < interval_min * 60:
                        continue  # not yet due
                except Exception:
                    pass  # corrupted watermark, treat as due
            candidates.append(key)

        if not candidates:
            logger.debug("heartbeat poll: {} enabled, 0 due this tick", len(enabled))
            return

        logger.info(
            "heartbeat poll: {} due group(s), max_concurrent={}",
            len(candidates), self.max_concurrent,
        )

        sem = asyncio.Semaphore(self.max_concurrent)

        async def _one(key: str) -> None:
            async with sem:
                channel, chat_id = self._route[key]
                try:
                    await self.agent_loop.run_heartbeat(
                        session_key=key,
                        channel=channel,
                        chat_id=chat_id,
                        history_window=self.history_window,
                    )
                except asyncio.CancelledError:
                    raise
                except Exception:
                    logger.exception("heartbeat run failed for {}", key)

        await asyncio.gather(*(_one(k) for k in candidates), return_exceptions=True)
