"""Group-chat heartbeat lurking service (SPEC §5.2 item 3).

Wakes up every `interval_seconds`, asks the session manager which group sessions have
opted in via ``/heartbeat on``, and for each one runs a full agent turn (delegated to
``AgentLoop.run_heartbeat``). The agent itself decides whether to chime in (normal reply)
or stay silent (``pass_heartbeat`` tool).

The per-group "last scanned" watermark lives in each session's metadata (persisted in
session.jsonl), so a gateway crash never silently drops messages that arrived between
the last scan and the crash. The in-memory ``_route`` map is allowed to reset on
restart — the channel hook repopulates it on the next inbound message.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    from bubbles.agent.loop import AgentLoop
    from bubbles.session.manager import SessionManager


class GroupHeartbeat:
    def __init__(
        self,
        *,
        agent_loop: "AgentLoop",
        session_manager: "SessionManager",
        interval_seconds: int,
        history_window: int,
        max_concurrent: int,
    ) -> None:
        self.agent_loop = agent_loop
        self.session_manager = session_manager
        self.interval_seconds = max(int(interval_seconds), 30)
        self.history_window = history_window
        self.max_concurrent = max(int(max_concurrent), 1)

        # Per-process state. The watermark itself is persisted in each session's
        # metadata (see AgentLoop.run_heartbeat), so a crash never silently drops
        # messages received before the crash. _route is fine to lose on restart —
        # the channel hook repopulates it on the next inbound message.
        self._route: dict[str, tuple[str, str]] = {}  # session_key -> (channel, chat_id)
        self._stop = asyncio.Event()
        self._task: asyncio.Task | None = None

    # Convenience for /heartbeat status / "interval is X minutes" messaging.
    @property
    def interval_minutes(self) -> int:
        return max(self.interval_seconds // 60, 1)

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
            "GroupHeartbeat started (interval={}s, max_concurrent={}, history_window={})",
            self.interval_seconds, self.max_concurrent, self.history_window,
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
                await asyncio.wait_for(self._stop.wait(), timeout=self.interval_seconds)
                # wait_for returned without timeout => stop requested
                break
            except asyncio.TimeoutError:
                pass  # normal: time to tick
            try:
                await self._tick()
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("heartbeat tick crashed")

    async def _tick(self) -> None:
        enabled = self.session_manager.list_heartbeat_enabled_groups()
        if not enabled:
            logger.debug("heartbeat tick: 0 enabled groups")
            return

        # Only scan groups we have an outbound route for (i.e., have seen at least one
        # message since gateway started). Groups freshly enabled but inactive get picked up
        # on the next inbound message.
        candidates = [k for k in enabled if k in self._route]
        if not candidates:
            logger.debug(
                "heartbeat tick: {} enabled but no routes yet, skip", len(enabled),
            )
            return

        logger.info(
            "heartbeat tick: {} candidate group(s), max_concurrent={}",
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
