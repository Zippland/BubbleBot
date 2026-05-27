"""Tool-set policy for system-triggered agent turns.

System-triggered turns (cron jobs, heartbeat ticks) differ from user-driven
turns in two ways:

- they gain ``stay_silent`` so the model can opt out of delivery when the
  triggering condition isn't met (see SPEC §5.6 / §5.7);
- they lose ``cron`` so a triggered turn cannot schedule further jobs.
  Without this restriction a single cron tick could spawn N more jobs, and
  those could each spawn more — there is no good reason for a triggered turn
  to reach back into the scheduler that woke it up (see SPEC §5.6).

Usage::

    with system_triggered_toolset(agent):
        await agent.process_direct(...)
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING, Iterator

if TYPE_CHECKING:
    from bubbles.agent.loop import AgentLoop


@contextmanager
def system_triggered_toolset(agent: "AgentLoop") -> Iterator[None]:
    """Apply the system-triggered turn tool-set for the duration of the block."""
    from bubbles.agent.tools.stay_silent import StaySilentTool

    saved_cron = agent.tools.get("cron")
    agent.tools.register(StaySilentTool())
    if saved_cron is not None:
        agent.tools.unregister("cron")
    try:
        yield
    finally:
        agent.tools.unregister("stay_silent")
        if saved_cron is not None:
            agent.tools.register(saved_cron)
