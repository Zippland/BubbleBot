"""stay_silent tool: ends a system-triggered turn without sending any outbound message."""

from typing import Any

from bubbles.agent.tools.base import Tool


# Sentinel returned by the tool. Any periodic / system-triggered turn runner (heartbeat
# scan, cron tick, future system events) detects this prefix to break the agent loop
# AND suppress the outbound channel send.
STAY_SILENT_SENTINEL = "[stay-silent]"


class StaySilentTool(Tool):
    """Lets the model say 'no, I shouldn't act' during a system-triggered turn."""

    @property
    def name(self) -> str:
        return "stay_silent"

    @property
    def parameters(self) -> dict[str, Any]:
        return {"type": "object", "properties": {}}

    async def execute(self) -> str:
        return STAY_SILENT_SENTINEL
