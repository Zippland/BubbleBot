"""pass_heartbeat tool: ends a heartbeat turn without sending any outbound message."""

from typing import Any

from bubbles.agent.tools.base import Tool


# Sentinel returned by the tool; the heartbeat runner detects it to break the agent loop
# AND suppress the outbound channel send.
HEARTBEAT_PASS_SENTINEL = "[heartbeat-pass]"


class PassHeartbeatTool(Tool):
    """Lets the model say 'no, I shouldn't chime in' during a heartbeat scan."""

    @property
    def name(self) -> str:
        return "pass_heartbeat"

    @property
    def parameters(self) -> dict[str, Any]:
        return {"type": "object", "properties": {}}

    async def execute(self) -> str:
        return HEARTBEAT_PASS_SENTINEL
