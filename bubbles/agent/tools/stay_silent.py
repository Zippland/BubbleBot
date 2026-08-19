"""stay_silent tool: ends the current turn without an outbound reply."""

from typing import Any

from bubbles.agent.tools.base import Tool


# Sentinel returned by the tool. The agent loop detects the exact value, ends the
# turn immediately, and suppresses automatic outbound delivery.
STAY_SILENT_SENTINEL = "[stay-silent]"


class StaySilentTool(Tool):
    """Lets the model explicitly decide that no reply should be sent."""

    @property
    def name(self) -> str:
        return "stay_silent"

    @property
    def parameters(self) -> dict[str, Any]:
        return {"type": "object", "properties": {}, "additionalProperties": False}

    async def execute(self) -> str:
        return STAY_SILENT_SENTINEL
