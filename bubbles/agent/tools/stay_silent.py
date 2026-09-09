"""End-turn tool, retaining the stable legacy name stay_silent."""

from typing import Any

from bubbles.agent.tools.base import Tool


# Sentinel returned by the tool. The agent loop detects the exact value, ends the
# turn after the current tool batch, without automatic outbound delivery.
STAY_SILENT_SENTINEL = "[stay-silent]"


class StaySilentTool(Tool):
    """Finish a turn, with or without messages already explicitly sent."""

    @property
    def name(self) -> str:
        return "stay_silent"

    @property
    def parameters(self) -> dict[str, Any]:
        return {"type": "object", "properties": {}, "additionalProperties": False}

    async def execute(self) -> str:
        return STAY_SILENT_SENTINEL
