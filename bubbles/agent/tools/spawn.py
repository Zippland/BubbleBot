"""Spawn tool for creating background subagents."""

from pathlib import Path
from typing import Any, TYPE_CHECKING

from bubbles.agent.tools.base import Tool

if TYPE_CHECKING:
    from bubbles.agent.subagent import SubagentManager


class SpawnTool(Tool):
    """Tool to spawn a subagent for background task execution."""

    def __init__(self, manager: "SubagentManager"):
        self._manager = manager
        self._origin_channel = "cli"
        self._origin_chat_id = "direct"
        self._session_key = "cli:direct"
        self._session_dir: Path | None = None

    def set_context(self, channel: str, chat_id: str, session_key: str | None = None) -> None:
        """Set the origin context for subagent announcements."""
        self._origin_channel = channel
        self._origin_chat_id = chat_id
        self._session_key = session_key or f"{channel}:{chat_id}"

    def set_session_dir(self, session_dir: Path | None) -> None:
        """Set session directory for subagent file operations."""
        self._session_dir = session_dir
    
    @property
    def name(self) -> str:
        return "spawn"

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "task": {
                    "type": "string",
                    "description": "The task for the subagent to complete",
                },
                "label": {
                    "type": "string",
                    "description": "Optional short label for the task (for display)",
                },
            },
            "required": ["task"],
        }
    
    async def execute(self, task: str, label: str | None = None, **kwargs: Any) -> str:
        """Spawn a subagent to execute the given task."""
        return await self._manager.spawn(
            task=task,
            label=label,
            origin_channel=self._origin_channel,
            origin_chat_id=self._origin_chat_id,
            session_key=self._session_key,
            session_dir=self._session_dir,
        )
