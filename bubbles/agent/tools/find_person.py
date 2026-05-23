"""Tool: search a group member by name to get a `<@id>` mention marker."""

from __future__ import annotations

from typing import Any

from bubbles.agent.tools.base import Tool

MAX_MATCHES_SHOWN = 10


class FindPersonTool(Tool):
    """Look up a chat member by name, return a `<@id>` marker the agent can embed."""

    def __init__(self) -> None:
        self._channel: str = ""
        self._chat_id: str = ""
        self._channel_manager: Any = None

    def set_context(self, channel: str, chat_id: str) -> None:
        self._channel = channel
        self._chat_id = chat_id

    def set_channel_manager(self, manager: Any) -> None:
        self._channel_manager = manager

    @property
    def name(self) -> str:
        return "find_person"

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Name fragment to search (case-insensitive substring).",
                },
            },
            "required": ["query"],
        }

    async def execute(self, query: str = "", **kwargs: Any) -> str:
        q = (query or "").strip()
        if not q:
            return "Error: `query` is required."
        if not self._channel_manager:
            return "Error: channel manager not configured."
        if not self._channel or not self._chat_id:
            return "Error: no chat context — only callable inside a chat session."

        channel = self._channel_manager.get_channel(self._channel)
        if channel is None:
            return f"Error: channel '{self._channel}' is not running."

        members = await channel.get_group_members(self._chat_id)
        if not members:
            return "No group members available (this isn't a group, or the roster is empty)."

        q_lower = q.lower()

        def _matches(m: dict) -> bool:
            for v in (m.get("names") or {}).values():
                if v and q_lower in str(v).lower():
                    return True
            return False

        matches = [m for m in members if _matches(m)]
        if not matches:
            return f"No member matched '{q}' (searched {len(members)} member(s))."

        total = len(matches)
        shown = matches[:MAX_MATCHES_SHOWN]
        tail = (
            f"\n({total - MAX_MATCHES_SHOWN} more matches truncated — refine the query)"
            if total > MAX_MATCHES_SHOWN else ""
        )

        lines = [
            f"Found {total} match{'es' if total > 1 else ''}. "
            "Embed the `<@id>` marker shown below into your reply text to @ them."
        ]
        for i, m in enumerate(shown, 1):
            lines.append("")
            lines.append(f"#{i}  <@{m['id']}>   ← put this in your reply")
            for label, value in (m.get("names") or {}).items():
                lines.append(f"    {label}: {value}")
        return "\n".join(lines) + tail
