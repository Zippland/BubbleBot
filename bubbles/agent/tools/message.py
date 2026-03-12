"""Message tool for sending messages to users."""

import os
from pathlib import Path
from typing import Any, Awaitable, Callable

from bubbles.agent.tools.base import Tool
from bubbles.bus.events import OutboundMessage


def _resolve_media_path(path: str, session_dir: Path | None) -> str:
    """Resolve media path: ~ means session_dir, relative paths resolved against it."""
    raw = (path or "").strip()
    if not raw:
        return raw

    # Handle ~ as session_dir
    if raw.startswith("~") and session_dir:
        raw = str(session_dir / raw[1:].lstrip("/\\"))

    p = Path(raw)
    if not p.is_absolute() and session_dir:
        p = session_dir / p

    return str(p.resolve())


class MessageTool(Tool):
    """Tool to send messages to users on chat channels."""

    def __init__(
        self,
        send_callback: Callable[[OutboundMessage], Awaitable[None]] | None = None,
        default_channel: str = "",
        default_chat_id: str = "",
        default_message_id: str | None = None,
    ):
        self._send_callback = send_callback
        self._default_channel = default_channel
        self._default_chat_id = default_chat_id
        self._default_message_id = default_message_id
        self._sent_in_turn: bool = False
        self._sent_messages: set[tuple[str, str]] = set()  # (target, content) pairs sent in this turn
        self._duplicate_detected: bool = False  # Flag for loop detection
        self._session_dir: Path | None = None

    def set_session_dir(self, session_dir: Path | None) -> None:
        """Set session directory for resolving media paths."""
        self._session_dir = session_dir

    def set_context(self, channel: str, chat_id: str, message_id: str | None = None) -> None:
        """Set the current message context."""
        self._default_channel = channel
        self._default_chat_id = chat_id
        self._default_message_id = message_id

    def set_send_callback(self, callback: Callable[[OutboundMessage], Awaitable[None]]) -> None:
        """Set the callback for sending messages."""
        self._send_callback = callback

    def start_turn(self) -> None:
        """Reset per-turn send tracking."""
        self._sent_in_turn = False
        self._sent_messages = set()
        self._duplicate_detected = False

    @property
    def name(self) -> str:
        return "message"

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "content": {
                    "type": "string",
                    "description": "The message content to send"
                },
                "channel": {
                    "type": "string",
                    "description": "Target channel: telegram, feishu, wechat, discord, slack, etc. Defaults to current channel."
                },
                "chat_id": {
                    "type": "string",
                    "description": "Target chat/user ID. Defaults to current chat."
                },
                "media": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional: list of file paths to attach (images, audio, documents)"
                }
            },
            "required": ["content"]
        }

    async def execute(
        self,
        content: str,
        channel: str | None = None,
        chat_id: str | None = None,
        message_id: str | None = None,
        media: list[str] | None = None,
        **kwargs: Any
    ) -> str:
        channel = channel or self._default_channel
        chat_id = chat_id or self._default_chat_id
        message_id = message_id or self._default_message_id

        if not channel or not chat_id:
            return "Error: No target channel/chat specified"

        if not self._send_callback:
            return "Error: Message sending not configured"

        # Check for duplicate message to same target (loop detection)
        current_target = f"{channel}:{chat_id}"
        content_key = content[:100]  # Use first 100 chars as key to detect similar messages
        msg_key = (current_target, content_key)
        if msg_key in self._sent_messages:
            self._duplicate_detected = True
            return f"Error: Duplicate message to {current_target} detected. Stop and wait for user response."
        self._sent_messages.add(msg_key)

        # Resolve and validate media files
        resolved_media: list[str] = []
        if media:
            for f in media:
                resolved = _resolve_media_path(f, self._session_dir)
                if not os.path.isfile(resolved):
                    return f"Error: Media file not found: {f} (resolved to {resolved})"
                resolved_media.append(resolved)

        msg = OutboundMessage(
            channel=channel,
            chat_id=chat_id,
            content=content,
            media=resolved_media,
            metadata={
                "message_id": message_id,
            }
        )

        try:
            await self._send_callback(msg)
            if channel == self._default_channel and chat_id == self._default_chat_id:
                self._sent_in_turn = True
            media_info = f" with {len(media)} attachments" if media else ""
            return f"Message sent to {channel}:{chat_id}{media_info}"
        except Exception as e:
            return f"Error sending message: {str(e)}"
