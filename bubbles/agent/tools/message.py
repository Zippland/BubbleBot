"""Message tool for sending messages to users."""

import json
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, Awaitable, Callable

from bubbles.agent.tools.base import Tool
from bubbles.bus.events import OutboundMessage

if TYPE_CHECKING:
    from bubbles.session.manager import Session

MESSAGE_TARGET_KEY = "message_target"


def _resolve_media_path(path: str, session_dir: Path | None) -> str:
    """Resolve a media path and keep it inside the current session workspace."""
    raw = (path or "").strip()
    if not raw:
        raise ValueError("Media path must not be empty")
    if session_dir is None:
        raise ValueError("Media attachments require a session workspace")

    # Handle ~ as session_dir
    if raw.startswith("~"):
        raw = str(session_dir / raw[1:].lstrip("/\\"))

    p = Path(raw)
    if not p.is_absolute():
        p = session_dir / p

    resolved = p.resolve()
    workspace = session_dir.resolve()
    try:
        resolved.relative_to(workspace)
    except ValueError as e:
        raise ValueError(f"Media path is outside the session workspace: {path}") from e

    return str(resolved)


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
        self._sent_messages: set[tuple[str, str, tuple[str, ...]]] = set()
        self._session_dir: Path | None = None
        self._session: Session | None = None
        self._save_target: Callable[[], None] | None = None

    @property
    def target(self) -> tuple[str, str]:
        """The selected outbound destination, not necessarily the input source."""
        return self._default_channel, self._default_chat_id

    def bind_session(self, session: "Session", save_target: Callable[[], None]) -> None:
        """Restore the workspace selection without retaining cross-chat reply IDs."""
        self._session = session
        self._save_target = save_target
        saved = session.metadata.get(MESSAGE_TARGET_KEY)
        if isinstance(saved, dict) and all(
            isinstance(saved.get(k), str) and saved[k].strip() for k in ("channel", "chat_id")
        ):
            if self.target != (saved["channel"], saved["chat_id"]):
                self.set_context(saved["channel"], saved["chat_id"])
        else:
            session.metadata[MESSAGE_TARGET_KEY] = dict(zip(("channel", "chat_id"), self.target))

    def switch_target(self, channel: str, chat_id: str) -> None:
        """Persist selection before reporting success; a failed save keeps the old target."""
        previous = (*self.target, self._default_message_id)
        previous_saved = self._session.metadata.get(MESSAGE_TARGET_KEY) if self._session else None
        self.set_context(channel, chat_id)  # Never forward a reply ID from another window.
        try:
            if self._session is not None:
                self._session.metadata[MESSAGE_TARGET_KEY] = {"channel": channel, "chat_id": chat_id}
            if self._save_target is not None:
                self._save_target()
        except Exception:
            self.set_context(*previous)
            if self._session is not None:
                if previous_saved is None:
                    self._session.metadata.pop(MESSAGE_TARGET_KEY, None)
                else:
                    self._session.metadata[MESSAGE_TARGET_KEY] = previous_saved
            raise

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
        self._sent_messages = set()

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
                    "description": "The message content to send. MUST be plain text only - no Markdown formatting (**, #, -, ```, etc.) as chat apps don't render it."
                },
                "media": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional: list of file paths to attach (images, audio, documents)"
                }
            },
            "required": ["content"],
            "additionalProperties": False,
        }

    async def execute(
        self,
        content: str,
        media: list[str] | None = None,
        **kwargs: Any
    ) -> str:
        # Snapshot before awaiting: the receipt must describe this submission.
        channel, chat_id = self.target
        message_id = self._default_message_id

        def receipt(status: str, error: str | None = None) -> str:
            data = {
                "status": status, "channel": channel, "chat_id": chat_id,
            }
            if error:
                data["error"] = error
            return ("Error: " if error else "") + json.dumps(data, ensure_ascii=False)

        if kwargs:
            return receipt("rejected", "message accepts only content/media. Use switch_message_target to change destination.")

        if not channel or not chat_id:
            return receipt("rejected", "No target channel/chat specified; use switch_message_target.")

        if not self._send_callback:
            return receipt("rejected", "Message sending not configured")

        # Resolve and validate media files
        resolved_media: list[str] = []
        if media:
            for f in media:
                try:
                    resolved = _resolve_media_path(f, self._session_dir)
                except ValueError as e:
                    return receipt("rejected", str(e))
                if not os.path.isfile(resolved):
                    return receipt("rejected", f"Media file not found: {f} (resolved to {resolved})")
                resolved_media.append(resolved)

        # Only exact duplicate payloads count. Common prefixes and distinct
        # image-only messages must not block legitimate follow-up replies.
        current_target = f"{channel}:{chat_id}"
        msg_key = (current_target, content, tuple(resolved_media))
        if msg_key in self._sent_messages:
            return receipt("duplicate", f"This message was already submitted to {current_target}. Do not resend it; call stay_silent if the turn is finished.")

        msg = OutboundMessage(
            channel=channel,
            chat_id=chat_id,
            content=content,
            media=resolved_media,
            metadata={
                "message_id": message_id,
                "_agent_message": True,
            }
        )

        try:
            self._sent_messages.add(msg_key)
            await self._send_callback(msg)
            return receipt("submitted")
        except Exception as e:
            self._sent_messages.discard(msg_key)
            return receipt("failed", f"Error sending message: {e}")


class SwitchMessageTargetTool(Tool):
    """Select a destination shared by explicit messages and assistant text."""

    def __init__(self, message: MessageTool, channel_manager: Any = None):
        self._message = message
        self._channel_manager = channel_manager

    @property
    def name(self) -> str:
        return "switch_message_target"

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "channel": {"type": "string", "description": "Destination channel, e.g. wechat or feishu."},
                "chat_id": {"type": "string", "description": "Destination group or user ID on that channel."},
            },
            "required": ["channel", "chat_id"],
            "additionalProperties": False,
        }

    async def execute(self, channel: str, chat_id: str, **kwargs: Any) -> str:
        channel, chat_id = channel.strip(), chat_id.strip()
        try:
            if kwargs or not channel or not chat_id:
                raise ValueError("Provide non-empty channel and chat_id only.")
            if self._channel_manager is not None and self._channel_manager.get_channel(channel) is None:
                raise ValueError(f"Channel '{channel}' is not running.")
            self._message.switch_target(channel, chat_id)
        except Exception as e:
            current_channel, current_chat = self._message.target
            return "Error: " + json.dumps({
                "error": str(e), "channel": current_channel, "chat_id": current_chat,
                "note": "Target unchanged; nothing was sent.",
            }, ensure_ascii=False)
        return json.dumps({
            "status": "selected", "channel": channel, "chat_id": chat_id,
            "note": "All subsequent message calls, assistant text and find_person lookups use this target. Selection persists across turns until switched again. Nothing was sent by this tool.",
        }, ensure_ascii=False)
