"""Context builder for assembling agent prompts."""

import base64
import mimetypes
import platform
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from bubbles.agent.skills import SkillsLoader


class ContextBuilder:
    """
    Builds the context (system prompt + messages) for the agent.

    Each session is an independent workspace. All files (SOUL.md, MEMORY.md,
    skills/, etc.) are loaded from the session directory.
    """

    # Bootstrap files loaded from session directory
    BOOTSTRAP_FILES = ["SOUL.md", "MEMORY.md"]

    def __init__(self, session_dir: Path):
        """Initialize context builder.

        Args:
            session_dir: Session directory containing all agent files.
        """
        self.session_dir = session_dir
        self.skills = SkillsLoader(session_dir)
    
    def build_system_prompt(self, skill_names: list[str] | None = None) -> str:
        """Build the system prompt from identity, bootstrap files, and skills."""
        parts = [self._get_identity()]

        bootstrap = self._load_bootstrap_files()
        if bootstrap:
            parts.append(bootstrap)

        always_skills = self.skills.get_always_skills()
        if always_skills:
            always_content = self.skills.load_skills_for_context(always_skills)
            if always_content:
                parts.append(f"# Active Skills\n\n{always_content}")

        skills_summary = self.skills.build_skills_summary()
        if skills_summary:
            parts.append(f"# Skills\n\n{skills_summary}")

        return "\n\n---\n\n".join(parts)
    
    def _get_identity(self) -> str:
        """Get the core identity section."""
        session_path = str(self.session_dir.expanduser().resolve())
        system = platform.system()
        runtime = f"{'macOS' if system == 'Darwin' else system} {platform.machine()}, Python {platform.python_version()}"

        return f"""# IDENTITY -- Who am I?

- **Name:** 泡泡（bubbles）
- **Creator:** Zylan
- **Vibe:** 诚实守信、温暖阳光、不偏不倚、乐于助人
- **Emoji:** 🫧

---

# ENVIROMENT -- Where am I?

## Runtime
{runtime}

## Session Workspace
<work_dir>: {session_path}
- Long-term memory: <work_dir>/MEMORY.md（editable）
- All Message history: <work_dir>/session.jsonl (grep-searchable)
- Data Storage: <work_dir>/data/ (editable)
- Custom skills: <work_dir>/skills/{{skill-name}}/SKILL.md

**<work_dir> is the only place you and your human can visit, edit, create or delete files.**

# INSTRUCTIONS

# File Management
- All file operations must be performed within <work_dir>/data. Do not create or download files anywhere else.
- If the /data directory does not exist, create it before use.
- Temporary and one-off intermediate files must be deleted immediately after use. Keep the /data directory clean.
- Before writing a file, confirm the target path to avoid accidentally overwriting existing files.

## Message Context
Reply directly with text for conversations. Only use the 'message' tool to send to a specific channel.
Each user message includes a [Runtime Context] block at the end with:
- Current time and timezone
- Channel and chat ID
- Sender information (name and ID when available)

## Tool Call Guidelines
- Before calling tools, you may briefly state your intent, but NEVER predict the result.
- Before modifying a file, read it first.
- Do not assume a file exists — use list_dir or read_file to verify.
- If a tool call fails, analyze the error before retrying.

## Memory
Write important facts immediately using `edit_file` or `write_file` in "<work_dir>/MEMORY.md":
- User preferences ("I prefer dark mode")
- Project context ("The API uses OAuth2")
- Relationships ("Alice is the project lead")
Keep it concise — only facts you'll need to recall later."""

    @staticmethod
    def _inject_runtime_context(
        user_content: str | list[dict[str, Any]],
        channel: str | None,
        chat_id: str | None,
        sender_id: str | None = None,
        sender_name: str | None = None,
    ) -> str | list[dict[str, Any]]:
        """Append dynamic runtime context to the tail of the user message."""
        now = datetime.now().strftime("%Y-%m-%d %H:%M (%A)")
        tz = time.strftime("%Z") or "UTC"
        lines = [f"Time: {now} ({tz})"]
        if channel and chat_id:
            lines += [f"Channel: {channel}", f"Chat ID: {chat_id}"]
        if sender_name:
            lines.append(f"Sender: {sender_name} ({sender_id})")
        elif sender_id:
            lines.append(f"Sender ID: {sender_id}")
        block = "[Runtime Context]\n" + "\n".join(lines)
        if isinstance(user_content, str):
            return f"{user_content}\n\n{block}"
        return [*user_content, {"type": "text", "text": block}]
    
    def _load_bootstrap_files(self) -> str:
        """Load bootstrap files from session directory."""
        parts = []

        for filename in self.BOOTSTRAP_FILES:
            file_path = self.session_dir / filename
            if file_path.exists():
                content = file_path.read_text(encoding="utf-8").strip()
                if content:
                    parts.append(f"## {filename} (<work_dir>/{filename})\n\n{content}")

        return "\n\n---\n\n".join(parts) if parts else ""
    
    def build_messages(
        self,
        history: list[dict[str, Any]],
        current_message: str,
        skill_names: list[str] | None = None,
        media: list[str] | None = None,
        channel: str | None = None,
        chat_id: str | None = None,
        sender_id: str | None = None,
        sender_name: str | None = None,
        system_prompt_extra: str | None = None,
        session_bindings: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Build the complete message list for an LLM call.

        Args:
            history: Previous conversation messages.
            current_message: The new user message.
            skill_names: Optional skills to include.
            media: Optional list of local file paths for images/media.
            channel: Current channel (telegram, feishu, etc.).
            chat_id: Current chat/user ID.
            sender_id: ID of the message sender.
            sender_name: Display name of the sender.
            system_prompt_extra: Optional extra system prompt to append.
            session_bindings: List of channel:chat_id bound to this session.

        Returns:
            List of messages including system prompt.
        """
        messages = []

        # System prompt
        system_prompt = self.build_system_prompt(skill_names)

        # Add session bindings info
        if session_bindings:
            bindings_list = "\n".join(f"- {b}" for b in session_bindings)
            system_prompt += f"\n\n## Connected Channels\n\nThis session is connected to the following channels. You can send messages to any of them using the `message` tool:\n\n{bindings_list}"

        if system_prompt_extra:
            system_prompt = f"{system_prompt}\n\n# Session Instructions\n\n{system_prompt_extra}"
        messages.append({"role": "system", "content": system_prompt})

        # History
        messages.extend(history)

        # Current message (with optional image attachments)
        user_content = self._build_user_content(current_message, media)
        user_content = self._inject_runtime_context(
            user_content, channel, chat_id, sender_id, sender_name
        )
        messages.append({"role": "user", "content": user_content})

        return messages

    def _build_user_content(self, text: str, media: list[str] | None) -> str | list[dict[str, Any]]:
        """Build user message content with optional base64-encoded images."""
        if not media:
            return text

        images = []
        for path in media:
            p = Path(path)
            if not p.is_file():
                continue
            mime, _ = mimetypes.guess_type(path)
            if not mime or not mime.startswith("image/"):
                continue
            b64 = base64.b64encode(p.read_bytes()).decode()
            images.append({"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}})

        if not images:
            return text
        return images + [{"type": "text", "text": text}]
    
    def add_tool_result(
        self, messages: list[dict[str, Any]],
        tool_call_id: str, tool_name: str, result: str | list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Add a tool result to the message list.

        Args:
            result: Can be a string or a list containing text/image_url elements
        """
        messages.append({"role": "tool", "tool_call_id": tool_call_id, "name": tool_name, "content": result})
        return messages
    
    def add_assistant_message(
        self, messages: list[dict[str, Any]],
        content: str | None,
        tool_calls: list[dict[str, Any]] | None = None,
        reasoning_content: str | None = None,
    ) -> list[dict[str, Any]]:
        """Add an assistant message to the message list."""
        msg: dict[str, Any] = {"role": "assistant", "content": content}
        if tool_calls:
            msg["tool_calls"] = tool_calls
        if reasoning_content is not None:
            msg["reasoning_content"] = reasoning_content
        messages.append(msg)
        return messages
