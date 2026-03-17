"""Session management for conversation history."""

import json
import shutil
from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from loguru import logger

from bubbles.utils.helpers import ensure_dir, safe_filename


def _is_compaction_marker(msg: dict[str, Any]) -> bool:
    """Check if a message is a compaction marker."""
    return msg.get("_type") == "compaction"


# Default number of recent images to retain when building history for LLM
DEFAULT_KEEP_LAST_IMAGES = 10

# Placeholder for pruned images
PRUNED_IMAGE_MARKER = "[image data removed - already processed by model]"


def prune_old_images_inplace(
    messages: list[dict[str, Any]],
    keep_last_n: int = DEFAULT_KEEP_LAST_IMAGES,
) -> bool:
    """
    原地清理旧图片，仅保留最近 N 张。

    应在 Agent loop 入口处显式调用，清理上一轮对话的旧图片。
    当前轮对话的图片在 loop 过程中加入，不会被清理。

    Args:
        messages: 消息列表，将被原地修改。
        keep_last_n: 保留的最近图片数量。

    Returns:
        True 如果有图片被清理，否则 False。
    """
    if keep_last_n < 0:
        keep_last_n = 0

    # 收集所有图片位置: (message_index, block_index)
    image_locations: list[tuple[int, int]] = []

    for msg_idx, msg in enumerate(messages):
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        for block_idx, block in enumerate(content):
            if isinstance(block, dict) and block.get("type") == "image_url":
                image_locations.append((msg_idx, block_idx))

    # 计算需要清理的图片数量
    prune_count = max(0, len(image_locations) - keep_last_n)
    if prune_count == 0:
        return False

    # 原地清理最旧的图片
    for i in range(prune_count):
        msg_idx, block_idx = image_locations[i]
        content = messages[msg_idx].get("content")
        if isinstance(content, list) and block_idx < len(content):
            content[block_idx] = {"type": "text", "text": PRUNED_IMAGE_MARKER}

    return True


@dataclass
class SessionConfig:
    """Session-level configuration overrides.

    When set, these values override the global config for this session only.
    None means use global default.
    """
    provider: str | None = None
    model: str | None = None
    temperature: float | None = None
    max_tokens: int | None = None
    system_prompt: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dict, excluding None values."""
        return {k: v for k, v in {
            "provider": self.provider,
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "system_prompt": self.system_prompt,
        }.items() if v is not None}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SessionConfig":
        """Create from dict."""
        return cls(
            provider=data.get("provider"),
            model=data.get("model"),
            temperature=data.get("temperature"),
            max_tokens=data.get("max_tokens"),
            system_prompt=data.get("system_prompt"),
        )


@dataclass
class Session:
    """
    A conversation session.

    Stores messages in JSONL format for easy reading and persistence.

    Messages are append-only. Compaction markers (_type="compaction") can be
    inserted to summarize old messages. get_history() returns only messages
    after the last compaction marker, with the summary injected as a system message.
    """

    key: str  # channel:chat_id
    messages: list[dict[str, Any]] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    metadata: dict[str, Any] = field(default_factory=dict)
    config: SessionConfig = field(default_factory=SessionConfig)  # Session-specific config
    directory: Path | None = None  # Session directory (set by SessionManager)

    def add_message(self, role: str, content: str, **kwargs: Any) -> None:
        """Add a message to the session."""
        msg = {
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat(),
            **kwargs
        }
        self.messages.append(msg)
        self.updated_at = datetime.now()

    def get_history(self, max_messages: int = 500) -> list[dict[str, Any]]:
        """返回供 LLM 使用的消息历史（纯读取，无副作用）。

        从最后一个 compaction marker 之后开始返回消息。
        如果存在 compaction marker，会将摘要作为 system 消息注入。

        注意：图片清理应在调用此方法前由调用方显式执行。
        """
        # Find the last compaction marker
        last_compaction_idx = -1
        compaction_summary = None
        for i in range(len(self.messages) - 1, -1, -1):
            if _is_compaction_marker(self.messages[i]):
                last_compaction_idx = i
                compaction_summary = self.messages[i].get("summary", "")
                break

        # Get messages after the compaction marker
        start_idx = last_compaction_idx + 1 if last_compaction_idx >= 0 else 0
        active = self.messages[start_idx:]

        # Filter out any compaction markers and limit
        active = [m for m in active if not _is_compaction_marker(m)]
        sliced = active[-max_messages:]

        # Drop leading non-user messages to avoid orphaned tool_result blocks
        for i, m in enumerate(sliced):
            if m.get("role") == "user":
                sliced = sliced[i:]
                break

        out: list[dict[str, Any]] = []

        # Inject compaction summary as system message
        if compaction_summary:
            out.append({
                "role": "system",
                "content": f"## Previous conversation summary:\n{compaction_summary}",
            })

        for m in sliced:
            entry: dict[str, Any] = {"role": m["role"], "content": m.get("content", "")}
            for k in ("tool_calls", "tool_call_id", "name"):
                if k in m:
                    entry[k] = m[k]
            out.append(entry)

        return out

    def clear(self) -> None:
        """Clear all messages and reset session to initial state."""
        self.messages = []
        self.updated_at = datetime.now()

    # ---- Task 操作方法 (Claude Code style) ----

    def _next_task_id(self) -> str:
        """生成下一个任务 ID（包括已删除的，确保 ID 不复用）。"""
        all_tasks = self.metadata.get("tasks", [])
        if not all_tasks:
            return "1"
        max_id = max(int(t.get("id", 0)) for t in all_tasks)
        return str(max_id + 1)

    def get_tasks(self) -> list[dict[str, Any]]:
        """获取所有任务（排除已删除的）。"""
        tasks = self.metadata.get("tasks", [])
        return [t for t in tasks if t.get("status") != "deleted"]

    def get_task(self, task_id: str) -> dict[str, Any] | None:
        """根据 ID 获取单个任务。"""
        for t in self.get_tasks():
            if t.get("id") == task_id:
                return t
        return None

    def create_task(
        self,
        subject: str,
        description: str,
        active_form: str | None = None,
    ) -> dict[str, Any]:
        """创建新任务。"""
        task = {
            "id": self._next_task_id(),
            "subject": subject,
            "description": description,
            "status": "pending",
            "active_form": active_form or "",
            "blocks": [],
            "blocked_by": [],
        }
        tasks = self.metadata.get("tasks", [])
        tasks.append(task)
        self.metadata["tasks"] = tasks
        self.updated_at = datetime.now()
        return task

    def update_task(
        self,
        task_id: str,
        status: str | None = None,
        subject: str | None = None,
        description: str | None = None,
        active_form: str | None = None,
        add_blocks: list[str] | None = None,
        add_blocked_by: list[str] | None = None,
    ) -> dict[str, Any] | None:
        """更新任务。"""
        tasks = self.metadata.get("tasks", [])
        for task in tasks:
            if task.get("id") == task_id:
                if status is not None:
                    task["status"] = status
                if subject is not None:
                    task["subject"] = subject
                if description is not None:
                    task["description"] = description
                if active_form is not None:
                    task["active_form"] = active_form
                if add_blocks:
                    existing = set(task.get("blocks", []))
                    existing.update(add_blocks)
                    task["blocks"] = list(existing)
                if add_blocked_by:
                    existing = set(task.get("blocked_by", []))
                    existing.update(add_blocked_by)
                    task["blocked_by"] = list(existing)
                self.metadata["tasks"] = tasks
                self.updated_at = datetime.now()
                return task
        return None


class SessionManager:
    """
    Manages conversation sessions.

    Each session is an independent workspace containing:
    - session.jsonl: message history
    - SOUL.md: persona/personality
    - MEMORY.md: long-term memory
    - skills/: session-specific skills
    """

    def __init__(self, sessions_dir: Path | None = None):
        from bubbles.utils.helpers import get_sessions_path
        self.sessions_dir = sessions_dir or get_sessions_path()
        self._cache: dict[str, Session] = {}

    def _get_session_dir(self, key: str) -> Path:
        """Get the directory for a session."""
        safe_key = safe_filename(key.replace(":", "_"))
        return ensure_dir(self.sessions_dir / safe_key)

    def _get_session_path(self, key: str) -> Path:
        """Get the file path for a session's messages."""
        return self._get_session_dir(key) / "session.jsonl"
    
    def get_or_create(self, key: str) -> Session:
        """
        Get an existing session or create a new one.

        Args:
            key: Session key (usually channel:chat_id).

        Returns:
            The session.
        """
        if key in self._cache:
            return self._cache[key]

        session = self._load(key)
        is_new = session is None
        if is_new:
            session = Session(key=key)

        # All sessions get their own directory with SOUL.md, MEMORY.md, skills/
        session.directory = self._get_session_dir(key)
        if is_new:
            self._init_session_dir(session.directory)

        self._cache[key] = session
        return session

    def _init_session_dir(self, session_dir: Path) -> None:
        """Initialize session directory by copying the session template."""
        from importlib.resources import files as pkg_files, as_file

        template_dir = pkg_files("bubbles.templates") / "session"
        with as_file(template_dir) as src:
            shutil.copytree(src, session_dir, dirs_exist_ok=True)
    
    def _load(self, key: str) -> Session | None:
        """Load a session from disk."""
        path = self._get_session_path(key)

        if not path.exists():
            return None

        try:
            messages = []
            metadata = {}
            created_at = None
            config = SessionConfig()

            with open(path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue

                    data = json.loads(line)

                    if data.get("_type") == "metadata":
                        metadata = data.get("metadata", {})
                        created_at = datetime.fromisoformat(data["created_at"]) if data.get("created_at") else None
                        if "config" in data:
                            config = SessionConfig.from_dict(data["config"])
                    else:
                        messages.append(data)

            return Session(
                key=key,
                messages=messages,
                created_at=created_at or datetime.now(),
                metadata=metadata,
                config=config,
            )
        except Exception as e:
            logger.warning("Failed to load session {}: {}", key, e)
            return None
    
    def save(self, session: Session) -> None:
        """Save a session to disk."""
        path = self._get_session_path(session.key)

        with open(path, "w", encoding="utf-8") as f:
            metadata_line = {
                "_type": "metadata",
                "key": session.key,
                "created_at": session.created_at.isoformat(),
                "updated_at": session.updated_at.isoformat(),
                "metadata": session.metadata,
                "config": session.config.to_dict(),
            }
            f.write(json.dumps(metadata_line, ensure_ascii=False) + "\n")
            for msg in session.messages:
                f.write(json.dumps(msg, ensure_ascii=False) + "\n")

        self._cache[session.key] = session
    
    def invalidate(self, key: str) -> None:
        """Remove a session from the in-memory cache."""
        self._cache.pop(key, None)
    
    def list_sessions(self) -> list[dict[str, Any]]:
        """
        List all sessions.
        
        Returns:
            List of session info dicts.
        """
        sessions = []
        
        for path in self.sessions_dir.glob("*.jsonl"):
            try:
                # Read just the metadata line
                with open(path, encoding="utf-8") as f:
                    first_line = f.readline().strip()
                    if first_line:
                        data = json.loads(first_line)
                        if data.get("_type") == "metadata":
                            key = data.get("key") or path.stem.replace("_", ":", 1)
                            sessions.append({
                                "key": key,
                                "created_at": data.get("created_at"),
                                "updated_at": data.get("updated_at"),
                                "path": str(path)
                            })
            except Exception:
                continue
        
        return sorted(sessions, key=lambda x: x.get("updated_at", ""), reverse=True)
