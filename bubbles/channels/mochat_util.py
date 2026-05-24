"""Mochat data classes + pure helpers, decoupled from MochatChannel.

Pulled out of ``mochat.py`` so the channel class stays focused on
Socket.IO / HTTP fallback / dispatch concerns. Everything here is
state-free and unit-testable.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from bubbles.config.schema import MochatConfig


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class MochatBufferedEntry:
    """Buffered inbound entry for delayed dispatch."""
    raw_body: str
    author: str
    sender_name: str = ""
    sender_username: str = ""
    timestamp: int | None = None
    message_id: str = ""
    group_id: str = ""


@dataclass
class DelayState:
    """Per-target delayed message state."""
    entries: list[MochatBufferedEntry] = field(default_factory=list)
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    timer: asyncio.Task | None = None


@dataclass
class MochatTarget:
    """Outbound target resolution result."""
    id: str
    is_panel: bool


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------

def _safe_dict(value: Any) -> dict:
    """Return *value* if it's a dict, else empty dict."""
    return value if isinstance(value, dict) else {}


def _str_field(src: dict, *keys: str) -> str:
    """Return the first non-empty str value found for *keys*, stripped."""
    for k in keys:
        v = src.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return ""


def _make_synthetic_event(
    message_id: str, author: str, content: Any,
    meta: Any, group_id: str, converse_id: str,
    timestamp: Any = None, *, author_info: Any = None,
) -> dict[str, Any]:
    """Build a synthetic ``message.add`` event dict."""
    payload: dict[str, Any] = {
        "messageId": message_id, "author": author,
        "content": content, "meta": _safe_dict(meta),
        "groupId": group_id, "converseId": converse_id,
    }
    if author_info is not None:
        payload["authorInfo"] = _safe_dict(author_info)
    return {
        "type": "message.add",
        "timestamp": timestamp or datetime.utcnow().isoformat(),
        "payload": payload,
    }


def normalize_mochat_content(content: Any) -> str:
    """Normalize content payload to text."""
    if isinstance(content, str):
        return content.strip()
    if content is None:
        return ""
    try:
        return json.dumps(content, ensure_ascii=False)
    except TypeError:
        return str(content)


def resolve_mochat_target(raw: str) -> MochatTarget:
    """Resolve id and target kind from user-provided target string."""
    trimmed = (raw or "").strip()
    if not trimmed:
        return MochatTarget(id="", is_panel=False)

    lowered = trimmed.lower()
    cleaned, forced_panel = trimmed, False
    for prefix in ("mochat:", "group:", "channel:", "panel:"):
        if lowered.startswith(prefix):
            cleaned = trimmed[len(prefix):].strip()
            forced_panel = prefix in {"group:", "channel:", "panel:"}
            break

    if not cleaned:
        return MochatTarget(id="", is_panel=False)
    return MochatTarget(id=cleaned, is_panel=forced_panel or not cleaned.startswith("session_"))


def extract_mention_ids(value: Any) -> list[str]:
    """Extract mention ids from heterogeneous mention payload."""
    if not isinstance(value, list):
        return []
    ids: list[str] = []
    for item in value:
        if isinstance(item, str):
            if item.strip():
                ids.append(item.strip())
        elif isinstance(item, dict):
            for key in ("id", "userId", "_id"):
                candidate = item.get(key)
                if isinstance(candidate, str) and candidate.strip():
                    ids.append(candidate.strip())
                    break
    return ids


def resolve_was_mentioned(payload: dict[str, Any], agent_user_id: str) -> bool:
    """Resolve mention state from payload metadata and text fallback."""
    meta = payload.get("meta")
    if isinstance(meta, dict):
        if meta.get("mentioned") is True or meta.get("wasMentioned") is True:
            return True
        for f in ("mentions", "mentionIds", "mentionedUserIds", "mentionedUsers"):
            if agent_user_id and agent_user_id in extract_mention_ids(meta.get(f)):
                return True
    if not agent_user_id:
        return False
    content = payload.get("content")
    if not isinstance(content, str) or not content:
        return False
    return f"<@{agent_user_id}>" in content or f"@{agent_user_id}" in content


def resolve_require_mention(config: MochatConfig, session_id: str, group_id: str) -> bool:
    """Resolve mention requirement for group/panel conversations."""
    groups = config.groups or {}
    for key in (group_id, session_id, "*"):
        if key and key in groups:
            return bool(groups[key].require_mention)
    return bool(config.mention.require_in_groups)


def build_buffered_body(entries: list[MochatBufferedEntry], is_group: bool) -> str:
    """Build text body from one or more buffered entries."""
    if not entries:
        return ""
    if len(entries) == 1:
        return entries[0].raw_body
    lines: list[str] = []
    for entry in entries:
        if not entry.raw_body:
            continue
        if is_group:
            label = entry.sender_name.strip() or entry.sender_username.strip() or entry.author
            if label:
                lines.append(f"{label}: {entry.raw_body}")
                continue
        lines.append(entry.raw_body)
    return "\n".join(lines).strip()


def parse_timestamp(value: Any) -> int | None:
    """Parse event timestamp to epoch milliseconds."""
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        return int(datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp() * 1000)
    except ValueError:
        return None
