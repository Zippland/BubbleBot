"""Cross-channel @ mention markup.

Agents emit text containing `<@id>` markers; each channel translates these to
its native @-mention format on send. Inbound messages convert the other way so
the model always sees `<@id>` regardless of channel.

`id` is the channel-native user identifier (wxid for WeChat, open_id for Feishu).
"""

from __future__ import annotations

import re
from typing import Callable

# Allow letters, digits, underscore, hyphen — covers wxid (wxid_xxx),
# Feishu open_id (ou_xxx), Telegram user IDs (numeric), etc.
MENTION_RE = re.compile(r"<@([\w\-]+)>")


def extract_mentions(text: str | None) -> list[str]:
    """Ordered list of ids referenced by `<@id>` markers, with duplicates preserved."""
    return MENTION_RE.findall(text or "")


def split_mention_text(text: str | None) -> list[tuple[str, str]]:
    """Split text into ordered segments: ('text', str) | ('mention', id).

    Empty text segments are omitted. A bare `<@id>` produces only mention segments.
    """
    if not text:
        return []
    segments: list[tuple[str, str]] = []
    pos = 0
    for m in MENTION_RE.finditer(text):
        if m.start() > pos:
            segments.append(("text", text[pos:m.start()]))
        segments.append(("mention", m.group(1)))
        pos = m.end()
    if pos < len(text):
        segments.append(("text", text[pos:]))
    return segments


def replace_mentions(text: str | None, fn: Callable[[str], str]) -> str:
    """Replace each `<@id>` marker with `fn(id)`."""
    return MENTION_RE.sub(lambda m: fn(m.group(1)), text or "")
