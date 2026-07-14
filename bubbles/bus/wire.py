"""Wire serialization for bus messages (cross-machine transport).

Converts ``InboundMessage`` / ``OutboundMessage`` to and from a JSON string for
transports that cross a process/host boundary (``RemoteBus``). The in-process
``LocalBus`` never touches this — it passes objects by reference.

Two facts drive this module:
- ``InboundMessage.timestamp`` is a ``datetime`` (events.py) → NOT JSON-native;
  serialized as ISO-8601 and parsed back.
- ``metadata`` is ``dict[str, Any]`` — an unenforced contract. Values MUST be
  JSON primitives; a non-serializable value is coerced to ``str`` with a logged
  warning rather than silently breaking the wire.

Media handling (2a): ``media`` is carried as-is (list of path/URL strings).
Phase 2c replaces raw paths with content-addressed descriptors and rehydrates
at the transport boundary — that change lives here (``_media_to_wire`` /
``_media_from_wire``) so tools and ContextBuilder stay unchanged.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any

from loguru import logger

from bubbles.bus.events import InboundMessage, OutboundMessage

# Bump when the wire format changes incompatibly so a consumer can reject or
# migrate an entry it does not understand.
WIRE_SCHEMA = 1


def _json_safe(value: Any, *, where: str) -> Any:
    """Coerce a metadata value to something JSON-serializable.

    Primitives / lists / dicts pass through; anything else is stringified with
    a warning (the ``dict[str, Any]`` metadata type is not enforced elsewhere).
    """
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, list):
        return [_json_safe(v, where=where) for v in value]
    if isinstance(value, dict):
        return {k: _json_safe(v, where=f"{where}.{k}") for k, v in value.items()}
    logger.warning("Non-JSON metadata value at {} ({}); coercing to str", where, type(value).__name__)
    return str(value)


def _clean_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    return {k: _json_safe(v, where=f"metadata.{k}") for k, v in metadata.items()}


def inbound_to_wire(msg: InboundMessage) -> str:
    """Serialize an InboundMessage to a JSON wire string."""
    payload = {
        "_schema": WIRE_SCHEMA,
        "_type": "inbound",
        "channel": msg.channel,
        "sender_id": msg.sender_id,
        "chat_id": msg.chat_id,
        "content": msg.content,
        "timestamp": msg.timestamp.isoformat(),
        "media": list(msg.media),  # 2c: descriptors
        "metadata": _clean_metadata(msg.metadata),
        "session_key_override": msg.session_key_override,
    }
    return json.dumps(payload, ensure_ascii=False)


def inbound_from_wire(data: str) -> InboundMessage:
    """Deserialize an InboundMessage from a JSON wire string."""
    d = json.loads(data)
    ts = d.get("timestamp")
    return InboundMessage(
        channel=d["channel"],
        sender_id=d["sender_id"],
        chat_id=d["chat_id"],
        content=d["content"],
        timestamp=datetime.fromisoformat(ts) if ts else datetime.now(),
        media=list(d.get("media") or []),
        metadata=d.get("metadata") or {},
        session_key_override=d.get("session_key_override"),
    )
    # session_key is a @property (events.py) — recomputed, never on the wire.


def outbound_to_wire(msg: OutboundMessage) -> str:
    """Serialize an OutboundMessage to a JSON wire string."""
    payload = {
        "_schema": WIRE_SCHEMA,
        "_type": "outbound",
        "channel": msg.channel,
        "chat_id": msg.chat_id,
        "content": msg.content,
        "reply_to": msg.reply_to,
        "media": list(msg.media),  # 2c: descriptors
        "metadata": _clean_metadata(msg.metadata),
    }
    return json.dumps(payload, ensure_ascii=False)


def outbound_from_wire(data: str) -> OutboundMessage:
    """Deserialize an OutboundMessage from a JSON wire string."""
    d = json.loads(data)
    return OutboundMessage(
        channel=d["channel"],
        chat_id=d["chat_id"],
        content=d["content"],
        reply_to=d.get("reply_to"),
        media=list(d.get("media") or []),
        metadata=d.get("metadata") or {},
    )
