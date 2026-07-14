"""Tests for the pluggable message bus (Phase 2a).

Covers: LocalBus behavior parity with the historical MessageBus, the wire
round-trip (esp. the datetime field that broke naive json.dumps), the
back-compat shim, and factory selection. RemoteBus (Redis) is not exercised
here — it needs a live Redis and lands functionally in 2c.
"""

import json
from datetime import datetime

import pytest

from bubbles.bus.base import MessageBus
from bubbles.bus.events import InboundMessage, OutboundMessage
from bubbles.bus.factory import make_bus
from bubbles.bus.local import LocalBus
from bubbles.bus.wire import (
    inbound_from_wire,
    inbound_to_wire,
    outbound_from_wire,
    outbound_to_wire,
)


# ---- LocalBus parity ----

async def test_localbus_inbound_roundtrip() -> None:
    bus = LocalBus()
    assert bus.inbound_size == 0
    msg = InboundMessage(channel="cli", sender_id="u", chat_id="c", content="hi")
    await bus.publish_inbound(msg)
    assert bus.inbound_size == 1
    got = await bus.consume_inbound()
    assert got is msg  # in-process passes by reference
    assert bus.inbound_size == 0


async def test_localbus_outbound_roundtrip() -> None:
    bus = LocalBus()
    msg = OutboundMessage(channel="cli", chat_id="c", content="yo")
    await bus.publish_outbound(msg)
    assert bus.outbound_size == 1
    got = await bus.consume_outbound()
    assert got is msg
    assert bus.outbound_size == 0


def test_localbus_is_messagebus() -> None:
    assert isinstance(LocalBus(), MessageBus)


def test_abc_not_instantiable() -> None:
    with pytest.raises(TypeError):
        MessageBus()  # abstract


def test_queue_shim_is_localbus() -> None:
    # Back-compat: `from bubbles.bus.queue import MessageBus` must stay instantiable.
    from bubbles.bus.queue import MessageBus as ShimBus
    b = ShimBus()
    assert isinstance(b, LocalBus)


# ---- wire round-trip ----

def test_inbound_wire_roundtrip_with_datetime() -> None:
    ts = datetime(2026, 7, 14, 10, 0, 0)
    msg = InboundMessage(
        channel="wechat", sender_id="s", chat_id="grp", content="hello",
        timestamp=ts, media=["/x/a.jpg"],
        metadata={"is_group": True, "msg_type": 3, "sender_name": "Bob"},
        session_key_override="wechat:grp",
    )
    wire = inbound_to_wire(msg)
    # Must be valid JSON (the datetime field previously broke json.dumps).
    parsed = json.loads(wire)
    assert parsed["_schema"] == 1 and parsed["_type"] == "inbound"
    assert parsed["timestamp"] == ts.isoformat()

    back = inbound_from_wire(wire)
    assert back.channel == "wechat"
    assert back.content == "hello"
    assert back.timestamp == ts
    assert back.media == ["/x/a.jpg"]
    assert back.metadata == {"is_group": True, "msg_type": 3, "sender_name": "Bob"}
    assert back.session_key == "wechat:grp"  # @property recomputed from override


def test_outbound_wire_roundtrip() -> None:
    msg = OutboundMessage(
        channel="telegram", chat_id="42", content="pong",
        reply_to="7", media=["/tmp/out.pdf"], metadata={"message_id": "m1"},
    )
    back = outbound_from_wire(outbound_to_wire(msg))
    assert back.channel == "telegram"
    assert back.chat_id == "42"
    assert back.content == "pong"
    assert back.reply_to == "7"
    assert back.media == ["/tmp/out.pdf"]
    assert back.metadata == {"message_id": "m1"}


def test_wire_coerces_non_json_metadata() -> None:
    # metadata is dict[str, Any] — a non-JSON value must not break the wire;
    # it's coerced to str with a warning.
    msg = InboundMessage(channel="cli", sender_id="u", chat_id="c", content="x",
                         metadata={"obj": object()})
    wire = inbound_to_wire(msg)  # must not raise
    back = inbound_from_wire(wire)
    assert isinstance(back.metadata["obj"], str)


# ---- factory ----

def test_factory_default_is_localbus() -> None:
    from bubbles.config.schema import Config
    assert isinstance(make_bus(Config()), LocalBus)


def test_factory_none_config_is_localbus() -> None:
    assert isinstance(make_bus(None), LocalBus)


def test_factory_unknown_backend_falls_back_to_local() -> None:
    from bubbles.config.schema import Config
    cfg = Config()
    cfg.bus.default = "bogus"
    assert isinstance(make_bus(cfg), LocalBus)


def test_factory_redis_without_url_errors() -> None:
    from bubbles.config.schema import Config
    cfg = Config()
    cfg.bus.default = "redis"
    cfg.bus.redis_url = ""
    # Either missing redis package or empty url — both are RuntimeError/ValueError,
    # never a silent LocalBus.
    with pytest.raises((RuntimeError, ValueError)):
        make_bus(cfg)
