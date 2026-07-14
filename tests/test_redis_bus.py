"""RemoteBus integration tests over fakeredis (Phase 2c).

Gated on fakeredis — skipped where it's not installed (e.g. this dev machine
without pip access), run in CI or once `pip install bubbles[redis] fakeredis`.
These prove the Redis lanes end-to-end in-memory; true cross-machine + real
WeChat verification is the deployment-side recipe in the plan.
"""

import asyncio

import pytest

fakeredis = pytest.importorskip("fakeredis")

from bubbles.bus.events import InboundMessage, OutboundMessage  # noqa: E402
from bubbles.bus.redis_bus import RemoteBus  # noqa: E402


@pytest.fixture
def patched_bus(monkeypatch):
    """A RemoteBus whose redis client is a fresh in-memory fakeredis."""
    import fakeredis.aioredis

    def _make(url, **kwargs):
        bus = RemoteBus("redis://fake", **kwargs)
        # share ONE fake server so two RemoteBus instances see the same data
        bus._shared = getattr(_make, "_server", None) or fakeredis.aioredis.FakeRedis()
        _make._server = bus._shared

        async def _redis():
            return bus._shared
        bus._redis = _redis  # type: ignore
        return bus

    return _make


async def test_inbound_roundtrip(patched_bus) -> None:
    make = patched_bus
    channels = make("redis://fake", consumer_name="channels")
    harness = make("redis://fake", consumer_name="harness")
    msg = InboundMessage(channel="wechat", sender_id="s", chat_id="c", content="hi")
    await channels.publish_inbound(msg)
    got = await asyncio.wait_for(harness.consume_inbound(), timeout=3)
    assert got.channel == "wechat" and got.content == "hi"


async def test_outbound_roundtrip(patched_bus) -> None:
    make = patched_bus
    harness = make("redis://fake", consumer_name="harness")
    channels = make("redis://fake", consumer_name="channels")
    await harness.publish_outbound(OutboundMessage(channel="wechat", chat_id="c", content="yo"))
    got = await asyncio.wait_for(channels.consume_outbound(), timeout=3)
    assert got.content == "yo"


async def test_blob_put_get(patched_bus) -> None:
    bus = patched_bus("redis://fake")
    sha = await bus.put_blob(b"payload-bytes")
    assert await bus.get_blob(sha) == b"payload-bytes"


async def test_blob_missing_raises(patched_bus) -> None:
    bus = patched_bus("redis://fake")
    with pytest.raises(KeyError):
        await bus.get_blob("deadbeef")


async def test_inbound_media_crosses_via_blob(patched_bus, tmp_path, monkeypatch) -> None:
    # Force blob path (tiny inline threshold) and confirm the harness side
    # rehydrates the bytes into a local file.
    make = patched_bus
    channels = make("redis://fake", consumer_name="channels", inline_media_max_bytes=1)
    harness = make("redis://fake", consumer_name="harness", inline_media_max_bytes=1)

    # channel-side source file
    src = tmp_path / "img.jpg"
    src.write_bytes(b"\xff\xd8imagebytes")

    # redirect the harness landing dir into tmp
    dest = tmp_path / "harness_data"
    monkeypatch.setattr(harness, "_inbound_dest", lambda msg: dest)

    await channels.publish_inbound(
        InboundMessage(channel="wechat", sender_id="s", chat_id="c",
                       content="see pic", media=[str(src)])
    )
    got = await asyncio.wait_for(harness.consume_inbound(), timeout=3)
    assert len(got.media) == 1
    from pathlib import Path
    assert Path(got.media[0]).read_bytes() == b"\xff\xd8imagebytes"
    assert Path(got.media[0]).parent == dest


async def test_rpc_roundtrip(patched_bus) -> None:
    make = patched_bus
    harness = make("redis://fake", consumer_name="harness")
    channels = make("redis://fake", consumer_name="channels")

    async def handler(verb, payload):
        assert verb == "roster"
        return [{"id": "u1", "names": {"nick": "Bob"}}]

    server = asyncio.create_task(channels.serve_rpc(handler))
    try:
        result = await asyncio.wait_for(
            harness.request("roster", {"channel": "wechat", "chat_id": "g"}, timeout=3),
            timeout=5,
        )
        assert result == [{"id": "u1", "names": {"nick": "Bob"}}]
    finally:
        server.cancel()


async def test_rpc_timeout_when_no_server(patched_bus) -> None:
    harness = patched_bus("redis://fake", consumer_name="harness")
    with pytest.raises(TimeoutError):
        await harness.request("roster", {"channel": "x", "chat_id": "y"}, timeout=1)
