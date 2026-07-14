"""Cross-machine message bus over Redis Streams (Phase 2 split deployment).

Lets the channel host (Windows/wcferry) and the harness host (Linux) run as
separate processes that talk over a shared Redis. This is the ``bus.default =
redis`` transport; the default ``local`` transport (LocalBus) is unaffected.

**Scope of this file right now (2a): inbound/outbound lanes only.** The RPC lane
(find_person roster) and the media blob store are Phase 2c — stubbed here with
``NotImplementedError`` so the surface is visible but split mode isn't claimed
to work until 2c lands.

Reliability note: the ``MessageBus.consume_*`` contract returns a bare message
with no ack handle, so this skeleton auto-acks on read (at-most-once). 2c moves
the XACK to *after* ``save_turn`` (loop.py) for at-least-once + ``message_id``
dedupe — that needs a small extension to carry the stream entry id.
"""

from __future__ import annotations

from typing import Any

from loguru import logger

from bubbles.bus.base import MessageBus
from bubbles.bus.events import InboundMessage, OutboundMessage
from bubbles.bus.wire import (
    inbound_from_wire,
    inbound_to_wire,
    outbound_from_wire,
    outbound_to_wire,
)

# Stream keys + consumer groups. Inbound = channels→harness; outbound =
# harness→channels. Groups let a restarted consumer reclaim pending entries.
STREAM_INBOUND = "bus:inbound"
STREAM_OUTBOUND = "bus:outbound"
GROUP_HARNESS = "harness"    # consumes inbound
GROUP_CHANNELS = "channels"  # consumes outbound

# Bounded block so asyncio.wait_for(timeout=1.0) can cancel the read.
_BLOCK_MS = 1000


def _require_redis():
    try:
        import redis.asyncio as aioredis  # type: ignore
    except ImportError as e:
        raise RuntimeError(
            "The 'redis' backend requires the redis package. "
            "Install it with: pip install 'redis>=5.0'"
        ) from e
    return aioredis


class RemoteBus(MessageBus):
    """Redis Streams transport. One instance per host (channels or harness)."""

    def __init__(self, redis_url: str, *, consumer_name: str = "default"):
        if not redis_url:
            raise ValueError("RemoteBus requires a redis_url (bus.redis_url in config)")
        self._url = redis_url
        self._consumer = consumer_name
        self._aioredis = _require_redis()
        self._client: Any = None
        self._groups_ready = False

    # ---- connection / group setup ----

    async def _redis(self) -> Any:
        if self._client is None:
            self._client = self._aioredis.from_url(self._url)
        return self._client

    async def _ensure_group(self, stream: str, group: str) -> None:
        """Create the consumer group (idempotent; ignores BUSYGROUP)."""
        r = await self._redis()
        try:
            await r.xgroup_create(name=stream, groupname=group, id="0", mkstream=True)
        except Exception as e:  # redis.exceptions.ResponseError BUSYGROUP
            if "BUSYGROUP" not in str(e):
                raise

    # ---- inbound lane (channels → harness) ----

    async def publish_inbound(self, msg: InboundMessage) -> None:
        r = await self._redis()
        await r.xadd(STREAM_INBOUND, {"data": inbound_to_wire(msg)})

    async def consume_inbound(self) -> InboundMessage:
        return await self._consume(STREAM_INBOUND, GROUP_HARNESS, inbound_from_wire)

    # ---- outbound lane (harness → channels) ----

    async def publish_outbound(self, msg: OutboundMessage) -> None:
        r = await self._redis()
        await r.xadd(STREAM_OUTBOUND, {"data": outbound_to_wire(msg)})

    async def consume_outbound(self) -> OutboundMessage:
        return await self._consume(STREAM_OUTBOUND, GROUP_CHANNELS, outbound_from_wire)

    async def _consume(self, stream: str, group: str, parse):
        """Block (bounded, so wait_for can cancel) until one entry arrives.

        Auto-acks on read (2a skeleton). Loops on empty reads so the caller's
        ``asyncio.wait_for(timeout=1.0)`` cancels cleanly between BLOCKs.
        """
        if not self._groups_ready:
            await self._ensure_group(stream, group)
            self._groups_ready = True
        r = await self._redis()
        while True:
            resp = await r.xreadgroup(
                groupname=group, consumername=self._consumer,
                streams={stream: ">"}, count=1, block=_BLOCK_MS,
            )
            if not resp:
                continue  # timed out with no entry — let the outer wait_for tick
            _stream, entries = resp[0]
            entry_id, fields = entries[0]
            raw = fields.get(b"data") or fields.get("data")
            if isinstance(raw, bytes):
                raw = raw.decode("utf-8")
            # 2c: XACK moves to after save_turn for at-least-once. Skeleton acks now.
            await r.xack(stream, group, entry_id)
            try:
                return parse(raw)
            except Exception as e:
                logger.error("Dropping unparseable bus entry {} on {}: {}", entry_id, stream, e)
                continue

    # ---- sizes (status display) ----

    @property
    def inbound_size(self) -> int:
        raise NotImplementedError("inbound_size is async on Redis; use ainbound_size()")

    @property
    def outbound_size(self) -> int:
        raise NotImplementedError("outbound_size is async on Redis; use aoutbound_size()")

    async def ainbound_size(self) -> int:
        r = await self._redis()
        return int(await r.xlen(STREAM_INBOUND))

    async def aoutbound_size(self) -> int:
        r = await self._redis()
        return int(await r.xlen(STREAM_OUTBOUND))

    # ---- RPC lane + blob store: Phase 2c ----

    async def request(self, verb: str, payload: dict, timeout: float = 5.0) -> Any:
        raise NotImplementedError("RPC lane lands in Phase 2c (find_person roster)")

    async def serve_rpc(self, handler) -> None:
        raise NotImplementedError("RPC lane lands in Phase 2c")

    async def put_blob(self, data: bytes) -> str:
        raise NotImplementedError("Media blob store lands in Phase 2c")

    async def get_blob(self, sha256: str) -> bytes:
        raise NotImplementedError("Media blob store lands in Phase 2c")

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None
