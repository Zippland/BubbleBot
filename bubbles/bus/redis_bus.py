"""Cross-machine message bus over Redis Streams (Phase 2 split deployment).

Lets the channel host (Windows/wcferry) and the harness host (Linux/macOS) run
as separate processes over a shared Redis. This is the ``bus.default = redis``
transport; the default ``local`` transport (LocalBus) is unaffected.

Lanes:
- inbound (channels→harness) / outbound (harness→channels): Redis Streams.
- RPC (find_person roster): request/reply over Redis lists.
- media blob store: content-addressed keys with TTL.

Media crosses by value: ``publish_*`` dehydrates local paths to wire
descriptors; ``consume_*`` rehydrates them into a local file the consumer owns,
so tools and ContextBuilder (which expect ``media`` = local paths) are
UNCHANGED — the translation lives entirely at this transport boundary.

Reliability note: ``MessageBus.consume_*`` returns a bare message with no ack
handle, so this auto-acks on read. That is at-most-once; moving XACK to after
``save_turn`` for at-least-once + ``message_id`` dedupe is a follow-up that
needs the interface to carry the stream entry id.
"""

from __future__ import annotations

import json
import tempfile
import uuid
from dataclasses import replace
from pathlib import Path
from typing import Any, Awaitable, Callable

from loguru import logger

from bubbles.bus.base import MessageBus
from bubbles.bus.events import InboundMessage, OutboundMessage
from bubbles.bus.media import dehydrate_media, rehydrate_media, sha256_hex
from bubbles.bus.wire import (
    inbound_from_wire,
    inbound_to_wire,
    outbound_from_wire,
    outbound_to_wire,
)

STREAM_INBOUND = "bus:inbound"
STREAM_OUTBOUND = "bus:outbound"
GROUP_HARNESS = "harness"    # consumes inbound
GROUP_CHANNELS = "channels"  # consumes outbound

RPC_REQ_LIST = "rpc:channels:req"
BLOB_PREFIX = "blob:"
BLOB_TTL_SECONDS = 3600  # ≥ redelivery/consume window so a slow consumer still finds bytes
RPC_REPLY_TTL_SECONDS = 60

# Bounded block so asyncio.wait_for(timeout=1.0) can cancel the read.
_BLOCK_MS = 1000
_INLINE_MAX_DEFAULT = 1_048_576


def _require_redis():
    try:
        import redis.asyncio as aioredis  # type: ignore
    except ImportError as e:
        raise RuntimeError(
            "The 'redis' backend requires the redis package. "
            "Install it with: pip install 'bubbles[redis]'  (or: pip install 'redis>=5.0')"
        ) from e
    return aioredis


class RemoteBus(MessageBus):
    """Redis Streams transport. One instance per host (channels or harness)."""

    def __init__(
        self,
        redis_url: str,
        *,
        consumer_name: str = "default",
        inline_media_max_bytes: int = _INLINE_MAX_DEFAULT,
    ):
        if not redis_url:
            raise ValueError("RemoteBus requires a redis_url (bus.redis_url in config)")
        self._url = redis_url
        self._consumer = consumer_name
        self._inline_max = inline_media_max_bytes
        self._aioredis = _require_redis()
        self._client: Any = None
        self._groups_ready: set[str] = set()

    async def _redis(self) -> Any:
        if self._client is None:
            self._client = self._aioredis.from_url(self._url)
        return self._client

    async def _ensure_group(self, stream: str, group: str) -> None:
        if (stream, group) in self._groups_ready:
            return
        r = await self._redis()
        try:
            await r.xgroup_create(name=stream, groupname=group, id="0", mkstream=True)
        except Exception as e:  # redis.exceptions.ResponseError BUSYGROUP
            if "BUSYGROUP" not in str(e):
                raise
        self._groups_ready.add((stream, group))

    # ---- inbound lane (channels → harness) ----

    async def publish_inbound(self, msg: InboundMessage) -> None:
        msg = await self._dehydrate(msg)
        r = await self._redis()
        await r.xadd(STREAM_INBOUND, {"data": inbound_to_wire(msg)})

    async def consume_inbound(self) -> InboundMessage:
        msg = await self._consume(STREAM_INBOUND, GROUP_HARNESS, inbound_from_wire)
        if msg.media:
            dest = self._inbound_dest(msg)
            paths = await rehydrate_media(msg.media, dest_dir=dest, get_blob=self.get_blob)
            msg = replace(msg, media=paths)
        return msg

    # ---- outbound lane (harness → channels) ----

    async def publish_outbound(self, msg: OutboundMessage) -> None:
        msg = await self._dehydrate(msg)
        r = await self._redis()
        await r.xadd(STREAM_OUTBOUND, {"data": outbound_to_wire(msg)})

    async def consume_outbound(self) -> OutboundMessage:
        msg = await self._consume(STREAM_OUTBOUND, GROUP_CHANNELS, outbound_from_wire)
        if msg.media:
            dest = Path(tempfile.mkdtemp(prefix="bubbles-outbound-"))
            paths = await rehydrate_media(msg.media, dest_dir=dest, get_blob=self.get_blob)
            msg = replace(msg, media=paths)
        return msg

    async def _dehydrate(self, msg):
        """Replace local media paths with wire descriptors (no-op if no media)."""
        if not msg.media:
            return msg
        descs = await dehydrate_media(
            list(msg.media), inline_max_bytes=self._inline_max, put_blob=self.put_blob
        )
        return replace(msg, media=descs)

    def _inbound_dest(self, msg: InboundMessage) -> Path:
        """Harness-side landing dir for inbound media: <session>/data/.

        relocate_media_to_session (loop.py) does final placement, so any
        harness-readable path works; we mirror the session data dir directly.
        """
        from bubbles.utils.helpers import get_sessions_path, safe_filename
        safe = safe_filename(msg.session_key.replace(":", "_"))
        return get_sessions_path() / safe / "data"

    async def _consume(self, stream: str, group: str, parse):
        """Block (bounded, so wait_for can cancel) until one entry arrives."""
        await self._ensure_group(stream, group)
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
            await r.xack(stream, group, entry_id)  # at-most-once (see module docstring)
            try:
                return parse(raw)
            except Exception as e:
                logger.error("Dropping unparseable bus entry {} on {}: {}", entry_id, stream, e)
                continue

    # ---- sizes ----

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

    # ---- media blob store ----

    async def put_blob(self, data: bytes) -> str:
        sha = sha256_hex(data)
        r = await self._redis()
        await r.set(f"{BLOB_PREFIX}{sha}", data, ex=BLOB_TTL_SECONDS)
        return sha

    async def get_blob(self, sha256: str) -> bytes:
        r = await self._redis()
        data = await r.get(f"{BLOB_PREFIX}{sha256}")
        if data is None:
            raise KeyError(f"blob {sha256} not found (expired or never stored)")
        return data if isinstance(data, bytes) else bytes(data)

    # ---- RPC lane (find_person roster) ----

    async def request(self, verb: str, payload: dict, timeout: float = 5.0) -> Any:
        """Harness side: RPUSH a request, BLPOP the reply. Raises on timeout/error."""
        req_id = uuid.uuid4().hex
        reply_key = f"rpc:resp:{req_id}"
        r = await self._redis()
        await r.rpush(RPC_REQ_LIST, json.dumps(
            {"id": req_id, "verb": verb, "payload": payload, "reply": reply_key}
        ))
        resp = await r.blpop(reply_key, timeout=int(timeout) or 1)
        if resp is None:
            raise TimeoutError(f"RPC '{verb}' timed out after {timeout}s")
        _key, raw = resp
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        d = json.loads(raw)
        if not d.get("ok"):
            raise RuntimeError(d.get("error") or "RPC failed")
        return d.get("result")

    async def serve_rpc(self, handler: Callable[[str, dict], Awaitable[Any]]) -> None:
        """Channels side: consume requests, dispatch to handler, push replies.

        Runs until cancelled. handler(verb, payload) -> result (JSON-serializable).
        """
        r = await self._redis()
        while True:
            resp = await r.blpop(RPC_REQ_LIST, timeout=1)  # cancellable
            if resp is None:
                continue
            _key, raw = resp
            if isinstance(raw, bytes):
                raw = raw.decode("utf-8")
            try:
                req = json.loads(raw)
                result = await handler(req["verb"], req.get("payload") or {})
                reply = {"ok": True, "result": result}
            except Exception as e:
                reply = {"ok": False, "error": str(e)}
                req = locals().get("req") or {}
            reply_key = req.get("reply")
            if reply_key:
                await r.rpush(reply_key, json.dumps(reply))
                await r.expire(reply_key, RPC_REPLY_TTL_SECONDS)

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None
