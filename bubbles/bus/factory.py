"""Bus factory: construct the configured transport.

Mirrors the pluggable-sandbox pattern (SandboxManager / SandboxConfig). The
``local`` backend returns the in-process ``LocalBus`` (default, unchanged
single-process behavior); ``redis`` returns a ``RemoteBus`` for split
deployment.
"""

from __future__ import annotations

from loguru import logger

from bubbles.bus.base import MessageBus
from bubbles.bus.local import LocalBus


def make_bus(config: "Config | None" = None, *, consumer_name: str = "default") -> MessageBus:
    """Return the bus implementation selected by ``config.bus.default``.

    ``consumer_name`` distinguishes Redis consumers within a group (harness vs
    channels host); ignored by LocalBus.
    """
    bus_cfg = getattr(getattr(config, "bus", None), "default", "local") if config else "local"
    backend = (bus_cfg or "local").lower()

    if backend == "redis":
        from bubbles.bus.redis_bus import RemoteBus
        redis_url = config.bus.redis_url  # type: ignore[union-attr]
        inline_max = getattr(config.bus, "inline_media_max_bytes", 1_048_576)  # type: ignore[union-attr]
        return RemoteBus(redis_url, consumer_name=consumer_name, inline_media_max_bytes=inline_max)

    if backend != "local":
        logger.warning("Unknown bus backend {!r}; falling back to 'local'", backend)
    return LocalBus()
