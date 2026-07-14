"""RPC proxy standing in for ChannelManager on the harness host.

After the Phase 2 split, channels live on the Windows host and the harness has
no in-process ChannelManager. But ``FindPersonTool`` (find_person.py:53-57)
still calls ``channel_manager.get_channel(name).get_group_members(chat_id)``
synchronously. This proxy preserves that exact shape but turns the roster
lookup into a cross-machine RPC over the bus.

Degradation contract: on timeout / transport error / RPC-not-available, return
``[]`` — identical to the existing empty-roster path (find_person.py:58), so a
slow or dead channel host never raises into the agent loop or hangs the turn.
"""

from __future__ import annotations

from typing import Any

from loguru import logger

ROSTER_RPC_TIMEOUT = 5.0


class _RpcChannelStub:
    """Stand-in for a single channel; only supports get_group_members via RPC."""

    def __init__(self, bus: Any, channel: str):
        self._bus = bus
        self._channel = channel

    async def get_group_members(self, chat_id: str) -> list[dict]:
        # bus.request is implemented in 2c; until then (or on any failure)
        # degrade to empty roster — find_person handles [] gracefully.
        request = getattr(self._bus, "request", None)
        if request is None:
            return []
        try:
            result = await request(
                "roster",
                {"channel": self._channel, "chat_id": chat_id},
                timeout=ROSTER_RPC_TIMEOUT,
            )
        except NotImplementedError:
            return []  # 2b: RPC lane not wired yet
        except Exception as e:
            logger.warning("roster RPC failed for {}:{} — degrading to empty: {}",
                           self._channel, chat_id, e)
            return []
        return result if isinstance(result, list) else []


class RpcChannelProxy:
    """Drop-in for ChannelManager on the harness side (get_channel only)."""

    def __init__(self, bus: Any):
        self._bus = bus

    def get_channel(self, name: str) -> _RpcChannelStub:
        # Always return a stub; the real channel liveness is decided on the
        # channels host. A dead channel just yields [] via the RPC timeout.
        return _RpcChannelStub(self._bus, name)
