"""Message bus interface.

The bus decouples chat channels from the agent core: channels publish inbound
messages and consume outbound ones; the agent consumes inbound and publishes
outbound. Phase 2 makes the transport pluggable — the in-process
``asyncio.Queue`` implementation (``LocalBus``) is the default, and a
cross-machine ``RemoteBus`` (Redis Streams) lets channels and harness run on
separate hosts.

Any implementation MUST preserve these exact signatures — every ``bus=``
injection site (AgentLoop, SubagentManager, BaseChannel, ChannelManager)
depends on them unchanged.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from bubbles.bus.events import InboundMessage, OutboundMessage


class MessageBus(ABC):
    """Abstract async message bus.

    ``consume_inbound`` / ``consume_outbound`` block until a message is
    available and MUST remain cancellable — callers wrap them in
    ``asyncio.wait_for(..., timeout=1.0)`` (loop.py, manager.py, agent_cmd.py),
    so a network-backed implementation must use a bounded blocking read (e.g.
    ``XREADGROUP BLOCK=1000``) and let ``CancelledError`` propagate.
    """

    @abstractmethod
    async def publish_inbound(self, msg: InboundMessage) -> None:
        """Publish a message from a channel to the agent."""

    @abstractmethod
    async def consume_inbound(self) -> InboundMessage:
        """Consume the next inbound message (blocks until available)."""

    @abstractmethod
    async def publish_outbound(self, msg: OutboundMessage) -> None:
        """Publish a response from the agent to channels."""

    @abstractmethod
    async def consume_outbound(self) -> OutboundMessage:
        """Consume the next outbound message (blocks until available)."""

    @property
    @abstractmethod
    def inbound_size(self) -> int:
        """Number of pending inbound messages."""

    @property
    @abstractmethod
    def outbound_size(self) -> int:
        """Number of pending outbound messages."""
