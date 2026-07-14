"""Message bus module for decoupled channel-agent communication."""

from bubbles.bus.base import MessageBus
from bubbles.bus.events import InboundMessage, OutboundMessage
from bubbles.bus.local import LocalBus

__all__ = ["MessageBus", "LocalBus", "InboundMessage", "OutboundMessage"]
