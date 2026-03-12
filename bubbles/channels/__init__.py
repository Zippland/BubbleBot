"""Chat channels module with plugin architecture."""

from bubbles.channels.base import BaseChannel
from bubbles.channels.manager import ChannelManager

__all__ = ["BaseChannel", "ChannelManager"]
