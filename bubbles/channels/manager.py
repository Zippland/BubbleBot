"""Channel manager for coordinating chat channels."""

from __future__ import annotations

import asyncio
import inspect
from typing import Any, Awaitable, Callable

from loguru import logger

from bubbles.bus.events import OutboundMessage
from bubbles.bus.queue import MessageBus
from bubbles.channels.base import BaseChannel
from bubbles.config.schema import Config


class ChannelManager:
    """
    Manages chat channels and coordinates message routing.

    Responsibilities:
    - Initialize enabled channels (Telegram, WhatsApp, etc.)
    - Start/stop channels
    - Route outbound messages
    """

    def __init__(self, config: Config, bus: MessageBus):
        self.config = config
        self.bus = bus
        self.channels: dict[str, BaseChannel] = {}
        self._dispatch_task: asyncio.Task | None = None
        self._stopping = False
        self.on_delivery_result: (
            Callable[[OutboundMessage, bool], Awaitable[None] | None] | None
        ) = None

        self._init_channels()

    def _init_channels(self) -> None:
        """Initialize channels based on config."""
        session_mode = self.config.channels.session_mode

        # Telegram channel
        if self.config.channels.telegram.enabled:
            try:
                from bubbles.channels.telegram import TelegramChannel
                self.channels["telegram"] = TelegramChannel(
                    self.config.channels.telegram,
                    self.bus,
                    groq_api_key=self.config.providers.groq.api_key,
                    session_mode=session_mode,
                )
                logger.info("Telegram channel enabled")
            except ImportError as e:
                logger.warning("Telegram channel not available: {}", e)

        # WhatsApp channel
        if self.config.channels.whatsapp.enabled:
            try:
                from bubbles.channels.whatsapp import WhatsAppChannel
                self.channels["whatsapp"] = WhatsAppChannel(
                    self.config.channels.whatsapp, self.bus, session_mode=session_mode
                )
                logger.info("WhatsApp channel enabled")
            except ImportError as e:
                logger.warning("WhatsApp channel not available: {}", e)

        # Discord channel
        if self.config.channels.discord.enabled:
            try:
                from bubbles.channels.discord import DiscordChannel
                self.channels["discord"] = DiscordChannel(
                    self.config.channels.discord, self.bus, session_mode=session_mode
                )
                logger.info("Discord channel enabled")
            except ImportError as e:
                logger.warning("Discord channel not available: {}", e)

        # Feishu channel
        if self.config.channels.feishu.enabled:
            try:
                from bubbles.channels.feishu import FeishuChannel
                self.channels["feishu"] = FeishuChannel(
                    self.config.channels.feishu, self.bus, session_mode=session_mode
                )
                logger.info("Feishu channel enabled")
            except ImportError as e:
                logger.warning("Feishu channel not available: {}", e)

        # Mochat channel
        if self.config.channels.mochat.enabled:
            try:
                from bubbles.channels.mochat import MochatChannel
                self.channels["mochat"] = MochatChannel(
                    self.config.channels.mochat, self.bus, session_mode=session_mode
                )
                logger.info("Mochat channel enabled")
            except ImportError as e:
                logger.warning("Mochat channel not available: {}", e)

        # DingTalk channel
        if self.config.channels.dingtalk.enabled:
            try:
                from bubbles.channels.dingtalk import DingTalkChannel
                self.channels["dingtalk"] = DingTalkChannel(
                    self.config.channels.dingtalk, self.bus, session_mode=session_mode
                )
                logger.info("DingTalk channel enabled")
            except ImportError as e:
                logger.warning("DingTalk channel not available: {}", e)

        # Email channel
        if self.config.channels.email.enabled:
            try:
                from bubbles.channels.email import EmailChannel
                self.channels["email"] = EmailChannel(
                    self.config.channels.email, self.bus, session_mode=session_mode
                )
                logger.info("Email channel enabled")
            except ImportError as e:
                logger.warning("Email channel not available: {}", e)

        # Slack channel
        if self.config.channels.slack.enabled:
            try:
                from bubbles.channels.slack import SlackChannel
                self.channels["slack"] = SlackChannel(
                    self.config.channels.slack, self.bus, session_mode=session_mode
                )
                logger.info("Slack channel enabled")
            except ImportError as e:
                logger.warning("Slack channel not available: {}", e)

        # QQ channel
        if self.config.channels.qq.enabled:
            try:
                from bubbles.channels.qq import QQChannel
                self.channels["qq"] = QQChannel(
                    self.config.channels.qq, self.bus, session_mode=session_mode
                )
                logger.info("QQ channel enabled")
            except ImportError as e:
                logger.warning("QQ channel not available: {}", e)

        # WeChat channel
        if self.config.channels.wechat.enabled:
            try:
                from bubbles.channels.wechat import WeChatChannel
                self.channels["wechat"] = WeChatChannel(
                    self.config.channels.wechat,
                    self.bus,
                    session_mode=session_mode,
                    groq_api_key=self.config.providers.groq.api_key,
                )
                logger.info("WeChat channel enabled")
            except ImportError as e:
                logger.warning("WeChat channel not available: {}", e)

    async def _start_channel(self, name: str, channel: BaseChannel) -> None:
        """Start a channel and log any exceptions."""
        try:
            await channel.start()
            if not self._stopping:
                raise RuntimeError(f"Channel {name} stopped unexpectedly")
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.error("Failed to start channel {}: {}", name, e)
            raise

    async def start_all(self) -> None:
        """Start all channels and the outbound dispatcher."""
        self._stopping = False
        if not self.channels:
            logger.warning("No channels enabled")
            return

        # Start outbound dispatcher
        self._dispatch_task = asyncio.create_task(self._dispatch_outbound())

        # Start channels
        tasks = []
        for name, channel in self.channels.items():
            logger.info("Starting {} channel...", name)
            tasks.append(asyncio.create_task(self._start_channel(name, channel)))

        # Any enabled channel ending is a gateway service failure.  Propagate
        # immediately, then cancel the remaining channel tasks as one unit.
        try:
            await asyncio.gather(*tasks)
        finally:
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)

    async def stop_all(self) -> dict[str, BaseException]:
        """Stop every channel and return failures without skipping later channels."""
        logger.info("Stopping all channels...")
        self._stopping = True
        failures: dict[str, BaseException] = {}

        # Stop dispatcher
        if self._dispatch_task:
            self._dispatch_task.cancel()
            try:
                await self._dispatch_task
            except asyncio.CancelledError:
                pass

        # WCFerry injects native code into the already-running WeChat process.
        # Detach it before any other channel cleanup can block.
        ordered_names = []
        if "wechat" in self.channels:
            ordered_names.append("wechat")
        ordered_names.extend(name for name in self.channels if name != "wechat")

        for name in ordered_names:
            channel = self.channels[name]
            try:
                await channel.stop()
                logger.info("Stopped {} channel", name)
            except BaseException as e:
                logger.error("Error stopping {}: {}", name, e)
                failures[name] = e
        return failures

    async def _dispatch_outbound(self) -> None:
        """Dispatch outbound messages to the appropriate channel."""
        logger.info("Outbound dispatcher started")

        while True:
            try:
                msg = await asyncio.wait_for(
                    self.bus.consume_outbound(),
                    timeout=1.0
                )

                if msg.metadata.get("_progress"):
                    if msg.metadata.get("_tool_hint") and not self.config.channels.send_tool_hints:
                        continue
                    if not msg.metadata.get("_tool_hint") and not self.config.channels.send_progress:
                        continue

                channel = self.channels.get(msg.channel)
                if channel:
                    delivered = False
                    try:
                        result = await channel.send(msg)
                        delivered = result is not False
                    except Exception as e:
                        logger.error("Error sending to {}: {}", msg.channel, e)
                    finally:
                        await self._report_delivery_result(msg, delivered)
                else:
                    logger.warning("Unknown channel: {}", msg.channel)
                    await self._report_delivery_result(msg, False)

            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break

    async def _report_delivery_result(
        self,
        msg: OutboundMessage,
        delivered: bool,
    ) -> None:
        if self.on_delivery_result is None:
            return
        try:
            callback_result = self.on_delivery_result(msg, delivered)
            if inspect.isawaitable(callback_result):
                await callback_result
        except Exception as exc:
            logger.error("Delivery-result callback failed: {}", exc)

    def get_channel(self, name: str) -> BaseChannel | None:
        """Get a channel by name."""
        return self.channels.get(name)

    def get_status(self) -> dict[str, Any]:
        """Get status of all channels."""
        return {
            name: {
                "enabled": True,
                "running": channel.is_running
            }
            for name, channel in self.channels.items()
        }

    @property
    def enabled_channels(self) -> list[str]:
        """Get list of enabled channel names."""
        return list(self.channels.keys())
