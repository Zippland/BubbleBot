"""WeChat channel implementation using wcferry."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from queue import Empty
from threading import Thread

from loguru import logger

from bubbles.bus.events import OutboundMessage
from bubbles.bus.queue import MessageBus
from bubbles.channels.base import BaseChannel
from bubbles.config.schema import WeChatConfig
from bubbles.utils.helpers import get_sessions_path, safe_filename

try:
    from wcferry import Wcf, WxMsg
except ImportError:
    Wcf = None  # type: ignore
    WxMsg = None  # type: ignore


# WeChat message types
MSG_TYPE_TEXT = 1
MSG_TYPE_IMAGE = 3
MSG_TYPE_VOICE = 34
MSG_TYPE_FILE = 49  # Also used for links, mini-programs, etc.


class WeChatChannel(BaseChannel):
    """
    WeChat channel using wcferry.

    - Private chats: Direct reply
    - Group chats: Only reply when @mentioned
    """

    name = "wechat"
    supports_markdown = False

    def __init__(
        self,
        config: WeChatConfig,
        bus: MessageBus,
        session_mode: str = "channel",
        groq_api_key: str | None = None,
    ):
        super().__init__(config, bus, session_mode)
        self.config: WeChatConfig = config
        self.wcf: Wcf | None = None
        self.wxid: str = ""
        self._recv_thread: Thread | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._contacts: dict[str, str] = {}  # wxid -> nickname cache
        self.groq_api_key = groq_api_key

    async def start(self) -> None:
        """Start WeChat client and begin listening for messages."""
        if Wcf is None:
            logger.error("wcferry not installed. Install with: pip install wcferry")
            return

        self._running = True
        self._loop = asyncio.get_event_loop()

        try:
            self.wcf = Wcf()
            self.wxid = self.wcf.get_self_wxid()
            logger.info("WeChat connected as {}", self.wxid)
            # Load all contacts
            self._load_contacts()
        except Exception as e:
            logger.error("Failed to connect to WeChat: {}", e)
            return

        # Start message receiving
        self.wcf.enable_receiving_msg()
        self._recv_thread = Thread(target=self._recv_loop, daemon=True)
        self._recv_thread.start()
        logger.info("WeChat message receiver started")

        # Keep running until stopped
        while self._running:
            await asyncio.sleep(1)

    def _load_contacts(self) -> None:
        """Load all contacts from WeChat database."""
        if not self.wcf:
            return
        try:
            contacts = self.wcf.query_sql(
                "MicroMsg.db",
                "SELECT UserName, NickName FROM Contact;"
            )
            self._contacts = {c["UserName"]: c["NickName"] for c in contacts}
            logger.info("Loaded {} contacts", len(self._contacts))
        except Exception as e:
            logger.warning("Failed to load contacts: {}", e)

    def _get_sender_name(self, sender_id: str, room_id: str | None = None) -> str | None:
        """Get sender display name. For groups, try group alias first."""
        if not self.wcf:
            return None

        # For group chat, try to get alias in chatroom first
        if room_id:
            try:
                alias = self.wcf.get_alias_in_chatroom(sender_id, room_id)
                if alias and alias.strip():
                    return alias
            except Exception:
                pass

        # Fallback to contact nickname
        return self._contacts.get(sender_id)

    def _recv_loop(self) -> None:
        """Background thread for receiving messages from wcferry."""
        while self.wcf and self.wcf.is_receiving_msg():
            try:
                msg = self.wcf.get_msg()
                if self._loop:
                    asyncio.run_coroutine_threadsafe(
                        self._process_msg(msg),
                        self._loop
                    )
            except Empty:
                continue
            except Exception as e:
                logger.error("Error receiving WeChat message: {}", e)

    async def _process_msg(self, msg: WxMsg) -> None:
        """Process incoming WeChat message."""
        # Skip self messages
        if msg.from_self():
            return

        # Only handle supported message types
        if msg.type not in (MSG_TYPE_TEXT, MSG_TYPE_IMAGE, MSG_TYPE_VOICE, MSG_TYPE_FILE):
            logger.debug("Skipping unsupported message type: {}", msg.type)
            return

        is_group = msg.from_group()

        if is_group:
            # Group chat: check if in allowed groups
            if self.config.groups and msg.roomid not in self.config.groups:
                return
            chat_id = msg.roomid
            sender_id = msg.sender
            room_id = msg.roomid
            is_at_me = msg.is_at(self.wxid)
        else:
            # Private chat: direct reply
            chat_id = msg.sender
            sender_id = msg.sender
            room_id = None
            is_at_me = True  # Always respond to private chat

        # Compute session key and media directory
        session_key = self._compute_session_key(sender_id, chat_id)
        safe_key = safe_filename(session_key.replace(":", "_"))
        media_dir = get_sessions_path() / safe_key / "data"
        media_dir.mkdir(parents=True, exist_ok=True)

        # Process message content and media
        content_parts = []
        media_paths = []

        if msg.type == MSG_TYPE_TEXT:
            text = self._strip_at_mention(msg.content) if is_at_me and is_group else msg.content
            if text:
                content_parts.append(text)

        elif msg.type == MSG_TYPE_IMAGE:
            file_path, content_text = await self._download_image(msg, media_dir)
            if file_path:
                media_paths.append(file_path)
            content_parts.append(content_text)
            is_at_me = True  # Always respond to images

        elif msg.type == MSG_TYPE_VOICE:
            file_path, content_text = await self._download_voice(msg, media_dir)
            if file_path:
                media_paths.append(file_path)
                # Transcribe voice to text
                transcription = await self._transcribe_voice(file_path)
                if transcription:
                    content_parts.append(f"[transcription: {transcription}]")
                else:
                    content_parts.append(content_text)
            else:
                content_parts.append(content_text)
            is_at_me = True  # Always respond to voice

        elif msg.type == MSG_TYPE_FILE:
            # Type 49 can be file, link, or mini-program
            # For now just note it in content
            content_parts.append("[file or link message]")

        content = "\n".join(content_parts) if content_parts else ""

        if not content and not media_paths:
            return

        # Get sender name (group alias or contact nickname)
        sender_name = self._get_sender_name(sender_id, room_id)

        logger.debug("WeChat message from {} ({}) in {}: {}{}",
                     sender_name or sender_id, sender_id, chat_id, content[:50],
                     " [@me]" if is_at_me else "")

        await self._handle_message(
            sender_id=sender_id,
            chat_id=chat_id,
            content=content,
            media=media_paths,
            metadata={
                "is_group": is_group,
                "sender_name": sender_name,
                "respond": is_at_me,  # Only respond when @mentioned or private chat
            }
        )

    def _strip_at_mention(self, content: str) -> str:
        """Remove @mention from message content."""
        import re
        # WeChat @mention format: @nickname followed by space or text
        # Pattern: @xxx followed by space or end
        return re.sub(r"@\S+\s*", "", content).strip()

    async def _download_image(self, msg: WxMsg, media_dir: Path) -> tuple[str | None, str]:
        """Download image from WeChat message."""
        if not self.wcf:
            return None, "[image: download failed]"

        try:
            # wcf.download_image runs synchronously, use executor
            loop = asyncio.get_running_loop()
            file_path = await loop.run_in_executor(
                None,
                lambda: self.wcf.download_image(msg.id, msg.extra, str(media_dir), timeout=30)
            )

            if file_path and os.path.isfile(file_path):
                logger.debug("Downloaded image to {}", file_path)
                return file_path, f"[image: {os.path.basename(file_path)}]"
            else:
                logger.warning("Failed to download image for msg {}", msg.id)
                return None, "[image: download failed]"
        except Exception as e:
            logger.error("Error downloading image: {}", e)
            return None, "[image: download failed]"

    async def _download_voice(self, msg: WxMsg, media_dir: Path) -> tuple[str | None, str]:
        """Download voice message and convert to MP3."""
        if not self.wcf:
            return None, "[voice: download failed]"

        try:
            # wcf.get_audio_msg runs synchronously, use executor
            loop = asyncio.get_running_loop()
            file_path = await loop.run_in_executor(
                None,
                lambda: self.wcf.get_audio_msg(msg.id, str(media_dir), timeout=30)
            )

            if file_path and os.path.isfile(file_path):
                logger.debug("Downloaded voice to {}", file_path)
                return file_path, f"[voice: {os.path.basename(file_path)}]"
            else:
                logger.warning("Failed to download voice for msg {}", msg.id)
                return None, "[voice: download failed]"
        except Exception as e:
            logger.error("Error downloading voice: {}", e)
            return None, "[voice: download failed]"

    async def _transcribe_voice(self, file_path: str) -> str | None:
        """Transcribe voice file to text using Groq Whisper."""
        if not self.groq_api_key:
            logger.debug("No Groq API key configured, skipping transcription")
            return None

        try:
            from bubbles.providers.transcription import GroqTranscriptionProvider
            transcriber = GroqTranscriptionProvider(api_key=self.groq_api_key)
            transcription = await transcriber.transcribe(file_path)
            if transcription:
                logger.info("Transcribed voice: {}...", transcription[:50])
            return transcription
        except Exception as e:
            logger.error("Error transcribing voice: {}", e)
            return None

    _IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"}

    async def send(self, msg: OutboundMessage) -> None:
        """Send a message through WeChat, including media if present."""
        if not self.wcf:
            logger.warning("WeChat not connected")
            return

        try:
            # Send media files first
            for file_path in msg.media:
                if not os.path.isfile(file_path):
                    logger.warning("Media file not found: {}", file_path)
                    continue
                ext = os.path.splitext(file_path)[1].lower()
                if ext in self._IMAGE_EXTS:
                    self.wcf.send_image(file_path, msg.chat_id)
                    logger.debug("Sent image to {}: {}", msg.chat_id, file_path)
                else:
                    self.wcf.send_file(file_path, msg.chat_id)
                    logger.debug("Sent file to {}: {}", msg.chat_id, file_path)

            # Send text content
            if msg.content and msg.content.strip():
                self.wcf.send_text(msg.content, msg.chat_id)
                logger.debug("Sent message to {}: {}...", msg.chat_id, msg.content[:50])
        except Exception as e:
            logger.error("Failed to send WeChat message: {}", e)

    async def stop(self) -> None:
        """Stop WeChat client."""
        self._running = False

        if self.wcf:
            try:
                self.wcf.disable_recv_msg()
                logger.info("WeChat message receiver stopped")
            except Exception as e:
                logger.error("Error stopping WeChat: {}", e)
            self.wcf = None

        self._recv_thread = None
