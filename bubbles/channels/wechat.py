"""WeChat channel implementation using wcferry."""

from __future__ import annotations

import asyncio
import html
import os
import re
from queue import Empty
from threading import Thread

from loguru import logger

from bubbles.bus.events import OutboundMessage
from bubbles.bus.queue import MessageBus
from bubbles.channels.base import BaseChannel
from bubbles.config.schema import WeChatConfig

try:
    from wcferry import Wcf, WxMsg
except ImportError:
    Wcf = None  # type: ignore
    WxMsg = None  # type: ignore


# WeChat message types
MSG_TYPE_TEXT = 1
MSG_TYPE_IMAGE = 3
MSG_TYPE_VOICE = 34
MSG_TYPE_VIDEO = 43
MSG_TYPE_EMOJI = 47
MSG_TYPE_APP = 49  # 文件、链接、小程序、引用等
MSG_TYPE_SYSTEM = 10000


class WeChatChannel(BaseChannel):
    """
    WeChat channel using wcferry.

    - Private chats: Direct reply
    - Group chats: Only reply when @mentioned
    - Supports: text, image, voice, video, file, and quoted messages
    """

    name = "wechat"
    supports_markdown = False

    def __init__(
        self,
        config: WeChatConfig,
        bus: MessageBus,
        session_mode: str = "channel",
        groq_api_key: str | None = None,  # kept for compatibility, not used
    ):
        super().__init__(config, bus, session_mode)
        self.config: WeChatConfig = config
        self.wcf: Wcf | None = None
        self.wxid: str = ""
        self._recv_thread: Thread | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._contacts: dict[str, str] = {}  # wxid -> nickname cache

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

        # Get sender name (group alias or contact nickname)
        sender_name = self._get_sender_name(sender_id, room_id)

        # Process different message types
        text = ""
        media_paths: list[str] = []

        if msg.type == MSG_TYPE_TEXT:
            text = self._strip_at_mention(msg.content) if is_at_me and is_group else msg.content

        elif msg.type == MSG_TYPE_IMAGE:
            file_path, content_text = await self._download_and_save_media(
                "image", msg, chat_id
            )
            if file_path:
                media_paths.append(file_path)
            text = content_text

        elif msg.type == MSG_TYPE_VOICE:
            file_path, content_text = await self._download_and_save_media(
                "voice", msg, chat_id
            )
            if file_path:
                media_paths.append(file_path)
            text = content_text

        elif msg.type == MSG_TYPE_VIDEO:
            file_path, content_text = await self._download_and_save_media(
                "video", msg, chat_id
            )
            if file_path:
                media_paths.append(file_path)
            text = content_text

        elif msg.type == MSG_TYPE_APP:
            # Handle app messages (files, links, quotes, etc.)
            text, media_paths = await self._process_app_msg(msg, chat_id, is_group)
            # For quoted messages in groups, check if user @mentioned
            if is_group and text:
                if self._is_at_in_text(text):
                    is_at_me = True
                    text = self._strip_at_mention(text)

        else:
            logger.debug("Skipping unsupported message type: {}", msg.type)
            return

        if not text and not media_paths:
            return

        logger.debug("WeChat message from {} ({}) in {}: {}{}",
                     sender_name or sender_id, sender_id, chat_id,
                     (text[:50] if text else "[media]"),
                     " [@me]" if is_at_me else "")

        await self._handle_message(
            sender_id=sender_id,
            chat_id=chat_id,
            content=text,
            media=media_paths,
            metadata={
                "is_group": is_group,
                "sender_name": sender_name,
                "respond": is_at_me,  # Only respond when @mentioned or private chat
                "msg_type": msg.type,
            }
        )

    def _strip_at_mention(self, content: str) -> str:
        """Remove @mention from message content."""
        # WeChat @mention format: @nickname followed by space or text
        # Pattern: @xxx followed by space or end
        return re.sub(r"@\S+\s*", "", content).strip()

    def _is_at_in_text(self, text: str) -> bool:
        """Check if text contains @mention to self."""
        if not self.wcf:
            return False
        my_name = self._contacts.get(self.wxid, "")
        if my_name and f"@{my_name}" in text:
            return True
        return False

    async def _download_and_save_media(
        self,
        media_type: str,
        msg: WxMsg,
        chat_id: str,
    ) -> tuple[str | None, str]:
        """
        Download media from WeChat and save to session's data directory.

        Returns:
            (file_path, content_text) - file_path is None if download failed
        """
        # Check session binding first
        media_dir = self._get_media_dir(chat_id)
        if media_dir is None:
            return None, f"[{media_type}: 请先使用 /session <name> 绑定工作区]"

        if not self.wcf:
            return None, f"[{media_type}: WeChat not connected]"

        loop = asyncio.get_running_loop()
        file_path: str | None = None
        filename = ""

        try:
            if media_type == "image":
                # Download image
                file_path = await loop.run_in_executor(
                    None,
                    lambda: self.wcf.download_image(msg.id, msg.extra, str(media_dir), timeout=30)
                )
                if file_path:
                    filename = os.path.basename(file_path)

            elif media_type == "voice":
                # Download voice (converted to MP3)
                file_path = await loop.run_in_executor(
                    None,
                    lambda: self.wcf.get_audio_msg(msg.id, str(media_dir), timeout=10)
                )
                if file_path:
                    filename = os.path.basename(file_path)

            elif media_type == "video":
                # Download video
                file_path = await loop.run_in_executor(
                    None,
                    lambda: self.wcf.download_video(msg.id, msg.thumb, str(media_dir), timeout=60)
                )
                if file_path:
                    filename = os.path.basename(file_path)

            if file_path and os.path.exists(file_path):
                logger.debug("Downloaded {} to {}", media_type, file_path)
                return file_path, f"[{media_type}: <work_dir>/data/{filename}]"

        except Exception as e:
            logger.error("Error downloading {}: {}", media_type, e)

        return None, f"[{media_type}: download failed]"

    async def _process_app_msg(
        self,
        msg: WxMsg,
        chat_id: str,
        is_group: bool,
    ) -> tuple[str, list[str]]:
        """
        Process APP type messages (type=49): files, links, quotes, etc.

        Returns:
            (text, media_paths)
        """
        content = msg.content
        media_paths: list[str] = []

        # Detect appmsg type
        appmsg_type_match = re.search(r'<type>(\d+)</type>', content)
        if not appmsg_type_match:
            return "", []

        appmsg_type = appmsg_type_match.group(1)

        # Type 57: Quote/Reply message
        if appmsg_type == "57":
            return self._parse_quote_msg(content)

        # Type 6: File
        if appmsg_type == "6":
            title_match = re.search(r'<title>(.*?)</title>', content)
            filename = title_match.group(1) if title_match else "file"
            # Note: File download requires additional handling via download_attach
            # For now, just indicate file was received
            return f"[文件: {filename}]", []

        # Type 5: Link
        if appmsg_type == "5":
            title_match = re.search(r'<title>(.*?)</title>', content)
            url_match = re.search(r'<url>(.*?)</url>', content)
            title = html.unescape(title_match.group(1)) if title_match else ""
            url = html.unescape(url_match.group(1)) if url_match else ""
            if title and url:
                return f"[链接: {title}]\n{url}", []
            elif title:
                return f"[链接: {title}]", []
            return "[链接]", []

        # Type 33/36: Mini Program
        if appmsg_type in ("33", "36"):
            title_match = re.search(r'<title>(.*?)</title>', content)
            title = html.unescape(title_match.group(1)) if title_match else "小程序"
            return f"[小程序: {title}]", []

        # Type 19: Chat record forward
        if appmsg_type == "19":
            title_match = re.search(r'<title>(.*?)</title>', content)
            title = html.unescape(title_match.group(1)) if title_match else "聊天记录"
            return f"[聊天记录: {title}]", []

        # Other types: try to extract title
        title_match = re.search(r'<title>(.*?)</title>', content)
        if title_match:
            title = html.unescape(title_match.group(1))
            return f"[消息: {title}]", []

        return "[未知消息类型]", []

    def _parse_quote_msg(self, content: str) -> tuple[str, list[str]]:
        """
        Parse quoted/reply message (appmsg type=57).

        Returns:
            (text, media_paths)
        """
        result_parts = []

        # Extract user's new message from <title>
        title_match = re.search(r'<title>(.*?)</title>', content)
        if title_match:
            new_content = html.unescape(title_match.group(1).strip())
            if new_content:
                result_parts.append(new_content)

        # Extract quoted content from <refermsg>
        refermsg_match = re.search(r'<refermsg>(.*?)</refermsg>', content, re.DOTALL)
        if refermsg_match:
            refermsg = refermsg_match.group(1)

            # Get quoted sender
            displayname_match = re.search(r'<displayname>(.*?)</displayname>', refermsg)
            quoted_sender = html.unescape(displayname_match.group(1)) if displayname_match else ""

            # Get quoted message type
            type_match = re.search(r'<type>(\d+)</type>', refermsg)
            quoted_type = type_match.group(1) if type_match else "1"

            # Get quoted content
            content_match = re.search(r'<content>(.*?)</content>', refermsg, re.DOTALL)
            if content_match:
                quoted_raw = content_match.group(1)
                # Decode HTML entities (content may be double-encoded)
                quoted_decoded = html.unescape(quoted_raw)

                # Type 3 = image
                if quoted_type == "3":
                    quoted_text = "[图片]"
                # Type 34 = voice
                elif quoted_type == "34":
                    quoted_text = "[语音]"
                # Type 43 = video
                elif quoted_type == "43":
                    quoted_text = "[视频]"
                # Type 49 = app message (file, link, etc.)
                elif quoted_type == "49":
                    # Try to extract title from the quoted app message
                    inner_title = re.search(r'<title>(.*?)</title>', quoted_decoded)
                    if inner_title:
                        quoted_text = f"[{html.unescape(inner_title.group(1))}]"
                    else:
                        quoted_text = "[消息]"
                else:
                    # Text or other
                    quoted_text = re.sub(r'<.*?>', '', quoted_decoded).strip()
                    quoted_text = re.sub(r'\s+', ' ', quoted_text)

                if quoted_sender and quoted_text:
                    result_parts.append(f"[引用 {quoted_sender}: {quoted_text}]")
                elif quoted_text:
                    result_parts.append(f"[引用: {quoted_text}]")

        return "\n".join(result_parts), []

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
