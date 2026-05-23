"""WeChat channel implementation using wcferry."""

from __future__ import annotations

import asyncio
import html
import os
import re
from dataclasses import dataclass, field
from queue import Empty
from threading import Thread

from loguru import logger

from bubbles.bus.events import OutboundMessage
from bubbles.bus.queue import MessageBus
from bubbles.channels.base import BaseChannel
from bubbles.channels.mentions import extract_mentions, replace_mentions
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


@dataclass
class WeChatContact:
    """Names attached to a wxid. Empty strings mean field is not set."""
    nickname: str = ""  # 微信昵称（profile name）
    alias: str = ""     # 微信号（user-chosen short id）
    remark: str = ""    # 备注名（this bot account's note on the contact）

    def all_names(self) -> list[str]:
        """All non-empty searchable names, in display-priority order."""
        return [n for n in (self.remark, self.nickname, self.alias) if n]

    def primary(self) -> str:
        """Best name for display: remark > nickname > alias > '' ."""
        return self.remark or self.nickname or self.alias or ""


class WeChatChannel(BaseChannel):
    """
    WeChat channel using wcferry.

    - Private chats: Direct reply
    - Group chats: Only reply when @mentioned
    - Supports: text, image, voice, video, file, and quoted messages
    """

    name = "wechat"

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
        self._wechat_home: str = ""  # WeChat file storage base directory
        self._recv_thread: Thread | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._contacts: dict[str, WeChatContact] = {}  # wxid -> WeChatContact

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
            # Get user info including home directory for file downloads
            user_info = self.wcf.get_user_info()
            self._wechat_home = user_info.get("home", "")
            logger.info("WeChat connected as {} (home: {})", self.wxid, self._wechat_home)
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
        """Load full Contact records from WeChat database (nickname + alias + remark)."""
        if not self.wcf:
            return
        try:
            contacts = self.wcf.query_sql(
                "MicroMsg.db",
                "SELECT UserName, NickName, Alias, Remark FROM Contact;",
            )
            self._contacts = {
                c["UserName"]: WeChatContact(
                    nickname=c.get("NickName") or "",
                    alias=c.get("Alias") or "",
                    remark=c.get("Remark") or "",
                )
                for c in contacts
            }
            logger.info("Loaded {} contacts", len(self._contacts))
        except Exception as e:
            logger.warning("Failed to load contacts: {}", e)

    def _get_sender_name(self, sender_id: str, room_id: str | None = None) -> str | None:
        """Resolve display name: 群昵称 → 备注名 → 微信昵称 → 微信号."""
        if not self.wcf:
            return None

        # For group chat, try to get the per-room alias (群昵称) first
        if room_id:
            try:
                alias = self.wcf.get_alias_in_chatroom(sender_id, room_id)
                if alias and alias.strip():
                    return alias
            except Exception:
                pass

        contact = self._contacts.get(sender_id)
        return contact.primary() if contact else None

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
            text = msg.content
            # In groups, convert WeChat-native "@nickname " mentions to `<@wxid>`
            # markers so the model sees a uniform format. Falls back to stripping
            # @me when no aters info is available.
            if is_group:
                aters = self._parse_atuserlist(msg.xml)
                if aters:
                    text = self._convert_inbound_mentions(text, aters)
                elif is_at_me:
                    text = self._strip_at_mention(text)

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
            text, media_paths, is_reply_to_me = await self._process_app_msg(msg, chat_id)
            # For quoted messages in groups, check if replying to bot or @mentioned
            if is_group:
                if is_reply_to_me:
                    is_at_me = True
                elif text and self._is_at_in_text(text):
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

    @staticmethod
    def _parse_atuserlist(xml: str | None) -> list[str]:
        """Extract @-ed wxids from msg.xml's <atuserlist> element (ordered)."""
        if not xml:
            return []
        m = re.search(r"<atuserlist>(?:<!\[CDATA\[)?(.*?)(?:\]\]>)?</atuserlist>", xml, re.S)
        if not m:
            return []
        return [w for w in m.group(1).strip().strip(",").split(",") if w]

    @staticmethod
    def _convert_inbound_mentions(text: str, aters: list[str]) -> str:
        """Replace `@<token>` substrings positionally with `<@wxid>` markers.

        Order in `aters` is assumed to match the order of `@` substrings in text
        (the WeChat client emits them aligned).
        """
        if not aters or not text:
            return text
        it = iter(aters)

        def repl(_m: re.Match) -> str:
            try:
                return f"<@{next(it)}>"
            except StopIteration:
                return _m.group(0)

        return re.sub(r"@\S+", repl, text)

    def _translate_outbound_mentions(self, text: str, chat_id: str) -> tuple[str, str]:
        """Convert `<@wxid>` markers to WeChat `@nickname` format + aters CSV.

        WeChat requires the outbound text to contain N `@` substrings matching
        the N wxids in `aters`. We resolve the nickname (group alias preferred,
        falling back to contact nickname or wxid) so the visible text is friendly.
        """
        aters: list[str] = []
        room_id = chat_id if chat_id.endswith("@chatroom") else None

        def repl(wxid: str) -> str:
            aters.append(wxid)
            nickname = self._get_sender_name(wxid, room_id) or wxid
            # U+2005 is the four-per-em space WeChat itself uses after @nickname
            return f"@{nickname} "

        return replace_mentions(text, repl), ",".join(aters)

    def _is_at_in_text(self, text: str) -> bool:
        """True if text contains `@<bot-name>` under any of the bot's known names."""
        if not self.wcf:
            return False
        contact = self._contacts.get(self.wxid)
        names = contact.all_names() if contact else []
        return any(name and f"@{name}" in text for name in names)

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
                logger.debug(
                    "wcf.download_image start msg_id={} extra={!r} media_dir={} timeout=30",
                    msg.id, msg.extra, media_dir,
                )
                file_path = await loop.run_in_executor(
                    None,
                    lambda: self.wcf.download_image(msg.id, msg.extra, str(media_dir), timeout=30)
                )
                logger.debug("wcf.download_image returned: {!r}", file_path)
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
            if media_type == "image" and file_path:
                logger.warning(
                    "wcf.download_image returned path but file not found: {}", file_path,
                )

        except Exception as e:
            logger.error("Error downloading {}: {}", media_type, e)

        if media_type == "image":
            logger.warning(
                "Image download gave up: msg_id={}, extra_present={}, last_return={!r}",
                msg.id, bool(msg.extra), file_path,
            )
        return None, f"[{media_type}: download failed]"

    def _find_file_in_wechat_storage(self, filename: str) -> str | None:
        """
        Search for a file in WeChat's FileStorage directory.

        WeChat stores auto-downloaded files at:
        {home}/{wxid}/FileStorage/File/{year-month}/{filename}

        Returns the full path if found, None otherwise.
        """
        from datetime import datetime
        from pathlib import Path

        if not self._wechat_home or not self.wxid:
            return None

        base_path = Path(self._wechat_home) / self.wxid / "FileStorage" / "File"
        if not base_path.exists():
            return None

        # Current month directory (most likely location)
        current_month = datetime.now().strftime("%Y-%m")
        current_dir = base_path / current_month
        if current_dir.exists():
            file_path = current_dir / filename
            if file_path.exists():
                return str(file_path)

        # Search in all month directories (descending order, recent first)
        try:
            for month_dir in sorted(base_path.iterdir(), reverse=True):
                if month_dir.is_dir():
                    file_path = month_dir / filename
                    if file_path.exists():
                        return str(file_path)
        except Exception as e:
            logger.debug("Error searching WeChat storage: {}", e)

        return None

    async def _download_file(
        self,
        msg: WxMsg,
        filename: str,
        chat_id: str,
        timeout: int = 30,
    ) -> tuple[str | None, str]:
        """
        Download a file attachment from WeChat.

        Args:
            msg: The WeChat message object
            filename: The filename from XML
            chat_id: Chat ID for session binding
            timeout: Download timeout in seconds

        Returns:
            (file_path, content_text)
        """
        import shutil

        # Check session binding first
        media_dir = self._get_media_dir(chat_id)
        if media_dir is None:
            return None, f"[文件: {filename}] (请先使用 /session <name> 绑定工作区)"

        if not self.wcf:
            return None, f"[文件: {filename}]"

        extra = msg.extra or ""
        thumb = msg.thumb or ""

        # Log full message info for debugging
        logger.debug("File message details: id={}, thumb='{}', extra='{}', xml_len={}, content_len={}",
                     msg.id, thumb[:80], extra[:80], len(msg.xml or ""), len(msg.content or ""))

        # For file messages, try to extract attach info from XML content
        # File XML format: <appattach><totallen>...</totallen><attachid>...</attachid>...</appattach>
        if not extra and msg.content:
            # Try to find file path in content XML
            attachid_match = re.search(r'<attachid>(.*?)</attachid>', msg.content)
            cdnurl_match = re.search(r'<cdnattachurl>(.*?)</cdnattachurl>', msg.content)
            if attachid_match:
                logger.debug("Found attachid in XML: {}", attachid_match.group(1)[:50])
            if cdnurl_match:
                logger.debug("Found cdnattachurl in XML: {}", cdnurl_match.group(1)[:80])
            # Use content XML as extra for download_attach
            extra = msg.content

        logger.debug("Attempting to download file '{}': id={}", filename, msg.id)

        # If extra already points to an existing file, use it directly
        if extra and os.path.exists(extra):
            dest_path = media_dir / filename
            shutil.copy2(extra, dest_path)
            logger.debug("File already exists, copied to {}", dest_path)
            return str(dest_path), f"[文件: <work_dir>/data/{filename}]"

        # For auto-downloaded files (like PDF/xlsx with auto-download enabled),
        # check WeChat's FileStorage directory
        wechat_file = self._find_file_in_wechat_storage(filename)
        if wechat_file:
            dest_path = media_dir / filename
            shutil.copy2(wechat_file, dest_path)
            logger.debug("Found file in WeChat storage, copied to {}", dest_path)
            return str(dest_path), f"[文件: <work_dir>/data/{filename}]"

        try:
            loop = asyncio.get_running_loop()

            # For PDF/xlsx etc., WeChat needs the user to "accept" the file first
            # msg.extra will be empty until the file is accepted in WeChat client
            original_extra = msg.extra or ""

            # Trigger the download
            ret = await loop.run_in_executor(
                None,
                lambda: self.wcf.download_attach(msg.id, thumb, extra)
            )

            if ret != 0:
                logger.warning("download_attach returned {} (file may need manual accept in WeChat)", ret)

            # Wait for file to be available
            await asyncio.sleep(2)  # Wait for download to complete

            # Check WeChat storage again after triggering download
            wechat_file = self._find_file_in_wechat_storage(filename)
            if wechat_file:
                dest_path = media_dir / filename
                shutil.copy2(wechat_file, dest_path)
                logger.debug("Found file in WeChat storage after download, copied to {}", dest_path)
                return str(dest_path), f"[文件: <work_dir>/data/{filename}]"

            # Check if original extra path exists (for small files that auto-download)
            if original_extra:
                for _ in range(timeout):
                    if os.path.exists(original_extra):
                        dest_path = media_dir / filename
                        shutil.copy2(original_extra, dest_path)
                        logger.debug("Downloaded file to {}", dest_path)
                        return str(dest_path), f"[文件: <work_dir>/data/{filename}]"
                    await asyncio.sleep(1)

            # Final check in WeChat storage
            wechat_file = self._find_file_in_wechat_storage(filename)
            if wechat_file:
                dest_path = media_dir / filename
                shutil.copy2(wechat_file, dest_path)
                logger.debug("Found file in WeChat storage (final check), copied to {}", dest_path)
                return str(dest_path), f"[文件: <work_dir>/data/{filename}]"

            # For large files (PDF, xlsx), they need manual accept in WeChat
            # Return a message indicating this
            logger.info("File '{}' not found in WeChat storage: {}/{}/FileStorage/File/",
                       filename, self._wechat_home, self.wxid)
            return None, f"[文件: {filename}] (请在微信中点击接收后重新发送)"

        except Exception as e:
            logger.error("Error downloading file '{}': {}", filename, e)
            return None, f"[文件: {filename}] (下载失败)"

    async def _process_app_msg(
        self,
        msg: WxMsg,
        chat_id: str,
    ) -> tuple[str, list[str], bool]:
        """
        Process APP type messages (type=49): files, links, quotes, etc.

        Returns:
            (text, media_paths, is_reply_to_me)
        """
        content = msg.content

        # Detect appmsg type
        appmsg_type_match = re.search(r'<type>(\d+)</type>', content)
        if not appmsg_type_match:
            return "", [], False

        appmsg_type = appmsg_type_match.group(1)

        # Type 57: Quote/Reply message
        if appmsg_type == "57":
            return await self._parse_quote_msg(content, chat_id)

        # Type 6: File
        if appmsg_type == "6":
            title_match = re.search(r'<title>(.*?)</title>', content)
            filename = title_match.group(1) if title_match else "file"
            file_path, content_text = await self._download_file(msg, filename, chat_id)
            if file_path:
                return content_text, [file_path], False
            return content_text, [], False

        # Type 5: Link
        if appmsg_type == "5":
            title_match = re.search(r'<title>(.*?)</title>', content)
            url_match = re.search(r'<url>(.*?)</url>', content)
            title = html.unescape(title_match.group(1)) if title_match else ""
            url = html.unescape(url_match.group(1)) if url_match else ""
            if title and url:
                return f"[链接: {title}]\n{url}", [], False
            elif title:
                return f"[链接: {title}]", [], False
            return "[链接]", [], False

        # Type 33/36: Mini Program
        if appmsg_type in ("33", "36"):
            title_match = re.search(r'<title>(.*?)</title>', content)
            title = html.unescape(title_match.group(1)) if title_match else "小程序"
            return f"[小程序: {title}]", [], False

        # Type 19: Chat record forward
        if appmsg_type == "19":
            title_match = re.search(r'<title>(.*?)</title>', content)
            title = html.unescape(title_match.group(1)) if title_match else "聊天记录"
            return f"[聊天记录: {title}]", [], False

        # Other types: try to extract title
        title_match = re.search(r'<title>(.*?)</title>', content)
        if title_match:
            title = html.unescape(title_match.group(1))
            return f"[消息: {title}]", [], False

        return "[未知消息类型]", [], False

    async def _parse_quote_msg(self, content: str, chat_id: str) -> tuple[str, list[str], bool]:
        """
        Parse quoted/reply message (appmsg type=57).

        Returns:
            (text, media_paths, is_reply_to_me)
        """
        result_parts = []
        media_paths: list[str] = []
        is_reply_to_me = False

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

            # Get quoted sender wxid (chatusr field)
            chatusr_match = re.search(r'<chatusr>(.*?)</chatusr>', refermsg)
            quoted_wxid = chatusr_match.group(1) if chatusr_match else ""

            # Check if replying to bot's message
            if quoted_wxid and quoted_wxid == self.wxid:
                is_reply_to_me = True

            # Get quoted sender display name
            displayname_match = re.search(r'<displayname>(.*?)</displayname>', refermsg)
            quoted_sender = html.unescape(displayname_match.group(1)) if displayname_match else ""

            # Get quoted message type
            type_match = re.search(r'<type>(\d+)</type>', refermsg)
            quoted_type = type_match.group(1) if type_match else "1"

            # Get quoted message ID (svrid)
            svrid_match = re.search(r'<svrid>(\d+)</svrid>', refermsg)
            quoted_msg_id = svrid_match.group(1) if svrid_match else None

            # Get quoted content
            content_match = re.search(r'<content>(.*?)</content>', refermsg, re.DOTALL)
            quoted_raw = content_match.group(1) if content_match else ""
            quoted_decoded = html.unescape(quoted_raw)

            # Type 3 = image - try to download
            if quoted_type == "3":
                file_path, quoted_text = await self._download_quoted_image(
                    quoted_msg_id, quoted_decoded, chat_id
                )
                if file_path:
                    media_paths.append(file_path)
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

        return "\n".join(result_parts), media_paths, is_reply_to_me

    async def _download_quoted_image(
        self, msg_id: str | None, content_xml: str, chat_id: str
    ) -> tuple[str | None, str]:
        """
        Download a quoted image message.

        Args:
            msg_id: The svrid of the quoted message
            content_xml: The decoded content XML of the quoted image (raw_content)
            chat_id: Chat ID for session binding

        Returns:
            (file_path, content_text)
        """
        if not msg_id or not self.wcf:
            return None, "[图片]"

        # Check session binding
        media_dir = self._get_media_dir(chat_id)
        if media_dir is None:
            return None, "[图片: 请先使用 /session <name> 绑定工作区]"

        # wcferry 下载图片时按 md5 命名落到 data/；quote XML 里 <img md5="..."/>
        # 就是同一张图。命中本地副本就直接复用，避开旧消息 cdn 已过期导致的
        # FUNC_DOWNLOAD_ATTACH 超时。
        md5_match = re.search(r'md5="([0-9a-f]{32})"', content_xml)
        if md5_match:
            md5 = md5_match.group(1)
            for ext in self._IMAGE_EXTS:
                cached = media_dir / f"{md5}{ext}"
                if cached.exists():
                    logger.debug("Reusing cached quoted image: {}", cached)
                    return str(cached), f"[图片: <work_dir>/data/{md5}{ext}]"

        # Use the decoded content XML directly as extra parameter
        # wcferry's download_attach can parse this XML to get the image
        extra = content_xml
        logger.debug("Attempting to download quoted image: msg_id={}, extra_len={}", msg_id, len(extra))

        try:
            loop = asyncio.get_running_loop()
            file_path = await loop.run_in_executor(
                None,
                lambda: self.wcf.download_image(int(msg_id), extra, str(media_dir), timeout=30)
            )
            if file_path and os.path.exists(file_path):
                filename = os.path.basename(file_path)
                logger.debug("Downloaded quoted image to {}", file_path)
                return file_path, f"[图片: <work_dir>/data/{filename}]"
            else:
                logger.warning("download_image returned empty or non-existent path for msg_id={}", msg_id)
        except Exception as e:
            logger.warning("Failed to download quoted image {}: {}", msg_id, e)

        return None, "[图片]"

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
                text, aters = self._translate_outbound_mentions(msg.content, msg.chat_id)
                self.wcf.send_text(text, msg.chat_id, aters)
                logger.debug(
                    "Sent message to {}: {}... (aters={})",
                    msg.chat_id, text[:50], aters or "-",
                )
        except Exception as e:
            logger.error("Failed to send WeChat message: {}", e)

    async def get_group_members(self, chat_id: str) -> list[dict[str, object]]:
        """Return member records for a WeChat group.

        Each record: ``{id: wxid, names: {label: value, ...}}`` where `names`
        contains every non-empty identifier we know about, labeled for the
        model (群昵称 / 备注名 / 微信昵称 / 微信号). The @ marker uses `id`.
        """
        if not self.wcf or not chat_id.endswith("@chatroom"):
            return []
        loop = asyncio.get_running_loop()
        try:
            members = await loop.run_in_executor(None, self.wcf.get_chatroom_members, chat_id)
        except Exception as e:
            logger.warning("Failed to fetch chatroom members for {}: {}", chat_id, e)
            return []

        def _build(wxid: str, room_nickname: str) -> dict[str, object]:
            try:
                room_alias = self.wcf.get_alias_in_chatroom(wxid, chat_id) or ""
            except Exception:
                room_alias = ""
            contact = self._contacts.get(wxid) or WeChatContact()
            names: dict[str, str] = {}
            if room_alias:
                names["群昵称"] = room_alias
            if contact.remark:
                names["备注名"] = contact.remark
            if contact.nickname:
                names["微信昵称"] = contact.nickname
            if room_nickname and room_nickname not in names.values():
                # Distinct from above (rare, but possible if wcferry rosters drift)
                names.setdefault("成员表昵称", room_nickname)
            if contact.alias:
                names["微信号"] = contact.alias
            return {"id": wxid, "names": names}

        return [
            await loop.run_in_executor(None, _build, wxid, nickname)
            for wxid, nickname in (members or {}).items()
        ]

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
