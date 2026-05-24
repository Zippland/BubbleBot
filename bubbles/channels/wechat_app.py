"""WeChat APP-type message (msg.type == 49) parsing: files, links, quotes.

Pulled out of `wechat.py` so the channel class only carries connection /
dispatch / mention concerns. Pure free functions — caller passes the wcferry
client and the resolved per-chat media dir.
"""

from __future__ import annotations

import asyncio
import html
import os
import re
import shutil
from datetime import datetime
from pathlib import Path

from loguru import logger


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"}


def find_file_in_wechat_storage(
    wechat_home: str, wxid: str, filename: str
) -> str | None:
    """Search WeChat's FileStorage directory for an auto-downloaded file.

    Files land at ``{wechat_home}/{wxid}/FileStorage/File/{YYYY-MM}/{filename}``.
    Search current month first, then any month in descending order.
    """
    if not wechat_home or not wxid:
        return None

    base_path = Path(wechat_home) / wxid / "FileStorage" / "File"
    if not base_path.exists():
        return None

    current_month = datetime.now().strftime("%Y-%m")
    current_dir = base_path / current_month
    if current_dir.exists():
        candidate = current_dir / filename
        if candidate.exists():
            return str(candidate)

    try:
        for month_dir in sorted(base_path.iterdir(), reverse=True):
            if month_dir.is_dir():
                candidate = month_dir / filename
                if candidate.exists():
                    return str(candidate)
    except Exception as e:
        logger.debug("Error searching WeChat storage: {}", e)

    return None


async def download_file(
    wcf,
    msg,
    filename: str,
    media_dir: Path | None,
    *,
    wechat_home: str,
    wxid: str,
    timeout: int = 30,
) -> tuple[str | None, str]:
    """Download a file attachment for an APP type=6 message."""
    if media_dir is None:
        return None, f"[文件: {filename}] (请先使用 /session <name> 绑定工作区)"
    if not wcf:
        return None, f"[文件: {filename}]"

    extra = msg.extra or ""
    thumb = msg.thumb or ""

    logger.debug(
        "File message details: id={}, thumb='{}', extra='{}', xml_len={}, content_len={}",
        msg.id, thumb[:80], extra[:80], len(msg.xml or ""), len(msg.content or ""),
    )

    if not extra and msg.content:
        attachid_match = re.search(r'<attachid>(.*?)</attachid>', msg.content)
        cdnurl_match = re.search(r'<cdnattachurl>(.*?)</cdnattachurl>', msg.content)
        if attachid_match:
            logger.debug("Found attachid in XML: {}", attachid_match.group(1)[:50])
        if cdnurl_match:
            logger.debug("Found cdnattachurl in XML: {}", cdnurl_match.group(1)[:80])
        extra = msg.content

    logger.debug("Attempting to download file '{}': id={}", filename, msg.id)

    if extra and os.path.exists(extra):
        dest_path = media_dir / filename
        shutil.copy2(extra, dest_path)
        logger.debug("File already exists, copied to {}", dest_path)
        return str(dest_path), f"[文件: <work_dir>/data/{filename}]"

    wechat_file = find_file_in_wechat_storage(wechat_home, wxid, filename)
    if wechat_file:
        dest_path = media_dir / filename
        shutil.copy2(wechat_file, dest_path)
        logger.debug("Found file in WeChat storage, copied to {}", dest_path)
        return str(dest_path), f"[文件: <work_dir>/data/{filename}]"

    try:
        loop = asyncio.get_running_loop()
        original_extra = msg.extra or ""

        ret = await loop.run_in_executor(
            None,
            lambda: wcf.download_attach(msg.id, thumb, extra),
        )
        if ret != 0:
            logger.warning(
                "download_attach returned {} (file may need manual accept in WeChat)",
                ret,
            )

        await asyncio.sleep(2)

        wechat_file = find_file_in_wechat_storage(wechat_home, wxid, filename)
        if wechat_file:
            dest_path = media_dir / filename
            shutil.copy2(wechat_file, dest_path)
            logger.debug(
                "Found file in WeChat storage after download, copied to {}", dest_path,
            )
            return str(dest_path), f"[文件: <work_dir>/data/{filename}]"

        if original_extra:
            for _ in range(timeout):
                if os.path.exists(original_extra):
                    dest_path = media_dir / filename
                    shutil.copy2(original_extra, dest_path)
                    logger.debug("Downloaded file to {}", dest_path)
                    return str(dest_path), f"[文件: <work_dir>/data/{filename}]"
                await asyncio.sleep(1)

        wechat_file = find_file_in_wechat_storage(wechat_home, wxid, filename)
        if wechat_file:
            dest_path = media_dir / filename
            shutil.copy2(wechat_file, dest_path)
            logger.debug(
                "Found file in WeChat storage (final check), copied to {}", dest_path,
            )
            return str(dest_path), f"[文件: <work_dir>/data/{filename}]"

        logger.info(
            "File '{}' not found in WeChat storage: {}/{}/FileStorage/File/",
            filename, wechat_home, wxid,
        )
        return None, f"[文件: {filename}] (请在微信中点击接收后重新发送)"

    except Exception as e:
        logger.error("Error downloading file '{}': {}", filename, e)
        return None, f"[文件: {filename}] (下载失败)"


def _find_cached_quoted_image(media_dir: Path, content_xml: str) -> Path | None:
    """Find a previously-downloaded image whose hex filename stem appears in the
    quote XML. wcferry names downloaded images by md5; quote messages reference
    the same image by its hash somewhere in the XML, so a verbatim stem-in-xml
    match avoids assuming any specific WeChat-version attribute format.
    """
    if not media_dir or not media_dir.is_dir():
        return None
    xml_lower = content_xml.lower()
    for p in media_dir.iterdir():
        if p.suffix.lower() not in IMAGE_EXTS:
            continue
        stem = p.stem.lower()
        # md5 hex is 32 chars; widen to 16–64 so a future hash change still matches.
        if 16 <= len(stem) <= 64 and stem in xml_lower:
            return p
    return None


async def download_quoted_image(
    wcf,
    msg_id: str | None,
    content_xml: str,
    media_dir: Path | None,
    msg_id_to_path: dict[int, str] | None = None,
) -> tuple[str | None, str]:
    """Resolve a quoted image — prefer in-process msg_id cache, fall back to
    media_dir scan, finally to wcferry download."""
    if not msg_id or not wcf:
        return None, "[图片]"
    if media_dir is None:
        return None, "[图片: 请先使用 /session <name> 绑定工作区]"

    # 1. 进程内 msg.id → 路径缓存（最稳：wcferry 的 <svrid> 就是原图 msg.id）
    if msg_id_to_path:
        try:
            mid = int(msg_id)
        except (TypeError, ValueError):
            mid = None
        if mid is not None:
            hit = msg_id_to_path.get(mid)
            if hit and os.path.exists(hit):
                logger.debug("Reusing image by msg_id {}: {}", mid, hit)
                return hit, f"[图片: <work_dir>/data/{os.path.basename(hit)}]"

    # 2. fallback：扫 media_dir 找文件名 stem 出现在 XML 里的图（重启后老消息
    #    走这条；命中率取决于 WeChat 版本是否在 quote XML 里写了解密后的 md5）
    cached = _find_cached_quoted_image(media_dir, content_xml)
    if cached:
        logger.debug("Reusing cached quoted image by stem-in-xml: {}", cached)
        return str(cached), f"[图片: <work_dir>/data/{cached.name}]"

    extra = content_xml
    logger.debug(
        "Attempting to download quoted image: msg_id={}, extra_len={}",
        msg_id, len(extra),
    )
    try:
        loop = asyncio.get_running_loop()
        file_path = await loop.run_in_executor(
            None,
            lambda: wcf.download_image(int(msg_id), extra, str(media_dir), timeout=30),
        )
        if file_path and os.path.exists(file_path):
            filename = os.path.basename(file_path)
            logger.debug("Downloaded quoted image to {}", file_path)
            return file_path, f"[图片: <work_dir>/data/{filename}]"
        logger.warning(
            "download_image returned empty or non-existent path for msg_id={}", msg_id,
        )
    except Exception as e:
        logger.warning("Failed to download quoted image {}: {}", msg_id, e)

    return None, "[图片]"


async def parse_quote_msg(
    wcf,
    content: str,
    media_dir: Path | None,
    *,
    bot_wxid: str,
    msg_id_to_path: dict[int, str] | None = None,
) -> tuple[str, list[str], bool]:
    """Parse APP appmsg type=57 (quote/reply).

    Returns ``(text, media_paths, is_reply_to_me)``.
    """
    result_parts: list[str] = []
    media_paths: list[str] = []
    is_reply_to_me = False

    title_match = re.search(r'<title>(.*?)</title>', content)
    if title_match:
        new_content = html.unescape(title_match.group(1).strip())
        if new_content:
            result_parts.append(new_content)

    refermsg_match = re.search(r'<refermsg>(.*?)</refermsg>', content, re.DOTALL)
    if refermsg_match:
        refermsg = refermsg_match.group(1)

        chatusr_match = re.search(r'<chatusr>(.*?)</chatusr>', refermsg)
        quoted_wxid = chatusr_match.group(1) if chatusr_match else ""

        if quoted_wxid and quoted_wxid == bot_wxid:
            is_reply_to_me = True

        displayname_match = re.search(r'<displayname>(.*?)</displayname>', refermsg)
        quoted_sender = html.unescape(displayname_match.group(1)) if displayname_match else ""

        type_match = re.search(r'<type>(\d+)</type>', refermsg)
        quoted_type = type_match.group(1) if type_match else "1"

        svrid_match = re.search(r'<svrid>(\d+)</svrid>', refermsg)
        quoted_msg_id = svrid_match.group(1) if svrid_match else None

        content_match = re.search(r'<content>(.*?)</content>', refermsg, re.DOTALL)
        quoted_raw = content_match.group(1) if content_match else ""
        quoted_decoded = html.unescape(quoted_raw)

        if quoted_type == "3":
            file_path, quoted_text = await download_quoted_image(
                wcf, quoted_msg_id, quoted_decoded, media_dir,
                msg_id_to_path=msg_id_to_path,
            )
            if file_path:
                media_paths.append(file_path)
        elif quoted_type == "34":
            quoted_text = "[语音]"
        elif quoted_type == "43":
            quoted_text = "[视频]"
        elif quoted_type == "49":
            inner_title = re.search(r'<title>(.*?)</title>', quoted_decoded)
            if inner_title:
                quoted_text = f"[{html.unescape(inner_title.group(1))}]"
            else:
                quoted_text = "[消息]"
        else:
            quoted_text = re.sub(r'<.*?>', '', quoted_decoded).strip()
            quoted_text = re.sub(r'\s+', ' ', quoted_text)

        if quoted_sender and quoted_text:
            result_parts.append(f"[引用 {quoted_sender}: {quoted_text}]")
        elif quoted_text:
            result_parts.append(f"[引用: {quoted_text}]")

    return "\n".join(result_parts), media_paths, is_reply_to_me


async def process_app_msg(
    wcf,
    msg,
    media_dir: Path | None,
    *,
    wechat_home: str,
    bot_wxid: str,
    msg_id_to_path: dict[int, str] | None = None,
) -> tuple[str, list[str], bool]:
    """Dispatch APP type=49 messages by inner appmsg <type>.

    Returns ``(text, media_paths, is_reply_to_me)``.
    """
    content = msg.content

    appmsg_type_match = re.search(r'<type>(\d+)</type>', content)
    if not appmsg_type_match:
        return "", [], False

    appmsg_type = appmsg_type_match.group(1)

    # Type 57: Quote/Reply
    if appmsg_type == "57":
        return await parse_quote_msg(
            wcf, content, media_dir,
            bot_wxid=bot_wxid, msg_id_to_path=msg_id_to_path,
        )

    # Type 6: File
    if appmsg_type == "6":
        title_match = re.search(r'<title>(.*?)</title>', content)
        filename = title_match.group(1) if title_match else "file"
        file_path, content_text = await download_file(
            wcf, msg, filename, media_dir,
            wechat_home=wechat_home, wxid=bot_wxid,
        )
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

    title_match = re.search(r'<title>(.*?)</title>', content)
    if title_match:
        title = html.unescape(title_match.group(1))
        return f"[消息: {title}]", [], False

    return "[未知消息类型]", [], False
