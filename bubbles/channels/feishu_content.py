"""Pure parsers for Feishu rich-content payloads.

Handles share cards / interactive cards / post (rich-text) message bodies.
Pulled out of `feishu.py` so the channel class only carries SDK / dispatch
concerns. State-free, unit-testable, no SDK dependency.
"""

from __future__ import annotations

import json


def _extract_share_card_content(content_json: dict, msg_type: str) -> str:
    """Extract text representation from share cards and interactive messages."""
    parts = []

    if msg_type == "share_chat":
        parts.append(f"[shared chat: {content_json.get('chat_id', '')}]")
    elif msg_type == "share_user":
        parts.append(f"[shared user: {content_json.get('user_id', '')}]")
    elif msg_type == "interactive":
        parts.extend(_extract_interactive_content(content_json))
    elif msg_type == "share_calendar_event":
        parts.append(f"[shared calendar event: {content_json.get('event_key', '')}]")
    elif msg_type == "system":
        parts.append("[system message]")
    elif msg_type == "merge_forward":
        parts.append("[merged forward messages]")

    return "\n".join(parts) if parts else f"[{msg_type}]"


def _extract_interactive_content(content: dict) -> list[str]:
    """Recursively extract text and links from interactive card content."""
    parts = []

    if isinstance(content, str):
        try:
            content = json.loads(content)
        except (json.JSONDecodeError, TypeError):
            return [content] if content.strip() else []

    if not isinstance(content, dict):
        return parts

    if "title" in content:
        title = content["title"]
        if isinstance(title, dict):
            title_content = title.get("content", "") or title.get("text", "")
            if title_content:
                parts.append(f"title: {title_content}")
        elif isinstance(title, str):
            parts.append(f"title: {title}")

    for element in content.get("elements", []) if isinstance(content.get("elements"), list) else []:
        parts.extend(_extract_element_content(element))

    card = content.get("card", {})
    if card:
        parts.extend(_extract_interactive_content(card))

    header = content.get("header", {})
    if header:
        header_title = header.get("title", {})
        if isinstance(header_title, dict):
            header_text = header_title.get("content", "") or header_title.get("text", "")
            if header_text:
                parts.append(f"title: {header_text}")

    return parts


def _extract_element_content(element: dict) -> list[str]:
    """Extract content from a single card element."""
    parts = []

    if not isinstance(element, dict):
        return parts

    tag = element.get("tag", "")

    if tag in ("markdown", "lark_md"):
        content = element.get("content", "")
        if content:
            parts.append(content)

    elif tag == "div":
        text = element.get("text", {})
        if isinstance(text, dict):
            text_content = text.get("content", "") or text.get("text", "")
            if text_content:
                parts.append(text_content)
        elif isinstance(text, str):
            parts.append(text)
        for field in element.get("fields", []):
            if isinstance(field, dict):
                field_text = field.get("text", {})
                if isinstance(field_text, dict):
                    c = field_text.get("content", "")
                    if c:
                        parts.append(c)

    elif tag == "a":
        href = element.get("href", "")
        text = element.get("text", "")
        if href:
            parts.append(f"link: {href}")
        if text:
            parts.append(text)

    elif tag == "button":
        text = element.get("text", {})
        if isinstance(text, dict):
            c = text.get("content", "")
            if c:
                parts.append(c)
        url = element.get("url", "") or element.get("multi_url", {}).get("url", "")
        if url:
            parts.append(f"link: {url}")

    elif tag == "img":
        alt = element.get("alt", {})
        parts.append(alt.get("content", "[image]") if isinstance(alt, dict) else "[image]")

    elif tag == "note":
        for ne in element.get("elements", []):
            parts.extend(_extract_element_content(ne))

    elif tag == "column_set":
        for col in element.get("columns", []):
            for ce in col.get("elements", []):
                parts.extend(_extract_element_content(ce))

    elif tag == "plain_text":
        content = element.get("content", "")
        if content:
            parts.append(content)

    else:
        for ne in element.get("elements", []):
            parts.extend(_extract_element_content(ne))

    return parts


def _extract_post_content(content_json: dict) -> tuple[str, list[str]]:
    """Extract text and image keys from Feishu post (rich text) message content.

    Supports two formats:
    1. Direct format: {"title": "...", "content": [...]}
    2. Localized format: {"zh_cn": {"title": "...", "content": [...]}}

    Returns:
        (text, image_keys) - extracted text and list of image keys
    """
    def extract_from_lang(lang_content: dict) -> tuple[str | None, list[str]]:
        if not isinstance(lang_content, dict):
            return None, []
        title = lang_content.get("title", "")
        content_blocks = lang_content.get("content", [])
        if not isinstance(content_blocks, list):
            return None, []
        text_parts = []
        image_keys = []
        if title:
            text_parts.append(title)
        for block in content_blocks:
            if not isinstance(block, list):
                continue
            for element in block:
                if isinstance(element, dict):
                    tag = element.get("tag")
                    if tag == "text":
                        text_parts.append(element.get("text", ""))
                    elif tag == "a":
                        text_parts.append(element.get("text", ""))
                    elif tag == "at":
                        text_parts.append(f"@{element.get('user_name', 'user')}")
                    elif tag == "img":
                        img_key = element.get("image_key")
                        if img_key:
                            image_keys.append(img_key)
        text = " ".join(text_parts).strip() if text_parts else None
        return text, image_keys

    # Try direct format first
    if "content" in content_json:
        text, images = extract_from_lang(content_json)
        if text or images:
            return text or "", images

    # Try localized format
    for lang_key in ("zh_cn", "en_us", "ja_jp"):
        lang_content = content_json.get(lang_key)
        text, images = extract_from_lang(lang_content)
        if text or images:
            return text or "", images

    return "", []


def _extract_post_text(content_json: dict) -> str:
    """Extract plain text from Feishu post (rich text) message content.

    Legacy wrapper for _extract_post_content, returns only text.
    """
    text, _ = _extract_post_content(content_json)
    return text
