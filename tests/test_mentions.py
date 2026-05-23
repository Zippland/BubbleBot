"""Tests for cross-channel `<@id>` mention markup + find_person tool."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from bubbles.agent.tools.find_person import FindPersonTool
from bubbles.channels.mentions import (
    MENTION_RE,
    extract_mentions,
    replace_mentions,
    split_mention_text,
)
from bubbles.channels.wechat import WeChatChannel


# ---------- mentions.py: markup utilities ----------

def test_mention_re_matches_common_id_shapes() -> None:
    assert MENTION_RE.match("<@abc>")
    assert MENTION_RE.match("<@wxid_3rxz4>")
    assert MENTION_RE.match("<@ou_abc-def_123>")  # Feishu open_id with hyphen/underscore
    assert MENTION_RE.match("<@12345>")           # numeric (Telegram-style)
    assert not MENTION_RE.match("<@>")            # empty id
    assert not MENTION_RE.match("<@id with space>")


def test_extract_mentions_preserves_order_and_duplicates() -> None:
    assert extract_mentions("hello <@a> world <@b> end") == ["a", "b"]
    assert extract_mentions("<@x> <@x> <@y>") == ["x", "x", "y"]
    assert extract_mentions("") == []
    assert extract_mentions(None) == []
    assert extract_mentions("no mentions here") == []


def test_split_mention_text() -> None:
    assert split_mention_text("hi <@a> there") == [
        ("text", "hi "),
        ("mention", "a"),
        ("text", " there"),
    ]
    # Bare mention at start, no surrounding text
    assert split_mention_text("<@a>") == [("mention", "a")]
    # Two mentions back-to-back, no separator
    assert split_mention_text("<@a><@b>") == [("mention", "a"), ("mention", "b")]
    assert split_mention_text("") == []
    assert split_mention_text("plain") == [("text", "plain")]


def test_replace_mentions_applies_fn() -> None:
    out = replace_mentions("hi <@a> and <@b>", lambda i: f"@{i.upper()}")
    assert out == "hi @A and @B"
    # Empty text → empty result
    assert replace_mentions("", lambda i: i) == ""
    assert replace_mentions(None, lambda i: i) == ""


# ---------- WeChat: static helpers (no real wcferry needed) ----------

def test_wechat_parse_atuserlist_cdata() -> None:
    xml = """
    <msgsource>
      <atuserlist><![CDATA[,wxid_1,wxid_2]]></atuserlist>
    </msgsource>
    """
    assert WeChatChannel._parse_atuserlist(xml) == ["wxid_1", "wxid_2"]


def test_wechat_parse_atuserlist_no_cdata() -> None:
    xml = "<atuserlist>wxid_a,wxid_b,wxid_c</atuserlist>"
    assert WeChatChannel._parse_atuserlist(xml) == ["wxid_a", "wxid_b", "wxid_c"]


def test_wechat_parse_atuserlist_missing_or_empty() -> None:
    assert WeChatChannel._parse_atuserlist(None) == []
    assert WeChatChannel._parse_atuserlist("") == []
    assert WeChatChannel._parse_atuserlist("<other>thing</other>") == []
    assert WeChatChannel._parse_atuserlist("<atuserlist></atuserlist>") == []


def test_wechat_convert_inbound_mentions_positional() -> None:
    out = WeChatChannel._convert_inbound_mentions(
        "@张三 @李四 你们看下", ["wxid_zs", "wxid_ls"],
    )
    assert out == "<@wxid_zs> <@wxid_ls> 你们看下"


def test_wechat_convert_inbound_mentions_extra_at_left_alone() -> None:
    """If text has more @s than aters, leftovers stay as-is."""
    out = WeChatChannel._convert_inbound_mentions(
        "@a @b @c", ["wxid_1"],
    )
    assert out == "<@wxid_1> @b @c"


def test_wechat_convert_inbound_mentions_no_aters() -> None:
    """No aters → text untouched (don't wrap stale @text)."""
    assert WeChatChannel._convert_inbound_mentions("@somebody hi", []) == "@somebody hi"


# ---------- Feishu: inbound mention resolution (no real SDK needed) ----------

def test_feishu_resolve_inbound_mentions() -> None:
    from bubbles.channels.feishu import FeishuChannel

    class FakeId:
        def __init__(self, open_id): self.open_id = open_id

    class FakeMention:
        def __init__(self, key, open_id):
            self.key = key
            self.id = FakeId(open_id)

    mentions = [FakeMention("@_user_1", "ou_aaa"), FakeMention("@_user_2", "ou_bbb")]
    out = FeishuChannel._resolve_inbound_mentions(
        "@_user_1 你看一下 @_user_2 这个", mentions,
    )
    # Trailing space preserved on each replacement
    assert out == "<@ou_aaa> 你看一下 <@ou_bbb> 这个"


def test_feishu_resolve_inbound_drops_unmapped_placeholders() -> None:
    """@_all and other placeholders without a mapping must be dropped (legacy behavior)."""
    from bubbles.channels.feishu import FeishuChannel
    out = FeishuChannel._resolve_inbound_mentions("@_all hi there", [])
    assert out.strip() == "hi there"


# ---------- find_person tool ----------

def _make_channel_manager(members_by_chat: dict[str, list[dict]] = None,
                          channel_name: str = "wechat"):
    """Build a mock ChannelManager with a single channel returning fixture members."""
    members_by_chat = members_by_chat or {}

    fake_channel = MagicMock()

    async def get_members(chat_id: str):
        return list(members_by_chat.get(chat_id, []))

    fake_channel.get_group_members = AsyncMock(side_effect=get_members)

    cm = MagicMock()
    cm.get_channel = MagicMock(return_value=fake_channel)
    cm.get_channel.return_value._fake_channel = fake_channel
    cm._channel_name = channel_name
    return cm, fake_channel


@pytest.fixture
def tool():
    t = FindPersonTool()
    return t


def test_find_person_requires_query(tool) -> None:
    out = asyncio.run(tool.execute(query=""))
    assert "query" in out.lower()


def test_find_person_requires_channel_manager(tool) -> None:
    tool.set_context("wechat", "room@chatroom")
    out = asyncio.run(tool.execute(query="bob"))
    assert "channel manager" in out.lower()


def test_find_person_requires_chat_context(tool) -> None:
    cm, _ = _make_channel_manager()
    tool.set_channel_manager(cm)
    out = asyncio.run(tool.execute(query="bob"))
    assert "no chat context" in out.lower()


def test_find_person_unknown_channel(tool) -> None:
    cm = MagicMock()
    cm.get_channel = MagicMock(return_value=None)
    tool.set_channel_manager(cm)
    tool.set_context("wechat", "room@chatroom")
    out = asyncio.run(tool.execute(query="bob"))
    assert "not running" in out.lower()


def test_find_person_no_members(tool) -> None:
    cm, _ = _make_channel_manager({"room@chatroom": []})
    tool.set_channel_manager(cm)
    tool.set_context("wechat", "room@chatroom")
    out = asyncio.run(tool.execute(query="bob"))
    assert "no group members" in out.lower()


def test_find_person_single_match_returns_marker(tool) -> None:
    cm, _ = _make_channel_manager({
        "room@chatroom": [
            {"id": "wxid_alice", "name": "Alice"},
            {"id": "wxid_bob", "name": "Bob"},
        ]
    })
    tool.set_channel_manager(cm)
    tool.set_context("wechat", "room@chatroom")
    out = asyncio.run(tool.execute(query="bob"))
    assert "Bob" in out
    assert "<@wxid_bob>" in out
    assert "Alice" not in out


def test_find_person_case_insensitive_substring(tool) -> None:
    cm, _ = _make_channel_manager({
        "room@chatroom": [
            {"id": "1", "name": "张三"},
            {"id": "2", "name": "张三丰"},
            {"id": "3", "name": "李四"},
        ]
    })
    tool.set_channel_manager(cm)
    tool.set_context("wechat", "room@chatroom")
    out = asyncio.run(tool.execute(query="张三"))
    # Both 张三 and 张三丰 should match (substring)
    assert "<@1>" in out
    assert "<@2>" in out
    assert "李四" not in out


def test_find_person_truncates_when_many(tool) -> None:
    members = [{"id": f"id_{i}", "name": f"User_{i}"} for i in range(15)]
    cm, _ = _make_channel_manager({"room@chatroom": members})
    tool.set_channel_manager(cm)
    tool.set_context("wechat", "room@chatroom")
    out = asyncio.run(tool.execute(query="user"))
    assert "Found 15" in out
    assert "truncated" in out.lower()
    # Only 10 markers shown
    assert out.count("<@id_") == 10
