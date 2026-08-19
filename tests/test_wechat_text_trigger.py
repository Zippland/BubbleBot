"""Wechat group-response tests for native mentions and the visible wake word."""

from __future__ import annotations

import pytest

from bubbles.channels.wechat import MSG_TYPE_TEXT, WeChatChannel
from bubbles.config.schema import WeChatConfig


class _FakeBus:
    def __init__(self) -> None:
        self.inbound = []

    async def publish_inbound(self, message) -> None:
        self.inbound.append(message)


class _FakeTextMessage:
    def __init__(
        self,
        content: str,
        *,
        native_at: bool = False,
        xml: str = "",
    ) -> None:
        self.type = MSG_TYPE_TEXT
        self.content = content
        self.xml = xml
        self.sender = "wxid_alice"
        self.roomid = "room@chatroom"
        self._native_at = native_at

    def from_self(self) -> bool:
        return False

    def from_group(self) -> bool:
        return True

    def is_at(self, wxid: str) -> bool:
        assert wxid == "wxid_bot"
        return self._native_at


def _make_channel() -> tuple[WeChatChannel, _FakeBus]:
    bus = _FakeBus()
    channel = WeChatChannel(WeChatConfig(enabled=True), bus)
    channel.wxid = "wxid_bot"
    return channel, bus


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("content", "expected_content"),
    [
        ("@泡泡\u2005帮我看下", "帮我看下"),
        ("@泡泡", "@泡泡"),
        ("大家让泡泡看看", "大家让泡泡看看"),
    ],
)
async def test_visible_bot_name_triggers_group_response(
    content: str,
    expected_content: str,
) -> None:
    channel, bus = _make_channel()

    await channel._process_msg(_FakeTextMessage(content))

    assert len(bus.inbound) == 1
    assert bus.inbound[0].metadata["respond"] is True
    assert bus.inbound[0].content == expected_content


@pytest.mark.asyncio
async def test_group_text_without_bot_name_stays_silent() -> None:
    channel, bus = _make_channel()

    await channel._process_msg(_FakeTextMessage("大家让小王看看"))

    assert len(bus.inbound) == 1
    assert bus.inbound[0].metadata["respond"] is False


@pytest.mark.asyncio
async def test_native_at_remains_a_compatible_trigger() -> None:
    channel, bus = _make_channel()
    xml = "<atuserlist><![CDATA[wxid_bot]]></atuserlist>"

    await channel._process_msg(
        _FakeTextMessage("@Bubbles\u2005帮我看下", native_at=True, xml=xml)
    )

    assert len(bus.inbound) == 1
    assert bus.inbound[0].metadata["respond"] is True
    assert bus.inbound[0].content == "<@wxid_bot>\u2005帮我看下"


@pytest.mark.asyncio
async def test_name_trigger_does_not_strip_another_persons_mention() -> None:
    channel, bus = _make_channel()

    await channel._process_msg(_FakeTextMessage("@张三 帮我问问泡泡"))

    assert bus.inbound[0].metadata["respond"] is True
    assert bus.inbound[0].content == "@张三 帮我问问泡泡"
