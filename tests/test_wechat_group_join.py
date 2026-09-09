"""Native WeChat membership events trigger user turns with real @ identities."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from wcferry import WxMsg, wcf_pb2

from bubbles.agent.loop import AgentLoop
from bubbles.bus.queue import MessageBus
from bubbles.channels.wechat import MSG_TYPE_SYSTEM, MSG_TYPE_TEXT, WeChatChannel
from bubbles.channels.wechat_group_events import parse_group_join_members
from bubbles.config.schema import WeChatConfig
from bubbles.providers.base import LLMResponse, ToolCallRequest
from bubbles.session.manager import SessionManager


def _message(content='"邀请人"邀请"张三"加入了群聊', **kwargs):
    fields = dict(
        type=MSG_TYPE_SYSTEM, id=123, is_group=True,
        roomid="room@chatroom", sender="wxid_inviter", content=content,
    )
    fields.update(kwargs)
    return WxMsg(wcf_pb2.WxMsg(**fields))


def _channel(**config):
    bus = MessageBus()
    channel = WeChatChannel(WeChatConfig(enabled=True, **config), bus)
    channel.wxid = "wxid_bot"
    channel.wcf = Mock()
    channel.wcf.get_chatroom_members.return_value = {
        "wxid_new": "张三", "wxid_inviter": "邀请人",
    }
    channel.wcf.query_sql.return_value = []
    return channel, bus


@pytest.mark.parametrize(("notice", "members"), [
    ('"邀请人"邀请"张三"加入了群聊', ["张三"]),
    ('你邀请"张三"、"李四"加入了群聊', ["张三", "李四"]),
    ('“邀请人”邀请“张三”和“李四”加入群聊。', ["张三", "李四"]),
    ('"张三"通过扫描"邀请人"分享的二维码加入群聊', ["张三"]),
    ('"张三"通过扫描二维码加入了群聊', ["张三"]),
    ('"张三"通过"邀请人"的邀请加入了群聊', ["张三"]),
    ('张三加入了群聊', ["张三"]),
    ('"邀请人"邀请"Tom Lee"、"和、平"加入了群聊', ["Tom Lee", "和、平"]),
    ('"邀请人"邀请你和"李四"加入了群聊', ["李四"]),
    ('你邀请"你"加入了群聊', ["你"]),
    ('"邀请人"邀请你加入了群聊', []),
    ('你通过扫描"邀请人"分享的二维码加入群聊', []),
    ('"张三"退出了群聊', []),
    ('"邀请人"修改群名为“张三加入了群聊”', []),
    ('"邀请人"邀请"张三"加入群聊，请等待确认', []),
    ('"张三"撤回了一条消息', []),
    (None, []),
])
def test_parse_native_join_notice(notice, members):
    assert parse_group_join_members(notice) == members


@pytest.mark.asyncio
async def test_join_emits_user_event_with_nickname_wxid_and_mention():
    channel, bus = _channel()

    await channel._process_msg(_message())

    event = bus.inbound.get_nowait()
    assert event.channel == "wechat"
    assert event.chat_id == "room@chatroom"
    assert event.content.startswith("【系统通知】张三 <@wxid_new> 入群了")
    assert event.content.count("wxid_new") == 1
    assert event.metadata["respond"] is True
    assert event.metadata["is_group"] is True
    assert event.metadata["event_type"] == "group_member_joined"
    assert event.metadata["joined_members"] == [{"name": "张三", "wxid": "wxid_new"}]
    assert "wxid_inviter" not in event.content
    assert bus.outbound.empty()


@pytest.mark.asyncio
async def test_contact_nickname_resolves_when_room_alias_differs():
    channel, bus = _channel()
    channel.wcf.get_chatroom_members.return_value = {"wxid_new": "群名片"}
    channel.wcf.query_sql.return_value = [{"UserName": "wxid_new", "NickName": "张三"}]

    await channel._process_msg(_message())

    assert "<@wxid_new>" in bus.inbound.get_nowait().content
    channel.wcf.get_alias_in_chatroom.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("roster", [{}, {"wxid_1": "张三", "wxid_2": "张三"}])
async def test_missing_or_ambiguous_identity_does_not_guess_wxid(roster):
    channel, bus = _channel()
    channel.wcf.get_chatroom_members.return_value = roster

    await channel._process_msg(_message())

    event = bus.inbound.get_nowait()
    assert "张三（wxid 暂未确认）" in event.content
    assert "<@" not in event.content
    assert event.metadata["joined_members"] == [{"name": "张三", "wxid": None}]


@pytest.mark.asyncio
async def test_roster_error_still_delivers_notice():
    channel, bus = _channel()
    channel.wcf.get_chatroom_members.side_effect = RuntimeError("roster unavailable")
    await channel._process_msg(_message())
    assert bus.inbound.get_nowait().metadata["respond"] is True


@pytest.mark.asyncio
async def test_duplicate_delivery_is_ignored_but_distinct_join_can_trigger():
    channel, bus = _channel()
    await channel._process_msg(_message())
    await channel._process_msg(_message())
    await channel._process_msg(_message(id=124))
    assert bus.inbound.qsize() == 2
    assert channel.wcf.get_chatroom_members.call_count == 2


@pytest.mark.asyncio
async def test_bot_inviting_a_newcomer_still_triggers():
    channel, bus = _channel()
    await channel._process_msg(_message('你邀请"张三"加入了群聊', is_self=True))
    assert bus.inbound.get_nowait().metadata["respond"] is True


@pytest.mark.asyncio
@pytest.mark.parametrize("overrides", [
    {"is_group": False, "roomid": ""},
    {"content": '"张三"退出了群聊'},
    {"content": '"邀请人"邀请你加入了群聊'},
])
async def test_non_join_system_notices_are_ignored(overrides):
    channel, bus = _channel()
    channel._get_sender_name = Mock(return_value=None)
    await channel._process_msg(_message(**overrides))
    assert bus.inbound.empty()


@pytest.mark.asyncio
async def test_regular_text_is_not_a_membership_event():
    channel, bus = _channel()
    channel._get_sender_name = Mock(return_value=None)
    await channel._process_msg(_message(type=MSG_TYPE_TEXT))
    event = bus.inbound.get_nowait()
    assert event.metadata["respond"] is False
    assert "event_type" not in event.metadata


@pytest.mark.asyncio
@pytest.mark.parametrize("config", [
    {"groups": ["other@chatroom"]}, {"allow_from": ["wxid_other"]},
])
async def test_existing_allowlists_still_apply(config):
    channel, bus = _channel(**config)
    await channel._process_msg(_message())
    assert bus.inbound.empty()
    channel.wcf.get_chatroom_members.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("silent", [False, True])
async def test_notice_reaches_bound_agent_as_user_prompt_and_model_can_choose(tmp_path, silent):
    channel, bus = _channel()
    result = LLMResponse(
        content=None,
        tool_calls=[ToolCallRequest(id="silent", name="stay_silent", arguments={})],
    ) if silent else LLMResponse(content=None, tool_calls=[
        ToolCallRequest(id="welcome", name="message", arguments={"content": "欢迎 <@wxid_new>！"}),
        ToolCallRequest(id="end", name="stay_silent", arguments={}),
    ])
    provider = SimpleNamespace(
        get_default_model=lambda: "test-model", chat=AsyncMock(return_value=result),
    )
    loop = AgentLoop(
        bus=bus, provider=provider, max_tokens=100, memory_window=100,
        context_limit=1_000_000,
        session_manager=SessionManager(sessions_dir=tmp_path),
    )
    loop._session_bindings = {"wechat:room@chatroom": "welcome-session"}
    loop._sandboxes = SimpleNamespace(get=AsyncMock(return_value=SimpleNamespace(root=tmp_path)))
    await channel._process_msg(_message())
    await loop._dispatch(bus.inbound.get_nowait())

    provider.chat.assert_awaited_once()
    prompt = provider.chat.call_args.kwargs["messages"]
    user_message = next(m for m in prompt if m["role"] == "user")
    assert user_message["content"].startswith("【系统通知】")
    assert "<@wxid_new>" in user_message["content"]
    history = loop.sessions.get_or_create("welcome-session").messages
    assert any(m["role"] == "user" and "【系统通知】" in m["content"] for m in history)
    if silent:
        assert bus.outbound.empty()
    else:
        reply = bus.outbound.get_nowait()
        assert (reply.channel, reply.chat_id) == ("wechat", "room@chatroom")
        channel._get_sender_name = Mock(return_value="张三")
        text, aters = channel._translate_outbound_mentions(reply.content, reply.chat_id)
        assert "@张三" in text
        assert aters == "wxid_new"
        assert bus.outbound.empty()
