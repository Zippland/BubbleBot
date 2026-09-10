"""Stable tool definitions and explicit channel-delivery/end-turn protocol."""

import asyncio
import json
from copy import deepcopy
from itertools import product
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from bubbles.agent.context import ContextBuilder
from bubbles.agent.loop import AgentLoop
from bubbles.agent.tools.cron import CronTool
from bubbles.agent.tools.image_generation import GenerateImageTool
from bubbles.agent.tools.message import MESSAGE_TARGET_KEY, MessageTool, SwitchMessageTargetTool
from bubbles.agent.turn import TurnState, _completed_tool_groups, process_system_message
from bubbles.bus.events import InboundMessage
from bubbles.bus.queue import MessageBus
from bubbles.cron.service import CronService
from bubbles.cron.types import CronSchedule
from bubbles.providers.base import LLMResponse, ToolCallRequest
from bubbles.providers.custom_provider import CustomProvider
from bubbles.providers.litellm_provider import LiteLLMProvider
from bubbles.session.manager import CONTEXT_EXCLUDED_KEY, SessionManager, _sanitize_for_api


def call(name, **arguments):
    return ToolCallRequest(id=f"call-{name}", name=name, arguments=arguments)


def response(*calls, text=None):
    return LLMResponse(content=text, tool_calls=list(calls))


class Script:
    def __init__(self, *responses):
        self.responses = iter(responses)
        self.requests = []

    def get_default_model(self):
        return "test-model"

    async def chat(self, **kwargs):
        self.requests.append(deepcopy(kwargs))
        return next(self.responses)


def make_loop(tmp_path, *responses):
    loop = AgentLoop(
        bus=MessageBus(), provider=Script(*responses), max_tokens=100,
        memory_window=500, context_limit=1_000_000,
        session_manager=SessionManager(sessions_dir=tmp_path / "sessions"),
    )
    loop.data_dir = tmp_path
    loop._session_bindings = {"wechat:room": "shared"}
    loop._sandboxes = SimpleNamespace(get=AsyncMock(return_value=SimpleNamespace(root=tmp_path)))
    return loop


def inbound(content="hello"):
    return InboundMessage(channel="wechat", sender_id="user", chat_id="room", content=content)


def drain(bus):
    messages = []
    while not bus.outbound.empty():
        messages.append(bus.outbound.get_nowait())
    return messages


def test_builtin_schemas_identical_across_all_availability_states(tmp_path):
    loop = make_loop(tmp_path)
    reference = loop.tools.get_definitions()
    assert len(reference) == 18
    names = [t["function"]["name"] for t in reference]
    assert names == sorted(names)
    assert {"message", "switch_message_target", "stay_silent", "cron", "find_person", "generate_image"} <= set(names)
    assert "end_turn" not in names  # one end tool, not two aliases
    for system, cron, image, manager in product((False, True), repeat=4):
        loop.cron_service = Mock() if cron else None
        loop.image_generation_backend = Mock() if image else None
        loop.channel_manager = Mock() if manager else None
        tools = loop.build_turn_tools(
            channel="wechat" if system else "cli", chat_id=str(manager), message_id=str(image),
            session_dir=tmp_path, session_key=str(cron), session=None, sandbox=None,
            system_triggered=system,
        )
        assert tools.get_definitions() == reference


@pytest.mark.asyncio
@pytest.mark.parametrize("action", ["add", "list", "remove"])
async def test_system_cron_returns_reason_without_touching_service(action):
    service = Mock()
    tool = CronTool(service, system_triggered=True)
    result = await tool.execute(action=action, message="nested", every_seconds=60, job_id="x")
    assert result.startswith("Error:")
    assert "系统触发" in result and "递归" in result
    assert service.mock_calls == []


@pytest.mark.asyncio
async def test_unavailable_tools_return_errors(tmp_path):
    assert "调度服务未配置" in await CronTool(None).execute(action="list")
    assert "生图后端未配置" in await GenerateImageTool(None).execute(prompt="draw")
    loop = make_loop(tmp_path)
    assert "channel manager not configured" in await loop.tools.execute("find_person", {"query": "Alice"})


@pytest.mark.asyncio
async def test_normal_cron_still_works_and_is_session_scoped(tmp_path):
    service = CronService(tmp_path / "jobs.json")
    tool = CronTool(service)
    tool.set_context("wechat", "room", "shared")
    other = service.add_job("other", CronSchedule(kind="every", every_ms=60_000), "private", session_key="other")
    assert "Created job" in await tool.execute(action="add", message="remind", every_seconds=60)
    listing = await tool.execute(action="list")
    assert "remind" in listing and other.id not in listing
    assert "not found" in await tool.execute(action="remove", job_id=other.id)
    own = next(j for j in service.list_jobs() if j.payload.session_key == "shared")
    assert "Removed" in await tool.execute(action="remove", job_id=own.id)
    assert [j.id for j in service.list_jobs()] == [other.id]


@pytest.mark.asyncio
async def test_message_does_not_end_and_optional_text_uses_same_sender(tmp_path):
    loop = make_loop(tmp_path,
        response(call("message", content="开始查询"), text="收到"),
        response(call("find_person", query="Alice"), text="找一下成员"),
        response(call("message", content="查询完成")),
        response(call("stay_silent")),
    )
    await loop._dispatch(inbound())
    assert [m.content for m in drain(loop.bus)] == ["收到", "开始查询", "找一下成员", "查询完成"]
    assert len(loop.provider.requests) == 4
    assert all(r["tools"] == loop.provider.requests[0]["tools"] for r in loop.provider.requests)
    history = loop.sessions.get_or_create("shared").get_history()
    assert [m["name"] for m in history if m["role"] == "tool"] == [
        "message", "find_person", "message", "stay_silent",
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("end_first", [False, True])
async def test_end_is_a_batch_barrier_not_a_cancellation(tmp_path, end_first):
    calls = [call("message", content="answer"), call("stay_silent")]
    if end_first:
        calls.reverse()
    loop = make_loop(tmp_path, response(*calls, text="preamble"))
    await loop._dispatch(inbound())
    assert [m.content for m in drain(loop.bus)] == ["preamble", "answer"]
    assert len(loop.provider.requests) == 1
    history = loop.sessions.get_or_create("shared").get_history()
    assert [m["tool_call_id"] for m in history if m["role"] == "tool"] == [c.id for c in calls]


@pytest.mark.asyncio
async def test_end_without_message_is_silence(tmp_path):
    loop = make_loop(tmp_path, response(call("stay_silent")))
    await loop._dispatch(inbound())
    assert drain(loop.bus) == []


@pytest.mark.asyncio
async def test_plain_text_has_one_bounded_protocol_repair(tmp_path):
    loop = make_loop(tmp_path, response(text="sent once"), response(text="sent once"))
    await loop._dispatch(inbound())
    assert len(loop.provider.requests) == 2
    assert "[Harness protocol]" in loop.provider.requests[1]["messages"][-1]["content"]
    assert [m.content for m in drain(loop.bus)] == ["sent once"]


@pytest.mark.asyncio
async def test_plain_text_can_continue_to_another_reply(tmp_path):
    loop = make_loop(tmp_path, response(text="draft"),
        response(call("message", content="real reply"), call("stay_silent")))
    await loop._dispatch(inbound())
    assert [m.content for m in drain(loop.bus)] == ["draft", "real reply"]


@pytest.mark.asyncio
async def test_duplicate_send_error_does_not_implicitly_end_turn(tmp_path):
    loop = make_loop(tmp_path,
        response(call("message", content="first")),
        response(call("message", content="first")),
        response(call("message", content="correction")),
        response(call("stay_silent")),
    )
    await loop._dispatch(inbound())
    assert [m.content for m in drain(loop.bus)] == ["first", "correction"]
    assert len(loop.provider.requests) == 4
    assert "already submitted" in loop.provider.requests[2]["messages"][-1]["content"]


@pytest.mark.asyncio
@pytest.mark.parametrize("channel", ["cli", "wechat"])
async def test_process_direct_collects_both_paths_without_duplicates(tmp_path, channel):
    loop = make_loop(tmp_path,
        response(call("message", content="a"), text="preamble"),
        response(call("message", content="b"), call("stay_silent")),
    )
    text, used = await loop.process_direct("hello", channel=channel, chat_id="room", system_triggered=True)
    assert text == "preamble\n\na\n\nb"
    assert used == ["message", "message", "stay_silent"]
    assert [m.content for m in drain(loop.bus)] == ([] if channel == "cli" else ["preamble", "a", "b"])


@pytest.mark.asyncio
async def test_system_message_uses_both_output_paths_without_rebroadcast(tmp_path):
    loop = make_loop(tmp_path, response(call("message", content="task done"), call("stay_silent"), text="preamble"))
    result = await process_system_message(loop, InboundMessage(
        channel="system", sender_id="subagent", chat_id="shared", content="result"))
    assert result is None
    assert [m.content for m in drain(loop.bus)] == ["preamble", "task done"]


@pytest.mark.asyncio
async def test_heartbeat_retired_but_records_and_regular_cron_preserved(tmp_path):
    path = tmp_path / "jobs.json"
    old = CronService(path)
    heartbeat = old.add_job("heartbeat:shared", CronSchedule(kind="every", every_ms=60_000),
        "old heartbeat", session_key="shared")
    regular = old.add_job("regular", CronSchedule(kind="every", every_ms=60_000), "normal")
    on_job = AsyncMock()
    service = CronService(path, on_job=on_job)
    await service.start()
    try:
        jobs = {j.id: j for j in service.list_jobs(include_disabled=True)}
        assert not jobs[heartbeat.id].enabled
        assert jobs[heartbeat.id].state.next_run_at_ms is None
        assert jobs[heartbeat.id].payload.message == "old heartbeat"
        assert jobs[regular.id].enabled
        assert not await service.run_job(heartbeat.id, force=True)
        assert service.enable_job(heartbeat.id) is None
        assert await service.run_job(regular.id, force=True)
        on_job.assert_awaited_once()
    finally:
        service.stop()
    reloaded = CronService(path).list_jobs(include_disabled=True)
    assert len(reloaded) == 2
    assert not next(j for j in reloaded if j.id == heartbeat.id).enabled


@pytest.mark.asyncio
async def test_heartbeat_command_and_prompt_removed_without_deleting_user_file(tmp_path):
    file = tmp_path / "HEARTBEATS.md"
    file.write_text("legacy user instructions", encoding="utf-8")
    prompt = ContextBuilder(session_dir=tmp_path).build_system_prompt()
    assert "legacy user instructions" not in prompt
    assert "HEARTBEATS.md" not in prompt
    assert "/heartbeat" not in prompt
    assert file.read_text(encoding="utf-8") == "legacy user instructions"
    loop = make_loop(tmp_path)
    result = await loop._process_message(inbound("/heartbeat 30m"))
    assert "心跳功能已移除" in result.content
    assert loop.provider.requests == []


@pytest.mark.asyncio
async def test_failed_message_can_be_retried():
    sender = AsyncMock(side_effect=[RuntimeError("unavailable"), None])
    tool = MessageTool(send_callback=sender, default_channel="wechat", default_chat_id="room")
    assert "Error sending" in await tool.execute(content="retry")
    assert '"status": "submitted"' in await tool.execute(content="retry")


@pytest.mark.asyncio
async def test_distinct_text_tails_and_image_only_messages_are_not_duplicates(tmp_path):
    sender = AsyncMock()
    tool = MessageTool(send_callback=sender, default_channel="wechat", default_chat_id="room")
    tool.set_session_dir(tmp_path)
    prefix = "x" * 100
    assert '"status": "submitted"' in await tool.execute(content=prefix + "a")
    assert '"status": "submitted"' in await tool.execute(content=prefix + "b")
    for name in ("a.png", "b.png"):
        (tmp_path / name).write_bytes(b"image")
        assert '"status": "submitted"' in await tool.execute(content="", media=[name])
    assert "already submitted" in await tool.execute(content="", media=["a.png"])
    assert sender.await_count == 4


@pytest.mark.asyncio
async def test_receipts_do_not_repeat_content_or_attachments(tmp_path):
    sender = AsyncMock()
    tool = MessageTool(send_callback=sender, default_channel="wechat", default_chat_id="room")
    tool.set_session_dir(tmp_path)
    attachment = tmp_path / "attachment.png"
    attachment.write_bytes(b"image")
    content = "正文保留在原始消息中" * 1000
    result = await tool.execute(content=content, media=[attachment.name])
    assert json.loads(result) == {"status": "submitted", "channel": "wechat", "chat_id": "room"}
    outbound = sender.call_args.args[0]
    assert outbound.content == content and outbound.media == [str(attachment.resolve())]
    duplicate = await tool.execute(content=content, media=[attachment.name])
    receipt = json.loads(duplicate.removeprefix("Error: "))
    assert set(receipt) == {"status", "channel", "chat_id", "error"}
    assert receipt["status"] == "duplicate"
    assert content not in duplicate and attachment.name not in duplicate


def text_receipts(messages):
    return [json.loads(m["content"].split("] ", 1)[1]) for m in messages
        if m["role"] == "system" and m["content"].startswith("[Assistant text receipt")]


@pytest.mark.asyncio
async def test_switch_routes_both_output_paths_in_order_with_receipts(tmp_path):
    loop = make_loop(tmp_path,
        response(call("switch_message_target", channel="wechat", chat_id="B"), text="在旧窗口说"),
        response(call("message", content="新窗口消息"), text="在新窗口说"),
        response(call("switch_message_target", channel="feishu", chat_id="C"),
                 call("message", content="第三个窗口")),
        response(call("stay_silent"), text="第三个窗口收尾"),
    )
    await loop._dispatch(inbound())
    actual = [(m.channel, m.chat_id, m.content) for m in drain(loop.bus)]
    assert actual == [
        ("wechat", "room", "在旧窗口说"),
        ("wechat", "B", "在新窗口说"), ("wechat", "B", "新窗口消息"),
        ("feishu", "C", "第三个窗口"), ("feishu", "C", "第三个窗口收尾"),
    ]
    history = loop.sessions.get_or_create("shared").get_history()
    assert _sanitize_for_api(history) == history
    receipts = text_receipts(history) + [json.loads(m["content"]) for m in history
        if m["role"] == "tool" and m["name"] == "message"]
    assert sorted((r["channel"], r["chat_id"]) for r in receipts) == sorted((channel, chat_id) for channel, chat_id, _ in actual)
    assert all(r["status"] == "submitted" for r in receipts)
    assert all(set(r) == {"status", "channel", "chat_id"} for r in receipts)
    assert [m["content"] for m in history if m["role"] == "assistant" and m["content"]] == [
        "在旧窗口说", "在新窗口说", "第三个窗口收尾",
    ]
    assert [json.loads(c["function"]["arguments"])["content"]
        for m in history for c in m.get("tool_calls", []) if c["function"]["name"] == "message"
    ] == ["新窗口消息", "第三个窗口"]
    # Changing destinations does not invalidate the stable prefix or schemas.
    first = loop.provider.requests[0]
    assert all(r["tools"] == first["tools"] for r in loop.provider.requests)
    assert all(r["messages"][0] == first["messages"][0] for r in loop.provider.requests)


@pytest.mark.asyncio
async def test_selection_survives_next_input_restart_and_system_trigger(tmp_path):
    loop = make_loop(tmp_path,
        response(call("switch_message_target", channel="wechat", chat_id="B")),
        response(call("stay_silent")),
        response(call("message", content="下一次任务"), call("stay_silent")),
    )
    await loop._dispatch(inbound())
    await loop._dispatch(inbound("A 群又来消息"))
    assert [(m.chat_id, m.content) for m in drain(loop.bus)] == [("B", "下一次任务")]
    # A fresh manager/loop reads the disk, not the old instance's routing state.
    restarted = make_loop(tmp_path, response(call("stay_silent"), text="重启后定时任务"))
    await process_system_message(restarted, InboundMessage(
        channel="system", sender_id="cron", chat_id="shared", content="check"))
    assert [(m.chat_id, m.content) for m in drain(restarted.bus)] == [("B", "重启后定时任务")]
    current = restarted.provider.requests[0]["messages"][-1]
    assert current["content"].startswith("[Current message target]")
    assert '"chat_id": "B"' in current["content"]


@pytest.mark.asyncio
async def test_switch_persists_before_turn_finishes_and_drops_reply_id(tmp_path):
    loop = make_loop(tmp_path)
    session = loop.sessions.get_or_create("shared")
    tools = loop.build_turn_tools("wechat", "room", "original-reply-id", tmp_path, "shared", session, None)
    switch = tools.get("switch_message_target")
    assert json.loads(await switch.execute(channel="wechat", chat_id="B"))["status"] == "selected"
    loaded = SessionManager(loop.sessions.sessions_dir).get_or_create("shared")
    assert loaded.metadata[MESSAGE_TARGET_KEY] == {"channel": "wechat", "chat_id": "B"}
    await tools.execute("message", {"content": "b"})
    await switch.execute(channel="wechat", chat_id="room")
    await tools.execute("message", {"content": "a"})
    assert all(m.metadata["message_id"] is None for m in drain(loop.bus))
    assert session.key == "shared" and session.directory == loaded.directory


@pytest.mark.asyncio
async def test_first_target_is_retained_without_an_explicit_switch(tmp_path):
    loop = make_loop(tmp_path, response(call("stay_silent")), response(call("stay_silent"), text="still A"))
    loop._session_bindings["wechat:another-input"] = "shared"
    await loop._dispatch(inbound())
    await loop._dispatch(InboundMessage(channel="wechat", sender_id="u", chat_id="another-input", content="hi"))
    assert [(m.chat_id, m.content) for m in drain(loop.bus)] == [("room", "still A")]


@pytest.mark.asyncio
async def test_concurrent_workspaces_have_independent_selections(tmp_path):
    loop = make_loop(tmp_path)
    async def work(key, destination):
        session = loop.sessions.get_or_create(key)
        tools = loop.build_turn_tools("wechat", "origin", "id", tmp_path, key, session, None)
        await tools.execute("switch_message_target", {"channel": "wechat", "chat_id": destination})
        await asyncio.sleep(0)
        await tools.execute("message", {"content": key})
    await asyncio.gather(work("first", "A"), work("second", "B"))
    assert sorted((m.chat_id, m.content) for m in drain(loop.bus)) == [("A", "first"), ("B", "second")]


@pytest.mark.asyncio
async def test_legacy_message_destination_args_are_rejected_without_sending(tmp_path):
    loop = make_loop(tmp_path)
    tools = loop.build_turn_tools("wechat", "room", None, tmp_path, "shared", None, None)
    assert set(tools.get("message").parameters["properties"]) == {"content", "media"}
    for extra in ({"channel": "feishu"}, {"chat_id": "B"}, {"message_id": "wrong"}):
        result = await tools.execute("message", {"content": "hello", **extra})
        assert result.startswith("Error:") and "switch_message_target" in result
        assert '"chat_id": "room"' in result
    assert drain(loop.bus) == []
    assert tools.get("message").target == ("wechat", "room")


@pytest.mark.asyncio
async def test_failed_switch_keeps_previous_target_and_saved_selection(tmp_path):
    loop = make_loop(tmp_path)
    session = loop.sessions.get_or_create("shared")
    message = MessageTool(default_channel="wechat", default_chat_id="room", default_message_id="id")
    message.bind_session(session, Mock(side_effect=OSError("disk unavailable")))
    switch = SwitchMessageTargetTool(message)
    for channel, chat_id in (("", "B"), ("wechat", " "), ("wechat", "B")):
        result = await switch.execute(channel=channel, chat_id=chat_id)
        assert result.startswith("Error:") and "Target unchanged" in result
        assert message.target == ("wechat", "room")
        assert message._default_message_id == "id"
        assert session.metadata[MESSAGE_TARGET_KEY]["chat_id"] == "room"
    unavailable = SwitchMessageTargetTool(message, SimpleNamespace(get_channel=lambda name: None))
    assert "not running" in await unavailable.execute(channel="feishu", chat_id="B")
    assert message.target == ("wechat", "room")


@pytest.mark.asyncio
async def test_member_lookup_follows_selected_output_window(tmp_path):
    loop = make_loop(tmp_path)
    channel = SimpleNamespace(get_group_members=AsyncMock(return_value=[{"id": "wxid_b", "names": {"nickname": "Bob"}}]))
    loop.channel_manager = SimpleNamespace(get_channel=lambda name: channel if name == "wechat" else None)
    tools = loop.build_turn_tools("wechat", "room", None, tmp_path, "shared", None, None)
    await tools.execute("switch_message_target", {"channel": "wechat", "chat_id": "B"})
    result = await tools.execute("find_person", {"query": "Bob"})
    assert "<@wxid_b>" in result
    channel.get_group_members.assert_awaited_once_with("B")


@pytest.mark.asyncio
async def test_preamble_failure_returns_destination_and_preserves_tool_protocol(tmp_path):
    loop = make_loop(tmp_path, response(call("stay_silent"), text="failed text"))
    sender = AsyncMock(side_effect=RuntimeError("offline"))
    await loop._process_message(inbound(), on_message=sender)
    history = loop.sessions.get_or_create("shared").get_history()
    receipt = next(m for m in history if m["content"] and "[Assistant text receipt" in m["content"])
    assert '"status": "failed"' in receipt["content"]
    assert '"channel": "wechat"' in receipt["content"] and '"chat_id": "room"' in receipt["content"]
    assert "failed text" not in receipt["content"] and "offline" in receipt["content"]
    assert next(m for m in history if m["role"] == "assistant")["content"] == "failed text"
    assert _sanitize_for_api(history) == history
    assert drain(loop.bus) == []


@pytest.mark.asyncio
async def test_sent_text_receipt_survives_later_tool_cancellation(tmp_path):
    loop = make_loop(tmp_path, response(call("find_person", query="Bob"), text="已开始"))
    async def cancel(name, args, result):
        raise asyncio.CancelledError()
    with pytest.raises(asyncio.CancelledError):
        await loop._process_message(inbound(), on_tool_call=cancel)
    loop.sessions.invalidate("shared")
    session = loop.sessions.get_or_create("shared")
    history = session.get_history()
    assert _sanitize_for_api(history) == history
    assert text_receipts(history) == [{"status": "submitted", "channel": "wechat", "chat_id": "room"}]
    # The original text remains exactly once in the audit, not in the receipt.
    original = next(m for m in session.messages if m["role"] == "assistant")
    assert original["content"] == "已开始" and original[CONTEXT_EXCLUDED_KEY] is True
    assert json.dumps(session.messages, ensure_ascii=False).count("已开始") == 1
    assert [m.content for m in drain(loop.bus)] == ["已开始"]


@pytest.mark.asyncio
async def test_reasoning_is_not_sent_but_optional_assistant_text_is(tmp_path):
    answer = response(call("stay_silent"), text="<think>secret</think>可以说")
    answer.reasoning_content = "more secret reasoning"
    loop = make_loop(tmp_path, answer)
    await loop._dispatch(inbound())
    assert [m.content for m in drain(loop.bus)] == ["可以说"]
    history = loop.sessions.get_or_create("shared").get_history()
    assert text_receipts(history) == [{"status": "submitted", "channel": "wechat", "chat_id": "room"}]
    assert next(m for m in history if m["role"] == "assistant")["content"] == answer.content
    assert AgentLoop._strip_think("<think>unfinished secret") is None


@pytest.mark.asyncio
@pytest.mark.parametrize("provider_class,method", [
    (LiteLLMProvider, "_parse_response"), (CustomProvider, "_parse"),
])
@pytest.mark.parametrize("reasoning", [None, "", "raw reasoning"])
async def test_plain_text_reasoning_survives_next_request_and_reload(tmp_path, provider_class, method, reasoning):
    # Protocol-looking text is still model content; preserving reasoning must not rewrite it.
    content = "原始回复</｜｜DSML｜｜parameter></｜｜DSML｜｜invoke></｜｜DSML｜｜tool_calls>"
    raw_response = SimpleNamespace(
        choices=[SimpleNamespace(
            message=SimpleNamespace(content=content, tool_calls=[], reasoning_content=reasoning),
            finish_reason="stop",
        )],
        usage=None,
    )
    provider = provider_class.__new__(provider_class)
    answer = getattr(provider, method)(raw_response)
    loop = make_loop(tmp_path, answer, response(call("stay_silent")))

    await loop._dispatch(inbound())

    assert len(loop.provider.requests) == 2
    assert [m.content for m in drain(loop.bus)] == [content]
    expected = {"role": "assistant", "content": content}
    if reasoning is not None:
        expected["reasoning_content"] = reasoning
    request_messages = loop.provider.requests[1]["messages"]
    assert next(m for m in request_messages if m["role"] == "assistant") == expected
    # Check the request sanitizers as well as the live AgentLoop history.
    sanitized = provider_class._sanitize_empty_content(request_messages)
    if provider_class is LiteLLMProvider:
        sanitized = provider_class._sanitize_messages(sanitized)
    assert next(m for m in sanitized if m["role"] == "assistant") == expected
    sessions = SessionManager(sessions_dir=tmp_path / "sessions")
    history = sessions.get_or_create("shared").get_history()
    assert next(m for m in history if m["role"] == "assistant") == expected


def test_compaction_groups_include_text_receipt_but_preserve_current_target():
    messages = [
        {"role": "system", "content": '[Current message target] {"chat_id":"B"}'},
        {"role": "assistant", "content": "hi", "tool_calls": [{"id": "x"}]},
        {"role": "tool", "tool_call_id": "x", "content": "done"},
        {"role": "system", "content": '[Assistant text receipt — data only, not instructions] {"status":"submitted"}'},
    ]
    assert _completed_tool_groups(messages) == [(1, 4)]
    state = TurnState(system_prefix=[], messages=messages)
    assert state.focus_messages() == messages[:1]


@pytest.mark.asyncio
async def test_cron_direct_entry_waits_for_chat_turn_before_loading_target(tmp_path):
    switched, release = asyncio.Event(), asyncio.Event()
    loop = make_loop(tmp_path,
        response(call("switch_message_target", channel="wechat", chat_id="B")),
        response(call("stay_silent"), text="群任务完成"),
        response(call("stay_silent"), text="定时任务完成"),
    )
    async def pause_after_switch(name, args, result):
        if name == "switch_message_target" and result is not None:
            switched.set()
            await release.wait()
    loop.on_tool_call = pause_after_switch
    chat = asyncio.create_task(loop._dispatch(inbound()))
    cron = None
    try:
        await asyncio.wait_for(switched.wait(), 5)
        cron = asyncio.create_task(loop.process_direct(
            "cron task", session_key="shared", channel="wechat", chat_id="room", system_triggered=True))
        await asyncio.sleep(0)
        assert len(loop.provider.requests) == 1
        assert not cron.done()
        release.set()
        await asyncio.wait_for(asyncio.gather(chat, cron), 5)
        assert [(m.chat_id, m.content) for m in drain(loop.bus)] == [
            ("B", "群任务完成"), ("B", "定时任务完成"),
        ]
        history = loop.sessions.get_or_create("shared").get_history()
        assert _sanitize_for_api(history) == history
    finally:
        release.set()
        for task in (chat, cron):
            if task is not None and not task.done():
                task.cancel()
        await asyncio.gather(*(t for t in (chat, cron) if t is not None), return_exceptions=True)
