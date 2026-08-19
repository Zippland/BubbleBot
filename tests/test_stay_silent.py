"""Regression tests for the globally available stay_silent turn outcome."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from bubbles.agent.loop import AgentLoop
from bubbles.agent.tools.message import MessageTool
from bubbles.agent.tools.registry import ToolRegistry
from bubbles.agent.tools.stay_silent import StaySilentTool
from bubbles.agent.turn import TurnState, persist_turn_state, process_system_message
from bubbles.bus.events import InboundMessage
from bubbles.bus.queue import MessageBus
from bubbles.providers.base import LLMResponse, ToolCallRequest
from bubbles.session.manager import CONTEXT_EXCLUDED_KEY, SessionManager


class _ScriptedProvider:
    def __init__(self, response: LLMResponse) -> None:
        self.response = response
        self.calls = 0

    def get_default_model(self) -> str:
        return "model"

    async def chat(self, **kwargs) -> LLMResponse:
        self.calls += 1
        return self.response


class _Context:
    def add_user_messages(self, messages, inbound):
        raise AssertionError("no injections expected")

    def add_assistant_message(
        self,
        messages,
        content,
        tool_calls=None,
        reasoning_content=None,
    ):
        message = {"role": "assistant", "content": content}
        if tool_calls:
            message["tool_calls"] = tool_calls
        if reasoning_content is not None:
            message["reasoning_content"] = reasoning_content
        messages.append(message)
        return messages

    def add_tool_result(self, messages, tool_call_id, tool_name, result):
        messages.append({
            "role": "tool",
            "tool_call_id": tool_call_id,
            "name": tool_name,
            "content": result,
        })
        return messages


class _SandboxPool:
    async def get(self, key, session_dir, sandbox_name):
        return SimpleNamespace(root=session_dir)


def _make_loop(tmp_path, provider: _ScriptedProvider, bus=None) -> AgentLoop:
    return AgentLoop(
        bus=bus or MessageBus(),
        provider=provider,
        max_tokens=100,
        memory_window=500,
        context_limit=1_000_000,
        model="model",
        session_manager=SessionManager(sessions_dir=tmp_path),
    )


def _silent_response(*, mixed: bool = False) -> LLMResponse:
    calls = [
        ToolCallRequest(id="silent-1", name="stay_silent", arguments={}),
    ]
    if mixed:
        calls.insert(0, ToolCallRequest(
            id="message-1",
            name="message",
            arguments={"content": "must not be sent"},
        ))
    return LLMResponse(content="这条也不能作为 progress 发出去", tool_calls=calls)


def test_stay_silent_is_registered_for_every_turn(tmp_path) -> None:
    loop = _make_loop(tmp_path, _ScriptedProvider(_silent_response()))
    kwargs = dict(
        channel="wechat",
        chat_id="room@chatroom",
        message_id=None,
        session_dir=tmp_path,
        session_key="session",
        session=None,
        sandbox=None,
    )

    assert loop.build_turn_tools(**kwargs, system_triggered=False).has("stay_silent")
    assert loop.build_turn_tools(**kwargs, system_triggered=True).has("stay_silent")
    assert loop.tools.has("stay_silent")
    assert StaySilentTool().parameters == {
        "type": "object",
        "properties": {},
        "additionalProperties": False,
    }


@pytest.mark.asyncio
async def test_stay_silent_stops_without_progress_or_parallel_side_effects(tmp_path) -> None:
    provider = _ScriptedProvider(_silent_response(mixed=True))
    loop = _make_loop(tmp_path, provider)
    loop.max_iterations = 1
    loop._get_context = lambda session: _Context()
    session = loop.sessions.get_or_create("wechat:room")
    initial = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "泡泡玛特上新了"},
    ]
    turn_state = TurnState.from_context_messages(initial, history_length=0)
    sent = []
    progress = []
    tools = ToolRegistry()
    message = MessageTool(send_callback=sent.append)
    message.set_context("wechat", "room@chatroom")
    tools.register(message)
    tools.register(StaySilentTool())

    async def collect_progress(content: str, *, tool_hint: bool = False) -> None:
        progress.append((content, tool_hint))

    final, tools_used, _ = await loop._run_agent_loop(
        initial,
        on_progress=collect_progress,
        session=session,
        tools=tools,
        turn_state=turn_state,
    )

    assert provider.calls == 1
    assert final is None
    assert tools_used == ["stay_silent"]
    assert turn_state.suppress_outbound is True
    assert progress == []
    assert sent == []

    persist_turn_state(session, turn_state)
    assert session.messages[0]["role"] == "user"
    assert all(
        message.get(CONTEXT_EXCLUDED_KEY) is True
        for message in session.messages[1:]
    )
    assert "stay-silent" not in repr(session.get_history())


@pytest.mark.asyncio
async def test_user_turn_stay_silent_has_no_outbound_or_fallback(tmp_path) -> None:
    bus = MessageBus()
    provider = _ScriptedProvider(_silent_response())
    loop = _make_loop(tmp_path, provider, bus=bus)
    loop._sandboxes = _SandboxPool()
    loop._session_bindings["wechat:room@chatroom"] = "shared-session"
    msg = InboundMessage(
        channel="wechat",
        sender_id="wxid_alice",
        chat_id="room@chatroom",
        content="大家在聊泡泡浴",
        metadata={"is_group": True, "respond": True},
    )

    response = await loop._process_message(msg)

    assert response is None
    assert bus.outbound.empty()
    session = loop.sessions.get_or_create("shared-session")
    assert "I've completed processing" not in repr(session.messages)
    assert any("大家在聊泡泡浴" in str(message.get("content")) for message in session.messages)
    assert "stay-silent" not in repr(session.get_history())


@pytest.mark.asyncio
async def test_system_turn_stay_silent_has_no_background_fallback(tmp_path) -> None:
    provider = _ScriptedProvider(_silent_response())
    loop = _make_loop(tmp_path, provider)
    loop._sandboxes = _SandboxPool()
    loop._session_bindings["wechat:room@chatroom"] = "shared-session"
    msg = InboundMessage(
        channel="system",
        sender_id="heartbeat",
        chat_id="shared-session",
        content="check whether anything needs attention",
    )

    response = await process_system_message(loop, msg)

    assert response is None
    session = loop.sessions.get_or_create("shared-session")
    assert "Background task completed" not in repr(session.messages)
    assert "stay-silent" not in repr(session.get_history())


@pytest.mark.asyncio
async def test_message_tool_still_sends_normally() -> None:
    sent = []

    async def send(message) -> None:
        sent.append(message)

    tool = MessageTool(send_callback=send, default_channel="wechat", default_chat_id="room")

    result = await tool.execute(content="hello")

    assert result == "Message sent to wechat:room"
    assert tool._sent_in_turn is True
    assert [message.content for message in sent] == ["hello"]
