"""Regression tests for stable prefixes and provider history/usage fidelity."""

from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from bubbles.agent.subagent import SubagentManager
from bubbles.agent.tools.registry import ToolRegistry
from bubbles.agent.turn import persist_turn_messages
from bubbles.bus.queue import MessageBus
from bubbles.providers.base import LLMResponse, ToolCallRequest, normalize_usage
from bubbles.providers.custom_provider import CustomProvider
from bubbles.providers.litellm_provider import LiteLLMProvider
from bubbles.providers.openai_codex_provider import _convert_messages, _prompt_cache_key
from bubbles.session.manager import SessionManager


def test_codex_cache_key_stays_stable_as_conversation_grows():
    messages = [{"role": "system", "content": "stable workspace instructions"}]
    key = _prompt_cache_key(messages)
    for message in [
        {"role": "user", "content": "question"},
        {"role": "assistant", "content": "reply"},
        {"role": "system", "content": "appended runtime notice"},
        {"role": "user", "content": "followup"},
    ]:
        messages.append(message)
        assert _prompt_cache_key(messages) == key
    changed = deepcopy(messages)
    changed[0]["content"] = "different workspace instructions"
    assert _prompt_cache_key(changed) != key
    assert _prompt_cache_key([]) == _prompt_cache_key([{"role": "user", "content": "no system"}])


def test_codex_append_preserves_instructions_and_input_prefix():
    messages = [
        {"role": "system", "content": "stable"},
        {"role": "user", "content": "question"},
        {"role": "assistant", "content": None, "tool_calls": [{
            "id": "call_a|fc_a", "type": "function",
            "function": {"name": "find_person", "arguments": '{"query":"Alice"}'},
        }]},
        {"role": "tool", "tool_call_id": "call_a|fc_a", "content": "result"},
    ]
    instructions, before = _convert_messages(messages)
    messages.extend([
        {"role": "system", "content": "runtime notice"},
        {"role": "user", "content": "next"},
    ])
    new_instructions, after = _convert_messages(messages)
    assert new_instructions == instructions == "stable"
    assert after[:len(before)] == before
    assert after[-2] == {"role": "developer", "content": [{"type": "input_text", "text": "runtime notice"}]}


def test_subagent_system_prompt_has_no_clock_or_task_content(tmp_path):
    manager = SubagentManager.__new__(SubagentManager)
    first = manager._build_subagent_prompt("task A", tmp_path)
    assert first == manager._build_subagent_prompt("task B", tmp_path)
    assert "Current Time" not in first
    assert "task A" not in first


@pytest.mark.parametrize("reasoning", ["", "reasoning"])
def test_reasoning_and_end_tool_survive_save_reload_verbatim(tmp_path, reasoning):
    manager = SessionManager(sessions_dir=tmp_path)
    session = manager.get_or_create("test")
    messages = [
        {"role": "user", "content": "question"},
        {"role": "assistant", "content": "reply", "reasoning_content": reasoning},
        {"role": "assistant", "content": None, "reasoning_content": reasoning,
         "tool_calls": [{"id": "end", "type": "function", "function": {"name": "stay_silent", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "end", "name": "stay_silent", "content": "[stay-silent]"},
    ]
    persist_turn_messages(session, deepcopy(messages))
    manager.save(session)
    manager.invalidate("test")
    assert manager.get_or_create("test").get_history() == messages


@pytest.mark.parametrize("as_dict", [False, True])
def test_usage_keeps_cache_counters_without_fabricating_missing_values(as_dict):
    data = dict(prompt_tokens=1000, completion_tokens=10, total_tokens=1010,
        prompt_tokens_details={"cached_tokens": 800}, cache_read_input_tokens=700,
        cache_creation_input_tokens=100)
    usage = data if as_dict else SimpleNamespace(**data)
    assert normalize_usage(usage) == dict(prompt_tokens=1000, completion_tokens=10,
        total_tokens=1010, cached_tokens=800, cache_read_input_tokens=700, cache_creation_input_tokens=100)
    assert normalize_usage(None) == {}
    assert normalize_usage({"prompt_tokens": 100}) == {"prompt_tokens": 100}


@pytest.mark.parametrize("provider_class,method", [(LiteLLMProvider, "_parse_response"), (CustomProvider, "_parse")])
def test_provider_parse_retains_reported_cache_usage(provider_class, method):
    provider = provider_class.__new__(provider_class)
    response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="ok", tool_calls=[]), finish_reason="stop")],
        usage=SimpleNamespace(prompt_tokens=1024, completion_tokens=10, total_tokens=1034,
            prompt_tokens_details=SimpleNamespace(cached_tokens=900)),
    )
    assert getattr(provider, method)(response).usage["cached_tokens"] == 900


@pytest.mark.parametrize("provider_class,method", [
    (LiteLLMProvider, "_parse_response"), (CustomProvider, "_parse"),
])
@pytest.mark.parametrize("reasoning_fields", [
    pytest.param({}, id="missing"),
    pytest.param({"reasoning_content": None}, id="null"),
    pytest.param({"reasoning_content": ""}, id="empty"),
    pytest.param({"reasoning_content": " raw reasoning\n"}, id="text"),
])
@pytest.mark.parametrize("with_tools", [False, True])
def test_provider_parse_preserves_reported_reasoning(provider_class, method, reasoning_fields, with_tools):
    provider = provider_class.__new__(provider_class)
    tool_calls = [SimpleNamespace(
        id="end", function=SimpleNamespace(name="stay_silent", arguments="{}"),
    )] if with_tools else []
    raw_response = SimpleNamespace(
        choices=[SimpleNamespace(
            message=SimpleNamespace(content="reply", tool_calls=tool_calls, **reasoning_fields),
            finish_reason="tool_calls" if with_tools else "stop",
        )],
        usage=None,
    )

    parsed = getattr(provider, method)(raw_response)

    assert parsed.reasoning_content == reasoning_fields.get("reasoning_content")
    assert parsed.content == "reply"
    assert parsed.tool_calls == ([ToolCallRequest("end", "stay_silent", {})] if with_tools else [])


@pytest.mark.asyncio
@pytest.mark.parametrize("reasoning", [None, "", "raw reasoning"])
async def test_subagent_preserves_reasoning_in_next_tool_request(monkeypatch, tmp_path, reasoning):
    responses = iter([
        LLMResponse(
            content=None, reasoning_content=reasoning,
            tool_calls=[ToolCallRequest("read", "read_file", {"path": "example.txt"})],
        ),
        LLMResponse(content="done"),
    ])
    requests = []

    async def chat(**kwargs):
        requests.append(deepcopy(kwargs))
        return next(responses)

    provider = SimpleNamespace(chat=chat, get_default_model=lambda: "test-model")
    manager = SubagentManager(provider=provider, bus=MessageBus())
    execute = AsyncMock(return_value="file contents")
    monkeypatch.setattr(ToolRegistry, "execute", execute)

    await manager._run_subagent(
        "test", "read example.txt", "test", {"channel": "wechat", "chat_id": "room"},
        session_dir=tmp_path,
    )

    assert len(requests) == 2
    execute.assert_awaited_once_with("read_file", {"path": "example.txt"})
    assistant = next(m for m in requests[1]["messages"] if m["role"] == "assistant")
    if reasoning is None:
        assert "reasoning_content" not in assistant
    else:
        assert assistant["reasoning_content"] == reasoning
