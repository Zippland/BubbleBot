"""Regression tests for stable prefixes and provider history/usage fidelity."""

from copy import deepcopy
from types import SimpleNamespace

import pytest

from bubbles.agent.subagent import SubagentManager
from bubbles.agent.turn import persist_turn_messages
from bubbles.providers.base import normalize_usage
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


def test_reasoning_and_end_tool_survive_save_reload_verbatim(tmp_path):
    manager = SessionManager(sessions_dir=tmp_path)
    session = manager.get_or_create("test")
    messages = [
        {"role": "user", "content": "question"},
        {"role": "assistant", "content": None, "reasoning_content": "reasoning",
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
