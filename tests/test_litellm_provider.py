"""Focused tests for LiteLLM request shaping."""

from __future__ import annotations

from copy import deepcopy

from bubbles.providers.litellm_provider import LiteLLMProvider


def _cache_breakpoint_count(messages: list[dict], tools: list[dict] | None) -> int:
    message_count = sum(
        1
        for message in messages
        for block in (
            message.get("content", [])
            if isinstance(message.get("content"), list)
            else []
        )
        if isinstance(block, dict) and "cache_control" in block
    )
    tool_count = sum(1 for tool in tools or [] if "cache_control" in tool)
    return message_count + tool_count


def test_cache_control_never_exceeds_anthropic_breakpoint_limit() -> None:
    provider = LiteLLMProvider(default_model="claude-sonnet-4-5")
    messages = [
        {"role": "system", "content": f"SYSTEM-{index}"}
        for index in range(5)
    ] + [{"role": "user", "content": "hello"}]
    tools = [
        {"type": "function", "function": {"name": "read", "parameters": {}}},
        {"type": "function", "function": {"name": "write", "parameters": {}}},
    ]
    original_messages = deepcopy(messages)
    original_tools = deepcopy(tools)

    cached_messages, cached_tools = provider._apply_cache_control(messages, tools)

    assert _cache_breakpoint_count(cached_messages, cached_tools) == 4
    assert cached_tools is not None
    assert cached_tools[-1]["cache_control"] == {"type": "ephemeral"}
    cached_systems = [
        message["content"][0]["text"]
        for message in cached_messages
        if message.get("role") == "system"
        and isinstance(message.get("content"), list)
    ]
    assert cached_systems == ["SYSTEM-0", "SYSTEM-3", "SYSTEM-4"]
    assert messages == original_messages
    assert tools == original_tools
