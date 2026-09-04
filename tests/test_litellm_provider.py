"""Focused tests for LiteLLM request shaping."""

from __future__ import annotations

import os
from copy import deepcopy
from types import SimpleNamespace

import litellm
import pytest

from bubbles.config.schema import Config
from bubbles.providers.litellm_provider import LiteLLMProvider
from bubbles.providers.registry import find_by_name


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


def test_glm_5_3_flash_routes_to_zhipu_china_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ZAI_API_BASE", raising=False)
    monkeypatch.delenv("ZAI_API_KEY", raising=False)
    monkeypatch.delenv("ZHIPUAI_API_KEY", raising=False)

    config = Config(
        agents={"defaults": {"model": "glm-5.3-flash"}},
        providers={"zhipu": {"apiKey": "test-key"}},
    )
    provider = LiteLLMProvider(
        api_key=config.get_api_key(),
        api_base=config.get_api_base(),
        default_model=config.agents.defaults.model,
        provider_name=config.get_provider_name(),
    )

    assert config.get_provider_name() == "zhipu"
    assert provider._resolve_model("glm-5.3-flash") == "zai/glm-5.3-flash"
    assert provider.api_base == "https://open.bigmodel.cn/api/paas/v4"
    assert os.environ["ZAI_API_BASE"] == "https://open.bigmodel.cn/api/paas/v4"
    assert os.environ["ZAI_API_KEY"] == "test-key"
    assert os.environ["ZHIPUAI_API_KEY"] == "test-key"


def test_api_base_is_instance_scoped(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(litellm, "api_base", "https://global.example/v1")

    provider = LiteLLMProvider(
        api_key="test-key",
        api_base="https://instance.example/v1",
        default_model="glm-5.3-flash",
        provider_name="zhipu",
    )

    assert provider.api_base == "https://instance.example/v1"
    assert litellm.api_base == "https://global.example/v1"


def test_zai_litellm_prefix_is_an_alias_for_zhipu_config() -> None:
    config = Config(
        agents={"defaults": {"model": "zai/glm-5.3-flash"}},
        providers={"zhipu": {"apiKey": "test-key"}},
    )

    assert find_by_name("zai").name == "zhipu"
    assert config.get_provider_name() == "zhipu"

    provider = LiteLLMProvider(
        api_key="test-key",
        default_model="zhipu/glm-5.3-flash",
        provider_name="zhipu",
    )
    assert provider._resolve_model("zhipu/glm-5.3-flash") == "zai/glm-5.3-flash"


@pytest.mark.asyncio
async def test_glm_5_3_flash_preserves_images_tools_and_agent_thinking(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict = {}

    async def fake_acompletion(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content="ok",
                        tool_calls=None,
                        reasoning_content="reasoning",
                    ),
                    finish_reason="stop",
                )
            ],
            usage=SimpleNamespace(
                prompt_tokens=1,
                completion_tokens=1,
                total_tokens=2,
            ),
        )

    monkeypatch.setattr("bubbles.providers.litellm_provider.acompletion", fake_acompletion)
    provider = LiteLLMProvider(
        api_key="test-key",
        api_base="https://open.bigmodel.cn/api/paas/v4",
        default_model="glm-5.3-flash",
        provider_name="zhipu",
    )
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/png;base64,AAAA"},
                },
                {"type": "text", "text": "描述图片"},
            ],
        }
    ]
    tools = [
        {
            "type": "function",
            "function": {"name": "ping", "parameters": {"type": "object"}},
        }
    ]

    response = await provider.chat(messages=messages, tools=tools)

    assert response.content == "ok"
    assert response.reasoning_content == "reasoning"
    assert captured["model"] == "zai/glm-5.3-flash"
    assert captured["api_base"] == "https://open.bigmodel.cn/api/paas/v4"
    assert captured["messages"] == messages
    assert captured["tools"] == tools
    assert captured["tool_choice"] == "auto"
    assert captured["extra_body"] == {
        "thinking": {"type": "enabled", "clear_thinking": False},
        "reasoning_effort": "max",
    }


def test_glm_model_policy_does_not_leak_to_other_models() -> None:
    provider = LiteLLMProvider(default_model="glm-5.3")
    kwargs = {"temperature": 1.0}

    provider._apply_param_policy("zai/glm-5.3", kwargs)

    assert kwargs == {"temperature": 1.0}
