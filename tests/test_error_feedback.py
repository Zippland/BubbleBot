"""Provider reasons reach the triggering chat without leaking SDK/request dumps."""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import litellm.exceptions as le
import pytest
from openai import BadRequestError

from bubbles.agent.loop import AgentLoop
from bubbles.bus.events import InboundMessage
from bubbles.bus.queue import MessageBus
from bubbles.providers.base import LLMCallError, LLMErrorKind, to_llm_call_error
from bubbles.providers.error_details import MAX_REASON_LENGTH

GLM_REASON = "系统检测到输入或生成内容可能包含不安全或敏感内容，请您避免输入易产生敏感内容的提示语，感谢您的配合。"


def sdk_error(body, *, status=400):
    response = httpx.Response(
        status, json=body,
        request=httpx.Request("POST", "https://provider.invalid/v1/chat/completions"),
    )
    return BadRequestError("upstream error", response=response, body=body)


def nested_glm_error():
    original = sdk_error({"error": {"message": GLM_REASON, "code": "1301"}})
    # Production LiteLLM/ZAI errors have multiple wrappers with absent/empty bodies.
    middle = RuntimeError("OpenAIException - request rejected")
    middle.body = {}
    middle.__cause__ = original
    outer = le.BadRequestError("ZaiException - request rejected", llm_provider="zai", model="glm-5.3-flash")
    outer.__cause__ = middle
    return outer


def test_nested_glm_error_preserves_vendor_reason_and_code():
    error = to_llm_call_error(nested_glm_error())
    assert error.kind is LLMErrorKind.PERMANENT
    assert error.user_message(1) == (
        "⚠️ 模型接口拒绝了这次请求。\nHTTP 400，错误码 1301\n原因：" + GLM_REASON
    )
    assert "ZaiException" in error.detail  # Raw detail remains available for logs.
    assert to_llm_call_error(error) is error


@pytest.mark.parametrize("wrapped", [False, True])
def test_sdk_flat_and_nested_error_payloads(wrapped):
    details = {"message": "The supported API model names are deepseek-v4-flash.", "code": "model_not_found"}
    body = {"error": details} if wrapped else details
    text = to_llm_call_error(sdk_error(body)).user_message(1)
    assert "HTTP 400" in text
    assert "错误码 model_not_found" in text
    assert details["message"] in text


@pytest.mark.parametrize("serialize", [json.dumps, repr])
def test_http_error_with_serialized_body(serialize):
    body = {"error": {"code": 1301, "message": GLM_REASON}, "request": {"messages": "do not publish"}}
    text = to_llm_call_error(RuntimeError(f"HTTP 400: {serialize(body)}")).user_message(1)
    assert GLM_REASON in text and "HTTP 400" in text and "1301" in text
    assert "do not publish" not in text


def test_response_body_is_used_when_exception_has_no_body():
    error = RuntimeError("upstream failure")
    error.response = httpx.Response(429, json={"error": {"message": "Too many concurrent requests", "code": "1302"}})
    text = to_llm_call_error(error).user_message(3)
    assert "HTTP 429" in text and "1302" in text
    assert "Too many concurrent requests" in text
    assert "已重试 2 次" in text
    assert "限流" in text


def test_unstructured_fallback_is_readable_without_sdk_prefixes_or_paths():
    detail = "litellm.InternalServerError: OpenAIException - /internal/path.py upstream unavailable"
    error = to_llm_call_error(le.RateLimitError(detail, llm_provider="p", model="m"))
    text = error.user_message(3)
    assert "upstream unavailable" in text and "已重试 2 次" in text
    for leak in ("litellm", "Exception", "/internal/path.py", "Traceback"):
        assert leak not in text


def test_validation_parameter_paths_are_not_mistaken_for_request_content():
    reason = "messages[2].content must be a string; received null; tool_calls[0].id is required"
    assert reason in to_llm_call_error(sdk_error({"message": reason})).user_message(1)


def test_only_message_and_code_are_selected_from_response():
    body = {
        "error": {"message": "Invalid model", "code": "invalid_model", "input": "secret prompt"},
        "request": {"messages": [{"content": "private conversation"}]},
        "headers": {"Authorization": "Bearer private-token"},
        "stack": "/internal/path.py",
    }
    text = to_llm_call_error(sdk_error(body)).user_message(1)
    assert "Invalid model" in text
    for leak in ("secret prompt", "private conversation", "private-token", "/internal/path.py"):
        assert leak not in text


def test_secrets_and_echoed_request_text_are_redacted():
    key = "configured-key.with-provider-specific-format"
    prompt = "This is a private message from another group."
    reasoning = "Private model reasoning that must not be echoed."
    arguments = '{"private_tool_argument":"secret data"}'
    reason = (
        f"Invalid request using {key}; echoed {prompt}; {reasoning}; {arguments}; "
        "Authorization: Bearer hidden-bearer-token; api_key=hidden-api-key; "
        'password="hidden password"; refresh_token=hidden-refresh-token; '
        "sk-abcdefghijklmno; https://user:password@internal.invalid/path?token=hidden-query; "
        r"C:\Users\Administrator\private.py"
    )
    text = to_llm_call_error(
        sdk_error({"message": reason}), sensitive_values=(key,),
        messages=[
            {"role": "user", "content": [{"type": "text", "text": prompt}]},
            {"role": "assistant", "reasoning_content": reasoning,
             "tool_calls": [{"function": {"name": "execute", "arguments": arguments}}]},
        ],
    ).user_message(1)
    assert "Invalid request" in text
    for leak in (key, prompt, reasoning, arguments, "secret data", "hidden-bearer-token", "hidden-api-key",
                 "hidden password", "hidden-refresh-token", "sk-abcdefghijklmno", "internal.invalid",
                 "hidden-query", "Administrator"):
        assert leak not in text


@pytest.mark.parametrize("tail", [
    '\nRequest body: {"messages": "private content"}',
    ' request_body={"messages": "private content"}',
    ' messages=[{"content": "private content"}]',
    '\nTraceback (most recent call last):\n  File "/private/content.py", line 2',
    ", input_value='private content'",
])
def test_debug_request_and_stack_suffixes_are_omitted(tail):
    text = to_llm_call_error(RuntimeError("Invalid parameter." + tail)).user_message(1)
    assert "Invalid parameter." in text
    assert "private" not in text and "Traceback" not in text


def test_unrecognized_embedded_payloads_are_not_forwarded():
    text = to_llm_call_error(RuntimeError('Invalid input: {"content": "private conversation"}')).user_message(1)
    assert "Invalid input" in text
    assert "private conversation" not in text


@pytest.mark.parametrize("detail", [
    "Traceback (most recent call last): private traceback",
    "",
])
def test_only_empty_or_fully_filtered_details_use_a_placeholder(detail):
    text = to_llm_call_error(RuntimeError(detail)).user_message(1)
    assert "未返回错误正文" in text
    assert "private" not in text


@pytest.mark.parametrize("body", [
    {"detail": "Missing thought_signature"},
    {"vendor_specific_diagnostic": "Missing thought_signature"},
    [{"loc": ["body", "messages", 2], "msg": "Missing thought_signature"}],
    {"error": {"type": "invalid_request_error", "description": "Missing thought_signature"}},
    {"error": {"message": "Invalid request", "details": {"hint": "Missing thought_signature"}}},
    {"error": {"message": "Invalid request"}, "additional_info": "Missing thought_signature"},
])
def test_unknown_shapes_preserve_native_diagnostics_without_vendor_adapters(body):
    text = to_llm_call_error(sdk_error(body)).user_message(1)
    assert "Missing thought_signature" in text
    assert "HTTP 400" in text
    assert "未返回错误正文" not in text
    # Raw JSON remains valid when it fits in the length budget.
    assert json.loads(text.split("原因：", 1)[1]) == body


def test_native_error_type_is_retained_without_a_type_field_mapping():
    body = {"error": {"type": "invalid_request_error", "message": "Invalid tool call history"}}
    text = to_llm_call_error(sdk_error(body)).user_message(1)
    assert "invalid_request_error" in text and "Invalid tool call history" in text


@pytest.mark.parametrize("serialize", [json.dumps, repr])
def test_unknown_serialized_errors_keep_diagnostics_and_hide_private_fields(serialize):
    body = {"details": {"why": "Missing signature", "headers": {"vendor-secret": "private-header"}},
            "request": {"custom_prompt_field": "private conversation"}}
    text = to_llm_call_error(RuntimeError(f"HTTP 400: {serialize(body)}")).user_message(1)
    assert "Missing signature" in text
    assert "private-header" not in text and "private conversation" not in text


def test_plain_text_response_body_is_used_instead_of_an_opaque_sdk_error():
    error = RuntimeError("The upstream request failed")
    error.response = httpx.Response(502, text="Gateway could not reach inference worker")
    text = to_llm_call_error(error).user_message(1)
    assert "HTTP 502" in text
    assert "Gateway could not reach inference worker" in text


def test_html_proxy_errors_retain_visible_reason_without_scripts_or_credentials():
    error = RuntimeError("HTTP request failed")
    error.response = httpx.Response(502, text=(
        "<html><head><title>502 Bad Gateway</title><script>private-script</script></head>"
        "<body>Upstream connection refused &amp; retry exhausted. password=private-password</body></html>"
    ))
    text = to_llm_call_error(error).user_message(1)
    assert "502 Bad Gateway" in text and "Upstream connection refused & retry exhausted" in text
    assert "private-script" not in text and "private-password" not in text and "<html>" not in text


@pytest.mark.parametrize("serialize", [json.dumps, repr])
def test_redacting_a_request_dump_does_not_remove_the_error_that_follows(serialize):
    detail = f'Request body: {serialize({"vendor_prompt": "private prompt"})}; rejection: missing tool_call_id'
    text = to_llm_call_error(RuntimeError(detail)).user_message(1)
    assert "missing tool_call_id" in text and "private prompt" not in text


def test_sdk_dict_repr_with_quoted_braces_and_trailing_diagnostics():
    detail = "SDK failure: " + repr({"why": "Invalid {signature}", "request": {"prompt": "private prompt"}})
    detail += "; position: messages[2]"
    text = to_llm_call_error(RuntimeError(detail)).user_message(1)
    assert "Invalid {signature}" in text and "position: messages[2]" in text
    assert "private prompt" not in text


def test_native_diagnostics_on_both_sides_of_embedded_json_survive():
    detail = 'Validation failed: {"details": "Missing signature"}; field: messages[2].content'
    text = to_llm_call_error(RuntimeError(detail)).user_message(1)
    assert "Validation failed" in text and "Missing signature" in text and "messages[2].content" in text


def test_a_message_field_in_embedded_json_does_not_hide_the_actual_root_cause():
    detail = 'HTTP 400: {"error": {"message": "Bad request"}}; root cause: missing tool_call_id'
    text = to_llm_call_error(RuntimeError(detail)).user_message(1)
    assert "Bad request" in text and "missing tool_call_id" in text


def test_traceback_keeps_the_terminal_error_but_not_stack_frames():
    detail = ('Traceback (most recent call last):\n'
              '  File "/private/source.py", line 2, in chat\n'
              '    private_source_code()\n'
              'ValueError: Tool response is missing tool_call_id')
    text = to_llm_call_error(RuntimeError(detail)).user_message(1)
    assert "Tool response is missing tool_call_id" in text
    assert "private" not in text and "Traceback" not in text


def test_validation_paths_with_colons_are_not_redacted_as_private_fields():
    reason = "messages[2].content: expected string, received null"
    assert reason in to_llm_call_error(RuntimeError(reason)).user_message(1)


def test_malformed_json_keeps_the_native_reason_and_redacts_credentials():
    detail = 'HTTP 400: {"vendor_diagnostic": "Signature missing", "api_key": "private-key"'
    text = to_llm_call_error(RuntimeError(detail)).user_message(1)
    assert "Signature missing" in text and "private-key" not in text


def test_request_only_error_payloads_are_redacted_without_hiding_their_shape():
    text = to_llm_call_error(RuntimeError('{"request": {"messages": "private content"}}')).user_message(1)
    assert json.loads(text.split("原因：", 1)[1]) == {"request": "[已隐藏]"}


def test_long_messages_are_bounded_and_keep_status_and_code():
    error = to_llm_call_error(sdk_error({"error": {"message": "详细错误" * 1000, "code": 1301}}))
    text = error.user_message(1)
    assert len(error.public_details.reason) == MAX_REASON_LENGTH
    assert text.endswith("…")
    assert "HTTP 400" in text and "错误码 1301" in text


def test_exception_context_and_cycles_are_handled():
    outer = RuntimeError("wrapped failure")
    inner = sdk_error({"error": {"message": GLM_REASON, "code": "1301"}})
    outer.__context__ = inner
    inner.__context__ = outer
    assert GLM_REASON in to_llm_call_error(outer).user_message(1)


def test_nested_plaintext_reason_is_not_lost_to_a_generic_wrapper():
    outer = RuntimeError("Request failed")
    outer.__cause__ = TimeoutError("Connection timed out after 180 seconds")
    assert "Connection timed out after 180 seconds" in to_llm_call_error(outer).user_message(1)


@pytest.mark.parametrize("code", [{"secret": "private"}, ["private"], True, "x" * 100])
def test_malformed_vendor_codes_are_not_published(code):
    text = to_llm_call_error(sdk_error({"error": {"message": "Bad parameter", "code": code}})).user_message(1)
    assert "Bad parameter" in text and "错误码" not in text


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["litellm", "custom", "codex"])
async def test_providers_preserve_reason_and_redact_their_own_credentials(monkeypatch, kind):
    secret = "provider-key-that-must-not-reach-chat"
    private_text = "Private prompt from a shared workspace"
    fail = AsyncMock(side_effect=sdk_error({"error": {
        "code": "bad_request", "message": f"Bad input; {secret}; {private_text}",
    }}))
    if kind == "litellm":
        from bubbles.providers import litellm_provider as module
        monkeypatch.setattr(module, "acompletion", fail)
        provider = module.LiteLLMProvider(api_key=secret, default_model="openai/test")
    elif kind == "custom":
        from bubbles.providers.custom_provider import CustomProvider
        provider = CustomProvider(api_key=secret)
        monkeypatch.setattr(provider._client.chat.completions, "create", fail)
    else:
        from bubbles.providers import openai_codex_provider as module
        monkeypatch.setattr(module, "get_codex_token", lambda: SimpleNamespace(account_id="account", access=secret))
        monkeypatch.setattr(module, "_request_codex", fail)
        provider = module.OpenAICodexProvider()
    with pytest.raises(LLMCallError) as caught:
        await provider.chat(messages=[{"role": "user", "content": private_text}])
    text = caught.value.user_message(1)
    assert "Bad input" in text and "错误码 bad_request" in text
    assert secret not in text and private_text not in text


@pytest.mark.asyncio
@pytest.mark.parametrize("status,kind", [(400, LLMErrorKind.PERMANENT), (401, LLMErrorKind.AUTH), (429, LLMErrorKind.RATE_LIMIT)])
async def test_codex_http_errors_keep_the_real_reason_and_status(monkeypatch, status, kind):
    from bubbles.providers import openai_codex_provider as module

    def respond(request):
        return httpx.Response(status, json={"error": {"message": "Vendor-specific rejection", "code": "test_code"}})

    client = httpx.AsyncClient(transport=httpx.MockTransport(respond))
    monkeypatch.setattr(module.httpx, "AsyncClient", lambda **kwargs: client)
    with pytest.raises(httpx.HTTPStatusError) as caught:
        await module._request_codex("https://provider.invalid", {}, {}, verify=True)
    error = to_llm_call_error(caught.value)
    assert error.kind is kind
    text = error.user_message(1)
    assert f"HTTP {status}" in text
    assert "Vendor-specific rejection" in text and "test_code" in text


@pytest.mark.asyncio
@pytest.mark.parametrize("event", [
    {"type": "error", "error": {"message": "Stream rejected", "code": "stream_error"}},
    {"type": "error", "message": "Stream rejected", "code": "stream_error"},
    {"type": "response.failed", "response": {
        "error": {"message": "Stream rejected", "code": "stream_error"}, "output": "private generated text",
    }},
])
async def test_codex_stream_errors_keep_vendor_details_without_generated_text(event):
    from bubbles.providers.openai_codex_provider import _consume_sse

    response = httpx.Response(200, text=f"data: {json.dumps(event)}\n\n")
    with pytest.raises(RuntimeError) as caught:
        await _consume_sse(response)
    text = to_llm_call_error(caught.value).user_message(1)
    assert "Stream rejected" in text and "stream_error" in text
    assert "private generated text" not in str(caught.value)


@pytest.mark.asyncio
@pytest.mark.parametrize("nested", [False, True])
async def test_codex_stream_unknown_error_fields_are_not_discarded(nested):
    from bubbles.providers.openai_codex_provider import _consume_sse

    details = {"vendor_specific_diagnostic": "Missing signature"}
    event = ({"type": "response.failed", "response": {"error": details}}
             if nested else {"type": "error", **details})
    response = httpx.Response(200, text=f"data: {json.dumps(event)}\n\n")
    with pytest.raises(RuntimeError) as caught:
        await _consume_sse(response)
    assert "Missing signature" in to_llm_call_error(caught.value).user_message(1)


def error_loop():
    return SimpleNamespace(bus=MessageBus(), _last_error_reply_at={})


@pytest.mark.asyncio
@pytest.mark.parametrize("channel,chat_id", [("wechat", "123@chatroom"), ("wechat", "wxid_user"), ("cli", "direct")])
async def test_reason_is_delivered_to_the_triggering_group_private_chat_or_cli(channel, chat_id):
    loop = error_loop()
    inbound = InboundMessage(channel=channel, chat_id=chat_id, sender_id="user", content="hello")
    error = to_llm_call_error(nested_glm_error())
    await AgentLoop._emit_error_reply(loop, inbound, error)
    message = loop.bus.outbound.get_nowait()
    assert (message.channel, message.chat_id) == (channel, chat_id)
    assert message.content == error.user_message(1)
    assert GLM_REASON in message.content and "1301" in message.content
    assert loop.bus.outbound.empty()


@pytest.mark.asyncio
@pytest.mark.parametrize("channel,metadata", [("system", {}), ("wechat", {"respond": False})])
async def test_background_and_nonresponding_turns_stay_silent(channel, metadata):
    loop = error_loop()
    inbound = InboundMessage(channel=channel, chat_id="room", sender_id="user", content="hi", metadata=metadata)
    await AgentLoop._emit_error_reply(loop, inbound, to_llm_call_error(nested_glm_error()))
    assert loop.bus.outbound.empty()


@pytest.mark.asyncio
async def test_error_feedback_still_throttles_per_session(monkeypatch):
    monkeypatch.setattr("bubbles.agent.loop.time.monotonic", lambda: 1000)
    loop = error_loop()
    inbound = InboundMessage(channel="wechat", chat_id="room", sender_id="user", content="hi")
    error = to_llm_call_error(nested_glm_error())
    await AgentLoop._emit_error_reply(loop, inbound, error)
    await AgentLoop._emit_error_reply(loop, inbound, error)
    assert loop.bus.outbound.qsize() == 1


@pytest.mark.asyncio
async def test_non_llm_exceptions_do_not_expose_details():
    loop = error_loop()
    inbound = InboundMessage(channel="wechat", chat_id="room", sender_id="user", content="hi")
    await AgentLoop._emit_error_reply(loop, inbound, RuntimeError("private internal failure"))
    assert loop.bus.outbound.get_nowait().content == "Sorry, I encountered an error."
