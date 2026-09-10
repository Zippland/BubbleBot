"""Keep native provider diagnostics, with shape-independent redaction for chat."""

from __future__ import annotations

import ast
import json
import re
from collections.abc import Iterable
from dataclasses import dataclass
from html import unescape
from itertools import islice
from typing import Any

MAX_REASON_LENGTH = 800
_NO_REASON = "接口未返回错误正文，或正文仅包含已隐藏的敏感信息。"
_HIDDEN = "[已隐藏]"
# A privacy denylist, not a whitelist of vendor error formats. Unknown diagnostic
# fields (detail, errors, type, nested validation data, etc.) remain visible.
_PRIVATE_FIELDS = frozenset({
    "apikey", "accesskey", "accesstoken", "refreshtoken", "token", "password", "secret",
    "authorization", "cookie", "appcode", "headers", "requestheaders", "request",
    "requestbody", "requestdata", "requestargs", "requestpayload", "messages", "prompt",
    "input", "inputs", "inputvalue", "output", "content", "instructions", "reasoningcontent",
    "arguments", "stack", "stacktrace", "traceback",
})


@dataclass(frozen=True)
class PublicErrorDetails:
    reason: str
    status_code: int | None = None
    code: str | None = None

    def render(self) -> str:
        labels = []
        if self.status_code is not None:
            labels.append(f"HTTP {self.status_code}")
        if self.code:
            labels.append(f"错误码 {self.code}")
        prefix = "，".join(labels) + "\n" if labels else ""
        return f"{prefix}原因：{self.reason}"


def _fragment(text: str, offset: int = 0) -> tuple[int, int, Any] | None:
    """Find embedded JSON/SDK dict repr without discarding surrounding error text."""
    for match in islice(re.finditer(r"[\[{]", text[offset:]), 32):
        start = offset + match.start()
        try:
            parsed, end = json.JSONDecoder().raw_decode(text, start)
        except (ValueError, RecursionError):
            end = _literal_end(text, start)
            if end is None:
                continue
            try:
                # OpenAI-compatible SDKs sometimes include a Python dict repr.
                parsed = ast.literal_eval(text[start:end])
            except (ValueError, SyntaxError, RecursionError):
                continue
        if isinstance(parsed, (dict, list)):
            return start, end, parsed
    return None


def _literal_end(text: str, start: int) -> int | None:
    """Bound an SDK dict repr, including quoted braces, without eating its suffix."""
    stack = []
    quote = None
    escaped = False
    for index in range(start, min(len(text), start + 65_536)):
        char = text[index]
        if quote:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == quote:
                quote = None
        elif char in "\"'":
            quote = char
        elif char in "{[(":
            stack.append(char)
        elif char in "}])":
            if not stack or stack.pop() != {"}": "{", "]": "[", ")": "("}[char]:
                return None
            if not stack:
                return index + 1
    return None


def _payload(value: Any) -> dict[str, Any] | list[Any] | None:
    if isinstance(value, (dict, list)):
        return value
    if isinstance(value, str):
        fragment = _fragment(value)
        if fragment:
            return fragment[2]
    return None


def _fields(payload: Any) -> tuple[str | None, str | None]:
    if not isinstance(payload, dict):
        return None, None
    error = payload.get("error", payload)
    if isinstance(error, str):
        return error, None
    if not isinstance(error, dict):
        return None, None
    message = error.get("message")
    code = error.get("code")
    # Codes are identifiers, not arbitrary response bodies or exception types.
    code = str(code) if isinstance(code, (str, int)) and not isinstance(code, bool) else None
    if code and not re.fullmatch(r"[\w.:-]{1,64}", code, flags=re.ASCII):
        code = None
    return message if isinstance(message, str) and message.strip() else None, code


def _request_texts(value: Any) -> Iterable[str]:
    """Only redact message bodies/reasoning/arguments, not parameter or model names."""
    if isinstance(value, dict):
        for key, item in value.items():
            if key in {"content", "text", "reasoning_content", "arguments"} and isinstance(item, str):
                if len(item) >= 8:
                    yield item
            elif isinstance(item, (list, dict)):
                yield from _request_texts(item)
    elif isinstance(value, list):
        for item in value:
            yield from _request_texts(item)


def _redact(text: str, sensitive_values: Iterable[str]) -> str:
    for value in sensitive_values:
        if value:
            text = text.replace(value, _HIDDEN)
            # SDK strings may contain escaped JSON instead of the original text.
            text = text.replace(json.dumps(value, ensure_ascii=True)[1:-1], _HIDDEN)
    text = re.sub(r"(?i)\b(?:bearer|basic)\s+[A-Za-z0-9._~+/=-]+", _HIDDEN, text)
    text = re.sub(
        r"(?i)\b(?:api[_ -]?key|access[_ -]?token|refresh[_ -]?token|password|secret|"
        r"authorization|cookie|app-code)\b[\"']?\s*[:=]\s*"
        r"(?:\"[^\"\n]*\"|'[^'\n]*'|[^\s,;}\]]+)",
        _HIDDEN, text,
    )
    text = re.sub(r"\b(?:sk-[\w-]{8,}|eyJ[\w-]+\.[\w-]+\.[\w-]+)", _HIDDEN, text)
    text = re.sub(r"(?i)https?://[^\s<>\"']+", "[链接已隐藏]", text)
    text = re.sub(r"(?<![\w/])(?:[A-Za-z]:[\\/]|/)[^\s\"'<>，；。]+", "[路径已隐藏]", text)
    text = re.sub(
        r"(?i)(?<![\w.\]])([\w-]+|request\s+(?:body|headers|payload))[\"']?\s*[:=]\s*"
        r"(?:\"[^\"\n]*\"|'[^'\n]*'|[^\s,;}\]]+)",
        lambda match: f"{match[1]}={_HIDDEN}" if _private_field(match[1]) else match[0],
        text,
    )
    return text


def _private_field(key: str) -> bool:
    return re.sub(r"[^a-z0-9]", "", key.lower()) in _PRIVATE_FIELDS


def _native_text(value: Any) -> str:
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)


def _scrub(value: Any, sensitive_values: Iterable[str], depth: int = 0) -> Any:
    """Retain arbitrary error shapes; redact only private fields and known secrets."""
    if depth > 20:
        return "[嵌套内容已截断]"
    if isinstance(value, dict):
        return {
            _redact(str(key), sensitive_values): (
                _HIDDEN if _private_field(str(key)) else _scrub(item, sensitive_values, depth + 1)
            )
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_scrub(item, sensitive_values, depth + 1) for item in value]
    if not isinstance(value, str):
        if value is None or isinstance(value, (int, float, bool)):
            return value
        value = value.decode("utf-8", "replace") if isinstance(value, bytes) else str(value)

    text = value
    # Strip stack frames, not the terminal error line at the bottom of a traceback.
    if "Traceback (most recent call last):" in text:
        prefix, trace = text.split("Traceback (most recent call last):", 1)
        text = prefix + "\n" + "\n".join(
            line for line in trace.splitlines() if line and not line[0].isspace()
        )
    # Proxy errors can be HTML. Preserve their visible error text, not scripts/styles.
    if re.search(r"(?i)<(?:!doctype|html|head|body|h[1-6]|title|script|style)\b", text):
        text = re.sub(r"(?is)<(script|style)\b[^>]*>.*?</\1>", "", text)
        text = unescape(re.sub(r"<[^>]+>", " ", text))
    text = re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", text)

    parts = []
    offset = 0
    while fragment := _fragment(text, offset):
        start, end, payload = fragment
        prefix = text[offset:start]
        label = re.search(r"(?i)([\w-]+|request\s+(?:body|headers|payload))[\"']?\s*[:=]\s*$", prefix)
        private = (label and _private_field(label[1])) or text[start:end] in sensitive_values
        cleaned = _HIDDEN if private else _native_text(_scrub(payload, sensitive_values, depth + 1))
        parts.extend((_redact(prefix, sensitive_values), cleaned))
        offset = end
    parts.append(_redact(text[offset:], sensitive_values))
    return "".join(parts)


def _safe_reason(value: Any, sensitive_values: Iterable[str]) -> str:
    text = _native_text(_scrub(value, sensitive_values))
    # Drop nested SDK class prefixes while retaining the vendor's actual words.
    prefix = r"^(?:[\w.]+(?:Error|Exception)\s*[:\-]\s*|(?:HTTP|Error code:)\s*\d{3}\s*[:\-]\s*)"
    text = text.strip()
    while True:
        cleaned = re.sub(prefix, "", text).strip()
        if cleaned == text:
            break
        text = cleaned
    text = " ".join(text.split()).strip(" ,;:-")
    if not text:
        return _NO_REASON
    if len(text) > MAX_REASON_LENGTH:
        return text[:MAX_REASON_LENGTH - 1] + "…"
    return text


def extract_error_details(
    exc: BaseException,
    *,
    sensitive_values: Iterable[str | None] = (),
    messages: list[dict[str, Any]] | None = None,
) -> PublicErrorDetails:
    """Use common fields for readability, with native errors as the universal fallback."""
    secrets = tuple(value for value in sensitive_values if value) + tuple(_request_texts(messages))
    seen: set[int] = set()
    current: BaseException | None = exc
    reason = code = status = None
    fallback = str(exc)
    while current is not None and id(current) not in seen and len(seen) < 10:
        seen.add(id(current))
        response = getattr(current, "response", None)
        raw_status = getattr(current, "status_code", None) or getattr(response, "status_code", None)
        text = str(current)
        if text.strip():
            fallback = text
        if raw_status is None:
            match = re.search(r"\b(?:HTTP|Error code:)\s*(\d{3})\b", text)
            raw_status = int(match[1]) if match else None
        if status is None and isinstance(raw_status, int) and 100 <= raw_status <= 599:
            status = raw_status

        body = getattr(current, "body", None)
        if not body and response is not None:
            try:
                body = response.json()
            except (ValueError, RuntimeError, AttributeError):
                try:
                    body = response.text
                except (RuntimeError, AttributeError):
                    pass
        if body:
            reason = body  # Keep even nonstandard bodies; never depend on a message field.
        payload = _payload(body if body else text)
        if payload:
            message, vendor_code = _fields(payload)
            error = payload.get("error", payload) if isinstance(payload, dict) else payload
            # Compact the familiar message/code shape only. Extra diagnostic fields
            # remain in native form, without per-provider field mappings.
            simple_envelope = not isinstance(payload, dict) or "error" not in payload or set(payload) == {"error"}
            if isinstance(body, dict) and message and simple_envelope and (isinstance(error, str) or set(error) <= {"message", "code"}):
                reason = message
            if vendor_code:
                code = vendor_code
        current = current.__cause__ or current.__context__

    # Keep raw detail on LLMCallError for logging, never in the public reply.
    return PublicErrorDetails(
        reason=_safe_reason(reason or fallback, secrets),
        status_code=status,
        code=_redact(code, secrets) if code else None,
    )
