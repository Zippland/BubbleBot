"""Base LLM provider interface."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class LLMErrorKind(str, Enum):
    """错误类别，决定重试策略与用户可见文案。

    分类的唯一目的是回答两个问题：值不值得重试、给用户看什么。
    """

    RATE_LIMIT = "rate_limit"            # 429，退避后重试
    TRANSIENT = "transient"              # 5xx / 超时 / 连接失败，退避后重试
    CONTEXT_OVERFLOW = "context_overflow"  # 上下文超限，压缩后重试
    AUTH = "auth"                        # 401/403，重试无意义
    PERMANENT = "permanent"              # 400 / 内容策略等，重试无意义


# 用户可见文案：给出类别，不给异常类型、堆栈、内部路径（SPEC §5.1）。
_KIND_MESSAGES = {
    LLMErrorKind.RATE_LIMIT: "模型接口触发限流",
    LLMErrorKind.TRANSIENT: "模型接口暂时不可用",
    LLMErrorKind.CONTEXT_OVERFLOW: "对话上下文超出模型上限",
    LLMErrorKind.AUTH: "模型接口认证失败，请检查 API key",
    LLMErrorKind.PERMANENT: "模型接口拒绝了这次请求",
}


class LLMCallError(Exception):
    """LLM 调用失败。

    Why: provider 曾把异常转成 ``LLMResponse(content="Error calling LLM: ...")``
    返回，让"传输层失败"和"模型的回答"在类型上不可区分——错误文本会被当成
    正常回复发给用户、写进历史、甚至被 compaction 当成摘要，同时 SPEC §5.1
    的错误反馈策略和 cron 退避都因为"没有异常"而从未触发。改为抛出本异常，
    由调用方决定重试与展示。
    """

    def __init__(self, kind: LLMErrorKind, detail: str, retry_after: float | None = None):
        self.kind = kind
        self.detail = detail          # 仅进日志
        self.retry_after = retry_after  # 服务端 Retry-After（秒），如果给了
        self.attempts = 1             # 实际尝试次数，由重试层回填
        super().__init__(f"{kind.value}: {detail}")

    @property
    def retryable(self) -> bool:
        return self.kind in (
            LLMErrorKind.RATE_LIMIT,
            LLMErrorKind.TRANSIENT,
            LLMErrorKind.CONTEXT_OVERFLOW,
        )

    def user_message(self, attempts: int) -> str:
        """面向用户的一句话：说清是接口问题 + 试了几次，不泄露内部细节。"""
        base = _KIND_MESSAGES.get(self.kind, "模型接口调用失败")
        if attempts > 1:
            return f"⚠️ {base}，已重试 {attempts - 1} 次仍失败，请稍后再试。"
        return f"⚠️ {base}。"


def classify_exception(exc: BaseException) -> LLMErrorKind:
    """把 provider SDK 抛出的异常映射到 LLMErrorKind。

    优先用 litellm 的异常类型（精确），退化到 status_code，最后才看字符串。
    """
    try:
        import litellm.exceptions as le

        if isinstance(exc, le.ContextWindowExceededError):
            return LLMErrorKind.CONTEXT_OVERFLOW
        if isinstance(exc, le.RateLimitError):
            return LLMErrorKind.RATE_LIMIT
        if isinstance(exc, (le.AuthenticationError, le.PermissionDeniedError)):
            return LLMErrorKind.AUTH
        if isinstance(exc, (le.Timeout, le.APIConnectionError, le.InternalServerError,
                            le.ServiceUnavailableError, le.BadGatewayError)):
            return LLMErrorKind.TRANSIENT
        if isinstance(exc, (le.ContentPolicyViolationError, le.BadRequestError,
                            le.UnprocessableEntityError, le.NotFoundError)):
            return LLMErrorKind.PERMANENT
    except ImportError:
        pass

    status = getattr(exc, "status_code", None) or getattr(exc, "code", None)
    if isinstance(status, int):
        if status == 429:
            return LLMErrorKind.RATE_LIMIT
        if status in (401, 403):
            return LLMErrorKind.AUTH
        if status >= 500 or status == 408:
            return LLMErrorKind.TRANSIENT
        if status >= 400:
            return LLMErrorKind.PERMANENT

    text = str(exc).lower()
    if "context" in text and ("length" in text or "window" in text or "too long" in text):
        return LLMErrorKind.CONTEXT_OVERFLOW
    if "rate limit" in text or "too many requests" in text:
        return LLMErrorKind.RATE_LIMIT
    if "timeout" in text or "timed out" in text or "connection" in text:
        return LLMErrorKind.TRANSIENT
    # 未知错误按可重试处理：一次网络抖动重试一下的代价，远低于把真实故障
    # 当成永久失败直接放弃。
    return LLMErrorKind.TRANSIENT


def _retry_after_of(exc: BaseException) -> float | None:
    """从异常上的响应头里取 Retry-After（秒）。"""
    headers = getattr(exc, "response_headers", None) or getattr(exc, "headers", None)
    if not headers:
        resp = getattr(exc, "response", None)
        headers = getattr(resp, "headers", None) if resp is not None else None
    if not headers:
        return None
    try:
        raw = headers.get("retry-after") or headers.get("Retry-After")
        return float(raw) if raw is not None else None
    except (TypeError, ValueError):
        return None


def to_llm_call_error(exc: BaseException) -> LLMCallError:
    """Wrap any provider-SDK exception into a classified LLMCallError."""
    if isinstance(exc, LLMCallError):
        return exc
    return LLMCallError(classify_exception(exc), str(exc), _retry_after_of(exc))


@dataclass
class ToolCallRequest:
    """A tool call request from the LLM."""
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class LLMResponse:
    """Response from an LLM provider."""
    content: str | None
    tool_calls: list[ToolCallRequest] = field(default_factory=list)
    finish_reason: str = "stop"
    usage: dict[str, int] = field(default_factory=dict)
    reasoning_content: str | None = None  # Kimi, DeepSeek-R1 etc.
    
    @property
    def has_tool_calls(self) -> bool:
        """Check if response contains tool calls."""
        return len(self.tool_calls) > 0


class LLMProvider(ABC):
    """
    Abstract base class for LLM providers.
    
    Implementations should handle the specifics of each provider's API
    while maintaining a consistent interface.
    """
    
    def __init__(self, api_key: str | None = None, api_base: str | None = None):
        self.api_key = api_key
        self.api_base = api_base

    @staticmethod
    def _sanitize_empty_content(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Replace empty text content that causes provider 400 errors.

        Empty content can appear when MCP tools return nothing. Most providers
        reject empty-string content or empty text blocks in list content.
        """
        result: list[dict[str, Any]] = []
        for msg in messages:
            content = msg.get("content")

            if isinstance(content, str) and not content:
                clean = dict(msg)
                clean["content"] = None if (msg.get("role") == "assistant" and msg.get("tool_calls")) else "(empty)"
                result.append(clean)
                continue

            if isinstance(content, list):
                filtered = [
                    item for item in content
                    if not (
                        isinstance(item, dict)
                        and item.get("type") in ("text", "input_text", "output_text")
                        and not item.get("text")
                    )
                ]
                if len(filtered) != len(content):
                    clean = dict(msg)
                    if filtered:
                        clean["content"] = filtered
                    elif msg.get("role") == "assistant" and msg.get("tool_calls"):
                        clean["content"] = None
                    else:
                        clean["content"] = "(empty)"
                    result.append(clean)
                    continue

            result.append(msg)
        return result
    
    @abstractmethod
    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
    ) -> LLMResponse:
        """
        Send a chat completion request.
        
        Args:
            messages: List of message dicts with 'role' and 'content'.
            tools: Optional list of tool definitions.
            model: Model identifier (provider-specific).
            max_tokens: Maximum tokens in response.
            temperature: Sampling temperature.
        
        Returns:
            LLMResponse with content and/or tool calls.
        """
        pass
    
    @abstractmethod
    def get_default_model(self) -> str:
        """Get the default model for this provider."""
        pass
