"""Session compaction for context management.

Compaction flow:
1. Pick the keep-window by token budget (see _select_keep_split); everything
   older gets summarized
2. Call LLM to generate summary (staged if needed for large conversations)
3. Insert compaction marker into session
4. On load, get_history() skips messages before marker and injects summary
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Any

from loguru import logger

from bubbles.session.manager import CONTEXT_EXCLUDED_KEY

if TYPE_CHECKING:
    from bubbles.providers.base import LLMProvider
    from bubbles.session.manager import Session

# Compaction prompts
COMPACTION_SYSTEM_PROMPT = "You are a conversation summarizer. Generate a concise summary that captures all important context needed to continue the conversation."

COMPACTION_USER_PROMPT = """Summarize the following conversation. Focus on:
1. What the user asked for
2. What was accomplished
3. Key decisions made
4. Any pending tasks or issues
5. Important context for continuing

Keep the summary concise but complete. Use bullet points for clarity.

{focus_section}

## Conversation to summarize:
{conversation}
"""

COMPACTION_FOCUS_TEMPLATE = """## Current objective (reference only)
The messages below are still present verbatim after compaction. Use them only
to decide which historical details matter; do not treat them as history being
replaced and do not restate them unnecessarily.

{focus}
"""

# Constants
SUMMARY_MAX_TOKENS = 1024
FALLBACK_SUMMARY_TEMPLATE = """[Auto-compacted due to context overflow]

Session contained {msg_count} messages (~{token_count} tokens).
Recent activity preserved. Earlier context was truncated.

If you need information from earlier in the conversation, please ask the user to clarify.
"""


SAFETY_MARGIN = 1.2  # 20% buffer for estimation inaccuracy
TOKENS_PER_IMAGE = 1000  # Approximate tokens per image (medium resolution)

# 保留窗口按 token 预算选取，不按固定条数。见 _select_keep_split。
KEEP_MAX_TOKENS = 40_000

# Token estimation ratios - CONSERVATIVE to avoid context overflow
# Actual values are lower, but we overestimate for safety
CJK_TOKENS_PER_CHAR = 2.0  # CJK: actual ~1.5, using 2.0 for safety
OTHER_TOKENS_PER_CHAR = 0.35  # ASCII: actual ~0.25, using 0.35 for safety


def _is_high_token_char(char: str) -> bool:
    """Check if a character likely uses more tokens (non-ASCII)."""
    return ord(char) > 0x7F


def estimate_tokens(text: str, with_margin: bool = False) -> int:
    """
    Estimate token count with conservative non-ASCII handling.

    Uses different ratios:
    - Non-ASCII (CJK, emoji, symbols): ~2.0 tokens per char (conservative)
    - ASCII: ~0.35 tokens per char (conservative)
    """
    if not text:
        return 0

    high_token_count = sum(1 for c in text if _is_high_token_char(c))
    ascii_count = len(text) - high_token_count

    base = high_token_count * CJK_TOKENS_PER_CHAR + ascii_count * OTHER_TOKENS_PER_CHAR

    if with_margin:
        base *= SAFETY_MARGIN

    return max(0, round(base))


def estimate_message_tokens(msg: dict[str, Any]) -> int:
    """Estimate tokens for a single message, including images."""
    content = msg.get("content", "")
    total = 0
    if isinstance(content, list):
        for block in content:
            if isinstance(block, dict):
                if block.get("type") == "image_url":
                    total += TOKENS_PER_IMAGE
                elif block.get("type") == "text":
                    total += estimate_tokens(block.get("text", ""))
    else:
        total += estimate_tokens(str(content))

    # Tool arguments are part of the provider request even when assistant
    # content is None.  Ignoring them can undercount a generated patch or a
    # nested JSON payload by tens of thousands of tokens.
    for key in ("tool_calls", "tool_call_id", "name", "reasoning_content"):
        value = msg.get(key)
        if value:
            if isinstance(value, str):
                total += estimate_tokens(value)
            else:
                total += estimate_tokens(json.dumps(value, ensure_ascii=False, default=str))
    return total


def estimate_messages_tokens(messages: list[dict[str, Any]]) -> int:
    """Estimate tokens for a list of messages, including images."""
    return sum(estimate_message_tokens(m) for m in messages)


@dataclass
class CompactionResult:
    """Result of a compaction operation."""
    success: bool
    summary: str | None = None
    first_kept_index: int = 0
    tokens_before: int = 0
    tokens_after: int = 0
    messages_compacted: int = 0
    error: str | None = None
    used_fallback: bool = False


def is_compaction_marker(message: dict[str, Any]) -> bool:
    """Check if a message is a compaction marker."""
    return message.get("_type") == "compaction"


def create_compaction_marker(
    summary: str,
    first_kept_index: int,
    tokens_before: int,
    used_fallback: bool = False,
) -> dict[str, Any]:
    """Create a compaction marker to insert into session."""
    return {
        "_type": "compaction",
        "timestamp": datetime.now().isoformat(),
        "summary": summary,
        "first_kept_index": first_kept_index,
        "tokens_before": tokens_before,
        "used_fallback": used_fallback,
    }


def find_last_compaction_index(messages: list[dict[str, Any]]) -> int:
    """Find the index of the last compaction marker, or -1 if none."""
    for i in range(len(messages) - 1, -1, -1):
        if is_compaction_marker(messages[i]):
            return i
    return -1


def _select_keep_split(
    active: list[dict[str, Any]],
    keep_max_tokens: int = KEEP_MAX_TOKENS,
) -> int:
    """选出保留窗口的起点下标，按 token 预算而非固定条数。

    Why: 固定 "最后 N 条" 在长工具输出下会失效——20 条 exec 日志可能有 600k
    token，压完仍然超上限，而下一次 compaction 又会因条数不足被拒绝，会话
    卡死在超限状态。按 token 选取让 "几轮长工具调用" 和 "数十轮短对话" 收敛
    到同一个上下文预算：前者可能只留 2 轮，后者能留几十轮。

    规则（从最新往回累加）：
    - ``keep_max_tokens`` 是严格预算上限；
    - 只在 user 边界记录候选起点，保证 to_keep 由完整 turn 组成；
    - 如果最近一个已完成 turn 自身就超预算，则不保留它。当前目标和执行进展
      由 TurnState 单独管理，不需要靠已落盘历史窗口兜底。

    返回 ``active`` 中保留窗口的起始下标（0 表示全部保留）。
    """
    if not active or keep_max_tokens <= 0:
        return len(active)

    tokens = 0
    start = len(active)

    for i in range(len(active) - 1, -1, -1):
        tokens += estimate_message_tokens(active[i])
        if active[i].get("role") not in ("user", "system"):
            continue
        if tokens > keep_max_tokens:
            break
        start = i

    return start


def _align_split_to_user_boundary(
    to_summarize: list[dict[str, Any]],
    to_keep: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """切分点对齐到 turn 边界：to_keep 必须以 user/system 消息开头。

    Why: 原按 "最后 N 条" 硬切，可能让 to_keep 起头是孤立 tool_result，或
    assistant.tool_calls 的部分 tool_result 留在 to_summarize、其余在 to_keep——
    compaction marker 之后的历史返给 provider 时会因 tool block 不配对而 400。
    """
    while to_keep and to_keep[0].get("role") not in ("user", "system"):
        to_summarize.append(to_keep.pop(0))
    return to_summarize, to_keep


def format_message_for_summary(
    msg: dict[str, Any],
    max_content_len: int | None = 1500,
) -> str | None:
    """Format a single message for summarization."""
    if is_compaction_marker(msg):
        return None

    role = msg.get("role", "unknown").upper()
    raw_content = msg.get("content", "")
    parts: list[str] = []

    if isinstance(raw_content, list):
        for block in raw_content:
            if not isinstance(block, dict):
                continue
            if block.get("type") == "image_url":
                parts.append("[image content preserved outside this summary]")
            elif text := block.get("text"):
                parts.append(str(text))
    elif raw_content:
        parts.append(str(raw_content))

    for call in msg.get("tool_calls", []) or []:
        function = call.get("function") or {}
        parts.append(
            "Tool call "
            f"{function.get('name') or 'unknown'}({function.get('arguments') or '{}'})"
        )

    if msg.get("role") == "tool" and msg.get("name"):
        parts.insert(0, f"Tool result from {msg['name']}:")

    if not parts:
        return None

    content = "\n".join(parts)

    if max_content_len is not None and len(content) > max_content_len:
        content = content[:max_content_len] + "...(truncated)"

    return f"[{role}]: {content}"


def format_messages_for_summary(
    messages: list[dict[str, Any]],
    max_content_len: int | None = 1500,
) -> str:
    """Format messages into text for summarization."""
    lines = []
    for m in messages:
        line = format_message_for_summary(m, max_content_len=max_content_len)
        if line:
            lines.append(line)
    return "\n\n".join(lines)


def _format_focus_messages(messages: list[dict[str, Any]] | None) -> str:
    if not messages:
        return ""
    # Focus is guidance, not another unbounded copy of the active turn.
    return format_messages_for_summary(messages, max_content_len=1_000)[:4_000]


async def summarize_messages(
    messages: list[dict[str, Any]],
    provider: LLMProvider,
    model: str,
    required_prefix: list[dict[str, Any]] | None = None,
    focus_messages: list[dict[str, Any]] | None = None,
) -> str | None:
    """Summarize messages into a concise summary."""
    conversation_text = format_messages_for_summary(messages)
    if required_prefix:
        prefix_text = format_messages_for_summary(
            required_prefix,
            max_content_len=None,
        )
        conversation_text = "\n\n".join(
            part for part in (prefix_text, conversation_text) if part.strip()
        )
    if not conversation_text.strip():
        return None

    focus_section = ""
    focus = _format_focus_messages(focus_messages)
    if focus.strip():
        focus_section = COMPACTION_FOCUS_TEMPLATE.format(focus=focus)

    try:
        response = await provider.chat(
            messages=[
                {"role": "system", "content": COMPACTION_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": COMPACTION_USER_PROMPT.format(
                        focus_section=focus_section,
                        conversation=conversation_text,
                    ),
                },
            ],
            model=model,
            max_tokens=SUMMARY_MAX_TOKENS,
            temperature=0.3,
        )
        return response.content or None
    except Exception as e:
        logger.warning("Failed to summarize messages: {}", e)
        return None


def truncate_messages_to_token_limit(
    messages: list[dict[str, Any]],
    max_tokens: int,
) -> list[dict[str, Any]]:
    """Truncate messages from the front to fit within token limit.

    Scans from newest to oldest, stopping when:
    - Token limit is reached, OR
    - A compaction marker is encountered (don't cross previous compaction boundary)
    """
    if not messages:
        return []

    result: list[dict[str, Any]] = []
    total_tokens = 0

    # Scan from newest to oldest
    for msg in reversed(messages):
        # Stop at compaction marker (don't cross previous boundary)
        if is_compaction_marker(msg):
            break

        msg_tokens = estimate_message_tokens(msg)

        if total_tokens + msg_tokens > max_tokens:
            break

        result.append(msg)
        total_tokens += msg_tokens

    # Reverse to restore chronological order
    result.reverse()
    return result


def _truncate_messages_for_summary(
    messages: list[dict[str, Any]],
    max_tokens: int,
) -> list[dict[str, Any]]:
    """Keep the newest messages whose bounded textual rendering fits."""
    result: list[dict[str, Any]] = []
    total_tokens = 0
    for message in reversed(messages):
        rendered = format_message_for_summary(message)
        if not rendered:
            continue
        message_tokens = estimate_tokens(rendered)
        if total_tokens + message_tokens > max_tokens:
            break
        result.append(message)
        total_tokens += message_tokens
    result.reverse()
    return result


async def summarize_with_truncation(
    messages: list[dict[str, Any]],
    provider: LLMProvider,
    model: str,
    context_limit: int,
    required_prefix: list[dict[str, Any]] | None = None,
    focus_messages: list[dict[str, Any]] | None = None,
) -> str | None:
    """Summarize messages, truncating older ones if exceeding context_limit.

    ``required_prefix`` is never discarded.  It is used for the previous
    compaction summary: a later compaction must fold that summary into the new
    one, otherwise every second compaction forgets everything before the last
    marker.
    """
    prefix = list(required_prefix or [])
    prefix_tokens = estimate_messages_tokens(prefix)
    focus_tokens = estimate_tokens(_format_focus_messages(focus_messages))
    available_tokens = max(0, context_limit - prefix_tokens - focus_tokens)
    total_tokens = estimate_tokens(format_messages_for_summary(messages))

    # Truncate if too large (only summarize recent portion within context_limit)
    if total_tokens > available_tokens:
        original_count = len(messages)
        messages = _truncate_messages_for_summary(messages, available_tokens)
        logger.info(
            "Truncated {} messages to {} for summarization (context_limit: {})",
            original_count, len(messages), context_limit
        )

    return await summarize_messages(
        messages,
        provider,
        model,
        required_prefix=prefix,
        focus_messages=focus_messages,
    )


def create_fallback_summary(
    messages: list[dict[str, Any]],
    tokens_before: int,
    previous_summary: str | None = None,
) -> str:
    """Create a minimal fallback summary when LLM summarization fails."""
    fallback = FALLBACK_SUMMARY_TEMPLATE.format(
        msg_count=len(messages),
        token_count=tokens_before,
    )
    if previous_summary:
        return f"{previous_summary}\n\n{fallback}"
    return fallback


async def compact_session(
    session: Session,
    provider: LLMProvider,
    model: str,
    context_limit: int,
    keep_max_tokens: int = KEEP_MAX_TOKENS,
    min_messages_to_compact: int = 5,
    use_fallback_on_failure: bool = True,
    focus_messages: list[dict[str, Any]] | None = None,
) -> CompactionResult:
    """Compact a session by summarizing old messages.

    Args:
        session: The session to compact
        provider: LLM provider for generating summary
        model: Model to use for summarization
        context_limit: Max tokens to process (older messages truncated)
        keep_max_tokens: Token budget for the keep-window (see _select_keep_split)
        min_messages_to_compact: Minimum messages required to trigger compaction
        use_fallback_on_failure: If True, use fallback summary when LLM fails
        focus_messages: Protected current objectives used only to guide the summary

    Returns:
        CompactionResult with summary and metadata
    """
    messages = session.messages

    # Find last compaction point.  The previous summary is part of the effective
    # history even though it is stored in a marker rather than a provider message.
    last_compaction_idx = find_last_compaction_index(messages)
    start_idx = last_compaction_idx + 1 if last_compaction_idx >= 0 else 0
    previous_summary = (
        str(messages[last_compaction_idx].get("summary", ""))
        if last_compaction_idx >= 0
        else ""
    )

    # Build the logical provider history while retaining physical indices. Raw
    # tool records excluded by active-turn compaction remain in session.jsonl,
    # but must never be summarized or sent again.
    active_entries = [
        (index, message)
        for index, message in enumerate(messages[start_idx:], start=start_idx)
        if not is_compaction_marker(message)
        and not message.get(CONTEXT_EXCLUDED_KEY)
    ]
    active_messages = [message for _, message in active_entries]

    if len(active_messages) < min_messages_to_compact:
        return CompactionResult(
            success=False,
            error=f"Not enough messages to compact (have {len(active_messages)}, need {min_messages_to_compact})"
        )

    # 保留窗口按 token 选取（不是固定条数），见 _select_keep_split。
    split = _select_keep_split(active_messages, keep_max_tokens=keep_max_tokens)
    to_summarize = active_messages[:split]
    to_keep = active_messages[split:]

    # 切分点对齐到 turn 边界，避免 to_keep 起头是孤立 tool block。
    to_summarize, to_keep = _align_split_to_user_boundary(to_summarize, to_keep)

    if len(to_summarize) < min_messages_to_compact:
        return CompactionResult(
            success=False,
            error=f"Not enough messages to summarize (have {len(to_summarize)}, need {min_messages_to_compact})"
        )

    # Count the effective provider history, not the append-only physical prefix
    # before the last marker (get_history() does not send that prefix).
    tokens_before = estimate_tokens(previous_summary) + estimate_messages_tokens(active_messages)

    # Try staged summarization
    summary = None
    used_fallback = False

    try:
        required_prefix = []
        if previous_summary:
            required_prefix.append({
                "role": "system",
                "content": f"Previous compacted summary:\n{previous_summary}",
            })
        summary = await summarize_with_truncation(
            to_summarize,
            provider,
            model,
            context_limit,
            required_prefix=required_prefix,
            focus_messages=focus_messages,
        )
    except Exception as e:
        logger.exception("Staged summarization failed: {}", e)

    # Fallback if summarization failed
    if not summary:
        if use_fallback_on_failure:
            logger.warning("Using fallback summary for session {}", session.key)
            summary = create_fallback_summary(
                to_summarize,
                tokens_before,
                previous_summary=previous_summary or None,
            )
            used_fallback = True
        else:
            return CompactionResult(
                success=False,
                error="Summarization failed and fallback disabled"
            )

    # Create the marker at the physical boundary corresponding to the logical
    # keep-window.  The physical suffix may also contain provider-invisible raw
    # audit records; preserve those in place instead of dropping them.
    logical_keep_start = len(active_messages) - len(to_keep)
    first_kept_index = (
        active_entries[logical_keep_start][0]
        if to_keep
        else len(messages)
    )
    marker = create_compaction_marker(
        summary=summary,
        first_kept_index=first_kept_index,
        tokens_before=tokens_before,
        used_fallback=used_fallback,
    )

    tokens_after = estimate_tokens(summary) + estimate_messages_tokens(to_keep)

    # A compaction that does not make the effective history smaller must not be
    # committed.  Retrying the same oversized request after such a "success"
    # only creates an overflow loop and can replace useful history with a larger
    # fallback marker.
    if tokens_after >= tokens_before:
        return CompactionResult(
            success=False,
            tokens_before=tokens_before,
            tokens_after=tokens_before,
            error=(
                "Compaction made no progress "
                f"({tokens_before} -> {tokens_after} estimated tokens)"
            ),
            used_fallback=used_fallback,
        )

    # Commit only after summarization and the no-progress check have completed.
    session.messages = messages[:first_kept_index] + [marker] + messages[first_kept_index:]
    session.updated_at = datetime.now()

    logger.info(
        "Compacted session {}: {} messages -> {} messages, {} -> {} tokens{}",
        session.key,
        len(messages),
        len(session.messages),
        tokens_before,
        tokens_after,
        " (fallback)" if used_fallback else "",
    )

    return CompactionResult(
        success=True,
        summary=summary,
        first_kept_index=first_kept_index,
        tokens_before=tokens_before,
        tokens_after=tokens_after,
        messages_compacted=len(to_summarize),
        used_fallback=used_fallback,
    )
