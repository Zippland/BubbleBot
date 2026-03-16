"""Session compaction for context management.

Compaction flow:
1. Collect messages to summarize (before keep_recent threshold)
2. Call LLM to generate summary (staged if needed for large conversations)
3. Insert compaction marker into session
4. On load, get_history() skips messages before marker and injects summary
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, TYPE_CHECKING

from loguru import logger

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

## Conversation to summarize:
{conversation}
"""

# Constants
SUMMARY_MAX_TOKENS = 1024
FALLBACK_SUMMARY_TEMPLATE = """[Auto-compacted due to context overflow]

Session contained {msg_count} messages (~{token_count} tokens).
Recent activity preserved. Earlier context was truncated.

If you need information from earlier in the conversation, please ask the user to clarify.
"""


CHARS_PER_TOKEN = 0.5  # 1 char ≈ 2 tokens (conservative for CJK)
SAFETY_MARGIN = 1.2  # 20% buffer for estimation inaccuracy
TOKENS_PER_IMAGE = 1000  # Approximate tokens per image (medium resolution)


def estimate_tokens(text: str, with_margin: bool = False) -> int:
    """Estimate token count from text. Conservative: 1 char = 2 tokens."""
    base = len(text or "") / CHARS_PER_TOKEN
    if with_margin:
        base *= SAFETY_MARGIN
    return max(0, round(base))


def estimate_message_tokens(msg: dict[str, Any]) -> int:
    """Estimate tokens for a single message, including images."""
    content = msg.get("content", "")
    if isinstance(content, list):
        total = 0
        for block in content:
            if isinstance(block, dict):
                if block.get("type") == "image_url":
                    total += TOKENS_PER_IMAGE
                elif block.get("type") == "text":
                    total += estimate_tokens(block.get("text", ""))
        return total
    return estimate_tokens(str(content))


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


def format_message_for_summary(msg: dict[str, Any], max_content_len: int = 1500) -> str | None:
    """Format a single message for summarization."""
    if is_compaction_marker(msg):
        return None

    role = msg.get("role", "unknown").upper()
    content = msg.get("content", "")

    if isinstance(content, list):
        content = " ".join(
            block.get("text", "") for block in content if isinstance(block, dict)
        )

    if not content:
        return None

    if len(content) > max_content_len:
        content = content[:max_content_len] + "...(truncated)"

    return f"[{role}]: {content}"


def format_messages_for_summary(messages: list[dict[str, Any]]) -> str:
    """Format messages into text for summarization."""
    lines = []
    for m in messages:
        line = format_message_for_summary(m)
        if line:
            lines.append(line)
    return "\n\n".join(lines)


async def summarize_messages(
    messages: list[dict[str, Any]],
    provider: LLMProvider,
    model: str,
) -> str | None:
    """Summarize messages into a concise summary."""
    conversation_text = format_messages_for_summary(messages)
    if not conversation_text.strip():
        return None

    try:
        response = await provider.chat(
            messages=[
                {"role": "system", "content": COMPACTION_SYSTEM_PROMPT},
                {"role": "user", "content": COMPACTION_USER_PROMPT.format(conversation=conversation_text)},
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


async def summarize_with_truncation(
    messages: list[dict[str, Any]],
    provider: LLMProvider,
    model: str,
    context_limit: int,
) -> str | None:
    """Summarize messages, truncating older ones if exceeding context_limit.

    Scans from newest to oldest, keeps messages within context_limit,
    then summarizes in one LLM call.
    """
    total_tokens = estimate_messages_tokens(messages)

    # Truncate if too large (only summarize recent portion within context_limit)
    if total_tokens > context_limit:
        original_count = len(messages)
        messages = truncate_messages_to_token_limit(messages, context_limit)
        logger.info(
            "Truncated {} messages to {} for summarization (context_limit: {})",
            original_count, len(messages), context_limit
        )

    return await summarize_messages(messages, provider, model)


def create_fallback_summary(
    messages: list[dict[str, Any]],
    tokens_before: int,
) -> str:
    """Create a minimal fallback summary when LLM summarization fails."""
    return FALLBACK_SUMMARY_TEMPLATE.format(
        msg_count=len(messages),
        token_count=tokens_before,
    )


async def compact_session(
    session: Session,
    provider: LLMProvider,
    model: str,
    context_limit: int,
    keep_recent: int,
    min_messages_to_compact: int = 5,
    use_fallback_on_failure: bool = True,
) -> CompactionResult:
    """Compact a session by summarizing old messages.

    Args:
        session: The session to compact
        provider: LLM provider for generating summary
        model: Model to use for summarization
        context_limit: Max tokens to process (older messages truncated)
        keep_recent: Number of recent messages to keep (not summarize)
        min_messages_to_compact: Minimum messages required to trigger compaction
        use_fallback_on_failure: If True, use fallback summary when LLM fails

    Returns:
        CompactionResult with summary and metadata
    """
    messages = session.messages

    # Find last compaction point
    last_compaction_idx = find_last_compaction_index(messages)
    start_idx = last_compaction_idx + 1 if last_compaction_idx >= 0 else 0

    # Get messages since last compaction (excluding markers)
    active_messages = [m for m in messages[start_idx:] if not is_compaction_marker(m)]

    if len(active_messages) < min_messages_to_compact + keep_recent:
        return CompactionResult(
            success=False,
            error=f"Not enough messages to compact (have {len(active_messages)}, need {min_messages_to_compact + keep_recent})"
        )

    # Split into messages to summarize and messages to keep
    to_summarize = active_messages[:-keep_recent]
    to_keep = active_messages[-keep_recent:]

    if len(to_summarize) < min_messages_to_compact:
        return CompactionResult(
            success=False,
            error=f"Not enough messages to summarize (have {len(to_summarize)}, need {min_messages_to_compact})"
        )

    # Estimate tokens
    tokens_before = estimate_messages_tokens(messages)

    # Try staged summarization
    summary = None
    used_fallback = False

    try:
        summary = await summarize_with_truncation(to_summarize, provider, model, context_limit)
    except Exception as e:
        logger.exception("Staged summarization failed: {}", e)

    # Fallback if summarization failed
    if not summary:
        if use_fallback_on_failure:
            logger.warning("Using fallback summary for session {}", session.key)
            summary = create_fallback_summary(to_summarize, tokens_before)
            used_fallback = True
        else:
            return CompactionResult(
                success=False,
                error="Summarization failed and fallback disabled"
            )

    # Create compaction marker
    first_kept_index = len(messages) - keep_recent
    marker = create_compaction_marker(
        summary=summary,
        first_kept_index=first_kept_index,
        tokens_before=tokens_before,
        used_fallback=used_fallback,
    )

    # Insert marker before the kept messages
    session.messages = messages[:first_kept_index] + [marker] + to_keep
    session.updated_at = datetime.now()

    tokens_after = estimate_tokens(summary) + estimate_messages_tokens(to_keep)

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
