"""Turn-level helpers for AgentLoop: persist new messages, decide / run compaction.

Pulled out so AgentLoop only orchestrates; the actual compaction policy and
persistence are testable in isolation.
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Any, Awaitable, Callable

from loguru import logger

from bubbles.agent.bindings import get_bindings_for_session
from bubbles.agent.commands import build_heartbeat_info
from bubbles.agent.compaction import (
    CompactionResult,
    compact_session,
    estimate_messages_tokens,
)
from bubbles.bus.events import InboundMessage, OutboundMessage
from bubbles.session.manager import Session, prune_old_images_inplace

if TYPE_CHECKING:
    from bubbles.agent.loop import AgentLoop


async def process_system_message(
    loop: "AgentLoop",
    msg: InboundMessage,
    on_tool_call: Callable[[str, dict, str | None], Awaitable[None]] | None = None,
) -> OutboundMessage | None:
    """Run a `channel == "system"` turn (cron / subagent fan-out / etc).

    `msg.chat_id` may be either `"channel:chat_id"` (standard) or a bare session
    key — in the latter case we look up the first channel binding for that
    session to find a reply target. Returns None if no binding exists.
    """
    if ":" in msg.chat_id:
        channel, chat_id = msg.chat_id.split(":", 1)
        key = f"{channel}:{chat_id}"
    else:
        key = msg.chat_id
        bindings = get_bindings_for_session(loop._session_bindings, key)
        if not bindings:
            logger.warning("No binding found for session {}, cannot route reply", key)
            return None
        channel, chat_id = bindings[0].split(":", 1)

    logger.info("Processing system message from {} to session {}", msg.sender_id, key)
    session = loop.sessions.get_or_create(key)
    prune_old_images_inplace(session.messages)
    context = loop._get_context(session)
    loop._set_tool_context(
        channel, chat_id, msg.metadata.get("message_id"),
        session.directory, key, session,
    )
    history = session.get_history(max_messages=loop.memory_window)
    messages = context.build_messages(
        history=history,
        current_message=msg.content, channel=channel, chat_id=chat_id,
        sender_id=msg.sender_id,
        sender_name=msg.metadata.get("sender_name"),
        system_prompt_extra=session.config.system_prompt,
        session_bindings=get_bindings_for_session(loop._session_bindings, session.key),
        heartbeat_info=build_heartbeat_info(loop.cron_service, session.key),
    )
    final_content, _, all_msgs = await loop._run_agent_loop(
        messages, session=session, on_tool_call=on_tool_call,
    )
    save_turn(session, all_msgs, 1 + len(history))
    loop.sessions.save(session)
    return OutboundMessage(
        channel=channel, chat_id=chat_id,
        content=final_content or "Background task completed.",
    )


def save_turn(session: Session, messages: list[dict], skip: int) -> None:
    """Append new-turn messages (after the pre-existing prefix) into session."""
    for m in messages[skip:]:
        entry = {k: v for k, v in m.items() if k != "reasoning_content"}
        entry.setdefault("timestamp", datetime.now().isoformat())
        session.messages.append(entry)
    session.updated_at = datetime.now()


def should_compact(loop: "AgentLoop", messages: list[dict[str, Any]]) -> bool:
    """Decide if messages exceed the compact threshold based on token estimation."""
    estimated = estimate_messages_tokens(messages)
    usable = loop.context_limit - loop.max_tokens
    return estimated > usable * loop.compact_threshold


async def do_compact(loop: "AgentLoop", session: Session) -> CompactionResult:
    """Compact session history using LLM-powered summarization."""
    return await compact_session(
        session=session,
        provider=loop.provider,
        model=loop.model,
        context_limit=loop.context_limit,
        keep_recent=loop.compact_keep_recent,
        min_messages_to_compact=loop.compact_min_messages,
        use_fallback_on_failure=True,
    )


async def mid_loop_compact(
    loop: "AgentLoop",
    session: Session,
    current_messages: list[dict[str, Any]],
    on_progress: Callable[..., Awaitable[None]] | None = None,
) -> list[dict[str, Any]]:
    """Execute compaction in the middle of a running loop and return rebuilt messages."""
    logger.info("Mid-loop compaction triggered for session {}", session.key)

    # 1. Save current progress to session (skip system messages)
    first_non_system = 0
    for i, m in enumerate(current_messages):
        if m.get("role") != "system":
            first_non_system = i
            break

    history_len = len(session.get_history(max_messages=loop.memory_window))
    new_messages = current_messages[first_non_system + history_len:]
    for m in new_messages:
        if m.get("role") != "system":
            session.messages.append(m)

    # 2. Execute compaction
    try:
        result = await do_compact(loop, session)
        if result.success:
            msg = (
                "Context overflow, history compacted"
                if not result.used_fallback
                else "Context overflow, history truncated"
            )
            if on_progress:
                await on_progress(
                    f"{msg} ({result.tokens_before} → {result.tokens_after} tokens)"
                )
            logger.info(
                "Mid-loop compaction successful: {} -> {} tokens",
                result.tokens_before, result.tokens_after,
            )
            loop.sessions.save(session)
        else:
            logger.warning("Mid-loop compaction skipped: {}", result.error)
    except Exception as e:
        logger.exception("Mid-loop compaction failed: {}", e)

    # 3. Rebuild: system prompt + new history + current query
    new_history = session.get_history(max_messages=loop.memory_window)

    current_query = None
    for m in reversed(current_messages):
        if m.get("role") == "user":
            current_query = m.get("content", "")
            break

    context = loop._get_context(session)
    return context.build_messages(
        history=new_history,
        current_message=current_query or "",
        channel="cli",
        chat_id="direct",
        heartbeat_info=build_heartbeat_info(loop.cron_service, session.key),
    )
