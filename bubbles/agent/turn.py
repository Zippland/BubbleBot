"""Turn-level helpers for AgentLoop: persist new messages, decide / run compaction.

Pulled out so AgentLoop only orchestrates; the actual compaction policy and
persistence are testable in isolation.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any, Awaitable, Callable

from loguru import logger

from bubbles.agent.bindings import get_bindings_for_session
from bubbles.agent.commands import build_heartbeat_info
from bubbles.agent.compaction import (
    SUMMARY_MAX_TOKENS,
    CompactionResult,
    compact_session,
    estimate_message_tokens,
    estimate_messages_tokens,
    summarize_with_truncation,
)
from bubbles.bus.events import InboundMessage, OutboundMessage
from bubbles.session.manager import (
    CONTEXT_EXCLUDED_KEY,
    Session,
    _sanitize_for_api,
    prune_old_images_inplace,
)

if TYPE_CHECKING:
    from bubbles.agent.loop import AgentLoop


@dataclass
class TurnState:
    """The non-persisted state of one running turn.

    ``messages`` is the full durable transcript. ``context_messages`` is its
    live provider projection and may replace completed tool groups with
    ``compacted_summary``. Keeping both explicit prevents compaction from
    guessing turn boundaries and preserves the verbatim log for persistence.
    """

    system_prefix: list[dict[str, Any]]
    messages: list[dict[str, Any]]
    context_messages: list[dict[str, Any]] | None = None
    compacted_summary: str | None = None
    compacted_message_ids: set[int] = field(default_factory=set, repr=False)
    context_excluded_message_ids: set[int] = field(default_factory=set, repr=False)
    suppress_outbound: bool = False

    def __post_init__(self) -> None:
        if self.context_messages is None:
            self.context_messages = list(self.messages)

    @classmethod
    def from_context_messages(
        cls,
        messages: list[dict[str, Any]],
        history_length: int,
    ) -> "TurnState":
        """Split ContextBuilder output at its known construction boundary."""
        if not messages or messages[0].get("role") != "system":
            raise ValueError("Context messages must start with the base system prompt")
        active_start = 1 + history_length
        if active_start > len(messages):
            raise ValueError("History length exceeds the built context")
        return cls(
            system_prefix=list(messages[:1]),
            messages=list(messages[active_start:]),
        )

    def rebuild(self, session: Session, memory_window: int) -> list[dict[str, Any]]:
        """Build provider input without recreating or reinjecting the user message."""
        summary_messages: list[dict[str, Any]] = []
        if self.compacted_summary:
            summary_messages.append(self.summary_message(self.compacted_summary))
        return [
            *self.system_prefix,
            *session.get_history(max_messages=memory_window),
            *summary_messages,
            *(self.context_messages or []),
        ]

    def capture_appended(
        self,
        provider_messages: list[dict[str, Any]],
        start: int,
    ) -> None:
        """Record newly appended provider messages in durable and live views."""
        appended = provider_messages[start:]
        self.messages.extend(appended)
        assert self.context_messages is not None
        self.context_messages.extend(appended)

    def focus_messages(self) -> list[dict[str, Any]]:
        """Current instructions that guide compaction but remain verbatim."""
        return [m for m in self.messages if m.get("role") in ("user", "system")]

    def persistence_messages(self) -> list[dict[str, Any]]:
        """Return one durable sequence containing both projection and raw audit.

        The progress summary is placed where the live provider projection puts
        it.  Compacted tool records remain verbatim but are tagged so history
        construction and future compaction skip them.
        """
        persisted: list[dict[str, Any]] = []
        if self.compacted_summary:
            persisted.append(self.summary_message(self.compacted_summary))
        for message in self.messages:
            if (
                id(message) in self.compacted_message_ids
                or id(message) in self.context_excluded_message_ids
            ):
                persisted.append({**message, CONTEXT_EXCLUDED_KEY: True})
            else:
                persisted.append(message)
        return persisted

    @staticmethod
    def summary_message(summary: str) -> dict[str, Any]:
        return {
            "role": "system",
            "content": f"## Current turn progress summary\n{summary}",
        }


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
    sandbox = await loop._sandboxes.get(key, session.directory, session.config.sandbox)
    turn_tools = loop.build_turn_tools(
        channel=channel, chat_id=chat_id,
        message_id=msg.metadata.get("message_id"),
        session_dir=session.directory, session_key=key,
        session=session, sandbox=sandbox,
        system_triggered=True,
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
        work_dir=sandbox.root,
    )
    turn_state = TurnState.from_context_messages(messages, len(history))
    try:
        final_content, _, _ = await loop._run_agent_loop(
            messages,
            session=session,
            on_tool_call=on_tool_call,
            tools=turn_tools,
            turn_state=turn_state,
        )
    except BaseException:
        try:
            persist_failed_turn(session, turn_state)
            loop.sessions.save(session)
        except Exception as persist_error:
            logger.exception("Failed to persist interrupted turn: {}", persist_error)
        raise
    persist_turn_state(session, turn_state)
    loop.sessions.save(session)
    if turn_state.suppress_outbound:
        logger.info("System turn ended silently for session {}", key)
        return None
    return OutboundMessage(
        channel=channel, chat_id=chat_id,
        content=final_content or "Background task completed.",
    )


def persist_turn_messages(session: Session, messages: list[dict[str, Any]]) -> None:
    """Persist one completed turn exactly once, without provider-only reasoning."""
    for m in messages:
        entry = {k: v for k, v in m.items() if k != "reasoning_content"}
        entry.setdefault("timestamp", datetime.now().isoformat())
        session.messages.append(entry)
    session.updated_at = datetime.now()


def persist_turn_state(session: Session, turn_state: TurnState) -> None:
    """Persist the active turn's provider projection plus its raw audit data."""
    persist_turn_messages(session, turn_state.persistence_messages())


def persist_failed_turn(session: Session, turn_state: TurnState) -> None:
    """Persist the user input and only protocol-complete progress on failure."""
    persist_turn_messages(session, _sanitize_for_api(turn_state.persistence_messages()))


def should_compact(
    loop: "AgentLoop",
    messages: list[dict[str, Any]],
    session: Session | None = None,
) -> bool:
    """Decide if messages exceed the compact threshold based on token estimation."""
    estimated = estimate_messages_tokens(messages)
    max_tokens = (
        session.config.max_tokens
        if session and session.config.max_tokens is not None
        else loop.max_tokens
    )
    usable = loop.context_limit - max_tokens
    return estimated > usable * loop.compact_threshold


async def do_compact(
    loop: "AgentLoop",
    session: Session,
    *,
    keep_max_tokens: int | None = None,
    min_messages_to_compact: int | None = None,
    focus_messages: list[dict[str, Any]] | None = None,
) -> CompactionResult:
    """Compact session history using LLM-powered summarization."""
    model = session.config.model or loop.model
    return await compact_session(
        session=session,
        provider=loop._provider_for(model),
        model=model,
        context_limit=loop.context_limit,
        keep_max_tokens=(
            loop.compact_keep_max_tokens
            if keep_max_tokens is None
            else keep_max_tokens
        ),
        min_messages_to_compact=(
            loop.compact_min_messages
            if min_messages_to_compact is None
            else min_messages_to_compact
        ),
        use_fallback_on_failure=True,
        focus_messages=focus_messages,
    )


def _message_has_image(message: dict[str, Any]) -> bool:
    content = message.get("content")
    return isinstance(content, list) and any(
        isinstance(block, dict) and block.get("type") == "image_url"
        for block in content
    )


def _completed_tool_groups(messages: list[dict[str, Any]]) -> list[tuple[int, int]]:
    """Return complete, text-only assistant/tool groups as half-open ranges."""
    groups: list[tuple[int, int]] = []
    i = 0
    while i < len(messages):
        message = messages[i]
        if message.get("role") != "assistant" or not message.get("tool_calls"):
            i += 1
            continue

        expected = {
            call.get("id")
            for call in message.get("tool_calls", [])
            if call.get("id") is not None
        }
        j = i + 1
        seen: set[str] = set()
        valid = bool(expected) and not _message_has_image(message)
        while j < len(messages) and messages[j].get("role") == "tool":
            tool_id = messages[j].get("tool_call_id")
            if tool_id not in expected or _message_has_image(messages[j]):
                valid = False
            elif tool_id is not None:
                seen.add(tool_id)
            j += 1

        if valid and seen == expected:
            groups.append((i, j))
        i = max(j, i + 1)
    return groups


async def _compact_active_turn(
    loop: "AgentLoop",
    session: Session,
    turn_state: TurnState,
    keep_max_tokens: int,
) -> CompactionResult:
    """Summarize completed tool groups while keeping instructions verbatim."""
    context_messages = list(turn_state.context_messages or [])
    groups = _completed_tool_groups(context_messages)
    if not groups:
        return CompactionResult(success=False, error="No completed active tool groups to compact")

    kept_tokens = 0
    summarize_group_count = len(groups)
    for pos in range(len(groups) - 1, -1, -1):
        start, end = groups[pos]
        group_tokens = estimate_messages_tokens(context_messages[start:end])
        if kept_tokens + group_tokens > keep_max_tokens:
            break
        kept_tokens += group_tokens
        summarize_group_count = pos

    ranges = groups[:summarize_group_count]
    if not ranges:
        return CompactionResult(success=False, error="Active tool groups already fit keep budget")

    compacted_indices = {
        index
        for start, end in ranges
        for index in range(start, end)
    }
    to_summarize = [
        message
        for index, message in enumerate(context_messages)
        if index in compacted_indices
    ]
    required_prefix = []
    if turn_state.compacted_summary:
        required_prefix.append({
            "role": "system",
            "content": (
                "Previous current-turn progress summary:\n"
                f"{turn_state.compacted_summary}"
            ),
        })

    # Keep model/provider routing identical to the main turn.
    model = session.config.model or loop.model
    summary = await summarize_with_truncation(
        to_summarize,
        loop._provider_for(model),
        model,
        loop.context_limit,
        required_prefix=required_prefix,
        focus_messages=turn_state.focus_messages(),
    )
    if not summary:
        return CompactionResult(success=False, error="Failed to summarize active tool progress")

    candidate_context = [
        message
        for index, message in enumerate(context_messages)
        if index not in compacted_indices
    ]
    before_tokens = estimate_messages_tokens(context_messages)
    if turn_state.compacted_summary:
        before_tokens += estimate_message_tokens(
            TurnState.summary_message(turn_state.compacted_summary)
        )
    after_tokens = estimate_messages_tokens(candidate_context)
    after_tokens += estimate_message_tokens(TurnState.summary_message(summary))
    if after_tokens >= before_tokens:
        return CompactionResult(
            success=False,
            tokens_before=before_tokens,
            tokens_after=before_tokens,
            error=f"Active compaction made no progress ({before_tokens} -> {after_tokens})",
        )

    turn_state.context_messages = candidate_context
    turn_state.compacted_summary = summary
    turn_state.compacted_message_ids.update(id(message) for message in to_summarize)
    return CompactionResult(
        success=True,
        summary=summary,
        tokens_before=before_tokens,
        tokens_after=after_tokens,
        messages_compacted=len(compacted_indices),
    )


async def compact_for_turn(
    loop: "AgentLoop",
    session: Session,
    turn_state: TurnState,
    on_progress: Callable[..., Awaitable[None]] | None = None,
    *,
    force_active: bool = False,
) -> tuple[list[dict[str, Any]], CompactionResult]:
    """Compact old history, then completed active tool progress if necessary."""
    logger.info("Context compaction triggered for session {}", session.key)

    before_messages = turn_state.rebuild(session, loop.memory_window)
    tokens_before = estimate_messages_tokens(before_messages)
    max_tokens = session.config.max_tokens or loop.max_tokens
    target_tokens = max(
        0,
        int((loop.context_limit - max_tokens) * loop.compact_threshold),
    )
    protected_messages = [
        *turn_state.system_prefix,
        *(turn_state.context_messages or []),
    ]
    if turn_state.compacted_summary:
        protected_messages.append(
            TurnState.summary_message(turn_state.compacted_summary)
        )
    protected_tokens = estimate_messages_tokens(protected_messages)
    keep_budget = max(0, target_tokens - protected_tokens - SUMMARY_MAX_TOKENS)
    keep_budget = min(loop.compact_keep_max_tokens, keep_budget)

    original_messages = session.messages
    original_updated_at = session.updated_at
    original_context_messages = list(turn_state.context_messages or [])
    original_turn_summary = turn_state.compacted_summary
    original_compacted_message_ids = set(turn_state.compacted_message_ids)
    try:
        history_result = await do_compact(
            loop,
            session,
            keep_max_tokens=keep_budget,
            # Automatic recovery is already gated by token pressure.  A single
            # giant completed message must be compactable even when the manual
            # /compact command would consider the history too short.
            min_messages_to_compact=1,
            focus_messages=turn_state.focus_messages(),
        )
        rebuilt_after_history = turn_state.rebuild(session, loop.memory_window)

        active_result = CompactionResult(success=False, error="Active compaction not needed")
        if force_active or estimate_messages_tokens(rebuilt_after_history) > target_tokens:
            context_messages = list(turn_state.context_messages or [])
            group_indices = {
                index
                for start, end in _completed_tool_groups(context_messages)
                for index in range(start, end)
            }
            noncompactable_active = [
                message
                for index, message in enumerate(context_messages)
                if index not in group_indices
            ]
            fixed_tokens = estimate_messages_tokens([
                *turn_state.system_prefix,
                *session.get_history(max_messages=loop.memory_window),
                *noncompactable_active,
            ])
            active_keep_budget = max(
                0,
                target_tokens - fixed_tokens - SUMMARY_MAX_TOKENS,
            )
            if force_active:
                # The provider has already rejected this exact transcript, so
                # estimator-based spare budget is not trustworthy.  Compact at
                # least the completed tool progress before the one allowed retry.
                active_keep_budget = 0
            active_result = await _compact_active_turn(
                loop,
                session,
                turn_state,
                keep_max_tokens=active_keep_budget,
            )

        rebuilt = turn_state.rebuild(session, loop.memory_window)
        tokens_after = estimate_messages_tokens(rebuilt)
        if tokens_after >= tokens_before:
            session.messages = original_messages
            session.updated_at = original_updated_at
            turn_state.context_messages = original_context_messages
            turn_state.compacted_summary = original_turn_summary
            turn_state.compacted_message_ids = original_compacted_message_ids
            errors = [
                item.error
                for item in (history_result, active_result)
                if item.error
            ]
            result = CompactionResult(
                success=False,
                tokens_before=tokens_before,
                tokens_after=tokens_before,
                error=(
                    "Compaction did not reduce provider context "
                    f"({tokens_before} -> {tokens_after} estimated tokens): "
                    + "; ".join(errors)
                ),
                used_fallback=history_result.used_fallback,
            )
            logger.warning("Context compaction rolled back: {}", result.error)
            return before_messages, result

        result = CompactionResult(
            success=True,
            summary=active_result.summary or history_result.summary,
            tokens_before=tokens_before,
            tokens_after=tokens_after,
            messages_compacted=(
                history_result.messages_compacted + active_result.messages_compacted
            ),
            used_fallback=history_result.used_fallback,
        )
        if session.messages is not original_messages:
            loop.sessions.save(session)

    except asyncio.CancelledError:
        session.messages = original_messages
        session.updated_at = original_updated_at
        turn_state.context_messages = original_context_messages
        turn_state.compacted_summary = original_turn_summary
        turn_state.compacted_message_ids = original_compacted_message_ids
        raise
    except Exception as e:
        session.messages = original_messages
        session.updated_at = original_updated_at
        turn_state.context_messages = original_context_messages
        turn_state.compacted_summary = original_turn_summary
        turn_state.compacted_message_ids = original_compacted_message_ids
        logger.exception("Context compaction failed: {}", e)
        return before_messages, CompactionResult(
            success=False,
            tokens_before=tokens_before,
            tokens_after=tokens_before,
            error=str(e),
        )

    # Progress delivery is outside the transaction.  A broken channel callback
    # must not roll in-memory history back after the compacted session was
    # already saved successfully.
    msg = (
        "Context compacted"
        if not result.used_fallback
        else "Context truncated"
    )
    if on_progress:
        try:
            await on_progress(
                f"{msg} ({result.tokens_before} → {result.tokens_after} tokens)"
            )
        except Exception as e:
            logger.warning("Failed to publish compaction progress: {}", e)
    logger.info(
        "Context compaction successful: {} -> {} tokens",
        result.tokens_before,
        result.tokens_after,
    )
    return rebuilt, result
