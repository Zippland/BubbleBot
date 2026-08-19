"""Regression tests for compaction boundaries and active-turn persistence."""

from __future__ import annotations

import asyncio
from copy import deepcopy
from types import SimpleNamespace

import pytest

from bubbles.agent.compaction import (
    COMPACTION_SYSTEM_PROMPT,
    _select_keep_split,
    compact_session,
    estimate_message_tokens,
)
from bubbles.agent.context import ContextBuilder
from bubbles.agent.loop import AgentLoop
from bubbles.agent.turn import TurnState, compact_for_turn, persist_turn_messages
from bubbles.bus.events import InboundMessage
from bubbles.providers.base import (
    LLMCallError,
    LLMErrorKind,
    LLMResponse,
    ToolCallRequest,
)
from bubbles.providers.openai_codex_provider import _convert_messages
from bubbles.session.manager import Session, SessionManager


class _SummaryProvider:
    def __init__(self, summaries: list[str]):
        self.summaries = list(summaries)
        self.calls: list[list[dict]] = []

    async def chat(self, *, messages, **kwargs):
        self.calls.append(messages)
        return SimpleNamespace(content=self.summaries.pop(0))


class _SessionSaver:
    def __init__(self, error: Exception | None = None):
        self.error = error
        self.saved = 0

    def save(self, session):
        if self.error:
            raise self.error
        self.saved += 1


def _compact_loop(provider, *, saver=None):
    loop = SimpleNamespace(
        model="model",
        context_limit=20_000,
        max_tokens=1_000,
        compact_threshold=0.8,
        compact_keep_max_tokens=0,
        compact_min_messages=1,
        memory_window=500,
        sessions=saver or _SessionSaver(),
    )
    loop._provider_for = lambda model: provider
    return loop


def _long(text: str) -> str:
    return (text + " ") * 400


@pytest.mark.asyncio
async def test_compact_for_turn_preserves_current_request_once() -> None:
    provider = _SummaryProvider(["CUMULATIVE-SUMMARY"])
    saver = _SessionSaver()
    loop = _compact_loop(provider, saver=saver)
    session = Session(key="cli:compact")
    session.messages = [
        {"role": "user", "content": _long("oldest")},
        {"role": "assistant", "content": _long("old answer")},
        {"_type": "compaction", "summary": "PREVIOUS-SUMMARY"},
        {"role": "user", "content": _long("recent history")},
        {"role": "assistant", "content": _long("recent answer")},
    ]
    current = "CURRENT-REQUEST\n\n[Runtime Context]\nChannel: feishu"
    active = [
        {"role": "user", "content": current},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [{"id": "t1", "type": "function", "function": {"name": "read", "arguments": "{}"}}],
        },
        {"role": "tool", "tool_call_id": "t1", "name": "read", "content": "TOOL-RESULT"},
    ]
    turn_state = TurnState(
        system_prefix=[{"role": "system", "content": "BASE-SYSTEM"}],
        messages=list(active),
    )

    rebuilt, result = await compact_for_turn(loop, session, turn_state)

    assert result.success
    assert saver.saved == 1
    assert [m["role"] for m in rebuilt] == ["system", "system", "user", "assistant", "tool"]
    rendered = repr(rebuilt)
    assert rendered.count("CURRENT-REQUEST") == 1
    assert rendered.count("[Runtime Context]") == 1
    assert rendered.count("TOOL-RESULT") == 1
    assert turn_state.messages == active
    assert "CURRENT-REQUEST" not in repr(session.messages)
    summary_prompt = provider.calls[0][1]["content"]
    assert "PREVIOUS-SUMMARY" in summary_prompt
    assert "CURRENT-REQUEST" in summary_prompt


@pytest.mark.asyncio
async def test_consecutive_compactions_fold_previous_summary() -> None:
    long_summary = "SUMMARY-ONE-" + ("x" * 2_000) + "-SUMMARY-TAIL"
    provider = _SummaryProvider([long_summary, "SUMMARY-TWO"])
    session = Session(key="cli:twice")
    session.messages = [
        {"role": "user", "content": _long("first request")},
        {"role": "assistant", "content": _long("first answer")},
    ]

    first = await compact_session(
        session=session,
        provider=provider,
        model="model",
        context_limit=20_000,
        keep_max_tokens=0,
        min_messages_to_compact=1,
    )
    assert first.success

    session.messages.extend([
        {"role": "user", "content": _long("second request")},
        {"role": "assistant", "content": _long("second answer")},
    ])
    second = await compact_session(
        session=session,
        provider=provider,
        model="model",
        context_limit=20_000,
        keep_max_tokens=0,
        min_messages_to_compact=1,
    )

    assert second.success
    assert long_summary in provider.calls[1][1]["content"]
    assert "SUMMARY-TAIL" in provider.calls[1][1]["content"]
    assert "SUMMARY-TWO" in session.get_history()[0]["content"]


@pytest.mark.asyncio
async def test_consecutive_active_compactions_fold_progress_summary() -> None:
    provider = _SummaryProvider(["ACTIVE-SUMMARY-ONE", "ACTIVE-SUMMARY-TWO"])
    loop = _compact_loop(provider)
    session = Session(key="cli:active-twice")
    current = {"role": "user", "content": "CURRENT-GOAL"}

    def tool_group(call_id: str, fill: str) -> list[dict]:
        return [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": call_id,
                    "type": "function",
                    "function": {"name": "read", "arguments": "{}"},
                }],
            },
            {
                "role": "tool",
                "tool_call_id": call_id,
                "name": "read",
                "content": fill * 10_000,
            },
        ]

    first_group = tool_group("t1", "A")
    turn_state = TurnState(
        system_prefix=[{"role": "system", "content": "BASE-SYSTEM"}],
        messages=[current, *first_group],
    )
    _, first = await compact_for_turn(
        loop,
        session,
        turn_state,
        force_active=True,
    )
    assert first.success

    second_group = tool_group("t2", "B")
    turn_state.messages.extend(second_group)
    assert turn_state.context_messages is not None
    turn_state.context_messages.extend(second_group)
    rebuilt, second = await compact_for_turn(
        loop,
        session,
        turn_state,
        force_active=True,
    )

    assert second.success
    assert "ACTIVE-SUMMARY-ONE" in provider.calls[1][1]["content"]
    assert "CURRENT-GOAL" in provider.calls[1][1]["content"]
    assert repr(rebuilt).count("ACTIVE-SUMMARY-TWO") == 1
    assert not [message for message in rebuilt if message.get("role") == "tool"]
    assert len([message for message in turn_state.messages if message.get("role") == "tool"]) == 2


def test_keep_budget_is_strict_and_tool_arguments_are_counted() -> None:
    oversized_turn = [
        {"role": "user", "content": "request"},
        {"role": "assistant", "content": _long("answer")},
    ]
    assert _select_keep_split(oversized_turn, keep_max_tokens=1) == len(oversized_turn)

    tool_call = {
        "role": "assistant",
        "content": None,
        "tool_calls": [{
            "id": "large",
            "type": "function",
            "function": {"name": "write", "arguments": "x" * 10_000},
        }],
    }
    assert estimate_message_tokens(tool_call) > 3_000


def test_turn_boundary_uses_built_history_length_with_summary_system() -> None:
    history = [
        {"role": "system", "content": "PREVIOUS-SUMMARY"},
        {"role": "user", "content": "old request"},
        {"role": "assistant", "content": "old answer"},
    ]
    current = {"role": "system", "content": "CURRENT-SUBAGENT-RESULT"}
    built = [{"role": "system", "content": "BASE-SYSTEM"}, *history, current]

    turn_state = TurnState.from_context_messages(built, history_length=len(history))

    assert turn_state.system_prefix == built[:1]
    assert turn_state.messages == [current]


def test_system_triggered_history_survives_reload_projection() -> None:
    session = Session(key="system:history")
    session.messages = [
        {"role": "system", "content": "SUBAGENT-RESULT"},
        {"role": "assistant", "content": "handled"},
    ]

    history = session.get_history()

    assert [message["role"] for message in history] == ["system", "assistant"]
    assert history[0]["content"] == "SUBAGENT-RESULT"


def test_codex_combines_all_system_messages() -> None:
    instructions, items = _convert_messages([
        {"role": "system", "content": "BASE-SYSTEM"},
        {"role": "system", "content": "COMPACTED-HISTORY"},
        {"role": "user", "content": "current"},
    ])

    assert instructions == "BASE-SYSTEM\n\nCOMPACTED-HISTORY"
    assert len(items) == 1


@pytest.mark.asyncio
async def test_no_progress_and_save_failure_do_not_mutate_session() -> None:
    no_progress_provider = _SummaryProvider([_long("larger replacement summary")])
    session = Session(key="cli:no-progress")
    original = [
        {"role": "user", "content": "short request"},
        {"role": "assistant", "content": "short answer"},
    ]
    session.messages = original
    original_updated_at = session.updated_at

    result = await compact_session(
        session=session,
        provider=no_progress_provider,
        model="model",
        context_limit=20_000,
        keep_max_tokens=0,
        min_messages_to_compact=1,
    )

    assert not result.success
    assert session.messages is original
    assert session.updated_at == original_updated_at

    save_failure_provider = _SummaryProvider(["small summary"])
    failing_loop = _compact_loop(
        save_failure_provider,
        saver=_SessionSaver(OSError("disk unavailable")),
    )
    session.messages = [
        {"role": "user", "content": _long("request")},
        {"role": "assistant", "content": _long("answer")},
    ]
    original = session.messages
    turn_state = TurnState(
        system_prefix=[{"role": "system", "content": "system"}],
        messages=[{"role": "user", "content": "current"}],
    )

    _, result = await compact_for_turn(failing_loop, session, turn_state)

    assert not result.success
    assert session.messages is original


class _ScriptedProvider:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls: list[list[dict]] = []

    def get_default_model(self):
        return "model"

    async def chat(self, *, messages, **kwargs):
        self.calls.append(messages)
        return self.responses.pop(0)


class _Context:
    def add_user_messages(self, messages, inbound):
        raise AssertionError("no injections expected")

    def add_assistant_message(self, messages, content, tool_calls=None, reasoning_content=None):
        message = {"role": "assistant", "content": content}
        if tool_calls:
            message["tool_calls"] = tool_calls
        if reasoning_content is not None:
            message["reasoning_content"] = reasoning_content
        messages.append(message)
        return messages

    def add_tool_result(self, messages, tool_call_id, tool_name, result):
        messages.append({
            "role": "tool",
            "tool_call_id": tool_call_id,
            "name": tool_name,
            "content": result,
        })
        return messages


class _Tools:
    def __init__(self, result="TOOL-RESULT"):
        self.result = result

    def get_definitions(self):
        return []

    async def execute(self, name, arguments):
        return self.result

    def get(self, name):
        return None


def _agent_loop(tmp_path, provider, *, context_limit=1_000_000) -> AgentLoop:
    async def _publish_outbound(message):
        return None

    loop = AgentLoop(
        bus=SimpleNamespace(publish_outbound=_publish_outbound),
        provider=provider,
        max_tokens=100,
        memory_window=500,
        context_limit=context_limit,
        model="model",
        session_manager=SessionManager(sessions_dir=tmp_path),
    )
    loop._get_context = lambda session: _Context()
    return loop


@pytest.mark.asyncio
async def test_active_turn_accumulates_and_persists_each_message_once(tmp_path) -> None:
    provider = _ScriptedProvider([
        LLMResponse(
            content=None,
            tool_calls=[ToolCallRequest(id="t1", name="read", arguments={})],
            reasoning_content="provider-only reasoning",
        ),
        LLMResponse(content="FINAL-ANSWER"),
    ])
    loop = _agent_loop(tmp_path, provider)
    session = loop.sessions.get_or_create("cli:turn")
    initial = [
        {"role": "system", "content": "BASE-SYSTEM"},
        {"role": "user", "content": "CURRENT-REQUEST\n\n[Runtime Context]"},
    ]
    turn_state = TurnState.from_context_messages(initial, history_length=0)

    final, _, _ = await loop._run_agent_loop(
        initial,
        session=session,
        tools=_Tools(),
        turn_state=turn_state,
    )

    assert final == "FINAL-ANSWER"
    assert [m["role"] for m in turn_state.messages] == ["user", "assistant", "tool", "assistant"]
    persist_turn_messages(session, turn_state.messages)
    assert [m["role"] for m in session.messages] == ["user", "assistant", "tool", "assistant"]
    assert repr(session.messages).count("CURRENT-REQUEST") == 1
    assert repr(session.messages).count("TOOL-RESULT") == 1
    assert repr(session.messages).count("FINAL-ANSWER") == 1
    assert "reasoning_content" not in session.messages[1]
    assert "reasoning_content" in turn_state.messages[1]


class _CompactingScriptedProvider:
    def __init__(self):
        self.main_calls: list[list[dict]] = []
        self.summary_calls: list[list[dict]] = []

    def get_default_model(self):
        return "model"

    async def chat(self, *, messages, **kwargs):
        if messages and messages[0].get("content") == COMPACTION_SYSTEM_PROMPT:
            self.summary_calls.append(deepcopy(messages))
            return LLMResponse(content="COMPACTED-HISTORY")

        self.main_calls.append(deepcopy(messages))
        if len(self.main_calls) == 1:
            return LLMResponse(
                content=None,
                tool_calls=[ToolCallRequest(id="t1", name="read", arguments={})],
            )
        return LLMResponse(content="FINAL-AFTER-COMPACT")


@pytest.mark.asyncio
async def test_tool_result_triggers_real_loop_compaction_and_rebuilt_transcript(tmp_path) -> None:
    provider = _CompactingScriptedProvider()
    loop = _agent_loop(tmp_path, provider, context_limit=10_000)
    loop.compact_keep_max_tokens = 0
    session = loop.sessions.get_or_create("cli:real-chain")
    session.messages = [
        {"role": "user", "content": _long("historical request")},
        {"role": "assistant", "content": _long("historical answer")},
    ]
    history = session.get_history(max_messages=loop.memory_window)
    current = "CURRENT-REQUEST\n\n[Runtime Context]\nChannel: cli"
    initial = [
        {"role": "system", "content": "BASE-SYSTEM"},
        *history,
        {"role": "user", "content": current},
    ]
    turn_state = TurnState.from_context_messages(initial, history_length=len(history))

    final, _, _ = await loop._run_agent_loop(
        initial,
        session=session,
        tools=_Tools(result="T" * 17_000),
        turn_state=turn_state,
    )

    assert final == "FINAL-AFTER-COMPACT"
    assert len(provider.summary_calls) == 1
    assert len(provider.main_calls) == 2
    rebuilt = repr(provider.main_calls[1])
    assert rebuilt.count("CURRENT-REQUEST") == 1
    assert rebuilt.count("[Runtime Context]") == 1
    assert rebuilt.count("COMPACTED-HISTORY") == 1
    tool_messages = [m for m in provider.main_calls[1] if m.get("role") == "tool"]
    assert len(tool_messages) == 1
    assert tool_messages[0]["content"] == "T" * 17_000

    persist_turn_messages(session, turn_state.messages)
    persisted = repr(session.messages)
    assert persisted.count("CURRENT-REQUEST") == 1
    assert persisted.count("FINAL-AFTER-COMPACT") == 1


class _OverflowProvider:
    def __init__(self):
        self.calls = 0

    def get_default_model(self):
        return "model"

    async def chat(self, **kwargs):
        self.calls += 1
        raise LLMCallError(LLMErrorKind.CONTEXT_OVERFLOW, "too large")


@pytest.mark.asyncio
async def test_overflow_is_not_retried_when_history_cannot_shrink(tmp_path) -> None:
    provider = _OverflowProvider()
    loop = _agent_loop(tmp_path, provider)
    session = loop.sessions.get_or_create("cli:overflow")
    turn_state = TurnState(
        system_prefix=[{"role": "system", "content": "system"}],
        messages=[{"role": "user", "content": _long("only current turn")}],
    )
    messages = turn_state.rebuild(session, loop.memory_window)

    with pytest.raises(LLMCallError):
        await loop._chat_with_retry(
            model="model",
            messages=messages,
            temperature=0,
            max_tokens=100,
            session=session,
            turn_state=turn_state,
            tools=_Tools(),
        )

    assert provider.calls == 1


class _PausingCompactionProvider:
    def __init__(self):
        self.summary_started = asyncio.Event()
        self.release_summary = asyncio.Event()
        self.main_calls: list[list[dict]] = []

    def get_default_model(self):
        return "model"

    async def chat(self, *, messages, **kwargs):
        if messages and messages[0].get("content") == COMPACTION_SYSTEM_PROMPT:
            self.summary_started.set()
            await self.release_summary.wait()
            return LLMResponse(content="HISTORY-SUMMARY")
        self.main_calls.append(deepcopy(messages))
        return LLMResponse(content="done with supplement")


@pytest.mark.asyncio
async def test_injection_arriving_during_compaction_reaches_main_call_with_media(tmp_path) -> None:
    provider = _PausingCompactionProvider()
    loop = _agent_loop(tmp_path, provider, context_limit=10_000)
    session = loop.sessions.get_or_create("cli:injection")
    loop._get_context = lambda current_session: ContextBuilder(
        session_dir=current_session.directory
    )
    session.messages = [
        {"role": "user", "content": _long("history one")},
        {"role": "assistant", "content": _long("answer one")},
        {"role": "user", "content": _long("history two")},
        {"role": "assistant", "content": _long("answer two")},
        {"role": "user", "content": _long("history three")},
        {"role": "assistant", "content": _long("answer three")},
    ]
    history = session.get_history(max_messages=loop.memory_window)
    initial = [
        {"role": "system", "content": "BASE-SYSTEM"},
        *history,
        {"role": "user", "content": "CURRENT-GOAL"},
    ]
    turn_state = TurnState.from_context_messages(initial, history_length=len(history))
    image_path = session.directory / "data" / "injected.png"
    image_path.parent.mkdir(parents=True, exist_ok=True)
    image_path.write_bytes(b"not-a-real-png-but-readable")

    task = asyncio.create_task(loop._run_agent_loop(
        initial,
        session=session,
        tools=_Tools(),
        turn_state=turn_state,
    ))
    await provider.summary_started.wait()
    loop._pending_injections[session.key] = [InboundMessage(
        channel="cli",
        chat_id="injection",
        sender_id="user",
        content="SUPPLEMENT-DURING-COMPACT",
        media=[str(image_path)],
    )]
    provider.release_summary.set()

    final, _, _ = await task

    assert final == "done with supplement"
    assert len(provider.main_calls) == 1
    rendered = repr(provider.main_calls[0])
    assert rendered.count("SUPPLEMENT-DURING-COMPACT") == 1
    image_blocks = [
        block
        for message in provider.main_calls[0]
        for block in (message.get("content") if isinstance(message.get("content"), list) else [])
        if isinstance(block, dict) and block.get("type") == "image_url"
    ]
    assert len(image_blocks) == 1
    assert not loop._pending_injections.get(session.key)


class _OverflowThenCompactProvider:
    def __init__(self):
        self.main_calls: list[list[dict]] = []
        self.summary_calls: list[list[dict]] = []

    def get_default_model(self):
        return "model"

    async def chat(self, *, messages, **kwargs):
        if messages and messages[0].get("content") == COMPACTION_SYSTEM_PROMPT:
            self.summary_calls.append(deepcopy(messages))
            return LLMResponse(content="ACTIVE-PROGRESS-SUMMARY")
        self.main_calls.append(deepcopy(messages))
        if len(self.main_calls) == 1:
            raise LLMCallError(LLMErrorKind.CONTEXT_OVERFLOW, "too large")
        return LLMResponse(content="recovered")


@pytest.mark.asyncio
async def test_overflow_recovers_by_compacting_completed_active_tool_group(tmp_path) -> None:
    provider = _OverflowThenCompactProvider()
    loop = _agent_loop(tmp_path, provider, context_limit=10_000)
    session = loop.sessions.get_or_create("cli:active-overflow")
    current = {"role": "user", "content": "CURRENT-GOAL"}
    assistant = {
        "role": "assistant",
        "content": None,
        "tool_calls": [{
            "id": "t1",
            "type": "function",
            "function": {"name": "read", "arguments": "{}"},
        }],
    }
    tool = {
        "role": "tool",
        "tool_call_id": "t1",
        "name": "read",
        "content": "R" * 20_000,
    }
    turn_state = TurnState(
        system_prefix=[{"role": "system", "content": "BASE-SYSTEM"}],
        messages=[current, assistant, tool],
    )

    response, rebuilt = await loop._chat_with_retry(
        model="model",
        messages=turn_state.rebuild(session, loop.memory_window),
        temperature=0,
        max_tokens=100,
        session=session,
        turn_state=turn_state,
        tools=_Tools(),
    )

    assert response.content == "recovered"
    assert len(provider.main_calls) == 2
    assert len(provider.summary_calls) == 1
    assert "CURRENT-GOAL" in provider.summary_calls[0][1]["content"]
    assert repr(rebuilt).count("CURRENT-GOAL") == 1
    assert repr(rebuilt).count("ACTIVE-PROGRESS-SUMMARY") == 1
    assert not [message for message in rebuilt if message.get("role") == "tool"]
    assert turn_state.messages == [current, assistant, tool]


class _ProactiveThenOverflowProvider:
    """Force a real overflow after estimator-triggered history compaction."""

    def __init__(self):
        self.main_calls: list[list[dict]] = []
        self.summary_calls: list[list[dict]] = []

    def get_default_model(self):
        return "model"

    async def chat(self, *, messages, **kwargs):
        if messages and messages[0].get("content") == COMPACTION_SYSTEM_PROMPT:
            self.summary_calls.append(deepcopy(messages))
            summary_number = len(self.summary_calls)
            return LLMResponse(content=(
                "HISTORY-SUMMARY"
                if summary_number == 1
                else "ACTIVE-PROGRESS-SUMMARY"
            ))

        self.main_calls.append(deepcopy(messages))
        if len(self.main_calls) == 1:
            raise LLMCallError(LLMErrorKind.CONTEXT_OVERFLOW, "provider disagrees with estimator")
        return LLMResponse(content="recovered after active compaction")


@pytest.mark.asyncio
async def test_real_overflow_can_force_active_compaction_after_proactive_history_compaction(
    tmp_path,
) -> None:
    provider = _ProactiveThenOverflowProvider()
    loop = _agent_loop(tmp_path, provider, context_limit=10_000)
    loop.compact_keep_max_tokens = 0
    session = loop.sessions.get_or_create("cli:proactive-then-overflow")
    session.messages = [
        {"role": "user", "content": _long("history request one")},
        {"role": "assistant", "content": _long("history answer one")},
        {"role": "user", "content": _long("history request two")},
        {"role": "assistant", "content": _long("history answer two")},
    ]
    history = session.get_history(max_messages=loop.memory_window)
    current = {"role": "user", "content": "CURRENT-GOAL"}
    assistant = {
        "role": "assistant",
        "content": None,
        "tool_calls": [{
            "id": "t1",
            "type": "function",
            "function": {"name": "read", "arguments": "{}"},
        }],
    }
    tool = {
        "role": "tool",
        "tool_call_id": "t1",
        "name": "read",
        "content": "R" * 4_000,
    }
    initial = [
        {"role": "system", "content": "BASE-SYSTEM"},
        *history,
        current,
        assistant,
        tool,
    ]
    turn_state = TurnState.from_context_messages(initial, history_length=len(history))

    final, _, _ = await loop._run_agent_loop(
        initial,
        session=session,
        tools=_Tools(),
        turn_state=turn_state,
    )

    assert final == "recovered after active compaction"
    assert len(provider.summary_calls) == 2
    assert len(provider.main_calls) == 2
    recovered = provider.main_calls[1]
    assert repr(recovered).count("CURRENT-GOAL") == 1
    assert repr(recovered).count("ACTIVE-PROGRESS-SUMMARY") == 1
    assert not [message for message in recovered if message.get("role") == "tool"]
    assert not [message for message in recovered if message.get("tool_calls")]


@pytest.mark.asyncio
async def test_process_message_persists_active_compaction_projection_across_reload(
    tmp_path,
) -> None:
    provider = _CompactingScriptedProvider()
    loop = _agent_loop(tmp_path, provider, context_limit=10_000)
    loop._get_context = lambda session: ContextBuilder(session_dir=session.directory)
    loop.build_turn_tools = lambda **kwargs: _Tools(result="R" * 40_000)
    message = InboundMessage(
        channel="cli",
        chat_id="active-persistence",
        sender_id="user",
        content="DURABLE-CURRENT-GOAL",
    )

    response = await loop._process_message(
        message,
        session_key="cli:active-persistence",
    )

    assert response is not None
    assert response.content == "FINAL-AFTER-COMPACT"
    assert len(provider.summary_calls) == 1
    loop.sessions.invalidate("cli:active-persistence")
    reloaded = loop.sessions.get_or_create("cli:active-persistence")
    history = reloaded.get_history(max_messages=loop.memory_window)
    rendered = repr(history)
    assert rendered.count("DURABLE-CURRENT-GOAL") == 1
    assert rendered.count("COMPACTED-HISTORY") == 1
    assert rendered.count("FINAL-AFTER-COMPACT") == 1
    assert not [message for message in history if message.get("role") == "tool"]
    assert not [message for message in history if message.get("tool_calls")]
    raw_tools = [
        message
        for message in reloaded.messages
        if message.get("role") == "tool" and message.get("_context_excluded")
    ]
    assert len(raw_tools) == 1
    assert raw_tools[0]["content"] == "R" * 40_000
    await loop.close_sandboxes()


@pytest.mark.asyncio
async def test_later_history_compaction_ignores_but_preserves_raw_active_audit() -> None:
    provider = _SummaryProvider(["FOLDED-HISTORY"])
    raw_assistant = {
        "role": "assistant",
        "content": None,
        "tool_calls": [{
            "id": "raw-tool",
            "type": "function",
            "function": {"name": "read", "arguments": "{}"},
        }],
        "_context_excluded": True,
    }
    raw_tool = {
        "role": "tool",
        "tool_call_id": "raw-tool",
        "name": "read",
        "content": "RAW-TOOL-PAYLOAD-" * 2_000,
        "_context_excluded": True,
    }
    session = Session(key="cli:projected-history")
    session.messages = [
        {"role": "system", "content": "ACTIVE-PROGRESS-SUMMARY"},
        {"role": "user", "content": _long("original goal")},
        raw_assistant,
        raw_tool,
        {"role": "assistant", "content": _long("final answer")},
    ]

    result = await compact_session(
        session=session,
        provider=provider,
        model="model",
        context_limit=20_000,
        keep_max_tokens=0,
        min_messages_to_compact=1,
    )

    assert result.success
    assert "RAW-TOOL-PAYLOAD" not in repr(provider.calls)
    assert raw_assistant in session.messages
    assert raw_tool in session.messages
    history = session.get_history()
    assert repr(history).count("FOLDED-HISTORY") == 1
    assert not [message for message in history if message.get("role") == "tool"]
    assert not [message for message in history if message.get("tool_calls")]


@pytest.mark.asyncio
async def test_process_message_persists_current_user_when_overflow_cannot_recover(tmp_path) -> None:
    provider = _OverflowProvider()
    loop = _agent_loop(tmp_path, provider, context_limit=10_000)
    loop._get_context = lambda session: ContextBuilder(session_dir=session.directory)
    message = InboundMessage(
        channel="cli",
        chat_id="failure",
        sender_id="user",
        content="DURABLE-CURRENT-REQUEST",
    )

    with pytest.raises(LLMCallError):
        await loop._process_message(message, session_key="cli:failure")

    loop.sessions.invalidate("cli:failure")
    reloaded = loop.sessions.get_or_create("cli:failure")
    assert len(reloaded.messages) == 1
    assert reloaded.messages[0]["role"] == "user"
    assert "DURABLE-CURRENT-REQUEST" in str(reloaded.messages[0]["content"])
    await loop.close_sandboxes()
