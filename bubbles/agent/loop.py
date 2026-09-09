"""Agent loop: the core processing engine."""

from __future__ import annotations

import asyncio
import json
import re
import time
from contextlib import AsyncExitStack
from pathlib import Path
from typing import TYPE_CHECKING, Any, Awaitable, Callable

ERROR_REPLY_THROTTLE_SEC = 60.0
DATA_CLEANUP_THROTTLE_SEC = 24 * 3600.0
# API 重试退避：1s → 2s → 4s…，封顶 30s（服务端给了 Retry-After 时优先用它）。
API_RETRY_BASE_DELAY_SEC = 1.0
API_RETRY_MAX_DELAY_SEC = 30.0


def _default_concurrency() -> int:
    """跨 session 的并发上限。

    LLM 调用是 IO 等待，不吃 CPU；真正吃 CPU 的是 exec（本机 subprocess）和
    沙箱启动，所以留一半核给它们。封顶 4：再往上主要是推高同一 provider 的
    限流概率，而个人 bot 场景里 "同时有 >4 个 session 在等" 基本不出现。
    """
    import os

    return min(4, max(2, (os.cpu_count() or 2) // 2))

from loguru import logger

from bubbles.agent.bindings import (
    get_bindings_for_session,
    load_session_bindings,
    relocate_media_to_session,
    save_session_bindings,
)
from bubbles.agent.commands import handle_config_command
from bubbles.agent.context import ContextBuilder
from bubbles.agent.turn import (
    ASSISTANT_TEXT_RECEIPT_PREFIX,
    TurnState,
    compact_for_turn,
    do_compact,
    persist_failed_turn,
    persist_turn_state,
    process_system_message,
    should_compact,
)
from bubbles.agent.subagent import SubagentManager
from bubbles.agent.tools.cron import CronTool
from bubbles.agent.tools.filesystem import EditFileTool, ListDirTool, ReadFileTool, WriteFileTool
from bubbles.agent.tools.find_person import FindPersonTool
from bubbles.agent.tools.image_generation import GenerateImageTool
from bubbles.agent.tools.message import MessageTool, SwitchMessageTargetTool
from bubbles.agent.tools.registry import ToolRegistry
from bubbles.agent.tools.shell import ExecTool
from bubbles.agent.tools.spawn import SpawnTool
from bubbles.agent.tools.stay_silent import STAY_SILENT_SENTINEL, StaySilentTool
from bubbles.agent.tools.task import TaskListTool, TaskGetTool, TaskCreateTool, TaskUpdateTool
from bubbles.agent.tools.web import WebFetchTool, WebSearchTool
from bubbles.sandbox.manager import SandboxManager
from bubbles.bus.events import InboundMessage, OutboundMessage
from bubbles.bus.queue import MessageBus
from bubbles.providers.base import LLMCallError, LLMErrorKind, LLMProvider
from bubbles.session.manager import (
    Session,
    SessionManager,
    cleanup_data_dir,
    prune_old_images_inplace,
)

if TYPE_CHECKING:
    from bubbles.config.schema import ChannelsConfig, ExecToolConfig, SandboxConfig
    from bubbles.gateway_control import GatewayControl
    from bubbles.image_generation import ImageGenerationBackend
    from bubbles.sandbox.base import Sandbox
    from bubbles.cron.service import CronService


class AgentLoop:
    """
    The agent loop is the core processing engine.

    It:
    1. Receives messages from the bus
    2. Builds context with history, memory, skills
    3. Calls the LLM
    4. Executes tool calls
    5. Sends responses back
    """

    def __init__(
        self,
        bus: MessageBus,
        provider: LLMProvider,
        max_tokens: int,
        memory_window: int,
        context_limit: int,
        model: str | None = None,
        max_iterations: int = 40,
        temperature: float = 0.1,
        tavily_api_key: str | None = None,
        exec_config: ExecToolConfig | None = None,
        sandbox_config: "SandboxConfig | None" = None,
        cron_service: CronService | None = None,
        session_manager: SessionManager | None = None,
        channel_manager: Any = None,
        provider_factory: Callable[[str], tuple[str, LLMProvider]] | None = None,
        default_provider_name: str | None = None,
        mcp_servers: dict | None = None,
        channels_config: ChannelsConfig | None = None,
        # Auto-compaction settings
        compact_threshold: float = 0.85,
        compact_keep_max_tokens: int = 40_000,
        compact_min_messages: int = 5,
        max_api_retries: int = 2,
        max_concurrent_sessions: int = 0,
        image_generation_backend: "ImageGenerationBackend | None" = None,
        gateway_control: "GatewayControl | None" = None,
    ):
        from bubbles.config.schema import ExecToolConfig, SandboxConfig
        from bubbles.utils.helpers import get_data_path
        self.bus = bus
        self.channels_config = channels_config
        self.provider = provider
        self.data_dir = get_data_path()  # ~/.bubbles/
        self.model = model or provider.get_default_model()
        self.max_iterations = max_iterations
        self.max_api_retries = max_api_retries
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.memory_window = memory_window
        self.context_limit = context_limit
        self.tavily_api_key = tavily_api_key
        self.exec_config = exec_config or ExecToolConfig()
        self.sandbox_config = sandbox_config or SandboxConfig()
        self.cron_service = cron_service
        self.channel_manager = channel_manager
        self.provider_factory = provider_factory
        self.default_provider_name = default_provider_name
        self.image_generation_backend = image_generation_backend
        self.gateway_control = gateway_control
        self._provider_cache: dict[str, LLMProvider] = {}
        if provider is not None and default_provider_name:
            self._provider_cache[default_provider_name] = provider

        # Auto-compaction settings
        self.compact_threshold = compact_threshold
        self.compact_keep_max_tokens = compact_keep_max_tokens
        self.compact_min_messages = compact_min_messages

        self._context_cache: dict[str, ContextBuilder] = {}  # session_key -> ContextBuilder
        self.sessions = session_manager or SessionManager()
        self.tools = ToolRegistry()
        self._sandboxes = SandboxManager(
            config=self.sandbox_config,
            path_append=self.exec_config.path_append,
        )
        self.subagents = SubagentManager(
            provider=provider,
            bus=bus,
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            tavily_api_key=tavily_api_key,
            exec_config=self.exec_config,
            sandbox_manager=self._sandboxes,
        )

        self._running = False
        self._mcp_servers = mcp_servers or {}
        self._mcp_stack: AsyncExitStack | None = None
        self._mcp_connected = False
        self._mcp_connecting = False
        # {channel}:{chat_id} -> custom session key
        self._session_bindings: dict[str, str] = load_session_bindings(self.data_dir)
        self._active_tasks: dict[str, list[asyncio.Task]] = {}  # session_key -> tasks
        self._last_error_reply_at: dict[str, float] = {}  # session_key -> monotonic ts of last user-visible error reply
        self._last_data_cleanup_at: dict[str, float] = {}  # session_key -> monotonic ts of last data/ cleanup
        # 同 session 串行；跨 session 并发，上限由 semaphore 控。
        self._session_locks: dict[str, asyncio.Lock] = {}
        self._concurrency = asyncio.Semaphore(max_concurrent_sessions or _default_concurrency())
        # session_key -> 等着插进当前 turn 的用户消息
        self._pending_injections: dict[str, list[InboundMessage]] = {}
        self.on_tool_call: Callable[[str, dict, str | None], Awaitable[None]] | None = None  # Debug callback
        self._register_default_tools()

    def _register_default_tools(self) -> None:
        """Register all built-in schemas, regardless of backend availability.

        Session-scoped tool instances are built separately by
        :meth:`build_turn_tools`; only stateless tools are shared across turns.

        ``self.tools`` remains the template registry: it backs
        ``get_definitions()`` (the schema list is identical for every session)
        and is where MCP servers register their tools on connect.
        """
        for cls in (ReadFileTool, WriteFileTool, EditFileTool, ListDirTool):
            self.tools.register(cls())
        self.tools.register(ExecTool(timeout=self.exec_config.timeout))
        self.tools.register(WebSearchTool(api_key=self.tavily_api_key))
        self.tools.register(WebFetchTool())
        message = MessageTool(send_callback=self.bus.publish_outbound)
        self.tools.register(message)
        self.tools.register(SwitchMessageTargetTool(message, self.channel_manager))
        self.tools.register(StaySilentTool())
        self.tools.register(GenerateImageTool(self.image_generation_backend))
        self.tools.register(SpawnTool(manager=self.subagents))
        self.tools.register(CronTool(self.cron_service))
        find_person = FindPersonTool()
        find_person.set_channel_manager(self.channel_manager)
        find_person.set_target_provider(lambda: message.target)
        self.tools.register(find_person)
        for cls in (TaskListTool, TaskGetTool, TaskCreateTool, TaskUpdateTool):
            self.tools.register(cls())

    def build_turn_tools(
        self,
        channel: str,
        chat_id: str,
        message_id: str | None,
        session_dir: Path | None,
        session_key: str,
        session: Session | None,
        sandbox: "Sandbox | None",
        system_triggered: bool = False,
    ) -> ToolRegistry:
        """Build a registry owned by one turn.

        Why: tools used to be shared singletons whose sandbox / channel /
        session were rewritten before each turn. That is only correct while a
        global lock serializes every turn — under per-session concurrency,
        session A's ``exec`` would run in session B's sandbox, and A's
        ``message`` would deliver into B's chat. Constructing per turn removes
        the shared mutable state instead of guarding it, which is also how
        subagents already build their tools.

        Stateless tools (web, MCP) are shared by reference — they read nothing
        but their call arguments.
        """
        reg = ToolRegistry()

        for cls in (ReadFileTool, WriteFileTool, EditFileTool, ListDirTool):
            tool = cls()
            tool.set_sandbox(sandbox)
            reg.register(tool)

        exec_tool = ExecTool(timeout=self.exec_config.timeout)
        exec_tool.set_sandbox(sandbox)
        reg.register(exec_tool)

        message = MessageTool(send_callback=self.bus.publish_outbound)
        message.set_context(channel, chat_id, message_id)
        if session is not None:
            message.bind_session(session, lambda: self.sessions.save(session))
        message.set_session_dir(session_dir)
        message.start_turn()
        reg.register(message)
        reg.register(SwitchMessageTargetTool(message, self.channel_manager))
        reg.register(StaySilentTool())

        image_generation_backend = getattr(self, "image_generation_backend", None)
        image_generation = GenerateImageTool(image_generation_backend)
        image_generation.set_sandbox(sandbox)
        reg.register(image_generation)

        spawn = SpawnTool(manager=self.subagents)
        spawn.set_context(channel, chat_id, session_key)
        spawn.set_session_dir(session_dir)
        reg.register(spawn)

        # Keep schemas stable; enforce availability in execute(), not registration.
        cron = CronTool(self.cron_service, system_triggered=system_triggered)
        cron.set_context(channel, chat_id, session_key)
        reg.register(cron)

        find_person = FindPersonTool()
        find_person.set_channel_manager(self.channel_manager)
        find_person.set_context(channel, chat_id)
        find_person.set_target_provider(lambda: message.target)
        reg.register(find_person)

        for cls in (TaskListTool, TaskGetTool, TaskCreateTool, TaskUpdateTool):
            tool = cls()
            if session is not None:
                tool.set_session(session)
            reg.register(tool)

        # Stateless: safe to share the same instances across concurrent turns.
        for name in ("web_search", "web_fetch"):
            if tool := self.tools.get(name):
                reg.register(tool)
        for name, tool in self._mcp_tools().items():
            reg.register(tool)

        return reg

    def _mcp_tools(self) -> dict[str, Any]:
        """MCP-provided tools from the template registry (stateless, shared)."""
        builtin = {
            "read_file", "write_file", "edit_file", "list_dir", "exec",
            "web_search", "web_fetch", "message", "switch_message_target", "spawn", "cron",
            "find_person", "task_list", "task_get", "task_create",
            "task_update", "stay_silent", "generate_image",
        }
        return {n: t for n, t in self.tools._tools.items() if n not in builtin}

    def _provider_for(self, model: str | None) -> LLMProvider:
        """Resolve the provider that should handle this model.

        Caches per provider_name; falls back to the default provider when no
        factory is configured or instantiation fails (logged warning).
        """
        if not model or self.provider_factory is None:
            return self.provider
        try:
            provider_name, prov = self.provider_factory(model)
        except Exception as e:
            logger.warning(
                "Provider factory failed for model {!r}: {} — falling back to default",
                model, e,
            )
            return self.provider
        cached = self._provider_cache.get(provider_name)
        if cached is not None:
            return cached
        self._provider_cache[provider_name] = prov
        return prov

    def _get_context(self, session: Session) -> ContextBuilder:
        """Get ContextBuilder for a session (cached per session key)."""
        if session.key not in self._context_cache:
            self._context_cache[session.key] = ContextBuilder(session_dir=session.directory)
        return self._context_cache[session.key]

    async def _connect_mcp(self) -> None:
        """Connect to configured MCP servers (one-time, lazy)."""
        if self._mcp_connected or self._mcp_connecting or not self._mcp_servers:
            return
        self._mcp_connecting = True
        from bubbles.agent.tools.mcp import connect_mcp_servers
        try:
            self._mcp_stack = AsyncExitStack()
            await self._mcp_stack.__aenter__()
            await connect_mcp_servers(self._mcp_servers, self.tools, self._mcp_stack)
            self._mcp_connected = True
        except Exception as e:
            logger.error("Failed to connect MCP servers (will retry next message): {}", e)
            if self._mcp_stack:
                try:
                    await self._mcp_stack.aclose()
                except Exception:
                    pass
                self._mcp_stack = None
        finally:
            self._mcp_connecting = False



    @staticmethod
    def _strip_think(text: str | None) -> str | None:
        """Remove <think>…</think> blocks that some models embed in content."""
        if not text:
            return None
        return re.sub(r"<think>[\s\S]*?(?:</think>|$)", "", text, flags=re.IGNORECASE).strip() or None

    @staticmethod
    def _tool_hint(tool_calls: list) -> str:
        """Format tool calls as concise hint, e.g. 'web_search("query")'."""
        def _fmt(tc):
            val = next(iter(tc.arguments.values()), None) if tc.arguments else None
            if not isinstance(val, str):
                return tc.name
            return f'{tc.name}("{val[:40]}…")' if len(val) > 40 else f'{tc.name}("{val}")'
        return ", ".join(_fmt(tc) for tc in tool_calls)

    async def _chat_with_retry(
        self,
        model: str | None,
        messages: list[dict],
        temperature: float,
        max_tokens: int,
        session: Session,
        turn_state: TurnState | None = None,
        tools: ToolRegistry | None = None,
        on_progress: Callable[..., Awaitable[None]] | None = None,
    ) -> tuple[Any, list[dict]]:
        """调一次 LLM，可重试的失败按类别退避重试。

        返回 ``(response, messages)`` —— messages 可能被 context_overflow 恢复
        路径重建过，调用方必须用返回的这份。

        重试放在这一层而不是 provider：只有这里能做 context_overflow 的恢复
        动作（压缩历史后重试），provider 看不到 session。
        """
        if turn_state is None:
            system_count = 1 if messages and messages[0].get("role") == "system" else 0
            turn_state = TurnState(
                system_prefix=list(messages[:system_count]),
                messages=list(messages[system_count:]),
            )

        attempt = 0
        overflow_recovery_attempted = False

        while True:
            attempt += 1
            try:
                response = await self._provider_for(model).chat(
                    messages=messages,
                    tools=(tools or self.tools).get_definitions(),
                    model=model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                return response, messages
            except LLMCallError as e:
                e.attempts = attempt  # 供上层文案说明"已重试 N 次"
                last_attempt = attempt > self.max_api_retries
                if not e.retryable or last_attempt:
                    logger.error(
                        "LLM call failed ({}) after {} attempt(s) for session {}: {}",
                        e.kind.value, attempt, session.key, e.detail,
                    )
                    raise

                if e.kind is LLMErrorKind.CONTEXT_OVERFLOW:
                    # Preflight compaction is estimator-driven and may only have
                    # reduced history.  A real provider overflow still gets one
                    # stronger recovery attempt that also compacts completed
                    # active tool groups.  A second real overflow is terminal.
                    if overflow_recovery_attempted:
                        logger.error(
                            "Context still overflowing after compaction for session {}", session.key,
                        )
                        raise
                    overflow_recovery_attempted = True
                    logger.warning("Context overflow for session {}; compacting and retrying", session.key)
                    messages, result = await compact_for_turn(
                        self,
                        session,
                        turn_state,
                        on_progress,
                        force_active=True,
                    )
                    if not result.success:
                        logger.error(
                            "Context compaction made no progress for session {}: {}",
                            session.key,
                            result.error,
                        )
                        raise
                    continue

                delay = e.retry_after if e.retry_after is not None else API_RETRY_BASE_DELAY_SEC * (2 ** (attempt - 1))
                delay = min(delay, API_RETRY_MAX_DELAY_SEC)
                logger.warning(
                    "LLM call failed ({}) attempt {}/{} for session {}; retrying in {:.1f}s: {}",
                    e.kind.value, attempt, self.max_api_retries + 1, session.key, delay, e.detail,
                )
                await asyncio.sleep(delay)

    async def _run_agent_loop(
        self,
        initial_messages: list[dict],
        on_progress: Callable[..., Awaitable[None]] | None = None,
        session: Session | None = None,
        should_stop: Callable[[], bool] | None = None,
        on_tool_call: Callable[[str, dict, str | None], Awaitable[None]] | None = None,
        tools: ToolRegistry | None = None,
        turn_state: TurnState | None = None,
        require_explicit_end: bool = False,
    ) -> tuple[str | None, list[str], list[dict]]:
        """Run the agent iteration loop. Returns (final_content, tools_used, messages)."""
        messages = list(initial_messages)
        iteration = 0
        final_content = None
        tools_used: list[str] = []
        text_only_retries = 0
        # 每轮自己的工具集；缺省回落到模板 registry（测试与旧调用方）。
        tools = tools if tools is not None else self.tools

        # Get context for this session (session is required in new architecture)
        if not session:
            raise ValueError("Session is required for agent loop")
        context = self._get_context(session)
        if turn_state is None:
            # Backward-compatible default for direct/internal callers. Production
            # callers pass the exact ContextBuilder boundary explicitly.
            system_count = 1 if messages and messages[0].get("role") == "system" else 0
            turn_state = TurnState(
                system_prefix=list(messages[:system_count]),
                messages=list(messages[system_count:]),
            )

        # Use session config if available, otherwise use defaults
        cfg = session.config if session else None
        model = (cfg.model if cfg and cfg.model else self.model)
        temperature = (cfg.temperature if cfg and cfg.temperature is not None else self.temperature)
        max_tokens = (cfg.max_tokens if cfg and cfg.max_tokens else self.max_tokens)

        message_tool = tools.get("message")
        if not isinstance(message_tool, MessageTool):
            message_tool = None

        def _record_notice(content: str) -> None:
            messages.append({"role": "system", "content": content})
            turn_state.capture_appended(messages, len(messages) - 1)

        def _record_target() -> None:
            if message_tool is not None:
                channel, chat_id = message_tool.target
                _record_notice("[Current message target] " + json.dumps({
                    "channel": channel, "chat_id": chat_id,
                    "note": "Persists until switch_message_target succeeds. Incoming message sources do not change it.",
                }, ensure_ascii=False))

        async def _send_assistant_text(text: str | None) -> str | None:
            content = self._strip_think(text)
            if content and message_tool is not None:
                return await message_tool.execute(content=content)
            return None

        def _record_text_receipt(receipt: str | None) -> None:
            if receipt is not None:
                _record_notice(
                    ASSISTANT_TEXT_RECEIPT_PREFIX
                    + ToolRegistry._cap_result("message", receipt)
                )

        # Dynamic destination belongs at the tail, never in the stable prompt.
        _record_target()

        def _drain_injections() -> int:
            nonlocal messages
            injected = self._take_injections(session.key)
            if not injected:
                return 0
            appended_from = len(messages)
            messages = context.add_user_messages(messages, injected)
            turn_state.capture_appended(messages, appended_from)
            logger.info(
                "Injected {} mid-turn message(s) into session {}",
                len(injected),
                session.key,
            )
            return len(injected)

        while iteration < self.max_iterations:
            iteration += 1

            # Check external stop signal
            if should_stop and should_stop():
                logger.warning("External stop signal received, ending agent loop")
                break

            # 把 turn 进行期间到达的用户消息插进来。位置只能是这里：
            # assistant(tool_calls) 与它的 tool_result 之间不允许插 user 消息，
            # 所以「下一次工具调用之前」就是循环顶部。放在 compaction 之前，
            # 注入的内容才会被计入 token 估算。
            _drain_injections()

            # Auto-compaction: check if context is overflowing (pre-call estimation)
            if should_compact(self, messages, session):
                messages, _ = await compact_for_turn(
                    self,
                    session,
                    turn_state,
                    on_progress,
                )
                # Summarization is an awaited provider call.  Messages that
                # arrived during it must influence the main request, not wait in
                # a queue that may outlive this turn.
                if _drain_injections() and should_compact(self, messages, session):
                    messages, _ = await compact_for_turn(
                        self,
                        session,
                        turn_state,
                        on_progress,
                    )

            response, messages = await self._chat_with_retry(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                session=session,
                turn_state=turn_state,
                tools=tools,
                on_progress=on_progress,
            )

            if response.has_tool_calls:
                text_only_retries = 0
                # Ending is a batch barrier, never a reason to discard other calls.
                effective_tool_calls = response.tool_calls

                tool_call_dicts = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.name,
                            "arguments": json.dumps(tc.arguments, ensure_ascii=False)
                        }
                    }
                    for tc in effective_tool_calls
                ]
                appended_from = len(messages)
                messages = context.add_assistant_message(
                    messages, response.content, tool_call_dicts,
                    reasoning_content=response.reasoning_content,
                )
                turn_state.capture_appended(messages, appended_from)
                end_requested = False
                # Provider text precedes its tool calls, including any switch.
                receipt = await _send_assistant_text(response.content)
                previous_target = message_tool.target if message_tool is not None else None
                try:
                    for tool_call in effective_tool_calls:
                        tools_used.append(tool_call.name)
                        args_str = json.dumps(tool_call.arguments, ensure_ascii=False)
                        logger.info("Tool call: {}({})", tool_call.name, args_str[:200])
                        if on_tool_call:
                            await on_tool_call(tool_call.name, tool_call.arguments, None)
                        result = await tools.execute(tool_call.name, tool_call.arguments)
                        if on_tool_call:
                            await on_tool_call(tool_call.name, tool_call.arguments, result)
                        appended_from = len(messages)
                        messages = context.add_tool_result(
                            messages, tool_call.id, tool_call.name, result
                        )
                        turn_state.capture_appended(messages, appended_from)
                        if tool_call.name == "stay_silent" and result == STAY_SILENT_SENTINEL:
                            end_requested = True
                finally:
                    # Never interrupt assistant/tool-result pairing. Preserve
                    # the text submission audit even if later work is cancelled.
                    _record_text_receipt(receipt)
                    if message_tool is not None and message_tool.target != previous_target:
                        _record_target()

                if end_requested:
                    turn_state.suppress_outbound = True
                    final_content = None
                    logger.info("stay_silent/end_turn completed after all tool results")
                    break
            else:
                final_content = self._strip_think(response.content)
                # Keep the actual provider message (including reasoning) for
                # round-trip fidelity; only stripped content is sent outward.
                if response.content or response.reasoning_content is not None:
                    appended_from = len(messages)
                    messages = context.add_assistant_message(
                        messages, response.content, None,
                        reasoning_content=response.reasoning_content,
                    )
                    turn_state.capture_appended(messages, appended_from)
                receipt = await _send_assistant_text(response.content)
                _record_text_receipt(receipt)
                if receipt is not None:
                    final_content = None  # Already submitted; never send twice.
                if require_explicit_end:
                    final_content = None
                    if text_only_retries >= 1:
                        logger.warning("Model omitted end_turn twice; stopping without additional output")
                        break
                    text_only_retries += 1
                    reminder = {
                        "role": "system",
                        "content": (
                            "[Harness protocol] Any assistant text was handled according to its receipt; "
                            "do not repeat successfully submitted text. "
                            "If your work is finished, call stay_silent (end_turn) without additional text. "
                            "Otherwise continue with the needed tools."
                        ),
                    }
                    messages.append(reminder)
                    turn_state.capture_appended(messages, len(messages) - 1)
                    continue
                if receipt is not None:
                    turn_state.suppress_outbound = True
                break

        if (
            final_content is None
            and iteration >= self.max_iterations
            and not turn_state.suppress_outbound
        ):
            logger.warning("Max iterations ({}) reached", self.max_iterations)
            if require_explicit_end:
                return None, tools_used, messages
            final_content = (
                f"I reached the maximum number of tool call iterations ({self.max_iterations}) "
                "without completing the task. You can try breaking the task into smaller steps."
            )
            appended_from = len(messages)
            messages = context.add_assistant_message(messages, final_content, None)
            turn_state.capture_appended(messages, appended_from)

        return final_content, tools_used, messages

    async def run(self) -> None:
        """Run the agent loop, dispatching messages as tasks to stay responsive to /stop."""
        self._running = True
        await self._connect_mcp()
        # Sweep stale files in every session's data/ directory at startup (SPEC §5.4).
        try:
            self.sessions.cleanup_all_data_dirs()
        except Exception as e:
            logger.warning("Startup data/ cleanup failed: {}", e)
        logger.info("Agent loop started")
        if self.gateway_control is not None:
            self.gateway_control.mark_agent_ready()

        while self._running:
            try:
                msg = await asyncio.wait_for(self.bus.consume_inbound(), timeout=1.0)
            except asyncio.TimeoutError:
                continue
            if not self._running:
                break

            if msg.content.strip().lower() == "/stop":
                await self._handle_stop(msg)
                continue

            key = self._resolve_session_key(msg)

            # 该 session 正在跑 turn 且这条不是命令 → 插进当前 turn，而不是
            # 排队等它跑完（最多 40 轮迭代，用户可能要等几分钟）。
            if self._has_active_turn(key) and not self._is_command(msg):
                if msg.media:
                    session = self.sessions.get_or_create(key)
                    msg.media = relocate_media_to_session(msg.media, session)
                self._pending_injections.setdefault(key, []).append(msg)
                logger.info("Queued mid-turn injection for session {}", key)
                continue

            task = asyncio.create_task(self._dispatch(msg))
            self._active_tasks.setdefault(key, []).append(task)
            task.add_done_callback(lambda t, k=key: self._forget_task(k, t))

    def _forget_task(self, key: str, task: asyncio.Task) -> None:
        tasks = self._active_tasks.get(key)
        if tasks and task in tasks:
            tasks.remove(task)
        if tasks == []:
            self._active_tasks.pop(key, None)
            # A message can arrive after the loop's last injection drain but
            # before this task's done callback.  Promote the oldest one to a new
            # turn and leave the remainder queued so that turn absorbs them.
            pending = self._pending_injections.get(key) if self._running else None
            if pending:
                next_message = pending.pop(0)
                if not pending:
                    self._pending_injections.pop(key, None)
                next_task = asyncio.create_task(self._dispatch(next_message))
                self._active_tasks.setdefault(key, []).append(next_task)
                next_task.add_done_callback(lambda t, k=key: self._forget_task(k, t))

    def _has_active_turn(self, key: str) -> bool:
        return any(not t.done() for t in self._active_tasks.get(key, []))

    @staticmethod
    def _is_command(msg: InboundMessage) -> bool:
        """命令（/new、/compact、/config…）要走完整的命令解析，不能当文本注入。"""
        text = msg.content
        while m := re.match(r"^\s*<@\S+>\s*", text):
            text = text[m.end():]
        return text.strip().startswith("/")

    def _take_injections(self, key: str) -> list[InboundMessage]:
        return self._pending_injections.pop(key, [])

    async def _handle_stop(self, msg: InboundMessage) -> None:
        """Cancel all active tasks and subagents for the session."""
        key = self._resolve_session_key(msg)
        self._pending_injections.pop(key, None)
        tasks = self._active_tasks.pop(key, [])
        cancelled = sum(1 for t in tasks if not t.done() and t.cancel())
        for t in tasks:
            try:
                await t
            except (asyncio.CancelledError, Exception):
                pass
        sub_cancelled = await self.subagents.cancel_by_session(key)
        total = cancelled + sub_cancelled
        content = f"⏹ Stopped {total} task(s)." if total else "No active task to stop."
        await self.bus.publish_outbound(OutboundMessage(
            channel=msg.channel, chat_id=msg.chat_id, content=content,
        ))

    def _resolve_session_key(self, msg: InboundMessage, session_key: str | None = None) -> str:
        """The session a message actually lands in, honoring /session bindings.

        Why this matters for locking: ``msg.session_key`` is ``channel:chat_id``,
        but two different chats (even on different channels) can be bound to the
        same session. Serializing on ``msg.session_key`` would let both write the
        same session's history concurrently.
        """
        if session_key is not None:
            return session_key
        if msg.channel == "system":
            chat = msg.chat_id
            return chat if ":" not in chat else chat
        return self._session_bindings.get(f"{msg.channel}:{msg.chat_id}") or msg.session_key

    def _session_lock(self, key: str) -> asyncio.Lock:
        """Per-session lock: turns within one session stay strictly serialized."""
        lock = self._session_locks.get(key)
        if lock is None:
            lock = asyncio.Lock()
            self._session_locks[key] = lock
        return lock

    async def _dispatch(self, msg: InboundMessage) -> None:
        """Process a message: serialized per session, concurrent across sessions.

        Two gates instead of the old single global lock:
        - a per-session lock, because concurrent turns in one session would
          interleave writes to the same ``session.messages``;
        - a global semaphore, so N sessions don't fan out into N simultaneous
          provider calls (the real ceiling is the provider's rate limit, not CPU).
        """
        key = self._resolve_session_key(msg)
        self._maybe_cleanup_session_data(key)
        async with self._session_lock(key):
            async with self._concurrency:
                try:
                    response = await self._process_message(msg, on_tool_call=self.on_tool_call)
                    if response is not None:
                        await self.bus.publish_outbound(response)
                    elif msg.channel == "cli":
                        await self.bus.publish_outbound(OutboundMessage(
                            channel=msg.channel, chat_id=msg.chat_id,
                            content="", metadata=msg.metadata or {},
                        ))
                except asyncio.CancelledError:
                    logger.info("Task cancelled for session {}", key)
                    raise
                except Exception as e:
                    logger.exception("Error processing message for session {}", key)
                    await self._emit_error_reply(msg, e)

    def _maybe_cleanup_session_data(self, session_key: str) -> None:
        """Sweep stale files in this session's data/ once per DATA_CLEANUP_THROTTLE_SEC (SPEC §5.4)."""
        now = time.monotonic()
        if (now - self._last_data_cleanup_at.get(session_key, 0.0)) < DATA_CLEANUP_THROTTLE_SEC:
            return
        self._last_data_cleanup_at[session_key] = now
        try:
            session_dir = self.sessions._get_session_dir(session_key)
            removed = cleanup_data_dir(session_dir)
            if removed:
                logger.info("Cleaned {} stale data/ files for session {}", removed, session_key)
        except Exception as e:
            logger.warning("Runtime data/ cleanup failed for session {}: {}", session_key, e)

    async def _emit_error_reply(self, msg: InboundMessage, exc: BaseException | None = None) -> None:
        """Send a user-visible error reply per SPEC §5.1 error policy.

        判据是"这一轮是不是用户主动触发的"，不是群聊/私聊：
        - 用户触发（私聊、群里 @ 机器人、CLI）→ 回一条，用户在等回应，静默才是坏体验；
        - 非用户触发（cron、subagent 汇报等 system turn，或群里没 @ 的旁听
          消息）→ 完全静默，只进日志。没人在等的消息不该让机器人在群里叫。

        ``exc`` 是 LLMCallError 时给出错误类别与重试次数（不含异常类型、堆栈、
        内部路径）；其他异常沿用固定文案。同一 session 60 秒内只发一条。
        """
        user_triggered = (
            msg.channel != "system"
            and bool(msg.metadata.get("respond", True) if msg.metadata else True)
        )
        now = time.monotonic()
        throttled = (now - self._last_error_reply_at.get(msg.session_key, 0.0)) < ERROR_REPLY_THROTTLE_SEC

        if user_triggered and not throttled:
            self._last_error_reply_at[msg.session_key] = now
            if isinstance(exc, LLMCallError):
                content = exc.user_message(getattr(exc, "attempts", 1))
            else:
                content = "Sorry, I encountered an error."
            await self.bus.publish_outbound(OutboundMessage(
                channel=msg.channel, chat_id=msg.chat_id, content=content,
            ))
        elif msg.channel == "cli":
            # Unblock the interactive prompt's turn_done waiter even when silent.
            await self.bus.publish_outbound(OutboundMessage(
                channel=msg.channel, chat_id=msg.chat_id,
                content="", metadata=msg.metadata or {},
            ))

    async def close_mcp(self) -> None:
        """Close MCP connections."""
        if self._mcp_stack:
            try:
                await self._mcp_stack.aclose()
            except (RuntimeError, BaseExceptionGroup):
                pass  # MCP SDK cancel scope cleanup is noisy but harmless
            self._mcp_stack = None

    async def close_sandboxes(self) -> None:
        """Tear down all per-session sandboxes."""
        await self._sandboxes.close_all()

    def stop(self) -> None:
        """Stop the agent loop."""
        self._running = False
        logger.info("Agent loop stopping")

    async def cancel_active_tasks(self) -> None:
        """Cancel and await every active turn and background subagent."""
        self._pending_injections.clear()
        tasks = {
            task
            for session_tasks in self._active_tasks.values()
            for task in session_tasks
            if not task.done()
        }
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._active_tasks.clear()
        await self.subagents.cancel_all()

    async def _process_message(
        self,
        msg: InboundMessage,
        session_key: str | None = None,
        on_progress: Callable[[str], Awaitable[None]] | None = None,
        on_tool_call: Callable[[str, dict, str | None], Awaitable[None]] | None = None,
        system_triggered: bool = False,
        on_message: Callable[[OutboundMessage], Awaitable[None]] | None = None,
    ) -> OutboundMessage | None:
        """Process a single inbound message and return the response."""
        if msg.channel == "system":
            return await process_system_message(self, msg, on_tool_call)

        preview = msg.content[:80] + "..." if len(msg.content) > 80 else msg.content
        logger.info("Processing message from {}:{}: {}", msg.channel, msg.sender_id, preview)

        # Parse command first (before session lookup). Strip leading <@id>
        # mentions so "<@bot> /config reset" 等同于 "/config reset"——SPEC §5.2
        # 已把入站 @ 统一成 <@id>，命令识别不该被 mention 前缀挡住。原 msg.content
        # 不改：history 与主聊天流程仍需要看到 <@bot> 标记。
        content_for_cmd = msg.content
        while m := re.match(r"^\s*<@\S+>\s*", content_for_cmd):
            content_for_cmd = content_for_cmd[m.end():]
        cmd = content_for_cmd.strip().lower()
        cmd_parts = content_for_cmd.strip().split(maxsplit=1)
        cmd_name = cmd_parts[0].lower() if cmd_parts else ""
        cmd_arg = cmd_parts[1] if len(cmd_parts) > 1 else ""

        # In groups, commands with side effects must require @bot — otherwise anyone in
        # the group could fire /new, /config reset, /session etc. /help is read-only so
        # we allow it without @. /stop is handled even earlier in agent.run() and is
        # intentionally permissive (emergency brake).
        should_respond = msg.metadata.get("respond", True)
        _SAFE_NO_AT = {"/help"}
        if (
            not should_respond
            and cmd_name.startswith("/")
            and cmd_name not in _SAFE_NO_AT
        ):
            cmd = ""
            cmd_name = ""
            cmd_arg = ""

        # Check session binding
        binding_key = f"{msg.channel}:{msg.chat_id}"
        bound_key = self._session_bindings.get(binding_key)

        # /session command (always allowed, no session needed)
        if cmd_name == "/session":
            if not cmd_arg:
                # Show current binding status
                if bound_key:
                    return OutboundMessage(
                        channel=msg.channel, chat_id=msg.chat_id,
                        content=f"Current session: `{bound_key}`\n\n"
                                f"Usage:\n"
                                f"• `/session <id>` — bind to session\n"
                                f"• `/session unbind` — unbind session"
                    )
                else:
                    return OutboundMessage(
                        channel=msg.channel, chat_id=msg.chat_id,
                        content="No session bound.\n\n"
                                "Usage: `/session <id>` — bind to session"
                    )
            elif cmd_arg.lower() == "unbind":
                self._session_bindings.pop(binding_key, None)
                save_session_bindings(self.data_dir, self._session_bindings)
                return OutboundMessage(
                    channel=msg.channel, chat_id=msg.chat_id,
                    content="Session unbound."
                )
            else:
                # Bind to specified session and create it if new
                new_session_key = cmd_arg.strip()
                self._session_bindings[binding_key] = new_session_key
                save_session_bindings(self.data_dir, self._session_bindings)
                # Create session directory immediately
                new_session = self.sessions.get_or_create(new_session_key)
                self.sessions.save(new_session)
                return OutboundMessage(
                    channel=msg.channel, chat_id=msg.chat_id,
                    content=f"Bound to session: `{new_session_key}`"
                )

        # A privileged lifecycle command must be deterministic and must not be
        # exposed as an LLM tool. It is handled before session binding because
        # recovering the gateway is independent from conversation state.
        if cmd_name == "/upgrade":
            if cmd != "/upgrade":
                return OutboundMessage(
                    channel=msg.channel,
                    chat_id=msg.chat_id,
                    content="热升级命令不接受参数，请单独发送 `/upgrade`。",
                )
            if self.gateway_control is None:
                return OutboundMessage(
                    channel=msg.channel,
                    chat_id=msg.chat_id,
                    content="当前进程不支持热升级。",
                )
            decision = self.gateway_control.prepare_upgrade(msg)
            metadata = dict(msg.metadata or {})
            metadata.update(decision.metadata)
            return OutboundMessage(
                channel=msg.channel,
                chat_id=msg.chat_id,
                content=decision.content,
                metadata=metadata,
            )

        # Require session binding before chatting (except CLI)
        if msg.channel != "cli" and not bound_key:
            if not should_respond:
                # Stay silent in groups when bot wasn't @'d and there's no session yet —
                # don't reply "请先 /session", don't create a stray session either.
                return None
            return OutboundMessage(
                channel=msg.channel, chat_id=msg.chat_id,
                content="⚠️ 无权限\n\n请先使用/session <name> 绑定工作区。"
            )

        # Now determine session key and create session
        if session_key is not None:
            key = session_key
        elif bound_key:
            key = bound_key
        else:
            key = msg.session_key  # CLI fallback
        logger.debug("Session lookup: binding_key={}, bound={}, explicit={}, final={}",
                     binding_key, bound_key, session_key, key)
        session = self.sessions.get_or_create(key)
        # 清理历史图片（每次对话入口只执行一次）
        prune_old_images_inplace(session.messages)

        # /config command - manage session-specific configuration
        if cmd_name == "/config":
            return await handle_config_command(self, msg, session, cmd_arg)

        # Retired command: do not send it to the model and recreate the feature.
        if cmd_name == "/heartbeat":
            return OutboundMessage(
                channel=msg.channel, chat_id=msg.chat_id,
                content="心跳功能已移除。需要定时执行时，请使用 cron 定时任务。",
            )

        if cmd == "/new":
            session.clear()
            self.sessions.save(session)
            self.sessions.invalidate(session.key)
            return OutboundMessage(channel=msg.channel, chat_id=msg.chat_id,
                                  content="New session started.")

        if cmd == "/compact":
            result = await do_compact(self,session)
            if result.success:
                self.sessions.save(session)
                return OutboundMessage(
                    channel=msg.channel, chat_id=msg.chat_id,
                    content=f"Compacted: {result.messages_compacted} messages summarized, "
                            f"{result.tokens_before} → {result.tokens_after} tokens"
                            + (" (fallback)" if result.used_fallback else "")
                )
            return OutboundMessage(
                channel=msg.channel, chat_id=msg.chat_id,
                content=f"Nothing to compact: {result.error or 'not enough messages'}"
            )
        if cmd == "/help":
            help_text = """/new
  开始一段新对话
/compact
  压缩历史
/stop
  中止当前任务
/session [<id>|unbind]
  绑定 / 解绑会话
/config [<key> <value>|reset]
  key: model | system_prompt | sandbox；reset 还原默认
/upgrade
  由管理员在微信私聊中触发受控升级并重启"""
            return OutboundMessage(channel=msg.channel, chat_id=msg.chat_id, content=help_text)

        # should_respond was computed at the top of this method (around the command-gate).
        # If not responding, just save the message to history and return.
        if not should_respond:
            # Move media files to session directory if present
            media = relocate_media_to_session(msg.media, session) if msg.media else None

            # Save as a simple user message to session history
            from datetime import datetime
            sender_name = msg.metadata.get("sender_name") or msg.sender_id
            content = f"[{sender_name}]: {msg.content}"
            # Append media paths to content if present
            if media:
                media_desc = ", ".join(f"<work_dir>/data/{Path(p).name}" for p in media)
                content = f"{content}\n[媒体文件: {media_desc}]"
            entry = {
                "role": "user",
                "content": content,
                "timestamp": datetime.now().isoformat(),
            }
            session.messages.append(entry)
            self.sessions.save(session)
            logger.debug("Saved non-respond message to history: {}", msg.content[:50])
            return None

        # Move media files to correct session directory if needed (handles session binding)
        media = relocate_media_to_session(msg.media, session) if msg.media else None

        sandbox = await self._sandboxes.get(key, session.directory, session.config.sandbox)
        turn_tools = self.build_turn_tools(
            channel=msg.channel, chat_id=msg.chat_id,
            message_id=msg.metadata.get("message_id"),
            session_dir=session.directory, session_key=key,
            session=session, sandbox=sandbox,
            system_triggered=system_triggered,
        )
        if on_message is not None:
            message_tool = turn_tools.get("message")
            if isinstance(message_tool, MessageTool):
                message_tool.set_send_callback(on_message)

        context = self._get_context(session)
        history = session.get_history(max_messages=self.memory_window)
        initial_messages = context.build_messages(
            history=history,
            current_message=msg.content,
            media=media,
            channel=msg.channel, chat_id=msg.chat_id,
            sender_id=msg.sender_id,
            sender_name=msg.metadata.get("sender_name"),
            system_prompt_extra=session.config.system_prompt,
            session_bindings=get_bindings_for_session(self._session_bindings, session.key),
            work_dir=sandbox.root,
        )
        turn_state = TurnState.from_context_messages(initial_messages, len(history))

        try:
            await self._run_agent_loop(
                initial_messages,
                on_progress=on_progress,
                session=session,
                on_tool_call=on_tool_call,
                tools=turn_tools,
                turn_state=turn_state,
                require_explicit_end=True,
            )
        except BaseException:
            try:
                persist_failed_turn(session, turn_state)
                self.sessions.save(session)
            except Exception as persist_error:
                logger.exception("Failed to persist interrupted turn: {}", persist_error)
            raise

        persist_turn_state(session, turn_state)
        self.sessions.save(session)

        # Conversational output has already gone through the shared sender. Never append a
        # final answer or a fallback merely because the model stopped generating.
        return None

    async def process_direct(
        self,
        content: str,
        session_key: str | None = None,
        channel: str = "cli",
        chat_id: str = "direct",
        on_progress: Callable[[str], Awaitable[None]] | None = None,
        on_tool_call: Callable[[str, dict, str | None], Awaitable[None]] | None = None,
        system_triggered: bool = False,
    ) -> tuple[str, list[str]]:
        """Process a message directly (for CLI or cron usage).

        If session_key is None, the session will be determined by:
        1. User's session binding (if exists)
        2. Default: f"{channel}:{chat_id}"

        ``system_triggered=True`` keeps the same schemas but makes cron return
        an explanatory error (no recursive scheduling, SPEC §5.6).

        Returns ``(response_text, tools_used)``. ``tools_used`` lists tool names
        invoked during the turn. response_text contains submissions to the input
        destination (or a command response), including ordinary assistant text.
        Non-CLI submissions are already published; callers must not resend them.
        """
        await self._connect_mcp()
        msg = InboundMessage(channel=channel, sender_id="user", chat_id=chat_id, content=content)

        # Capture which tools ran, while still forwarding to any user-provided on_tool_call.
        tools_used: list[str] = []
        sent_content: list[str] = []

        async def _send(message: OutboundMessage) -> None:
            current_target = message.channel == channel and message.chat_id == chat_id
            if current_target:
                if message.content:
                    sent_content.append(message.content)
                sent_content.extend(f"[附件: {path}]" for path in message.media)
            # Direct CLI has no bus consumer; return its messages below.
            if not (current_target and channel == "cli"):
                await self.bus.publish_outbound(message)

        async def _capture(name: str, args: dict, result: str | None) -> None:
            if on_tool_call is not None:
                await on_tool_call(name, args, result)
            if result is not None:
                tools_used.append(name)

        # Cron/direct entry points share history and destination with chat turns.
        # Use the same lock before constructing tools or loading routing state.
        async with self._session_lock(self._resolve_session_key(msg, session_key)):
            response = await self._process_message(
                msg, session_key=session_key, on_progress=on_progress, on_tool_call=_capture,
                system_triggered=system_triggered,
                on_message=_send,
            )
        return (response.content if response else "\n\n".join(sent_content)), tools_used
