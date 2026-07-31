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
from bubbles.agent.commands import (
    build_heartbeat_info,
    handle_config_command,
    handle_heartbeat_command,
)
from bubbles.agent.context import ContextBuilder
from bubbles.agent.turn import (
    do_compact,
    mid_loop_compact,
    process_system_message,
    save_turn,
    should_compact,
)
from bubbles.agent.subagent import SubagentManager
from bubbles.agent.tools.cron import CronTool
from bubbles.agent.tools.filesystem import EditFileTool, ListDirTool, ReadFileTool, WriteFileTool
from bubbles.agent.tools.find_person import FindPersonTool
from bubbles.agent.tools.message import MessageTool
from bubbles.agent.tools.registry import ToolRegistry
from bubbles.agent.tools.shell import ExecTool
from bubbles.agent.tools.spawn import SpawnTool
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
        """Register the stateless, session-independent tools.

        These carry no per-turn state, so a single shared instance is safe even
        with several sessions running concurrently. Session-scoped tools are
        built per turn by :meth:`build_turn_tools`.

        ``self.tools`` remains the template registry: it backs
        ``get_definitions()`` (the schema list is identical for every session)
        and is where MCP servers register their tools on connect.
        """
        for cls in (ReadFileTool, WriteFileTool, EditFileTool, ListDirTool):
            self.tools.register(cls())
        self.tools.register(ExecTool(timeout=self.exec_config.timeout))
        self.tools.register(WebSearchTool(api_key=self.tavily_api_key))
        self.tools.register(WebFetchTool())
        self.tools.register(MessageTool(send_callback=self.bus.publish_outbound))
        self.tools.register(SpawnTool(manager=self.subagents))
        if self.cron_service:
            self.tools.register(CronTool(self.cron_service))
        if self.channel_manager is not None:
            find_person = FindPersonTool()
            find_person.set_channel_manager(self.channel_manager)
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
        message.set_session_dir(session_dir)
        message.start_turn()
        reg.register(message)

        spawn = SpawnTool(manager=self.subagents)
        spawn.set_context(channel, chat_id, session_key)
        spawn.set_session_dir(session_dir)
        reg.register(spawn)

        # System-triggered turns may not schedule further jobs (SPEC §5.6),
        # and get stay_silent so the model can opt out of delivery.
        if system_triggered:
            from bubbles.agent.tools.stay_silent import StaySilentTool
            reg.register(StaySilentTool())
        elif self.cron_service:
            cron = CronTool(self.cron_service)
            cron.set_context(channel, chat_id, session_key)
            reg.register(cron)

        if self.channel_manager is not None:
            find_person = FindPersonTool()
            find_person.set_channel_manager(self.channel_manager)
            find_person.set_context(channel, chat_id)
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
            "web_search", "web_fetch", "message", "spawn", "cron",
            "find_person", "task_list", "task_get", "task_create",
            "task_update", "stay_silent",
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
        return re.sub(r"<think>[\s\S]*?</think>", "", text).strip() or None

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
        tools: ToolRegistry | None = None,
        on_progress: Callable[..., Awaitable[None]] | None = None,
    ) -> tuple[Any, list[dict]]:
        """调一次 LLM，可重试的失败按类别退避重试。

        返回 ``(response, messages)`` —— messages 可能被 context_overflow 恢复
        路径重建过，调用方必须用返回的这份。

        重试放在这一层而不是 provider：只有这里能做 context_overflow 的恢复
        动作（压缩历史后重试），provider 看不到 session。
        """
        attempt = 0
        compacted_once = False

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
                    # 压缩一次就够：压完还超说明不是历史长度的问题，
                    # 再压只会把上下文越削越少却仍然失败。
                    if compacted_once:
                        logger.error(
                            "Context still overflowing after compaction for session {}", session.key,
                        )
                        raise
                    compacted_once = True
                    logger.warning("Context overflow for session {}; compacting and retrying", session.key)
                    messages = await mid_loop_compact(self, session, messages, on_progress)
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
    ) -> tuple[str | None, list[str], list[dict]]:
        """Run the agent iteration loop. Returns (final_content, tools_used, messages)."""
        messages = initial_messages
        iteration = 0
        final_content = None
        tools_used: list[str] = []
        # 每轮自己的工具集；缺省回落到模板 registry（测试与旧调用方）。
        tools = tools if tools is not None else self.tools

        # Get context for this session (session is required in new architecture)
        if not session:
            raise ValueError("Session is required for agent loop")
        context = self._get_context(session)

        # Use session config if available, otherwise use defaults
        cfg = session.config if session else None
        model = (cfg.model if cfg and cfg.model else self.model)
        temperature = (cfg.temperature if cfg and cfg.temperature is not None else self.temperature)
        max_tokens = (cfg.max_tokens if cfg and cfg.max_tokens else self.max_tokens)

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
            if injected := self._take_injections(session.key):
                messages = context.add_user_messages(messages, injected)
                logger.info(
                    "Injected {} mid-turn message(s) into session {}", len(injected), session.key,
                )

            # Auto-compaction: check if context is overflowing (pre-call estimation)
            if should_compact(self,messages):
                messages = await mid_loop_compact(self, session, messages, on_progress)

            response, messages = await self._chat_with_retry(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                session=session,
                tools=tools,
                on_progress=on_progress,
            )

            if response.has_tool_calls:
                if on_progress:
                    clean = self._strip_think(response.content)
                    if clean:
                        await on_progress(clean)
                    await on_progress(self._tool_hint(response.tool_calls), tool_hint=True)

                tool_call_dicts = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.name,
                            "arguments": json.dumps(tc.arguments, ensure_ascii=False)
                        }
                    }
                    for tc in response.tool_calls
                ]
                messages = context.add_assistant_message(
                    messages, response.content, tool_call_dicts,
                    reasoning_content=response.reasoning_content,
                )

                for tool_call in response.tool_calls:
                    tools_used.append(tool_call.name)
                    args_str = json.dumps(tool_call.arguments, ensure_ascii=False)
                    logger.info("Tool call: {}({})", tool_call.name, args_str[:200])
                    if on_tool_call:
                        await on_tool_call(tool_call.name, tool_call.arguments, None)
                    result = await tools.execute(tool_call.name, tool_call.arguments)
                    if on_tool_call:
                        await on_tool_call(tool_call.name, tool_call.arguments, result)
                    messages = context.add_tool_result(
                        messages, tool_call.id, tool_call.name, result
                    )

                # Check for duplicate message sends (loop detection)
                if message_tool := tools.get("message"):
                    if isinstance(message_tool, MessageTool) and message_tool._duplicate_detected:
                        logger.warning("Duplicate message detected, stopping agent loop")
                        final_content = None  # Already sent via message tool
                        break
            else:
                final_content = self._strip_think(response.content)
                # Add final assistant message to history
                if final_content:
                    messages = context.add_assistant_message(messages, final_content, None)
                break

        if final_content is None and iteration >= self.max_iterations:
            logger.warning("Max iterations ({}) reached", self.max_iterations)
            final_content = (
                f"I reached the maximum number of tool call iterations ({self.max_iterations}) "
                "without completing the task. You can try breaking the task into smaller steps."
            )

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

        while self._running:
            try:
                msg = await asyncio.wait_for(self.bus.consume_inbound(), timeout=1.0)
            except asyncio.TimeoutError:
                continue

            if msg.content.strip().lower() == "/stop":
                await self._handle_stop(msg)
                continue

            key = self._resolve_session_key(msg)

            # 该 session 正在跑 turn 且这条不是命令 → 插进当前 turn，而不是
            # 排队等它跑完（最多 40 轮迭代，用户可能要等几分钟）。
            if self._has_active_turn(key) and not self._is_command(msg):
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
        - 非用户触发（cron、心跳、subagent 汇报等 system turn，或群里没 @ 的旁听
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

    async def _process_message(
        self,
        msg: InboundMessage,
        session_key: str | None = None,
        on_progress: Callable[[str], Awaitable[None]] | None = None,
        on_tool_call: Callable[[str, dict, str | None], Awaitable[None]] | None = None,
        system_triggered: bool = False,
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

        # /heartbeat - user-controlled periodic auto-wake (AI cannot enable)
        if cmd_name == "/heartbeat":
            return handle_heartbeat_command(self, msg, session, key, cmd_arg)

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
/heartbeat [<间隔>|off]
  开启（30m / 2h…）/ 关闭定时唤醒"""
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

        # Entry compaction: check if context is overflowing before entering loop
        if should_compact(self,initial_messages):
            logger.info("Entry compaction triggered for session {}", session.key)
            await do_compact(self,session)
            self.sessions.save(session)
            # Rebuild messages with compacted history
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
                heartbeat_info=build_heartbeat_info(self.cron_service, session.key),
                work_dir=sandbox.root,
            )

        # Track progress messages to detect loops
        _sent_progress: set[str] = set()
        _progress_loop_detected = False

        async def _bus_progress(content: str, *, tool_hint: bool = False) -> None:
            nonlocal _progress_loop_detected
            # Skip duplicate progress messages (loop detection)
            content_key = content[:100]
            if not tool_hint and content_key in _sent_progress:
                _progress_loop_detected = True
                logger.warning("Duplicate progress message detected, will stop loop")
                return
            if not tool_hint:
                _sent_progress.add(content_key)

            meta = dict(msg.metadata or {})
            meta["_progress"] = True
            meta["_tool_hint"] = tool_hint
            await self.bus.publish_outbound(OutboundMessage(
                channel=msg.channel, chat_id=msg.chat_id, content=content, metadata=meta,
            ))

        final_content, _, all_msgs = await self._run_agent_loop(
            initial_messages,
            on_progress=on_progress or _bus_progress,
            session=session,
            should_stop=lambda: _progress_loop_detected,
            on_tool_call=on_tool_call,
            tools=turn_tools,
        )

        if final_content is None:
            final_content = "I've completed processing but have no response to give."

        save_turn(session, all_msgs, 1 + len(history))
        self.sessions.save(session)

        if (mt := turn_tools.get("message")) and isinstance(mt, MessageTool) and mt._sent_in_turn:
            return None

        preview = final_content[:120] + "..." if len(final_content) > 120 else final_content
        logger.info("Response to {}:{}: {}", msg.channel, msg.sender_id, preview)
        return OutboundMessage(
            channel=msg.channel, chat_id=msg.chat_id, content=final_content,
            metadata=msg.metadata or {},
        )
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

        ``system_triggered=True`` marks a turn the user didn't ask for (cron /
        heartbeat): the model gains ``stay_silent`` and loses ``cron`` (no
        recursive job creation, SPEC §5.6), and failures stay silent.

        Returns ``(response_text, tools_used)``. ``tools_used`` lists tool names
        invoked during the turn — callers can check for sentinel tools like
        ``stay_silent`` to suppress outbound delivery.
        """
        await self._connect_mcp()
        msg = InboundMessage(channel=channel, sender_id="user", chat_id=chat_id, content=content)

        # Capture which tools ran, while still forwarding to any user-provided on_tool_call.
        tools_used: list[str] = []

        async def _capture(name: str, args: dict, result: str | None) -> None:
            if on_tool_call is not None:
                await on_tool_call(name, args, result)
            if result is not None:
                tools_used.append(name)

        response = await self._process_message(
            msg, session_key=session_key, on_progress=on_progress, on_tool_call=_capture,
            system_triggered=system_triggered,
        )
        return (response.content if response else ""), tools_used
