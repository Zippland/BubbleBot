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
from bubbles.bus.events import InboundMessage, OutboundMessage
from bubbles.bus.queue import MessageBus
from bubbles.providers.base import LLMProvider
from bubbles.session.manager import (
    Session,
    SessionManager,
    cleanup_data_dir,
    prune_old_images_inplace,
)

if TYPE_CHECKING:
    from bubbles.config.schema import ChannelsConfig, ExecToolConfig
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
        cron_service: CronService | None = None,
        session_manager: SessionManager | None = None,
        channel_manager: Any = None,
        provider_factory: Callable[[str], tuple[str, LLMProvider]] | None = None,
        default_provider_name: str | None = None,
        mcp_servers: dict | None = None,
        channels_config: ChannelsConfig | None = None,
        # Auto-compaction settings
        compact_threshold: float = 0.85,
        compact_keep_recent: int = 20,
        compact_min_messages: int = 5,
    ):
        from bubbles.config.schema import ExecToolConfig
        from bubbles.utils.helpers import get_data_path
        self.bus = bus
        self.channels_config = channels_config
        self.provider = provider
        self.data_dir = get_data_path()  # ~/.bubbles/
        self.model = model or provider.get_default_model()
        self.max_iterations = max_iterations
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.memory_window = memory_window
        self.context_limit = context_limit
        self.tavily_api_key = tavily_api_key
        self.exec_config = exec_config or ExecToolConfig()
        self.cron_service = cron_service
        self.channel_manager = channel_manager
        self.provider_factory = provider_factory
        self.default_provider_name = default_provider_name
        self._provider_cache: dict[str, LLMProvider] = {}
        if provider is not None and default_provider_name:
            self._provider_cache[default_provider_name] = provider

        # Auto-compaction settings
        self.compact_threshold = compact_threshold
        self.compact_keep_recent = compact_keep_recent
        self.compact_min_messages = compact_min_messages

        self._context_cache: dict[str, ContextBuilder] = {}  # session_key -> ContextBuilder
        self.sessions = session_manager or SessionManager()
        self.tools = ToolRegistry()
        self.subagents = SubagentManager(
            provider=provider,
            bus=bus,
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            tavily_api_key=tavily_api_key,
            exec_config=self.exec_config,
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
        self._processing_lock = asyncio.Lock()
        self.on_tool_call: Callable[[str, dict, str | None], Awaitable[None]] | None = None  # Debug callback
        self._register_default_tools()

    def _register_default_tools(self) -> None:
        """Register the default set of tools."""
        # File tools: session_dir is set dynamically via set_session_dir()
        for cls in (ReadFileTool, WriteFileTool, EditFileTool, ListDirTool):
            self.tools.register(cls())
        # Exec tool: working_dir is set per session via set_session_dir()
        self.tools.register(ExecTool(
            timeout=self.exec_config.timeout,
            path_append=self.exec_config.path_append,
        ))
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
        # Task tools: session is set dynamically via _set_tool_context()
        for cls in (TaskListTool, TaskGetTool, TaskCreateTool, TaskUpdateTool):
            self.tools.register(cls())

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

    def _set_tool_context(
        self, channel: str, chat_id: str, message_id: str | None = None,
        session_dir: Path | None = None, session_key: str | None = None,
        session: Session | None = None,
    ) -> None:
        """Update context for all tools that need routing info."""
        for name in ("message", "spawn", "cron", "find_person"):
            if tool := self.tools.get(name):
                if hasattr(tool, "set_context"):
                    if name == "message":
                        tool.set_context(channel, chat_id, message_id)
                    elif name in ("cron", "spawn"):
                        tool.set_context(channel, chat_id, session_key or f"{channel}:{chat_id}")
                    else:
                        tool.set_context(channel, chat_id)

        # Set session directory for file, exec, spawn, and message tools
        for tool_name in ("read_file", "write_file", "edit_file", "list_dir", "exec", "spawn", "message"):
            if tool := self.tools.get(tool_name):
                if hasattr(tool, "set_session_dir"):
                    tool.set_session_dir(session_dir)

        # Set session for task tools
        if session:
            for tool_name in ("task_list", "task_get", "task_create", "task_update"):
                if tool := self.tools.get(tool_name):
                    if hasattr(tool, "set_session"):
                        tool.set_session(session)


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

    async def _run_agent_loop(
        self,
        initial_messages: list[dict],
        on_progress: Callable[..., Awaitable[None]] | None = None,
        session: Session | None = None,
        should_stop: Callable[[], bool] | None = None,
        on_tool_call: Callable[[str, dict, str | None], Awaitable[None]] | None = None,
    ) -> tuple[str | None, list[str], list[dict]]:
        """Run the agent iteration loop. Returns (final_content, tools_used, messages)."""
        messages = initial_messages
        iteration = 0
        final_content = None
        tools_used: list[str] = []

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

            # Auto-compaction: check if context is overflowing (pre-call estimation)
            if should_compact(self,messages):
                messages = await mid_loop_compact(self, session, messages, on_progress)

            response = await self._provider_for(model).chat(
                messages=messages,
                tools=self.tools.get_definitions(),
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
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
                    result = await self.tools.execute(tool_call.name, tool_call.arguments)
                    if on_tool_call:
                        await on_tool_call(tool_call.name, tool_call.arguments, result)
                    messages = context.add_tool_result(
                        messages, tool_call.id, tool_call.name, result
                    )

                # Check for duplicate message sends (loop detection)
                if message_tool := self.tools.get("message"):
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
            else:
                task = asyncio.create_task(self._dispatch(msg))
                self._active_tasks.setdefault(msg.session_key, []).append(task)
                task.add_done_callback(lambda t, k=msg.session_key: self._active_tasks.get(k, []) and self._active_tasks[k].remove(t) if t in self._active_tasks.get(k, []) else None)

    async def _handle_stop(self, msg: InboundMessage) -> None:
        """Cancel all active tasks and subagents for the session."""
        tasks = self._active_tasks.pop(msg.session_key, [])
        cancelled = sum(1 for t in tasks if not t.done() and t.cancel())
        for t in tasks:
            try:
                await t
            except (asyncio.CancelledError, Exception):
                pass
        sub_cancelled = await self.subagents.cancel_by_session(msg.session_key)
        total = cancelled + sub_cancelled
        content = f"⏹ Stopped {total} task(s)." if total else "No active task to stop."
        await self.bus.publish_outbound(OutboundMessage(
            channel=msg.channel, chat_id=msg.chat_id, content=content,
        ))

    async def _dispatch(self, msg: InboundMessage) -> None:
        """Process a message under the global lock."""
        self._maybe_cleanup_session_data(msg.session_key)
        async with self._processing_lock:
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
                logger.info("Task cancelled for session {}", msg.session_key)
                raise
            except Exception:
                logger.exception("Error processing message for session {}", msg.session_key)
                await self._emit_error_reply(msg)

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

    async def _emit_error_reply(self, msg: InboundMessage) -> None:
        """Send a user-visible error reply per SPEC §5.1 error policy.

        Group chats stay silent; private chats get one fixed message, throttled per
        session_key. CLI always publishes something so the interactive turn can finish.
        """
        is_group = bool(msg.metadata and (
            msg.metadata.get("is_group")
            or msg.metadata.get("chat_type") == "group"
        ))
        now = time.monotonic()
        throttled = (now - self._last_error_reply_at.get(msg.session_key, 0.0)) < ERROR_REPLY_THROTTLE_SEC

        if not is_group and not throttled:
            self._last_error_reply_at[msg.session_key] = now
            await self.bus.publish_outbound(OutboundMessage(
                channel=msg.channel, chat_id=msg.chat_id,
                content="Sorry, I encountered an error.",
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
  key: model | system_prompt；reset 还原默认
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

        self._set_tool_context(msg.channel, msg.chat_id, msg.metadata.get("message_id"), session.directory, key, session)
        if message_tool := self.tools.get("message"):
            if isinstance(message_tool, MessageTool):
                message_tool.start_turn()

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
        )

        if final_content is None:
            final_content = "I've completed processing but have no response to give."

        save_turn(session, all_msgs, 1 + len(history))
        self.sessions.save(session)

        if (mt := self.tools.get("message")) and isinstance(mt, MessageTool) and mt._sent_in_turn:
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
    ) -> tuple[str, list[str]]:
        """Process a message directly (for CLI or cron usage).

        If session_key is None, the session will be determined by:
        1. User's session binding (if exists)
        2. Default: f"{channel}:{chat_id}"

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
        )
        return (response.content if response else ""), tools_used
