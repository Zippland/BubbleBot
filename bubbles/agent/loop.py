"""Agent loop: the core processing engine."""

from __future__ import annotations

import asyncio
import json
import re
from contextlib import AsyncExitStack
from pathlib import Path
from typing import TYPE_CHECKING, Any, Awaitable, Callable

from loguru import logger

from bubbles.agent.compaction import compact_session, CompactionResult, TokenTracker
from bubbles.agent.context import ContextBuilder
from bubbles.agent.subagent import SubagentManager
from bubbles.agent.tools.cron import CronTool
from bubbles.agent.tools.filesystem import EditFileTool, ListDirTool, ReadFileTool, WriteFileTool
from bubbles.agent.tools.message import MessageTool
from bubbles.agent.tools.registry import ToolRegistry
from bubbles.agent.tools.shell import ExecTool
from bubbles.agent.tools.spawn import SpawnTool
from bubbles.agent.tools.web import WebFetchTool, WebSearchTool
from bubbles.bus.events import InboundMessage, OutboundMessage
from bubbles.bus.queue import MessageBus
from bubbles.providers.base import LLMProvider
from bubbles.session.manager import Session, SessionConfig, SessionManager

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

    _TOOL_RESULT_MAX_CHARS = 500

    def __init__(
        self,
        bus: MessageBus,
        provider: LLMProvider,
        model: str | None = None,
        max_iterations: int = 40,
        temperature: float = 0.1,
        max_tokens: int,
        memory_window: int,
        context_limit: int,
        tavily_api_key: str | None = None,
        exec_config: ExecToolConfig | None = None,
        cron_service: CronService | None = None,
        session_manager: SessionManager | None = None,
        mcp_servers: dict | None = None,
        channels_config: ChannelsConfig | None = None,
        # Auto-compaction settings
        compact_threshold: float = 0.85,
        compact_keep_recent: int = 10,
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

        # Auto-compaction: use max_tokens as output reserve
        self.compact_threshold = compact_threshold
        self.compact_keep_recent = compact_keep_recent
        self.compact_min_messages = compact_min_messages
        self.token_tracker = TokenTracker(
            context_limit=context_limit,
            output_reserve=max_tokens,  # Reserve space for output
        )

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
        self._session_bindings: dict[str, str] = {}  # {channel}:{chat_id} -> custom session key
        self._load_session_bindings()
        self._active_tasks: dict[str, list[asyncio.Task]] = {}  # session_key -> tasks
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

    def _get_context(self, session: Session) -> ContextBuilder:
        """Get ContextBuilder for a session (cached per session key)."""
        if session.key not in self._context_cache:
            self._context_cache[session.key] = ContextBuilder(session_dir=session.directory)
        return self._context_cache[session.key]

    def _get_bindings_path(self) -> Path:
        """Get path to session bindings file."""
        return self.data_dir / "session_bindings.json"

    def _load_session_bindings(self) -> None:
        """Load session bindings from file."""
        path = self._get_bindings_path()
        if path.exists():
            try:
                with open(path, encoding="utf-8") as f:
                    self._session_bindings = json.load(f)
                logger.debug("Loaded {} session bindings", len(self._session_bindings))
            except Exception as e:
                logger.warning("Failed to load session bindings: {}", e)
                self._session_bindings = {}

    def _save_session_bindings(self) -> None:
        """Save session bindings to file."""
        path = self._get_bindings_path()
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(self._session_bindings, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.warning("Failed to save session bindings: {}", e)

    def _get_bindings_for_session(self, session_key: str) -> list[str]:
        """Get all channel:chat_id pairs bound to a session."""
        return [k for k, v in self._session_bindings.items() if v == session_key]

    def _relocate_media_to_session(
        self, media_paths: list[str], session: Session
    ) -> list[str]:
        """Move media files to the correct session directory if needed.

        Channel layer may download media before knowing the final session binding.
        This method moves files to session.directory/data/ if they're elsewhere.

        Returns:
            Updated list of media paths.
        """
        if not media_paths or not session.directory:
            return media_paths

        import shutil
        target_dir = session.directory / "data"
        target_dir.mkdir(parents=True, exist_ok=True)

        updated = []
        for path in media_paths:
            p = Path(path)
            if not p.is_file():
                updated.append(path)
                continue
            # Check if already in the correct directory
            try:
                p.relative_to(target_dir)
                updated.append(path)  # Already correct
            except ValueError:
                # Move to correct directory
                new_path = target_dir / p.name
                # Handle name collision
                if new_path.exists():
                    stem, suffix = new_path.stem, new_path.suffix
                    for i in range(1, 100):
                        new_path = target_dir / f"{stem}_{i}{suffix}"
                        if not new_path.exists():
                            break
                try:
                    shutil.move(str(p), str(new_path))
                    logger.debug("Moved media {} -> {}", p, new_path)
                    updated.append(str(new_path))
                except Exception as e:
                    logger.warning("Failed to move media {}: {}", p, e)
                    updated.append(path)  # Keep original on failure
        return updated

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
        session_dir: Path | None = None, session_key: str | None = None
    ) -> None:
        """Update context for all tools that need routing info."""
        for name in ("message", "spawn", "cron"):
            if tool := self.tools.get(name):
                if hasattr(tool, "set_context"):
                    if name == "message":
                        tool.set_context(channel, chat_id, message_id)
                    elif name == "cron":
                        tool.set_context(channel, chat_id, session_key or f"{channel}:{chat_id}")
                    else:
                        tool.set_context(channel, chat_id)

        # Set session directory for file, exec, spawn, and message tools
        for tool_name in ("read_file", "write_file", "edit_file", "list_dir", "exec", "spawn", "message"):
            if tool := self.tools.get(tool_name):
                if hasattr(tool, "set_session_dir"):
                    tool.set_session_dir(session_dir)

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

            # Auto-compaction: check if context is overflowing
            if self.token_tracker.is_overflow(threshold=self.compact_threshold):
                messages = await self._mid_loop_compact(session, messages, on_progress)

            response = await self.provider.chat(
                messages=messages,
                tools=self.tools.get_definitions(),
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
            )

            # Update token tracker from response usage
            if hasattr(response, "usage") and response.usage:
                self.token_tracker.update(response.usage)

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
                await self.bus.publish_outbound(OutboundMessage(
                    channel=msg.channel, chat_id=msg.chat_id,
                    content="Sorry, I encountered an error.",
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
        # System messages: parse origin from chat_id ("channel:chat_id")
        if msg.channel == "system":
            channel, chat_id = (msg.chat_id.split(":", 1) if ":" in msg.chat_id
                                else ("cli", msg.chat_id))
            logger.info("Processing system message from {}", msg.sender_id)
            key = f"{channel}:{chat_id}"
            session = self.sessions.get_or_create(key)
            context = self._get_context(session)
            self._set_tool_context(channel, chat_id, msg.metadata.get("message_id"), session.directory, key)
            history = session.get_history(max_messages=self.memory_window)
            messages = context.build_messages(
                history=history,
                current_message=msg.content, channel=channel, chat_id=chat_id,
                sender_id=msg.sender_id,
                sender_name=msg.metadata.get("sender_name"),
                system_prompt_extra=session.config.system_prompt,
                session_bindings=self._get_bindings_for_session(session.key),
            )
            final_content, _, all_msgs = await self._run_agent_loop(
                messages, session=session, on_tool_call=on_tool_call)
            self._save_turn(session, all_msgs, 1 + len(history))
            self.sessions.save(session)
            return OutboundMessage(channel=channel, chat_id=chat_id,
                                  content=final_content or "Background task completed.")

        preview = msg.content[:80] + "..." if len(msg.content) > 80 else msg.content
        logger.info("Processing message from {}:{}: {}", msg.channel, msg.sender_id, preview)

        # Parse command first (before session lookup)
        cmd = msg.content.strip().lower()
        cmd_parts = msg.content.strip().split(maxsplit=1)
        cmd_name = cmd_parts[0].lower() if cmd_parts else ""
        cmd_arg = cmd_parts[1] if len(cmd_parts) > 1 else ""

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
                self._save_session_bindings()
                return OutboundMessage(
                    channel=msg.channel, chat_id=msg.chat_id,
                    content="Session unbound."
                )
            else:
                # Bind to specified session and create it if new
                new_session_key = cmd_arg.strip()
                self._session_bindings[binding_key] = new_session_key
                self._save_session_bindings()
                # Create session directory immediately
                new_session = self.sessions.get_or_create(new_session_key)
                self.sessions.save(new_session)
                return OutboundMessage(
                    channel=msg.channel, chat_id=msg.chat_id,
                    content=f"Bound to session: `{new_session_key}`"
                )

        # Require session binding before chatting (except CLI)
        if msg.channel != "cli" and not bound_key:
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

        # /config command - manage session-specific configuration
        if cmd_name == "/config":
            return await self._handle_config_command(msg, session, cmd_arg)

        if cmd == "/new":
            session.clear()
            self.sessions.save(session)
            self.sessions.invalidate(session.key)
            return OutboundMessage(channel=msg.channel, chat_id=msg.chat_id,
                                  content="New session started.")

        if cmd == "/compact":
            result = await self._do_compact(session)
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
            return OutboundMessage(channel=msg.channel, chat_id=msg.chat_id,
                                  content="🫧 bubbles commands:\n"
                                          "/new — Start a new conversation\n"
                                          "/compact — Compress conversation history\n"
                                          "/stop — Stop the current task\n"
                                          "/session — View/switch session\n"
                                          "/config — View/modify session config\n"
                                          "/help — Show available commands")

        # Check if we should respond (default True for backward compatibility)
        should_respond = msg.metadata.get("respond", True)

        # If not responding, just save the message to history and return
        if not should_respond:
            # Move media files to session directory if present
            media = self._relocate_media_to_session(msg.media, session) if msg.media else None

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
        media = self._relocate_media_to_session(msg.media, session) if msg.media else None

        self._set_tool_context(msg.channel, msg.chat_id, msg.metadata.get("message_id"), session.directory, key)
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
            session_bindings=self._get_bindings_for_session(session.key),
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

        self._save_turn(session, all_msgs, 1 + len(history))
        self.sessions.save(session)

        if (mt := self.tools.get("message")) and isinstance(mt, MessageTool) and mt._sent_in_turn:
            return None

        preview = final_content[:120] + "..." if len(final_content) > 120 else final_content
        logger.info("Response to {}:{}: {}", msg.channel, msg.sender_id, preview)
        return OutboundMessage(
            channel=msg.channel, chat_id=msg.chat_id, content=final_content,
            metadata=msg.metadata or {},
        )

    async def _handle_config_command(
        self, msg: InboundMessage, session: Session, cmd_arg: str
    ) -> OutboundMessage:
        """Handle /config command for session-specific configuration."""
        cfg = session.config

        if not cmd_arg:
            # Show current session config
            lines = [f"Session: `{session.key}`", "", "**Current config:**"]
            lines.append(f"• model: `{cfg.model or '(default: ' + self.model + ')'}`")
            lines.append(f"• temperature: `{cfg.temperature if cfg.temperature is not None else '(default: ' + str(self.temperature) + ')'}`")
            lines.append(f"• max_tokens: `{cfg.max_tokens or '(default: ' + str(self.max_tokens) + ')'}`")
            if cfg.system_prompt:
                preview = cfg.system_prompt[:50] + "..." if len(cfg.system_prompt) > 50 else cfg.system_prompt
                lines.append(f"• system_prompt: `{preview}`")
            else:
                lines.append("• system_prompt: `(default)`")
            lines.append("")
            lines.append("**Usage:**")
            lines.append("• `/config <key> <value>` — Set config")
            lines.append("• `/config reset` — Reset to defaults")
            lines.append("")
            lines.append("**Keys:** model, temperature, max_tokens, system_prompt")
            return OutboundMessage(channel=msg.channel, chat_id=msg.chat_id, content="\n".join(lines))

        parts = cmd_arg.split(maxsplit=1)
        key = parts[0].lower()
        value = parts[1] if len(parts) > 1 else ""

        if key == "reset":
            session.config = SessionConfig()
            self.sessions.save(session)
            return OutboundMessage(
                channel=msg.channel, chat_id=msg.chat_id,
                content="Config reset to defaults."
            )

        if key == "model":
            cfg.model = value if value else None
            self.sessions.save(session)
            return OutboundMessage(
                channel=msg.channel, chat_id=msg.chat_id,
                content=f"model = `{value}`" if value else "model reset to default"
            )

        if key == "temperature":
            try:
                cfg.temperature = float(value) if value else None
                self.sessions.save(session)
                return OutboundMessage(
                    channel=msg.channel, chat_id=msg.chat_id,
                    content=f"temperature = `{value}`" if value else "temperature reset to default"
                )
            except ValueError:
                return OutboundMessage(
                    channel=msg.channel, chat_id=msg.chat_id,
                    content=f"Invalid temperature: `{value}` (must be a number)"
                )

        if key == "max_tokens":
            try:
                cfg.max_tokens = int(value) if value else None
                self.sessions.save(session)
                return OutboundMessage(
                    channel=msg.channel, chat_id=msg.chat_id,
                    content=f"max_tokens = `{value}`" if value else "max_tokens reset to default"
                )
            except ValueError:
                return OutboundMessage(
                    channel=msg.channel, chat_id=msg.chat_id,
                    content=f"Invalid max_tokens: `{value}` (must be an integer)"
                )

        if key == "system_prompt":
            cfg.system_prompt = value if value else None
            self.sessions.save(session)
            return OutboundMessage(
                channel=msg.channel, chat_id=msg.chat_id,
                content=f"system_prompt updated" if value else "system_prompt reset to default"
            )

        return OutboundMessage(
            channel=msg.channel, chat_id=msg.chat_id,
            content=f"Unknown config key: `{key}`\n\nValid keys: provider, model, temperature, max_tokens, system_prompt"
        )

    def _save_turn(self, session: Session, messages: list[dict], skip: int) -> None:
        """Save new-turn messages into session, truncating large tool results."""
        from datetime import datetime
        for m in messages[skip:]:
            entry = {k: v for k, v in m.items() if k != "reasoning_content"}
            if entry.get("role") == "tool":
                content = entry.get("content")
                if isinstance(content, str):
                    if len(content) > self._TOOL_RESULT_MAX_CHARS:
                        entry["content"] = content[:self._TOOL_RESULT_MAX_CHARS] + "\n... (truncated)"
                elif isinstance(content, list):
                    # Tool returned image content (e.g., view_image), strip base64
                    text_parts = [c for c in content if c.get("type") == "text"]
                    entry["content"] = text_parts[0].get("text", "[image viewed]") if text_parts else "[image viewed]"
            if entry.get("role") == "user" and isinstance(entry.get("content"), list):
                # Remove base64 images, keep text (which contains path descriptions)
                text_parts = [
                    c for c in entry["content"]
                    if c.get("type") == "text"
                ]
                # Simplify to string if only text remains
                if len(text_parts) == 1:
                    entry["content"] = text_parts[0].get("text", "")
                elif text_parts:
                    entry["content"] = text_parts
                else:
                    entry["content"] = "[media]"
            entry.setdefault("timestamp", datetime.now().isoformat())
            session.messages.append(entry)
        session.updated_at = datetime.now()

    async def _do_compact(self, session: Session) -> CompactionResult:
        """Compact session history using LLM-powered summarization."""
        return await compact_session(
            session=session,
            provider=self.provider,
            model=self.model,
            keep_recent=self.compact_keep_recent,
            min_messages_to_compact=self.compact_min_messages,
            use_fallback_on_failure=True,
        )

    async def _mid_loop_compact(
        self,
        session: Session,
        current_messages: list[dict[str, Any]],
        on_progress: Callable[..., Awaitable[None]] | None = None,
    ) -> list[dict[str, Any]]:
        """Execute compaction mid-loop, return rebuilt messages."""
        logger.info("Mid-loop compaction triggered for session {}", session.key)

        # 1. Save current progress to session (skip system messages)
        first_non_system = 0
        for i, m in enumerate(current_messages):
            if m.get("role") != "system":
                first_non_system = i
                break

        # Append new messages from current loop to session
        history_len = len(session.get_history(max_messages=self.memory_window))
        new_messages = current_messages[first_non_system + history_len:]
        for m in new_messages:
            if m.get("role") != "system":
                session.messages.append(m)

        # 2. Execute compaction
        try:
            result = await self._do_compact(session)
            if result.success:
                msg = "Context overflow, history compacted" if not result.used_fallback else "Context overflow, history truncated"
                if on_progress:
                    await on_progress(f"{msg} ({result.tokens_before} → {result.tokens_after} tokens)")
                logger.info("Mid-loop compaction successful: {} -> {} tokens", result.tokens_before, result.tokens_after)
                self.sessions.save(session)
            else:
                logger.warning("Mid-loop compaction skipped: {}", result.error)
        except Exception as e:
            logger.exception("Mid-loop compaction failed: {}", e)

        # 3. Rebuild messages: system prompt + new history + current query
        new_history = session.get_history(max_messages=self.memory_window)

        # Find current user query (last user message)
        current_query = None
        for m in reversed(current_messages):
            if m.get("role") == "user":
                current_query = m.get("content", "")
                break

        # Rebuild complete message list
        context = self._get_context(session)
        rebuilt = context.build_messages(
            history=new_history,
            current_message=current_query or "",
            channel="cli",
            chat_id="direct",
        )

        # Reset token tracker (compaction reduces token count)
        self.token_tracker.last_prompt_tokens = 0

        return rebuilt

    async def process_direct(
        self,
        content: str,
        session_key: str | None = None,
        channel: str = "cli",
        chat_id: str = "direct",
        on_progress: Callable[[str], Awaitable[None]] | None = None,
        on_tool_call: Callable[[str, dict, str | None], Awaitable[None]] | None = None,
    ) -> str:
        """Process a message directly (for CLI or cron usage).

        If session_key is None, the session will be determined by:
        1. User's session binding (if exists)
        2. Default: f"{channel}:{chat_id}"
        """
        await self._connect_mcp()
        msg = InboundMessage(channel=channel, sender_id="user", chat_id=chat_id, content=content)
        response = await self._process_message(
            msg, session_key=session_key, on_progress=on_progress, on_tool_call=on_tool_call)
        return response.content if response else ""
