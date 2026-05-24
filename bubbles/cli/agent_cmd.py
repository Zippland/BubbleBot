"""`bubbles agent` — single-shot and interactive REPL for the agent."""

from __future__ import annotations

import asyncio
import os
import signal

import typer

from bubbles import __logo__
from bubbles.agent.bindings import get_bindings_for_session, save_session_bindings
from bubbles.agent.turn import do_compact
from bubbles.cli._interactive import (
    _flush_pending_tty_input,
    _init_prompt_session,
    _is_exit_command,
    _print_agent_response,
    _read_interactive_input_async,
    _restore_terminal,
)
from bubbles.cli._providers import _make_provider, _make_provider_for_model
from bubbles.cli.commands import app, console


def _print_debug_info(agent_loop, session_id: str) -> None:
    """Print debug info: model config, tools, and system prompt (exactly as sent to API)."""
    from bubbles.agent.context import ContextBuilder
    from rich.markup import escape

    console.print("[dim]" + "─" * 60 + "[/dim]")

    # 1. Model & Config
    console.print("[bold]Model & Config:[/bold]")
    console.print(f"  model: {agent_loop.model}")
    console.print(f"  max_iterations: {agent_loop.max_iterations}")
    console.print(f"  temperature: {agent_loop.temperature}")
    console.print(f"  max_tokens: {agent_loop.max_tokens}")

    # 2. Registered Tools
    console.print("\n[bold]Registered Tools:[/bold]")
    for tool_def in agent_loop.tools.get_definitions():
        func = tool_def.get("function", {})
        name = func.get("name", "?")
        desc = func.get("description", "")
        params = func.get("parameters", {}).get("properties", {})
        param_names = ", ".join(params.keys()) if params else ""
        console.print(f"\n  [cyan]\\[{escape(name)}][/cyan]({escape(param_names)})")
        for line in desc.strip().split("\n"):
            console.print(f"    {escape(line)}")

    # 3. System Prompt (exactly as sent to API)
    console.print("\n[dim]" + "─" * 60 + "[/dim]")
    console.print(f"[bold]System Prompt[/bold] (session: {session_id})")
    console.print("[dim]" + "─" * 60 + "[/dim]")

    # Build messages exactly as sent to API
    session = agent_loop.sessions.get_or_create(session_id)
    context = ContextBuilder(session_dir=session.directory)
    messages = context.build_messages(
        history=[],
        current_message="(debug)",
        channel="cli",
        chat_id="direct",
        system_prompt_extra=session.config.system_prompt if session.config else None,
        session_bindings=get_bindings_for_session(agent_loop._session_bindings, session_id),
    )
    # Extract system prompt from messages
    system_prompt = messages[0].get("content", "") if messages else ""
    console.print(system_prompt, markup=False)

    console.print("[dim]" + "─" * 60 + "[/dim]\n")


@app.command()
def agent(
    message: str = typer.Option(None, "--message", "-m", help="Message to send to the agent"),
    session_id: str = typer.Option(None, "--session", "-s", help="Session ID (required for -m, or use /session in interactive mode)"),
    markdown: bool = typer.Option(True, "--markdown/--no-markdown", help="Render assistant output as Markdown"),
    logs: bool = typer.Option(False, "--logs/--no-logs", help="Show bubbles runtime logs during chat"),
    debug: bool = typer.Option(False, "--debug", "-d", help="Print tools and system prompt at startup"),
):
    """Interact with the agent directly."""
    from bubbles.config.loader import load_config, get_data_dir
    from bubbles.bus.queue import MessageBus
    from bubbles.agent.loop import AgentLoop
    from bubbles.cron.service import CronService
    from loguru import logger

    config = load_config()

    bus = MessageBus()
    provider = _make_provider(config)
    default_provider_name = config.get_provider_name(config.agents.defaults.model)
    provider_factory = lambda m: _make_provider_for_model(config, m)

    # Create cron service for tool usage (no callback needed for CLI unless running)
    cron_store_path = get_data_dir() / "cron" / "jobs.json"
    cron = CronService(cron_store_path)

    if logs:
        logger.enable("bubbles")
    else:
        logger.disable("bubbles")

    agent_loop = AgentLoop(
        bus=bus,
        provider=provider,
        provider_factory=provider_factory,
        default_provider_name=default_provider_name,
        model=config.agents.defaults.model,
        temperature=config.agents.defaults.temperature,
        max_tokens=config.agents.defaults.max_tokens,
        context_limit=config.agents.defaults.context_limit,
        max_iterations=config.agents.defaults.max_tool_iterations,
        memory_window=config.agents.defaults.memory_window,
        tavily_api_key=config.tools.web.search.api_key or None,
        exec_config=config.tools.exec,
        cron_service=cron,
        mcp_servers=config.tools.mcp_servers,
        channels_config=config.channels,
    )

    # Show spinner when logs are off (no output to miss); skip when logs are on
    def _thinking_ctx():
        if logs:
            from contextlib import nullcontext
            return nullcontext()
        # Animated spinner is safe to use with prompt_toolkit input handling
        return console.status("[dim]bubbles is thinking...[/dim]", spinner="dots")

    async def _cli_progress(content: str, *, tool_hint: bool = False) -> None:
        ch = agent_loop.channels_config
        if ch and tool_hint and not ch.send_tool_hints:
            return
        if ch and not tool_hint and not ch.send_progress:
            return
        from prompt_toolkit import print_formatted_text
        print_formatted_text(f"  ↳ {content}")

    async def _debug_tool_call(name: str, args: dict, result: str | None) -> None:
        """Debug callback for tool calls."""
        import json
        from prompt_toolkit import print_formatted_text
        if result is None:
            # Tool call start
            args_str = json.dumps(args, ensure_ascii=False, indent=2)
            print_formatted_text(f"\n┌─ {name}")
            for line in args_str.split("\n"):
                print_formatted_text(f"│ {line}")
        else:
            # Tool call result
            result_preview = result[:500] + "..." if len(result) > 500 else result
            print_formatted_text("├─ Result:")
            for line in result_preview.split("\n")[:20]:
                print_formatted_text(f"│ {line}")
            if len(result) > 500 or result.count("\n") > 20:
                print_formatted_text(f"│ ... ({len(result)} chars total)")
            print_formatted_text("└─")

    if message:
        # Single message mode — direct call, no bus needed
        if not session_id:
            console.print("[red]Error:[/red] Session ID required for -m mode. Use -s <session_id>")
            raise typer.Exit(1)
        async def run_once():
            if debug:
                _print_debug_info(agent_loop, session_id)
            with _thinking_ctx():
                response, _tools_used = await agent_loop.process_direct(
                    message, session_id,
                    on_progress=_cli_progress,
                    on_tool_call=_debug_tool_call if debug else None,
                )
            _print_agent_response(response, render_markdown=markdown)
            await agent_loop.close_mcp()

        asyncio.run(run_once())
    else:
        # Interactive mode — route through bus like other channels
        from bubbles.bus.events import InboundMessage
        _init_prompt_session()
        console.print(f"{__logo__} Interactive mode (type [bold]exit[/bold] or [bold]Ctrl+C[/bold] to quit)")

        # Use cli:interactive as binding key, actual session determined by /session command
        cli_channel, cli_chat_id = "cli", "interactive"
        binding_key = f"{cli_channel}:{cli_chat_id}"

        # If session_id provided via -s, auto-bind to it
        if session_id:
            agent_loop._session_bindings[binding_key] = session_id
            save_session_bindings(agent_loop.data_dir, agent_loop._session_bindings)
            console.print(f"Bound to session: [bold]{session_id}[/bold]\n")
        else:
            # Check if already bound from previous run
            existing = agent_loop._session_bindings.get(binding_key)
            if existing:
                console.print(f"Resuming session: [bold]{existing}[/bold]\n")
            else:
                console.print("[dim]Use /session <id> to bind a session before chatting[/dim]\n")

        if debug:
            bound_session = agent_loop._session_bindings.get(binding_key)
            if bound_session:
                _print_debug_info(agent_loop, bound_session)
            agent_loop.on_tool_call = _debug_tool_call

        def _exit_on_sigint(signum, frame):
            _restore_terminal()
            console.print("\nGoodbye!")
            os._exit(0)

        signal.signal(signal.SIGINT, _exit_on_sigint)

        async def run_interactive():
            bus_task = asyncio.create_task(agent_loop.run())
            turn_done = asyncio.Event()
            turn_done.set()
            turn_response: list[str] = []

            async def _consume_outbound():
                while True:
                    try:
                        msg = await asyncio.wait_for(bus.consume_outbound(), timeout=1.0)
                        if msg.metadata.get("_progress"):
                            is_tool_hint = msg.metadata.get("_tool_hint", False)
                            ch = agent_loop.channels_config
                            if ch and is_tool_hint and not ch.send_tool_hints:
                                pass
                            elif ch and not is_tool_hint and not ch.send_progress:
                                pass
                            else:
                                # Use print_formatted_text for async output (compatible with prompt_toolkit)
                                from prompt_toolkit import print_formatted_text
                                from prompt_toolkit.formatted_text import HTML
                                print_formatted_text(HTML(f"  <style fg='gray'>↳ {msg.content}</style>"))
                        elif not turn_done.is_set():
                            if msg.content:
                                turn_response.append(msg.content)
                            turn_done.set()
                        elif msg.content:
                            # Async notification (e.g., subagent done) - use plain text
                            from prompt_toolkit import print_formatted_text
                            print_formatted_text(f"\n{__logo__} bubbles\n{msg.content}\n")
                    except asyncio.TimeoutError:
                        continue
                    except asyncio.CancelledError:
                        break

            outbound_task = asyncio.create_task(_consume_outbound())

            try:
                while True:
                    try:
                        _flush_pending_tty_input()
                        user_input = await _read_interactive_input_async()
                        command = user_input.strip()
                        if not command:
                            continue

                        if _is_exit_command(command):
                            _restore_terminal()
                            console.print("\nGoodbye!")
                            break

                        # Handle /prompt command - use bound session if exists
                        if command.lower() in {"/prompt", "/p"}:
                            bound_session = agent_loop._session_bindings.get(binding_key)
                            if not bound_session:
                                console.print("[yellow]No session bound.[/yellow] Use /session <id> first.")
                                continue
                            _print_debug_info(agent_loop, bound_session)
                            continue

                        # Handle /compact command - compress conversation history
                        if command.lower() == "/compact":
                            bound_session = agent_loop._session_bindings.get(binding_key)
                            if not bound_session:
                                console.print("[yellow]No session bound.[/yellow] Use /session <id> first.")
                                continue
                            session = agent_loop.sessions.get_or_create(bound_session)
                            console.print("[dim]Compacting conversation history...[/dim]")
                            result = await do_compact(agent_loop, session)
                            if result.success:
                                agent_loop.sessions.save(session)
                                console.print(
                                    f"[green]✓[/green] Compacted: {result.messages_compacted} messages summarized, "
                                    f"{result.tokens_before} → {result.tokens_after} tokens"
                                    + (" (fallback)" if result.used_fallback else "")
                                )
                            else:
                                console.print(f"[yellow]Nothing to compact:[/yellow] {result.error or 'not enough messages'}")
                            continue

                        # /session command - always allow (handled by agent loop)
                        if command.lower().startswith("/session"):
                            turn_done.clear()
                            turn_response.clear()
                            await bus.publish_inbound(InboundMessage(
                                channel=cli_channel,
                                sender_id="user",
                                chat_id=cli_chat_id,
                                content=user_input,
                            ))
                            with _thinking_ctx():
                                await turn_done.wait()
                            if turn_response:
                                _print_agent_response(turn_response[0], render_markdown=markdown)
                            continue

                        # Regular messages require a bound session
                        bound_session = agent_loop._session_bindings.get(binding_key)
                        if not bound_session:
                            console.print("[yellow]No session bound.[/yellow] Use /session <id> to start.")
                            continue

                        turn_done.clear()
                        turn_response.clear()

                        await bus.publish_inbound(InboundMessage(
                            channel=cli_channel,
                            sender_id="user",
                            chat_id=cli_chat_id,
                            content=user_input,
                        ))

                        with _thinking_ctx():
                            await turn_done.wait()

                        if turn_response:
                            _print_agent_response(turn_response[0], render_markdown=markdown)
                    except KeyboardInterrupt:
                        _restore_terminal()
                        console.print("\nGoodbye!")
                        break
                    except EOFError:
                        _restore_terminal()
                        console.print("\nGoodbye!")
                        break
            finally:
                agent_loop.stop()
                outbound_task.cancel()
                await asyncio.gather(bus_task, outbound_task, return_exceptions=True)
                await agent_loop.close_mcp()

        asyncio.run(run_interactive())
