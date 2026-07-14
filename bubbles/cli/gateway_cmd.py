"""`bubbles gateway` — start the long-running gateway process.

Runs in one of three roles (Phase 2 split deployment):

- ``all`` (default): channels + harness in one process over an in-process bus.
  Identical to the pre-split single-process gateway.
- ``harness``: agent + cron + sandboxes only; channels reached via RPC over the
  bus. For the host that should run the agent (Linux or macOS).
- ``channels``: chat channels only (e.g. wcferry on Windows); no agent/session.

``harness`` and ``channels`` require a networked bus (``bus.default = redis``).
"""

from __future__ import annotations

import asyncio

import typer

from bubbles import __logo__
from bubbles.cli._providers import _make_provider, _make_provider_for_model
from bubbles.cli.commands import app, console


def _build_agent(config, bus, cron, session_manager, channel_manager):
    """Construct the AgentLoop (shared by `all` and `harness` roles)."""
    from bubbles.agent.loop import AgentLoop

    provider = _make_provider(config)
    default_provider_name = config.get_provider_name(config.agents.defaults.model)
    provider_factory = lambda m: _make_provider_for_model(config, m)
    return AgentLoop(
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
        sandbox_config=config.tools.sandbox,
        cron_service=cron,
        session_manager=session_manager,
        channel_manager=channel_manager,
        mcp_servers=config.tools.mcp_servers,
        channels_config=config.channels,
    )


def _wire_cron(agent, bus, cron, logger):
    """Attach the cron→agent callback (shared by `all` and `harness`)."""
    from bubbles.agent.system_turn import system_triggered_toolset
    from bubbles.bus.events import OutboundMessage
    from bubbles.cron.types import CronJob

    async def on_cron_job(job: CronJob) -> str | None:
        """Execute a cron job through the agent.

        System-triggered tool-set: ``stay_silent`` added (opt out of delivery),
        ``cron`` removed (no recursive job creation; SPEC §5.6).
        """
        session_key = job.payload.session_key or f"cron:{job.id}"
        with system_triggered_toolset(agent):
            response, tools_used = await agent.process_direct(
                job.payload.message,
                session_key=session_key,
                channel=job.payload.channel or "cli",
                chat_id=job.payload.to or "direct",
            )
        if "stay_silent" in tools_used:
            logger.info("cron: stay_silent for job {} ({})", job.id, job.name)
            return None
        if job.payload.deliver and job.payload.to and response:
            await bus.publish_outbound(OutboundMessage(
                channel=job.payload.channel or "cli",
                chat_id=job.payload.to,
                content=response,
            ))
        return response

    cron.on_job = on_cron_job


@app.command()
def gateway(
    port: int = typer.Option(18790, "--port", "-p", help="Gateway port"),
    role: str = typer.Option(
        "all", "--role",
        help="Deployment role: all (default, single process) | harness (agent only) | channels (channels only)",
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
):
    """Start the bubbles gateway."""
    from loguru import logger
    from bubbles.config.loader import load_config, get_data_dir
    from bubbles.bus.factory import make_bus
    from bubbles.channels.manager import ChannelManager
    from bubbles.session.manager import SessionManager
    from bubbles.cron.service import CronService

    if verbose:
        logger.enable("bubbles")
        import os
        os.environ["LITELLM_LOG"] = "DEBUG"
    else:
        logger.disable("bubbles")

    role = role.lower()
    if role not in ("all", "harness", "channels"):
        console.print(f"[red]Unknown --role '{role}'. Use: all | harness | channels[/red]")
        raise typer.Exit(1)

    config = load_config()
    if role != "all" and config.bus.default == "local":
        console.print(
            f"[red]--role={role} requires a networked bus. "
            f"Set bus.default='redis' + bus.redis_url in config.[/red]"
        )
        raise typer.Exit(1)

    console.print(f"{__logo__} Starting bubbles gateway (role={role})...")

    if role == "channels":
        _run_channels(config, make_bus, ChannelManager, logger)
    elif role == "harness":
        _run_harness(config, make_bus, SessionManager, CronService, get_data_dir, logger)
    else:
        _run_all(config, make_bus, ChannelManager, SessionManager, CronService, get_data_dir, logger)


def _run_all(config, make_bus, ChannelManager, SessionManager, CronService, get_data_dir, logger):
    """Single-process: channels + harness sharing an in-process bus (unchanged)."""
    bus = make_bus(config)
    session_manager = SessionManager()
    cron = CronService(get_data_dir() / "cron" / "jobs.json")
    channels = ChannelManager(config, bus)
    agent = _build_agent(config, bus, cron, session_manager, channels)
    _wire_cron(agent, bus, cron, logger)

    if channels.enabled_channels:
        console.print(f"[green]✓[/green] Channels enabled: {', '.join(channels.enabled_channels)}")
    else:
        console.print("[yellow]Warning: No channels enabled[/yellow]")
    _print_cron_status(cron)

    async def run():
        try:
            await cron.start()
            await asyncio.gather(agent.run(), channels.start_all())
        except KeyboardInterrupt:
            console.print("\nShutting down...")
        finally:
            await agent.close_mcp()
            await agent.close_sandboxes()
            cron.stop()
            agent.stop()
            await channels.stop_all()

    asyncio.run(run())


def _run_harness(config, make_bus, SessionManager, CronService, get_data_dir, logger):
    """Agent + cron + sandboxes; channels reached via RPC proxy over the bus."""
    from bubbles.bus.rpc_proxy import RpcChannelProxy

    bus = make_bus(config, consumer_name="harness")
    session_manager = SessionManager()
    cron = CronService(get_data_dir() / "cron" / "jobs.json")
    channel_proxy = RpcChannelProxy(bus)
    agent = _build_agent(config, bus, cron, session_manager, channel_proxy)
    _wire_cron(agent, bus, cron, logger)

    console.print("[green]✓[/green] Harness role: agent + cron + sandboxes")
    _print_cron_status(cron)

    async def run():
        try:
            await cron.start()
            await agent.run()
        except KeyboardInterrupt:
            console.print("\nShutting down...")
        finally:
            await agent.close_mcp()
            await agent.close_sandboxes()
            cron.stop()
            agent.stop()

    asyncio.run(run())


def _run_channels(config, make_bus, ChannelManager, logger):
    """Chat channels only (e.g. wcferry on Windows). No agent/session/cron."""
    bus = make_bus(config, consumer_name="channels")
    channels = ChannelManager(config, bus)

    if channels.enabled_channels:
        console.print(f"[green]✓[/green] Channels enabled: {', '.join(channels.enabled_channels)}")
    else:
        console.print("[yellow]Warning: No channels enabled[/yellow]")

    async def run():
        try:
            # serve_rpc handles harness→channel roster lookups (find_person).
            # Wired fully in 2c; skip gracefully if the bus has no RPC lane yet.
            serve = getattr(bus, "serve_rpc", None)
            rpc_task = None
            if serve is not None:
                async def _roster_handler(verb, payload):
                    if verb != "roster":
                        return None
                    ch = channels.get_channel(payload.get("channel", ""))
                    if ch is None:
                        return []
                    return await ch.get_group_members(payload.get("chat_id", ""))
                try:
                    rpc_task = asyncio.create_task(serve(_roster_handler))
                except NotImplementedError:
                    rpc_task = None  # 2b: RPC lane not wired yet

            await channels.start_all()
            if rpc_task:
                rpc_task.cancel()
        except KeyboardInterrupt:
            console.print("\nShutting down...")
        finally:
            await channels.stop_all()

    asyncio.run(run())


def _print_cron_status(cron):
    status = cron.status()
    if status["jobs"] > 0:
        console.print(f"[green]✓[/green] Cron: {status['jobs']} scheduled jobs")
