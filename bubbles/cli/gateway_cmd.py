"""`bubbles gateway` — start the long-running gateway process."""

from __future__ import annotations

import asyncio

import typer

from bubbles import __logo__
from bubbles.cli._providers import _make_provider, _make_provider_for_model
from bubbles.cli.commands import app, console


@app.command()
def gateway(
    port: int = typer.Option(18790, "--port", "-p", help="Gateway port"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
):
    """Start the bubbles gateway."""
    from loguru import logger
    from bubbles.config.loader import load_config, get_data_dir
    from bubbles.bus.queue import MessageBus
    from bubbles.agent.loop import AgentLoop
    from bubbles.channels.manager import ChannelManager
    from bubbles.session.manager import SessionManager
    from bubbles.cron.service import CronService
    from bubbles.cron.types import CronJob
    if verbose:
        logger.enable("bubbles")
        import os
        os.environ["LITELLM_LOG"] = "DEBUG"
    else:
        logger.disable("bubbles")

    console.print(f"{__logo__} Starting bubbles gateway on port {port}...")

    config = load_config()
    bus = MessageBus()
    provider = _make_provider(config)
    default_provider_name = config.get_provider_name(config.agents.defaults.model)
    provider_factory = lambda m: _make_provider_for_model(config, m)
    session_manager = SessionManager()  # Uses default ~/.bubbles/sessions/

    # Create cron service first (callback set after agent creation)
    cron_store_path = get_data_dir() / "cron" / "jobs.json"
    cron = CronService(cron_store_path)

    # Channels are needed by the agent (find_person tool dispatches to them),
    # so construct them before the agent. Started later, alongside the agent.
    channels = ChannelManager(config, bus)

    # Create agent with cron service + channel manager
    agent = AgentLoop(
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
        session_manager=session_manager,
        channel_manager=channels,
        mcp_servers=config.tools.mcp_servers,
        channels_config=config.channels,
    )

    # Set cron callback (needs agent)
    async def on_cron_job(job: CronJob) -> str | None:
        """Execute a cron job through the agent.

        Uses the system-triggered tool-set for the duration of the turn:
        - ``stay_silent`` is added so the model can opt out of delivery.
        - ``cron`` is removed so a triggered turn cannot schedule more jobs
          (no recursive job creation; see SPEC §5.6).
        """
        from bubbles.agent.system_turn import system_triggered_toolset
        from bubbles.bus.events import OutboundMessage

        # Use the saved session_key to inject history, fallback to cron:{job.id}
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

    if channels.enabled_channels:
        console.print(f"[green]✓[/green] Channels enabled: {', '.join(channels.enabled_channels)}")
    else:
        console.print("[yellow]Warning: No channels enabled[/yellow]")

    cron_status = cron.status()
    if cron_status["jobs"] > 0:
        console.print(f"[green]✓[/green] Cron: {cron_status['jobs']} scheduled jobs")

    async def run():
        try:
            await cron.start()
            await asyncio.gather(
                agent.run(),
                channels.start_all(),
            )
        except KeyboardInterrupt:
            console.print("\nShutting down...")
        finally:
            await agent.close_mcp()
            cron.stop()
            agent.stop()
            await channels.stop_all()

    asyncio.run(run())
