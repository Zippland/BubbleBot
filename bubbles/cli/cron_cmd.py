"""`bubbles cron` — manage scheduled jobs."""

from __future__ import annotations

import asyncio

import typer
from rich.table import Table

from bubbles.cli._interactive import _print_agent_response
from bubbles.cli._providers import _make_provider
from bubbles.cli.commands import app, console


cron_app = typer.Typer(help="Manage scheduled tasks")
app.add_typer(cron_app, name="cron")


@cron_app.command("list")
def cron_list(
    all: bool = typer.Option(False, "--all", "-a", help="Include disabled jobs"),
    session: str | None = typer.Option(None, "--session", help="Filter by session_key"),
    channel: str | None = typer.Option(None, "--channel", help="Filter by delivery channel"),
    as_json: bool = typer.Option(False, "--json", help="Emit machine-readable JSON"),
):
    """List scheduled jobs."""
    import json as _json

    from bubbles.config.loader import get_data_dir
    from bubbles.cron.format import (
        format_absolute,
        format_relative,
        format_schedule,
        format_status,
    )
    from bubbles.cron.service import CronService

    store_path = get_data_dir() / "cron" / "jobs.json"
    service = CronService(store_path)

    jobs = service.list_jobs(include_disabled=all)
    if session is not None:
        jobs = [j for j in jobs if j.payload.session_key == session]
    if channel is not None:
        jobs = [j for j in jobs if j.payload.channel == channel]

    if as_json:
        payload = [
            {
                "id": j.id,
                "name": j.name,
                "enabled": j.enabled,
                "schedule": format_schedule(j.schedule),
                "next_run_at_ms": j.state.next_run_at_ms,
                "next_relative": format_relative(j.state.next_run_at_ms) if j.state.next_run_at_ms else None,
                "last_status": j.state.last_status,
                "last_error": j.state.last_error,
                "backoff_until_ms": j.state.backoff_until_ms,
                "session_key": j.payload.session_key,
                "channel": j.payload.channel,
                "to": j.payload.to,
            }
            for j in jobs
        ]
        console.print_json(_json.dumps(payload, ensure_ascii=False))
        return

    if not jobs:
        console.print("No scheduled jobs.")
        return

    table = Table(title="Scheduled Jobs")
    table.add_column("ID", style="cyan")
    table.add_column("Name")
    table.add_column("Schedule")
    table.add_column("Enabled")
    table.add_column("Next Run")
    table.add_column("Status")

    for job in jobs:
        sched = format_schedule(job.schedule)

        next_run = ""
        if job.state.next_run_at_ms:
            abs_str = format_absolute(job.state.next_run_at_ms, job.schedule.tz)
            rel_str = format_relative(job.state.next_run_at_ms)
            next_run = f"{abs_str}\n[dim]{rel_str}[/dim]"

        enabled_cell = "[green]yes[/green]" if job.enabled else "[dim]no[/dim]"

        status_text = format_status(job.state)
        if job.state.last_status == "error":
            status_cell = f"[red]{status_text}[/red]"
        elif job.state.last_status == "ok":
            status_cell = f"[green]{status_text}[/green]"
        else:
            status_cell = f"[dim]{status_text}[/dim]"

        table.add_row(job.id, job.name, sched, enabled_cell, next_run, status_cell)

    console.print(table)


@cron_app.command("add")
def cron_add(
    name: str = typer.Option(..., "--name", "-n", help="Job name"),
    message: str = typer.Option(..., "--message", "-m", help="Message for agent"),
    every: int = typer.Option(None, "--every", "-e", help="Run every N seconds"),
    cron_expr: str = typer.Option(None, "--cron", "-c", help="Cron expression (e.g. '0 9 * * *')"),
    tz: str | None = typer.Option(None, "--tz", help="IANA timezone for cron (e.g. 'America/Vancouver')"),
    at: str = typer.Option(None, "--at", help="Run once at time (ISO format)"),
    deliver: bool = typer.Option(False, "--deliver", "-d", help="Deliver response to channel"),
    to: str = typer.Option(None, "--to", help="Recipient for delivery"),
    channel: str = typer.Option(None, "--channel", help="Channel for delivery (e.g. 'telegram', 'whatsapp')"),
):
    """Add a scheduled job."""
    from bubbles.config.loader import get_data_dir
    from bubbles.cron.service import CronService
    from bubbles.cron.types import CronSchedule

    if tz and not cron_expr:
        console.print("[red]Error: --tz can only be used with --cron[/red]")
        raise typer.Exit(1)

    if every:
        schedule = CronSchedule(kind="every", every_ms=every * 1000)
    elif cron_expr:
        schedule = CronSchedule(kind="cron", expr=cron_expr, tz=tz)
    elif at:
        import datetime
        dt = datetime.datetime.fromisoformat(at)
        schedule = CronSchedule(kind="at", at_ms=int(dt.timestamp() * 1000))
    else:
        console.print("[red]Error: Must specify --every, --cron, or --at[/red]")
        raise typer.Exit(1)

    store_path = get_data_dir() / "cron" / "jobs.json"
    service = CronService(store_path)

    try:
        job = service.add_job(
            name=name,
            schedule=schedule,
            message=message,
            deliver=deliver,
            to=to,
            channel=channel,
        )
    except ValueError as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1) from e

    console.print(f"[green]✓[/green] Added job '{job.name}' ({job.id})")


@cron_app.command("remove")
def cron_remove(
    job_id: str = typer.Argument(..., help="Job ID to remove"),
):
    """Remove a scheduled job."""
    from bubbles.config.loader import get_data_dir
    from bubbles.cron.service import CronService

    store_path = get_data_dir() / "cron" / "jobs.json"
    service = CronService(store_path)

    if service.remove_job(job_id):
        console.print(f"[green]✓[/green] Removed job {job_id}")
    else:
        console.print(f"[red]Job {job_id} not found[/red]")


@cron_app.command("enable")
def cron_enable(
    job_id: str = typer.Argument(..., help="Job ID"),
    disable: bool = typer.Option(False, "--disable", help="Disable instead of enable"),
):
    """Enable or disable a job."""
    from bubbles.config.loader import get_data_dir
    from bubbles.cron.service import CronService

    store_path = get_data_dir() / "cron" / "jobs.json"
    service = CronService(store_path)

    job = service.enable_job(job_id, enabled=not disable)
    if job:
        status = "disabled" if disable else "enabled"
        console.print(f"[green]✓[/green] Job '{job.name}' {status}")
    else:
        console.print(f"[red]Job {job_id} not found[/red]")


@cron_app.command("run")
def cron_run(
    job_id: str = typer.Argument(..., help="Job ID to run"),
    force: bool = typer.Option(False, "--force", "-f", help="Run even if disabled"),
):
    """Manually run a job."""
    from loguru import logger
    from bubbles.config.loader import load_config, get_data_dir
    from bubbles.cron.service import CronService
    from bubbles.cron.types import CronJob
    from bubbles.bus.queue import MessageBus
    from bubbles.agent.loop import AgentLoop
    logger.disable("bubbles")

    config = load_config()
    provider = _make_provider(config)
    bus = MessageBus()
    agent_loop = AgentLoop(
        bus=bus,
        provider=provider,
        model=config.agents.defaults.model,
        temperature=config.agents.defaults.temperature,
        max_tokens=config.agents.defaults.max_tokens,
        context_limit=config.agents.defaults.context_limit,
        max_iterations=config.agents.defaults.max_tool_iterations,
        memory_window=config.agents.defaults.memory_window,
        tavily_api_key=config.tools.web.search.api_key or None,
        exec_config=config.tools.exec,
        mcp_servers=config.tools.mcp_servers,
        channels_config=config.channels,
    )

    store_path = get_data_dir() / "cron" / "jobs.json"
    service = CronService(store_path)

    result_holder = []

    async def on_job(job: CronJob) -> str | None:
        response, _tools_used = await agent_loop.process_direct(
            job.payload.message,
            session_key=f"cron:{job.id}",
            channel=job.payload.channel or "cli",
            chat_id=job.payload.to or "direct",
        )
        result_holder.append(response)
        return response

    service.on_job = on_job

    async def run():
        return await service.run_job(job_id, force=force)

    if asyncio.run(run()):
        console.print("[green]✓[/green] Job executed")
        if result_holder:
            _print_agent_response(result_holder[0], render_markdown=True)
    else:
        console.print(f"[red]Failed to run job {job_id}[/red]")
