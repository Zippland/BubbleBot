"""`bubbles gateway` — start the long-running gateway process."""

from __future__ import annotations

import asyncio
import inspect
import os
import threading
from collections.abc import Callable
from typing import Any

import typer
from loguru import logger

from bubbles import __logo__
from bubbles.cli._image_generation import _make_image_generation_backend
from bubbles.cli._providers import _make_provider, _make_provider_for_model
from bubbles.cli.commands import app, console
from bubbles.gateway_control import (
    NO_RESTART_EXIT_CODE,
    SUPERVISOR_STOP_ACTION,
    UPGRADE_ACTION,
    UPGRADE_EXIT_CODE,
)

UPGRADE_HARD_EXIT_TIMEOUT_SECONDS = 30
SERVICE_FAILURE_HARD_EXIT_TIMEOUT_SECONDS = 2
WCFERRY_CLEANUP_HARD_EXIT_TIMEOUT_SECONDS = 30


class UnsafeWcferryShutdownError(RuntimeError):
    """WCFerry cleanup could not be proven; the supervisor must not restart."""


def _hard_exit(exit_code: int) -> None:
    """Terminate even if a third-party coroutine ignores cancellation."""
    os._exit(exit_code)


def _hard_exit_upgrade() -> None:
    _hard_exit(UPGRADE_EXIT_CODE)


def _hard_exit_no_restart() -> None:
    _hard_exit(NO_RESTART_EXIT_CODE)


def _arm_upgrade_exit_watchdog(
    *,
    timeout: float = UPGRADE_HARD_EXIT_TIMEOUT_SECONDS,
    timer_factory: Callable[..., Any] = threading.Timer,
) -> Any:
    """Arm a daemon watchdog that preserves the supervisor's exit-code contract."""
    timer = timer_factory(timeout, _hard_exit_upgrade)
    timer.daemon = True
    timer.start()
    return timer


def _arm_service_failure_exit_watchdog(
    *,
    timeout: float = SERVICE_FAILURE_HARD_EXIT_TIMEOUT_SECONDS,
    timer_factory: Callable[..., Any] = threading.Timer,
) -> Any:
    """Force a prompt non-zero exit if failed services refuse to shut down."""
    timer = timer_factory(timeout, lambda: _hard_exit(1))
    timer.daemon = True
    timer.start()
    return timer


def _arm_wcferry_cleanup_watchdog(
    *,
    timeout: float = WCFERRY_CLEANUP_HARD_EXIT_TIMEOUT_SECONDS,
    timer_factory: Callable[..., Any] = threading.Timer,
) -> Any:
    """Fail closed if native WCFerry detachment itself hangs."""
    timer = timer_factory(timeout, _hard_exit_no_restart)
    timer.daemon = True
    timer.start()
    return timer


async def _notify_upgrade_result(
    gateway_control: Any,
    channels: Any,
    *,
    timeout: float = 300,
    poll_interval: float = 1,
) -> None:
    """Wait for the supervisor result, then report it through its source channel."""
    from bubbles.bus.events import OutboundMessage

    expected_request_id = gateway_control.readiness_request_id
    result = gateway_control.load_upgrade_result()
    if result is None and expected_request_id is None:
        return

    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    reported_stale_request_id: str | None = None
    while loop.time() < deadline:
        if result is None:
            result = gateway_control.load_upgrade_result()

        if result is not None:
            if (
                expected_request_id is not None
                and result.request_id != expected_request_id
            ):
                stale_request_id = result.request_id or "<missing>"
                if stale_request_id != reported_stale_request_id:
                    logger.warning(
                        "Ignoring stale gateway upgrade result for request {} while waiting",
                        stale_request_id,
                    )
                    reported_stale_request_id = stale_request_id
                result = None
                await asyncio.sleep(poll_interval)
                continue

            committed = gateway_control.observe_upgrade_result(result)
            if expected_request_id is not None and result.ok and not committed:
                result = None
                await asyncio.sleep(poll_interval)
                continue
            channel = channels.get_channel(result.channel)
            if channel is not None and channel.is_ready:
                try:
                    delivered = await channel.send(OutboundMessage(
                        channel=result.channel,
                        chat_id=result.chat_id,
                        content=result.format_message(),
                    ))
                    if delivered is not False:
                        gateway_control.clear_upgrade_result()
                        return
                except Exception as exc:
                    logger.warning("Failed to report gateway upgrade result: {}", exc)
        await asyncio.sleep(poll_interval)

    logger.warning("Gateway upgrade result notification timed out; will retry next start")


async def _publish_gateway_readiness(
    gateway_control: Any,
    channels: Any,
    *,
    poll_interval: float = 1,
) -> None:
    """Publish readiness only after AgentLoop and the WeChat receiver are live."""
    if gateway_control.readiness_request_id is None:
        return

    await gateway_control.wait_for_agent_ready()
    while True:
        channel = channels.get_channel("wechat")
        wechat_ready = channel is not None and bool(channel.is_ready)
        if gateway_control.publish_readiness(wechat_ready=wechat_ready):
            return
        await asyncio.sleep(poll_interval)


async def _best_effort_cleanup_step(
    label: str,
    operation: Callable[[], Any],
) -> tuple[bool, Any]:
    """Run one cleanup step without preventing later steps or exit code 75."""
    try:
        result = operation()
        if inspect.isawaitable(result):
            result = await result
        return True, result
    except BaseException as exc:
        logger.opt(exception=exc).error(
            "Gateway cleanup step '{}' failed; continuing",
            label,
        )
        return False, None


async def _cleanup_gateway_runtime(
    *,
    agent: Any,
    channels: Any,
    cron: Any,
    service_tasks: tuple[asyncio.Task[Any] | None, ...],
    helper_tasks: tuple[asyncio.Task[Any] | None, ...],
    after_wcferry_cleanup: Callable[[], Any] | None = None,
    wcferry_watchdog_factory: Callable[[], Any] = _arm_wcferry_cleanup_watchdog,
) -> tuple[bool, bool]:
    """Stop WCFerry first, then drain the rest of the gateway runtime."""
    helpers = tuple(task for task in helper_tasks if task is not None)
    services = tuple(task for task in service_tasks if task is not None)
    wechat_enabled = "wechat" in getattr(channels, "enabled_channels", ())
    wcferry_watchdog: Any = None

    async def stop_helpers() -> None:
        for task in helpers:
            if not task.done():
                task.cancel()
        if helpers:
            await asyncio.gather(*helpers, return_exceptions=True)

    async def drain_services() -> bool:
        for task in services:
            if not task.done():
                task.cancel()
        if not services:
            return True

        done, pending = await asyncio.wait(set(services), timeout=3)
        for task in done:
            try:
                task.result()
            except BaseException:
                pass
        if pending:
            logger.error(
                "{} gateway service task(s) ignored cancellation; watchdog remains armed",
                len(pending),
            )
            return False
        return True

    if wechat_enabled:
        try:
            wcferry_watchdog = wcferry_watchdog_factory()
        except BaseException as exc:
            logger.opt(exception=exc).error("Unable to arm WCFerry cleanup watchdog")

    channel_step_ok, channel_failures = await _best_effort_cleanup_step(
        "channels",
        channels.stop_all,
    )
    if wcferry_watchdog is not None:
        await _best_effort_cleanup_step(
            "WCFerry cleanup watchdog",
            wcferry_watchdog.cancel,
        )
    failure_names = channel_failures if isinstance(channel_failures, dict) else {}
    wcferry_safe = channel_step_ok and "wechat" not in failure_names
    if wcferry_safe and after_wcferry_cleanup is not None:
        await _best_effort_cleanup_step(
            "post-WCFerry cleanup transition",
            after_wcferry_cleanup,
        )

    await _best_effort_cleanup_step("helper tasks", stop_helpers)
    await _best_effort_cleanup_step("agent stop", agent.stop)
    await _best_effort_cleanup_step("active agent tasks", agent.cancel_active_tasks)
    await _best_effort_cleanup_step("cron", cron.stop)
    await _best_effort_cleanup_step("MCP connections", agent.close_mcp)
    await _best_effort_cleanup_step("sandboxes", agent.close_sandboxes)
    drained, services_finished = await _best_effort_cleanup_step(
        "service tasks",
        drain_services,
    )
    return bool(drained and services_finished), wcferry_safe


async def _run_gateway_runtime(
    *,
    agent: Any,
    channels: Any,
    cron: Any,
    gateway_control: Any,
    watchdog_factory: Callable[[], Any] = _arm_upgrade_exit_watchdog,
    failure_watchdog_factory: Callable[[], Any] = _arm_service_failure_exit_watchdog,
    no_restart_watchdog_factory: Callable[[], Any] = _arm_wcferry_cleanup_watchdog,
    wcferry_watchdog_factory: Callable[[], Any] = _arm_wcferry_cleanup_watchdog,
) -> str | None:
    """Run gateway services until a lifecycle action or service termination."""
    requested_action: str | None = None
    agent_task: asyncio.Task[Any] | None = None
    channels_task: asyncio.Task[Any] | None = None
    action_task: asyncio.Task[Any] | None = None
    supervisor_stop_task: asyncio.Task[Any] | None = None
    notification_task: asyncio.Task[Any] | None = None
    readiness_task: asyncio.Task[Any] | None = None
    upgrade_watchdog: Any = None
    failure_watchdog: Any = None
    no_restart_watchdog: Any = None
    service_failed = False
    unsafe_wcferry_state = False
    stop_ack_published = False
    stop_ack_required = bool(getattr(gateway_control, "instance_id", None))

    def arm_upgrade_watchdog() -> None:
        nonlocal upgrade_watchdog
        if requested_action != UPGRADE_ACTION or upgrade_watchdog is not None:
            return
        try:
            upgrade_watchdog = watchdog_factory()
        except BaseException as exc:
            logger.opt(exception=exc).error("Unable to arm gateway upgrade exit watchdog")

    def arm_failure_watchdog() -> None:
        nonlocal failure_watchdog
        if failure_watchdog is not None:
            return
        try:
            failure_watchdog = failure_watchdog_factory()
        except BaseException as exc:
            logger.opt(exception=exc).error("Unable to arm gateway failure exit watchdog")

    def arm_no_restart_watchdog() -> None:
        nonlocal no_restart_watchdog
        if no_restart_watchdog is not None:
            return
        try:
            no_restart_watchdog = no_restart_watchdog_factory()
        except BaseException as exc:
            logger.opt(exception=exc).error("Unable to arm no-restart exit watchdog")

    def after_wcferry_cleanup() -> None:
        nonlocal stop_ack_published
        if stop_ack_required:
            stop_ack_published = bool(gateway_control.publish_stopped())
        if requested_action == UPGRADE_ACTION:
            arm_upgrade_watchdog()
        elif requested_action == SUPERVISOR_STOP_ACTION:
            arm_no_restart_watchdog()
        elif service_failed:
            arm_failure_watchdog()

    try:
        await cron.start()
        agent_task = asyncio.create_task(agent.run())
        channels_task = asyncio.create_task(channels.start_all())
        action_task = asyncio.create_task(gateway_control.wait_for_action())
        supervisor_stop_task = asyncio.create_task(
            gateway_control.wait_for_supervisor_stop()
        )
        notification_task = asyncio.create_task(
            _notify_upgrade_result(gateway_control, channels)
        )
        readiness_task = asyncio.create_task(
            _publish_gateway_readiness(gateway_control, channels)
        )
        wait_targets = {agent_task, action_task, supervisor_stop_task}
        if getattr(channels, "enabled_channels", ()):
            wait_targets.add(channels_task)
        done, _ = await asyncio.wait(
            wait_targets,
            return_when=asyncio.FIRST_COMPLETED,
        )
        if action_task in done:
            requested_action = action_task.result()
        elif supervisor_stop_task in done:
            requested_action = supervisor_stop_task.result()
        elif channels_task in done:
            service_failed = True
            try:
                await channels_task
            except BaseException as exc:
                if bool(getattr(exc, "no_restart", False)):
                    unsafe_wcferry_state = True
                raise
            raise RuntimeError("Enabled channel services stopped unexpectedly")
        else:
            service_failed = True
            await agent_task
            raise RuntimeError("Agent service stopped unexpectedly")
    except KeyboardInterrupt:
        console.print("\nShutting down...")
    finally:
        # If service termination and the lifecycle event raced, retain the
        # already-confirmed action before cleanup starts.
        if (
            requested_action is None
            and action_task is not None
            and action_task.done()
            and not action_task.cancelled()
        ):
            try:
                requested_action = action_task.result()
            except Exception:
                logger.exception("Unable to read completed gateway lifecycle action")
        if (
            requested_action is None
            and supervisor_stop_task is not None
            and supervisor_stop_task.done()
            and not supervisor_stop_task.cancelled()
        ):
            try:
                requested_action = supervisor_stop_task.result()
            except Exception:
                logger.exception("Unable to read completed supervisor stop action")

        services_finished, wcferry_safe = await _cleanup_gateway_runtime(
            agent=agent,
            channels=channels,
            cron=cron,
            service_tasks=(agent_task, channels_task),
            helper_tasks=(
                action_task,
                supervisor_stop_task,
                notification_task,
                readiness_task,
            ),
            after_wcferry_cleanup=after_wcferry_cleanup,
            wcferry_watchdog_factory=wcferry_watchdog_factory,
        )
        if services_finished and upgrade_watchdog is not None:
            await _best_effort_cleanup_step(
                "upgrade exit watchdog",
                upgrade_watchdog.cancel,
            )
        if services_finished and failure_watchdog is not None:
            await _best_effort_cleanup_step(
                "service failure exit watchdog",
                failure_watchdog.cancel,
            )
        if services_finished and no_restart_watchdog is not None:
            await _best_effort_cleanup_step(
                "no-restart exit watchdog",
                no_restart_watchdog.cancel,
            )
        if not wcferry_safe or unsafe_wcferry_state:
            raise UnsafeWcferryShutdownError(
                "WCFerry cleanup was not proven; refusing every automatic restart"
            )
        if stop_ack_required and not stop_ack_published:
            raise UnsafeWcferryShutdownError(
                "Safe WCFerry stop completed, but the supervisor acknowledgement failed"
            )

    return requested_action


@app.command()
def gateway(
    port: int = typer.Option(18790, "--port", "-p", help="Gateway port"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
):
    """Start the bubbles gateway."""
    from bubbles.agent.loop import AgentLoop
    from bubbles.bus.queue import MessageBus
    from bubbles.channels.manager import ChannelManager
    from bubbles.config.loader import get_data_dir, load_config
    from bubbles.cron.service import CronService
    from bubbles.cron.types import CronJob
    from bubbles.gateway_control import GatewayControl
    from bubbles.session.manager import SessionManager

    if verbose:
        logger.enable("bubbles")
        os.environ["LITELLM_LOG"] = "DEBUG"
    else:
        logger.disable("bubbles")

    console.print(f"{__logo__} Starting bubbles gateway on port {port}...")

    config = load_config()
    bus = MessageBus()
    gateway_control = GatewayControl(config.gateway.update, get_data_dir())
    provider = _make_provider(config)
    default_provider_name = config.get_provider_name(config.agents.defaults.model)

    def provider_factory(model: str):
        return _make_provider_for_model(config, model)

    image_generation_backend = _make_image_generation_backend(config)
    session_manager = SessionManager()  # Uses default ~/.bubbles/sessions/

    # Create cron service first (callback set after agent creation)
    cron_store_path = get_data_dir() / "cron" / "jobs.json"
    cron = CronService(cron_store_path)

    # Channels are needed by the agent (find_person tool dispatches to them),
    # so construct them before the agent. Started later, alongside the agent.
    channels = ChannelManager(config, bus)
    channels.on_delivery_result = gateway_control.handle_delivery_result

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
        compact_keep_max_tokens=config.agents.defaults.compact_keep_max_tokens,
        max_api_retries=config.agents.defaults.max_api_retries,
        max_concurrent_sessions=config.agents.defaults.max_concurrent_sessions,
        tavily_api_key=config.tools.web.search.api_key or None,
        exec_config=config.tools.exec,
        sandbox_config=config.tools.sandbox,
        cron_service=cron,
        session_manager=session_manager,
        channel_manager=channels,
        mcp_servers=config.tools.mcp_servers,
        channels_config=config.channels,
        image_generation_backend=image_generation_backend,
        gateway_control=gateway_control,
    )

    # Set cron callback (needs agent)
    async def on_cron_job(job: CronJob) -> str | None:
        """Execute a cron job through the agent.

        Uses the system-triggered tool-set for the duration of the turn:
        - ``stay_silent`` remains available, as it is in every turn.
        - ``cron`` is removed so a triggered turn cannot schedule more jobs
          (no recursive job creation; see SPEC §5.6).
        """
        from bubbles.bus.events import OutboundMessage

        # Use the saved session_key to inject history, fallback to cron:{job.id}
        session_key = job.payload.session_key or f"cron:{job.id}"

        response, tools_used = await agent.process_direct(
            job.payload.message,
            session_key=session_key,
            channel=job.payload.channel or "cli",
            chat_id=job.payload.to or "direct",
            system_triggered=True,
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

    try:
        action = asyncio.run(_run_gateway_runtime(
            agent=agent,
            channels=channels,
            cron=cron,
            gateway_control=gateway_control,
        ))
    except UnsafeWcferryShutdownError as exc:
        logger.critical("{}", exc)
        raise typer.Exit(NO_RESTART_EXIT_CODE) from exc
    if action == UPGRADE_ACTION:
        raise typer.Exit(UPGRADE_EXIT_CODE)
    if action == SUPERVISOR_STOP_ACTION:
        raise typer.Exit(NO_RESTART_EXIT_CODE)
