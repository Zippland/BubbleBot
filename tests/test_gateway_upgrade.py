"""Gateway remote-upgrade control and WeChat delivery ordering regressions.

The production implementation deliberately owns only control-plane state.  The
external supervisor performs the actual repository update and process restart.
"""

from __future__ import annotations

import ast
import asyncio
import inspect
import json
import os
import threading
import time
from pathlib import Path
from typing import Any

import pytest

import bubbles.channels.wechat as wechat_module
from bubbles.agent.loop import AgentLoop
from bubbles.bus.events import InboundMessage, OutboundMessage
from bubbles.bus.queue import MessageBus
from bubbles.channels.manager import ChannelManager
from bubbles.channels.wechat import (
    MSG_TYPE_TEXT,
    WCFERRY_COMMAND_PORT,
    WCFERRY_MUTEX_NAME,
    WcferryCleanupError,
    WeChatChannel,
    _assert_wcferry_ports_idle,
    _cleanup_wcferry_client,
    _construct_wcferry,
    _wait_for_wcferry_ports_released,
    _WcferryInstanceMutex,
    _WcferryLease,
    _WcferryRequestedExitError,
)
from bubbles.cli import commands as _cli_commands  # noqa: F401
from bubbles.cli.gateway_cmd import (
    UnsafeWcferryShutdownError,
    _arm_service_failure_exit_watchdog,
    _arm_upgrade_exit_watchdog,
    _arm_wcferry_cleanup_watchdog,
    _notify_upgrade_result,
    _publish_gateway_readiness,
    _run_gateway_runtime,
)
from bubbles.config.schema import Config, GatewayUpdateConfig, WeChatConfig
from bubbles.gateway_control import (
    CONTROL_PROTOCOL_VERSION,
    GATEWAY_INSTANCE_ID_ENV,
    NO_RESTART_EXIT_CODE,
    POST_DELIVERY_ACTION_KEY,
    SUPERVISOR_STOP_ACTION,
    UPGRADE_ACTION,
    UPGRADE_EXIT_CODE,
    UPGRADE_REQUEST_ID_ENV,
    UPGRADE_REQUEST_ID_KEY,
    GatewayControl,
)
from bubbles.session.manager import SessionManager

ADMIN_WXID = "wxid_owner"


def _make_control(
    data_dir,
    *,
    enabled: bool = True,
    admins: tuple[str, ...] = (ADMIN_WXID,),
    supervised: bool = True,
    remote: str = "bubblebot",
    branch: str = "main",
) -> GatewayControl:
    """Single adaptation point for the production constructor."""
    return GatewayControl(
        GatewayUpdateConfig(
            enabled=enabled,
            admins={"wechat": list(admins)} if admins else {},
            remote=remote,
            branch=branch,
        ),
        data_dir,
        supervised=supervised,
    )


def _upgrade_message(
    *,
    sender_id: str = ADMIN_WXID,
    chat_id: str | None = None,
    message_id: str = "1001",
    is_group: bool = False,
) -> InboundMessage:
    return InboundMessage(
        channel="wechat",
        sender_id=sender_id,
        chat_id=chat_id or ("ops@chatroom" if is_group else sender_id),
        content="/upgrade",
        metadata={
            "is_group": is_group,
            "message_id": message_id,
            "respond": True,
        },
    )


async def _maybe_await(value: Any) -> Any:
    if inspect.isawaitable(value):
        return await value
    return value


def _reply_for(request: InboundMessage, decision: Any) -> OutboundMessage:
    return OutboundMessage(
        channel=request.channel,
        chat_id=request.chat_id,
        content=decision.content,
        metadata=dict(decision.metadata),
    )


def _action_name(action: Any) -> str:
    value = getattr(action, "kind", getattr(action, "action", action))
    return str(getattr(value, "value", value))


@pytest.mark.asyncio
async def test_private_wechat_admin_upgrade_waits_for_delivery(tmp_path) -> None:
    control = _make_control(tmp_path)
    request = _upgrade_message()

    decision = await _maybe_await(control.prepare_upgrade(request))

    assert decision.accepted is True
    assert decision.content.strip()
    assert decision.metadata, "accepted reply needs a token for post-delivery correlation"

    action_task = asyncio.create_task(control.wait_for_action())
    await asyncio.sleep(0)
    assert not action_task.done(), "preparing an upgrade must not restart before reply delivery"

    reply = _reply_for(request, decision)
    await _maybe_await(control.handle_delivery_result(reply, delivered=True))

    action = await asyncio.wait_for(action_task, timeout=0.2)
    assert _action_name(action) == "upgrade"


class _ProviderMustNotRun:
    @staticmethod
    def get_default_model() -> str:
        return "test-model"

    async def chat(self, **_kwargs):
        raise AssertionError("/upgrade must be handled before the model and session binding")


@pytest.mark.asyncio
async def test_agent_loop_handles_upgrade_before_session_binding(tmp_path) -> None:
    control = _make_control(tmp_path)
    loop = AgentLoop(
        bus=MessageBus(),
        provider=_ProviderMustNotRun(),
        max_tokens=100,
        memory_window=10,
        context_limit=1_000_000,
        session_manager=SessionManager(sessions_dir=tmp_path / "sessions"),
        gateway_control=control,
    )
    loop._session_bindings = {}
    request = _upgrade_message(message_id="unbound-upgrade")
    assert not loop._session_bindings

    response = await loop._process_message(request)

    assert response is not None
    assert response.content.strip()
    assert response.metadata[POST_DELIVERY_ACTION_KEY] == "upgrade"
    assert response.metadata[UPGRADE_REQUEST_ID_KEY] == "unbound-upgrade"
    assert control.state == "waiting_delivery"


@pytest.mark.asyncio
async def test_upgrade_command_rejects_arguments_without_preparing(tmp_path) -> None:
    control = _make_control(tmp_path)
    loop = AgentLoop(
        bus=MessageBus(),
        provider=_ProviderMustNotRun(),
        max_tokens=100,
        memory_window=10,
        context_limit=1_000_000,
        session_manager=SessionManager(sessions_dir=tmp_path / "sessions"),
        gateway_control=control,
    )
    request = _upgrade_message(message_id="upgrade-with-arguments")
    request.content = "/upgrade now"

    response = await loop._process_message(request)

    assert response is not None
    assert "不接受参数" in response.content
    assert control.state == "idle"
    assert not control.request_path.exists()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("enabled", "admins", "supervised", "upgrade_msg"),
    [
        (False, (ADMIN_WXID,), True, _upgrade_message(message_id="disabled")),
        (True, (ADMIN_WXID,), True, _upgrade_message(sender_id="wxid_other", message_id="other")),
        (True, (), True, _upgrade_message(message_id="empty-admin-list")),
        (True, (ADMIN_WXID,), True, _upgrade_message(message_id="group", is_group=True)),
        (True, (ADMIN_WXID,), False, _upgrade_message(message_id="not-supervised")),
    ],
    ids=["disabled", "non-admin", "empty-admin-list", "group", "not-supervised"],
)
async def test_upgrade_rejects_unsafe_contexts(
    tmp_path,
    enabled: bool,
    admins: tuple[str, ...],
    supervised: bool,
    upgrade_msg: InboundMessage,
) -> None:
    control = _make_control(
        tmp_path,
        enabled=enabled,
        admins=admins,
        supervised=supervised,
    )

    decision = await _maybe_await(control.prepare_upgrade(upgrade_msg))

    assert decision.accepted is False
    assert decision.content.strip()
    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(control.wait_for_action(), timeout=0.01)


@pytest.mark.asyncio
async def test_concurrent_upgrade_requests_accept_exactly_one(tmp_path) -> None:
    control = _make_control(tmp_path)
    first = _upgrade_message(message_id="concurrent-1")
    second = _upgrade_message(message_id="concurrent-2")

    decisions = await asyncio.gather(
        _maybe_await(control.prepare_upgrade(first)),
        _maybe_await(control.prepare_upgrade(second)),
    )

    accepted = [(request, decision) for request, decision in zip((first, second), decisions)
                if decision.accepted]
    rejected = [decision for decision in decisions if not decision.accepted]
    assert len(accepted) == 1
    assert len(rejected) == 1
    assert rejected[0].content.strip()

    request, decision = accepted[0]
    action_task = asyncio.create_task(control.wait_for_action())
    await _maybe_await(
        control.handle_delivery_result(_reply_for(request, decision), delivered=True)
    )
    action = await asyncio.wait_for(action_task, timeout=0.2)
    assert _action_name(action) == "upgrade"


@pytest.mark.asyncio
async def test_failed_success_reply_delivery_rolls_back_pending_upgrade(tmp_path) -> None:
    control = _make_control(tmp_path)
    first = _upgrade_message(message_id="delivery-failed")
    first_decision = await _maybe_await(control.prepare_upgrade(first))
    assert first_decision.accepted is True

    await _maybe_await(
        control.handle_delivery_result(
            _reply_for(first, first_decision),
            delivered=False,
        )
    )

    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(control.wait_for_action(), timeout=0.01)

    retry = _upgrade_message(message_id="delivery-retry")
    retry_decision = await _maybe_await(control.prepare_upgrade(retry))
    assert retry_decision.accepted is True

    action_task = asyncio.create_task(control.wait_for_action())
    await _maybe_await(
        control.handle_delivery_result(
            _reply_for(retry, retry_decision),
            delivered=True,
        )
    )
    action = await asyncio.wait_for(action_task, timeout=0.2)
    assert _action_name(action) == "upgrade"


class _BlockingChannel:
    def __init__(self) -> None:
        self.send_started = asyncio.Event()
        self.release_send = asyncio.Event()
        self.send_finished = False

    async def send(self, _message: OutboundMessage) -> bool:
        self.send_started.set()
        await self.release_send.wait()
        self.send_finished = True
        return True


@pytest.mark.asyncio
async def test_channel_manager_reports_delivery_only_after_send_finishes(tmp_path) -> None:
    control = _make_control(tmp_path)
    request = _upgrade_message(message_id="delivery-order")
    decision = control.prepare_upgrade(request)
    reply = _reply_for(request, decision)

    bus = MessageBus()
    manager = ChannelManager(Config(), bus)
    channel = _BlockingChannel()
    manager.channels["wechat"] = channel
    manager.on_delivery_result = control.handle_delivery_result

    dispatch_task = asyncio.create_task(manager._dispatch_outbound())
    action_task = asyncio.create_task(control.wait_for_action())
    try:
        await bus.publish_outbound(reply)
        await asyncio.wait_for(channel.send_started.wait(), timeout=0.2)
        assert not channel.send_finished
        assert not action_task.done()

        channel.release_send.set()
        action = await asyncio.wait_for(action_task, timeout=0.2)

        assert channel.send_finished
        assert _action_name(action) == "upgrade"
    finally:
        dispatch_task.cancel()
        await asyncio.gather(dispatch_task, return_exceptions=True)


def test_upgrade_request_and_result_json_contract(tmp_path) -> None:
    control = _make_control(
        tmp_path,
        remote="origin",
        branch="release",
    )
    request = _upgrade_message(
        chat_id="wxid_owner",
        message_id="json-contract",
    )

    decision = control.prepare_upgrade(request)

    assert decision.accepted is True
    request_json = json.loads(control.request_path.read_text(encoding="utf-8"))
    assert request_json["request_id"] == "json-contract"
    assert request_json["remote"] == "origin"
    assert request_json["branch"] == "release"
    assert request_json["channel"] == "wechat"
    assert request_json["chat_id"] == "wxid_owner"

    control.result_path.write_text(
        json.dumps({
            "request_id": "json-contract",
            "channel": "wechat",
            "chat_id": "wxid_owner",
            "ok": True,
            "remote": "origin",
            "branch": "release",
            "old_revision": "a" * 40,
            "new_revision": "b" * 40,
        }),
        encoding="utf-8",
    )

    result = control.load_upgrade_result()

    assert result is not None
    assert result.request_id == "json-contract"
    formatted = result.format_message()
    assert "aaaaaaaa" in formatted
    assert "bbbbbbbb" in formatted
    assert "origin/release" in formatted

    control.clear_upgrade_result()
    assert not control.result_path.exists()
    assert control.load_upgrade_result() is None


def test_control_protocol_version_is_a_single_static_literal() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    source_path = repo_root / "bubbles" / "gateway_control.py"
    module = ast.parse(source_path.read_text(encoding="utf-8"))
    assignments = [
        node
        for node in module.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name)
            and target.id == "CONTROL_PROTOCOL_VERSION"
            for target in node.targets
        )
    ]

    assert len(assignments) == 1
    assert ast.literal_eval(assignments[0].value) == CONTROL_PROTOCOL_VERSION == 1

    supervisor = (repo_root / "scripts" / "bubbles-supervisor.ps1").read_text(
        encoding="utf-8"
    )
    assert "function Assert-GatewayControlProtocol" in supervisor
    assert '$stage = "control_protocol_validation"' in supervisor
    assert '$stage = "control_protocol_revalidation"' in supervisor
    startup_sync = supervisor[
        supervisor.index("function Invoke-StartupSync") :
        supervisor.index("function Invoke-Upgrade")
    ]
    assert startup_sync.count("Assert-GatewayControlProtocol") == 2
    start_gateway = supervisor[
        supervisor.index("function Start-GatewayProcess") :
        supervisor.index("function Stop-GatewayProcess")
    ]
    assert "Assert-GatewayControlProtocol" in start_gateway


def test_supervisor_never_restarts_an_unvalidated_checkout() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    supervisor = (repo_root / "scripts" / "bubbles-supervisor.ps1").read_text(
        encoding="utf-8-sig"
    )
    start_gateway = supervisor[
        supervisor.index("function Start-GatewayProcess") :
        supervisor.index("function Stop-GatewayProcess")
    ]
    invoke_upgrade = supervisor[
        supervisor.index("function Invoke-Upgrade") :
        supervisor.index("function Get-RestartDelaySeconds")
    ]

    assert "$null = Assert-CleanBranch" in start_gateway
    assert (
        "$canRestart = $candidateStopped -and -not "
        "[string]::IsNullOrWhiteSpace($oldRevision)"
    ) in invoke_upgrade


def test_supervisor_uses_utf8_bom_for_windows_powershell_51() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    supervisor_path = repo_root / "scripts" / "bubbles-supervisor.ps1"

    assert supervisor_path.read_bytes().startswith(b"\xef\xbb\xbf")


def test_wcferry_mutex_spans_windows_sessions() -> None:
    assert WCFERRY_MUTEX_NAME.startswith("Global\\")


@pytest.mark.parametrize(
    "invalid_result",
    [
        {"channel": None, "chat_id": "wxid_owner", "ok": False},
        {"channel": "wechat", "chat_id": None, "ok": False},
        {"channel": "wechat", "chat_id": "wxid_owner", "ok": "false"},
    ],
)
def test_invalid_upgrade_result_is_not_routable(tmp_path, invalid_result) -> None:
    control = _make_control(tmp_path)
    control.control_dir.mkdir(parents=True)
    control.result_path.write_text(json.dumps(invalid_result), encoding="utf-8")

    assert control.load_upgrade_result() is None
    assert not control.result_path.exists()


def test_gateway_readiness_requires_agent_and_wechat(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(UPGRADE_REQUEST_ID_ENV, "ready-contract")
    control = _make_control(tmp_path)

    assert control.publish_readiness(wechat_ready=True) is False
    assert not control.ready_path.exists()

    control.mark_agent_ready()
    assert control.publish_readiness(wechat_ready=False) is False
    assert not control.ready_path.exists()

    assert control.publish_readiness(wechat_ready=True) is True
    ready = json.loads(control.ready_path.read_text(encoding="utf-8"))
    assert ready["schema_version"] == 1
    assert ready["control_protocol_version"] == CONTROL_PROTOCOL_VERSION
    assert ready["request_id"] == "ready-contract"
    assert ready["ready_at"]
    assert not list(control.control_dir.glob(".gateway-ready.json.*.tmp"))


@pytest.mark.asyncio
async def test_agent_readiness_is_after_startup_work(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(UPGRADE_REQUEST_ID_ENV, "agent-startup")
    control = _make_control(tmp_path)
    loop = AgentLoop(
        bus=MessageBus(),
        provider=_ProviderMustNotRun(),
        max_tokens=100,
        memory_window=10,
        context_limit=1_000_000,
        session_manager=SessionManager(sessions_dir=tmp_path / "sessions"),
        gateway_control=control,
    )
    startup_entered = asyncio.Event()
    release_startup = asyncio.Event()

    async def blocked_startup() -> None:
        startup_entered.set()
        await release_startup.wait()

    loop._connect_mcp = blocked_startup
    run_task = asyncio.create_task(loop.run())
    try:
        await asyncio.wait_for(startup_entered.wait(), timeout=0.2)
        assert control.publish_readiness(wechat_ready=True) is False

        release_startup.set()
        await asyncio.wait_for(control.wait_for_agent_ready(), timeout=0.2)
        assert control.publish_readiness(wechat_ready=True) is True
    finally:
        loop.stop()
        run_task.cancel()
        await asyncio.gather(run_task, return_exceptions=True)


@pytest.mark.asyncio
async def test_readiness_publisher_waits_for_live_wechat(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(UPGRADE_REQUEST_ID_ENV, "late-wechat")
    control = _make_control(tmp_path)

    class _Channel:
        is_ready = False

    channel = _Channel()

    class _Channels:
        @staticmethod
        def get_channel(name: str):
            assert name == "wechat"
            return channel

    task = asyncio.create_task(
        _publish_gateway_readiness(control, _Channels(), poll_interval=0.001)
    )
    control.mark_agent_ready()
    await asyncio.sleep(0.01)
    assert not control.ready_path.exists()

    channel.is_ready = True
    await asyncio.wait_for(task, timeout=0.2)
    assert control.ready_path.exists()


@pytest.mark.asyncio
async def test_upgrade_result_notification_waits_for_supervisor_write(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(UPGRADE_REQUEST_ID_ENV, "result-after-ready")
    control = _make_control(tmp_path)
    sent: list[OutboundMessage] = []

    blocked = control.prepare_upgrade(_upgrade_message(message_id="too-early"))
    assert control.state == "startup_pending"
    assert blocked.accepted is False
    assert "supervisor" in blocked.content
    assert not control.request_path.exists()
    control.control_dir.mkdir(parents=True, exist_ok=True)
    control.in_progress_path.write_text("{}", encoding="utf-8")

    class _Channel:
        is_ready = True

        async def send(self, message: OutboundMessage) -> bool:
            sent.append(message)
            return True

    class _Channels:
        @staticmethod
        def get_channel(name: str):
            return _Channel() if name == "wechat" else None

    task = asyncio.create_task(
        _notify_upgrade_result(
            control,
            _Channels(),
            timeout=0.3,
            poll_interval=0.001,
        )
    )
    await asyncio.sleep(0.01)
    assert not task.done(), "new gateway must wait for the supervisor to write its result"

    control._write_json_atomic(control.result_path, {
        "request_id": "result-after-ready",
        "channel": "wechat",
        "chat_id": ADMIN_WXID,
        "ok": True,
        "remote": "bubblebot",
        "branch": "main",
        "old_revision": "a" * 40,
        "new_revision": "b" * 40,
    })

    await asyncio.sleep(0.01)
    assert sent == []
    assert control.state == "startup_pending"
    assert control.result_path.exists()
    control.in_progress_path.unlink()

    await asyncio.wait_for(task, timeout=0.3)
    assert len(sent) == 1
    assert "aaaaaaaa" in sent[0].content
    assert "bbbbbbbb" in sent[0].content
    assert not control.result_path.exists()
    assert control.state == "idle"

    after_commit = control.prepare_upgrade(_upgrade_message(message_id="after-commit"))
    assert after_commit.accepted is True
    assert control.request_path.exists()


@pytest.mark.asyncio
async def test_upgrade_result_notification_ignores_stale_request(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(UPGRADE_REQUEST_ID_ENV, "current-request")
    control = _make_control(tmp_path)

    class _Channel:
        is_ready = True
        sent: list[OutboundMessage] = []

        async def send(self, message: OutboundMessage) -> bool:
            self.sent.append(message)
            return True

    channel = _Channel()

    class _Channels:
        @staticmethod
        def get_channel(name: str):
            return channel if name == "wechat" else None

    control._write_json_atomic(control.result_path, {
        "request_id": "stale-request",
        "channel": "wechat",
        "chat_id": ADMIN_WXID,
        "ok": True,
    })
    task = asyncio.create_task(
        _notify_upgrade_result(
            control,
            _Channels(),
            timeout=0.3,
            poll_interval=0.001,
        )
    )
    await asyncio.sleep(0.01)
    assert channel.sent == []
    assert control.result_path.exists()

    control._write_json_atomic(control.result_path, {
        "request_id": "current-request",
        "channel": "wechat",
        "chat_id": ADMIN_WXID,
        "ok": False,
        "error": "test failure",
    })
    await asyncio.wait_for(task, timeout=0.3)
    assert len(channel.sent) == 1
    assert "test failure" in channel.sent[0].content
    assert control.state == "startup_pending"


@pytest.mark.asyncio
async def test_confirmed_upgrade_survives_non_wcf_cleanup_failures() -> None:
    calls: list[str] = []
    service_stop = asyncio.Event()

    class _Agent:
        async def run(self) -> None:
            await service_stop.wait()

        def stop(self) -> None:
            calls.append("agent.stop")
            service_stop.set()
            raise SystemExit("stop failed")

        async def cancel_active_tasks(self) -> None:
            calls.append("agent.cancel_active_tasks")
            raise RuntimeError("cancel failed")

        async def close_mcp(self) -> None:
            calls.append("agent.close_mcp")
            raise RuntimeError("mcp failed")

        async def close_sandboxes(self) -> None:
            calls.append("agent.close_sandboxes")
            raise RuntimeError("sandbox failed")

    class _Channels:
        enabled_channels = ["wechat"]

        async def start_all(self) -> None:
            await service_stop.wait()

        async def stop_all(self) -> dict[str, BaseException]:
            calls.append("channels.stop_all")
            service_stop.set()
            return {}

        @staticmethod
        def get_channel(_name: str):
            return None

    class _Cron:
        async def start(self) -> None:
            calls.append("cron.start")

        def stop(self) -> None:
            calls.append("cron.stop")
            raise RuntimeError("cron failed")

    class _Control:
        readiness_request_id = None

        @staticmethod
        async def wait_for_action() -> str:
            return UPGRADE_ACTION

        @staticmethod
        async def wait_for_supervisor_stop() -> str:
            await asyncio.Event().wait()
            raise AssertionError("unreachable")

        @staticmethod
        def load_upgrade_result():
            return None

    class _Watchdog:
        def cancel(self) -> None:
            calls.append("watchdog.cancel")

    def arm_watchdog() -> _Watchdog:
        calls.append("watchdog.arm")
        return _Watchdog()

    action = await asyncio.wait_for(
        _run_gateway_runtime(
            agent=_Agent(),
            channels=_Channels(),
            cron=_Cron(),
            gateway_control=_Control(),
            watchdog_factory=arm_watchdog,
        ),
        timeout=1,
    )

    assert action == UPGRADE_ACTION
    assert UPGRADE_EXIT_CODE == 75
    assert calls == [
        "cron.start",
        "channels.stop_all",
        "watchdog.arm",
        "agent.stop",
        "agent.cancel_active_tasks",
        "cron.stop",
        "agent.close_mcp",
        "agent.close_sandboxes",
        "watchdog.cancel",
    ]


@pytest.mark.parametrize(
    ("arm_watchdog", "expected_code"),
    [
        (_arm_upgrade_exit_watchdog, UPGRADE_EXIT_CODE),
        (_arm_service_failure_exit_watchdog, 1),
        (_arm_wcferry_cleanup_watchdog, NO_RESTART_EXIT_CODE),
    ],
)
def test_hard_exit_watchdogs_are_daemon_and_use_expected_code(
    monkeypatch: pytest.MonkeyPatch,
    arm_watchdog,
    expected_code: int,
) -> None:
    exit_codes: list[int] = []
    created: list[Any] = []

    class _Timer:
        def __init__(self, interval: float, callback) -> None:
            self.interval = interval
            self.callback = callback
            self.daemon = False
            self.started = False
            created.append(self)

        def start(self) -> None:
            self.started = True

    monkeypatch.setattr(os, "_exit", exit_codes.append)
    timer = arm_watchdog(timeout=7, timer_factory=_Timer)

    assert timer is created[0]
    assert timer.interval == 7
    assert timer.daemon is True
    assert timer.started is True
    timer.callback()
    assert exit_codes == [expected_code]


@pytest.mark.asyncio
async def test_enabled_channel_service_exit_propagates_nonzero_path() -> None:
    calls: list[str] = []
    stop_agent = asyncio.Event()

    class _Agent:
        async def run(self) -> None:
            await stop_agent.wait()

        def stop(self) -> None:
            calls.append("agent.stop")
            stop_agent.set()

        async def cancel_active_tasks(self) -> None:
            calls.append("agent.cancel_active_tasks")

        async def close_mcp(self) -> None:
            calls.append("agent.close_mcp")

        async def close_sandboxes(self) -> None:
            calls.append("agent.close_sandboxes")

    class _Channels:
        enabled_channels = ["wechat"]

        async def start_all(self) -> None:
            return

        async def stop_all(self) -> None:
            calls.append("channels.stop_all")

        @staticmethod
        def get_channel(_name: str):
            return None

    class _Cron:
        async def start(self) -> None:
            calls.append("cron.start")

        def stop(self) -> None:
            calls.append("cron.stop")

    class _Control:
        readiness_request_id = None

        @staticmethod
        async def wait_for_action() -> str:
            await asyncio.Event().wait()
            raise AssertionError("unreachable")

        @staticmethod
        async def wait_for_supervisor_stop() -> str:
            await asyncio.Event().wait()
            raise AssertionError("unreachable")

        @staticmethod
        def load_upgrade_result():
            return None

    class _FailureWatchdog:
        def cancel(self) -> None:
            calls.append("failure_watchdog.cancel")

    def arm_failure_watchdog() -> _FailureWatchdog:
        calls.append("failure_watchdog.arm")
        return _FailureWatchdog()

    with pytest.raises(RuntimeError, match="Enabled channel services stopped"):
        await _run_gateway_runtime(
            agent=_Agent(),
            channels=_Channels(),
            cron=_Cron(),
            gateway_control=_Control(),
            watchdog_factory=lambda: pytest.fail("upgrade watchdog must not arm"),
            failure_watchdog_factory=arm_failure_watchdog,
        )

    assert calls == [
        "cron.start",
        "channels.stop_all",
        "failure_watchdog.arm",
        "agent.stop",
        "agent.cancel_active_tasks",
        "cron.stop",
        "agent.close_mcp",
        "agent.close_sandboxes",
        "failure_watchdog.cancel",
    ]


@pytest.mark.asyncio
async def test_channel_manager_propagates_unexpected_channel_stop() -> None:
    blocking_cancelled = asyncio.Event()

    class _StoppedChannel:
        async def start(self) -> None:
            return

        async def stop(self) -> None:
            return

    class _BlockingChannel:
        async def start(self) -> None:
            try:
                await asyncio.Event().wait()
            finally:
                blocking_cancelled.set()

        async def stop(self) -> None:
            return

    manager = ChannelManager(Config(), MessageBus())
    manager.channels["wechat"] = _StoppedChannel()
    manager.channels["telegram"] = _BlockingChannel()

    with pytest.raises(RuntimeError, match="Channel wechat stopped unexpectedly"):
        await asyncio.wait_for(manager.start_all(), timeout=0.2)
    assert blocking_cancelled.is_set()
    await manager.stop_all()


def test_matrix_start_awaits_its_sync_task() -> None:
    matrix_path = Path(__file__).parents[1] / "bubbles" / "channels" / "matrix.py"
    module = ast.parse(matrix_path.read_text(encoding="utf-8"))
    matrix_class = next(
        node
        for node in module.body
        if isinstance(node, ast.ClassDef) and node.name == "MatrixChannel"
    )
    start_method = next(
        node
        for node in matrix_class.body
        if isinstance(node, ast.AsyncFunctionDef) and node.name == "start"
    )

    assert any(
        isinstance(node, ast.Await)
        and isinstance(node.value, ast.Attribute)
        and isinstance(node.value.value, ast.Name)
        and node.value.value.id == "self"
        and node.value.attr == "_sync_task"
        for node in ast.walk(start_method)
    )


class _FakeBus:
    def __init__(self) -> None:
        self.inbound: list[InboundMessage] = []

    async def publish_inbound(self, message: InboundMessage) -> None:
        self.inbound.append(message)


class _FakeWcferryMessageThread:
    def __init__(self) -> None:
        self.alive = True

    def is_alive(self) -> bool:
        return self.alive

    def join(self, _timeout: float | None = None) -> None:
        self.alive = False


class _FakePrivateTextMessage:
    type = MSG_TYPE_TEXT
    content = "/upgrade"
    xml = ""
    sender = ADMIN_WXID
    roomid = ""
    id = 987654321

    @staticmethod
    def from_self() -> bool:
        return False

    @staticmethod
    def from_group() -> bool:
        return False

    @staticmethod
    def is_at(_wxid: str) -> bool:
        return False


@pytest.mark.asyncio
async def test_wechat_forwards_message_id_as_string() -> None:
    bus = _FakeBus()
    channel = WeChatChannel(WeChatConfig(enabled=True), bus)
    channel.wxid = "wxid_bot"

    await channel._process_msg(_FakePrivateTextMessage())

    assert len(bus.inbound) == 1
    assert bus.inbound[0].metadata["message_id"] == "987654321"


def test_wechat_is_not_ready_from_partial_client_state() -> None:
    channel = WeChatChannel(WeChatConfig(enabled=True), _FakeBus())
    channel._running = True
    channel.wcf = object()

    assert channel.is_ready is False


def test_wechat_ready_requires_live_internal_message_thread() -> None:
    class _Thread:
        @staticmethod
        def is_alive() -> bool:
            return False

    class _Socket:
        pipes = (object(),)

    class _Wcf:
        msg_socket = _Socket()

        @staticmethod
        def is_receiving_msg() -> bool:
            return True

    channel = WeChatChannel(WeChatConfig(enabled=True), _FakeBus())
    channel._running = True
    channel._ready = True
    channel.wcf = _Wcf()
    channel._recv_thread = _FakeWcferryMessageThread()
    channel._wcferry_message_threads = (_Thread(),)

    assert channel.is_ready is False


@pytest.mark.asyncio
async def test_wechat_start_waits_for_internal_message_pipe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    receiving_enabled = threading.Event()

    class _Socket:
        def __init__(self) -> None:
            self.pipes: tuple[object, ...] = ()
            self.closed = False

        def close(self) -> None:
            self.closed = True

    class _Wcf:
        _local_mode = False
        _is_running = True
        _is_receiving_msg = False

        def __init__(self) -> None:
            self.cmd_socket = _Socket()
            self.msg_socket = _Socket()
            self.message_thread: threading.Thread | None = None

        @staticmethod
        def is_login() -> bool:
            return True

        @staticmethod
        def get_self_wxid() -> str:
            return "wxid_bot"

        @staticmethod
        def get_user_info() -> dict[str, str]:
            return {"home": ""}

        @staticmethod
        def query_sql(_database: str, _query: str) -> list[dict[str, str]]:
            return []

        def enable_receiving_msg(self) -> bool:
            self._is_receiving_msg = True

            def listen() -> None:
                while self._is_receiving_msg:
                    time.sleep(0.001)

            self.message_thread = threading.Thread(
                target=listen,
                name="GetMessage",
                daemon=True,
            )
            self.message_thread.start()
            receiving_enabled.set()
            return True

        def is_receiving_msg(self) -> bool:
            return self._is_receiving_msg

        @staticmethod
        def get_msg():
            time.sleep(0.001)
            raise wechat_module.Empty

        def disable_recv_msg(self) -> int:
            self._is_receiving_msg = False
            return 0

    fake_wcf = _Wcf()
    monkeypatch.setattr(
        wechat_module,
        "Wcf",
        lambda *, port, block: fake_wcf,
    )
    monkeypatch.setattr(
        wechat_module,
        "WECHAT_MESSAGE_TRANSPORT_POLL_INTERVAL_SECONDS",
        0.001,
    )
    monkeypatch.setattr(
        wechat_module,
        "WECHAT_RECEIVER_HEALTH_INTERVAL_SECONDS",
        0.001,
    )
    monkeypatch.setattr(
        wechat_module,
        "prune_wechat_image_cache",
        lambda *_args, **_kwargs: 0,
    )
    channel = WeChatChannel(WeChatConfig(enabled=True), _FakeBus())

    start_task = asyncio.create_task(channel.start())
    assert await asyncio.to_thread(receiving_enabled.wait, 1)
    await asyncio.sleep(0.01)
    assert channel.is_ready is False
    assert channel._recv_thread is None

    fake_wcf.msg_socket.pipes = (object(),)
    for _ in range(100):
        if channel.is_ready:
            break
        await asyncio.sleep(0.001)
    assert channel.is_ready is True

    await channel.stop()
    await asyncio.wait_for(start_task, timeout=1)


@pytest.mark.asyncio
async def test_wcferry_named_mutex_blocks_second_constructor_and_releases(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _Kernel32:
        def __init__(self) -> None:
            self.owned = False
            self.next_handle = 1
            self.closed: list[int] = []
            self.released: list[int] = []

        def CreateMutexW(self, _security, _initial_owner, _name: str) -> int:  # noqa: N802
            handle = self.next_handle
            self.next_handle += 1
            return handle

        def WaitForSingleObject(self, _handle: int, _timeout: int) -> int:  # noqa: N802
            if self.owned:
                return 0x00000102
            self.owned = True
            return 0x00000000

        def ReleaseMutex(self, handle: int) -> int:  # noqa: N802
            assert self.owned
            self.owned = False
            self.released.append(handle)
            return 1

        def CloseHandle(self, handle: int) -> int:  # noqa: N802
            self.closed.append(handle)
            return 1

    kernel32 = _Kernel32()
    first = _WcferryInstanceMutex(enabled=True, kernel32=kernel32)
    second = _WcferryInstanceMutex(enabled=True, kernel32=kernel32)
    first.acquire()
    constructor_calls = 0

    def construct_wcf(*, port: int, block: bool):
        assert port == wechat_module.WCFERRY_COMMAND_PORT
        assert block is False
        nonlocal constructor_calls
        constructor_calls += 1
        raise AssertionError("Wcf must not be constructed while the mutex is occupied")

    monkeypatch.setattr(wechat_module, "Wcf", construct_wcf)
    monkeypatch.setattr(wechat_module, "prune_wechat_image_cache", lambda *_args, **_kwargs: 0)
    channel = WeChatChannel(
        WeChatConfig(enabled=True),
        _FakeBus(),
        wcferry_mutex=second,
    )

    with pytest.raises(RuntimeError, match="另一 Bubblebot 进程"):
        await channel.start()

    assert constructor_calls == 0
    assert first.held is True
    assert second.held is False
    assert channel.is_running is False

    first.release()
    second.acquire()
    assert second.held is True
    second.release()
    assert kernel32.owned is False
    assert kernel32.released == [1, 3]
    assert kernel32.closed == [2, 1, 3]

    third = _WcferryInstanceMutex(enabled=True, kernel32=kernel32)
    failing_channel = WeChatChannel(
        WeChatConfig(enabled=True),
        _FakeBus(),
        wcferry_mutex=third,
    )
    with pytest.raises(
        WcferryCleanupError,
        match="WCFerry 构造未完整返回",
    ) as exc_info:
        await failing_channel.start()
    assert isinstance(exc_info.value.__cause__, AssertionError)
    assert "Wcf must not be constructed" in str(exc_info.value.__cause__)
    assert constructor_calls == 1
    assert third.held is False
    assert kernel32.owned is False
    assert kernel32.released == [1, 3, 4]
    assert kernel32.closed == [2, 1, 3, 4]


@pytest.mark.asyncio
async def test_wechat_receiver_loss_fails_channel_service(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _Wcf:
        def __init__(self) -> None:
            self.receiving = True

        @staticmethod
        def get_self_wxid() -> str:
            return "wxid_bot"

        @staticmethod
        def is_login() -> bool:
            return True

        @staticmethod
        def get_user_info() -> dict[str, str]:
            return {"home": ""}

        @staticmethod
        def query_sql(_database: str, _query: str) -> list[dict[str, str]]:
            return []

        def enable_receiving_msg(self) -> bool:
            self.receiving = True
            return True

        def is_receiving_msg(self) -> bool:
            return self.receiving

        def get_msg(self):
            self.receiving = False
            raise wechat_module.Empty

        def disable_recv_msg(self) -> None:
            self.receiving = False

    fake_wcf = _Wcf()
    monkeypatch.setattr(
        wechat_module,
        "Wcf",
        lambda *, port, block: fake_wcf,
    )
    monkeypatch.setattr(
        wechat_module,
        "WECHAT_RECEIVER_HEALTH_INTERVAL_SECONDS",
        0.001,
    )
    monkeypatch.setattr(wechat_module, "prune_wechat_image_cache", lambda *_args, **_kwargs: 0)
    message_thread = _FakeWcferryMessageThread()
    channel = WeChatChannel(
        WeChatConfig(enabled=True),
        _FakeBus(),
        wcferry_message_transport_check=lambda _client: True,
        wcferry_message_thread_capture=lambda _before: (message_thread,),
    )

    with pytest.raises(RuntimeError, match="receiver stopped unexpectedly"):
        await asyncio.wait_for(channel.start(), timeout=0.2)

    assert channel.is_ready is False
    assert channel.is_running is False
    await channel.stop()


@pytest.mark.asyncio
async def test_gateway_stop_request_requires_matching_instance_and_writes_ack(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(GATEWAY_INSTANCE_ID_ENV, "instance-current")
    control = _make_control(tmp_path)
    control._write_json_atomic(
        control.stop_request_path,
        {
            "schema_version": 1,
            "control_protocol_version": CONTROL_PROTOCOL_VERSION + 1,
            "instance_id": "instance-current",
            "reason": "test",
        },
    )
    task = asyncio.create_task(
        control.wait_for_supervisor_stop(poll_interval=0.001)
    )
    await asyncio.sleep(0.01)
    assert not task.done()

    control._write_json_atomic(
        control.stop_request_path,
        {
            "schema_version": 1,
            "control_protocol_version": CONTROL_PROTOCOL_VERSION,
            "instance_id": "instance-stale",
            "reason": "test",
        },
    )
    await asyncio.sleep(0.01)
    assert not task.done()

    control._write_json_atomic(
        control.stop_request_path,
        {
            "schema_version": 1,
            "control_protocol_version": CONTROL_PROTOCOL_VERSION,
            "instance_id": "instance-current",
            "reason": "test",
        },
    )
    assert await asyncio.wait_for(task, timeout=0.2) == SUPERVISOR_STOP_ACTION
    assert control.publish_stopped() is True
    ack = json.loads(control.stopped_path.read_text(encoding="utf-8"))
    assert ack["schema_version"] == 1
    assert ack["control_protocol_version"] == CONTROL_PROTOCOL_VERSION
    assert ack["instance_id"] == "instance-current"
    assert ack["stopped_at"]


def test_wcferry_lease_is_fail_closed_until_proven_cleanup(tmp_path: Path) -> None:
    path = tmp_path / "control" / "wcferry-lease.json"
    first = _WcferryLease(path, enabled=True)
    second = _WcferryLease(path, enabled=True)

    first.acquire()
    assert first.owned is True
    assert path.exists()
    with pytest.raises(RuntimeError, match="未清理的 WCFerry 安全标记"):
        second.acquire()

    first.release()
    second.acquire()
    assert second.owned is True
    second.release()
    assert not path.exists()


def test_wcferry_port_preflight_refuses_existing_rpc_listener(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        wechat_module,
        "_listening_ipv4_ports",
        lambda **_kwargs: {WCFERRY_COMMAND_PORT},
    )
    with pytest.raises(RuntimeError, match="旧 WCFerry RPC"):
        _assert_wcferry_ports_idle(enabled=True)


def test_wcferry_cleanup_refuses_lingering_rpc_listener(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        wechat_module,
        "_listening_ipv4_ports",
        lambda **_kwargs: {WCFERRY_COMMAND_PORT},
    )

    with pytest.raises(WcferryCleanupError, match="端口仍在监听"):
        _wait_for_wcferry_ports_released(enabled=True, timeout=0)


def test_wcferry_constructor_converts_python_os_exit() -> None:
    original_os = os

    def factory(*, port: int, block: bool):
        assert port == WCFERRY_COMMAND_PORT
        assert block is False
        os._exit(-2)

    with pytest.raises(_WcferryRequestedExitError) as exc_info:
        _construct_wcferry(factory, port=WCFERRY_COMMAND_PORT)
    assert exc_info.value.exit_code == -2
    assert os is original_os


def test_wcferry_cleanup_owns_native_destroy_after_disable_abort() -> None:
    class _Sdk:
        def __init__(self) -> None:
            self.calls = 0

        def WxDestroySDK(self) -> int:  # noqa: N802
            self.calls += 1
            return 0

    class _Socket:
        def __init__(self) -> None:
            self.closed = False

        def close(self) -> None:
            self.closed = True

    class _Client:
        _local_mode = True
        _is_running = True
        _is_receiving_msg = True

        def __init__(self) -> None:
            self.sdk = _Sdk()
            self.cmd_socket = _Socket()
            self.msg_socket = _Socket()

        @staticmethod
        def disable_recv_msg() -> None:
            raise SystemExit("RPC cleanup aborted")

    client = _Client()
    _cleanup_wcferry_client(client)

    assert client.sdk.calls == 1
    assert client.cmd_socket.closed is True
    assert client.msg_socket.closed is True
    assert client._is_running is False
    assert client._is_receiving_msg is False


@pytest.mark.parametrize("destroy_status", [1, -2])
def test_wcferry_cleanup_rejects_unconfirmed_native_detach(
    destroy_status: int,
) -> None:
    class _Sdk:
        def __init__(self) -> None:
            self.calls = 0

        def WxDestroySDK(self) -> int:  # noqa: N802
            self.calls += 1
            return destroy_status

    class _Client:
        _local_mode = True
        _is_running = True
        _is_receiving_msg = False

        def __init__(self) -> None:
            self.sdk = _Sdk()

        def cleanup(self) -> None:
            if self._is_running:
                self.sdk.WxDestroySDK()

    client = _Client()
    with pytest.raises(WcferryCleanupError, match="禁止自动重启"):
        _cleanup_wcferry_client(client)
    assert client._is_running is False
    client.cleanup()  # Simulate WCFerry's registered atexit callback.
    assert client.sdk.calls == 1


def test_wcferry_native_destroy_guard_refuses_a_second_attempt() -> None:
    class _Sdk:
        def __init__(self) -> None:
            self.calls = 0

        def WxDestroySDK(self) -> int:  # noqa: N802
            self.calls += 1
            return 1

    class _Client:
        sdk = _Sdk()

    client = _Client()
    channel = WeChatChannel(WeChatConfig(enabled=True), _FakeBus())

    with pytest.raises(WcferryCleanupError, match="禁止自动重启"):
        channel._destroy_wcferry_once(client=client)
    with pytest.raises(WcferryCleanupError, match="拒绝再次调用"):
        channel._destroy_wcferry_once(client=client)
    assert client.sdk.calls == 1


def test_wcferry_native_destroy_guard_rejects_concurrent_reentry() -> None:
    destroy_entered = threading.Event()
    allow_destroy_to_finish = threading.Event()
    errors: list[BaseException] = []

    class _Sdk:
        def __init__(self) -> None:
            self.calls = 0

        def WxDestroySDK(self) -> int:  # noqa: N802
            self.calls += 1
            destroy_entered.set()
            assert allow_destroy_to_finish.wait(timeout=1)
            return 0

    class _Client:
        sdk = _Sdk()

    client = _Client()
    channel = WeChatChannel(WeChatConfig(enabled=True), _FakeBus())

    def destroy() -> None:
        try:
            channel._destroy_wcferry_once(client=client)
        except BaseException as exc:
            errors.append(exc)

    first = threading.Thread(target=destroy)
    second = threading.Thread(target=destroy)
    first.start()
    assert destroy_entered.wait(timeout=1)
    second.start()
    second.join(timeout=1)
    assert not second.is_alive()

    allow_destroy_to_finish.set()
    first.join(timeout=1)
    assert not first.is_alive()

    assert client.sdk.calls == 1
    assert len(errors) == 1
    assert isinstance(errors[0], WcferryCleanupError)
    assert "并发重复卸载" in str(errors[0])
    assert channel._wcferry_native_destroy_state == "succeeded"


def test_wcferry_cleanup_joins_message_thread_before_destroy() -> None:
    calls: list[str] = []

    class _MessageThread:
        alive = True

        def join(self, _timeout: float) -> None:
            calls.append("message_thread.join")
            self.alive = False

        def is_alive(self) -> bool:
            return self.alive

    class _Socket:
        def __init__(self, name: str) -> None:
            self.name = name

        def close(self) -> None:
            calls.append(f"{self.name}.close")

    class _Sdk:
        @staticmethod
        def WxDestroySDK() -> int:  # noqa: N802
            calls.append("native.destroy")
            return 0

    class _Client:
        _local_mode = True
        _is_running = True
        _is_receiving_msg = True
        sdk = _Sdk()
        msg_socket = _Socket("msg")
        cmd_socket = _Socket("cmd")

        @staticmethod
        def disable_recv_msg() -> int:
            calls.append("receive.disable")
            return 0

    _cleanup_wcferry_client(
        _Client(),
        message_threads=(_MessageThread(),),
    )

    assert calls == [
        "receive.disable",
        "msg.close",
        "message_thread.join",
        "cmd.close",
        "native.destroy",
    ]


def test_wcferry_cleanup_refuses_destroy_while_message_thread_is_alive() -> None:
    class _MessageThread:
        @staticmethod
        def join(_timeout: float) -> None:
            return None

        @staticmethod
        def is_alive() -> bool:
            return True

    class _Sdk:
        def __init__(self) -> None:
            self.calls = 0

        def WxDestroySDK(self) -> int:  # noqa: N802
            self.calls += 1
            return 0

    class _Socket:
        @staticmethod
        def close() -> None:
            return None

    class _Client:
        _local_mode = True
        _is_running = True
        _is_receiving_msg = True

        def __init__(self) -> None:
            self.sdk = _Sdk()
            self.msg_socket = _Socket()
            self.cmd_socket = _Socket()

        def disable_recv_msg(self) -> int:
            self._is_receiving_msg = False
            return 0

    client = _Client()
    with pytest.raises(WcferryCleanupError, match="内部消息线程未退出"):
        _cleanup_wcferry_client(
            client,
            message_threads=(_MessageThread(),),
            message_thread_join_timeout=0,
        )

    assert client._is_running is False
    assert client.sdk.calls == 0


@pytest.mark.parametrize("captured_thread_count", [0, 2])
@pytest.mark.asyncio
async def test_unverified_wcferry_message_thread_preserves_singleton_guards(
    captured_thread_count: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _Lease:
        owned = False

        def acquire(self) -> None:
            self.owned = True

        def release(self) -> None:
            self.owned = False

    class _Mutex:
        held = False

        def acquire(self) -> None:
            self.held = True

        def release(self) -> None:
            self.held = False

    class _Watchdog:
        def cancel(self) -> None:
            return None

    class _Socket:
        pipes = (object(),)

        def close(self) -> None:
            return None

    class _Sdk:
        def __init__(self) -> None:
            self.calls = 0

        def WxDestroySDK(self) -> int:  # noqa: N802
            self.calls += 1
            return 0

    class _Wcf:
        _local_mode = True
        _is_running = True
        _is_receiving_msg = False

        def __init__(self) -> None:
            self.sdk = _Sdk()
            self.cmd_socket = _Socket()
            self.msg_socket = _Socket()

        @staticmethod
        def is_login() -> bool:
            return True

        @staticmethod
        def get_self_wxid() -> str:
            return "wxid_bot"

        @staticmethod
        def get_user_info() -> dict[str, str]:
            return {"home": ""}

        @staticmethod
        def query_sql(_database: str, _query: str) -> list[dict[str, str]]:
            return []

        def enable_receiving_msg(self) -> bool:
            self._is_receiving_msg = True
            return True

        def disable_recv_msg(self) -> int:
            self._is_receiving_msg = False
            return 0

    fake_wcf = _Wcf()
    lease = _Lease()
    mutex = _Mutex()
    captured_threads = tuple(
        _FakeWcferryMessageThread() for _ in range(captured_thread_count)
    )
    monkeypatch.setattr(wechat_module, "Wcf", lambda *, port, block: fake_wcf)
    monkeypatch.setattr(
        wechat_module,
        "prune_wechat_image_cache",
        lambda *_args, **_kwargs: 0,
    )
    channel = WeChatChannel(
        WeChatConfig(enabled=True),
        _FakeBus(),
        wcferry_mutex=mutex,
        wcferry_lease=lease,
        wcferry_port_check=lambda: None,
        wcferry_port_release_check=lambda: None,
        wcferry_cleanup_watchdog_factory=_Watchdog,
        wcferry_message_transport_check=lambda _client: True,
        wcferry_message_thread_capture=lambda _before: captured_threads,
    )

    with pytest.raises(WcferryCleanupError, match="无法唯一确认"):
        await channel.start()

    assert fake_wcf.sdk.calls == 0
    assert lease.owned is True
    assert mutex.held is True
    assert channel.wcf is fake_wcf


@pytest.mark.asyncio
async def test_wcferry_cleanup_failure_blocks_upgrade_restart() -> None:
    calls: list[str] = []
    stop = asyncio.Event()

    class _Agent:
        async def run(self) -> None:
            await stop.wait()

        def stop(self) -> None:
            stop.set()

        async def cancel_active_tasks(self) -> None:
            return None

        async def close_mcp(self) -> None:
            return None

        async def close_sandboxes(self) -> None:
            return None

    class _Channels:
        enabled_channels = ["wechat"]

        async def start_all(self) -> None:
            await stop.wait()

        async def stop_all(self) -> dict[str, BaseException]:
            stop.set()
            return {"wechat": WcferryCleanupError("detach failed")}

        @staticmethod
        def get_channel(_name: str):
            return None

    class _Cron:
        async def start(self) -> None:
            return None

        def stop(self) -> None:
            return None

    class _Control:
        readiness_request_id = None

        @staticmethod
        async def wait_for_action() -> str:
            return UPGRADE_ACTION

        @staticmethod
        async def wait_for_supervisor_stop() -> str:
            await asyncio.Event().wait()
            raise AssertionError("unreachable")

        @staticmethod
        def load_upgrade_result():
            return None

    class _Watchdog:
        def cancel(self) -> None:
            calls.append("wcf_watchdog.cancel")

    with pytest.raises(UnsafeWcferryShutdownError):
        await _run_gateway_runtime(
            agent=_Agent(),
            channels=_Channels(),
            cron=_Cron(),
            gateway_control=_Control(),
            watchdog_factory=lambda: pytest.fail("upgrade watchdog must not arm"),
            wcferry_watchdog_factory=lambda: _Watchdog(),
        )
    assert calls == ["wcf_watchdog.cancel"]


@pytest.mark.asyncio
async def test_supervisor_stop_ack_is_published_after_wcferry_cleanup() -> None:
    calls: list[str] = []
    stop = asyncio.Event()

    class _Agent:
        async def run(self) -> None:
            await stop.wait()

        def stop(self) -> None:
            calls.append("agent.stop")
            stop.set()

        async def cancel_active_tasks(self) -> None:
            return None

        async def close_mcp(self) -> None:
            return None

        async def close_sandboxes(self) -> None:
            return None

    class _Channels:
        enabled_channels = ["wechat"]

        async def start_all(self) -> None:
            await stop.wait()

        async def stop_all(self) -> dict[str, BaseException]:
            calls.append("wcf.cleanup")
            stop.set()
            return {}

        @staticmethod
        def get_channel(_name: str):
            return None

    class _Cron:
        async def start(self) -> None:
            return None

        def stop(self) -> None:
            return None

    class _Control:
        readiness_request_id = None
        instance_id = "instance-test"

        @staticmethod
        async def wait_for_action() -> str:
            await asyncio.Event().wait()
            raise AssertionError("unreachable")

        @staticmethod
        async def wait_for_supervisor_stop() -> str:
            return SUPERVISOR_STOP_ACTION

        @staticmethod
        def load_upgrade_result():
            return None

        @staticmethod
        def publish_stopped() -> bool:
            calls.append("stop.ack")
            return True

    class _Watchdog:
        def __init__(self, name: str) -> None:
            self.name = name

        def cancel(self) -> None:
            calls.append(f"{self.name}.cancel")

    action = await _run_gateway_runtime(
        agent=_Agent(),
        channels=_Channels(),
        cron=_Cron(),
        gateway_control=_Control(),
        no_restart_watchdog_factory=lambda: (
            calls.append("no_restart.arm") or _Watchdog("no_restart")
        ),
        wcferry_watchdog_factory=lambda: _Watchdog("wcf"),
    )

    assert action == SUPERVISOR_STOP_ACTION
    assert calls.index("wcf.cleanup") < calls.index("stop.ack")
    assert calls.index("stop.ack") < calls.index("agent.stop")
    assert "no_restart.arm" in calls
    assert "no_restart.cancel" in calls


@pytest.mark.asyncio
async def test_channel_manager_stops_wechat_before_other_channels() -> None:
    calls: list[str] = []

    class _Channel:
        def __init__(self, name: str) -> None:
            self.name = name

        async def stop(self) -> None:
            calls.append(self.name)

    manager = ChannelManager(Config(), MessageBus())
    manager.channels["telegram"] = _Channel("telegram")
    manager.channels["wechat"] = _Channel("wechat")
    manager.channels["matrix"] = _Channel("matrix")

    assert await manager.stop_all() == {}
    assert calls == ["wechat", "telegram", "matrix"]


@pytest.mark.asyncio
async def test_wechat_stop_cannot_race_past_startup_guard(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prune_started = threading.Event()
    allow_prune = threading.Event()
    constructor_calls = 0

    def blocking_prune(path, **_kwargs) -> int:
        if path.name == "wechat-outbound-images":
            prune_started.set()
            assert allow_prune.wait(timeout=1)
        return 0

    def construct_wcf(*, port: int, block: bool):
        assert block is False
        nonlocal constructor_calls
        constructor_calls += 1
        raise AssertionError(f"Wcf({port}) must not run after stop was requested")

    monkeypatch.setattr(wechat_module, "prune_wechat_image_cache", blocking_prune)
    monkeypatch.setattr(wechat_module, "Wcf", construct_wcf)
    channel = WeChatChannel(WeChatConfig(enabled=True), _FakeBus())

    start_task = asyncio.create_task(channel.start())
    assert await asyncio.to_thread(prune_started.wait, 1)
    stop_task = asyncio.create_task(channel.stop())
    await asyncio.sleep(0)
    assert not stop_task.done()

    allow_prune.set()
    await asyncio.wait_for(asyncio.gather(start_task, stop_task), timeout=1)

    assert constructor_calls == 0
    assert channel.wcf is None
    assert channel._wcferry_lease.owned is False
    assert channel._wcferry_mutex.held is False


@pytest.mark.asyncio
async def test_constructor_watchdog_failure_precedes_persistent_lease(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lease_path = tmp_path / "control" / "wcferry-lease.json"
    lease = _WcferryLease(lease_path, enabled=True)
    constructor_calls = 0

    def construct_wcf(*, port: int, block: bool):
        nonlocal constructor_calls
        constructor_calls += 1
        raise AssertionError(f"unexpected Wcf({port}, block={block})")

    def fail_watchdog():
        raise OSError("timer unavailable")

    monkeypatch.setattr(wechat_module, "Wcf", construct_wcf)
    monkeypatch.setattr(
        wechat_module,
        "prune_wechat_image_cache",
        lambda *_args, **_kwargs: 0,
    )
    channel = WeChatChannel(
        WeChatConfig(enabled=True),
        _FakeBus(),
        wcferry_lease=lease,
        wcferry_port_check=lambda: None,
        wcferry_cleanup_watchdog_factory=fail_watchdog,
    )

    with pytest.raises(WcferryCleanupError, match="构造阶段"):
        await channel.start()

    assert constructor_calls == 0
    assert lease.owned is False
    assert not lease_path.exists()


@pytest.mark.asyncio
async def test_wechat_constructor_thread_yields_to_safe_stop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    constructor_started = threading.Event()
    allow_constructor = threading.Event()
    cleanup_called = threading.Event()

    class _Watchdog:
        cancelled = False

        def cancel(self) -> None:
            self.cancelled = True

    class _Wcf:
        _local_mode = False
        _is_running = True
        _is_receiving_msg = False

        def disable_recv_msg(self) -> None:
            cleanup_called.set()
            self._is_running = False

    def construct_wcf(*, port: int, block: bool) -> _Wcf:
        assert port == WCFERRY_COMMAND_PORT
        assert block is False
        constructor_started.set()
        assert allow_constructor.wait(timeout=1)
        return _Wcf()

    watchdog = _Watchdog()
    monkeypatch.setattr(wechat_module, "Wcf", construct_wcf)
    monkeypatch.setattr(
        wechat_module,
        "prune_wechat_image_cache",
        lambda *_args, **_kwargs: 0,
    )
    channel = WeChatChannel(
        WeChatConfig(enabled=True),
        _FakeBus(),
        wcferry_cleanup_watchdog_factory=lambda: watchdog,
    )

    start_task = asyncio.create_task(channel.start())
    assert await asyncio.to_thread(constructor_started.wait, 1)
    stop_task = asyncio.create_task(channel.stop())
    await asyncio.sleep(0)
    assert channel._stop_requested is True
    assert not stop_task.done()

    allow_constructor.set()
    await asyncio.wait_for(asyncio.gather(start_task, stop_task), timeout=1)

    assert watchdog.cancelled is True
    assert cleanup_called.is_set()
    assert channel.wcf is None


@pytest.mark.asyncio
async def test_wechat_login_wait_yields_to_safe_stop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    login_polled = threading.Event()
    allow_login_result = threading.Event()
    constructed_with_block: list[bool] = []

    class _Wcf:
        _local_mode = False
        _is_running = True
        _is_receiving_msg = False

        @staticmethod
        def is_login() -> bool:
            login_polled.set()
            assert allow_login_result.wait(timeout=1)
            return False

        def disable_recv_msg(self) -> None:
            self._is_running = False

    def construct_wcf(*, port: int, block: bool) -> _Wcf:
        assert port == WCFERRY_COMMAND_PORT
        constructed_with_block.append(block)
        return _Wcf()

    monkeypatch.setattr(wechat_module, "Wcf", construct_wcf)
    monkeypatch.setattr(
        wechat_module,
        "WECHAT_LOGIN_POLL_INTERVAL_SECONDS",
        0.001,
    )
    monkeypatch.setattr(
        wechat_module,
        "prune_wechat_image_cache",
        lambda *_args, **_kwargs: 0,
    )
    channel = WeChatChannel(WeChatConfig(enabled=True), _FakeBus())

    start_task = asyncio.create_task(channel.start())
    assert await asyncio.to_thread(login_polled.wait, 1)
    stop_task = asyncio.create_task(channel.stop())
    await asyncio.sleep(0)
    assert channel._stop_requested is True
    assert not stop_task.done()

    allow_login_result.set()
    await asyncio.wait_for(asyncio.gather(start_task, stop_task), timeout=1)

    assert constructed_with_block == [False]
    assert channel.wcf is None
    assert channel.is_running is False


@pytest.mark.asyncio
async def test_wechat_rejects_duplicate_start_before_second_constructor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    constructor_calls = 0

    class _Wcf:
        _local_mode = False
        _is_running = True
        _is_receiving_msg = True

        @staticmethod
        def get_self_wxid() -> str:
            return "wxid_bot"

        @staticmethod
        def is_login() -> bool:
            return True

        @staticmethod
        def get_user_info() -> dict[str, str]:
            return {"home": ""}

        @staticmethod
        def query_sql(_database: str, _query: str) -> list[dict[str, str]]:
            return []

        def enable_receiving_msg(self) -> bool:
            return True

        def is_receiving_msg(self) -> bool:
            return self._is_receiving_msg

        @staticmethod
        def get_msg():
            time.sleep(0.001)
            raise wechat_module.Empty

        def cleanup(self) -> None:
            self._is_receiving_msg = False
            self._is_running = False

    def construct_wcf(*, port: int, block: bool) -> _Wcf:
        nonlocal constructor_calls
        assert port == WCFERRY_COMMAND_PORT
        assert block is False
        constructor_calls += 1
        return _Wcf()

    monkeypatch.setattr(wechat_module, "Wcf", construct_wcf)
    monkeypatch.setattr(
        wechat_module,
        "WECHAT_RECEIVER_HEALTH_INTERVAL_SECONDS",
        0.001,
    )
    monkeypatch.setattr(
        wechat_module,
        "prune_wechat_image_cache",
        lambda *_args, **_kwargs: 0,
    )
    message_thread = _FakeWcferryMessageThread()
    channel = WeChatChannel(
        WeChatConfig(enabled=True),
        _FakeBus(),
        wcferry_message_transport_check=lambda _client: True,
        wcferry_message_thread_capture=lambda _before: (message_thread,),
    )

    first_start = asyncio.create_task(channel.start())
    for _ in range(100):
        if channel.is_ready:
            break
        await asyncio.sleep(0.001)
    assert channel.is_ready

    with pytest.raises(WcferryCleanupError, match="拒绝重复启动"):
        await channel.start()

    assert constructor_calls == 1
    assert channel.is_ready
    assert channel.is_running

    await channel.stop()
    await asyncio.wait_for(first_start, timeout=1)
