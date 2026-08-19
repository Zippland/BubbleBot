"""Privileged gateway lifecycle control.

The running gateway never updates its own checkout. It accepts an explicit,
authenticated request, waits until the acknowledgement has actually been
delivered, then exits with a well-known code. An external supervisor owns
``git pull`` / dependency sync / process restart.
"""

from __future__ import annotations

import asyncio
import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from loguru import logger

if TYPE_CHECKING:
    from bubbles.bus.events import InboundMessage, OutboundMessage
    from bubbles.config.schema import GatewayUpdateConfig


CONTROL_PROTOCOL_VERSION = 1
UPGRADE_EXIT_CODE = 75
NO_RESTART_EXIT_CODE = 76
SUPERVISED_ENV = "BUBBLES_GATEWAY_SUPERVISED"
UPGRADE_REQUEST_ID_ENV = "BUBBLES_GATEWAY_UPGRADE_REQUEST_ID"
GATEWAY_INSTANCE_ID_ENV = "BUBBLES_GATEWAY_INSTANCE_ID"
POST_DELIVERY_ACTION_KEY = "_gateway_post_delivery_action"
UPGRADE_REQUEST_ID_KEY = "_gateway_upgrade_request_id"
UPGRADE_ACTION = "upgrade"
SUPERVISOR_STOP_ACTION = "supervisor_stop"


@dataclass(frozen=True)
class UpgradeDecision:
    """User-visible decision returned by :meth:`prepare_upgrade`."""

    accepted: bool
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class UpgradeResult:
    """Result written by the supervisor and reported after restart."""

    channel: str
    chat_id: str
    ok: bool
    remote: str
    branch: str
    request_id: str = ""
    old_revision: str = ""
    new_revision: str = ""
    error: str = ""

    def format_message(self) -> str:
        old = self.old_revision[:8] if self.old_revision else "unknown"
        new = self.new_revision[:8] if self.new_revision else "unknown"
        target = f"{self.remote}/{self.branch}"
        if self.ok:
            if self.old_revision and self.old_revision == self.new_revision:
                return f"✅ 热升级完成：代码已是最新（`{new}`），gateway 已重新启动。"
            return f"✅ 热升级完成：`{old}` → `{new}`（{target}），gateway 已重新启动。"
        detail = self.error.strip() or "git pull 未成功"
        return f"⚠️ 热升级失败，gateway 已重新启动原有版本。\n原因：{detail}"


class GatewayControl:
    """Single-process state machine for privileged gateway actions."""

    def __init__(
        self,
        config: "GatewayUpdateConfig",
        data_dir: Path,
        *,
        supervised: bool | None = None,
    ) -> None:
        self.config = config
        self.control_dir = data_dir / "control"
        self.request_path = self.control_dir / "upgrade-request.json"
        self.result_path = self.control_dir / "upgrade-result.json"
        self.in_progress_path = self.control_dir / "upgrade-in-progress.json"
        self.ready_path = self.control_dir / "gateway-ready.json"
        self.stop_request_path = self.control_dir / "gateway-stop-request.json"
        self.stopped_path = self.control_dir / "gateway-stopped.json"
        self.supervised = (
            os.environ.get(SUPERVISED_ENV) == "1" if supervised is None else supervised
        )
        startup_request_id = os.environ.get(UPGRADE_REQUEST_ID_ENV, "").strip()
        if len(startup_request_id) > 200:
            logger.warning(
                "Ignoring gateway readiness request id longer than 200 characters"
            )
            startup_request_id = ""
        self.readiness_request_id = startup_request_id or None
        instance_id = os.environ.get(GATEWAY_INSTANCE_ID_ENV, "").strip()
        if len(instance_id) > 200:
            logger.warning("Ignoring gateway instance id longer than 200 characters")
            instance_id = ""
        self.instance_id = instance_id or None
        self._agent_ready = asyncio.Event()
        self._readiness_published = False
        self._state = "startup_pending" if self.readiness_request_id else "idle"
        self._request_id: str | None = None
        self._action_event = asyncio.Event()

    @property
    def state(self) -> str:
        return self._state

    def prepare_upgrade(self, msg: "InboundMessage") -> UpgradeDecision:
        """Validate and reserve one upgrade request.

        No network or process work happens here. The request becomes actionable
        only after ``ChannelManager`` reports the acknowledgement as delivered.
        """
        if not self.config.enabled:
            return UpgradeDecision(False, "热升级未启用。请先配置 `gateway.update.enabled=true`。")
        if msg.channel != "wechat":
            return UpgradeDecision(False, "热升级目前只允许从微信私聊触发。")
        if bool((msg.metadata or {}).get("is_group")):
            return UpgradeDecision(False, "热升级只能在微信私聊中触发，群聊不会执行。")

        admins = self.config.admins.get(msg.channel, [])
        if not admins or msg.sender_id not in admins:
            return UpgradeDecision(False, "无权限执行热升级。")
        if not self.supervised:
            return UpgradeDecision(
                False,
                "当前 gateway 不是由 Windows supervisor 启动，不能安全自重启。"
                "请先运行 `scripts/install-bubbles-gateway-task.ps1`。",
            )

        trigger_id = str((msg.metadata or {}).get("message_id") or uuid4().hex)
        if self._state == "startup_pending":
            return UpgradeDecision(
                False,
                "gateway 正在等待 supervisor 确认刚完成的升级，请稍后再试。",
            )
        if self._state != "idle":
            if trigger_id == self._request_id:
                return UpgradeDecision(False, "这条热升级请求已经受理，正在重启。")
            return UpgradeDecision(False, "已有热升级请求正在处理，请等待 gateway 重新上线。")

        request = {
            "schema_version": 1,
            "action": UPGRADE_ACTION,
            "request_id": trigger_id,
            "channel": msg.channel,
            "chat_id": msg.chat_id,
            "sender_id": msg.sender_id,
            "remote": self.config.remote,
            "branch": self.config.branch,
            "requested_at": datetime.now(timezone.utc).isoformat(),
        }
        try:
            self._write_json_atomic(self.request_path, request)
        except OSError as exc:
            logger.error("Failed to persist gateway upgrade request: {}", exc)
            return UpgradeDecision(False, "无法写入升级请求，gateway 未重启。请检查本机日志。")

        self._state = "waiting_delivery"
        self._request_id = trigger_id
        return UpgradeDecision(
            True,
            "收到，准备热升级：回复送达后会退出当前 gateway，"
            f"由 supervisor 拉取 `{self.config.remote}/{self.config.branch}` 并重新启动。",
            {
                POST_DELIVERY_ACTION_KEY: UPGRADE_ACTION,
                UPGRADE_REQUEST_ID_KEY: trigger_id,
            },
        )

    async def handle_delivery_result(
        self,
        msg: "OutboundMessage",
        delivered: bool,
    ) -> None:
        """Advance or roll back the state after the acknowledgement send."""
        metadata = msg.metadata or {}
        if metadata.get(POST_DELIVERY_ACTION_KEY) != UPGRADE_ACTION:
            return
        if metadata.get(UPGRADE_REQUEST_ID_KEY) != self._request_id:
            return
        if self._state != "waiting_delivery":
            return

        if not delivered:
            self._state = "idle"
            self._request_id = None
            try:
                self.request_path.unlink(missing_ok=True)
            except OSError as exc:
                logger.warning("Failed to remove undelivered upgrade request: {}", exc)
            return

        self._state = "restart_pending"
        self._action_event.set()

    async def wait_for_action(self) -> str:
        """Wait until a delivered privileged action is ready to execute."""
        await self._action_event.wait()
        return UPGRADE_ACTION

    async def wait_for_supervisor_stop(
        self,
        *,
        poll_interval: float = 0.25,
    ) -> str:
        """Wait for a stop request addressed to this exact gateway instance."""
        if not self.supervised or self.instance_id is None:
            await asyncio.Event().wait()
            raise AssertionError("unreachable")

        last_error = ""
        while True:
            try:
                with self.stop_request_path.open(encoding="utf-8-sig") as file:
                    raw = json.load(file)
                if not isinstance(raw, dict):
                    raise ValueError("stop request must be an object")
                if raw.get("schema_version") != 1:
                    raise ValueError("unsupported schema_version")
                if raw.get("control_protocol_version") != CONTROL_PROTOCOL_VERSION:
                    raise ValueError("unsupported control_protocol_version")
                request_instance_id = raw.get("instance_id")
                if not isinstance(request_instance_id, str):
                    raise ValueError("instance_id must be a string")
                if request_instance_id == self.instance_id:
                    logger.warning(
                        "Supervisor requested a safe stop for gateway instance {}",
                        self.instance_id,
                    )
                    return SUPERVISOR_STOP_ACTION
            except FileNotFoundError:
                pass
            except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
                error = str(exc)
                if error != last_error:
                    logger.warning("Ignoring invalid gateway stop request: {}", exc)
                    last_error = error
            await asyncio.sleep(poll_interval)

    def publish_stopped(self) -> bool:
        """Acknowledge that WCFerry has been safely detached for this instance."""
        if self.instance_id is None:
            return False
        payload = {
            "schema_version": 1,
            "control_protocol_version": CONTROL_PROTOCOL_VERSION,
            "instance_id": self.instance_id,
            "stopped_at": datetime.now(timezone.utc).isoformat(),
        }
        try:
            self._write_json_atomic(self.stopped_path, payload)
        except OSError as exc:
            logger.error("Failed to publish safe gateway stop acknowledgement: {}", exc)
            return False
        return True

    def mark_agent_ready(self) -> None:
        """Record that AgentLoop completed its startup work."""
        self._agent_ready.set()

    async def wait_for_agent_ready(self) -> None:
        """Wait until AgentLoop has completed its startup work."""
        await self._agent_ready.wait()

    def publish_readiness(self, *, wechat_ready: bool) -> bool:
        """Publish the supervised startup handshake once both sides are ready.

        The caller owns the live WeChat channel and therefore supplies its
        readiness explicitly.  Keeping both predicates here prevents either
        AgentLoop startup or a half-constructed WCFerry client from publishing
        a false-positive handshake.
        """
        if self.readiness_request_id is None:
            return False
        if self._readiness_published:
            return True
        if not self._agent_ready.is_set() or not wechat_ready:
            return False

        payload = {
            "schema_version": 1,
            "control_protocol_version": CONTROL_PROTOCOL_VERSION,
            "request_id": self.readiness_request_id,
            "ready_at": datetime.now(timezone.utc).isoformat(),
        }
        try:
            self._write_json_atomic(self.ready_path, payload)
        except OSError as exc:
            logger.warning("Failed to publish gateway readiness: {}", exc)
            return False

        self._readiness_published = True
        logger.info(
            "Published gateway readiness for upgrade request {}",
            self.readiness_request_id,
        )
        return True

    def observe_upgrade_result(self, result: UpgradeResult) -> bool:
        """Open lifecycle commands after the supervisor's matching commit fence."""
        if not result.ok or result.request_id != self.readiness_request_id:
            return False
        try:
            if self.in_progress_path.exists():
                return False
        except OSError as exc:
            logger.warning("Unable to inspect gateway upgrade marker: {}", exc)
            return False
        if self._state == "idle":
            return True
        if self._state != "startup_pending":
            return False

        self._state = "idle"
        logger.info(
            "Observed committed gateway upgrade result for request {}",
            result.request_id,
        )
        return True

    def load_upgrade_result(self) -> UpgradeResult | None:
        """Load a supervisor result without consuming it."""
        try:
            with self.result_path.open(encoding="utf-8-sig") as file:
                raw = json.load(file)

            channel = raw["channel"]
            chat_id = raw["chat_id"]
            ok = raw["ok"]
            if not isinstance(channel, str) or not channel.strip():
                raise ValueError("channel must be a non-empty string")
            if not isinstance(chat_id, str) or not chat_id.strip():
                raise ValueError("chat_id must be a non-empty string")
            if not isinstance(ok, bool):
                raise ValueError("ok must be a boolean")
            request_id = raw.get("request_id")
            if request_id is not None and not isinstance(request_id, str):
                raise ValueError("request_id must be a string when present")

            return UpgradeResult(
                channel=channel,
                chat_id=chat_id,
                ok=ok,
                remote=str(raw.get("remote") or self.config.remote),
                branch=str(raw.get("branch") or self.config.branch),
                request_id=request_id or "",
                old_revision=str(raw.get("old_revision") or ""),
                new_revision=str(raw.get("new_revision") or ""),
                error=str(raw.get("error") or ""),
            )
        except FileNotFoundError:
            return None
        except OSError as exc:
            logger.warning("Unable to read gateway upgrade result: {}", exc)
            return None
        except (ValueError, KeyError, TypeError) as exc:
            logger.warning("Ignoring invalid gateway upgrade result: {}", exc)
            self.clear_upgrade_result()
            return None

    def clear_upgrade_result(self) -> None:
        try:
            self.result_path.unlink(missing_ok=True)
        except OSError as exc:
            logger.warning("Failed to remove delivered upgrade result: {}", exc)

    def _write_json_atomic(self, path: Path, payload: dict[str, Any]) -> None:
        self.control_dir.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
        try:
            temporary.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
