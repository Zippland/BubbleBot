"""Slash command handlers (/config, /heartbeat) and heartbeat helpers.

Pulled out of `loop.py`. Handlers take the AgentLoop instance explicitly so
they can mutate session state / cron service / providers, but the module
itself is import-light to avoid a circular dependency with loop.py.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from loguru import logger

from bubbles.bus.events import InboundMessage, OutboundMessage
from bubbles.session.manager import Session, SessionConfig

if TYPE_CHECKING:
    from bubbles.agent.loop import AgentLoop
    from bubbles.cron.service import CronService


HEARTBEAT_MIN_MINUTES = 1
HEARTBEAT_MAX_MINUTES = 24 * 60

HEARTBEATS_TEMPLATE = """# Heartbeats

Your periodic check-in list. The bot reads this file on every heartbeat tick.
Add items, remove them, rewrite framing freely.

## Items

- (no items yet — examples:)
  - Scan MEMORY.md for stale facts.
  - Light check-in if quiet for 8+ hours during daytime.
"""

HEARTBEAT_TICK_MESSAGE = (
    "[心跳触发] 严格按照 <work_dir>/HEARTBEATS.md（上方已加载）里的清单执行。"
    "不要从历史对话里翻补未完成的旧任务。\n\n"
    "默认沉默。清单里没有当下应处理的事项就调 stay_silent 结束。"
)


def heartbeat_job_name(session_key: str) -> str:
    return f"heartbeat:{session_key}"


def parse_heartbeat_interval(s: str) -> int | None:
    """Parse '30m', '2h', '90s', or bare number (= minutes). Return minutes, or None."""
    m = re.match(r"^(\d+)\s*([smh]?)$", s.strip().lower())
    if not m:
        return None
    n = int(m.group(1))
    unit = m.group(2) or "m"
    if unit == "s":
        return max(1, (n + 30) // 60)
    if unit == "h":
        return n * 60
    return n


def humanize_minutes_cn(minutes: int) -> str:
    if minutes >= 60 and minutes % 60 == 0:
        return f"{minutes // 60} 小时"
    return f"{minutes} 分钟"


def humanize_minutes_en(minutes: int) -> str:
    if minutes >= 60 and minutes % 60 == 0:
        h = minutes // 60
        return f"{h} hour{'s' if h > 1 else ''}"
    return f"{minutes} minute{'s' if minutes > 1 else ''}"


def get_heartbeat_job(cron_service: "CronService | None", session_key: str):
    """Return the heartbeat cron job for this session, or None."""
    if not cron_service:
        return None
    name = heartbeat_job_name(session_key)
    for j in cron_service.list_jobs(include_disabled=True):
        if j.name == name:
            return j
    return None


def build_heartbeat_info(
    cron_service: "CronService | None", session_key: str
) -> str | None:
    """A system-prompt block describing the active heartbeat, or None if off."""
    job = get_heartbeat_job(cron_service, session_key)
    if not job or not job.schedule.every_ms:
        return None
    minutes = job.schedule.every_ms // 60_000
    return (
        f"## Heartbeat: ON\n\n"
        f"Firing every {humanize_minutes_en(minutes)}. HEARTBEATS.md is loaded above. "
        f"On each tick, scan the checklist and act only when something genuinely warrants "
        f"attention. Call `stay_silent` to end the turn quietly when nothing applies."
    )


async def handle_config_command(
    loop: "AgentLoop",
    msg: InboundMessage,
    session: Session,
    cmd_arg: str,
) -> OutboundMessage:
    """Handle /config command for session-specific configuration."""
    cfg = session.config

    if not cmd_arg:
        model_val = cfg.model or loop.model
        prompt_val = (cfg.system_prompt[:30] + "...") if cfg.system_prompt else "-"
        config_text = f"""session: {session.key}
model: {model_val}
system_prompt: {prompt_val}"""
        return OutboundMessage(channel=msg.channel, chat_id=msg.chat_id, content=config_text)

    parts = cmd_arg.split(maxsplit=1)
    key = parts[0].lower()
    value = parts[1] if len(parts) > 1 else ""

    if key == "reset":
        session.config = SessionConfig()
        loop.sessions.save(session)
        return OutboundMessage(
            channel=msg.channel, chat_id=msg.chat_id,
            content="Config reset to defaults.",
        )

    if key == "model":
        if value and loop.provider_factory is not None:
            try:
                loop.provider_factory(value)
            except Exception as e:
                return OutboundMessage(
                    channel=msg.channel, chat_id=msg.chat_id,
                    content=f"无法切换到 `{value}`：{e}",
                )
        cfg.model = value if value else None
        loop.sessions.save(session)
        return OutboundMessage(
            channel=msg.channel, chat_id=msg.chat_id,
            content=f"model = `{value}`" if value else "model reset to default",
        )

    if key == "system_prompt":
        cfg.system_prompt = value if value else None
        loop.sessions.save(session)
        return OutboundMessage(
            channel=msg.channel, chat_id=msg.chat_id,
            content="system_prompt updated" if value else "system_prompt reset to default",
        )

    return OutboundMessage(
        channel=msg.channel, chat_id=msg.chat_id,
        content=f"Unknown config key: `{key}`\n\nValid keys: model, system_prompt",
    )


def handle_heartbeat_command(
    loop: "AgentLoop",
    msg: InboundMessage,
    session: Session,
    session_key: str,
    arg: str,
) -> OutboundMessage:
    """Handle /heartbeat command (status / set interval / off)."""
    if not loop.cron_service:
        return OutboundMessage(
            channel=msg.channel, chat_id=msg.chat_id,
            content="心跳功能不可用：cron 服务未配置。",
        )

    arg = (arg or "").strip().lower()

    # No-arg → status
    if not arg:
        job = get_heartbeat_job(loop.cron_service, session_key)
        if not job or not job.schedule.every_ms:
            return OutboundMessage(
                channel=msg.channel, chat_id=msg.chat_id, content="心跳未开启。",
            )
        minutes = job.schedule.every_ms // 60_000
        line = f"心跳每 {humanize_minutes_cn(minutes)} 触发"
        if job.state.next_run_at_ms:
            from datetime import datetime as _dt
            nxt = _dt.fromtimestamp(job.state.next_run_at_ms / 1000)
            line += f"，下次 {nxt.strftime('%H:%M:%S')}"
        return OutboundMessage(channel=msg.channel, chat_id=msg.chat_id, content=line + "。")

    # Off
    if arg == "off":
        job = get_heartbeat_job(loop.cron_service, session_key)
        if not job:
            return OutboundMessage(
                channel=msg.channel, chat_id=msg.chat_id, content="心跳本来就没开。",
            )
        loop.cron_service.remove_job(job.id)
        return OutboundMessage(
            channel=msg.channel, chat_id=msg.chat_id, content="心跳已关闭。",
        )

    # Else: try to parse as interval
    minutes = parse_heartbeat_interval(arg)
    if minutes is None:
        return OutboundMessage(
            channel=msg.channel, chat_id=msg.chat_id,
            content=(
                "格式不对。\n\n"
                "示例：`/heartbeat 30m`、`/heartbeat 2h`、`/heartbeat 30`（默认分钟）、`/heartbeat off`"
            ),
        )
    if minutes < HEARTBEAT_MIN_MINUTES or minutes > HEARTBEAT_MAX_MINUTES:
        return OutboundMessage(
            channel=msg.channel, chat_id=msg.chat_id,
            content=f"间隔需在 {HEARTBEAT_MIN_MINUTES} 分钟到 {HEARTBEAT_MAX_MINUTES // 60} 小时之间。",
        )

    # Drop any existing heartbeat job for this session, then register fresh
    existing = get_heartbeat_job(loop.cron_service, session_key)
    if existing:
        loop.cron_service.remove_job(existing.id)

    # Write template if HEARTBEATS.md missing
    created_template = False
    if session.directory:
        hb_path = session.directory / "HEARTBEATS.md"
        if not hb_path.exists():
            hb_path.write_text(HEARTBEATS_TEMPLATE, encoding="utf-8")
            created_template = True
            logger.info("Created HEARTBEATS.md at {}", hb_path)

    from bubbles.cron.types import CronSchedule
    loop.cron_service.add_job(
        name=heartbeat_job_name(session_key),
        schedule=CronSchedule(kind="every", every_ms=minutes * 60_000),
        message=HEARTBEAT_TICK_MESSAGE,
        deliver=True,
        channel=msg.channel,
        to=msg.chat_id,
        session_key=session_key,
    )

    body = f"心跳已开启，每 {humanize_minutes_cn(minutes)} 触发。"
    if created_template:
        body += " 已写入 HEARTBEATS.md 模板。"
    return OutboundMessage(channel=msg.channel, chat_id=msg.chat_id, content=body)
