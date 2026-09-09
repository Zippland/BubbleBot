"""Session /config command handling; independent from cron scheduling."""

from __future__ import annotations

from typing import TYPE_CHECKING

from bubbles.bus.events import InboundMessage, OutboundMessage
from bubbles.session.manager import Session, SessionConfig

if TYPE_CHECKING:
    from bubbles.agent.loop import AgentLoop


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
        sandbox_val = cfg.sandbox or f"{loop.sandbox_config.default} (default)"
        config_text = f"""session: {session.key}
model: {model_val}
system_prompt: {prompt_val}
sandbox: {sandbox_val}"""
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

    if key == "sandbox":
        valid = ("local", "local_isolated")
        if value and value.lower() not in valid:
            return OutboundMessage(
                channel=msg.channel, chat_id=msg.chat_id,
                content=f"Unknown sandbox backend `{value}`. Valid: {', '.join(valid)}",
            )
        cfg.sandbox = value.lower() if value else None
        loop.sessions.save(session)
        # Drop the cached sandbox so the new backend is built on the next turn.
        await loop._sandboxes.close(session.key)
        return OutboundMessage(
            channel=msg.channel, chat_id=msg.chat_id,
            content=f"sandbox = `{value.lower()}`" if value else "sandbox reset to default",
        )

    return OutboundMessage(
        channel=msg.channel, chat_id=msg.chat_id,
        content=f"Unknown config key: `{key}`\n\nValid keys: model, system_prompt, sandbox",
    )
