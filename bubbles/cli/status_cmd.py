"""`bubbles status` — at-a-glance summary of the local install."""

from __future__ import annotations

from bubbles import __logo__
from bubbles.cli.commands import app, console


@app.command()
def status():
    """Show bubbles status."""
    from bubbles.config.loader import load_config, get_config_path
    from bubbles.utils.helpers import get_sessions_path, get_data_path

    config_path = get_config_path()
    sessions_dir = get_sessions_path()

    console.print(f"{__logo__} bubbles Status\n")

    if not config_path.exists():
        console.print(f"Config:    [red]missing[/red] (expected at {config_path})")
        console.print("\nRun [cyan]bubbles onboard[/cyan] to initialize.")
        return

    config = load_config()
    console.print(f"Config:    {config_path}")

    # Sessions: path + count + last activity
    session_count, last_activity = _scan_sessions(sessions_dir)
    if session_count == 0:
        console.print(f"Sessions:  {sessions_dir}  [dim](none)[/dim]")
    else:
        activity = f"last activity {_humanize_ago(last_activity)}" if last_activity else "never used"
        console.print(f"Sessions:  {sessions_dir}  [dim]({session_count} session(s), {activity})[/dim]")

    # Agent: model + configured providers (folded summary)
    console.print(f"\nModel:     {config.agents.defaults.model}")
    configured, unconfigured = _split_providers(config)
    if configured:
        console.print(f"Providers: {', '.join(configured)}")
        if unconfigured:
            console.print(f"           [dim]({unconfigured} others not configured)[/dim]")
    else:
        console.print("Providers: [yellow]none configured[/yellow]")

    # Runtime capabilities
    enabled_channels = _enabled_channels(config)
    if enabled_channels:
        console.print(f"\nChannels:  {', '.join(enabled_channels)}")
    else:
        console.print("\nChannels:  [dim]none enabled[/dim]")

    # Cron count + heartbeat sessions (read store without starting the service)
    cron_count, heartbeats = _scan_cron_jobs(get_data_path())
    console.print(f"Cron:      {cron_count} scheduled job(s)")
    if not heartbeats:
        console.print("Heartbeat: [dim]none enabled[/dim]")
    elif len(heartbeats) <= 3:
        details = ", ".join(
            f"{h['session']} ({_format_interval(h['every_ms'])})" for h in heartbeats
        )
        console.print(f"Heartbeat: {details}")
    else:
        console.print(f"Heartbeat: {len(heartbeats)} session(s) enabled")


def _scan_sessions(sessions_dir) -> tuple[int, "datetime | None"]:
    """Return (session_count, most_recent_session_jsonl_mtime) by scanning sessions_dir."""
    from datetime import datetime
    if not sessions_dir.is_dir():
        return 0, None
    count = 0
    latest: float | None = None
    for child in sessions_dir.iterdir():
        if not child.is_dir():
            continue
        jsonl = child / "session.jsonl"
        if not jsonl.exists():
            continue
        count += 1
        try:
            mtime = jsonl.stat().st_mtime
            if latest is None or mtime > latest:
                latest = mtime
        except OSError:
            pass
    return count, datetime.fromtimestamp(latest) if latest else None


def _humanize_ago(when: "datetime") -> str:
    """Rough human time (e.g. '2h ago', '3d ago')."""
    from datetime import datetime
    delta = datetime.now() - when
    s = int(delta.total_seconds())
    if s < 60:
        return f"{s}s ago"
    if s < 3600:
        return f"{s // 60}m ago"
    if s < 86400:
        return f"{s // 3600}h ago"
    return f"{s // 86400}d ago"


def _split_providers(config) -> tuple[list[str], int]:
    """Return (configured_labels, count_unconfigured)."""
    from bubbles.providers.registry import PROVIDERS
    configured: list[str] = []
    unconfigured = 0
    for spec in PROVIDERS:
        p = getattr(config.providers, spec.name, None)
        if p is None:
            continue
        is_set = False
        suffix = ""
        if spec.is_oauth:
            # Actually check whether a token has been cached — don't just trust
            # the OAuth-type flag (otherwise users see (OAuth) even before login).
            if _is_oauth_authenticated(spec):
                is_set = True
                suffix = " (OAuth)"
        elif spec.is_local:
            if p.api_base:
                is_set = True
        else:
            is_set = bool(p.api_key)
        if is_set:
            configured.append(f"{spec.label}{suffix}")
        else:
            unconfigured += 1
    return configured, unconfigured


def _is_oauth_authenticated(spec) -> bool:
    """Best-effort check that the user has completed OAuth login for this provider.

    Reads the on-disk token cache without triggering any network refresh, so it's
    safe to call from a static status command. Any error → treat as not authenticated.
    """
    try:
        if spec.name == "openai_codex":
            from oauth_cli_kit import OPENAI_CODEX_PROVIDER
            from oauth_cli_kit.storage import FileTokenStorage
            tok = FileTokenStorage(
                token_filename=OPENAI_CODEX_PROVIDER.token_filename,
            ).load()
            return bool(tok and tok.access)
        if spec.name == "github_copilot":
            import os
            token_dir = os.getenv(
                "GITHUB_COPILOT_TOKEN_DIR",
                os.path.expanduser("~/.config/litellm/github_copilot"),
            )
            token_file = os.path.join(
                token_dir,
                os.getenv("GITHUB_COPILOT_ACCESS_TOKEN_FILE", "access-token"),
            )
            with open(token_file, encoding="utf-8") as f:
                return bool(f.read().strip())
    except Exception:
        return False
    return False


def _enabled_channels(config) -> list[str]:
    """List channel names whose .enabled is True."""
    channels_cfg = config.channels
    out: list[str] = []
    for name in (
        "telegram", "feishu", "wechat", "discord", "dingtalk",
        "qq", "slack", "whatsapp", "matrix", "mochat", "email",
    ):
        sub = getattr(channels_cfg, name, None)
        if sub is not None and getattr(sub, "enabled", False):
            out.append(name)
    return out


def _scan_cron_jobs(data_path) -> tuple[int, list[dict]]:
    """Read jobs.json without starting CronService.

    Returns (non_heartbeat_cron_count, heartbeats), where heartbeats is a list of
    {"session": session_key, "every_ms": int|None} for jobs named "heartbeat:<key>".
    """
    import json as _json
    path = data_path / "cron" / "jobs.json"
    if not path.exists():
        return 0, []
    try:
        with open(path, encoding="utf-8") as f:
            data = _json.load(f)
    except Exception:
        return 0, []

    cron_count = 0
    heartbeats: list[dict] = []
    for j in data.get("jobs", []):
        name = j.get("name", "")
        if name.startswith("heartbeat:"):
            # Disabled heartbeats: skip entirely (don't fall through to cron count)
            if j.get("enabled", True):
                heartbeats.append({
                    "session": name[len("heartbeat:"):],
                    "every_ms": j.get("schedule", {}).get("everyMs"),
                })
        else:
            cron_count += 1
    return cron_count, heartbeats


def _format_interval(every_ms: int | None) -> str:
    """Compact interval format matching the /heartbeat input syntax (30m / 1h)."""
    if not every_ms:
        return "?"
    minutes = every_ms // 60_000
    if minutes >= 60 and minutes % 60 == 0:
        return f"{minutes // 60}h"
    return f"{minutes}m"
