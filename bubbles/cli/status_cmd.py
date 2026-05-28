"""`bubbles status` — at-a-glance summary of the local install.

Design (see SPEC §5.4): the global section describes the install-wide baseline
(default model, providers, channels). Per-session state is only shown when it
*differs* from that baseline — a session that uses defaults stays silent. This
keeps `status` compact for the common case while surfacing customized sessions.
"""

from __future__ import annotations

from datetime import datetime, timedelta

from bubbles import __logo__
from bubbles.cli.commands import app, console


# Cap on Session-overrides rows. status is supposed to be at-a-glance, not a
# session audit log — overflow becomes a "(N more)" footer instead of scrolling.
SESSION_OVERRIDES_LIMIT = 10


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

    # Sessions: path + total + active-in-24h + last activity
    session_stats = _scan_sessions(sessions_dir)
    if session_stats["count"] == 0:
        console.print(f"Sessions:  {sessions_dir}  [dim](none)[/dim]")
    else:
        parts = [f"{session_stats['count']} session(s)"]
        if session_stats["active_24h"] > 0:
            parts.append(f"{session_stats['active_24h']} active in last 24h")
        if session_stats["last_activity"]:
            parts.append(f"last activity {_humanize_ago(session_stats['last_activity'])}")
        console.print(f"Sessions:  {sessions_dir}  [dim]({', '.join(parts)})[/dim]")

    # Agent: model + configured providers
    console.print(f"\nModel:     {config.agents.defaults.model}")
    configured, unconfigured = _split_providers(config)
    if configured:
        console.print(f"Providers: {', '.join(configured)}")
        if unconfigured:
            console.print(f"           [dim]({unconfigured} others not configured)[/dim]")
    else:
        console.print("Providers: [yellow]none configured[/yellow]")

    # Runtime: channels
    enabled_channels = _enabled_channels(config)
    if enabled_channels:
        console.print(f"\nChannels:  {', '.join(enabled_channels)}")
    else:
        console.print("\nChannels:  [dim]none enabled[/dim]")

    # Session overrides: only sessions whose model / cron / heartbeat differs.
    default_model = config.agents.defaults.model
    overrides, unassigned_cron = _collect_session_overrides(sessions_dir, get_data_path(), default_model)
    _render_overrides_section(overrides, total_sessions=session_stats["count"])
    if unassigned_cron > 0:
        console.print(
            f"\nUnassigned cron: {unassigned_cron} job(s)  "
            f"[dim][legacy: created via CLI without session binding][/dim]"
        )


# ============ Section: sessions ============

def _scan_sessions(sessions_dir) -> dict:
    """Return {count, active_24h, last_activity}.

    A session is "active in last 24h" if its session.jsonl mtime is within 24h.
    """
    out = {"count": 0, "active_24h": 0, "last_activity": None}
    if not sessions_dir.is_dir():
        return out
    now = datetime.now()
    cutoff = now - timedelta(hours=24)
    latest: float | None = None
    for child in sessions_dir.iterdir():
        if not child.is_dir():
            continue
        jsonl = child / "session.jsonl"
        if not jsonl.exists():
            continue
        out["count"] += 1
        try:
            mtime_ts = jsonl.stat().st_mtime
        except OSError:
            continue
        mtime = datetime.fromtimestamp(mtime_ts)
        if mtime >= cutoff:
            out["active_24h"] += 1
        if latest is None or mtime_ts > latest:
            latest = mtime_ts
    if latest is not None:
        out["last_activity"] = datetime.fromtimestamp(latest)
    return out


def _humanize_ago(when: "datetime") -> str:
    """Rough human time (e.g. '2h ago', '3d ago')."""
    delta = datetime.now() - when
    s = int(delta.total_seconds())
    if s < 60:
        return f"{s}s ago"
    if s < 3600:
        return f"{s // 60}m ago"
    if s < 86400:
        return f"{s // 3600}h ago"
    return f"{s // 86400}d ago"


# ============ Section: providers ============

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


# ============ Section: session overrides ============

def _collect_session_overrides(
    sessions_dir,
    data_path,
    default_model: str,
) -> tuple[list[dict], int]:
    """Build the per-session overrides list and count unassigned cron jobs.

    Returns (overrides, unassigned_cron) where each override is:
        {
            "session": str,                # session key
            "model": str | None,           # set if model differs from default
            "cron": int,                   # non-heartbeat job count for this session
            "heartbeat_ms": int | None,    # set if heartbeat enabled
            "last_activity": datetime | None,
        }

    A session appears in the list iff at least one of (model, cron, heartbeat)
    is a true override.
    """
    # Pass 1: read each session's config.json + jsonl mtime
    sessions: dict[str, dict] = {}
    if sessions_dir.is_dir():
        for child in sessions_dir.iterdir():
            if not child.is_dir():
                continue
            jsonl = child / "session.jsonl"
            if not jsonl.exists():
                continue
            entry = {
                "session": child.name,
                "model": None,
                "cron": 0,
                "heartbeat_ms": None,
                "last_activity": None,
            }
            try:
                entry["last_activity"] = datetime.fromtimestamp(jsonl.stat().st_mtime)
            except OSError:
                pass
            cfg = _load_session_config(child / "config.json")
            cfg_model = cfg.get("model")
            if cfg_model and cfg_model != default_model:
                entry["model"] = cfg_model
            sessions[child.name] = entry

    # Pass 2: walk cron jobs.json once and attribute to sessions
    unassigned = 0
    for j in _iter_cron_jobs(data_path):
        name = j.get("name", "")
        session_key = j.get("payload", {}).get("sessionKey")
        if name.startswith("heartbeat:"):
            if not j.get("enabled", True):
                continue
            hb_key = name[len("heartbeat:"):]
            entry = sessions.get(hb_key)
            if entry is not None:
                entry["heartbeat_ms"] = j.get("schedule", {}).get("everyMs")
            # If the heartbeat names a session that no longer exists on disk,
            # we silently drop it — status is a snapshot, not an integrity tool.
        else:
            if session_key and session_key in sessions:
                sessions[session_key]["cron"] += 1
            else:
                unassigned += 1

    overrides = [
        e for e in sessions.values()
        if e["model"] is not None or e["cron"] > 0 or e["heartbeat_ms"] is not None
    ]
    # Sort: most-customized first (number of override dimensions), then most-recent.
    def _rank(e: dict) -> tuple[int, float]:
        dims = (1 if e["model"] else 0) + (1 if e["cron"] else 0) + (1 if e["heartbeat_ms"] else 0)
        ts = e["last_activity"].timestamp() if e["last_activity"] else 0.0
        return (-dims, -ts)
    overrides.sort(key=_rank)
    return overrides, unassigned


def _load_session_config(path) -> dict:
    """Load a session's config.json, returning {} on any failure."""
    import json as _json
    if not path.exists():
        return {}
    try:
        with open(path, encoding="utf-8") as f:
            return _json.load(f) or {}
    except Exception:
        return {}


def _iter_cron_jobs(data_path):
    """Yield raw job dicts from jobs.json (empty on any failure)."""
    import json as _json
    path = data_path / "cron" / "jobs.json"
    if not path.exists():
        return
    try:
        with open(path, encoding="utf-8") as f:
            data = _json.load(f)
    except Exception:
        return
    for j in data.get("jobs", []):
        yield j


def _render_overrides_section(overrides: list[dict], total_sessions: int) -> None:
    """Render the 'Session overrides' block, or nothing if there's nothing to say."""
    if not overrides:
        return

    n = len(overrides)
    console.print(f"\nSession overrides ({n} of {total_sessions} differ from defaults):")

    name_width = max(len(e["session"]) for e in overrides[:SESSION_OVERRIDES_LIMIT])
    for entry in overrides[:SESSION_OVERRIDES_LIMIT]:
        parts: list[str] = []
        if entry["model"]:
            parts.append(f"model: {entry['model']}")
        if entry["cron"]:
            parts.append(f"cron: {entry['cron']}")
        if entry["heartbeat_ms"]:
            parts.append(f"heartbeat: {_format_interval(entry['heartbeat_ms'])}")
        console.print(f"  {entry['session'].ljust(name_width)}  {'  ·  '.join(parts)}")

    if n > SESSION_OVERRIDES_LIMIT:
        console.print(f"  [dim]({n - SESSION_OVERRIDES_LIMIT} more session(s) with overrides)[/dim]")


def _format_interval(every_ms: int | None) -> str:
    """Compact interval format matching the /heartbeat input syntax (30m / 1h)."""
    if not every_ms:
        return "?"
    minutes = every_ms // 60_000
    if minutes >= 60 and minutes % 60 == 0:
        return f"{minutes // 60}h"
    return f"{minutes}m"
