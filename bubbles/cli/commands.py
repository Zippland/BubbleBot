"""CLI commands for bubbles.

This module defines the root Typer ``app`` plus the simple top-level commands
(``onboard``, ``sync-skills``) and the shared ``console`` / ``EXIT_COMMANDS``
state. The rest of the command surface lives in sibling modules
(``agent_cmd``, ``gateway_cmd``, ``status_cmd``, ``channels_cmd``,
``cron_cmd``, ``provider_cmd``) which register themselves on ``app`` at import
time — they are imported at the bottom of this file purely for that side
effect.
"""

from pathlib import Path
import sys

# Force UTF-8 on stdio before rich touches it — otherwise on Windows zh_CN consoles
# (cp936/GBK) the 🫧 logo in __logo__ crashes the whole CLI via UnicodeEncodeError.
for _stream in (sys.stdout, sys.stderr):
    if _stream is not None and hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, OSError):
            pass

import typer
from rich.console import Console

from bubbles import __version__, __logo__


app = typer.Typer(
    name="bubbles",
    help=f"{__logo__} bubbles - Personal AI Assistant",
    no_args_is_help=True,
)

console = Console()
EXIT_COMMANDS = {"exit", "quit", "/exit", "/quit", ":q"}


def version_callback(value: bool):
    if value:
        console.print(f"{__logo__} bubbles v{__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: bool = typer.Option(
        None, "--version", "-v", callback=version_callback, is_eager=True
    ),
):
    """bubbles - Personal AI Assistant."""
    pass


@app.command()
def onboard():
    """Initialize bubbles configuration."""
    from bubbles.config.loader import get_config_path, load_config, save_config
    from bubbles.config.schema import Config
    from bubbles.utils.helpers import get_sessions_path

    config_path = get_config_path()

    if config_path.exists():
        console.print(f"[yellow]Config already exists at {config_path}[/yellow]")
        console.print("  [bold]y[/bold] = overwrite with defaults (existing values will be lost)")
        console.print("  [bold]N[/bold] = refresh config, keeping existing values and adding new fields")
        if typer.confirm("Overwrite?"):
            config = Config()
            save_config(config)
            console.print(f"[green]✓[/green] Config reset to defaults at {config_path}")
        else:
            config = load_config()
            save_config(config)
            console.print(f"[green]✓[/green] Config refreshed at {config_path} (existing values preserved)")
    else:
        save_config(Config())
        console.print(f"[green]✓[/green] Created config at {config_path}")

    sessions_dir = get_sessions_path()
    console.print(f"[green]✓[/green] Sessions directory: {sessions_dir}")

    console.print(f"\n{__logo__} bubbles is ready!")
    console.print("\nNext steps:")
    console.print("  1. Add your API key to [cyan]~/.bubbles/config.json[/cyan]")
    console.print("  2. Start chatting: [cyan]bubbles agent[/cyan]")
    console.print("  3. Create a session: [cyan]/session <name>[/cyan]")
    console.print("\n[dim]Each session is an isolated workspace with its own files[/dim]")


@app.command("sync-skills")
def sync_skills(
    session: str = typer.Option(None, "--session", "-s", help="Only sync to a specific session"),
    dry_run: bool = typer.Option(False, "--dry-run", "-n", help="Show what would be done"),
):
    """Sync skills from template to all session workspaces."""
    import shutil
    from bubbles.utils.helpers import get_sessions_path

    # Find template skills directory
    template_dir = Path(__file__).parent.parent / "templates" / "session" / "skills"
    sessions_dir = get_sessions_path()

    if not template_dir.exists():
        console.print(f"[red]Template not found: {template_dir}[/red]")
        raise typer.Exit(1)

    if not sessions_dir.exists():
        console.print(f"[red]Sessions directory not found: {sessions_dir}[/red]")
        raise typer.Exit(1)

    if session:
        session_dirs = [sessions_dir / session]
        if not session_dirs[0].exists():
            console.print(f"[red]Session not found: {session}[/red]")
            raise typer.Exit(1)
    else:
        session_dirs = [d for d in sessions_dir.iterdir() if d.is_dir()]

    if not session_dirs:
        console.print("No sessions found")
        return

    template_skills = [d for d in template_dir.iterdir() if d.is_dir() and (d / "SKILL.md").exists()]
    if not template_skills:
        console.print("No skills in template")
        return

    console.print(f"Skills: {', '.join(s.name for s in template_skills)}")
    console.print(f"Sessions: {', '.join(d.name for d in session_dirs)}\n")

    for session_dir in session_dirs:
        skills_dir = session_dir / "skills"

        if not skills_dir.exists():
            if dry_run:
                console.print(f"  [dim]Would create {skills_dir}[/dim]")
            else:
                skills_dir.mkdir(parents=True)

        for skill in template_skills:
            dest = skills_dir / skill.name
            action = "update" if dest.exists() else "add"

            if dry_run:
                console.print(f"  [dim]Would {action}: {session_dir.name}/skills/{skill.name}[/dim]")
            else:
                if dest.exists():
                    shutil.rmtree(dest)
                shutil.copytree(skill, dest)
                console.print(f"  [green]✓[/green] {action}: {session_dir.name}/skills/{skill.name}")

    console.print("\n[green]Done![/green]")


# ===== Trigger sibling sub-modules to register their commands on `app` =====
# These imports MUST live at the bottom: each sub-module imports `app` /
# `console` from this file, which only become defined above. Side-effect imports.
from bubbles.cli import agent_cmd  # noqa: E402, F401
from bubbles.cli import channels_cmd  # noqa: E402, F401
from bubbles.cli import cron_cmd  # noqa: E402, F401
from bubbles.cli import gateway_cmd  # noqa: E402, F401
from bubbles.cli import provider_cmd  # noqa: E402, F401
from bubbles.cli import status_cmd  # noqa: E402, F401


if __name__ == "__main__":
    app()
