"""Shell execution tool.

Delegates the actual subprocess to the session's ``Sandbox`` (see
``bubbles/sandbox/``). This tool keeps the best-effort safety guard
(dangerous-command patterns + path-containment for non-isolated backends) and
the output formatting; *where and how* the command runs is the sandbox's job.
The historical ``$HOME`` handling now lives in the sandbox's env construction,
so ``local_isolated`` gives each session its own credential home.
"""

import re
from pathlib import Path
from typing import Any

from bubbles.agent.tools.base import Tool
from bubbles.sandbox.base import Sandbox


class ExecTool(Tool):
    """Tool to execute shell commands."""

    def __init__(
        self,
        timeout: int = 60,
        deny_patterns: list[str] | None = None,
        allow_patterns: list[str] | None = None,
    ):
        self.timeout = timeout
        self._sandbox: Sandbox | None = None
        self.deny_patterns = deny_patterns or [
            r"\brm\s+-[rf]{1,2}\b",          # rm -r, rm -rf, rm -fr
            r"\bdel\s+/[fq]\b",              # del /f, del /q
            r"\brmdir\s+/s\b",               # rmdir /s
            r"(?:^|[;&|]\s*)format\b",       # format (as standalone command only)
            r"\b(mkfs|diskpart)\b",          # disk operations
            r"\bdd\s+if=",                   # dd
            r">\s*/dev/sd",                  # write to disk
            r"\b(shutdown|reboot|poweroff)\b",  # system power
            r":\(\)\s*\{.*\};\s*:",          # fork bomb
        ]
        self.allow_patterns = allow_patterns or []

    def set_sandbox(self, sandbox: Sandbox | None) -> None:
        """Set the sandbox for command execution (called per turn)."""
        self._sandbox = sandbox

    @property
    def name(self) -> str:
        return "exec"

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The shell command to execute"
                },
                "working_dir": {
                    "type": "string",
                    "description": "Optional working directory for the command"
                }
            },
            "required": ["command"]
        }

    async def execute(self, command: str, working_dir: str | None = None, **kwargs: Any) -> str:
        if self._sandbox is None:
            return "Error: no sandbox bound"

        # Resolve the working directory to a sandbox-internal path, enforcing
        # containment within the sandbox root (catches working_dir-with-..
        # escapes and planted symlinks — backend's half of the guarantee).
        try:
            if working_dir:
                cwd = await self._sandbox.resolve_within_root(working_dir)
            else:
                cwd = self._sandbox.root
        except PermissionError:
            return (
                f"Error: working_dir resolves outside session directory "
                f"({working_dir!r})"
            )
        except (ValueError, OSError) as e:
            return f"Error resolving working_dir: {e}"

        guard_error = self._guard_command(command)
        if guard_error:
            return guard_error

        try:
            result = await self._sandbox.exec(command, cwd=cwd, timeout=self.timeout)
        except Exception as e:
            return f"Error executing command: {str(e)}"

        if result.timed_out:
            return f"Error: Command timed out after {self.timeout} seconds"

        output_parts = []
        if result.stdout:
            output_parts.append(result.stdout)
        if result.stderr and result.stderr.strip():
            output_parts.append(f"STDERR:\n{result.stderr}")
        if result.returncode != 0:
            output_parts.append(f"\nExit code: {result.returncode}")

        output = "\n".join(output_parts) if output_parts else "(no output)"

        # Truncate very long output
        max_len = 10000
        if len(output) > max_len:
            output = output[:max_len] + f"\n... (truncated, {len(output) - max_len} more chars)"

        return output

    def _guard_command(self, command: str) -> str | None:
        """Best-effort safety guard for potentially destructive commands.

        Two layers: dangerous-command patterns (always applied), and
        path-containment on the command string (only for backends that are not
        structurally isolated — a container/VM contains escapes at the kernel).

        This is **application-layer best-effort**. Shell is Turing-complete —
        variables, command substitution, here-docs, and indirect file access
        all bypass static checks. For a hard guarantee, use a sandbox backend
        with ``provides_hard_isolation`` (see SECURITY.md §4.1).
        """
        cmd = command.strip()
        lower = cmd.lower()

        for pattern in self.deny_patterns:
            if re.search(pattern, lower):
                return "Error: Command blocked by safety guard (dangerous pattern detected)"

        if self.allow_patterns:
            if not any(re.search(p, lower) for p in self.allow_patterns):
                return "Error: Command blocked by safety guard (not in allowlist)"

        # Structurally-isolated backends contain path escapes at the kernel —
        # the static path checks below would false-positive on legit absolute
        # paths inside the container, so skip them there.
        if self._sandbox is None or self._sandbox.provides_hard_isolation:
            return None

        session_dir = Path(self._sandbox.root)

        if "..\\" in cmd or "../" in cmd:
            return "Error: Command blocked by safety guard (path traversal detected)"

        # `cd ..` (bare, no trailing slash) is a traversal too — the
        # `../` check above misses it because the regex needs a slash.
        if re.search(r"\b(?:cd|pushd|chdir)\s+\.\.(?=\s|;|&|\||$)", cmd, re.IGNORECASE):
            return "Error: Command blocked by safety guard (cd .. detected)"

        # `cd /...` / `cd ~ ...` / `pushd /...` to absolute paths outside session.
        # Doesn't catch every escape (env-var indirection, command substitution),
        # but blocks the obvious cases without false-positives on `cd subdir`.
        cd_match = re.search(
            r"\b(?:cd|pushd|chdir)\s+([^;&|`$]+)", cmd, re.IGNORECASE,
        )
        if cd_match:
            target = cd_match.group(1).strip().strip("'\"")
            if target.startswith(("/", "~")) or re.match(r"^[A-Za-z]:[\\/]", target):
                # absolute path — must resolve inside session_dir
                try:
                    resolved = Path(target.replace("~", str(session_dir))).resolve()
                    resolved.relative_to(session_dir.resolve())
                except (ValueError, OSError):
                    return f"Error: Command blocked by safety guard (cd target outside session directory: {target})"

        session_path = session_dir.resolve()

        win_paths = re.findall(r"[A-Za-z]:\\[^\\\"']+", cmd)
        # POSIX absolute paths. Accept many leading separators so we catch
        # quoted strings, var-assignments (`P='/path'`), and substituted
        # forms (`"$X"`). Trailing terminators: whitespace, quotes, redirects.
        posix_paths = re.findall(r"(?:^|[\s|>='\"`(),])(/[^\s\"'>;]+)", cmd)

        for raw in win_paths + posix_paths:
            try:
                p = Path(raw.strip()).resolve()
            except Exception:
                continue
            if p.is_absolute() and session_path not in p.parents and p != session_path:
                return "Error: Command blocked by safety guard (path outside session directory)"

        return None
