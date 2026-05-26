"""Shell execution tool."""

import asyncio
import os
import re
from pathlib import Path
from typing import Any

from bubbles.agent.tools.base import Tool


class ExecTool(Tool):
    """Tool to execute shell commands."""

    def __init__(
        self,
        timeout: int = 60,
        deny_patterns: list[str] | None = None,
        allow_patterns: list[str] | None = None,
        path_append: str = "",
    ):
        self.timeout = timeout
        self._session_dir: Path | None = None
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
        self.path_append = path_append

    def set_session_dir(self, session_dir: Path | None) -> None:
        """Set session directory for command execution."""
        self._session_dir = session_dir
    
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
        # Resolve working directory relative to session_dir
        if working_dir and self._session_dir:
            # Handle ~ as session_dir
            if working_dir.startswith("~"):
                working_dir = str(self._session_dir / working_dir[1:].lstrip("/\\"))
            elif not Path(working_dir).is_absolute():
                working_dir = str(self._session_dir / working_dir)
        cwd = working_dir or (str(self._session_dir) if self._session_dir else os.getcwd())

        # Hard sandbox: cwd MUST resolve inside session_dir. This catches the
        # working_dir-with-..  escape (the alternative would be the agent jumping
        # to another session's working tree). Symlinks are resolved so a planted
        # symlink can't break out either.
        if self._session_dir:
            try:
                resolved_cwd = Path(cwd).resolve()
                session_resolved = self._session_dir.resolve()
                resolved_cwd.relative_to(session_resolved)
            except (ValueError, OSError):
                return (
                    f"Error: working_dir resolves outside session directory "
                    f"({cwd!r} ↛ {self._session_dir})"
                )
            cwd = str(resolved_cwd)

        guard_error = self._guard_command(command, cwd)
        if guard_error:
            return guard_error
        
        env = os.environ.copy()
        if self.path_append:
            env["PATH"] = env.get("PATH", "") + os.pathsep + self.path_append

        try:
            process = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=cwd,
                env=env,
            )
            
            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=self.timeout
                )
            except asyncio.TimeoutError:
                process.kill()
                # Wait for the process to fully terminate so pipes are
                # drained and file descriptors are released.
                try:
                    await asyncio.wait_for(process.wait(), timeout=5.0)
                except asyncio.TimeoutError:
                    pass
                return f"Error: Command timed out after {self.timeout} seconds"
            
            output_parts = []
            
            if stdout:
                output_parts.append(stdout.decode("utf-8", errors="replace"))
            
            if stderr:
                stderr_text = stderr.decode("utf-8", errors="replace")
                if stderr_text.strip():
                    output_parts.append(f"STDERR:\n{stderr_text}")
            
            if process.returncode != 0:
                output_parts.append(f"\nExit code: {process.returncode}")
            
            result = "\n".join(output_parts) if output_parts else "(no output)"
            
            # Truncate very long output
            max_len = 10000
            if len(result) > max_len:
                result = result[:max_len] + f"\n... (truncated, {len(result) - max_len} more chars)"
            
            return result
            
        except Exception as e:
            return f"Error executing command: {str(e)}"

    def _guard_command(self, command: str, cwd: str) -> str | None:
        """Best-effort safety guard for potentially destructive commands.

        Note: This is **application-layer best-effort**. Shell is Turing-complete
        — variables, command substitution, here-docs, and indirect file access
        all bypass static checks. For true session isolation, run each session
        in its own container (see DISTRIBUTED.md).
        """
        cmd = command.strip()
        lower = cmd.lower()

        for pattern in self.deny_patterns:
            if re.search(pattern, lower):
                return "Error: Command blocked by safety guard (dangerous pattern detected)"

        if self.allow_patterns:
            if not any(re.search(p, lower) for p in self.allow_patterns):
                return "Error: Command blocked by safety guard (not in allowlist)"

        # Restrict to session directory if set
        if self._session_dir:
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
                        resolved = Path(target.replace("~", str(self._session_dir))).resolve()
                        resolved.relative_to(self._session_dir.resolve())
                    except (ValueError, OSError):
                        return f"Error: Command blocked by safety guard (cd target outside session directory: {target})"

            session_path = self._session_dir.resolve()

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
