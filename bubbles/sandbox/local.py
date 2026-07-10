"""Local sandbox backends: run on the harness host's own filesystem.

- ``LocalSandbox`` reproduces the pre-sandbox behavior exactly: the exec
  subprocess inherits the host environment (``os.environ.copy()``), and files
  live directly under the session directory. This is the default backend, so
  existing deployments behave byte-for-byte as before.

- ``LocalIsolatedSandbox`` gives each session its own ``$HOME`` (and XDG /
  Windows equivalents) so CLI tools (``gh`` / ``aws`` / ``gcloud`` / ``kubectl``
  / git) store and read credentials per session instead of sharing the host
  user's real home. The environment is built from an **allowlist** (not a
  denylist — a denylist is whack-a-mole), so host credential-pointer vars
  (``GH_CONFIG_DIR``, ``AWS_*``, ``KUBECONFIG``, …) simply never exist in the
  child.

  Honest boundary: this isolates *environment-directed credential lookup only*,
  NOT the whole filesystem. A command that reads an absolute path
  (``cat /home/other/.ssh/id_rsa``) is unaffected here — hard filesystem
  isolation is a container/OS backend (Phase 1). See SECURITY.md §4.1.

  The per-session home lives OUTSIDE the session working directory
  (``~/.bubbles/session_homes/<key>/``), so the model's own file tools cannot
  read the stored credentials — only the exec subprocess can.
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

from bubbles.sandbox.base import DirEntry, ExecResult, Sandbox, StatResult

# Environment variables that must always survive into an isolated shell or the
# interpreter/shell itself breaks. PATH is handled separately (path_append).
_POSIX_ESSENTIAL = ("PATH", "LANG", "LC_ALL", "LC_CTYPE", "TZ", "TERM", "SSL_CERT_FILE", "SSL_CERT_DIR")
# On Windows, cmd.exe / PowerShell won't even launch without these.
_WINDOWS_ESSENTIAL = (
    "PATH", "SystemRoot", "ComSpec", "PATHEXT", "WINDIR",
    "NUMBER_OF_PROCESSORS", "PROCESSOR_ARCHITECTURE", "TEMP", "TMP",
)


class LocalSandbox(Sandbox):
    """Run commands and file I/O on the host, under the session directory."""

    provides_hard_isolation = False

    def __init__(self, session_key: str, root: Path, path_append: str = ""):
        self.session_key = session_key
        self._root = Path(root)
        self.root = str(self._root)
        self.path_append = path_append

    # ---- environment ----

    def _build_env(self) -> dict[str, str]:
        """Full host environment (current behavior), plus optional PATH append."""
        env = os.environ.copy()
        if self.path_append:
            env["PATH"] = env.get("PATH", "") + os.pathsep + self.path_append
        return env

    # ---- exec ----

    async def exec(self, command: str, *, cwd: str, timeout: int) -> ExecResult:
        return await self._run(command, cwd=cwd, timeout=timeout, env=self._build_env())

    async def _run(self, command: str, *, cwd: str, timeout: int, env: dict[str, str]) -> ExecResult:
        process = await asyncio.create_subprocess_shell(
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=cwd,
            env=env,
        )
        try:
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=timeout)
        except asyncio.TimeoutError:
            process.kill()
            # Drain pipes / release fds before returning.
            try:
                await asyncio.wait_for(process.wait(), timeout=5.0)
            except asyncio.TimeoutError:
                pass
            return ExecResult(stdout="", stderr="", returncode=-1, timed_out=True)

        return ExecResult(
            stdout=stdout.decode("utf-8", errors="replace") if stdout else "",
            stderr=stderr.decode("utf-8", errors="replace") if stderr else "",
            returncode=process.returncode if process.returncode is not None else -1,
        )

    # ---- filesystem ----

    def _host_path(self, path: str) -> Path:
        """Resolve a sandbox path to a host path, enforcing containment in root.

        Ports the pre-sandbox ``_resolve_path`` semantics:
        - ``~`` refers to the session root (NOT the system $HOME)
        - relative paths resolve against root
        - symlinks are resolved; anything escaping root raises PermissionError
        """
        raw = (path or "").strip()
        if not raw:
            raise ValueError("empty path")

        if raw.startswith("~"):
            raw = str(self._root / raw[1:].lstrip("/\\"))

        p = Path(raw)
        if not p.is_absolute():
            p = self._root / p

        resolved = p.resolve()
        base = self._root.resolve()
        try:
            resolved.relative_to(base)
        except ValueError:
            raise PermissionError(f"Path '{path}' is outside allowed directory")
        return resolved

    async def resolve_within_root(self, path: str) -> str:
        return str(self._host_path(path))

    async def read_bytes(self, path: str) -> bytes:
        return self._host_path(path).read_bytes()

    async def write_bytes(self, path: str, data: bytes) -> None:
        p = self._host_path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)

    async def list_dir(self, path: str) -> list[DirEntry]:
        p = self._host_path(path)
        return [DirEntry(name=item.name, is_dir=item.is_dir()) for item in sorted(p.iterdir())]

    async def stat(self, path: str) -> StatResult | None:
        p = self._host_path(path)
        if not p.exists():
            return None
        st = p.stat()
        return StatResult(size=st.st_size, mtime=st.st_mtime, is_dir=p.is_dir(), is_file=p.is_file())

    async def mkdir(self, path: str, *, parents: bool = True, exist_ok: bool = True) -> None:
        self._host_path(path).mkdir(parents=parents, exist_ok=exist_ok)


class LocalIsolatedSandbox(LocalSandbox):
    """LocalSandbox with a per-session ``$HOME`` for exec (per-session CLI identity).

    Filesystem behavior is identical to LocalSandbox (same session root); only
    the exec environment differs.
    """

    def __init__(
        self,
        session_key: str,
        root: Path,
        home_dir: Path,
        path_append: str = "",
        env_passthrough: list[str] | None = None,
    ):
        super().__init__(session_key, root, path_append=path_append)
        self._home = Path(home_dir)
        self._extra_passthrough = tuple(env_passthrough or ())

    def _build_env(self) -> dict[str, str]:
        # Ensure the session home exists before the child looks for it.
        self._home.mkdir(parents=True, exist_ok=True)
        host = os.environ
        essential = _WINDOWS_ESSENTIAL if sys.platform == "win32" else _POSIX_ESSENTIAL

        env: dict[str, str] = {}
        for key in (*essential, *self._extra_passthrough):
            if key in host:
                env[key] = host[key]

        if self.path_append:
            env["PATH"] = env.get("PATH", "") + os.pathsep + self.path_append

        home = str(self._home)
        # POSIX + Windows home anchors. Setting both on either platform is
        # harmless — a var no CLI reads is just an unused key.
        env["HOME"] = home
        env["USERPROFILE"] = home
        env["XDG_CONFIG_HOME"] = str(self._home / ".config")
        env["XDG_CACHE_HOME"] = str(self._home / ".cache")
        env["XDG_DATA_HOME"] = str(self._home / ".local" / "share")
        env["XDG_STATE_HOME"] = str(self._home / ".local" / "state")
        env["APPDATA"] = str(self._home / "AppData" / "Roaming")
        env["LOCALAPPDATA"] = str(self._home / "AppData" / "Local")
        return env
