"""Sandbox interface + result types.

A `Sandbox` is an execution environment for one session. It owns two
capabilities the agent tools need:

- **exec**: run a shell command, return stdout/stderr/exit-code.
- **filesystem**: read/write/list/stat files, addressed by paths *relative to
  the sandbox root* (the session working directory).

All FS methods take sandbox-relative paths. The tool layer is responsible for
lexical containment (rejecting ``..`` / absolute escapes / ``~`` outside root);
the backend additionally enforces symlink-escape via ``resolve_within_root``.
Together they preserve the L1 path-sandbox guarantee (SPEC §7, SECURITY §4.1)
regardless of which backend is active.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class ExecResult:
    """Outcome of a shell command run inside a sandbox."""

    stdout: str
    stderr: str
    returncode: int
    timed_out: bool = False


@dataclass
class DirEntry:
    """One entry in a directory listing."""

    name: str
    is_dir: bool


@dataclass
class StatResult:
    """Metadata about a path inside a sandbox."""

    size: int
    mtime: float
    is_dir: bool
    is_file: bool


class Sandbox(ABC):
    """An execution environment bound to a single session.

    Attributes:
        session_key: The session this sandbox serves.
        root: The sandbox-internal absolute path of the workspace root. For
            local backends this equals the host session directory; for
            container/remote backends it is the path *inside* that environment
            (e.g. ``/workspace``).
        provides_hard_isolation: True when the backend structurally prevents
            access outside the workspace (mount namespace, VM, separate host).
            False for local backends, whose containment is best-effort L1/L2.
    """

    session_key: str
    root: str
    provides_hard_isolation: bool = False

    async def start(self) -> None:
        """Provision the environment. Idempotent; safe to call once per lifecycle."""

    async def close(self) -> None:
        """Tear down the environment and release resources."""

    @abstractmethod
    async def exec(self, command: str, *, cwd: str, timeout: int) -> ExecResult:
        """Run ``command`` with working directory ``cwd`` (sandbox-internal path)."""

    @abstractmethod
    async def read_bytes(self, path: str) -> bytes:
        """Read a file. Raises FileNotFoundError / IsADirectoryError as appropriate."""

    @abstractmethod
    async def write_bytes(self, path: str, data: bytes) -> None:
        """Write a file, creating parent directories as needed."""

    @abstractmethod
    async def list_dir(self, path: str) -> list[DirEntry]:
        """List directory entries (sorted by name)."""

    @abstractmethod
    async def stat(self, path: str) -> StatResult | None:
        """Return metadata for ``path``, or None if it does not exist."""

    @abstractmethod
    async def mkdir(self, path: str, *, parents: bool = True, exist_ok: bool = True) -> None:
        """Create a directory."""

    async def exists(self, path: str) -> bool:
        """Whether ``path`` exists. Default derives from ``stat``."""
        return await self.stat(path) is not None

    @abstractmethod
    async def resolve_within_root(self, path: str) -> str:
        """Resolve ``path`` to a sandbox-internal absolute path, enforcing that
        it stays within ``root`` (following symlinks). Raises PermissionError on
        escape. This is the backend's half of the path-sandbox guarantee.
        """
