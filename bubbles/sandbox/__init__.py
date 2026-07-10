"""Sandbox abstraction: pluggable execution backends for the agent.

The agent kernel (harness) is decoupled from *where* shell commands run and
files live. Each session binds to one `Sandbox` backend. Tools (`exec`,
`read_file`/`write_file`/`edit_file`/`list_dir`) no longer touch the OS
directly — they delegate every side effect to the session's sandbox.

Phase 0 ships local backends only (`LocalSandbox`, `LocalIsolatedSandbox`);
the interface is deliberately backend-agnostic so container / remote backends
(docker, firejail, e2b, …) drop in later without changing the tools.
"""

from bubbles.sandbox.base import DirEntry, ExecResult, Sandbox, StatResult
from bubbles.sandbox.local import LocalIsolatedSandbox, LocalSandbox
from bubbles.sandbox.manager import SandboxManager

__all__ = [
    "Sandbox",
    "ExecResult",
    "DirEntry",
    "StatResult",
    "LocalSandbox",
    "LocalIsolatedSandbox",
    "SandboxManager",
]
