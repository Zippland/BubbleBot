"""SandboxManager: resolve and cache one Sandbox per session.

Maps ``session_key`` → live ``Sandbox``. The backend for a session is chosen
by (in order): the explicit per-session override (``SessionConfig.sandbox`` →
``/config sandbox <backend>``), else the global default
(``config.tools.sandbox.default``).

Sandboxes are cached and kept warm across turns (matters once container/remote
backends land — no cold start per message). Switching a session's backend at
runtime transparently tears down the old sandbox and builds the new one.
"""

from __future__ import annotations

from pathlib import Path

from loguru import logger

from bubbles.sandbox.base import Sandbox
from bubbles.sandbox.local import LocalIsolatedSandbox, LocalSandbox

# Sibling of ~/.bubbles/sessions/ — per-session credential homes live here,
# OUTSIDE the session working dir, so the model's file tools cannot read them.
SESSION_HOMES_DIRNAME = "session_homes"


class SandboxManager:
    """Owns the session_key → Sandbox mapping and lifecycle."""

    def __init__(self, config: "SandboxConfig | None" = None, path_append: str = ""):
        from bubbles.config.schema import SandboxConfig

        self.config = config or SandboxConfig()
        self.path_append = path_append
        self._cache: dict[str, Sandbox] = {}
        # Remember which backend each cached sandbox was built for, so a
        # per-session backend change triggers a rebuild.
        self._backends: dict[str, str] = {}

    def _resolve_backend(self, override: str | None) -> str:
        return (override or self.config.default or "local").lower()

    def _session_home(self, session_dir: Path) -> Path:
        """``~/.bubbles/sessions/<key>`` → ``~/.bubbles/session_homes/<key>``."""
        return session_dir.parent.parent / SESSION_HOMES_DIRNAME / session_dir.name

    def _build(self, backend: str, session_key: str, session_dir: Path) -> Sandbox:
        if backend == "local_isolated":
            return LocalIsolatedSandbox(
                session_key=session_key,
                root=session_dir,
                home_dir=self._session_home(session_dir),
                path_append=self.path_append,
                env_passthrough=list(self.config.env_passthrough),
            )
        if backend not in ("local", ""):
            logger.warning(
                "Unknown sandbox backend {!r} for session {}; falling back to 'local'",
                backend, session_key,
            )
        return LocalSandbox(
            session_key=session_key,
            root=session_dir,
            path_append=self.path_append,
        )

    async def get(
        self, session_key: str, session_dir: Path, backend: str | None = None
    ) -> Sandbox:
        """Return the (cached) sandbox for a session, building it on first use."""
        resolved = self._resolve_backend(backend)

        cached = self._cache.get(session_key)
        if cached is not None and self._backends.get(session_key) == resolved:
            return cached

        # Backend changed (or first use): drop the old one before rebuilding.
        if cached is not None:
            await self.close(session_key)

        sandbox = self._build(resolved, session_key, Path(session_dir))
        await sandbox.start()
        self._cache[session_key] = sandbox
        self._backends[session_key] = resolved
        return sandbox

    async def close(self, session_key: str) -> None:
        sandbox = self._cache.pop(session_key, None)
        self._backends.pop(session_key, None)
        if sandbox is not None:
            try:
                await sandbox.close()
            except Exception as e:
                logger.warning("Error closing sandbox for {}: {}", session_key, e)

    async def close_all(self) -> None:
        for key in list(self._cache.keys()):
            await self.close(key)
