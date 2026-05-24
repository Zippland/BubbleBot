"""Session bindings store + media relocation helpers.

Pulled out of `loop.py` so AgentLoop only carries dispatch / loop concerns.
State-free; the caller owns the bindings dict and passes it in.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from loguru import logger

from bubbles.session.manager import Session


def get_bindings_path(data_dir: Path) -> Path:
    """Where the session_bindings.json file lives."""
    return data_dir / "session_bindings.json"


def load_session_bindings(data_dir: Path) -> dict[str, str]:
    """Load `{channel}:{chat_id} -> session_key` bindings (empty dict on miss/error)."""
    path = get_bindings_path(data_dir)
    if not path.exists():
        return {}
    try:
        with open(path, encoding="utf-8") as f:
            bindings = json.load(f)
        logger.debug("Loaded {} session bindings", len(bindings))
        return bindings
    except Exception as e:
        logger.warning("Failed to load session bindings: {}", e)
        return {}


def save_session_bindings(data_dir: Path, bindings: dict[str, str]) -> None:
    """Persist bindings dict to disk."""
    path = get_bindings_path(data_dir)
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(bindings, f, indent=2, ensure_ascii=False)
    except Exception as e:
        logger.warning("Failed to save session bindings: {}", e)


def get_bindings_for_session(
    bindings: dict[str, str], session_key: str
) -> list[str]:
    """All `{channel}:{chat_id}` pairs bound to this session."""
    return [k for k, v in bindings.items() if v == session_key]


def relocate_media_to_session(
    media_paths: list[str], session: Session
) -> list[str]:
    """Move media files to session.directory/data/ if they're elsewhere.

    Channel layer may download media before knowing the final session binding,
    so files might land in a fallback dir; move them onto the right session.
    Returns the updated list of paths.
    """
    if not media_paths or not session.directory:
        return media_paths

    target_dir = session.directory / "data"
    target_dir.mkdir(parents=True, exist_ok=True)

    updated: list[str] = []
    for path in media_paths:
        p = Path(path)
        if not p.is_file():
            updated.append(path)
            continue
        try:
            p.relative_to(target_dir)
            updated.append(path)  # Already correct
            continue
        except ValueError:
            pass
        new_path = target_dir / p.name
        if new_path.exists():
            stem, suffix = new_path.stem, new_path.suffix
            for i in range(1, 100):
                new_path = target_dir / f"{stem}_{i}{suffix}"
                if not new_path.exists():
                    break
        try:
            shutil.move(str(p), str(new_path))
            logger.debug("Moved media {} -> {}", p, new_path)
            updated.append(str(new_path))
        except Exception as e:
            logger.warning("Failed to move media {}: {}", p, e)
            updated.append(path)
    return updated
