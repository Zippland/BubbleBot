"""Cross-machine media transfer for the split bus.

The in-process bus passed ``media`` as local filesystem paths, silently relying
on channels and harness sharing a disk. After the split they don't. This module
converts a local file into a self-describing wire descriptor and back:

- **dehydrate** (producer side): local path → descriptor. Small files inline as
  base64; larger files (and everything non-trivial) go to a content-addressed
  blob store, leaving only ``{name, sha256, size, mime}`` on the wire.
- **rehydrate** (consumer side): descriptor → a local file the consumer owns
  (fetching blob bytes if needed). The result is a local path, so the tools and
  ContextBuilder that expect ``media: list[str]`` of paths are UNCHANGED.

The blob store is injected as two async callbacks (``put_blob``/``get_blob``),
so all of this logic is transport-agnostic and unit-testable without Redis.
"""

from __future__ import annotations

import base64
import hashlib
import mimetypes
from pathlib import Path
from typing import Any, Awaitable, Callable

from loguru import logger

PutBlob = Callable[[bytes], Awaitable[str]]   # bytes -> sha256
GetBlob = Callable[[str], Awaitable[bytes]]   # sha256 -> bytes


def _guess_mime(name: str) -> str:
    mime, _ = mimetypes.guess_type(name)
    return mime or "application/octet-stream"


async def dehydrate_media(
    paths: list[str],
    *,
    inline_max_bytes: int,
    put_blob: PutBlob,
) -> list[dict[str, Any]]:
    """Local paths → wire descriptors. Missing files are skipped with a warning."""
    out: list[dict[str, Any]] = []
    for p in paths:
        fp = Path(p)
        if not fp.is_file():
            logger.warning("dehydrate_media: skipping missing/non-file path {}", p)
            continue
        data = fp.read_bytes()
        name = fp.name
        mime = _guess_mime(name)
        if len(data) <= inline_max_bytes:
            out.append({
                "name": name,
                "mime": mime,
                "size": len(data),
                "inline_b64": base64.b64encode(data).decode("ascii"),
            })
        else:
            sha = await put_blob(data)
            out.append({"name": name, "mime": mime, "size": len(data), "sha256": sha})
    return out


async def rehydrate_media(
    descriptors: list[Any],
    *,
    dest_dir: Path,
    get_blob: GetBlob,
) -> list[str]:
    """Wire descriptors → local file paths under ``dest_dir``.

    Tolerates raw path strings too (LocalBus / same-disk case, or already-local
    entries): a plain string is passed through unchanged.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    out: list[str] = []
    for d in descriptors:
        # Same-disk / legacy: already a local path string.
        if isinstance(d, str):
            out.append(d)
            continue
        if not isinstance(d, dict):
            logger.warning("rehydrate_media: unexpected media entry {!r}, skipping", d)
            continue

        name = d.get("name") or "media"
        try:
            if "inline_b64" in d:
                data = base64.b64decode(d["inline_b64"])
            elif "sha256" in d:
                data = await get_blob(d["sha256"])
            else:
                logger.warning("rehydrate_media: descriptor without payload {}, skipping", d)
                continue
        except Exception as e:
            logger.error("rehydrate_media: failed to fetch {}: {}", name, e)
            continue

        target = _unique_path(dest_dir, name)
        target.write_bytes(data)
        out.append(str(target))
    return out


def _unique_path(dest_dir: Path, name: str) -> Path:
    """Avoid clobbering an existing file of the same name in dest_dir."""
    target = dest_dir / Path(name).name
    if not target.exists():
        return target
    stem, suffix = target.stem, target.suffix
    i = 1
    while True:
        cand = dest_dir / f"{stem}_{i}{suffix}"
        if not cand.exists():
            return cand
        i += 1


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()
