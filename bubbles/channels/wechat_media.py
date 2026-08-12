"""WeChat-specific outbound image preparation.

WCFerry accepts a filesystem path and forwards it to a native RPC. Large PNGs
can remain stuck in WeChat's sending state, so the channel sends a bounded JPEG
derivative while preserving the original session artifact.
"""

from __future__ import annotations

import os
import tempfile
import time
from dataclasses import dataclass, replace
from pathlib import Path

from loguru import logger
from PIL import Image, ImageOps, UnidentifiedImageError

WECHAT_IMAGE_CACHE_TTL_SECONDS = 10 * 60
WECHAT_IMAGE_MAX_DECODE_PIXELS = 20_000_000


@dataclass(frozen=True)
class PreparedWeChatImage:
    """A path ready for WCFerry plus its delivery fallback semantics."""

    path: str
    original_path: str
    derived: bool
    send_as_file: bool
    original_size_bytes: int
    prepared_size_bytes: int


def prune_wechat_image_cache(
    cache_dir: Path,
    *,
    max_age_seconds: int = WECHAT_IMAGE_CACHE_TTL_SECONDS,
) -> None:
    """Remove only expired derivatives created by this module."""
    if max_age_seconds < 0:
        raise ValueError("max_age_seconds must not be negative")
    try:
        if not cache_dir.exists():
            return
        candidates = list(cache_dir.rglob("bubbles-wechat-*.jpg"))
    except OSError as exc:
        logger.warning("Failed to inspect WeChat image cache {}: {}", cache_dir, exc)
        return

    cutoff = time.time() - max_age_seconds
    for candidate in candidates:
        try:
            if candidate.stat().st_mtime <= cutoff:
                candidate.unlink(missing_ok=True)
        except OSError as exc:
            logger.warning("Failed to prune WeChat image cache file {}: {}", candidate, exc)


def remove_wechat_cached_image(path: str) -> None:
    """Best-effort removal restricted to derivatives created by this module."""
    candidate = Path(path)
    if not candidate.name.startswith("bubbles-wechat-") or candidate.suffix != ".jpg":
        return
    try:
        candidate.unlink(missing_ok=True)
    except OSError as exc:
        logger.warning("Failed to remove WeChat image cache file {}: {}", candidate, exc)


def _to_rgb(image: Image.Image) -> Image.Image:
    has_alpha = "A" in image.getbands() or "transparency" in image.info
    if not has_alpha:
        return image.convert("RGB")

    rgba = image.convert("RGBA")
    try:
        background = Image.new("RGB", rgba.size, "white")
        try:
            alpha = rgba.getchannel("A")
            try:
                background.paste(rgba, mask=alpha)
            finally:
                alpha.close()
        except Exception:
            background.close()
            raise
        return background
    finally:
        rgba.close()


def _save_bounded_jpeg(
    image: Image.Image,
    destination: Path,
    *,
    max_bytes: int,
    max_edge: int,
) -> tuple[int, tuple[int, int]]:
    attempts = [
        (max_edge, 82),
        (max_edge, 72),
        (min(max_edge, 1024), 72),
        (min(max_edge, 816), 65),
        (min(max_edge, 640), 55),
        (min(max_edge, 512), 50),
        (min(max_edge, 384), 45),
        (min(max_edge, 256), 40),
    ]
    final_size = 0
    final_dimensions = image.size

    for target_edge, quality in dict.fromkeys(attempts):
        candidate = image.copy()
        try:
            candidate.thumbnail(
                (target_edge, target_edge),
                Image.Resampling.LANCZOS,
            )
            candidate.save(
                destination,
                format="JPEG",
                quality=quality,
                optimize=True,
                subsampling=2,
            )
            final_dimensions = candidate.size
        finally:
            candidate.close()

        final_size = destination.stat().st_size
        if final_size <= max_bytes:
            break

    return final_size, final_dimensions


def prepare_wechat_image(
    path: str,
    *,
    cache_dir: Path,
    max_bytes: int,
    max_edge: int,
) -> PreparedWeChatImage:
    """Return an original or cached JPEG path suitable for WCFerry.

    ``max_bytes`` is an operational delivery budget, not a protocol limit.
    Animated images are not flattened; oversized animations fall back to file
    delivery so they do not enter WCFerry's problematic image path.
    """
    if max_bytes <= 0:
        raise ValueError("max_bytes must be positive")
    if max_edge <= 0:
        raise ValueError("max_edge must be positive")

    source = Path(path)
    source_size = source.stat().st_size
    original = PreparedWeChatImage(
        path=str(source),
        original_path=str(source),
        derived=False,
        send_as_file=False,
        original_size_bytes=source_size,
        prepared_size_bytes=source_size,
    )
    temporary_path: Path | None = None
    try:
        with Image.open(source) as opened:
            width, height = opened.size
            oversized = source_size > max_bytes or max(width, height) > max_edge
            if not oversized:
                return original
            if (
                getattr(opened, "is_animated", False)
                or width * height > WECHAT_IMAGE_MAX_DECODE_PIXELS
            ):
                return replace(original, send_as_file=True)

            cache_dir.mkdir(parents=True, exist_ok=True)
            prune_wechat_image_cache(cache_dir)
            with ImageOps.exif_transpose(opened) as transposed:
                with _to_rgb(transposed) as rgb:
                    descriptor, temporary_name = tempfile.mkstemp(
                        prefix="bubbles-wechat-",
                        suffix=".jpg",
                        dir=cache_dir,
                    )
                    os.close(descriptor)
                    temporary_path = Path(temporary_name)
                    prepared_size, dimensions = _save_bounded_jpeg(
                        rgb,
                        temporary_path,
                        max_bytes=max_bytes,
                        max_edge=max_edge,
                    )
                    if prepared_size > max_bytes:
                        remove_wechat_cached_image(str(temporary_path))
                        logger.warning(
                            "Could not fit WeChat image {} within {} bytes; "
                            "falling back to file delivery",
                            source,
                            max_bytes,
                        )
                        return replace(original, send_as_file=True)

        logger.info(
            "Prepared WeChat image {}: {} bytes -> {} bytes, {}x{}",
            source,
            source_size,
            prepared_size,
            dimensions[0],
            dimensions[1],
        )
        return PreparedWeChatImage(
            path=str(temporary_path),
            original_path=str(source),
            derived=True,
            send_as_file=False,
            original_size_bytes=source_size,
            prepared_size_bytes=prepared_size,
        )
    except (Image.DecompressionBombError, UnidentifiedImageError, OSError, ValueError) as exc:
        if temporary_path is not None:
            remove_wechat_cached_image(str(temporary_path))
        logger.warning(
            "Could not prepare WeChat image {}; falling back to file delivery: {}",
            source,
            exc,
        )
        return replace(original, send_as_file=True)
