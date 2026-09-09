"""Agent-facing image generation tool."""

from __future__ import annotations

import json
from pathlib import PurePosixPath, PureWindowsPath
from typing import Any
from uuid import uuid4

from bubbles.agent.tools.base import Tool
from bubbles.image_generation import (
    GeneratedImage,
    ImageGenerationBackend,
    ImageGenerationError,
    ImageGenerationRequest,
    ReferenceImage,
)
from bubbles.sandbox.base import Sandbox

_ASPECT_RATIOS = ["auto", "1:1", "3:2", "2:3", "16:9", "9:16"]
_QUALITIES = ["low", "medium"]
_OUTPUT_FORMATS = ["png", "jpeg", "webp"]
_MIME_TYPES = {"png": "image/png", "jpeg": "image/jpeg", "webp": "image/webp"}
_REFERENCE_MIME_TYPES = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
}
_MAX_REFERENCE_IMAGES = 4
_MAX_REFERENCE_IMAGE_BYTES = 50 * 1024 * 1024


def _content_matches_image_format(data: bytes, suffix: str) -> bool:
    """Validate the small, stable file signatures accepted by the Image API."""
    if suffix == ".png":
        return data.startswith(b"\x89PNG\r\n\x1a\n")
    if suffix in {".jpg", ".jpeg"}:
        return data.startswith(b"\xff\xd8\xff")
    if suffix == ".webp":
        return len(data) >= 12 and data.startswith(b"RIFF") and data[8:12] == b"WEBP"
    return False


class GenerateImageTool(Tool):
    """Generate images through a configurable provider backend."""

    def __init__(self, backend: ImageGenerationBackend | None) -> None:
        self._backend = backend
        self._sandbox: Sandbox | None = None

    def set_sandbox(self, sandbox: Sandbox | None) -> None:
        """Bind the current session sandbox. Called once for each turn."""
        self._sandbox = sandbox

    @property
    def name(self) -> str:
        return "generate_image"

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "minLength": 1,
                    "description": "A detailed description of the image to create or edit.",
                },
                "reference_images": {
                    "type": "array",
                    "items": {"type": "string", "minLength": 1},
                    "maxItems": _MAX_REFERENCE_IMAGES,
                    "description": (
                        "Optional session-relative image paths to use as visual references. "
                        "When provided, the backend edits or composes from these images."
                    ),
                },
                "aspect_ratio": {
                    "type": "string",
                    "enum": _ASPECT_RATIOS,
                    "description": "Output aspect ratio.",
                    "default": "auto",
                },
                "quality": {
                    "type": "string",
                    "enum": _QUALITIES,
                    "description": (
                        "Cost-bounded rendering quality; auto and high are not exposed."
                    ),
                    "default": "low",
                },
                "output_format": {
                    "type": "string",
                    "enum": _OUTPUT_FORMATS,
                    "description": "Image file format.",
                    "default": "png",
                },
                "count": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 4,
                    "description": "Number of images to generate.",
                    "default": 1,
                },
            },
            "required": ["prompt"],
            "additionalProperties": False,
        }

    async def execute(
        self,
        prompt: str,
        reference_images: list[str] | None = None,
        aspect_ratio: str = "auto",
        quality: str = "low",
        output_format: str = "png",
        count: int = 1,
        **kwargs: Any,
    ) -> str:
        if self._backend is None:
            return "Error: 生图后端未配置，当前无法生成或编辑图片。请先配置生图服务。"
        if self._sandbox is None:
            return "Error: no sandbox bound"
        if not prompt.strip():
            return "Error: Image prompt must not be empty"
        if aspect_ratio not in _ASPECT_RATIOS:
            return f"Error: Unsupported aspect ratio: {aspect_ratio}"
        if quality not in _QUALITIES:
            return f"Error: Unsupported image quality: {quality}"
        if output_format not in _OUTPUT_FORMATS:
            return f"Error: Unsupported output format: {output_format}"
        if isinstance(count, bool) or not isinstance(count, int) or not 1 <= count <= 4:
            return "Error: Image count must be between 1 and 4"
        if reference_images is not None and not isinstance(reference_images, list):
            return "Error: reference_images must be an array"
        if reference_images and len(reference_images) > _MAX_REFERENCE_IMAGES:
            return f"Error: At most {_MAX_REFERENCE_IMAGES} reference images are allowed"

        try:
            references = await self._read_references(reference_images or [])
            generated = await self._backend.generate(
                ImageGenerationRequest(
                    prompt=prompt.strip(),
                    reference_images=references,
                    aspect_ratio=aspect_ratio,
                    quality=quality,
                    output_format=output_format,
                    count=count,
                )
            )
            if not generated:
                return "Error: Image backend returned no images"
            return await self._write_artifacts(generated)
        except ImageGenerationError as exc:
            return f"Error: Image generation failed: {exc}"
        except (FileNotFoundError, IsADirectoryError, PermissionError, ValueError) as exc:
            return f"Error: Could not read reference image: {exc}"
        except Exception:
            # Arbitrary backend exceptions can contain request headers. Keep the
            # tool boundary deliberately opaque so credentials never leak.
            return "Error: Image generation failed unexpectedly"

    async def _read_references(self, paths: list[str]) -> tuple[ReferenceImage, ...]:
        if self._sandbox is None:
            raise RuntimeError("no sandbox bound")

        references: list[ReferenceImage] = []
        for path in paths:
            normalized = self._session_relative_path(path)
            stat = await self._sandbox.stat(normalized)
            if stat is None:
                raise FileNotFoundError(normalized)
            if not stat.is_file:
                raise IsADirectoryError(normalized)
            if stat.size > _MAX_REFERENCE_IMAGE_BYTES:
                raise ValueError(f"Reference image exceeds the 50 MiB limit: {normalized}")

            filename = PurePosixPath(normalized).name
            suffix = PurePosixPath(filename).suffix.lower()
            mime_type = _REFERENCE_MIME_TYPES.get(suffix)
            if mime_type is None:
                raise ValueError(
                    f"Unsupported reference image format for {normalized}; use PNG, JPEG, or WebP"
                )
            data = await self._sandbox.read_bytes(normalized)
            # Re-check after reading: a local file may change between stat() and
            # read_bytes(), and remote sandbox implementations need not provide an
            # atomic stat+read operation.
            if len(data) > _MAX_REFERENCE_IMAGE_BYTES:
                raise ValueError(f"Reference image exceeds the 50 MiB limit: {normalized}")
            if not _content_matches_image_format(data, suffix):
                raise ValueError(
                    f"Reference image contents do not match its {suffix} extension: {normalized}"
                )
            references.append(
                ReferenceImage(data=data, filename=filename, mime_type=mime_type)
            )
        return tuple(references)

    @staticmethod
    def _session_relative_path(path: str) -> str:
        if not isinstance(path, str):
            raise ValueError("reference image paths must be strings")
        raw = path.strip()
        windows = PureWindowsPath(raw)
        normalized = raw.replace("\\", "/")
        posix = PurePosixPath(normalized)
        if (
            not raw
            or "\x00" in raw
            or raw.startswith("~")
            or posix.is_absolute()
            or bool(windows.drive)
            or ".." in posix.parts
        ):
            raise PermissionError(f"Reference image path must be session-relative: {path}")
        return normalized

    async def _write_artifacts(self, images: list[GeneratedImage]) -> str:
        if self._sandbox is None:
            raise RuntimeError("no sandbox bound")

        artifacts: list[dict[str, str | int]] = []
        for image in images:
            if image.output_format not in _OUTPUT_FORMATS:
                raise ImageGenerationError(
                    f"Backend returned unsupported output format: {image.output_format}"
                )
            if not image.data:
                raise ImageGenerationError("Backend returned an empty image")

            path = f"data/generated-images/{uuid4().hex}.{image.output_format}"
            await self._sandbox.write_bytes(path, image.data)
            artifacts.append(
                {
                    "path": path,
                    "mime_type": _MIME_TYPES[image.output_format],
                    "size_bytes": len(image.data),
                }
            )

        return json.dumps({"artifacts": artifacts}, ensure_ascii=False, separators=(",", ":"))
