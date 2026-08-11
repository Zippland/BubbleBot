"""Provider-neutral contracts for image generation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True, slots=True)
class ReferenceImage:
    """One reference image, already read from the current session sandbox."""

    data: bytes
    filename: str
    mime_type: str


@dataclass(frozen=True, slots=True)
class ImageGenerationRequest:
    """Stable request understood by image-generation backends."""

    prompt: str
    reference_images: tuple[ReferenceImage, ...] = ()
    aspect_ratio: str = "auto"
    quality: str = "low"
    output_format: str = "png"
    count: int = 1


@dataclass(frozen=True, slots=True)
class GeneratedImage:
    """Image bytes returned by a backend."""

    data: bytes
    output_format: str


class ImageGenerationError(RuntimeError):
    """A safe, user-presentable image-generation failure."""


class ImageGenerationBackend(Protocol):
    """Provider adapter used by the agent-facing image tool."""

    async def generate(self, request: ImageGenerationRequest) -> list[GeneratedImage]:
        """Generate images for ``request`` or raise ``ImageGenerationError``."""
        ...
