"""Image-generation backend construction shared by CLI entry points."""

from __future__ import annotations

from typing import TYPE_CHECKING

import typer

from bubbles.image_generation import ImageGenerationBackend, OpenAIImageGenerationBackend

if TYPE_CHECKING:
    from bubbles.config.schema import Config


def _build_image_generation_backend(config: "Config") -> ImageGenerationBackend | None:
    """Build the configured backend, raising ``ValueError`` for invalid config."""
    image_config = config.tools.image_generation
    if not image_config.enabled:
        return None

    provider_name = image_config.provider.strip().lower()
    if provider_name != "openai":
        raise ValueError(
            "Unsupported image generation provider "
            f"{image_config.provider!r}; currently supported: openai"
        )

    provider_config = config.providers.openai
    if not provider_config.api_key.strip():
        raise ValueError(
            "Image generation is enabled, but providers.openai.api_key is not configured"
        )

    return OpenAIImageGenerationBackend(
        api_key=provider_config.api_key,
        model=image_config.model,
        api_base=provider_config.api_base,
        extra_headers=provider_config.extra_headers,
        timeout=image_config.timeout,
    )


def _make_image_generation_backend(config: "Config") -> ImageGenerationBackend | None:
    """Build the backend for a CLI entry point, reporting configuration errors cleanly."""
    try:
        return _build_image_generation_backend(config)
    except ValueError as exc:
        # Lazy import avoids a cycle when this helper is imported directly:
        # commands imports agent_cmd, which imports this module.
        from bubbles.cli.commands import console

        console.print(f"[red]Error: {exc}[/red]")
        raise typer.Exit(1) from None
