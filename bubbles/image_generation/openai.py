"""OpenAI Image API backend."""

from __future__ import annotations

import asyncio
import base64
import binascii
from collections.abc import Mapping
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any

import httpx

from bubbles.image_generation.base import (
    GeneratedImage,
    ImageGenerationError,
    ImageGenerationRequest,
)

_SIZE_BY_ASPECT_RATIO = {
    "auto": "auto",
    "1:1": "816x816",
    "3:2": "1536x1024",
    "2:3": "1024x1536",
    "16:9": "1536x864",
    "9:16": "864x1536",
}
_QUALITIES = {"low", "medium"}
_OUTPUT_FORMATS = {"png", "jpeg", "webp"}


class OpenAIImageGenerationBackend:
    """Generate and edit images through OpenAI's Image API.

    ``api_base`` is the OpenAI-compatible API root (normally ending in
    ``/v1``), not an individual endpoint.
    """

    def __init__(
        self,
        api_key: str | None,
        model: str = "gpt-image-2",
        api_base: str | None = "https://api.openai.com/v1",
        extra_headers: Mapping[str, str] | None = None,
        timeout: float = 180.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        if timeout <= 0:
            raise ValueError("timeout must be positive")
        if not model.strip():
            raise ValueError("model must not be empty")

        self._api_key = (api_key or "").strip()
        self._model = model.strip()
        self._api_base = (api_base or "https://api.openai.com/v1").rstrip("/")
        self._extra_headers = dict(extra_headers or {})
        # Compatible gateways often carry a second credential in extra_headers.
        # Keep all credential values out of provider-authored error messages.
        header_secrets = {
            value.strip()
            for value in self._extra_headers.values()
            if isinstance(value, str) and len(value.strip()) >= 4
        }
        self._secret_values = tuple(
            sorted({self._api_key, *header_secrets} - {""}, key=len, reverse=True)
        )
        self._timeout = timeout
        self._transport = transport

    async def generate(self, request: ImageGenerationRequest) -> list[GeneratedImage]:
        self._validate_request(request)
        if not self._api_key:
            raise ImageGenerationError("OpenAI API key is not configured")

        headers = dict(self._extra_headers)
        headers["Authorization"] = f"Bearer {self._api_key}"

        client_kwargs: dict[str, Any] = {"timeout": self._timeout}
        if self._transport is not None:
            client_kwargs["transport"] = self._transport

        try:
            async with httpx.AsyncClient(**client_kwargs) as client:
                response = await self._request_with_retries(client, headers, request)
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise ImageGenerationError(self._format_http_error(exc.response)) from None
        except httpx.TimeoutException:
            raise ImageGenerationError("OpenAI image request timed out") from None
        except httpx.RequestError:
            raise ImageGenerationError("Could not reach the OpenAI image API") from None

        return self._decode_response(response, request.output_format)

    async def _request_with_retries(
        self,
        client: httpx.AsyncClient,
        headers: dict[str, str],
        request: ImageGenerationRequest,
    ) -> httpx.Response:
        """Retry only explicit throttling/server responses.

        Connection and timeout errors are intentionally not retried: the server
        may already have accepted a paid generation even if the client did not
        receive its response.
        """
        for attempt in range(3):
            if request.reference_images:
                response = await self._edit(client, headers, request)
            else:
                response = await self._generate(client, headers, request)

            retryable = response.status_code == 429 or response.status_code >= 500
            if not retryable or attempt == 2:
                return response
            await asyncio.sleep(self._retry_delay(response, attempt))

        raise AssertionError("unreachable")

    @staticmethod
    def _retry_delay(response: httpx.Response, attempt: int) -> float:
        """Return a bounded Retry-After delay, falling back to short backoff."""
        raw = response.headers.get("Retry-After", "").strip()
        if raw:
            try:
                return min(10.0, max(0.0, float(raw)))
            except ValueError:
                try:
                    retry_at = parsedate_to_datetime(raw)
                    if retry_at.tzinfo is None:
                        retry_at = retry_at.replace(tzinfo=timezone.utc)
                    seconds = (retry_at - datetime.now(timezone.utc)).total_seconds()
                    return min(10.0, max(0.0, seconds))
                except (TypeError, ValueError, OverflowError):
                    pass
        return min(2.0, 0.5 * (2**attempt))

    async def _generate(
        self,
        client: httpx.AsyncClient,
        headers: dict[str, str],
        request: ImageGenerationRequest,
    ) -> httpx.Response:
        return await client.post(
            f"{self._api_base}/images/generations",
            headers=headers,
            json=self._request_fields(request),
        )

    async def _edit(
        self,
        client: httpx.AsyncClient,
        headers: dict[str, str],
        request: ImageGenerationRequest,
    ) -> httpx.Response:
        fields = {key: str(value) for key, value in self._request_fields(request).items()}
        files = [
            ("image[]", (image.filename, image.data, image.mime_type))
            for image in request.reference_images
        ]
        return await client.post(
            f"{self._api_base}/images/edits",
            headers=headers,
            data=fields,
            files=files,
        )

    def _request_fields(self, request: ImageGenerationRequest) -> dict[str, str | int]:
        return {
            "model": self._model,
            "prompt": request.prompt,
            "n": request.count,
            "size": _SIZE_BY_ASPECT_RATIO[request.aspect_ratio],
            "quality": request.quality,
            "output_format": request.output_format,
        }

    @staticmethod
    def _validate_request(request: ImageGenerationRequest) -> None:
        if not request.prompt.strip():
            raise ImageGenerationError("Image prompt must not be empty")
        if request.aspect_ratio not in _SIZE_BY_ASPECT_RATIO:
            raise ImageGenerationError(f"Unsupported aspect ratio: {request.aspect_ratio}")
        if request.quality not in _QUALITIES:
            raise ImageGenerationError(f"Unsupported image quality: {request.quality}")
        if request.output_format not in _OUTPUT_FORMATS:
            raise ImageGenerationError(f"Unsupported output format: {request.output_format}")
        if not 1 <= request.count <= 4:
            raise ImageGenerationError("Image count must be between 1 and 4")
        if any(not image.data for image in request.reference_images):
            raise ImageGenerationError("Reference images must not be empty")

    def _format_http_error(self, response: httpx.Response) -> str:
        status = response.status_code
        code = ""
        message = ""
        try:
            payload = response.json()
            error = payload.get("error", {}) if isinstance(payload, dict) else {}
            if isinstance(error, dict):
                code = str(error.get("code") or error.get("type") or "").strip()
                message = str(error.get("message") or "").strip()
        except (ValueError, TypeError):
            pass

        # Never let a provider echo credentials back through a tool result.
        for secret in self._secret_values:
            code = code.replace(secret, "[REDACTED]")
            message = message.replace(secret, "[REDACTED]")
        code = code[:100]
        message = message[:500]

        detail = f" ({code})" if code else ""
        suffix = f": {message}" if message else ""
        return f"OpenAI image API returned HTTP {status}{detail}{suffix}"

    @staticmethod
    def _decode_response(response: httpx.Response, output_format: str) -> list[GeneratedImage]:
        try:
            payload = response.json()
            items = payload.get("data") if isinstance(payload, dict) else None
            if not isinstance(items, list) or not items:
                raise ValueError

            images: list[GeneratedImage] = []
            for item in items:
                encoded = item.get("b64_json") if isinstance(item, dict) else None
                if not isinstance(encoded, str) or not encoded:
                    raise ValueError
                images.append(
                    GeneratedImage(
                        data=base64.b64decode(encoded, validate=True),
                        output_format=output_format,
                    )
                )
            return images
        except (ValueError, TypeError, binascii.Error):
            raise ImageGenerationError("OpenAI image API returned an invalid response") from None
