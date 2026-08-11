"""Image generation backend/tool contract and workspace-boundary tests.

The HTTP assertions follow the OpenAI Image API contract: text-only requests
use JSON ``/images/generations`` while reference-image requests use multipart
``/images/edits`` with repeated ``image[]`` parts.
"""

from __future__ import annotations

import base64
import json
from email.parser import BytesParser
from email.policy import default
from pathlib import Path

import httpx
import pytest
import typer

import bubbles.agent.tools.image_generation as image_tool_module
from bubbles.agent.context import ContextBuilder
from bubbles.agent.loop import AgentLoop
from bubbles.agent.tools.image_generation import GenerateImageTool
from bubbles.agent.tools.message import _resolve_media_path
from bubbles.bus.queue import MessageBus
from bubbles.cli._image_generation import (
    _build_image_generation_backend,
    _make_image_generation_backend,
)
from bubbles.config.schema import Config
from bubbles.image_generation import (
    GeneratedImage,
    ImageGenerationError,
    ImageGenerationRequest,
    OpenAIImageGenerationBackend,
    ReferenceImage,
)
from bubbles.sandbox.base import StatResult
from bubbles.sandbox.local import LocalSandbox
from bubbles.session.manager import SessionManager

_PNG_BYTES = b"\x89PNG\r\n\x1a\nreference-png"


class _FakeBackend:
    """Deliberately does not inherit a concrete base: the tool accepts a Protocol."""

    def __init__(self, images: list[GeneratedImage] | None = None) -> None:
        self.images = images or []
        self.requests: list[ImageGenerationRequest] = []

    async def generate(self, request: ImageGenerationRequest) -> list[GeneratedImage]:
        self.requests.append(request)
        return self.images


def _multipart_parts(request: httpx.Request) -> list[dict[str, object]]:
    """Parse an httpx multipart request without depending on an extra package."""
    content_type = request.headers["content-type"]
    message = BytesParser(policy=default).parsebytes(
        f"Content-Type: {content_type}\r\nMIME-Version: 1.0\r\n\r\n".encode()
        + request.content
    )
    assert message.is_multipart()
    return [
        {
            "name": part.get_param("name", header="content-disposition"),
            "filename": part.get_filename(),
            "content_type": part.get_content_type(),
            "data": part.get_payload(decode=True),
        }
        for part in message.iter_parts()
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("aspect_ratio", "expected_size"),
    [
        ("1:1", "816x816"),
        ("3:2", "1008x672"),
        ("2:3", "672x1008"),
        ("16:9", "1280x720"),
        ("9:16", "720x1280"),
    ],
)
async def test_openai_generation_sends_json_and_decodes_base64(
    aspect_ratio: str, expected_size: str
) -> None:
    expected_images = [b"first-image", b"second-image"]
    captured: dict[str, object] = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        await request.aread()
        captured["request"] = request
        captured["payload"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "data": [
                    {"b64_json": base64.b64encode(image).decode()}
                    for image in expected_images
                ]
            },
        )

    backend = OpenAIImageGenerationBackend(
        api_key="test-secret",
        model="gpt-image-2",
        transport=httpx.MockTransport(handler),
    )
    images = await backend.generate(
        ImageGenerationRequest(
            prompt="Draw an otter",
            aspect_ratio=aspect_ratio,
            output_format="webp",
            count=2,
        )
    )

    request = captured["request"]
    assert isinstance(request, httpx.Request)
    assert request.method == "POST"
    assert request.url == httpx.URL("https://api.openai.com/v1/images/generations")
    assert request.headers["authorization"] == "Bearer test-secret"
    assert request.headers["content-type"].startswith("application/json")
    assert captured["payload"] == {
        "model": "gpt-image-2",
        "prompt": "Draw an otter",
        "n": 2,
        "size": expected_size,
        "quality": "low",
        "output_format": "webp",
    }
    assert [(image.data, image.output_format) for image in images] == [
        (b"first-image", "webp"),
        (b"second-image", "webp"),
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("quality", ["auto", "high"])
async def test_openai_backend_rejects_uncapped_quality(quality: str) -> None:
    backend = OpenAIImageGenerationBackend(api_key="test-secret")

    with pytest.raises(ImageGenerationError, match="Unsupported image quality"):
        await backend.generate(ImageGenerationRequest(prompt="Draw an otter", quality=quality))


@pytest.mark.asyncio
async def test_openai_reference_images_use_multipart_edits() -> None:
    captured: dict[str, httpx.Request] = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        await request.aread()
        captured["request"] = request
        return httpx.Response(
            200,
            json={"data": [{"b64_json": base64.b64encode(b"edited").decode()}]},
        )

    backend = OpenAIImageGenerationBackend(
        api_key="test-secret",
        api_base="https://gateway.example/v1/",
        transport=httpx.MockTransport(handler),
    )
    images = await backend.generate(
        ImageGenerationRequest(
            prompt="Combine these references",
            reference_images=(
                ReferenceImage(b"png-bytes", "one.png", "image/png"),
                ReferenceImage(b"jpeg-bytes", "two.jpg", "image/jpeg"),
            ),
            aspect_ratio="16:9",
            quality="medium",
            output_format="jpeg",
            count=1,
        )
    )

    request = captured["request"]
    assert request.url == httpx.URL("https://gateway.example/v1/images/edits")
    assert request.headers["content-type"].startswith("multipart/form-data;")
    parts = _multipart_parts(request)

    fields = {
        part["name"]: part["data"].decode()
        for part in parts
        if part["filename"] is None
    }
    assert fields == {
        "model": "gpt-image-2",
        "prompt": "Combine these references",
        "n": "1",
        "size": "1280x720",
        "quality": "medium",
        "output_format": "jpeg",
    }
    assert "input_fidelity" not in fields  # gpt-image-2 fixes this at high fidelity.

    files = [part for part in parts if part["filename"] is not None]
    assert files == [
        {
            "name": "image[]",
            "filename": "one.png",
            "content_type": "image/png",
            "data": b"png-bytes",
        },
        {
            "name": "image[]",
            "filename": "two.jpg",
            "content_type": "image/jpeg",
            "data": b"jpeg-bytes",
        },
    ]
    assert [(image.data, image.output_format) for image in images] == [(b"edited", "jpeg")]


@pytest.mark.asyncio
async def test_openai_error_does_not_leak_api_key() -> None:
    secret = "test-api-key-sensitive-value"
    gateway_secret = "gateway-sensitive-value"

    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            401,
            json={
                "error": {
                    "code": f"invalid_api_key:{secret}",
                    "message": (
                        f"The supplied credentials {secret} and {gateway_secret} are invalid"
                    ),
                }
            },
        )

    backend = OpenAIImageGenerationBackend(
        api_key=secret,
        extra_headers={"X-Gateway-Key": gateway_secret},
        transport=httpx.MockTransport(handler),
    )
    with pytest.raises(ImageGenerationError) as error:
        await backend.generate(ImageGenerationRequest(prompt="Draw a circle"))

    public_error = str(error.value)
    assert secret not in public_error
    assert gateway_secret not in public_error
    assert "[REDACTED]" in public_error
    assert "HTTP 401" in public_error


@pytest.mark.asyncio
async def test_openai_retries_only_explicit_transient_http_responses() -> None:
    statuses = [429, 503, 200]
    calls = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        status = statuses[calls]
        calls += 1
        if status == 200:
            return httpx.Response(
                200,
                json={"data": [{"b64_json": base64.b64encode(b"image").decode()}]},
            )
        return httpx.Response(
            status,
            headers={"Retry-After": "0"},
            json={"error": {"code": "transient", "message": "try again"}},
        )

    backend = OpenAIImageGenerationBackend(
        api_key="test-secret",
        transport=httpx.MockTransport(handler),
    )
    images = await backend.generate(ImageGenerationRequest(prompt="Draw a circle"))

    assert calls == 3
    assert [image.data for image in images] == [b"image"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("exception_type", "expected"),
    [
        (httpx.ConnectError, "Could not reach"),
        (httpx.ReadTimeout, "timed out"),
    ],
)
async def test_openai_does_not_retry_ambiguous_transport_failures(
    exception_type: type[httpx.RequestError], expected: str
) -> None:
    calls = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        raise exception_type("ambiguous transport failure", request=request)

    backend = OpenAIImageGenerationBackend(
        api_key="test-secret",
        transport=httpx.MockTransport(handler),
    )
    with pytest.raises(ImageGenerationError, match=expected):
        await backend.generate(ImageGenerationRequest(prompt="Draw a circle"))

    assert calls == 1


@pytest.mark.asyncio
async def test_tool_uses_replaceable_backend_and_writes_relative_artifacts(
    tmp_path: Path,
) -> None:
    fake = _FakeBackend(
        [
            GeneratedImage(data=b"image-one", output_format="webp"),
            GeneratedImage(data=b"image-two", output_format="webp"),
        ]
    )
    tool = GenerateImageTool(backend=fake)
    tool.set_sandbox(LocalSandbox("session", tmp_path))

    result = json.loads(
        await tool.execute(
            prompt="  Draw a durable abstraction  ",
            aspect_ratio="9:16",
            output_format="webp",
            count=2,
        )
    )

    assert len(fake.requests) == 1
    assert fake.requests[0] == ImageGenerationRequest(
        prompt="Draw a durable abstraction",
        aspect_ratio="9:16",
        quality="low",
        output_format="webp",
        count=2,
    )
    artifacts = result["artifacts"]
    assert len(artifacts) == 2
    assert artifacts[0]["path"] != artifacts[1]["path"]
    for artifact, expected in zip(artifacts, (b"image-one", b"image-two"), strict=True):
        path = artifact["path"]
        assert path.startswith("data/generated-images/")
        assert not Path(path).is_absolute()
        assert ".." not in Path(path).parts
        assert artifact["mime_type"] == "image/webp"
        assert artifact["size_bytes"] == len(expected)
        assert (tmp_path / path).read_bytes() == expected


@pytest.mark.asyncio
async def test_tool_reads_reference_from_sandbox_and_rejects_escapes(tmp_path: Path) -> None:
    workspace = tmp_path / "session"
    workspace.mkdir()
    (workspace / "refs").mkdir()
    (workspace / "refs" / "source.png").write_bytes(_PNG_BYTES)
    outside = tmp_path / "outside.png"
    outside.write_bytes(_PNG_BYTES)
    (workspace / "refs" / "linked.png").symlink_to(outside)

    fake = _FakeBackend([GeneratedImage(data=b"output", output_format="png")])
    tool = GenerateImageTool(backend=fake)
    tool.set_sandbox(LocalSandbox("session", workspace))

    success = await tool.execute(
        prompt="Use the reference",
        reference_images=["refs/source.png"],
    )
    assert "artifacts" in json.loads(success)
    assert fake.requests[0].reference_images == (
        ReferenceImage(
            data=_PNG_BYTES,
            filename="source.png",
            mime_type="image/png",
        ),
    )

    for escaped in ("../outside.png", str(outside), "~/outside.png", r"C:\outside.png"):
        error = await tool.execute(prompt="Steal a reference", reference_images=[escaped])
        assert error.startswith("Error:")
        assert "session-relative" in error

    symlink_error = await tool.execute(
        prompt="Follow an escaping symlink",
        reference_images=["refs/linked.png"],
    )
    assert symlink_error.startswith("Error:")
    assert "outside allowed directory" in symlink_error

    # Only the valid call reached the backend.
    assert len(fake.requests) == 1


@pytest.mark.asyncio
async def test_inbound_image_exposes_reusable_relative_path_without_host_leak(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "session"
    attachment = workspace / "data" / "reference.png"
    attachment.parent.mkdir(parents=True)
    attachment.write_bytes(_PNG_BYTES)

    content = ContextBuilder(session_dir=workspace)._build_user_content(
        "Please reuse my image",
        [str(attachment)],
    )
    assert isinstance(content, list)
    text_block = content[-1]
    assert text_block["type"] == "text"
    assert "[Attached image paths: data/reference.png]" in text_block["text"]
    assert str(workspace) not in text_block["text"]

    fake = _FakeBackend([GeneratedImage(data=b"output", output_format="png")])
    tool = GenerateImageTool(backend=fake)
    tool.set_sandbox(LocalSandbox("session", workspace))
    result = await tool.execute(
        prompt="Use the attached image",
        reference_images=["data/reference.png"],
    )

    assert "artifacts" in json.loads(result)
    assert fake.requests[0].reference_images[0].data == _PNG_BYTES


@pytest.mark.asyncio
async def test_tool_enforces_reference_count_format_and_size_limits(tmp_path: Path) -> None:
    workspace = tmp_path / "session"
    workspace.mkdir()
    unsupported = workspace / "reference.gif"
    unsupported.write_bytes(b"gif")
    disguised = workspace / "disguised.png"
    disguised.write_bytes(b"not really a png")
    oversized = workspace / "oversized.png"
    with oversized.open("wb") as file:
        file.truncate(50 * 1024 * 1024 + 1)

    fake = _FakeBackend([GeneratedImage(data=b"output", output_format="png")])
    tool = GenerateImageTool(backend=fake)
    tool.set_sandbox(LocalSandbox("session", workspace))

    too_many = await tool.execute(
        prompt="Too many references",
        reference_images=[f"reference-{index}.png" for index in range(5)],
    )
    wrong_format = await tool.execute(
        prompt="Unsupported input",
        reference_images=["reference.gif"],
    )
    wrong_contents = await tool.execute(
        prompt="Disguised input",
        reference_images=["disguised.png"],
    )
    too_large = await tool.execute(
        prompt="Oversized input",
        reference_images=["oversized.png"],
    )

    assert "At most 4" in too_many
    assert "PNG, JPEG, or WebP" in wrong_format
    assert "contents do not match" in wrong_contents
    assert "50 MiB" in too_large
    assert fake.requests == []


@pytest.mark.asyncio
async def test_tool_rechecks_reference_size_after_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class _StaleStatSandbox(LocalSandbox):
        async def stat(self, path: str) -> StatResult | None:
            result = await super().stat(path)
            if result is None:
                return None
            return StatResult(size=1, mtime=result.mtime, is_dir=False, is_file=True)

    workspace = tmp_path / "session"
    workspace.mkdir()
    (workspace / "growing.png").write_bytes(_PNG_BYTES)
    monkeypatch.setattr(image_tool_module, "_MAX_REFERENCE_IMAGE_BYTES", 8)

    fake = _FakeBackend([GeneratedImage(data=b"output", output_format="png")])
    tool = GenerateImageTool(backend=fake)
    tool.set_sandbox(_StaleStatSandbox("session", workspace))

    error = await tool.execute(
        prompt="Read after a stale stat",
        reference_images=["growing.png"],
    )
    assert "50 MiB" in error
    assert fake.requests == []


@pytest.mark.parametrize(
    ("params", "expected"),
    [
        ({"prompt": ""}, "prompt must be at least 1 chars"),
        ({"prompt": "x", "count": 0}, "count must be >= 1"),
        ({"prompt": "x", "count": 5}, "count must be <= 4"),
        ({"prompt": "x", "count": "2"}, "count should be integer"),
        ({"prompt": "x", "aspect_ratio": "4:3"}, "aspect_ratio must be one of"),
        ({"prompt": "x", "quality": "high"}, "quality must be one of"),
        ({"prompt": "x", "quality": "auto"}, "quality must be one of"),
        ({"prompt": "x", "output_format": "gif"}, "output_format must be one of"),
    ],
)
def test_tool_parameter_schema_rejects_invalid_boundaries(
    params: dict[str, object], expected: str
) -> None:
    tool = GenerateImageTool(backend=_FakeBackend())
    assert expected in "; ".join(tool.validate_params(params))


def test_tool_parameter_schema_exposes_only_provider_neutral_inputs() -> None:
    tool = GenerateImageTool(backend=_FakeBackend())
    schema = tool.parameters
    assert schema["required"] == ["prompt"]
    assert schema["additionalProperties"] is False
    assert schema["properties"]["reference_images"]["maxItems"] == 4
    assert schema["properties"]["quality"]["enum"] == ["low", "medium"]
    assert schema["properties"]["quality"]["default"] == "low"
    assert set(schema["properties"]) == {
        "prompt",
        "reference_images",
        "aspect_ratio",
        "quality",
        "output_format",
        "count",
    }
    assert tool.validate_params(
        {
            "prompt": "x",
            "reference_images": ["refs/a.png"],
            "aspect_ratio": "auto",
            "quality": "medium",
            "output_format": "png",
            "count": 1,
        }
    ) == []


def test_message_media_path_cannot_escape_session_workspace(tmp_path: Path) -> None:
    workspace = tmp_path / "session"
    workspace.mkdir()
    inside = workspace / "data" / "generated-images" / "safe.png"
    inside.parent.mkdir(parents=True)
    inside.write_bytes(b"safe")
    outside = tmp_path / "outside.png"
    outside.write_bytes(b"secret")

    assert _resolve_media_path("data/generated-images/safe.png", workspace) == str(
        inside.resolve()
    )
    for escaped in ("../outside.png", str(outside)):
        with pytest.raises(ValueError, match="outside the session workspace"):
            _resolve_media_path(escaped, workspace)
    with pytest.raises(ValueError, match="require a session workspace"):
        _resolve_media_path(str(outside), None)


class _Provider:
    def get_default_model(self) -> str:
        return "test-model"


def test_agent_registers_image_tool_only_when_backend_is_enabled(tmp_path: Path) -> None:
    common = {
        "bus": MessageBus(),
        "provider": _Provider(),
        "max_tokens": 100,
        "memory_window": 5,
        "context_limit": 10_000,
        "session_manager": SessionManager(sessions_dir=tmp_path / "sessions"),
    }

    disabled = AgentLoop(**common)
    assert not disabled.tools.has("generate_image")

    enabled = AgentLoop(
        **common,
        image_generation_backend=_FakeBackend(
            [GeneratedImage(data=b"output", output_format="png")]
        ),
    )
    assert enabled.tools.has("generate_image")
    turn_tools = enabled.build_turn_tools(
        channel="test",
        chat_id="chat",
        message_id=None,
        session_dir=tmp_path,
        session_key="test:chat",
        session=None,
        sandbox=LocalSandbox("test:chat", tmp_path),
    )
    assert turn_tools.has("generate_image")


def test_image_backend_factory_is_opt_in_and_cli_errors_cleanly() -> None:
    disabled = Config(tools={"image_generation": {"enabled": False}})
    assert _build_image_generation_backend(disabled) is None

    enabled = Config(
        tools={"image_generation": {"enabled": True}},
        providers={"openai": {"api_key": "test-secret"}},
    )
    assert isinstance(
        _build_image_generation_backend(enabled),
        OpenAIImageGenerationBackend,
    )

    missing_key = Config(
        tools={"image_generation": {"enabled": True}},
        providers={"openai": {"api_key": ""}},
    )
    with pytest.raises(typer.Exit) as error:
        _make_image_generation_backend(missing_key)
    assert error.value.exit_code == 1
