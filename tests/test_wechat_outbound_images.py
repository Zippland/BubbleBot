"""Outbound WeChat image preparation and WCFerry delivery tests."""

from __future__ import annotations

import asyncio
import os
import random
import threading
import time
from pathlib import Path

import pytest
from PIL import Image

from bubbles.bus.events import OutboundMessage
from bubbles.bus.queue import MessageBus
from bubbles.channels import wechat as wechat_channel_module
from bubbles.channels import wechat_media
from bubbles.channels.wechat import WeChatChannel
from bubbles.channels.wechat_media import (
    PreparedWeChatImage,
    prepare_wechat_image,
    prune_wechat_image_cache,
)
from bubbles.config.schema import WeChatConfig


def _write_noisy_png(path: Path, size: tuple[int, int] = (900, 900)) -> bytes:
    pixels = random.Random(0).randbytes(size[0] * size[1] * 3)
    image = Image.frombytes("RGB", size, pixels)
    try:
        image.save(path, format="PNG")
    finally:
        image.close()
    return path.read_bytes()


def test_prepare_wechat_image_bounds_large_png_without_changing_source(
    tmp_path: Path,
) -> None:
    source = tmp_path / "large.png"
    original_bytes = _write_noisy_png(source)
    cache_dir = tmp_path / "cache"

    prepared = prepare_wechat_image(
        str(source),
        cache_dir=cache_dir,
        max_bytes=1024 * 1024,
        max_edge=1280,
    )

    assert prepared.derived is True
    assert prepared.send_as_file is False
    assert prepared.path != str(source)
    assert Path(prepared.path).is_file()
    assert prepared.prepared_size_bytes <= 1024 * 1024
    assert prepared.prepared_size_bytes < prepared.original_size_bytes
    assert source.read_bytes() == original_bytes
    with Image.open(prepared.path) as image:
        assert image.format == "JPEG"
        assert max(image.size) <= 1280


def test_prepare_wechat_image_passes_small_jpeg_through(tmp_path: Path) -> None:
    source = tmp_path / "small.jpg"
    with Image.new("RGB", (64, 64), "blue") as image:
        image.save(source, format="JPEG")

    prepared = prepare_wechat_image(
        str(source),
        cache_dir=tmp_path / "cache",
        max_bytes=1024 * 1024,
        max_edge=1280,
    )

    assert prepared.path == str(source)
    assert prepared.derived is False
    assert prepared.send_as_file is False


def test_prepare_wechat_image_falls_back_to_file_for_invalid_image(
    tmp_path: Path,
) -> None:
    source = tmp_path / "broken.png"
    source.write_bytes(b"not-an-image" * 100)

    prepared = prepare_wechat_image(
        str(source),
        cache_dir=tmp_path / "cache",
        max_bytes=1,
        max_edge=1280,
    )

    assert prepared.path == str(source)
    assert prepared.derived is False
    assert prepared.send_as_file is True


def test_prepare_wechat_image_never_returns_an_over_budget_derivative(
    tmp_path: Path,
) -> None:
    source = tmp_path / "large.png"
    _write_noisy_png(source)
    cache_dir = tmp_path / "cache"

    prepared = prepare_wechat_image(
        str(source),
        cache_dir=cache_dir,
        max_bytes=1,
        max_edge=1280,
    )

    assert prepared.derived is False
    assert prepared.send_as_file is True
    assert list(cache_dir.glob("bubbles-wechat-*.jpg")) == []


def test_cache_failure_bypasses_small_image_and_degrades_large_image(
    tmp_path: Path,
) -> None:
    cache_blocker = tmp_path / "cache"
    cache_blocker.write_bytes(b"not-a-directory")
    small = tmp_path / "small.jpg"
    with Image.new("RGB", (64, 64), "blue") as image:
        image.save(small, format="JPEG")
    large = tmp_path / "large.png"
    _write_noisy_png(large)

    small_prepared = prepare_wechat_image(
        str(small),
        cache_dir=cache_blocker,
        max_bytes=1024 * 1024,
        max_edge=1280,
    )
    large_prepared = prepare_wechat_image(
        str(large),
        cache_dir=cache_blocker,
        max_bytes=1024 * 1024,
        max_edge=1280,
    )

    assert small_prepared.path == str(small)
    assert small_prepared.send_as_file is False
    assert large_prepared.path == str(large)
    assert large_prepared.send_as_file is True


def test_prepare_wechat_image_rejects_unsafe_decode_size(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "large-dimensions.png"
    with Image.new("RGB", (64, 64), "green") as image:
        image.save(source, format="PNG")
    monkeypatch.setattr(wechat_media, "WECHAT_IMAGE_MAX_DECODE_PIXELS", 100)

    prepared = prepare_wechat_image(
        str(source),
        cache_dir=tmp_path / "cache",
        max_bytes=1024 * 1024,
        max_edge=32,
    )

    assert prepared.send_as_file is True
    assert prepared.derived is False


def test_prune_wechat_image_cache_removes_only_expired_derivatives(
    tmp_path: Path,
) -> None:
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    expired = cache_dir / "bubbles-wechat-expired.jpg"
    recent = cache_dir / "bubbles-wechat-recent.jpg"
    unrelated = cache_dir / "keep.jpg"
    for path in (expired, recent, unrelated):
        path.write_bytes(b"x")
    old_timestamp = time.time() - 601
    os.utime(expired, (old_timestamp, old_timestamp))

    prune_wechat_image_cache(cache_dir, max_age_seconds=600)

    assert not expired.exists()
    assert recent.exists()
    assert unrelated.exists()


class _FakeWcf:
    def __init__(self, image_status: int = 0) -> None:
        self.image_status = image_status
        self.image_paths: list[str] = []
        self.file_paths: list[str] = []
        self.texts: list[str] = []

    def send_image(self, path: str, receiver: str) -> int:
        assert receiver == "alice"
        assert Path(path).is_file()
        self.image_paths.append(path)
        return self.image_status

    def send_file(self, path: str, receiver: str) -> int:
        assert receiver == "alice"
        assert Path(path).is_file()
        self.file_paths.append(path)
        return 0

    def send_text(self, text: str, receiver: str, aters: str) -> int:
        assert receiver == "alice"
        assert aters == ""
        self.texts.append(text)
        return 0

    def disable_recv_msg(self) -> int:
        return 0


@pytest.mark.asyncio
async def test_wechat_send_uses_cached_derivative_and_preserves_original(
    tmp_path: Path,
) -> None:
    source = tmp_path / "large.png"
    original_bytes = _write_noisy_png(source)
    channel = WeChatChannel(
        WeChatConfig(enabled=True),
        MessageBus(),
    )
    channel._outbound_image_cache = tmp_path / "cache"
    fake = _FakeWcf()
    channel.wcf = fake

    await channel.send(
        OutboundMessage(
            channel="wechat",
            chat_id="alice",
            content="caption",
            media=[str(source)],
        )
    )

    assert len(fake.image_paths) == 1
    derivative = Path(fake.image_paths[0])
    assert derivative.parent == channel._outbound_image_cache
    assert derivative.is_file()  # Kept briefly because native consumption may be async.
    assert derivative.suffix == ".jpg"
    assert source.read_bytes() == original_bytes
    assert fake.file_paths == []
    assert fake.texts == ["caption"]
    await channel.stop()
    assert not derivative.exists()


@pytest.mark.asyncio
async def test_wechat_send_falls_back_to_file_and_continues_text(
    tmp_path: Path,
) -> None:
    source = tmp_path / "large.png"
    _write_noisy_png(source)
    channel = WeChatChannel(
        WeChatConfig(enabled=True),
        MessageBus(),
    )
    channel._outbound_image_cache = tmp_path / "cache"
    fake = _FakeWcf(image_status=7)
    channel.wcf = fake

    await channel.send(
        OutboundMessage(
            channel="wechat",
            chat_id="alice",
            content="caption",
            media=[str(source)],
        )
    )

    assert len(fake.image_paths) == 1
    assert fake.file_paths == [str(source)]
    assert fake.texts == ["caption"]
    await channel.stop()


@pytest.mark.asyncio
async def test_wechat_cached_derivative_expires_without_another_send(
    tmp_path: Path,
) -> None:
    source = tmp_path / "large.png"
    _write_noisy_png(source)
    channel = WeChatChannel(
        WeChatConfig(enabled=True),
        MessageBus(),
    )
    channel._outbound_image_cache = tmp_path / "cache"
    channel._outbound_image_cache_ttl_seconds = 0.01
    fake = _FakeWcf()
    channel.wcf = fake

    await channel.send(
        OutboundMessage(
            channel="wechat",
            chat_id="alice",
            content="",
            media=[str(source)],
        )
    )
    derivative = Path(fake.image_paths[0])
    assert derivative.exists()

    await asyncio.sleep(0.05)

    assert not derivative.exists()
    await channel.stop()


@pytest.mark.asyncio
async def test_stopping_one_channel_does_not_prune_another_instance_cache(
    tmp_path: Path,
) -> None:
    first = WeChatChannel(WeChatConfig(enabled=True), MessageBus())
    second = WeChatChannel(WeChatConfig(enabled=True), MessageBus())
    first._outbound_image_cache = tmp_path / "first"
    second._outbound_image_cache = tmp_path / "second"
    first._outbound_image_cache.mkdir()
    second._outbound_image_cache.mkdir()
    first_file = first._outbound_image_cache / "bubbles-wechat-first.jpg"
    second_file = second._outbound_image_cache / "bubbles-wechat-second.jpg"
    first_file.write_bytes(b"first")
    second_file.write_bytes(b"second")

    await first.stop()

    assert not first_file.exists()
    assert second_file.exists()


@pytest.mark.asyncio
async def test_cancelling_during_prepare_removes_abandoned_derivative(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source.png"
    source.write_bytes(b"source")
    cache_dir = tmp_path / "cache"
    derivative = cache_dir / "bubbles-wechat-cancelled.jpg"
    started = threading.Event()
    release = threading.Event()

    def slow_prepare(path: str, **kwargs: object) -> PreparedWeChatImage:
        cache_dir.mkdir()
        derivative.write_bytes(b"derived")
        started.set()
        release.wait(timeout=2)
        return PreparedWeChatImage(
            path=str(derivative),
            original_path=path,
            derived=True,
            send_as_file=False,
            original_size_bytes=6,
            prepared_size_bytes=7,
        )

    monkeypatch.setattr(wechat_channel_module, "prepare_wechat_image", slow_prepare)
    channel = WeChatChannel(WeChatConfig(enabled=True), MessageBus())
    channel._outbound_image_cache = cache_dir
    fake = _FakeWcf()
    channel.wcf = fake
    send_task = asyncio.create_task(
        channel.send(
            OutboundMessage(
                channel="wechat",
                chat_id="alice",
                content="",
                media=[str(source)],
            )
        )
    )
    assert await asyncio.to_thread(started.wait, 1)

    send_task.cancel()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await send_task

    assert not derivative.exists()
    assert fake.image_paths == []
    await channel.stop()


def test_wechat_image_budget_config_accepts_camel_case() -> None:
    config = WeChatConfig(
        outboundImageMaxBytes=2 * 1024 * 1024,
        outboundImageMaxEdge=1600,
    )

    assert config.outbound_image_max_bytes == 2 * 1024 * 1024
    assert config.outbound_image_max_edge == 1600
