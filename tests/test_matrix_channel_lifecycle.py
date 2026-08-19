import asyncio
import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from bubbles.bus.queue import MessageBus
from bubbles.config.schema import MatrixConfig


@pytest.fixture
def matrix_module(monkeypatch):
    """Load the real Matrix module with its optional SDK surface stubbed."""

    class _NioType:
        pass

    class _Cleaner:
        def __init__(self, **_kwargs) -> None:
            pass

        def clean(self, value: str) -> str:
            return value

    nio = ModuleType("nio")
    nio.__path__ = []
    for name in (
        "AsyncClient",
        "AsyncClientConfig",
        "ContentRepositoryConfigError",
        "DownloadError",
        "InviteEvent",
        "JoinError",
        "MatrixRoom",
        "MemoryDownloadResponse",
        "RoomEncryptedMedia",
        "RoomMessage",
        "RoomMessageMedia",
        "RoomMessageText",
        "RoomSendError",
        "RoomTypingError",
        "SyncError",
        "UploadError",
    ):
        setattr(nio, name, _NioType)

    nio_crypto = ModuleType("nio.crypto")
    nio_crypto.__path__ = []
    nio_attachments = ModuleType("nio.crypto.attachments")
    nio_attachments.decrypt_attachment = lambda *_args, **_kwargs: b""
    nio_exceptions = ModuleType("nio.exceptions")
    nio_exceptions.EncryptionError = _NioType

    nh3 = ModuleType("nh3")
    nh3.Cleaner = _Cleaner
    mistune = ModuleType("mistune")
    mistune.create_markdown = lambda **_kwargs: lambda value: value

    for name, module in {
        "nio": nio,
        "nio.crypto": nio_crypto,
        "nio.crypto.attachments": nio_attachments,
        "nio.exceptions": nio_exceptions,
        "nh3": nh3,
        "mistune": mistune,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)

    module_name = "_bubblebot_matrix_lifecycle_under_test"
    module_path = Path(__file__).parents[1] / "bubbles" / "channels" / "matrix.py"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, module_name, module)
    spec.loader.exec_module(module)
    return module


class _FakeAsyncClient:
    def __init__(self, **_kwargs) -> None:
        self.user_id = ""
        self.access_token = ""
        self.device_id = ""

    def add_event_callback(self, _callback, _event_type) -> None:
        pass

    def add_response_callback(self, _callback, _response_type) -> None:
        pass

    def stop_sync_forever(self) -> None:
        pass

    async def close(self) -> None:
        pass


@pytest.mark.asyncio
async def test_start_waits_for_sync_task_lifecycle(
    monkeypatch, tmp_path, matrix_module
) -> None:
    sync_started = asyncio.Event()
    finish_sync = asyncio.Event()

    async def sync_loop() -> None:
        sync_started.set()
        await finish_sync.wait()

    monkeypatch.setattr(matrix_module, "get_data_dir", lambda: tmp_path)
    monkeypatch.setattr(matrix_module, "AsyncClient", _FakeAsyncClient)
    monkeypatch.setattr(
        matrix_module,
        "AsyncClientConfig",
        lambda **kwargs: SimpleNamespace(**kwargs),
    )

    channel = matrix_module.MatrixChannel(
        MatrixConfig(
            homeserver="https://matrix.example",
            access_token="token",
            user_id="@bubblebot:matrix.example",
        ),
        MessageBus(),
    )
    monkeypatch.setattr(channel, "_sync_loop", sync_loop)

    start_task = asyncio.create_task(channel.start())
    await asyncio.wait_for(sync_started.wait(), timeout=1)

    assert channel._sync_task is not None
    assert not channel._sync_task.done()
    assert not start_task.done()

    finish_sync.set()
    await asyncio.wait_for(start_task, timeout=1)
    await channel.stop()
