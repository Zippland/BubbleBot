"""WeChat channel implementation using wcferry."""

from __future__ import annotations

import asyncio
import ctypes
import importlib
import json
import logging
import os
import re
import socket
import time
from collections import OrderedDict
from collections.abc import Callable
from ctypes import wintypes
from dataclasses import dataclass
from pathlib import Path
from queue import Empty
from threading import Lock, Thread, Timer
from threading import enumerate as enumerate_threads
from uuid import uuid4

from loguru import logger

from bubbles.bus.events import OutboundMessage
from bubbles.bus.queue import MessageBus
from bubbles.channels import wechat_app
from bubbles.channels.base import BaseChannel
from bubbles.channels.mentions import replace_mentions
from bubbles.channels.wechat_app import IMAGE_EXTS
from bubbles.channels.wechat_group_events import parse_group_join_members
from bubbles.channels.wechat_media import (
    WECHAT_IMAGE_CACHE_TTL_SECONDS,
    prepare_wechat_image,
    prune_wechat_image_cache,
    remove_wechat_cached_image,
)
from bubbles.config.schema import WeChatConfig
from bubbles.gateway_control import NO_RESTART_EXIT_CODE
from bubbles.utils.helpers import get_data_path

try:
    from wcferry import Wcf, WxMsg
except ImportError:
    Wcf = None  # type: ignore
    WxMsg = None  # type: ignore


# WeChat message types
MSG_TYPE_TEXT = 1
MSG_TYPE_IMAGE = 3
MSG_TYPE_VOICE = 34
MSG_TYPE_VIDEO = 43
MSG_TYPE_EMOJI = 47
MSG_TYPE_APP = 49  # 文件、链接、小程序、引用等
MSG_TYPE_SYSTEM = 10000

# 联系人名单缓存：TTL 到点重读（抓改名），遇到未知 wxid 也重读一次（抓新好友），
# 但两次重读之间至少隔 MIN_REFRESH，避免未知发送者连发消息时反复查库。
CONTACTS_TTL_SEC = 30 * 60.0
CONTACTS_MIN_REFRESH_INTERVAL_SEC = 60.0

# 图片解密重试：微信的 download_attach 常在文件已落盘时仍报失败，wcferry 会因此
# 直接放弃解密。等一下再调，此时 msg.extra 已存在，wcferry 会跳过下载直接解密。
# 详见 _download_image_with_retry。
IMAGE_DOWNLOAD_ATTEMPTS = 4
IMAGE_DOWNLOAD_RETRY_DELAY_SEC = 1.5

# 微信复制出来的 ``@泡泡`` 只是普通文本，WCFerry 的 ``msg.is_at`` 不会
# 命中。群聊除了保留原生 @ 兼容性外，也把正文中的名字作为显式唤醒词。
WECHAT_BOT_TRIGGER_WORD = "泡泡"
WECHAT_RECEIVER_HEALTH_INTERVAL_SECONDS = 1.0
WECHAT_LOGIN_POLL_INTERVAL_SECONDS = 1.0
WECHAT_MESSAGE_TRANSPORT_READY_TIMEOUT_SECONDS = 10.0
WECHAT_MESSAGE_TRANSPORT_POLL_INTERVAL_SECONDS = 0.05
WCFERRY_MESSAGE_THREAD_JOIN_TIMEOUT_SECONDS = 5.0
# WCFerry injects into WeChat and binds machine-wide RPC ports.  The mutex must
# therefore span console/RDP sessions as well as Python processes.  A Local\
# mutex would allow the same Windows user to construct a second Wcf() from a
# different interactive session before the port listener becomes observable.
WCFERRY_MUTEX_NAME = r"Global\Bubblebot-WCFerry"
WCFERRY_COMMAND_PORT = 10086
WCFERRY_MESSAGE_PORT = WCFERRY_COMMAND_PORT + 1
WCFERRY_LEASE_FILE = "wcferry-lease.json"
WCFERRY_NATIVE_CLEANUP_TIMEOUT_SECONDS = 30

_WAIT_OBJECT_0 = 0x00000000
_WAIT_ABANDONED = 0x00000080
_WAIT_TIMEOUT = 0x00000102
_ERROR_INSUFFICIENT_BUFFER = 122
_AF_INET = 2
_TCP_TABLE_OWNER_PID_LISTENER = 3


class WcferryCleanupError(RuntimeError):
    """The process cannot prove that WCFerry detached from WeChat safely."""

    no_restart = True


class _WcferryRequestedExitError(RuntimeError):
    def __init__(self, exit_code: int) -> None:
        super().__init__(f"WCFerry requested an immediate process exit ({exit_code})")
        self.exit_code = exit_code


class _WcferryOsProxy:
    """Override only wcferry.client's os._exit during construction."""

    def __init__(self, wrapped) -> None:
        self._wrapped = wrapped

    def __getattr__(self, name: str):
        return getattr(self._wrapped, name)

    @staticmethod
    def _exit(exit_code: int) -> None:
        raise _WcferryRequestedExitError(exit_code)


class _MibTcpRowOwnerPid(ctypes.Structure):
    _fields_ = [
        ("dwState", wintypes.DWORD),
        ("dwLocalAddr", wintypes.DWORD),
        ("dwLocalPort", wintypes.DWORD),
        ("dwRemoteAddr", wintypes.DWORD),
        ("dwRemotePort", wintypes.DWORD),
        ("dwOwningPid", wintypes.DWORD),
    ]


def _load_kernel32_mutex_api():
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CreateMutexW.argtypes = [
        wintypes.LPVOID,
        wintypes.BOOL,
        wintypes.LPCWSTR,
    ]
    kernel32.CreateMutexW.restype = wintypes.HANDLE
    kernel32.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
    kernel32.WaitForSingleObject.restype = wintypes.DWORD
    kernel32.ReleaseMutex.argtypes = [wintypes.HANDLE]
    kernel32.ReleaseMutex.restype = wintypes.BOOL
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel32.CloseHandle.restype = wintypes.BOOL
    return kernel32


def _windows_mutex_error(operation: str, error_code: int | None = None) -> OSError:
    if error_code is None:
        error_code = ctypes.get_last_error()
    return OSError(error_code, f"{operation} failed: {ctypes.FormatError(error_code)}")


class _WcferryInstanceMutex:
    """Machine-wide WCFerry singleton guard across Windows processes/sessions."""

    def __init__(
        self,
        name: str = WCFERRY_MUTEX_NAME,
        *,
        enabled: bool | None = None,
        kernel32=None,
    ) -> None:
        self.name = name
        self.enabled = os.name == "nt" if enabled is None else enabled
        self._kernel32 = kernel32
        self._handle = None

    @property
    def held(self) -> bool:
        return self._handle is not None

    def acquire(self) -> None:
        if not self.enabled or self._handle is not None:
            return

        kernel32 = self._kernel32 or _load_kernel32_mutex_api()
        handle = kernel32.CreateMutexW(None, False, self.name)
        if not handle:
            raise _windows_mutex_error("CreateMutexW")

        try:
            wait_status = int(kernel32.WaitForSingleObject(handle, 0))
        except BaseException:
            kernel32.CloseHandle(handle)
            raise
        if wait_status in (_WAIT_OBJECT_0, _WAIT_ABANDONED):
            self._kernel32 = kernel32
            self._handle = handle
            return

        wait_error_code = (
            ctypes.get_last_error() if wait_status != _WAIT_TIMEOUT else None
        )
        close_succeeded = bool(kernel32.CloseHandle(handle))
        if not close_succeeded:
            raise _windows_mutex_error("CloseHandle")
        if wait_status == _WAIT_TIMEOUT:
            raise WcferryCleanupError(
                "另一 Bubblebot 进程正在使用 WCFerry；为避免微信连接冲突，本进程拒绝启动。"
            )
        raise _windows_mutex_error("WaitForSingleObject", wait_error_code)

    def release(self) -> None:
        if not self.enabled or self._handle is None:
            return

        kernel32 = self._kernel32
        handle = self._handle
        self._handle = None
        release_error: BaseException | None = None
        try:
            if not kernel32.ReleaseMutex(handle):
                release_error = _windows_mutex_error("ReleaseMutex")
        except BaseException as exc:
            release_error = exc
        try:
            if not kernel32.CloseHandle(handle) and release_error is None:
                release_error = _windows_mutex_error("CloseHandle")
        except BaseException as exc:
            if release_error is None:
                release_error = exc
        if release_error is not None:
            raise release_error


def _load_iphlpapi():
    api = ctypes.WinDLL("iphlpapi", use_last_error=True)
    api.GetExtendedTcpTable.argtypes = [
        wintypes.LPVOID,
        ctypes.POINTER(wintypes.ULONG),
        wintypes.BOOL,
        wintypes.ULONG,
        wintypes.ULONG,
        wintypes.ULONG,
    ]
    api.GetExtendedTcpTable.restype = wintypes.ULONG
    return api


def _listening_ipv4_ports(*, iphlpapi=None) -> set[int]:
    """Read the Windows listener table without connecting to WCFerry's NNG peer."""
    api = iphlpapi or _load_iphlpapi()
    size = wintypes.ULONG(0)
    status = int(
        api.GetExtendedTcpTable(
            None,
            ctypes.byref(size),
            False,
            _AF_INET,
            _TCP_TABLE_OWNER_PID_LISTENER,
            0,
        )
    )
    if status not in (0, _ERROR_INSUFFICIENT_BUFFER):
        raise OSError(status, "GetExtendedTcpTable size query failed")

    table = ctypes.create_string_buffer(size.value)
    status = int(
        api.GetExtendedTcpTable(
            table,
            ctypes.byref(size),
            False,
            _AF_INET,
            _TCP_TABLE_OWNER_PID_LISTENER,
            0,
        )
    )
    if status != 0:
        raise OSError(status, "GetExtendedTcpTable failed")

    row_count = ctypes.cast(table, ctypes.POINTER(wintypes.DWORD)).contents.value
    row_size = ctypes.sizeof(_MibTcpRowOwnerPid)
    first_row = ctypes.addressof(table) + ctypes.sizeof(wintypes.DWORD)
    ports: set[int] = set()
    for index in range(row_count):
        row = _MibTcpRowOwnerPid.from_address(first_row + index * row_size)
        ports.add(socket.ntohs(int(row.dwLocalPort) & 0xFFFF))
    return ports


def _assert_wcferry_ports_idle(*, enabled: bool | None = None, iphlpapi=None) -> None:
    if (os.name == "nt" if enabled is None else enabled) is False:
        return
    occupied = {WCFERRY_COMMAND_PORT, WCFERRY_MESSAGE_PORT} & _listening_ipv4_ports(
        iphlpapi=iphlpapi
    )
    if occupied:
        rendered = ", ".join(str(port) for port in sorted(occupied))
        raise WcferryCleanupError(
            f"检测到旧 WCFerry RPC 仍在监听端口 {rendered}；"
            "为避免重复注入微信，本进程拒绝启动。"
        )


def _wait_for_wcferry_ports_released(
    *,
    enabled: bool | None = None,
    timeout: float = 5.0,
    poll_interval: float = 0.1,
) -> None:
    if (os.name == "nt" if enabled is None else enabled) is False:
        return
    deadline = time.monotonic() + timeout
    occupied: set[int] = set()
    while True:
        occupied = {
            WCFERRY_COMMAND_PORT,
            WCFERRY_MESSAGE_PORT,
        } & _listening_ipv4_ports()
        if not occupied:
            return
        if time.monotonic() >= deadline:
            rendered = ", ".join(str(port) for port in sorted(occupied))
            raise WcferryCleanupError(
                f"WxDestroySDK 返回后 WCFerry 端口仍在监听：{rendered}。"
            )
        time.sleep(poll_interval)


def _wcferry_message_transport_ready(client) -> bool:
    """Whether pynng has established WCFerry's message-channel pipe."""
    msg_socket = getattr(client, "msg_socket", None)
    if msg_socket is None:
        return False
    try:
        return bool(msg_socket.pipes)
    except Exception:
        return False


def _capture_wcferry_message_threads(
    previous_threads: frozenset[Thread],
) -> tuple[Thread, ...]:
    """Find the internal receiver thread created by WCFerry 39.5.1."""
    return tuple(
        thread
        for thread in enumerate_threads()
        if thread not in previous_threads and thread.name == "GetMessage"
    )


class _WcferryLease:
    """Persistent dirty marker; only proven native cleanup may remove it."""

    def __init__(self, path: Path, *, enabled: bool | None = None) -> None:
        self.path = path
        self.enabled = os.name == "nt" if enabled is None else enabled
        self._owned = False

    @property
    def owned(self) -> bool:
        return self._owned

    def acquire(self) -> None:
        if not self.enabled or self._owned:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": 1,
            "pid": os.getpid(),
            "command_port": WCFERRY_COMMAND_PORT,
            "created_at_unix": time.time(),
        }
        try:
            with self.path.open("x", encoding="utf-8") as file:
                json.dump(payload, file, ensure_ascii=False, indent=2)
                file.flush()
                os.fsync(file.fileno())
        except FileExistsError as exc:
            raise WcferryCleanupError(
                f"检测到未清理的 WCFerry 安全标记：{self.path}。"
                "请先完全退出并重新启动微信，确认没有其他 WCFerry 后再人工移除该文件。"
            ) from exc
        self._owned = True

    def release(self) -> None:
        if not self.enabled or not self._owned:
            return
        self.path.unlink()
        self._owned = False


def _wcferry_client_module(factory):
    module_name = getattr(factory, "__module__", "")
    if not module_name:
        return None
    try:
        return importlib.import_module(module_name)
    except ImportError:
        return None


def _construct_wcferry(factory, *, port: int, block: bool = False):
    """Turn wcferry.client's Python-level os._exit calls into catchable errors."""
    module = _wcferry_client_module(factory)
    module_os = getattr(module, "os", None) if module is not None else None
    if module_os is None:
        return factory(port=port, block=block)

    module.os = _WcferryOsProxy(module_os)
    try:
        return factory(port=port, block=block)
    finally:
        module.os = module_os


async def _await_blocking_wcf_call(operation: Callable[[], object]):
    """Let lifecycle control run without abandoning an in-flight socket call."""
    task = asyncio.create_task(asyncio.to_thread(operation))
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError:
        # A WCFerry socket must not be closed while its worker thread is still
        # using it.  Finish the bounded call, then let cancellation clean up.
        try:
            await asyncio.shield(task)
        except BaseException:
            pass
        raise


def _load_wcferry_sdk(factory):
    module = _wcferry_client_module(factory)
    module_file = getattr(module, "__file__", None) if module is not None else None
    if not module_file:
        raise WcferryCleanupError("无法定位 WCFerry sdk.dll，不能证明微信注入已清理。")
    sdk_path = Path(module_file).resolve().parent / "sdk.dll"
    if not sdk_path.is_file():
        raise WcferryCleanupError(f"WCFerry sdk.dll 不存在：{sdk_path}")
    return ctypes.cdll.LoadLibrary(str(sdk_path))


def _destroy_wcferry_sdk(*, client=None, factory=None) -> None:
    sdk = getattr(client, "sdk", None) if client is not None else None
    if sdk is None:
        if factory is None:
            raise WcferryCleanupError("缺少 WCFerry SDK 句柄，不能证明微信注入已清理。")
        sdk = _load_wcferry_sdk(factory)
    destroy = sdk.WxDestroySDK
    try:
        destroy.argtypes = []
        destroy.restype = ctypes.c_int
    except AttributeError:
        # Test doubles and alternative wrappers may expose a normal callable.
        pass
    status = int(destroy())
    if status != 0:
        raise WcferryCleanupError(
            f"WxDestroySDK 返回 {status}；为避免第二个 WCFerry，本进程禁止自动重启。"
        )


def _cleanup_wcferry_client(
    client,
    *,
    destroy_native: Callable[[], None] | None = None,
    message_threads: tuple[Thread, ...] = (),
    message_thread_capture_verified: bool = True,
    message_thread_join_timeout: float = WCFERRY_MESSAGE_THREAD_JOIN_TIMEOUT_SECONDS,
) -> None:
    """Own the detach sequence so native success is observed exactly once."""
    cleanup_error: BaseException | None = None
    # Claim cleanup before any fallible operation. WCFerry's atexit and __del__
    # callbacks both check this flag, while disable_recv_msg does not.
    if hasattr(client, "_is_running"):
        client._is_running = False

    disable = getattr(client, "disable_recv_msg", None)
    if callable(disable):
        try:
            disable()
        except BaseException as exc:
            cleanup_error = exc

    if hasattr(client, "_is_receiving_msg"):
        client._is_receiving_msg = False

    # Closing the message socket must unblock WCFerry's hidden GetMessage
    # thread. Native detach is forbidden until every captured thread is gone.
    close = getattr(getattr(client, "msg_socket", None), "close", None)
    if callable(close):
        try:
            close()
        except BaseException as exc:
            logger.warning("Failed to close WCFerry msg_socket: {}", exc)

    if not message_thread_capture_verified:
        raise WcferryCleanupError(
            "无法唯一确认 WCFerry 内部消息线程身份；拒绝执行 native destroy。"
        )
    for thread in message_threads:
        try:
            thread.join(message_thread_join_timeout)
        except BaseException as exc:
            raise WcferryCleanupError(
                "等待 WCFerry 内部消息线程退出失败；拒绝执行 native destroy。"
            ) from exc
        if thread.is_alive():
            raise WcferryCleanupError(
                "WCFerry 内部消息线程未退出；拒绝执行 native destroy。"
            )

    for attribute in ("cmd_socket",):
        close = getattr(getattr(client, attribute, None), "close", None)
        if callable(close):
            try:
                close()
            except BaseException as exc:
                logger.warning("Failed to close WCFerry {}: {}", attribute, exc)

    is_local = getattr(client, "_local_mode", None)
    if is_local is True:
        if destroy_native is None:
            _destroy_wcferry_sdk(client=client)
        else:
            destroy_native()
    elif os.name == "nt" and is_local is not False:
        raise WcferryCleanupError("无法确认 WCFerry 是否为本地注入模式。")

    if cleanup_error is not None and is_local is not True:
        raise WcferryCleanupError("WCFerry 客户端清理失败。") from cleanup_error


def _arm_native_cleanup_watchdog(
    *,
    timeout: float = WCFERRY_NATIVE_CLEANUP_TIMEOUT_SECONDS,
    timer_factory=Timer,
):
    timer = timer_factory(timeout, lambda: os._exit(NO_RESTART_EXIT_CODE))
    timer.daemon = True
    timer.start()
    return timer


class _WcferryLoguruHandler(logging.Handler):
    """Route wcferry's stdlib logs into Loguru, demoting expected retry noise.

    Why this exists: wcferry logs through ``logging.getLogger("WCF")``, which
    has no handler, so stdlib's ``lastResort`` sink writes every WARNING+ record
    straight to stderr. That bypasses ``gateway``'s ``-v`` gate entirely — with
    verbosity off, all of bubbles' own logs are suppressed and wcferry's are
    the only thing left on screen.

    Worse, two of those records are now *expected*.
    ``download_image`` logs ``下载失败`` / ``下载超时`` on every attempt that
    finds the encrypted ``.dat`` not yet written, and
    :meth:`WeChatChannel._download_image_with_retry` deliberately provokes
    exactly that — measured 12 first-attempt failures followed by 12 successes
    in one burst. Reporting a per-attempt failure as ERROR describes the
    mechanism, not the outcome; the outcome is logged by the retry helper
    itself. So records from ``download_image`` are demoted to DEBUG.

    Matching is on ``record.funcName``, not on the message text: it pins the
    demotion to the one function we wrap, leaving connection, init and video
    errors at full severity. ``下载超时`` from ``download_video`` really is an
    outcome — that path has no retry wrapper.

    The wcferry function name is prefixed onto the message rather than recovered
    by frame-walking (as ``_NioLoguruHandler`` does). Frame-walking would
    re-attribute each record to ``wcferry.client``, and ``logger.disable
    ("bubbles")`` — the ``-v`` gate — only silences records attributed to
    ``bubbles``, so it would reintroduce the leak this class exists to stop.
    Prefixing also gives the message the only context it has: wcferry's own text
    is bare enough ("下载失败") to be unattributable on its own.
    """

    #: wcferry functions whose failures are handled by a retry loop of ours.
    _RETRIED_FUNCS = frozenset({"download_image"})

    def emit(self, record: logging.LogRecord) -> None:
        try:
            if record.funcName in self._RETRIED_FUNCS:
                level = "DEBUG"
            else:
                level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno
        logger.opt(exception=record.exc_info).log(
            level, "wcferry.{}: {}", record.funcName, record.getMessage(),
        )


def _configure_wcferry_logging_bridge() -> None:
    """Bridge wcferry logs to Loguru (idempotent)."""
    wcf_logger = logging.getLogger("WCF")
    if any(isinstance(h, _WcferryLoguruHandler) for h in wcf_logger.handlers):
        return
    handler = _WcferryLoguruHandler()
    # wcferry's DEBUG records are a full hex dump of every RPC response
    # (`_send_request`), so drop them at the handler rather than let bubbles'
    # verbose mode drown in hex. Filtering here and not via `wcf_logger
    # .setLevel` covers `Logger.handle()` too, and saves nothing either way —
    # the dump is built by the caller before any level check runs.
    handler.setLevel(logging.INFO)
    wcf_logger.handlers = [handler]
    wcf_logger.propagate = False
    wcf_logger.setLevel(logging.DEBUG)


@dataclass
class WeChatContact:
    """Names attached to a wxid. Empty strings mean field is not set."""
    nickname: str = ""  # 微信昵称（profile name）
    alias: str = ""     # 微信号（user-chosen short id）
    remark: str = ""    # 备注名（this bot account's note on the contact）

    def all_names(self) -> list[str]:
        """All non-empty searchable names, in display-priority order."""
        return [n for n in (self.remark, self.nickname, self.alias) if n]

    def primary(self) -> str:
        """Best name for display: remark > nickname > alias > '' ."""
        return self.remark or self.nickname or self.alias or ""


class WeChatChannel(BaseChannel):
    """
    WeChat channel using wcferry.

    - Private chats: Direct reply
    - Group chats: Native @mentions, ``泡泡``, and newcomer notices trigger the agent
    - Supports: text, image, voice, video, file, and quoted messages
    """

    name = "wechat"

    def __init__(
        self,
        config: WeChatConfig,
        bus: MessageBus,
        session_mode: str = "channel",
        groq_api_key: str | None = None,  # kept for compatibility, not used
        wcferry_mutex: _WcferryInstanceMutex | None = None,
        wcferry_lease: _WcferryLease | None = None,
        wcferry_port_check: Callable[[], None] = _assert_wcferry_ports_idle,
        wcferry_port_release_check: Callable[[], None] = (
            _wait_for_wcferry_ports_released
        ),
        wcferry_cleanup_watchdog_factory: Callable[[], object] = (
            _arm_native_cleanup_watchdog
        ),
        wcferry_message_transport_check: Callable[[object], bool] = (
            _wcferry_message_transport_ready
        ),
        wcferry_message_thread_capture: Callable[
            [frozenset[Thread]], tuple[Thread, ...]
        ] = _capture_wcferry_message_threads,
    ):
        super().__init__(config, bus, session_mode)
        self.config: WeChatConfig = config
        self.wcf: Wcf | None = None
        self._wcferry_mutex = wcferry_mutex or _WcferryInstanceMutex()
        self._wcferry_lease = wcferry_lease or _WcferryLease(
            get_data_path() / "control" / WCFERRY_LEASE_FILE
        )
        self._wcferry_port_check = wcferry_port_check
        self._wcferry_port_release_check = wcferry_port_release_check
        self._wcferry_cleanup_watchdog_factory = wcferry_cleanup_watchdog_factory
        self._wcferry_message_transport_check = wcferry_message_transport_check
        self._wcferry_message_thread_capture = wcferry_message_thread_capture
        self._wcferry_cleanup_watchdog = None
        self._wcferry_native_destroy_lock = Lock()
        self._wcferry_native_destroy_state = "not_attempted"
        self._wcferry_receiving_enable_attempted = False
        self._wcferry_message_thread_verified = False
        self._wcferry_message_threads: tuple[Thread, ...] = ()
        self._lifecycle_lock = asyncio.Lock()
        self._stop_requested = False
        self.wxid: str = ""
        self._ready = False
        self._wechat_home: str = ""  # WeChat file storage base directory
        self._recv_thread: Thread | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._contacts: dict[str, WeChatContact] = {}  # wxid -> WeChatContact
        self._contacts_loaded_at: float = 0.0  # monotonic ts of last roster load
        self._send_lock = asyncio.Lock()
        self._outbound_image_cache_root = (
            get_data_path() / "cache" / "wechat-outbound-images"
        )
        self._outbound_image_cache = self._outbound_image_cache_root / uuid4().hex
        self._outbound_image_cache_ttl_seconds = WECHAT_IMAGE_CACHE_TTL_SECONDS
        self._outbound_image_cleanup_tasks: set[asyncio.Task[None]] = set()
        # msg.id -> downloaded image path. Quote messages carry the original
        # msg.id as <svrid>; this lets us reuse the local copy instead of
        # re-downloading via wcferry (cdn handles expire fast).
        self._image_path_by_msg_id: dict[int, str] = {}
        self._processed_join_ids: OrderedDict[tuple[str, str], None] = OrderedDict()

    async def start(self) -> None:
        """Start WeChat client and begin listening for messages."""
        if Wcf is None:
            self._ready = False
            logger.error("wcferry not installed. Install with: pip install wcferry")
            return

        construction_incomplete = False
        startup_claimed = False
        try:
            async with self._lifecycle_lock:
                if (
                    self._running
                    or self.wcf is not None
                    or self._wcferry_mutex.held
                    or self._wcferry_lease.owned
                ):
                    raise WcferryCleanupError(
                        "WeChat channel 已经持有 WCFerry；拒绝重复启动第二个实例。"
                    )
                if self._stop_requested:
                    self._running = False
                    return

                startup_claimed = True
                self._ready = False
                self._running = True
                self._loop = asyncio.get_running_loop()
                await asyncio.to_thread(
                    prune_wechat_image_cache,
                    self._outbound_image_cache_root,
                    max_age_seconds=self._outbound_image_cache_ttl_seconds,
                )
                if self._stop_requested:
                    self._running = False
                    return

                # Must precede Wcf() — its __init__ already logs through the WCF logger.
                _configure_wcferry_logging_bridge()
                self._wcferry_mutex.acquire()
                self._wcferry_port_check()
                try:
                    self._wcferry_cleanup_watchdog = (
                        self._wcferry_cleanup_watchdog_factory()
                    )
                except BaseException as exc:
                    raise WcferryCleanupError(
                        "无法为 WCFerry 构造阶段建立失败关闭 watchdog。"
                    ) from exc

                try:
                    self._wcferry_lease.acquire()
                    construction_task = asyncio.create_task(
                        asyncio.to_thread(
                            _construct_wcferry,
                            Wcf,
                            port=WCFERRY_COMMAND_PORT,
                            block=False,
                        )
                    )
                    try:
                        try:
                            self.wcf = await asyncio.shield(construction_task)
                        except asyncio.CancelledError:
                            try:
                                self.wcf = await asyncio.shield(construction_task)
                            except BaseException:
                                construction_incomplete = True
                                raise
                            raise
                    except BaseException:
                        construction_incomplete = True
                        raise
                finally:
                    self._wcferry_cleanup_watchdog.cancel()
                    self._wcferry_cleanup_watchdog = None

                while True:
                    if self._stop_requested:
                        self._running = False
                        return
                    logged_in = await _await_blocking_wcf_call(self.wcf.is_login)
                    if self._stop_requested:
                        self._running = False
                        return
                    if logged_in:
                        break
                    await asyncio.sleep(WECHAT_LOGIN_POLL_INTERVAL_SECONDS)

                self.wxid = await _await_blocking_wcf_call(self.wcf.get_self_wxid)
                if self._stop_requested:
                    self._running = False
                    return
                # Get user info including home directory for file downloads
                user_info = await _await_blocking_wcf_call(self.wcf.get_user_info)
                if self._stop_requested:
                    self._running = False
                    return
                self._wechat_home = user_info.get("home", "")
                logger.info(
                    "WeChat connected as {} (home: {})",
                    self.wxid,
                    self._wechat_home,
                )
                # Load all contacts
                await _await_blocking_wcf_call(self._load_contacts)
                if self._stop_requested:
                    self._running = False
                    return
                # Start message receiving
                threads_before = frozenset(enumerate_threads())
                self._wcferry_receiving_enable_attempted = True
                try:
                    receiving_enabled = await _await_blocking_wcf_call(
                        self.wcf.enable_receiving_msg
                    )
                finally:
                    self._wcferry_message_threads = tuple(
                        self._wcferry_message_thread_capture(threads_before)
                    )
                    self._wcferry_message_thread_verified = (
                        len(self._wcferry_message_threads) == 1
                    )
                if self._stop_requested:
                    self._running = False
                    return
                if receiving_enabled is not True:
                    raise RuntimeError("WCFerry 拒绝启动消息接收；可能存在残留注入。")
                if len(self._wcferry_message_threads) != 1:
                    raise RuntimeError(
                        "无法唯一确认 WCFerry 内部消息线程，拒绝把微信渠道标记为 ready。"
                    )
                loop = asyncio.get_running_loop()
                transport_deadline = (
                    loop.time() + WECHAT_MESSAGE_TRANSPORT_READY_TIMEOUT_SECONDS
                )
                while not (
                    self._wcferry_message_threads[0].is_alive()
                    and self._wcferry_message_transport_check(self.wcf)
                ):
                    if self._stop_requested:
                        self._running = False
                        return
                    if not self._wcferry_message_threads[0].is_alive():
                        raise RuntimeError(
                            "WCFerry 内部消息线程已退出，拒绝把微信渠道标记为 ready。"
                        )
                    if loop.time() >= transport_deadline:
                        raise RuntimeError(
                            "WCFerry 消息 socket 未建立，拒绝把微信渠道标记为 ready。"
                        )
                    await asyncio.sleep(
                        WECHAT_MESSAGE_TRANSPORT_POLL_INTERVAL_SECONDS
                    )
                self._recv_thread = Thread(target=self._recv_loop, daemon=True)
                self._ready = True
                self._recv_thread.start()
                logger.info("WeChat message receiver started")

            # Keep running until stopped
            while self._running:
                await asyncio.sleep(WECHAT_RECEIVER_HEALTH_INTERVAL_SECONDS)
                if self._running and not self.is_ready:
                    raise RuntimeError("WeChat receiver stopped unexpectedly")
        except BaseException as exc:
            if not startup_claimed:
                raise
            self._ready = False
            self._running = False
            if isinstance(exc, Exception):
                logger.error("WeChat channel failed: {}", exc)
            try:
                await self.stop()
            except BaseException as stop_exc:
                logger.opt(exception=stop_exc).error(
                    "Failed to clean up WeChat after startup error"
                )
                raise stop_exc from exc
            if construction_incomplete:
                raise WcferryCleanupError(
                    "WCFerry 构造未完整返回；即使已尝试清理，也禁止自动重启。"
                ) from exc
            raise

    def _load_contacts(self) -> None:
        """Load full Contact records from WeChat database (nickname + alias + remark)."""
        if not self.wcf:
            return
        try:
            contacts = self.wcf.query_sql(
                "MicroMsg.db",
                "SELECT UserName, NickName, Alias, Remark FROM Contact;",
            )
            self._contacts = {
                c["UserName"]: WeChatContact(
                    nickname=c.get("NickName") or "",
                    alias=c.get("Alias") or "",
                    remark=c.get("Remark") or "",
                )
                for c in contacts
            }
            self._contacts_loaded_at = time.monotonic()
            logger.info("Loaded {} contacts", len(self._contacts))
        except Exception as e:
            logger.warning("Failed to load contacts: {}", e)

    def _refresh_contacts_if_stale(self, missing_wxid: str | None = None) -> None:
        """Re-read the roster when it can't answer, or when it's simply old.

        Why: the roster used to be read once at startup and never again, so a
        contact added after launch had no name at all — the model would only see
        ``Sender ID: wxid_xxx`` — and a renamed 备注名 stayed stale until restart.
        Group 群昵称 was unaffected (queried live per message), which is why this
        mostly bit private chats and name-based lookups.

        Two triggers: an unknown wxid (refresh now, the answer may exist), and a
        TTL (catch renames of contacts we already know). Both are rate-limited so
        a burst of messages from one unknown sender can't hammer the DB.
        """
        if not self.wcf:
            return
        now = time.monotonic()
        age = now - self._contacts_loaded_at
        unknown = missing_wxid is not None and missing_wxid not in self._contacts

        if unknown:
            if age < CONTACTS_MIN_REFRESH_INTERVAL_SEC:
                return  # 已经刚查过；这个 wxid 大概确实不在名单里
        elif age < CONTACTS_TTL_SEC:
            return

        self._load_contacts()

    def _get_sender_name(self, sender_id: str, room_id: str | None = None) -> str | None:
        """Resolve display name: 群昵称 → 备注名 → 微信昵称 → 微信号."""
        if not self.wcf:
            return None

        # For group chat, try to get the per-room alias (群昵称) first
        if room_id:
            try:
                alias = self.wcf.get_alias_in_chatroom(sender_id, room_id)
                if alias and alias.strip():
                    return alias
            except Exception:
                pass

        self._refresh_contacts_if_stale(sender_id)
        contact = self._contacts.get(sender_id)
        return contact.primary() if contact else None

    def _recv_loop(self) -> None:
        """Background thread for receiving messages from wcferry."""
        try:
            while self.wcf and self.wcf.is_receiving_msg():
                try:
                    msg = self.wcf.get_msg()
                    if self._loop:
                        asyncio.run_coroutine_threadsafe(
                            self._process_msg(msg),
                            self._loop
                        )
                except Empty:
                    continue
                except Exception as e:
                    logger.error("Error receiving WeChat message: {}", e)
        finally:
            self._ready = False

    async def _process_msg(self, msg: WxMsg) -> None:
        """Process incoming WeChat message."""
        # Native notices can describe the bot inviting somebody else. They
        # still need processing even when WCFerry marks them as from_self.
        if msg.from_self() and msg.type != MSG_TYPE_SYSTEM:
            return

        is_group = msg.from_group()

        if is_group:
            # Group chat: check if in allowed groups
            if self.config.groups and msg.roomid not in self.config.groups:
                return
            chat_id = msg.roomid
            sender_id = msg.sender
            room_id = msg.roomid
            if msg.type == MSG_TYPE_SYSTEM:
                await self._handle_group_join(msg)
                return
            native_at_me = msg.is_at(self.wxid)
            should_respond = native_at_me
        else:
            # Private chat: direct reply
            chat_id = msg.sender
            sender_id = msg.sender
            room_id = None
            native_at_me = True
            should_respond = True  # Always respond to private chat

        # Get sender name (group alias or contact nickname)
        sender_name = self._get_sender_name(sender_id, room_id)

        # Process different message types
        text = ""
        media_paths: list[str] = []

        if msg.type == MSG_TYPE_TEXT:
            text = msg.content
            name_triggered = is_group and self._contains_bot_trigger(text)
            should_respond = should_respond or name_triggered
            # In groups, convert WeChat-native "@nickname " mentions to `<@wxid>`
            # markers so the model sees a uniform format. Falls back to stripping
            # @me when no aters info is available.
            if is_group:
                aters = self._parse_atuserlist(msg.xml)
                if aters:
                    text = self._convert_inbound_mentions(text, aters)
                elif native_at_me:
                    text = self._strip_at_mention(text)
                elif name_triggered:
                    # A copied ``@泡泡`` has no atuserlist. Remove only this
                    # literal prefix; do not delete unrelated mentions such as
                    # ``@张三 帮我问泡泡``.
                    text = self._strip_copied_bot_mention(text)

        elif msg.type == MSG_TYPE_IMAGE:
            file_path, content_text = await self._download_and_save_media(
                "image", msg, chat_id
            )
            if file_path:
                media_paths.append(file_path)
            text = content_text

        elif msg.type == MSG_TYPE_VOICE:
            file_path, content_text = await self._download_and_save_media(
                "voice", msg, chat_id
            )
            if file_path:
                media_paths.append(file_path)
            text = content_text

        elif msg.type == MSG_TYPE_VIDEO:
            file_path, content_text = await self._download_and_save_media(
                "video", msg, chat_id
            )
            if file_path:
                media_paths.append(file_path)
            text = content_text

        elif msg.type == MSG_TYPE_APP:
            # Handle app messages (files, links, quotes, etc.)
            text, media_paths, is_reply_to_me = await wechat_app.process_app_msg(
                self.wcf, msg, self._get_media_dir(chat_id),
                wechat_home=self._wechat_home, bot_wxid=self.wxid,
                msg_id_to_path=self._image_path_by_msg_id,
            )
            # For quoted messages in groups, check if replying to bot or @mentioned
            if is_group:
                if is_reply_to_me:
                    should_respond = True
                elif text and self._contains_bot_trigger(text):
                    should_respond = True
                if native_at_me:
                    text = self._strip_at_mention(text)

        else:
            logger.debug("Skipping unsupported message type: {}", msg.type)
            return

        if not text and not media_paths:
            return

        logger.debug("WeChat message from {} ({}) in {}: {}{}",
                     sender_name or sender_id, sender_id, chat_id,
                     (text[:50] if text else "[media]"),
                     " [triggered]" if should_respond else "")

        await self._handle_message(
            sender_id=sender_id,
            chat_id=chat_id,
            content=text,
            media=media_paths,
            metadata={
                "is_group": is_group,
                "sender_name": sender_name,
                "respond": should_respond,
                "msg_type": msg.type,
                "message_id": str(msg.id) if getattr(msg, "id", None) is not None else None,
            }
        )

    async def _handle_group_join(self, msg: WxMsg) -> None:
        """Turn a native membership event into an ordinary user prompt."""
        members = parse_group_join_members(msg.content)
        if not members or not self.is_allowed(msg.sender):
            return

        message_id = str(msg.id) if getattr(msg, "id", None) else None
        if message_id:
            key = (msg.roomid, message_id)
            if key in self._processed_join_ids:
                return
            self._processed_join_ids[key] = None
            while len(self._processed_join_ids) > 1000:
                self._processed_join_ids.popitem(last=False)

        # A native text notice contains display names, not necessarily wxids.
        # Read one fresh roster/contact snapshot off the event loop, avoiding
        # one get_alias_in_chatroom RPC per member of a potentially large group.
        joined_members = await asyncio.to_thread(
            self._resolve_join_members, msg.roomid, members
        )
        labels = [
            f"{member['name']} <@{member['wxid']}>"
            if member["wxid"] else f"{member['name']}（wxid 暂未确认）"
            for member in joined_members
        ]
        content = (
            f"【系统通知】{'、'.join(labels)} 入群了。"
            "可以 @ 新人欢迎一下。"
        )
        if any(not member["wxid"] for member in joined_members):
            content += "未提供提及标记的成员不要猜测 wxid。"

        await self._handle_message(
            sender_id=msg.sender,
            chat_id=msg.roomid,
            content=content,
            metadata={
                "is_group": True,
                "sender_name": "微信系统",
                "respond": True,
                "msg_type": MSG_TYPE_SYSTEM,
                "message_id": message_id,
                "event_type": "group_member_joined",
                "joined_members": joined_members,
            },
        )

    def _resolve_join_members(
        self, chat_id: str, names: list[str]
    ) -> list[dict[str, str | None]]:
        """Resolve exact, unique names in this group; never infer an identity."""
        roster = {}
        if self.wcf:
            try:
                roster = self.wcf.get_chatroom_members(chat_id) or {}
                self._load_contacts()
            except Exception as exc:
                logger.warning("Failed to resolve newcomers in {}: {}", chat_id, exc)

        by_name: dict[str, set[str]] = {}
        for wxid, room_name in roster.items():
            contact = self._contacts.get(wxid) or WeChatContact()
            for name in (room_name, contact.nickname, contact.remark, contact.alias):
                if name:
                    by_name.setdefault(name, set()).add(wxid)
        resolved = []
        for name in names:
            matches = by_name.get(name, set())
            resolved.append({
                "name": name,
                "wxid": next(iter(matches)) if len(matches) == 1 else None,
            })
        return resolved

    @staticmethod
    def _contains_bot_trigger(content: str | None) -> bool:
        """Whether visible message text explicitly names the bot."""
        return bool(content and WECHAT_BOT_TRIGGER_WORD in content)

    @staticmethod
    def _strip_copied_bot_mention(content: str) -> str:
        """Remove copied ``@泡泡`` text without touching other @mentions."""
        stripped = re.sub(
            rf"@{re.escape(WECHAT_BOT_TRIGGER_WORD)}"
            rf"(?=$|[\s\u2005，。！？、,:：;；])[\s\u2005]*",
            "",
            content,
        ).strip()
        # A message containing only the copied mention is still a real wake-up;
        # keeping it prevents the empty-message guard from dropping the turn.
        return stripped or content.strip()

    def _strip_at_mention(self, content: str) -> str:
        """Remove @mention from message content."""
        # WeChat @mention format: @nickname followed by space or text
        # Pattern: @xxx followed by space or end
        return re.sub(r"@\S+\s*", "", content).strip()

    @staticmethod
    def _parse_atuserlist(xml: str | None) -> list[str]:
        """Extract @-ed wxids from msg.xml's <atuserlist> element (ordered)."""
        if not xml:
            return []
        m = re.search(r"<atuserlist>(?:<!\[CDATA\[)?(.*?)(?:\]\]>)?</atuserlist>", xml, re.S)
        if not m:
            return []
        return [w for w in m.group(1).strip().strip(",").split(",") if w]

    @staticmethod
    def _convert_inbound_mentions(text: str, aters: list[str]) -> str:
        """Replace `@<token>` substrings positionally with `<@wxid>` markers.

        Order in `aters` is assumed to match the order of `@` substrings in text
        (the WeChat client emits them aligned).
        """
        if not aters or not text:
            return text
        it = iter(aters)

        def repl(_m: re.Match) -> str:
            try:
                return f"<@{next(it)}>"
            except StopIteration:
                return _m.group(0)

        return re.sub(r"@\S+", repl, text)

    def _translate_outbound_mentions(self, text: str, chat_id: str) -> tuple[str, str]:
        """Convert `<@wxid>` markers to WeChat `@nickname` format + aters CSV.

        WeChat requires the outbound text to contain N `@` substrings matching
        the N wxids in `aters`. We resolve the nickname (group alias preferred,
        falling back to contact nickname or wxid) so the visible text is friendly.
        """
        aters: list[str] = []
        room_id = chat_id if chat_id.endswith("@chatroom") else None

        def repl(wxid: str) -> str:
            aters.append(wxid)
            nickname = self._get_sender_name(wxid, room_id) or wxid
            # U+2005 is the four-per-em space WeChat itself uses after @nickname
            return f"@{nickname} "

        return replace_mentions(text, repl), ",".join(aters)

    async def _download_and_save_media(
        self,
        media_type: str,
        msg: WxMsg,
        chat_id: str,
    ) -> tuple[str | None, str]:
        """
        Download media from WeChat and save to session's data directory.

        Returns:
            (file_path, content_text) - file_path is None if download failed
        """
        # Check session binding first
        media_dir = self._get_media_dir(chat_id)
        if media_dir is None:
            return None, f"[{media_type}: 请先使用 /session <name> 绑定工作区]"

        if not self.wcf:
            return None, f"[{media_type}: WeChat not connected]"

        loop = asyncio.get_running_loop()
        file_path: str | None = None
        filename = ""

        try:
            if media_type == "image":
                logger.debug(
                    "wcf.download_image start msg_id={} extra={!r} media_dir={}",
                    msg.id, msg.extra, media_dir,
                )
                file_path = await self._download_image_with_retry(msg, media_dir)
                if file_path:
                    filename = os.path.basename(file_path)

            elif media_type == "voice":
                # Download voice (converted to MP3)
                file_path = await loop.run_in_executor(
                    None,
                    lambda: self.wcf.get_audio_msg(msg.id, str(media_dir), timeout=10)
                )
                if file_path:
                    filename = os.path.basename(file_path)

            elif media_type == "video":
                # Download video
                file_path = await loop.run_in_executor(
                    None,
                    lambda: self.wcf.download_video(msg.id, msg.thumb, str(media_dir), timeout=60)
                )
                if file_path:
                    filename = os.path.basename(file_path)

            if file_path and os.path.exists(file_path):
                logger.debug("Downloaded {} to {}", media_type, file_path)
                if media_type == "image":
                    self._image_path_by_msg_id[msg.id] = file_path
                return file_path, f"[{media_type}: <work_dir>/data/{filename}]"
            if media_type == "image" and file_path:
                logger.warning(
                    "wcf.download_image returned path but file not found: {}", file_path,
                )

        except Exception as e:
            logger.error("Error downloading {}: {}", media_type, e)

        if media_type == "image":
            logger.warning(
                "Image download gave up: msg_id={}, extra_present={}, last_return={!r}",
                msg.id, bool(msg.extra), file_path,
            )
        return None, f"[{media_type}: download failed]"

    async def _download_image_with_retry(self, msg: WxMsg, media_dir) -> str | None:
        """Decrypt an inbound image, retrying so WeChat's own write can land first.

        Why this exists: ``wcferry.download_image`` bails out before ever trying
        to decrypt when WeChat's ``download_attach`` RPC returns non-zero::

            if (not os.path.exists(extra)) and (self.download_attach(...) != 0):
                return ""          # never reaches decrypt_image

        On this deployment that RPC reports failure while WeChat *has already
        written* the encrypted ``.dat`` to disk — measured 40 of 238 attempts
        (~17%), and the file was verifiably decryptable each time. The first
        call is therefore doomed for reasons unrelated to the file's existence.

        The fix rides wcferry's own short-circuit: once ``msg.extra`` exists on
        disk, ``download_attach`` is skipped entirely and it goes straight to
        decryption. So we wait for the path to appear and call again.

        Passing ``timeout=1`` instead of 30 is deliberate — the retry loop lives
        here, where it is async. wcferry's timeout is a blocking ``sleep`` inside
        the executor thread, and a 30s budget meant one image could pin a thread
        for half a minute.
        """
        loop = asyncio.get_running_loop()
        extra = msg.extra or ""

        for attempt in range(IMAGE_DOWNLOAD_ATTEMPTS):
            file_path = await loop.run_in_executor(
                None,
                lambda: self.wcf.download_image(msg.id, extra, str(media_dir), timeout=1),
            )
            if file_path and os.path.exists(file_path):
                if attempt:
                    logger.debug("Image decrypted on attempt {} for msg_id={}", attempt + 1, msg.id)
                return file_path

            if attempt == IMAGE_DOWNLOAD_ATTEMPTS - 1:
                break
            logger.debug(
                "download_image attempt {} empty (extra_on_disk={}), retrying in {}s",
                attempt + 1, bool(extra) and os.path.exists(extra), IMAGE_DOWNLOAD_RETRY_DELAY_SEC,
            )
            await asyncio.sleep(IMAGE_DOWNLOAD_RETRY_DELAY_SEC)

        return None

    async def _call_wcf(self, method_name: str, *args) -> int:
        """Call WCFerry and normalize its process-exit failure mode."""
        if not self.wcf:
            raise RuntimeError("WeChat not connected")
        method = getattr(self.wcf, method_name)

        started = time.monotonic()
        try:
            status = method(*args)
        except SystemExit as exc:
            raise RuntimeError(f"WCFerry {method_name} aborted") from exc
        logger.debug(
            "WCFerry {} completed: status={}, elapsed={:.2f}s",
            method_name,
            status,
            time.monotonic() - started,
        )
        return status

    def _schedule_outbound_image_cleanup(self, path: str) -> None:
        async def expire() -> None:
            try:
                await asyncio.sleep(self._outbound_image_cache_ttl_seconds)
            finally:
                await asyncio.to_thread(remove_wechat_cached_image, path)

        task = asyncio.create_task(expire())
        self._outbound_image_cleanup_tasks.add(task)
        task.add_done_callback(self._outbound_image_cleanup_tasks.discard)

    async def _send_outbound_image(self, file_path: str, chat_id: str) -> None:
        prepare_task = asyncio.create_task(
            asyncio.to_thread(
                prepare_wechat_image,
                file_path,
                cache_dir=self._outbound_image_cache,
                max_bytes=self.config.outbound_image_max_bytes,
                max_edge=self.config.outbound_image_max_edge,
            )
        )
        try:
            prepared = await asyncio.shield(prepare_task)
        except asyncio.CancelledError:
            try:
                abandoned = await asyncio.shield(prepare_task)
            except Exception:
                pass
            else:
                if abandoned.derived:
                    await asyncio.to_thread(
                        remove_wechat_cached_image,
                        abandoned.path,
                    )
            raise
        if prepared.derived:
            self._schedule_outbound_image_cleanup(prepared.path)
        if prepared.send_as_file:
            status = await self._call_wcf("send_file", prepared.original_path, chat_id)
            if status != 0:
                raise RuntimeError(f"WCFerry send_file failed with status {status}")
            logger.debug("Submitted image as file to {}: {}", chat_id, file_path)
            return

        status = await self._call_wcf("send_image", prepared.path, chat_id)
        if status == 0:
            logger.debug(
                "Submitted image to {}: {} (prepared_bytes={})",
                chat_id,
                file_path,
                prepared.prepared_size_bytes,
            )
            return

        logger.warning(
            "WCFerry send_image failed with status {}; falling back to file: {}",
            status,
            file_path,
        )
        fallback_status = await self._call_wcf("send_file", prepared.original_path, chat_id)
        if fallback_status != 0:
            raise RuntimeError(
                "WCFerry image and file delivery failed with statuses "
                f"{status}/{fallback_status}"
            )

    async def send(self, msg: OutboundMessage) -> bool:
        """Send a message through WeChat, including media if present."""
        if not self.wcf:
            logger.warning("WeChat not connected")
            return False

        async with self._send_lock:
            delivered = True
            # Send media files first
            for file_path in msg.media:
                if not os.path.isfile(file_path):
                    logger.warning("Media file not found: {}", file_path)
                    delivered = False
                    continue
                ext = os.path.splitext(file_path)[1].lower()
                try:
                    if ext in IMAGE_EXTS:
                        await self._send_outbound_image(file_path, msg.chat_id)
                    else:
                        status = await self._call_wcf("send_file", file_path, msg.chat_id)
                        if status != 0:
                            raise RuntimeError(
                                f"WCFerry send_file failed with status {status}"
                            )
                        logger.debug("Submitted file to {}: {}", msg.chat_id, file_path)
                except (Exception, SystemExit) as exc:
                    logger.error("Failed to send WeChat media {}: {}", file_path, exc)
                    delivered = False

            # Send text content
            if msg.content and msg.content.strip():
                try:
                    text, aters = self._translate_outbound_mentions(
                        msg.content,
                        msg.chat_id,
                    )
                    status = await self._call_wcf("send_text", text, msg.chat_id, aters)
                    if status != 0:
                        raise RuntimeError(
                            f"WCFerry send_text failed with status {status}"
                        )
                    logger.debug(
                        "Submitted message to {}: {}... (aters={})",
                        msg.chat_id,
                        text[:50],
                        aters or "-",
                    )
                except (Exception, SystemExit) as exc:
                    logger.error("Failed to send WeChat message: {}", exc)
                    delivered = False
            return delivered

    @property
    def is_ready(self) -> bool:
        """WCFerry is ready only after its receiver thread has started."""
        structurally_ready = (
            self._ready
            and self._running
            and self.wcf is not None
            and self._recv_thread is not None
            and self._recv_thread.is_alive()
            and len(self._wcferry_message_threads) == 1
            and self._wcferry_message_threads[0].is_alive()
            and self._wcferry_message_transport_check(self.wcf)
        )
        if not structurally_ready:
            return False
        try:
            return bool(self.wcf.is_receiving_msg())
        except Exception:
            return False

    async def get_group_members(self, chat_id: str) -> list[dict[str, object]]:
        """Return member records for a WeChat group.

        Each record: ``{id: wxid, names: {label: value, ...}}`` where `names`
        contains every non-empty identifier we know about, labeled for the
        model (群昵称 / 备注名 / 微信昵称 / 微信号). The @ marker uses `id`.
        """
        if not self.wcf or not chat_id.endswith("@chatroom"):
            return []
        # 名字搜索要按备注名/昵称匹配，命中率直接取决于名单新旧。
        self._refresh_contacts_if_stale()
        loop = asyncio.get_running_loop()
        try:
            members = await loop.run_in_executor(None, self.wcf.get_chatroom_members, chat_id)
        except Exception as e:
            logger.warning("Failed to fetch chatroom members for {}: {}", chat_id, e)
            return []

        def _build(wxid: str, room_nickname: str) -> dict[str, object]:
            try:
                room_alias = self.wcf.get_alias_in_chatroom(wxid, chat_id) or ""
            except Exception:
                room_alias = ""
            contact = self._contacts.get(wxid) or WeChatContact()
            names: dict[str, str] = {}
            if room_alias:
                names["群昵称"] = room_alias
            if contact.remark:
                names["备注名"] = contact.remark
            if contact.nickname:
                names["微信昵称"] = contact.nickname
            if room_nickname and room_nickname not in names.values():
                # Distinct from above (rare, but possible if wcferry rosters drift)
                names.setdefault("成员表昵称", room_nickname)
            if contact.alias:
                names["微信号"] = contact.alias
            return {"id": wxid, "names": names}

        return [
            await loop.run_in_executor(None, _build, wxid, nickname)
            for wxid, nickname in (members or {}).items()
        ]

    async def stop(self) -> None:
        """Stop WeChat client."""
        self._ready = False
        self._running = False
        self._stop_requested = True
        async with self._lifecycle_lock:
            await self._stop_locked()

    async def _stop_locked(self) -> None:
        """Stop WCFerry while holding the lifecycle lock."""
        cleanup_safe = self.wcf is None and not self._wcferry_lease.owned
        if (
            (self.wcf is not None or self._wcferry_lease.owned)
            and self._wcferry_cleanup_watchdog is None
        ):
            try:
                self._wcferry_cleanup_watchdog = (
                    self._wcferry_cleanup_watchdog_factory()
                )
            except BaseException as exc:
                logger.opt(exception=exc).error(
                    "Unable to arm native WCFerry cleanup watchdog"
                )
        try:
            if self.wcf:
                client = self.wcf
                _cleanup_wcferry_client(
                    client,
                    destroy_native=lambda: self._destroy_wcferry_once(client=client),
                    message_threads=self._wcferry_message_threads,
                    message_thread_capture_verified=(
                        not self._wcferry_receiving_enable_attempted
                        or self._wcferry_message_thread_verified
                    ),
                )
                self._wcferry_port_release_check()
                cleanup_safe = True
                logger.info("WCFerry detached from WeChat safely")
                self.wcf = None
            elif self._wcferry_lease.owned:
                self._destroy_wcferry_once(factory=Wcf)
                self._wcferry_port_release_check()
                cleanup_safe = True

            cleanup_tasks = list(self._outbound_image_cleanup_tasks)
            for task in cleanup_tasks:
                task.cancel()
            if cleanup_tasks:
                await asyncio.gather(*cleanup_tasks, return_exceptions=True)
            try:
                await asyncio.to_thread(
                    prune_wechat_image_cache,
                    self._outbound_image_cache,
                    max_age_seconds=0,
                )
                try:
                    self._outbound_image_cache.rmdir()
                except OSError:
                    pass
            except BaseException as exc:
                logger.opt(exception=exc).warning(
                    "Failed to clean WeChat outbound image cache"
                )
        finally:
            self._recv_thread = None
            if cleanup_safe:
                try:
                    if self.wcf is not None:
                        raise WcferryCleanupError(
                            "WCFerry cleanup state changed before lease release."
                        )
                    self._wcferry_lease.release()
                    if self.wcf is not None or self._wcferry_lease.owned:
                        raise WcferryCleanupError(
                            "WCFerry lease could not be proven clear."
                        )
                    self._wcferry_mutex.release()
                finally:
                    if self._wcferry_cleanup_watchdog is not None:
                        self._wcferry_cleanup_watchdog.cancel()
                        self._wcferry_cleanup_watchdog = None

    def _destroy_wcferry_once(self, *, client=None, factory=None) -> None:
        """Attempt native detach at most once for this channel instance."""
        with self._wcferry_native_destroy_lock:
            if self._wcferry_native_destroy_state == "succeeded":
                return
            if self._wcferry_native_destroy_state == "in_progress":
                raise WcferryCleanupError(
                    "本进程正在执行 WxDestroySDK；为避免并发重复卸载，拒绝再次调用。"
                )
            if self._wcferry_native_destroy_state == "failed":
                raise WcferryCleanupError(
                    "本进程此前的 WxDestroySDK 已失败；为避免重复卸载，拒绝再次调用。"
                )
            self._wcferry_native_destroy_state = "in_progress"
        try:
            _destroy_wcferry_sdk(client=client, factory=factory)
        except BaseException:
            with self._wcferry_native_destroy_lock:
                self._wcferry_native_destroy_state = "failed"
            raise
        with self._wcferry_native_destroy_lock:
            self._wcferry_native_destroy_state = "succeeded"
