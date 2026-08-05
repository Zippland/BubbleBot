"""Tests for the wcferry -> Loguru logging bridge.

wcferry logs through stdlib ``logging.getLogger("WCF")``, which has no handler
of its own. Two consequences the bridge exists to fix:

1. stdlib's ``lastResort`` sink wrote every WARNING+ record straight to stderr,
   bypassing `bubbles gateway`'s ``-v`` gate — with verbosity off, bubbles' own
   logs were suppressed while wcferry's kept printing.
2. ``download_image`` logs ``下载失败`` / ``下载超时`` on *every* attempt whose
   encrypted ``.dat`` has not landed yet, and `_download_image_with_retry`
   deliberately provokes exactly that. Those records describe the retry
   mechanism, not the outcome, so they must not surface as ERROR.
"""

from __future__ import annotations

import io
import logging
import sys

import pytest
from loguru import logger

from bubbles.channels.wechat import (
    _configure_wcferry_logging_bridge,
    _WcferryLoguruHandler,
)


# (funcName, message, stdlib level) mirroring the real wcferry call sites.
WCF_CALL_SITES = [
    ("download_image", "下载超时", logging.ERROR),
    ("download_image", "下载失败", logging.ERROR),
    ("download_video", "下载超时", logging.ERROR),
    ("__init__", "连接失败: nope", logging.ERROR),
    ("_send_request", "deadbeef" * 20, logging.DEBUG),
]


def _emit_like_wcferry(wcf_logger: logging.Logger) -> None:
    """Emit one record per real call site, with funcName spoofed to match."""
    for func, msg, level in WCF_CALL_SITES:
        wcf_logger.handle(
            wcf_logger.makeRecord("WCF", level, "client.py", 1, msg, (), None, func)
        )


@pytest.fixture
def captured() -> list[dict]:
    """Install a capturing Loguru sink plus the bridge; restore both after."""
    wcf_logger = logging.getLogger("WCF")
    saved_handlers, saved_level, saved_propagate = (
        list(wcf_logger.handlers),
        wcf_logger.level,
        wcf_logger.propagate,
    )
    records: list[dict] = []
    logger.remove()
    logger.add(lambda m: records.append(m.record), level="DEBUG")
    logger.enable("bubbles")
    _configure_wcferry_logging_bridge()
    try:
        yield records
    finally:
        logger.enable("bubbles")
        logger.remove()
        wcf_logger.handlers = saved_handlers
        wcf_logger.setLevel(saved_level)
        wcf_logger.propagate = saved_propagate


def test_bridge_is_idempotent_and_stops_propagation(captured: list[dict]) -> None:
    """Re-configuring must not stack handlers, and root must never see records.

    Propagation is what reached ``lastResort``; leaving it on would keep the
    raw-stderr writes even with the bridge installed.
    """
    wcf_logger = logging.getLogger("WCF")
    before = list(wcf_logger.handlers)

    _configure_wcferry_logging_bridge()

    assert wcf_logger.handlers == before
    assert sum(isinstance(h, _WcferryLoguruHandler) for h in before) == 1
    assert wcf_logger.propagate is False


def test_retried_download_image_failures_are_demoted(captured: list[dict]) -> None:
    """`download_image` failures are an expected step of our retry loop."""
    _emit_like_wcferry(logging.getLogger("WCF"))

    demoted = [r for r in captured if "wcferry.download_image:" in r["message"]]
    assert len(demoted) == 2
    assert {r["level"].name for r in demoted} == {"DEBUG"}
    assert {r["message"] for r in demoted} == {
        "wcferry.download_image: 下载超时",
        "wcferry.download_image: 下载失败",
    }


def test_unwrapped_failures_keep_their_severity(captured: list[dict]) -> None:
    """Demotion keys off funcName, so video/connection errors stay ERROR.

    Matching on the message text instead would have silenced ``下载超时`` from
    ``download_video`` too — that path has no retry wrapper, so a failure there
    really is the outcome.
    """
    _emit_like_wcferry(logging.getLogger("WCF"))

    kept = {
        (r["message"], r["level"].name)
        for r in captured
        if "wcferry.download_image:" not in r["message"]
    }
    assert ("wcferry.download_video: 下载超时", "ERROR") in kept
    assert ("wcferry.__init__: 连接失败: nope", "ERROR") in kept


def test_records_stay_attributed_to_bubbles(captured: list[dict]) -> None:
    """Records must keep bubbles' module name for the ``-v`` gate to work.

    ``logger.disable("bubbles")`` filters on the recorded module, so recovering
    wcferry's own frame (as the matrix-nio bridge does) would make these records
    unsilenceable — the exact leak this bridge fixes. Hence the funcName goes in
    the message text instead.
    """
    _emit_like_wcferry(logging.getLogger("WCF"))

    assert captured
    assert {r["name"] for r in captured} == {"bubbles.channels.wechat"}


def test_rpc_hex_dumps_are_dropped(captured: list[dict]) -> None:
    """wcferry DEBUG-logs a hex dump of every RPC response; keep it out."""
    _emit_like_wcferry(logging.getLogger("WCF"))

    assert not any("deadbeef" in r["message"] for r in captured)


def test_bridge_respects_the_verbose_gate(captured: list[dict]) -> None:
    """With `logger.disable("bubbles")` — i.e. gateway without -v — silence."""
    logger.disable("bubbles")

    _emit_like_wcferry(logging.getLogger("WCF"))

    assert captured == []


def test_nothing_reaches_raw_stderr(captured: list[dict]) -> None:
    """The original symptom: records printed to stderr regardless of config."""
    err = io.StringIO()
    real_stderr, sys.stderr = sys.stderr, err
    try:
        _emit_like_wcferry(logging.getLogger("WCF"))
    finally:
        sys.stderr = real_stderr

    assert err.getvalue() == ""
