"""Tests that the refactored file/exec tools correctly delegate to a Sandbox
and preserve their user-facing output contract (line numbers, image handling,
exec formatting, error messages).
"""

import base64
import sys
from pathlib import Path

import pytest

from bubbles.agent.tools.filesystem import (
    EditFileTool,
    ListDirTool,
    ReadFileTool,
    WriteFileTool,
)
from bubbles.agent.tools.shell import ExecTool
from bubbles.sandbox.local import LocalSandbox


def _sb(tmp_path: Path) -> LocalSandbox:
    return LocalSandbox("s", tmp_path)


async def test_read_file_line_numbers(tmp_path: Path) -> None:
    (tmp_path / "f.txt").write_text("alpha\nbeta\ngamma\n")
    tool = ReadFileTool()
    tool.set_sandbox(_sb(tmp_path))
    out = await tool.execute("f.txt")
    assert "1|alpha" in out and "3|gamma" in out


async def test_read_file_range(tmp_path: Path) -> None:
    (tmp_path / "f.txt").write_text("a\nb\nc\nd\n")
    tool = ReadFileTool()
    tool.set_sandbox(_sb(tmp_path))
    out = await tool.execute("f.txt", start_line=2, end_line=3)
    assert "2|b" in out and "3|c" in out and "1|a" not in out


async def test_read_missing_and_dir(tmp_path: Path) -> None:
    tool = ReadFileTool()
    tool.set_sandbox(_sb(tmp_path))
    assert "File not found" in await tool.execute("nope.txt")
    (tmp_path / "d").mkdir()
    assert "Not a file" in await tool.execute("d")


async def test_read_image_returns_image_url(tmp_path: Path) -> None:
    # 1x1 transparent PNG
    png = base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
    )
    (tmp_path / "p.png").write_bytes(png)
    tool = ReadFileTool()
    tool.set_sandbox(_sb(tmp_path))
    out = await tool.execute("p.png")
    assert isinstance(out, list)
    assert out[0]["type"] == "image_url"
    assert out[0]["image_url"]["url"].startswith("data:image/png;base64,")


async def test_write_reports_bytes(tmp_path: Path) -> None:
    tool = WriteFileTool()
    tool.set_sandbox(_sb(tmp_path))
    out = await tool.execute("sub/x.txt", "héllo")
    assert "wrote" in out and "x.txt" in out
    assert (tmp_path / "sub" / "x.txt").read_text() == "héllo"


async def test_edit_replace_modes(tmp_path: Path) -> None:
    (tmp_path / "f.txt").write_text("x x x")
    tool = EditFileTool()
    tool.set_sandbox(_sb(tmp_path))
    out = await tool.execute("f.txt", "x", "y", "FIRST")
    assert "replaced 1" in out
    assert (tmp_path / "f.txt").read_text() == "y x x"
    await tool.execute("f.txt", "x", "z", "ALL")
    assert (tmp_path / "f.txt").read_text() == "y z z"


async def test_edit_not_found_message(tmp_path: Path) -> None:
    (tmp_path / "f.txt").write_text("hello world")
    tool = EditFileTool()
    tool.set_sandbox(_sb(tmp_path))
    out = await tool.execute("f.txt", "nonexistent", "x", "ALL")
    assert "old_text not found" in out


async def test_list_dir_output(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("1")
    (tmp_path / "sub").mkdir()
    tool = ListDirTool()
    tool.set_sandbox(_sb(tmp_path))
    out = await tool.execute(".")
    assert "📄 a.txt" in out and "📁 sub" in out


async def test_tools_error_without_sandbox() -> None:
    tool = ReadFileTool()
    assert "no sandbox bound" in await tool.execute("x")


async def test_path_escape_rejected_through_tool(tmp_path: Path) -> None:
    root = tmp_path / "sessions" / "s"
    root.mkdir(parents=True)
    (tmp_path / "secret").write_text("top")
    tool = ReadFileTool()
    tool.set_sandbox(LocalSandbox("s", root))
    out = await tool.execute("../../secret")
    assert out.startswith("Error")


# ---- exec tool ----

async def test_exec_tool_formats_output(tmp_path: Path) -> None:
    tool = ExecTool(timeout=10)
    tool.set_sandbox(_sb(tmp_path))
    out = await tool.execute("echo hi")
    assert "hi" in out


async def test_exec_tool_exit_code_shown(tmp_path: Path) -> None:
    if sys.platform == "win32":
        pytest.skip("POSIX shell")
    tool = ExecTool(timeout=10)
    tool.set_sandbox(_sb(tmp_path))
    out = await tool.execute("exit 3")
    assert "Exit code: 3" in out


async def test_exec_tool_blocks_dangerous(tmp_path: Path) -> None:
    tool = ExecTool(timeout=10)
    tool.set_sandbox(_sb(tmp_path))
    out = await tool.execute("rm -rf /")
    assert "blocked by safety guard" in out


async def test_exec_tool_blocks_working_dir_escape(tmp_path: Path) -> None:
    root = tmp_path / "s"
    root.mkdir()
    tool = ExecTool(timeout=10)
    tool.set_sandbox(LocalSandbox("s", root))
    out = await tool.execute("echo hi", working_dir="../..")
    assert "outside session directory" in out


async def test_exec_tool_no_sandbox() -> None:
    tool = ExecTool(timeout=10)
    assert "no sandbox bound" in await tool.execute("echo hi")
