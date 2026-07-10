"""File system tools: read, write, edit, list_dir.

These tools own *presentation* (line numbers, image base64, diff-matching, size
limits) but delegate all raw I/O to the session's ``Sandbox``. The sandbox
decides where bytes actually live (host disk, container, remote) and enforces
path containment. See ``bubbles/sandbox/``.
"""

import difflib
from typing import Any

from bubbles.agent.tools.base import Tool
from bubbles.sandbox.base import Sandbox


def _with_line_numbers(content: str, *, start_line: int = 1) -> str:
    """Format content with line numbers (cat -n style)."""
    lines = content.splitlines()
    if not lines:
        return ""
    width = len(str(start_line + len(lines) - 1))
    return "\n".join(f"{start_line + i:>{width}}|{line}" for i, line in enumerate(lines))


class _SandboxFileTool(Tool):
    """Base for file tools: holds the per-turn sandbox handle."""

    def __init__(self):
        self._sandbox: Sandbox | None = None

    def set_sandbox(self, sandbox: Sandbox | None) -> None:
        """Set the sandbox for file operations (called per turn)."""
        self._sandbox = sandbox


class ReadFileTool(_SandboxFileTool):
    """Tool to read file contents. Automatically handles images and text files."""

    _IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp", ".ico", ".tiff", ".tif"}

    @property
    def name(self) -> str:
        return "read_file"

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "The file path to read (supports text files and images)"},
                "start_line": {"type": "integer", "description": "Start line number (1-based), for text files only"},
                "end_line": {"type": "integer", "description": "End line number (1-based), for text files only"},
            },
            "required": ["path"],
        }

    async def execute(
        self, path: str, start_line: int | None = None, end_line: int | None = None, **kwargs: Any
    ) -> str | list[dict[str, Any]]:
        import base64
        import mimetypes
        from pathlib import PurePosixPath, PureWindowsPath

        if self._sandbox is None:
            return "Error: no sandbox bound"

        try:
            st = await self._sandbox.stat(path)
            if st is None:
                return f"Error: File not found: {path}"
            if st.is_dir:
                return f"Error: Not a file: {path}"

            # Determine extension without touching the host FS.
            name = PureWindowsPath(path).name if "\\" in path else PurePosixPath(path).name
            ext = ("." + name.rsplit(".", 1)[1].lower()) if "." in name else ""

            # Handle image files - return as image_url for model to see
            if ext in self._IMAGE_EXTENSIONS:
                image_data = await self._sandbox.read_bytes(path)
                b64 = base64.b64encode(image_data).decode()
                mime, _ = mimetypes.guess_type(name)
                if not mime:
                    mime = "image/jpeg"
                return [
                    {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
                    {"type": "text", "text": f"[Image: {path}]"},
                ]

            # Handle text files
            content = (await self._sandbox.read_bytes(path)).decode("utf-8", errors="replace")
            all_lines = content.splitlines()
            total_lines = len(all_lines)

            # Apply line range
            actual_start = start_line or 1
            actual_end = end_line or total_lines
            start_idx = actual_start - 1
            end_idx = min(actual_end, total_lines)
            selected_lines = all_lines[start_idx:end_idx]

            # Check size limit, truncate if needed
            max_chars = 100000
            result_lines = []
            char_count = 0
            for line in selected_lines:
                line_len = len(line) + 1  # +1 for newline
                if char_count + line_len > max_chars:
                    break
                result_lines.append(line)
                char_count += line_len

            content = _with_line_numbers("\n".join(result_lines), start_line=actual_start)
            if len(result_lines) < len(selected_lines):
                content += f"\n\n[Truncated. File has {total_lines} lines total]"
            return content
        except PermissionError as e:
            return f"Error: {e}"
        except Exception as e:
            return f"Error reading file: {str(e)}"


class WriteFileTool(_SandboxFileTool):
    """Tool to write content to a file."""

    @property
    def name(self) -> str:
        return "write_file"

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "The file path to write to"},
                "content": {"type": "string", "description": "The content to write"},
            },
            "required": ["path", "content"],
        }

    async def execute(self, path: str, content: str, **kwargs: Any) -> str:
        if self._sandbox is None:
            return "Error: no sandbox bound"
        try:
            data = content.encode("utf-8", errors="replace")
            await self._sandbox.write_bytes(path, data)
            return f"Successfully wrote {len(data)} bytes to {path}"
        except PermissionError as e:
            return f"Error: {e}"
        except Exception as e:
            return f"Error writing file: {str(e)}"


class EditFileTool(_SandboxFileTool):
    """Tool to edit a file by replacing text."""

    @property
    def name(self) -> str:
        return "edit_file"

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "The file path to edit"},
                "old_text": {"type": "string", "description": "The exact text to find and replace"},
                "new_text": {"type": "string", "description": "The text to replace with"},
                "replace_mode": {
                    "type": "string",
                    "description": "How to handle occurrences: ALL, FIRST, LAST.",
                    "enum": ["ALL", "FIRST", "LAST"],
                },
            },
            "required": ["path", "old_text", "new_text", "replace_mode"],
        }

    async def execute(
        self, path: str, old_text: str, new_text: str, replace_mode: str, **kwargs: Any
    ) -> str:
        if self._sandbox is None:
            return "Error: no sandbox bound"
        try:
            if not await self._sandbox.exists(path):
                return f"Error: File not found: {path}"

            content = (await self._sandbox.read_bytes(path)).decode("utf-8", errors="replace")

            if old_text not in content:
                return self._not_found_message(old_text, content, path)

            count = content.count(old_text)

            if replace_mode == "ALL":
                new_content = content.replace(old_text, new_text)
                replaced = count
            elif replace_mode == "FIRST":
                new_content = content.replace(old_text, new_text, 1)
                replaced = 1
            elif replace_mode == "LAST":
                # Replace last occurrence
                idx = content.rfind(old_text)
                new_content = content[:idx] + new_text + content[idx + len(old_text):]
                replaced = 1
            else:
                return f"Error: Invalid replace_mode '{replace_mode}'. Use ALL, FIRST, or LAST."

            await self._sandbox.write_bytes(path, new_content.encode("utf-8", errors="replace"))
            return f"Successfully edited {path} (replaced {replaced} occurrence(s))"
        except PermissionError as e:
            return f"Error: {e}"
        except Exception as e:
            return f"Error editing file: {str(e)}"

    @staticmethod
    def _not_found_message(old_text: str, content: str, path: str) -> str:
        """Build a helpful error when old_text is not found."""
        lines = content.splitlines(keepends=True)
        old_lines = old_text.splitlines(keepends=True)
        window = len(old_lines)

        best_ratio, best_start = 0.0, 0
        for i in range(max(1, len(lines) - window + 1)):
            ratio = difflib.SequenceMatcher(None, old_lines, lines[i : i + window]).ratio()
            if ratio > best_ratio:
                best_ratio, best_start = ratio, i

        if best_ratio > 0.5:
            diff = "\n".join(
                difflib.unified_diff(
                    old_lines,
                    lines[best_start : best_start + window],
                    fromfile="old_text (provided)",
                    tofile=f"{path} (actual, line {best_start + 1})",
                    lineterm="",
                )
            )
            return f"Error: old_text not found in {path}.\nBest match ({best_ratio:.0%} similar) at line {best_start + 1}:\n{diff}"
        return f"Error: old_text not found in {path}. No similar text found. Verify the file content."


class ListDirTool(_SandboxFileTool):
    """Tool to list directory contents."""

    @property
    def name(self) -> str:
        return "list_dir"

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "The directory path to list"},
            },
            "required": ["path"],
        }

    async def execute(self, path: str, **kwargs: Any) -> str:
        if self._sandbox is None:
            return "Error: no sandbox bound"
        try:
            st = await self._sandbox.stat(path)
            if st is None:
                return f"Error: Directory not found: {path}"
            if not st.is_dir:
                return f"Error: Not a directory: {path}"

            entries = await self._sandbox.list_dir(path)
            items = [f"{'📁 ' if e.is_dir else '📄 '}{e.name}" for e in entries]

            if not items:
                return f"Directory {path} is empty"

            return "\n".join(items)
        except PermissionError as e:
            return f"Error: {e}"
        except Exception as e:
            return f"Error listing directory: {str(e)}"
