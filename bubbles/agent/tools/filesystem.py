"""File system tools: read, write, edit, list_dir."""

import difflib
from pathlib import Path
from typing import Any

from bubbles.agent.tools.base import Tool


def _resolve_path(path: str, base_dir: Path | None = None, restrict_to_base: bool = False) -> Path:
    """Resolve path against base_dir and optionally enforce directory restriction.

    Path handling:
    - `~` refers to base_dir (NOT the system $HOME)
    - Relative paths are resolved relative to base_dir
    - If restrict_to_base is True, paths outside base_dir will raise PermissionError
    """
    raw = (path or "").strip()
    if not raw:
        raise ValueError("empty path")

    # Handle ~ as base_dir (not system $HOME)
    if raw.startswith("~"):
        if base_dir:
            raw = str(base_dir / raw[1:].lstrip("/\\"))
        else:
            raw = raw[1:].lstrip("/\\") or "."

    p = Path(raw)
    if not p.is_absolute() and base_dir:
        p = base_dir / p

    resolved = p.resolve()

    if restrict_to_base and base_dir:
        base = base_dir.resolve()
        try:
            resolved.relative_to(base)
        except ValueError:
            raise PermissionError(f"Path '{path}' is outside allowed directory")

    return resolved


def _with_line_numbers(content: str, *, start_line: int = 1) -> str:
    """Format content with line numbers (cat -n style)."""
    lines = content.splitlines()
    if not lines:
        return ""
    width = len(str(start_line + len(lines) - 1))
    return "\n".join(f"{start_line + i:>{width}}|{line}" for i, line in enumerate(lines))


class ReadFileTool(Tool):
    """Tool to read file contents. Automatically handles images and text files."""

    _IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp", ".ico", ".tiff", ".tif"}

    def __init__(self):
        self._session_dir: Path | None = None

    def set_session_dir(self, session_dir: Path | None) -> None:
        """Set session directory for file operations."""
        self._session_dir = session_dir

    @property
    def _base_dir(self) -> Path | None:
        """Get base directory (session dir)."""
        return self._session_dir

    @property
    def _restrict(self) -> bool:
        """Whether to restrict paths to session directory."""
        return self._session_dir is not None

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

        try:
            file_path = _resolve_path(path, self._base_dir, restrict_to_base=self._restrict)
            if not file_path.exists():
                return f"Error: File not found: {path}"
            if not file_path.is_file():
                return f"Error: Not a file: {path}"

            ext = file_path.suffix.lower()

            # Handle image files - return as image_url for model to see
            if ext in self._IMAGE_EXTENSIONS:
                image_data = file_path.read_bytes()
                b64 = base64.b64encode(image_data).decode()
                mime, _ = mimetypes.guess_type(str(file_path))
                if not mime:
                    mime = "image/jpeg"
                return [
                    {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
                    {"type": "text", "text": f"[Image: {path}]"},
                ]

            # Handle text files
            content = file_path.read_text(encoding="utf-8")
            lines = content.splitlines()

            # Apply line range if specified
            actual_start = 1
            if start_line is not None or end_line is not None:
                start_idx = (start_line or 1) - 1
                end_idx = end_line or len(lines)
                lines = lines[start_idx:end_idx]
                actual_start = (start_line or 1)

            content = "\n".join(lines)
            if len(content) >= 100000:
                return "File content is too large. Please use start_line/end_line to read in chunks."

            return _with_line_numbers(content, start_line=actual_start)
        except PermissionError as e:
            return f"Error: {e}"
        except Exception as e:
            return f"Error reading file: {str(e)}"


class WriteFileTool(Tool):
    """Tool to write content to a file."""

    def __init__(self):
        self._session_dir: Path | None = None

    def set_session_dir(self, session_dir: Path | None) -> None:
        """Set session directory for file operations."""
        self._session_dir = session_dir

    @property
    def _base_dir(self) -> Path | None:
        """Get base directory (session dir)."""
        return self._session_dir

    @property
    def _restrict(self) -> bool:
        """Whether to restrict paths to session directory."""
        return self._session_dir is not None

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
        try:
            file_path = _resolve_path(path, self._base_dir, restrict_to_base=self._restrict)
            file_path.parent.mkdir(parents=True, exist_ok=True)
            bytes_written = len(content.encode("utf-8", errors="replace"))
            file_path.write_text(content, encoding="utf-8")
            return f"Successfully wrote {bytes_written} bytes to {file_path}"
        except PermissionError as e:
            return f"Error: {e}"
        except Exception as e:
            return f"Error writing file: {str(e)}"


class EditFileTool(Tool):
    """Tool to edit a file by replacing text."""

    def __init__(self):
        self._session_dir: Path | None = None

    def set_session_dir(self, session_dir: Path | None) -> None:
        """Set session directory for file operations."""
        self._session_dir = session_dir

    @property
    def _base_dir(self) -> Path | None:
        """Get base directory (session dir)."""
        return self._session_dir

    @property
    def _restrict(self) -> bool:
        """Whether to restrict paths to session directory."""
        return self._session_dir is not None

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
        try:
            file_path = _resolve_path(path, self._base_dir, restrict_to_base=self._restrict)
            if not file_path.exists():
                return f"Error: File not found: {path}"

            content = file_path.read_text(encoding="utf-8")

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

            file_path.write_text(new_content, encoding="utf-8")
            return f"Successfully edited {file_path} (replaced {replaced} occurrence(s))"
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


class ListDirTool(Tool):
    """Tool to list directory contents."""

    def __init__(self):
        self._session_dir: Path | None = None

    def set_session_dir(self, session_dir: Path | None) -> None:
        """Set session directory for file operations."""
        self._session_dir = session_dir

    @property
    def _base_dir(self) -> Path | None:
        """Get base directory (session dir)."""
        return self._session_dir

    @property
    def _restrict(self) -> bool:
        """Whether to restrict paths to session directory."""
        return self._session_dir is not None

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
        try:
            dir_path = _resolve_path(path, self._base_dir, restrict_to_base=self._restrict)
            if not dir_path.exists():
                return f"Error: Directory not found: {path}"
            if not dir_path.is_dir():
                return f"Error: Not a directory: {path}"

            items = []
            for item in sorted(dir_path.iterdir()):
                prefix = "📁 " if item.is_dir() else "📄 "
                items.append(f"{prefix}{item.name}")

            if not items:
                return f"Directory {path} is empty"

            return "\n".join(items)
        except PermissionError as e:
            return f"Error: {e}"
        except Exception as e:
            return f"Error listing directory: {str(e)}"
