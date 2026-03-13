"""Task tools for task management within a session (Claude Code style)."""

from __future__ import annotations

import json
from typing import Any, Callable, Awaitable, TYPE_CHECKING

from bubbles.agent.tools.base import Tool

if TYPE_CHECKING:
    from bubbles.session.manager import Session


class TaskListTool(Tool):
    """列出所有任务。"""

    def __init__(self):
        self._session: "Session | None" = None

    def set_session(self, session: "Session") -> None:
        self._session = session

    @property
    def name(self) -> str:
        return "task_list"

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {},
            "required": [],
        }

    async def execute(self, **kwargs: Any) -> str:
        if not self._session:
            return "Error: Session not initialized."
        tasks = self._session.get_tasks()
        if not tasks:
            return "No tasks."
        # 返回摘要视图
        lines = []
        for t in tasks:
            blocked = t.get("blocked_by", [])
            # 过滤掉已完成的 blocker
            open_blockers = [
                b for b in blocked
                if self._session.get_task(b) and self._session.get_task(b).get("status") != "completed"
            ]
            status_icon = {"pending": "○", "in_progress": "◐", "completed": "●"}.get(t["status"], "?")
            line = f"{status_icon} [{t['id']}] {t['subject']} ({t['status']})"
            if open_blockers:
                line += f" [blocked by: {', '.join(open_blockers)}]"
            lines.append(line)
        return "\n".join(lines)


class TaskGetTool(Tool):
    """获取单个任务详情。"""

    def __init__(self):
        self._session: "Session | None" = None

    def set_session(self, session: "Session") -> None:
        self._session = session

    @property
    def name(self) -> str:
        return "task_get"

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "task_id": {
                    "type": "string",
                    "description": "The ID of the task to retrieve",
                },
            },
            "required": ["task_id"],
        }

    async def execute(self, task_id: str, **kwargs: Any) -> str:
        if not self._session:
            return "Error: Session not initialized."
        task = self._session.get_task(task_id)
        if not task:
            return f"Error: Task {task_id} not found."
        return json.dumps(task, ensure_ascii=False, indent=2)


class TaskCreateTool(Tool):
    """创建新任务。"""

    def __init__(self):
        self._session: "Session | None" = None
        self._on_progress: Callable[[str, dict[str, Any]], Awaitable[None]] | None = None

    def set_session(self, session: "Session") -> None:
        self._session = session

    def set_on_progress(self, on_progress: Callable[[str, dict[str, Any]], Awaitable[None]]) -> None:
        self._on_progress = on_progress

    @property
    def name(self) -> str:
        return "task_create"

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "subject": {
                    "type": "string",
                    "description": "A brief title for the task (imperative form, e.g., 'Fix authentication bug')",
                },
                "description": {
                    "type": "string",
                    "description": "A detailed description of what needs to be done",
                },
                "active_form": {
                    "type": "string",
                    "description": "Present continuous form shown when in_progress (e.g., 'Fixing authentication bug')",
                },
            },
            "required": ["subject", "description"],
        }

    async def execute(
        self,
        subject: str,
        description: str,
        active_form: str | None = None,
        **kwargs: Any,
    ) -> str:
        if not self._session:
            return "Error: Session not initialized."
        task = self._session.create_task(
            subject=subject,
            description=description,
            active_form=active_form,
        )
        await self._emit_update()
        return f"Created task [{task['id']}]: {task['subject']}"

    async def _emit_update(self) -> None:
        if self._on_progress and self._session:
            tasks = self._session.get_tasks()
            await self._on_progress("task.updated", {"tasks": tasks})


class TaskUpdateTool(Tool):
    """更新任务状态或内容。"""

    def __init__(self):
        self._session: "Session | None" = None
        self._on_progress: Callable[[str, dict[str, Any]], Awaitable[None]] | None = None

    def set_session(self, session: "Session") -> None:
        self._session = session

    def set_on_progress(self, on_progress: Callable[[str, dict[str, Any]], Awaitable[None]]) -> None:
        self._on_progress = on_progress

    @property
    def name(self) -> str:
        return "task_update"

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "task_id": {
                    "type": "string",
                    "description": "The ID of the task to update",
                },
                "status": {
                    "type": "string",
                    "enum": ["pending", "in_progress", "completed", "deleted"],
                    "description": "New status for the task",
                },
                "subject": {
                    "type": "string",
                    "description": "New subject for the task",
                },
                "description": {
                    "type": "string",
                    "description": "New description for the task",
                },
                "active_form": {
                    "type": "string",
                    "description": "Present continuous form shown when in_progress",
                },
                "add_blocks": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Task IDs that this task blocks",
                },
                "add_blocked_by": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Task IDs that block this task",
                },
            },
            "required": ["task_id"],
        }

    async def execute(
        self,
        task_id: str,
        status: str | None = None,
        subject: str | None = None,
        description: str | None = None,
        active_form: str | None = None,
        add_blocks: list[str] | None = None,
        add_blocked_by: list[str] | None = None,
        **kwargs: Any,
    ) -> str:
        if not self._session:
            return "Error: Session not initialized."

        task = self._session.update_task(
            task_id=task_id,
            status=status,
            subject=subject,
            description=description,
            active_form=active_form,
            add_blocks=add_blocks,
            add_blocked_by=add_blocked_by,
        )

        if not task:
            return f"Error: Task {task_id} not found."

        await self._emit_update()

        if status == "deleted":
            return f"Deleted task [{task_id}]"
        return f"Updated task [{task_id}]: {task['subject']} ({task['status']})"

    async def _emit_update(self) -> None:
        if self._on_progress and self._session:
            tasks = self._session.get_tasks()
            await self._on_progress("task.updated", {"tasks": tasks})
