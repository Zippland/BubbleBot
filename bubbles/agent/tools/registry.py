"""Tool registry for dynamic tool management."""

from typing import Any

from loguru import logger

from bubbles.agent.tools.base import Tool

# 单条工具结果的 token 预算。compaction 的保留窗口是 40k token，单条结果超过
# 窗口就没有任何压缩策略能守住上限——read_file 上限 100k 字符（中文约 200k
# token）、MCP 工具则完全没有上限。在出口统一截断，让预算成为闭环。
MAX_RESULT_TOKENS = 8_000


class ToolRegistry:
    """
    Registry for agent tools.
    
    Allows dynamic registration and execution of tools.
    """
    
    def __init__(self):
        self._tools: dict[str, Tool] = {}
    
    def register(self, tool: Tool) -> None:
        """Register a tool."""
        self._tools[tool.name] = tool
    
    def unregister(self, name: str) -> None:
        """Unregister a tool by name."""
        self._tools.pop(name, None)
    
    def get(self, name: str) -> Tool | None:
        """Get a tool by name."""
        return self._tools.get(name)
    
    def has(self, name: str) -> bool:
        """Check if a tool is registered."""
        return name in self._tools
    
    def get_definitions(self) -> list[dict[str, Any]]:
        """Get all tool definitions in OpenAI format."""
        return [tool.to_schema() for tool in self._tools.values()]
    
    async def execute(self, name: str, params: dict[str, Any]) -> str:
        """Execute a tool by name with given parameters."""
        _HINT = "\n\n[Analyze the error above and try a different approach.]"

        tool = self._tools.get(name)
        if not tool:
            return f"Error: Tool '{name}' not found. Available: {', '.join(self.tool_names)}"

        try:
            errors = tool.validate_params(params)
            if errors:
                return f"Error: Invalid parameters for tool '{name}': " + "; ".join(errors) + _HINT
            result = await tool.execute(**params)
            if isinstance(result, str) and result.startswith("Error"):
                return result + _HINT
            return self._cap_result(name, result)
        except Exception as e:
            return f"Error executing {name}: {str(e)}" + _HINT

    @staticmethod
    def _cap_result(name: str, result: Any) -> Any:
        """把单条工具结果压到 token 预算内。

        Why: 各工具自己的上限是按字符定的、且互不一致（exec 10k 字符、
        read_file 100k、MCP 无上限），换算成 token 后最坏能到 200k，超出
        compaction 的保留窗口预算数倍——这种情况下压缩再怎么做都守不住上限。
        在这个出口统一收口，顺带覆盖所有 MCP 工具。

        保留头尾：错误信息通常在结尾，而结构性信息（表头、路径）在开头。
        """
        if not isinstance(result, str):
            return result

        from bubbles.agent.compaction import estimate_tokens

        tokens = estimate_tokens(result)
        if tokens <= MAX_RESULT_TOKENS:
            return result

        # 按实际 token 密度反推可保留的字符数，中英文都适用。
        budget_chars = max(1, int(len(result) * MAX_RESULT_TOKENS / tokens))
        head = budget_chars * 2 // 3
        tail = budget_chars - head
        logger.warning(
            "Tool '{}' result too large ({} tokens); truncated to ~{}",
            name, tokens, MAX_RESULT_TOKENS,
        )
        return (
            result[:head]
            + f"\n\n... [输出过长，中间省略约 {tokens - MAX_RESULT_TOKENS} tokens；"
              f"需要完整内容请缩小范围重新调用] ...\n\n"
            + result[-tail:]
        )
    
    @property
    def tool_names(self) -> list[str]:
        """Get list of registered tool names."""
        return list(self._tools.keys())
    
    def __len__(self) -> int:
        return len(self._tools)
    
    def __contains__(self, name: str) -> bool:
        return name in self._tools
