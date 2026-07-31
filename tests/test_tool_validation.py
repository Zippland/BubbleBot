from typing import Any

from bubbles.agent.tools.base import Tool
from bubbles.agent.tools.registry import ToolRegistry


class SampleTool(Tool):
    @property
    def name(self) -> str:
        return "sample"

    @property
    def description(self) -> str:
        return "sample tool"

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {"type": "string", "minLength": 2},
                "count": {"type": "integer", "minimum": 1, "maximum": 10},
                "mode": {"type": "string", "enum": ["fast", "full"]},
                "meta": {
                    "type": "object",
                    "properties": {
                        "tag": {"type": "string"},
                        "flags": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                    },
                    "required": ["tag"],
                },
            },
            "required": ["query", "count"],
        }

    async def execute(self, **kwargs: Any) -> str:
        return "ok"


def test_validate_params_missing_required() -> None:
    tool = SampleTool()
    errors = tool.validate_params({"query": "hi"})
    assert "missing required count" in "; ".join(errors)


def test_validate_params_type_and_range() -> None:
    tool = SampleTool()
    errors = tool.validate_params({"query": "hi", "count": 0})
    assert any("count must be >= 1" in e for e in errors)

    errors = tool.validate_params({"query": "hi", "count": "2"})
    assert any("count should be integer" in e for e in errors)


def test_validate_params_enum_and_min_length() -> None:
    tool = SampleTool()
    errors = tool.validate_params({"query": "h", "count": 2, "mode": "slow"})
    assert any("query must be at least 2 chars" in e for e in errors)
    assert any("mode must be one of" in e for e in errors)


def test_validate_params_nested_object_and_array() -> None:
    tool = SampleTool()
    errors = tool.validate_params(
        {
            "query": "hi",
            "count": 2,
            "meta": {"flags": [1, "ok"]},
        }
    )
    assert any("missing required meta.tag" in e for e in errors)
    assert any("meta.flags[0] should be string" in e for e in errors)


def test_validate_params_ignores_unknown_fields() -> None:
    tool = SampleTool()
    errors = tool.validate_params({"query": "hi", "count": 2, "extra": "x"})
    assert errors == []


async def test_registry_returns_validation_error() -> None:
    reg = ToolRegistry()
    reg.register(SampleTool())
    result = await reg.execute("sample", {"query": "hi"})
    assert "Invalid parameters" in result


# ---- 单条工具结果的 token 上限（保护 compaction 的保留窗口预算）----

def test_oversized_tool_result_is_capped() -> None:
    """read_file 上限 100k 字符、MCP 无上限；换算成 token 会数倍于保留窗口预算。"""
    from bubbles.agent.compaction import estimate_tokens
    from bubbles.agent.tools.registry import MAX_RESULT_TOKENS, ToolRegistry

    capped = ToolRegistry._cap_result("read_file", "中" * 100_000)
    assert estimate_tokens("中" * 100_000) > MAX_RESULT_TOKENS * 10
    assert estimate_tokens(capped) <= MAX_RESULT_TOKENS * 1.05
    assert "省略" in capped, "要告诉模型内容被截断了，否则它会以为读到了全文"


def test_small_tool_result_untouched() -> None:
    from bubbles.agent.tools.registry import ToolRegistry

    assert ToolRegistry._cap_result("exec", "hello") == "hello"


def test_capped_result_keeps_head_and_tail() -> None:
    """错误信息通常在结尾，结构性信息在开头，两头都不能丢。"""
    from bubbles.agent.tools.registry import ToolRegistry

    body = "H" * 60_000 + "TRACEBACK_AT_END"
    out = ToolRegistry._cap_result("exec", body)
    assert out.startswith("H")
    assert out.endswith("TRACEBACK_AT_END")
