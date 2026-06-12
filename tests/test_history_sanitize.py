"""Tests for session history sanitization & compaction split alignment.

Covers two safeguards introduced to fix "session broken" cases where the next
LLM API call would reject the history due to unpaired tool_calls/tool_result:

1. `_sanitize_for_api`: read-side cleanup that drops any dangling
   `assistant.tool_calls` (with partial or missing tool_result) and any orphan
   `role=tool` messages.
2. `_align_split_to_user_boundary`: write-side cleanup that forces a compaction
   split to fall on a `user` boundary so the kept tail is never a half-turn.
"""

from __future__ import annotations

from bubbles.agent.compaction import _align_split_to_user_boundary
from bubbles.session.manager import Session, _sanitize_for_api


def _user(text: str) -> dict:
    return {"role": "user", "content": text}


def _asst(text: str | None = None, tool_calls: list[dict] | None = None) -> dict:
    msg: dict = {"role": "assistant", "content": text}
    if tool_calls:
        msg["tool_calls"] = tool_calls
    return msg


def _tc(call_id: str, name: str = "echo", args: str = "{}") -> dict:
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": args},
    }


def _tool(call_id: str, name: str = "echo", content: str = "ok") -> dict:
    return {
        "role": "tool",
        "tool_call_id": call_id,
        "name": name,
        "content": content,
    }


class TestSanitizeForApi:
    """Tests for `_sanitize_for_api`."""

    def test_empty_input(self):
        assert _sanitize_for_api([]) == []

    def test_passes_through_clean_history(self):
        msgs = [
            _user("hi"),
            _asst("hello"),
            _user("call a tool"),
            _asst(None, [_tc("c1")]),
            _tool("c1"),
            _asst("done"),
        ]
        assert _sanitize_for_api(msgs) == msgs

    def test_drops_orphan_leading_tool_result(self):
        msgs = [
            _tool("c1"),  # 孤立——前面没有 assistant.tool_calls
            _user("hi"),
            _asst("hello"),
        ]
        out = _sanitize_for_api(msgs)
        assert out == msgs[1:]

    def test_drops_orphan_middle_tool_result(self):
        msgs = [
            _user("hi"),
            _asst("hi back"),
            _tool("ghost"),  # 孤立
            _user("again"),
        ]
        out = _sanitize_for_api(msgs)
        assert out == [msgs[0], msgs[1], msgs[3]]

    def test_drops_dangling_assistant_tool_calls_at_end(self):
        msgs = [
            _user("hi"),
            _asst(None, [_tc("c1")]),
            # 缺 tool_result
        ]
        out = _sanitize_for_api(msgs)
        assert out == [msgs[0]]

    def test_drops_partial_tool_result_at_end(self):
        msgs = [
            _user("hi"),
            _asst(None, [_tc("c1"), _tc("c2")]),
            _tool("c1"),
            # c2 缺
        ]
        out = _sanitize_for_api(msgs)
        assert out == [msgs[0]]

    def test_drops_dangling_then_keeps_following_clean_turn_when_user_arrives(self):
        # 新的 user 来到时，前一段未完成的 assistant.tool_calls 会被回退。
        msgs = [
            _user("first"),
            _asst(None, [_tc("c1")]),
            _user("second"),
            _asst("ok"),
        ]
        out = _sanitize_for_api(msgs)
        assert out == [msgs[0], msgs[2], msgs[3]]

    def test_keeps_multi_tool_call_when_all_paired(self):
        msgs = [
            _user("hi"),
            _asst(None, [_tc("c1"), _tc("c2"), _tc("c3")]),
            _tool("c2"),
            _tool("c1"),
            _tool("c3"),
            _asst("done"),
        ]
        out = _sanitize_for_api(msgs)
        assert out == msgs

    def test_drops_dangling_then_keeps_following_clean_turn_after_new_tool_call(self):
        msgs = [
            _user("first"),
            _asst(None, [_tc("c1")]),  # 半截
            _asst(None, [_tc("c2")]),  # 直接来了新 turn
            _tool("c2"),
            _asst("done"),
        ]
        out = _sanitize_for_api(msgs)
        assert out == [msgs[0], msgs[2], msgs[3], msgs[4]]

    def test_orphan_tool_inside_otherwise_clean_turn(self):
        msgs = [
            _user("hi"),
            _asst(None, [_tc("c1")]),
            _tool("ghost"),  # 孤立 id 不在 pending → drop
            _tool("c1"),
            _asst("done"),
        ]
        out = _sanitize_for_api(msgs)
        assert out == [msgs[0], msgs[1], msgs[3], msgs[4]]


class TestAlignSplitToUserBoundary:
    """Tests for `_align_split_to_user_boundary`."""

    def test_split_already_on_user(self):
        to_summ = [_user("a"), _asst("b")]
        to_keep = [_user("c"), _asst("d")]
        s, k = _align_split_to_user_boundary(list(to_summ), list(to_keep))
        assert s == to_summ
        assert k == to_keep

    def test_tool_at_keep_head_moves_to_summarize(self):
        to_summ = [_user("a"), _asst(None, [_tc("c1")])]
        to_keep = [_tool("c1"), _user("b"), _asst("c")]
        s, k = _align_split_to_user_boundary(list(to_summ), list(to_keep))
        assert s == to_summ + [_tool("c1")]
        assert k == [_user("b"), _asst("c")]

    def test_assistant_tool_calls_at_keep_head_moves_to_summarize(self):
        # assistant.tool_calls 起头不安全：保守做法是推到 to_summarize 直到 user。
        to_summ = [_user("a"), _asst("b")]
        to_keep = [_asst(None, [_tc("c1")]), _tool("c1"), _user("c"), _asst("d")]
        s, k = _align_split_to_user_boundary(list(to_summ), list(to_keep))
        assert k == [_user("c"), _asst("d")]
        assert s == to_summ + [_asst(None, [_tc("c1")]), _tool("c1")]

    def test_no_user_in_keep_empties_it(self):
        to_summ = [_user("a")]
        to_keep = [_asst("b"), _asst(None, [_tc("c1")]), _tool("c1")]
        s, k = _align_split_to_user_boundary(list(to_summ), list(to_keep))
        assert k == []
        assert s == to_summ + to_keep

    def test_empty_to_keep_is_noop(self):
        to_summ = [_user("a"), _asst("b")]
        s, k = _align_split_to_user_boundary(list(to_summ), [])
        assert s == to_summ
        assert k == []


class TestGetHistoryUsesSanitize:
    """End-to-end: get_history returns API-valid output even with broken jsonl."""

    def test_get_history_strips_dangling_tool_calls(self, tmp_path):
        s = Session(key="cli:test")
        s.messages = [
            _user("hi"),
            _asst("hello"),
            _user("call"),
            _asst(None, [_tc("c1")]),  # 半截：只有 assistant.tool_calls，没 tool_result
        ]
        out = s.get_history(max_messages=500)
        # 末尾 assistant.tool_calls 被剔除；上一条 user("call") 实际发生过，保留。
        assert all(not (m.get("role") == "assistant" and m.get("tool_calls")) for m in out)
        assert [m["role"] for m in out] == ["user", "assistant", "user"]
        assert out[-1].get("content") == "call"

    def test_get_history_drops_orphan_leading_tool(self):
        s = Session(key="cli:test")
        s.messages = [
            _tool("ghost"),
            _user("hi"),
            _asst("hello"),
        ]
        out = s.get_history(max_messages=500)
        assert out[0]["role"] == "user"

    def test_get_history_preserves_clean_history(self):
        s = Session(key="cli:test")
        clean = [
            _user("hi"),
            _asst(None, [_tc("c1")]),
            _tool("c1"),
            _asst("done"),
        ]
        s.messages = list(clean)
        out = s.get_history(max_messages=500)
        # role + content + tool_calls/tool_call_id/name 都应保留
        assert [m["role"] for m in out] == ["user", "assistant", "tool", "assistant"]
        assert out[1].get("tool_calls") == clean[1]["tool_calls"]
        assert out[2].get("tool_call_id") == "c1"
