"""Tests for Phase 2b: gateway roles + RpcChannelProxy.

Full split (real Redis, two hosts) is verified in the user's deployment; here
we test the role guard, the CLI surface, and the find_person degrade contract.
"""

import asyncio

from typer.testing import CliRunner

from bubbles.bus.local import LocalBus
from bubbles.bus.rpc_proxy import RpcChannelProxy
from bubbles.cli.commands import app


def test_gateway_help_lists_roles() -> None:
    out = CliRunner().invoke(app, ["gateway", "--help"]).output
    assert "--role" in out
    assert "harness" in out and "channels" in out


def test_split_role_requires_networked_bus() -> None:
    # Default config has bus.default="local"; harness/channels must refuse.
    for role in ("harness", "channels"):
        result = CliRunner().invoke(app, ["gateway", "--role", role])
        assert result.exit_code == 1
        assert "requires a networked bus" in result.output


def test_unknown_role_rejected() -> None:
    result = CliRunner().invoke(app, ["gateway", "--role", "bogus"])
    assert result.exit_code == 1
    assert "Unknown --role" in result.output


async def test_proxy_degrades_to_empty_without_rpc() -> None:
    # LocalBus has no .request lane → roster lookup must degrade to [], never raise.
    proxy = RpcChannelProxy(LocalBus())
    stub = proxy.get_channel("wechat")
    assert await stub.get_group_members("grp") == []


async def test_proxy_returns_rpc_result_when_available() -> None:
    class _FakeBus:
        async def request(self, verb, payload, timeout):
            assert verb == "roster"
            return [{"id": "u1", "names": {"nick": "Bob"}}]

    proxy = RpcChannelProxy(_FakeBus())
    members = await proxy.get_channel("wechat").get_group_members("grp")
    assert members == [{"id": "u1", "names": {"nick": "Bob"}}]


async def test_proxy_degrades_on_rpc_error() -> None:
    class _BoomBus:
        async def request(self, verb, payload, timeout):
            raise RuntimeError("channel host down")

    proxy = RpcChannelProxy(_BoomBus())
    assert await proxy.get_channel("wechat").get_group_members("grp") == []
