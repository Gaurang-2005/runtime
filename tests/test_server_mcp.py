"""gitm server, milestone 6: the MCP tools wrap the REST API one to one."""

from __future__ import annotations

import asyncio

import pytest

pytest.importorskip("mcp")
pytest.importorskip("httpx")

from gitm.server.mcp import build  # noqa: E402


class FakeClient:
    def __init__(self):
        self.calls = []

    def call(self, method, path, **kw):
        self.calls.append((method, path, kw))
        return {"ok": True}


def test_every_control_is_a_tool_and_calls_its_route():
    client = FakeClient()
    server = build(client)
    names = {t.name for t in asyncio.run(server.list_tools())}
    assert {"status", "start_intervention", "stop_intervention", "promotion_rules", "promote",
            "revert", "pause", "resume", "get_policy", "set_policy", "live_stats"} <= names
    asyncio.run(server.call_tool("start_intervention", {"knobs": {"max_num_seqs": 128}}))
    method, path, kw = client.calls[-1]
    assert (method, path) == ("POST", "/interventions")
    assert kw["json"]["knobs"] == {"max_num_seqs": 128} and kw["json"]["submitter"] == "mcp-agent"
