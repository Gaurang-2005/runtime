"""MCP server: the gitm server's control API as tools an agent can call.

    gitm mcp --url http://127.0.0.1:7070 --token-file <workdir>/api-token

Runs over stdio, the way MCP clients launch local servers. Every tool is a thin
call to the REST API, so an agent can do exactly what the dashboard and the CLI
can and nothing more: it goes through the same token and, for promotion, the
same rules. ``promote`` re-checks every rule; approval skips none.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

INSTRUCTIONS = """\
Controls a gitm server: a stable vLLM serving production traffic, and a canary
where configuration changes are A/B-tested against recorded traffic.

Typical flow: status -> start_intervention(knobs) -> poll get_intervention until
its status is final -> promotion_rules(id) to see why it was or was not promoted.
Promotion is automatic when every rule passes (unless the policy requires
approval); revert works only during probation. Knobs are vLLM engine argument
names, e.g. {"max_num_batched_tokens": 16384} or {"enforce_eager": false}.
"""


class Client:
    def __init__(self, url: str, token: str):
        import httpx

        self.http = httpx.Client(base_url=url.rstrip("/"), timeout=30.0,
                                 headers={"Authorization": f"Bearer {token}"})

    def call(self, method: str, path: str, **kw: Any) -> Any:
        r = self.http.request(method, path, **kw)
        try:
            body = r.json()
        except json.JSONDecodeError:
            body = r.text
        if r.status_code >= 400:
            detail = body.get("detail", body) if isinstance(body, dict) else body
            return {"error": detail, "status_code": r.status_code}
        return body


def _server_class():
    try:  # mcp >= 2
        from mcp.server.mcpserver import MCPServer
        return MCPServer
    except ImportError:  # mcp 1.x
        from mcp.server.fastmcp import FastMCP
        return FastMCP


def build(client: Client):
    server = _server_class()(name="gitm", instructions=INSTRUCTIONS)
    tool = server.tool

    @tool()
    def status() -> dict:
        """Engines, what is running or queued, probation, recent promotions."""
        return client.call("GET", "/status")

    @tool()
    def live_stats(window_s: float = 60.0) -> dict:
        """Production numbers through the proxy over the last window_s seconds:
        requests/s, errors, output tokens/s, TTFT and latency p50/p99."""
        return client.call("GET", "/stats", params={"window_s": window_s})

    @tool()
    def list_interventions() -> list:
        """Every intervention with its status and measured effect."""
        return client.call("GET", "/interventions")

    @tool()
    def get_intervention(intervention_id: str) -> dict:
        """One intervention in full: samples so far, verdict, promotion rules."""
        return client.call("GET", f"/interventions/{intervention_id}")

    @tool()
    def start_intervention(knobs: dict, note: str = "") -> dict:
        """Queue an A/B of these vLLM engine arguments against stable's on the
        canary. Refused if stable already runs them or vLLM rejects the flags."""
        return client.call("POST", "/interventions",
                           json={"knobs": knobs, "note": note, "submitter": "mcp-agent"})

    @tool()
    def stop_intervention(intervention_id: str) -> dict:
        """Stop a queued or running intervention; the canary is put back."""
        return client.call("POST", f"/interventions/{intervention_id}/stop")

    @tool()
    def promotion_rules(intervention_id: str) -> dict:
        """Every promotion rule evaluated now for this intervention, with evidence."""
        return client.call("GET", f"/interventions/{intervention_id}/rules")

    @tool()
    def promote(intervention_id: str) -> dict:
        """Promote an intervention awaiting approval. Every rule is checked again."""
        return client.call("POST", f"/interventions/{intervention_id}/promote",
                           json={"by": "mcp-agent"})

    @tool()
    def revert(reason: str) -> dict:
        """During probation, route traffic back to the previous engine."""
        return client.call("POST", "/revert", json={"reason": reason})

    @tool()
    def pause() -> dict:
        """Stop starting tests and promoting until resumed."""
        return client.call("POST", "/pause")

    @tool()
    def resume() -> dict:
        """Undo pause."""
        return client.call("POST", "/resume")

    @tool()
    def start_autonomy(mode: str, budget_h: float | None = None, max_tests: int | None = None,
                       schedule: list | None = None, allow_knobs: list | None = None,
                       deny_knobs: list | None = None) -> dict:
        """Start a session. mode "assisted" (gitm picks and tests, wins wait for
        approval) or "autonomous" (it also promotes). budget_h / max_tests end the
        session; schedule is [[start_hour, end_hour], ...] local time for gitm's own
        tests; allow/deny narrow which knobs it may touch."""
        return client.call("POST", "/autonomy/start", json={
            "mode": mode, "budget_h": budget_h, "max_tests": max_tests, "schedule": schedule,
            "allow_knobs": allow_knobs, "deny_knobs": deny_knobs})

    @tool()
    def stop_autonomy(cancel_current: bool = False) -> dict:
        """Back to manual mode. Stable is untouched; gitm's queued picks are dropped."""
        return client.call("POST", "/autonomy/stop", json={"cancel_current": cancel_current})

    @tool()
    def get_queue() -> list:
        """Queued tests in run order, and why any are waiting."""
        return client.call("GET", "/queue")

    @tool()
    def move_in_queue(intervention_id: str, position: int) -> list:
        """Move a queued test; position 0 runs next."""
        return client.call("POST", f"/queue/{intervention_id}/move", json={"position": position})

    @tool()
    def traffic_status() -> dict:
        """Live traffic against what stable was tuned for: regime, distance, changed?"""
        return client.call("GET", "/traffic/watch")

    @tool()
    def get_policy() -> dict:
        """The test and promotion policy in force."""
        return client.call("GET", "/policy")

    @tool()
    def set_policy(changes: dict) -> dict:
        """Change policy keys, e.g. {"require_approval": true}. Unknown keys refused."""
        return client.call("PUT", "/policy", json=changes)

    return server


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="gitm mcp")
    ap.add_argument("--url", default="http://127.0.0.1:7070", help="gitm server control API.")
    ap.add_argument("--token-file", type=Path, required=True,
                    help="The api-token file in the gitm server's working directory.")
    a = ap.parse_args(argv)
    client = Client(a.url, a.token_file.read_text().strip())
    build(client).run("stdio")
    return 0
