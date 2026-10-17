"""The control API: what the dashboard, the CLI and an agent (via MCP) call.

Every route needs the bearer token the controller wrote to ``<workdir>/api-token``.
It binds to localhost unless told otherwise; exposing it is a deliberate choice.
"""

from __future__ import annotations

import time
from typing import Any

from gitm.server.controller import Controller


def summarize_stats(stats: list[dict[str, Any]], window_s: float) -> dict[str, Any]:
    """Live numbers over the last ``window_s`` seconds of completed requests."""
    now = time.time()
    recent = [s for s in stats if s["t_end"] >= now - window_s]
    ok = [s for s in recent if s["ok"]]
    out_tokens = sum(s["output_tokens"] or 0 for s in ok)

    def pct(values: list[float], q: float) -> float | None:
        if not values:
            return None
        values = sorted(values)
        return values[min(len(values) - 1, int(q * (len(values) - 1) + 0.5))]

    ttft = [s["ttft_s"] for s in ok if s["ttft_s"] is not None]
    lat = [s["latency_s"] for s in ok]
    return {
        "window_s": window_s,
        "requests": len(recent),
        "errors": len(recent) - len(ok),
        "requests_per_s": round(len(recent) / window_s, 3),
        "output_tokens_per_s": round(out_tokens / window_s, 1),
        "ttft_p50_s": pct(ttft, 0.5),
        "ttft_p99_s": pct(ttft, 0.99),
        "latency_p50_s": pct(lat, 0.5),
        "latency_p99_s": pct(lat, 0.99),
    }


def build_api(ctl: Controller):
    from fastapi import Depends, FastAPI, Header, HTTPException

    def auth(authorization: str | None = Header(default=None)) -> None:
        if authorization != f"Bearer {ctl.token}":
            raise HTTPException(status_code=401, detail="missing or wrong bearer token")

    app = FastAPI(title="gitm server", dependencies=[Depends(auth)],
                  docs_url=None, redoc_url=None)

    @app.get("/status")
    def status():
        return ctl.status()

    @app.get("/engines")
    def engines():
        return {role: e.info() for role, e in ctl.engines.items()}

    @app.get("/recordings")
    def recordings():
        return ctl.proxy_call("GET", "/admin/recordings")

    @app.get("/stats")
    def stats(window_s: float = 30.0):
        raw = ctl.proxy_call("GET", "/admin/stats", params={"since": time.time() - window_s})
        return summarize_stats(raw, window_s)

    @app.get("/events")
    def events(since: float = 0.0):
        return [e for e in ctl.events if e["t"] > since]

    return app
