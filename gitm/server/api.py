"""The control API: what the dashboard, the CLI and an agent (via MCP) call.

Every route needs the bearer token the controller wrote to ``<workdir>/api-token``.
It binds to localhost unless told otherwise; exposing it is a deliberate choice.
"""

from __future__ import annotations

import time
from dataclasses import asdict
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

    from fastapi.middleware.cors import CORSMiddleware

    app = FastAPI(title="gitm server", dependencies=[Depends(auth)],
                  docs_url=None, redoc_url=None)
    # The dashboard is a separate page on its own port, so the browser needs
    # CORS to call this API. Only the configured origins; the token still
    # guards every route. Preflight requests carry no token, so they are
    # answered by the middleware before auth runs.
    app.add_middleware(CORSMiddleware, allow_origins=list(ctl.cfg.dashboard_origins),
                       allow_methods=["GET", "POST", "PUT"],
                       allow_headers=["Authorization", "Content-Type"])

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

    @app.get("/policy")
    def get_policy():
        return ctl.policy

    @app.get("/interventions")
    def list_interventions():
        return [iv.brief() for iv in ctl.interventions.values()]

    @app.post("/interventions", status_code=202)
    def submit(body: dict):
        knobs = body.get("knobs")
        if not isinstance(knobs, dict):
            raise HTTPException(status_code=400, detail='body needs "knobs": {knob: value}')
        try:
            iv = ctl.submit(knobs, submitter=str(body.get("submitter", "api")),
                            note=str(body.get("note", "")), front=bool(body.get("front")))
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from None
        return iv.brief()

    @app.get("/interventions/{ident}")
    def get_intervention(ident: str):
        iv = ctl.interventions.get(ident)
        if iv is None:
            raise HTTPException(status_code=404, detail="no such intervention")
        return asdict(iv)

    @app.post("/interventions/{ident}/stop")
    def stop_intervention(ident: str):
        if ident not in ctl.interventions:
            raise HTTPException(status_code=404, detail="no such intervention")
        return ctl.stop_intervention(ident).brief()

    def _iv(ident: str):
        iv = ctl.interventions.get(ident)
        if iv is None:
            raise HTTPException(status_code=404, detail="no such intervention")
        return iv

    @app.get("/interventions/{ident}/rules")
    def rules(ident: str):
        from gitm.server.promotion import all_pass, as_dicts

        try:
            r = ctl.rules_for(_iv(ident))
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from None
        return {"promotable": all_pass(r), "rules": as_dicts(r)}

    @app.post("/interventions/{ident}/promote")
    def promote(ident: str, body: dict | None = None):
        """Approve a win. The canary is put back on its flags if needed and every
        rule is checked again; approval skips none."""
        try:
            return ctl.approve(_iv(ident), by=str((body or {}).get("by", "api")))
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from None

    @app.post("/interventions/{ident}/decline")
    def decline(ident: str, body: dict | None = None):
        try:
            return ctl.decline(_iv(ident), by=str((body or {}).get("by", "api")),
                               reason=str((body or {}).get("reason", ""))).brief()
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from None

    @app.post("/revert")
    def revert(body: dict | None = None):
        try:
            return ctl.revert(str((body or {}).get("reason", "reverted by operator")),
                              cause="api")
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from None

    @app.post("/pause")
    def pause():
        ctl.set_policy({"paused": True})
        return {"paused": True}

    @app.post("/resume")
    def resume():
        ctl.set_policy({"paused": False})
        return {"paused": False}

    @app.put("/policy")
    def put_policy(body: dict):
        try:
            return ctl.set_policy(body)
        except (KeyError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc).strip('"')) from None

    @app.get("/policy/warnings")
    def policy_warnings():
        from gitm.server.policy import warnings_for

        return warnings_for(ctl.policy)

    def _sim():
        if ctl.simulator is None:
            raise HTTPException(status_code=409, detail="the server has not started serving yet")
        return ctl.simulator

    @app.get("/traffic/presets")
    def traffic_presets():
        from gitm.server.simulate import PRESETS

        return PRESETS

    @app.get("/traffic")
    def traffic_runs():
        return [r.brief() for r in _sim().runs.values()]

    @app.post("/traffic/simulate", status_code=202)
    def simulate(body: dict):
        try:
            run = _sim().start(str(body.get("preset", "chat")),
                               rate_rps=body.get("rate_rps"),
                               duration_s=float(body.get("duration_s", 120.0)),
                               burstiness=body.get("burstiness"), seed=body.get("seed"))
        except (ValueError, TypeError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from None
        ctl.event("traffic.started", id=run.id, preset=run.preset, rate_rps=run.rate_rps,
                  duration_s=run.duration_s)
        return run.brief()

    @app.post("/traffic/{ident}/stop")
    def stop_traffic(ident: str):
        sim = _sim()
        if ident not in sim.runs:
            raise HTTPException(status_code=404, detail="no such traffic run")
        return sim.stop(ident).brief()

    # --- autonomy: who decides what to test and what goes live

    @app.get("/autonomy")
    def autonomy():
        return {"mode": ctl.mode,
                "session": ctl.session.summary(time.time()) if ctl.session else None}

    @app.post("/autonomy/start")
    def autonomy_start(body: dict):
        """{"mode": "assisted"|"autonomous", "budget_h"?, "max_tests"?,
        "schedule"?: [[start_hour, end_hour], ...] (local), "allow_knobs"?,
        "deny_knobs"?, "tiers"?}"""
        try:
            return ctl.start_session(
                str(body.get("mode", "")), budget_h=body.get("budget_h"),
                max_tests=body.get("max_tests"), schedule=body.get("schedule"),
                allow_knobs=body.get("allow_knobs"), deny_knobs=body.get("deny_knobs"),
                tiers=body.get("tiers"))
        except (ValueError, TypeError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from None

    @app.post("/autonomy/stop")
    def autonomy_stop(body: dict | None = None):
        """Back to manual. {"cancel_current": true} also cancels a test gitm started."""
        return ctl.stop_session(cancel_current=bool((body or {}).get("cancel_current")),
                                reason="stopped by operator")

    @app.get("/explore/preview")
    def explore_preview(n: int = 5):
        try:
            return ctl.explore_preview(n)
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=409, detail=str(exc)) from None

    # --- where stable's time goes, and what the explorer ranks by

    @app.get("/profile")
    def profile():
        return ctl.profile_view()

    @app.post("/profile", status_code=202)
    def request_profile():
        try:
            return ctl.request_profile(by="api")
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from None

    # --- the queue, in run order

    @app.get("/queue")
    def queue_view():
        return ctl.queue_view()

    @app.post("/queue/{ident}/move")
    def queue_move(ident: str, body: dict):
        try:
            return ctl.move(ident, int(body.get("position", 0)))
        except (ValueError, TypeError) as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from None

    @app.post("/queue/{ident}/front")
    def queue_front(ident: str):
        try:
            return ctl.move(ident, 0)
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from None

    # --- traffic: what stable is tuned for vs what is arriving now

    @app.get("/traffic/watch")
    def traffic_watch():
        return ctl.watch.summary()

    @app.get("/promotions")
    def promotions():
        return {"promotions": ctl.state["promotions"], "reverted": ctl.state["reverted"],
                "probation": ctl.probation_summary()}

    return app
