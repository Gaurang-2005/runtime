"""The controller: owns the engines and the proxy, and answers the API.

It launches the stable engine, puts the proxy in front of it with recording on,
launches the canary beside it, and runs submitted interventions one at a time:
each is an A/B in the canary slot against a recorded window, decided by
:func:`gitm.experiment.evaluate.evaluate`. Promotion builds on this.

Shutdown is deliberate about what it takes down. A clean stop (Ctrl-C on
``gitm run --server``) stops the proxy and every engine this controller started.
A crash stops nothing: the proxy and the engines live in their own sessions, so
traffic keeps flowing to the last stable engine.
"""

from __future__ import annotations

import json
import os
import queue
import secrets
import socket
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from gitm.server import policy as policy_mod
from gitm.server.supervisor import EngineProcess, EngineSpec


@dataclass
class ServerConfig:
    model: str
    serve_args: list[str] = field(default_factory=list)
    workdir: Path = Path.home() / ".cache" / "gitm-server"
    host: str = "127.0.0.1"
    port: int = 8000            # what clients call (the proxy)
    api_port: int = 7070        # the control API
    stable_port: int = 8100
    canary_port: int = 8101
    stable_gpus: str | None = None
    canary_gpus: str | None = None
    mem_fraction: float | None = None
    record: bool = True
    canary: bool = True
    policy: dict[str, Any] = field(default_factory=dict)
    startup_timeout_s: float = 1800.0


#: Intervention states before a verdict exists. After one, the state is the
#: verdict's own (improved / regressed / inconclusive / invalid / failed_to_execute).
QUEUED, RUNNING, STOPPED, REJECTED = "queued", "running", "stopped", "rejected"


@dataclass
class Intervention:
    id: str
    knobs: dict[str, Any]
    submitter: str
    note: str
    created: float
    status: str = QUEUED
    started: float | None = None
    finished: float | None = None
    detail: str = ""
    window: str | None = None
    candidate_args: list[str] | None = None
    progress: list[dict[str, Any]] = field(default_factory=list)
    verdict: dict[str, Any] | None = None
    rules: list[dict[str, Any]] | None = None

    def brief(self) -> dict[str, Any]:
        v = self.verdict or {}
        return {"id": self.id, "knobs": self.knobs, "status": self.status,
                "submitter": self.submitter, "created": self.created,
                "finished": self.finished, "detail": self.detail,
                "effect": v.get("effect"), "interval": v.get("interval"),
                "samples": len(self.progress)}


def port_free(port: int, host: str = "127.0.0.1") -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.bind((host, port))
        except OSError:
            return False
        return True


def shares_gpus(a: EngineProcess | None, b: EngineProcess | None) -> bool:
    """Whether two engines can run on the same GPU. Unset means every GPU."""
    if a is None or b is None:
        return False
    if a.spec.gpus is None or b.spec.gpus is None:
        return True
    return bool(set(a.spec.gpus.split(",")) & set(b.spec.gpus.split(",")))


class Controller:
    def __init__(self, cfg: ServerConfig):
        self.cfg = cfg
        self.workdir = Path(cfg.workdir)
        self.started_at = time.time()
        self.token = secrets.token_urlsafe(24)
        self.engines: dict[str, EngineProcess] = {}
        self.proxy: subprocess.Popen | None = None
        self.events: list[dict[str, Any]] = []
        self.policy = policy_mod.merged(cfg.policy)
        self.interventions: dict[str, Intervention] = {}
        self._queue: queue.Queue[str] = queue.Queue()
        self._cancel: dict[str, threading.Event] = {}
        self._worker: threading.Thread | None = None
        self._stopping = threading.Event()
        # Promotion state, kept on disk so a restart knows what was promoted,
        # what was reverted (and is therefore blocked), and when.
        self.state: dict[str, Any] = {"last_promotion_t": None, "last_revert_t": None,
                                      "promotions": [], "reverted": []}
        self.probation: dict[str, Any] | None = None
        self._swap_lock = threading.Lock()
        # Held by whoever is relaunching or measuring on the canary, so a test
        # and a restore never relaunch it under each other.
        self._canary_lock = threading.Lock()
        self._audit = None

    @property
    def audit(self):
        if self._audit is None:
            from gitm.safety.audit import AuditLog

            self.workdir.mkdir(parents=True, exist_ok=True)
            self._audit = AuditLog(self.workdir / "audit.jsonl")
        return self._audit

    # ------------------------------------------------------------------ paths

    @property
    def admin_socket(self) -> Path:
        return self.workdir / "proxy-admin.sock"

    @property
    def record_dir(self) -> Path:
        return self.workdir / "recordings"

    @property
    def log_dir(self) -> Path:
        return self.workdir / "logs"

    # ------------------------------------------------------------------ lifecycle

    def preflight(self) -> list[str]:
        """Problems that would make starting pointless. Empty means go."""
        problems = []
        ports = [("proxy", self.cfg.port), ("api", self.cfg.api_port),
                 ("stable engine", self.cfg.stable_port)]
        if self.cfg.canary:
            ports.append(("canary engine", self.cfg.canary_port))
        for name, port in ports:
            if not port_free(port, self.cfg.host if name == "proxy" else "127.0.0.1"):
                problems.append(f"port {port} ({name}) is already in use")
        return problems

    def start(self) -> None:
        self.workdir.mkdir(parents=True, exist_ok=True)
        self._load_state()
        token_file = self.workdir / "api-token"
        token_file.write_text(self.token + "\n")
        os.chmod(token_file, 0o600)

        stable = EngineProcess(EngineSpec(
            role="stable", model=self.cfg.model, port=self.cfg.stable_port,
            serve_args=list(self.cfg.serve_args), gpus=self.cfg.stable_gpus,
            mem_fraction=self.cfg.mem_fraction), self.log_dir)
        self.engines["stable"] = stable
        self.event("engine.starting", role="stable", argv=stable.spec.argv())
        stable.start()
        stable.wait_healthy(self.cfg.startup_timeout_s)
        self.event("engine.healthy", role="stable", startup_s=stable.info()["startup_s"])

        self._start_proxy(stable.base_url)
        self.event("proxy.started", port=self.cfg.port, upstream=stable.base_url)

        if self.cfg.canary:
            # After stable is healthy, not beside it: two engines profiling GPU
            # memory at once on one card each see the other's allocations.
            canary = EngineProcess(EngineSpec(
                role="canary", model=self.cfg.model, port=self.cfg.canary_port,
                serve_args=list(self.cfg.serve_args), gpus=self.cfg.canary_gpus,
                mem_fraction=self.cfg.mem_fraction,
                env={"VLLM_SERVER_DEV_MODE": "1"}), self.log_dir)
            self.engines["canary"] = canary
            self.event("engine.starting", role="canary", argv=canary.spec.argv())
            canary.start()
            canary.wait_healthy(self.cfg.startup_timeout_s)
            self.event("engine.healthy", role="canary", startup_s=canary.info()["startup_s"])
            self._worker = threading.Thread(target=self._work, name="gitm-interventions",
                                            daemon=True)
            self._worker.start()

    def _start_proxy(self, upstream: str) -> None:
        cmd = [sys.executable, "-m", "gitm.server.proxy", "--host", self.cfg.host,
               "--port", str(self.cfg.port), "--upstream", upstream,
               "--admin-socket", str(self.admin_socket)]
        if self.cfg.record:
            cmd += ["--record-dir", str(self.record_dir)]
        self.log_dir.mkdir(parents=True, exist_ok=True)
        log = (self.log_dir / "proxy.log").open("ab")
        self.proxy = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT,
                                      start_new_session=True)
        log.close()
        deadline = time.time() + 30
        while time.time() < deadline:
            if self.proxy.poll() is not None:
                raise RuntimeError(f"proxy exited with {self.proxy.returncode}; "
                                   f"see {self.log_dir / 'proxy.log'}")
            try:
                self.proxy_call("GET", "/admin/state")
                return
            except Exception:
                time.sleep(0.2)
        raise RuntimeError("proxy did not come up within 30s")

    def stop(self) -> None:
        """Clean stop: proxy first (no new requests), then the engines."""
        self._stopping.set()
        for ev in self._cancel.values():
            ev.set()
        if self.proxy is not None and self.proxy.poll() is None:
            self.proxy.terminate()
            try:
                self.proxy.wait(timeout=15)
            except subprocess.TimeoutExpired:
                self.proxy.kill()
        for eng in self.engines.values():
            eng.stop()
        self.event("server.stopped")

    # ------------------------------------------------------------------ proxy admin

    def proxy_call(self, method: str, path: str, json: dict | None = None,
                   params: dict | None = None, timeout: float = 10.0) -> Any:
        import httpx

        transport = httpx.HTTPTransport(uds=str(self.admin_socket))
        with httpx.Client(transport=transport, base_url="http://proxy",
                          timeout=timeout) as c:
            r = c.request(method, path, json=json, params=params)
            r.raise_for_status()
            return r.json()

    # ------------------------------------------------------------------ state

    def event(self, kind: str, **detail: Any) -> None:
        self.events.append({"t": time.time(), "kind": kind, **detail})
        del self.events[:-1000]

    def status(self) -> dict[str, Any]:
        try:
            proxy = self.proxy_call("GET", "/admin/state")
        except Exception as exc:
            proxy = {"error": str(exc)}
        running = [i.brief() for i in self.interventions.values() if i.status == RUNNING]
        return {
            "model": self.cfg.model,
            "uptime_s": round(time.time() - self.started_at, 1),
            "endpoint": f"http://{self.cfg.host}:{self.cfg.port}/v1",
            "engines": {role: e.info() for role, e in self.engines.items()},
            "proxy": proxy,
            "running": running[0] if running else None,
            "queued": [i.id for i in self.interventions.values() if i.status == QUEUED],
            "probation": self.probation_summary(),
            "paused": self.policy["paused"],
            "promotions": self.state["promotions"][-5:],
            "workdir": str(self.workdir),
        }

    def probation_summary(self) -> dict[str, Any] | None:
        p = self.probation
        if p is None:
            return None
        return {k: p[k] for k in ("intervention", "started", "ends", "baseline_decode_tok_s")} | {
            "observations": p["watch"].observations[-10:]}

    # ------------------------------------------------------------------ state file

    def _load_state(self) -> None:
        path = self.workdir / "state.json"
        if path.exists():
            try:
                self.state.update(json.loads(path.read_text()))
            except (OSError, json.JSONDecodeError):
                pass

    def _save_state(self) -> None:
        path = self.workdir / "state.json"
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.state, indent=2, default=str))
        tmp.replace(path)

    # ------------------------------------------------------------------ policy

    def set_policy(self, changes: dict[str, Any]) -> dict[str, Any]:
        updated = policy_mod.merged({**self.policy, **changes})
        self.policy = updated
        self.event("policy.changed", changes=changes)
        self.audit.record("policy", "-", "api", changes=changes)
        return updated

    # ------------------------------------------------------------------ promotion

    def rules_for(self, iv: Intervention):
        """Every promotion rule evaluated for ``iv`` right now."""
        from gitm.server import promotion
        from gitm.server.canary import load_window

        if iv.verdict is None or iv.window is None:
            raise ValueError("this intervention has no verdict to promote on")
        _, tested = load_window(Path(iv.window), max_requests=self.policy["max_replay_requests"],
                                speed=self.policy["replay_speed"])
        live = None
        try:
            windows = [Path(p) for p in self.proxy_call("GET", "/admin/recordings")]
        except Exception:
            windows = []
        newer = [w for w in windows if w.name > Path(iv.window).name]
        for w in reversed(newer):
            try:
                trace, regime = load_window(w, max_requests=self.policy["max_replay_requests"],
                                            speed=self.policy["replay_speed"])
            except Exception:
                continue
            if len(trace.requests) >= self.policy["min_window_requests"]:
                live = regime
                break
        stable, canary = self.engines.get("stable"), self.engines.get("canary")

        def placement(e):
            if e is None:
                return {}
            gpus = e.spec.gpus
            return {"gpus_count": len(gpus.split(",")) if gpus else None,
                    "mem_fraction": e.spec.mem_fraction}

        ctx = promotion.PromotionContext(
            policy=self.policy, now=time.time(), verdict_finished=iv.finished,
            tested_regime=tested, live_regime=live,
            stable_placement=placement(stable), canary_placement=placement(canary),
            shared_gpu=shares_gpus(stable, canary),
            canary_alive=bool(canary and canary.alive),
            canary_runs_candidate=bool(canary and canary.spec.serve_args == iv.candidate_args),
            last_promotion_t=self.state["last_promotion_t"],
            last_revert_t=self.state["last_revert_t"], reverted=self.state["reverted"])
        return promotion.check(iv.knobs, iv.verdict, ctx)

    def _consider(self, iv: Intervention) -> None:
        """After an ``improved`` verdict: promote, wait for approval, or stand down."""
        from gitm.server import promotion

        rules = self.rules_for(iv)
        iv.rules = promotion.as_dicts(rules)
        failed = [r.name for r in rules if not r.passed]
        if failed:
            iv.detail = f"improved, not promoted: failed {failed}"
            self.event("promotion.refused", id=iv.id, failed=failed)
            self._restore_canary_async(cause=f"{iv.id} not promoted")
            return
        if self.policy["require_approval"] or not self.policy["auto_promote"]:
            iv.status = "awaiting_approval"
            iv.detail = "every promotion rule passed; waiting for POST /interventions/{id}/promote"
            self.event("promotion.awaiting_approval", id=iv.id)
            return
        self.promote(iv, cause="auto: every promotion rule passed")

    def promote(self, iv: Intervention, *, cause: str) -> dict[str, Any]:
        """Route traffic to the canary, keep the old engine on standby, start probation."""
        from gitm.server import promotion

        with self._swap_lock:
            if self.probation is not None:
                raise ValueError("another promotion is still in probation")
            rules = self.rules_for(iv)
            iv.rules = promotion.as_dicts(rules)
            failed = [r.name for r in rules if not r.passed]
            if failed:
                raise ValueError(f"promotion rules failed: {failed}")
            old, new = self.engines["stable"], self.engines["canary"]
            before = [s for s in self.proxy_call("GET", "/admin/stats",
                                                 params={"since": time.time() - 300})
                      if s["upstream"] == old.base_url]
            baseline = promotion.decode_speed(before)
            swap = self.proxy_call("POST", "/admin/upstream", timeout=200.0,
                                   json={"url": new.base_url, "drain_timeout_s": 120.0})
            old.spec.role, new.spec.role = "standby", "stable"
            self.engines = {"stable": new, "standby": old}
            now = time.time()
            record = {"t": now, "intervention": iv.id, "knobs": iv.knobs, "cause": cause,
                      "from_args": old.spec.serve_args, "to_args": new.spec.serve_args,
                      "drained": swap.get("drained"), "outcome": "probation"}
            self.state["promotions"].append(record)
            self.state["last_promotion_t"] = now
            self._save_state()
            for k, v in iv.knobs.items():
                self.audit.record_apply(iv.id, knob=k, value=v, cause=cause)
            iv.status, iv.detail = "promoted", f"in probation for {self.policy['probation_s']:.0f}s"
            self._save(iv)
            self.probation = {"intervention": iv.id, "started": now,
                              "ends": now + self.policy["probation_s"],
                              "baseline_decode_tok_s": baseline, "record": record,
                              "watch": promotion.Probation(baseline, self.policy)}
            self.event("promotion.swapped", id=iv.id, drained=swap.get("drained"),
                       baseline_decode_tok_s=baseline)
        threading.Thread(target=self._watch_probation, name="gitm-probation", daemon=True).start()
        return record

    def _watch_probation(self) -> None:
        p = self.probation
        if p is None:
            return
        last = p["started"]
        while not self._stopping.is_set() and self.probation is p:
            time.sleep(self.policy["probation_interval_s"])
            now = time.time()
            new = self.engines["stable"]
            try:
                stats = [s for s in self.proxy_call("GET", "/admin/stats", params={"since": last})
                         if s["upstream"] == new.base_url]
            except Exception as exc:
                self.revert(f"proxy unreachable during probation: {exc}", cause="probation")
                return
            last = now
            if not new.alive:
                self.revert("the promoted engine exited", cause="probation")
                return
            reason = p["watch"].observe(stats, now)
            if reason:
                self.revert(reason, cause="probation")
                return
            if now >= p["ends"]:
                self._finish_probation()
                return

    def revert(self, reason: str, *, cause: str) -> dict[str, Any]:
        """During probation: route back to the standby engine, which never stopped."""
        with self._swap_lock:
            p = self.probation
            if p is None or "standby" not in self.engines:
                raise ValueError("nothing to revert: no promotion is in probation")
            promoted, old = self.engines["stable"], self.engines["standby"]
            swap = self.proxy_call("POST", "/admin/upstream", timeout=200.0,
                                   json={"url": old.base_url, "drain_timeout_s": 120.0})
            old.spec.role, promoted.spec.role = "stable", "canary"
            self.engines = {"stable": old, "canary": promoted}
            now = time.time()
            p["record"]["outcome"] = f"reverted: {reason}"
            from gitm.server.promotion import knob_key

            iv = self.interventions.get(p["intervention"])
            knobs = iv.knobs if iv else {}
            self.state["reverted"].append({"t": now, "intervention": p["intervention"],
                                           "knobs": knob_key(knobs), "reason": reason})
            self.state["last_revert_t"] = now
            self._save_state()
            self.audit.record_revert(p["intervention"], reason=reason, cause=cause)
            if iv:
                iv.status, iv.detail = "reverted", reason
                self._save(iv)
            self.probation = None
            self.event("promotion.reverted", id=p["intervention"], reason=reason,
                       drained=swap.get("drained"))
        self._restore_canary_async(cause="after revert")
        return p["record"]

    def _finish_probation(self) -> None:
        """Probation passed: stop the old engine and reuse its slot as the new canary."""
        with self._swap_lock:
            p = self.probation
            if p is None:
                return
            old = self.engines.pop("standby")
            p["record"]["outcome"] = "kept"
            self._save_state()
            iv = self.interventions.get(p["intervention"])
            if iv:
                iv.status, iv.detail = "kept", "passed probation; now the stable engine"
                self._save(iv)
            self.probation = None
            self.event("promotion.kept", id=p["intervention"])
        with self._canary_lock:
            old.stop()
            old.spec.role = "canary"
            old.spec.serve_args = list(self.engines["stable"].spec.serve_args)
            old.spec.env = {**old.spec.env, "VLLM_SERVER_DEV_MODE": "1"}
            self.engines["canary"] = old
            old.start()
            try:
                old.wait_healthy(self.cfg.startup_timeout_s)
                self.event("engine.healthy", role="canary", startup_s=old.info()["startup_s"])
            except Exception as exc:  # noqa: BLE001
                self.event("canary.restore_failed", cause="after probation", error=str(exc))

    def _restore_canary_async(self, *, cause: str) -> None:
        """Put the canary back on stable's flags, off the caller's thread."""
        def run():
            with self._canary_lock:
                canary, stable = self.engines.get("canary"), self.engines.get("stable")
                if (canary is None or stable is None
                        or (canary.alive and canary.spec.serve_args == stable.spec.serve_args)):
                    return
                canary.stop()
                canary.spec.serve_args = list(stable.spec.serve_args)
                canary.start()
                try:
                    canary.wait_healthy(self.cfg.startup_timeout_s)
                    self.event("canary.restored", cause=cause)
                except Exception as exc:  # noqa: BLE001
                    self.event("canary.restore_failed", cause=cause, error=str(exc))

        threading.Thread(target=run, name="gitm-canary-restore", daemon=True).start()

    # ------------------------------------------------------------------ interventions

    def submit(self, knobs: dict[str, Any], *, submitter: str = "api",
               note: str = "") -> Intervention:
        """Queue an A/B of ``knobs`` against the stable configuration.

        Refused up front (ValueError) when there is no canary, or when stable
        already runs every requested value: an A/B of a configuration against
        itself is an A/A, and would be reported as a lever that did nothing.
        """
        from gitm.server.canary import unchanged_knobs

        if "canary" not in self.engines:
            raise ValueError("this server runs without a canary, so it cannot test anything")
        if not knobs:
            raise ValueError("an intervention needs at least one knob")
        same = unchanged_knobs(self.engines["stable"].spec.serve_args, knobs)
        if same:
            raise ValueError(f"stable already runs {same} at the requested value")
        iv = Intervention(id=uuid.uuid4().hex[:10], knobs=dict(knobs), submitter=submitter,
                          note=note, created=time.time())
        self.interventions[iv.id] = iv
        self._cancel[iv.id] = threading.Event()
        self._save(iv)
        self._queue.put(iv.id)
        self.event("intervention.queued", id=iv.id, knobs=iv.knobs, submitter=submitter)
        return iv

    def stop_intervention(self, ident: str) -> Intervention:
        iv = self.interventions[ident]
        self._cancel[ident].set()
        if iv.status == QUEUED:
            iv.status, iv.finished, iv.detail = STOPPED, time.time(), "stopped before it ran"
            self._save(iv)
        self.event("intervention.stop_requested", id=ident)
        return iv

    def _save(self, iv: Intervention) -> None:
        d = self.workdir / "interventions"
        d.mkdir(parents=True, exist_ok=True)
        (d / f"{iv.id}.json").write_text(json.dumps(asdict(iv), indent=2, default=str))

    def _work(self) -> None:
        while not self._stopping.is_set():
            try:
                ident = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue
            iv = self.interventions[ident]
            if iv.status != QUEUED:
                continue
            # The canary slot is the standby engine while a promotion is in
            # probation, and paused means paused: wait, keeping queue order.
            while (self.probation is not None or self.policy["paused"]) \
                    and not self._stopping.is_set() and not self._cancel[ident].is_set():
                time.sleep(1.0)
            if self._stopping.is_set() or iv.status != QUEUED:
                continue
            try:
                with self._canary_lock:
                    self._run(iv)
            except Exception as exc:  # noqa: BLE001 - one bad intervention must not kill the worker
                iv.status, iv.detail = "failed_to_execute", f"{type(exc).__name__}: {exc}"
            iv.finished = time.time()
            self._save(iv)
            self.event("intervention.finished", id=iv.id, status=iv.status,
                       effect=(iv.verdict or {}).get("effect"))
            if iv.status == "improved":
                try:
                    self._consider(iv)
                except Exception as exc:  # noqa: BLE001
                    iv.detail = f"improved; promotion check failed: {type(exc).__name__}: {exc}"
                    self.event("promotion.error", id=iv.id, error=str(exc))
                self._save(iv)

    def _run(self, iv: Intervention) -> None:
        from gitm.experiment.evaluate import evaluate
        from gitm.serve.vllm import check_serve_args
        from gitm.server.canary import (
            CanaryApplicator,
            ReplaySampler,
            build_contract,
            candidate_args,
            load_window,
            write_plan,
        )

        cancel = self._cancel[iv.id]
        canary, stable = self.engines["canary"], self.engines["stable"]
        iv.status, iv.started = RUNNING, time.time()
        self.event("intervention.started", id=iv.id)

        stable_args = list(stable.spec.serve_args)
        iv.candidate_args = candidate_args(stable_args, iv.knobs)
        checks = check_serve_args(["vllm", "serve", self.cfg.model, *iv.candidate_args])
        if any(c.status == "fail" for c in checks):
            iv.status, iv.detail = REJECTED, checks[0].detail
            return

        window = self._window_for_test()
        if window is None:
            iv.status = REJECTED
            iv.detail = (f"no recorded window with at least {self.policy['min_window_requests']}"
                         " requests yet; send traffic through the proxy first")
            return
        iv.window = str(window)
        trace, regime = load_window(window, max_requests=self.policy["max_replay_requests"],
                                    speed=self.policy["replay_speed"])
        out = self.workdir / "interventions" / iv.id
        plan = write_plan(trace, out / "replay.jsonl")
        contract = build_contract(intervention_id=iv.id, knobs=iv.knobs,
                                  stable_args=stable_args, trace=trace, regime=regime,
                                  policy=self.policy, submitter=iv.submitter, note=iv.note)
        (out / "contract.json").write_text(contract.model_dump_json(indent=2))

        # The canary must start the test on stable's configuration: the
        # evaluator's snapshot is what it restores to.
        applicator = CanaryApplicator(
            canary, startup_timeout_s=self.cfg.startup_timeout_s, cancel=cancel,
            on_change=lambda args: self.event("canary.relaunched", id=iv.id, serve_args=args))
        if canary.spec.serve_args != stable_args or not canary.alive:
            applicator.restore(stable_args)

        def on_sample(arm, index, s):
            iv.progress.append({"arm": arm, "index": index, "wall_s": round(s.wall_s, 1),
                                "correctness": s.correctness,
                                **{k: s.metrics.get(k) for k in
                                   (self.policy["primary_metric"]["name"], "output_throughput",
                                    "p99_ttft_ms", "p99_itl_ms", "stable_rps")}})
            self._save(iv)
            self.event("intervention.sample", id=iv.id, arm=arm, index=index)

        def stable_load(t0: float, t1: float) -> float:
            try:
                done = [r for r in self.proxy_call("GET", "/admin/stats", params={"since": t0})
                        if r["t_end"] <= t1 and r["upstream"] == stable.base_url]
            except Exception:
                return float("nan")
            return len(done) / max(t1 - t0, 1e-6)

        sampler = ReplaySampler(canary, plan, regime, result_dir=out / "bench",
                                tokenizer=self.cfg.model, cancel=cancel, on_sample=on_sample,
                                stable_load=stable_load)
        verdict = evaluate(contract, applicator, sampler,
                           provenance={"window": str(window), "policy": self.policy,
                                       "stable_args": stable_args,
                                       "candidate_args": iv.candidate_args},
                           accelerators_held=1)
        (out / "verdict.json").write_text(verdict.model_dump_json(indent=2))
        iv.verdict = verdict.model_dump(mode="json")
        iv.status = STOPPED if cancel.is_set() else verdict.state
        iv.detail = verdict.reason
        if verdict.rollback == "kept":
            # Milestone 2: an improved candidate stays on the canary, waiting for
            # promotion. Nothing reaches production without it.
            self.event("canary.holds_candidate", id=iv.id, serve_args=canary.spec.serve_args)

    def _window_for_test(self) -> Path | None:
        """The newest closed recording window big enough to measure with.

        The open window is closed first when it already qualifies, so a test can
        use the traffic that just arrived rather than waiting for a rotation.
        """
        need = self.policy["min_window_requests"]

        def size(p: Path) -> int:
            with p.open() as fh:
                return sum(1 for line in fh if line.strip())

        try:
            state = self.proxy_call("GET", "/admin/state")
            windows = [Path(p) for p in self.proxy_call("GET", "/admin/recordings")]
            if (state.get("recorded") or 0) and not any(size(p) >= need for p in windows):
                self.proxy_call("POST", "/admin/rotate")
                windows = [Path(p) for p in self.proxy_call("GET", "/admin/recordings")]
        except Exception:
            windows = sorted(self.record_dir.glob("window-*.jsonl"))
        for p in reversed(windows):
            if size(p) >= need:
                return p
        return None
