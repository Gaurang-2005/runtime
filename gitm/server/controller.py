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
    #: Browser origins allowed to call the API (the dashboard runs on its own port).
    dashboard_origins: tuple[str, ...] = ("http://127.0.0.1:7080", "http://localhost:7080")
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
    #: Stable's flags this was tested against: "tried" is per stable configuration.
    stable_args: list[str] | None = None
    #: The traffic it was measured on (a Regime as a dict, at real speed): "tried"
    #: is also per kind of traffic.
    regime: dict[str, Any] | None = None
    progress: list[dict[str, Any]] = field(default_factory=list)
    verdict: dict[str, Any] | None = None
    rules: list[dict[str, Any]] | None = None
    #: Who approved (or declined) this win, when someone did.
    decided_by: str | None = None

    def brief(self) -> dict[str, Any]:
        v = self.verdict or {}
        from gitm.server.traffic import brief

        r = brief(self.regime)
        return {"id": self.id, "knobs": self.knobs, "status": self.status,
                "submitter": self.submitter, "created": self.created,
                "finished": self.finished, "detail": self.detail,
                "effect": v.get("effect"), "interval": v.get("interval"),
                "samples": len(self.progress), "traffic": r["label"] if r else None}


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
        # The queue, in the order tests will run. A list rather than a FIFO so
        # an operator can see it, reorder it and pin a test to the front.
        self._order: list[str] = []
        self._qlock = threading.Lock()
        self._wake = threading.Event()
        # Assisted / autonomous session (None: manual). See gitm.server.autonomy.
        self.session = None
        from gitm.server.traffic import TrafficWatch

        self.watch = TrafficWatch(threshold=self.policy["traffic_change_distance"],
                                  steady_windows=self.policy["traffic_steady_windows"])
        self._watcher: threading.Thread | None = None
        self._cancel: dict[str, threading.Event] = {}
        self._worker: threading.Thread | None = None
        self._stopping = threading.Event()
        # Promotion state, kept on disk so a restart knows what was promoted,
        # what was reverted (and is therefore blocked), and when.
        self.state: dict[str, Any] = {"last_promotion_t": None, "last_revert_t": None,
                                      "promotions": [], "reverted": []}
        self.probation: dict[str, Any] | None = None
        self.simulator = None
        self._serve_defaults: dict[str, int] | None = None
        self._swap_lock = threading.Lock()
        # Held by whoever is relaunching or measuring on the canary, so a test
        # and a restore never relaunch it under each other.
        self._canary_lock = threading.Lock()
        self._audit = None
        # The latest kernel profile of stable's configuration (gitm.server.profile),
        # what the explorer ranks by while it is fresh.
        self.profile: dict[str, Any] | None = None
        self._profile_requested: str | None = None
        self._profiling: str | None = None      # id of the profile being taken
        self._ruled_out: list[dict[str, Any]] = []

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
        path = self.workdir / "proxy-admin.sock"
        # A Unix socket path is capped at 108 bytes; a deep --scratch went past
        # it and the proxy died at bind ("AF_UNIX path too long"). Too long, it
        # moves to the temp dir under a name unique to this workdir.
        if len(str(path).encode()) <= 100:
            return path
        import hashlib
        import tempfile

        key = hashlib.sha256(str(self.workdir.resolve()).encode()).hexdigest()[:16]
        return Path(tempfile.gettempdir()) / f"gitm-proxy-{key}.sock"

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
        self._load_history()
        self._load_profile()
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
        from gitm.server.simulate import Simulator

        self.simulator = Simulator(f"http://127.0.0.1:{self.cfg.port}",
                                   stable.served_model or self.cfg.model)

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
        if self.cfg.record:
            self._watcher = threading.Thread(target=self._watch_traffic, name="gitm-traffic",
                                             daemon=True)
            self._watcher.start()

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
            "mode": self.mode,
            "session": self.session.summary(time.time()) if self.session else None,
            "queue": self.queue_view(),
            "traffic": self.watch.summary(),
            "promotions": self.state["promotions"][-5:],
            "profile": self.profile_brief(),
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
        from gitm.server.autonomy import Session

        path = self.workdir / "state.json"
        if path.exists():
            try:
                self.state.update(json.loads(path.read_text()))
            except (OSError, json.JSONDecodeError):
                pass
        self.watch.load_state(self.state.get("traffic"))
        sess = self.state.get("session")
        if sess:
            try:
                self.session = Session(**sess)
            except TypeError:
                self.session = None

    def _save_state(self) -> None:
        self.state["traffic"] = self.watch.to_state()
        self.state["session"] = asdict(self.session) if self.session else None
        path = self.workdir / "state.json"
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.state, indent=2, default=str))
        tmp.replace(path)

    def _load_history(self) -> None:
        """Finished interventions from earlier runs of this server, as results memory.

        What was tested, on what traffic, with what outcome, survives a restart,
        so "already tried for this traffic" still holds. Anything that was queued
        or running when the controller stopped is marked stopped, not resumed.
        """
        d = self.workdir / "interventions"
        if not d.is_dir():
            return
        fields = set(Intervention.__dataclass_fields__)
        for f in sorted(d.glob("*.json")):
            try:
                data = json.loads(f.read_text())
                iv = Intervention(**{k: v for k, v in data.items() if k in fields})
            except (OSError, json.JSONDecodeError, TypeError):
                continue
            if iv.status in (QUEUED, RUNNING):
                iv.status, iv.detail = STOPPED, "the controller restarted before it finished"
            self.interventions.setdefault(iv.id, iv)

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
        from gitm.server.autonomy import promotes_itself

        if self.policy["require_approval"] or not promotes_itself(self.mode):
            iv.status = "awaiting_approval"
            iv.detail = (f"every promotion rule passed; {self.mode} mode waits for approval "
                         "(POST /interventions/{id}/promote)")
            self.event("promotion.awaiting_approval", id=iv.id, mode=self.mode)
            return
        self.promote(iv, cause="auto: autonomous mode, every promotion rule passed")

    # ------------------------------------------------------------------ approval

    APPROVABLE = ("awaiting_approval", "improved")

    def approve(self, iv: Intervention, *, by: str) -> dict[str, Any]:
        """A person approves a win. Promote it, putting it back on the canary if needed.

        While a win waits, gitm may test something else, which relaunches the
        canary with other flags; promoting then failed ``canary_ready`` even
        though the result was good. So the canary is relaunched with the approved
        flags first (already proven, ~35 s on a small model), then every rule is
        checked again. If a test holds the canary right now, the approval is kept
        and carried out as soon as that test finishes.
        """
        if iv.status not in self.APPROVABLE:
            raise ValueError(f"{iv.id} is {iv.status}; only a win awaiting approval can be "
                             "promoted")
        if not iv.candidate_args:
            raise ValueError(f"{iv.id} has no tested configuration to promote")
        iv.decided_by = by
        self.audit.record("approve", iv.id, by)
        if self._canary_lock.acquire(blocking=False):
            try:
                return self._promote_approved(iv)
            finally:
                self._canary_lock.release()
        iv.status = "approved"
        iv.detail = f"approved by {by}; promotes when the test running on the canary finishes"
        self._save(iv)
        self.event("promotion.approved", id=iv.id, by=by, pending=True)
        self._wake.set()
        return {"status": "approved", "pending": True, "detail": iv.detail}

    def decline(self, iv: Intervention, *, by: str, reason: str = "") -> Intervention:
        """A person turns a win down. Nothing changes in production."""
        if iv.status not in (*self.APPROVABLE, "approved"):
            raise ValueError(f"{iv.id} is {iv.status}; only a win awaiting approval can be "
                             "declined")
        iv.status, iv.decided_by = "declined", by
        iv.detail = f"declined by {by}" + (f": {reason}" if reason else "")
        self._save(iv)
        self.audit.record("decline", iv.id, by, reason=reason)
        self.event("promotion.declined", id=iv.id, by=by)
        canary = self.engines.get("canary")
        if canary is not None and canary.spec.serve_args == iv.candidate_args:
            self._restore_canary_async(cause=f"{iv.id} declined")
        return iv

    def _promote_approved(self, iv: Intervention) -> dict[str, Any]:
        """Put the approved flags on the canary if they are gone, then promote.

        Caller holds the canary lock. Rules failing here (traffic moved, the
        verdict aged out) leave the win waiting, with the reason.
        """
        from gitm.server.canary import warm_up

        canary = self.engines["canary"]
        if not canary.alive or canary.spec.serve_args != iv.candidate_args:
            self.event("canary.relaunched", id=iv.id, serve_args=iv.candidate_args,
                       reason="restoring the approved configuration")
            canary.stop()
            canary.spec.serve_args = list(iv.candidate_args)
            canary.start()
            canary.wait_healthy(self.cfg.startup_timeout_s)
            warm_up(canary)
        try:
            return self.promote(iv, cause=f"approved: {iv.decided_by or 'api'}")
        except ValueError as exc:
            iv.status = "awaiting_approval"
            iv.detail = f"approved, but not promoted: {exc}"
            self._save(iv)
            self.event("promotion.refused", id=iv.id, reason=str(exc))
            raise

    def _carry_out_approvals(self) -> None:
        """Worker: promote approvals that were waiting for the canary to be free."""
        if self.probation is not None:
            return
        pending = [i for i in self.interventions.values() if i.status == "approved"]
        if not pending:
            return
        with self._canary_lock:
            try:
                self._promote_approved(pending[0])
            except Exception:  # noqa: BLE001 - the reason is on the intervention
                pass

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
            baseline, baseline_source = self._probation_baseline(iv, old)
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
            record["probation_baseline"] = {"decode_tok_s": baseline, "source": baseline_source}
            # Stable is now tuned for the traffic this change won on. The old
            # baseline is kept so a revert puts it back with the old configuration.
            traffic_before = self.watch.to_state()
            self.probation = {"intervention": iv.id, "started": now,
                              "traffic_before": traffic_before,
                              "ends": now + self.policy["probation_s"],
                              "baseline_decode_tok_s": baseline,
                              "baseline_source": baseline_source, "record": record,
                              "watch": promotion.Probation(baseline, self.policy)}
            self.watch.validate(iv.regime, how=f"promoted {iv.id} {iv.knobs}")
            self._save_state()
            self.event("promotion.swapped", id=iv.id, drained=swap.get("drained"),
                       baseline_decode_tok_s=baseline, baseline_source=baseline_source)
        threading.Thread(target=self._watch_probation, name="gitm-probation", daemon=True).start()
        return record

    def _probation_baseline(self, iv: Intervention, old: EngineProcess
                            ) -> tuple[float | None, str]:
        """The old engine's per-request decode speed, for probation to compare with.

        Live traffic on the old engine is the right reference, but the A/B is
        run while stable is quiet (the undisturbed check), so the minutes just
        before a swap often have none: the first full promotion on the laptop
        started probation with no speed baseline at all and guarded on errors
        only. So: the last 5 minutes if they hold enough requests, else the last
        30; failing both, the verdict's own baseline arm (1000 / mean TPOT), which
        was measured on the replayed load rather than the live one.
        """
        from gitm.server import promotion

        need = self.policy["min_window_requests"]
        for lookback in (300, 1800):
            try:
                before = [s for s in self.proxy_call("GET", "/admin/stats",
                                                     params={"since": time.time() - lookback})
                          if s["upstream"] == old.base_url]
            except Exception:
                before = []
            if len(before) >= need:
                speed = promotion.decode_speed(before)
                if speed:
                    return speed, f"live traffic on the old engine, last {lookback // 60} min"
        metric = self.policy["primary_metric"]["name"]
        tpots = [r["metrics"][metric] for r in (iv.verdict or {}).get("raw", [])
                 if r["arm"] == "baseline" and not r.get("warmup")
                 and r["metrics"].get(metric, 0) > 0]
        if metric == "mean_tpot_ms" and tpots:
            return 1000.0 / (sum(tpots) / len(tpots)), "the A/B baseline arm (1000 / mean TPOT)"
        return None, "none available: probation guards on errors only"

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
            # The old configuration is back, and so is the traffic it was validated
            # on; if live traffic differs from that, the watcher reports a change.
            self.watch.load_state(p.get("traffic_before"))
            self.watch.threshold = self.policy["traffic_change_distance"]
            if self.watch.current and self.watch.reference:
                from gitm.server.traffic import distance

                self.watch.last_distance = distance(self.watch.reference, self.watch.current)
                self.watch.changed = bool(self.watch.last_distance is not None
                                          and self.watch.last_distance > self.watch.threshold)
            now = time.time()
            p["record"]["outcome"] = f"reverted: {reason}"
            from gitm.server.promotion import knob_key

            iv = self.interventions.get(p["intervention"])
            knobs = iv.knobs if iv else {}
            self.state["reverted"].append({"t": now, "intervention": p["intervention"],
                                           "knobs": knob_key(knobs), "reason": reason,
                                           "regime": iv.regime if iv else None})
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
               note: str = "", front: bool = False) -> Intervention:
        """Queue an A/B of ``knobs`` against the stable configuration.

        Refused up front (ValueError) when there is no canary, or when stable
        already runs every requested value: an A/B of a configuration against
        itself is an A/A, and would be reported as a lever that did nothing.
        ``front`` puts it at the head of the queue.
        """
        from gitm.server.canary import unchanged_knobs

        if "canary" not in self.engines:
            raise ValueError("this server runs without a canary, so it cannot test anything")
        if not knobs:
            raise ValueError("an intervention needs at least one knob")
        stable = self.engines["stable"]
        resolved = stable.resolved_config() if hasattr(stable, "resolved_config") else {}
        same = unchanged_knobs(stable.spec.serve_args, knobs, resolved)
        if same:
            raise ValueError(f"stable already runs {same} at the requested value")
        iv = Intervention(id=uuid.uuid4().hex[:10], knobs=dict(knobs), submitter=submitter,
                          note=note, created=time.time())
        self.interventions[iv.id] = iv
        self._cancel[iv.id] = threading.Event()
        self._save(iv)
        with self._qlock:
            if front:
                self._order.insert(0, iv.id)
            else:
                self._order.append(iv.id)
        self._wake.set()
        self.event("intervention.queued", id=iv.id, knobs=iv.knobs, submitter=submitter,
                   front=front)
        return iv

    def stop_intervention(self, ident: str) -> Intervention:
        iv = self.interventions[ident]
        self._cancel[ident].set()
        with self._qlock:
            if ident in self._order:
                self._order.remove(ident)
        if iv.status == QUEUED:
            iv.status, iv.finished, iv.detail = STOPPED, time.time(), "stopped before it ran"
            self._save(iv)
        self.event("intervention.stop_requested", id=ident)
        return iv

    def _save(self, iv: Intervention) -> None:
        d = self.workdir / "interventions"
        d.mkdir(parents=True, exist_ok=True)
        (d / f"{iv.id}.json").write_text(json.dumps(asdict(iv), indent=2, default=str))

    # ------------------------------------------------------------------ queue

    def queue_view(self) -> list[dict[str, Any]]:
        """The queue in run order, with why each item may be waiting."""
        from gitm.server.autonomy import AUTO_SUBMITTERS, in_schedule

        now = time.time()
        with self._qlock:
            order = list(self._order)
        out = []
        for pos, ident in enumerate(order):
            iv = self.interventions.get(ident)
            if iv is None or iv.status != QUEUED:
                continue
            auto = iv.submitter in AUTO_SUBMITTERS
            waits = None
            if auto and self.session and not in_schedule(self.session.schedule, now):
                waits = "outside the session's schedule"
            out.append({"position": pos, "id": iv.id, "knobs": iv.knobs,
                        "submitter": iv.submitter, "note": iv.note, "auto": auto,
                        "waits": waits})
        return out

    def move(self, ident: str, position: int) -> list[dict[str, Any]]:
        """Move a queued test to ``position`` (0 = next to run)."""
        iv = self.interventions.get(ident)
        if iv is None or iv.status != QUEUED:
            raise ValueError("only a queued test can be moved")
        with self._qlock:
            self._order.remove(ident)
            self._order.insert(max(0, min(int(position), len(self._order))), ident)
        self.event("queue.moved", id=ident, position=position)
        self._wake.set()
        return self.queue_view()

    def _next_runnable(self) -> Intervention | None:
        """The first queued test allowed to start now.

        A test someone queued runs when it reaches the front. A test gitm queued
        itself waits for an active session's schedule, and is skipped over while
        it waits, so a hand-queued test behind it is not held up.
        """
        from gitm.server.autonomy import AUTO_SUBMITTERS, in_schedule

        now = time.time()
        with self._qlock:
            self._order = [i for i in self._order
                           if self.interventions.get(i) and self.interventions[i].status == QUEUED]
            for ident in self._order:
                iv = self.interventions[ident]
                if iv.submitter in AUTO_SUBMITTERS:
                    if self.session is None or not in_schedule(self.session.schedule, now):
                        continue
                self._order.remove(ident)
                return iv
        return None

    def _work(self) -> None:
        while not self._stopping.is_set():
            self._wake.clear()
            self._end_session_if_over()
            self._carry_out_approvals()
            iv = None
            if self.probation is None and not self.policy["paused"]:
                if self._maybe_profile():
                    continue
                iv = self._next_runnable()
                if iv is None:
                    self._maybe_explore()
                    iv = self._next_runnable()
            if iv is None:
                self._wake.wait(1.0)
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
            if iv.verdict is not None and iv.status != "improved":
                # Not a promotion candidate, but the rules still say what a win
                # would have run into (e.g. the undisturbed check), so "why not?"
                # has an answer on every outcome.
                from gitm.server.promotion import as_dicts

                try:
                    iv.rules = as_dicts(self.rules_for(iv))
                except Exception:  # noqa: BLE001
                    pass
            if iv.status == "improved":
                try:
                    self._consider(iv)
                except Exception as exc:  # noqa: BLE001
                    iv.detail = f"improved; promotion check failed: {type(exc).__name__}: {exc}"
                    self.event("promotion.error", id=iv.id, error=str(exc))
                self._save(iv)

    # ------------------------------------------------------------------ autonomy

    @property
    def mode(self) -> str:
        return self.session.mode if self.session else "manual"

    def start_session(self, mode: str, **limits: Any) -> dict[str, Any]:
        from gitm.server.autonomy import new_session

        if "canary" not in self.engines:
            raise ValueError("this server runs without a canary, so it cannot test anything")
        self.session = new_session(mode, **limits)
        self._save_state()
        self.audit.record("session.start", "-", "api", mode=mode, **{
            k: v for k, v in limits.items() if v is not None})
        self.event("session.started", mode=mode,
                   **{k: v for k, v in limits.items() if v is not None})
        self._wake.set()
        return self.session.summary(time.time())

    def stop_session(self, *, cancel_current: bool = False, reason: str = "stopped") -> dict:
        """Back to manual. Stable is left as it is; gitm's own queued picks are dropped."""
        from gitm.server.autonomy import AUTO_SUBMITTERS

        ended = self.session
        self.session = None
        dropped = []
        for iv in list(self.interventions.values()):
            if iv.submitter not in AUTO_SUBMITTERS:
                continue
            if iv.status == QUEUED:
                self.stop_intervention(iv.id)
                dropped.append(iv.id)
            elif iv.status == RUNNING and cancel_current:
                self.stop_intervention(iv.id)
        self._save_state()
        self.audit.record("session.stop", "-", "api", reason=reason,
                          cancel_current=cancel_current)
        self.event("session.stopped", reason=reason, dropped=dropped,
                   cancel_current=cancel_current,
                   tests_started=ended.tests_started if ended else 0)
        return {"mode": "manual", "dropped": dropped, "reason": reason}

    def _end_session_if_over(self) -> None:
        if self.session is not None:
            why = self.session.exhausted(time.time())
            if why:
                self.stop_session(reason=why)

    def _results(self) -> list[dict[str, Any]]:
        """Every intervention as a result row, for the traffic-aware memory."""
        from gitm.server.promotion import knob_key

        return [{"id": i.id, "knob_key": knob_key(i.knobs), "knobs": i.knobs,
                 "status": i.status, "effect": (i.verdict or {}).get("effect"),
                 "regime": i.regime, "stable_args": i.stable_args, "finished": i.finished}
                for i in self.interventions.values()]

    def _auto_submit(self, knobs: dict[str, Any], submitter: str, note: str,
                     front: bool = False) -> Intervention | None:
        """Queue a test gitm chose itself, counted against the session's budget.

        Whatever chose it (explorer, re-check, playbook, rediscovery), it must be
        inside the session's scope: the scope is the set of knobs gitm may touch.
        """
        if self.session is None:
            return None
        if not self.session.allows(knobs, self.policy):
            self.event("explore.skipped", knobs=knobs, reason="outside the session's scope")
            return None
        if self.session.exhausted(time.time()):
            return None
        try:
            iv = self.submit(knobs, submitter=submitter, note=note, front=front)
        except ValueError as exc:
            self.event("explore.skipped", knobs=knobs, reason=str(exc))
            return None
        self.session.tests_started += 1
        self._save_state()
        return iv

    def explore_preview(self, n: int = 5) -> list[dict[str, Any]]:
        """What gitm would pick next, in order, without queueing anything."""
        return self._candidates()[:n]

    def _candidates(self) -> list[dict[str, Any]]:
        from gitm.server.explore import candidates, model_facts
        from gitm.server.promotion import knob_key
        from gitm.server.traffic import near, tried_for

        if "stable" not in self.engines:
            return []
        stable = self.engines["stable"]
        stable_args = list(stable.spec.serve_args)
        policy = self.session.scoped_policy(self.policy) if self.session else self.policy
        threshold = self.policy["traffic_change_distance"]
        live = self.watch.current
        results = self._results()
        # "Tried" is for this stable *and* this kind of traffic; a revert blocks a
        # change for traffic like the traffic it was reverted on.
        tried = {r["knob_key"] for r in results
                 if tried_for(results, r["knob_key"], stable_args, live, threshold)}
        blocked = {r["knobs"] for r in self.state["reverted"]
                   if r.get("regime") is None or live is None
                   or near(r.get("regime"), live, threshold)}
        facts = model_facts(self.cfg.model, stable_args)
        todo = candidates(stable_args, tried=tried, blocked=blocked, policy=policy,
                          resolved=stable.resolved_config(), facts=facts,
                          defaults=self.serve_defaults())
        from gitm.server.profile import order

        # Ranked by where stable's time goes, while the profile describes this
        # stable on this traffic; catalogue order otherwise.
        fresh = self.profile if self.profile_stale() is None else None
        todo, self._ruled_out = order(todo, (fresh or {}).get("analysis"))
        return [c | {"knob_key": knob_key(c["knobs"])} for c in todo]

    def serve_defaults(self) -> dict[str, int]:
        """The batch limits vLLM picks for stable's GPU when its flags leave them unset."""
        from gitm.server.explore import model_facts, serve_defaults

        if self._serve_defaults is None:
            stable = self.engines["stable"]
            tp = model_facts(self.cfg.model, list(stable.spec.serve_args))["tp"]
            self._serve_defaults = serve_defaults(tp, stable.spec.gpus)
        return self._serve_defaults

    def _maybe_explore(self) -> Intervention | None:
        """In a session, queue the next catalogue candidate when nothing is waiting."""
        from gitm.server.autonomy import in_schedule, picks_itself

        if (self.session is None or not picks_itself(self.session.mode)
                or self.policy["paused"] or self.probation is not None
                or "canary" not in self.engines
                or not in_schedule(self.session.schedule, time.time())
                or any(i.status in (QUEUED, RUNNING) for i in self.interventions.values())):
            return None
        try:
            todo = self._candidates()
        except Exception as exc:  # noqa: BLE001 - a catalogue problem must not stop serving
            self.event("explore.error", error=str(exc))
            return None
        stable_args = list(self.engines["stable"].spec.serve_args)
        if not todo:
            key = (tuple(stable_args), self.watch.current_window)
            if getattr(self, "_explore_done_for", None) != key:
                self._explore_done_for = key
                self.event("explore.exhausted", stable_args=stable_args)
            self._accept_if_revalidated()
            return None
        nxt = todo[0]
        iv = self._auto_submit(nxt["knobs"], "explorer", nxt["note"])
        if iv is not None:
            self.event("explore.queued", id=iv.id, lever=nxt["lever"], knobs=nxt["knobs"])
        return iv

    def _accept_if_revalidated(self) -> bool:
        """In a session: after a traffic change, stable is re-validated once re-tested.

        A promotion already validates the traffic it was tested on. When a
        session's re-tests find nothing better, that is evidence stable is right
        for the new traffic, so gitm records it as validated itself. A win still
        waiting for approval keeps it open until a person decides. Manual mode
        runs no re-tests, so nothing is validated; the watcher still re-arms on
        steady new traffic, so later changes are caught.
        """
        from gitm.server.autonomy import picks_itself

        if not (self.watch.changed and picks_itself(self.mode)
                and self.probation is None
                and not any(i.status in (QUEUED, RUNNING, "awaiting_approval", "approved")
                            for i in self.interventions.values())):
            return False
        self.watch.validate(self.watch.current,
                            how=f"re-tested in a {self.mode} session; nothing beat stable")
        self._save_state()
        self.audit.record("traffic.validated", "-", self.mode,
                          reason="re-tested on the new traffic; stable stays")
        self.event("traffic.validated", reason=f"{self.mode}: re-tested everything worth "
                   "trying on the new traffic; stable's configuration stays")
        return True

    # ------------------------------------------------------------------ profiling

    @property
    def profile_dir(self) -> Path:
        return self.workdir / "profiles"

    def _load_profile(self) -> None:
        """The newest finished profile from an earlier run, so a restart keeps its ranking."""
        done = []
        for f in self.profile_dir.glob("*/profile.json"):
            try:
                p = json.loads(f.read_text())
            except (OSError, json.JSONDecodeError):
                continue
            if p.get("status") == "running":
                # Its controller died mid-profile; nothing will finish it.
                p["status"], p["detail"] = "interrupted", "the controller stopped before it finished"
                f.write_text(json.dumps(p, indent=2, default=str))
            if p.get("status") == "done":
                done.append(p)
        if done:
            self.profile = max(done, key=lambda p: p.get("finished") or 0)

    def profile_stale(self, profile: dict[str, Any] | None = None) -> str | None:
        """Why the profile no longer describes stable on the traffic arriving, or None.

        A profile is of one stable configuration on one kind of traffic, as a
        verdict is. A promotion or revert changes the first; the traffic watcher
        says when the second has moved.
        """
        from gitm.server.traffic import distance

        p = profile if profile is not None else self.profile
        if p is None or p.get("status") != "done":
            return "no profile of stable yet"
        stable = self.engines.get("stable")
        if stable is not None and p.get("stable_args") != list(stable.spec.serve_args):
            return "stable's configuration changed since it was profiled"
        d = distance(p.get("regime"), self.watch.current)
        if d is not None and d > self.policy["traffic_change_distance"]:
            return f"traffic moved {d:.2f} (limit {self.policy['traffic_change_distance']:g}) " \
                   "since it was profiled"
        return None

    def profile_brief(self) -> dict[str, Any] | None:
        p = self.profile
        if p is None:
            return None
        a = p.get("analysis") or {}
        return {"id": p["id"], "status": p["status"], "finished": p.get("finished"),
                "running": self._profiling, "requested": self._profile_requested,
                "kernels": a.get("kernels"), "stale": self.profile_stale(),
                "top_ops": (a.get("ops") or [])[:4], "causes": [c["signal"] for c in
                                                                a.get("causes") or []]}

    def profile_view(self) -> dict[str, Any]:
        """The latest profile in full, whether it is still in use, and what it rules out."""
        try:
            nxt = self._candidates()[:8]
        except Exception as exc:  # noqa: BLE001 - the view must not fail on a catalogue problem
            nxt = [{"error": str(exc)}]
        return {"profile": self.profile, "stale": self.profile_stale(),
                "running": self._profiling, "requested": self._profile_requested,
                "last_attempt": self.profile_last_attempt, "next": nxt,
                "ruled_out": self._ruled_out}

    def request_profile(self, *, by: str = "api") -> dict[str, Any]:
        if "canary" not in self.engines:
            raise ValueError("this server runs without a canary, so it cannot profile")
        if self._profile_requested is None:
            self._profile_requested = f"requested by {by}"
        self.event("profile.requested", by=by)
        self._wake.set()
        return {"requested": self._profile_requested}

    def _maybe_profile(self) -> bool:
        """Take a profile when one is asked for, or before a session explores on a stale one.

        Asked for, it runs before the next queued test. On its own, only when a
        session would otherwise explore (nothing queued or running) and the
        profile no longer describes stable on this traffic; a failure is not
        retried for the same stable and traffic within ``profile_retry_s``.
        """
        from gitm.server.autonomy import in_schedule, picks_itself

        reason = self._profile_requested
        if reason is None:
            if (not self.policy["profile"] or "canary" not in self.engines
                    or self.session is None or not picks_itself(self.session.mode)
                    or not in_schedule(self.session.schedule, time.time())
                    or any(i.status in (QUEUED, RUNNING) for i in self.interventions.values())):
                return False
            reason = self.profile_stale()
            if reason is None:
                return False
            p = self.profile_last_attempt
            if (p and p.get("status") == "failed"
                    and p.get("stable_args") == list(self.engines["stable"].spec.serve_args)
                    and time.time() - (p.get("finished") or 0) < self.policy["profile_retry_s"]):
                return False
        window = self._window_for_test()
        if window is None:
            if self._profile_requested is not None:
                self._profile_requested = None
                self.event("profile.failed", reason="no recorded traffic to profile on yet")
            return False
        with self._canary_lock:
            self._run_profile(window, reason)
        return True

    profile_last_attempt: dict[str, Any] | None = None

    def _run_profile(self, window: Path, reason: str) -> dict[str, Any]:
        from gitm.server.canary import current_value
        from gitm.server.profile import run_analysis, take_profile
        from gitm.server.traffic import regime_of

        stable, canary = self.engines["stable"], self.engines["canary"]
        stable_args = list(stable.spec.serve_args)
        pid = uuid.uuid4().hex[:10]
        out = self.profile_dir / pid
        out.mkdir(parents=True, exist_ok=True)
        rec: dict[str, Any] = {"id": pid, "status": "running", "reason": reason,
                               "started": time.time(), "finished": None,
                               "stable_args": stable_args, "window": str(window),
                               "regime": regime_of(window, self.policy["max_replay_requests"])[0]
                               .model_dump(mode="json")}
        self._profile_requested, self._profiling = None, pid
        self.event("profile.started", id=pid, reason=reason)
        (out / "profile.json").write_text(json.dumps(rec, indent=2, default=str))
        try:
            # Occupancy is running / max_num_seqs; without the cap no
            # under-filled batch can be seen.
            seqs = (current_value(stable_args, "max_num_seqs")
                    or self.serve_defaults().get("max_num_seqs"))
            cap = take_profile(
                canary=canary, stable_args=stable_args, window=window, out=out,
                policy=self.policy, tokenizer=self.cfg.model,
                startup_timeout_s=self.cfg.startup_timeout_s,
                max_num_seqs=int(seqs) if seqs else None,
                on_change=lambda args: self.event("canary.relaunched", id=pid,
                                                  serve_args=args))
            rec["capture"] = cap
            rec["analysis"] = run_analysis(cap, model=self.cfg.model,
                                           out=out / "ranking.json", gpus=stable.spec.gpus)
            rec["status"] = "done"
        except Exception as exc:  # noqa: BLE001 - recorded on the profile, never raised
            rec["status"], rec["detail"] = "failed", f"{type(exc).__name__}: {exc}"
        rec["finished"], self._profiling = time.time(), None
        (out / "profile.json").write_text(json.dumps(rec, indent=2, default=str))
        self.profile_last_attempt = rec
        if rec["status"] == "done":
            self.profile = rec
            a = rec["analysis"]
            self.event("profile.finished", id=pid, kernels=a["kernels"],
                       top_ops=[o["name"] for o in a["ops"][:3]],
                       causes=[c["signal"] for c in a["causes"]])
        else:
            self.event("profile.failed", id=pid, reason=rec["detail"])
        self.audit.record("profile", pid, "gitm", status=rec["status"], reason=reason,
                          detail=rec.get("detail", ""))
        return rec

    # ------------------------------------------------------------------ traffic

    def _watch_traffic(self) -> None:
        """Close windows once they can be judged, read each new one's regime, react."""
        while not self._stopping.is_set():
            self._stopping.wait(self.policy["traffic_check_s"])
            if self._stopping.is_set():
                return
            try:
                state = self.proxy_call("GET", "/admin/state")
                # Close the open window only once it holds enough requests to
                # judge. Closing on time alone split 0.6 req/s of long documents
                # into windows of 7, 19, 13 and 4 requests, each below the
                # minimum, so the change was never seen.
                if ((state.get("open_rows") or 0) >= self.policy["min_window_requests"]
                        and (state.get("open_age_s") or 0) >= self.policy["traffic_window_s"]):
                    self.proxy_call("POST", "/admin/rotate")
                windows = [Path(p) for p in self.proxy_call("GET", "/admin/recordings")]
            except Exception:
                continue
            for w in windows:
                if str(w) in self.watch.seen:
                    continue
                self.observe_window(w)

    def observe_window(self, window: Path) -> str | None:
        """Read one closed window's traffic. Returns the transition, if any."""
        from gitm.server.traffic import brief, regime_of

        try:
            regime, n = regime_of(window, self.policy["max_replay_requests"])
        except Exception:
            self.watch.seen.append(str(window))
            return None
        if n < self.policy["min_window_requests"]:
            self.watch.seen.append(str(window))
            return None
        self.watch.threshold = self.policy["traffic_change_distance"]
        self.watch.steady_windows = self.policy["traffic_steady_windows"]
        was = self.watch.reference
        change = self.watch.observe(regime, str(window))
        self._save_state()
        if change == "changed":
            self.event("traffic.changed", distance=round(self.watch.last_distance or 0, 3),
                       measured_from=brief(was), now=brief(regime), window=str(window))
            self._react_to_traffic_change(was)
        elif change == "settled":
            self.event("traffic.settled", distance=round(self.watch.last_distance or 0, 3))
        elif change == "rearmed":
            # Detection re-arms on the new traffic; validation does not move.
            self.event("traffic.rearmed", now=brief(regime),
                       validated_on=brief(self.watch.validated))
        return change

    def _react_to_traffic_change(self, old_regime: dict | None) -> list[str]:
        """What gitm queues when traffic moves away from what stable was tuned for.

        Manual mode only reports the change. In a session, in this order:

        1. re-check: the last promoted change reversed, to see whether stable's
           current configuration still beats what it replaced on this traffic;
        2. playbook: changes already proven on traffic like the new one, best
           first: the fast path, still verified on the canary;
        3. rediscover: changes that won or nearly won on the old traffic.

        Everything else becomes untried for the new traffic, so the explorer
        picks it up after these.
        """
        from gitm.server.autonomy import picks_itself
        from gitm.server.canary import current_value
        from gitm.server.traffic import promising_under, proven_for

        if self.session is None or not picks_itself(self.session.mode):
            return []
        queued: list[str] = []
        threshold = self.policy["traffic_change_distance"]
        results = self._results()
        stable_args = list(self.engines["stable"].spec.serve_args)
        seen: set[str] = set()

        def add(knobs, submitter, note):
            from gitm.server.promotion import knob_key

            key = knob_key(knobs)
            if key in seen:
                return
            seen.add(key)
            iv = self._auto_submit(knobs, submitter, note)
            if iv is not None:
                queued.append(iv.id)

        kept = [p for p in self.state["promotions"] if p.get("outcome") == "kept"]
        if kept:
            last = kept[-1]
            back = {k: current_value(last.get("from_args") or [], k) for k in last["knobs"]}
            back = {k: v for k, v in back.items() if v is not None}
            if back:
                add(back, "recheck", f"traffic changed: does {last['knobs']} still beat "
                                     f"what it replaced?")
        for r in proven_for(results, self.watch.current, threshold):
            add(r["knobs"], "playbook", f"proven on traffic like this before ({r['id']})")
        for r in promising_under(results, old_regime, threshold):
            if r.get("stable_args") in (None, stable_args):
                add(r["knobs"], "rediscover", f"won or nearly won on the old traffic ({r['id']})")
        self.event("traffic.requeued", ids=queued)
        return queued

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
        iv.stable_args = stable_args
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
        # Keyed at the traffic's real speed: a replay sped up to stress the
        # canary is still a measurement *about* the traffic that was recorded.
        from gitm.server.traffic import regime_of

        iv.regime = regime_of(window, self.policy["max_replay_requests"])[0].model_dump(
            mode="json")
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
                                    "p99_ttft_ms", "p99_itl_ms", "stable_rps", "remeasured")}})
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
                                stable_load=stable_load, shared_gpu=shares_gpus(stable, canary),
                                max_stable_rps=self.policy["max_shared_gpu_stable_rps"],
                                settle_s=self.policy["settle_s"])
        verdict = evaluate(contract, applicator, sampler,
                           provenance={"window": str(window), "policy": self.policy,
                                       "stable_args": stable_args,
                                       "candidate_args": iv.candidate_args},
                           accelerators_held=1)
        (out / "verdict.json").write_text(verdict.model_dump_json(indent=2))
        iv.verdict = verdict.model_dump(mode="json")
        iv.status = STOPPED if cancel.is_set() else verdict.state
        iv.detail = verdict.reason
        if iv.status in ("improved", "regressed"):
            from gitm.server.canary import arm_spread

            spread = arm_spread(iv.verdict, self.policy["primary_metric"]["name"])
            worst = max((v for v in spread.values() if v is not None), default=None)
            if worst is not None and worst > self.policy["max_arm_spread"]:
                # The laptop's thermal clamp produced a "regressed" this way:
                # identical configurations measured 6.3 and 14.6 ms per token.
                iv.detail = (f"was {iv.status}, but the measurement is unstable: an arm's own "
                             f"samples spread {worst:.0%} (limit "
                             f"{self.policy['max_arm_spread']:.0%}); {verdict.reason}")
                iv.status = "unstable"
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
