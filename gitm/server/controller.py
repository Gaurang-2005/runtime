"""The controller: owns the engines and the proxy, and answers the API.

Milestone 1 scope: launch the stable engine, put the proxy in front of it with
recording on, and report state. Canary testing and promotion build on this.

Shutdown is deliberate about what it takes down. A clean stop (Ctrl-C on
``gitm run --server``) stops the proxy and every engine this controller started.
A crash stops nothing: the proxy and the engines live in their own sessions, so
traffic keeps flowing to the last stable engine.
"""

from __future__ import annotations

import os
import secrets
import socket
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

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
    startup_timeout_s: float = 1800.0


def port_free(port: int, host: str = "127.0.0.1") -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.bind((host, port))
        except OSError:
            return False
        return True


class Controller:
    def __init__(self, cfg: ServerConfig):
        self.cfg = cfg
        self.workdir = Path(cfg.workdir)
        self.started_at = time.time()
        self.token = secrets.token_urlsafe(24)
        self.engines: dict[str, EngineProcess] = {}
        self.proxy: subprocess.Popen | None = None
        self.events: list[dict[str, Any]] = []

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
        for name, port in (("proxy", self.cfg.port), ("api", self.cfg.api_port),
                           ("stable engine", self.cfg.stable_port)):
            if not port_free(port, self.cfg.host if name != "stable engine" else "127.0.0.1"):
                problems.append(f"port {port} ({name}) is already in use")
        return problems

    def start(self) -> None:
        self.workdir.mkdir(parents=True, exist_ok=True)
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
        return {
            "model": self.cfg.model,
            "uptime_s": round(time.time() - self.started_at, 1),
            "endpoint": f"http://{self.cfg.host}:{self.cfg.port}/v1",
            "engines": {role: e.info() for role, e in self.engines.items()},
            "proxy": proxy,
            "workdir": str(self.workdir),
        }
