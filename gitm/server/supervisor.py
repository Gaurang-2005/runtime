"""Run ``vllm serve`` engines, each with its own environment, port and GPUs.

:func:`gitm.serve.vllm.launch_and_capture` runs one server and sets its tracing
variables on *this* process's ``os.environ``, which every child then inherits.
Two engines cannot be launched that way with different settings, so an engine
here gets a copied environment of its own, with the collector variables cleared
by the same :func:`~gitm.serve.vllm.apply_tracing_env` the capture path uses.

The serving flags (``serve_args``) are kept apart from the ones this module adds
to place the engine (``--host``, ``--port``, ``--gpu-memory-utilization``): the
first are the configuration being tuned and compared, the second are where it
runs. That is the same split :data:`gitm.optimizer.harness_results.LAUNCH_ONLY_FLAGS`
draws, so comparing two engines never counts their different ports as a change.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from gitm.serve.vllm import _arg_of, apply_tracing_env, served_model_name, shutdown, wait_healthy

LAUNCH_MARKER = "=== gitm launch:"


def _vllm_executable() -> str:
    """The ``vllm`` beside this interpreter, so an un-activated venv still finds it."""
    local = Path(sys.executable).parent / "vllm"
    if local.exists():
        return str(local)
    return shutil.which("vllm") or "vllm"


@dataclass
class EngineSpec:
    """What to run and where.

    ``serve_args`` is the configuration (everything after ``vllm serve <model>``
    that changes how the engine behaves). Placement lives in the other fields.
    """

    role: str                     # "stable" | "canary" | "standby"
    model: str
    port: int
    serve_args: list[str] = field(default_factory=list)
    gpus: str | None = None       # CUDA_VISIBLE_DEVICES for this engine
    mem_fraction: float | None = None
    env: dict[str, str] = field(default_factory=dict)

    def argv(self) -> list[str]:
        cmd = [_vllm_executable(), "serve", self.model, *self.serve_args,
               "--host", "127.0.0.1", "--port", str(self.port)]
        if self.mem_fraction is not None and _arg_of(self.serve_args,
                                                     "--gpu-memory-utilization") is None:
            cmd += ["--gpu-memory-utilization", f"{self.mem_fraction:g}"]
        return cmd

    def config_argv(self) -> list[str]:
        """The configuration as ``vllm serve`` argv, without placement: what is compared."""
        return ["vllm", "serve", self.model, *self.serve_args]


class EngineProcess:
    """One ``vllm serve`` process: start, wait for health, stop."""

    #: Models an engine has come up healthy on in this process. Every file vLLM
    #: reads for them is then in the local cache, yet each relaunch still asks
    #: the Hub for the repo's file list, and a DNS blip there kills the start
    #: ("Temporary failure in name resolution"). That was a canary relaunch on
    #: a laptop on flaky Wi-Fi; on a revert it would leave nothing serving.
    #: Later launches of these models run with the Hub offline.
    loaded_models: set[str] = set()

    def __init__(self, spec: EngineSpec, log_dir: str | Path):
        self.spec = spec
        self.log_dir = Path(log_dir)
        self.proc: subprocess.Popen | None = None
        self.started_at: float | None = None
        self.healthy_at: float | None = None
        self.served_model: str | None = None

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.spec.port}"

    @property
    def log_path(self) -> Path:
        return self.log_dir / f"{self.spec.role}-{self.spec.port}.log"

    @property
    def alive(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def environment(self) -> dict[str, str]:
        env = os.environ.copy()
        # The engine shells out to tools installed beside this interpreter (vLLM
        # JIT-builds kernels with `ninja`). From an activated venv they are on
        # PATH; from `.venv/bin/gitm` they are not, and the engine dies at
        # warmup with "No such file or directory: 'ninja'".
        bindir = str(Path(sys.executable).parent)
        if bindir not in env.get("PATH", "").split(os.pathsep):
            env["PATH"] = os.pathsep.join(filter(None, [bindir, env.get("PATH")]))
        if self.spec.model in EngineProcess.loaded_models and "HF_HUB_OFFLINE" not in env:
            env["HF_HUB_OFFLINE"] = "1"
        env.update(self.spec.env)
        # Untraced: the injection variables are read once, at CUDA init, from
        # whatever the child inherits, so they are removed rather than left unset.
        apply_tracing_env(env, None, nvtx=False, no_trace=True)
        if self.spec.gpus is not None:
            env["CUDA_VISIBLE_DEVICES"] = self.spec.gpus
        return env

    def start(self) -> None:
        if self.alive:
            raise RuntimeError(f"{self.spec.role} engine already running (pid {self.proc.pid})")
        self.log_dir.mkdir(parents=True, exist_ok=True)
        log = self.log_path.open("ab")
        # One log per slot holds every relaunch; the marker scopes root_cause()
        # to this launch, so an old failure is never reported for a new one.
        log.write(f"\n{LAUNCH_MARKER} {self.spec.role} {time.strftime('%Y-%m-%d %H:%M:%S')} "
                  f"{' '.join(self.spec.serve_args)}\n".encode())
        log.flush()
        # Own session: the engine is a process tree (EngineCore plus a worker per
        # rank), and shutdown() signals the whole group. It also means a crash of
        # the controller does not take a serving engine with it.
        self.proc = subprocess.Popen(self.spec.argv(), env=self.environment(), stdout=log,
                                     stderr=subprocess.STDOUT, start_new_session=True)
        log.close()
        self.started_at = time.time()
        self.healthy_at = None

    def wait_healthy(self, timeout_s: float = 1800.0) -> None:
        if self.proc is None:
            raise RuntimeError("engine was never started")
        try:
            wait_healthy(self.base_url, self.proc, timeout_s, self.log_path)
        except RuntimeError as exc:
            cause = self.root_cause()
            if cause:
                raise RuntimeError(f"vllm serve did not start: {cause} (log: {self.log_path})"
                                   ) from exc
            raise
        self.healthy_at = time.time()
        EngineProcess.loaded_models.add(self.spec.model)
        self.served_model = served_model_name(self.base_url, self.spec.model)

    def stop(self) -> None:
        if self.proc is not None:
            shutdown(self.proc)

    def root_cause(self) -> str | None:
        """The engine's own reason for dying, from its log, or None.

        vLLM ends a failed start with a generic "Engine core initialization
        failed. See root cause above"; the reason (e.g. "0.0 GiB KV cache
        available") is an earlier EngineCore error line. Reported instead of the
        log's last lines, which only ever showed the generic traceback.
        """
        import re

        try:
            text = self.log_path.read_text(errors="replace")
        except OSError:
            return None
        text = text[text.rfind(LAUNCH_MARKER):] if LAUNCH_MARKER in text else text[-500_000:]
        hits = re.findall(r"\b((?:[A-Z][A-Za-z]*Error|torch\.OutOfMemoryError): .+)", text)
        hits = [h for h in hits if "Engine core initialization failed" not in h]
        return hits[-1].strip()[:500] if hits else None

    def resolved_config(self) -> dict[str, str]:
        """The configuration vLLM actually resolved, from this engine's own log.

        Some defaults (prefix caching, chunked prefill) are decided by vLLM at
        startup, so neither the flags nor ``EngineArgs`` can say what is on.
        The engine prints the resolved values once, in its "Initializing a V1
        LLM engine ... with config:" line; this reads the simple ``key=value``
        pairs from the most recent such line. Empty when there is none yet.
        """
        import re

        try:
            text = self.log_path.read_text(errors="replace")
        except OSError:
            return {}
        lines = re.findall(r"Initializing a V1 LLM engine.*", text)
        if not lines:
            return {}
        return dict(re.findall(r"(?:^|, |config: )([a-z_]+)=([^,{}\[\]()]+?)(?=, |$)",
                               lines[-1]))

    def info(self) -> dict[str, Any]:
        return {
            "role": self.spec.role,
            "model": self.spec.model,
            "served_model": self.served_model,
            "url": self.base_url,
            "pid": self.proc.pid if self.proc else None,
            "alive": self.alive,
            "gpus": self.spec.gpus,
            "mem_fraction": self.spec.mem_fraction,
            "serve_args": list(self.spec.serve_args),
            "started_at": self.started_at,
            "healthy_at": self.healthy_at,
            "startup_s": (round(self.healthy_at - self.started_at, 1)
                          if self.healthy_at and self.started_at else None),
            "log": str(self.log_path),
        }
