"""Rank what to try next from where the engine's time actually goes.

Without this the explorer tries catalogue levers in catalogue order. With it,
gitm traces stable's configuration on the canary, replaying recorded traffic,
and ranks the catalogue the way the batch loop does (``gitm run``, phase 3):

* the kernel trace, against the model's predicted graph, gives each op's time
  above its floor (:func:`~gitm.optimizer.monitor.recoverable_by_op`). A lever
  aimed at an op already at its floor is dropped; one aimed at time above a
  floor is ranked by how much;
* the engine's own gauges over the same window (running, waiting, KV cache,
  preemptions, from ``/metrics``) give scheduler causes
  (:func:`~gitm.optimizer.scheduler_attribution.scheduler_causes`). A lever a
  cause argues for ranks ahead of one nothing argues for.

Every number is the batch loop's: the graph comes from
``_execution_graph_basis``, the residuals, causes and order from the same
functions, so the server and ``gitm run`` cannot rank one trace two ways. What
differs is only where the inputs come from. The batch loop reads its in-process
engine; a ``vllm serve`` engine is another process, so the model's config is
read from its checkpoint and the batch from ``/metrics``.

A profile describes one stable configuration on one kind of traffic. It goes
stale when either moves (a promotion, a revert, a traffic change), and the
controller takes a new one before exploring further.

The analysis runs in a subprocess (``python -m gitm.server.profile``): pricing
the graph asks the device for its peaks, and the controller should not hold a
CUDA context or import the planner for a few seconds of work.
"""

from __future__ import annotations

import contextlib
import json
import os
import threading
import time
from collections.abc import Iterator
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any

#: ``vllm:num_preemptions_total`` is cumulative; the others are gauges.
_PREEMPTIONS = ("vllm:num_preemptions_total", "vllm_num_preemptions_total")


class MetricsSeries:
    """The engine's scheduler gauges, sampled from ``/metrics`` over a window.

    The batch loop reads the same quantities off its in-process scheduler
    (:class:`~gitm.tracer.vllm_stats.SchedulerStatsSampler`); a served engine
    publishes them as Prometheus gauges instead. Turned into the same
    :class:`~gitm.tracer.vllm_stats.SchedulerSample` so one summary and one set
    of causes serve both.
    """

    def __init__(self, base_url: str, *, interval_s: float = 0.5,
                 max_num_seqs: int | None = None):
        self.base_url = base_url
        self.interval_s = interval_s
        self.max_num_seqs = max_num_seqs
        self.samples: list[Any] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._t0 = 0
        self._t0_wall = 0

    def __enter__(self) -> MetricsSeries:
        self._t0, self._t0_wall = time.perf_counter_ns(), time.time_ns()
        self._thread = threading.Thread(target=self._run, name="gitm-profile-metrics",
                                        daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc: Any) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=10)

    def _run(self) -> None:
        from gitm.serve.metrics import fetch_metrics, parse_prometheus

        while not self._stop.is_set():
            snap = parse_prometheus(fetch_metrics(self.base_url, timeout=5.0))
            if snap:
                self.samples.append(sample_from_metrics(
                    snap, t_ns=time.perf_counter_ns() - self._t0,
                    max_num_seqs=self.max_num_seqs))
            self._stop.wait(self.interval_s)

    def summary(self):
        from gitm.tracer.vllm_stats import summarize

        return summarize(list(self.samples), t0_wall_ns=self._t0_wall,
                         max_num_seqs=self.max_num_seqs)


def sample_from_metrics(snap: dict[str, float], *, t_ns: int,
                        max_num_seqs: int | None = None):
    """One ``/metrics`` scrape as a scheduler sample."""
    from gitm.serve.metrics import _family
    from gitm.tracer.vllm_stats import SchedulerSample

    running, waiting = _family(snap, "running"), _family(snap, "waiting")
    preempt = next((snap[k] for k in _PREEMPTIONS if k in snap), None)

    def as_int(v: float | None) -> int | None:
        return None if v is None else int(v)

    return SchedulerSample(
        t_ns=t_ns, num_running=as_int(running), num_waiting=as_int(waiting),
        num_unfinished=(None if running is None else int(running + (waiting or 0))),
        preemptions_cumulative=as_int(preempt),
        gpu_cache_usage=_family(snap, "kv_cache_usage"),
        batch_occupancy=(running / max_num_seqs
                         if running is not None and max_num_seqs else None))


# ----------------------------------------------------------------- analysis


def engine_stand_in(model: str, resolved: dict[str, str] | None = None) -> Any:
    """What the batch loop's planner reads off an engine, for a served model.

    ``_execution_graph_basis`` reads ``engine.model_config.hf_config`` and the
    planner context its dtype, context length and world size. A served engine
    is another process, so the config comes from the checkpoint (the files it
    loaded) and the rest from the configuration the engine itself resolved at
    startup (:meth:`~gitm.server.supervisor.EngineProcess.resolved_config`):
    ``--dtype auto`` is decided there, not in the flags. ``None`` when the
    checkpoint config cannot be read offline; the graph then says it is a
    default, and nothing is gated on its floors.
    """
    from transformers import AutoConfig

    try:
        hf = AutoConfig.from_pretrained(model, local_files_only=True)
    except Exception:  # noqa: BLE001 - unreadable config: the graph says so
        return None
    r = resolved or {}

    def as_int(key: str) -> int | None:
        try:
            return int(r[key])
        except (KeyError, ValueError):
            return None

    tp = as_int("tensor_parallel_size") or 1
    return SimpleNamespace(
        model_config=SimpleNamespace(hf_config=hf, dtype=r.get("dtype"),
                                     max_model_len=as_int("max_seq_len")),
        parallel_config=SimpleNamespace(tensor_parallel_size=tp, world_size=tp))


def analyze(trace: Any, *, model: str, resolved: dict[str, str] | None = None,
            sched: Any = None, library: list[Any] | None = None) -> dict[str, Any]:
    """Rank the catalogue for ``trace``, as the batch loop's phase 3 does.

    Returns a JSON-able dict: the ranking (one row per lever, rejected ones
    with their reason), where the time went, the causes, and the basis the
    floors were priced on, so a ranking can be checked against its inputs.
    """
    from gitm.agents.policy import Policy, select_interventions
    from gitm.kernels.library import load_library
    from gitm.optimizer.monitor import recoverable_by_op, residuals
    from gitm.optimizer.scheduler_attribution import scheduler_causes
    from gitm.planner.context import build_planner_context, hardware_spec_for
    from gitm.scheduler.loop import (
        _batch_config_from_stats,
        _execution_graph_basis,
        _motivation,
    )

    kernels = trace.kernels()
    device_s = sum(max(0, k.end_ns - k.start_ns) for k in kernels) / 1e9
    if not kernels:
        raise ValueError("the trace holds no kernels, so there is nothing to rank on. "
                         "The collector saw no GPU work in the window: too short a "
                         "window (CUPTI only hands over full buffers), or the engine "
                         "was not traced")

    engine = engine_stand_in(model, resolved)
    pctx = build_planner_context(engine)
    hw = hardware_spec_for(pctx.peak)
    batch, batch_source = _batch_config_from_stats(sched)
    graph, family, default_why = _execution_graph_basis(engine, hw, batch)
    priced = default_why is None and getattr(pctx, "peak", None) is not None
    res = residuals(trace, graph)
    recoverable = recoverable_by_op(res) if priced else None

    causes = scheduler_causes(sched)
    motivated = _motivation(causes, [])
    specs = library if library is not None else load_library(workload="vllm-decode")
    ranked = select_interventions(
        trace, specs, Policy(require_qualification_commit=True, use_history=False),
        top_n=len(specs), ctx=pctx.gate, recoverable=recoverable, motivated=motivated)

    by_op: dict[str, float] = {}
    by_kernel: dict[str, float] = {}
    from gitm.optimizer.deviation import observed_op

    labelled = 0.0
    for k in kernels:
        d = max(0, k.end_ns - k.start_ns) / 1e9
        # The op the residuals pair it with: its NVTX layer range when the
        # capture has one, else its name.
        op = observed_op(k.name, getattr(k, "range_op", None)) or "unattributed"
        by_op[op] = by_op.get(op, 0.0) + d
        by_kernel[k.name] = by_kernel.get(k.name, 0.0) + d
        if getattr(k, "range_op", None):
            labelled += d

    def top(d: dict[str, float], n: int) -> list[dict[str, Any]]:
        return [{"name": name, "s": round(s, 4), "share": round(s / device_s, 4)}
                for name, s in sorted(d.items(), key=lambda kv: -kv[1])[:n]]

    eff = getattr(graph, "batch", None)
    return {
        "kernels": len(kernels),
        "device_s": round(device_s, 4),
        "dropped_records": getattr(trace, "dropped_records", None),
        # Share of GPU time whose kernels carry a layer range. Low means the
        # engine pushed none (compiled, or vLLM's V2 runner), and op-level
        # targeting is blind.
        "labelled_share": round(labelled / device_s, 4),
        "ops": top(by_op, 12),
        "top_kernels": top(by_kernel, 10),
        "graph": {"family": family, "hardware": hw.name, "gpu_sku": pctx.sku,
                  "batch": getattr(eff, "batch", None), "batch_source": batch_source or "default",
                  "total_pred_s": graph.total_pred_s,
                  "floors_priced": priced,
                  "not_priced_because": None if priced else (
                      default_why or "no GPU peaks resolved for this device")},
        "recoverable_s": ({op: (None if v is None else round(v, 4))
                           for op, v in recoverable.items()} if recoverable else None),
        "scheduler": asdict(sched) if sched is not None else None,
        "causes": [{"signal": c.signal, "severity": round(c.severity, 3), "note": c.note,
                    "motivates_knobs": c.motivates_knobs} for c in causes],
        "ranking": [{"lever": c.spec.name, "knob": c.spec.knob,
                     "predicted_delta": round(c.predicted_delta, 4),
                     "rejected": c.rejected_reason, "motivated_by": c.motivated_by,
                     "targets": [{"op": op, "recoverable_s": round(s, 4)}
                                 for op, s in c.targets]}
                    for c in ranked],
    }


def why(row: dict[str, Any]) -> str:
    """One line on why a lever ranks where it does."""
    bits = []
    if row.get("motivated_by"):
        bits.append(f"the engine showed {row['motivated_by']}")
    if row.get("targets"):
        t = row["targets"][0]
        bits.append(f"{t['op']} spent {t['recoverable_s']:.2f}s above its floor")
    if not bits:
        bits.append(f"catalogue estimate {row['predicted_delta']:+.1%}")
    return "; ".join(bits)


def order(cands: list[dict[str, Any]], profile: dict[str, Any] | None
          ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """``cands`` in the profile's order, and the ones the profile rules out.

    A candidate whose lever the profile rejected (its ops are at their floor,
    or it does not apply here) is dropped, with the reason. The rest follow the
    profile's ranking; a lever the profile did not rank keeps catalogue order,
    after the ranked ones. Without a profile nothing changes.
    """
    if not profile or not profile.get("ranking"):
        return cands, []
    rows = {r["lever"]: (i, r) for i, r in enumerate(profile["ranking"])}
    kept, dropped = [], []
    for pos, c in enumerate(cands):
        hit = rows.get(c.get("lever"))
        if hit is not None and hit[1]["rejected"]:
            dropped.append(c | {"reason": hit[1]["rejected"]})
            continue
        if hit is not None:
            c = c | {"rank": hit[0] + 1, "why": why(hit[1]),
                     "note": f"{c['note']}; ranked #{hit[0] + 1}: {why(hit[1])}"}
        kept.append((hit[0] if hit is not None else len(rows) + pos, pos, c))
    return [c for _, _, c in sorted(kept, key=lambda t: (t[0], t[1]))], dropped


# ----------------------------------------------------------------- capture


@contextlib.contextmanager
def collecting(trace_out: Path) -> Iterator[None]:
    """Point this process's environment at a traced engine's shards, briefly.

    :mod:`gitm.tracer.injection` finds the collector through this process's
    environment. Restored on exit, so nothing started meanwhile (the replay's
    ``vllm bench serve``, another engine) inherits it and adds a shard of its own.
    """
    from gitm.tracer import injection

    env = injection.run_env(trace_out)
    saved = {k: os.environ.get(k) for k in env}
    os.environ.update(env)
    try:
        yield
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def open_window(trace_out: Path) -> int | None:
    """Arm the collector; the record-clock reading the window starts at.

    :func:`gitm.tracer.capture`'s opening half. Its closing half merges every
    shard in the calling process, and a profile's shards outgrow a controller:
    4.8 GB from 2.5 minutes of a 0.5B model, eager with layer ranges, merged
    into 17 GB of Python objects, and the kernel killed the server (and the
    editor beside it) twice. So the window is split: the controller only arms
    and disarms, and the merge runs in the analysis subprocess.
    """
    from gitm.tracer import injection

    with collecting(trace_out):
        injection.clear_stale_shards()
        start = injection.clock_now()
        injection.arm()
    return start


def close_window(trace_out: Path) -> int | None:
    """Wait out in-flight buffers and disarm; the clock reading the window ends at."""
    from gitm.tracer import injection

    with collecting(trace_out):
        end = injection.clock_now()
        injection.settle()
        injection.disarm()
    return end


#: Peak memory of merging and analysing a trace, per byte of shard (measured 8.3x).
MEMORY_PER_SHARD_BYTE = 10


def shard_bytes(trace_out: Path) -> int:
    return sum(p.stat().st_size for p in Path(trace_out).parent.glob(Path(trace_out).name + ".*")
               if p.suffix != ".arm")


def load_window_trace(trace_out: Path, start: int | None, end: int | None) -> Any:
    """Merge the shards into one trace, refusing what this machine cannot hold.

    Merged and analysed, a trace peaks at about 9x its shards in memory
    (1.08 GB of shards, 9 GB on the laptop). Past a tenth of the memory
    available, the load is refused with how to shrink it rather than left to
    the kernel's OOM killer, which picks the biggest process it finds.
    """
    from gitm.tracer import injection
    from gitm.tracer.schema import Trace

    size, avail = shard_bytes(trace_out), _mem_available()
    if avail is not None and size * MEMORY_PER_SHARD_BYTE > avail:
        raise MemoryError(f"the profile's shards are {size / 1e9:.1f} GB and this machine has "
                          f"{avail / 1e9:.1f} GB free; lower profile_window_s")
    with collecting(trace_out):
        events, dropped = injection.read_shards_with_loss(start, end)
    return Trace(workload_id="gitm-server-profile", fingerprint="-", run_id="-",
                 device_count=1, vendor=injection.active_vendor() or "nvidia",
                 captured_at_ns=time.time_ns(), duration_ns=0, events=events,
                 dropped_records=dropped)


def _mem_available() -> int | None:
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024
    except OSError:
        pass
    return None


#: What a traced engine adds to stable's flags, as the batch loop's traced
#: engine does (gitm.workloads, _build_engine). Eager, because vLLM's layerwise
#: ranges fire only on the uncompiled model; the ranges, because by name alone a
#: GEMM maps to no op (on the laptop, 89% of GPU time was unattributed). A
#: profile only says where time goes; every keep or reject is still an A/B of
#: untraced engines, so none of this reaches a measurement.
TRACED_FLAGS = ("--enforce-eager", "--enable-layerwise-nvtx-tracing")
#: vLLM 0.30 registers those ranges only in its V1 model runner. V2, its default
#: for Qwen2 and DeepSeek-V3, accepts the flag and pushes nothing: the first
#: server profile recorded CUB's ranges and none of the model's.
TRACED_ENV = {"VLLM_USE_V2_MODEL_RUNNER": "0"}


def traced_args(stable_args: list[str]) -> list[str]:
    return [*stable_args, *(f for f in TRACED_FLAGS if f not in stable_args)]


def take_profile(*, canary: Any, stable_args: list[str], window: Path, out: Path,
                 policy: dict[str, Any], tokenizer: str, startup_timeout_s: float,
                 max_num_seqs: int | None, on_change=None) -> dict[str, Any]:
    """Trace stable's configuration on the canary, replaying a recorded window.

    The canary is relaunched traced and warmed up, then replays the window at
    its recorded speed. Kernels are collected for ``profile_window_s`` of that
    replay, after ``profile_skip_s`` of ramp-up: a profile needs where time goes
    under steady load, not every kernel, and the window bounds what is written
    whatever the traffic. ``/metrics`` is sampled over the same window. The
    canary is relaunched untraced on stable's flags afterwards, whatever
    happened. Returns where the capture is; :func:`run_analysis` ranks it.
    """
    from gitm.server.canary import load_window, warm_up, write_plan
    from gitm.traffic.runner import run_replay

    on_change = on_change or (lambda args: None)
    trace_out = out / "shards" / "trace"
    spec = canary.spec
    added_env = {k: v for k, v in TRACED_ENV.items()
                 if k not in spec.env and k not in os.environ}
    trace, regime = load_window(window, max_requests=policy["max_replay_requests"], speed=1.0)
    plan = write_plan(trace, out / "replay.jsonl")

    def relaunch(args: list[str], traced: bool) -> None:
        canary.stop()
        spec.serve_args = args
        spec.trace_out = trace_out if traced else None
        spec.trace_nvtx = traced
        for k, v in added_env.items():
            if traced:
                spec.env[k] = v
            else:
                spec.env.pop(k, None)
        canary.start()
        on_change(args)
        canary.wait_healthy(startup_timeout_s)
        warm_up(canary)    # before the window: nothing is armed yet

    result: dict[str, Any] = {"trace_out": str(trace_out), "sched": str(out / "scheduler.json"),
                              "requests": len(trace.requests)}
    try:
        relaunch(traced_args(stable_args), traced=True)
        result["resolved"] = canary.resolved_config()
        replay: dict[str, Any] = {}

        def run() -> None:
            replay["run"] = run_replay(plan, model=canary.served_model or spec.model,
                                       base_url=canary.base_url, tokenizer=tokenizer,
                                       result_dir=out / "bench", regime=regime,
                                       timeout_s=policy["max_wall_clock_s"])

        sender = threading.Thread(target=run, name="gitm-profile-replay", daemon=True)
        sender.start()
        sender.join(policy["profile_skip_s"])
        with MetricsSeries(canary.base_url, max_num_seqs=max_num_seqs) as series:
            t0 = time.time()
            result["start"] = open_window(trace_out)
            sender.join(policy["profile_window_s"])
            result["end"] = close_window(trace_out)
            result["window_s"] = round(time.time() - t0, 1)
        sender.join()
        run = replay.get("run")
        if run is None or not run.ok:
            why = run.summary() if run is not None else "the replay raised"
            raise RuntimeError(f"replay failed: {why}")
        sched = series.summary()
        Path(result["sched"]).write_text(json.dumps(asdict(sched) if sched.n_samples else None))
        result.update(shard_bytes=shard_bytes(trace_out), metrics_samples=sched.n_samples)
    finally:
        # Never skipped: a canary left traced would carry the collector, and
        # vLLM's V1 runner, into every A/B after it.
        relaunch(list(stable_args), traced=False)
    return result


def run_analysis(capture: dict[str, Any], *, model: str, out: Path, gpus: str | None,
                 timeout_s: float = 900.0) -> dict[str, Any]:
    """Merge the shards and :func:`analyze` them in a subprocess on stable's GPUs.

    The shards are deleted afterwards, kept or not: they are the bulk of a
    profile (gigabytes) and the ranking is what it is for.
    """
    import shutil
    import subprocess
    import sys

    from gitm.tracer import injection

    env = os.environ.copy()
    for var in (injection.ENV_LIB, injection.ENV_ROCP, injection.ENV_OUT):
        env.pop(var, None)
    if gpus is not None:
        env["CUDA_VISIBLE_DEVICES"] = gpus
    try:
        p = subprocess.run([sys.executable, "-m", "gitm.server.profile",
                            json.dumps(capture), model, str(out)],
                           env=env, capture_output=True, text=True, timeout=timeout_s)
    finally:
        shutil.rmtree(Path(capture["trace_out"]).parent, ignore_errors=True)
    if p.returncode != 0:
        tail = (p.stderr or p.stdout).strip().splitlines()
        raise RuntimeError(tail[-1] if tail else f"analysis exited {p.returncode}")
    return json.loads(out.read_text())


def _main(argv: list[str]) -> int:
    import warnings

    from gitm.tracer.vllm_stats import SchedulerStatsSummary

    warnings.filterwarnings("ignore")
    capture, model, out = json.loads(argv[0]), argv[1], argv[2]
    trace = load_window_trace(Path(capture["trace_out"]), capture.get("start"),
                              capture.get("end"))
    raw = json.loads(Path(capture["sched"]).read_text())
    sched = SchedulerStatsSummary(**raw) if raw else None
    result = analyze(trace, model=model, resolved=capture.get("resolved"), sched=sched)
    Path(out).write_text(json.dumps(result, indent=2, default=str))
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(_main(sys.argv[1:]))
