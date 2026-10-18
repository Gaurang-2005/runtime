"""Test a candidate configuration on the canary, against recorded traffic.

The whole A/B runs in the canary slot. Arm A is the stable engine's own
configuration, arm B the candidate; both are launched on the canary's GPUs and
replayed the same recorded window, in the order the contract sets (ABBA by
default). The stable engine is never measured, so production is never disturbed,
and both arms get the same hardware.

This module supplies the two seams :func:`gitm.experiment.evaluate.evaluate`
drives, and builds the contract it reads:

* :class:`CanaryApplicator` — snapshot / apply / restore by relaunching the
  canary ``vllm serve`` with different flags;
* :class:`ReplaySampler` — one ``vllm bench serve`` replay of the window, plus a
  correctness probe: fixed token sequences (probe prompts plus the baseline's
  greedy continuation) are run teacher-forced on each arm, and the arms' top-1
  choices at every position must (nearly) all agree.

The verdict — improved, regressed, inconclusive, invalid, failed to execute — is
the evaluator's, not recomputed here.
"""

from __future__ import annotations

import dataclasses
import json
import threading
import time
import urllib.request
from pathlib import Path
from typing import Any

from gitm.experiment.contract import (
    ArmConfig,
    Budget,
    CorrectnessGate,
    EffectInterval,
    ExperimentContract,
    Hypothesis,
    LatencyGate,
    Metric,
    Protocol,
    Workload,
)
from gitm.experiment.evaluate import Sample
from gitm.kernels.spec import InterventionSpec
from gitm.optimizer.experiment_specs import _with, _without
from gitm.optimizer.vllm_knobs import same_server_value, server_arg
from gitm.serve.vllm import _arg_of
from gitm.server.supervisor import EngineProcess
from gitm.traffic.adapters import read_mooncake
from gitm.traffic.regime import Regime, SourceKind
from gitm.traffic.replay import ReplayPlan, write_timed_trace
from gitm.traffic.runner import run_replay
from gitm.traffic.schema import Trace

#: Fixed prompts for the correctness probe. Short, so the probe costs seconds,
#: and varied, so one lucky match does not stand in for "same model".
PROBES = (
    "The capital of France is",
    "def fibonacci(n):",
    "List three prime numbers:",
    "Translate to German: good morning",
)
PROBE_TOKENS = 32


class Cancelled(RuntimeError):
    """The intervention was stopped through the API."""


# ----------------------------------------------------------------- flags


def flag_for(knob: str) -> str:
    return server_arg(knob, None)[0]


def current_value(serve_args: list[str], knob: str) -> Any:
    """What ``serve_args`` sets ``knob`` to: the string value, True for a bare
    flag, False for ``--no-<flag>``, or None when vLLM's default applies."""
    flag = flag_for(knob)
    if f"--no-{flag[2:]}" in serve_args:
        return False
    for i, arg in enumerate(serve_args):
        if arg.startswith(flag + "="):
            return arg.split("=", 1)[1]
        if arg == flag:
            # A bare flag is followed by the next flag or by nothing. _arg_of
            # would read that next flag as this one's value.
            nxt = serve_args[i + 1] if i + 1 < len(serve_args) else None
            return True if nxt is None or nxt.startswith("--") else nxt
    return None


def candidate_args(serve_args: list[str], knobs: dict[str, Any]) -> list[str]:
    """``serve_args`` with each knob set, replacing whatever the flag held."""
    out = list(serve_args)
    for knob, value in knobs.items():
        flag, rendered = server_arg(knob, value, _arg_of(out, flag_for(knob)))
        negated = f"--no-{flag[2:]}"
        out = [a for a in out if a != negated]
        if value is True:
            out = _with(_without(out, flag, frozenset({flag})), flag, True)
        elif value is False:
            out = [*_without(out, flag, frozenset({flag})), negated]
        else:
            out = _with(out, flag, rendered)
    return out


def unchanged_knobs(serve_args: list[str], knobs: dict[str, Any],
                    resolved: dict[str, str] | None = None) -> list[str]:
    """Knobs the stable configuration already runs at the requested value.

    The flags answer for what they set. For what they leave at vLLM's default,
    ``resolved`` (the engine's own resolved configuration, see
    :meth:`~gitm.server.supervisor.EngineProcess.resolved_config`) answers when
    it reports the knob; otherwise the knob is treated as unknown, not as same.
    """
    same = []
    for knob, value in knobs.items():
        cur = current_value(serve_args, knob)
        if cur is None and resolved and knob in resolved:
            cur = {"True": True, "False": False}.get(resolved[knob], resolved[knob])
        if cur is None:
            continue
        if isinstance(value, bool) or isinstance(cur, bool):
            if cur == value:
                same.append(knob)
        elif same_server_value(str(cur), str(server_arg(knob, value, cur)[1])):
            same.append(knob)
    return same


# ----------------------------------------------------------------- applicator


class CanaryApplicator:
    """The canary as an :class:`~gitm.optimizer.apply.Applicator`.

    Every change is a relaunch: the flags being tested are read once, when vLLM
    builds the engine. ``on_change`` is told each time, so the controller can
    report what the canary is running.
    """

    def __init__(self, engine: EngineProcess, *, startup_timeout_s: float = 1800.0,
                 on_change=None, cancel: threading.Event | None = None):
        self.engine = engine
        self.startup_timeout_s = startup_timeout_s
        self.on_change = on_change or (lambda args: None)
        self.cancel = cancel or threading.Event()

    def snapshot(self) -> list[str]:
        return list(self.engine.spec.serve_args)

    def apply(self, spec: InterventionSpec) -> None:
        self._relaunch(candidate_args(self.engine.spec.serve_args, spec.knob_values))

    def restore(self, snapshot: list[str]) -> None:
        # Never refused for a stop: a stop cancels the test, not its clean-up.
        # Refusing it left "restore failed" and the canary on the candidate.
        self._relaunch(list(snapshot), cancellable=False)

    def measure(self, spec: InterventionSpec) -> float | None:
        return None  # the evaluator measures through the sampler

    def _relaunch(self, serve_args: list[str], cancellable: bool = True) -> None:
        if cancellable and self.cancel.is_set():
            raise Cancelled("stopped before relaunching the canary")
        self.engine.stop()
        self.engine.spec.serve_args = serve_args
        self.engine.start()
        self.on_change(serve_args)
        self.engine.wait_healthy(self.startup_timeout_s)
        warm_up(self.engine)


def warm_up(engine: EngineProcess, *, concurrent: int = 16) -> None:
    """A short burst of short and long requests right after a (re)launch.

    The first replay on a fresh engine ran cold: on the laptop its p99 TTFT
    was 44.6 ms against 37.5 for every later baseline sample, which alone made
    an arm look worse. A few seconds of mixed traffic pays that cost before
    anything is measured, without a whole discarded replay per switch.
    """
    import concurrent.futures as cf

    model = engine.served_model or engine.spec.model
    short = "Summarize the idea of paged attention in one sentence."
    long = " ".join(["The quick brown fox jumps over the lazy dog."] * 220)

    def one(i: int) -> None:
        try:
            _post(engine.base_url + "/v1/completions",
                  {"model": model, "prompt": long if i % 4 == 0 else short,
                   "max_tokens": 32, "temperature": 0.0}, timeout=120)
        except Exception:
            pass  # warm-up is best effort; the measurement will fail loudly if needed

    with cf.ThreadPoolExecutor(concurrent) as pool:
        list(pool.map(one, range(concurrent * 2)))


# ----------------------------------------------------------------- sampler


def _post(url: str, body: dict, timeout: float = 60.0) -> dict:
    req = urllib.request.Request(url, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        raw = r.read()
    return json.loads(raw) if raw else {}


def probe_prompts(engine: EngineProcess) -> list[list[int]]:
    """Each probe prompt plus this engine's greedy continuation, as token ids.

    Made once, on the baseline arm, and then scored identically on every arm, so
    both arms are judged on exactly the same tokens.
    """
    model = engine.served_model or engine.spec.model
    out = []
    for prompt in PROBES:
        r = _post(engine.base_url + "/v1/completions",
                  {"model": model, "prompt": prompt, "max_tokens": PROBE_TOKENS,
                   "temperature": 0.0, "seed": 0, "return_token_ids": True})
        choice = r["choices"][0]
        out.append(list(choice["prompt_token_ids"]) + list(choice["token_ids"]))
    return out


def teacher_forced_top1(engine: EngineProcess, sequences: list[list[int]]) -> list[list[int]]:
    """For every position of every sequence, the token this engine ranks first.

    One prefill pass per sequence (``prompt_logprobs``), so the result does not
    depend on how decoding would have branched: it is the engine's next-token
    choice at each position of a fixed input.
    """
    model = engine.served_model or engine.spec.model
    out = []
    for seq in sequences:
        r = _post(engine.base_url + "/v1/completions",
                  {"model": model, "prompt": seq, "max_tokens": 1, "temperature": 0.0,
                   "prompt_logprobs": 1})
        top = []
        for entry in r["choices"][0]["prompt_logprobs"][1:]:
            best = next((int(t) for t, info in (entry or {}).items()
                         if (info or {}).get("rank") == 1), -1)
            top.append(best)
        out.append(top)
    return out


def agreement(a: list[list[int]], b: list[list[int]]) -> float:
    """Fraction of positions where two engines rank the same token first."""
    pairs = [(x, y) for sa, sb in zip(a, b, strict=True) for x, y in zip(sa, sb, strict=True)]
    return sum(x == y for x, y in pairs) / len(pairs) if pairs else 0.0


class ReplaySampler:
    """One replay of the recorded window against whatever the canary runs now."""

    def __init__(self, engine: EngineProcess, plan: ReplayPlan, regime: Regime, *,
                 result_dir: Path, tokenizer: str, timeout_s: float = 3600.0,
                 cancel: threading.Event | None = None, on_sample=None, stable_load=None,
                 shared_gpu: bool = False, max_stable_rps: float = 0.1, max_retries: int = 2,
                 quiet_wait_s: float = 300.0, settle_s: float = 0.0):
        self.engine = engine
        self.plan = plan
        self.regime = regime
        self.result_dir = result_dir
        self.tokenizer = tokenizer
        self.timeout_s = timeout_s
        self.cancel = cancel or threading.Event()
        self.on_sample = on_sample or (lambda arm, index, sample: None)
        self.stable_load = stable_load
        self.shared_gpu = shared_gpu
        self.max_stable_rps = max_stable_rps
        self.max_retries = max_retries
        self.quiet_wait_s = quiet_wait_s
        self.settle_s = settle_s
        self.sequences: list[list[int]] | None = None
        self.reference: list[list[int]] | None = None

    def sample(self, arm: str, index: int) -> Sample:
        """One measurement, repeated when production load spoiled it.

        On a GPU shared with stable, a sample during which stable served more
        than ``max_stable_rps`` measured the candidate *and* production. On the
        laptop one such sample (stable at 0.54 req/s) turned a no-effect change
        into "+6.6%, CI -14%..+28%". It is thrown away, the sampler waits for
        stable to go quiet, and measures again, up to ``max_retries`` times; the
        last attempt is kept either way, and isolated_measurement judges it.
        """
        attempts = 0
        while True:
            s = self._sample_once(arm, index)
            busy = s.metrics.get("stable_rps", 0.0)
            if (not self.shared_gpu or self.stable_load is None
                    or not busy > self.max_stable_rps or attempts >= self.max_retries):
                s.metrics["remeasured"] = float(attempts)
                self.on_sample(arm, index, s)
                return s
            attempts += 1
            self._wait_for_quiet()

    def _wait_for_quiet(self) -> None:
        deadline = time.time() + self.quiet_wait_s
        while time.time() < deadline and not self.cancel.is_set():
            now = time.time()
            if self.stable_load(now - 10.0, now) <= self.max_stable_rps:
                return
            time.sleep(5.0)

    def _sample_once(self, arm: str, index: int) -> Sample:
        if self.cancel.is_set():
            raise Cancelled("stopped between samples")
        if self.settle_s > 0:
            # Let the machine settle before measuring. On the laptop a replay
            # started straight after another ran 2x slower: the CPU hit 96 C and
            # firmware held it near 450 MHz, while the GPU clock never moved.
            self.cancel.wait(self.settle_s)
            if self.cancel.is_set():
                raise Cancelled("stopped between samples")
        # Every sample starts from an empty prefix cache, so the second sample on
        # an engine is not credited with the first one's cached prefixes. Needs
        # VLLM_SERVER_DEV_MODE=1 on the canary; without it the reset is skipped.
        try:
            _post(self.engine.base_url + "/reset_prefix_cache", {}, timeout=30)
        except Exception:
            pass
        started = time.monotonic()
        t0 = time.time()
        run = run_replay(self.plan, model=self.engine.served_model or self.engine.spec.model,
                         base_url=self.engine.base_url, tokenizer=self.tokenizer,
                         result_dir=self.result_dir, regime=self.regime,
                         timeout_s=self.timeout_s)
        if not run.ok or run.result is None:
            raise RuntimeError(f"replay failed: {run.summary()}; {run.stderr_tail[-400:]}")
        metrics = {k: float(v) for k, v in (run.joined.metrics if run.joined else run.result)
                   .items() if isinstance(v, int | float) and not isinstance(v, bool)}
        if self.stable_load is not None:
            # Production load on the stable engine while this sample ran. On a
            # shared GPU it competes with the measurement; the promotion rules
            # read it (isolated_measurement).
            metrics["stable_rps"] = float(self.stable_load(t0, time.time()))
        # Correctness: both arms are scored the same way, on the same tokens —
        # teacher-forced top-1 choices compared arm to arm. Comparing a prefill
        # pass with free-running decode instead disagreed with itself: on the
        # second GPU run the baseline scored 0.92 against its own greedy text.
        if arm == "baseline" and self.sequences is None:
            self.sequences = probe_prompts(self.engine)
            self.reference = teacher_forced_top1(self.engine, self.sequences)
        correctness = (None if self.reference is None else
                       agreement(teacher_forced_top1(self.engine, self.sequences),
                                 self.reference))
        wall = time.monotonic() - started
        return Sample(metrics=metrics, wall_s=wall, accelerator_s=wall, correctness=correctness)


# ----------------------------------------------------------------- the workload


def load_window(path: Path, *, max_requests: int, speed: float = 1.0) -> tuple[Trace, Regime]:
    """The recorded window as a replayable trace, optionally capped and sped up.

    ``speed`` > 1 compresses arrival times, raising the offered load by that
    factor. Both arms get the same compressed schedule, so the comparison stays
    fair; the regime recorded on the verdict is the compressed one, so the claim
    names the load it was measured at.
    """
    trace = read_mooncake(path, max_rows=max_requests, source="gitm-recording")
    if speed != 1.0:
        reqs = [dataclasses.replace(r, arrival_s=r.arrival_s / speed) for r in trace.requests]
        meta = trace.meta.model_copy(update={"span_s": (trace.meta.span_s or 0) / speed,
                                             "notes": [*trace.meta.notes,
                                                       f"arrivals compressed {speed:g}x"]})
        trace = Trace(meta=meta, requests=reqs)
    regime = Regime.from_trace(trace, source_kind=SourceKind.PRODUCTION)
    return trace, regime


def build_contract(*, intervention_id: str, knobs: dict[str, Any], stable_args: list[str],
                   trace: Trace, regime: Regime, policy: dict[str, Any],
                   submitter: str, note: str = "") -> ExperimentContract:
    base = {k: current_value(stable_args, k) for k in knobs}
    base = {k: ("<vllm default>" if v is None else v) for k, v in base.items()}
    cand = {**base, **knobs}
    metric = policy["primary_metric"]
    return ExperimentContract(
        candidate_id=intervention_id,
        hypothesis=Hypothesis(
            id=intervention_id, submitter=submitter,
            claim=note or f"setting {knobs} improves {metric['name']} on recorded traffic",
            mechanism="submitted for testing; no mechanism claimed",
            expected_effect=EffectInterval(mean=0.0, lo=-1.0, hi=1.0),
            reject_if_effect_below=-1.0),
        intervention=dict(knobs),
        primary_metric=Metric(name=metric["name"], unit=metric.get("unit", ""),
                              direction=metric["direction"]),
        workload=Workload(trace_id=trace.meta.sha256 or "unknown", source=trace.meta.source,
                          operating_point=regime.label(), duration_s=max(trace.meta.span_s or 1.0,
                                                                         1.0)),
        baseline=ArmConfig(label="stable", engine_args=base),
        candidate=ArmConfig(label="candidate", engine_args=cand),
        budget=Budget(max_wall_clock_s=policy["max_wall_clock_s"],
                      max_accelerator_s=policy["max_wall_clock_s"]),
        correctness_gate=CorrectnessGate(method="token_agreement",
                                         reference="teacher-forced top-1 choices of the "
                                                   "baseline arm, position by position, on "
                                                   f"{len(PROBES)} fixed probe prompts",
                                         min_score=policy["min_correctness"]),
        latency_gates=[LatencyGate(metric=g["metric"], max_regression=g["max_regression"])
                       for g in policy["latency_gates"]],
        protocol=Protocol(repetitions=policy["repetitions"], warmup_reps=policy["warmup_reps"],
                          comparison_order="ABBA", pairing="paired"),
    )


def arm_spread(verdict: dict[str, Any], metric: str) -> dict[str, float | None]:
    """Relative spread (stdev / mean) of ``metric`` within each arm's kept samples.

    Both arms of one configuration should measure alike. When one arm's own
    samples disagree by more than the effect being judged, the verdict is about
    the machine, not the change.
    """
    import statistics

    out: dict[str, float | None] = {}
    for arm in ("baseline", "candidate"):
        vals = [r["metrics"][metric] for r in verdict.get("raw", [])
                if r["arm"] == arm and not r.get("warmup") and metric in r.get("metrics", {})]
        out[arm] = (statistics.stdev(vals) / statistics.fmean(vals)
                    if len(vals) >= 2 and statistics.fmean(vals) > 0 else None)
    return out


def write_plan(trace: Trace, out: Path) -> ReplayPlan:
    out.parent.mkdir(parents=True, exist_ok=True)
    return write_timed_trace(trace, out)
