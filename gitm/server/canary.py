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
  correctness probe: the candidate is fed the baseline arm's greedy answers to
  fixed prompts and must rank (nearly) every one of those tokens first.

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


def unchanged_knobs(serve_args: list[str], knobs: dict[str, Any]) -> list[str]:
    """Knobs the stable configuration already runs at the requested value."""
    same = []
    for knob, value in knobs.items():
        cur = current_value(serve_args, knob)
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
        self._relaunch(list(snapshot))

    def measure(self, spec: InterventionSpec) -> float | None:
        return None  # the evaluator measures through the sampler

    def _relaunch(self, serve_args: list[str]) -> None:
        if self.cancel.is_set():
            raise Cancelled("stopped before relaunching the canary")
        self.engine.stop()
        self.engine.spec.serve_args = serve_args
        self.engine.start()
        self.on_change(serve_args)
        self.engine.wait_healthy(self.startup_timeout_s)


# ----------------------------------------------------------------- sampler


def _post(url: str, body: dict, timeout: float = 60.0) -> dict:
    req = urllib.request.Request(url, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        raw = r.read()
    return json.loads(raw) if raw else {}


def reference_tokens(engine: EngineProcess) -> list[tuple[list[int], list[int]]]:
    """(prompt token ids, greedy continuation token ids) for each probe prompt."""
    model = engine.served_model or engine.spec.model
    out = []
    for prompt in PROBES:
        r = _post(engine.base_url + "/v1/completions",
                  {"model": model, "prompt": prompt, "max_tokens": PROBE_TOKENS,
                   "temperature": 0.0, "seed": 0, "return_token_ids": True})
        choice = r["choices"][0]
        out.append((list(choice["prompt_token_ids"]), list(choice["token_ids"])))
    return out


def token_agreement(engine: EngineProcess, refs: list[tuple[list[int], list[int]]]) -> float:
    """Fraction of the reference's greedy tokens this engine also ranks first.

    Teacher-forced: the engine is given each reference prompt *plus* the
    reference continuation, and ``prompt_logprobs`` reports the rank of every
    token in it. A change that only moves numerics (CUDA graphs, another
    attention kernel, a different batch shape) keeps nearly every token at rank
    1 even where free-running greedy output would branch after a near-tie; a
    model that is actually broken does not. Exact match of free-running text
    cannot tell those apart: on the first GPU run, CUDA graphs on Qwen2.5-0.5B
    changed 3 of 4 greedy answers.
    """
    model = engine.served_model or engine.spec.model
    agree = total = 0
    for prompt_ids, cont in refs:
        if not cont:
            continue
        r = _post(engine.base_url + "/v1/completions",
                  {"model": model, "prompt": prompt_ids + cont, "max_tokens": 1,
                   "temperature": 0.0, "prompt_logprobs": 1})
        ranks = r["choices"][0]["prompt_logprobs"][len(prompt_ids):]
        for tok, entry in zip(cont, ranks, strict=False):
            total += 1
            info = (entry or {}).get(str(tok)) or (entry or {}).get(tok)
            agree += bool(info and info.get("rank") == 1)
    return agree / total if total else 0.0


class ReplaySampler:
    """One replay of the recorded window against whatever the canary runs now."""

    def __init__(self, engine: EngineProcess, plan: ReplayPlan, regime: Regime, *,
                 result_dir: Path, tokenizer: str, timeout_s: float = 3600.0,
                 cancel: threading.Event | None = None, on_sample=None):
        self.engine = engine
        self.plan = plan
        self.regime = regime
        self.result_dir = result_dir
        self.tokenizer = tokenizer
        self.timeout_s = timeout_s
        self.cancel = cancel or threading.Event()
        self.on_sample = on_sample or (lambda arm, index, sample: None)
        self.reference: list[str] | None = None

    def sample(self, arm: str, index: int) -> Sample:
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
        run = run_replay(self.plan, model=self.engine.served_model or self.engine.spec.model,
                         base_url=self.engine.base_url, tokenizer=self.tokenizer,
                         result_dir=self.result_dir, regime=self.regime,
                         timeout_s=self.timeout_s)
        if not run.ok or run.result is None:
            raise RuntimeError(f"replay failed: {run.summary()}; {run.stderr_tail[-400:]}")
        metrics = {k: float(v) for k, v in (run.joined.metrics if run.joined else run.result)
                   .items() if isinstance(v, int | float) and not isinstance(v, bool)}
        if arm == "baseline" and self.reference is None:
            self.reference = reference_tokens(self.engine)
        correctness = (None if self.reference is None
                       else token_agreement(self.engine, self.reference))
        wall = time.monotonic() - started
        s = Sample(metrics=metrics, wall_s=wall, accelerator_s=wall, correctness=correctness)
        self.on_sample(arm, index, s)
        return s


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
                                         reference="teacher-forced top-1 agreement with the "
                                                   "baseline arm's greedy tokens on "
                                                   f"{len(PROBES)} fixed probe prompts",
                                         min_score=policy["min_correctness"]),
        latency_gates=[LatencyGate(metric=g["metric"], max_regression=g["max_regression"])
                       for g in policy["latency_gates"]],
        protocol=Protocol(repetitions=policy["repetitions"], warmup_reps=policy["warmup_reps"],
                          comparison_order="ABBA", pairing="paired"),
    )


def write_plan(trace: Trace, out: Path) -> ReplayPlan:
    out.parent.mkdir(parents=True, exist_ok=True)
    return write_timed_trace(trace, out)
