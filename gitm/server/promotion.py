"""When a tested candidate may replace the stable engine, and how that is watched.

Whether a change is *good* is the test's decision, made once, by
:func:`gitm.experiment.evaluate.evaluate`: the effect clears ``min_effect``,
the answers match, no request failed, and latency meets the operator's targets
(``latency_targets``). Promotion does not judge the change again. It asks the
questions a verdict cannot answer, because they are about *after* the test:

1. **verdict** — the test said ``improved``;
2. **evidence_current** — the verdict still describes production: traffic like
   the tested window, a recent verdict, a measurement production load did not
   disturb;
3. **canary_ready** — the canary runs exactly the candidate, on stable's GPU
   count and memory budget, so it can carry all of stable's load;
4. **allowed_now** — the operator's change controls: not paused or frozen, an
   allowed knob, not reverted before, and not too soon after the last
   promotion or revert.

Each answer keeps its individual checks as evidence, stored with the promotion
or the refusal, so "why did gitm change production, or not?" has an answer.

This replaced 18 rules, seven of which re-judged the verdict with thresholds of
their own: two places deciding one fact, and the relative latency limit among
them refused a change that met every latency objective.

After a swap, :class:`Probation` watches live traffic. The old engine stays
loaded until probation passes, so a revert is a routing change, not a rebuild.
"""

from __future__ import annotations

import fnmatch
import math
import statistics
import time
from dataclasses import asdict, dataclass, field
from typing import Any

from gitm.safety.autorevert import AutoRevert

PROMOTION_DEFAULTS: dict[str, Any] = {
    # Even in autonomous mode, make every win wait for a person.
    "require_approval": False,
    "max_verdict_age_s": 7200.0,
    # log2 units on each regime axis (1.0 = traffic changed 2x on some axis).
    "max_regime_distance": 1.0,
    # When stable and canary share a GPU, stable's production load competes
    # with the canary's measurement. Above this many stable requests/s during
    # any sample, the measurement is not of the candidate alone. Measured: ~1
    # req/s on a shared RTX 4050 turned CUDA graphs' +21% into +2% (-16..+21%).
    "max_shared_gpu_stable_rps": 0.1,
    "min_interval_s": 1800.0,
    "revert_cooldown_s": 3600.0,
    "paused": False,
    # [[start_hour_utc, end_hour_utc], ...]; no promotion inside any of them.
    "freeze_hours_utc": [],
    "allow_knobs": [],   # empty = every knob; otherwise globs
    "deny_knobs": [],
    # --- probation -----------------------------------------------------------
    "probation_s": 600.0,
    "probation_interval_s": 30.0,
    "probation_window": 5,
    # Fractional drop in per-request decode speed that reverts the promotion.
    "probation_tolerance": 0.05,
    # Any error rate above this in a probation interval reverts immediately.
    "probation_max_error_rate": 0.02,
}


@dataclass
class Rule:
    name: str
    passed: bool
    detail: str
    #: The individual checks behind it: [{"name", "passed", "detail"}].
    checks: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class PromotionContext:
    """Everything the rules read besides the verdict. Built by the controller."""

    policy: dict[str, Any]
    now: float
    verdict_finished: float | None
    tested_regime: Any            # gitm.traffic.regime.Regime
    live_regime: Any | None       # the newest window's regime, if any
    stable_placement: dict[str, Any]
    canary_placement: dict[str, Any]
    shared_gpu: bool = False
    canary_alive: bool = True
    canary_runs_candidate: bool = True
    last_promotion_t: float | None = None
    last_revert_t: float | None = None
    reverted: list[dict[str, Any]] = field(default_factory=list)


def knob_key(knobs: dict[str, Any]) -> str:
    return ",".join(f"{k}={knobs[k]}" for k in sorted(knobs))


def _rule(name: str, checks: list[tuple[str, bool, str]]) -> Rule:
    """One question, answered by its checks; the failing ones lead the detail."""
    bad = [d if len(checks) == 1 else f"{n}: {d}" for n, ok, d in checks if not ok]
    detail = "; ".join(bad) if bad else "; ".join(d for _, _, d in checks)
    return Rule(name, not bad, detail,
                [{"name": n, "passed": bool(ok), "detail": d} for n, ok, d in checks])


def check(knobs: dict[str, Any], verdict: dict[str, Any], ctx: PromotionContext) -> list[Rule]:
    """The four promotion questions, each with its evidence. Promote only if all pass."""
    p = ctx.policy

    # 1. The test's own decision, not re-judged here.
    state = verdict.get("state")
    effect, interval = verdict.get("effect"), verdict.get("interval")
    shown = ("" if effect is None or not interval else
             f" (effect {effect:+.1%}, interval {interval[0]:+.1%}..{interval[1]:+.1%})")
    rules = [_rule("verdict", [("improved", state == "improved",
                                f"{state}: {verdict.get('reason', '')}{shown}")])]

    # 2. Is the verdict still about production as it is now?
    age = None if ctx.verdict_finished is None else ctx.now - ctx.verdict_finished
    current = [("recent", age is not None and age <= p["max_verdict_age_s"],
                f"verdict age {'unknown' if age is None else f'{age:.0f}s'}, "
                f"limit {p['max_verdict_age_s']:.0f}s")]
    if ctx.live_regime is None:
        current.append(("same_traffic", True, "no newer traffic recorded since the test"))
    else:
        from gitm.playbook.match import regime_distance

        d = regime_distance(ctx.tested_regime, ctx.live_regime)
        current.append(("same_traffic", d.linf <= p["max_regime_distance"],
                        f"tested vs live traffic: {d.render()}, limit "
                        f"{p['max_regime_distance']:g}"))
    loads = [(r.get("metrics") or {}).get("stable_rps") for r in verdict.get("raw", [])]
    loads = [x for x in loads if x is not None]
    if not ctx.shared_gpu:
        current.append(("undisturbed", True, "canary has its own GPUs"))
    elif not loads or any(math.isnan(x) for x in loads):
        current.append(("undisturbed", False, "canary shares stable's GPU and stable's load "
                                              "during the test is unknown"))
    else:
        worst = max(loads)
        current.append(("undisturbed", worst <= p["max_shared_gpu_stable_rps"],
                        f"canary shares stable's GPU; stable served up to {worst:.2f} req/s "
                        f"during a sample, limit {p['max_shared_gpu_stable_rps']:g}"))
    rules.append(_rule("evidence_current", current))

    # 3. Can the canary take over as it stands?
    same = (ctx.stable_placement.get("gpus_count") == ctx.canary_placement.get("gpus_count")
            and ctx.stable_placement.get("mem_fraction")
            == ctx.canary_placement.get("mem_fraction"))
    rules.append(_rule("canary_ready", [
        ("runs_candidate", ctx.canary_alive and ctx.canary_runs_candidate,
         "canary is up and running the candidate" if ctx.canary_alive
         and ctx.canary_runs_candidate
         else f"canary alive={ctx.canary_alive}, runs candidate={ctx.canary_runs_candidate}"),
        ("same_capacity", same, f"stable {ctx.stable_placement}, canary {ctx.canary_placement}"),
    ]))

    # 4. The operator's change controls.
    hour = time.gmtime(ctx.now).tm_hour
    frozen = [w for w in p["freeze_hours_utc"] if _in_window(hour, w)]
    denied = [k for k in knobs if any(fnmatch.fnmatch(k, g) for g in p["deny_knobs"])]
    unlisted = ([k for k in knobs if not any(fnmatch.fnmatch(k, g) for g in p["allow_knobs"])]
                if p["allow_knobs"] else [])
    before = [r for r in ctx.reverted if r.get("knobs") == knob_key(knobs)]
    since_promo = None if ctx.last_promotion_t is None else ctx.now - ctx.last_promotion_t
    since_revert = None if ctx.last_revert_t is None else ctx.now - ctx.last_revert_t
    rules.append(_rule("allowed_now", [
        ("not_paused", not p["paused"], "paused by operator" if p["paused"] else "running"),
        ("not_frozen", not frozen, f"inside freeze window {frozen[0]} (UTC hour {hour})"
         if frozen else f"UTC hour {hour}, no freeze"),
        ("knob_allowed", not denied and not unlisted, "knobs allowed" if not (denied or unlisted)
         else f"denied {denied}, not allowed {unlisted}"),
        ("not_reverted_before", not before, "never reverted" if not before else
         f"reverted {len(before)}x before: {before[-1].get('reason', '')}"),
        ("promotion_interval", since_promo is None or since_promo >= p["min_interval_s"],
         "no earlier promotion" if since_promo is None else
         f"{since_promo:.0f}s since the last promotion, need {p['min_interval_s']:.0f}s"),
        ("revert_cooldown", since_revert is None or since_revert >= p["revert_cooldown_s"],
         "no earlier revert" if since_revert is None else
         f"{since_revert:.0f}s since the last revert, need {p['revert_cooldown_s']:.0f}s"),
    ]))
    return rules


def _in_window(hour: int, window: list[int]) -> bool:
    start, end = int(window[0]) % 24, int(window[1]) % 24
    return start <= hour < end if start <= end else hour >= start or hour < end


def all_pass(rules: list[Rule]) -> bool:
    return all(r.passed for r in rules)


def as_dicts(rules: list[Rule]) -> list[dict[str, Any]]:
    return [asdict(r) for r in rules]


# ----------------------------------------------------------------- probation


def decode_speed(stats: list[dict[str, Any]]) -> float | None:
    """Median per-request decode speed (tokens/s after the first token).

    Per request rather than in aggregate, so a quiet minute and a busy one are
    judged on the same footing: offered load changes how many tokens are served,
    not how fast each request decodes.
    """
    speeds = []
    for s in stats:
        out, ttft, lat = s.get("output_tokens"), s.get("ttft_s"), s.get("latency_s")
        if s.get("ok") and out and out > 1 and ttft is not None and lat and lat > ttft:
            speeds.append((out - 1) / (lat - ttft))
    return statistics.median(speeds) if speeds else None


def error_rate(stats: list[dict[str, Any]]) -> float | None:
    return (sum(1 for s in stats if not s.get("ok")) / len(stats)) if stats else None


class Probation:
    """Watch the promoted engine against the speed the old one had.

    ``baseline`` is the old engine's decode speed over the minutes before the
    swap. Each interval's speed is fed to :class:`~gitm.safety.autorevert.AutoRevert`;
    an error rate above the limit reverts at once, without waiting for a window.
    """

    def __init__(self, baseline: float | None, policy: dict[str, Any]):
        self.policy = policy
        self.baseline = baseline
        self.guard = (AutoRevert(baseline, tolerance=policy["probation_tolerance"],
                                 window=policy["probation_window"])
                      if baseline and baseline > 0 and math.isfinite(baseline) else None)
        self.observations: list[dict[str, Any]] = []

    def observe(self, interval_stats: list[dict[str, Any]], now: float) -> str | None:
        """A reason to revert, or None to carry on."""
        err = error_rate(interval_stats)
        speed = decode_speed(interval_stats)
        obs: dict[str, Any] = {"t": now, "requests": len(interval_stats), "error_rate": err,
                               "decode_tok_s": speed}
        self.observations.append(obs)
        if err is not None and err > self.policy["probation_max_error_rate"]:
            return f"error rate {err:.1%} above {self.policy['probation_max_error_rate']:.1%}"
        if self.guard is not None and speed is not None:
            d = self.guard.observe(speed)
            obs["autorevert"] = d.reason
            if d.should_revert:
                return f"decode speed {d.reason} (baseline {self.baseline:.1f} tok/s)"
        return None
