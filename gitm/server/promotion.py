"""When a tested candidate may replace the stable engine, and how that is watched.

Promotion is automatic, so the rules are strict and every one is reported. A
candidate is promoted only when **all** of them pass; the list, with each rule's
evidence, is stored with the promotion (or the refusal), so "why did gitm change
production?" always has an answer.

The rules, in four groups:

* **the evidence** — the verdict is ``improved``; enough paired rounds; the
  interval clears a bar set by this test's own noise; correctness and every
  latency gate passed; no replay request failed; the verdict is recent;
* **the traffic** — the window it was tested on still looks like the traffic
  arriving now (regime distance), so a result from last night's traffic is not
  applied to this morning's;
* **the hardware** — the canary has the same GPUs and memory budget as stable,
  so it can carry all of stable's load once promoted;
* **the operator** — not paused, not inside a freeze window, the knobs allowed,
  not a change that was reverted before, and not too soon after the last
  promotion or revert.

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
    "auto_promote": True,
    "require_approval": False,
    "promote_min_rounds": 3,
    # The interval's lower bound must clear max(min_effect, noise_factor x the
    # baseline arm's own relative spread). The spread is this test's noise floor:
    # the same configuration measured repeatedly on the same canary.
    "min_effect": 0.02,
    "noise_factor": 2.0,
    "max_verdict_age_s": 7200.0,
    # log2 units on each regime axis (1.0 = traffic changed 2x on some axis).
    "max_regime_distance": 1.0,
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
    canary_alive: bool
    canary_runs_candidate: bool
    last_promotion_t: float | None = None
    last_revert_t: float | None = None
    reverted: list[dict[str, Any]] = field(default_factory=list)


def knob_key(knobs: dict[str, Any]) -> str:
    return ",".join(f"{k}={knobs[k]}" for k in sorted(knobs))


def noise_bar(verdict: dict[str, Any], policy: dict[str, Any]) -> tuple[float, float | None]:
    """(bar, relative spread of the baseline arm's primary metric)."""
    metric = None
    raws = [r for r in verdict.get("raw", []) if not r.get("warmup")]
    base = [r["metrics"] for r in raws if r["arm"] == "baseline"]
    if base:
        # The primary metric is the one every sample carries that the contract
        # named; the verdict does not repeat it, so it is passed in the policy.
        metric = policy["primary_metric"]["name"]
    values = [m[metric] for m in base if metric in m] if metric else []
    spread = None
    if len(values) >= 2 and statistics.fmean(values) > 0:
        spread = statistics.stdev(values) / statistics.fmean(values)
    bar = max(policy["min_effect"], policy["noise_factor"] * (spread or 0.0))
    return bar, spread


def check(knobs: dict[str, Any], verdict: dict[str, Any], ctx: PromotionContext) -> list[Rule]:
    """Every promotion rule, evaluated. Promote only if all pass."""
    p = ctx.policy
    rules: list[Rule] = []

    def rule(name: str, ok: bool, detail: str) -> None:
        rules.append(Rule(name, bool(ok), detail))

    # --- evidence
    state = verdict.get("state")
    rule("verdict_improved", state == "improved", f"verdict is {state!r}")
    n = verdict.get("n_candidate", 0)
    rule("enough_rounds", n >= p["promote_min_rounds"],
         f"{n} candidate samples, need {p['promote_min_rounds']}")
    interval = verdict.get("interval")
    bar, spread = noise_bar(verdict, p)
    lo = interval[0] if interval else None
    rule("clears_noise", lo is not None and lo > bar,
         f"interval low {lo if lo is None else f'{lo:+.2%}'} vs bar {bar:.2%} "
         f"(min_effect {p['min_effect']:.0%}, baseline spread "
         f"{'n/a' if spread is None else f'{spread:.2%}'} x {p['noise_factor']:g})")
    corr = verdict.get("correctness") or {}
    rule("correctness", corr.get("status") == "pass", corr.get("detail", "not run"))
    lat = verdict.get("latency") or []
    bad = [g["name"] for g in lat if g.get("status") != "pass"]
    rule("latency_gates", not bad and bool(lat),
         "all passed" if lat and not bad else f"not passed: {bad or 'none run'}")
    failed = [r["order_index"] for r in verdict.get("raw", [])
              if (r.get("metrics") or {}).get("failed", 0) > 0]
    rule("no_failed_requests", not failed,
         "every replay request completed" if not failed else f"failures in samples {failed}")
    age = None if ctx.verdict_finished is None else ctx.now - ctx.verdict_finished
    rule("verdict_recent", age is not None and age <= p["max_verdict_age_s"],
         f"verdict age {'unknown' if age is None else f'{age:.0f}s'}, "
         f"limit {p['max_verdict_age_s']:.0f}s")

    # --- traffic
    if ctx.live_regime is None:
        rule("traffic_unchanged", True, "no newer traffic recorded since the test")
    else:
        from gitm.playbook.match import regime_distance

        d = regime_distance(ctx.tested_regime, ctx.live_regime)
        rule("traffic_unchanged", d.linf <= p["max_regime_distance"],
             f"tested vs live regime: {d.render()}, limit {p['max_regime_distance']:g}")

    # --- hardware
    same = (ctx.stable_placement.get("gpus_count") == ctx.canary_placement.get("gpus_count")
            and ctx.stable_placement.get("mem_fraction") == ctx.canary_placement.get("mem_fraction"))
    rule("same_capacity", same,
         f"stable {ctx.stable_placement}, canary {ctx.canary_placement}")
    rule("canary_ready", ctx.canary_alive and ctx.canary_runs_candidate,
         "canary is up and running the candidate" if ctx.canary_alive and ctx.canary_runs_candidate
         else f"canary alive={ctx.canary_alive}, runs candidate={ctx.canary_runs_candidate}")

    # --- operator
    rule("not_paused", not p["paused"], "paused by operator" if p["paused"] else "running")
    hour = time.gmtime(ctx.now).tm_hour
    frozen = [w for w in p["freeze_hours_utc"] if _in_window(hour, w)]
    rule("not_frozen", not frozen,
         f"inside freeze window {frozen[0]} (UTC hour {hour})" if frozen else f"UTC hour {hour}")
    denied = [k for k in knobs if any(fnmatch.fnmatch(k, g) for g in p["deny_knobs"])]
    unlisted = ([k for k in knobs if not any(fnmatch.fnmatch(k, g) for g in p["allow_knobs"])]
                if p["allow_knobs"] else [])
    rule("knobs_allowed", not denied and not unlisted,
         "allowed" if not (denied or unlisted) else f"denied {denied}, not allowed {unlisted}")
    key = knob_key(knobs)
    before = [r for r in ctx.reverted if r.get("knobs") == key]
    rule("not_reverted_before", not before,
         "never reverted" if not before else f"reverted {len(before)}x before: "
         f"{before[-1].get('reason', '')}")
    since_promo = None if ctx.last_promotion_t is None else ctx.now - ctx.last_promotion_t
    rule("promotion_interval", since_promo is None or since_promo >= p["min_interval_s"],
         "no earlier promotion" if since_promo is None
         else f"{since_promo:.0f}s since the last promotion, need {p['min_interval_s']:.0f}s")
    since_revert = None if ctx.last_revert_t is None else ctx.now - ctx.last_revert_t
    rule("revert_cooldown", since_revert is None or since_revert >= p["revert_cooldown_s"],
         "no earlier revert" if since_revert is None
         else f"{since_revert:.0f}s since the last revert, need {p['revert_cooldown_s']:.0f}s")
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
