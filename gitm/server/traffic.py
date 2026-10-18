"""Results belong to the traffic they were measured on.

"max_num_batched_tokens=4096 is +4%" is a statement about *some* traffic: chat
with short prompts, or long documents, or bursts. Until now the server remembered
a result per stable configuration only, so once a change was tried it was never
tried again, whatever the traffic later became, and nothing noticed when traffic
moved away from what stable had been tuned for.

This module makes the traffic regime part of every result:

* :class:`TrafficWatch` turns each recorded window into a
  :class:`~gitm.traffic.regime.Regime` and compares it with the regime stable
  was tuned for (set when stable was promoted, or the first traffic seen). When
  the distance passes a threshold, traffic has *changed*;
* the helpers below answer "was this change already tried for traffic like
  this?", "what has already been proven for traffic like this?", and "what did
  well under the old traffic and should be tried again?".

Distance is :func:`gitm.playbook.match.regime_distance`: log2 ratios per axis
(input/output length, I/O ratio, burstiness), combined with L-inf. 1.0 means
some axis changed 2x. The playbook's own matcher reads the same coordinates, so
a result stored here can become a playbook row without translation.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from gitm.traffic.regime import Regime


def regime_of(window: Path, max_requests: int) -> tuple[Regime, int]:
    """The regime of a recorded window at its real speed, and its request count."""
    from gitm.server.canary import load_window

    trace, regime = load_window(window, max_requests=max_requests, speed=1.0)
    return regime, len(trace.requests)


def as_regime(d: dict[str, Any] | Regime | None) -> Regime | None:
    if d is None or isinstance(d, Regime):
        return d
    return Regime.model_validate(d)


#: Burstiness (variance / mean of arrivals per bin) is only measurable from
#: enough requests. From 30-60 it is mostly noise: two windows of the same
#: simulated chat traffic read 1.94 and 0.82, a "2x change" that made gitm
#: re-test block_size=16 four times as if the traffic were new each time.
BURSTINESS_MIN_REQUESTS = 300


def distance(a: dict | Regime | None, b: dict | Regime | None) -> float | None:
    """L-inf regime distance in log2 units, or None when either side is unknown.

    Lengths and the prompt:answer ratio always count. Burstiness counts only
    when both regimes come from at least BURSTINESS_MIN_REQUESTS requests.
    """
    from gitm.playbook.match import DEFAULT_AXES, UNCALIBRATED_POLICY, regime_distance

    ra, rb = as_regime(a), as_regime(b)
    if ra is None or rb is None:
        return None
    axes = DEFAULT_AXES
    if min(ra.requests, rb.requests) < BURSTINESS_MIN_REQUESTS:
        axes = tuple(x for x in DEFAULT_AXES if x != "burstiness")
    policy = UNCALIBRATED_POLICY.model_copy(update={"axes": axes})
    return float(regime_distance(ra, rb, policy).linf)


def brief(r: dict | Regime | None) -> dict[str, Any] | None:
    """The axes worth showing a person."""
    r = as_regime(r)
    if r is None:
        return None
    return {"label": r.label(), "rate_rps": round(r.rate_rps, 3), "input_p50": r.input_p50,
            "output_p50": r.output_p50, "io_ratio": round(r.io_ratio, 2),
            "burstiness": round(r.burstiness, 2), "requests": r.requests}


@dataclass
class TrafficWatch:
    """Live traffic, against two references kept apart on purpose.

    * ``validated`` — the traffic stable's configuration was last *proven* on:
      a promoted change's test traffic, or traffic a session re-tested and found
      nothing better for. Only evidence moves it (and a revert restores it).
      Before any test it is simply the first traffic seen, and says so.
    * ``reference`` — what changes are measured *from*. When traffic moves away
      from it, that is a change (reported once). Once the new traffic has held
      steady for ``steady_windows`` windows, the reference re-arms to it, so the
      next change is caught too, without anyone declaring a "new normal" and
      without claiming stable was validated on it.
    """

    threshold: float = 1.0
    steady_windows: int = 3
    reference: dict[str, Any] | None = None
    validated: dict[str, Any] | None = None
    validated_at: float | None = None
    validated_how: str | None = None
    current: dict[str, Any] | None = None
    current_window: str | None = None
    current_at: float | None = None
    changed: bool = False
    changed_at: float | None = None
    last_distance: float | None = None      # current vs reference
    _pending: dict[str, Any] | None = None  # the new traffic, while it settles
    _steady: int = 0
    seen: list[str] = field(default_factory=list)

    def observe(self, regime: Regime, window: str, now: float | None = None) -> str | None:
        """Take in a window's regime. Returns "changed", "settled" or "rearmed" on a transition."""
        now = time.time() if now is None else now
        d = regime.model_dump(mode="json")
        self.current, self.current_window, self.current_at = d, window, now
        self.seen = (self.seen + [window])[-200:]
        if self.reference is None:
            self.reference = d
            if self.validated is None:
                self.validated, self.validated_at = d, now
                self.validated_how = "first traffic seen (not yet tested)"
            self.last_distance = 0.0
            return None
        self.last_distance = distance(self.reference, d)
        if self.last_distance is None:
            return None
        if not self.changed:
            if self.last_distance > self.threshold:
                self.changed, self.changed_at = True, now
                self._pending, self._steady = d, 1
                return "changed"
            return None
        if self.last_distance <= self.threshold:
            self.changed, self.changed_at, self._pending, self._steady = False, None, None, 0
            return "settled"
        if near(self._pending, d, self.threshold):
            self._steady += 1
        else:
            self._pending, self._steady = d, 1
        if self._steady >= self.steady_windows:
            self.reference = d
            self.changed, self.changed_at, self._pending, self._steady = False, None, None, 0
            self.last_distance = 0.0
            return "rearmed"
        return None

    def validate(self, regime: dict[str, Any] | Regime | None, how: str,
                 now: float | None = None) -> None:
        """Stable's configuration is now proven on ``regime``; measure changes from it."""
        if regime is None:
            return
        r = as_regime(regime).model_dump(mode="json")
        now = time.time() if now is None else now
        self.validated, self.validated_at, self.validated_how = r, now, how
        self.reference = r
        self.changed, self.changed_at, self._pending, self._steady = False, None, None, 0
        self.last_distance = distance(r, self.current) if self.current else 0.0

    def summary(self) -> dict[str, Any]:
        return {"threshold": self.threshold, "steady_windows": self.steady_windows,
                "validated_on": brief(self.validated), "validated_at": self.validated_at,
                "validated_how": self.validated_how,
                "validated_distance": distance(self.validated, self.current),
                "reference": brief(self.reference), "current": brief(self.current),
                "current_window": self.current_window, "current_at": self.current_at,
                "distance": self.last_distance, "changed": self.changed,
                "changed_at": self.changed_at,
                "settling": {"windows": self._steady, "of": self.steady_windows}
                if self.changed else None}

    def to_state(self) -> dict[str, Any]:
        return {"reference": self.reference, "validated": self.validated,
                "validated_at": self.validated_at, "validated_how": self.validated_how,
                "changed": self.changed, "changed_at": self.changed_at}

    def load_state(self, d: dict[str, Any] | None) -> None:
        if not d:
            return
        # state.json from before the split kept one regime, "tuned".
        self.reference = d.get("reference", d.get("tuned"))
        self.validated = d.get("validated", d.get("tuned"))
        self.validated_at = d.get("validated_at", d.get("tuned_at"))
        self.validated_how = d.get("validated_how")
        self.changed, self.changed_at = d.get("changed", False), d.get("changed_at")
        self._pending, self._steady = None, 0


# ----------------------------------------------------------------- results memory

#: Verdicts that say the change helped on the traffic it was measured on.
GOOD = ("improved", "promoted", "kept", "awaiting_approval", "approved")


def near(a: dict | None, b: dict | None, threshold: float) -> bool:
    d = distance(a, b)
    return d is not None and d <= threshold


def tried_for(results: list[dict[str, Any]], knob_key: str, stable_args: list[str],
              regime: dict | None, threshold: float) -> bool:
    """Whether this change was already tested against this stable, for traffic like ``regime``.

    Queued or running counts as tried whatever the traffic. A finished test with
    no regime (it never reached a measurement) counts too, so a change that
    cannot be measured is not retried in a loop. A measured one counts only for
    traffic within ``threshold`` of what it was measured on.
    """
    for r in results:
        if r["knob_key"] != knob_key or r.get("stable_args") not in (None, stable_args):
            continue
        if r["status"] in ("queued", "running") or r.get("regime") is None:
            return True
        if regime is None or near(r["regime"], regime, threshold):
            return True
    return False


def proven_for(results: list[dict[str, Any]], regime: dict | None,
               threshold: float) -> list[dict[str, Any]]:
    """Changes that won on traffic like ``regime``, best effect first."""
    hits = [r for r in results if r["status"] in GOOD and r.get("regime") is not None
            and near(r["regime"], regime, threshold)]
    return sorted(hits, key=lambda r: -(r.get("effect") or 0.0))


def promising_under(results: list[dict[str, Any]], regime: dict | None,
                    threshold: float) -> list[dict[str, Any]]:
    """Changes that won or nearly won (positive effect) on traffic like ``regime``."""
    hits = [r for r in results if r.get("regime") is not None
            and (r["status"] in GOOD or (r.get("effect") or 0.0) > 0)
            and near(r["regime"], regime, threshold)]
    return sorted(hits, key=lambda r: -(r.get("effect") or 0.0))
