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


def distance(a: dict | Regime | None, b: dict | Regime | None) -> float | None:
    """L-inf regime distance in log2 units, or None when either side is unknown."""
    from gitm.playbook.match import regime_distance

    ra, rb = as_regime(a), as_regime(b)
    if ra is None or rb is None:
        return None
    return float(regime_distance(ra, rb).linf)


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
    """Live traffic against the traffic stable was tuned for."""

    threshold: float = 1.0
    tuned: dict[str, Any] | None = None      # Regime as a dict
    tuned_at: float | None = None
    current: dict[str, Any] | None = None
    current_window: str | None = None
    current_at: float | None = None
    changed: bool = False
    changed_at: float | None = None
    last_distance: float | None = None
    seen: list[str] = field(default_factory=list)

    def observe(self, regime: Regime, window: str, now: float | None = None) -> str | None:
        """Take in a window's regime. Returns "changed" or "settled" on a transition."""
        now = time.time() if now is None else now
        d = regime.model_dump(mode="json")
        self.current, self.current_window, self.current_at = d, window, now
        self.seen = (self.seen + [window])[-200:]
        if self.tuned is None:
            self.tuned, self.tuned_at = d, now
            self.last_distance = 0.0
            return None
        self.last_distance = distance(self.tuned, d)
        if self.last_distance is None:
            return None
        if self.last_distance > self.threshold and not self.changed:
            self.changed, self.changed_at = True, now
            return "changed"
        if self.last_distance <= self.threshold and self.changed:
            self.changed, self.changed_at = False, None
            return "settled"
        return None

    def retune(self, regime: dict[str, Any] | Regime | None, now: float | None = None) -> None:
        """Stable is now tuned for ``regime`` (a promotion tested on it)."""
        if regime is None:
            return
        r = as_regime(regime)
        self.tuned, self.tuned_at = r.model_dump(mode="json"), time.time() if now is None else now
        self.changed, self.changed_at = False, None
        self.last_distance = distance(self.tuned, self.current) if self.current else 0.0

    def summary(self) -> dict[str, Any]:
        return {"threshold": self.threshold, "tuned_for": brief(self.tuned),
                "tuned_at": self.tuned_at, "current": brief(self.current),
                "current_window": self.current_window, "current_at": self.current_at,
                "distance": self.last_distance, "changed": self.changed,
                "changed_at": self.changed_at}

    def to_state(self) -> dict[str, Any]:
        return {"tuned": self.tuned, "tuned_at": self.tuned_at, "changed": self.changed,
                "changed_at": self.changed_at}

    def load_state(self, d: dict[str, Any] | None) -> None:
        if d:
            self.tuned, self.tuned_at = d.get("tuned"), d.get("tuned_at")
            self.changed, self.changed_at = d.get("changed", False), d.get("changed_at")


# ----------------------------------------------------------------- results memory

#: Verdicts that say the change helped on the traffic it was measured on.
GOOD = ("improved", "promoted", "kept", "awaiting_approval")


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
