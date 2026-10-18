"""Who decides what gets tested and what goes live: the operator chooses.

Three modes:

* **manual** — gitm tests only what someone queues, and never promotes on its
  own: a win waits for approval;
* **assisted** — gitm also picks candidates itself (catalogue, re-checks after a
  traffic change), but every win still waits for approval;
* **autonomous** — gitm picks, tests and promotes by itself, within every
  promotion rule.

Assisted and autonomous run as a **session** someone starts, with limits:

* a budget — hours and/or a number of tests gitm may start on its own; when it
  is spent, the session ends and the server is back in manual;
* a schedule — local hours in which gitm may *start* its own tests (e.g. 1–6 am).
  Tests someone queues by hand run whenever they reach the front of the queue;
* a scope — which knobs it may touch and which risk tiers, on top of the policy.

Stopping a session never touches stable. It either lets the running test finish
or cancels it, as the operator asks.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Any

MODES = ("manual", "assisted", "autonomous")

#: Submitters whose tests gitm queued on its own. They count against a
#: session's test budget and wait for the session's schedule; anything else was
#: queued by a person or an agent and runs when it reaches the front.
AUTO_SUBMITTERS = ("explorer", "recheck", "playbook", "rediscover")


def picks_itself(mode: str) -> bool:
    return mode in ("assisted", "autonomous")


def promotes_itself(mode: str) -> bool:
    return mode == "autonomous"


def in_schedule(schedule: list[list[int]] | None, now: float) -> bool:
    """Whether local time ``now`` falls inside any [start_hour, end_hour) window.

    No schedule means always. A window may wrap midnight ([22, 6]).
    """
    if not schedule:
        return True
    hour = time.localtime(now).tm_hour
    for start, end in schedule:
        start, end = int(start) % 24, int(end) % 24
        if start == end:
            return True
        if (start <= hour < end) if start < end else (hour >= start or hour < end):
            return True
    return False


@dataclass
class Session:
    mode: str
    started: float
    ends: float | None = None          # wall-clock deadline, or None for no time limit
    max_tests: int | None = None       # tests gitm may start on its own, or None
    tests_started: int = 0
    schedule: list[list[int]] = field(default_factory=list)
    allow_knobs: list[str] = field(default_factory=list)
    deny_knobs: list[str] = field(default_factory=list)
    tiers: list[str] | None = None     # None: the policy's explore_tiers

    def exhausted(self, now: float) -> str | None:
        """Why this session is over, or None while it may continue."""
        if self.ends is not None and now >= self.ends:
            return "time budget spent"
        if self.max_tests is not None and self.tests_started >= self.max_tests:
            return f"test budget spent ({self.tests_started} of {self.max_tests})"
        return None

    def scoped_policy(self, policy: dict[str, Any]) -> dict[str, Any]:
        """The policy with this session's scope applied on top: it can only narrow."""
        out = dict(policy)
        if self.allow_knobs:
            out["allow_knobs"] = list(self.allow_knobs)
        out["deny_knobs"] = list(policy.get("deny_knobs", [])) + list(self.deny_knobs)
        if self.tiers is not None:
            out["explore_tiers"] = [t for t in self.tiers if t in policy["explore_tiers"]] \
                or list(self.tiers)
        return out

    def allows(self, knobs: dict[str, Any], policy: dict[str, Any]) -> bool:
        """Whether every knob is inside this session's scope (and the policy's lists)."""
        import fnmatch

        p = self.scoped_policy(policy)
        for k in knobs:
            if any(fnmatch.fnmatch(k, g) for g in p.get("deny_knobs", [])):
                return False
            if p.get("allow_knobs") and not any(fnmatch.fnmatch(k, g)
                                                for g in p["allow_knobs"]):
                return False
        return True

    def summary(self, now: float) -> dict[str, Any]:
        d = asdict(self)
        d["remaining_s"] = None if self.ends is None else max(0.0, round(self.ends - now))
        d["in_schedule"] = in_schedule(self.schedule, now)
        d["over"] = self.exhausted(now)
        return d


def new_session(mode: str, *, budget_h: float | None = None, max_tests: int | None = None,
                schedule: list[list[int]] | None = None, allow_knobs: list[str] | None = None,
                deny_knobs: list[str] | None = None, tiers: list[str] | None = None,
                now: float | None = None) -> Session:
    """A validated session. Raises ValueError on anything that cannot run."""
    if mode not in ("assisted", "autonomous"):
        raise ValueError(f"a session runs in 'assisted' or 'autonomous', not {mode!r}; "
                         "manual is what the server is in without one")
    if budget_h is not None and not budget_h > 0:
        raise ValueError("budget_h must be > 0 (or omitted for no time limit)")
    if max_tests is not None and int(max_tests) < 1:
        raise ValueError("max_tests must be at least 1 (or omitted for no limit)")
    for w in schedule or []:
        if len(w) != 2 or not all(isinstance(h, int) and 0 <= h <= 24 for h in w):
            raise ValueError(f"schedule windows are [start_hour, end_hour] in 0-24, got {w!r}")
    now = time.time() if now is None else now
    return Session(mode=mode, started=now,
                   ends=None if budget_h is None else now + float(budget_h) * 3600,
                   max_tests=None if max_tests is None else int(max_tests),
                   schedule=[list(w) for w in schedule or []],
                   allow_knobs=list(allow_knobs or []), deny_knobs=list(deny_knobs or []),
                   tiers=None if tiers is None else list(tiers))
