"""How a candidate is tested, and (from milestone 3) what it takes to promote one.

One dict, so the API can show and change it and every verdict can record the
exact policy it was judged under.
"""

from __future__ import annotations

import copy
from typing import Any

from gitm.server.explore import EXPLORE_DEFAULTS
from gitm.server.promotion import PROMOTION_DEFAULTS

DEFAULT_POLICY: dict[str, Any] = {
    # --- the test --------------------------------------------------------------
    # Per-token decode speed under the recorded load. Throughput is not the
    # default: a replay offers the load the trace recorded, so below saturation
    # every configuration serves the same tokens per second and only latency can
    # tell them apart.
    "primary_metric": {"name": "mean_tpot_ms", "unit": "ms", "direction": "lower_is_better"},
    "latency_gates": [
        {"metric": "p99_ttft_ms", "max_regression": 0.10},
        {"metric": "p99_itl_ms", "max_regression": 0.10},
    ],
    # Kept samples per arm, ABBA order, paired. 4, not 3: the first GPU run's
    # eager baseline wandered 6.9-9.2 ms/token across 3 samples, and a paired
    # t-interval on 3 pairs (t=4.3) could not separate a real +21% from that.
    "repetitions": 4,
    "warmup_reps": 0,
    # Teacher-forced top-1 agreement with the baseline arm's greedy tokens on
    # the probe prompts. Numerics-only changes score ~0.95-1.0; a broken model
    # far less. Below this the candidate is not the same model, whatever its speed.
    "min_correctness": 0.95,
    "max_wall_clock_s": 3600.0,
    # Seconds to let the machine settle before each sample. 0 on servers; a
    # laptop needs ~45 (its CPU throttles hard when run back to back).
    "settle_s": 0.0,
    # A verdict is "unstable" when either arm's own samples of the primary
    # metric spread more than this (stdev / mean): the machine moved, so neither
    # "improved" nor "regressed" can be trusted.
    "max_arm_spread": 0.15,
    # --- the traffic -----------------------------------------------------------
    # A window with fewer requests than this is not a measurement of anything.
    "min_window_requests": 20,
    # Cap per replay, so one sample stays a few minutes long.
    "max_replay_requests": 300,
    # >1 compresses the recorded arrivals to test at a higher load than was seen.
    "replay_speed": 1.0,
}
# Promotion and probation: see gitm.server.promotion for what each one means.
DEFAULT_POLICY.update(PROMOTION_DEFAULTS)
# Exploring the catalogue on its own when idle: see gitm.server.explore.
DEFAULT_POLICY.update(EXPLORE_DEFAULTS)
# Watching traffic: see gitm.server.traffic.
DEFAULT_POLICY.update({
    # log2 units on each regime axis: 1.0 = traffic changed 2x on some axis.
    # Past this, traffic has changed from what stable was tuned for, and a
    # result measured on the old traffic no longer counts as "tried".
    "traffic_change_distance": 1.0,
    "traffic_check_s": 60.0,
    # After a change, once this many windows in a row show the same new
    # traffic, changes are measured from it (so the next change is caught).
    "traffic_steady_windows": 3,
    # The open recording window is closed this often (if it has new requests),
    # so the watcher sees fresh traffic without waiting for a full window.
    "traffic_window_s": 300.0,
})


#: Bounds for the settings an operator tunes most: (type, low, high).
#: repetitions >= 2 because the paired interval needs two pairs per arm.
BOUNDS: dict[str, tuple[type, float, float]] = {
    "repetitions": (int, 2, 20),
    "settle_s": (float, 0.0, 600.0),
    "replay_speed": (float, 0.1, 64.0),
}


def merged(overrides: dict[str, Any] | None) -> dict[str, Any]:
    """The default policy with ``overrides`` applied.

    Unknown keys raise KeyError; a bounded setting outside its range, or of the
    wrong type, raises ValueError, before anything runs with it.
    """
    out = copy.deepcopy(DEFAULT_POLICY)
    for key, value in (overrides or {}).items():
        if key not in out:
            raise KeyError(f"unknown policy key {key!r}; known: {sorted(out)}")
        if key in BOUNDS:
            kind, low, high = BOUNDS[key]
            if isinstance(value, bool) or not isinstance(value, int | float) \
                    or (kind is int and float(value) != int(value)):
                raise ValueError(f"{key} must be a{'n integer' if kind is int else ' number'}, "
                                 f"got {value!r}")
            value = kind(value)
            if not low <= value <= high:
                raise ValueError(f"{key} must be between {low:g} and {high:g}, got {value:g}")
        out[key] = value
    return out


def warnings_for(policy: dict[str, Any]) -> list[str]:
    """Settings that are allowed but have a consequence worth saying out loud."""
    out = []
    if policy["repetitions"] < policy["promote_min_rounds"]:
        out.append(f"{policy['repetitions']} samples per arm can reject a change but never "
                   f"promote one: promotion needs {policy['promote_min_rounds']}")
    return out
