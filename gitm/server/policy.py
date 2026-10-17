"""How a candidate is tested, and (from milestone 3) what it takes to promote one.

One dict, so the API can show and change it and every verdict can record the
exact policy it was judged under.
"""

from __future__ import annotations

import copy
from typing import Any

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
    # Kept samples per arm, ABBA order, paired.
    "repetitions": 3,
    "warmup_reps": 0,
    # Fraction of the fixed probe prompts that must answer exactly as the
    # baseline arm did. 1.0: a candidate that changes greedy output is not the
    # same model, whatever its speed.
    "min_correctness": 1.0,
    "max_wall_clock_s": 3600.0,
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


def merged(overrides: dict[str, Any] | None) -> dict[str, Any]:
    """The default policy with ``overrides`` applied; unknown keys are refused."""
    out = copy.deepcopy(DEFAULT_POLICY)
    for key, value in (overrides or {}).items():
        if key not in out:
            raise KeyError(f"unknown policy key {key!r}; known: {sorted(out)}")
        out[key] = value
    return out
