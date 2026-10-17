"""What to test next when nobody has asked: candidates from gitm's lever catalogue.

With ``auto_explore`` on, the controller asks this module for the next candidate
whenever the canary is idle (nothing queued or running, no probation, not
paused) and queues it like any submitted change. Every candidate still goes
through the whole A/B and every promotion rule; exploring only decides *what*
is tried, never what is kept.

A catalogue lever becomes a candidate when:

* its tier is allowed (``explore_tiers``; high-risk levers such as tensor
  parallelism or quantization are excluded by default);
* it is a ``vllm serve`` flag — environment-variable levers need a different
  launch and are skipped;
* it is not a placement knob (``explore_skip_knobs``: memory fraction, swap,
  offload decide *where* an engine fits, and the capacity rule compares them);
* it is for something this model has: MoE levers only on a model whose config
  declares experts, multi-GPU levers only when stable runs TP > 1;
* stable does not already run that value — by its flags, or by the
  configuration the engine itself resolved — it was not tried against this
  exact stable configuration, and it was never reverted.

Relative levers ("x2 the per-step token budget") are resolved against the value
stable runs. When stable leaves the flag at vLLM's default, the catalogue's own
reference value is used and the note says so, because vLLM computes some of
those defaults at startup and the server's flags cannot show them.
"""

from __future__ import annotations

from typing import Any

from gitm.server.canary import current_value, unchanged_knobs
from gitm.server.promotion import knob_key

EXPLORE_DEFAULTS: dict[str, Any] = {
    "auto_explore": False,
    "explore_tiers": ["low_risk", "moderate"],
    "explore_skip_knobs": ["gpu_memory_utilization", "swap_space", "cpu_offload_gb",
                           "max_seq_len_to_capture"],
}


#: Levers for layers or parallelism a model may not have.
MOE_KNOBS = ("moe_backend", "enable_eplb", "enable_expert_parallel", "enable_dbo")
MULTI_GPU_KNOBS = ("disable_custom_all_reduce",)


def model_facts(model: str, stable_args: list[str]) -> dict[str, Any]:
    """What the model has, from its own HF config: MoE or not, and the TP size.

    ``moe`` is ``None`` when the config cannot be read offline, and then no MoE
    lever is ruled out — the same rule #180 applies in the batch loop.
    """
    tp = int(current_value(stable_args, "tensor_parallel_size") or 1)
    moe = None
    try:
        from transformers import AutoConfig

        cfg = AutoConfig.from_pretrained(model, local_files_only=True).to_dict()
        from gitm.planner.registry import text_config

        inner = text_config(cfg)
        moe = any((c.get(k) or 0) > 1 for c in (cfg, inner)
                  for k in ("num_experts", "n_routed_experts", "num_local_experts"))
    except Exception:  # noqa: BLE001 - unreadable config: rule nothing out
        moe = None
    return {"moe": moe, "tp": tp}


def _number(v: Any) -> float | None:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def candidates(stable_args: list[str], *, tried: set[str], blocked: set[str],
               policy: dict[str, Any], library=None, resolved: dict[str, str] | None = None,
               facts: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    """Every catalogue candidate worth testing against ``stable_args``, in catalogue order."""
    import fnmatch

    from gitm.kernels.library import load_library

    specs = library if library is not None else load_library(workload="vllm-decode")
    out: list[dict[str, Any]] = []
    for spec in specs:
        tier = getattr(spec.safety, "tier", None) if spec.safety else None
        tier = getattr(tier, "value", tier)
        if tier not in policy["explore_tiers"]:
            continue
        if spec.knobs:          # joint levers: one knob at a time for now
            continue
        knob = spec.knob
        if knob.isupper() or knob in policy["explore_skip_knobs"]:
            continue
        if any(fnmatch.fnmatch(knob, g) for g in policy.get("deny_knobs", [])):
            continue
        if knob in MOE_KNOBS and (facts or {}).get("moe") is False:
            continue
        if knob in MULTI_GPU_KNOBS and (facts or {}).get("tp", 1) == 1:
            continue
        if policy.get("allow_knobs") and not any(
                fnmatch.fnmatch(knob, g) for g in policy["allow_knobs"]):
            continue
        values: list[tuple[Any, str]]
        grid = list(spec.value_multiplier_grid or [])
        if grid:
            cur = _number(current_value(stable_args, knob))
            ref, how = (cur, "stable's value") if cur else (_number(spec.value),
                                                            "the catalogue's reference value")
            if not ref:
                continue
            values = [(int(round(ref * m)) if float(ref).is_integer() else round(ref * m, 4),
                       f"{m:g}x {how} ({ref:g})") for m in grid]
        else:
            values = [(spec.value, spec.summary or spec.name)]
        for value, why in values:
            knobs = {knob: value}
            key = knob_key(knobs)
            if key in tried or key in blocked or unchanged_knobs(stable_args, knobs, resolved):
                continue
            out.append({"knobs": knobs, "lever": spec.name, "note": f"explore: {why}"})
    return out
