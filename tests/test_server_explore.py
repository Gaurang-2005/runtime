"""gitm server, milestone 4: choosing what to test next from the catalogue."""

from __future__ import annotations

from gitm.server.explore import candidates
from gitm.server.policy import merged
from gitm.server.promotion import knob_key


def keys(cands):
    return [knob_key(c["knobs"]) for c in cands]


def test_relative_levers_resolve_against_stables_own_value():
    c = candidates(["--max-num-batched-tokens", "4096"], tried=set(), blocked=set(),
                   policy=merged(None))
    got = keys(c)
    assert {"max_num_batched_tokens=2048", "max_num_batched_tokens=8192",
            "max_num_batched_tokens=16384"} <= set(got)


def test_risky_env_placement_and_already_running_levers_are_not_candidates():
    c = keys(candidates(["--enforce-eager"], tried=set(), blocked=set(), policy=merged(None)))
    assert "enforce_eager=False" in c                      # stable is eager: graphs are a candidate
    assert not any(k.startswith(("tensor_parallel_size", "quantization", "VLLM_",
                                 "gpu_memory_utilization", "num_speculative_tokens")) for k in c)
    on = keys(candidates([], tried=set(), blocked=set(), policy=merged(None)))
    assert "enforce_eager=False" in on                      # vLLM default is unknown, so allowed


def test_tried_and_reverted_changes_are_not_offered_again():
    pol = merged(None)
    all_ = keys(candidates([], tried=set(), blocked=set(), policy=pol))
    first, second = all_[0], all_[1]
    rest = keys(candidates([], tried={first}, blocked={second}, policy=pol))
    assert first not in rest and second not in rest and len(rest) == len(all_) - 2


def test_operator_deny_list_applies_to_exploring_too():
    c = keys(candidates([], tried=set(), blocked=set(),
                        policy=merged({"deny_knobs": ["max_num_*"]})))
    assert not any(k.startswith("max_num_") for k in c)
