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


def test_what_the_engine_resolved_counts_as_already_running():
    from gitm.server.canary import unchanged_knobs

    resolved = {"enable_prefix_caching": "True", "kv_cache_dtype": "auto"}
    assert unchanged_knobs([], {"enable_prefix_caching": True}, resolved) == ["enable_prefix_caching"]
    assert unchanged_knobs([], {"kv_cache_dtype": "fp8"}, resolved) == []
    c = keys(candidates([], tried=set(), blocked=set(), policy=merged(None), resolved=resolved))
    assert "enable_prefix_caching=True" not in c


def test_moe_and_multi_gpu_levers_are_skipped_where_they_cannot_apply():
    dense = keys(candidates([], tried=set(), blocked=set(), policy=merged(None),
                            facts={"moe": False, "tp": 1}))
    assert not any(k.startswith(("moe_backend", "enable_eplb", "enable_dbo",
                                 "disable_custom_all_reduce")) for k in dense)
    unknown = keys(candidates([], tried=set(), blocked=set(), policy=merged(None),
                              facts={"moe": None, "tp": 2}))
    assert any(k.startswith("moe_backend") for k in unknown)   # unreadable config: kept
    assert any(k.startswith("disable_custom_all_reduce") for k in unknown)


def test_resolved_config_is_read_from_the_engines_log(tmp_path):
    from gitm.server.supervisor import EngineProcess, EngineSpec

    e = EngineProcess(EngineSpec("stable", "m", 8100, []), tmp_path)
    assert e.resolved_config() == {}
    e.log_path.write_text("x\nINFO Initializing a V1 LLM engine (v0.30.0) with config: "
                          "model='m', max_seq_len=8192, enforce_eager=True, "
                          "compilation_config={'mode': 0}, enable_prefix_caching=True, "
                          "enable_chunked_prefill=False\n")
    r = e.resolved_config()
    assert r["enable_prefix_caching"] == "True" and r["enable_chunked_prefill"] == "False"
    assert r["max_seq_len"] == "8192"


def test_relative_levers_use_vllms_own_default_for_this_gpu():
    c = candidates([], tried=set(), blocked=set(), policy=merged(None),
                   defaults={"max_num_batched_tokens": 2048, "max_num_seqs": 256})
    got = keys(c)
    assert {"max_num_batched_tokens=1024", "max_num_batched_tokens=4096",
            "max_num_batched_tokens=8192"} <= set(got)
    assert "max_num_batched_tokens=16384" not in got     # not built off the catalogue's 8192
    assert any("vLLM's default for this GPU (2048)" in x["note"] for x in c)
