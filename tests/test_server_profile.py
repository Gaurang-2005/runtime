"""gitm server: ranking what to try next from a kernel profile of stable (no GPU, no vLLM)."""

from __future__ import annotations

import json
import os
import time

import pytest

from gitm.server.profile import (
    TRACED_FLAGS,
    analyze,
    close_window,
    load_window_trace,
    open_window,
    order,
    sample_from_metrics,
    traced_args,
)
from gitm.tracer import injection
from gitm.tracer.schema import KernelEvent, Trace
from gitm.tracer.vllm_stats import summarize


def kernel(name, start, dur, op=None):
    return KernelEvent(name=name, start_ns=start, end_ns=start + dur, stream_id=1, device_id=0,
                       range_op=op)


def trace(kernels):
    return Trace(workload_id="t", fingerprint="f", run_id="r", device_count=1, vendor="nvidia",
                 captured_at_ns=0, duration_ns=10**9, events=kernels)


# ----------------------------------------------------------------- engine signals

def test_metrics_gauges_become_the_scheduler_summary_the_batch_loop_uses():
    snaps = [{"vllm:num_requests_running": 8, "vllm:num_requests_waiting": 30,
              "vllm:kv_cache_usage_perc": 0.5, "vllm:num_preemptions_total": 2},
             {"vllm:num_requests_running": 8, "vllm:num_requests_waiting": 40,
              "vllm:kv_cache_usage_perc": 0.95, "vllm:num_preemptions_total": 7}]
    s = summarize([sample_from_metrics(x, t_ns=i * 10**9, max_num_seqs=8)
                   for i, x in enumerate(snaps)], max_num_seqs=8)
    assert s.mean_running == 8 and s.peak_queue_depth == 40
    assert s.total_preemptions == 5 and s.peak_gpu_cache_usage == 0.95
    from gitm.optimizer.scheduler_attribution import scheduler_causes

    signals = {c.signal for c in scheduler_causes(s)}
    assert {"kv_cache_preemption", "kv_cache_pressure"} <= signals


def test_a_profile_engine_is_traced_the_way_the_batch_loop_traces_one():
    args = traced_args(["--max-model-len", "2048", "--enforce-eager"])
    assert args.count("--enforce-eager") == 1
    assert all(f in args for f in TRACED_FLAGS)


# ----------------------------------------------------------------- ranking

def profile_with(rows):
    return {"ranking": rows}


def row(lever, rejected=None, cause=None, targets=()):
    return {"lever": lever, "knob": lever, "predicted_delta": 0.05, "rejected": rejected,
            "motivated_by": cause, "targets": [{"op": o, "recoverable_s": s} for o, s in targets]}


def cand(lever, n=0):
    return {"knobs": {lever: n}, "lever": lever, "note": "explore: x"}


def test_the_profile_orders_candidates_and_rules_out_what_it_rejects():
    cands = [cand("a"), cand("b"), cand("c"), cand("b", 1), cand("d")]
    prof = profile_with([row("c", cause="admission_backlog"), row("b"),
                         row("a", rejected="no_recoverable_time: attention at its floor")])
    kept, out = order(cands, prof)
    assert [c["lever"] for c in kept] == ["c", "b", "b", "d"]   # d is unranked: last
    assert "admission_backlog" in kept[0]["why"] and "ranked #1" in kept[0]["note"]
    assert [c["lever"] for c in out] == ["a"] and "floor" in out[0]["reason"]


def test_without_a_profile_the_catalogue_order_stands():
    cands = [cand("a"), cand("b")]
    assert order(cands, None) == (cands, [])


@pytest.fixture
def tiny_model(tmp_path):
    d = tmp_path / "model"
    d.mkdir()
    (d / "config.json").write_text(json.dumps({
        "model_type": "qwen2", "architectures": ["Qwen2ForCausalLM"], "hidden_size": 896,
        "intermediate_size": 4864, "num_attention_heads": 14, "num_key_value_heads": 2,
        "num_hidden_layers": 24, "vocab_size": 151936, "torch_dtype": "bfloat16",
        "max_position_embeddings": 32768}))
    return str(d)


def test_analysis_ranks_the_catalogue_and_says_where_the_time_went(tiny_model, monkeypatch):
    monkeypatch.setenv("GITM_GPU_SKU", "NVIDIA H100 80GB HBM3")
    ks = [kernel("cutlass_80_wmma_tensorop_bf16_gemm", i * 1000, 600, op="mlp_gate_up")
          for i in range(200)]
    ks += [kernel("flash_fwd_splitkv_kernel", 300_000 + i * 1000, 300) for i in range(100)]
    out = analyze(trace(ks), model=tiny_model,
                  resolved={"dtype": "torch.bfloat16", "max_seq_len": "2048",
                            "tensor_parallel_size": "1"})
    ops = {o["name"]: o["share"] for o in out["ops"]}
    assert ops["mlp_gate_up"] == pytest.approx(0.8, abs=0.01)
    assert out["labelled_share"] == pytest.approx(0.8, abs=0.01)
    assert out["graph"]["family"] == "dense" and out["graph"]["floors_priced"]
    levers = {r["lever"]: r for r in out["ranking"]}
    # dtype came from the engine's resolved config, not "unknown"
    assert not (levers["kv_cache_block_size_16"]["rejected"] or "").startswith(
        "not_applicable: Requires dtype")


def test_an_empty_trace_is_refused_rather_than_ranked(tiny_model):
    with pytest.raises(ValueError, match="no kernels"):
        analyze(trace([]), model=tiny_model)


# ----------------------------------------------------------------- capture

def test_the_controller_carries_the_collector_only_while_opening_and_closing(tmp_path,
                                                                             monkeypatch):
    monkeypatch.setenv("GITM_TRACE_SETTLE_S", "0")
    monkeypatch.delenv(injection.ENV_OUT, raising=False)
    trace_out = tmp_path / "shards" / "trace"
    trace_out.parent.mkdir()
    open_window(trace_out)
    # A replay client started now must not inherit the collector...
    assert injection.ENV_OUT not in os.environ
    # ...but the window is open for the traced engine.
    assert trace_out.with_name("trace.arm").exists()
    close_window(trace_out)
    assert not trace_out.with_name("trace.arm").exists()
    assert injection.ENV_OUT not in os.environ


def test_shards_too_big_for_this_machine_are_refused_not_loaded(tmp_path, monkeypatch):
    import gitm.server.profile as prof

    trace_out = tmp_path / "shards" / "trace"
    trace_out.parent.mkdir()
    (trace_out.parent / "trace.123").write_bytes(b"x" * 1000)
    monkeypatch.setattr(prof, "_mem_available", lambda: 9000)    # 1000 B needs ~10 kB
    with pytest.raises(MemoryError, match="profile_window_s"):
        load_window_trace(trace_out, None, None)


def test_only_a_traced_engine_gets_the_collector_and_nvtx(tmp_path, monkeypatch):
    from gitm.server.supervisor import EngineProcess, EngineSpec

    monkeypatch.setenv(injection.ENV_OUT, "/somewhere/else")   # left over in the shell
    plain = EngineProcess(EngineSpec(role="canary", model="m", port=1), tmp_path).environment()
    assert injection.ENV_OUT not in plain
    traced = EngineProcess(EngineSpec(role="canary", model="m", port=1,
                                      trace_out=tmp_path / "s" / "trace", trace_nvtx=True),
                           tmp_path).environment()
    assert traced[injection.ENV_OUT].endswith("s/trace")
    if injection.detect_vendor() != "amd":
        assert traced[injection.ENV_NVTX] == "1"


# ----------------------------------------------------------------- the controller

class FakeEngine:
    def __init__(self, role, port, args):
        from gitm.server.supervisor import EngineSpec

        self.spec = EngineSpec(role, "m", port, list(args), mem_fraction=0.4)
        self.alive = True

    def resolved_config(self):
        return {}


@pytest.fixture
def ctl(tmp_path):
    from gitm.server.controller import Controller, ServerConfig

    c = Controller(ServerConfig(model="m", workdir=tmp_path))
    c.engines = {"stable": FakeEngine("stable", 8100, ["--max-num-seqs", "64"]),
                 "canary": FakeEngine("canary", 8101, ["--max-num-seqs", "64"])}
    return c


CHAT = {"source_kind": "production", "trace": "t", "requests": 100, "rate_rps": 2.0,
        "io_ratio": 5.0, "input_p50": 900, "input_p95": 950, "output_p50": 200,
        "output_p95": 380, "burstiness": 1.0, "concurrency": None, "in_envelope": True}


def test_a_profile_goes_stale_when_stable_or_the_traffic_moves(ctl):
    from gitm.traffic.regime import Regime

    assert ctl.profile_stale() == "no profile of stable yet"
    ctl.profile = {"status": "done", "stable_args": ["--max-num-seqs", "64"], "regime": CHAT}
    ctl.watch.current = Regime.model_validate(CHAT).model_dump(mode="json")
    assert ctl.profile_stale() is None
    ctl.engines["stable"].spec.serve_args = ["--max-num-seqs", "128"]      # promoted
    assert "configuration changed" in ctl.profile_stale()
    ctl.engines["stable"].spec.serve_args = ["--max-num-seqs", "64"]
    ctl.watch.current = Regime.model_validate(CHAT | {"input_p50": 4000, "input_p95": 5000,
                                                      "io_ratio": 25.0}).model_dump(mode="json")
    assert "traffic moved" in ctl.profile_stale()


def test_a_session_profiles_before_exploring_and_a_person_can_ask(ctl, monkeypatch, tmp_path):
    from gitm.server.autonomy import new_session

    ran = []
    monkeypatch.setattr(ctl, "_window_for_test", lambda: tmp_path / "w.jsonl")
    monkeypatch.setattr(ctl, "_run_profile", lambda w, why: ran.append(why))
    assert ctl._maybe_profile() is False                        # manual: never on its own
    ctl.request_profile(by="test")
    assert ctl._maybe_profile() is True and ran == ["requested by test"]
    ctl._profile_requested = None
    ctl.session = new_session("assisted")
    assert ctl._maybe_profile() is True and ran[-1] == "no profile of stable yet"
    ctl.submit({"max_num_seqs": 128})                           # something queued: test first
    assert ctl._maybe_profile() is False


def test_a_failed_profile_is_not_retried_for_the_same_stable_at_once(ctl, monkeypatch, tmp_path):
    from gitm.server.autonomy import new_session

    ctl.session = new_session("autonomous")
    monkeypatch.setattr(ctl, "_window_for_test", lambda: tmp_path / "w.jsonl")
    monkeypatch.setattr(ctl, "_run_profile", lambda w, why: None)
    ctl.profile_last_attempt = {"status": "failed", "finished": time.time(),
                                "stable_args": ["--max-num-seqs", "64"]}
    assert ctl._maybe_profile() is False
    ctl.profile_last_attempt["finished"] -= ctl.policy["profile_retry_s"] + 1
    assert ctl._maybe_profile() is True


def test_a_restart_keeps_the_last_finished_profile(ctl, tmp_path):
    for pid, status, t in (("a", "done", 1.0), ("b", "done", 2.0), ("c", "failed", 3.0),
                           ("d", "running", 4.0)):
        (tmp_path / "profiles" / pid).mkdir(parents=True)
        (tmp_path / "profiles" / pid / "profile.json").write_text(
            json.dumps({"id": pid, "status": status, "finished": t}))
    ctl._load_profile()
    assert ctl.profile["id"] == "b"
    d = json.loads((tmp_path / "profiles" / "d" / "profile.json").read_text())
    assert d["status"] == "interrupted"          # its controller died mid-profile


def test_the_explorer_ranks_by_the_profile_the_controller_stores(ctl, monkeypatch):
    """The record shape _run_profile writes, not a hand-built one: the ranking
    sits under "analysis", and an order() that looked for it at the top level
    ranked nothing on the first GPU run while every unit test passed."""
    import gitm.server.explore as explore

    monkeypatch.setattr(explore, "model_facts", lambda model, args: {"moe": False, "tp": 1})
    ctl._serve_defaults = {"max_num_batched_tokens": 2048, "max_num_seqs": 256}
    first = ctl._candidates()
    assert first and not any("why" in c for c in first)          # no profile: catalogue order
    last, out = first[-1]["lever"], first[0]["lever"]
    ctl.profile = {"status": "done", "stable_args": ["--max-num-seqs", "64"], "regime": None,
                   "analysis": {"ranking": [row(last, cause="admission_backlog"),
                                            row(out, rejected="not_applicable: no")]}}
    ranked = ctl._candidates()
    assert ranked[0]["lever"] == last and "admission_backlog" in ranked[0]["why"]
    assert out not in {c["lever"] for c in ranked}
    assert {c["lever"] for c in ctl._ruled_out} == {out}
