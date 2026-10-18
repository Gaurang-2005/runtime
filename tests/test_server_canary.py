"""gitm server, milestone 2: building and judging a canary A/B (no GPU, no vLLM)."""

from __future__ import annotations

import json

import pytest

from gitm.experiment.evaluate import Sample, evaluate
from gitm.server.canary import (
    build_contract,
    candidate_args,
    current_value,
    load_window,
    unchanged_knobs,
)
from gitm.server.policy import DEFAULT_POLICY, merged


def test_a_knob_replaces_the_flag_stable_already_sets():
    base = ["--max-model-len", "8192", "--max-num-batched-tokens", "8192"]
    out = candidate_args(base, {"max_num_batched_tokens": 16384})
    assert out == ["--max-model-len", "8192", "--max-num-batched-tokens", "16384"]


def test_a_new_knob_is_added_and_booleans_use_vllms_spelling():
    out = candidate_args(["--max-model-len", "8192"],
                         {"enable_prefix_caching": False, "enforce_eager": True})
    assert "--no-enable-prefix-caching" in out and "--enforce-eager" in out
    back = candidate_args(out, {"enable_prefix_caching": True})
    assert "--enable-prefix-caching" in back and "--no-enable-prefix-caching" not in back


def test_current_values_are_read_back_from_the_flags():
    args = ["--max-num-seqs", "64", "--enforce-eager", "--no-enable-prefix-caching"]
    assert current_value(args, "max_num_seqs") == "64"
    assert current_value(args, "enforce_eager") is True
    assert current_value(args, "enable_prefix_caching") is False
    assert current_value(args, "max_num_batched_tokens") is None


def test_an_ab_of_stable_against_itself_is_refused():
    args = ["--max-num-seqs", "64", "--enforce-eager"]
    assert unchanged_knobs(args, {"max_num_seqs": 64, "enforce_eager": True}) == [
        "max_num_seqs", "enforce_eager"]
    assert unchanged_knobs(args, {"max_num_seqs": 128}) == []
    assert unchanged_knobs([], {"max_num_seqs": 128}) == []   # vLLM default: unknown, allowed


def _window(tmp_path, n=30, gap_ms=1000):
    p = tmp_path / "window-test.jsonl"
    p.write_text("\n".join(json.dumps({"timestamp": 1_000_000 + i * gap_ms,
                                       "input_length": 600, "output_length": 50,
                                       "hash_ids": [0, i + 1]}) for i in range(n)) + "\n")
    return p


def test_a_window_can_be_capped_and_replayed_faster(tmp_path):
    p = _window(tmp_path)
    trace, regime = load_window(p, max_requests=20, speed=1.0)
    fast, fast_regime = load_window(p, max_requests=20, speed=4.0)
    assert len(trace.requests) == 20
    assert fast.requests[-1].arrival_s == pytest.approx(trace.requests[-1].arrival_s / 4)
    assert fast_regime.rate_rps == pytest.approx(regime.rate_rps * 4, rel=0.05)


class FakeCanary:
    def __init__(self):
        self.args = ["--max-num-batched-tokens", "8192"]
        self.history = []

    def snapshot(self):
        return list(self.args)

    def apply(self, spec):
        self.args = candidate_args(self.args, spec.knob_values)
        self.history.append(("apply", list(self.args)))

    def restore(self, snap):
        self.args = list(snap)
        self.history.append(("restore", list(self.args)))

    def measure(self, spec):
        return None


class FakeSampler:
    """Candidate decodes 10% faster per token, latency unchanged, same answers."""

    def __init__(self, canary, candidate_tpot=9.0, correct=1.0):
        self.canary, self.candidate_tpot, self.correct = canary, candidate_tpot, correct

    def sample(self, arm, index):
        cand = arm == "candidate"
        tpot = (self.candidate_tpot if cand else 10.0) + 0.01 * index
        return Sample(metrics={"mean_tpot_ms": tpot, "p99_ttft_ms": 50.0, "p99_itl_ms": 20.0},
                      wall_s=1.0, accelerator_s=1.0, correctness=self.correct if cand else 1.0)


def _contract(tmp_path, **policy):
    trace, regime = load_window(_window(tmp_path), max_requests=30)
    return build_contract(intervention_id="iv1", knobs={"max_num_batched_tokens": 16384},
                          stable_args=["--max-num-batched-tokens", "8192"], trace=trace,
                          regime=regime, policy=merged(policy), submitter="test")


def test_a_faster_candidate_is_improved_and_left_on_the_canary(tmp_path):
    canary = FakeCanary()
    v = evaluate(_contract(tmp_path), canary, FakeSampler(canary))
    assert v.state == "improved" and v.rollback == "kept"
    assert v.effect == pytest.approx(0.1, abs=0.01)
    assert "--max-num-batched-tokens" in canary.args and "16384" in canary.args
    assert v.n_baseline == DEFAULT_POLICY["repetitions"]


def test_a_candidate_that_changes_the_answers_is_invalid_and_rolled_back(tmp_path):
    canary = FakeCanary()
    v = evaluate(_contract(tmp_path), canary, FakeSampler(canary, correct=0.75))
    # ABBA with an even count ends on a baseline arm, so the candidate may
    # already be off: either way the canary is left on stable's flags.
    assert v.state == "invalid" and v.rollback in ("rolled_back", "not_applied")
    assert canary.args == ["--max-num-batched-tokens", "8192"]


def test_a_candidate_with_no_real_difference_is_not_kept(tmp_path):
    canary = FakeCanary()
    v = evaluate(_contract(tmp_path), canary, FakeSampler(canary, candidate_tpot=10.0))
    assert v.state != "improved" and canary.args == ["--max-num-batched-tokens", "8192"]


def test_unknown_policy_keys_are_refused():
    with pytest.raises(KeyError):
        merged({"replay_sped": 2})


def test_submitting_needs_a_canary_and_a_real_change(tmp_path):
    from gitm.server.controller import Controller, ServerConfig
    from gitm.server.supervisor import EngineProcess, EngineSpec

    ctl = Controller(ServerConfig(model="m", workdir=tmp_path, serve_args=["--max-num-seqs", "64"]))
    ctl.engines["stable"] = EngineProcess(EngineSpec("stable", "m", 1, ["--max-num-seqs", "64"]),
                                          tmp_path)
    with pytest.raises(ValueError, match="without a canary"):
        ctl.submit({"max_num_seqs": 128})
    ctl.engines["canary"] = EngineProcess(EngineSpec("canary", "m", 2, ["--max-num-seqs", "64"]),
                                          tmp_path)
    with pytest.raises(ValueError, match="already runs"):
        ctl.submit({"max_num_seqs": 64})
    iv = ctl.submit({"max_num_seqs": 128}, submitter="agent")
    assert iv.status == "queued" and (tmp_path / "interventions" / f"{iv.id}.json").exists()
    assert ctl.stop_intervention(iv.id).status == "stopped"


def test_a_sample_spoiled_by_production_load_is_measured_again(tmp_path):
    from gitm.server.canary import ReplaySampler

    loads = iter([0.54, 0.0])            # first attempt: stable busy; second: quiet
    s = ReplaySampler.__new__(ReplaySampler)
    s.shared_gpu, s.max_stable_rps, s.max_retries, s.quiet_wait_s = True, 0.1, 2, 0.0
    s.stable_load = lambda t0, t1: 0.0
    s.cancel = __import__("threading").Event()
    seen = []
    s.on_sample = lambda arm, index, smp: seen.append(smp)
    s._sample_once = lambda arm, index: Sample(metrics={"stable_rps": next(loads)},
                                               wall_s=1, accelerator_s=1, correctness=1.0)
    out = s.sample("baseline", 7)
    assert out.metrics["stable_rps"] == 0.0 and out.metrics["remeasured"] == 1.0
    assert len(seen) == 1                 # only the kept attempt is reported


def test_on_its_own_gpu_a_busy_stable_is_not_a_reason_to_remeasure(tmp_path):
    from gitm.server.canary import ReplaySampler

    s = ReplaySampler.__new__(ReplaySampler)
    s.shared_gpu, s.max_stable_rps, s.max_retries, s.quiet_wait_s = False, 0.1, 2, 0.0
    s.stable_load = lambda t0, t1: 5.0
    s.on_sample = lambda *a: None
    s._sample_once = lambda arm, index: Sample(metrics={"stable_rps": 5.0}, wall_s=1,
                                               accelerator_s=1, correctness=1.0)
    assert s.sample("candidate", 1).metrics["remeasured"] == 0.0


def test_arm_spread_flags_a_machine_that_moved():
    from gitm.server.canary import arm_spread

    def raw(arm, vals):
        return [{"arm": arm, "warmup": False, "metrics": {"mean_tpot_ms": v}} for v in vals]

    steady = {"raw": raw("baseline", [6.27, 6.30, 6.31, 6.30]) + raw("candidate", [6.3] * 4)}
    hot = {"raw": raw("baseline", [6.27, 6.30, 14.55, 6.30]) + raw("candidate", [6.3] * 4)}
    assert arm_spread(steady, "mean_tpot_ms")["baseline"] < 0.01
    assert arm_spread(hot, "mean_tpot_ms")["baseline"] > 0.15


def test_the_sampler_settles_before_measuring_and_stops_when_cancelled():
    import threading
    import time

    from gitm.server.canary import Cancelled, ReplaySampler

    s = ReplaySampler.__new__(ReplaySampler)
    s.cancel, s.settle_s = threading.Event(), 5.0
    threading.Timer(0.05, s.cancel.set).start()
    t0 = time.monotonic()
    with pytest.raises(Cancelled):
        s._sample_once("baseline", 0)
    assert time.monotonic() - t0 < 1.0          # a stop interrupts the settle
