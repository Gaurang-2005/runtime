"""gitm server, milestone 3: promotion rules, probation, swap and revert (no GPU)."""

from __future__ import annotations

import time

import pytest

from gitm.server import promotion
from gitm.server.policy import merged
from gitm.traffic.regime import Regime, SourceKind


def regime(**over):
    base = dict(source_kind=SourceKind.PRODUCTION, trace="t", requests=100, rate_rps=2.0,
                io_ratio=5.0, input_p50=900, input_p95=950, output_p50=200, output_p95=380,
                burstiness=1.0, concurrency=None, in_envelope=True)
    fields = set(Regime.model_fields)
    return Regime(**{k: v for k, v in {**base, **over}.items() if k in fields})


def verdict(state="improved", lo=0.10, hi=0.20, base_values=(10.0, 10.1, 9.9)):
    raw = [{"arm": "baseline", "order_index": i, "warmup": False,
            "metrics": {"mean_tpot_ms": v, "failed": 0}} for i, v in enumerate(base_values)]
    raw += [{"arm": "candidate", "order_index": 10 + i, "warmup": False,
             "metrics": {"mean_tpot_ms": 8.5, "failed": 0}} for i in range(3)]
    return {"state": state, "interval": [lo, hi], "effect": (lo + hi) / 2, "n_candidate": 3,
            "correctness": {"status": "pass", "detail": "4/4"},
            "latency": [{"name": "p99_ttft_ms", "status": "pass"}], "raw": raw}


def ctx(**over):
    now = time.time()
    base = dict(policy=merged(None), now=now, verdict_finished=now - 60,
                tested_regime=regime(), live_regime=None,
                stable_placement={"gpus_count": None, "mem_fraction": 0.4},
                canary_placement={"gpus_count": None, "mem_fraction": 0.4},
                canary_alive=True, canary_runs_candidate=True)
    base.update(over)
    return promotion.PromotionContext(**base)


def failed(rules):
    return [r.name for r in rules if not r.passed]


KNOBS = {"max_num_batched_tokens": 16384}


def test_the_first_real_cuda_graph_win_passes_the_noise_rules():
    """The RTX 4050 control: +11.0%, CI [+5.5%, +16.4%], baseline spread 3.73%."""
    v = verdict(lo=0.0555, hi=0.1645, base_values=(7.28, 7.18, 7.36, 6.76))
    v["effect"] = 0.110
    rules = {r.name: r for r in promotion.check(KNOBS, v, ctx())}
    assert rules["clears_noise"].passed and rules["interval_meaningful"].passed


def test_a_clean_improvement_passes_every_rule():
    rules = promotion.check(KNOBS, verdict(), ctx())
    assert failed(rules) == [] and promotion.all_pass(rules)
    assert len(rules) >= 15   # every rule reported, not just the failures


@pytest.mark.parametrize("make, rule", [
    (lambda: (verdict(state="inconclusive"), ctx()), "verdict_improved"),
    (lambda: (verdict(lo=0.01), ctx()), "interval_meaningful"),        # low end < min_effect
    (lambda: (verdict(lo=0.03, hi=0.09, base_values=(10, 12, 8)), ctx()), "clears_noise"),
    (lambda: (verdict(), ctx(verdict_finished=time.time() - 99999)), "verdict_recent"),
    (lambda: (verdict(), ctx(live_regime=regime(input_p50=4000, input_p95=5000))),
     "traffic_unchanged"),
    (lambda: (verdict(), ctx(canary_placement={"gpus_count": None, "mem_fraction": 0.3})),
     "same_capacity"),
    (lambda: (verdict(), ctx(canary_runs_candidate=False)), "canary_ready"),
    (lambda: (verdict(), ctx(policy=merged({"paused": True}))), "not_paused"),
    (lambda: (verdict(), ctx(policy=merged({"deny_knobs": ["max_num_*"]}))), "knobs_allowed"),
    (lambda: (verdict(), ctx(policy=merged({"allow_knobs": ["enforce_eager"]}))),
     "knobs_allowed"),
    (lambda: (verdict(), ctx(reverted=[{"knobs": promotion.knob_key(KNOBS), "reason": "slow"}])),
     "not_reverted_before"),
    (lambda: (verdict(), ctx(last_promotion_t=time.time() - 60)), "promotion_interval"),
    (lambda: (verdict(), ctx(last_revert_t=time.time() - 60)), "revert_cooldown"),
])
def test_each_rule_can_stop_a_promotion(make, rule):
    v, c = make()
    assert rule in failed(promotion.check(KNOBS, v, c))


def test_a_freeze_window_stops_promotion_in_its_hours():
    hour = time.gmtime().tm_hour
    rules = promotion.check(KNOBS, verdict(), ctx(policy=merged(
        {"freeze_hours_utc": [[hour, (hour + 1) % 24]]})))
    assert "not_frozen" in failed(rules)
    assert promotion._in_window(23, [22, 2]) and promotion._in_window(1, [22, 2])
    assert not promotion._in_window(12, [22, 2])


def stat(ok=True, out=101, ttft=0.1, lat=1.1):
    return {"ok": ok, "output_tokens": out, "ttft_s": ttft, "latency_s": lat}


def test_decode_speed_is_per_request_tokens_after_the_first():
    assert promotion.decode_speed([stat(), stat(lat=2.1)]) == pytest.approx(75.0)
    assert promotion.decode_speed([stat(ok=False)]) is None


def test_probation_reverts_on_a_sustained_slowdown_but_not_on_one_bad_interval():
    p = promotion.Probation(100.0, merged({"probation_window": 3, "probation_tolerance": 0.05}))
    assert p.observe([stat(lat=1.15)], 1) is None             # a mild blip (95 tok/s)
    assert p.observe([stat()], 2) is None
    assert p.observe([stat()], 3) is None                     # window mean -1.6%: fine
    for t in range(4, 7):
        reason = p.observe([stat(lat=2.1)], t)               # 50 tok/s sustained
    assert reason and "decode speed" in reason


def test_probation_reverts_at_once_on_errors():
    p = promotion.Probation(100.0, merged(None))
    assert "error rate" in p.observe([stat(), stat(ok=False)], 1)


def test_probation_with_no_baseline_still_watches_errors():
    p = promotion.Probation(None, merged(None))
    assert p.observe([stat()], 1) is None
    assert p.observe([stat(ok=False)] * 3, 2)


# ----------------------------------------------------------------- the swap


class FakeEngine:
    def __init__(self, role, port, args):
        from gitm.server.supervisor import EngineSpec

        self.spec = EngineSpec(role, "m", port, list(args), mem_fraction=0.4)
        self.alive, self.started, self.stopped = True, 0, 0

    @property
    def base_url(self):
        return f"http://127.0.0.1:{self.spec.port}"

    def start(self):
        self.started += 1
        self.alive = True

    def stop(self):
        self.stopped += 1
        self.alive = False

    def wait_healthy(self, timeout_s=0):
        pass

    def info(self):
        return {"role": self.spec.role, "startup_s": 0}


@pytest.fixture
def server(tmp_path, monkeypatch):
    from gitm.server.controller import Controller, Intervention, ServerConfig

    ctl = Controller(ServerConfig(model="m", workdir=tmp_path))
    stable = FakeEngine("stable", 8100, ["--max-num-batched-tokens", "8192"])
    canary = FakeEngine("canary", 8101, ["--max-num-batched-tokens", "16384"])
    ctl.engines = {"stable": stable, "canary": canary}
    routes = []

    def proxy_call(method, path, json=None, params=None, timeout=10.0):
        if path == "/admin/upstream":
            routes.append(json["url"])
            return {"drained": True}
        if path == "/admin/stats":
            return [{"upstream": routes[-1] if routes else stable.base_url, **stat()}]
        return []

    monkeypatch.setattr(ctl, "proxy_call", proxy_call)
    monkeypatch.setattr(ctl, "rules_for", lambda iv: [promotion.Rule("all", True, "")])
    iv = Intervention(id="iv1", knobs=KNOBS, submitter="t", note="", created=time.time(),
                      status="improved", candidate_args=canary.spec.serve_args)
    ctl.interventions["iv1"] = iv
    return ctl, stable, canary, routes, iv


def test_promotion_routes_to_the_canary_and_keeps_the_old_engine_on_standby(server):
    ctl, stable, canary, routes, iv = server
    ctl.policy["probation_interval_s"] = 3600   # keep the watcher thread asleep
    ctl.promote(iv, cause="test")
    assert routes == [canary.base_url]
    assert ctl.engines == {"stable": canary, "standby": stable}
    assert stable.alive and iv.status == "promoted" and ctl.probation is not None
    assert (ctl.workdir / "state.json").exists() and (ctl.workdir / "audit.jsonl").exists()
    with pytest.raises(ValueError, match="probation"):
        ctl.promote(iv, cause="again")


def test_a_revert_routes_back_and_blocks_the_change(server):
    ctl, stable, canary, routes, iv = server
    ctl.policy["probation_interval_s"] = 3600
    ctl.promote(iv, cause="test")
    ctl.revert("slower in production", cause="test")
    assert routes[-1] == stable.base_url and ctl.engines["stable"] is stable
    assert ctl.probation is None and iv.status == "reverted"
    assert ctl.state["reverted"][-1]["knobs"] == promotion.knob_key(KNOBS)
    with pytest.raises(ValueError, match="nothing to revert"):
        ctl.revert("again", cause="test")


def test_passing_probation_stops_the_old_engine_and_reuses_its_slot_as_canary(server):
    ctl, stable, canary, routes, iv = server
    ctl.policy["probation_interval_s"] = 3600
    ctl.promote(iv, cause="test")
    ctl._finish_probation()
    assert ctl.engines["stable"] is canary and ctl.engines["canary"] is stable
    assert stable.stopped == 1 and stable.started == 1
    assert stable.spec.serve_args == canary.spec.serve_args   # new canary runs new stable
    assert iv.status == "kept" and ctl.state["promotions"][-1]["outcome"] == "kept"


def test_probation_watcher_reverts_a_promoted_engine_that_dies(server):
    ctl, stable, canary, routes, iv = server
    ctl.policy["probation_interval_s"] = 0.01
    ctl.promote(iv, cause="test")
    canary.alive = False
    for _ in range(200):
        if ctl.probation is None:
            break
        time.sleep(0.01)
    assert iv.status == "reverted" and "exited" in iv.detail


def with_load(v, rps):
    for r in v["raw"]:
        r["metrics"]["stable_rps"] = rps
    return v


def test_a_shared_gpu_measurement_under_production_load_is_not_promoted():
    rules = promotion.check(KNOBS, with_load(verdict(), 1.0), ctx(shared_gpu=True))
    assert "isolated_measurement" in failed(rules)
    quiet = promotion.check(KNOBS, with_load(verdict(), 0.0), ctx(shared_gpu=True))
    assert "isolated_measurement" not in failed(quiet)
    unknown = promotion.check(KNOBS, verdict(), ctx(shared_gpu=True))
    assert "isolated_measurement" in failed(unknown)
    own_gpus = promotion.check(KNOBS, with_load(verdict(), 5.0), ctx(shared_gpu=False))
    assert "isolated_measurement" not in failed(own_gpus)


def test_engines_share_a_gpu_unless_both_are_pinned_apart():
    from gitm.server.controller import shares_gpus

    a, b = FakeEngine("stable", 1, []), FakeEngine("canary", 2, [])
    assert shares_gpus(a, b)                       # neither pinned: both see every GPU
    a.spec.gpus, b.spec.gpus = "0,1,2,3", "4,5,6,7"
    assert not shares_gpus(a, b)
    b.spec.gpus = "3,4"
    assert shares_gpus(a, b)


def test_correctness_compares_the_arms_choices_position_by_position():
    from gitm.server.canary import agreement

    ref = [[1, 2, 3, 4], [5, 6]]
    assert agreement(ref, ref) == 1.0              # an arm against itself
    assert agreement([[1, 2, 3, 9], [5, 6]], ref) == pytest.approx(5 / 6)


def test_probation_falls_back_to_the_ab_baseline_when_stable_was_quiet(server):
    ctl, stable, canary, routes, iv = server
    ctl.proxy_call = lambda method, path, json=None, params=None, timeout=10.0: (
        {"drained": True} if path == "/admin/upstream" else [])
    iv.verdict = verdict(base_values=(8.0, 8.0, 8.0))
    speed, source = ctl._probation_baseline(iv, stable)
    assert speed == pytest.approx(125.0) and "A/B baseline arm" in source
    iv.verdict = None
    assert ctl._probation_baseline(iv, stable)[0] is None
