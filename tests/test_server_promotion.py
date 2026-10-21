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


def failed_checks(rules):
    return {c["name"] for r in rules for c in r.checks if not c["passed"]}


KNOBS = {"max_num_batched_tokens": 16384}


def test_a_clean_improvement_answers_all_four_questions():
    rules = promotion.check(KNOBS, verdict(), ctx())
    assert [r.name for r in rules] == ["verdict", "evidence_current", "canary_ready",
                                       "allowed_now"]
    assert failed(rules) == [] and promotion.all_pass(rules)
    # every check is kept as evidence, not just the failures
    assert sum(len(r.checks) for r in rules) == 12


def test_promotion_takes_the_tests_verdict_as_it_is():
    """No second judgement of the effect: an improved verdict with a small
    interval or a noisy baseline is the test's call (its min_effect), not
    promotion's. The 18 rules re-judged it with thresholds of their own."""
    small = verdict(lo=0.021, hi=0.04, base_values=(10, 12, 8))
    assert promotion.all_pass(promotion.check(KNOBS, small, ctx()))
    rules = promotion.check(KNOBS, verdict(state="inconclusive"), ctx())
    assert failed(rules) == ["verdict"] and rules[0].detail.startswith("inconclusive")


@pytest.mark.parametrize("make, question, check", [
    (lambda: (verdict(), ctx(verdict_finished=time.time() - 99999)), "evidence_current",
     "recent"),
    (lambda: (verdict(), ctx(live_regime=regime(input_p50=4000, input_p95=5000))),
     "evidence_current", "same_traffic"),
    (lambda: (verdict(), ctx(canary_placement={"gpus_count": None, "mem_fraction": 0.3})),
     "canary_ready", "same_capacity"),
    (lambda: (verdict(), ctx(canary_runs_candidate=False)), "canary_ready", "runs_candidate"),
    (lambda: (verdict(), ctx(policy=merged({"paused": True}))), "allowed_now", "not_paused"),
    (lambda: (verdict(), ctx(policy=merged({"deny_knobs": ["max_num_*"]}))), "allowed_now",
     "knob_allowed"),
    (lambda: (verdict(), ctx(policy=merged({"allow_knobs": ["enforce_eager"]}))),
     "allowed_now", "knob_allowed"),
    (lambda: (verdict(), ctx(reverted=[{"knobs": promotion.knob_key(KNOBS), "reason": "slow"}])),
     "allowed_now", "not_reverted_before"),
    (lambda: (verdict(), ctx(last_promotion_t=time.time() - 60)), "allowed_now",
     "promotion_interval"),
    (lambda: (verdict(), ctx(last_revert_t=time.time() - 60)), "allowed_now",
     "revert_cooldown"),
])
def test_each_check_can_stop_a_promotion_and_names_itself(make, question, check):
    v, c = make()
    rules = promotion.check(KNOBS, v, c)
    assert failed(rules) == [question] and failed_checks(rules) == {check}
    assert next(r for r in rules if r.name == question).detail.startswith(check)


def test_a_freeze_window_stops_promotion_in_its_hours():
    hour = time.gmtime().tm_hour
    rules = promotion.check(KNOBS, verdict(), ctx(policy=merged(
        {"freeze_hours_utc": [[hour, (hour + 1) % 24]]})))
    assert failed_checks(rules) == {"not_frozen"}
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
    assert "undisturbed" in failed_checks(rules)
    quiet = promotion.check(KNOBS, with_load(verdict(), 0.0), ctx(shared_gpu=True))
    assert "undisturbed" not in failed_checks(quiet)
    unknown = promotion.check(KNOBS, verdict(), ctx(shared_gpu=True))
    assert "undisturbed" in failed_checks(unknown)
    own_gpus = promotion.check(KNOBS, with_load(verdict(), 5.0), ctx(shared_gpu=False))
    assert "undisturbed" not in failed_checks(own_gpus)


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


def test_a_revert_puts_back_the_traffic_baseline_with_the_configuration(server):
    from gitm.traffic.regime import Regime as R

    ctl, stable, canary, routes, iv = server
    ctl.policy["probation_interval_s"] = 3600
    chat = regime().model_dump(mode="json")
    docs = regime(input_p50=4000, input_p95=5000, io_ratio=25.0).model_dump(mode="json")
    ctl.watch.observe(R.model_validate(chat), "w1")          # stable tuned for chat
    ctl.watch.observe(R.model_validate(docs), "w2")          # traffic moves to docs
    iv.regime = docs
    ctl.promote(iv, cause="test")
    assert ctl.watch.validated["input_p50"] == 4000 and not ctl.watch.changed
    assert ctl.watch.validated_how.startswith("promoted")
    ctl.revert("slower live", cause="test")
    assert ctl.watch.validated["input_p50"] == 900           # the old config's baseline
    assert ctl.watch.reference["input_p50"] == 900
    assert ctl.watch.changed                                 # live docs != chat: flagged again
