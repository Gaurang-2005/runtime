"""gitm server: operator-controlled autonomy, an editable queue, and traffic-aware results."""

from __future__ import annotations

import time

import pytest

from gitm.server.autonomy import in_schedule, new_session
from gitm.server.traffic import (
    TrafficWatch,
    promising_under,
    proven_for,
    tried_for,
)
from gitm.traffic.regime import Regime, SourceKind


def regime(**over) -> Regime:
    base = dict(source_kind=SourceKind.PRODUCTION, trace="t", requests=100, rate_rps=2.0,
                io_ratio=5.0, input_p50=900, input_p95=950, output_p50=200, output_p95=380,
                burstiness=1.0, concurrency=None, in_envelope=True)
    fields = set(Regime.model_fields)
    return Regime(**{k: v for k, v in {**base, **over}.items() if k in fields})


CHAT = regime().model_dump(mode="json")
DOCS = regime(input_p50=4000, input_p95=5000, io_ratio=25.0).model_dump(mode="json")


# ----------------------------------------------------------------- sessions

def test_a_session_ends_on_its_time_or_test_budget():
    s = new_session("autonomous", budget_h=1, max_tests=2, now=1000.0)
    assert s.exhausted(1000.0 + 3599) is None
    assert "time" in s.exhausted(1000.0 + 3601)
    s2 = new_session("assisted", max_tests=2, now=0.0)
    s2.tests_started = 2
    assert "test budget" in s2.exhausted(10.0)


def test_sessions_refuse_what_cannot_run():
    for bad in [dict(mode="manual"), dict(mode="autonomous", budget_h=0),
                dict(mode="assisted", max_tests=0),
                dict(mode="assisted", schedule=[[1, 30]])]:
        mode = bad.pop("mode")
        with pytest.raises(ValueError):
            new_session(mode, **bad)


def test_schedule_windows_include_ones_that_wrap_midnight():
    t = time.mktime((2026, 10, 18, 2, 30, 0, 0, 0, -1))      # 02:30 local
    assert in_schedule([[1, 6]], t) and in_schedule([[22, 6]], t)
    assert not in_schedule([[9, 17]], t)
    assert in_schedule([], t)


def test_a_sessions_scope_can_only_narrow_the_policy():
    from gitm.server.policy import merged

    s = new_session("assisted", allow_knobs=["max_num_*"], deny_knobs=["max_num_seqs"],
                    tiers=["low_risk"])
    p = s.scoped_policy(merged({"deny_knobs": ["block_size"]}))
    assert p["allow_knobs"] == ["max_num_*"]
    assert p["deny_knobs"] == ["block_size", "max_num_seqs"]
    assert p["explore_tiers"] == ["low_risk"]


# ----------------------------------------------------------------- controller

class FakeEngine:
    def __init__(self, role, port, args):
        from gitm.server.supervisor import EngineSpec

        self.spec = EngineSpec(role, "m", port, list(args), mem_fraction=0.4)
        self.alive = True

    @property
    def base_url(self):
        return f"http://127.0.0.1:{self.spec.port}"

    def resolved_config(self):
        return {}


@pytest.fixture
def ctl(tmp_path, monkeypatch):
    from gitm.server.controller import Controller, ServerConfig

    c = Controller(ServerConfig(model="m", workdir=tmp_path))
    c.engines = {"stable": FakeEngine("stable", 8100, ["--max-num-seqs", "64"]),
                 "canary": FakeEngine("canary", 8101, ["--max-num-seqs", "64"])}
    return c


def test_the_queue_runs_in_order_and_can_be_reordered(ctl):
    a = ctl.submit({"max_num_seqs": 128})
    b = ctl.submit({"max_num_seqs": 256})
    c = ctl.submit({"max_num_seqs": 32}, front=True)
    assert [q["id"] for q in ctl.queue_view()] == [c.id, a.id, b.id]
    ctl.move(b.id, 0)
    assert [q["id"] for q in ctl.queue_view()] == [b.id, c.id, a.id]
    ctl.stop_intervention(c.id)
    assert [q["id"] for q in ctl.queue_view()] == [b.id, a.id]
    assert ctl._next_runnable() is b


def test_gitms_own_picks_wait_for_the_schedule_but_never_block_a_persons(ctl):
    ctl.session = new_session("assisted", schedule=[[(time.localtime().tm_hour + 2) % 24,
                                                     (time.localtime().tm_hour + 3) % 24]])
    auto = ctl._auto_submit({"max_num_seqs": 128}, "explorer", "x")
    mine = ctl.submit({"max_num_seqs": 256})
    view = ctl.queue_view()
    assert view[0]["id"] == auto.id and view[0]["waits"] == "outside the session's schedule"
    assert ctl._next_runnable() is mine          # skipped over, not blocking
    assert ctl._next_runnable() is None          # the auto pick still waits


def test_stopping_a_session_drops_gitms_picks_and_keeps_a_persons(ctl):
    ctl.session = new_session("autonomous", max_tests=5)
    auto = ctl._auto_submit({"max_num_seqs": 128}, "explorer", "x")
    mine = ctl.submit({"max_num_seqs": 256})
    out = ctl.stop_session()
    assert out["dropped"] == [auto.id] and ctl.mode == "manual"
    assert [q["id"] for q in ctl.queue_view()] == [mine.id]


def test_the_test_budget_counts_only_gitms_own_picks(ctl):
    ctl.session = new_session("assisted", max_tests=1)
    ctl.submit({"max_num_seqs": 512})                       # a person's: not counted
    assert ctl._auto_submit({"max_num_seqs": 128}, "explorer", "x") is not None
    assert ctl._auto_submit({"max_num_seqs": 256}, "explorer", "x") is None
    ctl._end_session_if_over()
    assert ctl.session is None


@pytest.mark.parametrize("mode, expect", [(None, "awaiting_approval"),
                                          ("assisted", "awaiting_approval"),
                                          ("autonomous", "promoted")])
def test_only_autonomous_mode_promotes_by_itself(ctl, monkeypatch, mode, expect):
    from gitm.server.controller import Intervention
    from gitm.server.promotion import Rule

    if mode:
        ctl.session = new_session(mode)
    monkeypatch.setattr(ctl, "rules_for", lambda iv: [Rule("all", True, "")])
    promoted = []
    monkeypatch.setattr(ctl, "promote", lambda iv, cause: promoted.append(cause))
    iv = Intervention(id="i", knobs={"max_num_seqs": 128}, submitter="t", note="",
                      created=0, status="improved")
    ctl._consider(iv)
    assert (iv.status == "awaiting_approval") == (expect == "awaiting_approval")
    assert bool(promoted) == (expect == "promoted")


# ----------------------------------------------------------------- traffic

BURST = regime(input_p50=150, input_p95=300, io_ratio=0.5,
               burstiness=16.0).model_dump(mode="json")


def test_the_watch_flags_a_change_and_a_return():
    w = TrafficWatch(threshold=1.0, steady_windows=3)
    assert w.observe(Regime.model_validate(CHAT), "w1") is None     # first sight
    assert "not yet tested" in w.validated_how
    assert w.observe(Regime.model_validate(CHAT), "w2") is None
    assert w.observe(Regime.model_validate(DOCS), "w3") == "changed"
    assert w.observe(Regime.model_validate(DOCS), "w4") is None     # reported once
    assert w.observe(Regime.model_validate(CHAT), "w5") == "settled"
    w.observe(Regime.model_validate(DOCS), "w6")
    w.validate(DOCS, how="test")
    assert not w.changed and w.last_distance == 0.0 and w.validated_how == "test"


def test_detection_rearms_on_steady_new_traffic_without_validating_it():
    """No one has to declare a new normal: the next change is still caught."""
    w = TrafficWatch(threshold=1.0, steady_windows=3)
    w.observe(Regime.model_validate(CHAT), "w1")
    assert w.observe(Regime.model_validate(DOCS), "w2") == "changed"
    assert w.observe(Regime.model_validate(DOCS), "w3") is None     # steady 2 of 3
    assert w.observe(Regime.model_validate(DOCS), "w4") == "rearmed"
    assert w.reference["input_p50"] == 4000                         # changes measured from docs
    assert w.validated["input_p50"] == 900                          # but validated on chat only
    assert w.observe(Regime.model_validate(BURST), "w5") == "changed"   # the next change


def test_a_flapping_change_does_not_rearm():
    w = TrafficWatch(threshold=1.0, steady_windows=3)
    w.observe(Regime.model_validate(CHAT), "w1")
    w.observe(Regime.model_validate(DOCS), "w2")
    w.observe(Regime.model_validate(BURST), "w3")                    # not steady: restart count
    w.observe(Regime.model_validate(BURST), "w4")
    assert w.changed and w.reference["input_p50"] == 900


def row(knob_key, status, regime_, effect=None, stable=("--x",)):
    return {"id": knob_key, "knob_key": knob_key, "knobs": {"k": knob_key}, "status": status,
            "effect": effect, "regime": regime_, "stable_args": list(stable)}


def test_tried_means_tried_for_this_kind_of_traffic():
    results = [row("a", "regressed", CHAT), row("b", "rejected", None),
               row("c", "queued", None)]
    stable = ["--x"]
    assert tried_for(results, "a", stable, CHAT, 1.0)          # same traffic: tried
    assert not tried_for(results, "a", stable, DOCS, 1.0)      # new traffic: try again
    assert tried_for(results, "b", stable, DOCS, 1.0)          # never measurable: don't loop
    assert tried_for(results, "c", stable, DOCS, 1.0)          # already waiting
    assert not tried_for(results, "a", ["--y"], CHAT, 1.0)     # another stable: untried


def test_memory_finds_what_was_proven_and_what_was_promising():
    results = [row("a", "kept", DOCS, 0.08), row("b", "improved", CHAT, 0.05),
               row("c", "inconclusive", CHAT, 0.01), row("d", "regressed", CHAT, -0.04)]
    assert [r["id"] for r in proven_for(results, DOCS, 1.0)] == ["a"]
    assert [r["id"] for r in promising_under(results, CHAT, 1.0)] == ["b", "c"]


def test_a_traffic_change_requeues_recheck_then_proven_then_promising(ctl):
    from gitm.server.controller import Intervention

    ctl.session = new_session("assisted")
    ctl.state["promotions"].append({"outcome": "kept", "knobs": {"max_num_seqs": 64},
                                    "from_args": ["--max-num-seqs", "32"]})
    for ident, knobs, status, reg, eff in [
            ("p", {"max_num_batched_tokens": 8192}, "kept", DOCS, 0.07),
            ("q", {"enable_chunked_prefill": False}, "improved", CHAT, 0.03),
            ("r", {"block_size": 32}, "regressed", CHAT, -0.02)]:
        ctl.interventions[ident] = Intervention(
            id=ident, knobs=knobs, submitter="t", note="", created=0, status=status,
            regime=reg, stable_args=["--max-num-seqs", "64"], verdict={"effect": eff})
    ctl.watch.observe(Regime.model_validate(CHAT), "w1")
    ctl.watch.observe(Regime.model_validate(DOCS), "w2")
    queued = ctl._react_to_traffic_change(CHAT)
    got = [ctl.interventions[i] for i in queued]
    assert [g.submitter for g in got] == ["recheck", "playbook", "rediscover"]
    assert got[0].knobs == {"max_num_seqs": "32"}                  # the change reversed
    assert got[1].knobs == {"max_num_batched_tokens": 8192}        # proven on docs
    assert got[2].knobs == {"enable_chunked_prefill": False}       # won on the old chat
    assert ctl.session.tests_started == 3


def test_manual_mode_only_reports_a_traffic_change(ctl):
    assert ctl._react_to_traffic_change(CHAT) == []


def test_results_memory_survives_a_controller_restart(tmp_path, ctl):
    from gitm.server.controller import Controller, ServerConfig

    iv = ctl.submit({"max_num_seqs": 128})
    iv.status, iv.regime = "regressed", CHAT
    ctl._save(iv)
    queued = ctl.submit({"max_num_seqs": 256})
    again = Controller(ServerConfig(model="m", workdir=tmp_path))
    again._load_history()
    assert again.interventions[iv.id].regime == CHAT
    assert again.interventions[queued.id].status == "stopped"


def test_a_sessions_scope_applies_to_every_pick_not_only_the_explorers(ctl):
    from gitm.server.controller import Intervention

    ctl.session = new_session("autonomous", allow_knobs=["enforce_eager"])
    ctl.interventions["p"] = Intervention(
        id="p", knobs={"max_num_batched_tokens": 8192}, submitter="t", note="", created=0,
        status="kept", regime=DOCS, stable_args=["--max-num-seqs", "64"],
        verdict={"effect": 0.07})
    ctl.watch.observe(Regime.model_validate(CHAT), "w1")
    ctl.watch.observe(Regime.model_validate(DOCS), "w2")
    assert ctl._react_to_traffic_change(CHAT) == []          # proven, but out of scope
    assert ctl._auto_submit({"enforce_eager": False}, "explorer", "x") is not None


@pytest.mark.parametrize("change", [{"repetitions": 1}, {"repetitions": 2.5},
                                    {"repetitions": True}, {"settle_s": -1},
                                    {"replay_speed": 0}, {"replay_speed": "fast"}])
def test_test_settings_outside_their_range_are_refused(change):
    from gitm.server.policy import merged

    with pytest.raises(ValueError):
        merged(change)


def test_two_samples_are_allowed_but_say_they_cannot_promote():
    from gitm.server.policy import merged, warnings_for

    assert merged({"repetitions": 2})["repetitions"] == 2
    assert "never promote" in warnings_for(merged({"repetitions": 2}))[0]
    assert warnings_for(merged({"repetitions": 4})) == []


def changed_watch(ctl):
    ctl.watch.observe(Regime.model_validate(CHAT), "w1")
    ctl.watch.observe(Regime.model_validate(DOCS), "w2")
    assert ctl.watch.changed


@pytest.mark.parametrize("mode, accepts", [("autonomous", True), ("assisted", True),
                                           (None, False)])
def test_a_session_validates_stable_on_new_traffic_once_retested(ctl, mode, accepts):
    if mode:
        ctl.session = new_session(mode)
    changed_watch(ctl)
    assert ctl._accept_if_revalidated() is accepts
    assert ctl.watch.changed is (not accepts)
    if accepts:
        assert ctl.watch.validated["input_p50"] == 4000      # stable now validated on docs
        assert "nothing beat stable" in ctl.watch.validated_how


def test_autonomous_does_not_accept_while_a_retest_is_still_queued(ctl):
    ctl.session = new_session("autonomous")
    changed_watch(ctl)
    ctl._auto_submit({"max_num_seqs": 128}, "rediscover", "x")
    assert ctl._accept_if_revalidated() is False and ctl.watch.changed


class RelaunchEngine(FakeEngine):
    def __init__(self, *a):
        super().__init__(*a)
        self.relaunches = 0

    def stop(self):
        self.alive = False

    def start(self):
        self.alive = True
        self.relaunches += 1

    def wait_healthy(self, timeout_s=0):
        pass


@pytest.fixture
def approving(ctl, monkeypatch):
    from gitm.server.controller import Intervention

    ctl.engines["canary"] = RelaunchEngine("canary", 8101, ["--max-num-seqs", "256"])
    monkeypatch.setattr("gitm.server.canary.warm_up", lambda engine: None)
    promoted = []

    def promote(iv, cause):
        promoted.append((iv.id, list(ctl.engines["canary"].spec.serve_args), cause))
        iv.status = "promoted"
        return {"promoted": iv.id}

    monkeypatch.setattr(ctl, "promote", promote)
    iv = Intervention(id="w", knobs={"max_num_seqs": 128}, submitter="t", note="", created=0,
                      status="awaiting_approval", candidate_args=["--max-num-seqs", "128"])
    ctl.interventions["w"] = iv
    return ctl, iv, promoted


def test_approving_puts_the_win_back_on_the_canary_before_promoting(approving):
    ctl, iv, promoted = approving                       # canary has moved on to another test
    ctl.approve(iv, by="gaurang")
    assert promoted == [("w", ["--max-num-seqs", "128"], "approved: gaurang")]
    assert ctl.engines["canary"].relaunches == 1


def test_approving_while_a_test_holds_the_canary_waits_for_it(approving):
    ctl, iv, promoted = approving
    with ctl._canary_lock:                              # a test is running
        out = ctl.approve(iv, by="gaurang")
    assert out["pending"] and iv.status == "approved" and not promoted
    ctl._carry_out_approvals()                          # the worker, once it is free
    assert promoted and iv.status == "promoted"


def test_failing_rules_on_approval_leave_the_win_waiting_with_the_reason(approving, monkeypatch):
    ctl, iv, promoted = approving

    def refuse(iv, cause):
        raise ValueError("promotion rules failed: ['traffic_unchanged']")

    monkeypatch.setattr(ctl, "promote", refuse)
    with pytest.raises(ValueError):
        ctl.approve(iv, by="gaurang")
    assert iv.status == "awaiting_approval" and "traffic_unchanged" in iv.detail


def test_declining_changes_nothing_and_cannot_be_approved_after(approving):
    ctl, iv, promoted = approving
    ctl.decline(iv, by="gaurang", reason="not during the launch")
    assert iv.status == "declined" and "not during the launch" in iv.detail and not promoted
    with pytest.raises(ValueError):
        ctl.approve(iv, by="gaurang")
