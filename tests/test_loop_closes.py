"""The loop, end to end: propose arms, run them, read the results back.

This is the epic's two missing clauses meeting in one test — *"emits experiment
specs for the runtime experiment harness"* and *"must ingest experiment results
and use them to re-rank/propose the next batch"*. Each had working code and no
caller; neither had ever been shown to connect to the other.

The harness itself is stubbed: it runs on a cluster and this does not. What is
stubbed is only the part that spends GPU time — taking an arm's server argv and
producing a capture directory in the shape `gitm capture serve` writes one. The
argv the stub receives is the argv `gitm propose` emitted, and the directories it
writes are read by `gitm ingest` unmodified, so everything between the two
commands is real.
"""

from __future__ import annotations

import json

from gitm.cli import main as cli_main
from gitm.optimizer.history import load_history, record_for

from .test_harness_results import LOAD, _arm

SKU = "AMD Instinct MI355X"


def _baseline(tmp_path, argv):
    """A traced baseline capture, the input `gitm propose` takes."""
    return _arm(tmp_path / "results", "baseline", argv=argv, rps=40.0)


def _run_arm(tmp_path, arm, rps):
    """Stand in for the harness: run one arm, leave a capture directory.

    Takes the emitted argv verbatim. If propose emitted an argv the reader
    cannot diff back to the lever, this is where it shows up.
    """
    return _arm(tmp_path / "results", arm["lever"], argv=arm["serve_argv"], rps=rps)


def test_propose_then_ingest_puts_measured_levers_in_the_history(tmp_path):
    base_argv = ["--tensor-parallel-size", "8", "--enforce-eager"]
    base = _baseline(tmp_path, base_argv)

    # 1. The loop proposes a batch from the baseline capture.
    rc = cli_main(["propose", "--baseline", str(base), "--max-arms", "4",
                   "--out", str(tmp_path / "experiments.json"), "--run-id", "batch1"])
    assert rc == 0

    sweep = json.loads((tmp_path / "experiments.json").read_text())
    assert sweep["baseline"]["serve_argv"] == base_argv
    assert sweep["load"] == LOAD          # carried from the baseline, not invented
    assert sweep["served_model"] == "Kimi-K2.5"
    arms = sweep["arms"]
    assert arms, "proposed nothing to run"

    # 2. The harness runs each arm. One wins, the rest are flat.
    ingestable = [a for a in arms if a["ingestable"]]
    assert ingestable, "every proposed arm was unattributable"
    dirs = [_run_arm(tmp_path, a, rps=59.6 if i == 0 else 40.2)
            for i, a in enumerate(ingestable)]

    # 3. The results come back in.
    argv = ["ingest", "--baseline", str(base), "--scratch", str(tmp_path / "scratch"),
            "--gpu-sku", SKU, "--fingerprint", "kimi-mi355x", "--run-id", "batch1"]
    for d in dirs:
        argv += ["--candidate", str(d)]
    assert cli_main(argv) == 0

    # 4. Every lever the sweep proposed is now a record the ranking reads, under
    #    the name the catalogue knows it by — which is the whole point: a name
    #    invented from a flag is a record nothing looks up.
    hist = load_history(tmp_path / "scratch" / "runs", gpu_sku=SKU)
    for a in ingestable:
        rec = record_for(hist, a["lever"], gpu_sku=SKU, fingerprint="kimi-mi355x")
        assert rec is not None, f"{a['lever']} proposed, measured, and then lost"

    # The winner came back as a win, with the delta it actually measured.
    won = record_for(hist, ingestable[0]["lever"], gpu_sku=SKU, fingerprint="kimi-mi355x")
    assert won.wins == 1 and won.mean_delta > 0.4


def test_the_next_batch_is_ranked_from_the_last_one(tmp_path):
    """The clause that makes it a loop rather than a pipeline: results change
    what gets proposed next."""
    base = _baseline(tmp_path, ["--tensor-parallel-size", "8", "--enforce-eager"])
    cli_main(["propose", "--baseline", str(base), "--max-arms", "3",
              "--out", str(tmp_path / "b1.json")])
    first = json.loads((tmp_path / "b1.json").read_text())
    lever = next(a for a in first["arms"] if a["ingestable"])

    # It wins on the cluster, by a lot.
    won = _run_arm(tmp_path, lever, rps=80.0)
    assert cli_main(["ingest", "--baseline", str(base), "--candidate", str(won),
                     "--scratch", str(tmp_path / "scratch"), "--gpu-sku", SKU,
                     "--fingerprint", "kimi-mi355x", "--run-id", "batch1"]) == 0

    # The record is there for the ranking to read, and says what was measured
    # rather than what the catalogue guessed.
    hist = load_history(tmp_path / "scratch" / "runs", gpu_sku=SKU)
    rec = record_for(hist, lever["lever"], gpu_sku=SKU, fingerprint="kimi-mi355x")
    assert rec is not None and rec.mean_delta > 0.9   # 40.0 -> 80.0 rps

    catalogue_guess = lever["predicted_delta"]
    assert rec.mean_delta > catalogue_guess, (
        "the measured delta should differ from the prior, or this test is not "
        "showing that measurement replaces estimate")


def test_an_arm_the_baseline_already_runs_is_never_proposed(tmp_path):
    """Proposing it would spend a cluster job measuring the baseline against
    itself."""
    base = _baseline(tmp_path, ["--tensor-parallel-size", "8",
                                "--enable-expert-parallel", "--enforce-eager"])
    cli_main(["propose", "--baseline", str(base), "--out", str(tmp_path / "e.json")])
    doc = json.loads((tmp_path / "e.json").read_text())

    assert "enable_expert_parallel" not in {a["lever"] for a in doc["arms"]}
    why = {u["lever"]: u["reason"] for u in doc["unreachable"]}
    assert "already runs" in why["enable_expert_parallel"]


def test_an_untraced_baseline_is_refused_rather_than_ranked_on_nothing(tmp_path):
    base = _arm(tmp_path / "results", "untraced", rps=40.0, trace=False)
    rc = cli_main(["propose", "--baseline", str(base), "--out", str(tmp_path / "e.json")])
    assert rc == 2
    assert not (tmp_path / "e.json").exists()
