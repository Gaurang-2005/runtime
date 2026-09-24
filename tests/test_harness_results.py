"""Reading a cluster experiment back into the record the loop ranks from.

The harness runs experiments across a fleet and hands the server to
``gitm capture serve``, which leaves ``serving_summary.json`` and
``run_manifest.json`` per arm. ``load_history`` reads
``runs/<id>/verification.json``, and only ``run_loop`` writes one — so every
result measured on the cluster was invisible to the ranking that reads history.
A loop that proposes experiments and cannot see their results is the failure the
history reader exists to prevent, one layer out.
"""

from __future__ import annotations

import json

import pytest

from gitm.optimizer.harness_results import (
    CaptureError,
    compare,
    knob_difference,
    read_capture,
    write_comparison,
)
from gitm.optimizer.history import load_history, record_for

BASE_ARGV = ["--tensor-parallel-size", "2"]
LOAD = {"requests": 512, "concurrency": 256, "input_tokens": 1024,
        "output_tokens": 256, "seed": 42}


def _arm(root, name, *, argv=None, rps=40.0, model="Kimi-K2.5", tracing="cupti",
         load=None, summary=None, manifest=None):
    """One arm's directory in the shape `gitm capture serve` writes it."""
    d = root / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "serving_summary.json").write_text(json.dumps(summary if summary is not None else {
        "mode": "drive", "tracing": tracing, "nvtx": False, "wall_s": 300.0,
        "client": {"latency_source": "client", "n_failed_requests": 0,
                   "n_requests": 512, "goodput_rps": rps, "window_s": 300.0},
    }))
    (d / "run_manifest.json").write_text(json.dumps(manifest if manifest is not None else {
        "workload_id": "vllm-serve", "capture_mode": "serve", "served_model": model,
        "serve_argv": BASE_ARGV if argv is None else argv,
        "load": LOAD if load is None else load,
    }))
    return d


# --------------------------------------------------------------------------- #
# the round trip                                                               #
# --------------------------------------------------------------------------- #
def test_a_cluster_result_reaches_the_record_the_loop_ranks_from(tmp_path):
    """The whole point: a pair of harness arms becomes a lever the ranking can
    see, keyed the same way a local run would be."""
    base = _arm(tmp_path, "tp2", rps=40.0)
    cand = _arm(tmp_path, "tp2-ep", argv=[*BASE_ARGV, "--enable-expert-parallel"], rps=59.6)

    write_comparison(read_capture(base), read_capture(cand),
                     out_dir=tmp_path / "runs" / "cluster-1",
                     gpu_sku="AMD Instinct MI355X", fingerprint="kimi-k2.5-mi355x")

    rec = record_for(load_history(tmp_path / "runs"), "enable_expert_parallel",
                     gpu_sku="AMD Instinct MI355X", fingerprint="kimi-k2.5-mi355x")
    assert rec is not None
    assert abs(rec.mean_delta - 0.49) < 1e-9
    assert (rec.wins, rec.losses) == (1, 0)


def test_a_measured_win_is_a_win_and_not_a_loss(tmp_path):
    """``kept`` maps to the verdict, and leaving it False because no rollback
    gate ran would record every cluster win as a loss — demoting the lever the
    result proves. A harness arm runs standalone, so there is nothing to roll
    back and the number is the whole question."""
    base = _arm(tmp_path, "b", rps=40.0)
    win = _arm(tmp_path, "w", argv=[*BASE_ARGV, "--enable-expert-parallel"], rps=59.6)
    loss = _arm(tmp_path, "l", argv=[*BASE_ARGV, "--enforce-eager"], rps=36.4)

    assert compare(read_capture(base), read_capture(win)).kept is True
    assert compare(read_capture(base), read_capture(loss)).kept is False


def test_a_change_inside_the_band_is_not_significant(tmp_path):
    base = _arm(tmp_path, "b", rps=40.0)
    noise = _arm(tmp_path, "n", argv=[*BASE_ARGV, "--enable-expert-parallel"], rps=40.4)

    rec = compare(read_capture(base), read_capture(noise))

    assert rec.significant is False
    assert rec.kept is False


# --------------------------------------------------------------------------- #
# arms that are not an A/B                                                     #
# --------------------------------------------------------------------------- #
def test_a_traced_arm_against_an_untraced_one_is_refused(tmp_path):
    """Tracing costs throughput. Comparing across arms would report the tracer's
    overhead as the lever's effect — and the harness's own default arm list makes
    this easy to do by accident."""
    base = _arm(tmp_path, "b", tracing="off", rps=44.0)
    cand = _arm(tmp_path, "c", argv=[*BASE_ARGV, "--enable-expert-parallel"],
                tracing="cupti", rps=40.0)

    with pytest.raises(CaptureError, match="not an A/B"):
        compare(read_capture(base), read_capture(cand))


def test_a_different_load_shape_is_refused(tmp_path):
    base = _arm(tmp_path, "b", rps=40.0)
    cand = _arm(tmp_path, "c", argv=[*BASE_ARGV, "--enable-expert-parallel"], rps=59.6,
                load={**LOAD, "concurrency": 32})

    with pytest.raises(CaptureError, match="not an A/B"):
        compare(read_capture(base), read_capture(cand))


def test_a_different_model_is_refused(tmp_path):
    base = _arm(tmp_path, "b", rps=40.0)
    cand = _arm(tmp_path, "c", argv=[*BASE_ARGV, "--enable-expert-parallel"],
                model="GLM-5.2", rps=59.6)

    with pytest.raises(CaptureError, match="not an A/B"):
        compare(read_capture(base), read_capture(cand))


def test_identical_flags_are_refused_rather_than_recorded_as_a_lever(tmp_path):
    """Two arms of the same config measure run-to-run scatter. Recording that as
    an intervention would put noise in the record under a lever's name."""
    base = _arm(tmp_path, "b", rps=40.0)
    same = _arm(tmp_path, "s", rps=41.0)

    with pytest.raises(CaptureError, match="no intervention"):
        compare(read_capture(base), read_capture(same))


# --------------------------------------------------------------------------- #
# reading a directory                                                          #
# --------------------------------------------------------------------------- #
def test_a_capture_with_no_throughput_raises_rather_than_half_reporting(tmp_path):
    """A comparison built from it would state a delta against nothing."""
    d = _arm(tmp_path, "b", summary={"mode": "drive", "tracing": "cupti",
                                     "client": {"n_failed_requests": 0}})

    with pytest.raises(CaptureError, match="no throughput"):
        read_capture(d)


def test_goodput_of_zero_is_a_measurement_not_a_missing_field(tmp_path):
    """A run that met no SLO really did achieve zero goodput. Falling back to
    raw request rate there would report throughput the run did not deliver."""
    d = _arm(tmp_path, "b", rps=0.0)

    assert read_capture(d).throughput == 0.0


def test_request_rate_is_used_only_when_goodput_is_absent(tmp_path):
    d = _arm(tmp_path, "b", summary={
        "mode": "drive", "tracing": "cupti", "wall_s": 256.0,
        "client": {"n_requests": 512, "window_s": 256.0}})

    assert read_capture(d).throughput == 2.0


def test_a_missing_directory_says_so(tmp_path):
    with pytest.raises(CaptureError, match="not a directory"):
        read_capture(tmp_path / "nope")


def test_malformed_json_names_the_file(tmp_path):
    d = _arm(tmp_path, "b")
    (d / "serving_summary.json").write_text("{ truncated")

    with pytest.raises(CaptureError, match="serving_summary.json"):
        read_capture(d)


# --------------------------------------------------------------------------- #
# which knob moved                                                             #
# --------------------------------------------------------------------------- #
def test_the_knob_is_the_flag_the_candidate_added(tmp_path):
    base = read_capture(_arm(tmp_path, "b"))
    cand = read_capture(_arm(tmp_path, "c", argv=[*BASE_ARGV, "--max-num-seqs", "512"]))

    assert knob_difference(base, cand) == {"--max-num-seqs": "512"}


def test_a_bare_switch_reads_as_true(tmp_path):
    base = read_capture(_arm(tmp_path, "b"))
    cand = read_capture(_arm(tmp_path, "c", argv=[*BASE_ARGV, "--enforce-eager"]))

    assert knob_difference(base, cand) == {"--enforce-eager": True}


def test_a_flag_the_baseline_carries_and_the_candidate_drops_is_not_reported(tmp_path):
    """Removing a knob is a different experiment. Reporting it under the same
    shape would claim the candidate *set* something it unset."""
    base = read_capture(_arm(tmp_path, "b", argv=[*BASE_ARGV, "--enforce-eager"]))
    cand = read_capture(_arm(tmp_path, "c", argv=BASE_ARGV))

    assert knob_difference(base, cand) == {}


def test_the_lever_name_is_stable_across_reads(tmp_path):
    """Read twice, the same experiment must key to one lever — otherwise it
    accumulates as several that were each tried once."""
    base = read_capture(_arm(tmp_path, "b"))
    cand = read_capture(_arm(tmp_path, "c",
                             argv=[*BASE_ARGV, "--enable-expert-parallel", "--max-num-seqs", "512"]))

    first = compare(base, cand).intervention_name
    assert first == compare(base, cand).intervention_name
    assert first == "enable_expert_parallel+max_num_seqs_512"
