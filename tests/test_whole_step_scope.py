"""A lever that acts on the whole step, on whatever architecture is running.

``applies_to_kernels`` names ops. Twelve engine and scheduler levers — batch
shape, admission order, graph capture, sharding degree — do not act on named ops
at all, and were written as a list of the *dense* model's six. On a sparse-MoE
checkpoint none of those op names occur, so every one of them scored zero
coverage, and ``coverage x effect`` made the whole catalog score zero: the
ranking fell through to its alphabetical tie-break on exactly the models being
run. A measured result could not rescue it either, because zero times anything
is still zero.
"""

from __future__ import annotations

from gitm.kernels.library import load_library
from gitm.kernels.spec import Applicability, InterventionSpec, SafetyGate
from gitm.optimizer.replay import predict_delta
from gitm.tracer.schema import KernelEvent, Trace

# Kernels a sparse-MoE decode step actually emits; none classify to a dense op.
MOE = ["fused_moe_kernel", "moe_align_block_size_kernel", "topk_softmax_kernel"]
DENSE = ["void gemm_kernel", "flash_fwd_kernel"]
UNMODELED = ["some_kernel_nobody_models", "void reduce_kernel"]


def _trace(names) -> Trace:
    events = [
        KernelEvent(name=n, start_ns=i * 1000, end_ns=i * 1000 + 1000, stream_id=7,
                    device_id=0, correlation_id=i)
        for i, n in enumerate(names)
    ]
    return Trace(workload_id="vllm-decode", fingerprint="f", run_id="r", device_count=1,
                 vendor="nvidia", captured_at_ns=0, duration_ns=len(names) * 1000,
                 events=events)


def _spec(name, *, whole_step=False, kernels=()) -> InterventionSpec:
    return InterventionSpec(
        name=name, summary="s", knob=name, value=1,
        expected_delta_mean=0.10, expected_delta_lo=0.05, expected_delta_hi=0.15,
        source="t", whole_step=whole_step, applies_to_kernels=list(kernels),
        applicability=Applicability(workloads=["vllm-decode"]),
        safety=SafetyGate(tier="low_risk"),
    )


def test_a_whole_step_lever_covers_an_architecture_it_has_never_heard_of():
    """Batch shape and admission order do not care which ops a checkpoint has.
    Enumerating one architecture's op names is what made them invisible on
    every other one."""
    lever = _spec("max_num_seqs", whole_step=True)

    assert predict_delta(_trace(MOE), lever) == 0.10        # full coverage
    assert predict_delta(_trace(DENSE), lever) == 0.10


def test_a_whole_step_lever_also_covers_what_the_graph_does_not_model():
    """The step includes kernels the predicted graph has no node for. A lever
    that reshapes the step affects those too, so counting only modeled ops
    understated it — on a dense trace as well as a sparse one."""
    lever = _spec("cuda_graphs", whole_step=True)

    assert predict_delta(_trace(UNMODELED), lever) == 0.10


def test_an_op_scoped_lever_is_unchanged_and_still_misses_what_it_does_not_name():
    """The fix must not turn every lever into a whole-step lever: a KV-cache
    layout knob really does only touch attention."""
    lever = _spec("kv_cache_block_size", kernels=["attn_score_value"])

    assert predict_delta(_trace(MOE), lever) == 0.0
    assert predict_delta(_trace(DENSE), lever) > 0.0


def test_a_blank_scope_still_means_no_coverage_rather_than_all_of_it():
    """The existing rule, and three catalog entries depend on it."""
    lever = _spec("orchestration_only")

    assert predict_delta(_trace(MOE), lever) == 0.0
    assert predict_delta(_trace(DENSE), lever) == 0.0


# --------------------------------------------------------------------------- #
# the catalog itself                                                           #
# --------------------------------------------------------------------------- #
def test_the_engine_and_scheduler_levers_are_marked_whole_step():
    """These act on the step, not on ops. If one is ever re-written as a list of
    op names it goes invisible on every architecture that lacks them, which is
    the defect this flag exists to prevent."""
    by_knob = {s.knob: s for s in load_library(workload="vllm-decode")}
    for knob in ("max_num_seqs", "max_num_batched_tokens", "gpu_memory_utilization",
                 "enforce_eager", "tensor_parallel_size", "pipeline_parallel_size",
                 "enable_chunked_prefill", "async_scheduling", "scheduling_policy",
                 "num_speculative_tokens", "max_seq_len_to_capture", "enable_dbo"):
        assert by_knob[knob].whole_step is True, knob


def test_the_catalog_ranks_a_sparse_moe_trace_at_all():
    """Every lever scored zero here, so the sort fell through to its
    alphabetical tie-break: the loop was choosing by name on the models it was
    actually pointed at."""
    lib = load_library(workload="vllm-decode")
    scored = [predict_delta(_trace(MOE), s) for s in lib]

    assert sum(1 for d in scored if d > 0) >= 10
    assert max(scored) > 0


def test_a_measured_result_can_reach_the_ranking_on_a_sparse_moe_trace():
    """coverage x measured was 0 x anything. A lever with a real number for this
    box and model could not outrank one that had never been tried."""
    lever = _spec("tp4", whole_step=True)

    measured = predict_delta(_trace(MOE), lever, delta_mean=0.21)

    assert measured > predict_delta(_trace(MOE), lever)   # beats its own prior
    assert abs(measured - 0.21) < 1e-9
