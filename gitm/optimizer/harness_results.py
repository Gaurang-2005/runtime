"""Turn a pair of harness captures into a verification record the loop can read.

    a = read_capture("results/tp2-baseline")
    b = read_capture("results/tp2-ep")
    write_comparison(a, b, out_dir=runs_dir() / run_id)

`runtime-experiment-harness` runs experiments across a cluster and hands the
server to ``gitm capture serve``, which leaves ``serving_summary.json`` and
``run_manifest.json`` in each arm's directory. Nothing read them back.

:mod:`gitm.optimizer.history` aggregates ``runs/<run_id>/verification.json``, and
**only ``run_loop`` writes one** — so every result measured on the cluster was
invisible to the ranking that reads history. A loop that proposes experiments and
then cannot see their results is the failure the history reader exists to
prevent, one layer out.

This converts rather than teaching the reader a second format. ``history.py`` is
the most-tested module here and has been through several rounds of review
findings; giving it another input shape would put that at risk to save a file
write. A converted capture lands beside the loop's own exports and is read by the
same code, so a cluster result and a local one are the same kind of evidence.

**Pairing is explicit, never inferred.** Which arm is the baseline is not
recoverable from two directories — the one with fewer flags is a guess, and a
wrong guess silently inverts the sign of every delta it produces. The caller
says, or nothing is written.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from gitm.optimizer.history import EXPORT_NAME
from gitm.optimizer.report import Provenance
from gitm.optimizer.verification_export import VerificationRecord, write_verification

__all__ = [
    "Capture",
    "CaptureError",
    "read_capture",
    "knob_difference",
    "compare",
    "write_comparison",
]

SUMMARY_NAME = "serving_summary.json"
MANIFEST_NAME = "run_manifest.json"


class CaptureError(ValueError):
    """A capture directory that cannot be read as one, with the reason."""


@dataclass(frozen=True)
class Capture:
    """One arm of a harness experiment, as it lands on disk."""

    path: Path
    served_model: str | None
    #: The server argv the harness launched. The knob under test is the
    #: difference between two of these, which is why the whole list is kept
    #: rather than a parsed subset.
    serve_argv: tuple[str, ...]
    #: Load shape: requests, concurrency, input/output tokens, seed. Two arms
    #: measured under different load are not an A/B, and this is what says so.
    load: dict[str, Any]
    #: ``off`` | ``cupti`` | ``cupti+nvtx``. Tracing costs throughput, so an arm
    #: traced against one that was not measures the tracer, not the knob.
    tracing: str | None
    throughput: float | None
    window_s: float | None

    @property
    def comparable_key(self) -> tuple:
        """What must match for two arms to be measuring the same thing."""
        return (self.served_model, self.tracing,
                tuple(sorted((k, str(v)) for k, v in self.load.items())))


def _read_json(path: Path) -> dict[str, Any]:
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise CaptureError(f"{path.name}: unreadable: {exc}") from exc
    except ValueError as exc:
        raise CaptureError(f"{path.name}: not valid JSON: {exc}") from exc
    if not isinstance(doc, dict):
        raise CaptureError(f"{path.name}: not a JSON object")
    return doc


def _throughput(summary: dict[str, Any]) -> tuple[float | None, float | None]:
    """Requests per second over the window, and the window.

    ``goodput_rps`` where the capture reported one — it counts only requests that
    met their SLO, which is the number a serving change should be judged on. A
    run that met none has a real goodput of 0.0 and is not missing data, so the
    fallback is only used when the field is absent entirely.
    """
    client = summary.get("client")
    client = client if isinstance(client, dict) else {}
    window = client.get("window_s") or summary.get("wall_s")
    window = float(window) if isinstance(window, int | float) else None

    goodput = client.get("goodput_rps")
    if isinstance(goodput, int | float):
        return float(goodput), window

    n = client.get("n_requests")
    if isinstance(n, int | float) and window:
        return float(n) / window, window
    return None, window


def read_capture(path: str | Path) -> Capture:
    """One arm's directory, read into a :class:`Capture`.

    Raises rather than returning a half-built record: a comparison assembled from
    a capture whose throughput is missing would report a delta against nothing.
    """
    path = Path(path)
    if not path.is_dir():
        raise CaptureError(f"{path}: not a directory")
    summary = _read_json(path / SUMMARY_NAME)
    manifest = _read_json(path / MANIFEST_NAME)

    throughput, window = _throughput(summary)
    if throughput is None:
        raise CaptureError(f"{path.name}: no throughput in {SUMMARY_NAME}")

    argv = manifest.get("serve_argv")
    load = manifest.get("load")
    return Capture(
        path=path,
        served_model=manifest.get("served_model"),
        serve_argv=tuple(str(a) for a in argv) if isinstance(argv, list) else (),
        load=load if isinstance(load, dict) else {},
        tracing=summary.get("tracing"),
        throughput=throughput,
        window_s=window,
    )


def knob_difference(baseline: Capture, candidate: Capture) -> dict[str, Any]:
    """The server flags the candidate has and the baseline does not.

    Returned as ``{flag: value}`` with ``True`` for a bare switch. Only the
    candidate's side: a flag the *baseline* carries and the candidate drops is a
    different experiment — removing a knob — and reporting it under the same
    shape would claim the candidate set something it unset.
    """
    def flags(argv: tuple[str, ...]) -> dict[str, Any]:
        out: dict[str, Any] = {}
        i = 0
        while i < len(argv):
            token = argv[i]
            if not token.startswith("--"):
                i += 1
                continue
            nxt = argv[i + 1] if i + 1 < len(argv) else None
            if nxt is not None and not nxt.startswith("--"):
                out[token], i = nxt, i + 2
            else:
                out[token], i = True, i + 1
        return out

    base, cand = flags(baseline.serve_argv), flags(candidate.serve_argv)
    return {k: v for k, v in cand.items() if base.get(k) != v}


def compare(
    baseline: Capture, candidate: Capture, *, name: str | None = None,
    agreement_band: float = 0.02,
) -> VerificationRecord:
    """One baseline↔candidate comparison, in the loop's own record shape.

    Refuses arms that are not measuring the same thing. A different model, a
    different load shape or a different tracing arm each make the number a
    comparison of something other than the knob — tracing especially, since it
    costs throughput, so a traced candidate against an untraced baseline reports
    the tracer's overhead as the lever's effect.

    ``significant`` is the gain clearing ``agreement_band``, not a statistical
    test: a harness arm is one measurement, not ``reps`` of them, so there is no
    scatter to compute and claiming a std of 0 would read as perfect precision
    rather than as one sample. ``reps=1`` says which it is.

    ``kept`` is the gate's own rule applied to this measurement: cleared the band
    and faster. Leaving it False because no rollback gate ran would be the
    tidier-sounding choice and the wrong one — the reader maps ``not kept`` to
    *loss*, so every cluster result, including a +49% win, would demote the lever
    it proves. There is nothing to roll back either way: a harness arm runs
    standalone, so "would the gate have kept this" is the whole question, and it
    is answerable from the number. ``via="harness"`` records which path decided.
    """
    if baseline.comparable_key != candidate.comparable_key:
        raise CaptureError(
            "these arms are not an A/B: "
            f"baseline {baseline.comparable_key} vs candidate {candidate.comparable_key}")
    if not baseline.throughput:
        raise CaptureError(f"{baseline.path.name}: baseline throughput is zero")

    knobs = knob_difference(baseline, candidate)
    if not knobs:
        raise CaptureError(
            f"{baseline.path.name} and {candidate.path.name} ran the same server "
            "flags: there is no intervention between them")

    knob, value = next(iter(knobs.items()))
    speedup = candidate.throughput / baseline.throughput
    delta = speedup - 1.0
    return VerificationRecord(
        intervention_name=name or _name_for(knobs),
        summary=f"harness capture: {candidate.path.name} vs {baseline.path.name}",
        knob=knob,
        value=value,
        source=str(candidate.path),
        baseline_tps=baseline.throughput,
        candidate_tps=candidate.throughput,
        speedup=speedup,
        delta=delta,
        baseline_std=0.0,
        candidate_std=0.0,
        reps=1,
        agreement_band=agreement_band,
        significant=abs(delta) > agreement_band,
        kept=delta > 0 and abs(delta) > agreement_band,
        via="harness",
        baseline_config={"serve_argv": list(baseline.serve_argv)},
        candidate_config={"serve_argv": list(candidate.serve_argv)},
    )


def _name_for(knobs: dict[str, Any]) -> str:
    """A lever name from the flags that moved.

    Deterministic, so the same experiment read twice keys to the same record
    rather than accumulating as two levers that were each tried once.
    """
    parts = []
    for flag, value in sorted(knobs.items()):
        stem = flag.lstrip("-").replace("-", "_")
        parts.append(stem if value is True else f"{stem}_{value}")
    return "+".join(parts)


def write_comparison(
    baseline: Capture, candidate: Capture, *, out_dir: str | Path,
    gpu_sku: str | None = None, fingerprint: str | None = None,
    run_id: str | None = None, name: str | None = None,
) -> str:
    """Write the comparison as a ``verification.json`` under ``out_dir``.

    ``fingerprint`` identifies the workload the record belongs to, and the reader
    keys on it: without one the result lands under ``None`` and merges with every
    other unfingerprinted run, which is the mistake that key exists to prevent.
    The served model is used when nothing better is given — coarser than a real
    trace fingerprint, and honest about which arms belong together.
    """
    record = compare(baseline, candidate, name=name)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    prov = Provenance(
        workload_id="vllm-serve",
        fingerprint=fingerprint or candidate.served_model or "unknown",
        run_id=run_id or out_dir.name,
        git_sha="", gitm_version="", started_at_ns=0, ended_at_ns=0,
    )
    return write_verification([record], prov, out_dir / EXPORT_NAME, gpu_sku=gpu_sku)
