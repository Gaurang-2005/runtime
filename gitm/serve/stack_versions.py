"""Host driver / CUDA versions for the run record (re-baseline visibility).

Written next to ``preflight.json`` as ``stack_versions.json`` and copied into
``run_manifest.json``. Best-effort: missing tools become null, never raise.

Only ``nvidia-smi`` fields are filled here. ``torch_cuda_version`` / ``nccl_version``
stay null: reading ``import torch`` would sample the capture process, not the
server, and mis-label observer versions as the run's stack.
"""

from __future__ import annotations

import re
import subprocess
from typing import Any


def _run(cmd: list[str], timeout: float = 15) -> str:
    try:
        return subprocess.check_output(
            cmd, text=True, stderr=subprocess.DEVNULL, timeout=timeout
        )
    except Exception:
        return ""


def collect_stack_versions() -> dict[str, Any]:
    """Snapshot host driver/CUDA from nvidia-smi (torch/NCCL left unknown)."""
    versions: dict[str, Any] = {
        "driver_version": None,
        "cuda_version": None,  # driver-reported CUDA (nvidia-smi banner)
        "torch_cuda_version": None,
        "nccl_version": None,
        "source": "host_nvidia_smi",
    }

    driver = _run(
        ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"]
    ).strip().splitlines()
    if driver:
        versions["driver_version"] = driver[0].strip() or None

    smi = _run(["nvidia-smi"])
    m = re.search(r"CUDA Version:\s*([\d.]+)", smi)
    if m:
        versions["cuda_version"] = m.group(1)

    return versions
