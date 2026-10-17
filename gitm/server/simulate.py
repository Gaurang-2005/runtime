"""Synthetic traffic, sent through the proxy like any client's.

The dashboard starts these to put a server under a known kind of load. They go
through the public proxy port, so they are served by the stable engine and
recorded exactly as production traffic is — which is the point: a simulated
chat workload becomes a recorded window the canary can be tested against.

Each preset is a traffic *shape*, not a benchmark score: how long prompts are,
how much of them is shared, how long the answers are, and how arrivals bunch.

* ``chat`` — a shared system prompt, short varied questions, medium answers;
* ``agent`` — a long shared tool/system preamble, short answers (prefix-heavy);
* ``long_doc`` — a different long document per request, summary-length answers
  (prefill-heavy, nothing shared);
* ``bursty_chat`` — ``chat`` with arrivals bunched into bursts.

Arrivals are open-loop: requests are sent on schedule whether or not earlier
ones have finished, as real users do. A closed loop would slow its own arrivals
exactly when the server is struggling and hide the problem.
"""

from __future__ import annotations

import asyncio
import random
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

_WORDS = ("the model serves requests in batches and each step reads the weights once while "
          "attention reads every cached key and value for the tokens seen so far so long "
          "prompts cost prefill time and long answers cost decode steps across the cluster").split()

PRESETS: dict[str, dict[str, Any]] = {
    "chat": {"system_words": 300, "user_words": (15, 150), "max_tokens": (64, 512),
             "shared_system": True, "burstiness": 1.0, "rate_rps": 2.0},
    "agent": {"system_words": 1500, "user_words": (20, 80), "max_tokens": (32, 192),
              "shared_system": True, "burstiness": 1.0, "rate_rps": 1.0},
    "long_doc": {"system_words": 0, "user_words": (1500, 4000), "max_tokens": (128, 256),
                 "shared_system": False, "burstiness": 1.0, "rate_rps": 0.3},
    "bursty_chat": {"system_words": 300, "user_words": (15, 150), "max_tokens": (64, 512),
                    "shared_system": True, "burstiness": 8.0, "rate_rps": 2.0},
}


def _text(rng: random.Random, n: int) -> str:
    return " ".join(rng.choice(_WORDS) for _ in range(n))


def arrivals(rng: random.Random, rate: float, duration_s: float, burstiness: float) -> list[float]:
    """Arrival offsets in seconds. burstiness 1 = Poisson; above 1, gamma gaps
    with the same mean and variance x burstiness, so arrivals bunch."""
    out, t = [], 0.0
    shape = 1.0 / max(burstiness, 1e-6)
    while True:
        gap = rng.gammavariate(shape, 1.0 / (rate * shape)) if burstiness != 1.0 \
            else rng.expovariate(rate)
        t += gap
        if t >= duration_s:
            return out
        out.append(t)


@dataclass
class SimRun:
    id: str
    preset: str
    rate_rps: float
    duration_s: float
    burstiness: float
    started: float
    planned: int = 0
    sent: int = 0
    ok: int = 0
    failed: int = 0
    status: str = "running"
    errors: list[str] = field(default_factory=list)

    def brief(self) -> dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if k != "errors"} | {
            "last_errors": self.errors[-3:]}


class Simulator:
    """Runs simulated traffic in a background event loop, one run at a time or several."""

    def __init__(self, base_url: str, model: str):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.runs: dict[str, SimRun] = {}
        self._stop: dict[str, threading.Event] = {}

    def start(self, preset: str, *, rate_rps: float | None = None, duration_s: float = 120.0,
              burstiness: float | None = None, seed: int | None = None) -> SimRun:
        if preset not in PRESETS:
            raise ValueError(f"unknown preset {preset!r}; known: {sorted(PRESETS)}")
        p = PRESETS[preset]
        rate = float(rate_rps if rate_rps is not None else p["rate_rps"])
        burst = float(burstiness if burstiness is not None else p["burstiness"])
        if rate <= 0 or duration_s <= 0 or duration_s > 24 * 3600:
            raise ValueError("rate_rps must be > 0 and duration_s within (0, 86400]")
        run = SimRun(id=uuid.uuid4().hex[:8], preset=preset, rate_rps=rate,
                     duration_s=duration_s, burstiness=burst, started=time.time())
        self.runs[run.id] = run
        self._stop[run.id] = threading.Event()
        threading.Thread(target=lambda: asyncio.run(self._run(run, p, seed)),
                         name=f"gitm-sim-{run.id}", daemon=True).start()
        return run

    def stop(self, ident: str) -> SimRun:
        self._stop[ident].set()
        return self.runs[ident]

    async def _run(self, run: SimRun, p: dict[str, Any], seed: int | None) -> None:
        import httpx

        rng = random.Random(seed)
        schedule = arrivals(rng, run.rate_rps, run.duration_s, run.burstiness)
        run.planned = len(schedule)
        system = _text(rng, p["system_words"]) if p["shared_system"] else ""
        stop = self._stop[run.id]
        t0 = time.monotonic()
        async with httpx.AsyncClient(timeout=httpx.Timeout(600.0, connect=10.0)) as http:
            tasks = []
            for at in schedule:
                if stop.is_set():
                    break
                while (delay := t0 + at - time.monotonic()) > 0 and not stop.is_set():
                    await asyncio.sleep(min(delay, 0.25))
                if stop.is_set():
                    break
                tasks.append(asyncio.create_task(self._one(http, run, rng, p, system)))
                run.sent += 1
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
        run.status = "stopped" if stop.is_set() else "done"

    async def _one(self, http, run: SimRun, rng: random.Random, p: dict[str, Any],
                   system: str) -> None:
        user = _text(rng, rng.randint(*p["user_words"]))
        messages = ([{"role": "system", "content": system}] if system else []) + [
            {"role": "user", "content": user}]
        body = {"model": self.model, "messages": messages, "stream": True,
                "max_tokens": rng.randint(*p["max_tokens"]), "temperature": 0.7}
        try:
            async with http.stream("POST", self.base_url + "/v1/chat/completions",
                                   json=body) as r:
                async for _ in r.aiter_bytes():
                    pass
                if r.status_code == 200:
                    run.ok += 1
                else:
                    run.failed += 1
                    run.errors.append(f"HTTP {r.status_code}")
        except Exception as exc:  # noqa: BLE001 - a failed request is a count, not a crash
            run.failed += 1
            run.errors.append(f"{type(exc).__name__}: {exc}"[:200])
