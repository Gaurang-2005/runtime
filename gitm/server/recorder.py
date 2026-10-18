"""Record served traffic as a trace the rest of gitm already reads.

Each request the proxy completes becomes one Mooncake-shaped JSONL row:
``timestamp`` (ms), ``input_length``, ``output_length`` and ``hash_ids``. That is
the format :func:`gitm.traffic.adapters.read_mooncake` reads, so a recorded
window feeds :func:`gitm.traffic.replay.write_timed_trace`, the regime fit and
validation unchanged — no second parser to drift from the first.

**No prompt text is stored.** Prefix identity is kept the way Mooncake keeps it:
the prompt's token ids are cut into fixed blocks and each block is named by a
hash *chained* over every block before it, so two requests share a block id only
when they share the whole prefix up to and including it. The hash is turned into
a small integer id that stays stable for the life of the recorder, so sharing
across windows is preserved too.
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

#: Tokens per ``hash_ids`` entry. The published Mooncake traces use 512, and the
#: replay path expands each id to exactly this many tokens.
BLOCK_TOKENS = 512

#: Distinct prefix blocks remembered before the oldest ids are forgotten. A
#: forgotten block that comes back gets a fresh id, which under-counts sharing
#: for that block rather than inventing any.
MAX_BLOCK_IDS = 1_000_000


@dataclass(frozen=True)
class RecordedRequest:
    """One completed request, as the recorder writes it."""

    arrival_s: float  # wall clock, seconds
    input_tokens: int
    output_tokens: int
    hash_ids: tuple[int, ...]

    def row(self) -> dict:
        return {
            "timestamp": int(round(self.arrival_s * 1000)),
            "input_length": self.input_tokens,
            "output_length": self.output_tokens,
            "hash_ids": list(self.hash_ids),
        }


class BlockIds:
    """Chained block hashes -> small, stable integer ids."""

    def __init__(self, block_tokens: int = BLOCK_TOKENS, max_ids: int = MAX_BLOCK_IDS):
        self.block_tokens = block_tokens
        self._max = max_ids
        self._ids: OrderedDict[bytes, int] = OrderedDict()
        self._next = 0
        self._lock = threading.Lock()

    def for_tokens(self, token_ids: list[int]) -> tuple[int, ...]:
        """The block-id chain for this prompt. A final partial block counts."""
        out: list[int] = []
        prev = b""
        for start in range(0, len(token_ids), self.block_tokens):
            block = token_ids[start:start + self.block_tokens]
            h = hashlib.blake2b(prev, digest_size=16)
            h.update(json.dumps(block, separators=(",", ":")).encode())
            prev = h.digest()
            out.append(self._id(prev))
        return tuple(out)

    def _id(self, digest: bytes) -> int:
        with self._lock:
            if digest in self._ids:
                self._ids.move_to_end(digest)
                return self._ids[digest]
            ident = self._next
            self._next += 1
            self._ids[digest] = ident
            if len(self._ids) > self._max:
                self._ids.popitem(last=False)
            return ident


class Recorder:
    """Append rows to rotating window files under ``root``.

    A window closes after ``window_requests`` rows or ``window_s`` seconds,
    whichever comes first, so a replay always has a bounded, complete file to
    read. Files are named by their opening time and never rewritten.
    """

    def __init__(self, root: str | Path, *, window_requests: int = 2000,
                 window_s: float = 600.0, clock=time.time):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.window_requests = window_requests
        self.window_s = window_s
        self._clock = clock
        self._lock = threading.Lock()
        self._fh = None
        self._path: Path | None = None
        self._opened_at = 0.0
        self._rows = 0
        self.total_rows = 0

    def write(self, rec: RecordedRequest) -> None:
        with self._lock:
            now = self._clock()
            if self._fh is not None and (self._rows >= self.window_requests
                                         or now - self._opened_at >= self.window_s):
                self._close()
            if self._fh is None:
                self._open(now)
            self._fh.write(json.dumps(rec.row()) + "\n")
            self._fh.flush()
            self._rows += 1
            self.total_rows += 1

    def rotate(self) -> Path | None:
        """Close the current window now. Returns the file closed, if any."""
        with self._lock:
            path = self._path if self._fh is not None else None
            self._close()
            return path

    def windows(self) -> list[Path]:
        """Closed windows, oldest first. The open one is excluded: it is still growing."""
        with self._lock:
            open_path = self._path if self._fh is not None else None
        return sorted(p for p in self.root.glob("window-*.jsonl") if p != open_path)

    def close(self) -> None:
        with self._lock:
            self._close()

    def _open(self, now: float) -> None:
        stamp = time.strftime("%Y%m%d-%H%M%S", time.gmtime(now))
        path = self.root / f"window-{stamp}-{int(now * 1000) % 1000:03d}.jsonl"
        n = 1
        while path.exists():
            path = self.root / f"window-{stamp}-{int(now * 1000) % 1000:03d}-{n}.jsonl"
            n += 1
        self._path = path
        self._fh = path.open("a", encoding="utf-8")
        self._opened_at = now
        self._rows = 0

    def _close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            if self._path is not None:
                _sort_by_arrival(self._path)
        self._fh = None

    @property
    def open_rows(self) -> int:
        with self._lock:
            return self._rows if self._fh is not None else 0

    @property
    def open_age_s(self) -> float:
        with self._lock:
            return self._clock() - self._opened_at if self._fh is not None else 0.0


def _sort_by_arrival(path: Path) -> None:
    """Rewrite a closed window in arrival order.

    A row is written when its request *finishes*, but stamped with when it
    *arrived*, so a long request that arrived first is written after shorter
    ones that arrived later. The trace reader drops rows that go back in time
    (``non_monotonic_arrival``): on the laptop that silently lost 21 of 50 chat
    rows, and 240 of 405 in a chat + long-document window, mostly the long
    requests, so every replay and every regime was skewed toward short ones.
    """
    lines = [ln for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]

    def ts(line: str) -> float:
        try:
            return float(json.loads(line).get("timestamp", 0))
        except (ValueError, AttributeError):
            return 0.0

    lines.sort(key=ts)  # stable: rows with equal stamps keep their order
    tmp = path.with_suffix(".sorting")
    tmp.write_text("".join(ln + "\n" for ln in lines), encoding="utf-8")
    tmp.replace(path)
