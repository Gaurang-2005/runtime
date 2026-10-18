"""The proxy: the only part of gitm on a request's path.

Clients talk to this exactly as they would to ``vllm serve``: it forwards the
OpenAI-compatible endpoints to whichever engine is stable right now, streaming
included. Around each request it does three things and nothing else:

* **routes** — one upstream pointer, swapped atomically. Requests already in
  flight finish on the engine they started on; a swap reports when the old
  engine has drained;
* **measures** — time to first token, latency and token counts per request, kept
  in a bounded ring for the controller and dashboard;
* **records** — after the response is complete, off the hot path, the request's
  shape goes to the :class:`~gitm.server.recorder.Recorder`. Token counts come
  from the engine's own ``usage``; prefix identity from the engine's
  ``/tokenize`` of the same prompt, so the hashes are of the tokens the engine
  actually saw.

To get ``usage`` on streamed responses it asks for ``stream_options.include_usage``.
When the client did not ask for that itself, the usage-only chunk is removed
before it reaches the client, so the client sees what it would have seen from
vLLM directly.

It runs as its own process (``python -m gitm.server.proxy``), in its own session,
so the controller dying never takes serving down with it. The controller talks
to it over a unix socket (the admin app), never over a port.
"""


import argparse
import asyncio
import json
import math
import threading
import time
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from gitm.server.recorder import BLOCK_TOKENS, BlockIds, RecordedRequest, Recorder

#: Endpoints forwarded with request recording. Everything else under /v1 is
#: forwarded as-is and not recorded.
RECORDED = ("/v1/chat/completions", "/v1/completions")

#: Hop-by-hop and length headers the proxy must not copy across.
_DROP_HEADERS = {"host", "content-length", "connection", "keep-alive", "transfer-encoding",
                 "te", "trailer", "upgrade", "proxy-authorization", "proxy-authenticate"}

STATS_RING = 20_000


@dataclass
class RequestStat:
    t_end: float
    upstream: str
    endpoint: str
    ok: bool
    status: int
    ttft_s: float | None
    latency_s: float
    input_tokens: int | None
    output_tokens: int | None


class ProxyState:
    """Routing, in-flight accounting and the stats ring. Shared by both apps."""

    def __init__(self, upstream: str, recorder: Recorder | None,
                 block_ids: BlockIds | None = None):
        self.upstream = upstream.rstrip("/")
        self.recorder = recorder
        self.block_ids = block_ids or BlockIds()
        self.inflight: dict[str, int] = {}
        self.stats: deque[RequestStat] = deque(maxlen=STATS_RING)
        self.started = time.time()
        self.swaps: list[dict[str, Any]] = []
        self.record_failures = 0
        self._private = 0
        self._lock = threading.Lock()

    def acquire(self) -> str:
        with self._lock:
            up = self.upstream
            self.inflight[up] = self.inflight.get(up, 0) + 1
            return up

    def release(self, up: str) -> None:
        with self._lock:
            self.inflight[up] = max(0, self.inflight.get(up, 0) - 1)

    def inflight_on(self, up: str) -> int:
        with self._lock:
            return self.inflight.get(up, 0)

    def private_ids(self, n_blocks: int) -> tuple[int, ...]:
        """Ids nobody else holds: for a prompt whose tokens could not be read.
        Negative, so they can never collide with a hashed block's id."""
        with self._lock:
            out = tuple(-(self._private + i + 1) for i in range(n_blocks))
            self._private += n_blocks
            return out

    def summary(self) -> dict[str, Any]:
        with self._lock:
            inflight = dict(self.inflight)
        return {
            "upstream": self.upstream,
            "inflight": inflight,
            "uptime_s": round(time.time() - self.started, 1),
            "requests_seen": len(self.stats),
            "recorded": self.recorder.total_rows if self.recorder else None,
            "record_failures": self.record_failures,
            "swaps": self.swaps[-10:],
        }


def _forward_headers(headers) -> dict[str, str]:
    return {k: v for k, v in headers.items() if k.lower() not in _DROP_HEADERS}


def build_apps(state: ProxyState, client=None):
    """The public app (what clients call) and the admin app (what the controller calls)."""
    import httpx
    from fastapi import FastAPI, Request
    from fastapi.responses import JSONResponse, Response, StreamingResponse

    http = client or httpx.AsyncClient(timeout=httpx.Timeout(None, connect=10.0))
    public = FastAPI(title="gitm proxy", docs_url=None, redoc_url=None, openapi_url=None)
    admin = FastAPI(title="gitm proxy admin", docs_url=None, redoc_url=None, openapi_url=None)

    async def record(up: str, endpoint: str, payload: dict, arrival: float,
                     input_tokens: int, output_tokens: int) -> None:
        if state.recorder is None:
            return
        try:
            tokens = await _prompt_tokens(http, up, endpoint, payload)
        except Exception:
            tokens = None
        n_blocks = max(1, math.ceil(input_tokens / BLOCK_TOKENS))
        if tokens:
            ids = state.block_ids.for_tokens(tokens)
        else:
            state.record_failures += 1
            ids = ()
        if len(ids) < n_blocks:
            # The replay expands each id to a full block and refuses a file whose
            # ids cannot cover the input, so a short chain is completed with ids
            # nobody shares: under-counting sharing, never inventing it.
            ids = ids + state.private_ids(n_blocks - len(ids))
        state.recorder.write(RecordedRequest(arrival, input_tokens, output_tokens, ids))

    def finish(up, endpoint, payload, arrival, status, ttft, usage) -> None:
        state.release(up)
        ok = 200 <= status < 300
        inp = usage.get("prompt_tokens") if usage else None
        out = usage.get("completion_tokens") if usage else None
        state.stats.append(RequestStat(time.time(), up, endpoint, ok, status, ttft,
                                       time.time() - arrival, inp, out))
        if ok and inp and out and endpoint in RECORDED:
            asyncio.get_running_loop().create_task(
                record(up, endpoint, payload, arrival, int(inp), int(out)))

    @public.api_route("/v1/{rest:path}", methods=["GET", "POST"])
    async def forward(rest: str, request: Request):
        endpoint = "/v1/" + rest
        raw = await request.body()
        payload: dict = {}
        if request.method == "POST" and raw:
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError:
                return JSONResponse({"error": "request body is not JSON"}, status_code=400)
        stream = bool(payload.get("stream")) and endpoint in RECORDED
        strip_usage = False
        if stream:
            opts = payload.get("stream_options") or {}
            if not opts.get("include_usage"):
                payload["stream_options"] = {**opts, "include_usage": True}
                strip_usage = True

        up = state.acquire()
        arrival = time.time()
        try:
            req = http.build_request(
                request.method, up + endpoint, headers=_forward_headers(request.headers),
                params=request.query_params,
                content=json.dumps(payload).encode() if payload else raw or None)
            resp = await http.send(req, stream=True)
        except Exception as exc:
            finish(up, endpoint, payload, arrival, 502, None, None)
            return JSONResponse({"error": f"upstream unavailable: {exc}"}, status_code=502)

        headers = _forward_headers(resp.headers)
        if not stream:
            try:
                body = await resp.aread()
            finally:
                await resp.aclose()
            usage = None
            if endpoint in RECORDED and resp.status_code == 200:
                try:
                    usage = json.loads(body).get("usage")
                except Exception:
                    usage = None
            finish(up, endpoint, payload, arrival, resp.status_code,
                   time.time() - arrival if usage else None, usage)
            return Response(body, status_code=resp.status_code, headers=headers)

        async def events():
            ttft = None
            usage = None
            event: list[str] = []
            try:
                async for line in resp.aiter_lines():
                    if line:
                        event.append(line)
                        continue
                    if not event:
                        continue
                    keep, first_token, seen_usage = _inspect_event(event, strip_usage)
                    if first_token and ttft is None:
                        ttft = time.time() - arrival
                    if seen_usage is not None:
                        usage = seen_usage
                    if keep:
                        yield ("\n".join(event) + "\n\n").encode()
                    event = []
                if event:
                    yield ("\n".join(event) + "\n\n").encode()
            finally:
                await resp.aclose()
                finish(up, endpoint, payload, arrival, resp.status_code, ttft, usage)

        return StreamingResponse(events(), status_code=resp.status_code, headers=headers)

    @public.get("/")
    async def chat_page():
        """A minimal chat page, so a phone browser can talk to the served model.

        It calls this proxy's own /v1, so its messages are served by stable and
        recorded like any other client's. Static: nothing here touches state.
        """
        from importlib import resources

        from fastapi.responses import HTMLResponse

        return HTMLResponse(resources.files("gitm.server").joinpath("chat.html").read_text(),
                            headers={"Cache-Control": "no-store"})

    @public.get("/health")
    async def health():
        try:
            r = await http.get(state.upstream + "/health", timeout=5.0)
            return Response(status_code=r.status_code)
        except Exception:
            return Response(status_code=503)

    @admin.get("/admin/state")
    async def admin_state():
        return state.summary()

    @admin.get("/admin/stats")
    async def admin_stats(since: float = 0.0):
        return [asdict(s) for s in state.stats if s.t_end > since]

    @admin.post("/admin/upstream")
    async def admin_upstream(body: dict):
        new = str(body["url"]).rstrip("/")
        timeout = float(body.get("drain_timeout_s", 120.0))
        with state._lock:
            old = state.upstream
            state.upstream = new
        deadline = time.time() + timeout
        while state.inflight_on(old) and time.time() < deadline:
            await asyncio.sleep(0.05)
        drained = state.inflight_on(old) == 0
        state.swaps.append({"t": time.time(), "from": old, "to": new, "drained": drained})
        return {"from": old, "to": new, "drained": drained,
                "still_inflight": state.inflight_on(old)}

    @admin.post("/admin/rotate")
    async def admin_rotate():
        path = state.recorder.rotate() if state.recorder else None
        return {"closed": str(path) if path else None}

    @admin.get("/admin/recordings")
    async def admin_recordings():
        if state.recorder is None:
            return []
        return [str(p) for p in state.recorder.windows()]

    return public, admin, http


def _inspect_event(lines: list[str], strip_usage: bool) -> tuple[bool, bool, dict | None]:
    """(forward this SSE event?, does it carry a token?, its usage if any)."""
    data = "".join(ln[5:].strip() for ln in lines if ln.startswith("data:"))
    if not data or data == "[DONE]":
        return True, False, None
    try:
        chunk = json.loads(data)
    except json.JSONDecodeError:
        return True, False, None
    choices = chunk.get("choices") or []
    usage = chunk.get("usage")
    token = any(
        (c.get("delta") or {}).get("content") or (c.get("delta") or {}).get("reasoning_content")
        or c.get("text")
        for c in choices)
    if usage and not choices and strip_usage:
        return False, False, usage
    return True, token, usage


async def _prompt_tokens(http, up: str, endpoint: str, payload: dict) -> list[int] | None:
    """The token ids the engine built from this prompt, from its own ``/tokenize``."""
    model = payload.get("model")
    if endpoint == "/v1/chat/completions":
        body = {"model": model, "messages": payload.get("messages", []),
                "add_generation_prompt": True}
    else:
        prompt = payload.get("prompt")
        if isinstance(prompt, list) and prompt and all(isinstance(t, int) for t in prompt):
            return list(prompt)  # already token ids
        if not isinstance(prompt, str):
            return None
        body = {"model": model, "prompt": prompt}
    r = await http.post(up + "/tokenize", json=body, timeout=30.0)
    if r.status_code != 200:
        return None
    tokens = r.json().get("tokens")
    return tokens if isinstance(tokens, list) else None


async def serve(*, host: str, port: int, upstream: str, admin_socket: Path,
                record_dir: Path | None) -> None:
    import uvicorn

    recorder = Recorder(record_dir) if record_dir else None
    state = ProxyState(upstream, recorder)
    public, admin, http = build_apps(state)
    admin_socket.parent.mkdir(parents=True, exist_ok=True)
    if admin_socket.exists():
        admin_socket.unlink()
    servers = [
        uvicorn.Server(uvicorn.Config(public, host=host, port=port, log_level="warning",
                                      access_log=False)),
        uvicorn.Server(uvicorn.Config(admin, uds=str(admin_socket), log_level="warning",
                                      access_log=False)),
    ]
    try:
        await asyncio.gather(*(s.serve() for s in servers))
    finally:
        await http.aclose()
        if recorder:
            recorder.close()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m gitm.server.proxy")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--upstream", required=True, help="Stable engine base URL.")
    ap.add_argument("--admin-socket", type=Path, required=True)
    ap.add_argument("--record-dir", type=Path, default=None)
    a = ap.parse_args(argv)
    asyncio.run(serve(host=a.host, port=a.port, upstream=a.upstream,
                      admin_socket=a.admin_socket, record_dir=a.record_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
