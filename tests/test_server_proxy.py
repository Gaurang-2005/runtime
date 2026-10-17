"""gitm server, milestone 1: the proxy, the recorder and the control API.

Everything runs against a fake ``vllm serve`` (an ASGI app answering the same
endpoints), so no GPU or vLLM is needed.
"""

from __future__ import annotations

import asyncio
import functools
import json

import pytest

fastapi = pytest.importorskip("fastapi")
httpx = pytest.importorskip("httpx")

from fastapi import FastAPI, Request  # noqa: E402
from fastapi.responses import JSONResponse, StreamingResponse  # noqa: E402

from gitm.server.api import summarize_stats  # noqa: E402
from gitm.server.proxy import ProxyState, build_apps  # noqa: E402
from gitm.server.recorder import BLOCK_TOKENS, BlockIds, Recorder  # noqa: E402
from gitm.traffic.adapters import read_mooncake  # noqa: E402
from gitm.traffic.replay import write_timed_trace  # noqa: E402


def run_async(fn):
    """Run an async test on a fresh loop (no pytest-asyncio in CI)."""
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        return asyncio.run(fn(*args, **kwargs))
    return wrapper


def _text(messages) -> str:
    return "".join(m.get("content", "") for m in messages)


def fake_vllm(name: str, *, tokenize_ok: bool = True, gate: asyncio.Event | None = None):
    """A stand-in for vllm serve: one token per prompt character, 3 output tokens."""
    app = FastAPI()
    app.state.hits = 0

    @app.post("/v1/chat/completions")
    async def chat(request: Request):
        app.state.hits += 1
        body = await request.json()
        if gate is not None:
            await gate.wait()
        n_in = len(_text(body["messages"]))
        usage = {"prompt_tokens": n_in, "completion_tokens": 3, "total_tokens": n_in + 3}
        if not body.get("stream"):
            return JSONResponse({"id": name, "choices": [{"message": {"content": "abc"}}],
                                 "usage": usage})

        async def gen():
            for tok in "abc":
                yield "data: " + json.dumps({"choices": [{"delta": {"content": tok}}]}) + "\n\n"
            if (body.get("stream_options") or {}).get("include_usage"):
                yield "data: " + json.dumps({"choices": [], "usage": usage}) + "\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(gen(), media_type="text/event-stream")

    @app.post("/tokenize")
    async def tokenize(request: Request):
        body = await request.json()
        if not tokenize_ok:
            return JSONResponse({"error": "no"}, status_code=500)
        return {"tokens": [ord(c) for c in _text(body["messages"])]}

    @app.get("/health")
    async def health():
        return {}

    return app


def make(tmp_path, upstreams: dict, record=True):
    mounts = {f"http://{k}": httpx.ASGITransport(app=v) for k, v in upstreams.items()}
    upstream_client = httpx.AsyncClient(mounts=mounts)
    first = next(iter(upstreams))
    state = ProxyState(f"http://{first}", Recorder(tmp_path / "rec") if record else None)
    public, admin, _ = build_apps(state, client=upstream_client)
    client = httpx.AsyncClient(transport=httpx.ASGITransport(app=public), base_url="http://proxy")
    admin_client = httpx.AsyncClient(transport=httpx.ASGITransport(app=admin),
                                     base_url="http://admin")
    return state, client, admin_client


async def settle(state, n):
    for _ in range(100):
        if state.recorder.total_rows >= n:
            return
        await asyncio.sleep(0.01)
    raise AssertionError(f"only {state.recorder.total_rows} of {n} requests recorded")


def chat(content, **extra):
    return {"model": "m", "messages": [{"role": "user", "content": content}], **extra}


@run_async
async def test_a_plain_request_passes_through_and_is_recorded(tmp_path):
    state, client, _ = make(tmp_path, {"a": fake_vllm("a")})
    r = await client.post("/v1/chat/completions", json=chat("hello"))
    assert r.status_code == 200 and r.json()["id"] == "a"
    await settle(state, 1)
    state.recorder.close()
    trace = read_mooncake(next((tmp_path / "rec").glob("window-*.jsonl")))
    assert [(q.input_tokens, q.output_tokens) for q in trace.requests] == [(5, 3)]
    assert state.stats[-1].ok and state.stats[-1].input_tokens == 5


@run_async
async def test_a_stream_reaches_the_client_as_vllm_would_send_it(tmp_path):
    """The proxy asks for usage to record token counts; a client that did not ask
    for it must not receive the usage-only chunk."""
    state, client, _ = make(tmp_path, {"a": fake_vllm("a")})
    r = await client.post("/v1/chat/completions", json=chat("hi", stream=True))
    events = [e for e in r.text.split("\n\n") if e]
    assert events[-1] == "data: [DONE]"
    assert not any('"usage"' in e for e in events)
    assert "".join(json.loads(e[6:])["choices"][0]["delta"]["content"]
                   for e in events[:-1]) == "abc"
    await settle(state, 1)
    assert state.stats[-1].ttft_s is not None and state.stats[-1].output_tokens == 3


@run_async
async def test_a_client_that_asked_for_usage_still_gets_it(tmp_path):
    state, client, _ = make(tmp_path, {"a": fake_vllm("a")})
    r = await client.post("/v1/chat/completions",
                          json=chat("hi", stream=True, stream_options={"include_usage": True}))
    assert '"usage"' in r.text


@run_async
async def test_requests_that_shared_a_prefix_share_its_block_ids(tmp_path):
    state, client, _ = make(tmp_path, {"a": fake_vllm("a")})
    system = "s" * BLOCK_TOKENS
    for tail in ("first question", "another question"):
        await client.post("/v1/chat/completions", json=chat(system + tail))
    await client.post("/v1/chat/completions", json=chat("x" * (BLOCK_TOKENS + 10)))
    await settle(state, 3)
    state.recorder.close()
    rows = [json.loads(ln) for ln in next((tmp_path / "rec").glob("*.jsonl")).read_text()
            .splitlines()]
    a, b, c = (r["hash_ids"] for r in rows)
    assert a[0] == b[0] and a[1] != b[1]      # same system prompt, different questions
    assert c[0] != a[0]                        # a different prompt shares nothing


@run_async
async def test_unreadable_prompt_tokens_record_unshared_blocks_that_still_replay(tmp_path):
    state, client, _ = make(tmp_path, {"a": fake_vllm("a", tokenize_ok=False)})
    await client.post("/v1/chat/completions", json=chat("y" * (2 * BLOCK_TOKENS + 1)))
    await settle(state, 1)
    state.recorder.close()
    path = next((tmp_path / "rec").glob("*.jsonl"))
    row = json.loads(path.read_text())
    assert len(row["hash_ids"]) == 3 and all(i < 0 for i in row["hash_ids"])
    assert state.record_failures == 1
    # the replay path accepts it: the ids cover the whole input
    write_timed_trace(read_mooncake(path), tmp_path / "replay.jsonl")


@run_async
async def test_a_swap_routes_new_requests_and_waits_for_the_old_engine_to_drain(tmp_path):
    gate = asyncio.Event()
    a, b = fake_vllm("a", gate=gate), fake_vllm("b")
    state, client, admin = make(tmp_path, {"a": a, "b": b})

    slow = asyncio.create_task(client.post("/v1/chat/completions", json=chat("slow")))
    for _ in range(100):
        if state.inflight_on("http://a"):
            break
        await asyncio.sleep(0.01)
    swap = asyncio.create_task(admin.post("/admin/upstream",
                                          json={"url": "http://b", "drain_timeout_s": 5}))
    await asyncio.sleep(0.05)
    r = await client.post("/v1/chat/completions", json=chat("new"))
    assert r.json()["id"] == "b"               # new traffic already on b
    assert not swap.done()                     # a still has a request in flight
    gate.set()
    assert (await slow).json()["id"] == "a"    # the in-flight request finished on a
    result = (await swap).json()
    assert result["drained"] and result["to"] == "http://b"


@run_async
async def test_an_unreachable_engine_is_a_502_not_a_hang(tmp_path):
    state, client, _ = make(tmp_path, {"a": fake_vllm("a")})
    state.upstream = "http://nowhere.invalid"
    r = await client.post("/v1/chat/completions", json=chat("hi"))
    assert r.status_code == 502
    assert state.inflight_on("http://nowhere.invalid") == 0


def test_recorder_rotates_windows_by_size(tmp_path):
    from gitm.server.recorder import RecordedRequest

    rec = Recorder(tmp_path, window_requests=2)
    for i in range(5):
        rec.write(RecordedRequest(1000.0 + i, 10, 2, (i,)))
    assert len(rec.windows()) == 2             # the third window is still open
    rec.close()
    assert len(rec.windows()) == 3


def test_block_ids_chain_over_the_whole_prefix():
    ids = BlockIds(block_tokens=2)
    a = ids.for_tokens([1, 2, 3, 4])
    b = ids.for_tokens([9, 9, 3, 4])           # same second block, different first
    assert a[1] != b[1]


def test_live_stats_summary():
    import time

    now = time.time()
    stats = [{"t_end": now, "ok": True, "output_tokens": 30, "ttft_s": 0.1, "latency_s": 1.0},
             {"t_end": now, "ok": False, "output_tokens": None, "ttft_s": None, "latency_s": 0.2}]
    s = summarize_stats(stats, 10.0)
    assert s["requests"] == 2 and s["errors"] == 1 and s["output_tokens_per_s"] == 3.0


def test_the_api_refuses_a_missing_token(tmp_path):
    from fastapi.testclient import TestClient

    from gitm.server.api import build_api
    from gitm.server.controller import Controller, ServerConfig

    ctl = Controller(ServerConfig(model="m", workdir=tmp_path))
    client = TestClient(build_api(ctl))
    assert client.get("/events").status_code == 401
    ok = client.get("/events", headers={"Authorization": f"Bearer {ctl.token}"})
    assert ok.status_code == 200 and ok.json() == []
