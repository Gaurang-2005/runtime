"""gitm server, milestone 5: the traffic simulator and the dashboard page."""

from __future__ import annotations

import random
import statistics
import threading
import urllib.request

import pytest

from gitm.server.simulate import PRESETS, Simulator, arrivals


def test_poisson_arrivals_hit_the_requested_rate():
    a = arrivals(random.Random(1), 5.0, 2000.0, 1.0)
    assert len(a) / 2000.0 == pytest.approx(5.0, rel=0.05)


def test_bursty_arrivals_keep_the_rate_but_bunch_up():
    rng = random.Random(2)
    calm, burst = arrivals(rng, 2.0, 5000.0, 1.0), arrivals(rng, 2.0, 5000.0, 8.0)
    assert len(burst) / 5000.0 == pytest.approx(2.0, rel=0.15)

    def gap_cv(a):
        gaps = [b - x for x, b in zip(a, a[1:], strict=False)]
        return statistics.pstdev(gaps) / statistics.fmean(gaps)

    assert gap_cv(burst) > 2 * gap_cv(calm)


def test_bad_simulation_requests_are_refused():
    sim = Simulator("http://127.0.0.1:1", "m")
    with pytest.raises(ValueError, match="unknown preset"):
        sim.start("nope")
    with pytest.raises(ValueError):
        sim.start("chat", rate_rps=0)
    assert set(PRESETS) >= {"chat", "agent", "long_doc", "bursty_chat"}


def test_the_dashboard_serves_its_page_with_a_strict_csp():
    from http.server import ThreadingHTTPServer

    from gitm.dashboard import Handler

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    port = httpd.server_address[1]
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/") as r:
            page = r.read().decode()
            csp = r.headers["Content-Security-Policy"]
        assert "<title>gitm dashboard</title>" in page and "script-src 'self'" in csp
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/app.js") as r:
            assert ".innerHTML" not in r.read().decode()    # server data is text only
        with pytest.raises(urllib.error.HTTPError):
            urllib.request.urlopen(f"http://127.0.0.1:{port}/../../etc/passwd")
    finally:
        httpd.shutdown()
