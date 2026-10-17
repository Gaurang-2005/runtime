"""gitm server: a long-running controller beside a real ``vllm serve``.

``gitm run --server`` launches a stable engine behind gitm's own OpenAI-compatible
proxy, records the traffic it serves, and exposes a control API. Candidates are
tested on a canary engine against that recorded traffic and promoted only when
every rule in the promotion policy passes.

Process layout, and why:

* the **proxy** (:mod:`gitm.server.proxy`) is its own process, started in its own
  session. It is the only part on a request's path, so a controller crash leaves
  traffic flowing to the last stable engine — the fail-open rule;
* the **engines** (:mod:`gitm.server.supervisor`) are ``vllm serve`` processes,
  each with its own environment, port and GPUs;
* the **controller** (:mod:`gitm.server.controller`) owns the decisions and the
  API, and talks to the proxy over a unix socket.
"""
