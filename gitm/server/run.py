"""``gitm run --server``: start the controller and serve the API until stopped."""

from __future__ import annotations

import shlex
import signal
import sys
import time
from pathlib import Path

from gitm.server.controller import Controller, ServerConfig


def config_from_args(args) -> ServerConfig:
    workdir = Path(args.scratch) if args.scratch else (
        Path.home() / ".cache" / f"gitm-server-{time.strftime('%Y%m%d-%H%M%S')}")
    return ServerConfig(
        model=args.model,
        serve_args=shlex.split(args.serve_args or ""),
        workdir=workdir,
        host=args.host,
        port=args.port,
        api_port=args.api_port,
        stable_port=args.stable_port,
        canary_port=args.canary_port,
        stable_gpus=args.stable_gpus,
        canary_gpus=args.canary_gpus,
        mem_fraction=args.mem_fraction,
        record=not args.no_record,
        canary=not args.no_canary,
        policy=_read_policy(args.policy),
    )


def _read_policy(path: Path | None) -> dict:
    if path is None:
        return {}
    import yaml

    from gitm.server.policy import merged

    data = yaml.safe_load(path.read_text()) or {}
    merged(data)  # refuse unknown keys now, not when the first test runs
    return data


def run_server(cfg: ServerConfig) -> int:
    import uvicorn

    from gitm.server.api import build_api

    ctl = Controller(cfg)
    problems = ctl.preflight()
    if problems:
        for p in problems:
            print(f"gitm server: {p}", file=sys.stderr)
        return 2

    what = "stable and canary engines" if cfg.canary else "stable engine"
    print(f"gitm server: starting {what} for {cfg.model} (logs in {ctl.log_dir})", flush=True)
    try:
        ctl.start()
    except BaseException as exc:
        ctl.stop()
        if isinstance(exc, KeyboardInterrupt):
            return 130
        print(f"gitm server: could not start: {exc}", file=sys.stderr)
        return 1

    print("gitm server: ready", flush=True)
    print(f"  serving    http://{cfg.host}:{cfg.port}/v1   (OpenAI-compatible; point clients here)")
    print(f"  control    http://127.0.0.1:{cfg.api_port}     (token in {ctl.workdir / 'api-token'})")
    print(f"  recording  {ctl.record_dir}" if cfg.record else "  recording  off")
    from urllib.parse import quote

    print(f"  dashboard  gitm dashboard, then open http://127.0.0.1:7080/#server="
          f"{quote(f'http://127.0.0.1:{cfg.api_port}', safe='')}&token={ctl.token}")
    print("  Ctrl-C stops the proxy and every engine this server started.", flush=True)

    server = uvicorn.Server(uvicorn.Config(build_api(ctl), host="127.0.0.1",
                                           port=cfg.api_port, log_level="warning"))
    # uvicorn handles SIGINT/SIGTERM by returning from run(); the finally below
    # is what turns that into a clean stop of everything this server started.
    signal.signal(signal.SIGTERM, signal.default_int_handler)
    try:
        server.run()
    except KeyboardInterrupt:
        pass
    finally:
        print("gitm server: stopping", flush=True)
        ctl.stop()
    return 0
