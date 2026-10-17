"""``gitm dashboard``: a page that links to a gitm server and drives it.

A separate process from the server on purpose: one dashboard can be pointed at
any gitm server by URL and token, and closing it changes nothing on the server.
It serves two static files and nothing else; every number on the page comes
from the server's own API (which allows this origin through CORS).

Same posture as :mod:`gitm.traffic.gui`: loopback only, a strict CSP, no build
step, standard library only.
"""

from __future__ import annotations

import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib import resources

HOST = "127.0.0.1"
FILES = {"/": ("index.html", "text/html; charset=utf-8"),
         "/app.js": ("app.js", "text/javascript; charset=utf-8")}
CSP = ("default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; "
       "connect-src http: https:; img-src 'self' data:; frame-ancestors 'none'")


def _read(name: str) -> bytes:
    return resources.files(__package__).joinpath(name).read_bytes()


class Handler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:  # noqa: N802 - http.server's name
        host = (self.headers.get("Host") or "").split(":")[0]
        if host not in (HOST, "localhost"):
            self.send_error(403, "loopback only")
            return
        entry = FILES.get(self.path.split("?")[0])
        if entry is None:
            self.send_error(404)
            return
        body = _read(entry[0])
        self.send_response(200)
        self.send_header("Content-Type", entry[1])
        self.send_header("Content-Security-Policy", CSP)
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt, *args) -> None:
        pass


def serve(port: int = 7080, *, open_browser: bool = True) -> int:
    httpd = ThreadingHTTPServer((HOST, port), Handler)
    url = f"http://{HOST}:{port}/"
    print(f"gitm dashboard: {url}  (Ctrl-C to stop)", flush=True)
    if open_browser:
        webbrowser.open(url)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
    return 0
