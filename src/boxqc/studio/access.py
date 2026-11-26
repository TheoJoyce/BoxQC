"""Serve mode: the studio opened to other computers and phones, behind an access key.

``boxqc studio --serve`` listens on every network interface, so a phone or a second
computer on the same network can use the web UI. The studio has no user accounts, so
serve mode adds one shared secret, the *access key*:

* It is made once and kept in ``<data_dir>/.studio-access-key`` (readable only by its
  owner, never part of the dataset export), so links and phones stay signed in across
  restarts. ``--new-key`` replaces it, which signs every other browser out.
* The studio prints a link per network address with the key in it (``/?key=...``) and
  a QR code for phones. Opening the link stores the key in a cookie and redirects to
  the same page without it, so the key does not linger in the address bar. Without the
  link, ``/access`` asks for the key.
* Scripts can send it in an ``X-Studio-Key`` header instead.
* Requests from the computer the studio runs on (loopback) need no key.

Everything is checked here, in one ASGI middleware in front of the whole app: pages,
the studio API, the device proxy (the live camera view streams through untouched),
``/docs`` and ``/openapi.json``. Outside serve mode the middleware is not installed and
nothing changes.
"""

from __future__ import annotations

import hmac
import html
import io
import ipaddress
import logging
import os
import secrets
import socket
from pathlib import Path
from urllib.parse import parse_qsl, quote, urlencode

from starlette.datastructures import Headers
from starlette.responses import HTMLResponse, JSONResponse, RedirectResponse, Response

log = logging.getLogger(__name__)

KEY_FILE = ".studio-access-key"
COOKIE = "boxqc_access"
HEADER = "x-studio-key"
COOKIE_MAX_AGE = 365 * 24 * 3600
# Reachable without the key: the key page itself and the icons it shows.
OPEN_PATHS = {"/access", "/favicon.ico", "/static/favicon.svg", "/static/apple-touch-icon.png"}


def load_or_create_key(data_dir: Path, renew: bool = False) -> str:
    """The studio's access key, made (or with ``renew``, replaced) on first use."""
    path = Path(data_dir) / KEY_FILE
    if not renew:
        try:
            key = path.read_text().strip()
            if len(key) >= 16:
                return key
        except FileNotFoundError:
            pass
    key = secrets.token_urlsafe(15)         # 20 characters, 120 bits
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as f:
        f.write(key + "\n")
    os.chmod(path, 0o600)       # also when the file already existed with wider rights
    return key


def _is_loopback(host: str | None) -> bool:
    try:
        return host is not None and ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def _safe_next(target: str | None) -> str:
    """Only a path on this site: never ``//other.site`` or ``http://...`` (open redirect)."""
    if not target or not target.startswith("/") or target.startswith("//") or "\\" in target:
        return "/"
    return target


class AccessKeyMiddleware:
    """Let a request through only from loopback, or with the access key (cookie, header,
    or a ``key=`` query parameter, which is turned into the cookie)."""

    def __init__(self, app, key: str):
        self.app = app
        self.key = key

    def _matches(self, candidate: str | None) -> bool:
        return bool(candidate) and hmac.compare_digest(candidate.encode(), self.key.encode())

    async def __call__(self, scope, receive, send):
        if scope["type"] not in ("http", "websocket"):
            await self.app(scope, receive, send)
            return
        client = scope.get("client")
        headers = Headers(scope=scope)
        cookies = _cookies(headers.get("cookie", ""))
        open_icon = scope.get("path") in OPEN_PATHS and scope.get("path") != "/access"
        if (open_icon or _is_loopback(client[0] if client else None) or self._matches(headers.get(HEADER))
                or self._matches(cookies.get(COOKIE))):
            if scope.get("path") == "/access" and scope["type"] == "http":
                # Already allowed in (e.g. Back after signing in): go where the form would have.
                query = dict(parse_qsl(scope.get("query_string", b"").decode(errors="replace")))
                response = RedirectResponse(_safe_next(query.get("next")), status_code=303)
                if self._matches(query.get("key")):
                    response.set_cookie(COOKIE, self.key, max_age=COOKIE_MAX_AGE, httponly=True, samesite="lax")
                await response(scope, receive, send)
                return
            await self.app(scope, receive, send)
            return
        response = self._challenge(scope)
        await response(scope, receive, send)

    def _challenge(self, scope) -> Response:
        path = scope.get("path", "/")
        query = dict(parse_qsl(scope.get("query_string", b"").decode(errors="replace")))
        if scope["type"] == "http" and self._matches(query.get("key")):
            # A link with the key: remember it in a cookie, then show the page without it.
            rest = urlencode([(k, v) for k, v in query.items() if k not in ("key", "next")])
            target = _safe_next(query.get("next")) if path == "/access" else path + (f"?{rest}" if rest else "")
            response = RedirectResponse(target, status_code=303)
            response.set_cookie(COOKIE, self.key, max_age=COOKIE_MAX_AGE, httponly=True, samesite="lax")
            return response
        if path == "/access":
            wrong = "key" in query
            return HTMLResponse(_access_page(_safe_next(query.get("next")), wrong),
                                status_code=401 if wrong else 200, headers={"Referrer-Policy": "no-referrer"})
        if path.startswith("/api/") or path in ("/openapi.json",) or scope["type"] == "websocket":
            return JSONResponse({"detail": "This studio needs its access key.", "access": "/access"},
                                status_code=401)
        query_string = scope.get("query_string", b"").decode(errors="replace")
        here = path + (f"?{query_string}" if query_string else "")
        return RedirectResponse(f"/access?next={quote(here, safe='')}", status_code=303)


def _cookies(header: str) -> dict[str, str]:
    out = {}
    for part in header.split(";"):
        name, _, value = part.strip().partition("=")
        if name:
            out[name] = value
    return out


def _access_page(next_path: str, wrong: bool) -> str:
    """The key form. Inline styles: nothing else on the site may be loaded yet."""
    note = '<p class="err">That is not the access key.</p>' if wrong else ""
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Access key · BoxQC</title>
<link rel="icon" href="/static/favicon.svg" type="image/svg+xml">
<style>
  body {{ margin: 0; min-height: 100vh; display: grid; place-items: center; background: #111418;
         color: #e6e9ec; font: 16px system-ui, -apple-system, "Segoe UI", Roboto, sans-serif; }}
  form {{ box-sizing: border-box; width: min(380px, calc(100vw - 2rem)); background: #1a1f25; border: 1px solid #2e3740;
         border-radius: 12px; padding: 1.4rem; }}
  h1 {{ font-size: 1.25rem; margin: 0 0 .5rem; }}
  p {{ color: #8b96a1; margin: 0 0 1rem; line-height: 1.4; }}
  .err {{ color: #ffb4b0; }}
  input, button {{ font: inherit; width: 100%; padding: .7rem .8rem; border-radius: 8px; box-sizing: border-box; }}
  input {{ background: #111418; color: inherit; border: 1px solid #2e3740; margin-bottom: .8rem; }}
  button {{ background: #3d8bfd; color: #fff; border: 0; font-weight: 600; cursor: pointer; }}
</style></head>
<body><form method="get" action="/access">
  <h1>BoxQC studio</h1>
  <p>This studio is shared on the network and needs its access key. Open the link or scan
  the QR code that <code>boxqc studio --serve</code> printed, or type the key here.</p>
  {note}
  <input name="key" autocomplete="off" autocapitalize="off" spellcheck="false" placeholder="Access key"
         required autofocus>
  <input type="hidden" name="next" value="{html.escape(next_path, quote=True)}">
  <button>Open</button>
</form></body></html>"""


# ---- links for other devices ---------------------------------------------------------
def primary_address() -> str | None:
    """The address this machine uses towards other networks (no packet is sent)."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("192.0.2.1", 9))     # TEST-NET-1: never routed, only selects a source address
            return s.getsockname()[0]
    except OSError:
        return None


def share_urls(port: int, key: str, host: str = "0.0.0.0") -> list[str]:
    """``http://<address>:<port>/?key=...`` for each address other devices can use,
    the one towards the default route first."""
    from ..discovery import ipv4_interfaces

    if host not in ("0.0.0.0", "", "::"):
        addresses = [host]
    else:
        addresses = [i.address for i in ipv4_interfaces()]
        first = primary_address()
        if first and not first.startswith("127."):
            addresses = [first, *[a for a in addresses if a != first]]
    return [f"http://{a}:{port}/?key={key}" for a in dict.fromkeys(addresses)]


def qr_code(url: str, kind: str = "terminal") -> str | None:
    """The URL as a QR code: ``terminal`` text or an ``svg`` string. None without the
    optional ``segno`` package (``pip install segno``)."""
    try:
        import segno
    except ImportError:
        return None
    qr = segno.make(url, error="m")
    if kind == "svg":
        buf = io.BytesIO()
        qr.save(buf, kind="svg", scale=4, border=2, dark="#000", light="#fff", xmldecl=False, svgns=True)
        return buf.getvalue().decode()
    buf = io.StringIO()
    qr.terminal(out=buf, compact=True, border=2)
    return buf.getvalue()
