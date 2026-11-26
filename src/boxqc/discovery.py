"""Finding a device on the local network without knowing its address.

A running device (``boxqc device``) answers *discovery queries*: small UDP datagrams
sent to port :data:`DISCOVERY_PORT` (47800), usually as a broadcast. The studio and the
command line send one when ``device.url`` is ``auto``, collect the answers for about a
second, and use the address each answer came from. ``boxqc devices`` lists them.

Why not mDNS / Bonjour (``_boxqc._tcp``)? It was the first choice and was tried on the
network this project was built on: a campus Wi-Fi that filters mDNS (UDP 5353) between
clients while letting ordinary broadcast through. Many managed Wi-Fi networks do this.
A plain broadcast works there, on home networks and over the USB link, and needs no
library on either side. The cost is that generic Bonjour browsers do not list the
device.

The wire format (``docs/device-api.md#discovery`` is the reference for integrators):

* **Query**: UTF-8 JSON ``{"boxqc": "discover", "v": 1, "nonce": "<32 hex digits>"}``,
  padded with spaces to at least :data:`MIN_QUERY_BYTES`. A shorter datagram is
  ignored, and every answer is smaller than that, so a forged sender address cannot
  make a device send more bytes than it was sent (no amplification).
* **Answer**: UTF-8 JSON sent back to the query's source address and port::

      {"boxqc": "device", "v": 1, "nonce": "<the query's>", "id": "<random, per start>",
       "name": "piqc", "hostname": "piqc", "version": "0.2.0", "api_version": "1",
       "port": 8000, "addresses": ["10.12.194.1", "192.168.1.50"],
       "auth": true, "proof": "<hex HMAC-SHA256 or null>"}

  ``auth`` says whether the API requires a token. When it does, ``proof`` is
  HMAC-SHA256, keyed with the token, over the answer without ``proof`` (JSON, keys
  sorted, no spaces). A client holding the token checks it, and only then sends the
  token to the device. Since ``addresses`` and ``nonce`` are covered, a machine that
  replays or relays a real device's answer from its own address is not accepted.
"""

from __future__ import annotations

import contextlib
import hashlib
import hmac
import ipaddress
import json
import logging
import re
import secrets
import socket
import subprocess
import threading
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field

log = logging.getLogger(__name__)

DISCOVERY_PORT = 47800
MIN_QUERY_BYTES = 1024       # queries are padded to this; answers stay well below it (< 600)
MAX_NAME = 64                # longer names are cut, to keep the answer small
PROTOCOL_VERSION = 1
DEFAULT_TIMEOUT = 1.0        # seconds a client listens for answers
# Wi-Fi does not acknowledge or resend broadcast frames, so one query can simply be lost:
# a client sends it again at these times (seconds) while it listens.
RESEND_AT = (0.0, 0.3, 0.6)
MAX_ANSWERS_PER_SECOND = 20  # a device answers at most this often, whoever asks


# ---- interfaces ---------------------------------------------------------------
@dataclass(frozen=True)
class Interface:
    name: str
    address: str
    broadcast: str | None


def ipv4_interfaces() -> list[Interface]:
    """This machine's IPv4 interfaces other than loopback, with their broadcast address.

    Read from ``ip`` on Linux and ``ifconfig`` on macOS, so no library is needed. An
    empty list when neither works: discovery then only uses 255.255.255.255.
    """
    for command, parse in ((["ip", "-4", "-o", "addr", "show"], _parse_ip),
                           (["ifconfig"], _parse_ifconfig)):
        try:
            out = subprocess.run(command, capture_output=True, text=True, timeout=2).stdout
        except (OSError, subprocess.SubprocessError):
            continue
        interfaces = [i for i in parse(out) if not i.address.startswith("127.")]
        if interfaces:
            return interfaces
    return []


def _parse_ip(out: str) -> list[Interface]:
    """``ip -4 -o addr show``: ``2: wlan0    inet 192.168.1.50/24 brd 192.168.1.255 scope ...``"""
    found = []
    for line in out.splitlines():
        m = re.search(r"^\d+:\s+(\S+)\s+inet\s+([\d.]+)/(\d+)(?:\s+brd\s+([\d.]+))?", line)
        if m:
            name, address, prefix, brd = m.groups()
            found.append(Interface(name, address, brd or _broadcast_of(address, int(prefix))))
    return found


def _parse_ifconfig(out: str) -> list[Interface]:
    """macOS ``ifconfig``: ``inet 192.168.1.20 netmask 0xffffff00 broadcast 192.168.1.255``"""
    found, name = [], None
    for line in out.splitlines():
        if m := re.match(r"^(\S+?):? ", line):
            name = m.group(1)
        m = re.search(r"\binet ([\d.]+) netmask (0x[0-9a-f]+)(?: broadcast ([\d.]+))?", line)
        if m and name:
            address, mask, brd = m.groups()
            found.append(Interface(name, address, brd or _broadcast_of(address, bin(int(mask, 16)).count("1"))))
    return found


def _broadcast_of(address: str, prefix: int) -> str | None:
    if prefix >= 31:        # point-to-point: no broadcast address
        return None
    return str(ipaddress.ip_network(f"{address}/{prefix}", strict=False).broadcast_address)


# ---- the answer, and its proof --------------------------------------------------
def _canonical(payload: dict) -> bytes:
    return json.dumps({k: v for k, v in payload.items() if k != "proof"}, sort_keys=True,
                      separators=(",", ":")).encode()


def sign(payload: dict, token: str) -> str:
    return hmac.new(token.encode(), _canonical(payload), hashlib.sha256).hexdigest()


def verify(payload: dict, token: str) -> bool:
    proof = payload.get("proof")
    return isinstance(proof, str) and hmac.compare_digest(proof, sign(payload, token))


@dataclass
class FoundDevice:
    """One device that answered. ``address`` is where its answer came from."""

    name: str
    hostname: str
    address: str
    port: int
    version: str
    auth: bool
    id: str
    addresses: list[str] = field(default_factory=list)
    # Only with a token to check against: whether the answer proved it holds that token
    # (None: no token, nothing to check). ``problem`` says why not.
    verified: bool | None = None
    problem: str | None = None

    @property
    def url(self) -> str:
        return f"http://{self.address}:{self.port}"

    def describe(self) -> str:
        return f"{self.name} at {self.url}"


def parse_answer(data: bytes, source: str, nonce: str, token: str | None = None) -> FoundDevice | None:
    """A :class:`FoundDevice` from an answer datagram, or None if it is not an answer.

    With ``token``, ``verified`` says whether the answer can be trusted with it: it must
    carry a valid proof (an answer claiming ``auth: false`` has none, so it fails) and
    come from one of the addresses it lists, since the token is about to be sent there.
    """
    try:
        payload = json.loads(data.decode())
        if not isinstance(payload, dict) or payload.get("boxqc") != "device" or payload.get("nonce") != nonce:
            return None
        found = FoundDevice(
            name=str(payload["name"]), hostname=str(payload.get("hostname", payload["name"])), address=source,
            port=int(payload["port"]), version=str(payload.get("version", "?")), auth=bool(payload.get("auth")),
            id=str(payload["id"]), addresses=[str(a) for a in payload.get("addresses") or []])
    except (ValueError, KeyError, TypeError, UnicodeDecodeError):
        return None
    if token is not None:
        if not found.auth:
            found.problem = "it requires no API token, so it cannot prove it holds this one"
        elif not verify(payload, token):
            found.problem = "its proof does not match this API token"
        elif source not in found.addresses:
            found.problem = "it came from an address it does not list"
        found.verified = found.problem is None
    return found


# ---- the client side ---------------------------------------------------------------
def query_targets(port: int = DISCOVERY_PORT) -> list[tuple[str, int]]:
    """Where a query goes: this machine itself, every interface's broadcast address, and
    255.255.255.255 (which most systems send out of one interface only)."""
    targets = ["127.0.0.1", *(i.broadcast for i in ipv4_interfaces() if i.broadcast), "255.255.255.255"]
    return [(t, port) for t in dict.fromkeys(targets)]


def discover(timeout: float = DEFAULT_TIMEOUT, token: str | None = None, port: int = DISCOVERY_PORT,
             targets: Iterable[tuple[str, int]] | None = None,
             include_unverified: bool = False) -> list[FoundDevice]:
    """Ask the local network for devices; return those that answered within ``timeout``.

    With ``token``, only devices that proved they hold it are returned, unless
    ``include_unverified`` (for listing them; their ``verified`` is False). A device
    reachable over two links (USB and Wi-Fi) answers once per link; the first answer is
    kept, which is normally the faster link. Sorted by name.
    """
    nonce = secrets.token_hex(16)
    query = json.dumps({"boxqc": "discover", "v": PROTOCOL_VERSION, "nonce": nonce}).encode()
    query = query.ljust(MIN_QUERY_BYTES)
    found: dict[str, FoundDevice] = {}
    rejected: set[str] = set()
    targets = list(targets if targets is not None else query_targets(port))
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        sock.bind(("", 0))
        start = time.monotonic()
        deadline = start + timeout
        sends = [start + t for t in RESEND_AT if t < timeout]
        while (left := deadline - time.monotonic()) > 0:
            if sends and time.monotonic() >= sends[0]:
                sends.pop(0)
                for target in targets:
                    try:
                        sock.sendto(query, target)
                    except OSError as exc:      # e.g. no route for a broadcast address
                        log.debug("discovery query to %s failed: %s", target, exc)
            wait = left if not sends else max(0.01, min(left, sends[0] - time.monotonic()))
            sock.settimeout(wait)
            try:
                data, (source, _) = sock.recvfrom(4096)
            except TimeoutError:
                continue        # time to send again, or the deadline (checked above)
            except OSError as exc:
                log.debug("discovery receive failed: %s", exc)
                break
            device = parse_answer(data, source, nonce, token)
            if device is None or device.id in found:
                continue
            if device.verified is False and not include_unverified:
                if source not in rejected:
                    rejected.add(source)
                    log.warning("Ignored the discovery answer from %s: %s", source, device.problem)
                continue
            found[device.id] = device
    return sorted(found.values(), key=lambda d: (d.name, d.address))


class DiscoveryError(RuntimeError):
    """No device, or more than one, answered for ``device.url: auto``.

    ``found`` holds the devices that did answer: empty if none did, otherwise the
    problem is which one to use (several, or none with the configured name).
    """

    def __init__(self, message: str, found: list[FoundDevice] | None = None):
        super().__init__(message)
        self.found = found or []


def choose(devices: list[FoundDevice], name: str | None = None) -> FoundDevice:
    """The one device to use: the only one found, or the one called ``name``.

    Several devices and no ``name`` is an error rather than a guess: deploying a model
    to the wrong station is worse than being asked to pick.
    """
    if name:
        matches = [d for d in devices if name in (d.name, d.hostname)]
        if matches:
            return matches[0]
        others = f" Found instead: {', '.join(d.describe() for d in devices)}." if devices else ""
        raise DiscoveryError(f"No device named {name!r} answered on the local network.{others}", devices)
    if not devices:
        raise DiscoveryError(
            "No BoxQC device answered on the local network (device.url: auto). Check that it is "
            "switched on, on this network, and runs a BoxQC that answers discovery (docs/upgrading.md), "
            "or set device.url to its address.")
    if len(devices) > 1:
        raise DiscoveryError(
            f"{len(devices)} devices answered: {', '.join(d.describe() for d in devices)}. "
            "Set device.name (or device.url) to choose one.", devices)
    return devices[0]


# ---- the device side ---------------------------------------------------------------
class DiscoveryResponder:
    """Answers discovery queries for one device, in a background thread.

    ``info`` returns the answer's fields at the time of each query, so a changed
    address (the USB link coming up after boot, a new Wi-Fi lease) is always current.
    The socket is bound to all interfaces, so interfaces that appear later are covered.
    """

    def __init__(self, info: Callable[[], dict], token: str | None = None, host: str = "0.0.0.0",
                 port: int = DISCOVERY_PORT):
        self._info = info
        self._token = token
        self._host, self._port = host, port
        self._id = secrets.token_hex(8)
        self._sock: socket.socket | None = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._window, self._count = 0.0, 0

    @property
    def port(self) -> int:
        """The bound port (useful when started with port 0)."""
        return self._sock.getsockname()[1] if self._sock else self._port

    def start(self) -> None:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind((self._host, self._port))
        except OSError:
            sock.close()
            raise
        sock.settimeout(0.5)        # so stop() is noticed
        self._sock = sock
        self._thread = threading.Thread(target=self._serve, name="boxqc-discovery", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)
        if self._sock:
            self._sock.close()

    def answer(self, data: bytes) -> bytes | None:
        """The answer to one datagram, or None to stay silent."""
        nonce = self._nonce_of(data)
        return None if nonce is None else self._build(nonce, len(data))

    @staticmethod
    def _nonce_of(data: bytes) -> str | None:
        """The nonce of a valid query (cheap: no subprocess, no HMAC), else None."""
        if len(data) < MIN_QUERY_BYTES:
            return None
        try:
            query = json.loads(data.decode())
        except (ValueError, UnicodeDecodeError):
            return None
        if not isinstance(query, dict) or query.get("boxqc") != "discover":
            return None
        nonce = query.get("nonce")
        if not isinstance(nonce, str) or not re.fullmatch(r"[0-9a-f]{8,64}", nonce):
            return None
        return nonce

    def _build(self, nonce: str, query_size: int) -> bytes | None:
        info = self._info()
        payload = {
            "boxqc": "device", "v": PROTOCOL_VERSION, "nonce": nonce, "id": self._id,
            "name": str(info["name"])[:MAX_NAME], "hostname": str(info["hostname"])[:MAX_NAME],
            "version": str(info["version"])[:16], "api_version": str(info["api_version"])[:8],
            "port": int(info["port"]), "addresses": list(info["addresses"])[:8], "auth": bool(self._token),
        }
        payload["proof"] = sign(payload, self._token) if self._token else None
        out = json.dumps(payload, separators=(",", ":")).encode()
        return out if len(out) < query_size else None

    def _allowed(self) -> bool:
        now = time.monotonic()
        if now - self._window >= 1.0:
            self._window, self._count = now, 0
        self._count += 1
        return self._count <= MAX_ANSWERS_PER_SECOND

    def _serve(self) -> None:
        assert self._sock is not None
        while not self._stop.is_set():
            try:
                data, source = self._sock.recvfrom(4096)
            except TimeoutError:
                continue
            except OSError:
                if self._stop.is_set():
                    break
                time.sleep(0.5)
                continue
            # Cheap checks first, then the rate limit, and only then the work of answering.
            nonce = self._nonce_of(data)
            if nonce is None or not self._allowed():
                continue
            try:
                reply = self._build(nonce, len(data))
            except Exception:       # a broken info() must never stop the responder
                log.exception("discovery: could not answer %s", source)
                continue
            if reply is not None:
                with contextlib.suppress(OSError):
                    self._sock.sendto(reply, source)
