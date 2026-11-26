"""Device discovery (boxqc.discovery) and ``device.url: auto``.

No broadcast is sent: the responder runs on loopback with an ephemeral port, and the
clients get a fake finder, so the tests behave the same with or without a network.
"""

import json

import httpx
import pytest
from fastapi.testclient import TestClient

from boxqc import discovery
from boxqc.config import load_config
from boxqc.device.app import start_discovery
from boxqc.discovery import (
    MIN_QUERY_BYTES,
    DiscoveryError,
    DiscoveryResponder,
    FoundDevice,
    _parse_ifconfig,
    _parse_ip,
    choose,
    discover,
    parse_answer,
)
from boxqc.studio.app import create_app as create_studio_app
from boxqc.studio.device_client import RELOCATE_INTERVAL, DeviceClient, DeviceError


def info(**extra):
    return lambda: {"name": "piqc", "hostname": "piqc", "version": "9.9", "api_version": "1",
                    "port": 8000, "addresses": ["127.0.0.1"], **extra}


@pytest.fixture
def responder():
    r = DiscoveryResponder(info(), host="127.0.0.1", port=0)
    r.start()
    yield r
    r.stop()


def query(nonce="ab" * 16, pad=True):
    data = json.dumps({"boxqc": "discover", "v": 1, "nonce": nonce}).encode()
    return data.ljust(MIN_QUERY_BYTES) if pad else data


# ---- the protocol ------------------------------------------------------------
def test_discover_finds_a_responder_over_loopback(responder):
    found = discover(timeout=0.5, targets=[("127.0.0.1", responder.port)])
    assert len(found) == 1
    device = found[0]
    assert (device.name, device.version, device.url, device.auth) == ("piqc", "9.9", "http://127.0.0.1:8000", False)


def test_answers_are_smaller_than_queries_and_short_queries_are_ignored():
    r = DiscoveryResponder(info(), token="secret")
    assert r.answer(query(pad=False)) is None                    # no amplification
    assert r.answer(b"x" * 600) is None                          # not JSON
    assert r.answer(json.dumps([1]).encode().ljust(600)) is None  # JSON, not a query
    assert r.answer(query(nonce="../etc")) is None               # nonce must be hex
    answer = r.answer(query())
    assert answer is not None and len(answer) < MIN_QUERY_BYTES
    assert json.loads(answer)["auth"] is True


def test_a_long_name_is_cut_so_the_answer_stays_small():
    r = DiscoveryResponder(info(name="n" * 500, hostname="h" * 500, addresses=[f"10.0.0.{i}" for i in range(50)]),
                           token="t")
    answer = r.answer(query())
    assert answer is not None and len(answer) < MIN_QUERY_BYTES


def test_the_token_proof_is_checked():
    nonce = "cd" * 16
    answer = DiscoveryResponder(info(), token="secret").answer(query(nonce))
    assert parse_answer(answer, "127.0.0.1", nonce, token="secret").verified is True
    assert parse_answer(answer, "127.0.0.1", nonce).verified is None    # no token: nothing to check
    assert parse_answer(answer, "127.0.0.1", nonce, token="wrong").verified is False
    # relayed from a machine that is not one of the addresses the device lists
    assert parse_answer(answer, "10.9.9.9", nonce, token="secret").verified is False
    # a tampered answer (another port) no longer matches its proof
    forged = json.loads(answer)
    forged["port"] = 9999
    assert parse_answer(json.dumps(forged).encode(), "127.0.0.1", nonce, token="secret").verified is False
    # an answer to someone else's query is no answer at all
    assert parse_answer(answer, "127.0.0.1", "ef" * 16) is None


def test_with_a_token_an_answer_claiming_no_auth_is_not_trusted():
    """Otherwise anything on the network could answer auth: false and be sent the token."""
    nonce = "cd" * 16
    spoof = DiscoveryResponder(info()).answer(query(nonce))      # no token: auth false, no proof
    found = parse_answer(spoof, "127.0.0.1", nonce, token="secret")
    assert found.verified is False and "requires no API token" in found.problem


def test_discover_with_a_token_drops_unproven_devices_unless_asked():
    impostor = DiscoveryResponder(info(name="impostor"), host="127.0.0.1", port=0)
    real = DiscoveryResponder(info(name="real"), token="secret", host="127.0.0.1", port=0)
    for r in (impostor, real):
        r.start()
    try:
        targets = [("127.0.0.1", impostor.port), ("127.0.0.1", real.port)]
        assert [d.name for d in discover(0.5, token="secret", targets=targets)] == ["real"]
        listed = discover(0.5, token="secret", targets=targets, include_unverified=True)
        assert [(d.name, d.verified) for d in listed] == [("impostor", False), ("real", True)]
    finally:
        for r in (impostor, real):
            r.stop()


def test_the_rate_limit_is_checked_before_any_work(monkeypatch):
    calls = []
    r = DiscoveryResponder(lambda: calls.append(1) or info()(), host="127.0.0.1", port=0)
    monkeypatch.setattr(r, "_allowed", lambda: False)
    r.start()
    try:
        assert discover(0.3, targets=[("127.0.0.1", r.port)]) == []
    finally:
        r.stop()
    assert calls == []      # info() (a subprocess on a real device) never ran


def test_interface_parsing():
    ip = ("2: wlan0    inet 192.168.1.50/24 brd 192.168.1.255 scope global dynamic wlan0\n"
          "3: usb0    inet 10.12.194.1/28 scope global usb0\n")
    assert [(i.name, i.address, i.broadcast) for i in _parse_ip(ip)] == [
        ("wlan0", "192.168.1.50", "192.168.1.255"), ("usb0", "10.12.194.1", "10.12.194.15")]
    mac = ("en0: flags=8863<UP,BROADCAST> mtu 1500\n"
           "\tinet 192.168.1.20 netmask 0xffffff00 broadcast 192.168.1.255\n"
           "en8: flags=8863<UP> mtu 1500\n\tinet 10.12.194.2 netmask 0xfffffff0\n")
    assert [(i.name, i.address, i.broadcast) for i in _parse_ifconfig(mac)] == [
        ("en0", "192.168.1.20", "192.168.1.255"), ("en8", "10.12.194.2", "10.12.194.15")]


def device_named(name, address="10.0.0.1"):
    return FoundDevice(name=name, hostname=name, address=address, port=8000, version="1", auth=False, id=name)


def test_choose_never_guesses():
    a, b = device_named("a"), device_named("b", "10.0.0.2")
    assert choose([a]) is a
    assert choose([a, b], "b") is b
    with pytest.raises(DiscoveryError, match="2 devices answered") as exc:
        choose([a, b])
    assert exc.value.found == [a, b]
    with pytest.raises(DiscoveryError, match="No device named 'c'"):
        choose([a, b], "c")
    with pytest.raises(DiscoveryError, match="No BoxQC device answered") as exc:
        choose([])
    assert exc.value.found == []


def test_the_device_answers_with_its_configured_name(device_cfg):
    device_cfg.device.name = "line-2"
    responder = start_discovery(device_cfg, "127.0.0.1", 8123, discovery_port=0)
    try:
        found = discover(timeout=0.5, targets=[("127.0.0.1", responder.port)])
    finally:
        responder.stop()
    assert [(d.name, d.url) for d in found] == [("line-2", "http://127.0.0.1:8123")]


def test_a_port_that_cannot_be_opened_only_turns_discovery_off(device_cfg, monkeypatch, caplog):
    def busy(self):
        raise OSError(48, "Address already in use")

    monkeypatch.setattr(DiscoveryResponder, "start", busy)
    assert start_discovery(device_cfg, "0.0.0.0", 8000) is None
    assert "Discovery is off" in caplog.text


def test_auto_is_the_default_and_any_spelling_works(tmp_path, monkeypatch):
    monkeypatch.delenv("BOXQC_DEVICE_URL", raising=False)
    (tmp_path / "boxqc.yaml").write_text("device:\n  url: AUTO\n")
    assert load_config(tmp_path / "boxqc.yaml").device.url == "auto"
    (tmp_path / "boxqc.yaml").write_text("{}\n")
    assert load_config(tmp_path / "boxqc.yaml").device.url == "auto"


# ---- DeviceClient in auto mode --------------------------------------------------
class Finder:
    """Stands in for the network: returns whatever ``devices`` holds, counting calls."""

    def __init__(self, *devices):
        self.devices = list(devices)
        self.calls = 0

    def __call__(self):
        self.calls += 1
        return list(self.devices)


class Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


def test_auto_finds_the_device_on_first_use(device):
    finder = Finder(device_named("piqc", "device"))
    client = DeviceClient("auto", http=device, finder=finder)
    assert client.url is None and client.base_url == "auto"
    assert client.status()["version"]
    assert client.url == "http://device:8000" and finder.calls == 1
    client.settings()
    assert finder.calls == 1        # found once, then reused


def test_an_explicit_url_never_searches():
    finder = Finder()
    client = DeviceClient("http://10.42.0.2:8000/", finder=finder)
    assert client.locate() == "http://10.42.0.2:8000"
    client.connection_failed()
    assert client.url == "http://10.42.0.2:8000" and finder.calls == 0


def _refuse(request):
    raise httpx.ConnectError("refused", request=request)


def test_auto_searches_again_after_the_device_moves():
    finder, clock = Finder(device_named("piqc", "10.0.0.1")), Clock()
    http = httpx.Client(transport=httpx.MockTransport(_refuse), base_url="http://placeholder")
    client = DeviceClient("auto", http=http, finder=finder, clock=clock)
    with pytest.raises(DeviceError, match="Cannot reach the device at http://10.0.0.1:8000"):
        client.status()
    assert client.url is None       # forgotten: it may be somewhere else now
    finder.devices = [device_named("piqc", "10.0.0.7")]
    assert client.locate() == "http://10.0.0.7:8000"
    assert str(http.base_url).rstrip("/") == "http://10.0.0.7:8000"


def test_a_failed_search_is_not_repeated_on_every_request():
    finder, clock = Finder(), Clock()
    client = DeviceClient("auto", finder=finder, clock=clock)
    for _ in range(3):
        with pytest.raises(DeviceError, match="No BoxQC device answered") as exc:
            client.status()
        assert exc.value.unreachable
    assert finder.calls == 1
    clock.now += RELOCATE_INTERVAL + 1
    finder.devices = [device_named("piqc")]
    assert client.locate() == "http://10.0.0.1:8000"
    assert finder.calls == 2


def test_several_devices_need_a_name():
    finder = Finder(device_named("a"), device_named("b", "10.0.0.2"))
    with pytest.raises(DeviceError, match="Set device.name"):
        DeviceClient("auto", finder=finder).locate()
    assert DeviceClient("auto", finder=finder, name="b").locate() == "http://10.0.0.2:8000"


# ---- the studio ------------------------------------------------------------------
def test_studio_in_auto_mode_reports_and_proxies_the_found_device(cfg, device):
    finder = Finder()
    client = DeviceClient("auto", http=device, finder=finder)
    proxy = httpx.AsyncClient(transport=httpx.ASGITransport(app=device.app))
    with TestClient(create_studio_app(cfg, device=client, proxy=proxy)) as studio:
        status = studio.get("/api/status").json()
        assert status["device"] is None and status["device_url"] is None and status["device_auto"]
        assert "No BoxQC device answered" in status["device_error"]
        assert studio.get("/system").status_code == 200
        response = studio.get("/api/device/settings")
        assert response.status_code == 502 and "No BoxQC device answered" in response.json()["detail"]

        finder.devices = [device_named("piqc", "device")]
        client.locate(force=True)
        assert studio.get("/api/device/settings").status_code == 200
        studio.app.state.link.invalidate()
        status = studio.get("/api/status").json()
        assert status["device_url"] == "http://device:8000" and status["device"]["version"]


def test_the_discovery_module_imports_nothing_outside_the_standard_library():
    """The device imports it, and the Pi installs no extra package for it."""
    import ast
    import sys

    tree = ast.parse(open(discovery.__file__).read())
    modules = {alias.name.split(".")[0] for node in ast.walk(tree) if isinstance(node, ast.Import)
               for alias in node.names}
    modules |= {node.module.split(".")[0] for node in ast.walk(tree)
                if isinstance(node, ast.ImportFrom) and node.module and node.level == 0}
    assert modules <= set(sys.stdlib_module_names) | {"__future__"}


def test_a_lost_query_is_sent_again(monkeypatch):
    """Wi-Fi does not resend lost broadcast frames: discover() repeats the query itself."""
    r = DiscoveryResponder(info(), host="127.0.0.1", port=0)
    real, seen = r._nonce_of, []

    def drop_the_first(data):
        seen.append(1)
        return None if len(seen) == 1 else real(data)

    monkeypatch.setattr(r, "_nonce_of", drop_the_first)
    r.start()
    try:
        assert [d.name for d in discover(1.0, targets=[("127.0.0.1", r.port)])] == ["piqc"]
    finally:
        r.stop()
    assert len(seen) >= 2
