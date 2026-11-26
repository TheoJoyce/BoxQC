"""Serve mode: the studio behind an access key (boxqc.studio.access).

Starlette's TestClient reports its client address as "testclient", which is not
loopback, so it stands for a phone on the network. ``local`` overrides it with 127.0.0.1.
"""

import stat

import httpx
import pytest
from fastapi.testclient import TestClient

from boxqc.studio.access import COOKIE, KEY_FILE, load_or_create_key
from boxqc.studio.app import create_app as create_studio_app
from boxqc.studio.device_client import DeviceClient

KEY = "k" * 20


@pytest.fixture
def served(cfg, device):
    app = create_studio_app(
        cfg, device=DeviceClient("http://device", http=device),
        proxy=httpx.AsyncClient(transport=httpx.ASGITransport(app=device.app), base_url="http://device"),
        access_key=KEY, share_urls=[f"http://192.168.1.20:8080/?key={KEY}"])
    with TestClient(app, follow_redirects=False) as client:
        yield client


def test_without_the_key_nothing_is_reachable(served):
    page = served.get("/inspect")
    assert page.status_code == 303 and page.headers["location"] == "/access?next=%2Finspect"
    for path in ("/api/status", "/api/images", "/api/device/status", "/api/share", "/openapi.json"):
        response = served.get(path)
        assert response.status_code == 401, path
        assert response.json()["access"] == "/access"
    assert served.post("/api/datasets", json={"name": "x"}).status_code == 401
    assert served.get("/docs").status_code == 303
    assert served.get("/static/style.css").status_code == 303
    # the key page and its icon are open
    assert served.get("/access").status_code == 200
    assert served.get("/static/favicon.svg").status_code == 200


def test_the_link_sets_a_cookie_and_drops_the_key_from_the_address(served):
    response = served.get(f"/inspect?key={KEY}&x=1")
    assert response.status_code == 303 and response.headers["location"] == "/inspect?x=1"
    cookie = response.headers["set-cookie"]
    assert cookie.startswith(f"{COOKIE}={KEY}") and "HttpOnly" in cookie and "samesite=lax" in cookie.lower()
    assert served.get("/inspect").status_code == 200          # the client kept the cookie
    assert served.get("/api/device/status").status_code == 200
    assert served.get("/api/share").json()["serve"] is True


def test_the_key_form(served):
    wrong = served.get("/access?key=nope&next=/dataset")
    assert wrong.status_code == 401 and "not the access key" in wrong.text
    right = served.get(f"/access?key={KEY}&next=/dataset")
    assert right.status_code == 303 and right.headers["location"] == "/dataset"
    # never a redirect to another site
    served.cookies.clear()
    assert served.get(f"/access?key={KEY}&next=//evil.example/").headers["location"] == "/"
    served.cookies.clear()
    assert served.get(f"/access?key={KEY}&next=https://evil.example/").headers["location"] == "/"


def test_the_key_page_with_a_valid_cookie_goes_on(served):
    """Back to /access after signing in must not end on a 404."""
    served.get(f"/?key={KEY}")
    response = served.get("/access?next=/training")
    assert response.status_code == 303 and response.headers["location"] == "/training"


def test_scripts_can_send_the_key_in_a_header(served):
    assert served.get("/api/status", headers={"X-Studio-Key": KEY}).status_code == 200
    assert served.get("/api/status", headers={"X-Studio-Key": "wrong"}).status_code == 401


def test_this_computer_needs_no_key(cfg, device):
    app = create_studio_app(cfg, device=DeviceClient("http://device", http=device), access_key=KEY)
    with TestClient(app, client=("127.0.0.1", 50000)) as local:
        assert local.get("/api/status").status_code == 200


def test_without_serve_mode_nothing_changes(studio):
    assert studio.get("/api/status").status_code == 200
    assert studio.get("/api/share").json() == {"serve": False, "urls": [], "qr_svg": None}


def test_the_key_is_kept_private_and_out_of_the_export(served, cfg):
    key = load_or_create_key(cfg.data_dir)
    path = cfg.data_dir / KEY_FILE
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert load_or_create_key(cfg.data_dir) == key              # kept across restarts
    assert load_or_create_key(cfg.data_dir, renew=True) != key  # --new-key
    served.get(f"/?key={KEY}")
    import io
    import zipfile

    names = zipfile.ZipFile(io.BytesIO(served.get("/api/dataset/export.zip").content)).namelist()
    assert not any(KEY_FILE in n for n in names)
