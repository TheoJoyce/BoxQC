from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient
from helpers import FakeTrainer

from boxqc.config import Config
from boxqc.device.app import create_app as create_device_app
from boxqc.studio.app import create_app as create_studio_app
from boxqc.studio.device_client import DeviceClient
from boxqc.studio.training import TRAINERS, register_trainer


def make_cfg(root: Path) -> Config:
    """A config rooted in a temp dir, with the mock camera and no GPIO."""
    c = Config()
    c.data_dir = root / "data"
    c.models_dir = root / "models"
    c.train.workdir = root / "work"
    c.train.trainer = "fake"
    c.train.base_model = "tiny"
    c.camera.backend = "mock"
    c.camera.capture_size = [640, 360]
    c.camera.preview_size = [320, 180]
    c.hardware.mock = True
    return c


@pytest.fixture(autouse=True)
def no_broadcasts(monkeypatch):
    """device.url defaults to auto: make sure no test sends discovery queries to the real
    network (a test that wants an answer passes its own loopback target)."""
    import boxqc.discovery

    monkeypatch.setattr(boxqc.discovery, "query_targets", lambda port=0: [])


@pytest.fixture(autouse=True)
def fake_trainer():
    register_trainer(FakeTrainer)
    yield
    TRAINERS.unregister("fake")


@pytest.fixture
def cfg(tmp_path: Path) -> Config:
    """The studio's (desktop) configuration."""
    return make_cfg(tmp_path / "desktop")


@pytest.fixture
def device_cfg(tmp_path: Path) -> Config:
    """The device's (Pi) configuration: its own data and models directories."""
    return make_cfg(tmp_path / "pi")


@pytest.fixture
def device(device_cfg):
    """A running device app; ``.state`` is its DeviceState, ``.app`` the ASGI app."""
    app = create_device_app(device_cfg)
    with TestClient(app) as client:
        client.state = app.state.boxqc
        yield client


@pytest.fixture
def studio(cfg, device):
    """A running studio app connected to the in-process device."""
    app = create_studio_app(
        cfg,
        device=DeviceClient("http://device", http=device),
        proxy=httpx.AsyncClient(transport=httpx.ASGITransport(app=device.app), base_url="http://device"),
    )
    with TestClient(app) as client:
        client.training = app.state.training
        yield client


def _refuse(request: httpx.Request) -> httpx.Response:
    raise httpx.ConnectError("connection refused", request=request)


@pytest.fixture
def offline_studio(cfg):
    """A running studio app whose device cannot be reached."""
    app = create_studio_app(
        cfg,
        device=DeviceClient("http://device", http=httpx.Client(transport=httpx.MockTransport(_refuse),
                                                                base_url="http://device")),
        proxy=httpx.AsyncClient(transport=httpx.MockTransport(_refuse), base_url="http://device"),
    )
    with TestClient(app) as client:
        client.training = app.state.training
        yield client
