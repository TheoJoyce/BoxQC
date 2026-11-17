"""The device API (docs/device-api.md), exercised against the app with a mock camera."""

import hashlib
import sys
import types

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient
from helpers import FakeDetector, make_onnx, one_box_preds

from boxqc.config import CameraConfig
from boxqc.device import camera as camera_module
from boxqc.device import inspector as inspector_module
from boxqc.device.app import create_app
from boxqc.device.camera import MockCamera, open_camera
from boxqc.device.inspector import model_usage

V1 = "/api/v1"


def decode(jpeg: bytes) -> np.ndarray:
    return cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)


def test_status_and_landing_page(device):
    s = device.get(f"{V1}/status").json()
    assert s["api_version"] == "1" and s["camera"] == "mock" and s["model"] is None
    assert set(s["settings"]) == {"conf_threshold", "roi", "active_model"}
    assert "dataset" not in s           # the device keeps no training data
    assert "BoxQC device" in device.get("/").text
    assert device.get("/docs").status_code == 200
    paths = device.get("/openapi.json").json()["paths"]
    assert f"{V1}/inspect" in paths and "/api/inspect" not in paths     # only v1 is documented


def test_unversioned_0_1_paths_still_answer(device):
    assert device.get("/api/status").json()["api_version"] == "1"
    assert device.post("/api/inspect").json()["verdict"] == "NO_MODEL"
    assert device.get("/api/snapshot.jpg").headers["content-type"] == "image/jpeg"


def test_camera_capture_returns_a_full_frame_and_stores_nothing(device, device_cfg):
    r = device.post(f"{V1}/camera/capture")
    assert r.headers["content-type"] == "image/jpeg"
    assert (r.headers["x-image-width"], r.headers["x-image-height"]) == ("640", "360")
    assert decode(r.content).shape == (360, 640, 3)
    assert decode(device.get(f"{V1}/camera/snapshot.jpg").content).shape == (180, 320, 3)
    assert not list(device_cfg.data_dir.rglob("*.jpg"))


def test_inspect_without_model_reports_no_model(device):
    assert device.post(f"{V1}/inspect").json()["verdict"] == "NO_MODEL"


def test_inspect_verdicts_follow_the_threshold(device):
    device.state.inspector.detector = FakeDetector(conf=0.6)
    device.put(f"{V1}/settings", json={"conf_threshold": 0.5})
    r = device.post(f"{V1}/inspect").json()
    assert r["verdict"] == "FAIL" and r["failing"] == 1

    device.put(f"{V1}/settings", json={"conf_threshold": 0.7})
    assert device.post(f"{V1}/inspect").json()["verdict"] == "PASS"
    assert device.put(f"{V1}/settings", json={"conf_threshold": 7}).status_code == 400

    assert device.get(f"{V1}/inspections/{r['id']}").json()["verdict"] == "FAIL"
    assert device.get(f"{V1}/inspections/{r['id']}/image.jpg?annotated=true").status_code == 200
    assert "FAIL" in device.get(f"{V1}/inspections.csv").text
    assert len(device.get(f"{V1}/inspections").json()) == 2
    assert device.get(f"{V1}/inspections/nope").status_code == 404
    assert device.get(f"{V1}/inspections/..%2F..%2Fetc/image.jpg").status_code == 404


def test_annotated_image_is_rendered_on_demand_when_not_stored(device_cfg):
    device_cfg.inference.save_annotated = False
    app = create_app(device_cfg)
    with TestClient(app) as device:
        app.state.boxqc.inspector.detector = FakeDetector(conf=0.9)
        rid = device.post(f"{V1}/inspect").json()["id"]
        stored = sorted(p.name for p in (device_cfg.data_dir / "inspections").iterdir())
        assert stored == [f"{rid}.jpg", f"{rid}.json"]
        raw = device.get(f"{V1}/inspections/{rid}/image.jpg").content
        annotated = device.get(f"{V1}/inspections/{rid}/image.jpg?annotated=true").content
        assert decode(annotated).shape == decode(raw).shape and annotated != raw


def test_roi_offsets_detections(device):
    device.state.inspector.detector = FakeDetector(conf=0.9)
    device.put(f"{V1}/settings", json={"roi": [0.5, 0.5, 1.0, 1.0]})
    r = device.post(f"{V1}/inspect").json()
    # detection at (10,10) inside the crop -> (330, 190) in the 640x360 frame
    assert r["detections"][0]["box"][:2] == [330.0, 190.0]


def test_model_upload_validate_activate_delete(device, device_cfg, tmp_path):
    assert device.get(f"{V1}/models?usage=true").json() == []
    r = device.post(f"{V1}/models", files={"file": ("x.onnx", b"not a model")})
    assert r.status_code == 400 and "Not a usable model" in r.json()["detail"]
    assert device.post(f"{V1}/models", files={"file": ("x.exe", b"")}).status_code == 400
    assert not list(device_cfg.models_dir.iterdir())

    onnx_bytes = make_onnx(tmp_path / "m.onnx", one_box_preds(0.9)).read_bytes()
    first = device.post(f"{V1}/models", files={
        "file": ("first.onnx", onnx_bytes), "meta": ("first.json", b'{"recommended_threshold": 0.35}')}).json()
    assert first == {"name": "first.onnx", "active": True}
    status = device.get(f"{V1}/status").json()
    assert status["model"] == "first.onnx" and status["model_info"]["classes"] == ["dirty", "defect"]
    assert device.post(f"{V1}/inspect").json()["verdict"] == "FAIL"     # the real model, end to end

    second = device.post(f"{V1}/models", files={"file": ("second.onnx", onnx_bytes)},
                         data={"activate": "false"}).json()
    assert second == {"name": "second.onnx", "active": False}
    listed = {m["name"]: m for m in device.get(f"{V1}/models?usage=true").json()}
    assert listed["first.onnx"]["active"] and listed["first.onnx"]["info"] == {"recommended_threshold": 0.35}
    assert listed["first.onnx"]["usage"]["FAIL"] == 1 and listed["second.onnx"]["usage"] is None

    assert device.delete(f"{V1}/models/first.onnx").status_code == 400      # active
    assert device.post(f"{V1}/models/second.onnx/activate").json() == {"active": "second.onnx"}
    assert device.delete(f"{V1}/models/first.onnx").json() == {"deleted": "first.onnx"}
    assert not (device_cfg.models_dir / "first.json").exists()
    assert device.post(f"{V1}/models/missing.onnx/activate").status_code == 404

    # a model that loads but whose output cannot be decoded is refused at upload
    bad = make_onnx(tmp_path / "bad.onnx", np.zeros((1, 9, 3), np.float32)).read_bytes()
    r = device.post(f"{V1}/models", files={"file": ("bad.onnx", bad)})
    assert r.status_code == 400 and "does not fit 2 classes" in r.json()["detail"]


def test_api_token_is_required_when_configured(device_cfg):
    device_cfg.device.api_token = "s3cret"
    with TestClient(create_app(device_cfg)) as device:
        assert device.get(f"{V1}/status").status_code == 401
        assert device.get("/api/status").status_code == 401
        assert device.get(f"{V1}/status", headers={"Authorization": "Bearer wrong"}).status_code == 401
        assert device.get(f"{V1}/status", headers={"Authorization": "Bearer s3cret"}).status_code == 200
        assert device.get(f"{V1}/status", headers={"X-API-Key": "s3cret"}).status_code == 200
        assert device.get("/").status_code == 200       # the landing page says nothing sensitive


def test_buzzer_test_reports_mock(device):
    assert device.post(f"{V1}/buzzer/test", json={"pattern": "fail"}).json() == {"played": "FAIL", "mock": True}


def test_model_usage_counts_and_median():
    records = [
        {"model": "a.onnx", "verdict": "PASS", "time": "2026-01-01T10:00:00", "timing_ms": {"inference": 100}},
        {"model": "a.onnx", "verdict": "FAIL", "time": "2026-01-02T10:00:00", "timing_ms": {"inference": 300}},
        {"model": "a.onnx", "verdict": "PASS", "time": "2026-01-01T12:00:00", "timing_ms": {"inference": 200}},
        {"model": None, "verdict": "NO_MODEL"},
    ]
    u = model_usage(records)["a.onnx"]
    assert (u["inspections"], u["PASS"], u["FAIL"]) == (3, 2, 1)
    assert u["inference_ms"] == 200 and u["last_used"] == "2026-01-02T10:00:00"


def test_activating_an_unusable_model_keeps_the_current_one(device, device_cfg, tmp_path):
    good = make_onnx(tmp_path / "m.onnx", one_box_preds(0.9)).read_bytes()
    device.post(f"{V1}/models", files={"file": ("good.onnx", good)})
    # a file that reached models/ some other way (scp, rsync) and cannot be loaded
    (device_cfg.models_dir / "broken.onnx").write_bytes(b"not a model")

    r = device.post(f"{V1}/models/broken.onnx/activate")
    assert r.status_code == 500 and "broken.onnx" in r.json()["detail"]
    status = device.get(f"{V1}/status").json()
    assert status["model"] == "good.onnx" and status["settings"]["active_model"] == "good.onnx"
    assert status["model_error"] is None
    assert device.post(f"{V1}/inspect").json()["verdict"] == "FAIL"     # still inspecting with it


def test_uploads_are_staged_and_listed_with_their_content_hash(device, device_cfg, tmp_path):
    onnx_bytes = make_onnx(tmp_path / "m.onnx", one_box_preds(0.9)).read_bytes()
    device.post(f"{V1}/models", files={"file": ("a.onnx", onnx_bytes)})
    (listed,) = device.get(f"{V1}/models").json()
    assert listed["sha256"] == hashlib.sha256(onnx_bytes).hexdigest()

    # a failed upload leaves nothing behind, and a sidecar left by an earlier model of
    # the same name does not become the new model's metadata
    device.post(f"{V1}/models", files={"file": ("junk.onnx", b"junk")})
    (device_cfg.models_dir / "b.json").write_text('{"recommended_threshold": 0.9}')
    device.post(f"{V1}/models", files={"file": ("b.onnx", onnx_bytes)}, data={"activate": "false"})
    assert sorted(p.name for p in device_cfg.models_dir.iterdir()) == ["a.onnx", "b.onnx"]

    # a reused name gets a new one; the first file is untouched
    other = make_onnx(tmp_path / "other.onnx", one_box_preds(0.3)).read_bytes()
    renamed = device.post(f"{V1}/models", files={"file": ("a.onnx", other)}).json()["name"]
    assert renamed != "a.onnx" and (device_cfg.models_dir / "a.onnx").read_bytes() == onnx_bytes
    hashes = {m["name"]: m["sha256"] for m in device.get(f"{V1}/models").json()}
    assert hashes[renamed] == hashlib.sha256(other).hexdigest() != hashes["a.onnx"]


def test_interrupted_uploads_are_cleaned_up_at_start(device_cfg):
    leftover = device_cfg.models_dir / ".upload-abc123"
    leftover.mkdir(parents=True)
    (leftover / "half.onnx").write_bytes(b"half a model")
    with TestClient(create_app(device_cfg)) as device:
        assert not leftover.exists()
        assert device.get(f"{V1}/models").json() == []


def test_inspection_ids_are_never_reused(device, monkeypatch):
    drawn = iter(["20260101-120000-aaaa", "20260101-120000-aaaa", "20260101-120000-bbbb"])
    monkeypatch.setattr(inspector_module, "new_id", lambda: next(drawn))
    ids = [device.post(f"{V1}/inspect").json()["id"] for _ in range(2)]
    assert ids == ["20260101-120000-aaaa", "20260101-120000-bbbb"]
    assert len(device.get(f"{V1}/inspections").json()) == 2
    assert len(device.get(f"{V1}/inspections?limit=-1").json()) == 0     # not "all but one"


def test_a_failed_upload_inspection_answers_with_a_json_error(device):
    class Broken:
        path = FakeDetector.path

        def detect(self, *args):
            raise RuntimeError("model exploded")

    device.state.inspector.detector = Broken()
    jpeg = cv2.imencode(".jpg", np.zeros((40, 60, 3), np.uint8))[1].tobytes()
    r = device.post(f"{V1}/inspect/upload", files={"file": ("x.jpg", jpeg)})
    assert r.status_code == 500 and "model exploded" in r.json()["detail"]


def test_auto_camera_is_the_mock_only_where_picamera2_is_not_installed(monkeypatch):
    """A station whose camera fails to open must report it, not inspect synthetic images."""
    cfg = CameraConfig(capture_size=[640, 360], preview_size=[320, 180])

    monkeypatch.setattr(camera_module, "find_spec", lambda name: None)
    assert isinstance(open_camera(cfg), MockCamera)                     # a development machine

    class NoCamera:
        def __init__(self):
            raise RuntimeError("no cameras available")

    fake = types.ModuleType("picamera2")
    fake.Picamera2 = NoCamera
    monkeypatch.setitem(sys.modules, "picamera2", fake)
    monkeypatch.setattr(camera_module, "find_spec", lambda name: object())   # the library is installed
    with pytest.raises(RuntimeError, match="no cameras available"):
        open_camera(cfg)                                                # a Pi with a loose cable
    cfg.backend = "mock"
    assert isinstance(open_camera(cfg), MockCamera)


def test_a_missing_camera_is_reported_and_inspections_are_refused(device_cfg, monkeypatch):
    def no_camera(cfg):
        raise RuntimeError("no cameras available")

    monkeypatch.setattr("boxqc.device.app.open_camera", no_camera)
    with TestClient(create_app(device_cfg)) as device:
        status = device.get(f"{V1}/status").json()
        assert status["camera"] is None and "no cameras available" in status["camera_error"]
        assert device.post(f"{V1}/inspect").status_code == 503
        assert device.get(f"{V1}/camera/snapshot.jpg").status_code == 503
