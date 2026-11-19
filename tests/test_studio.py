"""The studio app, connected to an in-process device (and to one that is switched off)."""

import json
import re

import numpy as np
from helpers import FakeDetector, make_onnx, one_box_preds

from boxqc.models import list_models
from boxqc.studio.dataset import Box

PARAMS = {"epochs": 3, "imgsz": 640, "batch": 16, "base_model": "tiny", "device": "auto"}


def test_pages_render(studio):
    for path in ("/inspect", "/capture", "/dataset", "/system", "/training"):
        assert studio.get(path).status_code == 200
    for path in ("/favicon.ico", "/static/favicon.svg", "/static/apple-touch-icon.png"):
        assert studio.get(path).status_code == 200
    assert studio.get("/", follow_redirects=False).headers["location"] == "/inspect"


def test_pages_link_versioned_assets(studio):
    html = studio.get("/dataset").text
    urls = re.findall(r'(?:src|href)="(/static/[\w.-]+\.(?:js|css)[^"]*)"', html)
    assert {u.split("?")[0] for u in urls} >= {"/static/common.js", "/static/style.css", "/static/dataset.js"}
    assert all(re.search(r"\?v=\d+$", u) for u in urls), urls  # cache-busting version on every asset
    assert all(studio.get(u).status_code == 200 for u in urls)


def test_status_combines_studio_and_device(studio):
    s = studio.get("/api/status").json()
    assert s["device"]["camera"] == "mock" and s["device_error"] is None
    assert s["classes"] == ["dirty", "defect"] and s["dataset"]["total"] == 0


def test_capture_takes_the_photo_from_the_device_and_keeps_it_here(studio, cfg, device_cfg):
    ann = studio.post("/api/capture", json={"status": "unlabeled", "tag": "defective"}).json()
    assert (ann["width"], ann["height"]) == (640, 360)
    assert (cfg.data_dir / "images" / f"{ann['id']}.jpg").is_file()
    assert not (device_cfg.data_dir / "images").exists()        # nothing lands on the device

    assert studio.get(f"/api/images/{ann['id']}/thumb").status_code == 200
    r = studio.put(f"/api/images/{ann['id']}", json={
        "status": "labeled", "boxes": [{"cls": "dirty", "x0": 0.5, "y0": 0.5, "x1": 0.1, "y1": 0.2}]})
    assert r.status_code == 200
    box = r.json()["boxes"][0]
    assert (box["x0"], box["x1"]) == (0.1, 0.5)  # normalised order

    bad = studio.put(f"/api/images/{ann['id']}", json={"boxes": [{"cls": "nope", "x0": 0, "y0": 0, "x1": 1, "y1": 1}]})
    assert bad.status_code == 400

    manifest = studio.get("/api/dataset/manifest").json()
    assert [i["id"] for i in manifest["items"]] == [ann["id"]]
    assert studio.get("/api/images/../../etc/passwd/file").status_code == 404
    assert studio.get("/api/dataset/export.zip").headers["content-type"] == "application/zip"


def test_device_proxy_forwards_methods_bodies_and_errors(studio, device):
    assert studio.get("/api/device/status").json()["camera"] == "mock"
    assert studio.get("/api/device/camera/snapshot.jpg?full=true").headers["content-type"] == "image/jpeg"
    r = studio.put("/api/device/settings", json={"roi": [0.1, 0.1, 0.9, 0.9]})
    assert r.json()["roi"] == [0.1, 0.1, 0.9, 0.9]
    assert device.get("/api/v1/settings").json()["roi"] == [0.1, 0.1, 0.9, 0.9]
    bad = studio.put("/api/device/settings", json={"conf_threshold": 9})
    assert bad.status_code == 400 and "conf_threshold" in bad.json()["detail"]     # the device's own error
    assert studio.post("/api/device/models", files={"file": ("x.onnx", b"junk")}).status_code == 400
    assert studio.get("/api/device/nope").status_code == 404
    assert "id,time" in studio.get("/api/device/inspections.csv").text


def test_roi_for_labeling_follows_the_device(studio, cfg):
    assert studio.get("/api/settings").json()["roi"] == [0.0, 0.0, 1.0, 1.0]
    studio.put("/api/device/settings", json={"roi": [0.2, 0.2, 0.8, 0.8]})
    studio.app.state.link.status(fresh=True)        # what any page load does
    assert studio.get("/api/settings").json()["roi"] == [0.2, 0.2, 0.8, 0.8]
    assert json.loads((cfg.data_dir / "studio.json").read_text())["roi"] == [0.2, 0.2, 0.8, 0.8]


def test_inspection_to_dataset(studio, device):
    device.state.inspector.detector = FakeDetector(conf=0.6)
    r = studio.post("/api/device/inspect?buzz=false").json()
    assert r["verdict"] == "FAIL"
    ann = studio.post(f"/api/inspections/{r['id']}/to-dataset?min_conf=0.5").json()
    assert ann["status"] == "unlabeled" and ann["source"] == "inspection" and len(ann["boxes"]) == 1
    assert studio.post(f"/api/inspections/{r['id']}/to-dataset?min_conf=0.9").json()["boxes"] == []
    assert studio.post("/api/inspections/missing/to-dataset").status_code == 404
    assert studio.post("/api/inspections/bad id!/to-dataset").status_code == 404


def test_dataset_api_and_capture_target(studio):
    old = studio.post("/api/capture", json={"status": "labeled"}).json()
    assert old["datasets"] == []

    assert studio.post("/api/datasets", json={"name": "v2", "capture": True}).status_code == 200
    assert studio.post("/api/datasets", json={"name": "v2"}).status_code == 400
    new = studio.post("/api/capture", json={"status": "unlabeled"}).json()
    assert new["datasets"] == ["v2"]                     # captures join the capture dataset

    summary = studio.get("/api/datasets").json()
    assert summary["capture_dataset"] == "v2"
    assert summary["all"]["total"] == 2 and summary["unassigned"]["total"] == 1
    assert summary["datasets"][0]["stats"]["total"] == 1

    assert [i["id"] for i in studio.get("/api/images?dataset=v2").json()] == [new["id"]]
    assert [i["id"] for i in studio.get("/api/images?dataset=__none__").json()] == [old["id"]]
    nav = studio.get(f"/api/images/{new['id']}?dataset=v2").json()["nav"]
    assert nav["prev"] is None and nav["next"] is None

    r = studio.post("/api/datasets/v2/items", json={"add": [old["id"]]})
    assert r.json() == {"changed": 1}
    assert studio.post("/api/datasets/missing/items", json={"add": [old["id"]]}).status_code == 404

    assert studio.patch("/api/datasets/v2", json={"name": "round 2"}).json()["name"] == "round 2"
    assert studio.get("/api/datasets").json()["capture_dataset"] == "round 2"  # follows the rename

    assert studio.put("/api/datasets/capture", json={"name": None}).json() == {"capture_dataset": None}
    assert studio.put("/api/datasets/capture", json={"name": "missing"}).status_code == 404

    assert studio.delete("/api/datasets/round%202").json()["images_released"] == 2
    assert studio.get("/api/datasets").json()["datasets"] == []
    assert len(studio.get("/api/images").json()) == 2


def test_upload_joins_the_viewed_dataset(studio):
    import cv2

    png = cv2.imencode(".png", np.zeros((40, 60, 3), np.uint8))[1].tobytes()
    studio.post("/api/datasets", json={"name": "v2"})
    r = studio.post("/api/images/upload", files=[("files", ("a.png", png)), ("files", ("b.txt", b"nope"))],
                    data={"dataset": "v2"})
    assert len(r.json()["added"]) == 1
    assert studio.get("/api/images?dataset=v2").json()[0]["source"] == "upload"
    assert studio.post("/api/images/upload", files=[("files", ("a.png", png))],
                       data={"dataset": "missing"}).status_code == 404


def test_models_are_merged_and_deployed(studio, device, cfg, tmp_path):
    cfg.models_dir.mkdir(parents=True)
    local = make_onnx(cfg.models_dir / "trained-here.onnx", one_box_preds())
    local.with_suffix(".json").write_text(json.dumps({"recommended_threshold": 0.4}))
    device.post("/api/v1/models", files={
        "file": ("on-device.onnx", make_onnx(tmp_path / "d.onnx", one_box_preds()).read_bytes())})

    listing = studio.get("/api/models").json()
    assert listing["device_online"] is True
    by_name = {m["name"]: m for m in listing["models"]}
    assert (by_name["trained-here.onnx"]["local"], by_name["trained-here.onnx"]["on_device"]) == (True, False)
    assert (by_name["on-device.onnx"]["local"], by_name["on-device.onnx"]["active"]) == (False, True)

    r = studio.post("/api/models/trained-here.onnx/deploy", json={"activate": True}).json()
    assert r == {"name": "trained-here.onnx", "active": True, "already_on_device": False}
    again = studio.post("/api/models/trained-here.onnx/deploy", json={"activate": True}).json()
    assert again["already_on_device"] is True
    merged = {m["name"]: m for m in studio.get("/api/models").json()["models"]}["trained-here.onnx"]
    assert merged["local"] and merged["on_device"] and merged["active"]
    assert merged["info"] == {"recommended_threshold": 0.4}     # the metadata went with it
    assert len(device.get("/api/v1/models").json()) == 2        # no duplicate from the second deploy

    assert studio.delete("/api/models/trained-here.onnx").json() == {"deleted": "trained-here.onnx"}
    assert list_models(cfg.models_dir) == []
    assert studio.post("/api/models/trained-here.onnx/deploy", json={}).status_code == 404


def test_training_rejects_bad_requests(studio):
    for bad in ({"epochs": 0}, {"batch": 0}, {"name": "a b"}, {"trainer": "nope"}, {"imgsz": 8}):
        assert studio.post("/api/training/jobs", json={**PARAMS, **bad}).status_code == 400, bad
    yolo = {**PARAMS, "trainer": "ultralytics", "base_model": "yolo11n.pt"}
    for bad in ({"imgsz": 650}, {"base_model": "../x.pt"}, {"device": "gpu"}):
        r = studio.post("/api/training/jobs", json={**yolo, **bad})
        assert r.status_code == 400, bad

    assert studio.post("/api/training/jobs", json={**PARAMS, "dataset": "missing"}).status_code == 400
    studio.post("/api/datasets", json={"name": "empty"})
    r = studio.post("/api/training/jobs", json={**PARAMS, "dataset": "empty"})
    assert r.status_code == 400 and "no labeled images" in r.json()["detail"]
    assert studio.get("/api/training/jobs/missing").status_code == 404


def label_some_images(studio, n=4):
    for i in range(n):
        ann = studio.post("/api/capture", json={"status": "labeled"}).json()
        if i % 2:
            studio.put(f"/api/images/{ann['id']}", json={
                "boxes": [{"cls": "dirty", "x0": 0.3, "y0": 0.3, "x1": 0.6, "y1": 0.6}]})


def test_training_run_trains_evaluates_and_deploys(studio, device, cfg):
    overview = studio.get("/api/training").json()
    assert overview["default_trainer"] == "fake" and overview["defaults"]["epochs"] == 150
    fake = next(t for t in overview["trainers"] if t["name"] == "fake")
    assert fake["available"] and fake["base_models"] == [{"value": "tiny", "label": "tiny test model"}]
    assert {t["name"] for t in overview["trainers"]} >= {"fake", "ultralytics"}

    label_some_images(studio)
    studio.put("/api/device/settings", json={"roi": [0.1, 0.1, 0.9, 0.9]})
    job = studio.post("/api/training/jobs", json={**PARAMS, "name": "run1"}).json()
    assert job["params"]["deploy"] is True and job["params"]["trainer"] == "fake"
    studio.training.wait(30)

    job = studio.get(f"/api/training/jobs/{job['id']}").json()
    assert job["state"] == "done", job["log"]
    assert job["history"] == [{"epoch": 1, "mAP50": 0.4}, {"epoch": 2, "mAP50": 0.8}]
    assert job["epochs"] == 2
    result = job["result"]
    assert result["model"] == "run1.onnx" and result["deployed"] and result["active"]
    assert result["evaluation"]["val_images"] >= 1 and result["recommended_threshold"] is not None

    info = json.loads((cfg.models_dir / "run1.json").read_text())
    assert info["format"] == "yolo-onnx" and info["trainer"] == "fake" and info["classes"] == ["dirty", "defect"]
    assert info["roi"] == [0.1, 0.1, 0.9, 0.9]                  # the device's ROI at training time
    assert device.get("/api/v1/status").json()["model"] == "run1.onnx"
    assert studio.get("/api/training").json()["jobs"][0]["model"] == "run1.onnx"

    # a second run may reuse the name without overwriting the first model
    studio.post("/api/training/jobs", json={**PARAMS, "name": "run1", "deploy": False})
    studio.training.wait(30)
    models = [m["name"] for m in list_models(cfg.models_dir)]
    assert len(models) == 2 and "run1.onnx" in models
    assert len(device.get("/api/v1/models").json()) == 1        # deploy: false stayed local


def test_studio_keeps_working_with_the_device_off(offline_studio, cfg):
    s = offline_studio.get("/api/status").json()
    assert s["device"] is None and "Cannot reach the device" in s["device_error"]
    r = offline_studio.post("/api/capture", json={})
    assert r.status_code == 502 and "Cannot reach the device" in r.json()["detail"]
    assert offline_studio.get("/api/device/status").status_code == 502
    assert offline_studio.get("/api/models").json() == {
        "models": [], "device_online": False, "device_error": s["device_error"]}

    # labeling and training do not need the device
    import cv2

    png = cv2.imencode(".png", np.full((60, 80, 3), 90, np.uint8))[1].tobytes()
    files = [("files", (f"{i}.png", png)) for i in range(4)]
    ids = offline_studio.post("/api/images/upload", files=files, data={"status": "labeled"}).json()["added"]
    assert len(ids) == 4
    assert offline_studio.get("/api/settings").json()["roi"] == [0.0, 0.0, 1.0, 1.0]
    offline_studio.post("/api/training/jobs", json={**PARAMS, "name": "offline"})
    offline_studio.training.wait(30)
    job = offline_studio.get("/api/training").json()["jobs"][0]
    assert job["state"] == "done"
    result = offline_studio.get(f"/api/training/jobs/{job['id']}").json()["result"]
    assert result["deployed"] is False and "Cannot reach the device" in result["deploy_error"]
    assert (cfg.models_dir / "offline.onnx").is_file()          # kept here, to deploy later


def test_a_reused_model_name_does_not_pass_for_the_model_on_the_device(studio, device, cfg, device_cfg, tmp_path):
    """Same file name, different content: the device's copy is an older model."""
    old = make_onnx(tmp_path / "old.onnx", one_box_preds(0.9)).read_bytes()
    device.post("/api/v1/models", files={"file": ("line2.onnx", old)})
    cfg.models_dir.mkdir(parents=True)
    new = make_onnx(cfg.models_dir / "line2.onnx", one_box_preds(0.3))     # retrained under the same name

    (listed,) = studio.get("/api/models").json()["models"]
    assert listed["on_device"] and listed["device_copy_differs"]

    r = studio.post("/api/models/line2.onnx/deploy", json={"activate": True}).json()
    assert r["already_on_device"] is False and r["active"] is True and r["name"] != "line2.onnx"
    assert (device_cfg.models_dir / r["name"]).read_bytes() == new.read_bytes()
    assert (device_cfg.models_dir / "line2.onnx").read_bytes() == old       # the old one is kept
    assert device.get("/api/v1/status").json()["model"] == r["name"]


def test_cross_site_requests_that_change_something_are_refused(studio, device):
    """A web page in the same browser must not be able to drive the studio or, through it, the device."""
    evil = {"Origin": "https://evil.example"}
    r = studio.post("/api/device/inspect", headers=evil)
    assert r.status_code == 403 and r.json() == {"detail": "Cross-site request refused"}
    assert studio.post("/api/device/models", headers=evil, files={"file": ("m.onnx", b"x")}).status_code == 403
    sandboxed = {"Origin": "null"}
    assert studio.post("/api/images/upload", headers=sandboxed, files={"files": ("a.jpg", b"x")}).status_code == 403
    assert studio.delete("/api/datasets/x", headers=evil).status_code == 403
    assert device.get("/api/v1/inspections").json() == []                   # nothing reached the device

    # the studio's own pages, scripts without an Origin header, and reads are unaffected
    assert studio.post("/api/device/inspect", headers={"Origin": "http://testserver"}).status_code == 200
    assert studio.post("/api/device/inspect").status_code == 200
    assert studio.get("/api/status", headers=evil).status_code == 200
    behind_proxy = {"Origin": "https://studio.example", "X-Forwarded-Host": "studio.example"}
    assert studio.post("/api/device/inspect", headers=behind_proxy).status_code == 200


def test_an_image_labeled_with_a_class_that_is_gone_can_still_be_saved(studio):
    store = studio.app.state.dataset
    ann = store.add_image(np.full((60, 80, 3), 128, np.uint8), boxes=[Box("scratch", 0.1, 0.1, 0.5, 0.5)])
    stale = {"cls": "scratch", "x0": 0.1, "y0": 0.1, "x1": 0.5, "y1": 0.5}

    # skipping it, or keeping the old box while adding a valid one, works...
    assert studio.put(f"/api/images/{ann.id}", json={"boxes": [stale], "status": "skip"}).status_code == 200
    fresh = {"cls": "dirty", "x0": 0.6, "y0": 0.6, "x1": 0.9, "y1": 0.9}
    assert studio.put(f"/api/images/{ann.id}", json={"boxes": [stale, fresh]}).status_code == 200
    # ...but a class that neither boxqc.yaml nor the image knows is still refused
    typo = {**fresh, "cls": "dirtty"}
    assert studio.put(f"/api/images/{ann.id}", json={"boxes": [typo]}).status_code == 400
    # once the stale box is gone it cannot come back
    assert studio.put(f"/api/images/{ann.id}", json={"boxes": [fresh]}).status_code == 200
    assert studio.put(f"/api/images/{ann.id}", json={"boxes": [stale]}).status_code == 400


def test_status_is_fresh_after_a_change_made_through_the_proxy(studio, device, tmp_path):
    onnx_bytes = make_onnx(tmp_path / "m.onnx", one_box_preds(0.9)).read_bytes()
    for name in ("a.onnx", "b.onnx"):
        device.post("/api/v1/models", files={"file": (name, onnx_bytes)})
    assert studio.get("/api/status").json()["device"]["model"] == "b.onnx"   # now remembered for 2 s
    studio.post("/api/device/models/a.onnx/activate")
    assert studio.get("/api/status").json()["device"]["model"] == "a.onnx"


def _blank_image(client, width=640, height=360) -> str:
    """Add an unlabeled grey image to the studio's dataset without going through the device."""
    return client.app.state.dataset.add_image(np.full((height, width, 3), 90, np.uint8)).id


def test_the_labeling_assistant_suggests_boxes_and_stores_nothing(offline_studio, cfg):
    """A local model proposes boxes; the device is not involved and the annotation is not touched."""
    item = _blank_image(offline_studio)
    assert offline_studio.get("/api/assistant").json() == {"models": [], "default": None}
    none = offline_studio.get(f"/api/images/{item}/suggestions")
    assert none.status_code == 409 and "No model" in none.json()["detail"]

    cfg.models_dir.mkdir(parents=True, exist_ok=True)
    make_onnx(cfg.models_dir / "first.onnx", one_box_preds(0.9))
    overview = offline_studio.get("/api/assistant").json()
    assert overview["default"] == "first.onnx" and [m["name"] for m in overview["models"]] == ["first.onnx"]

    run = offline_studio.get(f"/api/images/{item}/suggestions").json()
    assert run["model"] == "first.onnx" and run["ignored_classes"] == []
    # The model's 16x16 box at the centre of its 64x64 input, mapped back through the
    # letterbox onto the 640x360 image and given as fractions of it, like a label.
    (box,) = run["suggestions"]
    assert box["cls"] == "defect" and box["conf"] == 0.9
    assert [box[k] for k in ("x0", "y0", "x1", "y1")] == [0.375, 0.27778, 0.625, 0.72222]

    ann = offline_studio.get(f"/api/images/{item}").json()
    assert ann["status"] == "unlabeled" and ann["boxes"] == []      # a suggestion is not a label
    # ...and what the page sends back when the user accepts it is saved as an ordinary box
    saved = offline_studio.put(f"/api/images/{item}", json={"status": "labeled", "boxes": [box]}).json()
    assert saved["boxes"] == [{k: v for k, v in box.items() if k != "conf"}]

    html = offline_studio.get(f"/annotate/{item}").text
    assert all(f'id="{i}"' in html for i in ("assist-on", "assist-conf", "assist-add-all", "assist-list",
                                             "assist-model", "assist-run", "assist-status"))


def test_the_labeling_assistant_picks_models_and_reports_what_it_cannot_use(offline_studio, cfg):
    item = _blank_image(offline_studio)
    cfg.models_dir.mkdir(parents=True, exist_ok=True)
    make_onnx(cfg.models_dir / "sure.onnx", one_box_preds(0.9))
    faint = one_box_preds(0.02)
    faint[0, 4, 0] = 0.01                                           # both class scores under the floor
    make_onnx(cfg.models_dir / "faint.onnx", faint)
    make_onnx(cfg.models_dir / "other.onnx", one_box_preds(0.8), names="{0: 'rust', 1: 'scratch'}")
    (cfg.models_dir / "broken.onnx").write_bytes(b"not a model")

    def suggest(model):
        return offline_studio.get(f"/api/images/{item}/suggestions", params={"model": model})

    assert suggest("sure.onnx").json()["suggestions"][0]["conf"] == 0.9
    assert suggest("faint.onnx").json()["suggestions"] == []        # below the floor: not worth showing
    other = suggest("other.onnx").json()                            # a class boxqc.yaml does not list
    assert other["suggestions"] == [] and other["ignored_classes"] == ["scratch"]
    assert suggest("nope.onnx").status_code == 404
    assert suggest("../boxqc.yaml").status_code == 404
    broken = suggest("broken.onnx")
    assert broken.status_code == 409 and "Cannot load broken.onnx" in broken.json()["detail"]
    assert offline_studio.get("/api/images/missing/suggestions").status_code == 404

    # a model retrained under the same name is loaded again, not served from memory
    make_onnx(cfg.models_dir / "sure.onnx", one_box_preds(0.6))
    assert suggest("sure.onnx").json()["suggestions"][0]["conf"] == 0.6


def test_the_labeling_assistant_shows_the_model_the_roi_like_an_inspection(studio, cfg):
    item = studio.post("/api/capture", json={}).json()["id"]
    cfg.models_dir.mkdir(parents=True, exist_ok=True)
    make_onnx(cfg.models_dir / "m.onnx", one_box_preds(0.9))
    studio.put("/api/device/settings", json={"roi": [0.5, 0.0, 1.0, 1.0]})
    studio.app.state.link.status(fresh=True)
    run = studio.get(f"/api/images/{item}/suggestions").json()
    (box,) = run["suggestions"]
    assert run["roi"] == [0.5, 0.0, 1.0, 1.0]
    assert 0.5 < box["x0"] < box["x1"] < 1.0                        # found in the crop, placed in the full image
