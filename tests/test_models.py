"""Model formats: decoding, the built-in ONNX adapter and the format registry."""

import json

import numpy as np
import pytest
from helpers import make_onnx, one_box_preds

from boxqc.device.inspector import verdict_for
from boxqc.models import (
    FORMATS,
    Detection,
    ModelAdapter,
    ModelError,
    file_sha256,
    list_models,
    load_model,
    register_format,
    validate_model,
)
from boxqc.models.yolo_onnx import decode_output, letterbox

pytest.importorskip("onnx")


def test_letterbox_keeps_aspect_and_centres():
    img = np.zeros((100, 200, 3), np.uint8)
    out, scale, (px, py) = letterbox(img, (640, 640))
    assert out.shape == (640, 640, 3)
    assert scale == 3.2
    assert (px, py) == (0, 160)


CLASSIC = np.array([
    # cx,  cy,  w,  h, s0,  s1
    [100, 100, 40, 40, 0.9, 0.0],
    [102, 101, 40, 40, 0.8, 0.0],   # overlaps #0, same class -> suppressed
    [104, 100, 40, 40, 0.0, 0.7],   # overlaps #0, other class -> kept
    [300, 300, 20, 20, 0.02, 0.01],  # below threshold
], dtype=np.float32)


@pytest.mark.parametrize("num_classes", [None, 2])
def test_decode_classic_layout_with_nms(num_classes):
    out = CLASSIC.T[None]       # (1, 4 + nc, N)
    boxes, scores, cls = decode_output(out, conf_threshold=0.1, iou_threshold=0.5, num_classes=num_classes)
    assert len(boxes) == 2
    assert sorted(cls.tolist()) == [0, 1]
    i0 = int(np.argmax(scores))
    np.testing.assert_allclose(boxes[i0], [80, 80, 120, 120])


def test_decode_classic_layout_channels_last():
    three = np.concatenate([CLASSIC, np.zeros((4, 1), np.float32)], axis=1)     # a third class
    boxes, _, cls = decode_output(three[None], 0.1, 0.5, num_classes=3)         # (1, N, 4 + nc)
    assert len(boxes) == 2 and sorted(cls.tolist()) == [0, 1]
    # with two classes the shape is (N, 6), which reads as end2end unless the layout is named
    boxes, _, cls = decode_output(CLASSIC[None], 0.1, 0.5, num_classes=2, layout="yolo")
    assert len(boxes) == 2 and sorted(cls.tolist()) == [0, 1]


def test_decode_end2end_layout():
    out = np.zeros((1, 300, 6), np.float32)
    out[0, 0] = [10, 20, 50, 60, 0.95, 1]
    out[0, 1] = [0, 0, 5, 5, 0.01, 0]
    boxes, _, cls = decode_output(out, 0.25, 0.45)
    assert boxes.tolist() == [[10, 20, 50, 60]]
    assert cls.tolist() == [1]


def test_decode_objectness_layout():
    # YOLOv5 style (1, N, 5 + nc): score = objectness * class score
    out = np.array([[
        # cx, cy,  w,  h, obj, s0,  s1,  s2
        [50, 50, 20, 20, 0.9, 0.1, 0.8, 0.0],    # 0.72 for class 1
        [90, 90, 20, 20, 0.2, 0.9, 0.0, 0.0],    # 0.18: objectness drags it below the threshold
    ]], np.float32)
    boxes, scores, cls = decode_output(out, 0.25, 0.45, num_classes=3)
    assert cls.tolist() == [1]
    assert scores[0] == pytest.approx(0.72)
    np.testing.assert_allclose(boxes[0], [40, 40, 60, 60])
    # a single-class model of this kind has the shape of end2end and must be told apart
    one = np.array([[[50, 50, 20, 20, 0.9, 0.8]]], np.float32)
    boxes, scores, _ = decode_output(one, 0.25, 0.45, num_classes=1, layout="yolo-obj")
    assert scores[0] == pytest.approx(0.72)


def test_decode_rejects_output_that_does_not_fit_the_classes():
    with pytest.raises(ModelError, match="does not fit 5 classes"):
        decode_output(CLASSIC.T[None], 0.1, 0.5, num_classes=5)
    with pytest.raises(ModelError, match="Unknown output layout"):
        decode_output(CLASSIC.T[None], 0.1, 0.5, layout="nope")


def test_verdict():
    d = [Detection(0, "dirty", 0.4, (0, 0, 1, 1))]
    assert verdict_for(d, 0.5) == "PASS"
    assert verdict_for(d, 0.3) == "FAIL"
    assert verdict_for([], 0.5, model_loaded=False) == "NO_MODEL"


def test_onnx_model_maps_boxes_back_to_image(tmp_path):
    # one "defect" box centred at (32, 32), 16x16, in 64x64 model space
    path = make_onnx(tmp_path / "m.onnx", one_box_preds())
    det = load_model(path, threads=1)
    assert det.format == "yolo-onnx"
    assert det.names == ["dirty", "defect"]
    assert det.input_size == (64, 64)
    validate_model(det)

    image = np.zeros((64, 128, 3), np.uint8)  # 2:1 -> scale 0.5, pad_y 16
    dets, _ = det.detect(image, conf_threshold=0.5)
    assert len(dets) == 1
    d = dets[0]
    assert d.name == "defect"
    # model box x 24..40, y 24..40 -> image x 48..80, y (24-16)/0.5=16 .. 48
    np.testing.assert_allclose(d.box, (48, 16, 80, 48))


def test_class_names_fall_back_to_metadata_then_config(tmp_path):
    path = make_onnx(tmp_path / "m.onnx", one_box_preds(), names=None)
    assert load_model(path, fallback_classes=["x", "y"]).names == ["x", "y"]
    path.with_suffix(".json").write_text(json.dumps({"classes": ["a", "b"]}))
    assert load_model(path, fallback_classes=["x", "y"]).names == ["a", "b"]
    assert list_models(tmp_path)[0]["info"] == {"classes": ["a", "b"]}


def test_metadata_selects_a_registered_format(tmp_path):
    class Stub(ModelAdapter):
        format = "stub"
        suffixes = (".stub",)

        def __init__(self, path, *, threads=4, fallback_classes=None, info=None):
            self.path, self.info = path, info or {}
            self.names, self.input_size = ["thing"], (8, 8)

        def detect(self, image_bgr, conf_threshold=0.25, iou_threshold=0.45):
            return [Detection(0, "thing", 1.0, (0, 0, 4, 4))], 0.0

    register_format(Stub)
    try:
        by_suffix = tmp_path / "a.stub"
        by_suffix.write_bytes(b"")
        assert isinstance(load_model(by_suffix), Stub)                  # chosen by file extension

        by_meta = tmp_path / "b.onnx"
        by_meta.write_bytes(b"")
        by_meta.with_suffix(".json").write_text(json.dumps({"format": "stub"}))
        model = load_model(by_meta)                                     # chosen by metadata
        assert isinstance(model, Stub) and model.info == {"format": "stub"}
        assert {m["name"] for m in list_models(tmp_path)} == {"a.stub", "b.onnx"}
    finally:
        FORMATS.unregister("stub")

    with pytest.raises(ModelError, match="Unknown model format 'stub'"):
        load_model(by_meta)
    with pytest.raises(ModelError, match="No model format handles .stub"):
        load_model(by_suffix)


def test_file_hash_follows_the_content(tmp_path):
    import hashlib
    import os

    f = tmp_path / "m.onnx"
    f.write_bytes(b"first")
    assert file_sha256(f) == hashlib.sha256(b"first").hexdigest()
    f.write_bytes(b"other")                 # same size, different content
    os.utime(f, ns=(1, 1))                  # make sure the change is visible in the mtime
    assert file_sha256(f) == hashlib.sha256(b"other").hexdigest()
