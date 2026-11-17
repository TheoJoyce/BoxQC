"""Test doubles shared by the test modules."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from boxqc.models import Detection
from boxqc.studio.training import Trainer, TrainerOutput


def make_onnx(path, preds: np.ndarray, names: str | None = "{0: 'dirty', 1: 'defect'}") -> Path:
    """Write a tiny ONNX model that ignores its 64x64 input and always outputs ``preds``.

    It is shaped like a detector export, so the real adapter loads and decodes it.
    """
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    inp = helper.make_tensor_value_info("images", TensorProto.FLOAT, [1, 3, 64, 64])
    out = helper.make_tensor_value_info("output0", TensorProto.FLOAT, list(preds.shape))
    const = helper.make_node("Constant", [], ["c"], value=numpy_helper.from_array(preds.astype(np.float32)))
    # use the input so ORT doesn't prune it: out = c + 0 * sum(images)
    s = helper.make_node("ReduceSum", ["images"], ["s"], keepdims=0)
    zero = helper.make_node("Constant", [], ["z"], value=numpy_helper.from_array(np.float32(0)))
    mul = helper.make_node("Mul", ["s", "z"], ["m"])
    add = helper.make_node("Add", ["c", "m"], ["output0"])
    graph = helper.make_graph([const, s, zero, mul, add], "g", [inp], [out])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    props = {"imgsz": "[64, 64]"}
    if names:
        props["names"] = names
    helper.set_model_props(model, props)
    onnx.save(model, str(path))
    return Path(path)


def one_box_preds(conf: float = 0.9) -> np.ndarray:
    """(1, 4 + 2, 3) output with a single "defect" box centred at (32, 32), 16x16."""
    preds = np.zeros((1, 6, 3), np.float32)
    preds[0, :, 0] = [32, 32, 16, 16, 0.1, conf]
    return preds


class FakeDetector:
    """Stands in for a loaded model: always reports one "dirty" box with a fixed confidence."""

    path = Path("fake.onnx")

    def __init__(self, conf):
        self.conf = conf

    def detect(self, image, conf_threshold, iou_threshold):
        return [Detection(0, "dirty", self.conf, (10, 10, 50, 40))], 1.0


class FakeTrainer(Trainer):
    """A trainer that needs no ML libraries: reports two epochs and "exports" a canned model."""

    name = "fake"
    label = "Fake trainer"
    base_models = (("tiny", "tiny test model"),)

    def train(self, request, dataset, ctx):
        ctx.stage("training")
        for epoch in (1, 2):
            ctx.check_cancelled()
            ctx.epoch({"epoch": epoch, "mAP50": 0.4 * epoch, "note": "dropped"}, total_epochs=2)
        ctx.stage("exporting")
        out = self.cfg.train.workdir / "runs" / request.name
        out.mkdir(parents=True, exist_ok=True)
        return TrainerOutput(make_onnx(out / "best.onnx", one_box_preds()), metrics={"mAP50": 0.8},
                             info={"run_dir": str(out)})
