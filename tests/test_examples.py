"""The plug-ins under examples/ are documentation that must keep working."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

from boxqc.config import load_config
from boxqc.models import FORMATS, load_model
from boxqc.plugins import load_plugins
from boxqc.studio.dataset import Box, DatasetStore
from boxqc.studio.training import TRAINERS, TrainRequest
from boxqc.studio.training.pipeline import run_training

EXAMPLES = Path(__file__).parent.parent / "examples"
pytest.importorskip("onnx")


@pytest.fixture
def examples_on_path(monkeypatch):
    monkeypatch.syspath_prepend(str(EXAMPLES))
    yield
    for name in ("command_trainer", "boxes_scores_labels_format"):
        sys.modules.pop(name, None)
    TRAINERS.unregister("command")
    FORMATS.unregister("boxes-scores-labels")


def test_plugins_listed_in_the_config_are_imported(tmp_path, examples_on_path):
    (tmp_path / "boxqc.yaml").write_text(yaml.safe_dump({"plugins": ["command_trainer", "no_such_module"]}))
    cfg = load_config(tmp_path / "boxqc.yaml")
    assert load_plugins(cfg) == ["command_trainer"]         # the broken entry is skipped, not fatal
    assert "command" in TRAINERS


def test_boxes_scores_labels_format(tmp_path, examples_on_path):
    import boxes_scores_labels_format  # noqa: F401 - registers the format
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    # a model that always reports one box (x 8..24, y 16..48 of its 64x32 input) for label 1
    values = {"boxes": np.array([[8, 16, 24, 48]], np.float32), "scores": np.array([0.8], np.float32),
              "labels": np.array([1], np.int64)}
    nodes = [helper.make_node("Constant", [], [name], value=numpy_helper.from_array(v)) for name, v in values.items()]
    outputs = [helper.make_tensor_value_info(name, TensorProto.INT64 if name == "labels" else TensorProto.FLOAT,
                                             list(v.shape)) for name, v in values.items()]
    graph = helper.make_graph(nodes, "g", [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3, 64, 32])],
                              outputs)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    path = tmp_path / "det.onnx"
    onnx.save(model, str(path))
    path.with_suffix(".json").write_text(json.dumps({"format": "boxes-scores-labels", "classes": ["dirty", "defect"]}))

    detector = load_model(path)
    assert detector.format == "boxes-scores-labels" and detector.input_size == (64, 32)
    detections, _ = detector.detect(np.zeros((128, 64, 3), np.uint8), conf_threshold=0.5)   # twice the input size
    assert [(d.name, d.box) for d in detections] == [("defect", (16.0, 32.0, 48.0, 96.0))]
    assert detector.detect(np.zeros((128, 64, 3), np.uint8), conf_threshold=0.9)[0] == []


def test_command_trainer_runs_an_external_program(cfg, tmp_path, examples_on_path, monkeypatch):
    import command_trainer
    from helpers import make_onnx, one_box_preds

    assert command_trainer.parse_progress("epoch=3 mAP50=0.71") == {"epoch": 3, "mAP50": 0.71}
    assert command_trainer.parse_progress("Epoch 3 of 10") is None
    assert command_trainer.CommandTrainer.availability() is not None        # no command configured yet

    # the "external program": prints progress, then copies a ready-made model to {output}
    ready = make_onnx(tmp_path / "ready.onnx", one_box_preds())
    script = tmp_path / "train.py"
    script.write_text(
        "import shutil, sys\n"
        "data, out, epochs = sys.argv[1:4]\n"
        "assert open(data).read().startswith('path:')\n"
        "for e in range(1, int(epochs) + 1):\n"
        "    print(f'epoch={e} mAP50={e / 10}', flush=True)\n"
        f"shutil.copy({str(ready)!r}, out)\n")
    monkeypatch.setenv("BOXQC_TRAIN_COMMAND", f"{sys.executable} {script} {{data}} {{output}} {{epochs}}")

    store = DatasetStore(cfg.data_dir)
    for i in range(4):
        store.add_image(np.full((60, 80, 3), 128, np.uint8), status="labeled",
                        boxes=[Box("dirty", 0.2, 0.2, 0.6, 0.6)] if i % 2 else [])
    request = TrainRequest(epochs=2, imgsz=64, batch=4, base_model="none", trainer="command", name="ext")
    model = run_training(cfg, request)
    info = json.loads(model.with_suffix(".json").read_text())
    assert model.name == "ext.onnx" and info["trainer"] == "command" and "train.py" in info["command"]
    assert (cfg.train.workdir / "yolo_dataset" / "annotations" / "instances_train.json").is_file()
