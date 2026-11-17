"""The trainer plug-in point and the background job manager, without HTTP."""

import time

import numpy as np
import pytest
from helpers import FakeTrainer, make_onnx, one_box_preds

from boxqc.models import read_metadata
from boxqc.studio.dataset import Box, DatasetStore
from boxqc.studio.training import (
    TRAINERS,
    TrainContext,
    TrainerOutput,
    TrainerUnavailable,
    TrainRequest,
    get_trainer,
    register_trainer,
)
from boxqc.studio.training.jobs import TrainingManager
from boxqc.studio.training.pipeline import run_training

IMG = np.full((60, 80, 3), 128, np.uint8)


@pytest.fixture
def labeled(cfg):
    store = DatasetStore(cfg.data_dir)
    for i in range(4):
        store.add_image(IMG, status="labeled", boxes=[Box("dirty", 0.2, 0.2, 0.6, 0.6)] if i % 2 else [])
    return cfg


def request(**overrides) -> TrainRequest:
    return TrainRequest(**{"epochs": 2, "imgsz": 64, "batch": 4, "base_model": "tiny", "trainer": "fake", **overrides})


def wait_for(predicate, timeout=10.0):
    deadline = time.monotonic() + timeout
    while not predicate():
        assert time.monotonic() < deadline, "timed out"
        time.sleep(0.01)


def test_registry_and_request_defaults(cfg):
    assert get_trainer("fake") is FakeTrainer and "ultralytics" in TRAINERS
    with pytest.raises(KeyError, match="Unknown trainer 'nope'"):
        get_trainer("nope")
    cfg.train.epochs = 7
    req = TrainRequest.from_config(cfg, imgsz=320, name=None)
    assert (req.epochs, req.imgsz, req.trainer, req.name) == (7, 320, "fake", None)


def test_pipeline_writes_model_and_metadata(labeled):
    lines = []

    class Recorder(TrainContext):
        def log(self, line, *, echo=True):
            lines.append(line)

    path = run_training(labeled, request(name="m1"), Recorder(), roi=[0.0, 0.0, 1.0, 1.0])
    assert path == labeled.models_dir / "m1.onnx"
    info = read_metadata(path)
    assert info["metrics"] == {"mAP50": 0.8} and info["dataset_name"] is None
    assert info["dataset"]["train"]["images"] + info["dataset"]["val"]["images"] == 4
    assert info["evaluation"]["table"] and info["run_dir"].endswith("m1")
    assert any("Image-level PASS/FAIL" in line for line in lines)

    with pytest.raises(ValueError, match="Unknown trainer"):
        run_training(labeled, request(trainer="nope"))
    with pytest.raises(ValueError, match="epochs"):
        run_training(labeled, request(epochs=0))


def test_unavailable_trainer_is_reported_not_started(labeled):
    class Missing(FakeTrainer):
        name = "missing"

        @classmethod
        def availability(cls):
            return "needs the foo library"

    register_trainer(Missing)
    try:
        assert Missing.describe()["unavailable_reason"] == "needs the foo library"
        with pytest.raises(TrainerUnavailable, match="foo"):
            TrainingManager(labeled).start(request(trainer="missing"))
    finally:
        TRAINERS.unregister("missing")


def test_manager_runs_one_job_at_a_time_and_supports_stop(labeled):
    class Slow(FakeTrainer):
        name = "slow"

        def train(self, request, dataset, ctx):
            ctx.stage("training")
            epoch = 0
            while ctx.stop_requested is None:
                ctx.check_cancelled()
                time.sleep(0.01)
            ctx.check_cancelled()                       # "cancel" ends the run here
            epoch += 1                                  # "early" keeps what there is
            ctx.epoch({"epoch": epoch}, total_epochs=request.epochs)
            out = self.cfg.train.workdir / "slow"
            out.mkdir(parents=True, exist_ok=True)
            return TrainerOutput(make_onnx(out / "best.onnx", one_box_preds()))

    register_trainer(Slow)
    try:
        manager = TrainingManager(labeled)
        job = manager.start(request(trainer="slow", name="early"), deploy=False)
        assert manager.presence() == {"busy": True}
        with pytest.raises(RuntimeError, match="already in progress"):
            manager.start(request(trainer="slow"))
        wait_for(lambda: manager.get(job["id"])["state"] == "training")
        with pytest.raises(ValueError):
            manager.request_stop(job["id"], "nope")
        assert manager.request_stop(job["id"], "early")["stop_request"] == "early"
        manager.wait(10)
        done = manager.get(job["id"])
        assert done["state"] == "done" and done["stop_request"] is None and done["epoch"] == 1
        assert done["result"]["model"] == "early.onnx" and done["result"]["deployed"] is False
        with pytest.raises(RuntimeError, match="already done"):
            manager.request_stop(job["id"], "cancel")

        job = manager.start(request(trainer="slow", name="cancelled"), deploy=False)
        wait_for(lambda: manager.get(job["id"])["state"] == "training")
        manager.request_stop(job["id"], "cancel")
        manager.wait(10)
        assert manager.get(job["id"])["state"] == "cancelled"
        assert not (labeled.models_dir / "cancelled.onnx").exists()
        assert manager.overview()["active"] is None and len(manager.overview()["jobs"]) == 2
    finally:
        TRAINERS.unregister("slow")


def test_failed_run_reports_the_error(cfg):
    manager = TrainingManager(cfg)          # nothing labeled
    job = manager.start(request(), deploy=False)
    manager.wait(10)
    failed = manager.get(job["id"])
    assert failed["state"] == "failed" and "No labeled images" in failed["error"]
    assert any("EmptyDatasetError" in line for line in failed["log"])
