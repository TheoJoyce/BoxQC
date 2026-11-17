"""The built-in trainer: Ultralytics YOLO, exported to ONNX.

Needs the ``train`` extra (``pip install "boxqc[train]"``), which pulls in PyTorch.
Nothing from it is imported until a method here is called, so the studio starts - and
labeling works - on machines without it.

Licensing note: Ultralytics and its pretrained weights are AGPL-3.0 (or a commercial
licence from Ultralytics). That applies to this backend and the models it produces,
not to the rest of BoxQC. See the README.
"""

from __future__ import annotations

import importlib.util
import logging
import platform
import re
import socket
import time
from pathlib import Path

from ..export import PreparedDataset
from .base import TrainContext, Trainer, TrainerOutput, TrainRequest

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
_BASE_MODEL_RE = re.compile(r"^[A-Za-z0-9._-]{1,64}\.pt$")
_METRIC_KEYS = {  # Ultralytics name -> the name the studio charts
    "train/box_loss": "box_loss",
    "train/cls_loss": "cls_loss",
    "train/dfl_loss": "dfl_loss",
    "val/box_loss": "val_box_loss",
    "val/cls_loss": "val_cls_loss",
    "metrics/precision(B)": "precision",
    "metrics/recall(B)": "recall",
    "metrics/mAP50(B)": "mAP50",
    "metrics/mAP50-95(B)": "mAP50-95",
}


def pick_device(requested: str) -> str:
    """Resolve ``auto`` to the best PyTorch device here: Apple GPU, then CUDA GPU 0, then CPU."""
    if requested != "auto":
        return requested
    import torch

    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "0"
    return "cpu"


class _LogForwarder(logging.Handler):
    """Copies Ultralytics' log output into the run's log."""

    def __init__(self, ctx: TrainContext):
        super().__init__(logging.INFO)
        self._ctx = ctx

    def emit(self, record: logging.LogRecord) -> None:
        try:
            for line in _ANSI_RE.sub("", record.getMessage()).splitlines():
                if line.strip():
                    self._ctx.log(line, echo=False)     # Ultralytics already printed it
        except Exception:  # never let logging break training
            pass


class UltralyticsTrainer(Trainer):
    name = "ultralytics"
    label = "Ultralytics YOLO"
    base_models = (
        ("yolo11n.pt", "yolo11n: fastest on the Pi"),
        ("yolo11s.pt", "yolo11s: more accurate, ~2.5x slower"),
        ("yolo11m.pt", "yolo11m: most accurate, slow on the Pi"),
    )
    devices = (
        ("auto", "auto"),
        ("mps", "mps (Apple GPU)"),
        ("0", "0 (NVIDIA GPU)"),
        ("cpu", "cpu"),
    )

    @classmethod
    def availability(cls) -> str | None:
        # find_spec does not import anything: this is called on every Training page refresh
        missing = [name for name in ("torch", "ultralytics") if importlib.util.find_spec(name) is None]
        if missing:
            return f'Not installed: {", ".join(missing)}. Run: pip install "boxqc[train]"'
        return None

    @classmethod
    def host_info(cls) -> dict:
        info: dict = {}
        try:
            import torch

            info["torch"] = torch.__version__
            info["device"] = pick_device("auto")
            if torch.cuda.is_available():
                info["gpu"] = torch.cuda.get_device_name(0)
            elif torch.backends.mps.is_available():
                info["gpu"] = f"Apple GPU (MPS) · {platform.processor() or platform.machine()}"
        except Exception as exc:
            info["error"] = f"PyTorch unavailable: {exc}"
        return info

    def validate(self, request: TrainRequest) -> None:
        if request.imgsz % 32 or not 256 <= request.imgsz <= 1920:
            raise ValueError("image size must be a multiple of 32 between 256 and 1920")
        if request.batch != -1 and not 1 <= request.batch <= 256:
            raise ValueError("batch must be -1 (auto) or 1-256")
        if not _BASE_MODEL_RE.match(request.base_model):
            raise ValueError("base model must be a .pt file name, e.g. yolo11n.pt")
        if request.device not in ("auto", "mps", "cpu") and not request.device.isdigit():
            raise ValueError("device must be auto, mps, cpu or a GPU index")

    def weights_path(self, name: str) -> Path:
        """Where the pretrained weights ``name`` are, or should be downloaded to.

        Looked up in ``<workdir>/weights`` and then the current directory; if absent,
        Ultralytics downloads its published weights to ``<workdir>/weights``.
        """
        cached = self.cfg.train.workdir / "weights" / name
        if cached.is_file():
            return cached
        if (Path.cwd() / name).is_file():
            return Path.cwd() / name
        cached.parent.mkdir(parents=True, exist_ok=True)
        return cached

    def train(self, request: TrainRequest, dataset: PreparedDataset, ctx: TrainContext) -> TrainerOutput:
        from ultralytics import YOLO

        tc = self.cfg.train
        device = pick_device(request.device)
        ctx.log(f"\nTraining {request.base_model} for {request.epochs} epochs at {request.imgsz}px "
                f"on device={device}")

        model = YOLO(str(self.weights_path(request.base_model)))
        for event, fn in self._callbacks(ctx).items():
            model.add_callback(event, fn)
        forwarder = _LogForwarder(ctx)
        ul_logger = logging.getLogger("ultralytics")
        ul_logger.addHandler(forwarder)
        # Top-down view of a box: flips and small rotations are all plausible viewpoints.
        # The HSV jitter stands in for lighting drift on the line.
        arguments = {
            "fliplr": 0.5, "flipud": 0.5, "degrees": 5.0,
            "hsv_h": 0.01, "hsv_s": 0.4, "hsv_v": 0.35,
            "mosaic": 0.5, "plots": True, "patience": tc.patience,
        }
        arguments.update(request.options)       # trainer-specific overrides from the request
        arguments.update(                       # ...but never of what the pipeline depends on
            data=str(dataset.yaml_path), epochs=request.epochs, imgsz=request.imgsz, batch=request.batch,
            device=device, project=str(tc.workdir / "runs"), name=request.name, exist_ok=False,
        )
        t_start = time.time()
        try:
            model.train(**arguments)
            train_seconds = time.time() - t_start
            trainer = model.trainer
            best_pt = Path(trainer.best)
            raw = getattr(trainer, "metrics", None) or {}
            metrics = {
                key: float(raw.get(f"metrics/{key}(B)", 0))
                for key in ("mAP50", "mAP50-95", "precision", "recall")
            }

            ctx.stage("exporting")
            ctx.log(f"\nExporting {best_pt} to ONNX")
            onnx_path = Path(YOLO(str(best_pt)).export(format="onnx", imgsz=request.imgsz,
                                                       simplify=True, dynamic=False))
        finally:
            ul_logger.removeHandler(forwarder)

        run_dir = best_pt.parent.parent
        ctx.log(f"Training plots / confusion matrix: {run_dir}")
        return TrainerOutput(
            model_path=onnx_path,
            format="yolo-onnx",
            metrics=metrics,
            info={
                "epochs_completed": trainer.epoch + 1,
                "best_epoch": getattr(getattr(trainer, "stopper", None), "best_epoch", None),
                "patience": tc.patience,
                "train_seconds": round(train_seconds),
                "trained_on": {"hostname": socket.gethostname(), "device": device},
                "run_dir": str(run_dir),
            },
        )

    @staticmethod
    def _callbacks(ctx: TrainContext) -> dict:
        """Ultralytics trainer callbacks that report progress to ``ctx`` and act on stop requests."""
        t0 = time.time()
        in_epoch = False

        def on_train_start(trainer):
            ctx.stage("training")

        def on_train_batch_end(trainer):
            ctx.check_cancelled()
            if ctx.stop_requested == "early" and not trainer.stop:
                trainer.stop = True
                ctx.log("Stopping early: finishing this epoch, then exporting the best model so far.")

        def on_train_epoch_end(trainer):
            nonlocal in_epoch
            in_epoch = True

        def on_fit_epoch_end(trainer):
            # Ultralytics also fires this after training for the final best.pt validation
            # (with epoch bumped by one); only real epochs go into the history.
            nonlocal in_epoch
            if not in_epoch:
                return
            in_epoch = False
            raw = {}
            if trainer.tloss is not None:
                raw.update(trainer.label_loss_items(trainer.tloss, prefix="train"))
            raw.update(trainer.metrics or {})
            metrics = {"epoch": trainer.epoch + 1, "elapsed_s": round(time.time() - t0, 1)}
            for key, short in _METRIC_KEYS.items():
                if key in raw:
                    metrics[short] = round(float(raw[key]), 5)
            ctx.epoch(metrics, total_epochs=int(trainer.epochs))

        return {"on_train_start": on_train_start, "on_train_batch_end": on_train_batch_end,
                "on_train_epoch_end": on_train_epoch_end, "on_fit_epoch_end": on_fit_epoch_end}
