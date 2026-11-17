"""The trainer contract: how a training backend plugs into the studio.

A *trainer* turns a prepared dataset into one model file. Everything around that step is
shared and lives in :mod:`boxqc.studio.training.pipeline`: building the dataset,
installing the model, the PASS/FAIL evaluation, the metadata sidecar and deployment.
So a new backend only implements :meth:`Trainer.train`. See ``docs/trainers.md``.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar

from ...config import Config
from ..export import PreparedDataset

STOP_MODES = ("early", "cancel")
_RUN_NAME_RE = re.compile(r"^[A-Za-z0-9._-]{1,64}$")


class TrainingCancelled(Exception):
    """Raised inside a run when the user cancels it."""


class TrainerUnavailable(RuntimeError):
    """The trainer's dependencies are not installed on this machine."""


@dataclass
class TrainRequest:
    """What to train. ``options`` is passed through untouched for trainer-specific settings."""

    epochs: int
    imgsz: int
    batch: int
    base_model: str
    device: str = "auto"
    trainer: str = "ultralytics"
    dataset: str | None = None      # named dataset; None = every labeled image
    name: str | None = None         # run / model name; None = boxqc-<date>-<time>
    options: dict = field(default_factory=dict)

    @classmethod
    def from_config(cls, cfg: Config, **overrides) -> TrainRequest:
        """The defaults from ``boxqc.yaml`` (``train:``), with any non-None overrides applied."""
        t = cfg.train
        values = {"epochs": t.epochs, "imgsz": t.imgsz, "batch": t.batch, "base_model": t.base_model,
                  "device": t.device, "trainer": t.trainer}
        values.update({k: v for k, v in overrides.items() if v is not None})
        return cls(**values)

    def validate(self) -> None:
        """Checks that hold for every trainer. Raises ``ValueError``."""
        if not 1 <= self.epochs <= 2000:
            raise ValueError("epochs must be 1-2000")
        if not 64 <= self.imgsz <= 4096:
            raise ValueError("image size must be between 64 and 4096")
        if self.batch != -1 and not 1 <= self.batch <= 1024:
            raise ValueError("batch must be -1 (auto) or 1-1024")
        if self.name and not _RUN_NAME_RE.match(self.name):
            raise ValueError("name may only use letters, digits, '.', '_' and '-'")
        if not isinstance(self.options, dict):
            raise ValueError("options must be an object")


class TrainContext:
    """A trainer's line back to whoever started the run: progress out, stop requests in.

    This base implementation prints to the console and never asks to stop; it is what
    ``boxqc train`` uses. The studio substitutes one that feeds the Training page.
    """

    def log(self, line: str, *, echo: bool = True) -> None:
        """Report a line of output. Pass ``echo=False`` for text that the trainer's own
        library has already printed to the console."""
        if echo:
            print(line, flush=True)

    def stage(self, name: str) -> None:
        """Announce the current stage: ``preparing``, ``training``, ``exporting``,
        ``evaluating`` or ``deploying``. Trainers call ``stage("exporting")`` when they
        start converting the trained weights."""

    def epoch(self, metrics: dict, total_epochs: int | None = None) -> None:
        """Report one finished epoch. ``metrics`` should contain ``epoch`` (1-based) and
        any of: ``mAP50``, ``mAP50-95``, ``precision``, ``recall``, ``box_loss``,
        ``cls_loss``, ``elapsed_s``. Unknown keys are kept but not charted."""

    @property
    def stop_requested(self) -> str | None:
        """``"early"`` (finish the epoch, keep the model), ``"cancel"`` (abort) or None."""
        return None

    def check_cancelled(self) -> None:
        """Raise :class:`TrainingCancelled` if the run was cancelled. Call it regularly."""
        if self.stop_requested == "cancel":
            raise TrainingCancelled()


@dataclass
class TrainerOutput:
    """What :meth:`Trainer.train` hands back."""

    model_path: Path                    # the exported model file, in a registered model format
    format: str = "yolo-onnx"           # name of that format (boxqc.models)
    metrics: dict = field(default_factory=dict)     # mAP50, mAP50-95, precision, recall
    info: dict = field(default_factory=dict)        # extra fields for the model's metadata sidecar


class Trainer(ABC):
    """Base class for training backends.

    Register a subclass with :func:`boxqc.studio.training.register_trainer` or the
    ``boxqc.trainers`` entry-point group. Class attributes describe it to the UI;
    nothing heavy (PyTorch, ...) may be imported at module import time - do that inside
    :meth:`availability`, :meth:`host_info` and :meth:`train`.
    """

    #: Identifier used in requests and model metadata.
    name: ClassVar[str]
    #: Shown in the Training page.
    label: ClassVar[str]
    #: ``(value, description)`` pairs offered as the base model. Other values may be
    #: typed in ``boxqc.yaml``; :meth:`validate` decides what is acceptable.
    base_models: ClassVar[tuple[tuple[str, str], ...]] = ()
    #: ``(value, description)`` pairs offered as the compute device.
    devices: ClassVar[tuple[tuple[str, str], ...]] = (("auto", "auto"),)

    def __init__(self, cfg: Config):
        self.cfg = cfg

    @classmethod
    def availability(cls) -> str | None:
        """None if this trainer can run here, else a sentence saying what is missing."""
        return None

    @classmethod
    def host_info(cls) -> dict:
        """Facts about this machine for the Training page, e.g. ``{"device": "mps", "gpu": "..."}``."""
        return {}

    def validate(self, request: TrainRequest) -> None:  # noqa: B027 - optional hook, not abstract
        """Trainer-specific request checks. Raise ``ValueError`` with a message for the user."""

    @abstractmethod
    def train(self, request: TrainRequest, dataset: PreparedDataset, ctx: TrainContext) -> TrainerOutput:
        """Train on ``dataset`` and export the best model.

        ``request.name`` is always set here and is unique enough to name a run directory
        under ``self.cfg.train.workdir``. Report progress through ``ctx``, honour
        ``ctx.stop_requested``, and let :class:`TrainingCancelled` propagate.
        """

    @classmethod
    def describe(cls) -> dict:
        """JSON description for the studio API."""
        reason = cls.availability()
        return {
            "name": cls.name,
            "label": cls.label,
            "available": reason is None,
            "unavailable_reason": reason,
            "base_models": [{"value": v, "label": text} for v, text in cls.base_models],
            "devices": [{"value": v, "label": text} for v, text in cls.devices],
        }
