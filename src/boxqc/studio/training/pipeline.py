"""One training run, start to finish. Shared by ``boxqc train`` and the studio's Training page.

    prepare dataset -> Trainer.train (train + export) -> install model -> evaluate -> write metadata

Only the second step belongs to the trainer; the rest is the same for every backend.
The result is a model file in ``models_dir`` with a metadata sidecar next to it
(``docs/models.md``), ready to be deployed to a device.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import replace
from datetime import datetime
from pathlib import Path

from ... import __version__
from ...config import Config
from ...ids import now_iso
from ...models import metadata_path
from ..evaluate import evaluate_image_level
from ..export import prepare_dataset
from .base import TrainContext, TrainerUnavailable, TrainRequest


def _write_metadata(model_path: Path, info: dict) -> None:
    metadata_path(model_path).write_text(json.dumps(info, indent=2))


def run_training(cfg: Config, request: TrainRequest, ctx: TrainContext | None = None,
                 roi: list[float] | None = None) -> Path:
    """Train a model as described by ``request``. Returns the path of the installed model file.

    ``roi`` overrides the region of interest images are cropped to (default: the device's
    ROI as last seen by the studio, else ``boxqc.yaml``).

    Raises ``ValueError`` for a bad request (including an unknown trainer),
    :class:`TrainerUnavailable` if the trainer's libraries are missing,
    :class:`~boxqc.studio.export.EmptyDatasetError` if nothing is labeled, and
    :class:`~.base.TrainingCancelled` if ``ctx`` reports a cancel.
    """
    from . import get_trainer  # deferred: the package imports this module's siblings

    ctx = ctx or TrainContext()
    request.validate()
    try:
        trainer_cls = get_trainer(request.trainer)
    except KeyError as exc:
        raise ValueError(exc.args[0]) from None
    if reason := trainer_cls.availability():
        raise TrainerUnavailable(reason)
    trainer = trainer_cls(cfg)
    trainer.validate(request)
    request = replace(request, name=request.name or datetime.now().strftime("boxqc-%Y%m%d-%H%M%S"))

    ctx.stage("preparing")
    prepared = prepare_dataset(cfg, log=ctx.log, dataset=request.dataset, roi=roi)
    ctx.check_cancelled()

    ctx.stage("training")
    output = trainer.train(request, prepared, ctx)
    ctx.check_cancelled()

    cfg.models_dir.mkdir(parents=True, exist_ok=True)
    suffix = output.model_path.suffix
    dest = cfg.models_dir / f"{request.name}{suffix}"
    if dest.exists():   # a reused run name must not overwrite an earlier model
        dest = cfg.models_dir / f"{request.name}-{datetime.now().strftime('%Y%m%d-%H%M%S')}{suffix}"
    shutil.copy2(output.model_path, dest)
    info = {
        "format": output.format,
        "classes": list(cfg.classes),
        "imgsz": request.imgsz,
        "roi": prepared.roi,
    }
    _write_metadata(dest, info)     # the evaluation below loads the model through its metadata

    ctx.stage("evaluating")
    evaluation = evaluate_image_level(dest, prepared.root, cfg.classes, log=ctx.log)
    info.update({
        "trainer": request.trainer,
        "base_model": request.base_model,
        "epochs": request.epochs,
        "batch": request.batch,
        "trained_at": now_iso(),
        "dataset": prepared.counts,
        "dataset_name": prepared.dataset,     # None = every labeled image
        "metrics": output.metrics,
        "recommended_threshold": evaluation.get("recommended_threshold"),
        "evaluation": {k: evaluation[k] for k in ("at_recommended", "table", "val_images") if k in evaluation},
        "boxqc_version": __version__,
    })
    info.update(output.info)
    _write_metadata(dest, info)
    ctx.log(f"\nModel saved: {dest}")
    return dest
