"""The labeling assistant: a trained model proposes boxes for an image being labeled.

Drawing boxes is the slow part of the loop, and once a first model exists it can do most
of that work. The assistant runs a model over a dataset image the way an inspection
would (same ROI crop, same :func:`boxqc.models.load_model`, so every model format works)
and returns what it finds as *suggestions*.

A suggestion is only a proposal. Nothing here reads or writes an annotation: the
labeling page draws the suggestions over the image, and the ones the user accepts are
saved like hand-drawn boxes (``PUT /api/images/{id}``). The rest are simply never
stored, which is what keeps a model's mistakes out of the next training round.

The models are the ones on the studio machine (``models_dir``), so the assistant works
with the device switched off. Before a first model has been trained there is nothing to
suggest with; the usual way in is to label a small batch by hand, train, and let that
model pre-box the remaining images.
"""

from __future__ import annotations

import os
import threading
from collections.abc import Sequence
from pathlib import Path

import cv2

from ..config import Config
from ..imaging import crop_roi
from ..models import ModelAdapter, is_model_file, list_models, load_model

#: Detections below this confidence are not worth showing even as a suggestion. The
#: labeling page filters further with its slider; this only bounds what is sent to it.
MIN_CONF = 0.05


class AssistantUnavailable(Exception):
    """There is no model to suggest with, or the chosen one cannot be loaded or run."""


class LabelAssistant:
    """Runs local models over dataset images. One instance serves the whole studio.

    The model used last stays loaded, so stepping through images costs one inference
    each rather than one model load each. Calls are serialised with a lock: the web
    server's worker threads share the instance.
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self._threads = min(8, os.cpu_count() or 4)
        self._lock = threading.Lock()
        self._loaded: tuple[tuple, ModelAdapter] | None = None     # (file identity, model)

    def models(self) -> list[dict]:
        """The models on this computer, newest first. The first one is the default."""
        models = list_models(self.cfg.models_dir)
        models.sort(key=lambda m: str(m["info"].get("trained_at") or m["modified"]), reverse=True)
        return [{
            "name": m["name"],
            "trained_at": m["info"].get("trained_at"),
            "dataset_name": m["info"].get("dataset_name"),
            "classes": m["info"].get("classes"),
        } for m in models]

    def _detector(self, name: str | None) -> ModelAdapter:
        """The model called ``name`` (default: the newest), loaded. Call with the lock held."""
        if name is None:
            available = self.models()
            if not available:
                raise AssistantUnavailable("No model on this computer yet. Train one first.")
            name = available[0]["name"]
        path = self.cfg.models_dir / Path(name).name
        if not is_model_file(path) or not path.is_file():
            raise KeyError(name)
        stat = path.stat()
        identity = (str(path), stat.st_mtime_ns, stat.st_size)      # a retrained file is loaded again
        if self._loaded is None or self._loaded[0] != identity:
            try:
                model = load_model(path, threads=self._threads, fallback_classes=self.cfg.classes)
            except Exception as exc:
                raise AssistantUnavailable(f"Cannot load {path.name}: {exc}") from exc
            self._loaded = (identity, model)
        return self._loaded[1]

    def suggest(self, image_path: Path, roi: Sequence[float], model: str | None = None) -> dict:
        """Run a model over one image and return its detections as suggested boxes.

        ``roi`` is the region the model is shown, as for an inspection. ``model`` is a
        file name in ``models_dir``; ``None`` picks the newest.

        Returns ``{"model", "roi", "inference_ms", "suggestions", "ignored_classes"}``.
        Each suggestion is ``{"cls", "conf", "x0", "y0", "x1", "y1"}`` with the corners
        as fractions of the full image, i.e. an annotation box plus a confidence, most
        confident first. Detections of a class that ``boxqc.yaml`` does not list cannot
        be saved as labels, so they are left out and their names reported in
        ``ignored_classes``.

        Raises ``KeyError`` for an unknown model name, ``ValueError`` for an unreadable
        image and :class:`AssistantUnavailable` when there is no usable model.
        """
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Cannot read {image_path.name}")
        height, width = image.shape[:2]
        crop, (ox, oy) = crop_roi(image, roi)
        with self._lock:
            detector = self._detector(model)
            try:
                detections, ms = detector.detect(crop, MIN_CONF, self.cfg.inference.iou_threshold)
            except Exception as exc:
                raise AssistantUnavailable(f"{detector.path.name} failed on this image: {exc}") from exc

        suggestions = []
        for det in detections:
            if det.name not in self.cfg.classes:
                continue
            x0, y0, x1, y1 = det.shifted(ox, oy).box
            suggestions.append({
                "cls": det.name,
                "conf": round(float(det.conf), 4),
                "x0": round(x0 / width, 5), "y0": round(y0 / height, 5),
                "x1": round(x1 / width, 5), "y1": round(y1 / height, 5),
            })
        return {
            "model": detector.path.name,
            "roi": list(roi),
            "inference_ms": round(ms, 1),
            "suggestions": suggestions,
            "ignored_classes": sorted({d.name for d in detections} - set(self.cfg.classes)),
        }
