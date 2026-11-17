"""Inspection pipeline: ROI crop -> detect -> PASS/FAIL verdict, plus an on-disk log."""

from __future__ import annotations

import json
import logging
import threading
from pathlib import Path

import numpy as np

from ..config import Config
from ..ids import check_id, new_id, now_iso
from ..imaging import crop_roi, decode_image, draw_detections, encode_jpeg
from ..models import Detection, ModelAdapter, list_models, load_model
from .settings import RuntimeSettings

log = logging.getLogger(__name__)

VERDICTS = ("PASS", "FAIL", "NO_MODEL")


def verdict_for(detections: list[Detection], threshold: float, model_loaded: bool = True) -> str:
    """The rule of the whole system: any detection at or above the threshold fails the box."""
    if not model_loaded:
        return "NO_MODEL"
    return "FAIL" if any(d.conf >= threshold for d in detections) else "PASS"


class Inspector:
    """Holds the active model and turns a camera frame into an inspection result."""

    def __init__(self, cfg: Config, settings: RuntimeSettings):
        self.cfg = cfg
        self.settings = settings
        self._lock = threading.Lock()
        self.detector: ModelAdapter | None = None
        self.load_error: str | None = None
        self.reload()

    @property
    def model_name(self) -> str | None:
        return self.detector.path.name if self.detector else None

    def load(self, path: Path) -> ModelAdapter:
        """Load a model file the way inspections run it. Raises if it cannot be loaded.

        Nothing changes until the result is handed to :meth:`use`, so a caller can check
        a model first and keep the current one active if the new one is unusable.
        """
        return load_model(path, threads=self.cfg.inference.threads, fallback_classes=self.cfg.classes)

    def use(self, detector: ModelAdapter) -> None:
        """Make an already loaded model the active one."""
        with self._lock:
            self.detector, self.load_error = detector, None
        log.info("Using model %s (format %s, input %s, classes %s)", detector.path.name,
                 detector.format, detector.input_size, detector.names)

    def reload(self) -> None:
        """(Re)load the active model; falls back to the newest model in models_dir."""
        self.cfg.models_dir.mkdir(parents=True, exist_ok=True)
        name = self.settings.get("active_model")
        path = self.cfg.models_dir / name if name else None
        if path is None or not path.is_file():
            models = list_models(self.cfg.models_dir)
            path = self.cfg.models_dir / models[0]["name"] if models else None
        detector, error = None, None
        if path is not None:
            try:
                detector = self.load(path)
                log.info("Loaded model %s (format %s, input %s, classes %s)", path.name,
                         detector.format, detector.input_size, detector.names)
            except Exception as exc:
                error = f"Failed to load {path.name}: {exc}"
                log.exception(error)
        with self._lock:
            self.detector, self.load_error = detector, error

    def run(self, image: np.ndarray, item_id: str | None = None) -> dict:
        """Inspect a full frame. Returns a JSON-serialisable result dict.

        ``item_id`` is the id to give the result; the device passes one that is known to
        be free in its inspection log (:meth:`InspectionLog.new_id`).
        """
        threshold = float(self.settings.get("conf_threshold"))
        roi = list(self.settings.get("roi"))
        h, w = image.shape[:2]
        crop, (ox, oy) = crop_roi(image, roi)

        with self._lock:
            detector = self.detector
            detections: list[Detection] = []
            infer_ms = 0.0
            if detector is not None:
                raw, infer_ms = detector.detect(
                    crop, self.cfg.inference.min_report_conf, self.cfg.inference.iou_threshold
                )
                detections = [d.shifted(ox, oy) for d in raw]

        verdict = verdict_for(detections, threshold, detector is not None)
        return {
            "id": item_id or new_id(),
            "time": now_iso(),
            "verdict": verdict,
            "threshold": threshold,
            "model": detector.path.name if detector else None,
            "image_size": [w, h],
            "roi": roi,
            "detections": [d.to_dict() for d in detections],
            "failing": sum(d.conf >= threshold for d in detections),
            "timing_ms": {"inference": round(infer_ms, 1)},
        }

    @staticmethod
    def render(image: np.ndarray, result: dict, threshold: float | None = None) -> np.ndarray:
        """Draw a result's boxes and PASS/FAIL banner onto its image."""
        dets = [Detection.from_dict(d) for d in result["detections"]]
        thr = result["threshold"] if threshold is None else threshold
        return draw_detections(image, dets, thr, result["verdict"], result.get("roi"))


def model_usage(records: list[dict]) -> dict[str, dict]:
    """Per-model counts and median inference time from stored inspection records."""
    usage: dict[str, dict] = {}
    for r in records:
        name = r.get("model")
        if not name:
            continue
        u = usage.setdefault(name, {"inspections": 0, "PASS": 0, "FAIL": 0, "last_used": None, "_ms": []})
        u["inspections"] += 1
        if r.get("verdict") in ("PASS", "FAIL"):
            u[r["verdict"]] += 1
        u["last_used"] = max(u["last_used"] or "", r.get("time") or "") or None
        ms = (r.get("timing_ms") or {}).get("inference")
        if isinstance(ms, (int, float)):
            u["_ms"].append(ms)
    for u in usage.values():
        ms = sorted(u.pop("_ms"))
        u["inference_ms"] = round(ms[len(ms) // 2], 1) if ms else None  # median
    return usage


class InspectionLog:
    """Stores each inspection as ``<id>.json`` + ``<id>.jpg`` (raw frame), and optionally
    ``<id>_annotated.jpg``. The oldest records beyond ``keep`` are deleted."""

    def __init__(self, root: Path, keep: int = 1000, save_annotated: bool = True):
        self.root = root
        self.keep = keep
        self.save_annotated = save_annotated
        root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def path(self, item_id: str, suffix: str) -> Path:
        return self.root / f"{check_id(item_id)}{suffix}"

    def new_id(self) -> str:
        """A fresh id that no stored record uses.

        Ids are a timestamp to the second plus four random hex digits, so two inspections
        in the same second can draw the same one; a taken id is drawn again.
        """
        while True:
            item_id = new_id()
            if not self.path(item_id, ".json").exists() and not self.path(item_id, ".jpg").exists():
                return item_id

    def save(self, result: dict, raw: np.ndarray) -> None:
        """Store a result and its frame. The record is written last and atomically, so a
        record that can be listed always has its image and is never half a file."""
        rid = result["id"]
        self.path(rid, ".jpg").write_bytes(encode_jpeg(raw, 92))
        if self.save_annotated:
            self.path(rid, "_annotated.jpg").write_bytes(encode_jpeg(Inspector.render(raw, result), 85))
        tmp = self.path(rid, ".json.tmp")       # not matched by the "*.json" globs below
        tmp.write_text(json.dumps(result))
        tmp.replace(self.path(rid, ".json"))
        self._prune()

    def _prune(self) -> None:
        with self._lock:
            records = sorted(self.root.glob("*.json"))
            for old in records[: max(0, len(records) - self.keep)]:
                for suffix in (".json", ".jpg", "_annotated.jpg"):
                    (self.root / f"{old.stem}{suffix}").unlink(missing_ok=True)

    def get(self, item_id: str) -> dict:
        p = self.path(item_id, ".json")
        if not p.is_file():
            raise KeyError(item_id)
        return json.loads(p.read_text())

    def list(self, limit: int = 50) -> list[dict]:
        """The newest ``limit`` records, newest first."""
        out = []
        for p in sorted(self.root.glob("*.json"), reverse=True)[:max(0, limit)]:
            try:
                out.append(json.loads(p.read_text()))
            except (OSError, json.JSONDecodeError):
                continue
        return out

    def image_jpeg(self, item_id: str, annotated: bool = False) -> bytes:
        """The stored frame, or its annotated version (rendered now if it was not stored).

        Raises ``KeyError`` if the record or its image is gone.
        """
        raw_path = self.path(item_id, ".jpg")
        if annotated:
            stored = self.path(item_id, "_annotated.jpg")
            if stored.is_file():
                return stored.read_bytes()
        if not raw_path.is_file():
            raise KeyError(item_id)
        data = raw_path.read_bytes()
        if not annotated:
            return data
        return encode_jpeg(Inspector.render(decode_image(data), self.get(item_id)), 85)
