"""The built-in model format: YOLO-family detectors exported to ONNX, run with ONNX Runtime.

No PyTorch is needed, which is what makes inference practical on a Raspberry Pi.

Input contract
    One image tensor, ``(1, 3, H, W)``, RGB, ``float32`` (or ``float16``) scaled to 0..1.
    The image is letterboxed: resized keeping its aspect ratio and padded with grey (114).

Output layouts (the first output tensor; ``nc`` = number of classes)
    ``yolo``      ``(1, 4 + nc, N)``  cx, cy, w, h, class scores. Ultralytics YOLOv8 / YOLO11.
    ``yolo-obj``  ``(1, N, 5 + nc)``  cx, cy, w, h, objectness, class scores. YOLOv5 / YOLOv7.
    ``end2end``   ``(1, N, 6)``       x0, y0, x1, y1, score, class. Exports with NMS built in (YOLOv10).

    Box coordinates are in input pixels. ``yolo`` and ``yolo-obj`` are accepted with the
    channel axis first or last. The layout is detected from the output shape and the
    number of class names. An ``(N, 6)`` output is read as ``end2end``; set ``"layout"``
    in the metadata sidecar to say otherwise (a single-class ``yolo-obj`` model and a
    two-class channels-last ``yolo`` model have that same shape).

Class names come from, in order: the ONNX metadata key ``names`` (written by
Ultralytics), ``classes`` in the metadata sidecar, the configured fallback.
"""

from __future__ import annotations

import ast
import time
from pathlib import Path

import cv2
import numpy as np

from .base import Detection, ModelAdapter, ModelError

LAYOUTS = ("auto", "yolo", "yolo-obj", "end2end")


def letterbox(image: np.ndarray, size: tuple[int, int]) -> tuple[np.ndarray, float, tuple[int, int]]:
    """Resize keeping aspect ratio and pad to `size` (h, w). Returns (img, scale, (pad_x, pad_y))."""
    th, tw = size
    h, w = image.shape[:2]
    scale = min(th / h, tw / w)
    nw, nh = int(round(w * scale)), int(round(h * scale))
    resized = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR) if (nw, nh) != (w, h) else image
    pad_x, pad_y = (tw - nw) // 2, (th - nh) // 2
    out = np.full((th, tw, 3), 114, dtype=np.uint8)
    out[pad_y:pad_y + nh, pad_x:pad_x + nw] = resized
    return out, scale, (pad_x, pad_y)


def nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> list[int]:
    """Plain NumPy non-maximum suppression. boxes: (N, 4) xyxy. Returns the kept indices."""
    order = scores.argsort()[::-1]
    areas = (boxes[:, 2] - boxes[:, 0]).clip(0) * (boxes[:, 3] - boxes[:, 1]).clip(0)
    keep: list[int] = []
    while order.size:
        i = int(order[0])
        keep.append(i)
        rest = order[1:]
        xx0 = np.maximum(boxes[i, 0], boxes[rest, 0])
        yy0 = np.maximum(boxes[i, 1], boxes[rest, 1])
        xx1 = np.minimum(boxes[i, 2], boxes[rest, 2])
        yy1 = np.minimum(boxes[i, 3], boxes[rest, 3])
        inter = (xx1 - xx0).clip(0) * (yy1 - yy0).clip(0)
        iou = inter / (areas[i] + areas[rest] - inter + 1e-9)
        order = rest[iou <= iou_threshold]
    return keep


def _channels_first(shape: tuple[int, int], channels: int | None) -> bool:
    """Is a 2-D prediction array laid out (channels, N) rather than (N, channels)?"""
    if channels is not None:
        if shape[0] == channels:
            return True
        if shape[1] == channels:
            return False
    return True     # what Ultralytics exports


def resolve_layout(shape: tuple[int, int], num_classes: int | None, layout: str = "auto") -> tuple[str, bool]:
    """Work out (layout, channels_first) for a 2-D output of the given shape."""
    if layout not in LAYOUTS:
        raise ModelError(f"Unknown output layout {layout!r} (expected one of {', '.join(LAYOUTS)})")
    if layout == "end2end" or (layout == "auto" and shape[1] == 6 and shape[0] != 6):
        if 6 not in shape:
            raise ModelError(f"Output shape {shape} is not an end2end (N, 6) layout")
        return "end2end", shape[1] != 6
    if layout == "auto":
        layout = "yolo"
        if num_classes and 4 + num_classes not in shape and 5 + num_classes in shape:
            layout = "yolo-obj"
    channels = None if not num_classes else num_classes + (5 if layout == "yolo-obj" else 4)
    if channels is not None and channels not in shape:
        raise ModelError(f"Output shape {shape} does not fit {num_classes} classes in the "
                         f"{layout!r} layout ({channels} channels expected)")
    if channels is None and layout == "yolo-obj":
        return layout, shape[0] < shape[1]      # no class count to go by: the short axis is channels
    return layout, _channels_first(shape, channels)


def decode_output(
    output: np.ndarray, conf_threshold: float, iou_threshold: float,
    num_classes: int | None = None, layout: str = "auto",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Decode raw model output into (boxes_xyxy, scores, class_ids) in model-input pixels."""
    out = np.asarray(output, dtype=np.float32)
    if out.ndim == 3:
        out = out[0]
    if out.ndim != 2:
        raise ModelError(f"Expected a 2-D or 3-D output tensor, got shape {np.shape(output)}")
    kind, channels_first = resolve_layout(out.shape, num_classes, layout)
    preds = out.T if channels_first else out             # -> (N, channels)

    if kind == "end2end":                                # already NMS'd, xyxy
        sel = preds[preds[:, 4] >= conf_threshold]
        return sel[:, :4], sel[:, 4], sel[:, 5].astype(int)

    # yolo-obj: a class score is only meaningful multiplied by the box's objectness
    class_scores = preds[:, 5:] * preds[:, 4:5] if kind == "yolo-obj" else preds[:, 4:]
    if class_scores.shape[1] == 0:
        raise ModelError(f"Output shape {out.shape} has no class scores")
    cls_ids = class_scores.argmax(axis=1)
    scores = class_scores[np.arange(len(preds)), cls_ids]
    mask = scores >= conf_threshold
    preds, scores, cls_ids = preds[mask], scores[mask], cls_ids[mask]
    if not len(preds):
        return np.zeros((0, 4), np.float32), np.zeros(0, np.float32), np.zeros(0, int)

    cx, cy, w, h = preds[:, 0], preds[:, 1], preds[:, 2], preds[:, 3]
    boxes = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1)
    # class-aware NMS via coordinate offset per class
    offset = cls_ids[:, None].astype(np.float32) * 10_000.0
    keep = nms(boxes + offset, scores, iou_threshold)
    return boxes[keep], scores[keep], cls_ids[keep]


def _parse_names(raw: str | None) -> list[str] | None:
    """Class names from the ONNX metadata string Ultralytics writes (a dict or list literal)."""
    if not raw:
        return None
    try:
        names = ast.literal_eval(raw)
    except (ValueError, SyntaxError):
        return None
    if isinstance(names, dict):
        return [str(names[k]) for k in sorted(names)]
    if isinstance(names, (list, tuple)):
        return [str(n) for n in names]
    return None


def _parse_imgsz(raw: str | int | list | None) -> tuple[int, int]:
    """(height, width) from an ``imgsz`` value: 640, [640, 640] or their string forms."""
    try:
        value = ast.literal_eval(raw) if isinstance(raw, str) else raw
        h, w = (value, value) if isinstance(value, int) else value
        return int(h), int(w)
    except (ValueError, SyntaxError, TypeError):
        return 640, 640


class YoloOnnxModel(ModelAdapter):
    """YOLO-family ONNX detector on ONNX Runtime (CPU). See the module docstring for the contract."""

    format = "yolo-onnx"
    suffixes = (".onnx",)

    def __init__(self, path: str | Path, *, threads: int = 4,
                 fallback_classes: list[str] | None = None, info: dict | None = None):
        import onnxruntime as ort

        self.path = Path(path)
        self.info = dict(info or {})
        self.layout = str(self.info.get("layout", "auto"))
        if self.layout not in LAYOUTS:
            raise ModelError(f"Unknown output layout {self.layout!r} (expected one of {', '.join(LAYOUTS)})")

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = threads
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(str(self.path), opts, providers=["CPUExecutionProvider"])
        inp = self.session.get_inputs()[0]
        self.input_name = inp.name
        self._dtype = np.float16 if "float16" in inp.type else np.float32
        meta = self.session.get_modelmeta().custom_metadata_map

        if len(inp.shape) != 4:
            raise ModelError(f"Expected an image input of shape (1, 3, H, W), got {inp.shape}")
        h, w = inp.shape[2], inp.shape[3]
        if not (isinstance(h, int) and isinstance(w, int)):     # dynamic axes: use the declared size
            h, w = _parse_imgsz(meta.get("imgsz") or self.info.get("imgsz"))
        self.input_size = (int(h), int(w))

        self.names = (
            _parse_names(meta.get("names"))
            or self.info.get("classes")
            or fallback_classes
            or []
        )

    def detect(
        self, image_bgr: np.ndarray, conf_threshold: float = 0.25, iou_threshold: float = 0.45
    ) -> tuple[list[Detection], float]:
        t0 = time.perf_counter()
        padded, scale, (pad_x, pad_y) = letterbox(image_bgr, self.input_size)
        blob = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB).transpose(2, 0, 1)[None].astype(self._dtype) / 255.0
        output = self.session.run(None, {self.input_name: blob.astype(self._dtype, copy=False)})[0]
        boxes, scores, cls_ids = decode_output(output, conf_threshold, iou_threshold,
                                               num_classes=len(self.names) or None, layout=self.layout)

        h, w = image_bgr.shape[:2]
        dets: list[Detection] = []
        for box, score, cid in zip(boxes, scores, cls_ids, strict=True):
            x0 = float(np.clip((box[0] - pad_x) / scale, 0, w))
            y0 = float(np.clip((box[1] - pad_y) / scale, 0, h))
            x1 = float(np.clip((box[2] - pad_x) / scale, 0, w))
            y1 = float(np.clip((box[3] - pad_y) / scale, 0, h))
            if x1 - x0 < 1 or y1 - y0 < 1:
                continue
            dets.append(Detection(int(cid), self.name_for(int(cid)), float(score), (x0, y0, x1, y1)))
        dets.sort(key=lambda d: d.conf, reverse=True)
        return dets, (time.perf_counter() - t0) * 1000

    def describe(self) -> dict:
        return {**super().describe(), "layout": self.layout}
