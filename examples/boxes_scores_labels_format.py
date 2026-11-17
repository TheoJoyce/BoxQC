"""Example model format: detectors that output separate ``boxes``, ``scores`` and ``labels``.

Many non-YOLO exporters (torchvision detection models, some RT-DETR and SSD exports with
post-processing built in) produce three output tensors instead of YOLO's single one.
This adapter runs such a model, and shows everything a model format has to do.

What it assumes about the model:

* one input ``(1, 3, H, W)`` with a fixed H and W, RGB, float32 scaled to 0..1, and the
  image simply resized to H x W (no letterbox padding);
* outputs named ``boxes`` ``(N, 4)`` as x0, y0, x1, y1 in input pixels, ``scores``
  ``(N,)`` and ``labels`` ``(N,)``, already filtered by the model's own NMS.

To use it, make the module importable and list it in boxqc.yaml (see docs/models.md)::

    plugins: [boxes_scores_labels_format]

then give the model a metadata sidecar that selects the format and names the classes::

    {"format": "boxes-scores-labels", "classes": ["dirty", "defect"]}
"""

from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np

from boxqc.models import Detection, ModelAdapter, ModelError, register_format


@register_format
class BoxesScoresLabelsModel(ModelAdapter):
    format = "boxes-scores-labels"
    suffixes = (".onnx",)

    def __init__(self, path: str | Path, *, threads: int = 4,
                 fallback_classes: list[str] | None = None, info: dict | None = None):
        import onnxruntime as ort

        self.path = Path(path)
        self.info = dict(info or {})
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        self.session = ort.InferenceSession(str(self.path), options, providers=["CPUExecutionProvider"])

        model_input = self.session.get_inputs()[0]
        self.input_name = model_input.name
        height, width = model_input.shape[2], model_input.shape[3]
        if not (isinstance(height, int) and isinstance(width, int)):
            raise ModelError("This format needs a fixed input size")
        self.input_size = (height, width)

        outputs = {o.name for o in self.session.get_outputs()}
        if not {"boxes", "scores", "labels"} <= outputs:
            raise ModelError(f"Expected outputs named boxes, scores and labels; the model has {sorted(outputs)}")
        self.names = list(self.info.get("classes") or fallback_classes or [])
        # e.g. torchvision reserves label 0 for "background": set "label_offset": 1 in the metadata
        self.label_offset = int(self.info.get("label_offset", 0))

    def detect(self, image_bgr: np.ndarray, conf_threshold: float = 0.25,
               iou_threshold: float = 0.45) -> tuple[list[Detection], float]:
        started = time.perf_counter()
        height, width = self.input_size
        resized = cv2.resize(image_bgr, (width, height), interpolation=cv2.INTER_LINEAR)
        blob = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).transpose(2, 0, 1)[None].astype(np.float32) / 255.0
        boxes, scores, labels = self.session.run(["boxes", "scores", "labels"], {self.input_name: blob})

        # back from model-input pixels to pixels of the image we were given
        image_h, image_w = image_bgr.shape[:2]
        scale = np.array([image_w / width, image_h / height] * 2, np.float32)
        detections = []
        for box, score, label in zip(np.reshape(boxes, (-1, 4)), np.ravel(scores), np.ravel(labels), strict=True):
            if score < conf_threshold:
                continue
            x0, y0, x1, y1 = (float(v) for v in np.clip(box * scale, 0, [image_w, image_h] * 2))
            cls_id = int(label) - self.label_offset
            detections.append(Detection(cls_id, self.name_for(cls_id), float(score), (x0, y0, x1, y1)))
        detections.sort(key=lambda d: d.conf, reverse=True)
        return detections, (time.perf_counter() - started) * 1000
