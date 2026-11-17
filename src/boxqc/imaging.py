"""Image helpers shared by the device (inference, live view) and the studio (dataset export).

Conventions used throughout BoxQC:

* Images are NumPy arrays in OpenCV's layout: ``(height, width, 3)``, ``uint8``, **BGR**.
* A *region of interest* (ROI) is ``[x0, y0, x1, y1]`` as fractions (0..1) of the full
  frame, so it stays valid whatever the camera resolution.
* Pixel boxes are ``(x0, y0, x1, y1)`` with the origin at the top-left corner.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import cv2
import numpy as np

if TYPE_CHECKING:
    from .models.base import Detection

# BGR colours per class index (distinct, readable on a brown box)
_PALETTE = [
    (0, 200, 255),   # amber
    (255, 80, 200),  # magenta
    (255, 200, 0),   # cyan-ish
    (80, 255, 80),   # green
    (60, 60, 255),   # red
    (255, 120, 60),  # blue
]
PASS_COLOR = (60, 180, 60)
FAIL_COLOR = (40, 40, 220)
NEUTRAL_COLOR = (160, 160, 160)


def class_color(index: int) -> tuple[int, int, int]:
    """BGR colour for a class index. The web UI uses the same palette (``common.js``)."""
    return _PALETTE[index % len(_PALETTE)]


def roi_to_pixels(roi: Sequence[float], width: int, height: int) -> tuple[int, int, int, int]:
    """Convert a normalized ROI to pixel bounds ``(x0, y0, x1, y1)`` for an image of this size.

    The result is always at least one pixel wide and high, so it can be used to slice.
    Training and inference both go through this function, which is what guarantees
    that the model sees exactly the same crop in both.
    """
    x0, y0, x1, y1 = roi
    px0 = int(round(np.clip(x0, 0, 1) * width))
    py0 = int(round(np.clip(y0, 0, 1) * height))
    px1 = int(round(np.clip(x1, 0, 1) * width))
    py1 = int(round(np.clip(y1, 0, 1) * height))
    return px0, py0, max(px1, px0 + 1), max(py1, py0 + 1)


def crop_roi(image: np.ndarray, roi: Sequence[float]) -> tuple[np.ndarray, tuple[int, int]]:
    """Crop to the ROI. Returns (crop, (x_offset, y_offset))."""
    h, w = image.shape[:2]
    x0, y0, x1, y1 = roi_to_pixels(roi, w, h)
    return image[y0:y1, x0:x1], (x0, y0)


def encode_jpeg(image: np.ndarray, quality: int = 90) -> bytes:
    """Encode a BGR image as JPEG bytes (``quality`` 1-100)."""
    ok, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
    if not ok:
        raise RuntimeError("JPEG encoding failed")
    return buf.tobytes()


def decode_image(data: bytes) -> np.ndarray:
    """Decode JPEG/PNG/... bytes into a BGR image. Raises ``ValueError`` if they are not an image."""
    arr = np.frombuffer(data, dtype=np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Could not decode image")
    return image


def resize_max(image: np.ndarray, max_side: int) -> np.ndarray:
    """Shrink so the longer side is at most ``max_side``, keeping the aspect ratio. Never enlarges."""
    h, w = image.shape[:2]
    scale = max_side / max(h, w)
    if scale >= 1:
        return image
    return cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)


def draw_guides(image: np.ndarray, roi: Sequence[float]) -> np.ndarray:
    """Alignment overlay for the live preview: ROI rectangle + centre crosshair."""
    out = image.copy()
    h, w = out.shape[:2]
    x0, y0, x1, y1 = roi_to_pixels(roi, w, h)
    cv2.rectangle(out, (x0, y0), (x1 - 1, y1 - 1), (0, 255, 255), 2)
    cx, cy = w // 2, h // 2
    arm = max(10, min(w, h) // 20)
    cv2.line(out, (cx - arm, cy), (cx + arm, cy), (0, 255, 0), 1)
    cv2.line(out, (cx, cy - arm), (cx, cy + arm), (0, 255, 0), 1)
    # thirds grid, subtle
    for i in (1, 2):
        cv2.line(out, (w * i // 3, 0), (w * i // 3, h), (90, 90, 90), 1)
        cv2.line(out, (0, h * i // 3), (w, h * i // 3), (90, 90, 90), 1)
    return out


def draw_detections(
    image: np.ndarray,
    detections: Sequence[Detection],
    threshold: float,
    verdict: str,
    roi: Sequence[float] | None = None,
) -> np.ndarray:
    """Render the detections at or above `threshold` plus a PASS/FAIL banner.

    Returns a copy; ``image`` is not modified. ``detections`` are in pixels of ``image``.
    Line width and text size scale with the image so the result is readable at any
    resolution. When ``roi`` is given (and is not the full frame) its outline is drawn too.
    """
    out = image.copy()
    h, w = out.shape[:2]
    thick = max(2, round(max(h, w) / 600))
    font_scale = max(0.6, max(h, w) / 1600)

    if roi is not None and list(roi) != [0.0, 0.0, 1.0, 1.0]:
        x0, y0, x1, y1 = roi_to_pixels(roi, w, h)
        cv2.rectangle(out, (x0, y0), (x1 - 1, y1 - 1), NEUTRAL_COLOR, 1)

    for det in detections:
        if det.conf < threshold:
            continue
        color = class_color(det.cls_id)
        x0, y0, x1, y1 = (int(round(v)) for v in det.box)
        cv2.rectangle(out, (x0, y0), (x1, y1), color, thick)
        label = f"{det.name} {det.conf:.2f}"
        (tw, th), base = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thick)
        ty = y0 - 4 if y0 - th - base - 4 > 0 else y1 + th + base + 4
        cv2.rectangle(out, (x0, ty - th - base), (x0 + tw + 4, ty + base // 2), color, -1)
        cv2.putText(out, label, (x0 + 2, ty - base // 2), cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale, (0, 0, 0), max(1, thick - 1), cv2.LINE_AA)

    banner_color = {"PASS": PASS_COLOR, "FAIL": FAIL_COLOR}.get(verdict, NEUTRAL_COLOR)
    text = verdict.replace("_", " ")
    scale = font_scale * 1.8
    (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick + 1)
    pad = int(th * 0.4)
    cv2.rectangle(out, (0, 0), (tw + 2 * pad, th + base + 2 * pad), banner_color, -1)
    cv2.putText(out, text, (pad, th + pad), cv2.FONT_HERSHEY_SIMPLEX, scale,
                (255, 255, 255), thick + 1, cv2.LINE_AA)
    return out
