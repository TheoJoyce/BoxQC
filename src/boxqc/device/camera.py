"""Camera backends: Picamera2 (Pi camera modules), OpenCV (USB webcams), and a mock.

The rest of the device only uses the small :class:`Camera` interface, so supporting
another camera means one more subclass and a branch in :func:`open_camera`. The backend
is chosen by ``camera.backend`` in ``boxqc.yaml``.

Every backend may be called from several threads at once (the live stream, an
inspection, a capture request), so each guards its device with a lock.
"""

from __future__ import annotations

import logging
import threading
import time
from importlib.util import find_spec

import cv2
import numpy as np

from ..config import CameraConfig

log = logging.getLogger(__name__)


class Camera:
    """What the device needs from a camera. Frames are BGR ``uint8`` arrays (OpenCV layout)."""

    #: Name reported by ``GET /api/v1/status`` (``"picamera2"``, ``"opencv"``, ``"mock"``).
    backend = "base"

    def capture(self) -> np.ndarray:
        """A fresh full-resolution frame (``camera.capture_size``): used for inspections
        and training photos."""
        raise NotImplementedError

    def preview(self) -> np.ndarray:
        """A low-resolution frame (``camera.preview_size``) for the live view. Called
        ``camera.preview_fps`` times a second per viewer, so it must be cheap."""
        raise NotImplementedError

    def close(self) -> None:
        """Release the device. Called once when the server shuts down."""


class PiCamera(Camera):
    """Raspberry Pi camera modules through Picamera2 / libcamera.

    One video configuration provides both streams at once: ``main`` at full resolution
    for captures and ``lores`` for the preview, so switching between them costs nothing.
    """

    backend = "picamera2"

    def __init__(self, cfg: CameraConfig):
        from picamera2 import Picamera2

        self._lock = threading.Lock()
        self._preview_size = tuple(cfg.preview_size)
        self.cam = Picamera2()
        config = self.cam.create_video_configuration(
            # picamera2 "RGB888" is BGR byte order in memory, i.e. what OpenCV expects
            main={"size": tuple(cfg.capture_size), "format": "RGB888"},
            lores={"size": self._preview_size, "format": "YUV420"},
            buffer_count=4,
        )
        self.cam.configure(config)
        self.cam.start()
        if cfg.controls:
            self.cam.set_controls(dict(cfg.controls))
        time.sleep(1.0)  # let AE/AWB settle

    def capture(self) -> np.ndarray:
        with self._lock:
            frame = self.cam.capture_array("main")
        return np.ascontiguousarray(frame[:, :, :3])

    def preview(self) -> np.ndarray:
        with self._lock:
            yuv = self.cam.capture_array("lores")
        bgr = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_I420)
        return bgr[:, : self._preview_size[0]]   # drop the row padding libcamera may add

    def close(self) -> None:
        try:
            self.cam.stop()
            self.cam.close()
        except Exception:  # pragma: no cover - best effort on shutdown
            log.exception("Error closing camera")


class OpenCVCamera(Camera):
    """USB webcams (or anything else OpenCV can open), selected by ``camera.device``."""

    backend = "opencv"

    def __init__(self, cfg: CameraConfig):
        self._lock = threading.Lock()
        self._preview_size = tuple(cfg.preview_size)
        self.cap = cv2.VideoCapture(cfg.device)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open OpenCV camera {cfg.device}")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.capture_size[0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.capture_size[1])

    def _read(self, flush: int = 0) -> np.ndarray:
        with self._lock:
            for _ in range(flush):  # drop buffered (stale) frames
                self.cap.grab()
            ok, frame = self.cap.read()
        if not ok:
            raise RuntimeError("Camera read failed")
        return frame

    def capture(self) -> np.ndarray:
        return self._read(flush=3)

    def preview(self) -> np.ndarray:
        return cv2.resize(self._read(), self._preview_size, interpolation=cv2.INTER_AREA)

    def close(self) -> None:
        self.cap.release()


class MockCamera(Camera):
    """Synthetic top-down box images, so the whole stack runs on a laptop."""

    backend = "mock"

    def __init__(self, cfg: CameraConfig):
        self._size = tuple(cfg.capture_size)
        self._preview_size = tuple(cfg.preview_size)
        self._rng = np.random.default_rng()
        self._frame = self._render()
        self._last = time.monotonic()

    def _render(self) -> np.ndarray:
        w, h = self._size
        rng = self._rng
        img = np.full((h, w, 3), (70, 75, 80), np.uint8)  # grey table
        bw, bh = int(w * 0.55), int(h * 0.8)
        x0, y0 = (w - bw) // 2 + rng.integers(-10, 10), (h - bh) // 2 + rng.integers(-10, 10)
        cv2.rectangle(img, (x0, y0), (x0 + bw, y0 + bh), (60, 110, 160), -1)        # box walls
        inset = int(min(bw, bh) * 0.12)
        cv2.rectangle(img, (x0 + inset, y0 + inset), (x0 + bw - inset, y0 + bh - inset),
                      (80, 135, 185), -1)                                            # box floor
        noise = rng.normal(0, 6, img.shape).astype(np.int16)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        if rng.random() < 0.5:  # add some "dirt"
            for _ in range(rng.integers(1, 4)):
                cx = int(rng.integers(x0 + 20, x0 + bw - 20))
                cy = int(rng.integers(y0 + 20, y0 + bh - 20))
                cv2.ellipse(img, (cx, cy), (int(rng.integers(8, 40)), int(rng.integers(8, 30))),
                            float(rng.integers(0, 180)), 0, 360, (30, 40, 50), -1)
        return img

    def capture(self) -> np.ndarray:
        self._frame = self._render()
        return self._frame.copy()

    def preview(self) -> np.ndarray:
        if time.monotonic() - self._last > 3:  # change scene every few seconds
            self._frame, self._last = self._render(), time.monotonic()
        return cv2.resize(self._frame, self._preview_size, interpolation=cv2.INTER_AREA)


def open_camera(cfg: CameraConfig) -> Camera:
    """Create the camera named by ``cfg.backend``.

    ``auto`` means the Pi camera wherever the Picamera2 library is installed, and the
    mock camera where it is not, which is what lets the same configuration run on a
    development machine. A camera that is expected but fails to open always raises: the
    device then reports the error and refuses to inspect. It never falls back to the mock
    camera, because a station that inspects synthetic images would pass every box.
    """
    backend = cfg.backend.lower()
    if backend == "auto":
        if find_spec("picamera2") is None:
            log.warning("Picamera2 is not installed: using the mock camera (synthetic images)")
            return MockCamera(cfg)
        return PiCamera(cfg)
    if backend == "picamera2":
        return PiCamera(cfg)
    if backend == "opencv":
        return OpenCVCamera(cfg)
    if backend == "mock":
        return MockCamera(cfg)
    raise ValueError(f"Unknown camera backend: {cfg.backend}")
