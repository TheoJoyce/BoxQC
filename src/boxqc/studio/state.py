"""Studio state that outlives a restart, persisted to ``<data_dir>/studio.json``.

``capture_dataset``
    The named dataset that new captures and uploads join (``None`` = no dataset).
``roi``
    The device's region of interest as last seen. Training crops images with it, so a
    copy is kept here for when the device is switched off. ``None`` until the device
    has been reached once; :func:`effective_roi` then falls back to ``boxqc.yaml``.
"""

from __future__ import annotations

import contextlib
import json
import threading
from pathlib import Path
from typing import Any

from ..config import Config, validate_roi

STATE_FILE = "studio.json"
LEGACY_SETTINGS_FILE = "settings.json"   # BoxQC 0.1: the Pi's settings, as copied by `boxqc pull`


def _read_json(path: Path) -> dict:
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


class StudioState:
    """Thread-safe key/value store. Keys: ``capture_dataset``, ``roi``."""

    def __init__(self, data_dir: Path):
        self.path = data_dir / STATE_FILE
        self._lock = threading.Lock()
        self._values: dict[str, Any] = {"capture_dataset": None, "roi": None}
        # First run after upgrading from 0.1: start from what the old `boxqc pull` left behind.
        stored = _read_json(self.path) if self.path.is_file() else _read_json(data_dir / LEGACY_SETTINGS_FILE)
        for key in self._values:
            if key in stored:
                with contextlib.suppress(ValueError, TypeError):    # ignore a corrupt value
                    self._values[key] = self._clean(key, stored[key])

    @staticmethod
    def _clean(key: str, value: Any) -> Any:
        if key == "capture_dataset":
            return None if value in (None, "") else str(value)
        if key == "roi":
            return None if value is None else validate_roi(value)
        raise ValueError(f"Unknown studio setting: {key}")

    def get(self, key: str) -> Any:
        with self._lock:
            return self._values[key]

    def as_dict(self) -> dict[str, Any]:
        with self._lock:
            return dict(self._values)

    def update(self, **changes: Any) -> dict[str, Any]:
        """Validate and store ``changes``; the file is only rewritten if something changed."""
        clean = {key: self._clean(key, value) for key, value in changes.items()}
        with self._lock:
            if all(self._values[k] == v for k, v in clean.items()):
                return dict(self._values)
            self._values.update(clean)
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(json.dumps(self._values, indent=2))
            tmp.replace(self.path)
            return dict(self._values)


def effective_roi(cfg: Config) -> list[float]:
    """The ROI to crop training images with: the device's (as last seen), else ``boxqc.yaml``'s."""
    roi = StudioState(cfg.data_dir).get("roi")
    return roi if roi is not None else list(cfg.roi)
