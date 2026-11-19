"""Settings that change while the device runs, persisted to ``<data_dir>/settings.json``.

These are the values an operator tunes at the station - the FAIL threshold, the region
of interest, which model is active and the PASS/FAIL buzzer beeps - so they are owned by the device, changed
through its API, and survive restarts. They override the defaults in ``boxqc.yaml``.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

from ..config import Config, validate_roi
from .hardware import default_buzzer_setting, validate_buzzer


class RuntimeSettings:
    """Thread-safe key/value store with validation. Keys: ``conf_threshold``, ``roi``, ``active_model``, ``buzzer``."""

    def __init__(self, path: Path, cfg: Config):
        self.path = path
        self._lock = threading.Lock()
        self._values: dict[str, Any] = {
            "conf_threshold": cfg.inference.conf_threshold,
            "roi": list(cfg.roi),
            "active_model": None,
            "buzzer": default_buzzer_setting(),
        }
        if path.is_file():
            try:
                stored = json.loads(path.read_text())
                # unknown keys (e.g. from an older version) are dropped on the next save
                self._values.update({k: v for k, v in stored.items() if k in self._values})
            except (OSError, json.JSONDecodeError, AttributeError):
                pass
            try:    # a hand-edited file must not make the buzzer sleep on nonsense
                self._values["buzzer"] = validate_buzzer(self._values["buzzer"], default_buzzer_setting())
            except ValueError:
                self._values["buzzer"] = default_buzzer_setting()

    def get(self, key: str) -> Any:
        with self._lock:
            return self._values[key]

    def as_dict(self) -> dict[str, Any]:
        with self._lock:
            return dict(self._values)

    def update(self, **changes: Any) -> dict[str, Any]:
        """Validate and store ``changes``. Raises ``ValueError`` and stores nothing if any is invalid."""
        clean: dict[str, Any] = {}
        for key, value in changes.items():
            if key == "conf_threshold":
                value = float(value)
                if not 0.0 < value <= 1.0:
                    raise ValueError("conf_threshold must be in (0, 1]")
            elif key == "roi":
                value = validate_roi(value)
            elif key == "active_model":
                value = None if value in (None, "") else str(value)
            elif key == "buzzer":
                value = validate_buzzer(value, self.get("buzzer"))
            else:
                raise ValueError(f"Unknown setting: {key}")
            clean[key] = value
        with self._lock:
            self._values.update(clean)
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(json.dumps(self._values, indent=2))
            tmp.replace(self.path)
            return dict(self._values)
