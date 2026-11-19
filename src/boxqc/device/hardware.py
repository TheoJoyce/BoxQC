"""GPIO: PASS/FAIL buzzer and an optional hardware trigger button.

Uses gpiozero, imported lazily. On a machine without GPIO (or with ``hardware.mock``)
the buzzer is replaced by a log line and the button is ignored, so the device runs
unchanged on a development machine.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from typing import Any

from ..config import HardwareConfig

log = logging.getLogger(__name__)

# Buzzer pattern per verdict: (on_seconds, off_seconds, repeats). The three must be
# easy to tell apart by ear on a noisy line.
PATTERNS: dict[str, tuple[float, float, int]] = {
    "PASS": (0.12, 0.10, 1),       # one short chirp
    "FAIL": (0.33, 0.20, 3),       # three beeps, 1 s of sound in total
    "NO_MODEL": (0.08, 0.08, 6),   # rapid chatter = system problem
    "ERROR": (0.08, 0.08, 6),
}

# PASS and FAIL can be changed while the device runs (the ``buzzer`` runtime setting):
# the number of beeps and the length of each. The gap between beeps stays as above, and
# the ERROR chatter is fixed so a system problem always sounds the same.
CONFIGURABLE = ("PASS", "FAIL")
BEEPS_RANGE = (1, 10)
BEEP_SECONDS_RANGE = (0.05, 2.0)   # a long pattern delays the next one (they never overlap)


def default_buzzer_setting() -> dict[str, dict]:
    """``{"PASS": {"beeps": 1, "beep_seconds": 0.12}, "FAIL": {...}}`` from ``PATTERNS``."""
    return {v: {"beeps": PATTERNS[v][2], "beep_seconds": PATTERNS[v][0]} for v in CONFIGURABLE}


def validate_buzzer(value: Any, current: dict[str, dict]) -> dict[str, dict]:
    """Merge a (possibly partial) ``buzzer`` setting over ``current``. Raises ``ValueError``."""
    if not isinstance(value, dict):
        raise ValueError("buzzer must be an object like {\"FAIL\": {\"beeps\": 3, \"beep_seconds\": 0.3}}")
    merged = {v: dict(current.get(v) or default_buzzer_setting()[v]) for v in CONFIGURABLE}
    for verdict, pattern in value.items():
        key = str(verdict).upper()
        if key not in CONFIGURABLE:
            raise ValueError(f"buzzer: only {' and '.join(CONFIGURABLE)} can be changed, not {verdict!r}")
        if not isinstance(pattern, dict) or not set(pattern) <= {"beeps", "beep_seconds"}:
            raise ValueError(f"buzzer.{key} must be an object with beeps and/or beep_seconds")
        if "beeps" in pattern:
            beeps = pattern["beeps"]
            if isinstance(beeps, bool) or not isinstance(beeps, int | float) or beeps != int(beeps) \
                    or not BEEPS_RANGE[0] <= beeps <= BEEPS_RANGE[1]:
                raise ValueError(f"buzzer.{key}.beeps must be a whole number from {BEEPS_RANGE[0]} to {BEEPS_RANGE[1]}")
            merged[key]["beeps"] = int(beeps)
        if "beep_seconds" in pattern:
            seconds = pattern["beep_seconds"]
            if isinstance(seconds, bool) or not isinstance(seconds, int | float) \
                    or not BEEP_SECONDS_RANGE[0] <= seconds <= BEEP_SECONDS_RANGE[1]:
                raise ValueError(f"buzzer.{key}.beep_seconds must be from {BEEP_SECONDS_RANGE[0]} "
                                 f"to {BEEP_SECONDS_RANGE[1]}")
            merged[key]["beep_seconds"] = round(float(seconds), 3)
    return merged


def pattern_for(verdict: str, buzzer: dict[str, dict] | None = None) -> tuple[float, float, int]:
    """(on, off, repeats) for a verdict, with the PASS/FAIL beeps taken from ``buzzer`` if given."""
    on, off, repeats = PATTERNS.get(verdict, PATTERNS["ERROR"])
    custom = (buzzer or {}).get(verdict)
    if custom:
        on, repeats = custom["beep_seconds"], custom["beeps"]
    return on, off, repeats


def describe_patterns(buzzer: dict[str, dict] | None = None) -> dict[str, dict]:
    """Every verdict's pattern as ``{"beeps", "beep_seconds", "gap_seconds"}``, for clients
    that play the same sound themselves (the studio's Inspect page can, in the browser)."""
    out = {}
    for verdict in PATTERNS:
        on, off, repeats = pattern_for(verdict, buzzer)
        out[verdict] = {"beeps": repeats, "beep_seconds": on, "gap_seconds": off}
    return out


class Signaller:
    """The station's operator interface: sounds the buzzer and watches the trigger button.

    Construction never fails. If the GPIO pins cannot be claimed, ``error`` says why and
    the buzzer is mocked; the problem is reported through ``GET /api/v1/status``.
    """

    def __init__(self, cfg: HardwareConfig, buzzer_setting: Callable[[], dict] | None = None):
        self.cfg = cfg
        self._buzzer_setting = buzzer_setting   # returns the current PASS/FAIL beeps
        self._out = None          # gpiozero output device
        self._on_value = 1.0
        self._button = None
        self._play_lock = threading.Lock()
        self.last_pattern: str | None = None
        self.error: str | None = None

        if cfg.mock or cfg.buzzer_pin is None:
            return
        try:
            if cfg.buzzer_type == "passive":
                from gpiozero import PWMOutputDevice

                self._out = PWMOutputDevice(
                    cfg.buzzer_pin, active_high=cfg.buzzer_active_high,
                    initial_value=0, frequency=cfg.passive_frequency,
                )
                self._on_value = 0.5  # 50% duty square wave
            else:
                from gpiozero import DigitalOutputDevice

                self._out = DigitalOutputDevice(
                    cfg.buzzer_pin, active_high=cfg.buzzer_active_high, initial_value=False
                )
        except Exception as exc:
            self.error = f"Buzzer unavailable: {exc}"
            log.warning("%s - using mock buzzer", self.error)

    @property
    def is_mock(self) -> bool:
        """True when there is no real buzzer (no GPIO, disabled, or it failed to initialise)."""
        return self._out is None

    def signal(self, verdict: str) -> None:
        """Play the pattern for a verdict without blocking the caller."""
        pattern = pattern_for(verdict, self._buzzer_setting() if self._buzzer_setting else None)
        self.last_pattern = verdict
        if self._out is None:
            log.info("[mock buzzer] %s %s", verdict, pattern)
            return
        threading.Thread(target=self._play, args=pattern, daemon=True).start()

    def _play(self, on: float, off: float, repeats: int) -> None:
        with self._play_lock:    # patterns never overlap: a second one waits for the first
            for _ in range(repeats):
                self._out.value = self._on_value
                time.sleep(on)
                self._out.value = 0
                time.sleep(off)

    def on_trigger(self, callback: Callable[[], None]) -> None:
        """Run `callback` (in a worker thread) whenever the trigger button is pressed."""
        if self.cfg.mock or self.cfg.trigger_pin is None:
            return
        try:
            from gpiozero import Button

            self._button = Button(self.cfg.trigger_pin, pull_up=True, bounce_time=0.1)
            self._button.when_pressed = lambda: threading.Thread(target=callback, daemon=True).start()
        except Exception as exc:
            self.error = f"Trigger button unavailable: {exc}"
            log.warning(self.error)

    def status(self) -> dict:
        """The ``hardware`` section of the status endpoint."""
        return {
            "buzzer": "mock" if self.is_mock else f"GPIO{self.cfg.buzzer_pin} ({self.cfg.buzzer_type})",
            "trigger": None if self._button is None else f"GPIO{self.cfg.trigger_pin}",
            "last_pattern": self.last_pattern,
            "patterns": describe_patterns(self._buzzer_setting() if self._buzzer_setting else None),
            "error": self.error,
        }

    def close(self) -> None:
        for dev in (self._out, self._button):
            if dev is not None:
                dev.close()
