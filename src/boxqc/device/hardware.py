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

from ..config import HardwareConfig

log = logging.getLogger(__name__)

# Buzzer pattern per verdict: (on_seconds, off_seconds, repeats). The three must be
# easy to tell apart by ear on a noisy line; change the timings here to suit yours.
PATTERNS: dict[str, tuple[float, float, int]] = {
    "PASS": (0.12, 0.10, 1),       # one short chirp
    "FAIL": (0.33, 0.20, 3),       # three beeps, 1 s of sound in total
    "NO_MODEL": (0.08, 0.08, 6),   # rapid chatter = system problem
    "ERROR": (0.08, 0.08, 6),
}


class Signaller:
    """The station's operator interface: sounds the buzzer and watches the trigger button.

    Construction never fails. If the GPIO pins cannot be claimed, ``error`` says why and
    the buzzer is mocked; the problem is reported through ``GET /api/v1/status``.
    """

    def __init__(self, cfg: HardwareConfig):
        self.cfg = cfg
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
        pattern = PATTERNS.get(verdict, PATTERNS["ERROR"])
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
            "error": self.error,
        }

    def close(self) -> None:
        for dev in (self._out, self._button):
            if dev is not None:
                dev.close()
