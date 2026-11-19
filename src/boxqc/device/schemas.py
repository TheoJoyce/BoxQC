"""Request bodies of the device API."""

from __future__ import annotations

from pydantic import BaseModel, Field


class BeepPattern(BaseModel):
    """How one verdict sounds. Omitted fields are left unchanged."""

    beeps: int | None = Field(None, description="Number of beeps, 1-10.")
    beep_seconds: float | None = Field(None, description="Length of each beep in seconds, 0.05-2.")


class SettingsUpdate(BaseModel):
    """Any subset of the runtime settings. Omitted fields are left unchanged."""

    conf_threshold: float | None = Field(None, description="FAIL threshold, 0 < value <= 1.")
    roi: list[float] | None = Field(None, description="Region of interest as normalized [x0, y0, x1, y1].")
    buzzer: dict[str, BeepPattern] | None = Field(
        None, description='PASS and/or FAIL beeps, e.g. {"FAIL": {"beeps": 3, "beep_seconds": 0.3}}.')


class BuzzerTest(BaseModel):
    pattern: str = Field("PASS", description="PASS, FAIL or ERROR.")
