"""Request bodies of the device API."""

from __future__ import annotations

from pydantic import BaseModel, Field


class SettingsUpdate(BaseModel):
    """Any subset of the runtime settings. Omitted fields are left unchanged."""

    conf_threshold: float | None = Field(None, description="FAIL threshold, 0 < value <= 1.")
    roi: list[float] | None = Field(None, description="Region of interest as normalized [x0, y0, x1, y1].")


class BuzzerTest(BaseModel):
    pattern: str = Field("PASS", description="PASS, FAIL or ERROR.")
