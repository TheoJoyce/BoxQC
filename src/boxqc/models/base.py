"""The model contract: what the rest of BoxQC needs from an object detector.

A *model* is one file (for example ``my-model.onnx``) with an optional JSON metadata
sidecar of the same name (``my-model.json``). A *model format* is a
:class:`ModelAdapter` subclass that knows how to load such a file and turn a camera
image into :class:`Detection` objects. The inspection pipeline only ever talks to this
interface, so supporting another architecture or runtime means writing one adapter;
see ``docs/models.md``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

import numpy as np


class ModelError(Exception):
    """A model file could not be loaded, or its output could not be understood."""


@dataclass
class Detection:
    """One detected defect."""

    cls_id: int
    name: str
    conf: float
    box: tuple[float, float, float, float]  # x0, y0, x1, y1 in source-image pixels

    def to_dict(self) -> dict:
        return {
            "cls_id": self.cls_id,
            "name": self.name,
            "conf": round(float(self.conf), 4),
            "box": [round(float(v), 1) for v in self.box],
        }

    @classmethod
    def from_dict(cls, d: dict) -> Detection:
        return cls(int(d["cls_id"]), str(d["name"]), float(d["conf"]), tuple(d["box"]))

    def shifted(self, dx: float, dy: float) -> Detection:
        """The same detection moved by an offset (crop coordinates -> full-frame coordinates)."""
        x0, y0, x1, y1 = self.box
        return Detection(self.cls_id, self.name, self.conf, (x0 + dx, y0 + dy, x1 + dx, y1 + dy))


class ModelAdapter(ABC):
    """Base class for model formats.

    Subclasses set :attr:`format` and :attr:`suffixes`, load the file in ``__init__``
    and implement :meth:`detect`. Register the class with
    :func:`boxqc.models.register_format` or the ``boxqc.model_formats`` entry-point group.

    Constructor arguments (always passed by keyword, except ``path``):

    ``path``
        The model file.
    ``threads``
        CPU threads the runtime may use.
    ``fallback_classes``
        Class names to use when neither the model nor its metadata carries any.
    ``info``
        The parsed metadata sidecar (``{}`` when there is none).
    """

    #: Name used in the metadata sidecar (``"format": "..."``) to select this adapter.
    format: ClassVar[str]
    #: File extensions this adapter can load. The first adapter registered for an
    #: extension is the default for files whose metadata does not name a format.
    suffixes: ClassVar[tuple[str, ...]] = (".onnx",)

    path: Path
    #: Class names by class index.
    names: list[str]
    #: Model input as (height, width) in pixels.
    input_size: tuple[int, int]
    #: The metadata sidecar's content.
    info: dict

    @abstractmethod
    def __init__(self, path: str | Path, *, threads: int = 4,
                 fallback_classes: list[str] | None = None, info: dict | None = None): ...

    @abstractmethod
    def detect(
        self, image_bgr: np.ndarray, conf_threshold: float = 0.25, iou_threshold: float = 0.45
    ) -> tuple[list[Detection], float]:
        """Run the model on a BGR ``uint8`` image of any size.

        Returns the detections scoring at least ``conf_threshold``, with boxes in the
        pixel coordinates of ``image_bgr`` and sorted by confidence (highest first),
        plus the time taken in milliseconds (pre- and post-processing included).
        """

    def name_for(self, cls_id: int) -> str:
        return self.names[cls_id] if 0 <= cls_id < len(self.names) else f"class{cls_id}"

    def describe(self) -> dict:
        """What the status endpoint reports about the loaded model."""
        return {
            "name": self.path.name,
            "format": self.format,
            "classes": list(self.names),
            "input_size": list(self.input_size),
        }
