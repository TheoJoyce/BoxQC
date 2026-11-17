"""Model formats: loading a detector file and running it, whatever produced it.

Typical use::

    from boxqc.models import load_model

    model = load_model("models/my-model.onnx")
    detections, ms = model.detect(image_bgr, conf_threshold=0.25)

The file format, the metadata sidecar and how to add a format are described in
``docs/models.md``.
"""

from .base import Detection, ModelAdapter, ModelError
from .registry import (
    FORMATS,
    file_sha256,
    format_for,
    is_model_file,
    list_models,
    load_model,
    metadata_path,
    model_suffixes,
    read_metadata,
    register_format,
    validate_model,
)
from .yolo_onnx import YoloOnnxModel

# The built-in format. Registered first, so it is the default for ``.onnx`` files.
register_format(YoloOnnxModel)

__all__ = [
    "FORMATS",
    "Detection",
    "ModelAdapter",
    "ModelError",
    "YoloOnnxModel",
    "file_sha256",
    "format_for",
    "is_model_file",
    "list_models",
    "load_model",
    "metadata_path",
    "model_suffixes",
    "read_metadata",
    "register_format",
    "validate_model",
]
