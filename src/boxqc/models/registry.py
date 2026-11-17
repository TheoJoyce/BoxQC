"""Model files on disk: metadata sidecars, format selection and loading."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np

from ..plugins import Registry
from .base import ModelAdapter, ModelError

#: Every known model format, by name. Third-party formats join through the
#: ``boxqc.model_formats`` entry-point group or :func:`register_format`.
FORMATS: Registry[ModelAdapter] = Registry("boxqc.model_formats", "model format")


def register_format(cls: type[ModelAdapter]) -> type[ModelAdapter]:
    """Class decorator: make a :class:`ModelAdapter` available under its ``format`` name."""
    return FORMATS.register(cls, cls.format)


def model_suffixes() -> tuple[str, ...]:
    """File extensions that some registered format can load, e.g. ``(".onnx",)``."""
    seen: dict[str, None] = {}
    for _, cls in FORMATS.items():
        for suffix in cls.suffixes:
            seen.setdefault(suffix.lower())
    return tuple(seen)


def is_model_file(path: Path) -> bool:
    return path.suffix.lower() in model_suffixes()


def metadata_path(model_path: Path) -> Path:
    """``models/foo.onnx`` -> ``models/foo.json``."""
    return model_path.with_suffix(".json")


def read_metadata(model_path: Path) -> dict:
    """The model's metadata sidecar, or ``{}`` if it is missing or unreadable."""
    sidecar = metadata_path(model_path)
    if not sidecar.is_file():
        return {}
    try:
        info = json.loads(sidecar.read_text())
    except (OSError, json.JSONDecodeError):
        return {}
    return info if isinstance(info, dict) else {}


def format_for(model_path: Path, info: dict | None = None) -> type[ModelAdapter]:
    """Pick the adapter class: ``info["format"]`` if given, else the default for the file extension."""
    info = read_metadata(model_path) if info is None else info
    name = info.get("format")
    if name:
        try:
            return FORMATS.get(str(name))
        except KeyError as exc:
            raise ModelError(str(exc.args[0])) from None
    suffix = model_path.suffix.lower()
    for _, cls in FORMATS.items():
        if suffix in cls.suffixes:
            return cls
    raise ModelError(f"No model format handles {suffix or 'files without an extension'} "
                     f"(supported: {', '.join(model_suffixes())})")


def load_model(model_path: str | Path, *, threads: int = 4,
               fallback_classes: list[str] | None = None) -> ModelAdapter:
    """Load a model file with the adapter its metadata (or file extension) selects."""
    path = Path(model_path)
    info = read_metadata(path)
    cls = format_for(path, info)
    return cls(path, threads=threads, fallback_classes=fallback_classes, info=info)


def validate_model(model: ModelAdapter) -> None:
    """Run one inference on a blank frame, so a model that cannot be decoded is rejected
    when it is uploaded rather than at the first inspection."""
    h, w = model.input_size
    try:
        model.detect(np.zeros((h, w, 3), np.uint8), 0.5, 0.45)
    except ModelError:
        raise
    except Exception as exc:
        raise ModelError(f"Test inference failed: {exc}") from exc


# path -> (mtime_ns, size, digest) of the files hashed so far
_DIGESTS: dict[str, tuple[int, int, str]] = {}


def file_sha256(path: Path) -> str:
    """SHA-256 of a file's content, as hex.

    Two model files are the same model exactly when this matches; names say nothing
    (a run name can be reused) and neither do sizes (two trainings of one architecture
    produce files of equal size). The digest is remembered until the file changes,
    because models are large and are listed often.
    """
    stat = path.stat()
    known = _DIGESTS.get(str(path))
    if known is not None and known[:2] == (stat.st_mtime_ns, stat.st_size):
        return known[2]
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    _DIGESTS[str(path)] = (stat.st_mtime_ns, stat.st_size, digest.hexdigest())
    return digest.hexdigest()


def list_models(models_dir: Path) -> list[dict]:
    """Model files in ``models_dir`` with their metadata and content hash, newest first."""
    if not models_dir.is_dir():
        return []
    paths = [p for p in models_dir.iterdir() if p.is_file() and is_model_file(p)]
    models = []
    for p in sorted(paths, key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            stat, digest = p.stat(), file_sha256(p)
        except OSError:         # deleted while listing
            continue
        models.append({
            "name": p.name,
            "size_mb": round(stat.st_size / 1e6, 2),
            "modified": time.strftime("%Y-%m-%d %H:%M", time.localtime(stat.st_mtime)),
            "sha256": digest,
            "info": read_metadata(p),
        })
    return models
