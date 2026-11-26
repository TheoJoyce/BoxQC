"""Static configuration, loaded from ``boxqc.yaml``.

One file describes both halves of the system; each half reads the sections it needs:

* the device (``boxqc device``, on the Pi): ``camera``, ``hardware``, ``inference``,
  ``device``, ``roi``, ``models_dir``, ``data_dir``
* the studio (``boxqc studio``, on a desktop): ``classes``, ``train``, ``studio``,
  ``device.url``, ``models_dir``, ``data_dir``

Lookup order for the file: ``--config``, then ``$BOXQC_CONFIG``, then ``./boxqc.yaml``.
Relative paths inside it are resolved against the file's directory.

Machine-specific values go in an optional overlay next to it, ``boxqc.local.yaml``
(not committed, not synced to the Pi). It has the same structure and wins key by key::

    # boxqc.local.yaml on a MacBook: pin the device's address instead of "auto"
    device:
      url: http://192.168.1.50:8000

Two environment variables override both files: ``BOXQC_DEVICE_URL`` and
``BOXQC_API_TOKEN``. ``device.url`` may be ``auto`` (the default): the studio and the
command line then find the device on the local network (:mod:`boxqc.discovery`).

Settings that change at runtime (ROI, threshold, active model) are not here; see
:mod:`boxqc.device.settings`.
"""

from __future__ import annotations

import dataclasses
import logging
import os
import types
import typing
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

log = logging.getLogger(__name__)

CONFIG_ENV = "BOXQC_CONFIG"
DEVICE_URL_ENV = "BOXQC_DEVICE_URL"
API_TOKEN_ENV = "BOXQC_API_TOKEN"
DEFAULT_CONFIG_NAMES = ("boxqc.yaml", "boxqc.yml")
AUTO_URL = "auto"   # device.url value meaning "find the device on the network"


@dataclass
class CameraConfig:
    backend: str = "auto"
    capture_size: list[int] = field(default_factory=lambda: [2304, 1296])
    preview_size: list[int] = field(default_factory=lambda: [768, 432])
    preview_fps: float = 10.0
    device: int = 0
    jpeg_quality: int = 95
    controls: dict[str, Any] = field(default_factory=dict)


@dataclass
class HardwareConfig:
    buzzer_pin: int | None = 17
    buzzer_type: str = "active"
    buzzer_active_high: bool = True
    passive_frequency: float = 2700.0
    trigger_pin: int | None = None
    mock: bool = False


@dataclass
class InferenceConfig:
    conf_threshold: float = 0.5
    iou_threshold: float = 0.45
    min_report_conf: float = 0.05
    threads: int = 4
    save_inspections: bool = True
    save_annotated: bool = True
    keep_inspections: int = 1000


@dataclass
class TrainConfig:
    trainer: str = "ultralytics"
    base_model: str = "yolo11n.pt"
    epochs: int = 150
    imgsz: int = 640
    batch: int = 16
    patience: int = 40
    val_fraction: float = 0.2
    device: str = "auto"
    workdir: Path = Path("work")


@dataclass
class DeviceConfig:
    """The inspection station's API server, and how other machines reach it."""

    host: str = "0.0.0.0"
    port: int = 8000
    # "auto": find the device on the local network (boxqc.discovery); else its address
    url: str = AUTO_URL
    # The device's name in discovery answers (default: its hostname). On the studio side,
    # the device to pick when several answer.
    name: str | None = None
    discovery: bool = True      # the device answers discovery queries
    api_token: str | None = None


@dataclass
class StudioConfig:
    """The desktop web UI server."""

    host: str = "127.0.0.1"
    port: int = 8080
    serve: bool = False     # serve mode: open to the network behind an access key (studio/access.py)


@dataclass
class Config:
    data_dir: Path = Path("data")
    models_dir: Path = Path("models")
    classes: list[str] = field(default_factory=lambda: ["dirty", "defect"])
    roi: list[float] = field(default_factory=lambda: [0.0, 0.0, 1.0, 1.0])
    plugins: list[str] = field(default_factory=list)   # modules to import at start-up (boxqc.plugins)
    camera: CameraConfig = field(default_factory=CameraConfig)
    hardware: HardwareConfig = field(default_factory=HardwareConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    device: DeviceConfig = field(default_factory=DeviceConfig)
    studio: StudioConfig = field(default_factory=StudioConfig)
    source: Path | None = None   # the YAML file this was loaded from
    overlay: Path | None = None  # the boxqc.local.yaml that was merged over it, if any


_INTERNAL_KEYS = {"source", "overlay"}

# Keys from BoxQC 0.1 and where they live now: (old section, old key) -> (new section, new key)
_LEGACY_KEYS = {
    ("server", "host"): ("device", "host"),
    ("server", "port"): ("device", "port"),
    ("remote", "url"): ("device", "url"),
}


def _check(value: Any, hint: Any, name: str) -> Any:
    """Return ``value`` as the type a config field declares, or raise ``ValueError``.

    YAML decides types by how a value is written (``8000`` is a number, ``"8000"`` and
    ``eight`` are text), so a wrong type is an easy mistake that would otherwise only
    surface later, somewhere unrelated. Harmless differences are accepted: a whole
    number where a decimal is expected, a number where text is expected (``device: 0``).
    """
    origin, args = typing.get_origin(hint), typing.get_args(hint)
    if origin in (typing.Union, types.UnionType):       # "X | None"
        if value is None and type(None) in args:
            return None
        return _check(value, next(a for a in args if a is not type(None)), name)
    if hint is Any:
        return value
    if origin is list:
        if not isinstance(value, list):
            raise ValueError(f"Config key {name} must be a list, e.g. [a, b]")
        return [_check(v, args[0], f"{name}[{i}]") for i, v in enumerate(value)]
    if origin is dict:
        if not isinstance(value, dict):
            raise ValueError(f"Config key {name} must be a mapping")
        return value
    number = isinstance(value, (int, float)) and not isinstance(value, bool)
    if hint is bool and not isinstance(value, bool):
        raise ValueError(f"Config key {name} must be true or false, not {value!r}")
    if hint is int and not (number and isinstance(value, int)):
        raise ValueError(f"Config key {name} must be a whole number, not {value!r}")
    if hint is float:
        if not number:
            raise ValueError(f"Config key {name} must be a number, not {value!r}")
        return float(value)
    if hint is str:
        if number:
            return str(value)
        if not isinstance(value, str):
            raise ValueError(f"Config key {name} must be text, not {value!r}")
    if hint is Path:
        if not isinstance(value, str):
            raise ValueError(f"Config key {name} must be a path, not {value!r}")
        return Path(value)
    return value


def _apply(obj: Any, data: dict[str, Any], prefix: str = "") -> None:
    """Copy ``data`` onto the dataclass ``obj``, rejecting unknown keys and wrong types."""
    hints = typing.get_type_hints(type(obj))
    for key, value in data.items():
        if key not in hints or (not prefix and key in _INTERNAL_KEYS):
            raise ValueError(f"Unknown config key: {prefix}{key}")
        current = getattr(obj, key)
        if dataclasses.is_dataclass(current):
            if value is None:       # a section with every key commented out
                continue
            if not isinstance(value, dict):
                raise ValueError(f"Config key {prefix}{key} must be a mapping")
            _apply(current, value, prefix=f"{prefix}{key}.")
        else:
            setattr(obj, key, _check(value, hints[key], f"{prefix}{key}"))


def _merge(base: dict[str, Any], extra: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge ``extra`` over ``base`` (mappings merge, everything else replaces)."""
    out = dict(base)
    for key, value in extra.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge(out[key], value)
        else:
            out[key] = value
    return out


def _migrate_legacy(data: dict[str, Any], origin: Path) -> dict[str, Any]:
    """Move 0.1-era keys (``server``, ``remote``) to their current place. New keys win."""
    data = dict(data)
    for (old_section, old_key), (new_section, new_key) in _LEGACY_KEYS.items():
        section = data.get(old_section)
        if not isinstance(section, dict) or old_key not in section:
            continue
        target = dict(data.get(new_section) or {})
        target.setdefault(new_key, section[old_key])
        data[new_section] = target
        log.warning("%s: '%s.%s' is deprecated, use '%s.%s'", origin.name, old_section, old_key,
                    new_section, new_key)
    for old_section in {section for section, _ in _LEGACY_KEYS}:
        section = data.get(old_section)
        if isinstance(section, dict):
            rest = {k: v for k, v in section.items() if (old_section, k) not in _LEGACY_KEYS}
            if rest:
                data[old_section] = rest    # leftovers are reported as unknown keys
            else:
                del data[old_section]
    return data


def _read_yaml(path: Path) -> dict[str, Any]:
    try:
        data = yaml.safe_load(path.read_text()) or {}
    except yaml.YAMLError as exc:
        raise ValueError(f"{path} is not valid YAML: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return _migrate_legacy(data, path)


def find_config_file(path: str | os.PathLike | None = None) -> Path | None:
    if path:
        return Path(path)
    if env := os.environ.get(CONFIG_ENV):
        return Path(env)
    for name in DEFAULT_CONFIG_NAMES:
        if (candidate := Path.cwd() / name).is_file():
            return candidate
    return None


def local_overlay_path(config_file: Path) -> Path:
    """``boxqc.yaml`` -> ``boxqc.local.yaml`` (same directory)."""
    return config_file.with_name(f"{config_file.stem}.local{config_file.suffix}")


def load_config(path: str | os.PathLike | None = None) -> Config:
    """Load the configuration. With no file anywhere, the built-in defaults are used."""
    cfg = Config()
    cfg_file = find_config_file(path)
    base = Path.cwd()
    if cfg_file is not None:
        cfg_file = cfg_file.expanduser().resolve()
        data = _read_yaml(cfg_file)
        overlay = local_overlay_path(cfg_file)
        if overlay.is_file():
            data = _merge(data, _read_yaml(overlay))
            cfg.overlay = overlay
        _apply(cfg, data)
        cfg.source = cfg_file
        base = cfg_file.parent
    cfg.data_dir = (base / cfg.data_dir).resolve()
    cfg.models_dir = (base / cfg.models_dir).resolve()
    cfg.train.workdir = (base / cfg.train.workdir).resolve()
    if url := os.environ.get(DEVICE_URL_ENV):
        cfg.device.url = url
    if token := os.environ.get(API_TOKEN_ENV):
        cfg.device.api_token = token
    cfg.device.url = normalize_device_url(cfg.device.url)
    validate_roi(cfg.roi)
    return cfg


def normalize_device_url(url: str) -> str:
    """``http://pi:8000/`` -> ``http://pi:8000``; any spelling of ``auto`` -> ``auto``."""
    url = url.strip().rstrip("/")
    return AUTO_URL if url.lower() == AUTO_URL else url


def validate_roi(roi: Any) -> list[float]:
    """Check a region of interest: normalized ``[x0, y0, x1, y1]`` of the full frame."""
    if not isinstance(roi, (list, tuple)) or len(roi) != 4:
        raise ValueError("roi must be [x0, y0, x1, y1]")
    x0, y0, x1, y1 = (float(v) for v in roi)
    if not (0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1):
        raise ValueError("roi values must satisfy 0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1")
    if x1 - x0 < 0.05 or y1 - y0 < 0.05:
        raise ValueError("roi is too small")
    return [x0, y0, x1, y1]
