"""A small named registry that can also discover plug-ins from installed packages.

BoxQC has two plug-in points, each backed by one :class:`Registry`:

===================  ==========================  =====================================
What                 Entry-point group           Registry
===================  ==========================  =====================================
Model formats        ``boxqc.model_formats``     :data:`boxqc.models.FORMATS`
Trainers             ``boxqc.trainers``          :data:`boxqc.studio.training.TRAINERS`
===================  ==========================  =====================================

A plug-in package advertises a class in its ``pyproject.toml``::

    [project.entry-points."boxqc.trainers"]
    my-trainer = "my_package.trainer:MyTrainer"

Built-in implementations are registered in code, so they work without installed
package metadata (for example straight from a source checkout).

For a plug-in that is not packaged - a single file next to ``boxqc.yaml``, say - list
its module under ``plugins:`` in the config instead; :func:`load_plugins` imports it at
start-up and the module registers itself with ``@register_trainer`` /
``@register_format``.
"""

from __future__ import annotations

import importlib
import logging
import sys
from importlib import metadata
from typing import TYPE_CHECKING, Generic, TypeVar

if TYPE_CHECKING:
    from .config import Config

log = logging.getLogger(__name__)

T = TypeVar("T")


class Registry(Generic[T]):
    """Maps a name to a class. Entry points of ``group`` are loaded on first lookup."""

    def __init__(self, group: str, kind: str):
        self.group = group
        self.kind = kind            # human-readable, for error messages ("trainer")
        self._items: dict[str, type[T]] = {}
        self._discovered = False

    def register(self, cls: type[T], name: str | None = None) -> type[T]:
        """Register ``cls`` under ``name`` (default: its ``name`` attribute). Usable as a decorator."""
        key = name or getattr(cls, "name", None)
        if not key or not isinstance(key, str):
            raise ValueError(f"{cls!r} needs a string `name` to be registered as a {self.kind}")
        self._items[key] = cls
        return cls

    def unregister(self, name: str) -> None:
        self._items.pop(name, None)

    def _discover(self) -> None:
        if self._discovered:
            return
        self._discovered = True
        for entry in metadata.entry_points(group=self.group):
            if entry.name in self._items:
                continue        # built-ins and explicit registrations win
            try:
                self._items[entry.name] = entry.load()
            except Exception:   # a broken plug-in must not take the application down
                log.exception("Could not load %s plug-in %r (%s)", self.kind, entry.name, entry.value)

    def get(self, name: str) -> type[T]:
        self._discover()
        try:
            return self._items[name]
        except KeyError:
            known = ", ".join(sorted(self._items)) or "none"
            raise KeyError(f"Unknown {self.kind} {name!r} (available: {known})") from None

    def names(self) -> list[str]:
        self._discover()
        return list(self._items)

    def items(self) -> list[tuple[str, type[T]]]:
        self._discover()
        return list(self._items.items())

    def __contains__(self, name: str) -> bool:
        self._discover()
        return name in self._items


def load_plugins(cfg: Config) -> list[str]:
    """Import the modules named under ``plugins:`` in the config. Returns those that loaded.

    The config file's directory is put on the module search path first, so a plug-in
    file can simply live beside ``boxqc.yaml``. A module that fails to import is logged
    and skipped.
    """
    if cfg.plugins and cfg.source is not None and str(cfg.source.parent) not in sys.path:
        sys.path.insert(0, str(cfg.source.parent))
    loaded = []
    for name in cfg.plugins:
        try:
            importlib.import_module(name)
            loaded.append(name)
        except Exception:
            log.exception("Could not import plug-in module %r (listed under 'plugins' in the config)", name)
    return loaded
