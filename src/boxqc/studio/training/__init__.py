"""Training: pluggable backends behind one pipeline.

* :mod:`.base` - the :class:`Trainer` interface a backend implements
* :mod:`.pipeline` - the shared steps around it (dataset, evaluation, metadata)
* :mod:`.jobs` - runs the pipeline in the background for the web UI
* :mod:`.ultralytics_yolo` - the built-in backend
"""

from ...plugins import Registry
from .base import (
    STOP_MODES,
    TrainContext,
    Trainer,
    TrainerOutput,
    TrainerUnavailable,
    TrainingCancelled,
    TrainRequest,
)
from .ultralytics_yolo import UltralyticsTrainer

#: Every known trainer, by name. Third-party trainers join through the
#: ``boxqc.trainers`` entry-point group or :func:`register_trainer`.
TRAINERS: Registry[Trainer] = Registry("boxqc.trainers", "trainer")


def register_trainer(cls: type[Trainer]) -> type[Trainer]:
    """Class decorator: make a :class:`Trainer` available under its ``name``."""
    return TRAINERS.register(cls, cls.name)


def get_trainer(name: str) -> type[Trainer]:
    """Look a trainer up by name. Raises ``KeyError`` naming the available ones."""
    return TRAINERS.get(name)


register_trainer(UltralyticsTrainer)

__all__ = [
    "STOP_MODES",
    "TRAINERS",
    "TrainContext",
    "TrainRequest",
    "Trainer",
    "TrainerOutput",
    "TrainerUnavailable",
    "TrainingCancelled",
    "UltralyticsTrainer",
    "get_trainer",
    "register_trainer",
]
