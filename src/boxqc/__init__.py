"""BoxQC - top-down box interior inspection (dirty / defect detection).

The package has two halves that talk over HTTP:

* :mod:`boxqc.device` - the inspection station (a Raspberry Pi). A self-contained API
  server: camera, ONNX inference, PASS/FAIL buzzer, inspection log.
* :mod:`boxqc.studio` - the desktop side. Web UI, image dataset, labeling and training.

:mod:`boxqc.models` (model formats) and :mod:`boxqc.studio.training` (trainers) are the
two plug-in points. See ``docs/architecture.md``.
"""

__version__ = "0.2.0"
