# Changelog

Notable changes to BoxQC. Versions follow [semantic versioning](https://semver.org/);
the device API has its own version in its path (`/api/v1`).

## Unreleased

### Added

* **Labeling assistant.** On the labeling page, a trained model on the studio machine
  proposes boxes for the image as dashed suggestions with a confidence. Click one to
  add it, right-click to dismiss it, or raise the confidence slider and add everything
  still shown with **A**. Suggestions that are not added are never saved. It runs by
  itself on unlabeled images, on request on labeled ones, and needs no device
  ([docs/workflow.md](docs/workflow.md#the-labeling-assistant-let-a-model-draw-the-boxes)).
  Studio routes: `GET /api/assistant`, `GET /api/images/{id}/suggestions`.

### Changed

* On the labeling page the lists of boxes and suggestions scroll, so the Save buttons
  stay in reach on an image with many boxes, and the right mouse button no longer
  starts drawing a box.

## 0.2.0

The system is split in two. See [docs/upgrading.md](docs/upgrading.md) for the steps.

### Changed

* **The Pi runs a self-contained device API** (`boxqc device`) and nothing else: camera,
  inference, buzzer, models, runtime settings and a bounded inspection log. It no longer
  serves the web UI or stores training images.
* **The web UI, the dataset and training run on a desktop** (`boxqc studio`). The
  studio reaches the device over HTTP, forwards the browser's device requests, and
  keeps working for labeling and training while the device is off.
* Training runs inside the studio. `boxqc agent`, `boxqc pull` and the check-in protocol
  between the Pi and a training machine are gone.
* The device API is versioned under `/api/v1`. The 0.1 paths still answer.
* Configuration: `server:` and `remote:` became `device:`; a `studio:` section was
  added. The old keys are still read, with a warning.
* Model files and their metadata are no longer tied to the configured class list: a
  model carries its own class names.
* `camera.backend: auto` uses the mock camera only where Picamera2 is not installed. A
  Pi camera that fails to open is reported (`camera_error`, `503`, the ERROR buzzer)
  instead of being replaced by synthetic images.
* The USB link to a Linux host uses `10.42.0.1` (host) and `10.42.0.2` (Pi, the new
  default `device.url`), the address the current `rpi-usb-gadget` looks for.
  `linux_usb_share.sh` needs `--pi-name`, and `sync.sh` needs `PI_HOST`.
* Configuration values are type-checked at start-up, and a YAML syntax error is
  reported as a configuration error.
* `boxqc serve`, `push-model` and `prepare` are now `device`, `deploy` and `export`
  (the old names remain as aliases); `boxqc train --pull --push` is
  `boxqc train --deploy`.

### Added

* Plug-in point for **model formats** (`boxqc.models.ModelAdapter`), with a metadata
  key to select the format per model. The built-in YOLO ONNX format now also reads
  YOLOv5-style outputs with objectness, channels-last outputs and float16 inputs, and
  reports undecodable models with a clear message.
* Plug-in point for **trainers** (`boxqc.studio.training.Trainer`). The Ultralytics
  trainer is the built-in; `examples/command_trainer.py` runs any external program.
* Plug-ins load from entry points or from a `plugins:` list in `boxqc.yaml`;
  `boxqc plugins` lists them.
* Models are validated with a test inference when uploaded to the device and when
  activated; a model that fails leaves the active one in place.
* `GET /api/v1/models` reports each file's `sha256`. Deploying compares content, not
  only names, so a reused name can no longer activate an older model.
* The studio refuses cross-site requests that change something (see SECURITY.md).
* `/etc/default/boxqc` on the Pi for the API token; `setup_pi.sh` can be re-run without
  losing it.
* The Training page lists models on both the studio machine and the device, and deploys
  with one click. A finished run stays on the studio machine if the device is offline.
* `boxqc export --format coco`, and `POST /api/v1/camera/capture` on the device.
* Optional API token for the device (`BOXQC_API_TOKEN`).
* `boxqc.local.yaml` overlay and `BOXQC_DEVICE_URL` for per-machine settings.
* `inference.save_annotated: false` stores one image per inspection instead of two; the
  annotated image is then rendered on request.
* `deploy/sync.sh fetch-data` to bring a 0.1 dataset over from the Pi.
* Documentation for third parties under `docs/`: architecture, device API reference,
  model and trainer guides, data formats, configuration, command reference,
  development guide, and examples. Tests keep it in step with the code.
* Apache-2.0 licence and a CI workflow.

### Fixed

* Reusing a run name no longer overwrites the earlier model.
* Images added within the same second (a bulk upload) can no longer overwrite each
  other.
* `boxqc export --out DIR` refuses to replace a folder that is not an earlier export.
* No threshold is recommended when the validation images include no defective one
  (it used to be 0.90).
* Images labeled with a class that was since renamed or removed are left out of
  training instead of being trained on as clean.
* On the labeling and capture pages, browser shortcuts such as Ctrl/Cmd+C no longer
  trigger the page's hotkeys (C = "clean" erased the boxes), and hotkeys keep working
  after a checkbox, slider or dropdown was used.
* `sync.sh push` no longer deletes the Pi's data when `data_dir` or `models_dir` were
  renamed, and reports "up to date" correctly with macOS's rsync.
* Pretrained weights are kept in `work/weights/` instead of the project root.

## 0.1.0

Initial version: capture, labeling, inspection and model management in a web UI served
by the Pi; training on a separate machine through `boxqc agent`.
