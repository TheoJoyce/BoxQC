# Changelog

Notable changes to BoxQC. Versions follow [semantic versioning](https://semver.org/);
the device API has its own version in its path (`/api/v1`).

## Unreleased

### Added

* **The device can be found without knowing its address.** A running device answers a
  discovery broadcast on UDP port 47800. `device.url` now defaults to `auto`: the
  studio and the command line find the device on the local network, and find it again
  when its address changes. `boxqc devices` lists the devices that answer, `device.name`
  picks one when there are several, and `device.discovery: false` turns it off on the
  device. Plain broadcast rather than mDNS/Bonjour, which many managed Wi-Fi networks
  filter between clients ([docs/pi-setup.md](docs/pi-setup.md#finding-the-device),
  protocol: [docs/device-api.md](docs/device-api.md#discovery)).
  `deploy/sync.sh` takes `PI_HOST=<user>@auto` (and `PI_NAME`) to find the Pi the same
  way, and `examples/inspect_client.py` finds the device when given no address.
* **Serve mode: the studio on phones and other computers.** `boxqc studio --serve` (or
  `studio.serve: true`) listens on the network behind an access key, and prints a link
  and a QR code for each of the computer's addresses. The System page shows them too.
  The key is kept across restarts in `<data_dir>/.studio-access-key`; `--new-key`
  replaces it. The studio's own computer needs no key
  ([docs/cli.md](docs/cli.md#boxqc-studio-desktop), [SECURITY.md](SECURITY.md)). The QR
  code uses the small `segno` package, now part of the `studio` extra.
* **Every studio page works on a phone.** No page scrolls sideways down to 320 px wide.
  The top bar is two compact rows, the Inspect button and verdict sit under the live
  view, the dataset shows two images per row, and touch screens get finger-sized
  controls, a larger grab area for box corners, touch instructions instead of keyboard
  hints, and 16 px form fields (so iOS does not zoom in).
* **Labeling assistant.** On the labeling page, a trained model on the studio machine
  proposes boxes for the image as dashed suggestions with a confidence. Click one to
  add it, right-click to dismiss it, or raise the confidence slider and add everything
  still shown with **A**. Suggestions that are not added are never saved. It runs by
  itself on unlabeled images, on request on labeled ones, and needs no device
  ([docs/workflow.md](docs/workflow.md#the-labeling-assistant-let-a-model-draw-the-boxes)).
  Studio routes: `GET /api/assistant`, `GET /api/images/{id}/suggestions`.
* **Configurable buzzer beeps.** The number of beeps and the length of each beep for
  PASS and FAIL are a device setting, `buzzer`, changed on the studio's System page or
  with `PUT /api/v1/settings` ([docs/device-api.md](docs/device-api.md#put-settings)).
  The defaults sound as before; the ERROR chatter is unchanged.
* **Buzzer sound in the browser.** A second checkbox on the Inspect page, off by default,
  plays the verdict's buzzer pattern on the computer running the browser, as the device
  reports it in `GET /api/v1/status` (`hardware.patterns`).

### Changed

* **The default `device.url` is `auto`**, no longer `http://10.42.0.2:8000`. A device
  running older code does not answer discovery, so update the Pi as well, or set
  `device.url` explicitly (the USB link to a Linux host still gives the Pi 10.42.0.2). An
  explicit `device.url`, `--device-url` or `$BOXQC_DEVICE_URL` works as before and
  skips discovery.
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
