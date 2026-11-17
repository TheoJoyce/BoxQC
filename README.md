# BoxQC

Camera-based PASS/FAIL inspection of open boxes. A camera looks straight down into a
box, an object-detection model finds **dirty** areas and **defects** on the interior
walls and floor, and a Raspberry Pi sounds a **PASS** or **FAIL** buzzer. The model
draws a box around each problem area, so a failure can be explained and corrected.

BoxQC covers the whole loop: capture photos, label them, train a model, deploy it,
inspect, and feed mistakes back into the next training round.

> **Project status.** This repository is a complete, working snapshot and is not under
> active development here. It is published to be used and built upon: to take it
> further, fork it. The code is commented and documented with that in mind; start at
> [docs/](docs/README.md).

## How it is put together

```
 ┌────────── device: Raspberry Pi ──────────┐           ┌────────── studio: desktop ──────────┐
 │  boxqc device                            │   HTTP    │  boxqc studio                       │
 │                                          │  /api/v1  │                                     │
 │  camera ─► ONNX model ─► PASS / FAIL     │ ◄──────── │  web UI   capture · label · inspect │
 │  buzzer + trigger button (GPIO)          │           │  dataset  images + annotations      │
 │                                          │  photos   │  training pluggable trainers        │
 │  keeps: models, 3 settings,              │ ────────► │  models   trained here,             │
 │         a bounded inspection log         │ ◄──────── │           deployed to the device    │
 └──────────────────────────────────────────┘  models   └─────────────────────────────────────┘
```

* **The device** is a self-contained API server. With a model loaded it inspects boxes
  on a button press or an API call, whether or not anything else is connected. It has
  no web UI and stores no training data, so it needs neither PyTorch nor much disk.
* **The studio** runs on a desktop or laptop: the web interface, the image dataset,
  labeling and training. It talks to the device only through the device API, and keeps
  working (labeling, training) while the device is switched off.
* The **device API** is documented and versioned, so a PLC, a script or your own
  front end can use it directly: [docs/device-api.md](docs/device-api.md).
* **Model formats** and **trainers** are plug-ins. The built-ins are YOLO-family ONNX
  models and an Ultralytics YOLO trainer; adding another is one class
  ([docs/models.md](docs/models.md), [docs/trainers.md](docs/trainers.md)).

More detail: [docs/architecture.md](docs/architecture.md).

## Quick start

### Try it without any hardware

Both halves run on one computer, with a synthetic camera and a mocked buzzer.
Python 3.11 or newer is required.

```bash
git clone <this repository> boxqc && cd boxqc
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[studio]"

boxqc device --host 127.0.0.1 &                        # the "Pi": mock camera on port 8000
boxqc studio --device-url http://127.0.0.1:8000        # then open http://localhost:8080
```

Capture a few images on the **Capture** page and label them under **Dataset**. Run
from one folder like this, the two halves share its `data/` and `models/` directories;
on a real installation each machine has its own. To train as well, install the built-in
trainer (this pulls in PyTorch):

```bash
pip install -e ".[studio,train]"
```

### Real setup

| Where | Install | Run |
|---|---|---|
| Raspberry Pi with camera and buzzer | `./deploy/pi/setup_pi.sh` (installs a systemd service) | `boxqc device` (started by the service) |
| Desktop or laptop | `pip install -e ".[studio,train]"` | `boxqc studio` |

1. Set up the Pi: [docs/pi-setup.md](docs/pi-setup.md) (hardware, installation, the
   optional USB network link).
2. On the desktop, tell the studio where the device is. Either pass
   `--device-url http://<pi-address>:8000` each time, or create `boxqc.local.yaml` next
   to `boxqc.yaml`:

   ```yaml
   device:
     url: http://raspberrypi.local:8000
   ```

3. Run `boxqc studio` and open <http://localhost:8080>.

Training runs on the computer the studio runs on. To train on a GPU workstation and
click from a laptop, run the studio on the workstation with `--host 0.0.0.0` and open
it from the laptop's browser.

## Using it

The full walk-through is in [docs/workflow.md](docs/workflow.md). In short:

1. **System** page: drag the region of interest (ROI) around the box interior. The
   model only ever sees this region.
2. **Capture** page: photograph clean and defective boxes (keys **C** / **D**). Aim
   for 150 to 300 images to start, about half of them clean.
3. **Dataset** page: draw a box around each defect. Group images into named datasets
   to choose what a model learns from.
4. **Training** page: start a run and follow its progress. The finished model is scored
   as one PASS/FAIL decision per validation image, a threshold is recommended, and the
   model is deployed to the device.
5. **Inspect** page, the trigger button, or `POST /api/v1/inspect`: inspect a box.
   Send any image the model got wrong back to the dataset and retrain.

Everything the pages do is also available without a browser:

```bash
boxqc train --dataset "line 2" --deploy     # prepare, train, export, evaluate, deploy
boxqc deploy models/my-model.onnx           # upload a model (and its metadata) to the device
boxqc export --format coco --out my-data    # the labeled images as a YOLO / COCO dataset
boxqc inspect photo1.jpg photo2.jpg         # run a model on image files
boxqc plugins                               # installed model formats and trainers
```

## Integrating and extending

| You want to | Read |
|---|---|
| Trigger inspections from a PLC, a script or another application | [docs/device-api.md](docs/device-api.md), [examples/inspect_client.py](examples/inspect_client.py) |
| Run a model trained elsewhere | [docs/models.md](docs/models.md): the model file and metadata contract |
| Support another model architecture or runtime | [docs/models.md](docs/models.md): writing a model format |
| Train with another framework or an external program | [docs/trainers.md](docs/trainers.md), [examples/command_trainer.py](examples/command_trainer.py) |
| Understand the code before changing it | [docs/architecture.md](docs/architecture.md), [docs/development.md](docs/development.md) |
| Read or produce BoxQC's files with other tools | [docs/data-formats.md](docs/data-formats.md) |

## Configuration

Everything is in [boxqc.yaml](boxqc.yaml), with comments. The same file serves both
halves; each reads the sections it needs. Machine-specific values go in
`boxqc.local.yaml` (same structure, never committed), and `BOXQC_DEVICE_URL` /
`BOXQC_API_TOKEN` override both. Details: [docs/configuration.md](docs/configuration.md).

The ROI, the FAIL threshold and the active model change at runtime. The device owns
them and stores them in `data/settings.json`; change them from the web UI or through
the device API (`PUT /api/v1/settings` for the ROI and the threshold,
`POST /api/v1/models/{name}/activate` for the model).

By default the device API is open to anyone who can reach the Pi, which suits an
isolated link between the Pi and one computer. On a shared network, set an API token:
see [docs/device-api.md](docs/device-api.md#authentication).

## Repository layout

```
boxqc.yaml               configuration for both halves
src/boxqc/
  config.py              boxqc.yaml -> Config
  imaging.py             ROI cropping, drawing, JPEG helpers
  plugins.py             the plug-in registry
  cli.py                 the `boxqc` command
  models/                model formats: the contract, the registry, YOLO ONNX      (plug-in point)
  device/                the Pi: camera, buzzer, inspection pipeline, device API
  studio/                the desktop: web UI, dataset store, device client
    training/            trainer contract, pipeline, job manager, Ultralytics      (plug-in point)
deploy/                  Pi installation, USB network link, code sync
docs/                    architecture, API reference, guides (index: docs/README.md)
examples/                API client and plug-in examples
tests/                   run against a mock camera; no hardware needed
```

## Development

```bash
pip install -e ".[dev]"
pytest                   # the device and the studio are tested against a mock camera
ruff check .
for f in src/boxqc/studio/static/*.js; do node --check "$f"; done
```

[docs/development.md](docs/development.md) covers the test fixtures, the rules that
keep the two halves independent, and how to make common changes. Upgrading an
installation of BoxQC 0.1, where the web UI and the dataset lived on the Pi:
[docs/upgrading.md](docs/upgrading.md).

## Documentation

| | |
|---|---|
| [docs/workflow.md](docs/workflow.md) | Using it: ROI, capture, label, train, inspect |
| [docs/pi-setup.md](docs/pi-setup.md) | Hardware, installing the device on a Raspberry Pi, USB link, troubleshooting |
| [docs/configuration.md](docs/configuration.md) | `boxqc.yaml`, per-machine overrides, runtime settings |
| [docs/cli.md](docs/cli.md) | Every `boxqc` command and option |
| [docs/device-api.md](docs/device-api.md) | The device's HTTP API |
| [docs/models.md](docs/models.md) | Model files, metadata, model-format plug-ins |
| [docs/trainers.md](docs/trainers.md) | The training pipeline, trainer plug-ins |
| [docs/data-formats.md](docs/data-formats.md) | Every file BoxQC stores |
| [docs/architecture.md](docs/architecture.md) | How the code is organised and why |
| [docs/development.md](docs/development.md) | Working on the code |
| [docs/upgrading.md](docs/upgrading.md) | From BoxQC 0.1 to 0.2 |
| [CHANGELOG.md](CHANGELOG.md) | What changed between versions |

## Licence

BoxQC is licensed under the [Apache License 2.0](LICENSE): you may use, modify, fork
and redistribute it freely, including commercially, as long as the licence and
copyright notices are kept.

One optional dependency has different terms. The built-in trainer uses
[Ultralytics YOLO](https://github.com/ultralytics/ultralytics), which, like its
pretrained weights, is published under AGPL-3.0 with a commercial licence available from
Ultralytics. Those terms apply to that trainer and may apply to models trained with it;
check them against your use. Nothing else in BoxQC depends on Ultralytics: the device
runs models with ONNX Runtime, and you can train with a different backend
([docs/trainers.md](docs/trainers.md)).
