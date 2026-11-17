# Architecture

BoxQC is two programs from one Python package. They share a little code and otherwise
meet only at an HTTP API.

```
            device  (boxqc device, on the Pi)                    studio  (boxqc studio, on a desktop)
 ┌───────────────────────────────────────────────┐      ┌─────────────────────────────────────────────────┐
 │ device/app.py        device API  /api/v1      │◄─────┤ studio/device_client.py   blocking client       │
 │   ├ device/camera.py     Picamera2 / OpenCV   │ HTTP │ studio/app.py             web pages + studio API │
 │   ├ device/hardware.py   buzzer, button       │      │   ├ /api/device/*  ──►    forwarded to device   │
 │   ├ device/inspector.py  crop ► detect ► log  │      │   ├ studio/dataset.py     images + annotations  │
 │   └ device/settings.py   ROI, threshold, model│      │   ├ studio/export.py      dataset ► YOLO / COCO │
 │                                               │      │   └ studio/training/      trainers, runs        │
 └────────────────────┬──────────────────────────┘      └───────────────────┬─────────────────────────────┘
                      │                                                     │
                      └────────────── models/  (shared) ───────────────────┘
                             model formats: load a file, run it
```

## Why two programs

Inspection has to keep working on the line without a laptop attached, on a small
computer. Training needs a GPU, gigabytes of libraries and a growing pile of images.
Putting the boundary between them gives each side only what it needs:

| | Device | Studio |
|---|---|---|
| Runs on | Raspberry Pi | desktop / laptop / GPU workstation |
| Must be running to inspect | yes | no |
| Python dependencies | FastAPI, ONNX Runtime, OpenCV, camera + GPIO libraries | the above plus Jinja2, httpx, and a trainer's libraries (PyTorch, ...) |
| Stores | models, `settings.json`, the last N inspections | every training image, annotations, training runs, trained models |
| Imports from the other half | nothing | nothing (it calls the device API) |

The rule that keeps this true: **`boxqc.device` never imports `boxqc.studio`, and
`boxqc.studio` reaches the device only through `boxqc.studio.device_client`.** Both may
import the shared modules (`config`, `imaging`, `ids`, `plugins`, `models`).

## The device

`device/app.py` builds a FastAPI application around one `DeviceState` object, which
owns the camera, the buzzer, the active model, the runtime settings and the inspection
log.

An inspection is the same whether it starts from the GPIO button, the API or an uploaded
image:

```
frame ─► crop to the ROI ─► model.detect() ─► shift boxes back to frame coordinates
      ─► verdict: FAIL if any detection >= threshold, else PASS (NO_MODEL without a model)
      ─► buzzer pattern ─► record + photo written to data/inspections/ (oldest pruned)
```

`Inspector` (in `device/inspector.py`) does the first three steps, `DeviceState.inspect`
the rest. Detections down to `inference.min_report_conf` are kept in the record even
when they are below the threshold; that is what lets a client show near-misses and
re-evaluate a stored inspection at a different threshold without running the model
again.

Three values change while the device runs and therefore live on the device, in
`data/settings.json` (`device/settings.py`): the FAIL threshold, the ROI and the name
of the active model. `boxqc.yaml` only supplies their initial values.

The API is served under `/api/v1`. The same routes also answer under the unversioned
`/api` prefix that BoxQC 0.1 used. See [device-api.md](device-api.md).

## The studio

`studio/app.py` serves the web pages (Jinja2 templates in `studio/templates`, plain
JavaScript in `studio/static`, no build step) and the JSON API behind them.

The browser only ever talks to the studio. That keeps the pages on one origin and the
device's API token out of the browser. Requests fall into three groups:

| Path | Handled by | Example |
|---|---|---|
| `/api/device/...` | forwarded unchanged to the device's `/api/v1/...`, streaming | the live camera view, `POST /api/device/inspect`, ROI and threshold changes |
| `/api/...` that needs both sides | the studio, calling the device | `POST /api/capture`: ask the device for a photo, store it in the dataset |
| the rest of `/api/...` | the studio alone | images, datasets, training runs |

If the device does not answer, forwarded requests return `502` with a readable message
and the studio-only parts keep working. `DeviceLink` caches the reachability check for
a few seconds so an absent device costs one timeout, not one per request.

The studio's own `/api` routes exist to serve its pages. They are documented at the
studio's `/docs`, but unlike the device API they are not a versioned public interface.

### Data on the studio machine

```
data/images/<id>.jpg           one photo per item, exactly as the device's camera encoded it
data/annotations/<id>.json     boxes (normalized to the full frame), status, datasets it belongs to
data/datasets.json             the named datasets
data/thumbs/                   thumbnail cache
data/studio.json               capture dataset; the device's ROI as last seen
models/<name>.onnx + .json     trained models and their metadata
work/                          prepared datasets, training runs, downloaded weights
```

Plain files, one per image: easy to inspect, back up and hand to other tools.
`studio/dataset.py` is the only code that reads or writes the first four.

### Training

```
TrainingManager (jobs.py)            one run at a time, in a worker thread; feeds the Training page
  └─ run_training (pipeline.py)      the steps every trainer shares
       ├─ prepare_dataset (export.py)    labeled images ─► ROI crops + labels, train/val split
       ├─ Trainer.train (plug-in)        train, export a model file
       ├─ evaluate_image_level           PASS/FAIL per validation image, at each threshold
       └─ write the metadata sidecar     classes, ROI, metrics, recommended threshold
  └─ DeviceClient.push_model         deploy (optional)
```

`boxqc train` calls the same `run_training` without the manager.

Two details matter for correctness:

* **The ROI is part of the model.** Images are cropped to the ROI before training, and
  the device crops the same way before inference. A run therefore asks the device for
  its current ROI and records it in the model's metadata; the studio warns when a
  model's ROI no longer matches the device's.
* **The train/val split is deterministic** (a hash of the image id), so an image never
  moves between the two from one run to the next and scores stay comparable.

## Plug-in points

Both are a base class plus a registry (`plugins.py`). A plug-in registers itself with a
decorator, through a Python entry point, or by being listed under `plugins:` in
`boxqc.yaml`.

| | Base class | Registry | Guide |
|---|---|---|---|
| Model format | `boxqc.models.ModelAdapter` | `boxqc.models.FORMATS` | [models.md](models.md) |
| Trainer | `boxqc.studio.training.Trainer` | `boxqc.studio.training.TRAINERS` | [trainers.md](trainers.md) |

A model format is used on both sides: by the device to inspect, and by the studio to
evaluate a freshly trained model exactly as the device will run it. A trainer is used
only by the studio.

Cameras are a third, smaller seam: `device/camera.py` has a `Camera` base class with
Picamera2, OpenCV and mock implementations, chosen by `camera.backend`.

## Module map

| Module | Responsibility |
|---|---|
| `boxqc/config.py` | `boxqc.yaml` + `boxqc.local.yaml` + environment ► `Config` dataclasses |
| `boxqc/imaging.py` | ROI cropping, JPEG encode/decode, drawing boxes and guides |
| `boxqc/ids.py` | sortable ids and timestamps |
| `boxqc/plugins.py` | `Registry`, entry-point discovery, `load_plugins` |
| `boxqc/models/base.py` | `Detection`, `ModelAdapter` (the model contract) |
| `boxqc/models/registry.py` | metadata sidecars, format selection, `load_model`, `list_models` |
| `boxqc/models/yolo_onnx.py` | the built-in format: YOLO-family ONNX on ONNX Runtime |
| `boxqc/device/app.py` | the device API and `DeviceState` |
| `boxqc/device/inspector.py` | `Inspector`, `InspectionLog`, the verdict rule |
| `boxqc/device/camera.py`, `hardware.py`, `settings.py` | camera backends; buzzer and button; runtime settings |
| `boxqc/studio/app.py` | web pages, studio API, device proxy |
| `boxqc/studio/dataset.py` | `DatasetStore`: images, annotations, named datasets |
| `boxqc/studio/device_client.py` | `DeviceClient`: Python client for the device API |
| `boxqc/studio/export.py` | labeled images ► YOLO / COCO training data |
| `boxqc/studio/evaluate.py` | image-level PASS/FAIL evaluation |
| `boxqc/studio/state.py` | studio state that survives a restart |
| `boxqc/studio/training/base.py` | `Trainer`, `TrainRequest`, `TrainContext` (the trainer contract) |
| `boxqc/studio/training/pipeline.py` | `run_training` |
| `boxqc/studio/training/jobs.py` | `TrainingManager` |
| `boxqc/studio/training/ultralytics_yolo.py` | the built-in trainer |
| `boxqc/cli.py` | the `boxqc` command |

## Testing

The tests run both applications in-process with the mock camera and no GPIO. The
studio fixture is connected to a device fixture, so `tests/test_studio.py` exercises
the real HTTP contract between the two (including the proxy). Training is tested with
a stand-in trainer that writes a tiny hand-built ONNX model, so the full pipeline runs
in under a second without PyTorch. See `tests/conftest.py` and `tests/helpers.py`.
