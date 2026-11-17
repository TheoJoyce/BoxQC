# Working on the code

For anyone who forks BoxQC or wants to change it. Read
[architecture.md](architecture.md) first; this page is the practical side.

## Setup

Python 3.11 or newer. No Raspberry Pi is needed: the camera and the buzzer are mocked.

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

pytest                                                       # about a second
ruff check .
for f in src/boxqc/studio/static/*.js; do node --check "$f"; done
```

The `dev` extra installs `node` into the virtualenv for the last line. CI
(`.github/workflows/ci.yml`) runs the same three checks on Python 3.11 to 3.13, plus a
job that proves the device still starts with only the `device` extra installed.

To click through the application, run both halves on one machine:

```bash
boxqc device --host 127.0.0.1 &
boxqc studio --device-url http://127.0.0.1:8000        # http://localhost:8080
```

The mock camera produces synthetic boxes, some with dark blobs, so capture, labeling
and inspection all have something to work with. Training for real needs
`pip install -e ".[dev,train]"` (PyTorch, large); the test suite does not.

Started from one folder like this, the device and the studio share `data/` and
`models/`, which two machines never do. Two effects to know about: a trained model is
"on the device" as soon as it is written, and deleting the studio's copy of a model
deletes the device's file too. To keep them apart, give one of them its own
`--config` in another folder.

## Rules that keep the design intact

* **The device stays small and independent.** `boxqc.device` must not import
  `boxqc.studio`, and must run with the `device` extra only: no PyTorch, Jinja2 or
  httpx on the Pi. CI checks this.
* **The studio reaches the device only through the device API**: from Python through
  `boxqc.studio.device_client`, from the browser through the `/api/device/` proxy.
* **The device API is a public contract.** Within `/api/v1`, add fields and routes; do
  not remove or repurpose them. Every route must be described in
  [device-api.md](device-api.md) (a test fails otherwise).
* **`ModelAdapter` and `Trainer` are public contracts too.** Other people's plug-ins
  subclass them; change them only in ways that keep existing subclasses working.
* **Heavy libraries are imported where they are used**, inside functions, so importing
  a module never pulls in PyTorch, a camera stack or GPIO.
* **Hardware code cannot be tested in CI.** `device/camera.py` and `device/hardware.py`
  only run for real on a Pi. Keep changes there small and try them on a board.

## Tests

| File | Covers |
|---|---|
| `tests/test_config.py` | config loading, overlay, legacy keys, runtime settings, studio state |
| `tests/test_models.py` | output decoding, the ONNX adapter, the format registry |
| `tests/test_dataset.py` | the image store and the training export |
| `tests/test_device_api.py` | every device route, against the device app with a mock camera |
| `tests/test_studio.py` | the studio app connected to an in-process device, and to one that is off |
| `tests/test_training.py` | the trainer contract, the pipeline and the job manager |
| `tests/test_examples.py` | the plug-ins under `examples/` |
| `tests/test_docs.py` | the documentation matches the code (routes, config keys, commands, links) |

Fixtures in `tests/conftest.py`: `device` (a running device app), `studio` (a studio
connected to it), `offline_studio` (a studio whose device refuses connections), and
`cfg` / `device_cfg` (configs in a temp directory). `tests/helpers.py` has `make_onnx`,
which writes a tiny ONNX model with a fixed output, a `FakeDetector`, and a
`FakeTrainer` that makes the whole training pipeline run in milliseconds.

## Common changes

**Add a device route.** Define it on the `api` router in `device/app.py` with a `tags`
and `summary`, add a method to `DeviceClient` if the studio needs it, document it in
`docs/device-api.md`, and test it in `tests/test_device_api.py`. The browser can
already reach it at `/api/device/<path>`.

**Add a studio page.** Add a template extending `base.html`, a script in `static/`, a
route in the *pages* section of `studio/app.py`, and a tab in `base.html`. Use `api()`
and `el()` from `common.js`.

**Add a setting.** Add the field to the right dataclass in `config.py` and a commented
entry to `boxqc.yaml` (a test checks that every key is there). For a value that must
change at runtime, extend `RuntimeSettings` (device) or `StudioState` (studio) instead.

**Add a model format or a trainer.** See [models.md](models.md) and
[trainers.md](trainers.md). Most plug-ins can live outside this repository.

**Support another camera.** Subclass `Camera` in `device/camera.py` and add a branch to
`open_camera`.

## Conventions

* Python: `ruff check` with the settings in `pyproject.toml`, 120 columns, type hints
  on public functions. Docstrings say what a caller needs to know; comments say why
  something is the way it is.
* JavaScript: plain scripts, no build step, no framework. `"use strict"`, and DOM built
  with `el()` rather than HTML strings, so untrusted text is never parsed as markup.
  A page's `keydown` handler starts with `if (notAHotkey(e)) return;` (`common.js`), so
  that typing in a field and browser shortcuts such as Ctrl/Cmd+C never trigger a hotkey.
* Files that are rewritten in place (annotations, `datasets.json`, both settings files)
  and inspection records are written atomically: temp file, then rename. A model upload
  is received in a staging folder and moved into `models/` only once it has passed its
  test inference. Image files are written once, under an id that cannot be taken.
* The studio refuses state-changing requests that a browser sends for another site
  (`SameOriginMiddleware` in `studio/app.py`). A new route needs nothing extra, as long
  as reads stay on `GET`.
* User-visible changes go into [CHANGELOG.md](../CHANGELOG.md), under a heading for the
  next version.
* Keep dependencies few, above all the device's.

## Releasing a version

The version is in two places: `__version__` in `src/boxqc/__init__.py` and `version`
in `pyproject.toml`. Update both and the changelog. The device API version
(`API_VERSION` in `device/app.py`) changes only with an incompatible API change.
