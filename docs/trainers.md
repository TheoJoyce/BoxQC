# Trainers

A *trainer* is the part of a training run that is specific to one framework: it turns a
prepared dataset into a model file. BoxQC ships one, built on Ultralytics YOLO. This
page explains how training is organised and how to add another backend.

If all you want is to train **outside** BoxQC and bring the model in, you do not need a
trainer at all: export the dataset with `boxqc export`, train however you like, and
upload the result ([models.md](models.md#running-a-model-trained-elsewhere)).

## What the pipeline does for you

Every run, from the Training page or `boxqc train`, goes through
[`run_training`](../src/boxqc/studio/training/pipeline.py):

| Step | Who | What |
|---|---|---|
| 1. prepare | pipeline | Labeled images are cropped to the ROI and written as a YOLO-layout dataset with a fixed train/val split. |
| 2. train + export | **trainer** | Train, then write the best model as one file in a [model format](models.md) the device can run. |
| 3. install | pipeline | The file is copied into `models/` under the run's name. |
| 4. evaluate | pipeline | The exported model is run over the validation images the way the device runs it: one PASS/FAIL per image, at each threshold. A threshold is recommended. |
| 5. describe | pipeline | The metadata sidecar is written (classes, ROI, metrics, evaluation, provenance). |
| 6. deploy | job manager | Optionally uploaded to the device and activated. |

So a trainer implements step 2 and nothing else.

## The interface

```python
from boxqc.studio.training import Trainer, TrainerOutput, register_trainer

@register_trainer
class MyTrainer(Trainer):
    name = "my-trainer"                     # used in requests, boxqc.yaml and model metadata
    label = "My framework"                  # shown on the Training page
    base_models = (("small", "small: fast"), ("large", "large: accurate"))
    devices = (("auto", "auto"), ("cpu", "cpu"))

    @classmethod
    def availability(cls):
        """None if it can run here, else a sentence saying what is missing."""

    def validate(self, request):
        """Raise ValueError for settings this trainer cannot use."""

    def train(self, request, dataset, ctx):
        ...
        return TrainerOutput(model_path=path_to_exported_model, format="yolo-onnx",
                             metrics={"mAP50": 0.93}, info={"run_dir": "..."})
```

The pieces, all in [base.py](../src/boxqc/studio/training/base.py):

**`request: TrainRequest`** holds `epochs`, `imgsz`, `batch`, `base_model`, `device`,
`dataset`, `name` (always set, and safe as a directory name) and `options`, a free-form
dict passed through from the API for settings only your trainer understands.

**`dataset: PreparedDataset`** is the dataset on disk
([export.py](../src/boxqc/studio/export.py)):

```
dataset.root/
  data.yaml                       Ultralytics-style description        (dataset.yaml_path)
  images/train/*.jpg  images/val/*.jpg       ROI crops
  labels/train/*.txt  labels/val/*.txt       YOLO labels: class cx cy w h, normalized
```

`dataset.classes` are the class names, `dataset.samples` the same boxes as Python
objects in pixels, and `write_coco(dataset)` adds
`annotations/instances_{train,val}.json` if your framework prefers COCO.

**`ctx: TrainContext`** is the line back to the user:

| Call | Effect |
|---|---|
| `ctx.log(line)` | Appends to the run's log. Use `echo=False` for text your library already printed. |
| `ctx.epoch(metrics, total_epochs=n)` | Reports a finished epoch. `metrics` needs `epoch` (1-based); `mAP50`, `mAP50-95`, `precision`, `recall`, `box_loss`, `cls_loss` and `elapsed_s` are charted. |
| `ctx.stage("exporting")` | Announce that training is over and the export has begun. |
| `ctx.stop_requested` | `"early"` (finish the current epoch, keep the best model), `"cancel"` or `None`. |
| `ctx.check_cancelled()` | Raises `TrainingCancelled` on cancel. Call it often and let it propagate. |

**`TrainerOutput`**: `model_path` is the exported file, `format` names its model format,
`metrics` are the final detection metrics, and `info` is merged into the model's
metadata sidecar.

**`self.cfg`** is the loaded configuration; write run artefacts under
`self.cfg.train.workdir`.

Rules worth following:

* Import your framework inside the methods, never at module level. The studio imports
  every trainer at start-up, and labeling must work on machines without it.
  `availability()` should not import it either; `importlib.util.find_spec` is enough.
* Do not write into `models/` or touch the metadata; the pipeline does that.
* Raise ordinary exceptions for failures. The message is shown on the Training page
  with the last lines of the traceback in the log.

## Examples

* [ultralytics_yolo.py](../src/boxqc/studio/training/ultralytics_yolo.py): the built-in
  trainer, with per-epoch progress and early stopping through framework callbacks.
* [examples/command_trainer.py](../examples/command_trainer.py): runs any external
  program on the prepared dataset and collects the model it writes. A good starting
  point when your training code is a script, a container or another language.
* `FakeTrainer` in [tests/helpers.py](../tests/helpers.py): the smallest possible
  trainer, used by the test suite.

## Making it available

A trainer is only needed on the studio machine.

*As a file.* Put the module next to `boxqc.yaml` (or anywhere importable) and list it:

```yaml
plugins: [my_trainer]
train:
  trainer: my-trainer        # make it the default (optional)
```

*As a package.* Declare an entry point, then `pip install` the package:

```toml
[project.entry-points."boxqc.trainers"]
my-trainer = "my_package.trainer:MyTrainer"
```

`boxqc plugins` lists the installed trainers and whether each can run. With more than
one installed, the Training page shows a trainer selector; from the command line use
`boxqc train --trainer my-trainer`.

## Starting runs from other programs

The studio's API can be scripted. Start a run and poll it:

```bash
curl -X POST http://localhost:8080/api/training/jobs -H 'Content-Type: application/json' -d '{
  "epochs": 150, "imgsz": 640, "batch": 16, "base_model": "yolo11n.pt",
  "dataset": "line 2", "deploy": true, "activate": true, "options": {}}'
curl http://localhost:8080/api/training/jobs/<id>          # state, epoch, history, log, result
curl -X POST http://localhost:8080/api/training/jobs/<id>/stop -d '{"mode": "early"}' -H 'Content-Type: application/json'
```

`trainer` selects a backend (default: `train.trainer`), and `options` is handed to it
unchanged; the Ultralytics trainer passes it on as extra training arguments. The
studio's routes are listed at its `/docs`. They serve its own pages and may change
between versions, unlike the [device API](device-api.md).
