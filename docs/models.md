# Models

How BoxQC represents a detection model, how to run one that was trained elsewhere, and
how to add support for another architecture or runtime.

## A model is a file plus optional metadata

```
models/my-model.onnx      the model
models/my-model.json      metadata sidecar (optional, same name)
```

The device and the studio both keep models this way, and `POST /api/v1/models`
(or `boxqc deploy my-model.onnx`) uploads the pair.

### The metadata sidecar

A JSON object. Every field is optional; unknown fields are kept and shown as they are.

| Field | Meaning |
|---|---|
| `format` | Name of the [model format](#model-formats) that loads the file. Default: the format registered for the file extension (`yolo-onnx` for `.onnx`). |
| `classes` | Class names by class index. Used when the model file itself carries none. |
| `layout` | `yolo-onnx` only: force the output layout (see below). |
| `imgsz` | Input size, used for models with dynamic input axes. |
| `roi` | The region of interest the training images were cropped to. The studio warns when it differs from the device's current ROI. |
| `recommended_threshold` | FAIL threshold suggested by the evaluation; offered in the UI with one click. `null` when the validation images included no defective one. |
| `metrics` | `{"mAP50": ..., "mAP50-95": ..., "precision": ..., "recall": ...}` |
| `evaluation` | PASS/FAIL results on the validation images at each threshold. |
| `trainer`, `base_model`, `epochs`, `batch`, `trained_at`, `trained_on`, `dataset`, `dataset_name`, `run_dir`, ... | Provenance, written by the training pipeline and shown on the Training page. |

A minimal sidecar for a model from elsewhere:

```json
{"classes": ["dirty", "defect"], "recommended_threshold": 0.4}
```

## Running a model trained elsewhere

If your model is a YOLO-family detector exported to ONNX, it very likely runs as it is:

```bash
boxqc deploy path/to/model.onnx          # uploads model.json too if it is there
```

The device loads it, runs it once on a blank image and refuses it with an explanation
if it cannot be decoded.

Keep two things in mind:

* **Train on the ROI crop.** The device crops each frame to the ROI before the model
  sees it. `boxqc export --format yolo` (or `coco`) writes your labeled images already
  cropped, with the boxes relative to the crop; train on that export.
* **The verdict rule is fixed:** any detection at or above the threshold fails the box.
  Train a model that detects defects, not one that classifies boxes.

## The built-in format: `yolo-onnx`

Runs with ONNX Runtime on the CPU ([yolo_onnx.py](../src/boxqc/models/yolo_onnx.py)).

**Input.** One tensor `(1, 3, H, W)`: RGB, `float32` or `float16`, scaled to 0..1. The
image is letterboxed, that is resized keeping its aspect ratio and padded with grey
(114). H and W come from the model, or from `imgsz` when the axes are dynamic.

**Output.** The first output tensor, in one of these layouts (`nc` = number of classes,
box coordinates in input pixels):

| `layout` | Shape | Row content | Typical source |
|---|---|---|---|
| `yolo` | `(1, 4 + nc, N)` | cx, cy, w, h, class scores | Ultralytics YOLOv8, YOLO11 |
| `yolo-obj` | `(1, N, 5 + nc)` | cx, cy, w, h, objectness, class scores | YOLOv5, YOLOv7 |
| `end2end` | `(1, N, 6)` | x0, y0, x1, y1, score, class | exports with the NMS built in (e.g. YOLOv10) |

`yolo` and `yolo-obj` are accepted with the channel axis first or last, and are followed
by class-aware non-maximum suppression. With `layout` unset (`auto`) the layout is
worked out from the output shape and the number of class names. One shape is ambiguous:
`(N, 6)` is read as `end2end`, so a single-class `yolo-obj` model or a two-class
channels-last `yolo` model must say `"layout": "yolo-obj"` or `"layout": "yolo"` in
its sidecar.

**Class names** are taken from the ONNX metadata key `names` (Ultralytics writes it),
else from `classes` in the sidecar, else from `classes` in `boxqc.yaml`.

Exporting from Ultralytics:

```python
YOLO("best.pt").export(format="onnx", imgsz=640, simplify=True, dynamic=False)
```

## Model formats

A *model format* is a class that loads a model file and turns an image into detections.
Everything else (the inspection pipeline, the evaluation after training, the
`boxqc inspect` command) goes through this interface, so one class is all it takes to
support, say, an RT-DETR export, a TFLite model or an accelerator's runtime.

### The interface

```python
from boxqc.models import Detection, ModelAdapter, register_format

@register_format
class MyModel(ModelAdapter):
    format = "my-format"          # the name used in the metadata sidecar
    suffixes = (".onnx",)         # file extensions it can load

    def __init__(self, path, *, threads=4, fallback_classes=None, info=None):
        self.path = Path(path)
        self.info = dict(info or {})                  # the metadata sidecar
        self.names = self.info.get("classes") or fallback_classes or []
        self.input_size = (640, 640)                  # (height, width)
        ...                                           # load the model

    def detect(self, image_bgr, conf_threshold=0.25, iou_threshold=0.45):
        ...
        return detections, milliseconds
```

`detect` receives a BGR `uint8` image of any size (the ROI crop) and returns:

* a list of `Detection(cls_id, name, conf, (x0, y0, x1, y1))` with boxes in the pixel
  coordinates **of the image it was given**, sorted by confidence, highest first, and
  limited to `conf >= conf_threshold`;
* the time taken in milliseconds.

Raise `boxqc.models.ModelError` with a message meant for a person when the file cannot
be used. Import heavy runtimes inside `__init__`, not at module level.

[examples/boxes_scores_labels_format.py](../examples/boxes_scores_labels_format.py) is
a complete, tested format for detectors with separate `boxes` / `scores` / `labels`
outputs.

### Selecting a format

`load_model(path)` picks the class named by `"format"` in the sidecar. Without that, it
picks the first registered format that lists the file's extension. So a second format
for `.onnx` files is selected per model, by its sidecar:

```json
{"format": "my-format", "classes": ["dirty", "defect"]}
```

A format with its own extension (say `.tflite`) needs no sidecar entry.

### Making it available

A format must be installed **on the device** (which runs the model) and on the studio
machine if you train with it there (the studio evaluates the model after training).

*As a file.* Put the module next to `boxqc.yaml` (or anywhere importable) and list it:

```yaml
plugins: [my_format]
```

*As a package.* Declare an entry point, then `pip install` the package:

```toml
[project.entry-points."boxqc.model_formats"]
my-format = "my_package.my_format:MyModel"
```

Check with `boxqc plugins`. The device's `GET /api/v1/status` lists the accepted file
extensions under `model_formats`.
