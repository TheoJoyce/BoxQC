# Data on disk

Everything BoxQC stores is a plain file: JPEG images, JSON documents, model files.
Nothing needs a database, and other tools can read or produce the same files. This page
specifies each of them.

Ids (`20261112-154347-ef25`) are a local timestamp plus four random hex digits. They
sort chronologically, and are used for images and inspections alike. An id that is
already taken is never handed out again, so adding many images within one second is
safe. Timestamps inside files are ISO 8601 with a UTC offset. Both come from the clock
of the machine that writes the file; on a Pi without network time that clock can be
behind ([pi-setup.md](pi-setup.md#clock)).

## On the studio machine

```
data/
  images/<id>.jpg           one photo per item, full camera resolution
  annotations/<id>.json     labels and bookkeeping for that photo
  datasets.json             the named datasets
  studio.json               studio state
  thumbs/<id>.jpg           thumbnail cache; safe to delete
models/
  <name>.onnx               trained models ...
  <name>.json               ... and their metadata (see models.md)
work/
  yolo_dataset/             the last prepared training dataset (rebuilt for every run)
  runs/<name>/              per-run output of the trainer (plots, weights)
  weights/                  downloaded pretrained weights
```

### `annotations/<id>.json`

```json
{
  "id": "20261112-154347-ef25",
  "width": 2304,
  "height": 1296,
  "created": "2026-11-12T15:43:47-05:00",
  "updated": "2026-11-12T15:50:45-05:00",
  "status": "labeled",
  "source": "capture",
  "tag": "defective",
  "notes": "",
  "boxes": [
    {"cls": "dirty", "x0": 0.5670, "y0": 0.6107, "x1": 0.6170, "y1": 0.7185}
  ],
  "datasets": ["Testing Data"]
}
```

| Field | Meaning |
|---|---|
| `id` | Same as the file names. |
| `width`, `height` | Size of the image in pixels. |
| `status` | `unlabeled`: not reviewed yet (may hold boxes suggested by a model). `labeled`: reviewed; **no boxes means the box is clean**, a negative example. `skip`: excluded from training. |
| `source` | `capture` (device camera), `upload`, or `inspection` (sent from the Inspect page). |
| `tag` | The operator's hint at capture time: `clean`, `defective` or empty. Informational. |
| `notes` | Free text. |
| `boxes` | One entry per defect. `cls` is a class name from `classes` in `boxqc.yaml`. Coordinates are fractions (0..1) of the **full image**, top-left origin, with `x0 < x1` and `y0 < y1`. They are not relative to the ROI, so labels survive an ROI change. |
| `datasets` | Names of the datasets this image belongs to (possibly none). |

Only `labeled` images are used for training. Unknown fields are ignored on load.

### `datasets.json`

```json
[{"name": "Testing Data", "description": "", "created": "2026-11-12T15:16:19-05:00"}]
```

The list of named datasets, oldest first. Membership is stored in each annotation's
`datasets` list, not here. Names are 1 to 40 characters (letters, digits, spaces,
`.`, `_`, `-`) and unique ignoring case. If annotations mention a dataset that is not
listed (files copied in from elsewhere), the studio adds it at start-up.

### `studio.json`

```json
{"capture_dataset": "Testing Data", "roi": [0.1, 0.05, 0.9, 0.95]}
```

| Field | Meaning |
|---|---|
| `capture_dataset` | The dataset that new captures and "send to dataset" join, or `null`. Uploads join it too unless they are made while another dataset is shown on the Dataset page. |
| `roi` | The device's ROI as last seen by the studio, or `null` if it has never been reached. Used to crop training images while the device is off. |

### Importing and exporting

* **Import:** drop JPEGs into `data/images/` with a matching annotation file each, or
  upload them on the Dataset page.
* **Export for training elsewhere:** `boxqc export --format yolo|coco` writes ROI-cropped
  images with labels ([trainers.md](trainers.md#the-interface) shows the layout).
* **Backup:** the Dataset and System pages offer a zip of `images/`, `annotations/`,
  `datasets.json` and `studio.json`; copying `data/` does the same.

## On the device

```
data/
  settings.json                    runtime settings
  inspections/<id>.json            one record per inspection
  inspections/<id>.jpg             the inspected frame
  inspections/<id>_annotated.jpg   the frame with boxes drawn (only with inference.save_annotated)
models/
  <name>.onnx + <name>.json        deployed models
```

The device holds nothing else. With the defaults an inspection takes roughly 1 MB, and
the newest `inference.keep_inspections` (1000) are kept.

### `settings.json`

```json
{"conf_threshold": 0.5, "roi": [0.0, 0.0, 1.0, 1.0], "active_model": "boxqc-20261112-155617.onnx"}
```

The three runtime settings ([configuration.md](configuration.md#runtime-settings)).
`roi` is `[x0, y0, x1, y1]` as fractions of the full frame. `active_model` is a file
name in `models/`, or `null` to use the newest model.

### `inspections/<id>.json`

The inspection record, identical to the response of `POST /api/v1/inspect`
([device-api.md](device-api.md#post-inspectbuzztrue)). Detection boxes there are in
**pixels of the full frame**, unlike annotation boxes, which are fractions.

## Models

A model file with an optional metadata file of the same name, on both machines. The
metadata fields and the model contract are in [models.md](models.md).
