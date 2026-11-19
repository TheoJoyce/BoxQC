# Configuration

BoxQC is configured by one YAML file, [boxqc.yaml](../boxqc.yaml). Every key is
explained by a comment in that file, which is the reference; this page explains how
the pieces fit together.

## Where the configuration comes from

Later sources win over earlier ones:

1. Built-in defaults (the dataclasses in [config.py](../src/boxqc/config.py)).
2. `boxqc.yaml`: the file given with `--config`, else `$BOXQC_CONFIG`, else
   `./boxqc.yaml`. Without any, the defaults are used.
3. `boxqc.local.yaml` next to it, if present. Same structure, merged key by key. It is
   never committed and never synced to the Pi, so it is the place for values that
   differ per machine.
4. Environment: `BOXQC_DEVICE_URL` (overrides `device.url`) and `BOXQC_API_TOKEN`
   (overrides `device.api_token`).

Relative paths are resolved against the directory of `boxqc.yaml`. An unknown key, a
value of the wrong type (`port: "eight thousand"`, `classes: dirty` instead of a list)
and a file that is not valid YAML are errors at start-up, each with a message naming the
key, so a typo cannot silently do nothing.

`boxqc.yaml` travels with the code: `git pull` and `deploy/sync.sh push` replace it. On
the Pi as on the desktop, a value that belongs to one machine (pins, camera controls,
the device address, a token) goes in that machine's `boxqc.local.yaml`.

```yaml
# boxqc.local.yaml on a laptop
device:
  url: http://raspberrypi.local:8000
```

## Who reads what

The same file can be used on both machines; each half ignores the other's sections.

| Section | Device (`boxqc device`) | Studio (`boxqc studio`, CLI) |
|---|---|---|
| `data_dir` | `settings.json`, `inspections/` | `images/`, `annotations/`, `datasets.json`, `studio.json` |
| `models_dir` | deployed models | trained models |
| `classes` | fallback names for a model that carries none | the classes offered for labeling, and the class order for training |
| `roi` | initial ROI | fallback when the device's ROI is unknown |
| `plugins` | model formats | model formats and trainers |
| `device.host`, `device.port` | where the API listens | – |
| `device.url` | – | how to reach the device |
| `device.api_token` | token to require | token to send |
| `camera`, `hardware`, `inference` | yes | – |
| `studio` | – | where the web UI listens |
| `train` | – | training defaults, `workdir` |

Keeping `classes` in the same order everywhere matters only for models without embedded
class names. Never reorder the list once models exist; append new classes at the end.

Labels store the class **name**. If a class is renamed or removed while images are
labeled with it, those images are left out of training (the export says how many, and
for which class) rather than being trained on as if they were clean. They can still be
opened, skipped and relabeled on the labeling page. To rename a class, relabel the
images, or change `cls` in the annotation files ([data-formats.md](data-formats.md)).

## Runtime settings

Four values are tuned while the station runs, so they are not read from `boxqc.yaml`
after the first start. The device stores them in `<data_dir>/settings.json`:

| Setting | Changed with | Initial value |
|---|---|---|
| `conf_threshold` | Inspect page, `PUT /api/v1/settings` | `inference.conf_threshold` |
| `roi` | System page, `PUT /api/v1/settings` | `roi` |
| `active_model` | model menu, `POST /api/v1/models/{name}/activate` | newest model |
| `buzzer` | System page, `PUT /api/v1/settings` | PASS 1 × 0.12 s, FAIL 3 × 0.33 s |

To go back to the YAML values, stop the device and delete `settings.json`.

The studio keeps two values of its own in `<data_dir>/studio.json`: the capture dataset,
and a copy of the device's ROI for training while the device is off. See
[data-formats.md](data-formats.md).

## Plug-ins

`plugins:` lists Python modules to import at start-up, each of which may register a
model format or a trainer. A module is looked up next to `boxqc.yaml` and on the normal
module path. See [models.md](models.md) and [trainers.md](trainers.md).

## Keys from BoxQC 0.1

`server.host`, `server.port` and `remote.url` are still read (as `device.host`,
`device.port` and `device.url`), with a warning in the log.
