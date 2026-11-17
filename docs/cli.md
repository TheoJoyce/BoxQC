# Command reference

One command, `boxqc`, with a sub-command per task. `boxqc --help` and
`boxqc <command> --help` print the same information.

```
boxqc [-c CONFIG] [-v] COMMAND [options]
```

| Global option | |
|---|---|
| `-c`, `--config FILE` | Path to `boxqc.yaml`. Default: `$BOXQC_CONFIG`, else `./boxqc.yaml`, else built-in defaults. |
| `-v`, `--verbose` | Debug logging. |

Commands marked *Pi* run on the inspection station; *desktop* commands need the
`studio` extra (and `train` for training).

## `boxqc device` (Pi)

Run the device API: camera, inference, buzzer. `boxqc serve` is an alias.

| Option | |
|---|---|
| `--host HOST` | Address to listen on. Default `device.host` (`0.0.0.0`). |
| `--port PORT` | Default `device.port` (`8000`). |

On the Pi this is started by the `boxqc` systemd service
([pi-setup.md](pi-setup.md)). API reference: [device-api.md](device-api.md).

## `boxqc studio` (desktop)

Run the web UI with the dataset and training. Open the address it prints.

| Option | |
|---|---|
| `--host HOST` | Address to listen on. Default `studio.host` (`127.0.0.1`); `0.0.0.0` opens it to the network. |
| `--port PORT` | Default `studio.port` (`8080`). |
| `--device-url URL` | Where the device is. Default `device.url`. |

## `boxqc train` (desktop)

Prepare the dataset, train, export, evaluate and write the model to `models/`. The
same pipeline as the Training page, in the foreground.

| Option | |
|---|---|
| `--dataset NAME` | Train on one named dataset. Default: every labeled image. |
| `--epochs N`, `--imgsz PX`, `--batch N`, `--base-model NAME`, `--device DEV` | Override the `train:` section of the config. `--device` is the compute device (`auto`, `mps`, `cpu`, `0`, ...). |
| `--trainer NAME` | Training backend. Default `train.trainer`. |
| `--name NAME` | Run and model name. Default `boxqc-<date>-<time>`. |
| `--deploy` | Upload the model to the device afterwards and activate it. |
| `--device-url URL` | Where the device is (for the ROI and `--deploy`). |

The images are cropped to the device's current ROI; if the device cannot be reached,
the ROI it had when the studio last saw it is used.

## `boxqc deploy MODEL` (desktop)

Upload a model file to the device. A metadata file with the same name and a `.json`
extension is uploaded with it. `boxqc push-model` is an alias.

| Option | |
|---|---|
| `--no-activate` | Store the model without making it the active one. |
| `--device-url URL` | Where the device is. |

## `boxqc export` (desktop)

Write the labeled images as a training dataset: ROI crops with YOLO label files, split
into train and val. `boxqc prepare` is an alias. Images labeled with a class that
`boxqc.yaml` no longer lists are left out, with a message.

| Option | |
|---|---|
| `--format yolo\|coco` | `coco` adds `annotations/instances_train.json` and `instances_val.json`. Default `yolo`. |
| `--out DIR` | Output directory. An earlier export there is **replaced**; a folder that holds anything else is refused, so a slip such as `--out .` deletes nothing. Default `<workdir>/yolo_dataset`. |
| `--dataset NAME` | Use one named dataset. Default: every labeled image. |

## `boxqc evaluate MODEL` (desktop)

Run a model over the validation images of an exported dataset and print the PASS/FAIL
table per threshold, as after training.

| Option | |
|---|---|
| `--dataset DIR` | Exported dataset directory. Default `<workdir>/yolo_dataset`. |
| `--json` | Also print the full result as JSON. |

## `boxqc inspect IMAGE...`

Run a model on image files and print a verdict for each. Works on either machine.

| Option | |
|---|---|
| `--model FILE` | Default: the newest model in `models_dir`. |
| `--conf T` | FAIL threshold. Default `inference.conf_threshold`. |
| `--out DIR` | Write the images with boxes and verdict drawn on them. |
| `--config-roi` | Use `roi` from `boxqc.yaml` instead of the saved runtime ROI. |

## `boxqc plugins`

List the installed model formats and trainers, and whether each trainer can run on
this machine. Use it to check that a plug-in was found.

## Removed in 0.2

`boxqc agent` and `boxqc pull` no longer exist; running them prints what replaced
them. See [upgrading.md](upgrading.md).
