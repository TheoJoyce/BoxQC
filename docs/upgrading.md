# Upgrading

## Device discovery (after 0.2.0)

`device.url` now defaults to `auto`: the studio finds the device by broadcasting on the
local network ([pi-setup.md](pi-setup.md#finding-the-device)). Only a device running the
new code answers. So:

* **Update the Pi too** (`./deploy/sync.sh push`, or `git pull && sudo systemctl
  restart boxqc`). No new packages are needed on either machine. Until then, the studio
  reports that no device answered, and `PI_HOST=<user>@auto` in `.sync.env` cannot find
  the Pi either, so push this first update to its address.
* **If you relied on the old default** `http://10.42.0.2:8000` (the USB link to a Linux
  host), nothing else changes once the Pi is updated: `auto` finds it over the link.
  Before then, set `device.url: http://10.42.0.2:8000` in `boxqc.local.yaml`.
* An address you set yourself (`boxqc.local.yaml`, `--device-url`, `$BOXQC_DEVICE_URL`)
  still wins, and no discovery happens. Remove it to use `auto`.

## From BoxQC 0.1 to 0.2

In 0.1 the Pi served the web UI and stored every training image, and a training machine
ran `boxqc agent` / `boxqc pull`. From 0.2 the Pi runs only the device API, and the web
UI, the dataset and training live on a desktop (`boxqc studio`). Nothing is deleted by
the upgrade; your images, labels, models and settings carry over.

### Steps

Do these from the desktop that will run the studio, with the Pi reachable. `sync.sh`
needs to know the Pi's SSH login: put `PI_HOST=<user>@<address>` in `.sync.env` at the
project root, or add `--host <user>@<address>` to each command. A Pi still on 0.1 does
not answer discovery, so `PI_HOST=<user>@auto` works only after step 2. To find the
address, run `hostname -I` on the Pi.

1. **Copy the dataset off the Pi.** It is no longer used there.

   ```bash
   ./deploy/sync.sh fetch-data
   ```

   This copies `data/images`, `data/annotations` and `data/datasets.json` from the Pi
   into the local `data/` folder (or the folder `data_dir` names in the local
   configuration). If you used `boxqc pull` before, most of it is already there and
   only the differences are transferred.

2. **Update the code on the Pi** and restart the service.

   ```bash
   ./deploy/sync.sh push
   ```

   The installed systemd unit keeps working (`boxqc serve` is now an alias of
   `boxqc device`), and no new packages are needed on the Pi. The Pi's models,
   `data/settings.json` (ROI, threshold, active model) and inspection log are untouched.

3. **Install and start the studio** on the desktop.

   ```bash
   pip install -e ".[studio,train]"
   boxqc studio
   ```

   The studio finds the Pi on the network now that it runs the new code (`boxqc devices`
   shows what it finds). On a network that drops broadcasts, use
   `boxqc studio --device-url http://<pi-address>:8000` instead.

   Open <http://localhost:8080>. On first start the studio registers the named datasets
   it finds in the annotations, and takes over the capture dataset from the settings
   file that `boxqc pull` left in `data/`.

4. **Check** the Dataset page (image and label counts) and run an inspection.

5. *Optional:* once you are satisfied, free the space on the Pi. `fetch-data` prints
   the exact command; it removes `data/images`, `data/annotations` and `data/thumbs`
   there.

### What changed

| 0.1 | 0.2 |
|---|---|
| Web UI at `http://<pi>:8000` | Web UI at `http://localhost:8080` (`boxqc studio`); the Pi shows a short landing page and `/docs` |
| `boxqc serve` | `boxqc device` (`serve` still works) |
| `boxqc agent` on the training machine | gone: training runs inside the studio. The Training tab is always there |
| `boxqc pull` | gone: the dataset is already on the studio machine |
| `boxqc train --pull --push` | `boxqc train --deploy` |
| `boxqc push-model` | `boxqc deploy` (`push-model` still works) |
| `boxqc prepare` | `boxqc export` (`prepare` still works) |
| `/api/...` on the Pi | `/api/v1/...` (the old paths for inspection, models, settings and status still answer) |
| `/api/stream.mjpg`, `/api/snapshot.jpg` | `/api/v1/camera/stream.mjpg`, `/api/v1/camera/snapshot.jpg` (old paths still answer) |
| Dataset routes on the Pi (`/api/images`, `/api/datasets`, `/api/capture`, ...) | on the studio; the Pi offers `POST /api/v1/camera/capture` to take a photo |
| `server:` and `remote:` in `boxqc.yaml` | `device:` (`host`, `port`, `url`). The old keys are still read, with a warning |
| "Keep `boxqc.yaml` identical on both machines" | no longer required: a model carries its class names, and `classes` only matters where you label |
| USB link to a Linux host at `10.55.0.1`, Pi at `10.55.0.2` | host `10.42.0.1`, Pi `10.42.0.2`, which the current `rpi-usb-gadget` needs. Re-run `deploy/host/linux_usb_share.sh --pi-name <hostname>` on the host, and update `device.url` if you had set it ([pi-setup.md](pi-setup.md#usb-link-to-a-development-machine-optional)) |
| `camera.backend: auto` fell back to the mock camera when the Pi camera did not open | it reports the error instead and refuses to inspect; the mock is only used where Picamera2 is not installed |

New in 0.2: named plug-in points for model formats and trainers, COCO export, an
optional API token, `boxqc.local.yaml` for per-machine settings, and
`inference.save_annotated: false` to halve the inspection log's size on the Pi.

### Existing integrations

A PLC or script that calls `POST /api/inspect`, `GET /api/status` or
`GET /api/inspections/{id}/image.jpg` on the Pi keeps working. Two fields left
`GET /status`, because the device no longer has that information: `dataset` and
`trainer`. `classes` there is now the class list of the loaded model. Move to `/api/v1` when convenient.
