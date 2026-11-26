# Device API (v1)

The HTTP API of the inspection station (`boxqc device`). Use it to trigger inspections
from a PLC or a script, to deploy models, or to build your own front end. The studio
itself uses nothing else.

* Base URL: `http://<device>:8000/api/v1` (port: `device.port` in `boxqc.yaml`). To find
  `<device>` without knowing it, see [Discovery](#discovery).
* Requests and responses are JSON unless noted. Errors are
  `{"detail": "<what went wrong>"}` with a 4xx or 5xx status.
* A running device serves interactive documentation at `/docs` and the OpenAPI schema
  at `/openapi.json`.
* [examples/inspect_client.py](../examples/inspect_client.py) is a complete client using only
  the Python standard library;
  [`DeviceClient`](../src/boxqc/studio/device_client.py) is a fuller one.

```bash
boxqc devices                       # prints the device's address, e.g. http://192.168.1.50:8000
curl -X POST http://192.168.1.50:8000/api/v1/inspect
```

## Versioning and compatibility

The version is in the path. Within `v1`, fields and routes are only added, never
removed or changed in meaning; a client should ignore fields it does not know.
`GET /status` reports `api_version`.

The routes below also answer under the unversioned prefix `/api` (for example
`POST /api/inspect`), which is what BoxQC 0.1 used. New integrations should use
`/api/v1`.

## Authentication

By default there is none: anyone who can reach the device can use the API. That is
intended for a private link between the Pi and one computer.

To require a token, set it on the device and on every client:

```bash
# on the device, in a file that only root can read and that setup_pi.sh never replaces:
echo 'BOXQC_API_TOKEN=<long random string>' | sudo tee /etc/default/boxqc
sudo chmod 600 /etc/default/boxqc && sudo systemctl restart boxqc
# on the studio machine:
export BOXQC_API_TOKEN=<the same string>
```

(`device.api_token` in `boxqc.local.yaml` works too, on either machine. Do not add the
token to the installed systemd unit: re-running `setup_pi.sh` rewrites that file.)
Every request must then carry
`Authorization: Bearer <token>` or `X-API-Key: <token>`; otherwise the answer is `401`.
The token travels in clear text over HTTP, so it keeps casual clients out but is not a
substitute for a trusted network.

## Status

### `GET /status`

Health of the station. Suitable for polling.

```json
{
  "version": "0.2.0",
  "api_version": "1",
  "hostname": "raspberrypi",
  "addresses": {"usb0": ["10.42.0.2/24"], "wlan0": ["192.168.1.40/24"]},
  "camera": "picamera2",
  "camera_error": null,
  "hardware": {"buzzer": "GPIO17 (active)", "trigger": null, "last_pattern": "PASS", "error": null,
               "patterns": {"PASS": {"beeps": 1, "beep_seconds": 0.12, "gap_seconds": 0.1}, "FAIL": {...}, ...}},
  "model": "boxqc-20261112-155617.onnx",
  "model_error": null,
  "model_info": {"name": "boxqc-20261112-155617.onnx", "format": "yolo-onnx",
                 "classes": ["dirty", "defect"], "input_size": [640, 640], "layout": "auto"},
  "classes": ["dirty", "defect"],
  "model_formats": {"suffixes": [".onnx"]},
  "settings": {"conf_threshold": 0.5, "roi": [0.0, 0.0, 1.0, 1.0], "active_model": "boxqc-20261112-155617.onnx",
               "buzzer": {"PASS": {"beeps": 1, "beep_seconds": 0.12}, "FAIL": {"beeps": 3, "beep_seconds": 0.33}}},
  "session_counts": {"PASS": 41, "FAIL": 3},
  "disk_free_gb": 5.9,
  "config_file": "/home/pi/boxqc/boxqc.yaml"
}
```

* `camera`: `picamera2`, `opencv` or `mock`; `null` if the camera failed to start
  (`camera_error` says why). `mock` means synthetic images: expected on a development
  machine, a fault on a station. A camera that fails to open is never replaced by the
  mock.
* `model`: the loaded model's file name, or `null` (then every inspection is
  `NO_MODEL`). `model_error` is set when the active model failed to load.
* `session_counts`: verdicts since the service started.

## Settings

The four values an operator tunes. They are stored on the device and survive restarts.

### `GET /settings`

```json
{"conf_threshold": 0.5, "roi": [0.0, 0.0, 1.0, 1.0], "active_model": "my-model.onnx",
 "buzzer": {"PASS": {"beeps": 1, "beep_seconds": 0.12}, "FAIL": {"beeps": 3, "beep_seconds": 0.33}}}
```

### `PUT /settings`

Body: any of `conf_threshold` (0 < value <= 1), `roi` (`[x0, y0, x1, y1]` as
fractions of the full frame, at least 5 % wide and high) and `buzzer` (see below).
Returns the settings after the change; `400` if a value is invalid, in which case
nothing is changed.
`active_model` is changed with `POST /models/{name}/activate`.

The ROI is applied before inference: only that part of the frame reaches the model.
Changing it after a model was trained usually means retraining.

`buzzer` sets how PASS and FAIL sound: the number of beeps (1 to 10) and the length of
each beep in seconds (0.05 to 2). Give only what changes; the rest stays as it is:

```json
{"buzzer": {"FAIL": {"beeps": 5, "beep_seconds": 0.25}}}
```

The pause between beeps (0.1 s for PASS, 0.2 s for FAIL) and the ERROR chatter are
fixed. Patterns play one after another, so a long one delays the next. `GET /status`
lists every verdict's pattern as it will sound under `hardware.patterns`, for a client
that wants to play the same sound itself.

## Camera

| Route | Returns |
|---|---|
| `GET /camera/stream.mjpg?guides=true` | Live low-resolution preview as `multipart/x-mixed-replace` (works as the `src` of an `<img>`). `guides` draws the ROI and a centre cross. |
| `GET /camera/snapshot.jpg?full=false` | One JPEG frame without guides. `full=true` gives a full-resolution capture. |
| `POST /camera/capture` | A full-resolution JPEG (`camera.jpeg_quality`), with `X-Image-Width` / `X-Image-Height` headers. Nothing is stored on the device. |

All three return `503` while the camera is unavailable.

## Inspection

### `POST /inspect?buzz=true`

Take a photo, run the model, sound the buzzer and log the result. One inspection runs
at a time; a second request waits for the first.

```json
{
  "id": "20261112-160114-a3f9",
  "time": "2026-11-12T16:01:14-05:00",
  "verdict": "FAIL",
  "threshold": 0.5,
  "model": "boxqc-20261112-155617.onnx",
  "image_size": [2304, 1296],
  "roi": [0.1, 0.05, 0.9, 0.95],
  "detections": [
    {"cls_id": 0, "name": "dirty", "conf": 0.9132, "box": [1210.5, 402.0, 1388.2, 561.7]},
    {"cls_id": 0, "name": "dirty", "conf": 0.2104, "box": [640.0, 880.3, 702.9, 930.0]}
  ],
  "failing": 1,
  "timing_ms": {"inference": 412.6, "capture": 95.0, "total": 520.4},
  "source": "camera"
}
```

* `verdict`: `PASS`, `FAIL` (at least one detection with `conf >= threshold`) or
  `NO_MODEL`.
* `detections`: sorted by confidence, in pixels of the full frame (`x0, y0, x1, y1`).
  They include weak detections down to `inference.min_report_conf`, so compare `conf`
  with `threshold` to find the ones that caused a FAIL; `failing` counts those.
* `source`: `camera`, `trigger` (the GPIO button) or `upload`.
* `id` and `time` come from the device's clock, which on a Pi without network time can
  be behind ([pi-setup.md](pi-setup.md#clock)).
* `buzz=false` skips the buzzer.

Errors: `503` camera unavailable, `500` if the inspection itself failed (the buzzer
then plays the ERROR pattern).

### `POST /inspect/upload?buzz=false`

The same on an image you supply: multipart form with one `file` field. `400` if it is
not an image, `500` if the inspection itself failed.

### Inspection log

| Route | Returns |
|---|---|
| `GET /inspections?limit=50` | The newest records (as above), newest first. |
| `GET /inspections/{id}` | One record. `404` if unknown or already pruned. |
| `GET /inspections/{id}/image.jpg?annotated=false` | The inspected frame as JPEG. `annotated=true` draws the boxes and the verdict on it. |
| `GET /inspections.csv?limit=1000` | `id,time,source,verdict,threshold,failing,top_conf,top_class,model,inference_ms` |

The device keeps the newest `inference.keep_inspections` records (default 1000) and
deletes older ones. With `inference.save_inspections: false` nothing is kept and these
routes return nothing useful.

## Models

A model is a file plus an optional metadata file; see [models.md](models.md).

### `GET /models?usage=false`

Models stored on the device, newest first.

```json
[{
  "name": "boxqc-20261112-155617.onnx",
  "size_mb": 10.61,
  "modified": "2026-11-12 16:01",
  "sha256": "9f2c41d0...e7a3",
  "info": {"format": "yolo-onnx", "classes": ["dirty", "defect"], "recommended_threshold": 0.35},
  "active": true,
  "usage": null
}]
```

`sha256` is the hash of the model file's content (64 hex digits). Use it, not the name,
to tell whether the device already has a particular model: a name can be reused for a
different model. `info` is the model's metadata (`{}` if it has none). `usage=true` fills `usage` from
the inspection log: `{"inspections": 44, "PASS": 41, "FAIL": 3, "last_used": "...",
"inference_ms": 410.2}` (median inference time), or `null` for a model that has not
been used.

### `POST /models`

Upload a model. Multipart form:

| Field | |
|---|---|
| `file` | the model file (required); its extension must belong to an installed format |
| `meta` | its metadata JSON (optional) |
| `activate` | `true` (default) to make it the active model |

```bash
curl -F file=@my-model.onnx -F meta=@my-model.json -F activate=true http://192.168.1.50:8000/api/v1/models
```

The device loads the model and runs it once on a blank image before accepting it. A
file that fails is discarded and the answer is `400` with the reason; nothing on the
device changes. If the name is taken, a timestamp is appended, so read the name from
the response: `{"name": "my-model.onnx", "active": true}`.

### `POST /models/{name}/activate`

Switch the active model. `{"active": "<name>"}`; `404` unknown. The model is loaded and
run once first: if that fails the answer is `500` and the model that was active stays
active.

### `DELETE /models/{name}`

Delete a model and its metadata. `400` for the active model.

## Hardware

### `POST /buzzer/test`

Body `{"pattern": "PASS"}` (`PASS`, `FAIL` or `ERROR`). Returns
`{"played": "PASS", "mock": false}`; `mock` is true when no GPIO buzzer is available.

Buzzer patterns: by default PASS is one short chirp and FAIL three longer beeps (both
can be changed with the `buzzer` [setting](#put-settings)); ERROR / NO_MODEL is a rapid
chatter. The test plays the saved patterns.

## Discovery

Besides the HTTP API, a running device answers **discovery queries** on **UDP port
47800**, so that a client can find it without knowing its address. This is what
`device.url: auto` and `boxqc devices` use. It is not part of `/api/v1`, but it follows
the same rule: fields are only ever added. The implementation, client and device side, is
[boxqc/discovery.py](../src/boxqc/discovery.py).

A **query** is one UDP datagram, UTF-8 JSON padded with spaces to at least **1024
bytes**:

```json
{"boxqc": "discover", "v": 1, "nonce": "<8 to 64 lowercase hex digits, random per query>"}
```

Send it to the broadcast address of each network interface (for example
`192.168.1.255`), to `255.255.255.255`, or straight to an address you suspect. Send it
more than once while you wait (the studio sends it at 0, 0.3 and 0.6 s and listens for
1 s), because Wi-Fi does not resend a lost broadcast frame.
[examples/inspect_client.py](../examples/inspect_client.py) has a minimal client. The device
ignores datagrams shorter than 1024 bytes and anything that is not a query. Its answer is
always shorter than the query, so a forged sender address cannot be used to make it send
more traffic than it receives. It also answers at most 20 queries a second.

The **answer** goes back to the address and port the query came from:

```json
{"boxqc": "device", "v": 1, "nonce": "<the query's>", "id": "3f9c0a1b2d4e5f60",
 "name": "piqc", "hostname": "piqc", "version": "0.2.0", "api_version": "1",
 "port": 8000, "addresses": ["10.12.194.1", "192.168.1.50", "127.0.0.1"],
 "auth": false, "proof": null}
```

* The API is at `http://<address the answer came from>:<port>/api/v1`. That address is
  one the client can reach. `addresses` lists all of the device's IPv4 addresses.
* `name` is `device.name` on the device, or its hostname. `id` is random each time the
  device starts. A device on two links (USB and Wi-Fi) answers once per link with the same
  `id`. The first answer to arrive is usually over the faster link.
* `auth` is true when the API requires a [token](#authentication). `proof` is then
  HMAC-SHA256, keyed with the token, of the answer without `proof`: JSON with sorted
  keys and no spaces (`json.dumps(answer, sort_keys=True, separators=(",", ":"))`).
  A client holding the token should check the proof, and that the answer came from one of
  `addresses`, before sending the token. It should treat an answer with `auth: false`
  as unproven too, since anyone can claim that. The studio and the command line do all
  of this, so another machine on the network cannot pose as the device to collect the
  token.
* Without a token, nothing checks that an answer came from a real device. On a network
  where that matters, set a token or give the device's address in `device.url`.

To turn discovery off, set `device.discovery: false` on the device. If the port cannot
be opened, the device logs a warning and runs without discovery.
