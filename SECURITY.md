# Security

## What to expect

BoxQC is built for a private network: a Pi and a desktop on a direct cable or a trusted
LAN.

* The device API has **no authentication by default**. Anyone who can reach port 8000
  can trigger inspections, change settings and upload models. Set an API token on shared
  networks ([docs/device-api.md](docs/device-api.md#authentication)).
* A token is sent in clear text over HTTP. It keeps casual clients out; it does not
  protect against someone who can read the network traffic. Do not expose the device or
  the studio to the internet. If you need remote access, put it behind a VPN or a
  TLS-terminating reverse proxy with its own authentication.
* The studio listens on `127.0.0.1` by default and has no user accounts. **Serve mode**
  (`--serve`) opens it to the network behind one shared **access key**. Everyone with the
  key, or with the link or QR code that carries it, can use the dataset, training and,
  through the studio, the device (the studio adds the device's token to the requests it
  forwards). Requests from the studio's own computer need no key. Behind a reverse proxy
  on that same computer, every request looks local, so the proxy must do its own
  authentication. The key travels in clear text over HTTP, like the device token.
  `--new-key` replaces it. Starting the studio with `--host 0.0.0.0` and no `--serve`
  opens it to everyone on the network with no key at all.
* Web pages open in the same browser cannot drive the studio: it refuses requests that
  change something when the browser says another site caused them (the `Origin` header
  must name the studio itself). This does not cover an attacker who can make a name of
  their own resolve to the studio's address (DNS rebinding), and the **device API has
  no such check**: on a device without a token, a page open in a browser on the same
  network can trigger inspections. Set a token if that matters.
* The device answers **discovery** broadcasts on UDP port 47800 with its name, version,
  port and addresses ([docs/device-api.md](docs/device-api.md#discovery)). With
  `device.url: auto`, the studio connects to whichever device answers. Without a token,
  another machine on the network could answer in its place. With a token, answers carry a
  proof made with it, and the studio sends the token only to a device that proves it holds
  it. Set `device.discovery: false` to stop answering, and give the address in
  `device.url` to stop searching.
* On the Pi, keep the token in `/etc/default/boxqc` (mode 600) or `boxqc.local.yaml`,
  not in the systemd unit, which is world-readable and rewritten by `setup_pi.sh`.
* `deploy/host/linux_usb_share.sh` turns the development machine into a DHCP server and
  NAT router for USB network gadgets (driver `cdc_ether`), so that the Pi gets internet
  access through it. It does this for every such device that is plugged in, not only the
  Pi. `--remove` undoes it.
* Model files are executed by ONNX Runtime, and plug-ins and trainer options run as
  code. Only deploy models and install plug-ins from sources you trust.

## Maintenance

This repository is not under active development, so do not expect security fixes to be
published here. If you deploy BoxQC, treat the points above as your responsibility, and
if you maintain a fork, keep its dependencies (FastAPI, ONNX Runtime, OpenCV, and
PyTorch / Ultralytics on the training side) up to date.
