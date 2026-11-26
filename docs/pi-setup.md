# Setting up the device (Raspberry Pi)

The device is the inspection station: a Raspberry Pi with a camera and a buzzer,
running `boxqc device`. This page covers the hardware, the installation, the optional
USB network link to a development machine, and keeping the code on the Pi up to date.

## Hardware

| Item | Notes |
|---|---|
| Raspberry Pi 4 or 5 (Pi Zero 2 W works but is slow) | 64-bit Raspberry Pi OS Bookworm or Trixie |
| Pi Camera Module 3 (or HQ camera) | Mount it straight down over the box, centred |
| Consistent lighting | Diffuse LED ring or panel. **Lighting consistency matters more than the model.** |
| Buzzer | Default GPIO17 (pin 11) → buzzer, GND. Drive it through an NPN transistor or use a buzzer module: GPIO pins only supply about 16 mA |
| Optional trigger button | Set `hardware.trigger_pin` (e.g. 27). Button between the pin and GND |

Buzzer patterns: **PASS** is one short chirp, **FAIL** is three longer beeps, and
**ERROR / no model** is rapid chatter. The number and length of the PASS and FAIL
beeps can be changed on the studio's **System** page. Set `buzzer_type: passive` for a buzzer that
needs a PWM tone.

A USB webcam works too (`camera.backend: opencv`), and so does any other Linux
computer; without GPIO the buzzer is simply mocked.

## Installation

Get the project onto the Pi, with `git clone` on the Pi or by copying it from your
development machine ([Keeping the code in sync](#keeping-the-code-in-sync)), then:

```bash
cd ~/boxqc                       # wherever the project is
./deploy/pi/setup_pi.sh          # apt packages, virtualenv, systemd service "boxqc"
```

Run it as the user the device should run as, not with `sudo` (it asks for sudo where it
needs it). The script installs only what the device needs: no web UI, no training
libraries.

* Picamera2, NumPy, OpenCV and gpiozero come from **apt** so they match the Pi's
  libcamera. The venv uses `--system-site-packages`, and a pip constraint stops pip
  from replacing the apt NumPy.
* The script stops with an error if Picamera2 cannot be imported while
  `camera.backend` is `auto` or `picamera2`. The device would otherwise have no real
  camera.
* The service runs `boxqc device` at boot. Logs: `journalctl -u boxqc -f`. Restart it
  after changing the configuration: `sudo systemctl restart boxqc`.
* The script can be run again at any time. It reinstalls and restarts the service; a
  changed copy of the installed unit file is kept as `boxqc.service.bak`.
  `./deploy/pi/setup_pi.sh --no-service` reinstalls the Python side only, for example
  after a dependency change.

### Settings for this Pi

Put what belongs to this station (GPIO pins, camera controls, a USB webcam) in
`boxqc.local.yaml` next to `boxqc.yaml` **on the Pi**, not in `boxqc.yaml` itself:

```yaml
# boxqc.local.yaml on the Pi
hardware:
  trigger_pin: 27
camera:
  controls: {AfMode: 0, LensPosition: 4.0}
```

`boxqc.yaml` is part of the code: `deploy/sync.sh push` and `git pull` replace it with
the development machine's version, while `boxqc.local.yaml` is never synced or
committed. See [configuration.md](configuration.md).

### Clock

A Raspberry Pi 4 has no battery-backed clock (a Pi 5 has one only with the optional
battery), and sets its time from the network. On a station with no internet access,
for example one reached only over a USB link that is not shared, the clock resumes from
roughly where it stopped at the last shutdown and falls behind. Inspection ids and
timestamps come from that clock, so they can be wrong, although their order normally
stays right. If the times matter, give the Pi network time (the shared USB link below is
enough) or a real-time clock module.

Check it from another computer on the same network: `boxqc devices` should list it
([Finding the device](#finding-the-device)). Then start the studio there with
`boxqc studio`.

The Pi and the studio machine only need to reach each other over the network: Wi-Fi,
Ethernet or the USB link below.

### Power

When the Pi 4 is powered through the same USB-C cable that carries data (gadget mode,
below), it wants up to 3 A. A hub or front-panel port often cannot supply that, and
under-voltage makes the Pi throttle and can cause USB errors.

* Plug the Pi directly into a rear motherboard USB-C port, or into a laptop's USB-C
  port. Avoid hubs, front-panel ports and monitor ports.
* Check on the Pi with `vcgencmd get_throttled`. It should print `0x0`; any other value
  means under-voltage or throttling has happened. The camera pipeline pushes a marginal
  supply over the edge.
* For a fully reliable setup, use a USB-C power/data splitter: the official 5.1 V / 3 A
  supply powers the Pi, and the host cable carries data only.

## Finding the device

The device's address changes: a Wi-Fi lease can give the Pi a different one at each
boot, and the USB link gives it another. So by default (`device.url: auto`) the studio
and the command line don't need it. They ask the local network, and the device answers.

```bash
$ boxqc devices
piqc                 http://192.168.1.50:8000    BoxQC 0.2.0
```

How it works: the studio sends a small UDP broadcast to port 47800 on each of its network
interfaces. `boxqc device` answers with its name and API port, and the studio uses the
address the answer came from. Over USB and Wi-Fi at once, the first answer usually comes
over the faster link. When the device stops answering on that address, the studio
searches again. The protocol is in [device-api.md](device-api.md#discovery).

* **Several devices on one network:** `auto` refuses to guess. Name the one to use in
  `boxqc.local.yaml` on the studio machine (`device.name: piqc`). The name is the Pi's
  hostname, or the `device.name` set on the Pi.
* **Discovery finds nothing:**
  * The machines must be on the same network segment, because broadcasts do not cross
    routers or VPNs.
  * A firewall must let UDP port 47800 into the Pi, and the answer back to the studio
    machine. On macOS, allow incoming connections for Python if the firewall asks.
  * Some networks drop broadcasts between clients (guest Wi-Fi with client isolation).
    Then give the address instead, `device.url: http://<pi-address>:8000` in
    `boxqc.local.yaml`, or use the USB link.
* **Why not `<hostname>.local`?** mDNS (Bonjour/Avahi) is the usual answer, but many
  managed Wi-Fi networks filter it between clients while letting broadcasts through, and
  this project's own network did exactly that. A `.local` name can also take seconds to
  resolve, longer than the studio waits for the device. Discovery hands the studio a
  plain IP address.
* **Turning it off:** `device.discovery: false` on the Pi.

## USB link to a development machine (optional)

The Pi's USB-C port (on a Zero 2 W, the port marked "USB") can act as a **USB network
adapter**, using Raspberry Pi's own **`rpi-usb-gadget`** package. Plugged into a
computer, the Pi gets:

* **internet through the host**, much faster than the Pi's Wi-Fi for `apt` and `pip`.
  The USB route (metric 100) beats Wi-Fi's (600) while the host is sharing, and the Pi
  falls back to Wi-Fi by itself otherwise.
* a **fixed address, 10.42.0.2**, on a Linux host. Discovery finds the Pi there
  too, or you can set `device.url: http://10.42.0.2:8000`.

```
 Linux dev machine 10.42.0.1  ──USB──  Pi 10.42.0.2
   NetworkManager "shared": DHCP + NAT     rpi-usb-gadget "USB Gadget (client)": DHCP client
   reservation: hostname of the Pi -> .2
```

`rpi-usb-gadget` is published for Raspberry Pi OS **Trixie**. On a release where apt
cannot find it, `usb_gadget.sh` stops without changing anything; use Wi-Fi or Ethernet
there.

**Pi side**, once: `sudo ./deploy/pi/usb_gadget.sh`, then reboot. It installs and
enables `rpi-usb-gadget`, which:

* loads the `g_ether` gadget
* adds `dtoverlay=dwc2,dr_mode=peripheral`
* creates two NetworkManager profiles:
  * *USB Gadget (client)*: DHCP from a sharing host
  * *USB Gadget (shared)*: the Pi serves DHCP on 10.12.194.1 when the host doesn't
    share. The `rpi-usb-gadget-ics` watcher switches between the two.

`rpi-usb-gadget status` shows the current mode.

**Host side**, once per machine:

* **Linux (NetworkManager):** run `./deploy/host/linux_usb_share.sh --pi-name <hostname of the Pi>`
  (the name is required; `hostname` on the Pi prints it).
  It creates a `shared` profile that matches USB-gadget network interfaces (driver
  `cdc_ether`). The Pi's gadget MAC changes every boot, so the profile cannot match on
  it. It also adds a DHCP reservation, so the client whose DHCP hostname is the Pi's
  always gets 10.42.0.2.
  * The host's address must be 10.42.0.1, NetworkManager's default for a shared
    connection: the Pi only asks a host for an address when it finds one at an address
    its watcher knows (that one, macOS's 192.168.2.1 or Windows' 192.168.137.1). Another
    shared connection on the same machine, such as a Wi-Fi hotspot, uses the same
    address by default, and only one of the two can work at a time.
  * Other `cdc_ether` devices plugged into this machine (some USB Ethernet adapters,
    docks and modems) get this profile too, and the machine then acts as DHCP server
    and router for them. `./deploy/host/linux_usb_share.sh --remove` takes it away.
* **macOS:** System Settings → General → Sharing → **Internet Sharing**. Share from
  Wi-Fi to the Pi's USB gadget interface. The Pi gets internet and an address in
  macOS's sharing subnet (usually `192.168.2.x`), not 10.42.0.2. Without Internet
  Sharing, the Pi serves the link itself and is at `10.12.194.1`. Either way
  discovery finds it (`boxqc devices`). To pin an address, put it in `boxqc.local.yaml`:

  ```yaml
  device:
    url: http://10.12.194.1:8000
  ```

**Check that it works (Linux host):** `ping 10.42.0.2`, then
`curl http://10.42.0.2:8000/api/v1/status`. On the Pi, `ip route get 1.1.1.1` should
show `dev usb0`. Give the Pi up to a minute after plugging in: it first serves the link
itself and then switches over.

Why not `http://<hostname>.local` everywhere? mDNS returns *all* of the Pi's addresses,
including its Wi-Fi one, so traffic can go over the slow Wi-Fi even with the cable
plugged in.

### If the link wedges

The host sees the Pi, the Pi's DHCP requests reach the host, but nothing reaches the
Pi. The host kernel logs `transmit queue 0 timed out`, and on the Pi,
`ip -s link show usb0` shows RX stuck. Two things help:

* **Un-wedge it now:** on the Pi, run `sudo modprobe -r g_ether && sudo modprobe g_ether`.
* **What fixed it on a Pi 4:** a fixed Pi-side MAC. Add
  `options g_ether dev_addr=<a fixed MAC>` in `/etc/modprobe.d/`. A file named
  `g_ether.conf` there *replaces* the package's `/usr/lib/modprobe.d/g_ether.conf`,
  which only drops Raspberry Pi's USB IDs. Give it another name, such as
  `g_ether-mac.conf`, to keep both. Scripts must never overwrite this file.

### macOS host without Internet Sharing

Here the Pi serves the link itself (`10.12.194.1`, DHCP range `.3`–`.14`). Two problems
come up, and both are fixed by hand as shown below.

**Give the Mac a fixed address on the link (recommended).** After the Pi boots, it
spends about 15 seconds as a DHCP *client* (waiting for a sharing host) before it starts
serving DHCP itself. The Mac's first DHCP requests go unanswered, macOS backs off, and the
interface can sit without an IPv4 address for several minutes. During that time
`<hostname>.local` doesn't resolve or doesn't connect, even though the cable and the Pi
are fine. A manual address outside the DHCP range removes the race:

```bash
networksetup -listallnetworkservices          # find the Pi's service name
sudo networksetup -setmanual "Raspberry Pi USB Gadget" 10.12.194.2 255.255.255.240
```

Undo it with `sudo networksetup -setdhcp "Raspberry Pi USB Gadget"`, and do that before
you turn on Internet Sharing to the Pi, which needs the interface back on DHCP.

**Manual recovery when the link is down.** Check the Mac side first:

```bash
ifconfig en8            # the Pi's interface; the number varies, see networksetup above
```

* *`status: active` but no `inet 10.12.194.x` line* (on DHCP): the race above. Wait a
  few minutes, re-request with `sudo ipconfig set en8 DHCP`, or switch to the fixed
  address.
* *`status: inactive` after the Mac slept:* the Pi is usually fine, but the gadget
  doesn't come back after the host's USB suspend. Reload it on the Pi over Wi-Fi or
  Ethernet: `sudo modprobe -r g_ether; sleep 2; sudo modprobe g_ether`. The Mac
  interface gets its address about 10 seconds later. Don't reload while the Mac is
  still asleep: macOS then drops the interface until the cable is replugged.
* *Active and addressed, but no reply from `10.12.194.1`:* see
  [If the link wedges](#if-the-link-wedges). On the Pi, `ip -s link show usb0` shows
  whether anything is received.

To reach the Pi over Wi-Fi when the USB link is down, find its address with
`hostname -I` on the Pi (screen and keyboard) or in the router's client list, because
the Wi-Fi address can change between boots. `<hostname>.local` may only return the
USB address then. If SSH warns about the host key, add
`-o HostKeyAlias=<hostname>.local`: it is the same machine.

## Keeping the code in sync

With git on the Pi, `git pull && sudo systemctl restart boxqc` is all it takes.

Without it, `deploy/sync.sh` copies the code between a development machine and the Pi
with rsync over SSH. The development machine is the source of truth. The script runs
from any shell and any working directory: the local side is always the project folder
the script lives in.

```bash
./deploy/sync.sh status      # what push / pull would change; changes nothing
./deploy/sync.sh push        # dev -> Pi, then restarts the boxqc service
./deploy/sync.sh pull        # Pi -> dev, to bring back a quick fix made on the Pi
./deploy/sync.sh ssh-key     # once per dev machine: install your SSH key on the Pi
```

* Set the target once per machine in `.sync.env` at the project root (not committed):

  ```
  PI_HOST=pi@auto
  PI_DIR=boxqc
  ```

  `PI_HOST` (the SSH login) has no default and must be set here or given with `--host`.
  The host part `auto` finds the Pi on the network, like `boxqc devices`
  ([Finding the device](#finding-the-device)), so only the user name has to be right. It
  needs the Pi to be running `boxqc device`, and Python 3.10 or newer on the dev machine.
  With several devices, add `PI_NAME=<name>` or `--name <name>`. Otherwise give the
  address, `PI_HOST=pi@192.168.1.50`. With `auto`, SSH keeps the Pi's host key under
  `<hostname>.local` rather than under the address. A changing address then causes no
  host-key warning, and SSH still refuses a machine with a different key that answers in
  the Pi's place.
  `PI_DIR` defaults to a folder in the Pi user's home with the same name as the local
  project folder; `--dir` overrides it for one command.
* Every push and pull shows the list of changes first and asks before doing anything.
  `-y` skips the question.
* **push** makes the Pi's copy match the dev machine, including **deleting** files on
  the Pi that no longer exist on the dev machine. Deletions are listed with the word
  `deleting` in the preview. **pull** never deletes anything locally.
* **push replaces the Pi's `boxqc.yaml`** with the dev machine's, like any other file.
  Keep the Pi's own settings in its `boxqc.local.yaml`
  ([Settings for this Pi](#settings-for-this-pi)).
* Never synced ([deploy/rsync-exclude.txt](../deploy/rsync-exclude.txt)): `.venv/`,
  `data/`, `models/`, `work/`, `boxqc.local.yaml`, caches and `.sync.env`. If
  `data_dir`, `models_dir` or `train.workdir` name other folders in the configuration
  of either machine, those are left alone as well. Each machine keeps its own
  environment and data; models reach the Pi with `boxqc deploy`.
* If a push changes `pyproject.toml`, the script reminds you to re-run
  `./deploy/pi/setup_pi.sh --no-service` on the Pi so it installs any new dependencies.
* The restart runs `sudo systemctl restart boxqc` and may ask for the Pi's password.
  Use `--no-restart` to skip it.
* With two dev machines, the one you switch **to** may be out of date. Before pushing
  from it, run `status` there, and `pull` if the Pi has newer code. Otherwise the push
  overwrites that newer code on the Pi.

## Troubleshooting

* **Status shows `"camera": null` and a `camera_error`; inspections answer 503 and the
  buzzer plays the ERROR pattern:** the camera did not open. Check that `rpicam-hello`
  works and that no other process has the camera (`sudo systemctl stop boxqc` before
  testing), then `sudo systemctl restart boxqc`.
* **Status shows `"camera": "mock"`:** the device is producing synthetic images. That
  happens only where Picamera2 is not installed in the device's environment (or with
  `camera.backend: mock`). On a Pi, run `./deploy/pi/setup_pi.sh` again and read its
  sanity check.
* **The studio says "device offline" or "no device answered":** run `boxqc devices`
  ([Finding the device](#finding-the-device)), or check `device.url` if you set one. Then
  run `curl http://<pi-address>:8000/api/v1/status` from the studio machine, then
  `systemctl status boxqc` and `journalctl -u boxqc -n 50` on the Pi.
* **Requests answer 401:** the device has an API token set and the client does not send
  it. See [device-api.md](device-api.md#authentication).
* **Host doesn't see the Pi over USB:** use a data-capable USB-C cable.
  * macOS: look for a new Ethernet interface in *System Settings → Network*.
  * Linux: check `nmcli device` for a `cdc_ether` interface using `boxqc-usb`.
  * On the Pi: `rpi-usb-gadget status` and `ip addr show usb0`.
* **Link is up but ping fails; the host kernel log shows `transmit queue 0 timed
  out`:** the Pi's gadget isn't reading anything the host sends. Check that nothing
  overrides `rpi-usb-gadget`: no `/etc/modprobe.d/g_ether.conf`, no
  `modules-load=...g_ether` in `cmdline.txt`, and no extra NetworkManager profiles on
  `usb0`. Re-running `sudo ./deploy/pi/usb_gadget.sh` removes the ones older versions
  of this project added. Then reboot the Pi. See also
  [If the link wedges](#if-the-link-wedges).
* **Mac: the USB link worked before, but now `<hostname>.local` doesn't connect (after
  a Pi reboot or after the Mac slept):** see
  [macOS host without Internet Sharing](#macos-host-without-internet-sharing).
* **Pi has no internet over USB:** the host isn't sharing. Check the host profile /
  Internet Sharing. A host firewall that blocks forwarding (ufw/firewalld with a deny
  routing policy) can also stop NAT.
* **Model loads but finds nothing:** make sure the ROI hasn't changed since training,
  and that lighting and focus match the training images.
