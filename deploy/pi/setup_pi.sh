#!/usr/bin/env bash
# Install the BoxQC device (the inspection station's API server) on a Raspberry Pi
# (Raspberry Pi OS Bookworm/Trixie, 64-bit). Only what the device needs is installed:
# no web UI and no training libraries - those run on a desktop (`boxqc studio`).
#
# Camera, NumPy, OpenCV and GPIO libraries come from apt so they match the
# system libcamera build; the venv sees them via --system-site-packages.
# Usage:  ./deploy/pi/setup_pi.sh [--no-service]
# Run it as the user the device should run as (it asks for sudo where needed), not as
# root. Safe to re-run: it reinstalls the Python side and the service, and restarts it.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
INSTALL_SERVICE=1
for arg in "$@"; do
    case "$arg" in
        --no-service) INSTALL_SERVICE=0 ;;
        *) echo "unknown option: $arg   (usage: ./deploy/pi/setup_pi.sh [--no-service])" >&2; exit 2 ;;
    esac
done
if [[ $EUID -eq 0 ]]; then
    echo "Run this as the user the device should run as, without sudo: the virtualenv and" >&2
    echo "the service would otherwise belong to root." >&2
    exit 1
fi

echo "==> apt packages"
sudo apt-get update
sudo apt-get install -y \
    python3-venv python3-pip \
    python3-picamera2 python3-opencv python3-numpy python3-yaml \
    python3-gpiozero python3-lgpio

echo "==> virtualenv at $REPO_DIR/.venv"
cd "$REPO_DIR"
# A .venv copied from another machine has symlinks to a Python that doesn't exist
# here (or lacks system site-packages, hiding apt's picamera2). Rebuild it.
if [[ -e .venv || -L .venv ]]; then
    if ! .venv/bin/python -c 'import sys' >/dev/null 2>&1 \
        || ! grep -qi '^include-system-site-packages *= *true' .venv/pyvenv.cfg 2>/dev/null; then
        echo "    existing .venv is broken or from another machine - recreating"
        rm -rf .venv src/*.egg-info
    fi
fi
python3 -m venv --system-site-packages .venv
.venv/bin/pip install --upgrade pip

# Never let pip replace the apt NumPy (picamera2/simplejpeg are built against it).
CONSTRAINTS="$(mktemp)"
python3 -c 'import numpy; print(f"numpy=={numpy.__version__}")' > "$CONSTRAINTS"

# Keep this list in step with the "device" extra in pyproject.toml.
.venv/bin/pip install -c "$CONSTRAINTS" \
    "fastapi>=0.110" "uvicorn[standard]>=0.29" "python-multipart>=0.0.9" \
    "onnxruntime>=1.17"
# --no-deps: numpy/opencv/pyyaml/gpiozero are already provided by apt
.venv/bin/pip install --no-deps -e .
rm -f "$CONSTRAINTS"

echo "==> sanity check"
# With camera.backend "auto" (the default) or "picamera2" the station needs Picamera2.
# If it cannot be imported, stop here: "auto" would otherwise run the mock camera, and a
# station that inspects synthetic images passes every box.
.venv/bin/python - <<'EOF'
import sys
import cv2, numpy, onnxruntime, fastapi
from boxqc.config import load_config

print("numpy", numpy.__version__, "| opencv", cv2.__version__, "| onnxruntime", onnxruntime.__version__)
backend = load_config("boxqc.yaml").camera.backend.lower()
try:
    import picamera2  # noqa: F401
    print("picamera2 OK")
except Exception as e:
    print("picamera2 NOT available:", e)
    if backend in ("auto", "picamera2"):
        sys.exit("ERROR: camera.backend is %r, which needs Picamera2 on this machine.\n"
                 "       Fix the import above (sudo apt-get install python3-picamera2), or, for a USB\n"
                 "       webcam, set camera.backend: opencv in boxqc.local.yaml. Then run this script again."
                 % backend)
EOF

if [[ $INSTALL_SERVICE -eq 1 ]]; then
    echo "==> systemd service"
    UNIT=/etc/systemd/system/boxqc.service
    ENV_FILE=/etc/default/boxqc
    # sed replacement text: escape the characters that are special there
    esc() { printf '%s' "$1" | sed -e 's/[&|\\]/\\&/g'; }
    NEW_UNIT="$(sed -e "s|@USER@|$(esc "$USER")|g" -e "s|@DIR@|$(esc "$REPO_DIR")|g" deploy/pi/boxqc.service)"

    if sudo test -f "$UNIT"; then
        # Earlier instructions said to put the API token into the unit file, which this
        # script rewrites. Move it to $ENV_FILE, which it never touches.
        old_token="$(sudo sed -n -E 's/^Environment="?BOXQC_API_TOKEN=([^"]*)"?[[:space:]]*$/\1/p' "$UNIT" | tail -n 1)"
        if [[ -n "$old_token" ]] && ! sudo grep -qs '^BOXQC_API_TOKEN=' "$ENV_FILE"; then
            printf 'BOXQC_API_TOKEN=%s\n' "$old_token" | sudo tee -a "$ENV_FILE" > /dev/null
            echo "    moved the API token from the old unit file to $ENV_FILE"
        fi
        # Any other change made to the installed unit would be lost: keep a copy.
        if [[ "$(sudo cat "$UNIT")" != "$NEW_UNIT" ]]; then
            sudo cp "$UNIT" "$UNIT.bak"
            echo "    the installed unit differed from the template; the old one is kept as $UNIT.bak"
        fi
    fi
    if sudo test -f "$ENV_FILE"; then
        sudo chown root:root "$ENV_FILE"
        sudo chmod 600 "$ENV_FILE"      # it may hold the API token
    fi
    printf '%s\n' "$NEW_UNIT" | sudo tee "$UNIT" > /dev/null
    sudo systemctl daemon-reload
    sudo systemctl enable boxqc.service
    sudo systemctl restart boxqc.service     # restart, so a re-run picks up new code
    sleep 3
    if systemctl is-active --quiet boxqc.service; then
        echo "Service running. Logs: journalctl -u boxqc -f"
    else
        echo "ERROR: the boxqc service did not stay up. Check: journalctl -u boxqc -n 50" >&2
        exit 1
    fi
fi

echo
echo "Done. Device API: http://$(hostname).local:8000  (docs at /docs)"
echo "On a desktop, open the web UI with:  boxqc studio --device-url http://$(hostname).local:8000"
echo "Settings for this Pi only (pins, camera controls) go in boxqc.local.yaml next to"
echo "boxqc.yaml; see docs/pi-setup.md."
echo "USB networking to a dev machine is set up separately: sudo ./deploy/pi/usb_gadget.sh"
