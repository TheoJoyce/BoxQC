#!/usr/bin/env bash
# USB networking between the Pi and a dev machine, using Raspberry Pi's own
# `rpi-usb-gadget` package (Pi 4 / Pi 5 USB-C port, Zero 2 W "USB" port).
#
# The package does all the Pi-side work:
#   * loads the g_ether gadget (Raspberry Pi USB IDs) and the dwc2 peripheral overlay
#   * "USB Gadget (client)": DHCP client - when the host shares its internet, the Pi
#     gets an address + internet over USB (route metric 100 beats Wi-Fi's 600)
#   * "USB Gadget (shared)": the Pi serves DHCP itself on 10.12.194.1/28 when the host
#     does not share; the rpi-usb-gadget-ics watcher switches between the two
#
# The fixed Pi address 10.42.0.2 comes from the *host* (deploy/host/linux_usb_share.sh).
#
# rpi-usb-gadget is published for Raspberry Pi OS Trixie. Where apt cannot find it, this
# script stops before changing anything.
#
# Your own gadget tweaks are left alone - e.g. a fixed Pi MAC in
# /etc/modprobe.d/g_ether-mac.conf (`options g_ether dev_addr=...`), which some Pis need
# to stop the gadget wedging. Don't name it g_ether.conf: that replaces the package's
# /usr/lib/modprobe.d/g_ether.conf (Raspberry Pi USB IDs) instead of adding to it. This script only removes what earlier BoxQC versions of
# it added (their 02:b0:c5:* MAC file, cmdline modules-load, extra NM profile,
# DHCP/uplink services).
#
# Usage: sudo ./deploy/pi/usb_gadget.sh     (then reboot). Safe to re-run.
set -euo pipefail
[[ $EUID -eq 0 ]] || { echo "run with sudo"; exit 1; }

if [[ -f /boot/firmware/cmdline.txt ]]; then BOOT=/boot/firmware; else BOOT=/boot; fi

# Install the package first. The clean-up below removes the gadget set-up that earlier
# versions of this script made, and must not run unless its replacement is in place.
echo "==> rpi-usb-gadget"
if ! command -v rpi-usb-gadget >/dev/null; then
    apt-get update
    if ! apt-cache show rpi-usb-gadget >/dev/null 2>&1; then
        echo "The rpi-usb-gadget package is not available for this OS release (it is published" >&2
        echo "for Raspberry Pi OS Trixie). Nothing was changed. Use Wi-Fi or Ethernet to reach" >&2
        echo "the Pi, or upgrade the OS." >&2
        exit 1
    fi
    apt-get install -y rpi-usb-gadget
fi

echo "==> removing changes made by earlier versions of this script"
# /etc/modprobe.d/g_ether.conf fully replaced the package's /usr/lib/modprobe.d/g_ether.conf
if grep -qs '02:b0:c5:00:00' /etc/modprobe.d/g_ether.conf; then
    rm -f /etc/modprobe.d/g_ether.conf
    echo "    removed /etc/modprobe.d/g_ether.conf"
fi
# the package loads g_ether via /etc/modules-load.d/usb-gadget.conf instead
if grep -q ' modules-load=dwc2,g_ether' "$BOOT/cmdline.txt"; then
    sed -i 's/ modules-load=dwc2,g_ether//' "$BOOT/cmdline.txt"
    echo "    removed modules-load=dwc2,g_ether from $BOOT/cmdline.txt"
fi
if nmcli -t -f NAME connection show 2>/dev/null | grep -qx 'usb-gadget'; then
    nmcli connection delete usb-gadget >/dev/null
    echo "    removed NetworkManager profile 'usb-gadget'"
fi
for unit in boxqc-usb-dhcp.service boxqc-usb-uplink.timer boxqc-usb-uplink.service; do
    systemctl disable --now "$unit" >/dev/null 2>&1 || true
    rm -f "/etc/systemd/system/$unit"
done
rm -f /usr/local/sbin/boxqc-usb-uplink /etc/boxqc/usb-dhcp.conf
rmdir /etc/boxqc 2>/dev/null || true
systemctl daemon-reload

echo "==> gadget mode"
if [[ -f /etc/modules-load.d/usb-gadget.conf ]]; then
    echo "    already on - keeping your existing profiles/mode"
else
    rpi-usb-gadget on
fi
rpi-usb-gadget status || true

echo
echo "Done. Reboot the Pi, then on each Linux dev machine run:"
echo "    ./deploy/host/linux_usb_share.sh --pi-name $(hostname)"
echo "The device API will then be at http://10.42.0.2:8000 over USB."
echo "(On a Mac, use Internet Sharing instead; see docs/pi-setup.md.)"
