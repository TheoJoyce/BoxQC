#!/usr/bin/env bash
# Linux dev machine (NetworkManager): share this machine's internet with the Pi over
# the USB gadget link (Pi side: rpi-usb-gadget, see deploy/pi/usb_gadget.sh).
#
#   this machine = 10.42.0.1, runs DHCP + NAT for the link
#   the Pi       = 10.42.0.2, always: DHCP reservation for the hostname the Pi sends
#
# Why 10.42.0.1: the Pi starts out serving the link itself, and rpi-usb-gadget's watcher
# only switches it to "get an address from the host" when it finds a sharing host at one
# of the addresses it knows: 10.42.0.1 (NetworkManager's default for a shared
# connection), 192.168.2.1 (macOS) or 192.168.137.1 (Windows). With any other address
# here the Pi would never ask for one.
#
# The Pi's gadget MAC changes every boot, so the profile matches the USB-gadget
# network driver (cdc_ether) and the reservation matches the Pi's hostname. Note that
# the profile therefore applies to every cdc_ether device plugged into this machine
# (some USB Ethernet adapters, docks and modems): this machine would act as DHCP server
# and router for those too. Remove the profile with --remove if that is in the way.
# One-time setup; persists across reboots and re-plugs.
#
# Usage: ./deploy/host/linux_usb_share.sh --pi-name NAME     (NAME = the Pi's hostname; asks for sudo)
#        ./deploy/host/linux_usb_share.sh --remove
set -euo pipefail

NAME="boxqc-usb"
HOST_ADDR="10.42.0.1/24"
PI_IP="10.42.0.2"
PI_NAME=""
DNSMASQ_CONF="/etc/NetworkManager/dnsmasq-shared.d/boxqc-usb.conf"

REMOVE=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --pi-name) PI_NAME="$2"; shift ;;
        --remove) REMOVE=1 ;;
        *) echo "unknown option: $1" >&2; exit 2 ;;
    esac
    shift
done

command -v nmcli >/dev/null || { echo "NetworkManager (nmcli) is required" >&2; exit 1; }
if [[ $REMOVE -eq 0 && -z "$PI_NAME" ]]; then
    echo "Which Pi? Pass its hostname (run 'hostname' on the Pi): --pi-name NAME" >&2
    echo "The DHCP reservation that gives the Pi $PI_IP is made for that name." >&2
    exit 2
fi

sudo nmcli connection delete "$NAME" >/dev/null 2>&1 || true
sudo rm -f "$DNSMASQ_CONF"
if [[ $REMOVE -eq 1 ]]; then
    echo "Removed profile $NAME and $DNSMASQ_CONF"
    exit 0
fi

# NM's shared-mode dnsmasq reads this directory. Only affects a client named $PI_NAME.
sudo mkdir -p "$(dirname "$DNSMASQ_CONF")"
sudo tee "$DNSMASQ_CONF" >/dev/null <<EOF
# Managed by BoxQC deploy/host/linux_usb_share.sh
# The Pi (DHCP hostname "$PI_NAME") always gets $PI_IP on the USB link.
dhcp-host=$PI_NAME,$PI_IP
EOF

# Only properties that every NetworkManager with "shared" mode knows. The DHCP pool is
# NetworkManager's default for the subnet; the Pi's reserved address lies outside it.
sudo nmcli connection add type ethernet con-name "$NAME" \
    match.driver cdc_ether \
    ipv4.method shared ipv4.addresses "$HOST_ADDR" \
    ipv6.method disabled \
    connection.autoconnect yes connection.autoconnect-priority 50
echo "Created NetworkManager profile '$NAME' (shared, $HOST_ADDR, Pi '$PI_NAME' -> $PI_IP)"

# Bring it up now on any USB gadget interface that's already plugged in
for dev in /sys/class/net/*; do
    iface="$(basename "$dev")"
    [[ "$(basename "$(readlink -f "$dev/device/driver" 2>/dev/null)")" == "cdc_ether" ]] || continue
    sudo nmcli connection up "$NAME" ifname "$iface" >/dev/null && echo "Activated on $iface"
done

# Other shared profiles that aren't bound to a device can grab the Pi instead
while IFS=: read -r cname ctype; do
    [[ "$ctype" == "802-3-ethernet" && "$cname" != "$NAME" ]] || continue
    method="$(nmcli -g ipv4.method connection show "$cname")"
    mac="$(nmcli -g 802-3-ethernet.mac-address connection show "$cname")"
    ifname="$(nmcli -g connection.interface-name connection show "$cname")"
    if [[ "$method" == "shared" && -z "$mac" && -z "$ifname" ]]; then
        echo "Note: profile '$cname' is also a shared profile not bound to any device."
        echo "      It is no longer needed:  nmcli connection delete '$cname'"
    fi
done < <(nmcli -t -f NAME,TYPE connection show)

# Another shared connection (a Wi-Fi hotspot, say) uses the same default address.
if [[ "$(ip -4 -o addr show 2>/dev/null | grep -c " ${HOST_ADDR%/*}/")" -gt 1 ]]; then
    echo "Note: ${HOST_ADDR%/*} may also be in use by another shared connection on this machine"
    echo "      (check 'ip -4 addr'). Only one of the two can work at a time."
fi

echo "Device API: http://$PI_IP:8000  (the Pi may need ~30 s to pick up the new address)"
echo "Web UI: run 'boxqc studio' on this machine; it finds the Pi over the link by itself"
echo "        (or pass --device-url http://$PI_IP:8000)."
