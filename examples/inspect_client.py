#!/usr/bin/env python3
"""Trigger an inspection on a BoxQC device and act on the verdict.

This is everything a line controller needs. It uses only the Python standard library,
so it runs anywhere; the same three HTTP calls work from any language (see
docs/device-api.md).

    python examples/inspect_client.py                      # find the device on the network
    python examples/inspect_client.py http://192.168.1.50:8000
    python examples/inspect_client.py --save result.jpg --token SECRET

Without an address it finds the device itself, with the discovery broadcast described
in docs/device-api.md#discovery (find_device() below is the whole protocol). It then
uses the first device that answers, so on a network with several, pass the address.

Exit status: 0 = PASS, 1 = FAIL, 2 = the device could not give a verdict.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import secrets
import socket
import sys
import urllib.error
import urllib.request

DISCOVERY_PORT = 47800


def call(base: str, path: str, *, method: str = "GET", token: str | None = None) -> bytes:
    request = urllib.request.Request(f"{base.rstrip('/')}/api/v1{path}", method=method)
    if token:
        request.add_header("Authorization", f"Bearer {token}")
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read()


def find_device(timeout: float = 1.0) -> str | None:
    """Broadcast a discovery query; return the first answering device's URL, or None.

    (Without a check of the answer's ``proof``, so pass the address when using a token
    on a shared network; boxqc.discovery has the full client.)
    """
    nonce = secrets.token_hex(16)
    query = json.dumps({"boxqc": "discover", "v": 1, "nonce": nonce}).encode().ljust(1024)
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        sock.settimeout(timeout / 3)
        for _ in range(3):      # Wi-Fi can lose a broadcast: send it up to three times
            for target in ("255.255.255.255", "127.0.0.1"):
                with contextlib.suppress(OSError):     # e.g. no route for this address
                    sock.sendto(query, (target, DISCOVERY_PORT))
            try:
                while True:
                    data, (address, _) = sock.recvfrom(4096)
                    try:
                        answer = json.loads(data)
                    except ValueError:
                        continue
                    if answer.get("boxqc") == "device" and answer.get("nonce") == nonce:
                        return f"http://{address}:{answer['port']}"
            except OSError:     # timed out (socket.timeout is an OSError): send again
                continue
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("device", nargs="?", help="device address, e.g. http://192.168.1.50:8000 "
                                                  "(default: find it on the network)")
    parser.add_argument("--no-buzz", action="store_true", help="do not sound the buzzer")
    parser.add_argument("--save", metavar="FILE", help="save the annotated photo here")
    parser.add_argument("--token", help="API token, if the device requires one")
    args = parser.parse_args()
    if args.device is None:
        args.device = find_device()
        if args.device is None:
            print("No device answered on the network; pass its address.", file=sys.stderr)
            return 2
        print(f"Device: {args.device}")

    try:
        buzz = "false" if args.no_buzz else "true"
        result = json.loads(call(args.device, f"/inspect?buzz={buzz}", method="POST", token=args.token))
    except (urllib.error.URLError, TimeoutError) as exc:
        print(f"Inspection failed: {exc}", file=sys.stderr)
        return 2

    print(f"{result['verdict']}  ({result['failing']} detection(s) at or above {result['threshold']}, "
          f"model {result['model']}, {result['timing_ms']['total']} ms)")
    for det in result["detections"]:
        if det["conf"] >= result["threshold"]:
            print(f"  {det['name']:10s} {det['conf']:.2f}  box {det['box']}")
    if args.save:
        image = call(args.device, f"/inspections/{result['id']}/image.jpg?annotated=true", token=args.token)
        with open(args.save, "wb") as f:
            f.write(image)
        print(f"Saved {args.save}")
    return {"PASS": 0, "FAIL": 1}.get(result["verdict"], 2)


if __name__ == "__main__":
    sys.exit(main())
