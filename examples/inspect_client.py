#!/usr/bin/env python3
"""Trigger an inspection on a BoxQC device and act on the verdict.

This is everything a line controller needs. It uses only the Python standard library,
so it runs anywhere; the same three HTTP calls work from any language (see
docs/device-api.md).

    python examples/inspect_client.py http://10.42.0.2:8000
    python examples/inspect_client.py http://10.42.0.2:8000 --save result.jpg --token SECRET

Exit status: 0 = PASS, 1 = FAIL, 2 = the device could not give a verdict.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request


def call(base: str, path: str, *, method: str = "GET", token: str | None = None) -> bytes:
    request = urllib.request.Request(f"{base.rstrip('/')}/api/v1{path}", method=method)
    if token:
        request.add_header("Authorization", f"Bearer {token}")
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("device", help="device address, e.g. http://10.42.0.2:8000")
    parser.add_argument("--no-buzz", action="store_true", help="do not sound the buzzer")
    parser.add_argument("--save", metavar="FILE", help="save the annotated photo here")
    parser.add_argument("--token", help="API token, if the device requires one")
    args = parser.parse_args()

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
