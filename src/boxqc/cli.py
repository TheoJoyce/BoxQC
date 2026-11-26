"""`boxqc` command line entry point.

    boxqc device     run the inspection station's API server (on the Pi)
    boxqc studio     run the web UI, dataset and training (on a desktop)
    boxqc devices    find the devices on the local network

The remaining commands are the studio's functions without the browser.
"""

# Each sub-command is one `cmd_*` function taking (cfg, args). They import what they
# need inside the function: `boxqc device` must start on a Pi that has none of the
# studio's or the trainers' dependencies installed, and `--help` should be instant.
# docs/cli.md is the user-facing reference; keep it in step with build_parser().

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from .config import Config, load_config, normalize_device_url
from .plugins import load_plugins


def cmd_device(cfg: Config, args) -> None:
    """Serve the device API until interrupted, and answer discovery queries alongside it."""
    import uvicorn

    from .device.app import create_app, start_discovery

    host, port = args.host or cfg.device.host, args.port or cfg.device.port
    responder = start_discovery(cfg, host, port) if cfg.device.discovery else None
    try:
        uvicorn.run(create_app(cfg), host=host, port=port, log_level="info")
    finally:
        if responder is not None:
            responder.stop()


def cmd_studio(cfg: Config, args) -> None:
    """Serve the studio (web UI + its API) until interrupted."""
    import uvicorn

    from .studio.app import create_app
    from .studio.device_client import DeviceClient, DeviceError

    if args.device_url:
        cfg.device.url = normalize_device_url(args.device_url)
    device = DeviceClient.from_config(cfg)
    try:    # with device.url: auto, find it now so the address is printed
        device.locate()
        where = f"{device.base_url}, found automatically" if device.auto else device.base_url
    except DeviceError as exc:
        where = f"none found yet; the studio searches again when it needs the device. {exc}"
    serve = args.serve or cfg.studio.serve
    host = args.host or cfg.studio.host
    if serve and host in ("127.0.0.1", "localhost", "::1"):   # the default: serve on every interface
        host = "0.0.0.0"
    port = args.port or cfg.studio.port
    print(f"BoxQC studio: http://{'localhost' if host in ('127.0.0.1', '0.0.0.0') else host}:{port}"
          f"   (device: {where})", flush=True)
    key = urls = None
    if serve:
        from .studio.access import load_or_create_key, qr_code, share_urls

        key = load_or_create_key(cfg.data_dir, renew=args.new_key)
        urls = share_urls(port, key, host)
        print("\nServe mode: open the studio on a phone or another computer on this network with")
        for url in urls:
            print(f"    {url}")
        if urls and (qr := qr_code(urls[0])):
            print(qr)
        elif urls:
            print("    (pip install segno to get a QR code here and on the System page)")
        print("The link holds the access key: share it only with people who may use the station.\n"
              "boxqc studio --serve --new-key replaces the key and signs out every other browser.\n", flush=True)
    elif host not in ("127.0.0.1", "localhost", "::1"):
        print(f"Warning: listening on {host} with no access key: anyone on the network can use the studio "
              "and, through it, the device. `--serve` adds an access key.", flush=True)
    uvicorn.run(create_app(cfg, device=device, access_key=key, share_urls=urls), host=host, port=port,
                log_level="info")


def cmd_inspect(cfg: Config, args) -> None:
    """Offline inspection of image files (works on any machine that has the model)."""
    import cv2

    from .device.inspector import verdict_for
    from .imaging import crop_roi, draw_detections
    from .models import list_models, load_model
    from .studio.state import effective_roi

    model = Path(args.model) if args.model else None
    if model is None:
        models = list_models(cfg.models_dir)
        if not models:
            sys.exit(f"No model given and none found in {cfg.models_dir}")
        model = cfg.models_dir / models[0]["name"]
    detector = load_model(model, threads=cfg.inference.threads, fallback_classes=cfg.classes)
    threshold = args.conf if args.conf is not None else cfg.inference.conf_threshold
    roi = cfg.roi if args.config_roi else effective_roi(cfg)
    out_dir = Path(args.out) if args.out else None
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)
    print(f"model={model.name} threshold={threshold} roi={roi}")

    failures = 0
    for path in map(Path, args.images):
        image = cv2.imread(str(path))
        if image is None:
            print(f"{path}: unreadable")
            continue
        crop, (ox, oy) = crop_roi(image, roi)
        dets, ms = detector.detect(crop, cfg.inference.min_report_conf, cfg.inference.iou_threshold)
        dets = [d.shifted(ox, oy) for d in dets]
        verdict = verdict_for(dets, threshold)
        failures += verdict == "FAIL"
        shown = [f"{d.name}:{d.conf:.2f}" for d in dets if d.conf >= threshold]
        print(f"{path.name}: {verdict} ({ms:.0f} ms) {' '.join(shown)}")
        if out_dir:
            cv2.imwrite(str(out_dir / f"{path.stem}_{verdict}.jpg"),
                        draw_detections(image, dets, threshold, verdict, roi))
    print(f"{failures} FAIL / {len(args.images)} images")


def cmd_export(cfg: Config, args) -> None:
    """Write the labeled images as a training dataset (YOLO layout, optionally with COCO files)."""
    from .studio.export import prepare_dataset, write_coco

    try:
        prepared = prepare_dataset(cfg, out_dir=Path(args.out) if args.out else None, dataset=args.dataset)
    except ValueError as exc:       # nothing labeled, or --out names a folder that is not an export
        sys.exit(str(exc))
    if args.format == "coco":
        for split, path in write_coco(prepared).items():
            print(f"  COCO annotations ({split}): {path}")


def cmd_train(cfg: Config, args) -> None:
    """Run the training pipeline in the foreground, then optionally deploy the model."""
    from .discovery import DiscoveryError
    from .studio.device_client import DeviceClient, DeviceError
    from .studio.export import EmptyDatasetError
    from .studio.training import TrainerUnavailable, TrainRequest
    from .studio.training.pipeline import run_training

    request = TrainRequest.from_config(
        cfg, epochs=args.epochs, imgsz=args.imgsz, batch=args.batch, base_model=args.base_model,
        device=args.device, trainer=args.trainer, dataset=args.dataset, name=args.name)
    device = DeviceClient.from_config(cfg, args.device_url)
    if args.deploy and device.auto:
        # Devices answered but none can be chosen (several, or none with device.name): a
        # configuration problem, so say so now rather than after training. A device that
        # is simply off is not: training goes ahead and the deploy reports it.
        try:
            device.locate()
        except DeviceError as exc:
            if isinstance(exc.__cause__, DiscoveryError) and exc.__cause__.found:
                sys.exit(str(exc))
    roi = None
    try:    # crop with the ROI the device uses right now, when it can be asked
        roi = device.settings()["roi"]
    except (DeviceError, KeyError):
        print("Device not reachable: using the last known ROI.")
    try:
        model = run_training(cfg, request, roi=roi)
    except (ValueError, TrainerUnavailable, EmptyDatasetError) as exc:
        sys.exit(str(exc))
    if args.deploy:
        cmd_deploy(cfg, argparse.Namespace(model=str(model), device_url=args.device_url, no_activate=False))


def cmd_evaluate(cfg: Config, args) -> None:
    """Print the image-level PASS/FAIL table of a model over an exported dataset's val split."""
    from .studio.evaluate import evaluate_image_level

    root = Path(args.dataset) if args.dataset else cfg.train.workdir / "yolo_dataset"
    result = evaluate_image_level(Path(args.model), root, cfg.classes)
    if args.json:
        print(json.dumps(result, indent=2))


def cmd_deploy(cfg: Config, args) -> None:
    """Upload a model file (and its metadata sidecar) to the device."""
    from .studio.device_client import DeviceClient, DeviceError

    device = DeviceClient.from_config(cfg, args.device_url)
    model = Path(args.model)
    if not model.is_file():
        sys.exit(f"No such model file: {model}")
    try:
        device.locate()
        print(f"Uploading {model.name} to {device.base_url} ...")
        result = device.push_model(model, activate=not args.no_activate)
    except DeviceError as exc:
        sys.exit(str(exc))
    print(f"The device stored it as {result['name']} (active={result['active']})")


def cmd_devices(cfg: Config, args) -> None:
    """List the devices that answer discovery queries on the local network."""
    from .discovery import discover

    devices = discover(timeout=args.timeout, token=cfg.device.api_token, include_unverified=True)
    if args.json:
        print(json.dumps([{**vars(d), "url": d.url} for d in devices], indent=2))
        return
    if not devices:
        sys.exit("No BoxQC device answered. Is it switched on, on this network, and running a version "
                 "with discovery? See docs/pi-setup.md#finding-the-device.")
    for d in devices:
        note = "  (API token required)" if d.auth and d.verified is None else ""
        if d.verified is False:     # device.url: auto will not use it
            note = f"  NOT VERIFIED: {d.problem}; auto will not use it"
        print(f"{d.name:20s} {d.url:28s} BoxQC {d.version}{note}")
    if len(devices) > 1 and not cfg.device.name:
        print("\nSeveral devices: set device.name in boxqc.local.yaml to choose one for device.url: auto.")


def cmd_plugins(cfg: Config, args) -> None:
    """List the registered model formats and trainers: the way to check a plug-in is found."""
    from .models import FORMATS

    print("Model formats:")
    for name, cls in FORMATS.items():
        print(f"  {name:16s} {', '.join(cls.suffixes):10s} {cls.__module__}.{cls.__qualname__}")
    try:
        from .studio.training import TRAINERS
    except ImportError as exc:      # the studio's dependencies are not installed (e.g. on the Pi)
        print(f"Trainers: unavailable here ({exc})")
        return
    print("Trainers:")
    for name, cls in TRAINERS.items():
        reason = cls.availability()
        print(f"  {name:16s} {'ready' if reason is None else 'unavailable: ' + reason}")


def _removed(message: str):
    """A command handler that only explains what replaced a command of BoxQC 0.1."""
    def run(cfg: Config, args) -> None:
        sys.exit(message)
    return run


def build_parser() -> argparse.ArgumentParser:
    """Define every sub-command and its options. Each parser's ``func`` is its handler."""
    p = argparse.ArgumentParser(prog="boxqc", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-c", "--config", help="path to boxqc.yaml (default: ./boxqc.yaml or $BOXQC_CONFIG)")
    p.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    sub = p.add_subparsers(dest="command", required=True, metavar="COMMAND")

    def device_url(parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--device-url", "--url", dest="device_url",
                            help="device address (default: device.url from the config)")

    s = sub.add_parser("device", aliases=["serve"], help="[Pi] run the device API: camera, inference, buzzer")
    s.add_argument("--host", help="address to listen on (default: device.host from the config, 0.0.0.0)")
    s.add_argument("--port", type=int, help="port to listen on (default: device.port, 8000)")
    s.set_defaults(func=cmd_device)

    s = sub.add_parser("studio", help="[desktop] run the web UI: capture, label, train, deploy")
    s.add_argument("--host", help="address to listen on (default 127.0.0.1; 0.0.0.0 to share on the network)")
    s.add_argument("--port", type=int, help="port to listen on (default: studio.port, 8080)")
    s.add_argument("--serve", action="store_true",
                   help="open the web UI to phones and other computers on the network, behind an access key")
    s.add_argument("--new-key", action="store_true", help="with --serve: replace the access key (signs others out)")
    device_url(s)
    s.set_defaults(func=cmd_studio)

    s = sub.add_parser("train", help="[desktop] prepare, train, export, evaluate")
    s.add_argument("--epochs", type=int, help="training epochs (default: train.epochs)")
    s.add_argument("--imgsz", type=int, help="model input size in pixels (default: train.imgsz)")
    s.add_argument("--batch", type=int, help="batch size, -1 = largest that fits (default: train.batch)")
    s.add_argument("--base-model", help="pretrained model to start from (default: train.base_model)")
    s.add_argument("--device", help="compute device, e.g. auto | mps | cpu | 0")
    s.add_argument("--trainer", help="training backend (default: train.trainer from the config)")
    s.add_argument("--name", help="run / model name (default: boxqc-<date>-<time>)")
    s.add_argument("--dataset", help="named dataset to train on (default: every labeled image)")
    s.add_argument("--deploy", "--push", dest="deploy", action="store_true",
                   help="upload the model to the device afterwards and activate it")
    device_url(s)
    s.set_defaults(func=cmd_train)

    s = sub.add_parser("deploy", aliases=["push-model"], help="[desktop] upload a model to the device")
    s.add_argument("model", help="model file; a metadata .json next to it is uploaded too")
    s.add_argument("--no-activate", action="store_true", help="store it without making it the active model")
    device_url(s)
    s.set_defaults(func=cmd_deploy)

    s = sub.add_parser("export", aliases=["prepare"],
                       help="[desktop] write the labeled images as a YOLO or COCO dataset")
    s.add_argument("--format", choices=["yolo", "coco"], default="yolo",
                   help="coco adds annotations/instances_*.json to the YOLO layout")
    s.add_argument("--out", help="output directory; an earlier export there is replaced, any other "
                                 "non-empty folder is refused (default: <workdir>/yolo_dataset)")
    s.add_argument("--dataset", help="named dataset to use (default: every labeled image)")
    s.set_defaults(func=cmd_export)

    s = sub.add_parser("evaluate", help="[desktop] image-level PASS/FAIL evaluation of a model")
    s.add_argument("model", help="model file to evaluate")
    s.add_argument("--dataset", help="exported dataset directory (default: <workdir>/yolo_dataset)")
    s.add_argument("--json", action="store_true", help="also print the full result as JSON")
    s.set_defaults(func=cmd_evaluate)

    s = sub.add_parser("inspect", help="run a model on image files")
    s.add_argument("images", nargs="+", help="image files to inspect")
    s.add_argument("--model", help="model file (default: newest in models_dir)")
    s.add_argument("--conf", type=float, help="FAIL threshold (default: inference.conf_threshold)")
    s.add_argument("--out", help="directory to write annotated images to")
    s.add_argument("--config-roi", action="store_true", help="use roi from boxqc.yaml, ignoring saved settings")
    s.set_defaults(func=cmd_inspect)

    s = sub.add_parser("devices", help="find the devices on the local network and print their addresses")
    s.add_argument("--timeout", type=float, default=1.0, help="seconds to wait for answers (default 1)")
    s.add_argument("--json", action="store_true", help="print the answers as JSON")
    s.set_defaults(func=cmd_devices)

    s = sub.add_parser("plugins", help="list the model formats and trainers that are installed")
    s.set_defaults(func=cmd_plugins)

    # Commands of BoxQC 0.1 that no longer exist: explain instead of "invalid choice".
    s = sub.add_parser("agent")
    s.add_argument("rest", nargs=argparse.REMAINDER)
    s.set_defaults(func=_removed(
        "`boxqc agent` is gone: training now runs inside the studio. Start `boxqc studio` instead."))
    s = sub.add_parser("pull")
    s.add_argument("rest", nargs=argparse.REMAINDER)
    s.set_defaults(func=_removed(
        "`boxqc pull` is gone: the dataset now lives on this computer and the device stores none.\n"
        "To copy a dataset that BoxQC 0.1 left on the Pi, run: ./deploy/sync.sh fetch-data"))
    return p


def main(argv: list[str] | None = None) -> None:
    """Parse the command line, load the configuration and plug-ins, run the command."""
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    if not args.verbose:
        logging.getLogger("httpx").setLevel(logging.WARNING)   # no log line per request to the device
    try:
        cfg = load_config(args.config)
    except (ValueError, OSError) as exc:
        sys.exit(f"Configuration error: {exc}")
    load_plugins(cfg)
    args.func(cfg, args)


if __name__ == "__main__":
    main()
