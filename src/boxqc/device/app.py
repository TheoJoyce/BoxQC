"""The device API: the HTTP server that runs on the inspection station (the Pi).

It is self-contained. With a model loaded it inspects boxes on a button press or an API
call and sounds the buzzer, whether or not any other machine is connected. It keeps
only what it needs for that: the models, four runtime settings and a bounded
inspection log. It has no web UI and stores no training data; those live in
:mod:`boxqc.studio`, which is one client of this API among others (a PLC, a script).

Routes are served under ``/api/v1``. The same routes also answer under ``/api`` (the
unversioned prefix of BoxQC 0.1) so existing integrations keep working. The reference
is ``docs/device-api.md``; a running device serves interactive docs at ``/docs``.
"""

from __future__ import annotations

import logging
import re
import secrets
import shutil
import socket
import subprocess
import tempfile
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import APIRouter, Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, Response, StreamingResponse

from .. import __version__
from ..config import Config
from ..discovery import DISCOVERY_PORT, DiscoveryResponder, ipv4_interfaces
from ..imaging import decode_image, draw_guides, encode_jpeg
from ..models import ModelError, is_model_file, list_models, load_model, metadata_path, model_suffixes, validate_model
from .camera import Camera, open_camera
from .hardware import Signaller
from .inspector import InspectionLog, Inspector, model_usage
from .schemas import BuzzerTest, SettingsUpdate
from .settings import RuntimeSettings

log = logging.getLogger(__name__)

API_VERSION = "1"
_MODEL_NAME_RE = re.compile(r"[^A-Za-z0-9._-]+")
UPLOAD_PREFIX = ".upload-"      # staging folders inside models_dir, one per upload in progress


def local_addresses() -> dict[str, list[str]]:
    """IPv4 addresses per interface (Linux `ip`; empty elsewhere)."""
    try:
        out = subprocess.run(["ip", "-4", "-o", "addr", "show"], capture_output=True, text=True,
                             timeout=2).stdout
    except (OSError, subprocess.SubprocessError):
        return {}
    addrs: dict[str, list[str]] = {}
    for line in out.splitlines():
        parts = line.split()
        if len(parts) >= 4 and parts[1] != "lo":
            addrs.setdefault(parts[1], []).append(parts[3])
    return addrs


class DeviceState:
    """Everything the device owns: camera, buzzer, model, settings and the inspection log."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.settings = RuntimeSettings(cfg.data_dir / "settings.json", cfg)
        self.log = InspectionLog(cfg.data_dir / "inspections", cfg.inference.keep_inspections,
                                 cfg.inference.save_annotated)
        self.inspector = Inspector(cfg, self.settings)
        self.camera: Camera | None = None
        self.signaller: Signaller | None = None
        self.camera_error: str | None = None
        self.session_counts: dict[str, int] = {}
        self.stopping = threading.Event()
        self._inspect_lock = threading.Lock()

    def start(self) -> None:
        """Open the camera and GPIO. A camera that fails to open is not fatal: the API
        still starts, reports the error in its status and answers camera routes with 503."""
        try:
            self.camera = open_camera(self.cfg.camera)
        except Exception as exc:
            self.camera_error = str(exc)
            log.exception("Camera failed to start")
        # uploads that a crash or power cut interrupted (see upload_model)
        for leftover in self.cfg.models_dir.glob(f"{UPLOAD_PREFIX}*"):
            shutil.rmtree(leftover, ignore_errors=True)
        self.signaller = Signaller(self.cfg.hardware, buzzer_setting=lambda: self.settings.get("buzzer"))
        self.signaller.on_trigger(lambda: self.inspect(source="trigger"))

    def stop(self) -> None:
        """Release the hardware; ``stopping`` also ends any live-stream generators."""
        self.stopping.set()
        if self.camera:
            self.camera.close()
        if self.signaller:
            self.signaller.close()

    def require_camera(self) -> Camera:
        """The camera, or an HTTP 503 explaining why there is none."""
        if self.camera is None:
            raise HTTPException(503, f"Camera unavailable: {self.camera_error}")
        return self.camera

    def inspect(self, image=None, source: str = "camera", buzz: bool = True) -> dict:
        """Capture (unless an image is given), run the model, signal and log the result."""
        with self._inspect_lock:  # one inspection at a time (button + API)
            t0 = time.perf_counter()
            try:
                if image is None:
                    image = self.require_camera().capture()
                capture_ms = (time.perf_counter() - t0) * 1000
                result = self.inspector.run(image, item_id=self.log.new_id())
            except Exception:
                if buzz and self.signaller:
                    self.signaller.signal("ERROR")
                raise
            result["source"] = source
            result["timing_ms"]["capture"] = round(capture_ms, 1)
            result["timing_ms"]["total"] = round((time.perf_counter() - t0) * 1000, 1)
            if buzz and self.signaller:
                self.signaller.signal(result["verdict"])
            self.session_counts[result["verdict"]] = self.session_counts.get(result["verdict"], 0) + 1
            if self.cfg.inference.save_inspections:
                self.log.save(result, image)
            return result


def _token_guard(token: str | None):
    """A dependency that requires ``Authorization: Bearer <token>`` (or ``X-API-Key``) when a token is set."""

    def guard(request: Request) -> None:
        if not token:
            return
        auth = request.headers.get("authorization", "")
        supplied = auth[7:] if auth.lower().startswith("bearer ") else request.headers.get("x-api-key", "")
        if not secrets.compare_digest(supplied.encode(), token.encode()):
            raise HTTPException(401, "Missing or wrong API token", headers={"WWW-Authenticate": "Bearer"})

    return guard


_LANDING = """<!doctype html>
<html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>BoxQC device</title>
<body style="font: 16px/1.5 system-ui, sans-serif; max-width: 40rem; margin: 3rem auto; padding: 0 1rem">
<h1>BoxQC device</h1>
<p>This is the inspection station's API server (version {version}, API v{api}).
It has no web interface of its own.</p>
<ul>
  <li><a href="/docs">API documentation</a></li>
  <li><a href="/api/v1/status">Status</a></li>
</ul>
<p>For the web interface, run <code>boxqc studio</code> on a desktop computer on the same
network: it finds this device by itself (<code>boxqc devices</code> lists the devices it
finds). Otherwise, pass <code>--device-url http://&lt;this-address&gt;:{port}</code>.</p>
</body></html>
"""


def start_discovery(cfg: Config, host: str, port: int,
                    discovery_port: int = DISCOVERY_PORT) -> DiscoveryResponder | None:
    """Answer discovery queries (:mod:`boxqc.discovery`) for an API served on ``host:port``.

    Started by ``boxqc device`` next to the HTTP server, not by :func:`create_app`, so
    that building the app (as the tests do) opens no sockets. Returns None, after a
    warning, if the port cannot be opened: the station works without discovery, it
    just has to be addressed directly.
    """
    loopback = host.startswith("127.") or host == "localhost"
    hostname = socket.gethostname().split(".")[0]
    cache = {"at": -1e9, "addresses": []}   # the interface list, re-read at most every 5 s

    def info() -> dict:
        if loopback:
            addresses = ["127.0.0.1"]
        elif host not in ("0.0.0.0", ""):
            addresses = [host]
        else:
            if time.monotonic() - cache["at"] > 5.0:
                cache.update(at=time.monotonic(), addresses=[i.address for i in ipv4_interfaces()] + ["127.0.0.1"])
            addresses = cache["addresses"]
        return {"name": cfg.device.name or hostname, "hostname": hostname, "version": __version__,
                "api_version": API_VERSION, "port": port, "addresses": addresses}

    responder = DiscoveryResponder(info, token=cfg.device.api_token,
                                   host="127.0.0.1" if loopback else "0.0.0.0", port=discovery_port)
    try:
        responder.start()
    except OSError as exc:
        log.warning("Discovery is off: UDP port %s could not be opened (%s). Address this device "
                    "directly (device.url on the studio machine).", discovery_port, exc)
        return None
    log.info("Answering discovery queries on UDP port %s as %r", responder.port, info()["name"])
    return responder


def create_app(cfg: Config) -> FastAPI:
    """Build the device application. The camera and GPIO are opened when the server starts."""
    state = DeviceState(cfg)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        state.start()
        yield
        state.stop()

    app = FastAPI(
        title="BoxQC device API",
        version=__version__,
        description="Camera, inference, PASS/FAIL signalling and model management for one "
                    "inspection station. Routes are under `/api/v1`.",
        lifespan=lifespan,
    )
    app.state.boxqc = state     # reachable from outside (the tests swap in a fake model)
    upload_lock = threading.Lock()
    # Every API route is defined once on `api` and mounted twice at the bottom of this
    # function: under /api/v1 (documented) and under /api (the 0.1 prefix, hidden).
    # `legacy` holds the two paths that were also renamed, not just re-prefixed.
    api = APIRouter(dependencies=[Depends(_token_guard(cfg.device.api_token))])
    legacy = APIRouter(dependencies=[Depends(_token_guard(cfg.device.api_token))])

    @app.get("/", include_in_schema=False, response_class=HTMLResponse)
    def landing():
        return _LANDING.format(version=__version__, api=API_VERSION, port=cfg.device.port)

    # ---- status / settings ----------------------------------------------
    @api.get("/status", tags=["status"], summary="Health of the camera, model and hardware")
    def status():
        disk = shutil.disk_usage(cfg.data_dir)
        detector = state.inspector.detector
        return {
            "version": __version__,
            "api_version": API_VERSION,
            "hostname": socket.gethostname(),
            "addresses": local_addresses(),
            "camera": state.camera.backend if state.camera else None,
            "camera_error": state.camera_error,
            "hardware": state.signaller.status() if state.signaller else None,
            "model": state.inspector.model_name,
            "model_error": state.inspector.load_error,
            "model_info": detector.describe() if hasattr(detector, "describe") else None,
            # class names in use: the loaded model's, else the configured fallback
            "classes": list(getattr(detector, "names", None) or cfg.classes),
            "model_formats": {"suffixes": list(model_suffixes())},
            "settings": state.settings.as_dict(),
            "session_counts": state.session_counts,
            "disk_free_gb": round(disk.free / 1e9, 1),
            "config_file": str(cfg.source) if cfg.source else None,
        }

    @api.get("/settings", tags=["settings"], summary="Runtime settings")
    def get_settings():
        return state.settings.as_dict()

    @api.put("/settings", tags=["settings"], summary="Change the threshold, the ROI and/or the buzzer beeps")
    def put_settings(body: SettingsUpdate):
        try:
            return state.settings.update(**body.model_dump(exclude_none=True))
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc

    # ---- camera ---------------------------------------------------------
    @api.get("/camera/stream.mjpg", tags=["camera"], summary="Live preview (MJPEG)")
    def stream(guides: bool = True):
        """Low-resolution live view as `multipart/x-mixed-replace`. `guides` draws the ROI and a centre cross."""
        camera = state.require_camera()
        interval = 1.0 / max(1.0, cfg.camera.preview_fps)

        def frames():
            # One JPEG per multipart chunk, paced to preview_fps. The generator ends when
            # the client disconnects (the write fails) or the server shuts down.
            while not state.stopping.is_set():
                t = time.perf_counter()
                frame = camera.preview()
                if guides:
                    frame = draw_guides(frame, state.settings.get("roi"))
                jpg = encode_jpeg(frame, 75)
                yield (b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                       + str(len(jpg)).encode() + b"\r\n\r\n" + jpg + b"\r\n")
                time.sleep(max(0.0, interval - (time.perf_counter() - t)))

        return StreamingResponse(frames(), media_type="multipart/x-mixed-replace; boundary=frame")

    @api.get("/camera/snapshot.jpg", tags=["camera"], summary="One preview frame (JPEG)")
    def snapshot(full: bool = False):
        """A single frame without guides. `full=true` returns a full-resolution capture."""
        camera = state.require_camera()
        frame = camera.capture() if full else camera.preview()
        return Response(encode_jpeg(frame, 90), media_type="image/jpeg",
                        headers={"Cache-Control": "no-store"})

    @api.post("/camera/capture", tags=["camera"], summary="Take a full-resolution photo (JPEG)")
    def capture():
        """Returns the photo; nothing is stored on the device. This is how the studio
        collects training images."""
        frame = state.require_camera().capture()
        h, w = frame.shape[:2]
        return Response(encode_jpeg(frame, cfg.camera.jpeg_quality), media_type="image/jpeg",
                        headers={"Cache-Control": "no-store", "X-Image-Width": str(w), "X-Image-Height": str(h)})

    # ---- inspection -----------------------------------------------------
    def _inspect(**kwargs) -> dict:
        """Run an inspection, turning any failure into a JSON error like every other route."""
        try:
            return state.inspect(**kwargs)
        except HTTPException:
            raise
        except Exception as exc:
            log.exception("Inspection failed")
            raise HTTPException(500, f"Inspection failed: {exc}") from exc

    @api.post("/inspect", tags=["inspection"], summary="Capture, run the model, signal PASS/FAIL")
    def inspect(buzz: bool = True):
        return _inspect(source="camera", buzz=buzz)

    @api.post("/inspect/upload", tags=["inspection"], summary="Run the model on an uploaded image")
    def inspect_upload(file: UploadFile = File(...), buzz: bool = False):
        try:
            image = decode_image(file.file.read())
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        return _inspect(image=image, source="upload", buzz=buzz)

    @api.get("/inspections", tags=["inspection"], summary="Recent inspection records, newest first")
    def inspections(limit: int = 50):
        return state.log.list(limit)

    @api.get("/inspections.csv", tags=["inspection"], summary="The inspection log as CSV")
    def inspections_csv(limit: int = 1000):
        lines = ["id,time,source,verdict,threshold,failing,top_conf,top_class,model,inference_ms"]
        for r in state.log.list(limit):
            top = max(r["detections"], key=lambda d: d["conf"], default=None)
            lines.append(",".join(str(v) for v in (
                r["id"], r["time"], r.get("source", ""), r["verdict"], r["threshold"], r["failing"],
                top["conf"] if top else "", top["name"] if top else "", r["model"] or "",
                r["timing_ms"].get("inference", ""),
            )))
        return Response("\n".join(lines) + "\n", media_type="text/csv",
                        headers={"Content-Disposition": "attachment; filename=inspections.csv"})

    def _get_record(item_id: str) -> dict:
        try:
            return state.log.get(item_id)
        except (KeyError, ValueError) as exc:
            raise HTTPException(404, "inspection not found") from exc

    @api.get("/inspections/{item_id}", tags=["inspection"], summary="One inspection record")
    def inspection(item_id: str):
        return _get_record(item_id)

    @api.get("/inspections/{item_id}/image.jpg", tags=["inspection"], summary="The inspected frame (JPEG)")
    def inspection_image(item_id: str, annotated: bool = False):
        """`annotated=true` returns the frame with the boxes and the verdict drawn on it."""
        _get_record(item_id)
        try:
            data = state.log.image_jpeg(item_id, annotated)
        except KeyError as exc:
            raise HTTPException(404, "image not stored") from exc
        return Response(data, media_type="image/jpeg")

    # ---- models ---------------------------------------------------------
    @api.get("/models", tags=["models"], summary="Models stored on the device")
    def models(usage: bool = False):
        """`usage=true` adds per-model inspection counts and the median inference time."""
        active = state.inspector.model_name
        stats = model_usage(state.log.list(limit=cfg.inference.keep_inspections)) if usage else {}
        return [{**m, "active": m["name"] == active, "usage": stats.get(m["name"])}
                for m in list_models(cfg.models_dir)]

    @api.post("/models", tags=["models"], summary="Upload a model (and its metadata)")
    def upload_model(file: UploadFile = File(...), meta: UploadFile | None = File(None),
                     activate: bool = Form(True)):
        """`file` is the model, `meta` its optional JSON metadata sidecar (see docs/models.md).
        The model is loaded and run once before it is accepted; a model that fails is discarded."""
        source = Path(file.filename or "model.onnx")
        if not is_model_file(source):
            raise HTTPException(400, f"Unsupported model file type {source.suffix!r} "
                                     f"(supported: {', '.join(model_suffixes())})")
        # The file name comes from the client: keep only safe characters, never a path.
        stem = _MODEL_NAME_RE.sub("_", source.stem) or "model"
        suffix = source.suffix.lower()
        cfg.models_dir.mkdir(parents=True, exist_ok=True)
        # Receive and check the upload in a staging folder, then move it into place. An
        # upload that fails, or that a power cut interrupts, therefore never appears in
        # models_dir as a model (where it could be picked up as "the newest model").
        staging = Path(tempfile.mkdtemp(prefix=UPLOAD_PREFIX, dir=cfg.models_dir))
        try:
            staged = staging / f"{stem}{suffix}"
            with staged.open("wb") as out:
                shutil.copyfileobj(file.file, out)
            if meta is not None:
                metadata_path(staged).write_bytes(meta.file.read())
            try:
                validate_model(load_model(staged, threads=1, fallback_classes=cfg.classes))
            except Exception as exc:
                detail = str(exc) if isinstance(exc, ModelError) else f"{type(exc).__name__}: {exc}"
                raise HTTPException(400, f"Not a usable model: {detail}") from exc
            with upload_lock:       # choosing a free name and taking it must not interleave
                if (cfg.models_dir / f"{stem}{suffix}").exists():
                    stem = f"{stem}-{time.strftime('%Y%m%d-%H%M%S')}"
                dest = cfg.models_dir / f"{stem}{suffix}"
                if meta is not None:
                    metadata_path(staged).replace(metadata_path(dest))
                else:               # a sidecar left behind by a model of that name is not this model's
                    metadata_path(dest).unlink(missing_ok=True)
                staged.replace(dest)    # last, so the model never shows up without its metadata
        finally:
            shutil.rmtree(staging, ignore_errors=True)
        if activate:
            state.settings.update(active_model=dest.name)
            state.inspector.reload()
        return {"name": dest.name, "active": state.inspector.model_name == dest.name}

    def _model_path(name: str) -> Path:
        """Resolve a model name from the URL to a file inside models_dir, or 404."""
        path = cfg.models_dir / Path(name).name      # .name: no directory traversal
        if not is_model_file(path) or not path.is_file():
            raise HTTPException(404, "model not found")
        return path

    @api.post("/models/{name}/activate", tags=["models"], summary="Make a stored model the active one")
    def activate_model(name: str):
        """The model is loaded and run once first. If that fails the answer is 500 and the
        model that was active stays active."""
        path = _model_path(name)
        try:
            detector = state.inspector.load(path)
            validate_model(detector)
        except Exception as exc:
            log.exception("Could not activate %s", path.name)
            raise HTTPException(500, f"Failed to load {path.name}: {exc}") from exc
        state.settings.update(active_model=path.name)
        state.inspector.use(detector)
        return {"active": path.name}

    @api.delete("/models/{name}", tags=["models"], summary="Delete a stored model")
    def delete_model(name: str):
        path = _model_path(name)
        if state.inspector.model_name == path.name:
            raise HTTPException(400, "cannot delete the active model")
        path.unlink()
        metadata_path(path).unlink(missing_ok=True)
        return {"deleted": path.name}

    # ---- hardware -------------------------------------------------------
    @api.post("/buzzer/test", tags=["hardware"], summary="Play a buzzer pattern")
    def buzzer_test(body: BuzzerTest):
        if state.signaller is None:
            raise HTTPException(503, "hardware not started")
        state.signaller.signal(body.pattern.upper())
        return {"played": body.pattern.upper(), "mock": state.signaller.is_mock}

    # ---- pre-v1 paths that were renamed -----------------------------------
    legacy.add_api_route("/stream.mjpg", stream, methods=["GET"])
    legacy.add_api_route("/snapshot.jpg", snapshot, methods=["GET"])

    app.include_router(api, prefix=f"/api/v{API_VERSION}")
    app.include_router(api, prefix="/api", include_in_schema=False)      # BoxQC 0.1 prefix
    app.include_router(legacy, prefix="/api", include_in_schema=False)
    return app
