"""The studio: the web UI and its API, run on a desktop computer as ``boxqc studio``.

It owns everything that is not needed to inspect a box: the image dataset, labeling,
training and the trained models. The device is reached over HTTP:

* ``/api/device/...`` forwards a request to the device API unchanged, so the browser
  only ever talks to the studio (one origin, and the device's API token stays here).
* A few routes combine both sides, e.g. ``POST /api/capture`` asks the device for a
  photo and files it in the dataset.

With the device switched off, the dataset, labeling and training keep working.

The studio's own ``/api`` routes back its web pages. They are documented at ``/docs``
but, unlike the device API, are not a versioned public interface.
"""

from __future__ import annotations

import contextlib
import logging
import shutil
import socket
import tempfile
import threading
import time
import zipfile
from contextlib import asynccontextmanager
from pathlib import Path
from urllib.parse import urlsplit

import anyio
import httpx
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field
from starlette.background import BackgroundTask
from starlette.datastructures import Headers

from .. import __version__
from ..config import Config
from ..models import is_model_file, list_models, metadata_path
from .assistant import AssistantUnavailable, LabelAssistant
from .dataset import STATUSES, Box, DatasetStore
from .device_client import API_PREFIX, DeviceClient, DeviceError
from .state import StudioState
from .training import TrainerUnavailable, TrainRequest
from .training.jobs import TrainingManager

log = logging.getLogger(__name__)
HERE = Path(__file__).parent
PROXY_METHODS = ["GET", "POST", "PUT", "PATCH", "DELETE"]
PROXY_RESPONSE_HEADERS = ("content-type", "content-disposition", "cache-control")
SAFE_METHODS = ("GET", "HEAD", "OPTIONS")


# ---- request bodies ---------------------------------------------------------
class CaptureRequest(BaseModel):
    status: str = "unlabeled"
    tag: str = ""
    notes: str = ""


class AnnotationUpdate(BaseModel):
    status: str | None = None
    tag: str | None = None
    notes: str | None = None
    boxes: list[dict] | None = None


class DatasetCreate(BaseModel):
    name: str
    description: str = ""
    capture: bool = False          # also make it the dataset new captures go into


class DatasetUpdate(BaseModel):
    name: str | None = None
    description: str | None = None


class DatasetMembership(BaseModel):
    add: list[str] = []
    remove: list[str] = []


class CaptureTarget(BaseModel):
    name: str | None = None        # None = new captures join no dataset


class TrainJobRequest(BaseModel):
    epochs: int
    imgsz: int
    batch: int
    base_model: str
    device: str = "auto"
    trainer: str | None = Field(None, description="Trainer name; default: train.trainer from boxqc.yaml.")
    name: str = ""
    dataset: str = ""              # named dataset to train on; "" = every labeled image
    deploy: bool = True            # upload the finished model to the device
    activate: bool = True          # ...and make it the active model there
    options: dict = Field(default_factory=dict, description="Trainer-specific settings, passed through.")


class TrainStop(BaseModel):
    mode: str = "early"


class DeployRequest(BaseModel):
    activate: bool = True


class SameOriginMiddleware:
    """Refuse requests that change something when a browser sent them for another site.

    The studio has no login, and it forwards ``/api/device/...`` with the device's API
    token attached. Without this check, any web page open in the same browser could post
    a form to ``http://localhost:8080/api/device/models`` and so upload a model to the
    device, or trigger inspections. A browser names the page that caused such a request
    in the ``Origin`` header; it must be the studio itself (the request's own ``Host``,
    or ``X-Forwarded-Host`` behind a reverse proxy). Requests without the header (curl,
    scripts, a PLC) pass, and so do reads, which change nothing.

    Written as plain ASGI middleware so that streamed responses (the live camera view)
    pass through untouched.
    """

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http" and scope["method"] not in SAFE_METHODS:
            headers = Headers(scope=scope)
            origin = headers.get("origin")
            if origin is not None:
                ours = {headers.get("host"), headers.get("x-forwarded-host")} - {None}
                if urlsplit(origin).netloc not in ours:
                    response = JSONResponse({"detail": "Cross-site request refused"}, status_code=403)
                    await response(scope, receive, send)
                    return
        await self.app(scope, receive, send)


class DeviceLink:
    """Remembers for a moment whether the device answered, so that a page making several
    requests (or a device that is switched off) does not cost one timeout per request."""

    OK_TTL = 2.0        # seconds a successful status is reused
    FAIL_TTL = 5.0      # seconds before an unreachable device is tried again

    def __init__(self, client: DeviceClient, state: StudioState, clock=time.monotonic):
        self.client = client
        self._state = state
        self._clock = clock
        self._lock = threading.Lock()
        self._at: float | None = None
        self._status: dict | None = None
        self._error: str | None = None

    def status(self, fresh: bool = False) -> tuple[dict | None, str | None]:
        """(device status, None) if it is reachable, else (None, why not)."""
        with self._lock:
            ttl = self.OK_TTL if self._status is not None else self.FAIL_TTL
            if fresh or self._at is None or self._clock() - self._at > ttl:
                try:
                    self._status, self._error = self.client.status(), None
                except DeviceError as exc:
                    self._status, self._error = None, str(exc)
                self._at = self._clock()
                self._remember_roi()
            return self._status, self._error

    def invalidate(self) -> None:
        """Forget the remembered status: something on the device was just changed."""
        with self._lock:
            self._at = None

    def _remember_roi(self) -> None:
        roi = ((self._status or {}).get("settings") or {}).get("roi")
        if roi is not None:
            with contextlib.suppress(ValueError, TypeError):
                self._state.update(roi=roi)


def _device_http_error(exc: DeviceError) -> HTTPException:
    """Translate a failed device call for the browser: 502 if unreachable, else the device's own status."""
    return HTTPException(502 if exc.unreachable else exc.status_code, str(exc))


def create_app(cfg: Config, device: DeviceClient | None = None,
               proxy: httpx.AsyncClient | None = None) -> FastAPI:
    """Build the studio application.

    ``device`` and ``proxy`` are the blocking and the async HTTP clients for the device;
    by default both are created from ``cfg.device``. Tests pass clients wired to an
    in-process device app.
    """
    auth = {"Authorization": f"Bearer {cfg.device.api_token}"} if cfg.device.api_token else {}
    device = device or DeviceClient.from_config(cfg)
    proxy_client = proxy or httpx.AsyncClient(base_url=cfg.device.url, timeout=httpx.Timeout(120.0, connect=5.0))
    store = DatasetStore(cfg.data_dir)
    state = StudioState(cfg.data_dir)
    link = DeviceLink(device, state)
    training = TrainingManager(cfg, device=device, state=state)
    assistant = LabelAssistant(cfg)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        adopted = store.adopt_unregistered_datasets()
        if adopted:
            log.info("Registered datasets found in existing annotations: %s", ", ".join(adopted))
        yield
        await proxy_client.aclose()

    app = FastAPI(
        title="BoxQC studio",
        version=__version__,
        description="Dataset, labeling and training. Requests under `/api/device/` are "
                    "forwarded to the device API (see the device's own `/docs`).",
        lifespan=lifespan,
    )
    app.add_middleware(SameOriginMiddleware)
    app.state.dataset = store
    app.state.studio = state
    app.state.training = training
    app.state.link = link
    app.mount("/static", StaticFiles(directory=HERE / "static"), name="static")
    templates = Jinja2Templates(directory=HERE / "templates")

    def static_url(name: str) -> str:
        """/static/<name>?v=<mtime>: a changed file gets a new URL, so browsers never run a
        stale cached copy after an update (e.g. a page script newer than common.js)."""
        try:
            version = int((HERE / "static" / name).stat().st_mtime)
        except OSError:
            version = 0
        return f"/static/{name}?v={version}"

    templates.env.globals["static_url"] = static_url

    def page(request: Request, name: str, **ctx):
        ctx.update(active=name, classes=cfg.classes, version=__version__, device_url=cfg.device.url)
        return templates.TemplateResponse(request, f"{name}.html", ctx)

    # ---- pages ----------------------------------------------------------
    @app.get("/", include_in_schema=False)
    def index():
        return RedirectResponse("/inspect")

    @app.get("/favicon.ico", include_in_schema=False)
    def favicon():
        return FileResponse(HERE / "static" / "favicon.ico", media_type="image/x-icon")

    @app.get("/capture", include_in_schema=False)
    def capture_page(request: Request):
        return page(request, "capture")

    @app.get("/dataset", include_in_schema=False)
    def dataset_page(request: Request):
        return page(request, "dataset")

    @app.get("/annotate/{item_id}", include_in_schema=False)
    def annotate_page(request: Request, item_id: str):
        return page(request, "annotate", item_id=item_id)

    @app.get("/inspect", include_in_schema=False)
    def inspect_page(request: Request):
        return page(request, "inspect")

    @app.get("/system", include_in_schema=False)
    def system_page(request: Request):
        return page(request, "system")

    @app.get("/training", include_in_schema=False)
    def training_page(request: Request):
        return page(request, "training")

    # ---- status / settings ----------------------------------------------
    def _capture_sets() -> list[str]:
        name = state.get("capture_dataset")
        return [name] if name and name in store.dataset_names() else []

    @app.get("/api/status", tags=["status"], summary="The studio, and the device as seen from it")
    def status():
        """`device` is the device's own status, or null with `device_error` saying why it is unreachable."""
        device_status, device_error = link.status()
        disk = shutil.disk_usage(cfg.data_dir)
        capture = _capture_sets()
        return {
            "version": __version__,
            "hostname": socket.gethostname(),
            "classes": cfg.classes,
            "capture_dataset": capture[0] if capture else None,
            "dataset": store.stats(cfg.classes),
            "training": training.presence(),
            "disk_free_gb": round(disk.free / 1e9, 1),
            "config_file": str(cfg.source) if cfg.source else None,
            "device_url": cfg.device.url,
            "device": device_status,
            "device_error": device_error,
        }

    @app.get("/api/settings", tags=["status"], summary="What the labeling pages need")
    def get_settings():
        """The ROI is the device's as last seen (it is refreshed whenever the device answers)."""
        roi = state.get("roi")
        capture = _capture_sets()
        return {"roi": roi if roi is not None else list(cfg.roi), "classes": cfg.classes,
                "capture_dataset": capture[0] if capture else None}

    # ---- device proxy ---------------------------------------------------
    @app.api_route("/api/device/{path:path}", methods=PROXY_METHODS, include_in_schema=False)
    async def device_proxy(path: str, request: Request):
        """Forward to the device API. Streams, so the live camera view passes through."""
        headers = dict(auth)
        if content_type := request.headers.get("content-type"):
            headers["content-type"] = content_type
        upstream = proxy_client.build_request(
            request.method, f"{API_PREFIX}/{path}", params=request.query_params.multi_items(),
            content=await request.body() or None, headers=headers)
        try:
            response = await proxy_client.send(upstream, stream=True)
        except httpx.HTTPError as exc:
            raise HTTPException(502, f"Cannot reach the device at {cfg.device.url} "
                                     f"({type(exc).__name__})") from exc
        if request.method not in SAFE_METHODS:
            link.invalidate()       # e.g. a new active model or threshold: the next status must show it

        async def body():
            try:
                async for chunk in response.aiter_raw():
                    yield chunk
            finally:    # also runs when the browser goes away mid-stream
                with anyio.CancelScope(shield=True):
                    await response.aclose()

        return StreamingResponse(
            body(), status_code=response.status_code,
            headers={k: response.headers[k] for k in PROXY_RESPONSE_HEADERS if k in response.headers})

    # ---- images ---------------------------------------------------------
    @app.post("/api/capture", tags=["dataset"], summary="Take a photo with the device and add it to the dataset")
    def capture(body: CaptureRequest):
        if body.status not in STATUSES:
            raise HTTPException(400, "bad status")
        try:
            jpeg = device.capture_jpeg()
        except DeviceError as exc:
            raise _device_http_error(exc) from exc
        ann = store.add_image(jpeg, source="capture", status=body.status, tag=body.tag,
                                notes=body.notes, datasets=_capture_sets())
        return ann.to_dict()

    @app.post("/api/images/upload", tags=["dataset"], summary="Add image files to the dataset")
    def upload_images(files: list[UploadFile] = File(...), status: str = Form("unlabeled"),
                      dataset: str = Form("")):
        """Uploads join `dataset` when given (the one being viewed), else the capture dataset."""
        if status not in STATUSES:
            raise HTTPException(400, "bad status")
        if dataset and dataset not in store.dataset_names():
            raise HTTPException(404, f"no dataset called {dataset!r}")
        sets = [dataset] if dataset else _capture_sets()
        added = []
        for f in files:
            try:
                added.append(store.add_image(f.file.read(), source="upload", status=status, datasets=sets).id)
            except ValueError:
                continue    # not an image
        return {"added": added}

    @app.get("/api/images", tags=["dataset"], summary="List images, newest first")
    def list_images(status: str | None = None, limit: int = 0, dataset: str | None = None):
        """`dataset`: a dataset name, "__none__" for images in no dataset, or omit for all."""
        items = [a.to_dict() for a in store.list(status, dataset=dataset or None)]
        return items[:limit] if limit else items

    def _get_ann(item_id: str):
        try:
            return store.get(item_id)
        except (KeyError, ValueError) as exc:
            raise HTTPException(404, "image not found") from exc

    @app.get("/api/images/{item_id}", tags=["dataset"], summary="One image's annotation, with its neighbours")
    def get_image(item_id: str, dataset: str | None = None):
        ann = _get_ann(item_id)
        return {**ann.to_dict(), "nav": store.neighbours(item_id, dataset=dataset or None)}

    @app.get("/api/images/{item_id}/file", tags=["dataset"], summary="The image (JPEG)")
    def image_file(item_id: str):
        _get_ann(item_id)
        return FileResponse(store.image_path(item_id), media_type="image/jpeg")

    @app.get("/api/images/{item_id}/thumb", tags=["dataset"], summary="A thumbnail (JPEG)")
    def image_thumb(item_id: str):
        _get_ann(item_id)
        return FileResponse(store.thumbnail(item_id), media_type="image/jpeg")

    @app.put("/api/images/{item_id}", tags=["dataset"], summary="Save boxes, status, tag or notes")
    def update_image(item_id: str, body: AnnotationUpdate):
        ann = _get_ann(item_id)
        if body.status is not None:
            if body.status not in STATUSES:
                raise HTTPException(400, "bad status")
            ann.status = body.status
        if body.tag is not None:
            ann.tag = body.tag
        if body.notes is not None:
            ann.notes = body.notes
        if body.boxes is not None:
            try:
                boxes = [Box.from_dict(b) for b in body.boxes]
            except (KeyError, TypeError, ValueError) as exc:
                raise HTTPException(400, f"bad box: {exc}") from exc
            # A class that boxqc.yaml no longer lists is accepted on boxes the image already
            # has (so the image can still be skipped, annotated or relabeled), never on new ones.
            unknown = {b.cls for b in boxes} - set(cfg.classes) - {b.cls for b in ann.boxes}
            if unknown:
                raise HTTPException(400, f"unknown classes: {sorted(unknown)}")
            ann.boxes = [b for b in boxes if b.x1 - b.x0 > 1e-3 and b.y1 - b.y0 > 1e-3]
        return store.save(ann).to_dict()

    @app.delete("/api/images/{item_id}", tags=["dataset"], summary="Delete an image and its annotation")
    def delete_image(item_id: str):
        _get_ann(item_id)
        store.delete(item_id)
        return {"deleted": item_id}

    # ---- labeling assistant -----------------------------------------------
    @app.get("/api/assistant", tags=["labeling"], summary="The models the labeling assistant can use")
    def assistant_overview():
        """Models on this computer, newest first. `default` is the one used when none is
        named, or null while there is no model at all."""
        available = assistant.models()
        return {"models": available, "default": available[0]["name"] if available else None}

    @app.get("/api/images/{item_id}/suggestions", tags=["labeling"],
             summary="Boxes a model proposes for this image")
    def image_suggestions(item_id: str, model: str | None = None):
        """Runs `model` (default: the newest on this computer) over the image, cropped to
        the ROI as for an inspection. Nothing is stored: the labeling page saves the
        suggestions the user accepts as ordinary boxes."""
        _get_ann(item_id)
        try:
            return assistant.suggest(store.image_path(item_id), get_settings()["roi"], model or None)
        except KeyError as exc:
            raise HTTPException(404, f"no model called {model!r} on this computer") from exc
        except ValueError as exc:
            raise HTTPException(404, str(exc)) from exc
        except AssistantUnavailable as exc:
            raise HTTPException(409, str(exc)) from exc

    @app.get("/api/dataset/manifest", tags=["dataset"], summary="Classes, datasets and every annotation")
    def manifest():
        return {
            "classes": cfg.classes,
            "roi": get_settings()["roi"],
            "datasets": store.datasets(),
            "items": [a.to_dict() for a in store.list()],
        }

    @app.get("/api/dataset/export.zip", tags=["dataset"], summary="All images and annotations as a zip")
    def export_zip():
        tmp = tempfile.NamedTemporaryFile(suffix=".zip", delete=False)
        with zipfile.ZipFile(tmp, "w", zipfile.ZIP_STORED) as zf:
            for sub in ("images", "annotations"):
                for p in sorted((cfg.data_dir / sub).glob("*")):
                    zf.write(p, f"{sub}/{p.name}")
            for name in ("datasets.json", "studio.json"):
                if (cfg.data_dir / name).is_file():
                    zf.write(cfg.data_dir / name, name)
        tmp.close()
        return FileResponse(tmp.name, media_type="application/zip",
                            filename=f"boxqc-dataset-{time.strftime('%Y%m%d-%H%M')}.zip",
                            background=BackgroundTask(Path(tmp.name).unlink, missing_ok=True))

    # ---- named datasets ---------------------------------------------------
    def _dataset_or_404(name: str) -> str:
        if name not in store.dataset_names():
            raise HTTPException(404, f"no dataset called {name!r}")
        return name

    @app.get("/api/datasets", tags=["datasets"], summary="Every dataset with its statistics")
    def datasets():
        capture = _capture_sets()
        return {**store.dataset_summaries(cfg.classes), "capture_dataset": capture[0] if capture else None}

    @app.post("/api/datasets", tags=["datasets"], summary="Create a dataset")
    def create_dataset(body: DatasetCreate):
        try:
            entry = store.create_dataset(body.name, body.description)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        if body.capture:
            state.update(capture_dataset=entry["name"])
        return entry

    @app.put("/api/datasets/capture", tags=["datasets"], summary="Choose the dataset new captures join")
    def set_capture_dataset(body: CaptureTarget):
        if body.name:
            _dataset_or_404(body.name)
        state.update(capture_dataset=body.name or None)
        return {"capture_dataset": body.name or None}

    @app.patch("/api/datasets/{name}", tags=["datasets"], summary="Rename or re-describe a dataset")
    def update_dataset(name: str, body: DatasetUpdate):
        _dataset_or_404(name)
        try:
            entry = store.update_dataset(name, new_name=body.name, description=body.description)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        if state.get("capture_dataset") == name and entry["name"] != name:
            state.update(capture_dataset=entry["name"])
        return entry

    @app.delete("/api/datasets/{name}", tags=["datasets"], summary="Delete a dataset (its images are kept)")
    def delete_dataset(name: str):
        _dataset_or_404(name)
        released = store.delete_dataset(name)
        if state.get("capture_dataset") == name:
            state.update(capture_dataset=None)
        return {"deleted": name, "images_released": released}

    @app.post("/api/datasets/{name}/items", tags=["datasets"], summary="Add or remove images")
    def dataset_items(name: str, body: DatasetMembership):
        _dataset_or_404(name)
        try:
            changed = store.set_membership(name, add=body.add, remove=body.remove)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        return {"changed": changed}

    # ---- inspections -> dataset -------------------------------------------
    @app.post("/api/inspections/{item_id}/to-dataset", tags=["dataset"],
              summary="Copy a device inspection into the dataset for labeling")
    def inspection_to_dataset(item_id: str, min_conf: float | None = None):
        """The image arrives unlabeled, pre-filled with the model's boxes at or above
        `min_conf` (default: the threshold used for that inspection) to correct."""
        try:
            record = device.inspection(item_id)
            jpeg = device.inspection_image(item_id)
        except ValueError as exc:
            raise HTTPException(404, "inspection not found") from exc
        except DeviceError as exc:
            raise _device_http_error(exc) from exc
        w, h = record["image_size"]
        thr = record["threshold"] if min_conf is None else min_conf
        boxes = [
            Box(d["name"], d["box"][0] / w, d["box"][1] / h, d["box"][2] / w, d["box"][3] / h)
            for d in record["detections"] if d["conf"] >= thr and d["name"] in cfg.classes
        ]
        ann = store.add_image(jpeg, source="inspection", status="unlabeled",
                                notes=f"from inspection {item_id} ({record['verdict']})",
                                boxes=boxes, datasets=_capture_sets())
        return ann.to_dict()

    # ---- models -----------------------------------------------------------
    def _local_model(name: str) -> Path:
        path = cfg.models_dir / Path(name).name
        if not is_model_file(path) or not path.is_file():
            raise HTTPException(404, "model not found on this computer")
        return path

    @app.get("/api/models", tags=["models"], summary="Models on this computer and on the device, merged by name")
    def models():
        """Each model says where it is (`local`, `on_device`) and whether it is the device's `active` one.
        `device_copy_differs` is true when both places have a file of that name but with
        different content: the device's is then another (usually older) model."""
        local = {m["name"]: m for m in list_models(cfg.models_dir)}
        remote: dict[str, dict] = {}
        device_status, device_error = link.status()
        if device_status is not None:
            try:
                remote = {m["name"]: m for m in device.models(usage=True)}
            except DeviceError as exc:
                device_error = str(exc)
        merged = []
        for name in {**remote, **local}:
            here, there = local.get(name), remote.get(name)
            first = here or there
            merged.append({
                "name": name,
                "size_mb": first["size_mb"],
                "modified": first["modified"],
                "info": (here or {}).get("info") or (there or {}).get("info") or {},
                "local": here is not None,
                "on_device": there is not None,
                "active": bool(there and there.get("active")),
                "device_copy_differs": bool(here and there and here.get("sha256") and there.get("sha256")
                                            and here["sha256"] != there["sha256"]),
                "usage": there.get("usage") if there else None,
            })
        merged.sort(key=lambda m: m["info"].get("trained_at") or m["modified"], reverse=True)
        return {"models": merged, "device_online": device_status is not None, "device_error": device_error}

    @app.post("/api/models/{name}/deploy", tags=["models"], summary="Upload a local model to the device")
    def deploy_model(name: str, body: DeployRequest):
        path = _local_model(name)
        try:
            return device.deploy_model(path, activate=body.activate)
        except DeviceError as exc:
            raise _device_http_error(exc) from exc
        finally:
            link.invalidate()

    @app.delete("/api/models/{name}", tags=["models"], summary="Delete the local copy of a model")
    def delete_model(name: str):
        path = _local_model(name)
        path.unlink()
        metadata_path(path).unlink(missing_ok=True)
        return {"deleted": path.name}

    # ---- training ---------------------------------------------------------
    @app.get("/api/training", tags=["training"], summary="Trainers, the running job and recent runs")
    def training_overview():
        t = cfg.train
        return {**training.overview(),
                "defaults": {"epochs": t.epochs, "imgsz": t.imgsz, "batch": t.batch,
                             "base_model": t.base_model, "device": t.device, "trainer": t.trainer},
                "classes": cfg.classes}

    @app.post("/api/training/jobs", tags=["training"], summary="Start a training run")
    def training_start(body: TrainJobRequest):
        if body.dataset:
            if body.dataset not in store.dataset_names():
                raise HTTPException(400, f"no dataset called {body.dataset!r}")
            if not store.list("labeled", dataset=body.dataset):
                raise HTTPException(400, f"dataset {body.dataset!r} has no labeled images yet")
        request = TrainRequest(
            epochs=body.epochs, imgsz=body.imgsz, batch=body.batch, base_model=body.base_model,
            device=body.device, trainer=body.trainer or cfg.train.trainer, dataset=body.dataset or None,
            name=body.name or None, options=body.options)
        try:
            return training.start(request, deploy=body.deploy, activate=body.activate)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        except (TrainerUnavailable, RuntimeError) as exc:
            raise HTTPException(409, str(exc)) from exc

    @app.get("/api/training/jobs/{job_id}", tags=["training"], summary="One run: progress, log and result")
    def training_job(job_id: str):
        try:
            return training.get(job_id)
        except KeyError as exc:
            raise HTTPException(404, "run not found") from exc

    @app.post("/api/training/jobs/{job_id}/stop", tags=["training"], summary="Stop early or cancel a run")
    def training_stop(job_id: str, body: TrainStop):
        try:
            return training.request_stop(job_id, body.mode)
        except KeyError as exc:
            raise HTTPException(404, "run not found") from exc
        except (ValueError, RuntimeError) as exc:
            raise HTTPException(409, str(exc)) from exc

    return app
