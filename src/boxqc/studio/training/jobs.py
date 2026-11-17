"""Background training runs for the studio's Training page.

One run at a time executes :func:`~.pipeline.run_training` in a worker thread; the web
UI polls :meth:`TrainingManager.overview` for progress. Run history is kept in memory
and is gone when the studio restarts - the models and their metadata are what persist.
"""

from __future__ import annotations

import copy
import platform
import socket
import threading
import traceback
import uuid
from collections import deque
from dataclasses import asdict
from datetime import datetime

from ... import __version__
from ...config import Config
from ...ids import now_iso
from ...models import read_metadata
from ..device_client import DeviceClient, DeviceError
from ..state import StudioState
from . import TRAINERS, get_trainer
from .base import STOP_MODES, TrainContext, TrainerUnavailable, TrainingCancelled, TrainRequest
from .pipeline import run_training

LOG_LINES = 500             # log lines kept per run
KEEP_JOBS = 20              # finished runs kept for the history list
ACTIVE = ("queued", "preparing", "training", "exporting", "evaluating", "deploying")
FINISHED = ("done", "failed", "cancelled")


class _JobContext(TrainContext):
    """Routes a trainer's progress reports into the job record the UI reads."""

    def __init__(self, manager: TrainingManager, job: dict):
        self._manager = manager
        self._job = job

    def log(self, line: str, *, echo: bool = True) -> None:
        if echo:
            print(line, flush=True)
        with self._manager._lock:
            self._job["log"].extend(part[:1000] for part in str(line).splitlines() if part.strip())

    def stage(self, name: str) -> None:
        with self._manager._lock:
            if name in ACTIVE and self._job["state"] in ACTIVE:
                self._job["state"] = name

    def epoch(self, metrics: dict, total_epochs: int | None = None) -> None:
        entry = {k: v for k, v in metrics.items() if isinstance(v, (int, float)) and not isinstance(v, bool)}
        with self._manager._lock:
            self._job["history"].append(entry)
            self._job["epoch"] = int(entry.get("epoch", len(self._job["history"])))
            if total_epochs:
                self._job["epochs"] = int(total_epochs)

    @property
    def stop_requested(self) -> str | None:
        return self._job["stop_request"]


class TrainingManager:
    """Starts, tracks and stops training runs.

    ``device`` (optional) is where finished models are deployed and where the current
    ROI is read from; ``state`` remembers that ROI for when the device is offline.
    ``runner`` is the function that does the work, replaceable in tests.
    """

    def __init__(self, cfg: Config, device: DeviceClient | None = None, state: StudioState | None = None,
                 runner=run_training):
        self.cfg = cfg
        self.device = device
        self.state = state or StudioState(cfg.data_dir)
        self._runner = runner
        self._lock = threading.Lock()
        self._jobs: list[dict] = []      # oldest first
        self._worker: threading.Thread | None = None
        self._host: dict | None = None
        self._host_lock = threading.Lock()

    # ---- what the UI reads ------------------------------------------------
    def host(self) -> dict:
        """This machine, as shown on the Training page (computed once; importing PyTorch is slow)."""
        with self._host_lock:
            if self._host is None:
                info = {
                    "hostname": socket.gethostname(),
                    "platform": f"{platform.system()} {platform.release()} ({platform.machine()})",
                    "python": platform.python_version(),
                    "boxqc": __version__,
                }
                try:
                    trainer = get_trainer(self.cfg.train.trainer)
                    if trainer.availability() is None:
                        info.update(trainer.host_info())
                except KeyError as exc:
                    info["error"] = exc.args[0]
                self._host = info
            return dict(self._host)

    def trainers(self) -> list[dict]:
        return [cls.describe() for _, cls in TRAINERS.items()]

    def presence(self) -> dict:
        with self._lock:
            return {"busy": self._active() is not None}

    def overview(self) -> dict:
        trainers = self.trainers()
        host = self.host()
        with self._lock:
            active = self._active()
            return {
                "host": host,
                "trainers": trainers,
                "default_trainer": self.cfg.train.trainer,
                "active": self._public(active) if active else None,
                "jobs": [self._summary(j) for j in reversed(self._jobs)],
            }

    def get(self, job_id: str) -> dict:
        with self._lock:
            return self._public(self._find(job_id))

    # ---- control ------------------------------------------------------------
    def start(self, request: TrainRequest, *, deploy: bool = True, activate: bool = True) -> dict:
        """Validate ``request`` and start a run in the background.

        Raises ``ValueError`` for a bad request, :class:`TrainerUnavailable` if the
        trainer cannot run here, and ``RuntimeError`` if a run is already in progress.
        """
        request.validate()
        try:
            trainer_cls = get_trainer(request.trainer)
        except KeyError as exc:
            raise ValueError(exc.args[0]) from None
        trainer_cls(self.cfg).validate(request)
        if reason := trainer_cls.availability():
            raise TrainerUnavailable(reason)
        with self._lock:
            if self._active():
                raise RuntimeError("A training run is already in progress")
            job = {
                "id": datetime.now().strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:4],
                "params": {**asdict(request), "dataset": request.dataset or "", "name": request.name or "",
                           "deploy": deploy, "activate": activate},
                "state": "queued",
                "stop_request": None,
                "created": now_iso(),
                "started": None,
                "finished": None,
                "epoch": 0,
                "epochs": request.epochs,
                "history": [],
                "log": deque(maxlen=LOG_LINES),
                "result": None,
                "error": None,
                "host": socket.gethostname(),
            }
            self._jobs.append(job)
            finished = [j for j in self._jobs if j["state"] in FINISHED]
            for old in finished[:-KEEP_JOBS]:
                self._jobs.remove(old)
            self._worker = threading.Thread(target=self._run, args=(job, request, deploy, activate),
                                            name=f"train-{job['id']}", daemon=True)
            self._worker.start()
            return self._public(job)

    def request_stop(self, job_id: str, mode: str) -> dict:
        """Ask a run to stop: ``early`` keeps the best model so far, ``cancel`` discards the run."""
        if mode not in STOP_MODES:
            raise ValueError(f"mode must be one of {STOP_MODES}")
        with self._lock:
            job = self._find(job_id)
            if job["state"] not in ACTIVE:
                raise RuntimeError(f"Run is already {job['state']}")
            job["stop_request"] = mode
            return self._public(job)

    def wait(self, timeout: float | None = None) -> None:
        """Block until the current run (if any) has finished."""
        worker = self._worker
        if worker is not None:
            worker.join(timeout)

    # ---- the worker -----------------------------------------------------------
    def _run(self, job: dict, request: TrainRequest, deploy: bool, activate: bool) -> None:
        ctx = _JobContext(self, job)
        with self._lock:
            job["started"] = now_iso()
        print(f"\n=== Training run {job['id']} started: {job['params']}", flush=True)
        try:
            ctx.check_cancelled()
            ctx.stage("preparing")
            model_path = self._runner(self.cfg, request, ctx, roi=self._current_roi(ctx))
            info = read_metadata(model_path)
            result = {
                "model": model_path.name,
                "deployed": False,
                "active": False,
                "deploy_error": None,
                "metrics": info.get("metrics"),
                "recommended_threshold": info.get("recommended_threshold"),
                "dataset": info.get("dataset"),
                "dataset_name": info.get("dataset_name"),
                "evaluation": info.get("evaluation"),
                "run_dir": info.get("run_dir"),
            }
            if deploy and self.device is not None:
                ctx.check_cancelled()
                ctx.stage("deploying")
                try:
                    pushed = self.device.deploy_model(model_path, activate=activate)
                    result.update(deployed=True, active=bool(pushed.get("active")), device_model=pushed.get("name"))
                    ctx.log(f"Deployed to the device as {pushed.get('name')} (active={pushed.get('active')}).")
                except DeviceError as exc:
                    result["deploy_error"] = str(exc)
                    ctx.log(f"Not deployed: {exc}")
                    ctx.log("The model is saved on this computer; deploy it from the Models panel "
                            "once the device is reachable.")
            with self._lock:
                job["result"] = result
                self._finish(job, "done")
        except TrainingCancelled:
            ctx.log("Run cancelled.")
            with self._lock:
                self._finish(job, "cancelled")
        except BaseException as exc:    # a library calling sys.exit() must not kill the report
            trace = traceback.format_exc()
            print(trace, flush=True)
            ctx.log("\n".join(trace.splitlines()[-15:]), echo=False)
            with self._lock:
                job["error"] = f"{type(exc).__name__}: {exc}"
                self._finish(job, "failed")
            if not isinstance(exc, Exception):
                raise
        finally:
            print(f"=== Training run {job['id']} finished: {job['state']}", flush=True)

    def _current_roi(self, ctx: TrainContext) -> list[float]:
        """The device's ROI, fetched now if possible, else as last seen (or from boxqc.yaml)."""
        if self.device is not None:
            try:
                self.state.update(roi=self.device.settings()["roi"])
            except (DeviceError, KeyError, ValueError) as exc:
                ctx.log(f"Could not read the ROI from the device ({exc}); using the last known one.")
        roi = self.state.get("roi")
        return roi if roi is not None else list(self.cfg.roi)

    # ---- helpers (call with the lock held) ---------------------------------
    def _active(self) -> dict | None:
        return next((j for j in reversed(self._jobs) if j["state"] in ACTIVE), None)

    def _find(self, job_id: str) -> dict:
        for job in self._jobs:
            if job["id"] == job_id:
                return job
        raise KeyError(job_id)

    @staticmethod
    def _finish(job: dict, state: str) -> None:
        job["state"] = state
        job["finished"] = now_iso()
        job["stop_request"] = None

    @staticmethod
    def _public(job: dict) -> dict:
        out = copy.deepcopy({k: v for k, v in job.items() if k != "log"})
        out["log"] = list(job["log"])
        return out

    @staticmethod
    def _summary(job: dict) -> dict:
        result = job["result"] or {}
        return {
            "id": job["id"],
            "state": job["state"],
            "created": job["created"],
            "finished": job["finished"],
            "epoch": job["epoch"],
            "epochs": job["epochs"],
            "params": dict(job["params"]),
            "model": result.get("model"),
            "mAP50": (result.get("metrics") or {}).get("mAP50"),
            "error": job["error"],
        }
