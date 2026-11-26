"""A Python client for the device API (``docs/device-api.md``).

Used by the studio and the command line; equally usable from your own scripts::

    from boxqc.studio.device_client import DeviceClient

    device = DeviceClient("http://10.42.0.2:8000")
    result = device.inspect()
    print(result["verdict"])

``DeviceClient("auto")`` finds the device on the local network instead
(:mod:`boxqc.discovery`), on first use and again whenever it stops answering.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from pathlib import Path

import httpx

from ..config import AUTO_URL, Config, normalize_device_url
from ..discovery import DiscoveryError, FoundDevice, choose, discover
from ..ids import check_id
from ..models import file_sha256, metadata_path

API_PREFIX = "/api/v1"
CONNECT_TIMEOUT = 5.0   # seconds to wait for a device that is switched off or unplugged
RELOCATE_INTERVAL = 10.0    # auto mode: seconds between searches while no device answers


class DeviceError(RuntimeError):
    """The device could not be reached (``status_code`` is None) or refused the request."""

    def __init__(self, message: str, status_code: int | None = None):
        super().__init__(message)
        self.status_code = status_code

    @property
    def unreachable(self) -> bool:
        return self.status_code is None


class DeviceClient:
    """Blocking client for one device.

    ``base_url`` is the device's address, or ``"auto"`` to find it on the local network:
    on first use, and again after a request fails to connect (the device may have a new
    address). In that mode ``name`` picks one device when several answer, and ``finder``
    replaces the network search (tests use this).

    ``http`` lets a caller supply its own ``httpx.Client`` (tests use this to talk to an
    in-process device app); otherwise one is created for ``base_url``.
    """

    def __init__(self, base_url: str, token: str | None = None, timeout: float = 10.0,
                 http: httpx.Client | None = None, name: str | None = None,
                 finder: Callable[[], list[FoundDevice]] | None = None, clock=time.monotonic):
        base_url = normalize_device_url(base_url)
        self.auto = base_url == AUTO_URL
        self.name = name
        self._url: str | None = None if self.auto else base_url
        self._finder = finder or (lambda: discover(token=token))
        self._clock = clock
        self._lock = threading.Lock()
        self._searched_at: float | None = None
        self._search_error: str | None = None
        self.found: FoundDevice | None = None       # what discovery last chose (auto mode)
        self._headers = {"Authorization": f"Bearer {token}"} if token else {}
        # In auto mode the base URL is replaced once the device is found.
        self.http = http or httpx.Client(base_url=self._url or "http://device.invalid",
                                         timeout=httpx.Timeout(timeout, connect=CONNECT_TIMEOUT))

    @classmethod
    def from_config(cls, cfg: Config, url: str | None = None) -> DeviceClient:
        """A client for ``url`` if given, else for ``device.url`` (which may be ``auto``)."""
        return cls(url or cfg.device.url, cfg.device.api_token, name=cfg.device.name)

    @property
    def url(self) -> str | None:
        """The device's current address; None while ``auto`` has not found it."""
        return self._url

    @property
    def base_url(self) -> str:
        """The current address, or ``auto`` while there is none (for messages)."""
        return self._url or AUTO_URL

    def locate(self, force: bool = False) -> str:
        """Return the device's address, searching the network first if needed (auto mode).

        Raises :class:`DeviceError` (unreachable) if no single device answers. A failed
        search is repeated at most every ``RELOCATE_INTERVAL`` seconds unless ``force``;
        in between, the same error is raised at once.
        """
        if self._url is not None:
            return self._url
        with self._lock:
            if self._url is not None:       # found by another thread meanwhile
                return self._url
            if (not force and self._searched_at is not None
                    and self._clock() - self._searched_at < RELOCATE_INTERVAL and self._search_error):
                raise DeviceError(self._search_error)
            self._searched_at = self._clock()
            try:
                found = choose(self._finder(), self.name)
            except DiscoveryError as exc:
                self._search_error = str(exc)
                raise DeviceError(str(exc)) from exc
            except OSError as exc:      # no network at all, for instance
                self._search_error = f"Could not search the network for the device ({exc})"
                raise DeviceError(self._search_error) from exc
            self.found, self._search_error = found, None
            self._url = found.url
            self.http.base_url = found.url
            return found.url

    def connection_failed(self) -> None:
        """A request could not reach the device: in auto mode, search again next time."""
        if self.auto:
            with self._lock:
                self._url = None
                self._searched_at = None    # the next request searches at once

    def _request(self, method: str, path: str, **kwargs) -> httpx.Response:
        if "timeout" in kwargs:     # a long allowance is for the answer, never for connecting
            kwargs["timeout"] = httpx.Timeout(kwargs["timeout"], connect=min(kwargs["timeout"], CONNECT_TIMEOUT))
        url = self.locate()
        try:
            response = self.http.request(method, API_PREFIX + path, headers=self._headers, **kwargs)
        except httpx.HTTPError as exc:
            if isinstance(exc, (httpx.ConnectError, httpx.ConnectTimeout)):
                self.connection_failed()    # not a slow answer: the device is not there (any more)
            raise DeviceError(f"Cannot reach the device at {url} ({type(exc).__name__}: {exc})") from exc
        if response.status_code >= 400:
            try:
                detail = response.json().get("detail", response.text)
            except (ValueError, AttributeError):
                detail = response.text
            raise DeviceError(f"Device answered {response.status_code}: {detail}", response.status_code)
        return response

    # ---- status / settings ----------------------------------------------
    def status(self, timeout: float = 3.0) -> dict:
        return self._request("GET", "/status", timeout=timeout).json()

    def settings(self) -> dict:
        return self._request("GET", "/settings").json()

    def update_settings(self, **changes) -> dict:
        """Change ``conf_threshold``, ``roi`` and/or ``buzzer``."""
        return self._request("PUT", "/settings", json=changes).json()

    # ---- camera / inspection --------------------------------------------
    def capture_jpeg(self) -> bytes:
        """A full-resolution photo from the device's camera, as JPEG bytes."""
        return self._request("POST", "/camera/capture", timeout=30).content

    def inspect(self, buzz: bool = True) -> dict:
        """Run one inspection on the device and return its record."""
        return self._request("POST", "/inspect", params={"buzz": str(buzz).lower()}, timeout=60).json()

    def inspections(self, limit: int = 50) -> list[dict]:
        return self._request("GET", "/inspections", params={"limit": limit}).json()

    def inspection(self, inspection_id: str) -> dict:
        return self._request("GET", f"/inspections/{check_id(inspection_id)}").json()

    def inspection_image(self, inspection_id: str, annotated: bool = False) -> bytes:
        return self._request("GET", f"/inspections/{check_id(inspection_id)}/image.jpg",
                             params={"annotated": str(annotated).lower()}, timeout=30).content

    # ---- models ---------------------------------------------------------
    def models(self, usage: bool = False) -> list[dict]:
        return self._request("GET", "/models", params={"usage": str(usage).lower()}).json()

    def push_model(self, model_path: Path, activate: bool = True) -> dict:
        """Upload a model file and, if present, its metadata sidecar. Returns ``{name, active}``."""
        model_path = Path(model_path)
        files = {"file": (model_path.name, model_path.read_bytes(), "application/octet-stream")}
        meta = metadata_path(model_path)
        if meta.is_file():
            files["meta"] = (meta.name, meta.read_bytes(), "application/json")
        return self._request("POST", "/models", files=files, data={"activate": str(activate).lower()},
                             timeout=300).json()

    def deploy_model(self, model_path: Path, activate: bool = True) -> dict:
        """Make sure the device has this model, uploading it only if it is not there yet.

        "There" means a file of the same name **and** the same content (SHA-256). The name
        alone is not enough: a run name can be reused, and the device may still hold the
        earlier model under it. In that case this one is uploaded, and the device stores
        it under a new name. Returns ``{name, active, already_on_device}``, where ``name``
        is what the device calls the model. (:meth:`push_model` always uploads.)
        """
        model_path = Path(model_path)
        digest = file_sha256(model_path)
        twin = next((m for m in self.models() if m["name"] == model_path.name), None)
        if twin is not None and twin.get("sha256") == digest:
            if activate:
                self.activate_model(model_path.name)
            return {"name": model_path.name, "active": activate or bool(twin.get("active")),
                    "already_on_device": True}
        return {**self.push_model(model_path, activate=activate), "already_on_device": False}

    def activate_model(self, name: str) -> dict:
        return self._request("POST", f"/models/{name}/activate", timeout=60).json()

    def delete_model(self, name: str) -> dict:
        return self._request("DELETE", f"/models/{name}").json()
