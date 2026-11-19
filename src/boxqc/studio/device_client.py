"""A Python client for the device API (``docs/device-api.md``).

Used by the studio and the command line; equally usable from your own scripts::

    from boxqc.studio.device_client import DeviceClient

    device = DeviceClient("http://10.42.0.2:8000")
    result = device.inspect()
    print(result["verdict"])
"""

from __future__ import annotations

from pathlib import Path

import httpx

from ..config import Config
from ..ids import check_id
from ..models import file_sha256, metadata_path

API_PREFIX = "/api/v1"
CONNECT_TIMEOUT = 5.0   # seconds to wait for a device that is switched off or unplugged


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

    ``http`` lets a caller supply its own ``httpx.Client`` (tests use this to talk to an
    in-process device app); otherwise one is created for ``base_url``.
    """

    def __init__(self, base_url: str, token: str | None = None, timeout: float = 10.0,
                 http: httpx.Client | None = None):
        self.base_url = base_url.rstrip("/")
        self._headers = {"Authorization": f"Bearer {token}"} if token else {}
        self.http = http or httpx.Client(base_url=self.base_url,
                                         timeout=httpx.Timeout(timeout, connect=CONNECT_TIMEOUT))

    @classmethod
    def from_config(cls, cfg: Config, url: str | None = None) -> DeviceClient:
        return cls(url or cfg.device.url, cfg.device.api_token)

    def _request(self, method: str, path: str, **kwargs) -> httpx.Response:
        if "timeout" in kwargs:     # a long allowance is for the answer, never for connecting
            kwargs["timeout"] = httpx.Timeout(kwargs["timeout"], connect=min(kwargs["timeout"], CONNECT_TIMEOUT))
        try:
            response = self.http.request(method, API_PREFIX + path, headers=self._headers, **kwargs)
        except httpx.HTTPError as exc:
            raise DeviceError(f"Cannot reach the device at {self.base_url} ({type(exc).__name__}: {exc})") from exc
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
