"""Dataset storage: one JPEG + one JSON sidecar per image, kept on the studio machine.

    data/images/<id>.jpg
    data/annotations/<id>.json
    data/thumbs/<id>.jpg          (cache, regenerated on demand)
    data/datasets.json            named datasets (name, description, created)

Named datasets are collections of images: each annotation lists the datasets it
belongs to, so an image can be in several (or none), and training can use one
dataset instead of every image ever captured. Removing an image from a dataset
never deletes the image.

Plain files keep the dataset easy to inspect, back up and feed to other tools.
The file formats are specified in ``docs/data-formats.md``.

Annotation status:
  unlabeled - captured, not reviewed yet (may contain model-suggested boxes)
  labeled   - reviewed; zero boxes means "clean" (a valuable negative example)
  skip      - excluded from training (blurry, wrong product, ...)
"""

from __future__ import annotations

import json
import re
import threading
from dataclasses import asdict, dataclass, field
from pathlib import Path

import cv2
import numpy as np

from ..ids import check_id, new_id, now_iso
from ..imaging import decode_image, encode_jpeg, resize_max

STATUSES = ("unlabeled", "labeled", "skip")
_DATASET_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9 ._-]{0,39}$")
UNASSIGNED = "__none__"   # list/stats filter: images that are in no dataset


def check_dataset_name(name: str) -> str:
    """Return the trimmed name if it is acceptable for a dataset, else raise ``ValueError``."""
    name = str(name).strip()
    if not _DATASET_RE.match(name):
        raise ValueError("Dataset names are 1-40 characters: letters, digits, spaces, '.', '_' or '-', "
                         "starting with a letter or digit")
    return name


@dataclass
class Box:
    """One labeled defect. Coordinates are fractions (0..1) of the **full** image, not of
    the ROI, so labels stay valid if the ROI is changed later."""

    cls: str    # class name, one of `classes` in boxqc.yaml
    x0: float
    y0: float
    x1: float
    y1: float

    @classmethod
    def from_dict(cls, d: dict) -> Box:
        """Build from untrusted input: corners are put in order and clipped to the image."""
        x0, x1 = sorted((float(d["x0"]), float(d["x1"])))
        y0, y1 = sorted((float(d["y0"]), float(d["y1"])))
        clip = lambda v: min(1.0, max(0.0, v))
        return cls(str(d["cls"]), clip(x0), clip(y0), clip(x1), clip(y1))


@dataclass
class Annotation:
    """Everything known about one image; stored as ``annotations/<id>.json``."""

    id: str
    width: int
    height: int
    created: str = field(default_factory=now_iso)
    updated: str = field(default_factory=now_iso)
    status: str = "unlabeled"
    source: str = "capture"      # capture | upload | inspection
    tag: str = ""                # operator hint at capture time: clean | defective | ""
    notes: str = ""
    boxes: list[Box] = field(default_factory=list)
    datasets: list[str] = field(default_factory=list)  # named datasets this image belongs to

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> Annotation:
        """Build from a stored file. Unknown keys are ignored, so files written by a
        newer version still load."""
        d = dict(d)
        d["boxes"] = [Box.from_dict(b) for b in d.get("boxes", [])]
        known = cls.__dataclass_fields__
        return cls(**{k: v for k, v in d.items() if k in known})


class DatasetStore:
    """The image dataset on disk, rooted at ``data_dir``.

    All reads and writes of images, annotations and the dataset registry go through
    this class. Annotation files are written atomically (temp file, then rename), and
    the registry is updated under a lock, so the web server's worker threads can use
    one instance concurrently.
    """

    def __init__(self, root: Path):
        self.root = root
        self.images_dir = root / "images"
        self.ann_dir = root / "annotations"
        self.thumbs_dir = root / "thumbs"
        for d in (self.images_dir, self.ann_dir, self.thumbs_dir):
            d.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    # ---- paths -----------------------------------------------------------
    def image_path(self, item_id: str) -> Path:
        return self.images_dir / f"{check_id(item_id)}.jpg"

    def ann_path(self, item_id: str) -> Path:
        return self.ann_dir / f"{check_id(item_id)}.json"

    # ---- CRUD ------------------------------------------------------------
    def add_image(
        self,
        image: np.ndarray | bytes,
        *,
        source: str = "capture",
        status: str = "unlabeled",
        tag: str = "",
        notes: str = "",
        boxes: list[Box] | None = None,
        datasets: list[str] | None = None,
        jpeg_quality: int = 95,
    ) -> Annotation:
        """Store a new image. ``image`` is a BGR array, or encoded image bytes.

        JPEG bytes are stored exactly as given (a photo from the device is not
        re-compressed); other encodings and arrays are saved as JPEG at ``jpeg_quality``.
        Names in ``datasets`` that are not registered datasets are ignored.
        """
        if status not in STATUSES:
            raise ValueError(f"Invalid status {status}")
        if isinstance(image, (bytes, bytearray, memoryview)):
            data = bytes(image)
            decoded = decode_image(data)            # also proves the bytes are an image
            if not data.startswith(b"\xff\xd8"):    # not a JPEG: convert
                data = encode_jpeg(decoded, jpeg_quality)
        else:
            decoded, data = image, encode_jpeg(image, jpeg_quality)
        item_id = self._store_new_image(data)
        h, w = decoded.shape[:2]
        ann = Annotation(id=item_id, width=w, height=h, status=status, source=source,
                         tag=tag, notes=notes, boxes=boxes or [],
                         datasets=[d for d in (datasets or []) if d in self.dataset_names()])
        self.save(ann, touch=False)
        return ann

    def _store_new_image(self, data: bytes) -> str:
        """Write ``data`` as a new image file under a fresh id, and return that id.

        An id is a timestamp to the second plus four random hex digits, so images added
        in the same second (a bulk upload) can draw the same one. The file is therefore
        created exclusively, and a taken id is simply drawn again: an existing image is
        never overwritten.
        """
        while True:
            item_id = new_id()
            if self.ann_path(item_id).exists():
                continue
            try:
                with self.image_path(item_id).open("xb") as out:
                    out.write(data)
            except FileExistsError:
                continue
            return item_id

    def get(self, item_id: str) -> Annotation:
        """Load one annotation. Raises ``KeyError`` if there is none, ``ValueError`` for a bad id."""
        path = self.ann_path(item_id)
        if not path.is_file():
            raise KeyError(item_id)
        return Annotation.from_dict(json.loads(path.read_text()))

    def save(self, ann: Annotation, touch: bool = True) -> Annotation:
        """Write an annotation. ``touch`` updates its ``updated`` timestamp (pass False for
        bookkeeping changes such as dataset membership)."""
        if ann.status not in STATUSES:
            raise ValueError(f"Invalid status {ann.status}")
        if touch:
            ann.updated = now_iso()
        path = self.ann_path(ann.id)
        with self._lock:
            tmp = path.with_suffix(".tmp")
            tmp.write_text(json.dumps(ann.to_dict(), indent=1))
            tmp.replace(path)
        return ann

    def delete(self, item_id: str) -> None:
        """Remove an image, its annotation and its thumbnail. Missing files are not an error."""
        for p in (self.image_path(item_id), self.ann_path(item_id), self.thumbs_dir / f"{item_id}.jpg"):
            p.unlink(missing_ok=True)

    def list(self, status: str | None = None, dataset: str | None = None) -> list[Annotation]:
        """All annotations, newest first. `dataset` filters to one named dataset, or to
        images in no dataset with UNASSIGNED; None means every image."""
        items = []
        for path in self.ann_dir.glob("*.json"):
            try:
                ann = Annotation.from_dict(json.loads(path.read_text()))
            except (OSError, ValueError, KeyError, TypeError):
                continue
            if status is not None and ann.status != status:
                continue
            if dataset == UNASSIGNED and ann.datasets:
                continue
            if dataset not in (None, UNASSIGNED) and dataset not in ann.datasets:
                continue
            items.append(ann)
        items.sort(key=lambda a: a.id, reverse=True)  # ids start with a timestamp
        return items

    def neighbours(self, item_id: str, dataset: str | None = None) -> dict[str, str | None]:
        """Newer/older image and the next unlabeled one, within `dataset` (see list())."""
        items = self.list(dataset=dataset)
        ids = [a.id for a in items]
        unlabeled = [a.id for a in items if a.status == "unlabeled" and a.id != item_id]
        try:
            i = ids.index(item_id)
        except ValueError:
            return {"prev": None, "next": None, "next_unlabeled": unlabeled[0] if unlabeled else None}
        older = [x for x in unlabeled if x < item_id]
        return {
            "prev": ids[i - 1] if i > 0 else None,          # newer
            "next": ids[i + 1] if i + 1 < len(ids) else None,  # older
            "next_unlabeled": (older[0] if older else (unlabeled[0] if unlabeled else None)),
        }

    def thumbnail(self, item_id: str, max_side: int = 320) -> Path:
        """Path of the image's thumbnail, creating or refreshing it if needed."""
        thumb = self.thumbs_dir / f"{check_id(item_id)}.jpg"
        src = self.image_path(item_id)
        if not thumb.is_file() or thumb.stat().st_mtime < src.stat().st_mtime:
            img = cv2.imread(str(src))
            if img is None:
                raise KeyError(item_id)
            thumb.write_bytes(encode_jpeg(resize_max(img, max_side), 80))
        return thumb

    def stats(self, classes: list[str] | None = None, dataset: str | None = None,
              items: list[Annotation] | None = None) -> dict:
        """Counts for the UI: images by status, clean vs defective, boxes per class.

        Covers ``dataset`` (see :meth:`list`), or exactly ``items`` when given, which
        lets a caller reuse one directory scan for several summaries.
        """
        items = self.list(dataset=dataset) if items is None else items
        by_status = {s: 0 for s in STATUSES}
        by_class: dict[str, int] = {c: 0 for c in (classes or [])}
        clean = defective = 0
        for a in items:
            by_status[a.status] = by_status.get(a.status, 0) + 1
            if a.status == "labeled":
                if a.boxes:
                    defective += 1
                else:
                    clean += 1
                for b in a.boxes:
                    by_class[b.cls] = by_class.get(b.cls, 0) + 1
        return {
            "total": len(items),
            "by_status": by_status,
            "labeled_clean": clean,
            "labeled_defective": defective,
            "boxes_by_class": by_class,
        }

    # ---- named datasets ----------------------------------------------------
    @property
    def _sets_path(self) -> Path:
        return self.root / "datasets.json"

    def datasets(self) -> list[dict]:
        """Registered datasets ({name, description, created}), oldest first."""
        try:
            sets = json.loads(self._sets_path.read_text())
        except (OSError, json.JSONDecodeError):
            return []
        return [d for d in sets if isinstance(d, dict) and d.get("name")]

    def dataset_names(self) -> list[str]:
        """Names of the registered datasets, oldest first."""
        return [d["name"] for d in self.datasets()]

    def _write_sets(self, sets: list[dict]) -> None:
        tmp = self._sets_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(sets, indent=1))
        tmp.replace(self._sets_path)

    def create_dataset(self, name: str, description: str = "") -> dict:
        """Register a new, empty dataset. Raises ``ValueError`` for a bad or duplicate name."""
        name = check_dataset_name(name)
        with self._lock:
            sets = self.datasets()
            if any(d["name"].lower() == name.lower() for d in sets):
                raise ValueError(f"A dataset called {name!r} already exists")
            entry = {"name": name, "description": str(description)[:200], "created": now_iso()}
            self._write_sets([*sets, entry])
        return entry

    def update_dataset(self, name: str, *, new_name: str | None = None, description: str | None = None) -> dict:
        """Rename and/or re-describe a dataset; a rename is applied to every member image."""
        with self._lock:
            sets = self.datasets()
            entry = next((d for d in sets if d["name"] == name), None)
            if entry is None:
                raise KeyError(name)
            if new_name is not None and new_name != name:
                new_name = check_dataset_name(new_name)
                if any(d["name"].lower() == new_name.lower() for d in sets if d is not entry):
                    raise ValueError(f"A dataset called {new_name!r} already exists")
                entry["name"] = new_name
            if description is not None:
                entry["description"] = str(description)[:200]
            self._write_sets(sets)
        if entry["name"] != name:
            for ann in self.list(dataset=name):
                ann.datasets = [entry["name"] if d == name else d for d in ann.datasets]
                self.save(ann, touch=False)
        return entry

    def delete_dataset(self, name: str) -> int:
        """Forget a dataset. Its images are kept (only their membership is removed)."""
        with self._lock:
            sets = self.datasets()
            if not any(d["name"] == name for d in sets):
                raise KeyError(name)
            self._write_sets([d for d in sets if d["name"] != name])
        return self.set_membership(name, remove=[a.id for a in self.list(dataset=name)])

    def set_membership(self, name: str, add: list[str] = (), remove: list[str] = ()) -> int:
        """Add/remove images to/from a dataset. Returns how many images changed."""
        add_ids, remove_ids = {check_id(i) for i in add}, {check_id(i) for i in remove}
        if add_ids and name not in self.dataset_names():
            raise KeyError(name)
        changed = 0
        for item_id in add_ids | remove_ids:
            try:
                ann = self.get(item_id)
            except KeyError:
                continue
            before = list(ann.datasets)
            if item_id in add_ids and name not in ann.datasets:
                ann.datasets.append(name)
            if item_id in remove_ids and item_id not in add_ids:
                ann.datasets = [d for d in ann.datasets if d != name]
            if ann.datasets != before:
                self.save(ann, touch=False)
                changed += 1
        return changed

    def dataset_summaries(self, classes: list[str] | None = None) -> dict:
        """Stats for every dataset plus "all images" and "in no dataset", from one scan."""
        items = self.list()
        return {
            "all": self.stats(classes, items=items),
            "unassigned": self.stats(classes, items=[a for a in items if not a.datasets]),
            "datasets": [{**d, "stats": self.stats(classes, items=[a for a in items if d["name"] in a.datasets])}
                         for d in self.datasets()],
        }

    def adopt_unregistered_datasets(self) -> list[str]:
        """Register dataset names that images refer to but ``datasets.json`` does not list.

        This happens when annotations are copied in from elsewhere without the registry
        (for example a dataset pulled from a BoxQC 0.1 device). Returns the adopted names.
        """
        known = {name.lower() for name in self.dataset_names()}
        first_seen: dict[str, str] = {}
        for ann in reversed(self.list()):               # oldest first
            for name in ann.datasets:
                if name.lower() not in known and name not in first_seen:
                    first_seen[name] = ann.created
        adopted = []
        with self._lock:
            sets = self.datasets()
            for name, created in first_seen.items():
                try:
                    check_dataset_name(name)
                except ValueError:
                    continue
                sets.append({"name": name, "description": "", "created": created})
                adopted.append(name)
            if adopted:
                self._write_sets(sets)
        return adopted
