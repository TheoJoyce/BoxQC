"""Turn the labeled images into a training-ready dataset on disk.

The model only ever sees the region of interest, so every image is cropped to the ROI
and its boxes are re-expressed relative to that crop. The result is a directory in the
standard YOLO layout, which most detection tools read directly::

    <out>/data.yaml
    <out>/images/train/<id>.jpg     <out>/labels/train/<id>.txt
    <out>/images/val/<id>.jpg       <out>/labels/val/<id>.txt

:func:`write_coco` adds COCO-format annotation files over the same images for tools
that prefer them. Trainers receive the :class:`PreparedDataset` and use either.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import yaml

from ..config import Config
from ..imaging import crop_roi, roi_to_pixels
from .dataset import DatasetStore
from .state import effective_roi

MIN_BOX_PX = 3  # boxes smaller than this after ROI clipping are dropped
SPLITS = ("train", "val")
# What prepare_dataset() and write_coco() put at the top of an export directory.
_EXPORT_ENTRIES = {"data.yaml", "images", "labels", "annotations"}


class EmptyDatasetError(ValueError):
    """There are no labeled images to train on."""


@dataclass
class PreparedBox:
    cls_id: int
    x0: float   # pixels in the cropped image
    y0: float
    x1: float
    y1: float


@dataclass
class PreparedSample:
    id: str
    split: str          # "train" | "val"
    width: int          # of the cropped image
    height: int
    boxes: list[PreparedBox]

    @property
    def file_name(self) -> str:
        return f"{self.id}.jpg"


@dataclass
class PreparedDataset:
    """A dataset written by :func:`prepare_dataset`."""

    yaml_path: Path                 # Ultralytics-style data.yaml
    root: Path
    counts: dict                    # per split: images, defective, clean, boxes
    roi: list[float]
    dataset: str | None = None      # the named dataset it was built from; None = every labeled image
    classes: list[str] = field(default_factory=list)
    samples: list[PreparedSample] = field(default_factory=list)

    def images_dir(self, split: str) -> Path:
        return self.root / "images" / split

    def labels_dir(self, split: str) -> Path:
        return self.root / "labels" / split


def _split_for(item_id: str, val_fraction: float) -> str:
    """Deterministic split so an image never hops between train and val across runs."""
    h = int(hashlib.sha1(item_id.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    return "val" if h < val_fraction else "train"


def check_replaceable(root: Path) -> None:
    """Raise ``ValueError`` unless ``root`` can safely be deleted and rebuilt as an export.

    The export directory is replaced on every run, and ``boxqc export --out`` lets the
    user name it, so a slip such as ``--out .`` must not erase a folder of other files.
    Only a missing or empty directory, or one that holds an earlier export and nothing
    else, is accepted.
    """
    if not root.exists():
        return
    if not root.is_dir():
        raise ValueError(f"Cannot export to {root}: it exists and is not a directory")
    entries = {p.name for p in root.iterdir()} - {".DS_Store"}
    if entries and not ("data.yaml" in entries and entries <= _EXPORT_ENTRIES):
        raise ValueError(f"Refusing to replace {root}: it is not empty and does not look like a dataset "
                         f"exported by BoxQC. Choose a new or empty directory.")


def clip_boxes(boxes, classes: list[str], width: int, height: int, roi_px) -> tuple[list[PreparedBox], int]:
    """Map full-image normalized boxes into pixel boxes relative to the ROI crop.

    Returns (boxes, number skipped). A box is skipped if its class is unknown or if
    what is left of it inside the ROI is smaller than ``MIN_BOX_PX``.
    """
    rx0, ry0, rx1, ry1 = roi_px
    out, skipped = [], 0
    for b in boxes:
        if b.cls not in classes:
            skipped += 1
            continue
        x0 = min(max(b.x0 * width, rx0), rx1) - rx0
        x1 = min(max(b.x1 * width, rx0), rx1) - rx0
        y0 = min(max(b.y0 * height, ry0), ry1) - ry0
        y1 = min(max(b.y1 * height, ry0), ry1) - ry0
        if x1 - x0 < MIN_BOX_PX or y1 - y0 < MIN_BOX_PX:
            skipped += 1
            continue
        out.append(PreparedBox(classes.index(b.cls), x0, y0, x1, y1))
    return out, skipped


def yolo_lines(boxes: list[PreparedBox], crop_width: int, crop_height: int) -> list[str]:
    """YOLO label lines: ``class cx cy w h``, normalized to the crop."""
    return [
        f"{b.cls_id} {(b.x0 + b.x1) / 2 / crop_width:.6f} {(b.y0 + b.y1) / 2 / crop_height:.6f} "
        f"{(b.x1 - b.x0) / crop_width:.6f} {(b.y1 - b.y0) / crop_height:.6f}"
        for b in boxes
    ]


def prepare_dataset(cfg: Config, out_dir: Path | None = None, log=print,
                    dataset: str | None = None, roi: list[float] | None = None) -> PreparedDataset:
    """Build the YOLO-layout dataset from labeled images - only those in `dataset` when given.

    ``out_dir`` (default ``<workdir>/yolo_dataset``) is deleted and rebuilt; a directory
    that holds anything other than an earlier export is refused (``ValueError``, see
    :func:`check_replaceable`). Raises :class:`EmptyDatasetError` when there is nothing
    labeled.

    An image with a box whose class is not in ``classes`` (a class that was renamed or
    removed in ``boxqc.yaml``) is left out altogether. Dropping only the box would turn a
    defective image into a clean example and teach the model that the defect is fine.
    """
    store = DatasetStore(cfg.data_dir)
    roi = effective_roi(cfg) if roi is None else list(roi)
    root = out_dir or (cfg.train.workdir / "yolo_dataset")

    items = store.list("labeled", dataset=dataset)
    if not items:
        where = f" in dataset {dataset!r}" if dataset else ""
        raise EmptyDatasetError(f"No labeled images{where}. Capture and label some in the studio first.")

    # Classes that boxqc.yaml no longer lists: leave those images out (see the docstring).
    stale: dict[str, int] = {}
    usable = []
    for ann in items:
        unknown = {b.cls for b in ann.boxes} - set(cfg.classes)
        for name in unknown:
            stale[name] = stale.get(name, 0) + 1
        if not unknown:
            usable.append(ann)
    if stale:
        log(f"  ! {len(items) - len(usable)} labeled image(s) left out: they have boxes of classes that "
            f"boxqc.yaml does not list ({', '.join(f'{n}: {c}' for n, c in sorted(stale.items()))}). "
            f"Add the class back to `classes`, or relabel those images.")
    items = usable
    if not items:
        raise EmptyDatasetError("Every labeled image has boxes of classes that boxqc.yaml does not list "
                                f"({', '.join(sorted(stale))}). Add them to `classes`, or relabel the images.")

    check_replaceable(root)
    if root.exists():
        shutil.rmtree(root)
    for split in SPLITS:
        (root / "images" / split).mkdir(parents=True)
        (root / "labels" / split).mkdir(parents=True)

    # Force at least one clean + one defective image into val when possible
    assignments = {a.id: _split_for(a.id, cfg.train.val_fraction) for a in items}
    for want_defect in (True, False):
        group = [a for a in items if bool(a.boxes) == want_defect]
        if len(group) >= 2 and not any(assignments[a.id] == "val" for a in group):
            assignments[group[0].id] = "val"

    counts = {s: {"images": 0, "defective": 0, "clean": 0, "boxes": 0} for s in SPLITS}
    samples: list[PreparedSample] = []
    skipped_boxes = 0
    for ann in items:
        split = assignments[ann.id]
        image = cv2.imread(str(store.image_path(ann.id)))
        if image is None:
            log(f"  ! unreadable image {ann.id}, skipping")
            continue
        h, w = image.shape[:2]
        crop, _ = crop_roi(image, roi)
        ch, cw = crop.shape[:2]
        boxes, skipped = clip_boxes(ann.boxes, cfg.classes, w, h, roi_to_pixels(roi, w, h))
        skipped_boxes += skipped
        lines = yolo_lines(boxes, cw, ch)
        cv2.imwrite(str(root / "images" / split / f"{ann.id}.jpg"), crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
        (root / "labels" / split / f"{ann.id}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
        samples.append(PreparedSample(ann.id, split, cw, ch, boxes))
        c = counts[split]
        c["images"] += 1
        c["boxes"] += len(lines)
        c["defective" if lines else "clean"] += 1

    yaml_path = root / "data.yaml"
    yaml_path.write_text(yaml.safe_dump({
        "path": str(root.resolve()),
        "train": "images/train",
        "val": "images/val",
        "names": {i: n for i, n in enumerate(cfg.classes)},
    }, sort_keys=False))

    log(f"Prepared dataset at {root}  (dataset={dataset or 'all labeled images'}, roi={roi})")
    for split, c in counts.items():
        log(f"  {split:5s}: {c['images']:4d} images  ({c['defective']} defective, {c['clean']} clean, "
            f"{c['boxes']} boxes)")
    if skipped_boxes:
        log(f"  ! {skipped_boxes} boxes skipped (outside the ROI, or under {MIN_BOX_PX} px of them inside it)")
    if counts["val"]["images"] == 0:
        log("  ! validation split is empty - add more labeled images")
    return PreparedDataset(yaml_path, root, counts, roi, dataset, list(cfg.classes), samples)


def write_coco(prepared: PreparedDataset) -> dict[str, Path]:
    """Write ``annotations/instances_<split>.json`` (COCO detection format) next to the images.

    Category ids are the class indices plus one (COCO ids start at 1). Returns the file per split.
    """
    out_dir = prepared.root / "annotations"
    out_dir.mkdir(parents=True, exist_ok=True)
    categories = [{"id": i + 1, "name": name} for i, name in enumerate(prepared.classes)]
    paths: dict[str, Path] = {}
    for split in SPLITS:
        images, annotations = [], []
        for sample in (s for s in prepared.samples if s.split == split):
            image_id = len(images) + 1
            images.append({"id": image_id, "file_name": sample.file_name,
                           "width": sample.width, "height": sample.height})
            for b in sample.boxes:
                w, h = b.x1 - b.x0, b.y1 - b.y0
                annotations.append({
                    "id": len(annotations) + 1, "image_id": image_id, "category_id": b.cls_id + 1,
                    "bbox": [round(b.x0, 2), round(b.y0, 2), round(w, 2), round(h, 2)],
                    "area": round(w * h, 2), "iscrowd": 0,
                })
        paths[split] = out_dir / f"instances_{split}.json"
        paths[split].write_text(json.dumps(
            {"images": images, "annotations": annotations, "categories": categories}))
    return paths
