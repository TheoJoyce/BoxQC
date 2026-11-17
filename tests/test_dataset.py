"""The studio's image store and the dataset export used for training."""

import json

import cv2
import numpy as np
import pytest
from helpers import make_onnx, one_box_preds

from boxqc.imaging import encode_jpeg
from boxqc.studio import dataset as dataset_module
from boxqc.studio.dataset import UNASSIGNED, Box, DatasetStore
from boxqc.studio.evaluate import evaluate_image_level
from boxqc.studio.export import EmptyDatasetError, prepare_dataset, write_coco

IMG = np.full((60, 80, 3), 128, np.uint8)
QUIET = lambda *_: None  # noqa: E731


def test_store_datasets_membership_rename_delete(tmp_path):
    store = DatasetStore(tmp_path)
    old = store.add_image(IMG)
    store.create_dataset("v2", "fresh start")
    with pytest.raises(ValueError):
        store.create_dataset("V2")                     # names are unique, ignoring case
    with pytest.raises(ValueError):
        store.create_dataset("../etc")
    new = store.add_image(IMG, datasets=["v2", "nope"])  # unknown names are dropped
    assert store.get(new.id).datasets == ["v2"]

    assert [a.id for a in store.list(dataset="v2")] == [new.id]
    assert [a.id for a in store.list(dataset=UNASSIGNED)] == [old.id]
    assert len(store.list()) == 2

    assert store.set_membership("v2", add=[old.id]) == 1
    assert store.set_membership("v2", add=[old.id]) == 0  # already a member
    store.update_dataset("v2", new_name="round 2")
    assert {a.id for a in store.list(dataset="round 2")} == {old.id, new.id}

    assert store.delete_dataset("round 2") == 2
    assert store.dataset_names() == []
    assert len(store.list()) == 2                       # images are never deleted
    assert all(a.datasets == [] for a in store.list())


def test_neighbours_stay_inside_dataset(tmp_path):
    store = DatasetStore(tmp_path)
    store.create_dataset("a")
    outside = store.add_image(IMG)
    inside = [store.add_image(IMG, datasets=["a"]) for _ in range(2)]
    nav = store.neighbours(inside[0].id, dataset="a")
    assert nav["next_unlabeled"] == inside[1].id
    assert outside.id not in nav.values()


def test_add_image_stores_jpeg_bytes_untouched_and_converts_the_rest(tmp_path):
    store = DatasetStore(tmp_path)
    jpeg = encode_jpeg(IMG, 60)
    ann = store.add_image(jpeg)
    assert store.image_path(ann.id).read_bytes() == jpeg       # a device photo is not re-compressed
    assert (ann.width, ann.height) == (80, 60)

    png = cv2.imencode(".png", IMG)[1].tobytes()
    converted = store.image_path(store.add_image(png).id).read_bytes()
    assert converted.startswith(b"\xff\xd8")
    with pytest.raises(ValueError):
        store.add_image(b"not an image")


def test_adopt_registers_datasets_that_only_annotations_mention(tmp_path):
    store = DatasetStore(tmp_path)
    store.create_dataset("known")
    ann = store.add_image(IMG, datasets=["known"])
    # an annotation copied in from another machine, without datasets.json
    raw = json.loads(store.ann_path(ann.id).read_text())
    raw["datasets"] = ["known", "Testing Data"]
    store.ann_path(ann.id).write_text(json.dumps(raw))

    assert store.adopt_unregistered_datasets() == ["Testing Data"]
    assert store.dataset_names() == ["known", "Testing Data"]
    assert store.adopt_unregistered_datasets() == []


def test_prepare_crops_roi_and_remaps_boxes(cfg):
    store = DatasetStore(cfg.data_dir)
    img = np.zeros((100, 200, 3), np.uint8)
    # box covering x 120..160, y 20..60 in a 200x100 image
    store.add_image(img, status="labeled", boxes=[Box("dirty", 0.6, 0.2, 0.8, 0.6)])
    store.add_image(img, status="labeled")            # clean negative
    store.add_image(img, status="unlabeled")          # excluded

    prepared = prepare_dataset(cfg, log=QUIET, roi=[0.5, 0.0, 1.0, 1.0])
    total = sum(c["images"] for c in prepared.counts.values())
    assert total == 2
    labels = [p.read_text().strip() for p in prepared.root.glob("labels/*/*.txt")]
    non_empty = [line for line in labels if line]
    assert len(non_empty) == 1
    cls, cx, cy, w, h = non_empty[0].split()
    # crop is x 100..200 (100 px wide): box 20..60 -> centre 0.4, width 0.4
    assert cls == "0"
    assert float(cx) == pytest.approx(0.4)
    assert float(cy) == pytest.approx(0.4)
    assert float(w) == pytest.approx(0.4)
    assert float(h) == pytest.approx(0.4)
    crops = list(prepared.root.glob("images/*/*.jpg"))
    assert all(cv2.imread(str(p)).shape[:2] == (100, 100) for p in crops)

    # the same boxes in COCO form: pixels of the crop, [x, y, width, height]
    coco = [json.loads(p.read_text()) for p in write_coco(prepared).values()]
    annotations = [a for split in coco for a in split["annotations"]]
    assert len(annotations) == 1 and annotations[0]["bbox"] == [20.0, 20.0, 40.0, 40.0]
    assert annotations[0]["category_id"] == 1
    assert coco[0]["categories"] == [{"id": 1, "name": "dirty"}, {"id": 2, "name": "defect"}]
    assert sum(len(split["images"]) for split in coco) == 2


def test_prepare_uses_only_the_chosen_dataset(cfg):
    store = DatasetStore(cfg.data_dir)
    store.create_dataset("new")
    store.add_image(IMG, status="labeled")                                  # old data
    for i in range(4):
        boxes = [Box("dirty", 0.1, 0.1, 0.5, 0.5)] if i % 2 else []
        store.add_image(IMG, status="labeled", boxes=boxes, datasets=["new"])
    prepared = prepare_dataset(cfg, log=QUIET, dataset="new")
    total = sum(c["images"] for c in prepared.counts.values())
    assert total == 4 and prepared.dataset == "new"
    assert sum(c["images"] for c in prepare_dataset(cfg, log=QUIET).counts.values()) == 5
    store.create_dataset("empty")
    with pytest.raises(EmptyDatasetError, match="empty"):
        prepare_dataset(cfg, log=QUIET, dataset="empty")


def test_images_added_in_the_same_second_never_overwrite_each_other(tmp_path, monkeypatch):
    """Ids are a timestamp plus four random hex digits, so a bulk upload can draw one twice."""
    drawn = iter(["20260101-120000-aaaa", "20260101-120000-aaaa", "20260101-120000-aaaa", "20260101-120000-bbbb"])
    monkeypatch.setattr(dataset_module, "new_id", lambda: next(drawn))
    store = DatasetStore(tmp_path)
    first = store.add_image(IMG, notes="first")
    second = store.add_image(np.zeros_like(IMG), notes="second")
    assert (first.id, second.id) == ("20260101-120000-aaaa", "20260101-120000-bbbb")
    assert store.get(first.id).notes == "first"
    assert cv2.imread(str(store.image_path(first.id))).mean() > 100      # still the grey image
    assert len(store.list()) == 2


def test_export_refuses_to_replace_a_folder_that_is_not_an_export(cfg, tmp_path):
    store = DatasetStore(cfg.data_dir)
    store.add_image(IMG, status="labeled")
    precious = tmp_path / "my-documents"
    precious.mkdir()
    (precious / "thesis.txt").write_text("years of work")
    with pytest.raises(ValueError, match="Refusing to replace"):
        prepare_dataset(cfg, out_dir=precious, log=QUIET)
    assert (precious / "thesis.txt").read_text() == "years of work"
    with pytest.raises(ValueError, match="not a directory"):
        prepare_dataset(cfg, out_dir=precious / "thesis.txt", log=QUIET)

    # a new folder, an empty one and an earlier export (with its COCO files) are all fine
    out = tmp_path / "export"
    write_coco(prepare_dataset(cfg, out_dir=out, log=QUIET))
    (out / "images" / "stale.jpg").write_bytes(b"x")
    prepared = prepare_dataset(cfg, out_dir=out, log=QUIET)
    assert not (out / "images" / "stale.jpg").exists() and prepared.yaml_path.is_file()
    empty = tmp_path / "empty"
    empty.mkdir()
    assert prepare_dataset(cfg, out_dir=empty, log=QUIET).root == empty


def test_export_leaves_out_images_labeled_with_a_class_that_is_gone(cfg):
    """Dropping only the box would turn a defective image into a clean training example."""
    store = DatasetStore(cfg.data_dir)
    store.add_image(IMG, status="labeled", boxes=[Box("dirty", 0.1, 0.1, 0.5, 0.5)])
    store.add_image(IMG, status="labeled", boxes=[Box("scratch", 0.1, 0.1, 0.5, 0.5)])    # renamed since
    lines = []
    prepared = prepare_dataset(cfg, log=lines.append)
    assert sum(c["images"] for c in prepared.counts.values()) == 1
    assert sum(c["clean"] for c in prepared.counts.values()) == 0
    assert any("1 labeled image(s) left out" in line and "scratch" in line for line in lines)

    cfg.classes = ["something-else"]
    with pytest.raises(EmptyDatasetError, match="dirty, scratch"):
        prepare_dataset(cfg, log=QUIET)


def test_no_threshold_is_recommended_without_a_defective_validation_image(tmp_path):
    root = tmp_path / "ds"
    for sub_dir in ("images/val", "labels/val"):
        (root / sub_dir).mkdir(parents=True)
    for i in range(3):
        cv2.imwrite(str(root / "images" / "val" / f"clean{i}.jpg"), np.zeros((64, 64, 3), np.uint8))
        (root / "labels" / "val" / f"clean{i}.txt").write_text("")
    model = make_onnx(tmp_path / "m.onnx", one_box_preds(0.6))
    lines = []
    result = evaluate_image_level(model, root, ["dirty", "defect"], log=lines.append)
    assert result["recommended_threshold"] is None and result["at_recommended"] is None
    assert result["val_images"] == 3 and len(result["table"]) == 18
    assert any("no threshold can be recommended" in line for line in lines)

    # with one defective image the rule applies again: the highest threshold that catches it
    cv2.imwrite(str(root / "images" / "val" / "bad.jpg"), np.zeros((64, 64, 3), np.uint8))
    (root / "labels" / "val" / "bad.txt").write_text("1 0.5 0.5 0.2 0.2\n")
    assert evaluate_image_level(model, root, ["dirty", "defect"], log=QUIET)["recommended_threshold"] == 0.6
