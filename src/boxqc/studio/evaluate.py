"""Score a trained model the way the device will use it: one PASS/FAIL per image.

Detection metrics such as mAP say how well boxes are drawn. The line only cares whether
a defective box is caught and a clean one is passed, so after training the exported
model is run over the validation images and that decision is tallied at each threshold.
This works for any model format, because it goes through :func:`boxqc.models.load_model`.
"""

from __future__ import annotations

from pathlib import Path

import cv2

from ..models import load_model

THRESHOLDS = [round(0.05 * i, 2) for i in range(1, 19)]  # 0.05 .. 0.90


def evaluate_image_level(model_path: Path, dataset_root: Path, classes: list[str], log=print) -> dict:
    """Run the exported model over the val split; report PASS/FAIL accuracy per threshold.

    ``dataset_root`` is a directory written by :func:`boxqc.studio.export.prepare_dataset`.
    Ground truth: an image is FAIL if its label file has any boxes.
    The recommended threshold is the highest one that still catches every defective
    val image (missing a bad box is worse than a false alarm); if none does, best F1.
    Without any defective val image there is nothing to base one on, so
    ``recommended_threshold`` is ``None``. Returns ``{}`` when there are no validation
    images at all.

    The recommendation is a starting point, not a guarantee: it is fitted to the same
    validation images that training used to choose its best epoch, and it sits exactly
    at the weakest defect among them, with no margin.
    """
    detector = load_model(model_path, threads=8, fallback_classes=classes)
    img_dir = dataset_root / "images" / "val"
    lbl_dir = dataset_root / "labels" / "val"
    samples = []
    for img_path in sorted(img_dir.glob("*.jpg")):
        label = lbl_dir / f"{img_path.stem}.txt"
        is_bad = label.is_file() and label.read_text().strip() != ""
        image = cv2.imread(str(img_path))
        if image is None:
            log(f"  ! unreadable validation image {img_path.name}, skipping")
            continue
        dets, _ = detector.detect(image, conf_threshold=0.01)
        top = max((d.conf for d in dets), default=0.0)
        samples.append((img_path.stem, is_bad, top))
    if not samples:
        log("No validation images - skipping image-level evaluation.")
        return {}

    rows = []
    for t in THRESHOLDS:
        tp = sum(bad and top >= t for _, bad, top in samples)
        fn = sum(bad and top < t for _, bad, top in samples)
        fp = sum((not bad) and top >= t for _, bad, top in samples)
        tn = sum((not bad) and top < t for _, bad, top in samples)
        recall = tp / (tp + fn) if tp + fn else 1.0
        precision = tp / (tp + fp) if tp + fp else 1.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        rows.append({"threshold": t, "tp": tp, "fn": fn, "fp": fp, "tn": tn,
                     "recall": recall, "precision": precision, "f1": f1,
                     "accuracy": (tp + tn) / len(samples)})

    n_bad = sum(bad for _, bad, _ in samples)
    perfect_recall = [r for r in rows if r["recall"] == 1.0]
    best = None         # without a defective image, recall is trivially perfect at every threshold
    if n_bad:
        best = perfect_recall[-1] if perfect_recall else max(rows, key=lambda r: r["f1"])

    log(f"\nImage-level PASS/FAIL on {len(samples)} val images ({n_bad} defective, "
        f"{len(samples) - n_bad} clean):")
    log("  thr   missed-bad  false-alarm  accuracy")
    for r in rows:
        mark = "  <= recommended" if r is best else ""
        log(f"  {r['threshold']:.2f}  {r['fn']:>6d}/{r['tp'] + r['fn']:<4d} {r['fp']:>6d}/{r['fp'] + r['tn']:<5d}"
            f" {r['accuracy']:8.1%}{mark}")
    if best is None:
        log("  ! no defective validation images, so no threshold can be recommended. "
            "Label more defective images and retrain.")
        return {"recommended_threshold": None, "at_recommended": None, "table": rows,
                "val_images": len(samples)}
    misses = [sid for sid, bad, top in samples if bad and top < best["threshold"]]
    alarms = [sid for sid, bad, top in samples if not bad and top >= best["threshold"]]
    if misses:
        log(f"  missed at recommended threshold: {', '.join(misses)}")
    if alarms:
        log(f"  false alarms at recommended threshold: {', '.join(alarms)}")
    return {"recommended_threshold": best["threshold"], "at_recommended": best, "table": rows,
            "val_images": len(samples)}
