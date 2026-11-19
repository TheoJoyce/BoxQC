# Workflow: from an empty station to a working inspection

All of this happens in the studio's web UI (`boxqc studio`, <http://localhost:8080>)
with the device connected. The top bar shows the device's camera and active model, or
"device offline".

## 1. Set up the camera station (System page)

1. Fix the camera and the box position (a jig or corner stops help a lot).
2. Take a snapshot and drag the **ROI** around the box interior, then save. The model
   only sees this region, both when training and when inspecting. The ROI is stored on
   the device.
3. For repeatable images, lock focus and exposure in `camera.controls` (see
   `boxqc.yaml`) on the device.

## 2. Capture (Capture page)

* Use the live view's yellow ROI and green crosshair to centre each box.
* Press **C** (clean), **D** (defective / dirty) or **Space** (unsorted).
  * *Clean* is saved as labeled with no boxes. These negative examples are just as
    important as the defective ones.
  * *Defective* is saved as unlabeled so you can draw boxes on it next.
* Aim for **at least 150 to 300 images to start**, with about as many clean as
  defective, covering every defect type, location, wall and box variant. You can also
  upload photos from the Dataset page.

The device takes the photo; the studio stores it. Nothing is kept on the device.

### Datasets: choosing which images a model learns from

A **dataset** is a named set of images. Training uses one dataset (or every labeled
image), so you can start over without the old data and without deleting anything:

1. **Dataset → + New dataset**, with *Save new captures to this dataset* ticked.
2. Capture and label as usual; new photos go into that dataset.
3. Train with that dataset chosen on the Training page (`boxqc train --dataset NAME`).

* An image can be in several datasets, or none. Removing an image from a dataset, or
  deleting a dataset, never deletes the photo.
* On the Dataset page, pick a dataset to see only its images. Tick images (shift-click
  for a range) to add them to a dataset or remove them from the one shown.
* The **capture dataset** (set on the Capture or Dataset page) is where new captures
  and Inspect's *Send to dataset* go. Uploads on the Dataset page join the dataset you
  are looking at, or the capture dataset when you are looking at all images.
* When labeling, newer/older and *Save & next unlabeled* stay inside the dataset you
  opened the image from. The side panel shows and changes which datasets the image is in.
* Each model records the dataset it was trained on (Training page → Models).

## 3. Label (Dataset → "Label next unlabeled")

* Drag to draw a box, press **1–9** to set its class, drag a box to move it, drag a
  corner to resize, and press **Del** to remove it.
* **N** saves and opens the next unlabeled image. **C** marks the image clean. **X**
  skips it (excluded from training).
* Draw tight boxes around each separate defect. Where one wall is heavily soiled, one
  box around the soiled area is fine; be consistent.

The classes come from `classes` in `boxqc.yaml` on the studio machine.

### The labeling assistant: let a model draw the boxes

Once there is a trained model on the studio machine, the labeling page runs it over
each unlabeled image you open and shows what it finds as **suggestions**: dashed
outlines tagged `+ class NN%`, where NN is how sure the model is. A suggestion is not a
box yet. You pick the good ones, and the rest are never saved:

* **Click** a suggestion to add it as a box. It stays selected, so **1–9** changes its
  class, its corners can be dragged tighter, and **Del** takes it out again.
* **Right-click** a suggestion (or **✕** in the Assistant panel) to dismiss it.
* The **slider** in the Assistant panel hides suggestions the model is less sure of.
  Raise it until only good ones are left, then press **A** (*Add all shown*).
* Draw what the model missed by hand, as before, and press **N**. Suggestions you did
  not add are simply dropped, so a wrong one costs nothing.
* **H** turns the suggestions off and on. The panel also chooses which model suggests;
  by default it is the newest. Your choices are remembered in the browser.

The assistant only proposes. Look at each suggestion before adding it: whatever you add
is what the next model learns from, including the mistakes of the current one.

To get started without a model, label a first batch by hand (a few dozen images,
clean ones included), train on it, and let that model pre-box the remaining images.
Expect it to miss things and to be wrong at first; its suggestions improve with every
round. The assistant uses the models in `models/` on
the studio machine and works with the device switched off. On an image that is already
labeled it stays quiet until you click *Suggest boxes for this image*; it then shows
only what your boxes do not cover.

## 4. Train (Training page)

Pick a dataset, set epochs, base model, image size, batch and device, and start. While
it runs you can follow:

* the stage (prepare → train → export → evaluate → deploy) and epoch progress with an
  estimate of the time left
* mAP and loss charts, and the live log
* **Stop early & keep model**: finish the current epoch, then export and deploy the
  best model so far. **Cancel run**: abort; no model is kept.

After training, the model is scored the way the device will use it: one PASS/FAIL per
validation image, at each threshold.

```
  thr   missed-bad  false-alarm  accuracy
  0.30       0/22        3/41      95.2%
  0.35       0/22        1/41      98.4%  <= recommended
  0.40       1/22        1/41      96.8%
```

The **recommended threshold** is the highest one that still catches every defective
validation image. It is stored in the model's metadata and can be applied to the device
with one click. Training plots, the confusion matrix and PR curves are in
`work/runs/<name>/`.

Treat it as a starting point and confirm it on real boxes:

* It is fitted to the validation images, the same ones training used to pick its best
  epoch, and it sits exactly at the weakest defect among them, with no margin. A defect
  the model is less sure about than any in the validation set will pass. With few
  defective validation images, set the threshold a step or two lower.
* Several photos of the same box can land on both sides of the train/validation split,
  which makes the scores look better than they are. Vary the boxes, not just the shots.
* With no defective image in the validation set, no threshold is recommended at all.

With *Deploy to the device when done* ticked, the model is uploaded and activated. If
the device is off at that moment, the model stays on the studio machine; deploy it
later from the **Models** panel, which lists the models on both sides.

The same from the command line:

```bash
boxqc train --dataset "line 2" --deploy    # prepare → train → export → evaluate → deploy
boxqc train --epochs 150                   # writes models/boxqc-YYYYmmdd-HHMMSS.onnx (+ .json)
boxqc deploy models/boxqc-....onnx
```

### Which machine to train on

For a few hundred to a few thousand images and `yolo11n` / `yolo11s`, an Apple Silicon
laptop is enough: expect minutes to tens of minutes per run. An NVIDIA GPU is several
times faster and worth it for larger base models, image sizes of 960 or more, datasets
in the thousands, or many experiments.

*Apple Silicon:* `pip install -e ".[studio,train]"`; the device `auto` selects `mps`.
If Ultralytics hits an operator bug on MPS, choose `cpu` (slow).

*NVIDIA (Linux):* install a PyTorch build for your CUDA version **before** the rest, so
pip does not pick another build. RTX 50-series cards need CUDA 12.8 or newer:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -e ".[studio,train]"
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Only the NVIDIA driver is needed; PyTorch bundles the CUDA runtime. A batch size of
`-1` lets Ultralytics choose the largest batch that fits.

**Model size and speed:** `yolo11n` at 640 px is the default and runs well under a
second per image on a Pi 4/5. If defects are tiny, first tighten the ROI, then try an
image size of 960. Use `yolo11s` if accuracy plateaus. Both make the device slower.

## 5. Inspect (Inspect page, the trigger button, or the API)

* **Space** (or the hardware button, or `POST /api/v1/inspect`) captures an image, runs
  the model, beeps and logs the result. This part needs only the device.
* The **threshold slider** redraws the result live and shows whether the verdict would
  change. Click *Save as default threshold* to keep a new value.
* *Show sub-threshold detections* shows the weaker detections as dashed boxes, which
  helps explain near-misses.
* **Send to dataset** copies the image into the dataset with the model's boxes already
  filled in. Correct them, retrain, and the model learns from its mistakes.
* The history table is exportable as CSV. The device keeps the newest
  `inference.keep_inspections` records.

To roll back to an earlier model, pick it from the model menu in the top bar.
