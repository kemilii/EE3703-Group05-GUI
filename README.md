# EE3703 Group 05 — Detection GUI

A browser GUI for the group's two workbench tool detectors (EE3703 bonus challenge). It lets you:

1. **upload** a photo with a button (JPG, PNG or WebP, any size),
2. **choose** the model: **YOLO** (YOLOv2 / ResNet-18) or **Faster R-CNN** (ResNet-18),
3. **see** the photo with each detection's **bounding box, class label and confidence**.

Classes: drill, scissor, screwdriver.

## How it works

```
browser (static/index.html, app.js)
   │  photo + chosen model, as JSON
   ▼
server.py ── checks the upload ── adapters/<model>.py ── detectors.py
                                                         ├─ preprocessing.py: letterbox to 640 × 640, normalize
                                                         ├─ models/yolov2.py or models/faster_rcnn.py
                                                         └─ boxes mapped back to the photo's own pixels
   ▲
   │  boxes, labels, confidences (checked: labels known, boxes inside the photo, confidences in [0, 1])
browser draws them over the photo
```

Each detector uses the **confidence threshold and NMS IoU chosen on validation** during training
(saved in the run's `evaluation_protocol.json`). Photos go through the same letterboxing and normalization as
the training images. Uploads are processed in memory and never saved.

| Path | Contents |
|---|---|
| `app.py` | Starts the GUI |
| `workbench_gui/server.py` | Local web server: serves the page, checks uploads, calls the chosen model, checks its output |
| `workbench_gui/static/` | The page: upload button, model choice, results drawn on the photo |
| `workbench_gui/adapters/` | One adapter per model, connecting the server to a trained run |
| `workbench_gui/detectors.py` | Loads a run's checkpoint and settings; photo in, detections out |
| `workbench_gui/models/` | The two networks and their box decoding |
| `workbench_gui/preprocessing.py`, `boxes.py`, `config.py` | Letterboxing, normalization, box operations, constants |
| `tests/test_gui.py` | Upload handling, the model boundary and coordinate mapping |

**Implementation boundary.** The GUI (server, page, upload checks, drawing) was written by the group using
Python's standard-library `http.server` and plain HTML/CSS/JavaScript, with no web framework. The models are the
group's own implementations; they use PyTorch layers and, for Faster R-CNN, torchvision's
ResNet-18, anchor generator, RoIAlign and NMS operators.

## Running it

Requires Python 3.10+.

```sh
pip install -r requirements.txt
```

**Model files.** The trained checkpoints (about 150 MB each) are not stored here because they exceed GitHub's
100 MB file limit. Each model needs two files from its training run, placed like this:

```
runs/<yolo run>/work/final_train/best.pt            (or work/train/best.pt, the development model)
runs/<yolo run>/work/validation/evaluation_protocol.json
runs/<faster r-cnn run>/work/final_train/best.pt
runs/<faster r-cnn run>/work/validation/evaluation_protocol.json
```

Then start the GUI and open <http://127.0.0.1:8765>:

```sh
python app.py --yolo-run runs/<yolo run> --frcnn-run runs/<faster r-cnn run>
```

A model whose run is not given appears as "not connected". Add `--device cuda` to run on a GPU.

Tests (no model files needed):

```sh
python -m unittest discover -s tests
```
