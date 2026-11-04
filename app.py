"""Start the browser demo: upload a photo, choose YOLO or Faster R-CNN, see boxes, labels and confidences.

    python download_models.py      # once: the trained models, into runs/
    python app.py

Then open http://127.0.0.1:8765. By default the models in runs/yolov2 and runs/faster_rcnn are used; a model
that is not there shows as "not connected".
"""

import argparse
from pathlib import Path

from workbench_gui.server import serve

RUNS = Path(__file__).resolve().parent / "runs"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--yolo-run", type=Path, default=RUNS / "yolov2", help="YOLOv2 run folder")
    parser.add_argument("--frcnn-run", type=Path, default=RUNS / "faster_rcnn", help="Faster R-CNN run folder")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--device", default="cpu", help="cpu (default) or cuda")
    args = parser.parse_args()
    present = {name: run if run.is_dir() else None for name, run in (("YOLOv2", args.yolo_run), ("Faster R-CNN", args.frcnn_run))}
    for name, run in present.items():
        print(f"{name}: {run if run else 'no model found (run python download_models.py)'}")
    serve(args.port, run=present["YOLOv2"], frcnn_run=present["Faster R-CNN"], device=args.device)


if __name__ == "__main__":
    main()
