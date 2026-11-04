"""Start the browser demo: upload a photo, choose YOLO or Faster R-CNN, see boxes, labels and confidences.

    python app.py --yolo-run runs/<yolo run> --frcnn-run runs/<faster r-cnn run>

Then open http://127.0.0.1:8765. A model whose run is not given shows as "not connected".
"""

import argparse
from pathlib import Path

from workbench_gui.server import serve


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--yolo-run", type=Path, help="YOLOv2 run folder")
    parser.add_argument("--frcnn-run", type=Path, help="Faster R-CNN run folder")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--device", default="cpu", help="cpu (default) or cuda")
    args = parser.parse_args()
    serve(args.port, run=args.yolo_run, frcnn_run=args.frcnn_run, device=args.device)


if __name__ == "__main__":
    main()
