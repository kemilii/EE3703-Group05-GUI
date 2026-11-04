"""YOLOv2 in the GUI: the run's best checkpoint with the confidence and NMS IoU chosen on validation."""

from pathlib import Path

READY = False
STATUS = "No YOLOv2 run connected. Start the app with: python app.py --yolo-run runs/<name>"
detector = None


def connect(run_dir, device="cpu"):
    """Load the run's final checkpoint if the test has been run, otherwise its development checkpoint."""
    global READY, STATUS, detector
    from ..detectors import YOLOv2Detector

    detector = YOLOv2Detector(run_dir, device)
    READY = True
    STATUS = f"YOLOv2 loaded: {Path(run_dir).name} ({detector.checkpoint.parent.name})"


def predict(image):
    if not READY:
        raise RuntimeError(STATUS)
    return detector.predict_image(image)
