"""The two trained detectors behind one call: predict_image(photo) -> boxes, labels and confidences.

Each detector loads a training run folder: the checkpoint, plus the operating confidence and NMS IoU chosen on
validation and frozen in work/validation/evaluation_protocol.json. The inference steps are the ones used to evaluate
the models.
"""

import json
from pathlib import Path

import numpy as np
import torch

from .config import CLASSES, IMAGE_SIZE
from .models.faster_rcnn import ASPECT_RATIOS, INFERENCE_PROPOSALS, FasterRCNN_Modular, decode_detections, filter_proposals
from .models.yolov2 import YOLOv2ResNet18, postprocess
from .preprocessing import letterbox, normalize, restore_box


def run_files(run_dir, model_name):
    """(checkpoint, frozen settings) of a run folder.

    Uses the final model (retrained on train + validation for the test) if present, otherwise the development
    model that validation chose the settings for.
    """
    run_dir = Path(run_dir)
    protocol_path = run_dir / "work" / "validation" / "evaluation_protocol.json"
    if not protocol_path.is_file():
        raise FileNotFoundError(f"{protocol_path} not found: copy the run's evaluation protocol there.")
    settings = json.loads(protocol_path.read_text(encoding="utf-8"))["models"][model_name]
    for checkpoint in (run_dir / "work" / "final_train" / "best.pt", run_dir / "work" / "train" / "best.pt"):
        if checkpoint.is_file():
            return checkpoint, settings
    raise FileNotFoundError(f"No best.pt under {run_dir / 'work'}/final_train or /train.")


def boxes_to_photo(result, transform):
    """Canvas detections -> [{"label", "confidence", "box"}] in the photo's own pixels."""
    detections = []
    for box, label, score in zip(result["boxes"].tolist(), result["labels"].tolist(), result["scores"].tolist()):
        restored = restore_box(box, transform)
        if restored[2] > restored[0] and restored[3] > restored[1]:
            detections.append({"label": CLASSES[label], "confidence": score, "box": restored})
    return detections


class YOLOv2Detector:
    """YOLOv2 at its operating settings: confidence threshold and class-wise NMS IoU chosen on validation."""

    model_name = "YOLOv2"

    def __init__(self, run_dir, device="cpu"):
        self.checkpoint, settings = run_files(run_dir, self.model_name)
        self.device = torch.device(device)
        saved = torch.load(self.checkpoint, map_location="cpu", weights_only=True)
        if tuple(saved["classes"]) != CLASSES:
            raise ValueError("Checkpoint class order differs from the GUI's.")
        self.model = YOLOv2ResNet18(saved["anchors"], stride=saved["config"].get("stride", 32))
        self.model.load_state_dict(saved["state_dict"])
        self.model.to(self.device).eval()
        self.confidence, self.nms_iou = settings["operating_confidence"], settings["nms"]["final_nms_iou"]

    @torch.inference_mode()
    def predict_image(self, image):
        canvas, transform = letterbox(image)
        raw = self.model(normalize(np.array(canvas))[None].to(self.device)).float()
        result = postprocess(raw.cpu(), self.model.anchors.cpu(), conf=self.confidence, nms_iou=self.nms_iou)[0]
        result["boxes"] *= IMAGE_SIZE  # normalized -> canvas pixels
        return boxes_to_photo(result, transform)


class FasterRCNNDetector:
    """Faster R-CNN at its operating settings: confidence threshold and RPN NMS IoU chosen on validation."""

    model_name = "Faster R-CNN"

    def __init__(self, run_dir, device="cpu"):
        self.checkpoint, settings = run_files(run_dir, self.model_name)
        self.device = torch.device(device)
        saved = torch.load(self.checkpoint, map_location="cpu", weights_only=True)
        if tuple(saved["classes"]) != CLASSES:
            raise ValueError("Checkpoint class order differs from the GUI's.")
        self.model = FasterRCNN_Modular(len(CLASSES) + 1, pretrained=False, anchor_scales=tuple(saved["anchor_scales"]),
                                        aspect_ratios=tuple(saved.get("aspect_ratios", ASPECT_RATIOS)))
        self.model.load_state_dict(saved["state_dict"])
        self.model.to(self.device).eval()
        self.confidence, self.nms_iou = settings["operating_confidence"], settings["nms"]["nms_iou"]

    @torch.inference_mode()
    def predict_image(self, image):
        canvas, transform = letterbox(image)
        tensor = normalize(np.array(canvas)).to(self.device)
        size = tensor.shape[-2:]
        features, anchors, logits, deltas = self.model([tensor])
        proposals, objectness = filter_proposals(logits, deltas, anchors[0], size, nms_iou=self.nms_iou,
                                                 limit=INFERENCE_PROPOSALS if self.model.dense else 100)
        roi_cls, roi_reg = self.model.forward_roi(features, [proposals])
        probabilities = roi_cls.softmax(1)
        # Discount RoI predictions attached to proposals with weak RPN support (dense profile).
        scores = probabilities * objectness[:, None].sqrt() if self.model.dense else probabilities
        result = decode_detections(proposals, scores, roi_reg.reshape(-1, 3, 4), size, self.confidence,
                                   limit=100 if self.model.dense else 20)
        result["labels"] = result["labels"] - 1  # detector IDs 1-3 -> class IDs 0-2
        return boxes_to_photo(result, transform)
