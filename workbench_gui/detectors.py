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
from .models.faster_rcnn import (ANCHOR_SIZES, ASPECT_RATIOS, INFERENCE_PROPOSALS, FasterRCNN_Modular, decode_detections,
                                 filter_proposals)
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


def check_settings(saved, expected, model_name, defaults=None):
    """Refuse a checkpoint trained with a different architecture than the GUI's model code."""
    values = {**(defaults or {}), **saved}
    different = [key for key, value in expected.items()
                 if key in values and (list(values[key]) if isinstance(values[key], (list, tuple)) else values[key]) != value]
    if different:
        raise ValueError(f"This {model_name} checkpoint uses a different architecture ({', '.join(different)}).")


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
        # 40 x 40 grid, 9 anchors; settings missing from older runs had YOLOv2's original values (stride 32, 5 anchors)
        check_settings(saved["config"], {"stride": 16, "anchors": 9, "objectness_target": "iou", "anchors_per_object": 1},
                       self.model_name, defaults={"stride": 32, "anchors": 5})
        self.model = YOLOv2ResNet18(saved["anchors"])
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
    """Faster R-CNN at its operating settings: confidence threshold, RPN NMS IoU and the number of anchors kept
    before RPN NMS, all chosen on validation."""

    model_name = "Faster R-CNN"

    def __init__(self, run_dir, device="cpu"):
        self.checkpoint, settings = run_files(run_dir, self.model_name)
        self.device = torch.device(device)
        saved = torch.load(self.checkpoint, map_location="cpu", weights_only=True)
        if tuple(saved["classes"]) != CLASSES:
            raise ValueError("Checkpoint class order differs from the GUI's.")
        check_settings(saved, {"features": "c3c4", "anchor_scales": list(ANCHOR_SIZES),
                               "aspect_ratios": list(ASPECT_RATIOS), "pool_size": 7, "roi_fc_layers": 2,
                               "roi_box_coder": "standard"}, self.model_name)
        self.model = FasterRCNN_Modular(len(CLASSES) + 1, pretrained=False)
        self.model.load_state_dict(saved["state_dict"])
        self.model.to(self.device).eval()
        self.confidence, self.nms_iou = settings["operating_confidence"], settings["nms"]["nms_iou"]
        self.pre_nms = settings.get("rpn_pre_nms") or 10 * INFERENCE_PROPOSALS

    @torch.inference_mode()
    def predict_image(self, image):
        canvas, transform = letterbox(image)
        tensor = normalize(np.array(canvas)).to(self.device)
        size = tensor.shape[-2:]
        features, anchors, logits, deltas = self.model([tensor])
        proposals, objectness = filter_proposals(logits, deltas, anchors[0], size, nms_iou=self.nms_iou,
                                                 limit=INFERENCE_PROPOSALS, pre_nms=self.pre_nms)
        roi_cls, roi_reg = self.model.forward_roi(features, [proposals])
        probabilities = roi_cls.softmax(1)
        # Discount RoI predictions attached to proposals with weak RPN support.
        scores = probabilities * objectness[:, None].sqrt()
        result = decode_detections(proposals, scores, roi_reg.reshape(-1, 3, 4), size, self.confidence, limit=100)
        result["labels"] = result["labels"] - 1  # detector IDs 1-3 -> class IDs 0-2
        return boxes_to_photo(result, transform)
