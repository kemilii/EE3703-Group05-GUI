"""Faster R-CNN with a ResNet-18 backbone (inference only): RPN, RoI head and decoding.

Copied from the project repository's workbench_detection/models/faster_rcnn.py (settings, model and decoding
sections; the training losses are left out), so the GUI runs exactly the inference that was evaluated.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.models import ResNet18_Weights, resnet18
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.image_list import ImageList


# Anchors: 11 sizes in half-octave steps (x1.41) from 16 to 512 px, so a box between two octave sizes still fits an
# anchor well, and 5 aspect ratios (height / width) for long, thin tools: 55 anchors per feature cell.
ANCHOR_SIZES = (16, 23, 32, 45, 64, 91, 128, 181, 256, 362, 512)
ASPECT_RATIOS = (0.25, 0.5, 1.0, 2.0, 4.0)
FEATURE_STRIDE = 8                  # layer 3 upsampled and added to layer 2: an 80 x 80 map for 640 x 640 images
ROI_POOL = 7                        # RoIAlign output size (7 x 7)
ROI_FC_LAYERS = 2                   # fully connected layers of 1024 in the RoI head
# RoI head box offsets are scaled by these weights and trained with smooth L1 at beta 1/9 (torchvision's Faster
# R-CNN), so errors of a few percent of the box size still give a full-size gradient. The RPN uses raw offsets.
BOX_WEIGHTS = (10.0, 10.0, 5.0, 5.0)
BOX_BETA = 1 / 9
RPN_BOX_LAMBDA = 10.0               # weight of the RPN box loss
RPN_POSITIVE_IOU = 0.60             # an anchor overlapping an object above this IoU is foreground
RPN_NMS_IOU = 0.70
FINAL_NMS_IOU = 0.50
TRAIN_PROPOSALS = 600
INFERENCE_PROPOSALS = 600


class RPN(nn.Module):

    def __init__(self, in_channels=256, num_anchors=55):
        super().__init__()

        # Intermediate 3x3 convolution
        self.rpn_head = nn.Sequential(
            nn.Conv2d(
                in_channels, in_channels,
                kernel_size=3, stride=1, padding=1
            ),
            nn.ReLU()
        )

        # Objectness classification head
        self.cls_logits = nn.Conv2d(
            in_channels, num_anchors,
            kernel_size=1, stride=1
        )

        # Bounding box regression head
        self.bbox_pred = nn.Conv2d(
            in_channels, num_anchors * 4,
            kernel_size=1, stride=1
        )

    def forward(self, x):
        t = self.rpn_head(x)
        return self.cls_logits(t), self.bbox_pred(t)

class Stage2_network(nn.Module):

    def __init__(self, in_channels=256, num_classes=4, *, stride=FEATURE_STRIDE, pool_size=ROI_POOL,
                 fc_layers=ROI_FC_LAYERS):
        super().__init__()

        # Region of Interest (RoI) Align
        self.roi_align = torchvision.ops.RoIAlign(
            output_size=(pool_size, pool_size),
            spatial_scale=1.0 / stride,
            sampling_ratio=2,
            aligned=True
        )

        # Fully connected layers of 1024 (two, as in torchvision's Faster R-CNN head)
        layers, width = [], in_channels * pool_size ** 2
        for _ in range(fc_layers):
            layers += [nn.Linear(in_features=width, out_features=1024), nn.LeakyReLU(0.01)]
            width = 1024
        self.fc = nn.Sequential(*layers)

        # Classification head
        self.cls_score = nn.Linear(
            1024, num_classes
        )

        # Bounding box regression head
        self.bbox_head = nn.Linear(
            1024, (num_classes - 1) * 4
        )

    def forward(self, features, proposals):

        pooled = self.roi_align(features, proposals)

        x = pooled.flatten(start_dim=1)

        x = self.fc(x)

        return self.cls_score(x), self.bbox_head(x)

class FPNMerge(nn.Module):
    """Layer 3 (stride 16) upsampled and added to a 1x1 projection of layer 2 (stride 8), then a 3x3 smoothing
    convolution: the stride-8 map keeps layer 2's position detail and layer 3's semantics (one FPN level)."""

    def __init__(self, fine_channels=128, coarse_channels=256):
        super().__init__()
        self.lateral = nn.Conv2d(fine_channels, coarse_channels, kernel_size=1)
        self.smooth = nn.Conv2d(coarse_channels, coarse_channels, kernel_size=3, padding=1)

    def forward(self, fine, coarse):
        merged = self.lateral(fine) + F.interpolate(coarse, size=fine.shape[-2:], mode="nearest")
        return self.smooth(merged)


class FasterRCNN_Modular(nn.Module):

    def __init__(self, num_classes, pretrained=True):
        super().__init__()

        # STEP 2.3. CNN Backbone
        backbone_model = resnet18(
            weights=ResNet18_Weights.DEFAULT if pretrained else None
        )

        # ResNet-18 up to layer 2 (stride 8, 128 channels) and layer 3 (stride 16, 256 channels), merged by the neck
        # into one stride-8 map with 256 channels.
        children = list(backbone_model.children())
        self.backbone = nn.Sequential(*children[:6])   # stem + layers 1-2
        self.backbone_top = children[6]                # layer 3
        self.neck = FPNMerge(128, 256)
        channels = 256

        # STEP 2.4. Anchor Generator
        # Use the same anchor count in the generator and both RPN heads.
        self.anchor_generator = AnchorGenerator(
          sizes=(ANCHOR_SIZES,),
          aspect_ratios=(ASPECT_RATIOS,)
        )

        # Assemble RPN and Stage 2 detector
        self.rpn = RPN(
            in_channels=channels,
            num_anchors=self.anchor_generator.num_anchors_per_location()[0]
        )

        self.stage2 = Stage2_network(
            in_channels=channels,
            num_classes=num_classes,
        )
        # Small initial weights for the output layers: objectness and class scores start near uniform
        for layer in (self.rpn.cls_logits, self.rpn.bbox_pred,
                      self.stage2.cls_score, self.stage2.bbox_head):
            nn.init.normal_(layer.weight, std=0.01)
            nn.init.zeros_(layer.bias)

    def pretrained_parts(self):
        """The ResNet-18 layers (ImageNet weights); the neck, RPN and RoI head are trained from scratch."""
        return [self.backbone, self.backbone_top]

    def new_parts(self):
        return [self.neck, self.rpn, self.stage2]

    def forward(self, images):

        # STEP 2.5. Stage 1: Backbone + RPN forward pass

        # Prepare ImageList for anchor generation
        image_sizes = [
            (img.shape[-2], img.shape[-1])
            for img in images
        ]

        image_list = ImageList(
            torch.stack(images),
            image_sizes
        )

        # 1. Feature extraction
        fine = self.backbone(image_list.tensors)
        features = self.neck(fine, self.backbone_top(fine))

        # 2. Anchor generation
        anchors = self.anchor_generator(
            image_list,
            [features]
        )

        # 3. RPN predictions
        rpn_cls, rpn_reg = self.rpn(features)

        return features, anchors, rpn_cls, rpn_reg

    def forward_roi(self, features, proposals):

        # STEP 2.5. Stage 2: RoI Head forward pass
        return self.stage2(features, proposals)


def decode_boxes(ref_boxes, transforms, image_size=(640, 640), weights=None):
    if weights is not None:  # undo the offset scaling of the box coder
        transforms = transforms / transforms.new_tensor(weights)

    # Reference box dimensions and centres
    aw = (ref_boxes[:, 2] - ref_boxes[:, 0]).clamp(min=1.0)
    ah = (ref_boxes[:, 3] - ref_boxes[:, 1]).clamp(min=1.0)

    ax = ref_boxes[:, 0] + 0.5 * aw
    ay = ref_boxes[:, 1] + 0.5 * ah

    # Predicted transforms
    tx = transforms[:, 0]
    ty = transforms[:, 1]
    tw = transforms[:, 2].clamp(-4.0, 4.0)
    th = transforms[:, 3].clamp(-4.0, 4.0)

    # Apply transforms
    px = ax + aw * tx
    py = ay + ah * ty
    pw = aw * torch.exp(tw)
    ph = ah * torch.exp(th)

    # Convert to (xmin, ymin, xmax, ymax)
    refined_boxes = torch.stack([
        px - 0.5 * pw,
        py - 0.5 * ph,
        px + 0.5 * pw,
        py + 0.5 * ph
    ], dim=1)

    # --- STEP 4.3. Clip Boundaries ---
    H, W = image_size

    refined_boxes[:, [0, 2]] = refined_boxes[:, [0, 2]].clamp(0, W)
    refined_boxes[:, [1, 3]] = refined_boxes[:, [1, 3]].clamp(0, H)

    return refined_boxes

def filter_proposals(rpn_cls, rpn_reg, curr_anchors, image_size=(640, 640), *, limit=100, nms_iou=RPN_NMS_IOU,
                     pre_nms=None):
    """The `limit` best proposals after NMS, from the `pre_nms` highest-scoring anchors (default 10 x limit)."""

    with torch.no_grad():

        # Flatten RPN predictions
        scores = rpn_cls[0].permute(1, 2, 0).flatten()
        transforms = rpn_reg[0].permute(1, 2, 0).reshape(-1, 4)

        # STEP 4.1. Select top-scoring anchors
        pre_nms_topk = min(pre_nms or 10 * limit, scores.numel())

        top_scores, top_idx = scores.topk(pre_nms_topk)

        top_anchors = curr_anchors[top_idx]
        top_transforms = transforms[top_idx]

        # STEP 4.2. Refine bounding boxes
        refined_boxes = decode_boxes(
            top_anchors,
            top_transforms,
            image_size
        )

        # Remove invalid boxes
        valid = (
            torch.isfinite(refined_boxes).all(dim=1)
            & (refined_boxes[:, 2] > refined_boxes[:, 0])
            & (refined_boxes[:, 3] > refined_boxes[:, 1])
        )

        refined_boxes = refined_boxes[valid]
        top_scores = top_scores[valid]

        # STEP 4.4. Non-Maximum Suppression (NMS)
        keep_idx = torchvision.ops.nms(
            refined_boxes,
            top_scores,
            iou_threshold=nms_iou
        )

        keep_idx = keep_idx[:limit]
        return refined_boxes[keep_idx], top_scores[keep_idx].sigmoid()


def decode_detections(proposals, probabilities, regression, image_size, score_floor, *, limit=100):
    boxes_out, scores_out, labels_out = [], [], []
    for label in range(1, 4):
        scores = probabilities[:, label]
        keep = scores >= score_floor
        boxes = decode_boxes(proposals[keep], regression[keep, label - 1], image_size, BOX_WEIGHTS)
        scores = scores[keep]
        valid = torch.isfinite(boxes).all(1) & ((boxes[:, 2:] - boxes[:, :2]) > 2).all(1)
        boxes, scores = boxes[valid], scores[valid]
        chosen = torchvision.ops.nms(boxes, scores, FINAL_NMS_IOU)
        boxes_out.append(boxes[chosen])
        scores_out.append(scores[chosen])
        labels_out.append(torch.full((len(chosen),), label, dtype=torch.long, device=boxes.device))
    scores = torch.cat(scores_out)
    order = scores.argsort(descending=True)[:limit]
    return {"boxes": torch.cat(boxes_out)[order].cpu(), "labels": torch.cat(labels_out)[order].cpu(),
            "scores": scores[order].cpu()}
