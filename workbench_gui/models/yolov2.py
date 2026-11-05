"""YOLOv2 with a ResNet-18 backbone (inference only): model and decoding.

Copied from the project repository's workbench_detection/models/yolov2.py (backbone, detector and decoding
sections), so the GUI runs exactly the inference that was evaluated.
"""

import torch
import torch.nn.functional as F
from torch import nn

from ..boxes import nms, xywh_to_xyxy
from ..config import CLASSES


def conv_bn(cin, cout, k=3, stride=1):
    """Convolution, batch norm and leaky ReLU: the building block of the YOLOv2 layers."""
    return nn.Sequential(nn.Conv2d(cin, cout, k, stride, k // 2, bias=False), nn.BatchNorm2d(cout),
                         nn.LeakyReLU(.1, inplace=True))


class BasicBlock(nn.Module):
    """ResNet basic block: two 3x3 convolutions with a skip connection (1x1 projection when shapes change)."""

    def __init__(self, cin, cout, stride=1):
        super().__init__()
        self.c1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False)
        self.b1 = nn.BatchNorm2d(cout)
        self.c2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False)
        self.b2 = nn.BatchNorm2d(cout)
        self.skip = (nn.Identity() if cin == cout and stride == 1 else
                     nn.Sequential(nn.Conv2d(cin, cout, 1, stride, bias=False), nn.BatchNorm2d(cout)))

    def forward(self, x):
        return F.relu(self.b2(self.c2(F.relu(self.b1(self.c1(x))))) + self.skip(x))


GRID_STRIDE = 16       # output grid 40 x 40 for 640 x 640 images
ANCHOR_COUNT = 9       # k-means anchor shapes per grid cell (YOLOv2 chose 5; 9 gives thin screwdrivers their own)


class YOLOv2ResNet18(nn.Module):
    """ResNet-18 stages (stem, s1-s4) feed the YOLOv2 head.

    The deep stride-32 features are upsampled onto the 40 x 40 grid of s3's stride-16 features and concatenated
    with them (the passthrough route), so objects smaller than a 32-pixel cell get cells of their own. One output
    scale, 1280 channels into the head.
    """

    def __init__(self, anchors, classes=len(CLASSES)):
        super().__init__()
        self.register_buffer("anchors", anchors.clone().float())  # (A, 2) normalized width, height
        self.classes, self.stride = classes, GRID_STRIDE
        self.stem = nn.Sequential(nn.Conv2d(3, 64, 7, 2, 3, bias=False), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(3, 2, 1))
        self.s1 = nn.Sequential(BasicBlock(64, 64), BasicBlock(64, 64))
        self.s2 = nn.Sequential(BasicBlock(64, 128, 2), BasicBlock(128, 128))
        self.s3 = nn.Sequential(BasicBlock(128, 256, 2), BasicBlock(256, 256))
        self.s4 = nn.Sequential(BasicBlock(256, 512, 2), BasicBlock(512, 512))
        self.deep = nn.Sequential(conv_bn(512, 1024), conv_bn(1024, 1024))
        self.passthrough = conv_bn(256, 256, 1)
        self.head = nn.Sequential(conv_bn(1280, 1024), nn.Conv2d(1024, len(anchors) * (5 + classes), 1))

    def forward(self, x):
        """(B, 3, 640, 640) images -> raw predictions (B, A, 40, 40, 5 + C)."""
        fine = self.s3(self.s2(self.s1(self.stem(x))))                  # stride 16, 256 channels
        deep = self.deep(self.s4(fine))                                   # stride 32, 1024 channels
        deep = F.interpolate(deep, scale_factor=2, mode="nearest")        # stride 16, 1024 channels
        route = self.passthrough(fine)                                    # stride 16, 256 channels
        raw = self.head(torch.cat((deep, route), 1))
        b, _, h, w = raw.shape
        return raw.view(b, len(self.anchors), 5 + self.classes, h, w).permute(0, 1, 3, 4, 2).contiguous()


def decode(raw, anchors):
    """Raw offsets -> normalized centre-format boxes (B, A, H, W, 4).

    x, y: sigmoid offset within the grid cell plus the cell index; w, h: anchor size times exp(offset).
    """
    _, _, grid_h, grid_w, _ = raw.shape
    cell_y, cell_x = torch.meshgrid(torch.arange(grid_h, device=raw.device), torch.arange(grid_w, device=raw.device),
                                    indexing="ij")
    cell_index = torch.stack((cell_x, cell_y), -1).to(raw.dtype)
    centre = (raw[..., :2].sigmoid() + cell_index) / raw.new_tensor([grid_w, grid_h])
    size = raw[..., 2:4].clamp(-8, 8).exp() * anchors[None, :, None, None, :]  # clamp keeps exp() finite
    return torch.cat((centre, size), -1)


def postprocess(raw, anchors, conf=.05, nms_iou=.5, topk=300):
    """Decode, filter by confidence and apply class-wise NMS. Returns normalized xyxy boxes per image.

    Score = objectness x class probability. At most 1000 candidates enter NMS and topk boxes are kept.
    """
    boxes = xywh_to_xyxy(decode(raw, anchors)).clamp(0, 1)
    class_scores = raw[..., 4:5].sigmoid() * raw[..., 5:].softmax(-1)  # objectness x class probability
    results = []
    for image_index in range(len(raw)):
        candidates = boxes[image_index].reshape(-1, 4)
        scores, classes = class_scores[image_index].reshape(-1, len(CLASSES)).max(1)  # best class per anchor
        # Keep confident boxes with a positive width and height, then the 1000 highest scores
        keep = (scores >= conf) & ((candidates[:, 2:] - candidates[:, :2]).min(1).values > 0)
        candidates, scores, classes = candidates[keep], scores[keep], classes[keep]
        order = scores.argsort(descending=True)[:1000]
        candidates, scores, classes = candidates[order], scores[order], classes[order]
        # Class-wise NMS: a box only suppresses boxes of its own class
        kept = []
        for class_id in range(len(CLASSES)):
            members = torch.where(classes == class_id)[0]
            kept.extend(members[nms(candidates[members], scores[members], nms_iou)].tolist())
        kept = torch.tensor(kept, device=raw.device, dtype=torch.long)
        kept = kept[scores[kept].argsort(descending=True)[:topk]]
        results.append({"boxes": candidates[kept].detach().cpu(), "scores": scores[kept].detach().cpu(),
                        "labels": classes[kept].detach().cpu()})
    return results
