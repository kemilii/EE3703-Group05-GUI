"""YOLOv2 with a ResNet-18 backbone, built from individual PyTorch layers: the network and box decoding.

Training-only parts (ImageNet initialization, anchor clustering) are left out.
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


class YOLOv2ResNet18(nn.Module):
    """ResNet-18 stages (stem, s1-s4) feed the YOLOv2 head.

    Stride 32 (YOLOv2): the passthrough layer reorganizes the stride-16 features of s3 onto the 20 x 20 grid
    (pixel_unshuffle) and concatenates them with the deep features, keeping finer detail for small tools.
    Stride 16: the deep features are upsampled onto the 40 x 40 grid of s3 instead, so objects smaller than a
    32-pixel cell get cells of their own. Both feed the same 1280-channel head; it is still one output scale.
    """

    def __init__(self, anchors, classes=len(CLASSES), stride=32):
        super().__init__()
        if stride not in (16, 32):
            raise ValueError("stride must be 16 or 32")
        self.register_buffer("anchors", anchors.clone().float())  # (A, 2) normalized width, height
        self.classes, self.stride = classes, stride
        self.stem = nn.Sequential(nn.Conv2d(3, 64, 7, 2, 3, bias=False), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(3, 2, 1))
        self.s1 = nn.Sequential(BasicBlock(64, 64), BasicBlock(64, 64))
        self.s2 = nn.Sequential(BasicBlock(64, 128, 2), BasicBlock(128, 128))
        self.s3 = nn.Sequential(BasicBlock(128, 256, 2), BasicBlock(256, 256))
        self.s4 = nn.Sequential(BasicBlock(256, 512, 2), BasicBlock(512, 512))
        self.deep = nn.Sequential(conv_bn(512, 1024), conv_bn(1024, 1024))
        self.passthrough = conv_bn(256, 64 if stride == 32 else 256, 1)
        self.head = nn.Sequential(conv_bn(1280, 1024), nn.Conv2d(1024, len(anchors) * (5 + classes), 1))

    def forward(self, x):
        fine = self.s3(self.s2(self.s1(self.stem(x))))                  # stride 16, 256 channels
        deep = self.deep(self.s4(fine))                                   # stride 32, 1024 channels
        if self.stride == 32:
            route = F.pixel_unshuffle(self.passthrough(fine), 2)          # stride 32, 256 channels
        else:
            deep = F.interpolate(deep, scale_factor=2, mode="nearest")    # stride 16, 1024 channels
            route = self.passthrough(fine)                                # stride 16, 256 channels
        raw = self.head(torch.cat((deep, route), 1))
        b, _, h, w = raw.shape
        return raw.view(b, len(self.anchors), 5 + self.classes, h, w).permute(0, 1, 3, 4, 2).contiguous()


def decode(raw, anchors):
    """Raw offsets -> normalized centre-format boxes (B, A, H, W, 4).

    x, y: sigmoid offset within the grid cell plus the cell index; w, h: anchor size times exp(offset).
    """
    _, a, h, w, _ = raw.shape
    gy, gx = torch.meshgrid(torch.arange(h, device=raw.device), torch.arange(w, device=raw.device), indexing="ij")
    grid = torch.stack((gx, gy), -1).to(raw.dtype)
    xy = (raw[..., :2].sigmoid() + grid) / raw.new_tensor([w, h])
    wh = raw[..., 2:4].clamp(-8, 8).exp() * anchors[None, :, None, None, :]
    return torch.cat((xy, wh), -1)


def postprocess(raw, anchors, conf=.05, nms_iou=.5, topk=300):
    """Decode, filter by confidence and apply class-wise NMS. Returns normalized xyxy boxes per image.

    Score = objectness x class probability. At most 1000 candidates enter NMS and topk boxes are kept.
    """
    boxes = xywh_to_xyxy(decode(raw, anchors)).clamp(0, 1)
    results = []
    probs = raw[..., 4:5].sigmoid() * raw[..., 5:].softmax(-1)
    for b in range(len(raw)):
        flat = boxes[b].reshape(-1, 4)
        scores, classes = probs[b].reshape(-1, len(CLASSES)).max(1)
        ok = (scores >= conf) & ((flat[:, 2:] - flat[:, :2]).min(1).values > 0)
        flat, scores, classes = flat[ok], scores[ok], classes[ok]
        idx = scores.argsort(descending=True)[:1000]
        flat, scores, classes = flat[idx], scores[idx], classes[idx]
        chosen = []
        for c in range(len(CLASSES)):
            idx = torch.where(classes == c)[0]
            chosen.extend(idx[nms(flat[idx], scores[idx], nms_iou)].tolist())
        idx = torch.tensor(chosen, device=raw.device, dtype=torch.long)
        idx = idx[scores[idx].argsort(descending=True)[:topk]]
        results.append({"boxes": flat[idx].detach().cpu(), "scores": scores[idx].detach().cpu(),
                        "labels": classes[idx].detach().cpu()})
    return results
