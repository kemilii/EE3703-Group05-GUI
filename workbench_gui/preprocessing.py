"""Letterboxing photos of any size to the 640 x 640 model input, mapping boxes back, and normalization.

These are the same steps the training images went through, so photos reach the models the same way.
"""

import torch
from PIL import Image

from .config import IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD, LETTERBOX_FILL


def letterbox(image, size=IMAGE_SIZE, fill=LETTERBOX_FILL):
    """Return a padded RGB image and the transform needed to map boxes back to the original pixels."""
    width, height = image.size
    scale = min(size / width, size / height)
    resized_width = max(1, round(width * scale))
    resized_height = max(1, round(height * scale))
    left, top = (size - resized_width) // 2, (size - resized_height) // 2
    canvas = Image.new("RGB", (size, size), fill)
    canvas.paste(image.resize((resized_width, resized_height), Image.Resampling.BILINEAR), (left, top))
    transform = {"scale_x": resized_width / width, "scale_y": resized_height / height,
                 "left": left, "top": top, "width": width, "height": height}
    return canvas, transform


def restore_box(box, transform):
    """Map a 640-canvas xyxy box back to original-image pixels, clipping away the padding."""
    x1, y1, x2, y2 = box

    def restore_x(x):
        return max(0, min(transform["width"], (x - transform["left"]) / transform["scale_x"]))

    def restore_y(y):
        return max(0, min(transform["height"], (y - transform["top"]) / transform["scale_y"]))

    return [restore_x(x1), restore_y(y1), restore_x(x2), restore_y(y2)]


def normalize(image):
    """RGB uint8 HWC array -> float tensor CHW with ImageNet normalization (what the model takes)."""
    tensor = torch.from_numpy(image.copy()).permute(2, 0, 1).float() / 255
    return (tensor - torch.tensor(IMAGENET_MEAN).view(3, 1, 1)) / torch.tensor(IMAGENET_STD).view(3, 1, 1)
