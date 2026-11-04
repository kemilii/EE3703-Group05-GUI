"""Constants shared by preprocessing, the models and the GUI."""

CLASSES = ("drill", "scissor", "screwdriver")  # class IDs 0, 1, 2
IMAGE_SIZE = 640                                # every image is letterboxed to 640 x 640
LETTERBOX_FILL = (114, 114, 114)
IMAGENET_MEAN = (0.485, 0.456, 0.406)           # the ResNet-18 backbones expect ImageNet statistics
IMAGENET_STD = (0.229, 0.224, 0.225)
