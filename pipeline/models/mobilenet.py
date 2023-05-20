import torch
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights

from constants import N_CLASSES


def get_mobilenet():
    return mobilenet_v2(num_classes=N_CLASSES)


def get_mobilenet_pretrained():
    model = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)
    model.classifier[1] = torch.nn.Linear(1280, N_CLASSES)
    return model
