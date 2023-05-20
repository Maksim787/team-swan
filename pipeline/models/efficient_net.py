from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from torch import nn

from constants import N_CLASSES


def get_efficientnet():
    model = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
    model.classifier[1] = nn.Linear(in_features=1280, out_features=N_CLASSES)
    return model
