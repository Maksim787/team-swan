from torch import nn
from torchvision.models import resnet18, ResNet18_Weights
from torchvision.models import resnet152, ResNet152_Weights

from constants import N_CLASSES


def get_resnet18():
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, N_CLASSES)
    return model

def get_resnet152():
    model = resnet152(weights=ResNet152_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, N_CLASSES)
    return model
