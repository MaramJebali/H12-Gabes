import torchvision.models as models
from torch import nn


def build_resnet18_model(num_classes: int = 3):
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

    num_ftrs = model.fc.in_features

    model.fc = nn.Sequential(
        nn.Linear(num_ftrs, num_classes),
        nn.Dropout(p=0.2),
    )

    return model