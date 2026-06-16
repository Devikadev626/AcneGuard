import torch.nn as nn
from torchvision import models


def get_model(num_classes=4):

    model = models.resnet18(weights="DEFAULT")

    # Replace final layer
    model.fc = nn.Linear(
        model.fc.in_features,
        num_classes
    )

    return model