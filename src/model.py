import torch.nn as nn
import timm


class AcneSeverityModel(nn.Module):

    def __init__(self):

        super().__init__()

        self.model = timm.create_model(
            "efficientnet_b0",
            pretrained=True,
            num_classes=4
        )

    def forward(self, x):

        return self.model(x)