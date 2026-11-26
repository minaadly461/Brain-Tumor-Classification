
import torch.nn as nn
from torchvision import models


class ResNet50Classifier(nn.Module):
    def __init__(self, num_classes=4, pretrained=True):
        super().__init__()

        weights = (
            models.ResNet50_Weights.DEFAULT
            if pretrained else None
        )

        self.model = models.resnet50(weights=weights)

        # Freeze pretrained backbone
        for param in self.model.parameters():
            param.requires_grad = False

        # Replace final classification layer
        in_features = self.model.fc.in_features

        self.model.fc = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x):
        return self.model(x)
