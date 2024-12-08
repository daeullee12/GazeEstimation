import torch
import torch.nn as nn
from efficientnet_pytorch import EfficientNet  # Import EfficientNet
from torchvision.models import mobilenet_v2  # Import MobileNetV2
from resnet_2 import resnet18  # Import ResNet18 from resnet_2


class Model(nn.Module):
    def __init__(self, config, backbone='efficientnet-b0'):
        super(Model, self).__init__()
        self.config = config
        self.backbone = backbone

        if backbone == 'efficientnet-b0':
            self.base_model = EfficientNet.from_pretrained('efficientnet-b0')  # Initialize EfficientNet with pre-trained weights
            in_features = self.base_model._fc.in_features
            self.base_model._fc = nn.Linear(in_features, 2)  # Modify the final layer for regression (yaw, pitch)

        elif backbone == 'mobilenet_v2':
            self.base_model = mobilenet_v2(pretrained=True)  # Initialize MobileNetV2 with pre-trained weights
            in_features = self.base_model.classifier[1].in_features
            self.base_model.classifier[1] = nn.Linear(in_features, 2)  # Modify the final layer for regression (yaw, pitch)

        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

    def forward(self, x):
        return self.base_model(x)

    def loss(self, data, anno):
        output = self.forward(data["face"])
        criterion = nn.MSELoss()
        loss = criterion(output, anno)
        return loss