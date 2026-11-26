"""
Spatial ResNet18 Model for Deepfake Detection
=============================================
Modular CNN-based spatial feature extractor and binary classifier based on
pretrained ResNet18.

Key Features:
- Pretrained ImageNet ResNet18 backbone.
- Binary classification head: REAL (0) vs DEEPFAKE (1).
- Modular `extract_features(x)` returning the 512-dim penultimate embedding,
  enabling direct reuse in Phase 6 (CNN + FFT + DWT feature fusion).
- Configurable backbone freezing for transfer learning and fine-tuning.
- Automatic CPU / GPU device management.
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
from torchvision.models import ResNet18_Weights, resnet18


class SpatialResNet18(nn.Module):
    """ResNet18-based spatial feature extractor and classifier for deepfake detection."""

    def __init__(
        self,
        pretrained: bool = True,
        num_classes: int = 2,
        dropout_rate: float = 0.3,
        freeze_backbone: bool = False,
    ):
        super().__init__()
        self.num_classes = num_classes

        # Load ResNet18
        if pretrained:
            try:
                weights = ResNet18_Weights.DEFAULT
                self.backbone = resnet18(weights=weights)
            except Exception:
                self.backbone = resnet18(weights=None)
        else:
            self.backbone = resnet18(weights=None)

        in_features = self.backbone.fc.in_features  # 512 for ResNet18

        # Replace classification head with custom binary classifier
        self.backbone.fc = nn.Identity()

        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features, num_classes),
        )

        if freeze_backbone:
            self.freeze_backbone_layers(freeze=True)

    def freeze_early_layers(self) -> None:
        """Freeze conv1, bn1, layer1, layer2, layer3. Keep layer4 and classifier trainable."""
        for name, child in self.backbone.named_children():
            if name != "layer4":
                for param in child.parameters():
                    param.requires_grad = False
            else:
                for param in child.parameters():
                    param.requires_grad = True

    def freeze_backbone_layers(self, freeze: bool = True) -> None:
        """Freeze or unfreeze the convolutional backbone parameters."""
        for param in self.backbone.parameters():
            param.requires_grad = not freeze

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract the 512-dimensional spatial feature vector from the penultimate layer.
        Essential for Phase 6 feature fusion (CNN + FFT + DWT).

        Parameters
        ----------
        x : torch.Tensor of shape (B, 3, 224, 224)

        Returns
        -------
        torch.Tensor of shape (B, 512)
        """
        return self.backbone(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass returning raw unnormalized logits.

        Parameters
        ----------
        x : torch.Tensor of shape (B, 3, 224, 224)

        Returns
        -------
        torch.Tensor of shape (B, 2)
        """
        features = self.extract_features(x)
        logits = self.classifier(features)
        return logits

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning softmax probability distributions."""
        logits = self.forward(x)
        return torch.softmax(logits, dim=1)


def get_default_device() -> torch.device:
    """Return CUDA device if available, otherwise CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def build_spatial_model(
    pretrained: bool = True,
    dropout_rate: float = 0.3,
    freeze_backbone: bool = False,
    fine_tune_layer4: bool = False,
    device: Optional[torch.device] = None,
) -> Tuple[SpatialResNet18, torch.device]:
    """Helper factory to instantiate and device-bind the spatial ResNet18 model."""
    if device is None:
        device = get_default_device()

    model = SpatialResNet18(
        pretrained=pretrained,
        num_classes=2,
        dropout_rate=dropout_rate,
        freeze_backbone=freeze_backbone,
    )
    if fine_tune_layer4:
        model.freeze_early_layers()
    model.to(device)
    return model, device
