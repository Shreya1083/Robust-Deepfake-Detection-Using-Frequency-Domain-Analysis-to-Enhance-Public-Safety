"""
Grad-CAM Spatial Feature Visualization Module
=============================================
Computes Gradient-weighted Class Activation Mapping (Grad-CAM) for ResNet18
spatial feature analysis in deepfake detection.

Key Features:
- Attaches hooks to the final convolutional layer (layer4).
- Computes actual gradients of the predicted class with respect to layer activations.
- Weighting via Global Average Pooling of gradients.
- ReLU activation to isolate positive contributions.
- Generates:
  * Grayscale normalized heatmap [0, 1]
  * Colorized heatmap using OpenCV JET colormap
  * Heatmap overlay on original RGB face image
  * Salient facial region identification based on peak activation coordinates
  * Base64 encoding for seamless API and JSON transport
"""

import base64
import io
import logging
from typing import Dict, Optional, Tuple, Union

import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class GradCAM:
    """
    Grad-CAM implementation for PyTorch CNN models (specifically ResNet18).
    """

    def __init__(self, model: nn.Module, target_layer: Optional[nn.Module] = None):
        self.model = model
        self.device = next(model.parameters()).device

        # If target layer is not explicitly provided, find the last conv layer in ResNet18
        if target_layer is None:
            if hasattr(model, "backbone") and hasattr(model.backbone, "layer4"):
                self.target_layer = model.backbone.layer4[-1]
            elif hasattr(model, "layer4"):
                self.target_layer = model.layer4[-1]
            else:
                # Find last Conv2d layer
                last_conv = None
                for module in model.modules():
                    if isinstance(module, nn.Conv2d):
                        last_conv = module
                self.target_layer = last_conv
        else:
            self.target_layer = target_layer

        if self.target_layer is None:
            raise ValueError("Could not automatically identify a target convolutional layer for Grad-CAM.")

        self.activations: Optional[torch.Tensor] = None
        self.gradients: Optional[torch.Tensor] = None

        # Register forward and backward hooks
        self._register_hooks()

    def _register_hooks(self) -> None:
        def forward_hook(module, input_tensor, output_tensor):
            self.activations = output_tensor.detach()

        def backward_hook(module, grad_input, grad_output):
            if grad_output[0] is not None:
                self.gradients = grad_output[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        # register_full_backward_hook is preferred in modern PyTorch
        if hasattr(self.target_layer, "register_full_backward_hook"):
            self.target_layer.register_full_backward_hook(backward_hook)
        else:
            self.target_layer.register_backward_hook(backward_hook)

    def generate(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None,
        original_image_rgb: Optional[np.ndarray] = None,
    ) -> Dict:
        """
        Compute Grad-CAM heatmap and overlay.

        Parameters
        ----------
        input_tensor : torch.Tensor of shape (1, 3, H, W)
            Preprocessed input image tensor.
        target_class : int, optional
            Index of target class. If None, uses the class with the highest logit.
        original_image_rgb : np.ndarray, optional
            Original RGB image (H_orig, W_orig, 3) as uint8 [0..255].

        Returns
        -------
        Dict containing:
            - heatmap : np.ndarray of shape (H, W), values in [0, 1]
            - heatmap_colored : np.ndarray of shape (H, W, 3), RGB uint8
            - overlay : np.ndarray of shape (H, W, 3), RGB uint8
            - heatmap_b64 : str (base64 JPEG)
            - overlay_b64 : str (base64 JPEG)
            - salient_regions : str (description of influential area)
            - target_layer_name : str
            - target_class : int
        """
        self.model.eval()
        input_tensor = input_tensor.to(self.device).requires_grad_(True)

        # 1. Forward pass
        logits = self.model(input_tensor)
        if target_class is None:
            target_class = int(torch.argmax(logits, dim=1).item())

        score = logits[0, target_class]

        # 2. Backward pass
        self.model.zero_grad()
        score.backward(retain_graph=True)

        if self.activations is None or self.gradients is None:
            raise RuntimeError("Grad-CAM failed to capture activations or gradients from the target layer.")

        # 3. Compute weights via Global Average Pooling of gradients
        # activations shape: (1, C, H_feat, W_feat)
        # gradients shape:   (1, C, H_feat, W_feat)
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)  # (1, C, 1, 1)

        # 4. Weighted combination of activation maps
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)  # (1, 1, H_feat, W_feat)

        # 5. ReLU to isolate positive contributions
        cam = F.relu(cam)

        # 6. Normalize cam to [0, 1]
        cam = cam.squeeze().cpu().numpy()
        cam_min, cam_max = np.min(cam), np.max(cam)
        if cam_max - cam_min > 1e-8:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)

        # 7. Resize heatmap to match image dimensions
        if original_image_rgb is not None:
            target_h, target_w = original_image_rgb.shape[:2]
        else:
            target_h, target_w = input_tensor.shape[2], input_tensor.shape[3]

        heatmap_resized = cv2.resize(cam, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

        # 8. Create colored heatmap (JET colormap converted to RGB)
        heatmap_uint8 = np.uint8(255 * heatmap_resized)
        heatmap_jet_bgr = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        heatmap_jet_rgb = cv2.cvtColor(heatmap_jet_bgr, cv2.COLOR_BGR2RGB)

        # 9. Create overlay
        if original_image_rgb is not None:
            base_rgb = original_image_rgb.copy()
            if base_rgb.dtype != np.uint8:
                base_rgb = np.uint8(np.clip(base_rgb, 0, 255))
        else:
            # Reconstruct from input_tensor
            t = input_tensor.squeeze(0).detach().cpu().numpy()
            t = np.transpose(t, (1, 2, 0))
            # Denormalize ImageNet
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            base_rgb = np.uint8(np.clip((t * std + mean) * 255, 0, 255))
            if (base_rgb.shape[0], base_rgb.shape[1]) != (target_h, target_w):
                base_rgb = cv2.resize(base_rgb, (target_w, target_h))

        # Alpha blend: 0.65 original + 0.35 heatmap
        overlay = cv2.addWeighted(base_rgb, 0.65, heatmap_jet_rgb, 0.35, 0)

        # 10. Analyze salient regions based on peak activation coordinates
        salient_desc = self._identify_salient_regions(heatmap_resized)

        # 11. Encode to base64
        heatmap_b64 = self._to_base64_jpeg(heatmap_jet_rgb)
        overlay_b64 = self._to_base64_jpeg(overlay)

        return {
            "heatmap": heatmap_resized,
            "heatmap_colored": heatmap_jet_rgb,
            "overlay": overlay,
            "heatmap_b64": heatmap_b64,
            "overlay_b64": overlay_b64,
            "salient_regions": salient_desc,
            "target_layer_name": "layer4 (Conv2d, 512 channels, 7x7)",
            "target_class": target_class,
        }

    @staticmethod
    def _identify_salient_regions(heatmap: np.ndarray) -> str:
        """
        Dynamically determine the facial anatomical sector with the strongest activation.
        Uses facial sector coordinate grid (upper, mid, lower, peripheral).
        """
        h, w = heatmap.shape
        threshold = 0.6 * np.max(heatmap)
        high_act = heatmap >= threshold

        if not np.any(high_act):
            return "Diffuse activation across facial contours"

        y_indices, x_indices = np.where(high_act)
        mean_y = float(np.mean(y_indices)) / h
        mean_x = float(np.mean(x_indices)) / w

        # Vertical sector
        if mean_y < 0.35:
            vert_sector = "Forehead and upper ocular boundary"
        elif mean_y < 0.65:
            vert_sector = "Periorbital, nasal, and cheek dermal texture"
        else:
            vert_sector = "Perioral (mouth), jawline, and chin boundary"

        # Horizontal sector
        if mean_x < 0.35:
            horiz_sector = "left facial periphery"
        elif mean_x > 0.65:
            horiz_sector = "right facial periphery"
        else:
            horiz_sector = "central facial features"

        # Check edge / boundary proportion
        edge_mask = np.zeros_like(heatmap, dtype=bool)
        edge_margin_y = int(0.15 * h)
        edge_margin_x = int(0.15 * w)
        edge_mask[:edge_margin_y, :] = True
        edge_mask[-edge_margin_y:, :] = True
        edge_mask[:, :edge_margin_x] = True
        edge_mask[:, -edge_margin_x:] = True

        edge_ratio = np.sum(high_act & edge_mask) / max(np.sum(high_act), 1)

        if edge_ratio > 0.45:
            return f"Facial blending margin and {vert_sector} ({horiz_sector})"
        return f"{vert_sector} ({horiz_sector})"

    @staticmethod
    def _to_base64_jpeg(img_rgb: np.ndarray, quality: int = 90) -> str:
        """Encode an RGB uint8 image array to base64 JPEG string."""
        pil_img = Image.fromarray(img_rgb)
        buf = io.BytesIO()
        pil_img.save(buf, format="JPEG", quality=quality)
        return base64.b64encode(buf.getvalue()).decode("utf-8")
