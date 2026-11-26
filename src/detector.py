"""
Unified Deepfake Detector Module (Spatial CNN + Grad-CAM + FFT + DWT)
=====================================================================
Modular deepfake detection engine featuring:
1. Spatial ResNet18 CNN inference with calibrated probabilities.
2. Actual Grad-CAM spatial activation heatmaps & overlays from layer4.
3. 2D Fast Fourier Transform (FFT) magnitude spectrum and radial profile.
4. Discrete Wavelet Transform (DWT) 2D Haar multi-scale sub-band analysis.
5. Unified analytical summary synthesizing all domain findings.
"""

import logging
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import cv2
import numpy as np
from PIL import Image
import torch

from models.spatial_model import SpatialResNet18, get_default_device
from src.config import CHECKPOINTS_DIR, CLASS_LABELS
from src.dataset_loader import get_spatial_transforms
from src.face_extraction import FaceExtractor
from src.frequency import run_fft_analysis
from src.gradcam import GradCAM
from src.wavelet import run_dwt_analysis

logger = logging.getLogger(__name__)


class SpatialDetector:
    """
    Forensic deepfake detector combining ResNet18 spatial analysis with
    Grad-CAM heatmaps, 2D FFT spectral decomposition, and 2D DWT wavelets.
    """

    def __init__(
        self,
        checkpoint_path: Optional[Union[str, Path]] = None,
        device: Optional[torch.device] = None,
    ):
        self.device = device or get_default_device()
        self.checkpoint_path = Path(checkpoint_path or (CHECKPOINTS_DIR / "resnet18_best.pth"))

        self.face_extractor = FaceExtractor()
        self.transform = get_spatial_transforms(is_train=False, input_size=(224, 224))

        self.model: Optional[SpatialResNet18] = None
        self.gradcam: Optional[GradCAM] = None
        self.checkpoint_meta: Optional[Dict] = None
        self.model_type: str = "spatial_resnet18"
        self.is_ready: bool = False

        self._load_model()

    def _load_model(self) -> bool:
        """Load trained ResNet18 weights if available; fallback to ImageNet pretrained."""
        if self.checkpoint_path.exists():
            try:
                ckpt = torch.load(self.checkpoint_path, map_location=self.device)
                self.model = SpatialResNet18(
                    pretrained=False,
                    num_classes=ckpt.get("num_classes", 2),
                )
                self.model.load_state_dict(ckpt["model_state_dict"])
                self.model.to(self.device)
                self.model.eval()
                self.checkpoint_meta = ckpt
                self.model_type = "spatial_resnet18_trained"
                self.is_ready = True
                logger.info("Loaded trained CNN checkpoint from: %s", self.checkpoint_path)
            except Exception as e:
                logger.warning("Could not load checkpoint %s: %s. Using pretrained backbone.", self.checkpoint_path, e)
                self.model = SpatialResNet18(pretrained=True).to(self.device)
                self.model.eval()
                self.model_type = "spatial_resnet18_pretrained"
                self.is_ready = True
        else:
            try:
                self.model = SpatialResNet18(pretrained=True).to(self.device)
                self.model.eval()
                self.model_type = "spatial_resnet18_pretrained"
                self.is_ready = True
                logger.info("No checkpoint at %s; running with pretrained ResNet18 backbone.", self.checkpoint_path)
            except Exception:
                self.model = SpatialResNet18(pretrained=False).to(self.device)
                self.model.eval()
                self.model_type = "spatial_resnet18_baseline"
                self.is_ready = True

        # Initialize Grad-CAM on layer4
        try:
            self.gradcam = GradCAM(self.model)
            logger.info("Initialized Grad-CAM on target layer: %s", self.gradcam.target_layer)
        except Exception as e:
            logger.error("Failed to initialize Grad-CAM: %s", e)
            self.gradcam = None

        return self.is_ready

    def predict_image(
        self,
        image_input: Union[str, Path, np.ndarray, Image.Image],
    ) -> Dict:
        """
        Run deepfake inference on an uploaded image, generating:
        - REAL vs DEEPFAKE verdict & calibrated class probabilities
        - Spatial Grad-CAM heatmap, overlay & salient region description
        - FFT frequency spectrum, multi-band energy & radial profile
        - DWT wavelet sub-bands & multi-scale detail ratios
        - Dynamic holistic analysis summary
        """
        if not self.is_ready or self.model is None:
            return {
                "is_ready": False,
                "error": "CNN model is not initialized.",
            }

        # 1. Standardize input to numpy BGR array
        np_frame: Optional[np.ndarray] = None
        if isinstance(image_input, (str, Path)):
            path = Path(image_input)
            if not path.exists():
                return {"is_ready": True, "error": f"File does not exist: {path}"}
            np_frame = cv2.imread(str(path))
        elif isinstance(image_input, Image.Image):
            rgb = np.array(image_input)
            np_frame = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        elif isinstance(image_input, np.ndarray):
            if image_input.size == 0:
                np_frame = None
            elif image_input.ndim == 2:
                np_frame = cv2.cvtColor(image_input, cv2.COLOR_GRAY2BGR)
            elif image_input.ndim == 3 and image_input.shape[2] == 4:
                np_frame = cv2.cvtColor(image_input, cv2.COLOR_BGRA2BGR)
            elif image_input.ndim == 3 and image_input.shape[2] == 3:
                np_frame = image_input

        if np_frame is None or np_frame.size == 0:
            return {"is_ready": True, "error": "Invalid or empty image provided."}

        # 2. Face Detection & Padded Cropping
        face_result = self.face_extractor.extract_face(np_frame)
        face_detected = face_result.success and face_result.face_image is not None

        if face_detected:
            face_bgr = face_result.face_image
            face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(face_rgb)
        else:
            face_rgb = cv2.cvtColor(np_frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(face_rgb)

        # 3. Spatial CNN Forward Pass
        tensor = self.transform(pil_image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            cnn_logits = self.model(tensor)
            cnn_probs = torch.softmax(cnn_logits, dim=1).squeeze(0).cpu().numpy()

        real_prob = float(cnn_probs[0])
        deepfake_prob = float(cnn_probs[1])

        # Use calibrated optimal threshold from validation set if available
        threshold = 0.50
        if self.checkpoint_meta and isinstance(self.checkpoint_meta, dict):
            threshold = float(self.checkpoint_meta.get("optimal_threshold", 0.50))

        if deepfake_prob >= threshold:
            pred_label = 1
            prediction = CLASS_LABELS.get(1, "DEEPFAKE")
            confidence = deepfake_prob
        else:
            pred_label = 0
            prediction = CLASS_LABELS.get(0, "REAL")
            confidence = real_prob

        # 4. Spatial Grad-CAM Generation
        spatial_analysis = {
            "status": "inactive",
            "architecture": "ResNet18",
            "target_layer": "layer4",
            "salient_regions": "N/A",
            "heatmap_b64": None,
            "overlay_b64": None,
            "explanation": "The CNN spatial analysis highlights facial regions that contributed most to the classification.",
        }

        if self.gradcam is not None:
            try:
                # Re-compute with gradient tracking enabled
                cam_tensor = self.transform(pil_image).unsqueeze(0).to(self.device)
                gradcam_out = self.gradcam.generate(
                    input_tensor=cam_tensor,
                    target_class=pred_label,
                    original_image_rgb=face_rgb,
                )
                spatial_analysis = {
                    "status": "active",
                    "architecture": "ResNet18",
                    "target_layer": gradcam_out["target_layer_name"],
                    "salient_regions": gradcam_out["salient_regions"],
                    "heatmap_b64": gradcam_out["heatmap_b64"],
                    "overlay_b64": gradcam_out["overlay_b64"],
                    "explanation": (
                        "The CNN spatial analysis highlights facial regions that contributed most to the classification. "
                        f"Primary model activations are localized on the {gradcam_out['salient_regions'].lower()}."
                    ),
                }
            except Exception as e:
                logger.error("Error computing Grad-CAM: %s", e)
                spatial_analysis["error"] = str(e)

        # 5. FFT Frequency Domain Analysis
        fft_analysis = {
            "status": "inactive",
            "magnitude_b64": None,
            "log_spectrum_b64": None,
            "metrics": None,
            "explanation": None,
        }
        try:
            fft_res = run_fft_analysis(pil_image, generate_visuals=True)
            if fft_res.success and fft_res.features:
                fft_analysis = {
                    "status": "active",
                    "magnitude_b64": fft_res.magnitude_b64,
                    "log_spectrum_b64": fft_res.log_spectrum_b64,
                    "metrics": fft_res.features.to_dict(),
                    "radial_profile": fft_res.features.radial_spectrum,
                    "explanation": fft_res.explanation,
                }
        except Exception as e:
            logger.error("Error during FFT analysis: %s", e)
            fft_analysis["error"] = str(e)

        # 6. DWT Discrete Wavelet Analysis
        dwt_analysis = {
            "status": "inactive",
            "subbands_b64": None,
            "combined_b64": None,
            "ll_b64": None,
            "lh_b64": None,
            "hl_b64": None,
            "hh_b64": None,
            "features": None,
            "wavelet_type": "haar",
            "explanation": None,
        }
        try:
            dwt_res = run_dwt_analysis(pil_image, generate_visuals=True)
            if dwt_res.success and dwt_res.features:
                dwt_feat = dwt_res.features
                dwt_explanation = (
                    f"2D Haar Wavelet decomposition isolates approximation (LL: {dwt_feat.ll.energy_ratio*100:.1f}%) "
                    f"and high-frequency directional subbands (LH: {dwt_feat.lh.energy_ratio*100:.2f}%, "
                    f"HL: {dwt_feat.hl.energy_ratio*100:.2f}%, HH: {dwt_feat.hh.energy_ratio*100:.2f}%). "
                    f"Detail-to-approximation ratio: {dwt_feat.detail_to_approx_ratio:.4f}."
                )
                dwt_analysis = {
                    "status": "active",
                    "subbands_b64": dwt_res.combined_b64,
                    "combined_b64": dwt_res.combined_b64,
                    "ll_b64": dwt_res.ll_b64,
                    "lh_b64": dwt_res.lh_b64,
                    "hl_b64": dwt_res.hl_b64,
                    "hh_b64": dwt_res.hh_b64,
                    "features": dwt_feat.to_dict(),
                    "wavelet_type": "haar",
                    "explanation": dwt_explanation,
                }
        except Exception as e:
            logger.error("Error during DWT analysis: %s", e)
            dwt_analysis["error"] = str(e)

        # 7. Holistic Combined Analysis Summary
        analysis_summary = self._generate_analysis_summary(
            prediction=prediction,
            confidence=confidence,
            spatial_analysis=spatial_analysis,
            fft_analysis=fft_analysis,
            dwt_analysis=dwt_analysis,
        )

        return {
            "is_ready": True,
            "model_type": self.model_type,
            "is_fusion": False,
            "prediction": prediction,
            "label": pred_label,
            "confidence": round(confidence, 4),
            "real_prob": round(real_prob, 4),
            "deepfake_prob": round(deepfake_prob, 4),
            "face_detected": face_detected,
            "cropped_face": face_rgb if face_detected else None,
            "spatial_analysis": spatial_analysis,
            "frequency_analysis": fft_analysis,
            "dwt_analysis": dwt_analysis,
            "analysis_summary": analysis_summary,
            "device": str(self.device),
            "error": None,
        }

    @staticmethod
    def _generate_analysis_summary(
        prediction: str,
        confidence: float,
        spatial_analysis: Dict,
        fft_analysis: Dict,
        dwt_analysis: Dict,
    ) -> str:
        """Create a cohesive, scientifically grounded combined analysis summary."""
        parts = []

        # Model determination
        conf_pct = confidence * 100
        parts.append(f"Model Determination: Classified as {prediction} with {conf_pct:.1f}% confidence.")

        # Spatial findings
        if spatial_analysis.get("status") == "active":
            salient = spatial_analysis.get("salient_regions", "facial contours")
            parts.append(
                f"Spatial Analysis (ResNet18): Feature attribution concentrated predominantly on the {salient.lower()}, "
                "identifying visual boundaries and textural gradients that influenced the classification."
            )

        # FFT findings
        fft_metrics = fft_analysis.get("metrics")
        if fft_metrics:
            high_e_pct = fft_metrics.get("high_energy_pct", 0.0)
            low_e_pct = fft_metrics.get("low_energy_pct", 0.0)
            parts.append(
                f"FFT Spectral Analysis: Exhibits {low_e_pct:.1f}% structural low-frequency energy and "
                f"{high_e_pct:.2f}% high-frequency components."
            )

        # DWT findings
        dwt_features = dwt_analysis.get("features")
        if dwt_features:
            hh_ratio = dwt_features.get("hh", {}).get("energy_ratio", 0.0) * 100
            parts.append(
                f"DWT Wavelet Decomposition: Diagonal detail subband (HH) contains {hh_ratio:.2f}% of total wavelet energy."
            )

        # Forensic conclusion
        if prediction == "DEEPFAKE":
            parts.append(
                "Conclusion: Elevated spatial saliency and localized frequency/wavelet cues are consistent with digital face manipulation artifacts."
            )
        else:
            parts.append(
                "Conclusion: Coherent spatial texture gradients and smooth spectral falloff are consistent with authentic photographic imagery."
            )

        return " ".join(parts)
