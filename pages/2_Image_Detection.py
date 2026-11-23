"""
Image Detection
===============
Upload an image for CNN-based spatial deepfake analysis.
Uses pretrained ResNet18 spatial feature extraction and classification.
"""

import io
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
from PIL import Image
import streamlit as st

from src.detector import SpatialDetector
from src.ui_helpers import (
    render_card,
    render_footer,
    render_hero,
    render_info_banner,
    render_placeholder,
    render_section_header,
    setup_page,
)

# ---- Page Config ----
setup_page("Image Detection", icon="🖼️")

# ================================================================
# HERO & STATUS
# ================================================================
render_hero(
    "🖼️ Image Detection",
    "Analyze images for deepfake manipulation using CNN spatial feature extraction",
)

# Initialize detector
detector = SpatialDetector()

if detector.is_ready:
    render_info_banner(
        f"Spatial ResNet18 model is active and running on device: <strong>{detector.device}</strong>"
    )
else:
    render_info_banner(
        "Spatial ResNet18 model checkpoint not yet trained. Upload an image to preview "
        "and test the pipeline. Run <code>python scripts/train_model.py</code> to train weights."
    )
st.markdown("")

# ================================================================
# UPLOAD SECTION
# ================================================================
render_section_header("📤 Upload Image")

uploaded_file = st.file_uploader(
    "Choose an image file",
    type=["jpg", "jpeg", "png"],
    help="Supported formats: JPG, JPEG, PNG",
)

if uploaded_file is not None:
    # Read bytes once
    image_bytes = uploaded_file.getvalue()
    pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    # ---- Preview ----
    col_img, col_info = st.columns([1.2, 1])

    with col_img:
        render_section_header("🔎 Image Preview")
        st.image(pil_image, use_container_width=True)

    with col_info:
        render_section_header("📄 File Details")
        file_size_kb = len(image_bytes) / 1024
        w, h = pil_image.size
        render_card(
            f"""
            <table style="width:100%; border-collapse:collapse; color:#94A3B8; font-size:0.9rem;">
            <tr style="border-bottom:1px solid #1E3050;">
                <td style="padding:0.5rem; font-weight:600; color:#E8EDF5;">File Name</td>
                <td style="padding:0.5rem;">{uploaded_file.name}</td>
            </tr>
            <tr style="border-bottom:1px solid #1E3050;">
                <td style="padding:0.5rem; font-weight:600; color:#E8EDF5;">Dimensions</td>
                <td style="padding:0.5rem;">{w} × {h} px</td>
            </tr>
            <tr style="border-bottom:1px solid #1E3050;">
                <td style="padding:0.5rem; font-weight:600; color:#E8EDF5;">File Size</td>
                <td style="padding:0.5rem;">{file_size_kb:.1f} KB</td>
            </tr>
            <tr>
                <td style="padding:0.5rem; font-weight:600; color:#E8EDF5;">Model Status</td>
                <td style="padding:0.5rem;">{"✅ Ready (ResNet18)" if detector.is_ready else "⏳ Checkpoint Awaited"}</td>
            </tr>
            </table>
            """
        )

        st.markdown("")

        # ---- Analyze Button ----
        analyze_clicked = st.button(
            "🔬  Analyze Image",
            use_container_width=True,
            type="primary",
        )

    st.markdown("")
    st.divider()

    # ---- Prediction Result Area ----
    render_section_header("📊 Detection Results")

    if analyze_clicked:
        if not detector.is_ready:
            st.warning(
                "⚠️ **Spatial model checkpoint not found at `models/checkpoints/resnet18_best.pth`**.\n\n"
                "To train the model on FaceForensics++ datasets:\n"
                "1. Place raw videos into `data/raw/faceforensics/`\n"
                "2. Extract frames: `python scripts/extract_frames.py`\n"
                "3. Crop faces and split: `python scripts/prepare_dataset.py`\n"
                "4. Train the spatial ResNet18: `python scripts/train_model.py`",
                icon="🚧",
            )
        else:
            with st.spinner("Analyzing image with Spatial ResNet18..."):
                res = detector.predict_image(pil_image)

            if res.get("error"):
                st.error(f"Inference error: {res['error']}")
            else:
                pred_label = res["prediction"]
                conf = res["confidence"] * 100
                real_p = res["real_prob"] * 100
                fake_p = res["deepfake_prob"] * 100

                # Metric Cards
                res_col1, res_col2 = st.columns(2)
                with res_col1:
                    is_real = pred_label == "REAL"
                    border_color = "#2ECC71" if is_real else "#E74C3C"
                    badge_color = "#2ECC71" if is_real else "#E74C3C"
                    icon = "✅" if is_real else "⚠️"

                    render_card(
                        f"""
                        <div style="text-align:center; padding:0.5rem 0;">
                            <span style="font-size:2rem;">{icon}</span>
                            <div style="font-size:1.8rem; font-weight:700; color:{badge_color}; margin:0.3rem 0;">
                                {pred_label}
                            </div>
                            <div style="font-size:0.85rem; color:#94A3B8;">Spatial CNN Classification</div>
                        </div>
                        """
                    )

                with res_col2:
                    render_card(
                        f"""
                        <div style="text-align:center; padding:0.5rem 0;">
                            <span style="font-size:2rem;">📈</span>
                            <div style="font-size:1.8rem; font-weight:700; color:#48CAE4; margin:0.3rem 0;">
                                {conf:.1f}%
                            </div>
                            <div style="font-size:0.85rem; color:#94A3B8;">Prediction Confidence</div>
                        </div>
                        """
                    )

                st.markdown("")

                # Probability Distribution
                render_section_header("🎯 Class Probability Breakdown")
                pcol1, pcol2 = st.columns(2)
                with pcol1:
                    st.write(f"**REAL:** {real_p:.1f}%")
                    st.progress(float(res["real_prob"]))
                with pcol2:
                    st.write(f"**DEEPFAKE:** {fake_p:.1f}%")
                    st.progress(float(res["deepfake_prob"]))

                st.markdown("")

                # Face extraction visual evidence
                if res.get("cropped_face") is not None:
                    render_section_header("👤 Extracted Face Region (Model Input)")
                    st.image(res["cropped_face"], caption="Padded Face Crop (224×224 input)", width=224)
                elif not res.get("face_detected"):
                    st.info("ℹ️ No distinct face bounding box was detected; analyzed full image context.")

    else:
        # Default placeholder when analyze hasn't been clicked yet
        res_col1, res_col2 = st.columns(2)
        with res_col1:
            render_placeholder(
                "🏷️",
                "Prediction result (REAL / DEEPFAKE) will appear here after clicking Analyze.",
            )
        with res_col2:
            render_placeholder(
                "📈",
                "Confidence score and probabilities will appear here after clicking Analyze.",
            )

    st.markdown("")

    # ---- Frequency Evidence Placeholder (Phase 4 / 5) ----
    render_section_header("🔬 Frequency Domain Evidence (Planned for Phase 4)")

    ev_col1, ev_col2 = st.columns(2)
    with ev_col1:
        render_placeholder(
            "📊",
            "FFT magnitude spectrum visualization (To be integrated in Phase 4).",
        )
    with ev_col2:
        render_placeholder(
            "🌊",
            "DWT wavelet sub-band visualization (To be integrated in Phase 4).",
        )

else:
    # ---- No file uploaded ----
    st.markdown("")
    render_placeholder(
        "📁",
        "Upload a JPG, JPEG or PNG image above to get started.",
    )

# ================================================================
# FOOTER
# ================================================================
render_footer()
