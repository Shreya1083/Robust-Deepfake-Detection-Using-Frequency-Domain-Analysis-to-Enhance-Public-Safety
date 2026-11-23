"""
Image Detection
===============
Upload an image for deepfake analysis.
Detection logic will be connected in a later phase.
"""

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import streamlit as st
from src.ui_helpers import (
    setup_page,
    render_hero,
    render_card,
    render_placeholder,
    render_section_header,
    render_info_banner,
    render_footer,
)

# ---- Page Config ----
setup_page("Image Detection", icon="🖼️")

# ================================================================
# HERO
# ================================================================
render_hero(
    "🖼️ Image Detection",
    "Upload an image to analyse it for deepfake manipulation",
)

render_info_banner(
    "Detection model is not connected yet.  You can upload and preview "
    "images — analysis will be enabled in a future phase."
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
    # ---- Preview ----
    col_img, col_info = st.columns([1.2, 1])

    with col_img:
        render_section_header("🔎 Image Preview")
        st.image(uploaded_file, use_container_width=True)

    with col_info:
        render_section_header("📄 File Details")
        file_size_kb = len(uploaded_file.getvalue()) / 1024
        render_card(
            f"""
            <table style="width:100%; border-collapse:collapse; color:#94A3B8; font-size:0.9rem;">
            <tr style="border-bottom:1px solid #1E3050;">
                <td style="padding:0.5rem; font-weight:600; color:#E8EDF5;">File Name</td>
                <td style="padding:0.5rem;">{uploaded_file.name}</td>
            </tr>
            <tr style="border-bottom:1px solid #1E3050;">
                <td style="padding:0.5rem; font-weight:600; color:#E8EDF5;">File Type</td>
                <td style="padding:0.5rem;">{uploaded_file.type}</td>
            </tr>
            <tr>
                <td style="padding:0.5rem; font-weight:600; color:#E8EDF5;">File Size</td>
                <td style="padding:0.5rem;">{file_size_kb:.1f} KB</td>
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

        if analyze_clicked:
            st.warning(
                "⚙️ Detection model is not connected yet. "
                "This feature will be available after model integration.",
                icon="🚧",
            )

    st.markdown("")
    st.divider()

    # ---- Prediction Result Placeholder ----
    render_section_header("📊 Detection Results")

    res_col1, res_col2 = st.columns(2)

    with res_col1:
        render_placeholder(
            "🏷️",
            "Prediction result (REAL / DEEPFAKE) will appear here after analysis.",
        )

    with res_col2:
        render_placeholder(
            "📈",
            "Confidence score will appear here after analysis.",
        )

    st.markdown("")

    # ---- Frequency Evidence Placeholder ----
    render_section_header("🔬 Frequency Domain Evidence")

    ev_col1, ev_col2 = st.columns(2)

    with ev_col1:
        render_placeholder(
            "📊",
            "FFT magnitude spectrum visualization will appear here.",
        )

    with ev_col2:
        render_placeholder(
            "🌊",
            "DWT wavelet sub-band visualization will appear here.",
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
