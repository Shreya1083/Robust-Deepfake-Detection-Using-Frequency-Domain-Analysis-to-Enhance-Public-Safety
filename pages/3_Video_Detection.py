"""
Video Detection
===============
Upload a video for frame-level deepfake analysis.
Processing logic will be connected in a later phase.
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
setup_page("Video Detection", icon="🎥")

# ================================================================
# HERO
# ================================================================
render_hero(
    "🎥 Video Detection",
    "Upload a video for frame-by-frame deepfake analysis",
)

render_info_banner(
    "Video processing is not connected yet.  You can upload and preview "
    "videos — analysis will be enabled in a future phase."
)
st.markdown("")

# ================================================================
# UPLOAD SECTION
# ================================================================
render_section_header("📤 Upload Video")

uploaded_video = st.file_uploader(
    "Choose a video file",
    type=["mp4", "avi", "mov"],
    help="Supported formats: MP4, AVI, MOV",
)

if uploaded_video is not None:
    # ---- Preview ----
    col_vid, col_info = st.columns([1.2, 1])

    with col_vid:
        render_section_header("🔎 Video Preview")
        st.video(uploaded_video)

    with col_info:
        render_section_header("📄 File Details")
        file_size_mb = len(uploaded_video.getvalue()) / (1024 * 1024)
        render_card(
            f"""
            <table style="width:100%; border-collapse:collapse; color:#94A3B8; font-size:0.9rem;">
            <tr style="border-bottom:1px solid #1E3050;">
                <td style="padding:0.5rem; font-weight:600; color:#E8EDF5;">File Name</td>
                <td style="padding:0.5rem;">{uploaded_video.name}</td>
            </tr>
            <tr style="border-bottom:1px solid #1E3050;">
                <td style="padding:0.5rem; font-weight:600; color:#E8EDF5;">File Type</td>
                <td style="padding:0.5rem;">{uploaded_video.type}</td>
            </tr>
            <tr>
                <td style="padding:0.5rem; font-weight:600; color:#E8EDF5;">File Size</td>
                <td style="padding:0.5rem;">{file_size_mb:.2f} MB</td>
            </tr>
            </table>
            """
        )

        st.markdown("")

        # ---- Analyze Button ----
        analyze_clicked = st.button(
            "🔬  Analyze Video",
            use_container_width=True,
            type="primary",
        )

        if analyze_clicked:
            st.warning(
                "⚙️ Video processing is not connected yet. "
                "This feature will be available after model integration.",
                icon="🚧",
            )

    st.markdown("")
    st.divider()

    # ---- Results Placeholder ----
    render_section_header("📊 Video Detection Results")

    r1, r2, r3 = st.columns(3)
    with r1:
        render_placeholder("🎞️", "Total frames analysed will appear here.")
    with r2:
        render_placeholder("🏷️", "Overall prediction (REAL / DEEPFAKE) will appear here.")
    with r3:
        render_placeholder("📈", "Aggregate confidence score will appear here.")

    st.markdown("")

    # ---- Frame-level Placeholder ----
    render_section_header("🖼️ Frame-Level Analysis")
    render_placeholder(
        "📋",
        "Per-frame detection results and timeline chart will appear here after processing.",
    )

else:
    # ---- No file uploaded ----
    st.markdown("")
    render_placeholder(
        "📁",
        "Upload an MP4, AVI or MOV video above to get started.",
    )

# ================================================================
# FOOTER
# ================================================================
render_footer()
