"""
About Project
=============
Project objectives, existing system limitations, proposed system,
methodology and expected outcomes.
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
    render_section_header,
    render_info_banner,
    render_footer,
)

# ---- Page Config ----
setup_page("About Project", icon="📖")

# ================================================================
# HERO
# ================================================================
render_hero(
    "📖 About the Project",
    "Robust Deepfake Detection Using Frequency Domain Analysis to Enhance Public Safety",
)

# ================================================================
# PROJECT OBJECTIVES
# ================================================================
render_section_header("🎯 Project Objectives")

objectives = [
    ("1️⃣", "Develop a deepfake detection system that analyses both spatial and frequency domain features."),
    ("2️⃣", "Implement FFT and DWT-based frequency feature extraction to capture manipulation artifacts invisible in pixel space."),
    ("3️⃣", "Design a CNN-based spatial feature extractor to detect blending boundaries and texture inconsistencies."),
    ("4️⃣", "Fuse CNN, FFT and DWT features into a unified multi-domain representation for robust classification."),
    ("5️⃣", "Support both image-level and video-level deepfake detection with confidence scoring."),
    ("6️⃣", "Provide frequency-domain visual evidence (spectrum plots, wavelet sub-bands) for explainability."),
    ("7️⃣", "Evaluate robustness under real-world degradations — JPEG compression, Gaussian blur, additive noise and resizing."),
]
for icon, text in objectives:
    render_card(
        f'<span style="font-size:1.3rem; margin-right:0.6rem;">{icon}</span>'
        f'<span style="color:#94A3B8;">{text}</span>',
        accent=True,
    )

st.markdown("")

# ================================================================
# EXISTING SYSTEM LIMITATIONS
# ================================================================
render_section_header("🔻 Existing System Limitations")

render_card(
    """
    <table style="width:100%; border-collapse:collapse;">
    <thead>
    <tr style="border-bottom:1px solid #1E3050;">
        <th style="text-align:left; padding:0.6rem; color:#E8EDF5;">Limitation</th>
        <th style="text-align:left; padding:0.6rem; color:#E8EDF5;">Impact</th>
    </tr>
    </thead>
    <tbody style="color:#94A3B8; font-size:0.9rem;">
    <tr style="border-bottom:1px solid #1E3050;">
        <td style="padding:0.6rem;">Rely solely on spatial / pixel-level features</td>
        <td style="padding:0.6rem;">Miss frequency-domain artifacts left by generative models</td>
    </tr>
    <tr style="border-bottom:1px solid #1E3050;">
        <td style="padding:0.6rem;">Poor generalisation across datasets and generators</td>
        <td style="padding:0.6rem;">High accuracy on training set but fail on unseen data</td>
    </tr>
    <tr style="border-bottom:1px solid #1E3050;">
        <td style="padding:0.6rem;">No robustness to post-processing</td>
        <td style="padding:0.6rem;">Simple compression or blurring degrades detection accuracy</td>
    </tr>
    <tr style="border-bottom:1px solid #1E3050;">
        <td style="padding:0.6rem;">Lack of explainability</td>
        <td style="padding:0.6rem;">Users cannot understand why a prediction was made</td>
    </tr>
    <tr>
        <td style="padding:0.6rem;">No multi-domain feature fusion</td>
        <td style="padding:0.6rem;">Single-domain detectors have an inherent accuracy ceiling</td>
    </tr>
    </tbody>
    </table>
    """
)

st.markdown("")

# ================================================================
# PROPOSED SYSTEM
# ================================================================
render_section_header("✅ Proposed System")

render_card(
    """
    <p style="color:#94A3B8; line-height:1.75; margin:0;">
    Our proposed system addresses each limitation above through a
    <strong style="color:#48CAE4;">multi-domain feature fusion architecture</strong>:
    </p>
    <ul style="color:#94A3B8; line-height:2; margin-top:0.8rem;">
        <li><strong style="color:#E8EDF5;">CNN branch</strong> — Extracts spatial features
            (edges, textures, blending boundaries) using a convolutional backbone.</li>
        <li><strong style="color:#E8EDF5;">FFT branch</strong> — Transforms the face crop
            into the frequency domain and extracts spectral features that reveal
            generator fingerprints.</li>
        <li><strong style="color:#E8EDF5;">DWT branch</strong> — Decomposes the image into
            wavelet sub-bands to capture multi-resolution detail coefficients.</li>
        <li><strong style="color:#E8EDF5;">Feature Fusion</strong> — Concatenates the three
            feature vectors and passes them through fully-connected layers for
            binary classification (REAL / DEEPFAKE) with a confidence score.</li>
        <li><strong style="color:#E8EDF5;">Visual Evidence</strong> — Displays FFT magnitude
            spectra and DWT sub-band heatmaps so users can understand the
            decision.</li>
        <li><strong style="color:#E8EDF5;">Robustness Suite</strong> — Automated testing
            under JPEG compression, Gaussian blur, additive noise and resizing
            to validate real-world reliability.</li>
    </ul>
    """
)

st.markdown("")

# ================================================================
# PROJECT METHODOLOGY
# ================================================================
render_section_header("📐 Project Methodology")

phases = [
    ("Phase 0", "Project scaffold, folder structure and environment setup.", "✅"),
    ("Phase 1", "Streamlit frontend — all pages, navigation and UI components.", "✅"),
    ("Phase 2", "Face extraction and image/video preprocessing pipeline.", "⬜"),
    ("Phase 3", "FFT and DWT frequency-domain feature extraction modules.", "⬜"),
    ("Phase 4", "CNN spatial feature extractor and multi-domain fusion model.", "⬜"),
    ("Phase 5", "Model training, evaluation and checkpoint management.", "⬜"),
    ("Phase 6", "Integration — connect backend to Streamlit frontend.", "⬜"),
    ("Phase 7", "Robustness testing, dashboard analytics and final polish.", "⬜"),
]

render_card(
    """
    <table style="width:100%; border-collapse:collapse;">
    <thead>
    <tr style="border-bottom:1px solid #1E3050;">
        <th style="text-align:left; padding:0.6rem; color:#E8EDF5; width:15%;">Phase</th>
        <th style="text-align:left; padding:0.6rem; color:#E8EDF5;">Description</th>
        <th style="text-align:center; padding:0.6rem; color:#E8EDF5; width:10%;">Status</th>
    </tr>
    </thead>
    <tbody style="color:#94A3B8; font-size:0.9rem;">
    """
    + "".join(
        f'<tr style="border-bottom:1px solid #1E3050;">'
        f'<td style="padding:0.6rem; color:#48CAE4; font-weight:600;">{p}</td>'
        f'<td style="padding:0.6rem;">{d}</td>'
        f'<td style="padding:0.6rem; text-align:center; font-size:1.1rem;">{s}</td>'
        f"</tr>"
        for p, d, s in phases
    )
    + """
    </tbody>
    </table>
    """
)

st.markdown("")

# ================================================================
# EXPECTED OUTCOMES
# ================================================================
render_section_header("🏆 Expected Outcomes")

col1, col2 = st.columns(2)

with col1:
    render_card(
        """
        <h4 style="color:#48CAE4; margin-top:0;">Technical Outcomes</h4>
        <ul style="color:#94A3B8; line-height:2; margin:0;">
            <li>High detection accuracy through multi-domain feature fusion</li>
            <li>Resilience to common post-processing operations</li>
            <li>Frame-level video analysis capability</li>
            <li>Interpretable frequency-domain visual evidence</li>
        </ul>
        """
    )

with col2:
    render_card(
        """
        <h4 style="color:#48CAE4; margin-top:0;">Practical Outcomes</h4>
        <ul style="color:#94A3B8; line-height:2; margin:0;">
            <li>User-friendly web application for non-technical users</li>
            <li>Confidence scoring for every prediction</li>
            <li>Dashboard for tracking analysis history</li>
            <li>Modular architecture for future extension</li>
        </ul>
        """
    )

render_info_banner(
    "This project is developed as an academic mini project to demonstrate "
    "the effectiveness of frequency-domain analysis in deepfake detection."
)

# ================================================================
# FOOTER
# ================================================================
render_footer()
