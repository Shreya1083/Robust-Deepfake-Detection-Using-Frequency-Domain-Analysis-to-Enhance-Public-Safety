"""
Robust Deepfake Detection — Home Page
======================================
Main Streamlit entry point.
"""

import sys
from pathlib import Path

# Ensure project root is in sys.path to guarantee clean imports across environments
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import streamlit as st
from src.config import PROJECT_SHORT_TITLE, PROJECT_SUBTITLE
from src.ui_helpers import (
    setup_page,
    render_hero,
    render_card,
    render_nav_card,
    render_section_header,
    render_footer,
)

# ---- Page Configuration & Sidebar Setup ----
setup_page("Home", icon="🔍")


def main():
    """Main application content."""
    # ================================================================
    # HERO
    # ================================================================
    render_hero(
        f"🔍 {PROJECT_SHORT_TITLE}",
        PROJECT_SUBTITLE,
    )

    # ================================================================
    # PROBLEM STATEMENT
    # ================================================================
    render_section_header("⚠️ Problem Statement")
    render_card(
        """
        <p style="color:#94A3B8; line-height:1.75; margin:0;">
        Deepfake technology uses advanced generative models to create hyper-realistic
        manipulated media — swapped faces, synthesised voices and fabricated videos.
        These manipulations are increasingly being weaponised for misinformation,
        identity fraud and political destabilisation. Existing detectors that rely
        solely on spatial (pixel-level) analysis struggle against modern deepfake
        generators that leave <strong>minimal visible artifacts</strong>, yet
        consistently introduce <strong>detectable anomalies in the frequency
        domain</strong>.
        </p>
        """,
        accent=True,
    )

    # ================================================================
    # WHY DEEPFAKE DETECTION MATTERS
    # ================================================================
    render_section_header("🛡️ Why Deepfake Detection Is Important")

    col1, col2, col3 = st.columns(3)

    with col1:
        render_card(
            """
            <div style="text-align:center;">
                <span style="font-size:2rem;">🏛️</span>
                <h4 style="color:#E8EDF5; margin:0.5rem 0 0.3rem;">Public Safety</h4>
                <p style="color:#94A3B8; font-size:0.85rem; margin:0;">
                Fake videos can incite violence, manipulate elections and erode
                public trust in institutions.
                </p>
            </div>
            """
        )
    with col2:
        render_card(
            """
            <div style="text-align:center;">
                <span style="font-size:2rem;">🔐</span>
                <h4 style="color:#E8EDF5; margin:0.5rem 0 0.3rem;">Identity Protection</h4>
                <p style="color:#94A3B8; font-size:0.85rem; margin:0;">
                Deepfakes enable identity theft and fraud, putting individuals'
                reputations and finances at risk.
                </p>
            </div>
            """
        )
    with col3:
        render_card(
            """
            <div style="text-align:center;">
                <span style="font-size:2rem;">📰</span>
                <h4 style="color:#E8EDF5; margin:0.5rem 0 0.3rem;">Media Integrity</h4>
                <p style="color:#94A3B8; font-size:0.85rem; margin:0;">
                Reliable detection preserves the credibility of journalism,
                evidence and digital communications.
                </p>
            </div>
            """
        )

    # ================================================================
    # OUR DETECTION APPROACH
    # ================================================================
    render_section_header("🧠 Our Detection Approach")
    st.markdown("")

    render_card(
        """
        <p style="color:#94A3B8; line-height:1.75; margin:0;">
        Our system fuses <strong style="color:#48CAE4;">three complementary
        feature domains</strong> to achieve robust detection:<br><br>
        <strong style="color:#E8EDF5;">1. CNN Spatial Features</strong> —
        A convolutional neural network extracts pixel-level manipulation
        artifacts such as blending boundaries, texture inconsistencies and
        unnatural lighting.<br><br>
        <strong style="color:#E8EDF5;">2. FFT Frequency Features</strong> —
        The Fast Fourier Transform converts the image into the frequency
        domain, exposing periodic artifacts and spectral anomalies
        introduced by generative models.<br><br>
        <strong style="color:#E8EDF5;">3. DWT Wavelet Features</strong> —
        The Discrete Wavelet Transform decomposes the image into
        multi-resolution sub-bands, capturing both spatial and
        frequency information simultaneously.<br><br>
        These three feature streams are <strong style="color:#48CAE4;">fused
        </strong> into a unified representation and fed to a classifier that
        outputs a <em>REAL / DEEPFAKE</em> prediction with a confidence score.
        </p>
        """
    )

    # ================================================================
    # CNN · FFT · DWT EXPLAINERS
    # ================================================================
    render_section_header("📚 Core Techniques Explained")

    tab_cnn, tab_fft, tab_dwt = st.tabs(["🧠 CNN", "📊 FFT", "🌊 DWT"])

    with tab_cnn:
        render_card(
            """
            <h4 style="color:#48CAE4; margin-top:0;">Convolutional Neural Network (CNN)</h4>
            <p style="color:#94A3B8; line-height:1.7; margin:0;">
            A CNN is a deep-learning architecture designed to automatically learn
            hierarchical spatial features from images. Layers of learnable
            convolutional filters detect edges, textures and high-level patterns.
            In deepfake detection the CNN captures subtle spatial artifacts —
            blending seams, inconsistent skin texture and distorted facial
            geometry — that are difficult for the human eye to notice.
            </p>
            """
        )

    with tab_fft:
        render_card(
            """
            <h4 style="color:#48CAE4; margin-top:0;">Fast Fourier Transform (FFT)</h4>
            <p style="color:#94A3B8; line-height:1.7; margin:0;">
            The FFT converts an image from the spatial domain into the frequency
            domain, representing it as a sum of sinusoidal components at different
            frequencies and orientations. Deepfake generators often leave
            characteristic "fingerprints" in the frequency spectrum — unusual
            peaks, missing high-frequency detail or periodic grid-like patterns —
            that are invisible in pixel space but clearly visible in the
            magnitude spectrum.
            </p>
            """
        )

    with tab_dwt:
        render_card(
            """
            <h4 style="color:#48CAE4; margin-top:0;">Discrete Wavelet Transform (DWT)</h4>
            <p style="color:#94A3B8; line-height:1.7; margin:0;">
            The DWT decomposes an image into approximation and detail sub-bands
            (horizontal, vertical, diagonal) at multiple resolution levels.
            Unlike FFT, it preserves <em>both</em> spatial and frequency
            information, making it effective at localising where in the image
            manipulation artifacts occur. Deepfakes tend to show anomalous
            energy distributions in the high-frequency detail sub-bands.
            </p>
            """
        )

    # ================================================================
    # NAVIGATION CARDS
    # ================================================================
    st.markdown("")
    render_section_header("🚀 Get Started")
    st.markdown("")

    nav1, nav2 = st.columns(2)

    with nav1:
        render_nav_card(
            "🖼️",
            "Image Detection",
            "Upload an image to analyze it for deepfake manipulation.",
        )
        st.page_link(
            "pages/2_Image_Detection.py",
            label="Open Image Detection →",
            use_container_width=True,
        )

    with nav2:
        render_nav_card(
            "🎥",
            "Video Detection",
            "Upload a video for frame-by-frame deepfake analysis.",
        )
        st.page_link(
            "pages/3_Video_Detection.py",
            label="Open Video Detection →",
            use_container_width=True,
        )

    # ================================================================
    # FOOTER
    # ================================================================
    render_footer()


# Execute UI
main()

# Standalone execution handler:
# If user runs `python app.py` directly, auto-launch through streamlit runner
if __name__ == "__main__":
    import streamlit.runtime as rt

    if not rt.exists():
        import subprocess

        print("Launching Streamlit server for app.py...")
        subprocess.run(
            [sys.executable, "-m", "streamlit", "run", str(Path(__file__).resolve())]
        )
