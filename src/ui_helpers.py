"""
Reusable UI Components
======================
Helper functions that render common widgets used across pages.
Every page can call these instead of duplicating HTML / markdown.
"""

import streamlit as st


# ============================================================
# Page Configuration Helper
# ============================================================

def setup_page(title: str, icon: str = "🔍", layout: str = "wide"):
    """Configure page metadata and inject global styles."""
    st.set_page_config(
        page_title=f"{title} | Deepfake Detection",
        page_icon=icon,
        layout=layout,
        initial_sidebar_state="expanded",
    )
    # Inject global CSS
    from src.styles import inject_global_styles
    inject_global_styles()
    # Render sidebar
    _render_sidebar()


# ============================================================
# Sidebar
# ============================================================

def _render_sidebar():
    """Render the shared sidebar present on every page."""
    with st.sidebar:
        st.markdown(
            """
            <div style="text-align:center; padding: 0.5rem 0 1rem 0;">
                <span style="font-size:2rem;">🔍</span>
                <h3 style="margin:0.2rem 0 0 0; font-size:1rem; color:#48CAE4;">
                    Deepfake Detection
                </h3>
                <p style="font-size:0.72rem; color:#64748B; margin-top:0.15rem;">
                    Frequency Domain Analysis
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.divider()

        st.caption("NAVIGATION")
        st.page_link("app.py", label="🏠  Home", use_container_width=True)
        st.page_link("pages/1_About_Project.py", label="📖  About Project", use_container_width=True)
        st.page_link("pages/2_Image_Detection.py", label="🖼️  Image Detection", use_container_width=True)
        st.page_link("pages/3_Video_Detection.py", label="🎥  Video Detection", use_container_width=True)
        st.page_link("pages/4_Frequency_Analysis.py", label="📊  Frequency Analysis", use_container_width=True)
        st.page_link("pages/5_Dashboard.py", label="📈  Dashboard", use_container_width=True)

        st.divider()
        st.caption("TECHNOLOGY STACK")
        st.markdown(
            """
            <div style="font-size:0.78rem; color:#94A3B8; line-height:1.7;">
            ⚡ PyTorch &nbsp;•&nbsp; Streamlit<br>
            👁️ OpenCV &nbsp;•&nbsp; NumPy<br>
            🌊 PyWavelets &nbsp;•&nbsp; SciPy<br>
            📊 Matplotlib &nbsp;•&nbsp; Scikit-learn
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.divider()
        st.markdown(
            '<p style="text-align:center; font-size:0.7rem; color:#475569;">'
            "🎓 College Mini Project</p>",
            unsafe_allow_html=True,
        )


# ============================================================
# Hero Banner
# ============================================================

def render_hero(title: str, subtitle: str):
    """Render a full-width hero section at the top of a page."""
    st.markdown(
        f"""
        <div class="page-hero">
            <h1>{title}</h1>
            <p>{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# Custom Cards
# ============================================================

def render_card(content_html: str, accent: bool = False):
    """Render a styled card.  Set *accent=True* for a left-border card."""
    css_class = "accent-card" if accent else "custom-card"
    st.markdown(
        f'<div class="{css_class}">{content_html}</div>',
        unsafe_allow_html=True,
    )


def render_stat_card(icon: str, value: str, label: str):
    """Render a single statistic card with icon, big value & label."""
    st.markdown(
        f"""
        <div class="stat-card">
            <div class="stat-icon">{icon}</div>
            <div class="stat-value">{value}</div>
            <div class="stat-label">{label}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_nav_card(icon: str, title: str, description: str):
    """Render a navigation card for the home page."""
    st.markdown(
        f"""
        <div class="nav-card">
            <div class="nav-card-icon">{icon}</div>
            <div class="nav-card-title">{title}</div>
            <div class="nav-card-desc">{description}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# Placeholder Box
# ============================================================

def render_placeholder(icon: str, message: str):
    """Render a dashed placeholder box for features not yet implemented."""
    st.markdown(
        f"""
        <div class="placeholder-box">
            <div class="placeholder-icon">{icon}</div>
            <div class="placeholder-text">{message}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# Section Header
# ============================================================

def render_section_header(text: str):
    """Render a styled section header with accent underline."""
    st.markdown(
        f'<div class="section-header">{text}</div>',
        unsafe_allow_html=True,
    )


# ============================================================
# Info Banner
# ============================================================

def render_info_banner(message: str):
    """Render a subtle information banner."""
    st.markdown(
        f'<div class="info-banner">ℹ️ &nbsp;{message}</div>',
        unsafe_allow_html=True,
    )


# ============================================================
# Footer
# ============================================================

def render_footer():
    """Render a consistent page footer."""
    st.divider()
    st.markdown(
        '<p style="text-align:center; font-size:0.75rem; color:#475569;">'
        "🎓 Robust Deepfake Detection Using Frequency Domain Analysis "
        "to Enhance Public Safety — College Mini Project</p>",
        unsafe_allow_html=True,
    )
