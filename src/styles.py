"""
Centralized CSS Styles
======================
All custom CSS for the Streamlit application lives here.
Import `inject_global_styles()` at the top of every page.
"""

# ============================================================
# Theme Palette
# ============================================================

COLORS = {
    "bg_primary": "#0B1120",
    "bg_secondary": "#111B2E",
    "bg_card": "#15213A",
    "bg_card_hover": "#1A2845",
    "accent": "#00B4D8",
    "accent_light": "#48CAE4",
    "accent_dark": "#0077B6",
    "success": "#2ECC71",
    "danger": "#E74C3C",
    "warning": "#F39C12",
    "text_primary": "#E8EDF5",
    "text_secondary": "#94A3B8",
    "text_muted": "#64748B",
    "border": "#1E3050",
    "divider": "#1E3A5F",
}


def _global_css() -> str:
    """Return the full global stylesheet as a single string."""
    c = COLORS
    return f"""
    <style>
    /* ---- Import Google Font ---- */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    /* ---- Root / Main Container ---- */
    .stApp {{
        background-color: {c["bg_primary"]};
        color: {c["text_primary"]};
        font-family: 'Inter', sans-serif;
    }}

    /* ---- Sidebar ---- */
    section[data-testid="stSidebar"] {{
        background-color: {c["bg_secondary"]};
        border-right: 1px solid {c["border"]};
    }}
    section[data-testid="stSidebar"] .stMarkdown p,
    section[data-testid="stSidebar"] .stMarkdown li,
    section[data-testid="stSidebar"] .stMarkdown span {{
        color: {c["text_secondary"]};
    }}

    /* ---- Headers ---- */
    h1, h2, h3, h4 {{
        color: {c["text_primary"]} !important;
        font-family: 'Inter', sans-serif !important;
    }}
    h1 {{ font-weight: 700 !important; }}
    h2 {{ font-weight: 600 !important; }}
    h3 {{ font-weight: 600 !important; }}

    /* ---- Paragraphs & captions ---- */
    p, li, span, label {{
        color: {c["text_secondary"]};
    }}
    .stCaption, [data-testid="stCaptionContainer"] {{
        color: {c["text_muted"]} !important;
    }}

    /* ---- Tabs ---- */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 8px;
    }}
    .stTabs [data-baseweb="tab"] {{
        background-color: {c["bg_card"]};
        border-radius: 8px;
        color: {c["text_secondary"]};
        border: 1px solid {c["border"]};
        padding: 8px 20px;
    }}
    .stTabs [aria-selected="true"] {{
        background-color: {c["accent_dark"]} !important;
        color: #FFFFFF !important;
        border-color: {c["accent"]} !important;
    }}

    /* ---- Buttons ---- */
    .stButton > button {{
        background: linear-gradient(135deg, {c["accent_dark"]}, {c["accent"]});
        color: #FFFFFF;
        border: none;
        border-radius: 8px;
        padding: 0.55rem 1.8rem;
        font-weight: 600;
        font-size: 0.95rem;
        transition: all 0.2s ease;
        font-family: 'Inter', sans-serif;
    }}
    .stButton > button:hover {{
        transform: translateY(-1px);
        box-shadow: 0 4px 15px rgba(0, 180, 216, 0.35);
    }}
    .stButton > button:active {{
        transform: translateY(0);
    }}

    /* ---- File Uploader ---- */
    [data-testid="stFileUploader"] {{
        background-color: {c["bg_card"]};
        border: 2px dashed {c["border"]};
        border-radius: 12px;
        padding: 1rem;
    }}
    [data-testid="stFileUploader"]:hover {{
        border-color: {c["accent"]};
    }}

    /* ---- Metrics ---- */
    [data-testid="stMetric"] {{
        background-color: {c["bg_card"]};
        border: 1px solid {c["border"]};
        border-radius: 12px;
        padding: 1rem 1.2rem;
    }}
    [data-testid="stMetricValue"] {{
        color: {c["accent_light"]} !important;
        font-weight: 700 !important;
    }}
    [data-testid="stMetricLabel"] {{
        color: {c["text_secondary"]} !important;
    }}

    /* ---- Expanders ---- */
    .streamlit-expanderHeader {{
        background-color: {c["bg_card"]};
        border-radius: 8px;
        color: {c["text_primary"]} !important;
        font-weight: 600;
    }}

    /* ---- Divider ---- */
    hr {{
        border-color: {c["divider"]} !important;
    }}

    /* ---- Custom card ---- */
    .custom-card {{
        background-color: {c["bg_card"]};
        border: 1px solid {c["border"]};
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        transition: border-color 0.2s ease;
    }}
    .custom-card:hover {{
        border-color: {c["accent"]};
    }}

    /* ---- Accent left border card ---- */
    .accent-card {{
        background-color: {c["bg_card"]};
        border-left: 4px solid {c["accent"]};
        border-radius: 0 12px 12px 0;
        padding: 1.2rem 1.5rem;
        margin-bottom: 1rem;
    }}

    /* ---- Stat card ---- */
    .stat-card {{
        background-color: {c["bg_card"]};
        border: 1px solid {c["border"]};
        border-radius: 14px;
        padding: 1.4rem;
        text-align: center;
        transition: border-color 0.2s ease, transform 0.2s ease;
    }}
    .stat-card:hover {{
        border-color: {c["accent"]};
        transform: translateY(-2px);
    }}
    .stat-icon {{
        font-size: 2rem;
        margin-bottom: 0.4rem;
    }}
    .stat-value {{
        font-size: 2rem;
        font-weight: 700;
        color: {c["accent_light"]};
        margin: 0.3rem 0;
    }}
    .stat-label {{
        font-size: 0.85rem;
        color: {c["text_muted"]};
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }}

    /* ---- Hero / Page banner ---- */
    .page-hero {{
        background: linear-gradient(135deg, {c["bg_secondary"]}, {c["bg_card"]});
        border: 1px solid {c["border"]};
        border-radius: 16px;
        padding: 2.5rem 2rem;
        margin-bottom: 2rem;
        text-align: center;
    }}
    .page-hero h1 {{
        font-size: 2.2rem !important;
        margin-bottom: 0.5rem !important;
    }}
    .page-hero p {{
        color: {c["text_secondary"]};
        font-size: 1.05rem;
        max-width: 700px;
        margin: 0 auto;
    }}

    /* ---- Placeholder box ---- */
    .placeholder-box {{
        background-color: {c["bg_card"]};
        border: 2px dashed {c["border"]};
        border-radius: 12px;
        padding: 2.5rem 1.5rem;
        text-align: center;
        color: {c["text_muted"]};
    }}
    .placeholder-box .placeholder-icon {{
        font-size: 2.5rem;
        margin-bottom: 0.6rem;
    }}
    .placeholder-box .placeholder-text {{
        font-size: 0.95rem;
        color: {c["text_muted"]};
    }}

    /* ---- Section header ---- */
    .section-header {{
        color: {c["text_primary"]};
        font-size: 1.3rem;
        font-weight: 600;
        margin-bottom: 0.8rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid {c["accent_dark"]};
        display: inline-block;
    }}

    /* ---- Nav card (home page) ---- */
    .nav-card {{
        background: linear-gradient(135deg, {c["bg_card"]}, {c["bg_card_hover"]});
        border: 1px solid {c["border"]};
        border-radius: 14px;
        padding: 2rem 1.5rem;
        text-align: center;
        transition: border-color 0.25s ease, transform 0.2s ease;
        min-height: 180px;
    }}
    .nav-card:hover {{
        border-color: {c["accent"]};
        transform: translateY(-3px);
    }}
    .nav-card-icon {{
        font-size: 2.5rem;
        margin-bottom: 0.6rem;
    }}
    .nav-card-title {{
        font-size: 1.1rem;
        font-weight: 600;
        color: {c["text_primary"]};
        margin-bottom: 0.4rem;
    }}
    .nav-card-desc {{
        font-size: 0.85rem;
        color: {c["text_muted"]};
    }}

    /* ---- Info banner ---- */
    .info-banner {{
        background-color: rgba(0, 180, 216, 0.08);
        border: 1px solid rgba(0, 180, 216, 0.25);
        border-radius: 10px;
        padding: 1rem 1.3rem;
        color: {c["accent_light"]};
        font-size: 0.9rem;
    }}

    /* ---- Hide default streamlit elements ---- */
    #MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden;}}
    [data-testid="stSidebarNav"] {{
        display: none !important;
    }}
    </style>
    """


def inject_global_styles():
    """Inject the global stylesheet into the current Streamlit page."""
    import streamlit as st
    st.markdown(_global_css(), unsafe_allow_html=True)
