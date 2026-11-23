"""
Dashboard
=========
Analytics dashboard with placeholder statistics and history.
No fabricated data — all counters start at zero.
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
    render_stat_card,
    render_placeholder,
    render_section_header,
    render_info_banner,
    render_footer,
)

# ---- Page Config ----
setup_page("Dashboard", icon="📈")

# ================================================================
# HERO
# ================================================================
render_hero(
    "📈 Detection Dashboard",
    "Track your analysis history and detection statistics at a glance",
)

render_info_banner(
    "Statistics will be populated automatically once the detection backend "
    "is integrated.  All counters currently display zero."
)
st.markdown("")

# ================================================================
# STAT CARDS
# ================================================================
render_section_header("📊 Overview Statistics")
st.markdown("")

s1, s2, s3, s4 = st.columns(4)

with s1:
    render_stat_card("🖼️", "0", "Total Images Analyzed")
with s2:
    render_stat_card("✅", "0", "Real Images")
with s3:
    render_stat_card("⚠️", "0", "Deepfake Images")
with s4:
    render_stat_card("🎥", "0", "Videos Analyzed")

st.markdown("")
st.markdown("")

# ================================================================
# DETECTION ACCURACY (placeholder)
# ================================================================
render_section_header("🎯 Detection Summary")

sum_col1, sum_col2 = st.columns(2)

with sum_col1:
    render_card(
        """
        <h4 style="color:#48CAE4; margin-top:0;">Classification Breakdown</h4>
        <table style="width:100%; border-collapse:collapse; color:#94A3B8; font-size:0.9rem;">
        <thead>
        <tr style="border-bottom:1px solid #1E3050;">
            <th style="text-align:left; padding:0.5rem; color:#E8EDF5;">Category</th>
            <th style="text-align:center; padding:0.5rem; color:#E8EDF5;">Count</th>
            <th style="text-align:center; padding:0.5rem; color:#E8EDF5;">Percentage</th>
        </tr>
        </thead>
        <tbody>
        <tr style="border-bottom:1px solid #1E3050;">
            <td style="padding:0.5rem;">✅ Real</td>
            <td style="padding:0.5rem; text-align:center;">0</td>
            <td style="padding:0.5rem; text-align:center;">—</td>
        </tr>
        <tr style="border-bottom:1px solid #1E3050;">
            <td style="padding:0.5rem;">⚠️ Deepfake</td>
            <td style="padding:0.5rem; text-align:center;">0</td>
            <td style="padding:0.5rem; text-align:center;">—</td>
        </tr>
        <tr>
            <td style="padding:0.5rem; font-weight:600; color:#E8EDF5;">Total</td>
            <td style="padding:0.5rem; text-align:center; font-weight:600; color:#E8EDF5;">0</td>
            <td style="padding:0.5rem; text-align:center;">—</td>
        </tr>
        </tbody>
        </table>
        """
    )

with sum_col2:
    render_placeholder(
        "📊",
        "Detection distribution chart (Real vs Deepfake) will appear here once analyses are recorded.",
    )

st.markdown("")

# ================================================================
# RECENT DETECTION HISTORY
# ================================================================
render_section_header("🕑 Recent Detection History")

render_card(
    """
    <table style="width:100%; border-collapse:collapse; color:#94A3B8; font-size:0.9rem;">
    <thead>
    <tr style="border-bottom:1px solid #1E3050;">
        <th style="text-align:left; padding:0.6rem; color:#E8EDF5;">Timestamp</th>
        <th style="text-align:left; padding:0.6rem; color:#E8EDF5;">File Name</th>
        <th style="text-align:center; padding:0.6rem; color:#E8EDF5;">Type</th>
        <th style="text-align:center; padding:0.6rem; color:#E8EDF5;">Prediction</th>
        <th style="text-align:center; padding:0.6rem; color:#E8EDF5;">Confidence</th>
    </tr>
    </thead>
    <tbody>
    <tr>
        <td colspan="5" style="padding:1.5rem; text-align:center; color:#64748B;">
            No analyses recorded yet.  Upload an image or video to get started.
        </td>
    </tr>
    </tbody>
    </table>
    """
)

st.markdown("")

# ================================================================
# ROBUSTNESS (placeholder)
# ================================================================
render_section_header("🛡️ Robustness Test Results")

render_placeholder(
    "🧪",
    "Robustness test results (compression, blur, noise, resizing) will be displayed here after evaluation.",
)

# ================================================================
# FOOTER
# ================================================================
render_footer()
