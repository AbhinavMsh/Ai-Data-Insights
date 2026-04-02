import streamlit as st
from profiler import profiler_main
from patterns import patterns_main
from patterns_2 import patterns_2_main
from problem_type import problem_type_main
from model_engine import model_main
import pandas as pd
import time

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Data Insight System",
    page_icon="✦",
    layout="centered",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:ital,wght@0,300;0,400;0,500;1,300&display=swap');

/* ── Reset & base ── */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

/* ── Page background ── */
.stApp {
    background: #0b0f1a;
    background-image:
        radial-gradient(ellipse 80% 50% at 50% -10%, rgba(99,179,237,0.13) 0%, transparent 70%),
        radial-gradient(ellipse 60% 40% at 80% 90%,  rgba(56,189,248,0.08) 0%, transparent 60%);
}

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 4rem; padding-bottom: 4rem; max-width: 680px; }

/* ── Accent rule above heading ── */
.accent-rule {
    width: 48px;
    height: 3px;
    background: linear-gradient(90deg, #38bdf8, #7dd3fc);
    border-radius: 2px;
    margin: 0 auto 1.4rem auto;
}

/* ── Main heading ── */
.main-heading {
    font-family: 'Syne', sans-serif;
    font-size: 2.6rem;
    font-weight: 800;
    text-align: center;
    letter-spacing: -0.02em;
    line-height: 1.15;
    margin-bottom: 1rem;
    background: linear-gradient(135deg, #e2f0ff 30%, #7dd3fc 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

/* ── Description ── */
.description {
    font-size: 1rem;
    font-weight: 300;
    color: #94a3b8;
    text-align: center;
    line-height: 1.7;
    max-width: 520px;
    margin: 0 auto 2.6rem auto;
    letter-spacing: 0.01em;
}

/* ── Upload card ── */
.upload-label {
    font-family: 'Syne', sans-serif;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #38bdf8;
    margin-bottom: 0.6rem;
    display: block;
}

[data-testid="stFileUploader"] {
    background: rgba(255,255,255,0.035) !important;
    border: 1.5px dashed rgba(56,189,248,0.35) !important;
    border-radius: 14px !important;
    padding: 2rem 1.5rem !important;
    transition: border-color 0.2s, background 0.2s;
}
[data-testid="stFileUploader"]:hover {
    border-color: rgba(56,189,248,0.7) !important;
    background: rgba(56,189,248,0.05) !important;
}
[data-testid="stFileUploader"] label {
    color: #cbd5e1 !important;
    font-size: 0.95rem !important;
}
[data-testid="stFileUploader"] small {
    color: #64748b !important;
}
[data-testid="stFileUploadDropzone"] svg {
    color: #38bdf8 !important;
}

/* ── Divider ── */
.divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(56,189,248,0.2), transparent);
    margin: 2rem 0;
}

/* ── Buttons ── */
.stButton > button {
    font-family: 'Syne', sans-serif !important;
    font-size: 0.85rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.06em !important;
    border-radius: 10px !important;
    padding: 0.65rem 1.6rem !important;
    transition: all 0.18s ease !important;
    width: 100% !important;
    cursor: pointer !important;
}

/* Submit — primary */
div[data-testid="column"]:nth-child(2) .stButton > button {
    background: linear-gradient(135deg, #0ea5e9, #38bdf8) !important;
    color: #0b0f1a !important;
    border: none !important;
    box-shadow: 0 4px 24px rgba(56,189,248,0.3) !important;
}
div[data-testid="column"]:nth-child(2) .stButton > button:hover {
    box-shadow: 0 6px 32px rgba(56,189,248,0.5) !important;
    transform: translateY(-1px) !important;
}

/* Reset — ghost */
div[data-testid="column"]:nth-child(1) .stButton > button {
    background: transparent !important;
    color: #94a3b8 !important;
    border: 1.5px solid rgba(148,163,184,0.3) !important;
}
div[data-testid="column"]:nth-child(1) .stButton > button:hover {
    border-color: rgba(148,163,184,0.6) !important;
    color: #cbd5e1 !important;
    background: rgba(255,255,255,0.04) !important;
}

/* ── Success / error messages ── */
[data-testid="stAlert"] {
    border-radius: 10px !important;
    font-size: 0.9rem !important;
}
</style>
""", unsafe_allow_html=True)

# ── Session state ─────────────────────────────────────────────────────────────
if "uploaded_file" not in st.session_state:
    st.session_state.uploaded_file = None
if "reset_trigger" not in st.session_state:
    st.session_state.reset_trigger = 0
if "target_column" not in st.session_state:
    st.session_state.target_column = None

# ── Helper function to extract column names ────────────────────────────────────
def get_column_names(uploaded_file):
    """Extract column names from Excel file"""
    try:
        df = pd.read_excel(uploaded_file)
        return df.columns.tolist()
    except Exception as e:
        st.error(f"Error reading file: {e}")
        return []

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown('<div class="accent-rule"></div>', unsafe_allow_html=True)
st.markdown('<h1 class="main-heading">AI Data Insight System</h1>', unsafe_allow_html=True)
st.markdown("""
<p class="description">
    Upload your Excel dataset and let our AI surface patterns, anomalies, and
    actionable insights — automatically. No code, no configuration, just clarity.
</p>
""", unsafe_allow_html=True)

# ── File uploader ─────────────────────────────────────────────────────────────
st.markdown('<span class="upload-label">✦ &nbsp;Upload Dataset</span>', unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    label="Drop your Excel file here, or click to browse",
    type=["xlsx", "xls"],
    accept_multiple_files=False,
    key=f"file_uploader_{st.session_state.reset_trigger}",
    help="Only .xlsx and .xls files are accepted.",
    label_visibility="collapsed",
)

if uploaded_file:
    st.session_state.uploaded_file = uploaded_file
    st.success(f"✓ &nbsp;&nbsp;**{uploaded_file.name}** ready for analysis "
               f"({uploaded_file.size / 1024:.1f} KB)")

    # ── Target column selector ────────────────────────────────────────────────
    st.markdown('<span class="upload-label">✦ &nbsp;Select Target Column</span>', unsafe_allow_html=True)
    
    column_names = get_column_names(uploaded_file)
    
    
    if column_names:
        st.session_state.target_column = st.selectbox(
            label="Choose the target column for analysis",
            options=[None] + column_names,
            index=0 if st.session_state.target_column is None else (
                column_names.index(st.session_state.target_column)
                if st.session_state.target_column in column_names else 0
            ),
            label_visibility="collapsed",
        )
    else:
        st.error("Unable to extract columns from the file.")

# ── Divider ───────────────────────────────────────────────────────────────────
st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

# ── Buttons ───────────────────────────────────────────────────────────────────
col_reset, col_submit = st.columns([1, 1], gap="medium")

with col_reset:
    if st.button("↺ &nbsp; Reset", use_container_width=True):
        st.session_state.uploaded_file = None
        st.session_state.target_column = None
        st.session_state.reset_trigger += 1
        st.rerun()

with col_submit:
    if st.button("Analyse &nbsp; →", use_container_width=True, type="primary"):
        if st.session_state.uploaded_file is None:
            st.warning("⚠ Please upload an Excel file before submitting.")
        elif st.session_state.target_column is None:
            st.warning("⚠ Please select a target column before submitting.")
        else:
            with st.spinner("Analysing your data…"):
                dataframe = pd.read_excel(st.session_state.uploaded_file)
                target_col = st.session_state.target_column
                
                cleaned_df, metadata = profiler_main(dataframe,target_col)
                correlations = patterns_main(cleaned_df, metadata)
                figures = patterns_2_main(correlations, top_n=10)
                problem = problem_type_main(cleaned_df, metadata, target_col)
                module4 = model_main( cleaned_df ,metadata,target_col, problem)

                time.sleep(1.5)
            st.balloons()
            st.success(f"✓ Analysis complete! Target column: **{st.session_state.target_column}**")