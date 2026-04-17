import streamlit as st
from profiler import profiler_main
from patterns import patterns_main
from patterns_2 import patterns_2_main
from problem_type import problem_type_main
from model_engine import model_main
from insights import insights_main
from styles import get_css
import pandas as pd
import time

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Data Insight System",
    page_icon="✦",
    layout="centered",
)
api_key = st.secrets["API_KEY"]

st.markdown(f"<style>{get_css()}</style>", unsafe_allow_html=True)

# ── Session state ─────────────────────────────────────────────────────────────
for key, default in {
    "uploaded_file": None,
    "reset_trigger": 0,
    "target_column": None,
    "analysis_done": False,
    "metadata": None,
    "correlations": None,
    "figures": None,
    "problem": None,
    "model_stats": None,
    "summary": None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# ── Helper ────────────────────────────────────────────────────────────────────
def get_column_names(uploaded_file):
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

    st.markdown('<span class="upload-label">✦ &nbsp;Select Target Column</span>', unsafe_allow_html=True)
    column_names = get_column_names(uploaded_file)

    if column_names:
        options = [None] + column_names

        selected = st.selectbox(
            label="Choose the target column for analysis",
            options=options,
            index=options.index(st.session_state.target_column)
                if st.session_state.target_column in options else 0,
            label_visibility="collapsed",
        )

        if selected is not None:
            st.session_state.target_column = selected
    else:
        st.error("Unable to extract columns from the file.")

# ── Divider ───────────────────────────────────────────────────────────────────
st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

# ── Buttons ───────────────────────────────────────────────────────────────────
col_reset, col_submit = st.columns([1, 1], gap="medium")

with col_reset:
    if st.button("↺ &nbsp; Reset", use_container_width=True):
        for key in ["uploaded_file", "target_column", "analysis_done",
                    "metadata", "correlations", "figures", "problem", "model_stats", "summary"]:
            st.session_state[key] = None
        st.session_state.analysis_done  = False
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
                dataframe  = pd.read_excel(st.session_state.uploaded_file)
                target_col = st.session_state.target_column

                cleaned_df, metadata = profiler_main(dataframe, target_col)
                correlations         = patterns_main(cleaned_df, metadata)
                figures              = patterns_2_main(correlations, top_n=10)
                problem              = problem_type_main(cleaned_df, metadata, target_col)
                model_stats          = model_main(cleaned_df, metadata, target_col, problem)
                summary              = insights_main(api_key, metadata, correlations, problem, model_stats, target_col)

                st.session_state.metadata     = metadata
                st.session_state.correlations = correlations
                st.session_state.figures      = figures
                st.session_state.problem      = problem
                st.session_state.model_stats  = model_stats
                st.session_state.summary      = summary
                st.session_state.analysis_done = True

                time.sleep(1.5)
            st.balloons()
            st.success(f"✓ Analysis complete! Target column: **{st.session_state.target_column}**")


# ═══════════════════════════════════════════════════════════════════════════════
#  RESULTS SECTION — only shows after analysis is done
# ═══════════════════════════════════════════════════════════════════════════════

if st.session_state.analysis_done and st.session_state.summary is not None:

    metadata    = st.session_state.metadata
    correlations= st.session_state.correlations
    figures     = st.session_state.figures
    problem     = st.session_state.problem
    model_stats = st.session_state.model_stats
    summary     = st.session_state.summary

    st.markdown('<div class="results-divider"></div>', unsafe_allow_html=True)

    # ── 1. DATASET OVERVIEW ───────────────────────────────────────────────────
    st.markdown('<span class="section-label">✦ &nbsp;Dataset Overview</span>', unsafe_allow_html=True)

    s = metadata.get("summary", {})
    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-value">{s.get('total_rows', '—')}</div>
            <div class="stat-label">Rows</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-value">{s.get('total_columns', '—')}</div>
            <div class="stat-label">Columns</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-value">{s.get('missing_values', '—')}</div>
            <div class="stat-label">Missing Cells</div>
        </div>""", unsafe_allow_html=True)
    with c4:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-value">{s.get('duplicate_rows', '—')}</div>
            <div class="stat-label">Duplicates</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Column types as pills ─────────────────────────────────────────────────
    col_types = metadata.get("column_types", {})
    pills_html = ""
    for col, dtype in col_types.items():
        css_class = {
            "numeric":     "pill-numeric",
            "categorical": "pill-categorical",
            "datetime":    "pill-datetime",
        }.get(dtype, "pill-other")
        pills_html += f'<span class="col-type-pill {css_class}">{col} <span style="opacity:0.6;">· {dtype}</span></span>'

    st.markdown(f'<div style="margin-bottom:0.5rem">{pills_html}</div>', unsafe_allow_html=True)

    # ── Missing values breakdown ──────────────────────────────────────────────
    missing_pct = metadata.get("missing_pct", {})
    if missing_pct:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<span class="section-label">✦ &nbsp;Missing Values</span>', unsafe_allow_html=True)
        for col, pct in missing_pct.items():
            bar_width = min(pct, 100)
            bar_color = "#ef4444" if pct > 50 else "#f97316" if pct > 20 else "#38bdf8"
            st.markdown(f"""
            <div style="margin-bottom:0.8rem">
                <div style="display:flex; justify-content:space-between; margin-bottom:0.2rem">
                    <span style="font-size:0.85rem; color:#cbd5e1">{col}</span>
                    <span style="font-size:0.82rem; color:{bar_color}; font-weight:600">{pct}%</span>
                </div>
                <div class="missing-bar-bg">
                    <div class="missing-bar-fill" style="width:{bar_width}%; background:{bar_color}"></div>
                </div>
            </div>""", unsafe_allow_html=True)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # ── 2. PROBLEM TYPE ───────────────────────────────────────────────────────
    st.markdown('<span class="section-label">✦ &nbsp;Problem Type</span>', unsafe_allow_html=True)

    problem_type   = problem.get("problem_type", "unknown").replace("_", " ").title()
    problem_reason = problem.get("reason", "")
    target_col_name= problem.get("target_column", "—")

    st.markdown(f"""
    <div class="insight-card">
        <div style="margin-bottom:0.8rem">
            <span class="problem-badge">{problem_type}</span>
        </div>
        <div style="font-size:0.85rem; color:#64748b; margin-bottom:0.4rem">
            Target column: <span style="color:#cbd5e1">{target_col_name}</span>
        </div>
        <div style="font-size:0.88rem; color:#94a3b8; font-weight:300; line-height:1.6">{problem_reason}</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # ── 3. PATTERNS & CORRELATIONS ────────────────────────────────────────────
    st.markdown('<span class="section-label">✦ &nbsp;Patterns & Correlations</span>', unsafe_allow_html=True)

    pearson_sig  = correlations.get("pearson",  {}).get("significant", [])
    spearman_sig = correlations.get("spearman", {}).get("significant", [])
    cramers_sig  = correlations.get("cramers",  {}).get("significant", [])

    if pearson_sig or spearman_sig or cramers_sig:

        if pearson_sig:
            st.markdown('<p style="font-size:0.78rem; color:#64748b; margin-bottom:0.5rem; letter-spacing:0.05em">PEARSON — NUMERIC</p>', unsafe_allow_html=True)
            for p in pearson_sig[:3]:
                st.markdown(f"""
                <div class="corr-row">
                    <span class="corr-cols">{p['column_a']} &nbsp;↔&nbsp; {p['column_b']}</span>
                    <span>
                        <span class="corr-score">r = {p['pearson_r']}</span>
                        <span class="corr-tag">{p['strength']} · {p['direction']}</span>
                    </span>
                </div>""", unsafe_allow_html=True)

        if spearman_sig:
            st.markdown('<p style="font-size:0.78rem; color:#64748b; margin: 0.8rem 0 0.5rem; letter-spacing:0.05em">SPEARMAN — NON-LINEAR</p>', unsafe_allow_html=True)
            for s in spearman_sig[:3]:
                st.markdown(f"""
                <div class="corr-row">
                    <span class="corr-cols">{s['column_a']} &nbsp;↔&nbsp; {s['column_b']}</span>
                    <span>
                        <span class="corr-score">r = {s['spearman_r']}</span>
                        <span class="corr-tag">{s['strength']} · {s['direction']}</span>
                    </span>
                </div>""", unsafe_allow_html=True)

        if cramers_sig:
            st.markdown('<p style="font-size:0.78rem; color:#64748b; margin: 0.8rem 0 0.5rem; letter-spacing:0.05em">CRAMÉR\'S V — CATEGORICAL</p>', unsafe_allow_html=True)
            for c in cramers_sig[:3]:
                st.markdown(f"""
                <div class="corr-row">
                    <span class="corr-cols">{c['column_a']} &nbsp;↔&nbsp; {c['column_b']}</span>
                    <span>
                        <span class="corr-score">V = {c['cramers_v']}</span>
                        <span class="corr-tag">{c['strength']}</span>
                    </span>
                </div>""", unsafe_allow_html=True)

        # ── Correlation figures ───────────────────────────────────────────────
        if figures:
            st.markdown("<br>", unsafe_allow_html=True)
            for fig in figures.values():
                if fig is not None:
                    st.plotly_chart(fig, use_container_width=True)
    else:
        st.markdown("""
        <div class="insight-card" style="text-align:center; padding: 2rem">
            <div style="font-size:0.9rem; color:#64748b">No significant correlations found above the threshold.</div>
        </div>""", unsafe_allow_html=True)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # ── 4. MODEL RESULTS ─────────────────────────────────────────────────────
    st.markdown('<span class="section-label">✦ &nbsp;Model Results</span>', unsafe_allow_html=True)

    all_results  = model_stats.get("all_results", [])
    best_model   = model_stats.get("best_model", "")
    problem_t    = model_stats.get("problem_type", "")
    is_classif   = "classification" in str(problem_t)

    if all_results:
        for r in all_results:
            is_best    = r["model"] == best_model
            card_class = "model-row best-model" if is_best else "model-row"
            badge      = '<span class="badge">Best</span>' if is_best else ""

            if is_classif:
                score_text = f"Accuracy: {r.get('accuracy', '—')} &nbsp;|&nbsp; F1: {r.get('f1_score', '—')} &nbsp;|&nbsp; ROC-AUC: {r.get('roc_auc', '—')}"
            else:
                score_text = f"R²: {r.get('r2_score', '—')} &nbsp;|&nbsp; MAE: {r.get('mae', '—')} &nbsp;|&nbsp; RMSE: {r.get('rmse', '—')}"

            st.markdown(f"""
            <div class="{card_class}">
                <span class="model-name">{r['model']}{badge}</span>
                <span class="model-score">{score_text}</span>
            </div>""", unsafe_allow_html=True)
    else:
        error_msg = model_stats.get("error", "No models could be evaluated.")
        st.markdown(f"""
        <div class="insight-card" style="text-align:center; padding:2rem">
            <div style="font-size:0.9rem; color:#ef4444">{error_msg}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # ── 5. AI INSIGHTS ────────────────────────────────────────────────────────
    st.markdown('<span class="section-label">✦ &nbsp;AI Insights</span>', unsafe_allow_html=True)

    if summary.get("success"):

        st.markdown(f"""
        <div class="insight-card">
            <div class="insight-card-title">Executive Summary</div>
            <div class="insight-card-body">{summary.get('executive_summary', '')}</div>
        </div>""", unsafe_allow_html=True)

        st.markdown(f"""
        <div class="insight-card">
            <div class="insight-card-title">Technical Summary</div>
            <div class="insight-card-body">{summary.get('technical_summary', '')}</div>
        </div>""", unsafe_allow_html=True)

        st.markdown(f"""
        <div class="recommendation-card">
            <div class="insight-card-title">✦ &nbsp;Top Recommendation</div>
            <div class="insight-card-body">{summary.get('top_recommendation', '')}</div>
        </div>""", unsafe_allow_html=True)

    else:
        st.error(f"Could not generate AI insights: {summary.get('technical_summary', 'Unknown error')}")

    # ── Footer ────────────────────────────────────────────────────────────────
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.markdown("""
    <p style="text-align:center; font-size:0.75rem; color:#334155; letter-spacing:0.05em">
        AI DATA INSIGHT SYSTEM &nbsp;·&nbsp; POWERED BY GEMINI
    </p>""", unsafe_allow_html=True)
