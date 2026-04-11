import google.generativeai as genai
import json

# ─────────────────────────────────────────
#  CONFIGURE GEMINI API
# ─────────────────────────────────────────

def configure_api(api_key: str):
    genai.configure(api_key=api_key)


# ─────────────────────────────────────────
#  EXTRACT FROM MODULE 1 (profiler_main)
# ─────────────────────────────────────────

def extract_profiler_info(metadata: dict) -> str:
    """
    Extracts and formats all findings from Module 1 output (metadata dict).

    Module 1 keys used:
        metadata["summary"]           → total_rows, total_columns, missing_values, duplicate_rows
        metadata["column_types"]      → { col: "numeric"/"categorical"/"datetime"/... }
        metadata["missing_pct"]       → { col: float }  — only cols with missing data
        metadata["numeric_stats"]     → { col: { mean, median, std_dev, min, max, skewness } }
        metadata["dominated_columns"] → { col: "yes"/"no" }
        metadata["target_column"]     → str or None
    """
    summary    = metadata.get("summary", {})
    col_types  = metadata.get("column_types", {})
    missing    = metadata.get("missing_pct", {})
    num_stats  = metadata.get("numeric_stats", {})
    dominated  = metadata.get("dominated_columns", {})
    target_col = metadata.get("target_column", "Not specified")

    # Group columns by type
    numeric_cols     = [c for c, t in col_types.items() if t == "numeric"]
    categorical_cols = [c for c, t in col_types.items() if t == "categorical"]
    datetime_cols    = [c for c, t in col_types.items() if t == "datetime"]
    other_cols       = [c for c, t in col_types.items() if t not in ("numeric", "categorical", "datetime")]

    # Format missing values
    if missing:
        missing_text = "\n".join([f"  - '{col}': {pct}% missing" for col, pct in missing.items()])
    else:
        missing_text = "  - No missing values found."

    # Format numeric stats
    if num_stats:
        stats_lines = []
        for col, s in num_stats.items():
            stats_lines.append(
                f"  - '{col}': mean={s.get('mean')}, median={s.get('median')}, "
                f"std_dev={s.get('std_dev')}, min={s.get('min')}, "
                f"max={s.get('max')}, skewness={s.get('skewness')}"
            )
        stats_text = "\n".join(stats_lines)
    else:
        stats_text = "  - No numeric columns found."

    # Format dominated columns
    dominated_yes = [c for c, v in dominated.items() if v == "yes"]
    dominated_text = (
        f"  - Imbalanced columns (one value > 70%): {', '.join(dominated_yes)}"
        if dominated_yes else
        "  - No heavily dominated categorical columns found."
    )

    return f"""
=== MODULE 1: DATASET PROFILE ===
- Total Rows:          {summary.get('total_rows', 'N/A')}
- Total Columns:       {summary.get('total_columns', 'N/A')}
- Total Missing Cells: {summary.get('missing_values', 'N/A')}
- Duplicate Rows:      {summary.get('duplicate_rows', 'N/A')}
- Target Column:       {target_col}

Column Types:
  - Numeric:     {', '.join(numeric_cols) if numeric_cols else 'None'}
  - Categorical: {', '.join(categorical_cols) if categorical_cols else 'None'}
  - Datetime:    {', '.join(datetime_cols) if datetime_cols else 'None'}
  - Other:       {', '.join(other_cols) if other_cols else 'None'}

Missing Values (only columns with missing data):
{missing_text}

Numeric Statistics:
{stats_text}

Class Imbalance:
{dominated_text}
"""


# ─────────────────────────────────────────
#  EXTRACT FROM MODULE 2 (patterns_main)
# ─────────────────────────────────────────

def extract_patterns_info(correlations: dict) -> str:
    """
    Extracts and formats all findings from Module 2 output.

    Module 2 keys used:
        correlations["pearson"]["significant"]   → list of dicts: column_a, column_b, pearson_r, strength, direction, p_value
        correlations["pearson"]["summary"]        → significant_pairs_found, strongest_pair, threshold_used
        correlations["spearman"]["significant"]  → list of dicts: column_a, column_b, spearman_r, strength, direction
        correlations["cramers"]["significant"]   → list of dicts: column_a, column_b, cramers_v, strength
    """

    # ── Pearson ───────────────────────────────────────────────────────────
    pearson      = correlations.get("pearson", {})
    pearson_sig  = pearson.get("significant", [])
    pearson_summ = pearson.get("summary", {})

    if pearson_sig:
        # Top 3 only
        pearson_lines = []
        for p in pearson_sig[:3]:
            pearson_lines.append(
                f"  - '{p['column_a']}' & '{p['column_b']}': r={p['pearson_r']} "
                f"({p['strength']}, {p['direction']}, p={p.get('p_value', 'N/A')})"
            )
        pearson_text = "\n".join(pearson_lines)
    else:
        pearson_text = "  - No significant Pearson correlations found above threshold."

    # ── Spearman ──────────────────────────────────────────────────────────
    spearman     = correlations.get("spearman", {})
    spearman_sig = spearman.get("significant", [])

    if spearman_sig:
        spearman_lines = []
        for s in spearman_sig[:3]:
            spearman_lines.append(
                f"  - '{s['column_a']}' & '{s['column_b']}': r={s['spearman_r']} "
                f"({s['strength']}, {s['direction']})"
            )
        spearman_text = "\n".join(spearman_lines)
    else:
        spearman_text = "  - No significant Spearman correlations found above threshold."

    # ── Cramér's V ────────────────────────────────────────────────────────
    cramers     = correlations.get("cramers", {})
    cramers_sig = cramers.get("significant", [])

    if cramers_sig:
        cramers_lines = []
        for c in cramers_sig[:3]:
            cramers_lines.append(
                f"  - '{c['column_a']}' & '{c['column_b']}': V={c['cramers_v']} ({c['strength']})"
            )
        cramers_text = "\n".join(cramers_lines)
    else:
        cramers_text = "  - No significant categorical associations found above threshold."

    return f"""
=== MODULE 2: PATTERN DETECTION ===
Pearson Correlations (numeric, top 3):
{pearson_text}
  (Total significant pairs: {pearson_summ.get('significant_pairs_found', 0)}, threshold: {pearson_summ.get('threshold_used', 0.7)})

Spearman Correlations (non-linear, top 3):
{spearman_text}

Cramér's V — Categorical Associations (top 3):
{cramers_text}
"""


# ─────────────────────────────────────────
#  EXTRACT FROM MODULE 3 (problem_type_main)
# ─────────────────────────────────────────

def extract_problem_info(problem: dict) -> str:
    """
    Extracts and formats findings from Module 3 output.

    Module 3 keys used:
        problem["problem_type"]  → 'regression', 'binary_classification',
                                   'multiclass_classification', 'unsupervised_clustering', 'unknown'
        problem["target_column"] → str or None
        problem["reason"]        → plain english explanation
    """
    return f"""
=== MODULE 3: PROBLEM TYPE ===
- Detected Problem Type: {problem.get('problem_type', 'N/A')}
- Target Column:         {problem.get('target_column', 'None')}
- Reason:                {problem.get('reason', 'N/A')}
"""


# ─────────────────────────────────────────
#  EXTRACT FROM MODULE 4 (model_main)
# ─────────────────────────────────────────

def extract_model_info(model_stats: dict) -> str:
    """
    Extracts and formats findings from Module 4 output.

    Module 4 keys used:
        model_stats["best_model"]   → name of best model e.g. "Random Forest"
        model_stats["best_metric"]  → float score of best model
        model_stats["problem_type"] → problem type string
        model_stats["all_results"]  → list of dicts with per-model scores
        model_stats["error"]        → None or error string
    """
    error        = model_stats.get("error")
    best_model   = model_stats.get("best_model", "N/A")
    best_metric  = model_stats.get("best_metric", "N/A")
    problem_type = model_stats.get("problem_type", "N/A")
    all_results  = model_stats.get("all_results", [])

    if error:
        return f"""
=== MODULE 4: MODEL SELECTION ===
- Error: {error}
"""

    # Determine metric label based on problem type
    if "classification" in str(problem_type):
        metric_label = "F1 Score (best metric)"
        score_lines = []
        for r in all_results:
            score_lines.append(
                f"  - {r['model']}: Accuracy={r.get('accuracy')}, "
                f"F1={r.get('f1_score')}, ROC-AUC={r.get('roc_auc')}"
            )
    else:
        metric_label = "R² Score (best metric)"
        score_lines = []
        for r in all_results:
            score_lines.append(
                f"  - {r['model']}: R²={r.get('r2_score')}, "
                f"MAE={r.get('mae')}, RMSE={r.get('rmse')}"
            )

    all_scores_text = "\n".join(score_lines) if score_lines else "  - No results available."

    return f"""
=== MODULE 4: MODEL SELECTION ===
- Best Model:      {best_model}
- {metric_label}: {best_metric}

All Models Compared:
{all_scores_text}
"""


# ─────────────────────────────────────────
#  BUILD THE FULL PROMPT
# ─────────────────────────────────────────

def build_prompt(metadata: dict, correlations: dict, problem: dict, model_stats: dict) -> str:
    """
    Assembles all 4 module extracts into one structured prompt for Gemini.

    Parameters:
        metadata     : output from profiler_main()    → Module 1
        correlations : output from patterns_main()    → Module 2
        problem      : output from problem_type_main()→ Module 3
        model_stats  : output from model_main()       → Module 4
    """
    profiler_section = extract_profiler_info(metadata)
    patterns_section = extract_patterns_info(correlations)
    problem_section  = extract_problem_info(problem)
    model_section    = extract_model_info(model_stats)

    prompt = f"""
You are a data science assistant. Below are findings from an automated ML pipeline.
Analyse all sections and generate insights.

{profiler_section}
{patterns_section}
{problem_section}
{model_section}

=== YOUR TASK ===
Based on ALL the above findings, respond in this exact JSON format:

{{
  "executive_summary": "2-3 sentences explaining the dataset and key findings in simple language for someone with no technical background.",
  "technical_summary": "3-4 sentences covering column types, missing data, correlations, anomalies, skewness, class imbalance, problem type, and model performance for a data science audience.",
  "top_recommendation": "1-2 sentences on the single most important next step based on the full analysis."
}}

Return ONLY the JSON. No extra text, no markdown, no explanation outside the JSON.
"""
    return prompt


# ─────────────────────────────────────────
#  CALL GEMINI API
# ─────────────────────────────────────────

def call_gemini(prompt: str) -> str:
    """
    Sends the prompt to Gemini and returns raw response text.
    """
    model    = genai.GenerativeModel("gemini-2.5-flash-lite")
    response = model.generate_content(prompt)
    return response.text


# ─────────────────────────────────────────
#  PARSE THE RESPONSE
# ─────────────────────────────────────────

def parse_response(raw_text: str) -> dict:
    """
    Safely parses Gemini's JSON response into a clean dictionary.
    Returns fallback messages if parsing fails.
    """
    try:
        cleaned = raw_text.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("```")[1]
            if cleaned.startswith("json"):
                cleaned = cleaned[4:]

        result = json.loads(cleaned)

        return {
            "executive_summary" : result.get("executive_summary",  "Not available."),
            "technical_summary" : result.get("technical_summary",  "Not available."),
            "top_recommendation": result.get("top_recommendation", "Not available."),
            "success"           : True
        }

    except Exception as e:
        return {
            "executive_summary" : "Could not generate summary. Please check your API key.",
            "technical_summary" : str(e),
            "top_recommendation": "Try again or check your internet connection.",
            "success"           : False
        }


# ─────────────────────────────────────────
#  MAIN FUNCTION
# ─────────────────────────────────────────

def insights_main(api_key: str, metadata: dict, correlations: dict, problem: dict, model_stats: dict) -> dict:
    """
    Master function — the only one you call from dashboard.py.

    Parameters match the exact outputs of each module:
        api_key      : your Gemini API key string
        metadata     : output of profiler_main()     → Module 1 (the metadata dict, not cleaned_df)
        correlations : output of patterns_main()     → Module 2
        problem      : output of problem_type_main() → Module 3
        model_stats  : output of model_main()        → Module 4

    Returns:
        {
            "executive_summary" : "...",
            "technical_summary" : "...",
            "top_recommendation": "...",
            "success"           : True / False
        }

    Usage in dashboard.py:
        cleaned_df, metadata = profiler_main(uploaded_file, target_col)
        correlations         = patterns_main(cleaned_df, metadata)
        problem              = problem_type_main(cleaned_df, metadata, target_col)
        model_stats          = model_main(cleaned_df, metadata, target_col, problem)

        insights = insights_main(
            api_key      = "your_gemini_api_key",
            metadata     = metadata,
            correlations = correlations,
            problem      = problem,
            model_stats  = model_stats
        )

        st.write(insights["executive_summary"])
        st.write(insights["technical_summary"])
        st.write(insights["top_recommendation"])
    """
    try:
        configure_api(api_key)

        prompt       = build_prompt(metadata, correlations, problem, model_stats)
        raw_response = call_gemini(prompt)

        return parse_response(raw_response)

    except Exception as e:
        return {
            "executive_summary" : "API call failed.",
            "technical_summary" : f"Error: {str(e)}",
            "top_recommendation": "Check your API key and internet connection.",
            "success"           : False
        }