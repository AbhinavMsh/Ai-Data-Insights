# Module 3: Problem Type Detector
import pandas as pd
def detect_problem_type(df: pd.DataFrame, metadata: dict, target_column: str = None, multiclass_max: int = 14) -> dict:
    """
    Detects the ML problem type based on the target column.

    Rules:
        - Target numeric & continuous        → Regression
        - Target has exactly 2 unique values → Binary Classification
        - Target has 3-14 unique categories  → Multi-class Classification
        - Target numeric but few unique vals → Classification (low unique count wins)
        - No target column provided          → Unsupervised Clustering

    Args:
        df             : cleaned DataFrame
        metadata       : output from Module 1
        target_column  : optional — user provided target column name
        multiclass_max : upper limit for multiclass unique values (default 14)

    Returns:
        dict with:
            - problem_type  : one of 'regression', 'binary_classification',
                              'multiclass_classification', 'unsupervised_clustering'
            - target_column : name of target column or None
            - reason        : plain english explanation of why
    """

    # ── Step 1: Resolve target column ────────────────────────────────────
    # Use user-provided target, fallback to metadata
    if target_column is "None":
        target_column = metadata.get("target_column", "None")

    # No target found anywhere → Unsupervised
    if target_column is "None":
        return {
            "problem_type" : "unsupervised_clustering",
            "target_column": "None",
            "reason"       : "No target column provided — defaulting to unsupervised clustering."
        }

    # Target column not in dataframe
    if target_column not in df.columns:
        return {
            "problem_type" : "None",
            "target_column": target_column,
            "reason"       : f"Target column '{target_column}' not found in DataFrame."
        }

    # ── Step 2: Analyze target column ────────────────────────────────────
    target_series  = df[target_column].dropna()
    n_unique       = target_series.nunique()
    column_types   = metadata.get("column_types", {})
    detected_dtype = column_types.get(target_column, "unknown")

    # ── Step 3: Apply rules ───────────────────────────────────────────────

    # Rule 1 — exactly 2 unique values → Binary Classification
    if n_unique == 2:
        return {
            "problem_type" : "binary_classification",
            "target_column": target_column,
            "reason"       : f"Target '{target_column}' has exactly 2 unique values → Binary Classification."
        }

    # Rule 2 — 3 to multiclass_max unique values → Multiclass Classification
    if 3 <= n_unique <= multiclass_max:
        return {
            "problem_type" : "multiclass_classification",
            "target_column": target_column,
            "reason"       : f"Target '{target_column}' has {n_unique} unique values (3–{multiclass_max}) → Multi-class Classification."
        }

    # Rule 3 — numeric but low unique count wins over regression
    if detected_dtype == "numeric" and n_unique <= multiclass_max:
        return {
            "problem_type" : "multiclass_classification",
            "target_column": target_column,
            "reason"       : f"Target '{target_column}' is numeric but has only {n_unique} unique values → treated as Multi-class Classification."
        }

    # Rule 4 — numeric and continuous → Regression
    if detected_dtype == "numeric":
        return {
            "problem_type" : "regression",
            "target_column": target_column,
            "reason"       : f"Target '{target_column}' is numeric and continuous ({n_unique} unique values) → Regression."
        }

    # Rule 5 — anything else with too many unique values → unknown
    return {
        "problem_type" : "Unknown",
        "target_column": target_column,
        "reason"       : f"Target '{target_column}' has {n_unique} unique values and dtype '{detected_dtype}' — could not determine problem type."
    }


def problem_type_main(df: pd.DataFrame,metadata: dict,target_column: str = "None") -> dict:
    """
    Main entry point for Module 3.

    Returns:
        dict with problem_type, target_column, reason
    """
    return detect_problem_type(df, metadata, target_column=target_column)
