# Module 3: Problem Type Detector
import pandas as pd
def detect_problem_type(df: pd.DataFrame, metadata: dict, target_column: str = None, multiclass_max: int = 14) -> dict:

 # ── Step 1: Resolve target column ────────────────────────────────────
    
    if target_column is None or target_column == "None":
        target_column = metadata.get("target_column", None)

    if target_column is None or target_column == "None":
        return {
            "problem_type" : "unsupervised_clustering",
            "target_column": None,
            "reason"       : "No target column provided — defaulting to unsupervised clustering."
        }

    if target_column not in df.columns:
        return {
            "problem_type" : "unknown",
            "target_column": target_column,
            "reason"       : f"Target column '{target_column}' not found in DataFrame."
        }

    # ── Step 2: Analyze target column ────────────────────────────────────
    target_series  = df[target_column].dropna()
    n_unique       = target_series.nunique()
    column_types   = metadata.get("column_types", {})
    detected_dtype = column_types.get(target_column, "unknown")

    # ── Step 3: Apply rules ───────────────────────────────────────────────
    if n_unique == 2:
        return {
            "problem_type" : "binary_classification",
            "target_column": target_column,
            "reason"       : f"Target '{target_column}' has exactly 2 unique values → Binary Classification."
        }

    if 3 <= n_unique <= multiclass_max:
        return {
            "problem_type" : "multiclass_classification",
            "target_column": target_column,
            "reason"       : f"Target '{target_column}' has {n_unique} unique values (3–{multiclass_max}) → Multi-class Classification."
        }

    if detected_dtype == "numeric":
        return {
            "problem_type" : "regression",
            "target_column": target_column,
            "reason"       : f"Target '{target_column}' is numeric and continuous ({n_unique} unique values) → Regression."
        }

    return {
        "problem_type" : "unknown",
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