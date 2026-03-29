import pandas as pd
import numpy as np
from scipy import stats
from itertools import combinations

def get_numeric_columns(metadata: dict) -> list[str]:
    """
    Uses metadata from Module 1 if available,
    falls back to pandas inference.
    """
    column_types = metadata.get("column_types", {})
    return [col for col, dtype in column_types.items() if dtype == "numeric"]


def get_categorical_columns(metadata: dict) -> list:
    """
    Returns list of categorical column names 
    using column_types from metadata dict.
    """
    column_types = metadata.get("column_types", {})
    return [col for col, dtype in column_types.items() if dtype == "categorical"]


def classify_strength(abs_r: float) -> str:
    """Classify correlation strength based on absolute r value."""
    if abs_r >= 0.9:
        return "very strong"
    elif abs_r >= 0.7:
        return "strong"
    elif abs_r >= 0.5:
        return "moderate"
    elif abs_r >= 0.3:
        return "weak"
    else:
        return "negligible"


def compute_pearson_correlation(df: pd.DataFrame, metadata: dict, threshold: float = 0.7) -> dict:
    """
    Computes Pearson correlation matrix and extracts
    significant pairs above the given threshold.

    Returns:
        dict with:
            - matrix        : Full Pearson correlation DataFrame
            - significant   : List of dicts for pairs above threshold
            - summary       : High-level stats
    """
    numeric_cols = get_numeric_columns(metadata)
    if len(numeric_cols) < 2:
        return {"matrix": None, "significant": [], "summary": {"error": "Need at least 2 numeric columns"}}

    subset = df[numeric_cols].copy()

    # Full correlation matrix (pairwise, skipping NaNs)
    pearson_matrix = subset.corr(method="pearson")

    # Extract unique pairs (upper triangle, excluding diagonal)
    significant_pairs = []
    for col_a, col_b in combinations(numeric_cols, 2):
        r_val = pearson_matrix.loc[col_a, col_b]

        if pd.isna(r_val):
            continue

        # p-value calculation using scipy
        valid_mask = subset[[col_a, col_b]].dropna()
        if len(valid_mask) < 3:
            p_val = None
        else:
            _, p_val = stats.pearsonr(valid_mask[col_a], valid_mask[col_b])

        pair_info = {
            "column_a"   : col_a,
            "column_b"   : col_b,
            "pearson_r"  : round(r_val, 4),
            "abs_r"      : round(abs(r_val), 4),
            "p_value"    : round(p_val, 6) if p_val is not None else None,
            "significant": abs(r_val) >= threshold,
            "direction"  : "positive" if r_val > 0 else "negative",
            "strength"   : classify_strength(abs(r_val))
        }
        if abs(r_val) >= threshold:
            significant_pairs.append(pair_info)

    # Sort by absolute correlation descending
    significant_pairs.sort(key=lambda x: x["abs_r"], reverse=True)

    summary = {
        "total_numeric_cols"     : len(numeric_cols),
        "total_pairs_checked"    : len(list(combinations(numeric_cols, 2))),
        "threshold_used"         : threshold,
        "significant_pairs_found": len(significant_pairs),
        "strongest_pair"         : significant_pairs[0] if significant_pairs else None
    }

    return {
        "matrix"    : pearson_matrix,
        "significant": significant_pairs,
        "summary"   : summary
    }


def compute_spearman_correlation(df: pd.DataFrame, metadata: dict, threshold: float = 0.7) -> dict:
    """
    Computes Spearman correlation matrix and extracts
    significant pairs above the given threshold.

    Returns:
        dict with:
            - matrix        : Full Spearman correlation DataFrame
            - significant   : List of dicts for pairs above threshold
            - summary       : High-level stats
    """
    numeric_cols = get_numeric_columns(metadata)
    if len(numeric_cols) < 2:
        return {"matrix": None, "significant": [], "summary": {"error": "Need at least 2 numeric columns"}}

    subset = df[numeric_cols].copy()

    # Full correlation matrix (pairwise, skipping NaNs)
    spearman_matrix = subset.corr(method="spearman")

    # Extract unique pairs (upper triangle, excluding diagonal)
    significant_pairs = []
    for col_a, col_b in combinations(numeric_cols, 2):
        r_val = spearman_matrix.loc[col_a, col_b]

        if pd.isna(r_val):
            continue

        # p-value calculation using scipy
        valid_mask = subset[[col_a, col_b]].dropna()
        if len(valid_mask) < 3:
            p_val = None
        else:
            _, p_val = stats.spearmanr(valid_mask[col_a], valid_mask[col_b])

        pair_info = {
            "column_a"   : col_a,
            "column_b"   : col_b,
            "spearman_r" : round(r_val, 4),
            "abs_r"      : round(abs(r_val), 4),
            "p_value"    : round(p_val, 6) if p_val is not None else None,
            "significant": abs(r_val) >= threshold,
            "direction"  : "positive" if r_val > 0 else "negative",
            "strength"   : classify_strength(abs(r_val))
        }
        if abs(r_val) >= threshold:
            significant_pairs.append(pair_info)

    # Sort by absolute correlation descending
    significant_pairs.sort(key=lambda x: x["abs_r"], reverse=True)

    summary = {
        "total_numeric_cols"     : len(numeric_cols),
        "total_pairs_checked"    : len(list(combinations(numeric_cols, 2))),
        "threshold_used"         : threshold,
        "significant_pairs_found": len(significant_pairs),
        "strongest_pair"         : significant_pairs[0] if significant_pairs else None
    }

    return {
        "matrix"     : spearman_matrix,
        "significant": significant_pairs,
        "summary"    : summary
    }


def compute_cramers_v_association(df: pd.DataFrame, metadata: dict, threshold: float = 0.3) -> dict:
    """
    Computes Cramér's V association matrix and extracts
    significant pairs above the given threshold.

    Returns:
        dict with:
            - matrix        : Full Cramér's V association DataFrame
            - significant   : List of dicts for pairs above threshold
            - summary       : High-level stats
    """
    categorical_cols = get_categorical_columns(metadata)
    if len(categorical_cols) < 2:
        return {
            "matrix"     : None,
            "significant": [],
            "summary"    : {"error": "Need at least 2 categorical columns"}
        }

    subset = df[categorical_cols].copy()

    # ── Helper: Cramér's V for a single pair ─────────────────────────────
    def _cramers_v(col_a: str, col_b: str) -> tuple:
        """Returns (v_val, p_val) for a pair of categorical columns."""
        valid_mask = subset[[col_a, col_b]].dropna()
        if len(valid_mask) < 3:
            return None, None

        contingency = pd.crosstab(valid_mask[col_a], valid_mask[col_b])
        chi2, p_val, _, _ = stats.chi2_contingency(contingency)

        n    = contingency.sum().sum()
        r, k = contingency.shape

        denom = n * (min(r, k) - 1)
        if denom == 0:
            return None, None

        v_val = np.sqrt(chi2 / denom)
        return round(v_val, 4), round(p_val, 6)

    # ── Build matrix and cache results in one pass ────────────────────────
    matrix_data = pd.DataFrame(index=categorical_cols, columns=categorical_cols, dtype=float)

    for col in categorical_cols:
        matrix_data.loc[col, col] = 1.0  # self association is always 1

    pair_cache = {}
    for col_a, col_b in combinations(categorical_cols, 2):
        v_val, p_val = _cramers_v(col_a, col_b)
        pair_cache[(col_a, col_b)] = (v_val, p_val)

        val = v_val if v_val is not None else np.nan
        matrix_data.loc[col_a, col_b] = val
        matrix_data.loc[col_b, col_a] = val  # symmetric

    # ── Extract significant pairs from cache ──────────────────────────────
    significant_pairs = []
    for (col_a, col_b), (v_val, p_val) in pair_cache.items():
        if v_val is None:
            continue

        pair_info = {
            "column_a"   : col_a,
            "column_b"   : col_b,
            "cramers_v"  : v_val,
            "abs_v"      : v_val,       # already 0-1, no negatives
            "p_value"    : p_val,
            "significant": v_val >= threshold,
            "direction"  : "n/a",       # no direction in categorical
            "strength"   : classify_strength(v_val)
        }
        if v_val >= threshold:
            significant_pairs.append(pair_info)

    # Sort by association strength descending
    significant_pairs.sort(key=lambda x: x["abs_v"], reverse=True)

    summary = {
        "total_categorical_cols" : len(categorical_cols),
        "total_pairs_checked"    : len(pair_cache),
        "threshold_used"         : threshold,
        "significant_pairs_found": len(significant_pairs),
        "strongest_pair"         : significant_pairs[0] if significant_pairs else None
    }

    return {
        "matrix"     : matrix_data,
        "significant": significant_pairs,
        "summary"    : summary
    }

def patterns_main(cleaned_df: pd.DataFrame,metadata: dict) -> dict:

    numeric_threshold: float = 0.7
    categoric_threshold: float = 0.3
    
    correlations = {
        "pearson" : compute_pearson_correlation(cleaned_df, metadata, threshold=numeric_threshold),
        "spearman": compute_spearman_correlation(cleaned_df, metadata, threshold=numeric_threshold),
        "cramers" : compute_cramers_v_association(cleaned_df, metadata, threshold=categoric_threshold)
    }

    return correlations