# Module 1:Data Profiler

import pandas as pd
import numpy as np
from scipy import stats

def detect_column_types(df: pd.DataFrame) -> dict:
    """
    Detect the semantic type of each column in a DataFrame.
    
    Returns a dict mapping column name -> detected type:
      'numeric', 'categorical', 'datetime', 'boolean', 'string', 'unknown'
    """
    results = {}

    for col in df.columns:
        series = df[col].dropna()

        if series.empty:
            results[col] = "unknown"
            continue

        # --- Boolean ---
        if pd.api.types.is_bool_dtype(series):
            results[col] = "boolean"

        # --- Numeric ---
        elif pd.api.types.is_numeric_dtype(series):
            results[col] = "numeric"

        # --- Datetime ---
        elif pd.api.types.is_datetime64_any_dtype(series):
            results[col] = "datetime"

        # --- Object / string columns — needs deeper inspection ---
        elif pd.api.types.is_object_dtype(series):
            # Try parsing as datetime
            try:
                pd.to_datetime(series.sample(min(50, len(series))), infer_datetime_format=True)
                results[col] = "datetime"
                continue
            except Exception:
                pass

            # Boolean-like strings
            bool_values = {"true", "false", "yes", "no", "1", "0", "t", "f", "y", "n"}
            if set(series.str.lower().unique()).issubset(bool_values):
                results[col] = "boolean"
                continue

            unique_ratio = series.nunique() / len(series)

            # Low cardinality → categorical, everything else → string
            if series.nunique() <= 20 or unique_ratio < 0.05:
                results[col] = "categorical"
            else:
                results[col] = "string"

        else:
            results[col] = "unknown"

    return results

def missing_data_percentage(df) -> dict:
    '''
     Takes a dataframe and returns a dictionary of columns with missing data and their percentage
     Only columns with at least one missing value are included
     Percentage is rounded to 2 decimal places
    '''
    missing = {}
    
    for col in df.columns:
        total_rows = len(df)
        missing_count = df[col].isna().sum()
        
        if missing_count > 0:
            percentage = (missing_count / total_rows) * 100
            percentage = round(percentage, 2)
            missing[col] = percentage
    
    return missing

def calculate_stats(df) -> dict:
    '''
    Calculates mean, median, std dev, min, max and skewness
    for all numeric columns in a DataFrame.
    Skips non-numeric columns and drops NaN values before computing.
    Returns a nested dictionary with column names as keys and stats as values.
    '''
    stats_dict = {}
    
    for col in df.select_dtypes(include=[np.number]).columns:
        data = df[col].dropna()
        stats_dict[col] = {
            'mean':     round(data.mean(), 4),
            'median':   round(data.median(), 4),
            'std_dev':  round(data.std(), 4),
            'min':      round(data.min(), 4),
            'max':      round(data.max(), 4),
            'skewness': round(stats.skew(data), 4)
        }
    
    return stats_dict

def check_dominated_categorical(df, threshold=0.70):
    """
    Checks if any categorical column is dominated by a single value (default > 70%).
    Returns a dict with column names as keys and 'yes' (dominated) or 'no' (not dominated).
    """
    col_types = detect_column_types(df)
    categorical_cols = [col for col, dtype in col_types.items() if dtype == 'categorical']
    
    result = {}
    for col in categorical_cols:
        top_freq = df[col].value_counts(normalize=True).iloc[0]
        result[col] = 'yes' if top_freq > threshold else 'no'
    
    return result

def dataframe_summary(df):
    """
    Calculates basic structural statistics for a DataFrame.
    Returns a dictionary with total rows, columns, missing values count, and duplicate rows.
    """
    return {
        "total_rows": len(df),
        "total_columns": len(df.columns),
        "missing_values": int(df.isnull().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum())
    }

def full_profile(df: pd.DataFrame, targetCol, dominated_threshold: float = 0.70) -> dict:
    """
    Runs all profiling functions and returns a single unified dictionary.

    Keys:
      - 'summary'            → dataframe_summary()
      - 'column_types'       → detect_column_types()
      - 'missing_pct'        → missing_data_percentage()
      - 'numeric_stats'      → calculate_stats()
      - 'dominated_columns'  → check_dominated_categorical()
    """
    return {
        "summary":           dataframe_summary(df),
        "column_types":      detect_column_types(df),
        "target_column":     targetCol,
        "missing_pct":       missing_data_percentage(df),
        "numeric_stats":     calculate_stats(df),
        "dominated_columns": check_dominated_categorical(df, threshold=dominated_threshold),
    }

def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans the dataframe using the following strategy:
      - Numeric missing values  → fill with column mean
      - Categorical missing values → drop the row
      - Duplicate rows → dropped
    Returns a cleaned copy of the dataframe.
    """
    df = df.copy()

    # Step 1: Drop duplicate rows
    df = df.drop_duplicates()

    # Step 2: Detect column types
    col_types = detect_column_types(df)

    # Step 3: Fill numeric missing values with column mean
    numeric_cols = [col for col, dtype in col_types.items() if dtype == "numeric"]
    for col in numeric_cols:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].mean())

    # Step 4: Drop rows where categorical columns have missing values
    categorical_cols = [col for col, dtype in col_types.items() if dtype == "categorical"]
    df = df.dropna(subset=categorical_cols)

    return df 

def profiler_main(uploaded_file, target_col: str = "None"):
    df       = uploaded_file.copy()
    metadata = full_profile(df, targetCol=target_col)
    return clean_dataframe(df), metadata

'''
Expected output
{
    "summary": {
        "total_rows": <int>,          # e.g. 1000
        "total_columns": <int>,       # e.g. 8
        "missing_values": <int>,      # total count of NaN cells
        "duplicate_rows": <int>       # count of duplicate rows
    },

    "column_types": {
        "col_A": "numeric",           # one of: 'numeric', 'categorical',
        "col_B": "categorical",       #   'datetime', 'boolean', 'string', 'unknown'
        "col_C": "datetime",
        ...
    },

    "missing_pct": {
        "col_A": 12.50,               # only columns WITH missing values appear here
        "col_C": 3.75,                # rounded to 2 decimal places
        ...
    },

    "numeric_stats": {
        "col_A": {
            "mean":     123.4567,
            "median":   120.0,
            "std_dev":  15.2345,
            "min":      80.0,
            "max":      200.0,
            "skewness": 0.3421        # all rounded to 4 decimal places
        },
        ...
    },

    "dominated_columns": {
        "col_B": "yes",               # top value > 70% of rows
        "col_D": "no"                 # only categorical columns appear here
    }
}
'''