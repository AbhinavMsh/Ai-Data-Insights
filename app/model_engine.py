# Module 4: Auto Model Selection Engine

import pandas as pd
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import StratifiedKFold, KFold, cross_validate
from sklearn.preprocessing import LabelEncoder


def _prepare_features(df: pd.DataFrame, target_col: str, metadata: dict):
    column_types = metadata.get("column_types", {})
    
    y = df[target_col].copy()
    X = df.drop(columns=[target_col]).copy()

    # Encode categorical features
    cat_cols = [col for col in X.columns if column_types.get(col) == "categorical"]
    if cat_cols:
        X = pd.get_dummies(X, columns=cat_cols, drop_first=True)

    # Remove non-numeric columns
    X = X.select_dtypes(include=["number"])

    # Encode target if needed
    if y.dtype == object or str(y.dtype) == "category":
        y = pd.Series(LabelEncoder().fit_transform(y), name=target_col)

    return X, y


def _evaluate(X, y, model, problem_type):
    try:
        if problem_type in ("binary_classification", "multiclass_classification"):
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            scoring = {
                "accuracy": "accuracy",
                "f1": "f1" if y.nunique() == 2 else "f1_macro",
            }

        else:
            cv = KFold(n_splits=5, shuffle=True, random_state=42)
            scoring = {
                "r2": "r2",
                "mae": "neg_mean_absolute_error",
            }

        cv_results = cross_validate(
            model, X, y,
            cv=cv,
            scoring=scoring,
            return_train_score=False
        )

        if problem_type.startswith("binary") or problem_type.startswith("multiclass"):
            return {
                "accuracy": round(cv_results["test_accuracy"].mean(), 4),
                "f1_score": round(cv_results["test_f1"].mean(), 4),
                "best_metric": round(cv_results["test_f1"].mean(), 4),
                "error": None
            }
        else:
            return {
                "r2_score": round(cv_results["test_r2"].mean(), 4),
                "mae": round(abs(cv_results["test_mae"].mean()), 4),
                "best_metric": round(cv_results["test_r2"].mean(), 4),
                "error": None
            }

    except Exception as e:
        return {
            "best_metric": -1,
            "error": str(e)
        }


def model_main(cleaned_df: pd.DataFrame, metadata: dict, target_col: str, problem: dict):
    problem_type = problem.get("problem_type")

    # Basic validation
    if problem_type in ("unsupervised_clustering", "unknown", None):
        return {"error": f"Invalid problem type: {problem_type}"}

    if not target_col or target_col not in cleaned_df.columns:
        return {"error": f"Target column '{target_col}' not found"}

    X, y = _prepare_features(cleaned_df, target_col, metadata)

    if X.shape[1] == 0:
        return {"error": "No usable features after preprocessing"}

    # Select model directly (removed model_type function)
    if problem_type in ("binary_classification", "multiclass_classification"):
        model = LogisticRegression(max_iter=5000, random_state=42, solver="saga")
    else:
        model = LinearRegression()

    results = _evaluate(X, y, model, problem_type)

    return {
        "problem_type": problem_type,
        "target_column": target_col,
        "best_model": type(model).__name__,
        "best_metric": results.get("best_metric"),
        "results": results,
        "error": results.get("error")
    }
