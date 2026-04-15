# Module 4: Auto Model Selection Engine
import pandas as pd
import numpy as np

from sklearn.linear_model    import LogisticRegression, LinearRegression
from sklearn.ensemble        import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import StratifiedKFold, KFold, cross_validate
from sklearn.preprocessing   import LabelEncoder
from xgboost                 import XGBClassifier, XGBRegressor


def _prepare_features(df: pd.DataFrame, target_col: str, metadata: dict) -> tuple:
    column_types = metadata.get("column_types", {})
    y            = df[target_col].copy()
    X            = df.drop(columns=[target_col]).copy()

    cat_cols = [col for col in X.columns if column_types.get(col) == "categorical"]
    if cat_cols:
        X = pd.get_dummies(X, columns=cat_cols, drop_first=True)

    non_numeric = [col for col in X.columns if not pd.api.types.is_numeric_dtype(X[col])]
    if non_numeric:
        X = X.drop(columns=non_numeric)

    if y.dtype == object or str(y.dtype) == "category":
        le = LabelEncoder()
        y  = pd.Series(le.fit_transform(y), name=target_col)

    return X, y


def model_type(problem: dict) -> dict:
    problem_type = problem.get("problem_type")

    if problem_type in ("binary_classification", "multiclass_classification"):
        return {
            "Logistic Regression": LogisticRegression(max_iter=5000, random_state=42, solver="saga"),
            "Random Forest"      : RandomForestClassifier(n_estimators=100, random_state=42),
            "XGBoost"            : XGBClassifier(n_estimators=100, random_state=42,
                                                  eval_metric="logloss", verbosity=0)
        }
    elif problem_type == "regression":
        return {
            "Linear Regression": LinearRegression(),
            "Random Forest"    : RandomForestRegressor(n_estimators=100, random_state=42),
            "XGBoost"          : XGBRegressor(n_estimators=100, random_state=42, verbosity=0)
        }
    else:
        return {}


def _evaluate_classification(X: pd.DataFrame, y: pd.Series, models: dict, problem_type: str) -> list:
    n_classes = y.nunique()
    f1_scorer = "f1" if problem_type == "binary_classification" else "f1_macro"
    cv        = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    results = []
    for model_name, model in models.items():
        try:
            cv_results = cross_validate(
                model, X, y,
                cv     = cv,
                scoring= {
                    "accuracy": "accuracy",
                    "f1"      : f1_scorer,
                    "roc_auc" : "roc_auc" if n_classes == 2 else "roc_auc_ovr_weighted"
                },
                return_train_score = False,
                error_score        = "raise"
            )
            results.append({
                "model"      : model_name,
                "accuracy"   : round(cv_results["test_accuracy"].mean(), 4),
                "f1_score"   : round(cv_results["test_f1"].mean(), 4),
                "roc_auc"    : round(cv_results["test_roc_auc"].mean(), 4),
                "best_metric": round(cv_results["test_f1"].mean(), 4),
                "error"      : None
            })
        except Exception as e:
            results.append({
                "model"      : model_name,
                "accuracy"   : None,
                "f1_score"   : None,
                "roc_auc"    : None,
                "best_metric": -1,
                "error"      : str(e)
            })

    results.sort(key=lambda x: x["best_metric"], reverse=True)
    return results


def _evaluate_regression(X: pd.DataFrame, y: pd.Series, models: dict) -> list:
    cv      = KFold(n_splits=5, shuffle=True, random_state=42)
    results = []

    for model_name, model in models.items():
        try:
            cv_results = cross_validate(
                model, X, y,
                cv     = cv,
                scoring= {
                    "r2"  : "r2",
                    "mae" : "neg_mean_absolute_error",
                    "rmse": "neg_root_mean_squared_error"
                },
                return_train_score = False,
                error_score        = "raise"
            )
            results.append({
                "model"      : model_name,
                "r2_score"   : round(cv_results["test_r2"].mean(), 4),
                "mae"        : round(abs(cv_results["test_mae"].mean()), 4),
                "rmse"       : round(abs(cv_results["test_rmse"].mean()), 4),
                "best_metric": round(cv_results["test_r2"].mean(), 4),
                "error"      : None
            })
        except Exception as e:
            results.append({
                "model"      : model_name,
                "r2_score"   : None,
                "mae"        : None,
                "rmse"       : None,
                "best_metric": -1,
                "error"      : str(e)
            })

    results.sort(key=lambda x: x["best_metric"], reverse=True)
    return results


def model_main(cleaned_df: pd.DataFrame, metadata: dict, target_col: str, problem: dict) -> dict:
    problem_type = problem.get("problem_type")

    if problem_type in ("unsupervised_clustering", "unknown", None):
        return {
            "problem_type" : problem_type,
            "target_column": None,
            "best_model"   : None,
            "best_metric"  : None,
            "all_results"  : [],
            "error"        : f"Cannot train models — problem type is '{problem_type}'."
        }

    if not target_col or target_col not in cleaned_df.columns:
        return {
            "problem_type" : problem_type,
            "target_column": target_col,
            "best_model"   : None,
            "best_metric"  : None,
            "all_results"  : [],
            "error"        : f"Target column '{target_col}' not found in DataFrame."
        }

    X, y = _prepare_features(cleaned_df, target_col, metadata)

    if X.shape[1] == 0:
        return {
            "problem_type" : problem_type,
            "target_column": target_col,
            "best_model"   : None,
            "best_metric"  : None,
            "all_results"  : [],
            "error"        : "No feature columns remaining after preprocessing."
        }

    models = model_type(problem)

    if problem_type in ("binary_classification", "multiclass_classification"):
        all_results = _evaluate_classification(X, y, models, problem_type)
    else:
        all_results = _evaluate_regression(X, y, models)

    best        = all_results[0] if all_results else {}
    best_model  = best.get("model")
    best_metric = best.get("best_metric")

    return {
        "problem_type" : problem_type,
        "target_column": target_col,
        "best_model"   : best_model,
        "best_metric"  : best_metric,
        "all_results"  : all_results,
        "error"        : None
    }
