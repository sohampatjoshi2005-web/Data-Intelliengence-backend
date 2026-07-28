from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split

try:
    import shap
except Exception:
    shap = None


def _clean_shap_reason(exc: Exception) -> str:
    raw = str(exc or "").lower()
    if "nopython" in raw or "numba" in raw:
        return "SHAP is not compatible with this model/pipeline in the current environment (numba backend issue)."
    if "mask" in raw or "explainer" in raw:
        return "SHAP explainer could not be constructed for this estimator."
    return "SHAP is unavailable for this model in the current setup."


def _split(df: pd.DataFrame, target: str, problem: str):
    X = df.drop(columns=[target])
    y = df[target]
    stratify = None
    test_size = 0.2
    
    # For small datasets, use a larger train set to ensure sufficient training data
    if len(df) < 10:
        test_size = 0.3  # 30% test, 70% train for small datasets
    
    if problem == "classification" and y.nunique(dropna=False) > 1:
        vc = y.value_counts(dropna=False)
        n_classes = len(vc)
        min_class_count = int(vc.min()) if not vc.empty else 1
        
        # Calculate expected test set size
        expected_test_size = int(len(df) * test_size)
        
        # Only use stratification if:
        # 1. We have at least 2 samples per class in the original data
        # 2. We have enough test samples to represent all classes
        if min_class_count >= 2 and expected_test_size >= n_classes:
            stratify = y
        # For very small datasets with all classes represented once, don't stratify
    
    return train_test_split(X, y, test_size=test_size, random_state=42, stratify=stratify)


def explain_model(df: pd.DataFrame, target: str, problem: str, model: Any, top_n: int = 15) -> Dict[str, Any]:
    X_train, X_test, y_train, y_test = _split(df, target, problem)
    est = clone(model)
    est.fit(X_train, y_train)

    score_metric = "f1_weighted" if problem == "classification" else "r2"
    scoring = "f1_weighted" if problem == "classification" else "r2"

    out: Dict[str, Any] = {
        "score_metric": score_metric,
        "permutation_importance": [],
        "shap": {"enabled": False, "reason": "shap unavailable" if shap is None else "not_computed"},
    }

    try:
        pi = permutation_importance(est, X_test, y_test, n_repeats=5, random_state=42, scoring=scoring)
        feats: List[Dict[str, Any]] = []
        for idx, col in enumerate(X_test.columns):
            feats.append(
                {
                    "feature": str(col),
                    "importance_mean": float(pi.importances_mean[idx]),
                    "importance_std": float(pi.importances_std[idx]),
                }
            )
        feats.sort(key=lambda x: x["importance_mean"], reverse=True)
        out["permutation_importance"] = feats[:top_n]
    except Exception as exc:
        out["permutation_importance_error"] = str(exc)

    if shap is not None:
        try:
            sample = X_test.head(min(100, len(X_test)))
            explainer = shap.Explainer(est.predict, sample)
            sv = explainer(sample)
            mean_abs = np.abs(sv.values).mean(axis=0)
            rows = []
            for i, col in enumerate(sample.columns):
                rows.append({"feature": str(col), "mean_abs_shap": float(mean_abs[i])})
            rows.sort(key=lambda x: x["mean_abs_shap"], reverse=True)
            out["shap"] = {
                "enabled": True,
                "global_mean_abs": rows[:top_n],
                "sample_size": int(len(sample)),
            }
        except Exception as exc:
            out["shap"] = {"enabled": False, "reason": _clean_shap_reason(exc)}

    return out
