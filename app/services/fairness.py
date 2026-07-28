from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split


def _pick_sensitive_columns(df: pd.DataFrame, target: str) -> List[str]:
    cols: List[str] = []
    for c in df.columns:
        if c == target:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            continue
        nun = int(df[c].nunique(dropna=True))
        if 2 <= nun <= 10:
            cols.append(c)
    return cols[:3]


def fairness_report(df: pd.DataFrame, target: str, problem: str, model: Any) -> Dict[str, Any]:
    sensitive_cols = _pick_sensitive_columns(df, target)
    if not sensitive_cols:
        return {
            "checked": False,
            "reason": "No low-cardinality categorical columns found for group fairness checks",
            "sensitive_columns": [],
            "group_metrics": {},
        }

    X = df.drop(columns=[target])
    y = df[target]
    stratify = y if problem == "classification" and y.nunique(dropna=False) > 1 else None
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=stratify)
    est = clone(model)
    est.fit(X_train, y_train)
    preds = est.predict(X_test)

    reports: Dict[str, Any] = {}
    for col in sensitive_cols:
        sub = pd.DataFrame({
            "group": X_test[col].astype(str).fillna("<NA>"),
            "y_true": y_test,
            "y_pred": preds,
        })
        rows = []
        for g, gdf in sub.groupby("group"):
            if len(gdf) < 5:
                continue
            if problem == "classification":
                metric = float(f1_score(gdf["y_true"], gdf["y_pred"], average="weighted", zero_division=0))
                acc = float(accuracy_score(gdf["y_true"], gdf["y_pred"]))
                rows.append({"group": g, "count": int(len(gdf)), "f1_weighted": metric, "accuracy": acc})
            else:
                mae = float(mean_absolute_error(gdf["y_true"], gdf["y_pred"]))
                r2 = float(r2_score(gdf["y_true"], gdf["y_pred"])) if len(gdf) > 1 else float("nan")
                rows.append({"group": g, "count": int(len(gdf)), "mae": mae, "r2": r2})

        rows = sorted(rows, key=lambda x: x["count"], reverse=True)
        if rows:
            if problem == "classification":
                vals = [r["f1_weighted"] for r in rows if np.isfinite(r["f1_weighted"])]
                disparity = float(max(vals) - min(vals)) if vals else None
                reports[col] = {"metric": "f1_weighted", "disparity": disparity, "groups": rows}
            else:
                vals = [r["mae"] for r in rows if np.isfinite(r["mae"])]
                disparity = float(max(vals) - min(vals)) if vals else None
                reports[col] = {"metric": "mae", "disparity": disparity, "groups": rows}

    return {
        "checked": bool(reports),
        "sensitive_columns": sensitive_cols,
        "group_metrics": reports,
    }
