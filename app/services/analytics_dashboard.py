from __future__ import annotations

from typing import Any, Dict, List

import pandas as pd


def build_dashboard_stats(df: pd.DataFrame) -> Dict[str, List[Dict[str, Any]]]:
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    cat_cols = [c for c in df.columns if c not in numeric_cols]

    numeric_summary: List[Dict[str, Any]] = []
    for col in numeric_cols:
        s = pd.to_numeric(df[col], errors="coerce").dropna()
        if s.empty:
            continue
        numeric_summary.append(
            {
                "column": col,
                "count": int(s.count()),
                "mean": float(s.mean()),
                "std": float(s.std(ddof=1)) if len(s) > 1 else 0.0,
                "min": float(s.min()),
                "max": float(s.max()),
                "median": float(s.median()),
            }
        )

    categorical_summary: List[Dict[str, Any]] = []
    for col in cat_cols:
        s = df[col].dropna().astype(str)
        if s.empty:
            continue
        top_counts = s.value_counts().head(10)
        categorical_summary.append(
            {
                "column": col,
                "unique": int(s.nunique()),
                "top_values": [{"value": k, "count": int(v)} for k, v in top_counts.items()],
            }
        )

    correlations: List[Dict[str, Any]] = []
    if len(numeric_cols) >= 2:
        corr = df[numeric_cols].corr(numeric_only=True)
        for c1 in numeric_cols:
            for c2 in numeric_cols:
                if c1 == c2:
                    continue
                correlations.append(
                    {
                        "pair": f"{c1} vs {c2}",
                        "corr": float(corr.loc[c1, c2]) if pd.notna(corr.loc[c1, c2]) else 0.0,
                    }
                )

    return {
        "numeric_summary": numeric_summary,
        "categorical_summary": categorical_summary,
        "correlations": correlations,
    }
