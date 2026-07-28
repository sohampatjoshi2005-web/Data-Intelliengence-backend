from __future__ import annotations

import base64
import io
import re
from typing import Any, Dict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_FIGSIZE = (6, 4)
DEFAULT_DPI = 100
MAX_RESULT_DISPLAY_LENGTH = 300
MAX_LIST_PREVIEW_ITEMS = 100
MAX_DICT_ITEMS = 200


def extract_first_code_block(text: str) -> str:
    match = re.search(r"```python\n(.*?)\n```", text, re.DOTALL)
    return match.group(1).strip() if match else ""


def is_plot_query(query: str) -> bool:
    q = query.lower()
    plot_terms = [
        "plot",
        "chart",
        "graph",
        "distribution",
        "histogram",
        "scatter",
        "bar",
        "line",
        "boxplot",
        "pie",
        "heatmap",
        "visual",
    ]
    return any(t in q for t in plot_terms)


def serialize_result(result: Any) -> Dict[str, Any]:
    if isinstance(result, pd.DataFrame):
        return {"type": "dataframe", "value": result.head(100).to_dict(orient="records")}
    if isinstance(result, pd.Series):
        return {"type": "series", "value": result.head(100).to_dict()}
    if isinstance(result, np.ndarray):
        return {"type": "ndarray", "value": result[:100].tolist()}
    if isinstance(result, dict):
        return {"type": "object", "value": _jsonable_value(result)}
    if isinstance(result, list):
        return {"type": "list", "value": _jsonable_value(result[:MAX_LIST_PREVIEW_ITEMS])}
    return {"type": "text", "value": str(result)[:MAX_RESULT_DISPLAY_LENGTH]}


def _jsonable_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, (pd.Series, np.ndarray)):
        return _jsonable_value(value.tolist())
    if isinstance(value, pd.DataFrame):
        return value.head(MAX_LIST_PREVIEW_ITEMS).to_dict(orient="records")
    if isinstance(value, dict):
        out = {}
        for idx, (k, v) in enumerate(value.items()):
            if idx >= MAX_DICT_ITEMS:
                break
            out[str(k)] = _jsonable_value(v)
        return out
    if isinstance(value, (list, tuple, set)):
        return [_jsonable_value(v) for v in list(value)[:MAX_LIST_PREVIEW_ITEMS]]
    return str(value)[:MAX_RESULT_DISPLAY_LENGTH]


def is_low_quality_text(text: str) -> bool:
    t = (text or "").strip().lower()
    if not t:
        return True
    bad_markers = [
        "reasoning unavailable",
        "i don't have the results",
        "i do not have the results",
        "no reasoning available",
        "insights unavailable",
        "cannot provide",
        "unable to",
    ]
    return any(marker in t for marker in bad_markers)


def summarize_execution_result(result: Dict[str, Any]) -> str:
    rtype = result.get("type", "text")
    value = result.get("value")
    if rtype == "dataframe" and isinstance(value, list):
        rows = len(value)
        cols = sorted({k for row in value if isinstance(row, dict) for k in row.keys()})
        return f"Result type=dataframe, preview_rows={rows}, columns={cols}"
    if rtype == "series" and isinstance(value, dict):
        keys = list(value.keys())[:10]
        return f"Result type=series, items={len(value)}, sample_keys={keys}"
    if rtype == "ndarray" and isinstance(value, list):
        return f"Result type=ndarray, length={len(value)}, sample={value[:10]}"
    if rtype == "object" and isinstance(value, dict):
        keys = list(value.keys())[:20]
        return f"Result type=object, keys={keys}"
    if rtype == "list" and isinstance(value, list):
        first_type = type(value[0]).__name__ if value else "none"
        return f"Result type=list, length={len(value)}, first_item_type={first_type}"
    text_value = str(value or "")
    return f"Result type=text, value={text_value[:MAX_RESULT_DISPLAY_LENGTH]}"


def build_reasoning_fallback(query: str, execution: Dict[str, Any]) -> str:
    if not execution:
        return "No execution payload available to derive reasoning."
    if not execution.get("ok", False):
        err = execution.get("error") or "unknown execution error"
        return f"Execution failed while answering '{query}': {err}. Verify query intent and generated code."

    result = execution.get("result", {})
    summary = summarize_execution_result(result)
    plot_present = bool(execution.get("plot_base64"))
    if result.get("type") == "object" and isinstance(result.get("value"), dict):
        obj = result.get("value", {})
        notable_keys = [k for k in ("class_distribution", "anova", "model_metrics", "conclusions") if k in obj]
        if notable_keys:
            return (
                f"Query analyzed: '{query}'. Structured result generated with sections {notable_keys}. "
                "Interpretation should be read from numeric tables and model metrics in execution output."
            )
    return (
        f"Query analyzed: '{query}'. {summary}. "
        f"Query-specific plot generated={plot_present}. "
        "Dataset-level visuals are generated separately from the query result. "
        "Interpretation is based on executed output, not template text."
    )


def build_insights_fallback(df: pd.DataFrame) -> str:
    row_count = len(df)
    col_count = len(df.columns)
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = [c for c in df.columns if c not in numeric_cols]
    missing_cells = int(df.isna().sum().sum())
    dup_count = int(df.duplicated().sum())

    target_col = ""
    for candidate in ("target", "label", "class", "species", "variety"):
        if candidate in {str(c).lower() for c in df.columns}:
            target_col = next(str(c) for c in df.columns if str(c).lower() == candidate)
            break

    summary_lines: list[str] = [
        f"This dataset has {row_count} rows and {col_count} columns.",
        f"It contains {len(numeric_cols)} numeric columns and {len(categorical_cols)} non-numeric columns.",
        f"Data quality check: missing cells={missing_cells}, duplicate rows={dup_count}.",
    ]

    if target_col:
        vc = df[target_col].astype(str).value_counts(dropna=False).head(5)
        dist_txt = ", ".join([f"{k}={int(v)}" for k, v in vc.items()])
        summary_lines.append(f"Likely target/group column is '{target_col}' with distribution: {dist_txt}.")

    if numeric_cols:
        top_num = numeric_cols[:3]
        stat_bits = []
        for c in top_num:
            s = pd.to_numeric(df[c], errors="coerce").dropna()
            if len(s) == 0:
                continue
            stat_bits.append(
                f"{c} (mean={float(s.mean()):.3f}, std={float(s.std(ddof=1)) if len(s) > 1 else 0.0:.3f}, "
                f"min={float(s.min()):.3f}, max={float(s.max()):.3f})"
            )
        if stat_bits:
            summary_lines.append("Sample numeric profile: " + "; ".join(stat_bits) + ".")

    if categorical_cols:
        top_cat = categorical_cols[0]
        vc = df[top_cat].astype(str).value_counts(dropna=False).head(5)
        cat_txt = ", ".join([f"{k}={int(v)}" for k, v in vc.items()])
        summary_lines.append(f"Top categories for '{top_cat}': {cat_txt}.")

    questions = [
        "Which features provide the strongest separation between groups/classes?",
        "Are there outliers, skewed distributions, or data-quality issues that could bias analysis?",
        "Which baseline model and validation setup should be used first for robust performance measurement?",
    ]
    if target_col:
        questions[0] = f"Which features are most predictive of '{target_col}' across its groups?"
    if numeric_cols and len(numeric_cols) >= 2:
        questions[1] = f"Which relationships among {numeric_cols[:3]} indicate multicollinearity or strong signal?"

    return (
        "Summary:\n"
        + "\n".join([f"- {line}" for line in summary_lines])
        + "\n\nAnalysis Questions:\n"
        + "\n".join([f"- {q}" for q in questions[:3]])
    )


def figure_to_base64(fig: plt.Figure) -> str:
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=DEFAULT_DPI, bbox_inches="tight")
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode("utf-8")
