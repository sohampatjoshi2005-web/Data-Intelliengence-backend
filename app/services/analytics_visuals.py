from __future__ import annotations

import base64
import io
from typing import Any, Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
try:
    import seaborn as sns
except Exception:  # pragma: no cover - optional dependency in local dev
    sns = None


def _fig_to_base64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    buf.seek(0)
    data = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return data


def _plot_hist(ax, values: pd.Series) -> None:
    clean = values.dropna()
    if sns is not None:
        sns.histplot(clean, kde=True, ax=ax)
        return
    ax.hist(clean, bins=min(20, max(5, int(clean.nunique()) if not clean.empty else 5)), color="#4f46e5", alpha=0.8)
    ax.set_ylabel("Count")


def _plot_scatter(ax, df: pd.DataFrame, x_col: str, y_col: str) -> None:
    subset = df[[x_col, y_col]].dropna()
    if sns is not None:
        sns.scatterplot(data=subset, x=x_col, y=y_col, ax=ax)
        return
    ax.scatter(subset[x_col], subset[y_col], alpha=0.75, color="#2563eb")
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)


def _plot_heatmap(ax, corr: pd.DataFrame) -> None:
    if sns is not None:
        sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
        return
    image = ax.imshow(corr.values, cmap="coolwarm", aspect="auto")
    ax.set_xticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(corr.index)))
    ax.set_yticklabels(corr.index)
    for row_idx in range(len(corr.index)):
        for col_idx in range(len(corr.columns)):
            ax.text(col_idx, row_idx, f"{corr.iloc[row_idx, col_idx]:.2f}", ha="center", va="center", fontsize=8)
    ax.figure.colorbar(image, ax=ax, fraction=0.046, pad=0.04)


def _plot_box(ax, df: pd.DataFrame, category_col: str, numeric_col: str) -> None:
    subset = df[[category_col, numeric_col]].dropna()
    if sns is not None:
        sns.boxplot(data=subset, x=category_col, y=numeric_col, ax=ax)
        return
    grouped = [group[numeric_col].to_numpy() for _, group in subset.groupby(category_col)]
    labels = [str(label) for label, _ in subset.groupby(category_col)]
    if grouped:
        ax.boxplot(grouped, tick_labels=labels)
    ax.set_xlabel(category_col)
    ax.set_ylabel(numeric_col)


def _plot_line(ax, series: pd.DataFrame, x_col: str, y_col: str) -> None:
    if sns is not None:
        sns.lineplot(data=series, x=x_col, y=y_col, ax=ax)
        return
    ax.plot(series[x_col], series[y_col], color="#0f766e", linewidth=2)
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)


def generate_visual_pack(df: pd.DataFrame) -> List[Dict[str, Any]]:
    visuals: List[Dict[str, Any]] = []
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    if num_cols:
        col = num_cols[0]
        fig, ax = plt.subplots(figsize=(7, 4))
        _plot_hist(ax, df[col])
        ax.set_title(f"Distribution + KDE: {col}")
        visuals.append({"title": f"Distribution of {col}", "image_base64": _fig_to_base64(fig)})

    if len(num_cols) >= 2:
        fig, ax = plt.subplots(figsize=(7, 4))
        _plot_scatter(ax, df, num_cols[0], num_cols[1])
        ax.set_title(f"Scatter: {num_cols[0]} vs {num_cols[1]}")
        visuals.append({"title": "Scatter Impact", "image_base64": _fig_to_base64(fig)})

        corr = df[num_cols].corr(numeric_only=True)
        fig, ax = plt.subplots(figsize=(7, 5))
        _plot_heatmap(ax, corr)
        ax.set_title("Correlation Heatmap")
        visuals.append({"title": "Correlation Heatmap", "image_base64": _fig_to_base64(fig)})

    if cat_cols and num_cols:
        fig, ax = plt.subplots(figsize=(8, 4))
        ccol = cat_cols[0]
        ncol = num_cols[0]
        _plot_box(ax, df, ccol, ncol)
        ax.set_title(f"Boxplot: {ncol} by {ccol}")
        ax.tick_params(axis="x", rotation=25)
        visuals.append({"title": "Category Variance", "image_base64": _fig_to_base64(fig)})

        fig, ax = plt.subplots(figsize=(6, 6))
        counts = df[ccol].value_counts().head(8)
        ax.pie(counts.values, labels=counts.index, autopct="%1.1f%%")
        ax.set_title(f"Category Split: {ccol}")
        visuals.append({"title": "Category Ratio", "image_base64": _fig_to_base64(fig)})

    # Date trend if any datetime-like column exists.
    dt_col = None
    for c in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[c]):
            dt_col = c
            break
        if df[c].dtype == object:
            converted = pd.to_datetime(df[c], errors="coerce")
            if converted.notna().sum() > max(10, int(0.5 * len(df))):
                df[c] = converted
                dt_col = c
                break

    if dt_col is not None:
        trend = df.dropna(subset=[dt_col]).copy()
        if not trend.empty:
            trend["__date"] = trend[dt_col].dt.date
            series = trend.groupby("__date").size().reset_index(name="count")
            fig, ax = plt.subplots(figsize=(8, 4))
            _plot_line(ax, series, "__date", "count")
            ax.set_title(f"Time Trend by {dt_col}")
            ax.tick_params(axis="x", rotation=25)
            visuals.append({"title": "Time Trend", "image_base64": _fig_to_base64(fig)})

    return visuals
