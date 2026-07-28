from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from app.agents.base import BaseAgent
from app.graph.state import AgentState


class DataUnderstandingAgent(BaseAgent):
    name = "data_understanding"

    def _suggest_target(self, df: pd.DataFrame) -> Optional[str]:
        candidates = [c for c in df.columns if df[c].nunique() < len(df)]
        return candidates[-1] if candidates else (df.columns[-1] if len(df.columns) > 0 else None)

    def _column_types(self, df: pd.DataFrame) -> Dict[str, str]:
        out: Dict[str, str] = {}
        for col in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                out[col] = "time_series"
            elif pd.api.types.is_numeric_dtype(df[col]):
                out[col] = "numeric"
            else:
                out[col] = "categorical"
        return out

    def _detect_leakage(self, df: pd.DataFrame, target: Optional[str]) -> list[str]:
        if not target or target not in df.columns:
            return []

        leaks: list[str] = []
        for col in df.columns:
            if col == target:
                continue
            if df[col].equals(df[target]):
                leaks.append(f"{col} is identical to target")
                continue

            if pd.api.types.is_numeric_dtype(df[col]) and pd.api.types.is_numeric_dtype(df[target]):
                corr = df[[col, target]].corr(numeric_only=True).iloc[0, 1]
                if np.isfinite(corr) and abs(corr) > 0.98:
                    leaks.append(f"{col} is highly correlated with target ({corr:.3f})")
        return leaks

    def run(self, state: AgentState) -> AgentState:
        df = state["dataframe"]
        target = state.get("target_column") or self._suggest_target(df)
        column_types = self._column_types(df)
        ts_cols = [c for c, t in column_types.items() if t == "time_series"]
        if not ts_cols:
            for c in df.columns:
                if c == target:
                    continue
                if pd.api.types.is_numeric_dtype(df[c]):
                    continue
                try:
                    parsed = pd.to_datetime(df[c], errors="coerce")
                    if parsed.notna().mean() > 0.8:
                        ts_cols.append(c)
                        break
                except Exception:
                    continue

        profile = {
            "shape": {"rows": int(df.shape[0]), "columns": int(df.shape[1])},
            "column_types": column_types,
            "missing_values": df.isna().sum().to_dict(),
            "duplicate_rows": int(df.duplicated().sum()),
            "target_suggestion": target,
            "leakage_signals": self._detect_leakage(df, target),
            "time_series_candidates": ts_cols,
            "recommended_validation_strategy": "time_aware_split" if ts_cols else "stratified_or_random_split",
            "describe": df.describe(include="all").fillna("").astype(str).to_dict(),
        }

        state["target_column"] = target
        state["data_profile"] = profile
        return state
