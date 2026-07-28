from __future__ import annotations

import io
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from app.agents.base import BaseAgent
from app.graph.state import AgentState
from app.services.analytics_helpers import figure_to_base64, serialize_result


class AnalyticsExecutionAgent(BaseAgent):
    name = "analytics_execution"

    @staticmethod
    def _looks_like_file_io_error(code: str, exc: Exception) -> bool:
        msg = str(exc).lower()
        file_markers = [
            "no such file or directory",
            "filenotfounderror",
            "does not exist",
            "cannot find the file",
        ]
        io_patterns = [
            r"\bread_csv\s*\(",
            r"\bread_excel\s*\(",
            r"\bread_json\s*\(",
            r"\bread_table\s*\(",
            r"\bopen\s*\(",
        ]
        return any(m in msg for m in file_markers) and any(re.search(p, code) for p in io_patterns)

    def run(self, state: AgentState) -> AgentState:
        df = state["dataframe"]
        code = state.get("analytics_code", "")
        should_plot = state.get("analytics_should_plot", False)

        env = {"pd": pd, "np": np, "df": df.copy(), "plt": plt, "io": io}
        raw_result = None
        error = None

        try:
            exec(code, {"__builtins__": __builtins__}, env)
            if "result" in env:
                raw_result = env["result"]
            elif should_plot:
                num_cols = df.select_dtypes(include=["number"]).columns.tolist()
                if num_cols:
                    fig, ax = plt.subplots(figsize=(6, 4))
                    df[num_cols[0]].dropna().plot(kind="hist", ax=ax, title=num_cols[0])
                    raw_result = fig
                else:
                    raw_result = "No numeric columns available for plotting."
            else:
                raw_result = df.describe(include="all").transpose()
        except Exception as exc:
            if self._looks_like_file_io_error(code, exc):
                if should_plot:
                    num_cols = df.select_dtypes(include=["number"]).columns.tolist()
                    if num_cols:
                        fig, ax = plt.subplots(figsize=(6, 4))
                        df[num_cols[0]].dropna().plot(kind="hist", ax=ax, title=num_cols[0])
                        raw_result = fig
                    else:
                        raw_result = "No numeric columns available for plotting."
                else:
                    raw_result = df.describe(include="all").transpose()
                error = None
            else:
                error = str(exc)

        if error is not None:
            state["analytics_execution"] = {
                "ok": False,
                "error": error,
                "result": {"type": "text", "value": f"Execution error: {error}"},
                "plot_base64": None,
            }
            return state

        plot_base64 = None
        if should_plot:
            try:
                if isinstance(raw_result, plt.Axes):
                    fig = raw_result.figure
                    plot_base64 = figure_to_base64(fig)
                elif isinstance(raw_result, plt.Figure):
                    plot_base64 = figure_to_base64(raw_result)
            except Exception:
                plot_base64 = None

        state["analytics_execution"] = {
            "ok": True,
            "error": None,
            "result": serialize_result(raw_result),
            "plot_base64": plot_base64,
        }
        return state
