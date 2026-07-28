from __future__ import annotations

from app.agents.base import BaseAgent
from app.core.llm_clients import LLMRouter
from app.graph.state import AgentState


class ProblemClassificationAgent(BaseAgent):
    name = "problem_classification"

    def __init__(self) -> None:
        self.router = LLMRouter()

    def _rules(self, text: str) -> str:
        q = text.lower()
        if any(k in q for k in ["forecast", "time series", "seasonality", "demand"]):
            return "time_series"
        if any(k in q for k in ["cluster", "segmentation", "segment"]):
            return "clustering"
        if any(k in q for k in ["rank", "recommend", "retrieval"]):
            return "ranking"
        if any(k in q for k in ["churn", "fraud", "classify", "classification", "binary"]):
            return "classification"
        if any(k in q for k in ["predict", "regress", "price", "value"]):
            return "regression"
        return "classification"

    def run(self, state: AgentState) -> AgentState:
        prompt = (state.get("business_problem") or state.get("user_prompt") or "").strip()
        provider = state.get("llm_provider", "bedrock")
        profile = state.get("data_profile", {})
        df = state.get("dataframe")
        target = state.get("target_column")

        problem_type = self._rules(prompt)
        if profile.get("time_series_candidates"):
            if problem_type in {"classification", "regression"}:
                state.setdefault("warnings", []).append(
                    "Detected time-series candidate columns; you may switch intent to time_series for forecasting workflows."
                )
        if prompt and provider in self.router.available_providers():
            llm_prompt = (
                "Classify this ML intent into one label only: "
                "classification, regression, time_series, ranking, clustering.\n"
                f"Intent: {prompt}"
            )
            response = self.router.complete(llm_prompt, provider=provider).lower().strip()
            for label in ["classification", "regression", "time_series", "ranking", "clustering"]:
                if label in response:
                    problem_type = label
                    break

        # Data-grounded override: for structured tasks, trust target nature over noisy prompt text.
        if df is not None and target in getattr(df, "columns", []):
            y = df[target]
            n_unique = int(y.nunique(dropna=False))
            if profile.get("time_series_candidates"):
                # keep time-series only when intent explicitly asks it.
                if "time" in prompt.lower() or "forecast" in prompt.lower():
                    problem_type = "time_series"
                else:
                    problem_type = "classification" if n_unique <= 15 else "regression"
            else:
                problem_type = "classification" if (not getattr(y, "dtype", None) is None and (not hasattr(y, "dtype") or str(y.dtype) == "object" or n_unique <= 15)) else "regression"

        state["problem_type"] = problem_type
        return state
