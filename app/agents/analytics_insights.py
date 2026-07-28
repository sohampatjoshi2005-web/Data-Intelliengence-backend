from __future__ import annotations

from app.agents.base import BaseAgent
from app.graph.state import AgentState
from app.services.analytics_helpers import build_insights_fallback, is_low_quality_text
from app.services.analytics_llm import AnalyticsModelConfig, chat_complete


class AnalyticsInsightsAgent(BaseAgent):
    name = "analytics_insights"

    def run(self, state: AgentState) -> AgentState:
        df = state["dataframe"]
        query = str(state.get("analytics_query", "") or "")
        fallback = build_insights_fallback(df)
        col_markers = [str(c) for c in df.columns[:8]]
        prompt = (
            "You are an analytics copilot.\n"
            f"User query: {query or 'N/A'}\n"
            f"Dataset rows: {len(df)}\n"
            f"Columns: {list(df.columns)}\n"
            "Return plain text with exactly two sections:\n"
            "Summary:\n"
            "- 2 to 4 concise lines grounded in this dataset and query.\n"
            "Analysis Questions:\n"
            "- exactly 3 concrete follow-up questions using exact column names where possible.\n"
            "No placeholders, no apologies, no generic filler."
        )

        insight = fallback
        try:
            cfg = AnalyticsModelConfig()
            llm_insight = chat_complete(
                prompt=prompt,
                temperature=cfg.insights_temperature,
                max_tokens=cfg.insights_max_tokens,
                trace_name="analytics_insights",
                provider=state.get("llm_provider", ""),
            )
            has_required_sections = ("Summary" in llm_insight) and ("Analysis Questions" in llm_insight)
            mentions_columns = any(c in llm_insight for c in col_markers)
            if (not is_low_quality_text(llm_insight)) and has_required_sections and mentions_columns:
                insight = llm_insight
        except Exception:
            pass

        state["analytics_insights"] = insight
        return state
