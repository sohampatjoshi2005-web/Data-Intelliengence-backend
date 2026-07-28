from __future__ import annotations

from app.agents.base import BaseAgent
from app.graph.state import AgentState
from app.services.analytics_helpers import is_plot_query
from app.services.analytics_llm import AnalyticsModelConfig, chat_complete


class AnalyticsQueryUnderstandingAgent(BaseAgent):
    name = "analytics_query_understanding"

    def run(self, state: AgentState) -> AgentState:
        query = state.get("analytics_query", "")
        provider_hint = state.get("llm_provider", "bedrock")

        # Local rules fallback first for speed/reliability.
        should_plot = is_plot_query(query)

        # LLM check to improve intent detection.
        try:
            cfg = AnalyticsModelConfig()
            prompt = (
                "Task: Determine if the query asks for a visual chart/plot. "
                "Respond only with true or false.\n"
                f"Query: {query}"
            )
            llm_decision = chat_complete(
                prompt=prompt,
                temperature=cfg.query_understanding_temperature,
                max_tokens=cfg.query_understanding_max_tokens,
                trace_name="analytics_query_understanding",
                provider=provider_hint,
            ).strip().lower()
            if "true" in llm_decision:
                should_plot = True
            elif "false" in llm_decision:
                should_plot = False
        except Exception:
            # Keep rules decision if LLM unavailable.
            pass

        state["analytics_should_plot"] = should_plot
        state["llm_provider"] = provider_hint
        return state
