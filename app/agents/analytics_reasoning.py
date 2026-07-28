from __future__ import annotations

from app.agents.base import BaseAgent
from app.graph.state import AgentState
from app.services.analytics_helpers import build_reasoning_fallback


class AnalyticsReasoningAgent(BaseAgent):
    name = "analytics_reasoning"

    def run(self, state: AgentState) -> AgentState:
        query = state.get("analytics_query", "")
        execution = state.get("analytics_execution", {})
        # Hallucination guard: always derive reasoning from computed execution payload.
        state["analytics_reasoning"] = build_reasoning_fallback(query, execution)
        return state
