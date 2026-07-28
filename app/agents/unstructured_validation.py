from __future__ import annotations

from app.agents.base import BaseAgent
from app.graph.state import AgentState
from app.services.rag_validation import validate_rag_answer


class UnstructuredValidationAgent(BaseAgent):
    name = "unstructured_validation"

    def run(self, state: AgentState) -> AgentState:
        query = state.get("validation_query", "What is this document about?")
        chunks = state.get("enriched_chunks", [])

        context = "\n".join([c.summary for c in chunks[:4]])
        answer = context[:400] if context else "No context available."

        state["rag_validation"] = validate_rag_answer(query=query, context=context, answer=answer)
        return state
