from __future__ import annotations

from app.agents.base import BaseAgent
from app.graph.state import AgentState
from app.services.unstructured_processing import anonymize_text, convert_to_markdown


class UnstructuredIngestionAgent(BaseAgent):
    name = "unstructured_ingestion"

    def run(self, state: AgentState) -> AgentState:
        file_path = state.get("document_path")
        if not file_path:
            raise ValueError("document_path is required for unstructured ingestion")

        markdown = convert_to_markdown(file_path)
        cleaned = anonymize_text(markdown)
        state["document_markdown"] = cleaned
        return state
