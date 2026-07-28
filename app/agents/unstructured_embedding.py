from __future__ import annotations

from app.agents.base import BaseAgent
from app.graph.state import AgentState
from app.services.vector_store import GeneralVectorStore


class UnstructuredEmbeddingAgent(BaseAgent):
    name = "unstructured_embedding"

    def run(self, state: AgentState) -> AgentState:
        dataset_id = state.get("dataset_id") or "unstructured"
        chunks = state.get("enriched_chunks", [])

        store = GeneralVectorStore()
        count = store.add_chunks(
            dataset_id=dataset_id,
            chunks=chunks,
        )
        state["kb_store_status"] = {
            "vector_store_available": store.available(),
            "stored_chunks": count,
            "dataset_id": dataset_id,
        }
        return state
