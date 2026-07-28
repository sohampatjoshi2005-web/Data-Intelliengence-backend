from __future__ import annotations

import os

from app.agents.base import BaseAgent
from app.graph.state import AgentState
from app.services.atom_tkg import build_atom_tkg
from app.services.unstructured_processing import enrich_chunks


class UnstructuredEnrichmentAgent(BaseAgent):
    name = "unstructured_enrichment"

    def run(self, state: AgentState) -> AgentState:
        markdown = state.get("document_markdown", "")
        provider = state.get("llm_provider", "bedrock")
        enable_atom = os.getenv("UNSTRUCTURED_ENABLE_ATOM", "1").lower() not in {"0", "false", "no"}
        atom_res = build_atom_tkg(markdown) if enable_atom else None
        state["atom_status"] = {
            "ok": atom_res.ok if atom_res is not None else False,
            "reason": atom_res.reason if atom_res is not None else "ATOM disabled by UNSTRUCTURED_ENABLE_ATOM",
            "atomic_fact_count": len(atom_res.atomic_facts) if atom_res is not None else 0,
        }
        state["atom_graph_payload"] = atom_res.graph_payload if atom_res is not None else {"nodes": [], "edges": []}

        # Keep enriched chunks for vector-RAG and downstream fallback.
        chunks = enrich_chunks(markdown, provider=provider)
        if not chunks and atom_res is not None and atom_res.atomic_facts:
            # Build minimal chunk wrappers from atomic facts if LLM enrichment is unavailable.
            from app.services.unstructured_processing import EnrichedChunk

            chunks = [
                EnrichedChunk(
                    chunk_id=f"atom_fact_{i}",
                    text=fact,
                    summary=fact[:220],
                    questions=[],
                    headings=[],
                    entities=[],
                    relationships=[],
                    sentiment=0.0,
                )
                for i, fact in enumerate(atom_res.atomic_facts)
            ]

        state["enriched_chunks"] = chunks
        return state
