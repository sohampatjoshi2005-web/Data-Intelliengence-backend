from __future__ import annotations

from app.unstructured.pipeline import (
    chunking,
    contextualization,
    data_acquisition,
    data_standardization,
    enrichment,
    feature_extraction,
    gen_embeddings,
    language_detection,
    markdownizer,
    redact_processing,
    structure_extraction,
)
from app.unstructured.schemas import KBState

try:
    from langgraph.graph import END, StateGraph
except Exception:
    END = "END"
    StateGraph = None


def build_kb_workflow():
    if StateGraph is None:
        return None

    graph = StateGraph(KBState)
    graph.add_node("data_acquisition", data_acquisition)
    graph.add_node("markdownizer", markdownizer)
    graph.add_node("language_detection", language_detection)
    graph.add_node("data_standardization", data_standardization)
    graph.add_node("feature_extraction", feature_extraction)
    graph.add_node("structure_extraction", structure_extraction)
    graph.add_node("redact_processing", redact_processing)
    graph.add_node("chunking", chunking)
    graph.add_node("contextualization", contextualization)
    graph.add_node("enrichment", enrichment)
    graph.add_node("gen_embeddings", gen_embeddings)

    graph.set_entry_point("data_acquisition")
    graph.add_edge("data_acquisition", "markdownizer")
    graph.add_edge("markdownizer", "language_detection")
    graph.add_edge("language_detection", "data_standardization")
    graph.add_edge("data_standardization", "feature_extraction")
    graph.add_edge("feature_extraction", "structure_extraction")
    graph.add_edge("structure_extraction", "redact_processing")
    graph.add_edge("redact_processing", "chunking")
    graph.add_edge("chunking", "contextualization")
    graph.add_edge("contextualization", "enrichment")
    graph.add_edge("enrichment", "gen_embeddings")
    graph.add_edge("gen_embeddings", END)
    return graph.compile()


def run_kb_sequential_fallback(state: KBState) -> KBState:
    for node in [
        data_acquisition,
        markdownizer,
        language_detection,
        data_standardization,
        feature_extraction,
        structure_extraction,
        redact_processing,
        chunking,
        contextualization,
        enrichment,
        gen_embeddings,
    ]:
        state = node(state)
    return state

