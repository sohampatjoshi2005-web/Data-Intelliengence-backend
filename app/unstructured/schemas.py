from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, TypedDict


@dataclass
class KBChunk:
    chunk_id: str
    chunk_text: str
    contextual_text: str
    summary: str
    tags: List[str]
    questions: List[str]
    keywords: List[str]
    tod_items: List[str]
    metadata: Dict[str, Any]
    sentiment: float
    embedding: List[float]


class KBState(TypedDict, total=False):
    dataset_id: str
    file_path: str
    llm_provider: str
    raw_text: str
    markdown_text: str
    language: str
    standardized_text: str
    redacted_text: str
    headings: List[str]
    entities: List[str]
    chunks_text: List[str]
    contextual_chunks: List[str]
    chunks: List[KBChunk]
    warnings: List[str]
    persistence: Dict[str, Any]
    fast_mode: bool
    chunk_cap: int
    chunk_size_tokens: int
    chunk_overlap_tokens: int
    skip_ner: bool
    skip_pii: bool
    skip_enrichment: bool
    enrichment_batch_size: int
    enrichment_workers: int
    embedding_batch_size: int
    embedding_workers: int
