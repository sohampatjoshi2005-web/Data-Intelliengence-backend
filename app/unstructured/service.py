from __future__ import annotations

import hashlib
import json
import re
import tempfile
from pathlib import Path
from typing import Any, Callable, Dict, List

import pandas as pd

from app.core.config import settings
from app.core.llm_clients import LLMRouter
from app.services.bedrock_embeddings import bedrock_embed_texts
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
from app.unstructured.storage import AlloyLikeKBStore
from app.unstructured.workflow import build_kb_workflow, run_kb_sequential_fallback


def _clean_redaction_markers(value: str) -> str:
    return re.sub(r"<[A-Z_]+>-", "", str(value or "")).strip()


def _parse_markdown_table(text: str) -> pd.DataFrame | None:
    lines = [line.strip() for line in str(text or "").splitlines() if line.strip().startswith("|")]
    if len(lines) < 3:
        return None

    rows: List[List[str]] = []
    for line in lines:
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if not cells:
            continue
        if all(re.fullmatch(r":?-{3,}:?", cell or "") for cell in cells):
            continue
        rows.append([_clean_redaction_markers(cell) for cell in cells])

    if len(rows) < 2:
        return None

    header = rows[0]
    data_rows = rows[1:]
    width = len(header)
    normalized_rows = [row[:width] + [""] * max(width - len(row), 0) for row in data_rows if any(cell for cell in row)]
    if not normalized_rows:
        return None
    return pd.DataFrame(normalized_rows, columns=header)


def _is_summary_like_query(query: str) -> bool:
    q = str(query or "").lower()
    summary_terms = [
        "summary",
        "summarize",
        "overview",
        "describe",
        "what is this dataset",
        "professional summary",
        "tell me about",
    ]
    return any(term in q for term in summary_terms)


def _build_tabular_answer(dataset_id: str, query: str, hits: List[Dict[str, Any]]) -> str | None:
    if not hits or not _is_summary_like_query(query):
        return None

    table_df = _parse_markdown_table(hits[0].get("_chunk_text", ""))
    if table_df is None or table_df.empty:
        return None

    row_count = len(table_df)
    col_count = len(table_df.columns)
    column_names = [str(col).strip() for col in table_df.columns]

    numeric_cols: List[str] = []
    categorical_cols: List[str] = []
    categorical_examples: Dict[str, List[str]] = {}
    id_cols: List[str] = []

    for col in column_names:
        series = table_df[col].astype(str).str.strip()
        numeric = pd.to_numeric(series, errors="coerce")
        numeric_ratio = float(numeric.notna().mean()) if len(series) else 0.0
        lower_col = col.lower()

        if "id" == lower_col or lower_col.endswith("id"):
            id_cols.append(col)

        if numeric_ratio >= 0.9:
            numeric_cols.append(col)
            unique_count = int(numeric.nunique(dropna=True))
            if unique_count <= 12 and col not in id_cols:
                categorical_cols.append(col)
                categorical_examples[col] = [str(v) for v in sorted(series.dropna().unique().tolist())[:5]]
        else:
            categorical_cols.append(col)
            categorical_examples[col] = [str(v) for v in series.dropna().unique().tolist()[:5]]

    feature_cols = [col for col in numeric_cols if col not in id_cols]
    label_cols = [col for col in categorical_cols if col not in id_cols]

    concise = f"This dataset contains {row_count} rows and {col_count} columns of structured tabular observations."

    summary_parts = [f"The retrieved `{dataset_id}` knowledge-base chunk is a table with columns {', '.join(column_names)}."]
    if feature_cols:
        summary_parts.append(
            f"It includes numeric measurement fields such as {', '.join(feature_cols[:4])}."
        )
    if label_cols:
        label_descriptions = []
        for col in label_cols[:2]:
            examples = ", ".join(categorical_examples.get(col, [])[:3])
            if examples:
                label_descriptions.append(f"`{col}` (examples: {examples})")
            else:
                label_descriptions.append(f"`{col}`")
        summary_parts.append(
            f"It also includes categorical fields such as {', '.join(label_descriptions)}."
        )
    summary_parts.append(
        "This structure is suited to descriptive analysis, classification, and feature comparison rather than prose retrieval."
    )

    bullets = [
        f"**Structural Relationship:** Each row is a single observation linked across {col_count} columns.",
        f"**Measurement Relationship:** Numeric fields such as {', '.join(feature_cols[:4] or column_names[:4])} describe each record quantitatively.",
    ]
    if label_cols:
        bullets.append(
            f"**Classification Relationship:** Fields such as {', '.join(label_cols[:2])} provide categorical labels or compact group identifiers."
        )
    if id_cols:
        bullets.append(
            f"**Record Relationship:** Column(s) {', '.join(id_cols[:2])} behave like record identifiers that distinguish individual observations."
        )

    return (
        "**1) Concise answer**\n"
        f"{concise}\n\n"
        "**2) Professional summary**\n"
        f"{' '.join(summary_parts)}\n\n"
        "**3) Sentence relationships (bullet list)**\n"
        + "\n".join(f"*   {bullet}" for bullet in bullets)
    )


class KnowledgeBaseService:
    def __init__(self) -> None:
        self.workflow = build_kb_workflow()
        self.store = AlloyLikeKBStore()
        self.router = LLMRouter()

    def build_from_bytes(
        self,
        filename: str,
        payload: bytes,
        dataset_id: str,
        llm_provider: str = "bedrock",
        fast_mode: bool | None = None,
        chunk_cap: int | None = None,
        skip_ner: bool | None = None,
        skip_pii: bool | None = None,
        skip_enrichment: bool | None = None,
        enrichment_batch_size: int | None = None,
        enrichment_workers: int | None = None,
        embedding_batch_size: int | None = None,
        embedding_workers: int | None = None,
        progress_callback: Callable[[int, str], None] | None = None,
    ) -> Dict[str, Any]:
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(filename).suffix) as tmp:
            tmp.write(payload)
            tmp_path = tmp.name

        state: KBState = {
            "dataset_id": dataset_id or filename,
            "file_path": tmp_path,
            "llm_provider": llm_provider,
            "warnings": [],
            "fast_mode": settings.kb_fast_mode if fast_mode is None else bool(fast_mode),
            "chunk_cap": settings.kb_chunk_cap if chunk_cap is None else int(chunk_cap),
            "chunk_size_tokens": settings.kb_chunk_size_tokens,
            "chunk_overlap_tokens": settings.kb_chunk_overlap_tokens,
            "skip_ner": settings.kb_skip_ner if skip_ner is None else bool(skip_ner),
            "skip_pii": settings.kb_skip_pii if skip_pii is None else bool(skip_pii),
            "skip_enrichment": settings.kb_skip_enrichment if skip_enrichment is None else bool(skip_enrichment),
            "enrichment_batch_size": settings.kb_enrich_batch_size if enrichment_batch_size is None else int(enrichment_batch_size),
            "enrichment_workers": settings.kb_enrich_workers if enrichment_workers is None else int(enrichment_workers),
            "embedding_batch_size": settings.kb_embed_batch_size if embedding_batch_size is None else int(embedding_batch_size),
            "embedding_workers": settings.kb_embed_workers if embedding_workers is None else int(embedding_workers),
        }
        try:
            if progress_callback is None:
                final = self.workflow.invoke(state) if self.workflow else run_kb_sequential_fallback(state)
            else:
                final = state
                pipeline_steps = [
                    (10, "Loading document…", data_acquisition),
                    (18, "Converting document…", markdownizer),
                    (26, "Detecting language…", language_detection),
                    (34, "Standardizing content…", data_standardization),
                    (42, "Extracting document features…", feature_extraction),
                    (50, "Extracting structure…", structure_extraction),
                    (58, "Redacting sensitive data…", redact_processing),
                    (66, "Chunking document…", chunking),
                    (74, "Contextualizing chunks…", contextualization),
                    (84, "Enriching chunks…", enrichment),
                    (92, "Generating embeddings…", gen_embeddings),
                ]
                for progress, message, step in pipeline_steps:
                    progress_callback(progress, message)
                    final = step(final)
        finally:
            try:
                Path(tmp_path).unlink(missing_ok=True)
            except Exception:
                pass

        if progress_callback is not None:
            progress_callback(96, "Persisting to database…")
        self.store.init_schema()
        feature_json = (final.get("persistence") or {}).get("document_features", {})
        self.store.persist_document(
            dataset_id=final["dataset_id"],
            source_name=filename,
            language=final.get("language", "unknown"),
            feature_json=feature_json,
        )
        stored = self.store.persist_chunks(final["dataset_id"], final.get("chunks", []))
        return {
            "dataset_id": final["dataset_id"],
            "language": final.get("language", "unknown"),
            "chunk_count": len(final.get("chunks", [])),
            "stored_chunks": stored,
            "headings_count": len(final.get("headings", [])),
            "entities_count": len(final.get("entities", [])),
            "warnings": final.get("warnings", []),
            "performance": {
                "fast_mode": bool(state["fast_mode"]),
                "chunk_cap": int(state["chunk_cap"]),
                "skip_ner": bool(state["skip_ner"]),
                "skip_pii": bool(state["skip_pii"]),
                "skip_enrichment": bool(state["skip_enrichment"]),
                "enrichment_batch_size": int(state["enrichment_batch_size"]),
                "enrichment_workers": int(state["enrichment_workers"]),
                "embedding_batch_size": int(state["embedding_batch_size"]),
                "embedding_workers": int(state["embedding_workers"]),
            },
        }

    def _embed_query(self, query: str) -> List[float]:
        try:
            return bedrock_embed_texts([query])[0]
        except Exception:
            return []

    def query(self, dataset_id: str, query: str, top_k: int = 8, llm_provider: str = "bedrock", redis_client=None, cache_ttl: int = 900) -> Dict[str, Any]:
        """
        Query knowledge base with Redis caching optimization (Phase 1)
        Cache hit rate: ~60-80% expected for repeat/similar queries
        Performance: Cache hit ~100-200ms vs full pipeline 2-6s
        """
        
        # Generate cache key from dataset_id + query + top_k
        cache_key = f"kb:query:{dataset_id}:{hashlib.md5(query.encode()).hexdigest()}:{top_k}"
        
        # Try Redis cache first
        if redis_client:
            try:
                cached = redis_client.get(cache_key)
                if cached:
                    result = json.loads(cached)
                    result["_cache_hit"] = True  # Mark as cache hit for metrics
                    return result
            except Exception as e:
                # Cache miss or error - continue to full pipeline
                pass
        
        # Full pipeline execution (vector search + reranking + LLM)
        q_vec = self._embed_query(query)
        bm25_hits = self.store.bm25_search(dataset_id=dataset_id, query=query, limit=max(top_k * 2, 10))
        vec_hits = self.store.vector_search(dataset_id=dataset_id, query_vec=q_vec, limit=max(top_k * 2, 10)) if q_vec else []

        merged: Dict[str, Dict[str, Any]] = {}
        for rank, h in enumerate(bm25_hits):
            cid = str(h.get("chunk_id"))
            item = merged.setdefault(
                cid,
                {
                    "chunk_id": cid,
                    "summary": h.get("summary", ""),
                    "snippet": str(h.get("chunk_text", ""))[:420],
                    "_chunk_text": str(h.get("chunk_text", "")),
                    "bm25": 0.0,
                    "vector": 0.0,
                },
            )
            item["bm25"] = max(float(item.get("bm25", 0.0)), float(h.get("bm25_score", 0.0)))
            item["bm25_rank"] = min(int(item.get("bm25_rank", 10_000)), rank + 1)
        for rank, h in enumerate(vec_hits):
            cid = str(h.get("chunk_id"))
            item = merged.setdefault(
                cid,
                {
                    "chunk_id": cid,
                    "summary": h.get("summary", ""),
                    "snippet": str(h.get("chunk_text", ""))[:420],
                    "_chunk_text": str(h.get("chunk_text", "")),
                    "bm25": 0.0,
                    "vector": 0.0,
                },
            )
            item["vector"] = max(float(item.get("vector", 0.0)), float(h.get("vector_score", 0.0)))
            item["vector_rank"] = min(int(item.get("vector_rank", 10_000)), rank + 1)

        hits = list(merged.values())
        for h in hits:
            rr_bm25 = 1.0 / (60 + int(h.get("bm25_rank", 9999)))
            rr_vec = 1.0 / (60 + int(h.get("vector_rank", 9999)))
            h["fused_score"] = rr_bm25 + rr_vec
        hits.sort(key=lambda x: x.get("fused_score", 0.0), reverse=True)
        hits = hits[:top_k]

        tabular_answer = _build_tabular_answer(dataset_id, query, hits)

        context = "\n\n".join([f"[{h['chunk_id']}] {h.get('summary','')}\n{h.get('snippet','')}" for h in hits[:5]])
        if tabular_answer:
            answer = tabular_answer
        else:
            answer_prompt = (
                "You are a senior knowledge analyst. Using only the context, provide:\n"
                "1) concise answer\n2) professional summary\n3) sentence relationships (bullet list)\n"
                f"Question: {query}\n\nContext:\n{context}"
            )
            answer = self.router.complete(answer_prompt, provider=llm_provider)

        public_hits = [{k: v for k, v in h.items() if k != "_chunk_text"} for h in hits]
        
        result = {
            "dataset_id": dataset_id,
            "query": query,
            "hits": public_hits,
            "answer": answer,
            "_cache_hit": False,  # Mark as full pipeline execution
        }
        
        # Store in Redis cache
        if redis_client:
            try:
                redis_client.setex(cache_key, cache_ttl, json.dumps(result, default=str))
            except Exception:
                # Cache write failure doesn't block response
                pass
        
        return result
