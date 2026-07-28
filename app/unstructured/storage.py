from __future__ import annotations

import json
from typing import Any, Dict, List

from sqlalchemy import JSON, Column, DateTime, Float, Index, MetaData, Table, Text, create_engine, func, select, text
from sqlalchemy.engine import Engine

from app.core.config import settings
from app.unstructured.schemas import KBChunk

try:
    from pgvector.sqlalchemy import Vector
except Exception:
    Vector = None

try:
    from rank_bm25 import BM25Okapi
except Exception:
    BM25Okapi = None

try:
    import spacy
except Exception:
    spacy = None


def _clean_text_for_pg(value: Any) -> str:
    return str(value or "").replace("\x00", " ")


def _tokenize_for_bm25(text_value: str) -> List[str]:
    if not text_value:
        return []
    if spacy is not None:
        try:
            nlp = spacy.blank("en")
            doc = nlp(text_value)
            return [t.text.lower() for t in doc if t.text and not t.is_space]
        except Exception:
            pass
    return [tok.lower() for tok in text_value.split() if tok]


class AlloyLikeKBStore:
    """
    Persistence layout aligned to flow:
    1) AlloyDB metadata (kb_documents)
    2) AlloyDB pgvector (kb_chunks.embedding)
    3) AlloyDB BM25 (library-backed BM25 over persisted searchable_text)
    """

    def __init__(self, dsn: str | None = None) -> None:
        self.dsn = dsn or settings.kb_pg_dsn
        self.engine: Engine | None = create_engine(self.dsn, future=True) if self.dsn else None
        self.meta = MetaData()
        self.vector_enabled = Vector is not None
        self.documents: Table | None = None
        self.chunks: Table | None = None

    def available(self) -> bool:
        return self.engine is not None

    def init_schema(self) -> None:
        if not self.engine:
            raise RuntimeError("KB_PG_DSN is not configured")

        with self.engine.begin() as conn:
            if self.vector_enabled:
                try:
                    conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
                except Exception:
                    self.vector_enabled = False

        self.documents = Table(
            "kb_documents",
            self.meta,
            Column("dataset_id", Text, primary_key=True),
            Column("source_name", Text),
            Column("language", Text),
            Column("feature_json", JSON, nullable=False, server_default=text("'{}'::jsonb")),
            Column("created_at", DateTime(timezone=True), server_default=func.now()),
            extend_existing=True,
        )

        if self.vector_enabled:
            self.chunks = Table(
                "kb_chunks",
                self.meta,
                Column("id", Text, primary_key=True),
                Column("dataset_id", Text, nullable=False),
                Column("chunk_id", Text, nullable=False),
                Column("chunk_text", Text, nullable=False),
                Column("contextual_text", Text, nullable=False),
                Column("summary", Text),
                Column("tags", JSON, nullable=False, server_default=text("'[]'::jsonb")),
                Column("questions", JSON, nullable=False, server_default=text("'[]'::jsonb")),
                Column("keywords", JSON, nullable=False, server_default=text("'[]'::jsonb")),
                Column("tod_items", JSON, nullable=False, server_default=text("'[]'::jsonb")),
                Column("meta_json", JSON, nullable=False, server_default=text("'{}'::jsonb")),
                Column("sentiment", Float, nullable=False, server_default=text("0")),
                Column("searchable_text", Text, nullable=False),
                Column("search_tokens", JSON, nullable=False, server_default=text("'[]'::jsonb")),
                Column("embedding", Vector(768)),
                Column("created_at", DateTime(timezone=True), server_default=func.now()),
                Index("idx_kb_chunks_dataset", "dataset_id"),
                extend_existing=True,
            )
        else:
            self.chunks = Table(
                "kb_chunks",
                self.meta,
                Column("id", Text, primary_key=True),
                Column("dataset_id", Text, nullable=False),
                Column("chunk_id", Text, nullable=False),
                Column("chunk_text", Text, nullable=False),
                Column("contextual_text", Text, nullable=False),
                Column("summary", Text),
                Column("tags", JSON, nullable=False, server_default=text("'[]'::jsonb")),
                Column("questions", JSON, nullable=False, server_default=text("'[]'::jsonb")),
                Column("keywords", JSON, nullable=False, server_default=text("'[]'::jsonb")),
                Column("tod_items", JSON, nullable=False, server_default=text("'[]'::jsonb")),
                Column("meta_json", JSON, nullable=False, server_default=text("'{}'::jsonb")),
                Column("sentiment", Float, nullable=False, server_default=text("0")),
                Column("searchable_text", Text, nullable=False),
                Column("search_tokens", JSON, nullable=False, server_default=text("'[]'::jsonb")),
                Column("embedding_json", JSON, nullable=False, server_default=text("'[]'::jsonb")),
                Column("created_at", DateTime(timezone=True), server_default=func.now()),
                Index("idx_kb_chunks_dataset", "dataset_id"),
                extend_existing=True,
            )

        self.meta.create_all(self.engine)

    def persist_document(self, dataset_id: str, source_name: str, language: str, feature_json: Dict[str, Any]) -> None:
        if self.engine is None or self.documents is None:
            raise RuntimeError("Store not initialized. Call init_schema() first.")

        with self.engine.begin() as conn:
            conn.execute(
                text(
                    """
                    INSERT INTO kb_documents(dataset_id, source_name, language, feature_json)
                    VALUES (:dataset_id, :source_name, :language, CAST(:feature_json AS jsonb))
                    ON CONFLICT(dataset_id) DO UPDATE
                    SET source_name = excluded.source_name,
                        language = excluded.language,
                        feature_json = excluded.feature_json
                    """
                ),
                {
                    "dataset_id": dataset_id,
                    "source_name": source_name,
                    "language": language,
                    "feature_json": json.dumps(feature_json),
                },
            )

    def persist_chunks(self, dataset_id: str, chunks: List[KBChunk]) -> int:
        if self.engine is None or self.chunks is None:
            raise RuntimeError("Store not initialized. Call init_schema() first.")

        rows: List[Dict[str, Any]] = []
        for i, ch in enumerate(chunks):
            searchable_text = " ".join(
                [
                    _clean_text_for_pg(ch.chunk_text),
                    _clean_text_for_pg(ch.summary),
                    " ".join(ch.tags),
                    " ".join(ch.questions),
                    " ".join(ch.keywords),
                    " ".join(ch.tod_items),
                ]
            )[:12000]

            row: Dict[str, Any] = {
                "id": f"{dataset_id}_{ch.chunk_id}_{i}",
                "dataset_id": _clean_text_for_pg(dataset_id),
                "chunk_id": _clean_text_for_pg(ch.chunk_id),
                "chunk_text": _clean_text_for_pg(ch.chunk_text),
                "contextual_text": _clean_text_for_pg(ch.contextual_text),
                "summary": _clean_text_for_pg(ch.summary),
                "tags": ch.tags,
                "questions": ch.questions,
                "keywords": ch.keywords,
                "tod_items": ch.tod_items,
                "meta_json": ch.metadata,
                "sentiment": float(ch.sentiment),
                "searchable_text": searchable_text,
                "search_tokens": _tokenize_for_bm25(searchable_text),
            }

            if self.vector_enabled:
                row["embedding"] = (ch.embedding or [])[:768]
            else:
                row["embedding_json"] = ch.embedding or []

            rows.append(row)

        with self.engine.begin() as conn:
            # Rebuild mode: replace previous chunk rows for this dataset.
            conn.execute(
                self.chunks.delete().where(self.chunks.c.dataset_id == _clean_text_for_pg(dataset_id))
            )
            conn.execute(self.chunks.insert(), rows)
        return len(rows)

    def vector_search(self, dataset_id: str, query_vec: List[float], limit: int = 12) -> List[Dict[str, Any]]:
        if self.engine is None or self.chunks is None or not self.vector_enabled or not query_vec:
            return []

        qv = query_vec[:768]
        stmt = (
            select(
                self.chunks.c.chunk_id,
                self.chunks.c.summary,
                self.chunks.c.chunk_text,
                (1.0 - self.chunks.c.embedding.cosine_distance(qv)).label("vector_score"),
            )
            .where(self.chunks.c.dataset_id == dataset_id)
            .order_by(self.chunks.c.embedding.cosine_distance(qv))
            .limit(int(limit))
        )
        with self.engine.connect() as conn:
            rows = conn.execute(stmt).mappings().all()
        return [dict(r) for r in rows]

    def bm25_search(self, dataset_id: str, query: str, limit: int = 12) -> List[Dict[str, Any]]:
        if self.engine is None or self.chunks is None:
            return []

        # Pull candidate corpus from persisted chunks, then score with BM25Okapi library.
        with self.engine.connect() as conn:
            rows = conn.execute(
                select(
                    self.chunks.c.chunk_id,
                    self.chunks.c.summary,
                    self.chunks.c.chunk_text,
                    self.chunks.c.search_tokens,
                )
                .where(self.chunks.c.dataset_id == dataset_id)
                .limit(500)
            ).mappings().all()

        if not rows:
            return []

        query_tokens = _tokenize_for_bm25(query)
        if not query_tokens:
            return []

        if BM25Okapi is None:
            # Minimal deterministic fallback if rank_bm25 is unavailable.
            scored: List[Dict[str, Any]] = []
            qset = set(query_tokens)
            for r in rows:
                tokens = [str(t) for t in (r.get("search_tokens") or [])]
                overlap = sum(1 for t in tokens if t in qset)
                scored.append(
                    {
                        "chunk_id": r["chunk_id"],
                        "summary": r.get("summary", ""),
                        "chunk_text": r.get("chunk_text", ""),
                        "bm25_score": float(overlap),
                    }
                )
            scored.sort(key=lambda x: x["bm25_score"], reverse=True)
            return scored[:limit]

        corpus = [[str(tok) for tok in (r.get("search_tokens") or [])] for r in rows]
        bm25 = BM25Okapi(corpus)
        scores = bm25.get_scores(query_tokens)

        scored_rows: List[Dict[str, Any]] = []
        for r, score in zip(rows, scores):
            scored_rows.append(
                {
                    "chunk_id": r["chunk_id"],
                    "summary": r.get("summary", ""),
                    "chunk_text": r.get("chunk_text", ""),
                    "bm25_score": float(score),
                }
            )
        scored_rows.sort(key=lambda x: x["bm25_score"], reverse=True)
        return scored_rows[:limit]
