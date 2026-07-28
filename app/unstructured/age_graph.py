from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List

import psycopg2
from psycopg2 import sql
from psycopg2.extensions import connection as PGConnection

from app.core.config import settings
from app.core.llm_clients import LLMRouter
from app.unstructured.storage import AlloyLikeKBStore


def _dsn_to_psycopg2(dsn: str) -> str:
    if dsn.startswith("postgresql+psycopg2://"):
        return "postgresql://" + dsn.split("postgresql+psycopg2://", 1)[1]
    return dsn


def _esc(val: str) -> str:
    return val.replace("\\", "\\\\").replace("'", "\\'")


def _norm_entity(raw: str) -> str:
    return " ".join(str(raw).strip().split())[:180]


def _is_placeholder_entity(value: str) -> bool:
    text = str(value or "").strip()
    if not text:
        return True
    if text.startswith("<") and text.endswith(">"):
        return True
    if re.fullmatch(r"[A-Z_]+", text):
        return True
    return False


def _ag_to_text(val: Any) -> str:
    s = str(val)
    if s.startswith('"') and s.endswith('"') and len(s) >= 2:
        return s[1:-1]
    return s


def _extract_json_block(text: str) -> str | None:
    if not text:
        return None
    match = re.search(r"\{.*\}", text, re.S)
    if match:
        return match.group(0)
    match = re.search(r"\[.*\]", text, re.S)
    if match:
        return match.group(0)
    return None


@dataclass
class KGBuildResult:
    dataset_id: str
    chunks: int
    entities: int
    edges: int
    graph_name: str


class AGEGraphService:
    """
    Phase-1 AGE graph adapter:
    - Build graph from existing kb_chunks rows.
    - Query neighbors for an entity.
    """

    def __init__(self, dsn: str | None = None, graph_name: str | None = None) -> None:
        self.dsn = dsn or settings.kb_age_dsn or settings.kb_pg_dsn
        self.graph_name = graph_name or settings.kb_age_graph_name
        # KB chunks are persisted in the primary KB store. The AGE DSN is for graph writes/queries.
        self.kb_store_dsn = settings.kb_pg_dsn or self.dsn
        self.store = AlloyLikeKBStore(dsn=self.kb_store_dsn)
        self.router = LLMRouter()
        self.dataset_label = "DatasetNode"
        self.chunk_label = "ChunkNode"
        self.entity_label = "EntityNode"

    def _extract_triplets(self, text: str) -> List[Dict[str, str]]:
        provider = os.getenv("KG_LLM_PROVIDER", "bedrock")
        prompt = (
            "Extract subject-predicate-object triples from the text.\n"
            "Return strict JSON only with shape: {\"triples\": ["
            "{\"subject\":\"...\",\"predicate\":\"...\",\"object\":\"...\"}"
            "]}\n"
            "Rules:\n"
            "- Use short, clean entity names.\n"
            "- Use a concise verb phrase for predicate.\n"
            "- Maximum 20 triples.\n\n"
            f"TEXT:\n{text[:4000]}"
        )
        raw = self.router.complete(prompt, provider=provider)
        block = _extract_json_block(raw) or raw
        triples: List[Dict[str, str]] = []
        try:
            data = json.loads(block)
            triples = data.get("triples") or []
        except Exception:
            try:
                triples = json.loads(block)
            except Exception:
                triples = []
        cleaned: List[Dict[str, str]] = []
        for t in triples or []:
            if not isinstance(t, dict):
                continue
            s = _norm_entity(t.get("subject", ""))
            p = " ".join(str(t.get("predicate", "")).strip().split())[:120]
            o = _norm_entity(t.get("object", ""))
            if s and p and o:
                cleaned.append({"subject": s, "predicate": p, "object": o})
        return cleaned[:20]

    def _connect(self) -> PGConnection:
        if not self.dsn:
            raise RuntimeError("KB_AGE_DSN (or KB_PG_DSN fallback) is not configured")
        conn = psycopg2.connect(_dsn_to_psycopg2(self.dsn))
        conn.autocommit = True
        return conn

    def _init_age(self, conn: PGConnection) -> None:
        with conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS age")
            cur.execute("LOAD 'age'")
            cur.execute("SET search_path = ag_catalog, \"$user\", public")
            try:
                cur.execute("SELECT create_graph(%s)", (self.graph_name,))
            except Exception:
                # Graph already exists or extension limitations; continue.
                conn.rollback()
                cur.execute("SET search_path = ag_catalog, \"$user\", public")

    def _run_cypher(self, conn: PGConnection, cypher: str) -> List[tuple]:
        with conn.cursor() as cur:
            cur.execute(
                sql.SQL("SELECT * FROM cypher({graph}, $$ {cypher} $$) AS (r agtype)").format(
                    graph=sql.Literal(self.graph_name),
                    cypher=sql.SQL(cypher.replace("$$", "")),
                )
            )
            rows = cur.fetchall()
        return rows

    def _extract_entities(self, row: Dict[str, Any]) -> List[str]:
        entities: List[str] = []
        for key in ("keywords", "tags"):
            vals = row.get(key) or []
            if isinstance(vals, str):
                try:
                    vals = json.loads(vals)
                except Exception:
                    vals = []
            if isinstance(vals, list):
                for v in vals:
                    e = _norm_entity(v)
                    if len(e) >= 2:
                        entities.append(e)

        meta = row.get("meta_json") or {}
        if isinstance(meta, str):
            try:
                meta = json.loads(meta)
            except Exception:
                meta = {}
        if isinstance(meta, dict):
            for key in ("entities", "named_entities"):
                vals = meta.get(key) or []
                if isinstance(vals, list):
                    for v in vals:
                        e = _norm_entity(v)
                        if len(e) >= 2:
                            entities.append(e)

        # Fallback for structured plain-text chunks:
        # Example lines: "Organization: Delta Manufacturing"
        chunk_text = str(row.get("chunk_text") or "")
        summary_text = str(row.get("summary") or "")
        for line in chunk_text.splitlines():
            clean = " ".join(line.split()).strip()
            if ":" not in clean:
                continue
            left, right = clean.split(":", 1)
            _ = left  # label not currently used
            values = [v.strip() for v in right.split(",")]
            for v in values:
                e = _norm_entity(v)
                if len(e) >= 2 and any(ch.isalpha() for ch in e) and not _is_placeholder_entity(e):
                    entities.append(e)

        # Free-text fallback for prose documents like resumes and cover letters.
        free_text = f"{summary_text}\n{chunk_text}"
        uppercase_spans = re.findall(r"\b[A-Z][A-Z]+(?:\s+[A-Z][A-Z]+){0,3}\b", free_text)
        titlecase_spans = re.findall(r"\b[A-Z][a-zA-Z0-9./+-]+(?:\s+[A-Z][a-zA-Z0-9./+-]+){1,3}\b", free_text)
        tech_spans = re.findall(
            r"\b(?:Node\.js|Express|JavaScript|TypeScript|React|Next\.js|NextJS|CI/CD|UX|UI|API|PostgreSQL|MongoDB|Docker|AWS)\b",
            free_text,
            flags=re.I,
        )

        banned = {
            "AV",
            "Cover Letter",
            "Dear Hiring Manager",
            "Hiring Manager",
            "Sincerely",
            "India",
            "Powered By",
            "Full Stack Developer",
        }
        stop_starts = {"the", "this", "that", "these", "those", "my", "your", "our", "in", "on", "at", "for", "with", "please", "moreover", "specifically"}
        allowed_singletons = {"javascript", "typescript", "react", "express", "docker", "mongodb", "postgresql", "aws", "ux", "ui", "api"}
        for candidate in [*uppercase_spans, *titlecase_spans, *tech_spans]:
            e = _norm_entity(candidate)
            if len(e) < 2 or _is_placeholder_entity(e):
                continue
            if e in banned:
                continue
            if not any(ch.isalpha() for ch in e):
                continue
            words = e.split()
            if len(words) == 1 and words[0].lower() not in allowed_singletons:
                continue
            if words[0].lower() in stop_starts:
                continue
            if len(words) > 3 and "/" not in e and "." not in e:
                continue
            entities.append(e)

        # Unique preserve order.
        seen = set()
        uniq: List[str] = []
        for e in entities:
            normalized = e.lower()
            if normalized in seen:
                continue
            seen.add(normalized)
            uniq.append(e)
        return uniq[:40]

    def build_for_dataset(self, dataset_id: str) -> KGBuildResult:
        self.store.init_schema()
        if self.store.engine is None or self.store.chunks is None:
            raise RuntimeError("kb_chunks table is unavailable")

        with self.store.engine.connect() as conn:
            rows = conn.execute(
                self.store.chunks.select().where(self.store.chunks.c.dataset_id == dataset_id)
            ).mappings().all()

        if not rows:
            return KGBuildResult(dataset_id=dataset_id, chunks=0, entities=0, edges=0, graph_name=self.graph_name)

        chunk_count = 0
        entity_count = 0
        edge_count = 0

        with self._connect() as conn:
            self._init_age(conn)

            # Dataset vertex
            self._run_cypher(
                conn,
                f"MERGE (d:{self.dataset_label} {{id:'{_esc(dataset_id)}'}})",
            )

            for row in rows:
                chunk_id = str(row.get("chunk_id") or "")
                node_id = f"{dataset_id}:{chunk_id}"
                summary = str(row.get("summary") or "")[:800]
                snippet = str(row.get("chunk_text") or "")[:1000]

                cy_chunk = (
                    f"MERGE (c:{self.chunk_label} {{id:'{_esc(node_id)}'}}) "
                    f"SET c.dataset_id='{_esc(dataset_id)}', c.chunk_id='{_esc(chunk_id)}', "
                    f"c.summary='{_esc(summary)}', c.snippet='{_esc(snippet)}' "
                    f"WITH c "
                    f"MATCH (d:{self.dataset_label} {{id:'{_esc(dataset_id)}'}}) "
                    f"MERGE (d)-[:HAS_CHUNK]->(c)"
                )
                self._run_cypher(conn, cy_chunk)
                chunk_count += 1

                text_for_kg = f"{summary}\n{snippet}"
                ents = self._extract_entities(dict(row))
                for ent in ents:
                    cy_ent = (
                        f"MATCH (c:{self.chunk_label} {{id:'{_esc(node_id)}'}}) "
                        f"MERGE (e:{self.entity_label} {{name:'{_esc(ent)}'}}) "
                        f"MERGE (c)-[:MENTIONS]->(e)"
                    )
                    self._run_cypher(conn, cy_ent)
                    entity_count += 1
                    edge_count += 1

                # Deterministic co-occurrence relations keep the graph useful even when
                # LLM triplet extraction is slow or unavailable.
                rel_entities = ents[:8]
                for i, src in enumerate(rel_entities):
                    for dst in rel_entities[i + 1 : i + 4]:
                        cy_related = (
                            f"MATCH (a:{self.entity_label} {{name:'{_esc(src)}'}}), (b:{self.entity_label} {{name:'{_esc(dst)}'}}) "
                            f"MERGE (a)-[:RELATION {{predicate:'co_occurs_with', dataset_id:'{_esc(dataset_id)}'}}]->(b)"
                        )
                        self._run_cypher(conn, cy_related)
                        edge_count += 1

                triples: List[Dict[str, str]] = []
                if len(rel_entities) < 2:
                    triples = self._extract_triplets(text_for_kg)
                if not triples:
                    continue

                for t in triples:
                    s = t["subject"]
                    p = t["predicate"]
                    o = t["object"]
                    cy_rel = (
                        f"MERGE (a:{self.entity_label} {{name:'{_esc(s)}'}}) "
                        f"MERGE (b:{self.entity_label} {{name:'{_esc(o)}'}}) "
                        f"MERGE (a)-[:RELATION {{predicate:'{_esc(p)}', dataset_id:'{_esc(dataset_id)}'}}]->(b)"
                    )
                    self._run_cypher(conn, cy_rel)
                    entity_count += 2
                    edge_count += 1
                    cy_m1 = (
                        f"MATCH (c:{self.chunk_label} {{id:'{_esc(node_id)}'}}), (e:{self.entity_label} {{name:'{_esc(s)}'}}) "
                        f"MERGE (c)-[:MENTIONS]->(e)"
                    )
                    cy_m2 = (
                        f"MATCH (c:{self.chunk_label} {{id:'{_esc(node_id)}'}}), (e:{self.entity_label} {{name:'{_esc(o)}'}}) "
                        f"MERGE (c)-[:MENTIONS]->(e)"
                    )
                    self._run_cypher(conn, cy_m1)
                    self._run_cypher(conn, cy_m2)
                    edge_count += 2

        return KGBuildResult(
            dataset_id=dataset_id,
            chunks=chunk_count,
            entities=entity_count,
            edges=edge_count,
            graph_name=self.graph_name,
        )

    def query_neighbors(self, dataset_id: str, entity: str, limit: int = 20) -> Dict[str, Any]:
        target = _norm_entity(entity)
        if not target:
            return {
                "dataset_id": dataset_id,
                "entity": entity,
                "neighbors": [],
                "graph_name": self.graph_name,
            }

        cypher = (
            f"MATCH (e:{self.entity_label} {{name:'{_esc(target)}'}})-[r:RELATION]->(n:{self.entity_label}) "
            f"WHERE r.dataset_id = '{_esc(dataset_id)}' "
            f"RETURN e.name, r.predicate, n.name LIMIT {int(limit)}"
        )
        neighbors: List[Dict[str, Any]] = []
        with self._connect() as conn:
            self._init_age(conn)
            with conn.cursor() as cur:
                cur.execute(
                    sql.SQL("SELECT * FROM cypher({graph}, $$ {cypher} $$) AS (src agtype, rel agtype, dst agtype)").format(
                        graph=sql.Literal(self.graph_name),
                        cypher=sql.SQL(cypher.replace("$$", "")),
                    )
                )
                rows = cur.fetchall()
            for row in rows:
                neighbors.append(
                    {
                        "source": _ag_to_text(row[0]),
                        "relation": _ag_to_text(row[1]),
                        "target": _ag_to_text(row[2]),
                    }
                )

        return {
            "dataset_id": dataset_id,
            "entity": target,
            "neighbors": neighbors,
            "graph_name": self.graph_name,
        }

    def subgraph(self, dataset_id: str, seed_entity: str, hops: int = 1, limit: int = 100) -> Dict[str, Any]:
        seed = _norm_entity(seed_entity)
        if not seed:
            # Dataset-wide graph view (edge-limited) when no seed is provided.
            lim = max(1, min(int(limit), 500))
            nodes: Dict[str, Dict[str, Any]] = {}
            edges: List[Dict[str, Any]] = []
            with self._connect() as conn:
                self._init_age(conn)
                # Layer edges: Dataset -> Chunk
                cy_chunks = (
                    f"MATCH (d:{self.dataset_label} {{id:'{_esc(dataset_id)}'}})-[r:HAS_CHUNK]->(c:{self.chunk_label}) "
                    f"RETURN d.id, type(r), c.id LIMIT {lim}"
                )
                with conn.cursor() as cur:
                    cur.execute(
                        sql.SQL("SELECT * FROM cypher({graph}, $$ {cypher} $$) AS (src agtype, rel agtype, dst agtype)").format(
                            graph=sql.Literal(self.graph_name),
                            cypher=sql.SQL(cy_chunks.replace("$$", "")),
                        )
                    )
                    rows = cur.fetchall()
                for row in rows:
                    src = _ag_to_text(row[0])
                    rel = _ag_to_text(row[1])
                    dst = _ag_to_text(row[2])
                    nodes.setdefault(src, {"id": src, "label": src, "type": "Dataset"})
                    nodes.setdefault(dst, {"id": dst, "label": dst.split(":", 1)[-1], "type": "Chunk"})
                    edges.append({"source": src, "target": dst, "label": rel})

                # Layer edges: Chunk -> Entity
                cy_mentions = (
                    f"MATCH (c:{self.chunk_label})-[r:MENTIONS]->(e:{self.entity_label}) "
                    f"WHERE c.dataset_id = '{_esc(dataset_id)}' "
                    f"RETURN c.id, type(r), e.name LIMIT {lim}"
                )
                with conn.cursor() as cur:
                    cur.execute(
                        sql.SQL("SELECT * FROM cypher({graph}, $$ {cypher} $$) AS (src agtype, rel agtype, dst agtype)").format(
                            graph=sql.Literal(self.graph_name),
                            cypher=sql.SQL(cy_mentions.replace("$$", "")),
                        )
                    )
                    rows = cur.fetchall()
                for row in rows:
                    src = _ag_to_text(row[0])
                    rel = _ag_to_text(row[1])
                    dst = _ag_to_text(row[2])
                    nodes.setdefault(src, {"id": src, "label": src.split(":", 1)[-1], "type": "Chunk"})
                    nodes.setdefault(dst, {"id": dst, "label": dst, "type": "Entity"})
                    edges.append({"source": src, "target": dst, "label": rel})

                # Entity relation edges
                cy_rel = (
                    f"MATCH (a:{self.entity_label})-[r:RELATION]->(b:{self.entity_label}) "
                    f"WHERE r.dataset_id = '{_esc(dataset_id)}' "
                    f"RETURN a.name, r.predicate, b.name LIMIT {lim}"
                )
                with conn.cursor() as cur:
                    cur.execute(
                        sql.SQL("SELECT * FROM cypher({graph}, $$ {cypher} $$) AS (src agtype, rel agtype, dst agtype)").format(
                            graph=sql.Literal(self.graph_name),
                            cypher=sql.SQL(cy_rel.replace("$$", "")),
                        )
                    )
                    rows = cur.fetchall()
                for row in rows:
                    src = _ag_to_text(row[0])
                    rel = _ag_to_text(row[1])
                    dst = _ag_to_text(row[2])
                    nodes.setdefault(src, {"id": src, "label": src, "type": "Entity"})
                    nodes.setdefault(dst, {"id": dst, "label": dst, "type": "Entity"})
                    edges.append({"source": src, "target": dst, "label": rel})
            return {
                "dataset_id": dataset_id,
                "seed_entity": "",
                "graph_name": self.graph_name,
                "nodes": list(nodes.values()),
                "edges": edges,
            }
        h = max(1, min(int(hops), 3))
        lim = max(1, min(int(limit), 500))
        nodes: Dict[str, Dict[str, Any]] = {}
        edges: List[Dict[str, Any]] = []

        frontier = {seed}
        visited = {seed}
        edge_seen = set()

        with self._connect() as conn:
            self._init_age(conn)
            for _ in range(h):
                next_frontier = set()
                for current in list(frontier):
                    cypher = (
                        f"MATCH (a:{self.entity_label} {{name:'{_esc(current)}'}})-[r:RELATION]->(b:{self.entity_label}) "
                        f"WHERE r.dataset_id = '{_esc(dataset_id)}' "
                        f"RETURN a.name, r.predicate, b.name LIMIT {lim}"
                    )
                    with conn.cursor() as cur:
                        cur.execute(
                            sql.SQL("SELECT * FROM cypher({graph}, $$ {cypher} $$) AS (src agtype, rel agtype, dst agtype)").format(
                                graph=sql.Literal(self.graph_name),
                                cypher=sql.SQL(cypher.replace("$$", "")),
                            )
                        )
                        rows = cur.fetchall()

                    for row in rows:
                        src = _ag_to_text(row[0])
                        rel = _ag_to_text(row[1])
                        dst = _ag_to_text(row[2])
                        nodes.setdefault(src, {"id": src, "label": src, "type": "Entity"})
                        nodes.setdefault(dst, {"id": dst, "label": dst, "type": "Entity"})
                        key = (src, rel, dst)
                        if key not in edge_seen:
                            edge_seen.add(key)
                            edges.append({"source": src, "target": dst, "label": rel})
                        if dst not in visited:
                            visited.add(dst)
                            next_frontier.add(dst)
                    if len(edges) >= lim:
                        break

                frontier = next_frontier
                if not frontier or len(edges) >= lim:
                    break

        return {
            "dataset_id": dataset_id,
            "seed_entity": seed,
            "graph_name": self.graph_name,
            "nodes": list(nodes.values()),
            "edges": edges,
        }
