from __future__ import annotations

import asyncio
import concurrent.futures
import inspect
import os
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List


@dataclass
class AtomResult:
    ok: bool
    reason: str
    atomic_facts: List[str]
    graph_payload: Dict[str, Any]


def _chunk_atomic_facts(text: str, max_chars: int = 1600) -> List[str]:
    text = text.strip()
    if not text:
        return []

    # Prefer sentence boundaries for atomicity fallback.
    sentences = re.split(r"(?<=[.!?])\s+", text)
    facts: List[str] = []
    current = ""
    for sent in sentences:
        s = sent.strip()
        if not s:
            continue
        if len(s) > max_chars:
            # Hard split for very long lines
            for i in range(0, len(s), max_chars):
                facts.append(s[i : i + max_chars])
            continue
        if len(current) + len(s) + 1 <= max_chars:
            current = (current + " " + s).strip()
        else:
            if current:
                facts.append(current)
            current = s
    if current:
        facts.append(current)
    return facts


def _extract_nodes_edges_from_atom_kg(kg: Any) -> Dict[str, Any]:
    # Best-effort extraction across package versions.
    nodes: Dict[str, Dict[str, Any]] = {}
    edges: List[Dict[str, Any]] = []

    candidate_rel_lists = []
    for attr in ["relations", "relationship_list", "relation_list", "edges", "triples", "quintuples"]:
        if hasattr(kg, attr):
            val = getattr(kg, attr)
            if isinstance(val, list):
                candidate_rel_lists.append(val)

    for rels in candidate_rel_lists:
        for rel in rels:
            # dict-style relation
            if isinstance(rel, dict):
                s = rel.get("subject") or rel.get("source") or rel.get("entity_1") or rel.get("head")
                p = rel.get("predicate") or rel.get("relation") or rel.get("label")
                o = rel.get("object") or rel.get("target") or rel.get("entity_2") or rel.get("tail")
                t_start = rel.get("t_start") or rel.get("valid_from") or rel.get("start_time")
                t_end = rel.get("t_end") or rel.get("valid_to") or rel.get("end_time")
            else:
                # tuple-style relation, expect at least (s,p,o)
                try:
                    s = rel[0]
                    p = rel[1]
                    o = rel[2]
                    t_start = rel[3] if len(rel) > 3 else None
                    t_end = rel[4] if len(rel) > 4 else None
                except Exception:
                    continue

            if not s or not p or not o:
                continue

            s_id = str(s)
            o_id = str(o)
            nodes.setdefault(s_id, {"id": s_id, "type": "entity"})
            nodes.setdefault(o_id, {"id": o_id, "type": "entity"})
            edges.append(
                {
                    "source": s_id,
                    "predicate": str(p),
                    "target": o_id,
                    "t_start": t_start,
                    "t_end": t_end,
                }
            )

    # If no explicit relation list was found, try node containers for fallback visualization.
    if not edges:
        for attr in ["entities", "entity_list", "nodes"]:
            if hasattr(kg, attr):
                val = getattr(kg, attr)
                if isinstance(val, list):
                    for n in val:
                        nid = str(n.get("name") if isinstance(n, dict) else n)
                        nodes.setdefault(nid, {"id": nid, "type": "entity"})

    return {"nodes": list(nodes.values()), "edges": edges}


def build_atom_tkg(markdown: str, obs_timestamp: str | None = None) -> AtomResult:
    max_facts = int(os.getenv("ATOM_MAX_FACTS", "60"))
    timeout_sec = int(os.getenv("ATOM_TIMEOUT_SEC", "120"))
    atomic_facts = _chunk_atomic_facts(markdown)[:max_facts]
    if not atomic_facts:
        return AtomResult(ok=False, reason="No atomic facts extracted", atomic_facts=[], graph_payload={"nodes": [], "edges": []})

    if obs_timestamp is None:
        obs_timestamp = datetime.utcnow().strftime("%Y-%m-%d")

    try:
        try:
            from itext2kg.atom import Atom
        except Exception:
            from itext2kg import Atom  # legacy compatibility
    except Exception as exc:
        return AtomResult(
            ok=False,
            reason=f"ATOM dependencies unavailable: {exc}",
            atomic_facts=atomic_facts,
            graph_payload={"nodes": [], "edges": []},
        )

    try:
        llm = None
        embeddings = None
        backend = "unknown"

        from langchain_aws import ChatBedrock, BedrockEmbeddings

        bedrock_region = os.getenv("BEDROCK_REGION", "us-east-2")
        bedrock_model_id = os.getenv("BEDROCK_MODEL_ID", "")
        bedrock_embed_model_id = os.getenv("BEDROCK_EMBED_MODEL_ID", "")
        if not bedrock_model_id or not bedrock_embed_model_id:
            raise RuntimeError("BEDROCK_MODEL_ID and BEDROCK_EMBED_MODEL_ID are required for ATOM KG.")

        llm = ChatBedrock(
            model_id=bedrock_model_id,
            region_name=bedrock_region,
            temperature=0,
        )
        embeddings = BedrockEmbeddings(
            model_id=bedrock_embed_model_id,
            region_name=bedrock_region,
        )
        backend = "langchain_aws->bedrock"

        atom = Atom(llm_model=llm, embeddings_model=embeddings)

        # Prefer direct single-time builder if available.
        if hasattr(atom, "build_graph"):
            method = getattr(atom, "build_graph")
            if inspect.iscoroutinefunction(method):
                kg = asyncio.run(
                    asyncio.wait_for(
                        method(atomic_facts=atomic_facts, obs_timestamp=obs_timestamp),
                        timeout=timeout_sec,
                    )
                )
            else:
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
                    out = ex.submit(method, atomic_facts=atomic_facts, obs_timestamp=obs_timestamp).result(timeout=timeout_sec)
                kg = asyncio.run(asyncio.wait_for(out, timeout=timeout_sec)) if inspect.iscoroutine(out) else out
        else:
            # Fallback to multi-time API.
            method = getattr(atom, "build_graph_from_different_obs_times")
            payload = {obs_timestamp: atomic_facts}
            if inspect.iscoroutinefunction(method):
                kg = asyncio.run(
                    asyncio.wait_for(
                        method(atomic_facts_with_obs_timestamps=payload),
                        timeout=timeout_sec,
                    )
                )
            else:
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
                    out = ex.submit(method, atomic_facts_with_obs_timestamps=payload).result(timeout=timeout_sec)
                kg = asyncio.run(asyncio.wait_for(out, timeout=timeout_sec)) if inspect.iscoroutine(out) else out

        graph_payload = _extract_nodes_edges_from_atom_kg(kg)
        return AtomResult(
            ok=True,
            reason=f"ATOM graph built ({backend})",
            atomic_facts=atomic_facts,
            graph_payload=graph_payload,
        )

    except Exception as exc:
        return AtomResult(
            ok=False,
            reason=f"ATOM build failed: {exc}",
            atomic_facts=atomic_facts,
            graph_payload={"nodes": [], "edges": []},
        )
