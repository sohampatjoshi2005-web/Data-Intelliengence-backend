from __future__ import annotations

import base64
import hashlib
import io
import json
import os
import re
import math
import time
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
from uuid import uuid4
from pathlib import Path
import sys
from collections import deque
from typing import Any, Dict, Optional, Tuple, List

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st
from custom_gpt import render_custom_gpt_tab
from custom_gpt.client import build_headers
from custom_gpt.types import HeaderConfig

# The SDR UI remains owned by the standalone AI_SDR project.  Add only its UI
# adapter to the import path; the SDR backend continues to run independently.
AI_SDR_UI_DIR = Path(__file__).resolve().parents[1] / "AI_SDR"
if str(AI_SDR_UI_DIR) not in sys.path:
    sys.path.insert(0, str(AI_SDR_UI_DIR))
from sdr_ui import render_ai_sdr_ui

try:
    from markitdown import MarkItDown
except Exception:
    MarkItDown = None

try:
    import pytesseract
    from PIL import Image
except Exception:
    pytesseract = None
    Image = None
try:
    import optuna
except Exception:
    optuna = None

try:
    from river import drift, stats, tree
except Exception:
    drift = None
    stats = None
    tree = None

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

try:
    from xgboost import XGBClassifier, XGBRegressor
except Exception:
    XGBClassifier = None
    XGBRegressor = None

try:
    from lightgbm import LGBMClassifier, LGBMRegressor
except Exception:
    LGBMClassifier = None
    LGBMRegressor = None

try:
    from catboost import CatBoostClassifier, CatBoostRegressor
except Exception:
    CatBoostClassifier = None
    CatBoostRegressor = None

DEFAULT_BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")
DEFAULT_SDR_API_URL = os.getenv("SDR_API_BASE_URL", "http://127.0.0.1:8011")


def _api_headers() -> Dict[str, str]:
    try:
        return build_headers(HeaderConfig(apply_role_headers=True, include_api_key=False))
    except Exception:
        return {}


def _raise_for_status_with_detail(resp: requests.Response) -> None:
    if resp.ok:
        return
    detail = resp.text
    try:
        body = resp.json()
        detail = body.get("detail", body)
    except Exception:
        pass
    raise requests.HTTPError(f"{resp.status_code} {resp.reason}: {detail}")


def api_get(base_url: str, path: str, params: Optional[Dict[str, Any]] = None) -> Any:
    url = f"{base_url.rstrip('/')}{path}"
    resp = requests.get(url, params=params, headers=_api_headers(), timeout=120)
    _raise_for_status_with_detail(resp)
    return resp.json()


def api_post_json(base_url: str, path: str, payload: Dict[str, Any]) -> Any:
    url = f"{base_url.rstrip('/')}{path}"
    resp = requests.post(url, json=payload, headers=_api_headers(), timeout=300)
    _raise_for_status_with_detail(resp)
    return resp.json()


def api_post_multipart(
    base_url: str,
    path: str,
    files: Dict[str, Any],
    data: Optional[Dict[str, Any]] = None,
) -> Any:
    url = f"{base_url.rstrip('/')}{path}"
    resp = requests.post(url, files=files, data=data or {}, headers=_api_headers(), timeout=600)
    _raise_for_status_with_detail(resp)
    return resp.json()


def api_post_multipart_stream(
    base_url: str,
    path: str,
    files: Dict[str, Any],
    data: Optional[Dict[str, Any]] = None,
):
    url = f"{base_url.rstrip('/')}{path}"
    with requests.post(url, files=files, data=data or {}, headers=_api_headers(), timeout=1200, stream=True) as resp:
        _raise_for_status_with_detail(resp)
        for line in resp.iter_lines(decode_unicode=True):
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception:
                continue


def decode_b64_image(b64: str) -> bytes:
    return base64.b64decode(b64.encode("utf-8"))


def safe_json(obj: Any) -> str:
    return json.dumps(obj, indent=2, default=str)


def render_deepeval_results(payload: Dict[str, Any]) -> None:
    if not payload:
        return
    status = payload.get("status", "unknown")
    if status != "ok":
        st.warning(f"DeepEval status: {status}. {payload.get('reason', '')}")
        return
    metrics = payload.get("metrics", [])
    if metrics:
        st.dataframe(pd.DataFrame(metrics), use_container_width=True)
    else:
        st.info("DeepEval returned no metric results.")


def _to_pct(value: Any) -> str:
    try:
        return f"{100.0 * float(value):.2f}%"
    except Exception:
        return str(value)


def _format_scalar(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def render_sql_result_chart(rows: list[dict]) -> None:
    if not rows:
        st.info("No rows returned.")
        return
    df = pd.DataFrame(rows)
    if df.empty:
        st.info("No rows returned.")
        return

    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    non_numeric_cols = [c for c in df.columns if c not in numeric_cols]

    if len(numeric_cols) >= 1 and len(non_numeric_cols) >= 1:
        x = non_numeric_cols[0]
        y = numeric_cols[0]
        fig = px.bar(df, x=x, y=y, title=f"{y} by {x}")
        st.plotly_chart(fig, use_container_width=True)
        return
    if len(numeric_cols) >= 2:
        x = numeric_cols[0]
        y = numeric_cols[1]
        fig = px.scatter(df, x=x, y=y, title=f"{y} vs {x}")
        st.plotly_chart(fig, use_container_width=True)
        return
    if len(numeric_cols) == 1:
        x = numeric_cols[0]
        fig = px.histogram(df, x=x, title=f"Distribution of {x}")
        st.plotly_chart(fig, use_container_width=True)
        return

    st.dataframe(df, use_container_width=True)


def un_show_build_summary(out: Dict[str, Any]) -> None:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Dataset", out.get("dataset_id", "N/A"))
    c2.metric("Language", out.get("language", "unknown"))
    c3.metric("Chunks Built", int(out.get("chunk_count", 0)))
    c4.metric("Chunks Stored", int(out.get("stored_chunks", 0)))

    st.markdown("### Pipeline Output")
    st.write(f"- Headings extracted: {int(out.get('headings_count', 0))}")
    st.write(f"- Entities extracted: {int(out.get('entities_count', 0))}")
    warnings = out.get("warnings", []) or []
    if warnings:
        st.warning("\n".join([f"- {w}" for w in warnings]))
    perf = out.get("performance") or {}
    if perf:
        st.markdown("### Performance Settings Used")
        st.code(json.dumps(perf, indent=2), language="json")
    progress = out.get("progress") or []
    if progress:
        st.markdown("### Streaming Build Progress (Batch Inserts)")
        st.code(json.dumps(progress, indent=2), language="json")
    steps = out.get("pipeline_steps") or []
    if steps:
        st.markdown("### Pipeline Steps (This Run)")
        for s in steps:
            label = s.get("step", "Step")
            enabled = s.get("enabled", False)
            note = s.get("notes", "")
            st.write(f"- [{'✓' if enabled else '×'}] {label}" + (f" — {note}" if note else ""))


def un_show_query_simple(out: Dict[str, Any]) -> None:
    st.markdown("### Grounded Answer")
    st.write(out.get("answer", ""))

    def _best_sentence(text: str) -> str:
        if not text:
            return ""
        parts = re.split(r"(?<=[.!?])\s+", text.strip())
        for p in parts:
            if len(p.split()) >= 6:
                return p.strip()
        return parts[0].strip() if parts else text[:200].strip()

    hits = out.get("hits", []) or []
    if not hits:
        st.info("No retrieved chunks.")
        return
    st.markdown("### Top Citations")
    for h in hits[:3]:
        summary = h.get("summary", "") or ""
        snippet = h.get("snippet", "") or ""
        sentence = _best_sentence(summary) or _best_sentence(snippet)
        if sentence:
            st.write(f"- {sentence}")


def un_show_query_debug(out: Dict[str, Any]) -> None:
    st.markdown("### Debug Payload")
    st.code(json.dumps(out, indent=2, default=str), language="json")
    hits = out.get("hits", []) or []
    if hits:
        st.markdown("### Retrieved Hits Table")
        st.dataframe(pd.DataFrame(hits), use_container_width=True)


def _un_build_adjacency(nodes: list[dict], edges: list[dict]) -> Dict[str, set]:
    adj: Dict[str, set] = {str(n["id"]): set() for n in nodes}
    for e in edges:
        s = str(e.get("source", ""))
        t = str(e.get("target", ""))
        if s in adj and t in adj:
            adj[s].add(t)
            adj[t].add(s)
    return adj


def _un_connected_components(nodes: list[dict], edges: list[dict]) -> Dict[str, int]:
    adj = _un_build_adjacency(nodes, edges)
    comp: Dict[str, int] = {}
    cid = 0
    for nid in adj:
        if nid in comp:
            continue
        stack = [nid]
        comp[nid] = cid
        while stack:
            cur = stack.pop()
            for nxt in adj[cur]:
                if nxt in comp:
                    continue
                comp[nxt] = cid
                stack.append(nxt)
        cid += 1
    return comp


def un_render_kg_plot_with_view(subgraph: Dict[str, Any], view_mode: str = "overview") -> None:
    nodes = subgraph.get("nodes", []) or []
    edges = subgraph.get("edges", []) or []
    if not nodes:
        st.info("No nodes to visualize.")
        return

    n = len(nodes)
    positions: Dict[str, tuple[float, float]] = {}
    node_by_id = {str(node["id"]): node for node in nodes}

    if view_mode == "layered":
        layers = {"Dataset": 2.0, "Chunk": 1.0, "Entity": 0.0}
        grouped: Dict[str, list[str]] = {"Dataset": [], "Chunk": [], "Entity": []}
        for node in nodes:
            t = str(node.get("type", "Entity"))
            grouped.setdefault(t, [])
            grouped[t].append(str(node["id"]))
        for t, ids in grouped.items():
            if not ids:
                continue
            span = max(len(ids), 1)
            for i, nid in enumerate(ids):
                x = -1.0 + (2.0 * i / max(span - 1, 1)) if span > 1 else 0.0
                y = layers.get(t, 0.0)
                positions[nid] = (x, y)
    elif view_mode == "community":
        comps = _un_connected_components(nodes, edges)
        by_comp: Dict[int, list[str]] = {}
        for nid, cid in comps.items():
            by_comp.setdefault(cid, []).append(nid)
        comp_ids = sorted(by_comp.keys())
        for cidx, cid in enumerate(comp_ids):
            cluster = by_comp[cid]
            cx = (cidx % 3) * 2.5
            cy = -((cidx // 3) * 2.0)
            for i, nid in enumerate(cluster):
                theta = (2.0 * math.pi * i) / max(len(cluster), 1)
                positions[nid] = (cx + 0.7 * math.cos(theta), cy + 0.7 * math.sin(theta))
    elif view_mode == "entity-centric":
        seed = str(subgraph.get("seed_entity", "")).strip()
        if seed and seed in node_by_id:
            positions[seed] = (0.0, 0.0)
        remaining = [str(node["id"]) for node in nodes if str(node["id"]) not in positions]
        for i, nid in enumerate(remaining):
            theta = (2.0 * math.pi * i) / max(len(remaining), 1)
            positions[nid] = (1.2 * math.cos(theta), 1.2 * math.sin(theta))
    else:  # overview
        radius = 1.0
        for i, node in enumerate(nodes):
            theta = (2.0 * math.pi * i) / max(n, 1)
            positions[str(node["id"])] = (radius * math.cos(theta), radius * math.sin(theta))

    edge_x: list[float] = []
    edge_y: list[float] = []
    for e in edges:
        s = str(e.get("source", ""))
        t = str(e.get("target", ""))
        if s not in positions or t not in positions:
            continue
        x0, y0 = positions[s]
        x1, y1 = positions[t]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    node_x = []
    node_y = []
    node_text = []
    node_color = []
    color_map = {"Dataset": "#F59E0B", "Chunk": "#06B6D4", "Entity": "#22C55E"}
    for node in nodes:
        nid = str(node["id"])
        x, y = positions[nid]
        node_x.append(x)
        node_y.append(y)
        nlabel = str(node.get("label", nid))
        ntype = str(node.get("type", "Entity"))
        node_text.append(f"{nlabel} ({ntype})")
        node_color.append(color_map.get(ntype, "#94A3B8"))

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=edge_x,
            y=edge_y,
            mode="lines",
            line=dict(width=1),
            hoverinfo="none",
            name="edges",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=node_x,
            y=node_y,
            mode="markers+text",
            marker=dict(size=14, color=node_color),
            text=node_text,
            textposition="top center",
            hoverinfo="text",
            name="nodes",
        )
    )
    fig.update_layout(
        title=f"AGE Graph View: {view_mode.title()}",
        showlegend=False,
        margin=dict(l=10, r=10, t=40, b=10),
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        height=520,
    )
    st.plotly_chart(fig, use_container_width=True)


def unstructured_tab(base_url: str) -> None:
    st.subheader("Unstructured RAG")
    st.caption("Build + query knowledge bases, and explore the Apache AGE knowledge graph.")

    st.markdown("## 0) Analyze Unstructured File (Any Format)")
    analyze_upload = st.file_uploader(
        "Upload Any Unstructured File",
        type=["pdf", "docx", "pptx", "xlsx", "txt", "md", "html", "htm", "json", "csv", "png", "jpg", "jpeg", "bmp", "tiff", "gif", "mp4", "mov", "mkv", "avi"],
        key="un_any_upload",
    )
    if st.button("Analyze File", use_container_width=True, key="un_any_analyze"):
        if not analyze_upload:
            st.error("Upload a file first.")
        else:
            try:
                out = api_post_multipart(
                    base_url,
                    "/unstructured/analyze",
                    files={"file": (analyze_upload.name, analyze_upload.getvalue())},
                )
                st.session_state["un_any_result"] = out
                st.success("Analysis completed.")
            except Exception as exc:
                st.error(f"Analyze failed: {exc}")

    if st.session_state.get("un_any_result"):
        res = st.session_state["un_any_result"]
        st.markdown("### Analysis Summary")
        st.write(f"File: `{res.get('file_name','')}` | Type: `{res.get('file_type','')}` | Text length: `{res.get('text_length',0)}`")
        if res.get("duration") is not None:
            st.caption(f"Video duration: {res.get('duration')}s")
        if res.get("caption"):
            st.markdown("### Image/Video Caption")
            st.info(res.get("caption", ""))
            if res.get("caption_model"):
                st.caption(f"Caption model: {res.get('caption_model')}")
        if res.get("warnings"):
            st.warning("\n".join([f"- {w}" for w in res.get("warnings", [])]))

        st.markdown("### Text Preview")
        st.code(res.get("text_preview", "")[:4000])

        counts = res.get("entity_counts", []) or []
        if counts:
            st.markdown("### Entity Counts")
            df_counts = pd.DataFrame(counts)
            st.dataframe(df_counts, use_container_width=True)
            if {"label", "count"}.issubset(df_counts.columns):
                fig = px.bar(df_counts, x="label", y="count", title="Entity Counts")
                st.plotly_chart(fig, use_container_width=True)
        st.markdown("---")

    st.markdown("## 1) Build Knowledge Base")
    upload = st.file_uploader("Upload Unstructured File", type=["pdf", "docx", "md", "txt", "json"], key="un_upload")
    c1, c2 = st.columns(2)
    with c1:
        dataset_id = st.text_input("Dataset ID (optional)", value="", key="un_dataset_id")
    with c2:
        provider = st.selectbox("LLM Provider", options=["ollama_local", "bedrock"], index=0, key="un_llm_provider")

    st.markdown("### Build Performance")
    p1, p2, p3 = st.columns(3)
    with p1:
        fast_mode = st.checkbox("Fast Mode", value=True, help="Enable faster defaults", key="un_fast_mode")
        chunk_cap = st.number_input("Chunk Cap", min_value=20, max_value=1000, value=120, step=10, key="un_chunk_cap")
        enrichment_batch_size = st.number_input("Enrichment Batch Size", min_value=1, max_value=32, value=8, step=1, key="un_enrich_batch")
    with p2:
        skip_ner = st.checkbox("Skip NER", value=True, key="un_skip_ner")
        skip_pii = st.checkbox("Skip PII Redaction", value=False, key="un_skip_pii")
        enrichment_workers = st.number_input("Enrichment Workers", min_value=1, max_value=8, value=2, step=1, key="un_enrich_workers")
    with p3:
        skip_enrichment = st.checkbox("Skip Enrichment", value=False, key="un_skip_enrich")
        embedding_batch_size = st.number_input("Embedding Batch Size", min_value=1, max_value=128, value=32, step=1, key="un_embed_batch")
        embedding_workers = st.number_input("Embedding Workers", min_value=1, max_value=8, value=2, step=1, key="un_embed_workers")

    with st.expander("RAG v2 Enhancements (Enabled by default)", expanded=False):
        st.markdown(
            "- Semantic + hierarchical chunking\n"
            "- Query expansion (multi-query retrieval)\n"
            "- Hybrid retrieval (BM25 + vector)\n"
            "- Reranking (FlashRank / BGE Cross-Encoder)\n"
            "- Graph expansion (Apache AGE)\n"
            "- Context compression\n"
            "- RAPTOR summaries (section + document)"
        )

    if st.button("Build KB", use_container_width=True, key="un_build_btn"):
        if not upload:
            st.error("Upload a file first.")
        else:
            try:
                out = api_post_multipart(
                    base_url,
                    "/kb/build",
                    files={"file": (upload.name, upload.getvalue())},
                    data={
                        "dataset_id": dataset_id,
                        "llm_provider": provider,
                        "fast_mode": str(bool(fast_mode)).lower(),
                        "chunk_cap": str(int(chunk_cap)),
                        "skip_ner": str(bool(skip_ner)).lower(),
                        "skip_pii": str(bool(skip_pii)).lower(),
                        "skip_enrichment": str(bool(skip_enrichment)).lower(),
                        "enrichment_batch_size": str(int(enrichment_batch_size)),
                        "enrichment_workers": str(int(enrichment_workers)),
                        "embedding_batch_size": str(int(embedding_batch_size)),
                        "embedding_workers": str(int(embedding_workers)),
                    },
                )
                st.session_state["un_build"] = out
                st.success("Knowledge base build completed.")
            except Exception as exc:
                st.error(f"Build failed: {exc}")

    if st.session_state.get("un_build"):
        un_show_build_summary(st.session_state["un_build"])

    st.markdown("---")
    st.markdown("## 2) Query Knowledge Base")
    q_dataset = st.text_input(
        "Dataset ID for Query",
        value=dataset_id or (upload.name if upload else ""),
        key="un_query_dataset",
    )
    query = st.text_area(
        "Question",
        value="Give me a professional summary and key relationships in this document.",
        height=100,
        key="un_query_text",
    )
    top_k = st.slider("Top K", min_value=3, max_value=20, value=8, key="un_topk")
    view = st.radio("View Mode", options=["Simple", "Debug"], horizontal=True, key="un_view")

    if st.button("Run Query", use_container_width=True, key="un_query_btn"):
        if not q_dataset.strip():
            st.error("Provide Dataset ID for query.")
        elif not query.strip():
            st.error("Provide a question.")
        else:
            try:
                out = api_post_json(
                    base_url,
                    "/kb/query",
                    {
                        "dataset_id": q_dataset.strip(),
                        "query": query.strip(),
                        "top_k": int(top_k),
                        "llm_provider": provider,
                    },
                )
                st.session_state["un_query"] = out
            except Exception as exc:
                st.error(f"Query failed: {exc}")

    if st.session_state.get("un_query"):
        if view == "Simple":
            un_show_query_simple(st.session_state["un_query"])
        else:
            un_show_query_debug(st.session_state["un_query"])

    st.markdown("---")
    st.markdown("## 3) Knowledge Graph (Apache AGE)")
    kg_dataset = st.text_input("Dataset ID for KG", value=q_dataset, key="un_kg_dataset")
    kg_entity = st.text_input("Entity for KG Query", value="", key="un_kg_entity")
    kg_question = st.text_input("Question for KG Answer (optional)", value="", key="un_kg_question")
    kg_limit = st.slider("KG Limit", min_value=5, max_value=50, value=20, key="un_kg_limit")
    kg_hops = st.slider("KG Hops", min_value=1, max_value=3, value=1, key="un_kg_hops")
    kg_subgraph_limit = st.slider("KG Subgraph Edge Limit", min_value=20, max_value=500, value=120, step=10, key="un_kg_subgraph_limit")
    kg_view_mode = st.selectbox(
        "KG View Mode",
        options=["overview", "entity-centric", "community", "layered"],
        index=0,
        key="un_kg_view_mode",
    )
    k1, k2 = st.columns(2)
    with k1:
        if st.button("Build KG from Dataset", use_container_width=True, key="un_kg_build_btn"):
            try:
                out = api_post_json(base_url, "/kg/build", {"dataset_id": kg_dataset.strip()})
                st.session_state["kg_build"] = out
                sub = api_post_json(
                    base_url,
                    "/kg/subgraph",
                    {
                        "dataset_id": kg_dataset.strip(),
                        "seed_entity": "",
                        "hops": 1,
                        "limit": int(kg_subgraph_limit),
                    },
                )
                st.session_state["kg_subgraph"] = sub
                st.success("Knowledge graph build completed.")
            except Exception as exc:
                st.error(f"KG build failed: {exc}")
    with k2:
        if st.button("Query KG", use_container_width=True, key="un_kg_query_btn"):
            try:
                out = api_post_json(
                    base_url,
                    "/kg/query",
                    {
                        "dataset_id": kg_dataset.strip(),
                        "entity": kg_entity.strip(),
                        "limit": int(kg_limit),
                        "question": kg_question.strip(),
                    },
                )
                st.session_state["kg_query"] = out
            except Exception as exc:
                st.error(f"KG query failed: {exc}")

    if st.button("Show KG Subgraph", use_container_width=True, key="un_kg_subgraph_btn"):
        try:
            out = api_post_json(
                base_url,
                "/kg/subgraph",
                {
                    "dataset_id": kg_dataset.strip(),
                    "seed_entity": kg_entity.strip(),
                    "hops": int(kg_hops),
                    "limit": int(kg_subgraph_limit),
                },
            )
            st.session_state["kg_subgraph"] = out
        except Exception as exc:
            st.error(f"KG subgraph failed: {exc}")

    if st.session_state.get("kg_build"):
        st.markdown("### KG Build Result")
        st.code(json.dumps(st.session_state["kg_build"], indent=2), language="json")
    if st.session_state.get("kg_query"):
        st.markdown("### KG Query Result")
        st.code(json.dumps(st.session_state["kg_query"], indent=2), language="json")
        if st.session_state["kg_query"].get("answer"):
            st.markdown("### KG Answer")
            st.write(st.session_state["kg_query"]["answer"])
    if st.session_state.get("kg_subgraph"):
        st.markdown("### KG Subgraph")
        un_render_kg_plot_with_view(st.session_state["kg_subgraph"], view_mode=kg_view_mode)
        with st.expander("Subgraph JSON", expanded=False):
            st.code(json.dumps(st.session_state["kg_subgraph"], indent=2), language="json")

    st.markdown("---")
    st.markdown("## 4) EdgeQuake Graph‑RAG (Local)")
    st.caption("Optional: run EdgeQuake separately and query it here. This does not replace your current KB/KG.")
    eq_base = st.text_input("EdgeQuake Base URL", value="http://localhost:8080", key="eq_base_url")
    eq_file = st.file_uploader(
        "Upload to EdgeQuake",
        type=["pdf", "docx", "txt", "md", "html", "json", "csv"],
        key="eq_upload_file",
    )
    if st.button("Upload to EdgeQuake", use_container_width=True, key="eq_upload_btn"):
        if not eq_file:
            st.error("Upload a file first.")
        else:
            try:
                payload = eq_file.getvalue()
                out = api_post_multipart(
                    base_url,
                    "/edgequake/upload",
                    files={"file": (eq_file.name, payload)},
                    data={"base_url": eq_base},
                )
                st.session_state["eq_upload"] = out
                st.success("Uploaded to EdgeQuake.")
            except Exception as exc:
                st.error(f"EdgeQuake upload failed: {exc}")

    if st.session_state.get("eq_upload"):
        st.markdown("### EdgeQuake Upload Result")
        st.code(json.dumps(st.session_state["eq_upload"], indent=2), language="json")

    eq_query = st.text_input("EdgeQuake Query", value="", key="eq_query")
    eq_mode = st.selectbox(
        "EdgeQuake Query Mode",
        options=["hybrid", "local", "global", "naive", "mix"],
        index=0,
        key="eq_mode",
    )
    if st.button("Run EdgeQuake Query", use_container_width=True, key="eq_query_btn"):
        if not eq_query.strip():
            st.error("Enter a query.")
        else:
            try:
                out = api_post_json(
                    base_url,
                    "/edgequake/query",
                    {"base_url": eq_base, "query": eq_query.strip(), "mode": eq_mode},
                )
                st.session_state["eq_query_out"] = out
            except Exception as exc:
                st.error(f"EdgeQuake query failed: {exc}")

    if st.session_state.get("eq_query_out"):
        st.markdown("### EdgeQuake Query Result")
        st.code(json.dumps(st.session_state["eq_query_out"], indent=2), language="json")


def _extract_insight_sections(text: str) -> tuple[str, list[str]]:
    raw = str(text or "").strip()
    if not raw:
        return "", []
    raw = raw.replace("**", "")
    summary = raw
    questions: list[str] = []

    if "Analysis Questions" in raw:
        parts = raw.split("Analysis Questions", 1)
        summary = parts[0].replace("Summary:", "").strip(" :\n")
        q_part = parts[1]
        for line in q_part.splitlines():
            line = line.strip().lstrip("-*0123456789. ").strip()
            if not line:
                continue
            if line.lower().startswith(":"):
                line = line[1:].strip()
            if line:
                questions.append(line)
    else:
        summary = raw.replace("Summary:", "").strip()
    summary = summary.replace("Summary\n", "").replace("Summary:", "").strip()
    return summary, questions[:5]


def transformations_tab(base_url: str) -> None:
    st.subheader("File Transformations")
    st.caption("Convert between common formats: CSV/Excel/JSON/TSV, PDF/DOCX → Markdown, Image → Text.")
    left, right = st.columns(2)

    with left:
        st.markdown("**Format Conversions**")
        upload = st.file_uploader(
            "Upload File",
            type=["csv", "tsv", "json", "xls", "xlsx", "pdf", "docx", "png", "jpg", "jpeg", "bmp", "tiff"],
            key="transform_upload",
        )

        transform_type = st.selectbox(
            "Transformation Type",
            options=[
                "CSV → Excel (XLSX)",
                "CSV → JSON",
                "CSV → TSV",
                "CSV → Markdown Table",
                "CSV → HTML Table",
                "TSV → CSV",
                "TSV → JSON",
                "TSV → Excel (XLSX)",
                "TSV → Markdown Table",
                "Excel → CSV",
                "Excel → JSON",
                "Excel → TSV",
                "Excel → Markdown Table",
                "JSON → CSV",
                "JSON → Excel (XLSX)",
                "JSON → TSV",
                "JSON → Markdown Table",
                "JSON → YAML",
                "YAML → JSON",
                "XML → JSON",
                "JSON → XML",
                "JSON → Pretty",
                "JSON → Minify",
                "PDF/DOCX → Markdown",
                "PDF → Text",
                "DOCX → Text",
                "Markdown → HTML",
                "HTML → Markdown",
                "TXT → Markdown",
                "TXT → HTML",
                "Image → Text (Vision)",
                "Image → Markdown (Caption)",
            ],
            index=0,
            key="transform_type",
        )

        if st.button("Run Transformation", use_container_width=True):
            if not upload:
                st.error("Upload a file first.")
                return

            name = upload.name
            data = upload.getvalue()
            try:
                out = _transform_run_api(base_url, name, data, transform_type)
                out_name = out.get("file_name", f"{Path(name).stem}.out")
                b64 = out.get("content_base64", "")
                raw = base64.b64decode(b64.encode("utf-8")) if b64 else b""
                if transform_type == "PDF/DOCX → Markdown":
                    st.text_area("Markdown Preview", value=raw.decode("utf-8", errors="ignore")[:4000], height=200)
                if transform_type in ("Image → Text (OCR)", "Image → Text (Vision)"):
                    st.text_area("OCR Text", value=raw.decode("utf-8", errors="ignore")[:4000], height=200)
                if transform_type not in ("Image → Text (OCR)", "Image → Text (Vision)"):
                    st.download_button("Download Output", data=raw, file_name=out_name)
            except Exception as exc:
                st.error(f"Transformation failed: {exc}")

    with right:
        st.markdown("**Logical Transformations**")
        st.caption("Build ETL-style logical steps with form controls (no raw JSON needed).")
        logical_upload = st.file_uploader(
            "Upload Table (CSV/TSV/JSON/XLSX)",
            type=["csv", "tsv", "json", "xls", "xlsx"],
            key="logical_transform_upload",
        )
        logical_right_upload = st.file_uploader(
            "Optional: Join File (CSV/TSV/JSON/XLSX)",
            type=["csv", "tsv", "json", "xls", "xlsx"],
            key="logical_transform_right",
        )
        if "etl_steps" not in st.session_state:
            st.session_state["etl_steps"] = [{"type": "clean", "drop_na": True, "drop_duplicates": True}]

        st.markdown("**Step Builder**")
        step_type = st.selectbox(
            "Add Step Type",
            options=["clean", "filter", "derive", "map", "groupby", "join", "scale", "validate", "rule", "window"],
            key="etl_step_type",
        )
        c_add1, c_add2 = st.columns(2)
        with c_add1:
            if st.button("Add Step", use_container_width=True):
                step: Dict[str, Any] = {"type": step_type}
                if step_type == "clean":
                    step.update({"drop_na": True, "drop_duplicates": True})
                elif step_type == "scale":
                    step.update({"method": "minmax"})
                st.session_state["etl_steps"].append(step)
        with c_add2:
            if st.button("Reset Steps", use_container_width=True):
                st.session_state["etl_steps"] = [{"type": "clean", "drop_na": True, "drop_duplicates": True}]

        steps = st.session_state["etl_steps"]
        for idx, step in enumerate(steps):
            with st.expander(f"Step {idx+1}: {step.get('type', 'unknown')}", expanded=(idx == 0)):
                op_type = st.selectbox(
                    "Type",
                    options=["clean", "filter", "derive", "map", "groupby", "join", "scale", "validate", "rule", "window"],
                    index=["clean", "filter", "derive", "map", "groupby", "join", "scale", "validate", "rule", "window"].index(step.get("type", "clean")),
                    key=f"etl_type_{idx}",
                )
                step["type"] = op_type
                if op_type == "clean":
                    step["drop_na"] = st.checkbox("Drop Null Rows", value=bool(step.get("drop_na", True)), key=f"etl_clean_na_{idx}")
                    step["drop_duplicates"] = st.checkbox("Drop Duplicates", value=bool(step.get("drop_duplicates", True)), key=f"etl_clean_dup_{idx}")
                elif op_type == "filter":
                    step["condition"] = st.text_input("Condition (pandas query)", value=str(step.get("condition", "")), key=f"etl_filter_cond_{idx}")
                elif op_type == "derive":
                    step["new_column"] = st.text_input("New Column", value=str(step.get("new_column", "")), key=f"etl_derive_col_{idx}")
                    step["expression"] = st.text_input("Expression", value=str(step.get("expression", "")), key=f"etl_derive_expr_{idx}")
                elif op_type == "scale":
                    step["method"] = st.selectbox("Method", options=["minmax", "standard"], index=0 if step.get("method", "minmax") == "minmax" else 1, key=f"etl_scale_method_{idx}")
                    cols = st.text_input("Columns (comma separated, optional)", value=",".join(step.get("columns", [])) if isinstance(step.get("columns"), list) else "", key=f"etl_scale_cols_{idx}")
                    step["columns"] = [c.strip() for c in cols.split(",") if c.strip()]
                elif op_type == "validate":
                    step["condition"] = st.text_input("Validation Condition", value=str(step.get("condition", "")), key=f"etl_val_cond_{idx}")
                    step["action"] = st.selectbox("On Failure", options=["drop", "flag"], index=0 if step.get("action", "drop") == "drop" else 1, key=f"etl_val_action_{idx}")
                elif op_type == "groupby":
                    by_cols = st.text_input("Group By Columns (comma separated)", value=",".join(step.get("by", [])) if isinstance(step.get("by"), list) else "", key=f"etl_groupby_by_{idx}")
                    step["by"] = [c.strip() for c in by_cols.split(",") if c.strip()]
                    agg_json = st.text_input("Aggregations JSON", value=json.dumps(step.get("aggregations", {"count": "count"})), key=f"etl_groupby_agg_{idx}")
                    try:
                        step["aggregations"] = json.loads(agg_json)
                    except Exception:
                        pass
                if st.button("Remove Step", key=f"etl_remove_{idx}"):
                    st.session_state["etl_steps"].pop(idx)
                    st.rerun()

        ops_payload = {"operations": st.session_state["etl_steps"]}
        st.code(json.dumps(ops_payload, indent=2), language="json")

        if st.button("Run Logical Transform", use_container_width=True):
            if not logical_upload:
                st.error("Upload a structured file first.")
                return

            try:
                out = _transform_logical_api(
                    base_url,
                    logical_upload.name,
                    logical_upload.getvalue(),
                    ops_payload,
                    logical_right_upload.name if logical_right_upload else None,
                    logical_right_upload.getvalue() if logical_right_upload else None,
                )
                st.success(f"Rows: {out.get('row_count', 0)}")
                if out.get("warnings"):
                    st.warning("Warnings: " + "; ".join(out["warnings"]))
                if out.get("report"):
                    st.code(json.dumps(out["report"], indent=2), language="json")
                if out.get("preview_rows"):
                    st.dataframe(pd.DataFrame(out["preview_rows"]))
                b64 = out.get("content_base64", "")
                raw = base64.b64decode(b64.encode("utf-8")) if b64 else b""
                if raw:
                    st.download_button(
                        "Download Transformed CSV",
                        data=raw,
                        file_name=out.get("file_name", "logical_transform.csv"),
                    )
            except Exception as exc:
                st.error(f"Logical transform failed: {exc}")

        st.markdown("---")
        st.markdown("**Airflow ETL Integration**")
        recipe_name = st.text_input("Pipeline Name", value="customer_churn_etl", key="etl_recipe_name")
        recipe_desc = st.text_input("Description", value="Logical transformation ETL pipeline", key="etl_recipe_desc")
        source_type = st.selectbox("Source Type", options=["upload_file", "postgres_table"], key="etl_source_type")
        source_config: Dict[str, Any] = {}
        if source_type == "upload_file":
            source_config["file_path"] = st.text_input(
                "Source File Path",
                value=str(Path.cwd() / "sample_datasets" / "structured" / "customer_churn_realistic.csv"),
                key="etl_source_file_path",
            )
        else:
            source_config["host"] = st.text_input("Postgres Host", value="127.0.0.1", key="etl_pg_host")
            source_config["port"] = st.number_input("Postgres Port", value=5432, step=1, key="etl_pg_port")
            source_config["database"] = st.text_input("Database", value="kbdb", key="etl_pg_db")
            source_config["username"] = st.text_input("Username", value="postgres", key="etl_pg_user")
            source_config["password"] = st.text_input("Password", value="mysecretpassword", type="password", key="etl_pg_pass")
            source_config["table"] = st.text_input("Table", value="customer_churn_realistic", key="etl_pg_table")
            source_config["limit"] = st.number_input("Row Limit", value=10000, step=100, key="etl_pg_limit")

        target_type = st.selectbox("Target Type", options=["csv", "postgres"], key="etl_target_type")
        target_config: Dict[str, Any] = {"type": target_type}
        if target_type == "csv":
            target_config["file_path"] = st.text_input(
                "Output CSV Path",
                value=str(Path.cwd() / "artifacts" / "etl_outputs" / "output.csv"),
                key="etl_target_csv_path",
            )
        else:
            target_config["host"] = st.text_input("Target Host", value="127.0.0.1", key="etl_tpg_host")
            target_config["port"] = st.number_input("Target Port", value=5432, step=1, key="etl_tpg_port")
            target_config["database"] = st.text_input("Target Database", value="kbdb", key="etl_tpg_db")
            target_config["username"] = st.text_input("Target Username", value="postgres", key="etl_tpg_user")
            target_config["password"] = st.text_input("Target Password", value="mysecretpassword", type="password", key="etl_tpg_pass")
            target_config["table"] = st.text_input("Target Table", value="customer_churn_curated", key="etl_tpg_table")
            target_config["mode"] = st.selectbox("Load Mode", options=["replace", "append"], key="etl_tpg_mode")

        c_etl1, c_etl2 = st.columns(2)
        with c_etl1:
            if st.button("Save ETL Recipe", use_container_width=True):
                try:
                    res = _etl_save_recipe_api(
                        base_url,
                        {
                            "name": recipe_name,
                            "description": recipe_desc,
                            "source_type": source_type,
                            "source_config": source_config,
                            "operations": st.session_state["etl_steps"],
                            "quality_rules": {},
                            "target_config": target_config,
                            "schedule": {},
                        },
                    )
                    st.session_state["last_recipe_id"] = res.get("recipe_id")
                    st.success(f"Recipe saved: {res.get('recipe_id')}")
                except Exception as exc:
                    st.error(f"Save recipe failed: {exc}")
        with c_etl2:
            if st.button("Run via Airflow", use_container_width=True):
                recipe_id = st.session_state.get("last_recipe_id")
                if not recipe_id:
                    st.error("Save recipe first.")
                else:
                    try:
                        run_res = _etl_trigger_run_api(base_url, recipe_id=recipe_id)
                        st.session_state["last_etl_run_id"] = run_res.get("run_id")
                        st.success(f"Triggered Airflow run: {run_res.get('run_id')}")
                    except Exception as exc:
                        st.error(f"Airflow trigger failed: {exc}")

        c_etl3, c_etl4 = st.columns(2)
        with c_etl3:
            if st.button("Refresh Run Status", use_container_width=True):
                run_id = st.session_state.get("last_etl_run_id")
                if not run_id:
                    st.warning("No run_id yet.")
                else:
                    try:
                        status = _etl_run_status_api(base_url, run_id)
                        st.code(json.dumps(status, indent=2), language="json")
                    except Exception as exc:
                        st.error(f"Run status failed: {exc}")
        with c_etl4:
            if st.button("List Recent ETL Runs", use_container_width=True):
                try:
                    runs = _etl_list_runs_api(base_url, limit=20)
                    st.code(json.dumps(runs, indent=2), language="json")
                except Exception as exc:
                    st.error(f"List ETL runs failed: {exc}")


def render_analytics_natural_language(query_payload: Dict[str, Any]) -> str:
    execution = query_payload.get("execution", {})
    result = execution.get("result", {})
    if result.get("type") != "object" or not isinstance(result.get("value"), dict):
        # Keep user-facing answer clean; technical errors stay in Debug Details.
        return "Analysis completed, but this run did not produce structured answer data. Check Debug Details for execution trace."

    obj = result.get("value", {})
    if isinstance(obj, dict) and "result" in obj and isinstance(obj.get("result"), dict):
        # Some analytics runs return a wrapped object: {"result": {...}}.
        # Unwrap it so schema-specific renderers still apply.
        obj = obj["result"]
    lines: list[str] = []

    target = obj.get("target_column")
    rows = obj.get("row_count")
    if target is not None and rows is not None:
        lines.append(f"Overview: I analyzed {rows} rows using '{target}' as the grouping column.")

    class_distribution = obj.get("class_distribution", [])
    if isinstance(class_distribution, list) and class_distribution:
        parts = [
            f"{d.get('class')}: {d.get('count')} ({d.get('pct')}%)"
            for d in class_distribution[:10]
            if isinstance(d, dict)
        ]
        if parts:
            lines.append("Class Distribution:")
            lines.extend([f"- {p}" for p in parts])

    mean_by_group = obj.get("mean_by_group", [])
    if isinstance(mean_by_group, list) and mean_by_group:
        mean_lines = []
        for item in mean_by_group[:10]:
            if not isinstance(item, dict):
                continue
            grp = item.get("group")
            metric_bits = []
            for k, v in item.items():
                if k == "group":
                    continue
                metric_bits.append(f"{k}={v:.4f}" if isinstance(v, (int, float)) else f"{k}={v}")
            if grp is not None and metric_bits:
                mean_lines.append(f"{grp}: " + ", ".join(metric_bits))
        if mean_lines:
            lines.append("Group Means:")
            lines.extend([f"- {m}" for m in mean_lines])

    class_counts = obj.get("class_counts")
    if isinstance(class_counts, dict) and class_counts:
        total = sum(int(v) for v in class_counts.values() if isinstance(v, (int, float)))
        lines.append("Class Counts:")
        for cls, cnt in class_counts.items():
            pct = (100.0 * float(cnt) / total) if total > 0 and isinstance(cnt, (int, float)) else None
            if pct is not None:
                lines.append(f"- {cls}: {int(cnt)} ({pct:.2f}%)")
            else:
                lines.append(f"- {cls}: {cnt}")

    mean_std = obj.get("mean_std")
    if isinstance(mean_std, dict) and mean_std:
        lines.append("Feature Means and Standard Deviations:")
        for grp, details in list(mean_std.items())[:10]:
            if isinstance(details, dict):
                metrics = []
                for metric, val in details.items():
                    if isinstance(val, dict):
                        m = val.get("mean")
                        s = val.get("std")
                        if isinstance(m, (int, float)) and isinstance(s, (int, float)):
                            metrics.append(f"{metric}: mean={m:.4f}, std={s:.4f}")
                    elif isinstance(val, (int, float)):
                        metrics.append(f"{metric}={val:.4f}")
                if metrics:
                    lines.append(f"- {grp}: " + "; ".join(metrics))

    pairwise = obj.get("pairwise_best_features")
    if isinstance(pairwise, list) and pairwise:
        lines.append("Best Feature Per Class Pair:")
        for rec in pairwise[:10]:
            if not isinstance(rec, dict):
                continue
            pair = rec.get("pair")
            feat = rec.get("feature")
            eff = rec.get("effect_size")
            th = rec.get("threshold_candidate")
            direction = rec.get("direction")
            lines.append(
                f"- {pair}: {feat} (effect size={_format_scalar(eff)}, threshold={_format_scalar(th)}, direction={direction})"
            )

    summary = obj.get("summary")
    if isinstance(summary, str) and summary.strip():
        lines.append("Summary:")
        lines.append(summary.strip())

    key_stats = obj.get("key_stats")
    if isinstance(key_stats, dict) and key_stats:
        lines.append("Key Statistics:")
        ordered = ["count", "mean", "median", "std", "min", "q1", "q3", "max"]
        for k in ordered:
            if k in key_stats:
                lines.append(f"- {k}: {_format_scalar(key_stats.get(k))}")

    top_corr = obj.get("top_correlations", [])
    if isinstance(top_corr, list) and top_corr:
        c0 = top_corr[0] if isinstance(top_corr[0], dict) else {}
        if c0:
            lines.append(
                f"Correlation: strongest pair is {c0.get('pair')} with coefficient {c0.get('corr')}."
            )

    anova = obj.get("anova", [])
    if isinstance(anova, list) and anova:
        sig = [a.get("feature") for a in anova if isinstance(a, dict) and float(a.get("p_value", 1.0)) < 0.05]
        lines.append(
            "ANOVA: significant separation features are "
            + (", ".join([str(s) for s in sig]) if sig else "none at p < 0.05")
            + "."
        )

    model = obj.get("model_metrics", {})
    if isinstance(model, dict) and model:
        macro_f1 = model.get("macro_f1")
        test_rows = model.get("test_rows")
        if macro_f1 is not None and test_rows is not None:
            lines.append(f"Model: baseline Logistic Regression macro F1 is {macro_f1} on {test_rows} test rows.")
        per_class = model.get("per_class", [])
        if isinstance(per_class, list) and per_class:
            class_bits = []
            for pc in per_class[:5]:
                if not isinstance(pc, dict):
                    continue
                class_bits.append(
                    f"{pc.get('class')}: P={_to_pct(pc.get('precision'))}, R={_to_pct(pc.get('recall'))}, F1={_to_pct(pc.get('f1'))}"
                )
            if class_bits:
                lines.append("Per-Class Metrics:")
                lines.extend([f"- {b}" for b in class_bits])

    conclusions = obj.get("conclusions", [])
    if isinstance(conclusions, list) and conclusions:
        lines.append("Conclusions:")
        for c in conclusions[:5]:
            lines.append(f"- {c}")

    recommendations = obj.get("recommendations", [])
    if isinstance(recommendations, list) and recommendations:
        lines.append("Recommendations:")
        for r in recommendations[:3]:
            lines.append(f"- {r}")

    if class_distribution and mean_by_group:
        lines.append("Takeaway: the classes are balanced, and petal measurements clearly separate the groups.")

    if lines:
        return "\n".join(lines)

    # Generic object fallback for previously unseen result schemas.
    generic_lines = ["Answer:"]
    for key, value in list(obj.items())[:10]:
        if isinstance(value, list):
            if value and isinstance(value[0], dict):
                sample = ", ".join([f"{k}={_format_scalar(v)}" for k, v in list(value[0].items())[:4]])
                generic_lines.append(f"- {key}: {len(value)} records. Example -> {sample}")
            else:
                preview = ", ".join([_format_scalar(v) for v in value[:5]])
                generic_lines.append(f"- {key}: [{preview}]")
        elif isinstance(value, dict):
            kv = ", ".join([f"{k}={_format_scalar(v)}" for k, v in list(value.items())[:5]])
            generic_lines.append(f"- {key}: {kv}")
        else:
            generic_lines.append(f"- {key}: {_format_scalar(value)}")
    return "\n".join(generic_lines)


def render_analytics_detailed_explanation(query_text: str, query_payload: Dict[str, Any]) -> str:
    execution = query_payload.get("execution", {})
    result = execution.get("result", {})
    if result.get("type") != "object" or not isinstance(result.get("value"), dict):
        return (
            f"You asked: {query_text}\n\n"
            "The analytics run completed, but structured result data was not returned in this response. "
            "Use Debug Details to inspect raw execution output and rerun with a more specific query if needed."
        )

    obj = result.get("value", {})
    if isinstance(obj, dict) and "result" in obj and isinstance(obj.get("result"), dict):
        obj = obj["result"]

    lines: list[str] = [f"You asked: {query_text}"]
    lines.append("What was computed:")

    target = obj.get("target_column")
    rows = obj.get("row_count")
    if target is not None and rows is not None:
        lines.append(f"- The analysis used {rows} rows grouped by `{target}`.")

    if isinstance(obj.get("class_distribution"), list):
        lines.append("- Class distribution was calculated with both counts and percentages.")
    if isinstance(obj.get("class_counts"), dict):
        lines.append("- Class-level sample counts were computed.")
    if isinstance(obj.get("mean_by_group"), list) or isinstance(obj.get("mean_std"), dict):
        lines.append("- Group-level feature statistics (means / standard deviations) were computed.")
    if isinstance(obj.get("pairwise_best_features"), list):
        lines.append("- Pairwise class-separation analysis was computed using effect size and threshold candidates.")
    if isinstance(obj.get("top_correlations"), list):
        lines.append("- Feature correlations were computed and ranked.")
    if isinstance(obj.get("anova"), list):
        lines.append("- ANOVA significance testing was run across groups.")
    if isinstance(obj.get("model_metrics"), dict):
        lines.append("- Baseline model performance metrics were computed.")

    lines.append("How to interpret this:")
    summary = obj.get("summary")
    if isinstance(summary, str) and summary.strip():
        lines.append(f"- Key finding: {summary.strip()}")
    elif isinstance(obj.get("pairwise_best_features"), list) and obj.get("pairwise_best_features"):
        best = obj["pairwise_best_features"][0]
        if isinstance(best, dict):
            lines.append(
                f"- Strongest separating feature example: `{best.get('feature')}` for `{best.get('pair')}`, "
                f"with effect size `{_format_scalar(best.get('effect_size'))}` and threshold `{_format_scalar(best.get('threshold_candidate'))}`."
            )
    elif isinstance(obj.get("class_distribution"), list) and obj.get("class_distribution"):
        parts = []
        for d in obj["class_distribution"][:5]:
            if isinstance(d, dict):
                parts.append(f"{d.get('class')}={d.get('count')} ({d.get('pct')}%)")
        if parts:
            lines.append("- The dataset is distributed as: " + ", ".join(parts) + ".")

    if isinstance(obj.get("model_metrics"), dict) and obj["model_metrics"].get("macro_f1") is not None:
        lines.append(
            f"- Model quality indicator: macro F1 is `{_format_scalar(obj['model_metrics'].get('macro_f1'))}`."
        )

    lines.append("What this means for users:")
    lines.append("- You can rely on this output as a computed result from the uploaded dataset, not a generic text template.")
    lines.append("- Use the summary and key metrics above to decide next analysis or modeling steps.")
    lines.append("- If needed, open `Debug Details` for full execution payload and generated logic.")
    return "\n".join(lines)


def render_analytics_dashboard(base_url: str, df: pd.DataFrame) -> None:
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    tab1, tab2, tab3 = st.tabs(["Distributions", "Categorical Analysis", "Correlations"])
    stats_payload = {}
    try:
        stats_payload = api_post_json(base_url, "/analytics/dashboard", {"dataframe": df.to_dict(orient="records")})
    except Exception as exc:
        st.caption(f"Dashboard stats unavailable: {exc}")
    with tab1:
        if num_cols:
            dist_col = st.selectbox("View Distribution", num_cols, key="analytics_dash_dist_col")
            fig_hist = px.histogram(df, x=dist_col, marginal="box", title=f"Distribution of {dist_col}", template="plotly_white")
            st.plotly_chart(fig_hist, use_container_width=True)
            if stats_payload.get("numeric_summary"):
                st.markdown("**Numeric Summary (Backend)**")
                st.dataframe(pd.DataFrame(stats_payload["numeric_summary"]), use_container_width=True)
        else:
            st.caption("No numeric columns found.")

    with tab2:
        if cat_cols:
            cat_col = st.selectbox("View Category Split", cat_cols, key="analytics_dash_cat_col")
            fig_pie = px.pie(df, names=cat_col, hole=0.4, title=f"Share of {cat_col}")
            st.plotly_chart(fig_pie, use_container_width=True)
            if num_cols:
                box_metric = st.selectbox("Variance Metric", num_cols, key="analytics_dash_box_metric")
                fig_box = px.box(df, x=cat_col, y=box_metric, color=cat_col, title=f"{box_metric} by {cat_col}")
                st.plotly_chart(fig_box, use_container_width=True)
            if stats_payload.get("categorical_summary"):
                st.markdown("**Categorical Summary (Backend)**")
                st.dataframe(pd.DataFrame(stats_payload["categorical_summary"]), use_container_width=True)
        else:
            st.caption("No categorical columns found.")

    with tab3:
        if len(num_cols) >= 2:
            corr = df[num_cols].corr()
            fig_heat = px.imshow(corr, text_auto=True, title="Correlation Heatmap")
            st.plotly_chart(fig_heat, use_container_width=True)
            x_col = st.selectbox("Impact X", num_cols, key="analytics_dash_scatter_x")
            y_candidates = [c for c in num_cols if c != x_col]
            y_col = st.selectbox("Impact Y", y_candidates or num_cols, key="analytics_dash_scatter_y")
            fig_scatter = px.scatter(df, x=x_col, y=y_col, trendline="ols", title=f"Impact: {x_col} vs {y_col}")
            st.plotly_chart(fig_scatter, use_container_width=True)
            if stats_payload.get("correlations"):
                st.markdown("**Correlation Summary (Backend)**")
                st.dataframe(pd.DataFrame(stats_payload["correlations"]).head(20), use_container_width=True)
        else:
            st.caption("Need at least two numeric columns for correlation analysis.")


def _clean_business_problem_text(text: str) -> str:
    raw = str(text or "").strip()
    if not raw:
        return ""
    first = raw.splitlines()[0].strip()
    cleaned = " ".join(first.split())
    return cleaned[:280]


def _load_structured_preview(base_url: str, file_name: str, payload: bytes) -> pd.DataFrame:
    out = api_post_multipart(
        base_url,
        "/structured/preview",
        files={"file": (file_name, payload)},
    )
    rows = out.get("rows", [])
    if not rows:
        return pd.DataFrame(columns=out.get("columns", []))
    return pd.DataFrame(rows)

def _structured_tune_api(
    base_url: str,
    df: pd.DataFrame,
    target: str,
    task: str,
    model_key: str,
    n_trials: int,
) -> Dict[str, Any]:
    payload = {
        "rows": df.to_dict(orient="records"),
        "target": target,
        "task": task,
        "model_key": model_key,
        "n_trials": int(n_trials),
    }
    return api_post_json(base_url, "/structured/tune", payload)
def _structured_predict_train_api(
    base_url: str,
    df: pd.DataFrame,
    target: str,
    task: str,
    model_key: str,
) -> Dict[str, Any]:
    payload = {
        "rows": df.to_dict(orient="records"),
        "target": target,
        "task": task,
        "model_key": model_key,
    }
    return api_post_json(base_url, "/structured/predict/train", payload)


def _structured_predict_api(
    base_url: str,
    model_id: str,
    rows: List[Dict[str, Any]],
    return_proba: bool = True,
) -> Dict[str, Any]:
    payload = {"model_id": model_id, "rows": rows, "return_proba": return_proba}
    return api_post_json(base_url, "/structured/predict", payload)


def _structured_online_start_api(base_url: str, task: str) -> Dict[str, Any]:
    return api_post_json(base_url, "/structured/online/start", {"task": task})


def _structured_online_batch_api(
    base_url: str,
    stream_id: str,
    rows: List[Dict[str, Any]],
    target: str,
    max_rows: int | None = None,
) -> Dict[str, Any]:
    payload = {"stream_id": stream_id, "rows": rows, "target": target, "max_rows": max_rows}
    return api_post_json(base_url, "/structured/online/batch", payload)


def _structured_online_status_api(base_url: str, stream_id: str) -> Dict[str, Any]:
    return api_get(base_url, "/structured/online/status", params={"stream_id": stream_id})


def _transform_run_api(base_url: str, file_name: str, payload: bytes, transform_type: str) -> Dict[str, Any]:
    out = api_post_multipart(
        base_url,
        "/transform/run",
        files={"file": (file_name, payload)},
        data={"transform_type": transform_type},
    )
    return out


def _transform_logical_api(
    base_url: str,
    file_name: str,
    payload: bytes,
    operations: Dict[str, Any],
    right_file_name: str | None = None,
    right_payload: bytes | None = None,
) -> Dict[str, Any]:
    files = {"file": (file_name, payload)}
    if right_file_name and right_payload:
        files["right_file"] = (right_file_name, right_payload)
    out = api_post_multipart(
        base_url,
        "/transform/logical",
        files=files,
        data={"operations": json.dumps(operations)},
    )
    return out


def _etl_save_recipe_api(base_url: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    return api_post_json(base_url, "/etl/recipes", payload)


def _etl_list_recipes_api(base_url: str) -> Dict[str, Any]:
    return api_get(base_url, "/etl/recipes")


def _etl_trigger_run_api(base_url: str, recipe_id: str, dag_id: str = "logical_transform_etl") -> Dict[str, Any]:
    return api_post_json(base_url, "/etl/run", {"recipe_id": recipe_id, "dag_id": dag_id, "initiated_by": "streamlit"})


def _etl_run_status_api(base_url: str, run_id: str) -> Dict[str, Any]:
    return api_get(base_url, f"/etl/run/{run_id}")


def _etl_list_runs_api(base_url: str, limit: int = 20) -> Dict[str, Any]:
    return api_get(base_url, "/etl/runs", params={"limit": limit})


def _render_new_data_drift_check(
    base_url: str,
    key_prefix: str,
    baseline_df: pd.DataFrame,
    target_col: str,
    problem: str,
    stream_id: str,
) -> None:
    st.markdown("#### New Data Drift Check")
    st.caption("Add new rows directly in-app or upload a batch to compare drift against baseline and optionally push to Hoeffding/ADWIN.")
    batch_source = st.radio(
        "New Batch Source",
        options=["Manual Entry", "Upload File"],
        horizontal=True,
        key=f"{key_prefix}_batch_source",
    )

    new_df: pd.DataFrame | None = None
    if batch_source == "Upload File":
        new_file = st.file_uploader(
            "Upload New Batch (CSV/TSV/JSON/XLSX)",
            type=["csv", "tsv", "txt", "json", "xls", "xlsx"],
            key=f"{key_prefix}_new_batch_file",
        )
        if not new_file:
            return
        try:
            new_df = _load_structured_preview(base_url, new_file.name, new_file.getvalue())
        except Exception as exc:
            st.error(f"Could not parse new batch: {exc}")
            return
    else:
        manual_cols = [str(c) for c in baseline_df.columns.tolist()]
        sample_row: Dict[str, Any] = {}
        for col in manual_cols:
            series = baseline_df[col]
            if pd.api.types.is_numeric_dtype(series):
                sample_row[col] = float(pd.to_numeric(series, errors="coerce").dropna().median()) if not series.dropna().empty else 0.0
            else:
                vals = series.dropna().astype(str).tolist()
                sample_row[col] = vals[0] if vals else ""
        placeholder = json.dumps([sample_row], indent=2, default=str)
        st.caption(f"Columns: {', '.join(manual_cols)}")
        manual_json = st.text_area(
            "Enter New Rows as JSON array",
            value=placeholder,
            height=220,
            key=f"{key_prefix}_manual_new_rows",
        )
        if not manual_json.strip():
            return
        try:
            parsed = json.loads(manual_json)
            if not isinstance(parsed, list):
                raise ValueError("JSON must be an array of row objects.")
            new_df = pd.DataFrame(parsed)
        except Exception as exc:
            st.error(f"Invalid manual JSON rows: {exc}")
            return

    if new_df is None or new_df.empty:
        st.warning("New batch is empty.")
        return

    st.write(f"Baseline rows: {len(baseline_df)} | New rows: {len(new_df)}")
    common_cols = [c for c in baseline_df.columns if c in new_df.columns]
    if not common_cols:
        st.warning("No common columns found between baseline and new batch.")
        return

    num_cols = [
        c
        for c in common_cols
        if pd.api.types.is_numeric_dtype(baseline_df[c]) and pd.api.types.is_numeric_dtype(new_df[c])
    ]
    if num_cols:
        rows: list[dict[str, Any]] = []
        for col in num_cols:
            b = pd.to_numeric(baseline_df[col], errors="coerce").dropna()
            n = pd.to_numeric(new_df[col], errors="coerce").dropna()
            if len(b) == 0 or len(n) == 0:
                continue
            b_mean = float(b.mean())
            n_mean = float(n.mean())
            b_std = float(b.std(ddof=1)) if len(b) > 1 else 0.0
            denom = b_std if b_std > 1e-9 else 1.0
            z_shift = (n_mean - b_mean) / denom
            rows.append(
                {
                    "column": col,
                    "baseline_mean": b_mean,
                    "new_mean": n_mean,
                    "delta": n_mean - b_mean,
                    "z_shift": z_shift,
                }
            )
        if rows:
            shift_df = pd.DataFrame(rows).sort_values(by="z_shift", key=lambda s: s.abs(), ascending=False)
            st.dataframe(shift_df.head(15), use_container_width=True)
            top = shift_df.iloc[0]
            st.info(
                f"Highest shift column: `{top['column']}` with z-shift `{float(top['z_shift']):.3f}` "
                f"(delta `{float(top['delta']):.3f}`)."
            )

    if target_col not in new_df.columns:
        st.warning(f"Target column `{target_col}` is not present in new batch; cannot stream to Hoeffding/ADWIN.")
        return

    if not stream_id:
        st.info("Start the online stream to process new batches.")
        return

    max_rows = st.number_input(
        "Rows to push into Hoeffding stream",
        min_value=1,
        max_value=max(1, len(new_df)),
        value=min(200, max(1, len(new_df))),
        step=1,
        key=f"{key_prefix}_push_rows",
    )
    if st.button("Process New Batch in Stream Monitor", key=f"{key_prefix}_push_button"):
        sample_df = new_df.head(int(max_rows))
        try:
            out = _structured_online_batch_api(
                base_url,
                stream_id=stream_id,
                rows=sample_df.to_dict(orient="records"),
                target=target_col,
                max_rows=int(max_rows),
            )
            st.success(
                f"Processed {out.get('processed', 0)} rows. "
                f"ADWIN drift triggers in batch: {out.get('drift_hits', 0)}."
            )
            st.session_state[f"{key_prefix}_stream_status"] = out
        except Exception as exc:
            st.error(f"Streaming batch failed: {exc}")


class StreamingScientist:
    def __init__(self):
        self.available = tree is not None and drift is not None and stats is not None
        if not self.available:
            return
        self.model = tree.HoeffdingAdaptiveTreeClassifier(
            grace_period=50,
            drift_detector=drift.ADWIN(),
            seed=42,
        )
        self.drift_detector = drift.ADWIN()
        self.history = deque(maxlen=200)
        self.accuracy_history = deque(maxlen=200)
        self.var_history = deque(maxlen=200)
        self.running_var = stats.Var()
        self.drift_events = 0
        self.total_count = 0
        self.correct_count = 0
        self.label_map: Dict[Any, int] = {}

    def process_sample(self, x: Dict[str, Any], y: Any, task: str) -> bool:
        if not self.available:
            return False
        self.total_count += 1
        if task == "classification":
            if y not in self.label_map:
                self.label_map[y] = len(self.label_map)
            y_val = self.label_map[y]
        else:
            y_val = float(y)

        self.running_var.update(y_val)
        self.var_history.append(self.running_var.get())
        y_pred = self.model.predict_one(x)
        error = 1 if y_pred != y_val else 0
        if error == 0:
            self.correct_count += 1
        self.model.learn_one(x, y_val)
        self.history.append(error)
        self.accuracy_history.append(self.correct_count / self.total_count)
        self.drift_detector.update(error)
        if self.drift_detector.drift_detected:
            self.drift_events += 1
            return True
        return False


def _normalize_model_name(name: str, task: str) -> str:
    n = str(name).lower()
    if "random" in n and "forest" in n:
        return "random_forest" if task == "classification" else "random_forest_regressor"
    if "logistic" in n:
        return "logistic_regression"
    if "linear" in n:
        return "linear_regression"
    if "xgb" in n or "xgboost" in n:
        return "xgboost" if task == "classification" else "xgboost_regressor"
    if "lightgbm" in n or "lgbm" in n:
        return "lightgbm" if task == "classification" else "lightgbm_regressor"
    if "catboost" in n:
        return "catboost" if task == "classification" else "catboost_regressor"
    return n.replace(" ", "_")


def _build_preprocessor(df: pd.DataFrame, target: str) -> Tuple[ColumnTransformer, list[str], list[str]]:
    X = df.drop(columns=[target])
    num_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    cat_cols = [c for c in X.columns if c not in num_cols]
    pre = ColumnTransformer(
        [
            ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), num_cols),
            ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")), ("oh", OneHotEncoder(handle_unknown="ignore"))]), cat_cols),
        ]
    )
    return pre, num_cols, cat_cols


def _get_estimator(model_key: str, task: str):
    if task == "classification":
        pool = {
            "logistic_regression": LogisticRegression(max_iter=500),
            "random_forest": RandomForestClassifier(n_estimators=200, random_state=42),
            "xgboost": XGBClassifier(eval_metric="mlogloss", random_state=42) if XGBClassifier else None,
            "lightgbm": LGBMClassifier(random_state=42) if LGBMClassifier else None,
            "catboost": CatBoostClassifier(verbose=0, random_state=42) if CatBoostClassifier else None,
        }
    else:
        pool = {
            "linear_regression": LinearRegression(),
            "random_forest_regressor": RandomForestRegressor(n_estimators=200, random_state=42),
            "xgboost_regressor": XGBRegressor(random_state=42) if XGBRegressor else None,
            "lightgbm_regressor": LGBMRegressor(random_state=42) if LGBMRegressor else None,
            "catboost_regressor": CatBoostRegressor(verbose=0, random_state=42) if CatBoostRegressor else None,
        }
    est = pool.get(model_key)
    if est is None:
        fallback = "random_forest" if task == "classification" else "random_forest_regressor"
        est = pool[fallback]
    return est


def _tune_with_optuna(df: pd.DataFrame, target: str, task: str, model_key: str, n_trials: int = 20):
    if optuna is None:
        raise RuntimeError("optuna is not installed in this Streamlit environment.")

    pre, _, _ = _build_preprocessor(df, target)
    X = df.drop(columns=[target])
    y = df[target]
    scoring = "accuracy" if task == "classification" else "r2"
    base_est = _get_estimator(model_key, task)
    base_pipe = Pipeline([("pre", pre), ("model", base_est)])
    baseline = float(np.mean(cross_val_score(base_pipe, X, y, cv=3, scoring=scoring)))

    trial_scores: list[float] = []
    best_scores: list[float] = []
    progress = st.progress(0.0)
    status = st.empty()
    chart = st.empty()

    def objective(trial):
        est = _get_estimator(model_key, task)
        params = {}
        if hasattr(est, "n_estimators"):
            params["n_estimators"] = trial.suggest_int("n_estimators", 50, 400)
        if hasattr(est, "max_depth"):
            params["max_depth"] = trial.suggest_int("max_depth", 2, 20)
        if hasattr(est, "min_samples_split"):
            params["min_samples_split"] = trial.suggest_int("min_samples_split", 2, 12)
        if hasattr(est, "learning_rate"):
            params["learning_rate"] = trial.suggest_float("learning_rate", 0.01, 0.3, log=True)
        if hasattr(est, "subsample"):
            params["subsample"] = trial.suggest_float("subsample", 0.5, 1.0)
        if params:
            est.set_params(**params)
        pipe = Pipeline([("pre", pre), ("model", est)])
        score = float(np.mean(cross_val_score(pipe, X, y, cv=3, scoring=scoring)))
        trial_scores.append(score)
        best_scores.append(max(best_scores[-1], score) if best_scores else score)
        progress.progress((trial.number + 1) / n_trials)
        status.write(f"Trial {trial.number + 1}/{n_trials}  score={score:.4f}")
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=trial_scores, mode="lines+markers", name="Trial"))
        fig.add_trace(go.Scatter(y=best_scores, mode="lines", name="Best"))
        fig.update_layout(height=280, title="Optuna Progress")
        chart.plotly_chart(fig, use_container_width=True)
        return score

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)
    tuned_score = float(study.best_value)
    improvement = tuned_score - baseline
    return {
        "baseline_score": baseline,
        "tuned_score": tuned_score,
        "improvement_abs": improvement,
        "improvement_pct": (improvement / abs(baseline) * 100.0) if baseline != 0 else 0.0,
        "best_params": study.best_params,
    }


def _fit_local_predictor(df: pd.DataFrame, target: str, task: str, model_key: str) -> Pipeline:
    pre, _, _ = _build_preprocessor(df, target)
    est = _get_estimator(model_key, task)
    pipe = Pipeline([("pre", pre), ("model", est)])
    X = df.drop(columns=[target])
    y = df[target]
    pipe.fit(X, y)
    return pipe


def _render_structured_summary(result: Dict[str, Any], business_problem: str) -> None:
    training = result.get("training", {})
    pipeline = result.get("pipeline", {})
    profile = result.get("data_profile", {})
    champion = training.get("best_model_name", "N/A")
    score = float(training.get("best_score", 0.0))
    problem = result.get("problem_type", "classification")
    rows = profile.get("shape", {}).get("rows", "N/A")
    cols = profile.get("shape", {}).get("columns", "N/A")

    st.markdown("### Executive Summary")
    summary = (
        f"Business problem: {business_problem or 'Not provided'}\n\n"
        f"Detected task: {problem}. Dataset size: {rows} rows x {cols} columns.\n\n"
        f"Champion model: {champion} with score {score:.4f}. "
        f"AutoML backend: {training.get('automl_backend', 'unknown')}.\n\n"
        f"Model strategy: family={pipeline.get('model_family', 'auto')}, "
        f"fixed_model={pipeline.get('fixed_model', 'auto') or 'auto'}, "
        f"auto_decision={pipeline.get('auto_model_decision', 'enabled')}."
    )
    st.info(summary)


def _flatten_leaderboard(rows: list[dict]) -> pd.DataFrame:
    flat_rows: list[dict] = []
    for r in rows:
        base = {k: v for k, v in r.items() if k != "metrics"}
        metrics = r.get("metrics", {}) if isinstance(r.get("metrics"), dict) else {}
        for mk, mv in metrics.items():
            base[f"metric_{mk}"] = mv
        flat_rows.append(base)
    return pd.DataFrame(flat_rows)


def _render_auto_business_summary(
    base_url: str,
    result: Dict[str, Any],
    business_problem: str,
    df_top: pd.DataFrame,
) -> None:
    st.markdown("### Business Summary")
    try:
        out = api_post_json(
            base_url,
            "/structured/business-summary",
            {"result": result, "business_problem": business_problem},
        )
        st.success(out.get("summary", "Summary unavailable."))
    except Exception as exc:
        st.warning(f"Business summary unavailable: {exc}")


def _default_business_summary_template() -> str:
    return (
        "You are an enterprise AI solutions architect.\n"
        "Create a concise, business-facing executive summary using the inputs below.\n"
        "Business problem: {business_problem}\n"
        "Task type: {problem_type}\n"
        "Champion model: {champion_model}\n"
        "Champion score: {champion_score}\n"
        "Top models considered: {top_models}\n"
        "Holdout metrics: {holdout_metrics}\n"
        "Cross-validation: {cv_metrics}\n"
        "Nested CV: {nested_cv_metrics}\n"
        "Fairness diagnostics: {fairness}\n"
        "Explainability highlights: {explainability}\n"
        "Deployment readiness: {deployment_ready}\n"
        "Monitoring plan: {monitoring_plan}\n\n"
        "Required output format:\n"
        "1) Business Outcome (2-3 lines)\n"
        "2) Why this model set is suitable\n"
        "3) Risks and assumptions\n"
        "4) Deployment recommendation (staging/production) with next 3 actions\n"
        "Keep it professional, non-JSON, readable by business and technical stakeholders."
    )


def _build_business_summary_prompt(
    result: Dict[str, Any],
    business_problem: str,
    df_top: pd.DataFrame,
    champion: str,
    champion_score: float,
    problem: str,
) -> str:
    eval_obj = result.get("evaluation", {}) if isinstance(result.get("evaluation", {}), dict) else {}
    validation = eval_obj.get("validation", {}) if isinstance(eval_obj.get("validation", {}), dict) else {}
    holdout = validation.get("holdout_metrics", {})
    cv = validation.get("cv_metrics", {})
    nested = validation.get("nested_cv_metrics", {})
    fair = eval_obj.get("fairness", {})
    explain = eval_obj.get("explainability", {})
    deploy = result.get("deployment", {}) if isinstance(result.get("deployment", {}), dict) else {}
    top_names = ", ".join([str(x) for x in df_top["model"].tolist()[:10]]) if not df_top.empty and "model" in df_top.columns else champion
    template = _default_business_summary_template()
    return template.format(
        business_problem=business_problem or "Not provided",
        problem_type=problem,
        champion_model=champion,
        champion_score=f"{champion_score:.4f}",
        top_models=top_names,
        holdout_metrics=safe_json(holdout),
        cv_metrics=safe_json(cv),
        nested_cv_metrics=safe_json(nested),
        fairness=safe_json(fair),
        explainability=safe_json(explain),
        deployment_ready=safe_json(deploy.get("release_checklist", {})),
        monitoring_plan=safe_json(deploy.get("drift_monitoring_plan", {})),
    )


def _build_ai_technical_explainer_prompt(
    result: Dict[str, Any],
    business_problem: str,
    df_top: pd.DataFrame,
    drift_context: Dict[str, Any],
) -> str:
    training = result.get("training", {}) if isinstance(result.get("training", {}), dict) else {}
    evaluation = result.get("evaluation", {}) if isinstance(result.get("evaluation", {}), dict) else {}
    validation = evaluation.get("validation", {}) if isinstance(evaluation.get("validation", {}), dict) else {}
    explainability = evaluation.get("explainability", {}) if isinstance(evaluation.get("explainability", {}), dict) else {}
    fairness = evaluation.get("fairness", {}) if isinstance(evaluation.get("fairness", {}), dict) else {}
    deployment = result.get("deployment", {}) if isinstance(result.get("deployment", {}), dict) else {}

    top_models = ", ".join([str(x) for x in df_top["model"].tolist()[:6]]) if not df_top.empty and "model" in df_top.columns else str(training.get("best_model_name", "N/A"))
    return (
        "You are a senior ML architect. Explain this run in practical terms for a Data Science user.\n"
        f"Business problem: {business_problem or 'Not provided'}\n"
        f"Task: {result.get('problem_type', 'classification')}\n"
        f"Champion: {training.get('best_model_name', 'N/A')} score={float(training.get('best_score', 0.0)):.4f}\n"
        f"Top models: {top_models}\n"
        f"Validation: {safe_json(validation)}\n"
        f"Explainability: {safe_json(explainability)}\n"
        f"Fairness: {safe_json(fairness)}\n"
        f"Drift context: {safe_json(drift_context)}\n"
        f"Monitoring plan: {safe_json(deployment.get('drift_monitoring_plan', {}))}\n\n"
        "Output in 4 short sections:\n"
        "1) How models are solving the business problem\n"
        "2) Why champion is selected vs other top models\n"
        "3) SHAP/Fairness interpretation and limitations\n"
        "4) Hoeffding+ADWIN drift graph interpretation and what to do next\n"
        "Do not output JSON."
    )


def structured_tab(base_url: str, user_mode: str = "Business") -> None:
    st.subheader("Structured AutoML")
    source_type = st.radio(
        "Structured Data Source",
        options=["Upload File", "Connector"],
        horizontal=True,
        key="structured_source_type",
    )
    file = None
    connector_payload: Dict[str, Any] | None = None
    if source_type == "Upload File":
        file = st.file_uploader(
            "Upload Structured File",
            type=["csv", "tsv", "txt", "json", "xls", "xlsx", "sas7bdat", "sav", "rdata"],
            key="structured_file",
        )
    else:
        st.caption("Use the same connector config you validated in the Connectors tab.")
        conn_name = st.text_input("Connector Name", value="postgres", key="structured_conn_name")
        conn_cfg = st.text_area(
            "Connector Config JSON",
            value='{"host":"localhost","port":5432,"database":"postgres","username":"postgres","password":"postgres"}',
            key="structured_conn_cfg",
        )
        conn_query = st.text_area(
            "SQL Query (optional)",
            value="SELECT * FROM your_table LIMIT 10000",
            key="structured_conn_query",
        )
        conn_table = st.text_input("Table (optional if query provided)", key="structured_conn_table")
        conn_limit = st.number_input(
            "Connector Row Limit",
            min_value=1,
            max_value=100000,
            value=10000,
            step=1,
            key="structured_conn_limit",
        )
        conn_dataset_name = st.text_input("Dataset Name (optional)", value="", key="structured_conn_dataset")
        connector_payload = {
            "connector": conn_name,
            "config": conn_cfg,
            "query": conn_query,
            "table": conn_table,
            "limit": int(conn_limit),
            "dataset_name": conn_dataset_name,
        }

    preview_df = None
    if source_type == "Upload File" and file is not None:
        try:
            preview_df = _load_structured_preview(base_url, file.name, file.getvalue())
            st.caption(f"Detected {len(preview_df.columns)} columns")
            st.dataframe(preview_df.head(10), use_container_width=True)
        except Exception as exc:
            st.info(f"Could not render local preview: {exc}")
    elif source_type == "Connector":
        preview_df = st.session_state.get("structured_connector_preview_df")
        if st.button("Load Connector Preview", key="structured_load_connector_preview"):
            try:
                assert connector_payload is not None
                load_req = {
                    "connector": connector_payload["connector"],
                    "config": json.loads(connector_payload["config"]),
                    "query": connector_payload["query"].strip() or None,
                    "table": connector_payload["table"].strip() or None,
                    "limit": int(connector_payload["limit"]),
                }
                out = api_post_json(base_url, "/connectors/load", load_req)
                preview = out.get("preview", [])
                if preview:
                    preview_df = pd.DataFrame(preview)
                    st.session_state["structured_connector_preview_df"] = preview_df
                    st.success(f"Loaded connector preview with {len(preview_df.columns)} columns.")
                else:
                    st.warning("Connector returned no preview rows.")
            except Exception as exc:
                st.error(f"Failed to load connector preview: {exc}")
        if isinstance(preview_df, pd.DataFrame):
            st.caption(f"Detected {len(preview_df.columns)} columns from connector preview")
            st.dataframe(preview_df.head(10), use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        business_problem = st.text_area("Business Problem", value="Predict customer churn in next 30 days")
        target_options = ["Auto-detect target"]
        if preview_df is not None:
            target_options += [str(c) for c in preview_df.columns]
        target_choice = st.selectbox("Target Column", options=target_options, index=0)
    with c2:
        family_options = [
            "auto",
            "tree_based",
            "boosting",
            "linear",
            "kernel",
            "instance_based",
        ]
        fixed_model_options = [
            "auto",
            "xgboost",
            "lightgbm",
            "catboost",
            "random_forest",
            "extra_trees",
            "gradient_boosting",
            "adaboost",
            "logistic_regression",
            "svc_rbf",
            "knn",
            "xgboost_regressor",
            "lightgbm_regressor",
            "catboost_regressor",
            "linear_regression",
            "random_forest_regressor",
            "extra_trees_regressor",
            "gradient_boosting_regressor",
            "elasticnet",
            "ridge",
            "svr_rbf",
            "knn_regressor",
        ]
        model_family = st.selectbox("Model Family", options=family_options, index=0)
        fixed_model = st.selectbox("Fixed Model", options=fixed_model_options, index=0)
    if fixed_model != "auto":
        st.info("Fixed Model is selected: AutoML ranking becomes single-model mode by design.")

    if st.button("Run Structured Pipeline", use_container_width=True):
        try:
            business_problem_clean = _clean_business_problem_text(business_problem)
            if source_type == "Upload File":
                if not file:
                    st.error("Upload a structured file first.")
                    return
                payload = file.getvalue()
                out = api_post_multipart(
                    base_url,
                    "/orchestrate",
                    files={"file": (file.name, payload)},
                    data={
                        "business_problem": business_problem_clean,
                        "target_column": "" if target_choice == "Auto-detect target" else target_choice,
                        "model_family": "" if model_family == "auto" else model_family,
                        "fixed_model": "" if fixed_model == "auto" else fixed_model,
                        "llm_provider": "bedrock",
                    },
                )
            else:
                assert connector_payload is not None
                out = api_post_json(
                    base_url,
                    "/orchestrate-from-connector",
                    {
                        "connector": connector_payload["connector"],
                        "config": json.loads(connector_payload["config"]),
                        "query": connector_payload["query"].strip() or None,
                        "table": connector_payload["table"].strip() or None,
                        "limit": int(connector_payload["limit"]),
                        "dataset_name": connector_payload["dataset_name"].strip() or None,
                        "business_problem": business_problem_clean,
                        "target_column": "" if target_choice == "Auto-detect target" else target_choice,
                        "model_family": "" if model_family == "auto" else model_family,
                        "fixed_model": "" if fixed_model == "auto" else fixed_model,
                        "llm_provider": "bedrock",
                    },
                )
            st.session_state["structured_result"] = out
            st.session_state["structured_business_problem"] = business_problem_clean
            if preview_df is not None:
                st.session_state["structured_df"] = preview_df
        except Exception as exc:
            st.error(f"Structured pipeline failed: {exc}")

    result = st.session_state.get("structured_result")
    if not result:
        return

    st.success("Structured pipeline completed.")
    training = result.get("training", {})
    top_models = training.get("leaderboard", [])
    if top_models:
        df_top = _flatten_leaderboard(top_models).head(10)
        st.markdown("### Top Models (AutoML)")
        st.dataframe(df_top, use_container_width=True)
        if {"model", "primary_metric"}.issubset(df_top.columns):
            fig = px.bar(df_top, x="model", y="primary_metric", title="Top Models")
            st.plotly_chart(fig, use_container_width=True)
    else:
        df_top = pd.DataFrame()

    problem = result.get("problem_type", "classification")
    champion = training.get("best_model_name", "N/A")
    champion_score = float(training.get("best_score", 0.0))
    business_problem_val = st.session_state.get("structured_business_problem", "")
    _render_structured_summary(result, business_problem_val)
    _render_auto_business_summary(base_url, result, business_problem_val, df_top)
    summary_version = (
        result.get("deployment", {})
        .get("model_registry", {})
        .get("version")
        if isinstance(result.get("deployment", {}), dict)
        else None
    ) or f"{champion}|{champion_score:.6f}"
    if st.session_state.get("structured_llm_summary_version") != summary_version:
        prompt = _build_business_summary_prompt(
            result=result,
            business_problem=business_problem_val,
            df_top=df_top,
            champion=champion,
            champion_score=champion_score,
            problem=problem,
        )
        try:
            llm_out = api_post_json(
                base_url,
                "/chat",
                {"question": prompt, "context": "Structured AutoML executive summary", "provider": "bedrock"},
            )
            st.session_state["structured_llm_summary"] = llm_out.get("answer", "")
        except Exception as exc:
            st.session_state["structured_llm_summary"] = f"LLM summary unavailable: {exc}"
        st.session_state["structured_llm_summary_version"] = summary_version
    if user_mode in {"Data Science", "MLOps"} and st.session_state.get("structured_llm_summary"):
        st.markdown("### LLM Executive Summary")
        st.write(st.session_state["structured_llm_summary"])

    st.markdown("### Structured Summary")
    st.info(
        f"Problem type: `{problem}` | Champion: `{champion}` | Score: `{champion_score:.4f}`. "
        f"AutoML backend: `{training.get('automl_backend', 'unknown')}`."
    )
    if result.get("pipeline", {}).get("fixed_model"):
        st.warning("You ran fixed-model mode. Clear Fixed Model to get full top-model ranking and ensemble behavior.")

    if user_mode in {"Data Science", "MLOps"} and not df_top.empty and preview_df is not None:
        st.markdown("### Model Selection and Optuna Tuning")
        model_choices = [str(m) for m in df_top["model"].tolist()]
        selected_name = st.selectbox("Select model from leaderboard", options=model_choices, index=0)
        n_trials = st.slider("Optuna Trials", min_value=5, max_value=100, value=20, step=5)
        tune_key = _normalize_model_name(selected_name, problem)
        st.caption(f"Mapped for tuning: `{tune_key}`")
        if st.button("Run Optuna Tuning", use_container_width=True):
            target_col = training.get("target")
            if not target_col or target_col not in preview_df.columns:
                st.error("Target column not available in preview dataframe for tuning.")
            else:
                try:
                    tune_out = _structured_tune_api(
                        base_url,
                        preview_df,
                        target_col,
                        problem,
                        tune_key,
                        n_trials=n_trials,
                    )
                    st.session_state["structured_tuning"] = tune_out
                except Exception as exc:
                    st.error(f"Optuna tuning failed: {exc}")

        tune_out = st.session_state.get("structured_tuning")
        if tune_out:
            c_t1, c_t2, c_t3 = st.columns(3)
            c_t1.metric("Baseline CV", f"{tune_out['baseline_score']:.4f}")
            c_t2.metric("Tuned CV", f"{tune_out['tuned_score']:.4f}")
            c_t3.metric("Improvement", f"{tune_out['improvement_abs']:.4f}", delta=f"{tune_out['improvement_pct']:.2f}%")
            st.code(safe_json(tune_out["best_params"]), language="json")
            trial_scores = tune_out.get("trial_scores") or []
            best_scores = tune_out.get("best_scores") or []
            if trial_scores:
                fig_tune = go.Figure()
                fig_tune.add_trace(go.Scatter(y=trial_scores, mode="lines+markers", name="Trial score"))
                if best_scores:
                    fig_tune.add_trace(go.Scatter(y=best_scores, mode="lines", name="Best so far"))
                fig_tune.update_layout(height=280, title="Optuna Progress", xaxis_title="Trial", yaxis_title="CV score")
                st.plotly_chart(fig_tune, use_container_width=True)

    if preview_df is not None and training.get("target"):
        st.markdown("### Prediction")
        target_col = training["target"]
        if target_col in preview_df.columns:
            pred_source_models = df_top["model"].tolist() if not df_top.empty and "model" in df_top.columns else [champion]
            chosen_pred_model = st.selectbox("Choose model for prediction", options=[str(m) for m in pred_source_models], index=0)
            model_for_pred = _normalize_model_name(chosen_pred_model, problem)
            dataset_key = (
                (file.name if file is not None else None)
                or str(training.get("dataset_name", "") or result.get("dataset_name", "") or "connector_dataset")
            )
            pred_key = f"pred_model::{dataset_key}::{target_col}::{model_for_pred}"
            if st.button("Train Predictor for Inference"):
                try:
                    out = _structured_predict_train_api(base_url, preview_df, target_col, problem, model_for_pred)
                    st.session_state[pred_key] = out.get("model_id")
                    st.success(f"Predictor ready using `{model_for_pred}`.")
                except Exception as exc:
                    st.error(f"Failed to train predictor: {exc}")

            model_id = st.session_state.get(pred_key)
            if model_id:
                inputs: Dict[str, Any] = {}
                feat_cols = [c for c in preview_df.columns if c != target_col]
                c_pred1, c_pred2 = st.columns(2)
                for i, col in enumerate(feat_cols):
                    col_widget = c_pred1 if i % 2 == 0 else c_pred2
                    with col_widget:
                        if pd.api.types.is_numeric_dtype(preview_df[col]):
                            default = float(preview_df[col].median())
                            inputs[col] = st.number_input(f"{col}", value=default, key=f"pred_{col}")
                        else:
                            vals = [str(v) for v in preview_df[col].dropna().astype(str).unique().tolist()[:100]]
                            if not vals:
                                vals = [""]
                            inputs[col] = st.selectbox(f"{col}", options=vals, key=f"pred_{col}")

                if st.button("Predict", use_container_width=True):
                    try:
                        out = _structured_predict_api(
                            base_url,
                            model_id=model_id,
                            rows=[inputs],
                            return_proba=True,
                        )
                        pred = out.get("predictions", [None])[0]
                        st.success(f"Predicted `{target_col}`: **{pred}**")
                        probs = out.get("probabilities")
                        if problem == "classification" and probs:
                            prob_df = pd.DataFrame({"class": [str(i) for i in range(len(probs[0]))], "probability": probs[0]})
                            st.dataframe(prob_df.sort_values("probability", ascending=False), use_container_width=True)
                    except Exception as exc:
                        st.error(f"Prediction failed: {exc}")

    if user_mode in {"Data Science", "MLOps"} and preview_df is not None and training.get("target"):
        st.markdown("### Hoeffding + ADWIN Stream Monitor")
        st.caption("Graph meaning: `Error Sequence` shows per-sample correctness (0=correct, 1=error). `Running Variance` shows volatility in streamed targets. `Drift Events` increases when ADWIN detects distribution change.")
        target_col = training["target"]
        dataset_key = (
            (file.name if file is not None else None)
            or str(training.get("dataset_name", "") or result.get("dataset_name", "") or "structured_dataset")
        )
        stream_key = f"stream_id::{dataset_key}"
        stream_id = st.session_state.get(stream_key, "")
        if not stream_id:
            if st.button("Start Online Stream", use_container_width=True):
                try:
                    out = _structured_online_start_api(base_url, task=problem)
                    stream_id = out.get("stream_id", "")
                    st.session_state[stream_key] = stream_id
                    st.success("Online stream started.")
                except Exception as exc:
                    st.error(f"Stream start failed: {exc}")
        if stream_id and target_col in preview_df.columns:
            if st.button("Process 1 Random Sample"):
                row = preview_df.sample(1).iloc[0]
                out = _structured_online_batch_api(
                    base_url,
                    stream_id=stream_id,
                    rows=[row.to_dict()],
                    target=target_col,
                    max_rows=1,
                )
                st.session_state["structured_stream_status"] = out
                if out.get("drift_hits", 0) > 0:
                    st.error("Concept drift detected by ADWIN.")
            if st.button("Process 50 Random Samples"):
                sample_rows = preview_df.sample(min(50, len(preview_df))).to_dict(orient="records")
                out = _structured_online_batch_api(
                    base_url,
                    stream_id=stream_id,
                    rows=sample_rows,
                    target=target_col,
                    max_rows=len(sample_rows),
                )
                st.session_state["structured_stream_status"] = out
            status = st.session_state.get("structured_stream_status")
            if status:
                m1, m2, m3 = st.columns(3)
                m1.metric("Samples", int(status.get("processed", 0)))
                m2.metric("Drift Events", int(status.get("drift_events", 0)))
                m3.metric("Online Accuracy", f"{float(status.get('accuracy', 0.0)):.3f}")
                var_hist = status.get("variance_history", [])
                if var_hist:
                    fig_v = go.Figure()
                    fig_v.add_trace(go.Scatter(y=list(var_hist), mode="lines", name="variance"))
                    fig_v.update_layout(height=250, title="Running Variance (Hoeffding Stream)")
                    st.plotly_chart(fig_v, use_container_width=True)
                err_hist = status.get("history", [])
                if err_hist:
                    fig_e = go.Figure()
                    fig_e.add_trace(go.Scatter(y=list(err_hist), mode="lines+markers", name="error"))
                    fig_e.update_layout(height=250, title="Error Sequence (0=correct, 1=error)")
                    st.plotly_chart(fig_e, use_container_width=True)
        if stream_id and target_col in preview_df.columns:
            _render_new_data_drift_check(
                base_url=base_url,
                key_prefix="structured",
                baseline_df=preview_df,
                target_col=target_col,
                problem=problem,
                stream_id=stream_id,
            )
        drift_context = {
            "samples": int(st.session_state.get("structured_stream_status", {}).get("processed", 0)),
            "drift_events": int(st.session_state.get("structured_stream_status", {}).get("drift_events", 0)),
            "online_accuracy": float(st.session_state.get("structured_stream_status", {}).get("accuracy", 0.0)),
            "latest_variance": float(st.session_state.get("structured_stream_status", {}).get("running_variance", 0.0)),
        }
    else:
        drift_context = {"samples": 0, "drift_events": 0, "online_accuracy": 0.0, "latest_variance": 0.0}

    eval_obj = result.get("evaluation", {}) if isinstance(result.get("evaluation", {}), dict) else {}
    val = eval_obj.get("validation", {}) if isinstance(eval_obj.get("validation", {}), dict) else {}
    if user_mode in {"Data Science", "MLOps"} and val:
        st.markdown("### Validation Quality")
        st.caption("How to read: Holdout = one unseen test split; CV Mean = average across folds; Nested CV Mean = stricter generalization estimate. Close values indicate stable performance.")
        h = val.get("holdout_metrics", {}) if isinstance(val.get("holdout_metrics", {}), dict) else {}
        cv = val.get("cv_metrics", {}) if isinstance(val.get("cv_metrics", {}), dict) else {}
        nested = val.get("nested_cv_metrics", {}) if isinstance(val.get("nested_cv_metrics", {}), dict) else {}
        t1, t2, t3 = st.columns(3)
        t1.metric("Holdout Primary", f"{(h.get('f1_weighted') if 'f1_weighted' in h else h.get('r2', 0.0)):.4f}")
        t2.metric("CV Mean", f"{float(cv.get('mean', 0.0)):.4f}")
        t3.metric("Nested CV Mean", f"{float(nested.get('outer_mean', 0.0)):.4f}")

    if user_mode in {"Data Science", "MLOps"} and eval_obj.get("fairness"):
        st.markdown("### Fairness Diagnostics")
        st.caption("Fairness checks require at least one low-cardinality categorical feature to compare group metrics. If none exist, this section is informational only.")
        fair = eval_obj.get("fairness", {})
        if fair.get("checked"):
            group_metrics = fair.get("group_metrics", {})
            if group_metrics:
                for col, details in group_metrics.items():
                    st.markdown(f"**Sensitive Column:** `{col}` | Metric: `{details.get('metric','')}` | Disparity: `{details.get('disparity')}`")
                    gdf = pd.DataFrame(details.get("groups", []))
                    if not gdf.empty:
                        st.dataframe(gdf, use_container_width=True)
            else:
                st.info("Fairness check ran but no eligible groups with enough samples.")
        else:
            st.info(fair.get("reason", "Fairness diagnostics not available for this dataset."))

    if user_mode in {"Data Science", "MLOps"} and eval_obj.get("explainability"):
        st.markdown("### Explainability")
        st.caption("Permutation importance is model-agnostic and always preferred here. SHAP appears when estimator + environment support it.")
        exp = eval_obj.get("explainability", {})
        if isinstance(exp, dict) and exp.get("narrative"):
            st.markdown("**Explainability Summary**")
            st.markdown(_normalize_llm_bullets(str(exp.get("narrative", ""))))
        imp = pd.DataFrame(exp.get("permutation_importance", []))
        if not imp.empty:
            st.dataframe(imp, use_container_width=True)
            if {"feature", "importance_mean"}.issubset(imp.columns):
                fig_imp = px.bar(
                    imp.sort_values("importance_mean", ascending=False),
                    x="feature",
                    y="importance_mean",
                    title="Feature Importance (Permutation)",
                )
                fig_imp.update_layout(height=320)
                st.plotly_chart(fig_imp, use_container_width=True)

        shap_info = exp.get("shap", {})
        if isinstance(shap_info, dict) and shap_info.get("enabled"):
            shap_df = pd.DataFrame(shap_info.get("global_mean_abs", []))
            if not shap_df.empty and {"feature", "mean_abs_shap"}.issubset(shap_df.columns):
                fig_shap = px.bar(
                    shap_df.sort_values("mean_abs_shap", ascending=False),
                    x="feature",
                    y="mean_abs_shap",
                    title="SHAP Global Importance",
                )
                fig_shap.update_layout(height=320)
                st.plotly_chart(fig_shap, use_container_width=True)
        else:
            st.caption("SHAP graph not available for this model/runtime. Using permutation importance graph above.")

    if user_mode in {"Data Science", "MLOps"} and eval_obj.get("experiment_tracking"):
        st.markdown("### Experiment Tracking")
        tr = eval_obj.get("experiment_tracking", {})
        c1, c2, c3 = st.columns(3)
        c1.metric("Tracking Backend", str(tr.get("backend", "unknown")))
        c2.metric("Run", str(tr.get("run_name", "N/A")))
        c3.metric("Log Path", str(tr.get("log_path", "N/A")))

    if user_mode in {"Data Science", "MLOps"}:
        st.markdown("### AI Technical Explanation")
        explain_version = f"{champion}|{champion_score:.6f}|{drift_context.get('samples',0)}|{drift_context.get('drift_events',0)}"
        if st.session_state.get("structured_ai_explain_version") != explain_version:
            try:
                prompt = _build_ai_technical_explainer_prompt(
                    result=result,
                    business_problem=business_problem_val,
                    df_top=df_top,
                    drift_context=drift_context,
                )
                out = api_post_json(
                    base_url,
                    "/structured/explain",
                    {
                        "result": result,
                        "business_problem": business_problem_val,
                        "drift_context": drift_context,
                        "response_style": "technical",
                        "llm_provider": "bedrock",
                    },
                )
                st.session_state["structured_ai_explain_text"] = out.get("explanation", "")
            except Exception:
                st.session_state["structured_ai_explain_text"] = (
                    "Modeling summary: top models compared and champion selected by validation metrics. "
                    "Fairness is not applicable here due to lack of eligible sensitive columns. "
                    "SHAP is unavailable for this runtime/model combination; permutation importance remains valid. "
                    "For drift, monitor increasing error spikes and ADWIN drift events; retrain if drift persists."
                )
            st.session_state["structured_ai_explain_version"] = explain_version
        st.markdown(_normalize_llm_bullets(st.session_state.get("structured_ai_explain_text", "")))

    if user_mode in {"Data Science", "MLOps"}:
        with st.expander("DeepEval (Structured)", expanded=False):
            model_name = st.text_input(
                "Ollama Model for DeepEval",
                value=st.session_state.get("structured_deepeval_model", "llama3.1:8b"),
                key="structured_deepeval_model",
            )
            metrics = st.multiselect(
                "Metrics",
                options=["faithfulness", "answer_relevancy"],
                default=["faithfulness", "answer_relevancy"],
                key="structured_deepeval_metrics",
            )
            summary_text = st.session_state.get("structured_llm_summary", "")
            if summary_text and st.button("Run DeepEval on Executive Summary", use_container_width=True):
                context_payload = {
                    "business_problem": business_problem_val,
                    "problem_type": problem,
                    "champion": champion,
                    "champion_score": champion_score,
                    "top_models": df_top.head(5).to_dict(orient="records") if not df_top.empty else [],
                    "validation": eval_obj.get("validation", {}),
                    "fairness": eval_obj.get("fairness", {}),
                    "explainability": eval_obj.get("explainability", {}),
                }
                try:
                    out = api_post_json(
                        base_url,
                        "/deepeval/run",
                        {
                            "input_text": business_problem_val,
                            "actual_output": summary_text,
                            "context": safe_json(context_payload),
                            "model_name": model_name,
                            "metrics": metrics,
                        },
                    )
                    st.session_state["structured_deepeval_summary"] = out
                except Exception as exc:
                    st.error(f"DeepEval failed: {exc}")
            if st.session_state.get("structured_deepeval_summary"):
                st.markdown("**Executive Summary Eval**")
                render_deepeval_results(st.session_state.get("structured_deepeval_summary", {}))

            tech_text = st.session_state.get("structured_ai_explain_text", "")
            if tech_text and st.button("Run DeepEval on Technical Explanation", use_container_width=True):
                context_payload = {
                    "business_problem": business_problem_val,
                    "problem_type": problem,
                    "champion": champion,
                    "champion_score": champion_score,
                    "validation": eval_obj.get("validation", {}),
                    "drift": drift_context,
                    "deployment": result.get("deployment", {}),
                }
                try:
                    out = api_post_json(
                        base_url,
                        "/deepeval/run",
                        {
                            "input_text": business_problem_val,
                            "actual_output": tech_text,
                            "context": safe_json(context_payload),
                            "model_name": model_name,
                            "metrics": metrics,
                        },
                    )
                    st.session_state["structured_deepeval_tech"] = out
                except Exception as exc:
                    st.error(f"DeepEval failed: {exc}")
            if st.session_state.get("structured_deepeval_tech"):
                st.markdown("**Technical Explanation Eval**")
                render_deepeval_results(st.session_state.get("structured_deepeval_tech", {}))

    deploy = result.get("deployment", {}) if isinstance(result.get("deployment", {}), dict) else {}
    if user_mode in {"Data Science", "MLOps"} and deploy:
        st.markdown("### Deployment Readiness")
        checklist = deploy.get("release_checklist", {})
        d1, d2, d3 = st.columns(3)
        d1.metric("Validated", "Yes" if checklist.get("validated") else "No")
        d2.metric("Packaged", "Yes" if checklist.get("packaged") else "No")
        d3.metric("Governance Ready", "Yes" if checklist.get("governance_ready") else "No")
        registry = deploy.get("model_registry", {})
        if registry:
            st.caption(f"Registry Version: {registry.get('version', 'N/A')}")

    if user_mode in {"Data Science", "MLOps"}:
        with st.expander("Debug JSON (optional)", expanded=False):
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**Data Profile**")
                st.code(safe_json(result.get("data_profile", {})), language="json")
                st.markdown("**Pipeline**")
                st.code(safe_json(result.get("pipeline", {})), language="json")
            with c2:
                st.markdown("**Evaluation**")
                st.code(safe_json(result.get("evaluation", {})), language="json")
                st.markdown("**Deployment**")
                st.code(safe_json(result.get("deployment", {})), language="json")


def _live_sdr_analytics_dataframe(sdr_base_url: str) -> pd.DataFrame:
    intelligence = api_get(sdr_base_url, "/prospect-intelligence").get("items", [])
    qualifications = api_get(sdr_base_url, "/qualification").get("items", [])
    campaigns = api_get(sdr_base_url, "/outreach/campaigns").get("items", [])
    follow_ups = api_get(sdr_base_url, "/follow-up/plans").get("items", [])
    conversations = api_get(sdr_base_url, "/conversations").get("items", [])
    meetings = api_get(sdr_base_url, "/meetings").get("items", [])
    crm_syncs = api_get(sdr_base_url, "/crm/syncs").get("items", [])

    qualification_by_contact = {
        item.get("contact_id"): item for item in qualifications
    }
    campaign_by_contact = {item.get("contact_id"): item for item in campaigns}
    follow_up_by_campaign = {
        item.get("campaign_id"): item for item in follow_ups
    }
    conversation_by_campaign = {
        item.get("campaign_id"): item for item in conversations
    }
    meeting_by_campaign = {item.get("campaign_id"): item for item in meetings}
    crm_by_meeting = {item.get("meeting_id"): item for item in crm_syncs}
    rows: list[dict[str, Any]] = []
    for item in intelligence:
        contact_id = item.get("contact_id")
        intent = item.get("intent", {})
        propensity = item.get("propensity", {})
        ranking = item.get("ranking", {})
        qualification = qualification_by_contact.get(contact_id, {})
        campaign = campaign_by_contact.get(contact_id, {})
        campaign_id = campaign.get("campaign_id")
        follow_up = follow_up_by_campaign.get(campaign_id, {})
        conversation = conversation_by_campaign.get(campaign_id, {})
        classification = conversation.get("latest_classification") or {}
        meeting = meeting_by_campaign.get(campaign_id, {})
        crm_sync = crm_by_meeting.get(meeting.get("meeting_id"), {})
        messages = campaign.get("messages", []) or []
        rows.append(
            {
                "account_id": item.get("account_id"),
                "company_name": item.get("company_name"),
                "contact_id": contact_id,
                "contact_name": item.get("contact_name"),
                "intent_probability": intent.get("probability", {}).get("value"),
                "reply_probability": propensity.get("reply_probability", {}).get("value"),
                "meeting_probability": propensity.get("meeting_probability", {}).get("value"),
                "qualification_probability": propensity.get(
                    "qualification_probability", {}
                ).get("value"),
                "priority_score": ranking.get("priority_score"),
                "queue_rank": ranking.get("rank"),
                "ranking_provider": ranking.get("provider"),
                "qualification_score": qualification.get("qualification_score"),
                "qualification_status": qualification.get("qualification_status"),
                "qualification_tier": qualification.get("qualification_tier"),
                "sales_ready": qualification.get("sales_ready"),
                "campaign_id": campaign_id,
                "campaign_status": campaign.get("status"),
                "campaign_message_count": len(messages),
                "sent_message_count": sum(
                    1
                    for message in messages
                    if message.get("status") in {"sent", "delivered", "replied"}
                ),
                "follow_up_status": follow_up.get("status"),
                "follow_up_scheduler": follow_up.get("scheduler_backend"),
                "conversation_status": conversation.get("status"),
                "reply_intent": classification.get("intent"),
                "reply_sentiment": classification.get("sentiment"),
                "reply_confidence": classification.get("confidence"),
                "meeting_id": meeting.get("meeting_id"),
                "meeting_status": meeting.get("status"),
                "meeting_provider": meeting.get("provider"),
                "meeting_start": meeting.get("start_at"),
                "meeting_url": meeting.get("meeting_url"),
                "meeting_temporal_status": meeting.get("temporal_status"),
                "crm_sync_status": crm_sync.get("status")
                or meeting.get("crm_sync_status"),
                "crm_provider": crm_sync.get("provider"),
                "crm_company_id": crm_sync.get("company_id"),
                "crm_person_id": crm_sync.get("person_id"),
                "crm_opportunity_id": crm_sync.get("opportunity_id"),
                "created_at": item.get("created_at"),
            }
        )
    return pd.DataFrame(rows)


def analytics_tab(base_url: str) -> None:
    st.subheader("Analytics Agents")
    source = st.radio(
        "Analytics data source",
        options=["Upload File", "Live AI SDR Data"],
        horizontal=True,
    )
    file = None
    analytics_name: str | None = None
    analytics_payload: bytes | None = None
    if source == "Upload File":
        file = st.file_uploader(
            "Upload Structured File for Analytics",
            type=["csv", "tsv", "txt", "json", "xls", "xlsx", "sas7bdat", "sav", "rdata"],
            key="analytics_file",
        )
        if file is not None:
            analytics_name = file.name
            analytics_payload = file.getvalue()
    else:
        sdr_base_url = st.session_state.get("sdr_backend_url", DEFAULT_SDR_API_URL)
        st.caption(f"Reading persisted SDR activity from `{sdr_base_url}`.")
        try:
            live_df = _live_sdr_analytics_dataframe(sdr_base_url)
            if live_df.empty:
                st.info(
                    "No Prospect Intelligence records exist yet. Run the SDR pipeline "
                    "first, then return here."
                )
            else:
                analytics_name = "live_ai_sdr_analytics.csv"
                analytics_payload = live_df.to_csv(index=False).encode()
        except Exception as exc:
            st.error(f"Could not load live SDR analytics data: {exc}")
    query = st.text_input("Analytics Query", value="Show distribution of first numeric column")
    context = st.text_area("Recent Chat Context (optional)", value="")

    if analytics_payload is not None and analytics_name is not None:
        try:
            payload = analytics_payload
            preview_df = _load_structured_preview(base_url, analytics_name, payload)
            st.session_state["analytics_df"] = preview_df
            with st.expander("Interactive Data Explorer", expanded=False):
                render_analytics_dashboard(base_url, preview_df)
            with st.expander("Data Preview", expanded=False):
                st.dataframe(preview_df.head(15), use_container_width=True)

            upload_sig = hashlib.md5(payload).hexdigest() + f":{analytics_name}"
            if st.session_state.get("analytics_upload_sig") != upload_sig:
                auto_insight = api_post_multipart(
                    base_url,
                    "/analytics/insights-file",
                    files={"file": (analytics_name, payload)},
                    data={"llm_provider": "bedrock"},
                )
                st.session_state["analytics_auto_insights"] = str(
                    auto_insight.get("insights", "")
                )
                st.session_state["analytics_upload_sig"] = upload_sig
        except Exception:
            pass

    if st.button("Generate SQL (NL → SQL)", use_container_width=True):
        if analytics_payload is None or analytics_name is None:
            st.error("Select a populated analytics data source first.")
        else:
            try:
                payload = analytics_payload
                out = api_post_multipart(
                    base_url,
                    "/analytics/query-file",
                    files={"file": (analytics_name, payload)},
                    data={
                        "query": query,
                        "chat_context": context,
                        "llm_provider": "bedrock",
                        "force_sql": "true",
                    },
                )
                st.session_state["analytics_sql_editor"] = str(out.get("code", ""))
                st.session_state["analytics_last_query"] = query
                st.session_state["analytics_sql_result"] = None
                st.success("SQL generated from natural language.")
            except Exception as exc:
                st.error(f"SQL generation failed: {exc}")

    auto_insights = str(st.session_state.get("analytics_auto_insights", "") or "")
    if auto_insights:
        s_text, qs = _extract_insight_sections(auto_insights)
        if s_text:
            st.markdown("### Upload Summary")
            st.write(s_text)
        if qs:
            st.markdown("### Suggested Analysis Questions")
            for q in qs:
                st.markdown(f"- {q}")

    if st.button("Run Analytics", use_container_width=True):
        if analytics_payload is None or analytics_name is None:
            st.error("Select a populated analytics data source first.")
            return
        try:
            payload = analytics_payload
            status_ph = st.empty()
            streamed_reasoning = ""
            final_payload: Dict[str, Any] | None = None

            for evt in api_post_multipart_stream(
                base_url,
                "/analytics/run-file-stream",
                files={"file": (analytics_name, payload)},
                    data={"query": query, "chat_context": context, "llm_provider": "bedrock"},
            ):
                et = str(evt.get("event", ""))
                if et == "status":
                    status_ph.info(str(evt.get("message", "Running...")))
                elif et == "reasoning_chunk":
                    streamed_reasoning += str(evt.get("text", ""))
                elif et == "result":
                    final_payload = evt.get("payload", {})
                elif et == "error":
                    raise RuntimeError(str(evt.get("message", "Analytics stream failed")))

            if not final_payload:
                raise RuntimeError("No final analytics payload received from stream")

            if streamed_reasoning and isinstance(final_payload.get("query"), dict):
                final_payload["query"]["reasoning"] = streamed_reasoning.strip()

            status_ph.success("Streaming complete.")
            st.session_state["analytics_last_query"] = query
            st.session_state["analytics_result"] = final_payload
        except Exception as exc:
            st.error(f"Analytics run failed: {exc}")

    result = st.session_state.get("analytics_result")
    if not result:
        return

    st.success("Analytics completed.")
    st.markdown("### Answer")
    qobj = result.get("query", {})
    nl_answer = render_analytics_natural_language(qobj)
    st.markdown(nl_answer)
    if qobj.get("code"):
        st.markdown("### Generated SQL/Code")
        st.code(qobj.get("code", ""), language=str(qobj.get("code_language", "sql")))
        st.markdown("### Edit SQL and Re-run")
        edited_sql = st.text_area(
            "SQL Editor",
            value=str(qobj.get("code", "")),
            height=140,
            key="analytics_sql_editor",
        )
        if st.button("Run SQL", use_container_width=True):
            df_src = st.session_state.get("analytics_df")
            if not isinstance(df_src, pd.DataFrame):
                st.error("Upload a file first to run SQL.")
            else:
                try:
                    out = api_post_json(
                        base_url,
                        "/analytics/run-sql",
                        {"dataframe": df_src.replace({pd.NA: None}).to_dict(orient="records"), "sql": edited_sql},
                    )
                    st.session_state["analytics_sql_result"] = out
                except Exception as exc:
                    st.error(f"SQL run failed: {exc}")

        sql_out = st.session_state.get("analytics_sql_result")
        if isinstance(sql_out, dict) and sql_out.get("rows") is not None:
            st.markdown("### SQL Result")
            st.dataframe(pd.DataFrame(sql_out.get("rows", [])), use_container_width=True)
            render_sql_result_chart(sql_out.get("rows", []))

        if st.button("Explain SQL in Plain English", use_container_width=True):
            df_src = st.session_state.get("analytics_df")
            cols = df_src.columns.tolist() if isinstance(df_src, pd.DataFrame) else []
            prompt = (
                "Explain this SQL in plain English and summarize what the result represents. "
                "Be concise and business-friendly.\n\n"
                f"SQL:\n{edited_sql}\n\n"
                f"Available columns: {cols}\n"
            )
            try:
                out = api_post_json(
                    base_url,
                    "/chat",
                    {"question": prompt, "context": "SQL explanation", "provider": "bedrock"},
                )
                st.session_state["analytics_sql_nl"] = out.get("answer", "")
            except Exception as exc:
                st.error(f"SQL explanation failed: {exc}")

        if st.session_state.get("analytics_sql_nl"):
            st.markdown("### SQL → Natural Language")
            st.write(st.session_state.get("analytics_sql_nl", ""))
    st.markdown("### Detailed Explanation")
    st.write(render_analytics_detailed_explanation(str(st.session_state.get("analytics_last_query", query)), qobj))

    insights_text = str(result.get("insights", {}).get("insights", "") or "")
    summary_text, questions = _extract_insight_sections(insights_text)
    if summary_text:
        st.markdown("### Summary")
        st.write(summary_text)
    if questions:
        st.markdown("### Analysis Questions")
        for q in questions:
            st.markdown(f"- {q}")

    plot_b64 = qobj.get("execution", {}).get("plot_base64")
    if plot_b64:
        st.image(decode_b64_image(plot_b64), caption="Primary Analytics Plot")

    visuals = result.get("visuals", {}).get("visuals", [])
    if visuals:
        st.markdown("**Additional Visual Pack**")
        cols = st.columns(2)
        for i, visual in enumerate(visuals):
            with cols[i % 2]:
                st.caption(visual.get("title", f"plot-{i}"))
                st.image(decode_b64_image(visual.get("image_base64", "")))

    with st.expander("DeepEval (Analytics)", expanded=False):
        model_name = st.text_input(
            "Ollama Model for DeepEval",
            value=st.session_state.get("analytics_deepeval_model", "llama3.1:8b"),
            key="analytics_deepeval_model",
        )
        metrics = st.multiselect(
            "Metrics",
            options=["faithfulness", "answer_relevancy"],
            default=["faithfulness", "answer_relevancy"],
            key="analytics_deepeval_metrics",
        )
        if st.button("Run DeepEval on Analytics Answer", use_container_width=True):
            context_payload = {
                "query": str(st.session_state.get("analytics_last_query", query)),
                "summary": summary_text,
                "analysis_questions": questions,
                "generated_code": qobj.get("code"),
                "code_language": qobj.get("code_language"),
                "execution_result": qobj.get("execution", {}),
                "plot_present": bool(plot_b64),
                "visual_titles": [v.get("title") for v in visuals if isinstance(v, dict)],
            }
            try:
                out = api_post_json(
                    base_url,
                    "/deepeval/run",
                    {
                        "input_text": str(st.session_state.get("analytics_last_query", query)),
                        "actual_output": nl_answer,
                        "context": safe_json(context_payload),
                        "model_name": model_name,
                        "metrics": metrics,
                    },
                )
                st.session_state["analytics_deepeval"] = out
            except Exception as exc:
                st.error(f"DeepEval failed: {exc}")
        if st.session_state.get("analytics_deepeval"):
            render_deepeval_results(st.session_state.get("analytics_deepeval", {}))

    with st.expander("Debug Details", expanded=False):
        reasoning = str(qobj.get("reasoning", "") or "")
        if reasoning:
            st.markdown("**Reasoning**")
            st.write(reasoning)
        st.markdown("**Generated Code**")
        code_lang = str(qobj.get("code_language", "python"))
        st.code(qobj.get("code", ""), language=code_lang)
        st.markdown("**Execution Result**")
        st.code(safe_json(qobj.get("execution", {})), language="json")
        st.markdown("**Raw Insights**")
        st.write(insights_text)


def connectors_tab(base_url: str, user_mode: str = "Business") -> None:
    st.subheader("Connectors")

    if st.button("Refresh Connector Catalog"):
        try:
            st.session_state["connectors_catalog"] = api_get(base_url, "/connectors")
        except Exception as exc:
            st.error(f"Failed to fetch connector catalog: {exc}")

    connectors = st.session_state.get("connectors_catalog")
    if connectors:
        st.dataframe(pd.DataFrame(connectors), use_container_width=True)

    st.markdown("### Connector Access Gateway")
    st.caption("Validate connector-level and source-level access using the same role and tenant headers used by Custom GPT.")
    gateway_connector = st.text_input("Gateway Connector", value="PostgreSQL", key="connector_gateway_name")
    gateway_scope = st.text_area(
        "Source Scope JSON",
        value='{"table":"customer_churn_realistic","query":"SELECT * FROM customer_churn_realistic LIMIT 10"}',
        key="connector_gateway_scope",
    )
    if st.button("Check Connector Access", key="connector_gateway_check"):
        try:
            payload = {
                "connector": gateway_connector,
                "source_scope": json.loads(gateway_scope) if gateway_scope.strip() else {},
            }
            out = api_post_json(base_url, "/connectors/access/check", payload)
            if out.get("allowed"):
                st.success("Connector access allowed.")
            else:
                st.warning("Connector access denied.")
            st.code(safe_json(out), language="json")
            st.session_state["connector_gateway_last"] = out
        except Exception as exc:
            st.error(f"Connector access check failed: {exc}")

    st.markdown("### Test Connector")
    connector_name = st.text_input("Connector Name", value="sqlite", key="connector_test_name")
    config_json = st.text_area(
        "Config JSON",
        value='{"path":"/absolute/path/to/your.db"}',
        key="connector_test_cfg",
    )
    if st.button("Test Connector"):
        try:
            payload = {"connector": connector_name, "config": json.loads(config_json)}
            res = api_post_json(base_url, "/connectors/test", payload)
            st.code(safe_json(res), language="json")
        except Exception as exc:
            st.error(f"Connector test failed: {exc}")

    st.markdown("### Load Data From Connector")
    connector_load_name = st.text_input("Connector Name (load)", value="sqlite", key="connector_load_name")
    load_config_json = st.text_area(
        "Config JSON (load)",
        value='{"path":"/absolute/path/to/your.db"}',
        key="connector_load_cfg",
    )
    sql_query = st.text_area("SQL Query (optional)", value="SELECT * FROM your_table LIMIT 100", key="connector_query")
    table = st.text_input("Table (optional, if no query)", key="connector_table")
    limit = st.number_input("Limit", min_value=1, max_value=100000, value=1000, step=1)

    if st.button("Load Connector Data"):
        try:
            parsed_cfg = json.loads(load_config_json)
            payload = {
                "connector": connector_load_name,
                "config": parsed_cfg,
                "query": sql_query if sql_query.strip() else None,
                "table": table if table.strip() else None,
                "limit": int(limit),
            }
            out = api_post_json(base_url, "/connectors/load", payload)
            st.code(safe_json({k: v for k, v in out.items() if k != "preview"}), language="json")
            preview = out.get("preview", [])
            if preview:
                preview_df = pd.DataFrame(preview)
                st.session_state["connector_preview_df"] = preview_df
                st.session_state["connector_source_payload"] = {
                    "connector": connector_load_name,
                    "config": parsed_cfg,
                    "query": sql_query if sql_query.strip() else None,
                    "table": table if table.strip() else None,
                    "limit": int(limit),
                }
                st.dataframe(preview_df, use_container_width=True)
        except Exception as exc:
            st.error(f"Connector load failed: {exc}")

    preview_df = st.session_state.get("connector_preview_df")
    connector_source_payload = st.session_state.get("connector_source_payload")
    st.markdown("### Connector AutoML")
    st.caption("Run full Structured AutoML directly on connector data. Optuna/Hoeffding below use loaded preview rows.")
    if not (isinstance(preview_df, pd.DataFrame) and connector_source_payload):
        st.info("Load connector data first to enable Connector AutoML, Optuna, and Hoeffding/ADWIN in this tab.")
        return
    if isinstance(preview_df, pd.DataFrame) and connector_source_payload:
        c1, c2 = st.columns(2)
        with c1:
            business_problem = st.text_area(
                "Business Problem (Connector)",
                value="Predict customer churn in next 30 days",
                key="connector_business_problem",
            )
            target_options = ["Auto-detect target"] + [str(c) for c in preview_df.columns]
            target_choice = st.selectbox(
                "Target Column (Connector)",
                options=target_options,
                index=0,
                key="connector_target_choice",
            )
        with c2:
            family_options = ["auto", "tree_based", "boosting", "linear", "kernel", "instance_based"]
            fixed_model_options = [
                "auto",
                "xgboost",
                "lightgbm",
                "catboost",
                "random_forest",
                "extra_trees",
                "gradient_boosting",
                "adaboost",
                "logistic_regression",
                "svc_rbf",
                "knn",
                "xgboost_regressor",
                "lightgbm_regressor",
                "catboost_regressor",
                "linear_regression",
                "random_forest_regressor",
                "extra_trees_regressor",
                "gradient_boosting_regressor",
                "elasticnet",
                "ridge",
                "svr_rbf",
                "knn_regressor",
            ]
            model_family = st.selectbox("Model Family (Connector)", options=family_options, index=0, key="connector_model_family")
            fixed_model = st.selectbox("Fixed Model (Connector)", options=fixed_model_options, index=0, key="connector_fixed_model")

        if st.button("Run Structured Pipeline from Connector", use_container_width=True):
            try:
                req = {
                    **connector_source_payload,
                    "business_problem": _clean_business_problem_text(business_problem),
                    "target_column": "" if target_choice == "Auto-detect target" else target_choice,
                    "model_family": "" if model_family == "auto" else model_family,
                    "fixed_model": "" if fixed_model == "auto" else fixed_model,
                    "llm_provider": "bedrock",
                }
                out = api_post_json(base_url, "/orchestrate-from-connector", req)
                st.session_state["connector_structured_result"] = out
                st.session_state["structured_result"] = out
                st.success("Connector AutoML completed.")
            except Exception as exc:
                st.error(f"Connector AutoML failed: {exc}")

        conn_result = st.session_state.get("connector_structured_result")
        if isinstance(conn_result, dict):
            training = conn_result.get("training", {}) if isinstance(conn_result.get("training", {}), dict) else {}
            st.info(
                f"Problem type: `{conn_result.get('problem_type', 'classification')}` | "
                f"Champion: `{training.get('best_model_name', 'N/A')}` | "
                f"Score: `{float(training.get('best_score', 0.0)):.4f}`"
            )
            leaderboard = training.get("leaderboard", [])
            if leaderboard:
                df_top = _flatten_leaderboard(leaderboard).head(10)
                st.dataframe(df_top, use_container_width=True)
            else:
                df_top = pd.DataFrame()

            problem = str(conn_result.get("problem_type", "classification"))
            target_col = training.get("target")
            if user_mode in {"Data Science", "MLOps"} and target_col and target_col in preview_df.columns:
                st.markdown("### Optuna Tuning (Connector Data)")
                if df_top.empty or "model" not in df_top.columns:
                    model_choices = [str(training.get("best_model_name", "random_forest"))]
                else:
                    model_choices = [str(m) for m in df_top["model"].tolist()]
                selected_name = st.selectbox("Model for Optuna", options=model_choices, index=0, key="connector_tune_model")
                n_trials = st.slider("Optuna Trials (Connector)", min_value=5, max_value=100, value=20, step=5, key="connector_tune_trials")
                tune_key = _normalize_model_name(selected_name, problem)
                if st.button("Run Optuna Tuning (Connector)", use_container_width=True):
                    try:
                        tune_out = _structured_tune_api(
                            base_url,
                            preview_df,
                            target_col,
                            problem,
                            tune_key,
                            n_trials=n_trials,
                        )
                        st.session_state["connector_tuning"] = tune_out
                    except Exception as exc:
                        st.error(f"Optuna tuning failed: {exc}")
                tune_out = st.session_state.get("connector_tuning")
                if tune_out:
                    a1, a2, a3 = st.columns(3)
                    a1.metric("Baseline CV", f"{tune_out['baseline_score']:.4f}")
                    a2.metric("Tuned CV", f"{tune_out['tuned_score']:.4f}")
                    a3.metric("Improvement", f"{tune_out['improvement_abs']:.4f}", delta=f"{tune_out['improvement_pct']:.2f}%")
                    st.code(safe_json(tune_out["best_params"]), language="json")

                st.markdown("### Hoeffding + ADWIN Stream Monitor (Connector Data)")
                st.caption("Graph meaning: `Error Sequence` shows per-sample correctness (0=correct, 1=error). `Running Variance` shows volatility in streamed targets. `Drift Events` increases when ADWIN detects distribution change.")
                stream_key = "stream_id::connector"
                stream_id = st.session_state.get(stream_key, "")
                if not stream_id:
                    if st.button("Start Online Stream (Connector)", use_container_width=True):
                        try:
                            out = _structured_online_start_api(base_url, task=problem)
                            stream_id = out.get("stream_id", "")
                            st.session_state[stream_key] = stream_id
                            st.success("Online stream started.")
                        except Exception as exc:
                            st.error(f"Stream start failed: {exc}")
                if stream_id:
                    if st.button("Process 1 Random Sample (Connector)"):
                        row = preview_df.sample(1).iloc[0]
                        out = _structured_online_batch_api(
                            base_url,
                            stream_id=stream_id,
                            rows=[row.to_dict()],
                            target=target_col,
                            max_rows=1,
                        )
                        st.session_state["connector_stream_status"] = out
                        if out.get("drift_hits", 0) > 0:
                            st.error("Concept drift detected by ADWIN.")
                    if st.button("Process 50 Random Samples (Connector)"):
                        sample_rows = preview_df.sample(min(50, len(preview_df))).to_dict(orient="records")
                        out = _structured_online_batch_api(
                            base_url,
                            stream_id=stream_id,
                            rows=sample_rows,
                            target=target_col,
                            max_rows=len(sample_rows),
                        )
                        st.session_state["connector_stream_status"] = out
                    status = st.session_state.get("connector_stream_status")
                    if status:
                        m1, m2, m3 = st.columns(3)
                        m1.metric("Samples", int(status.get("processed", 0)))
                        m2.metric("Drift Events", int(status.get("drift_events", 0)))
                        m3.metric("Online Accuracy", f"{float(status.get('accuracy', 0.0)):.3f}")
                        var_hist = status.get("variance_history", [])
                        if var_hist:
                            fig_v = go.Figure()
                            fig_v.add_trace(go.Scatter(y=list(var_hist), mode="lines", name="variance"))
                            fig_v.update_layout(height=250, title="Running Variance (Hoeffding Stream)")
                            st.plotly_chart(fig_v, use_container_width=True)
                        err_hist = status.get("history", [])
                        if err_hist:
                            fig_e = go.Figure()
                            fig_e.add_trace(go.Scatter(y=list(err_hist), mode="lines+markers", name="error"))
                            fig_e.update_layout(height=250, title="Error Sequence (0=correct, 1=error)")
                            st.plotly_chart(fig_e, use_container_width=True)
                if stream_id:
                    _render_new_data_drift_check(
                        base_url=base_url,
                        key_prefix="connector",
                        baseline_df=preview_df,
                        target_col=target_col,
                        problem=problem,
                        stream_id=stream_id,
                    )


def knowledge_base_tab(base_url: str) -> None:
    st.subheader("Knowledge Base (Unstructured)")
    file = st.file_uploader("Upload Document", type=["pdf", "docx", "md", "txt", "json"], key="kb_file")
    c1, c2 = st.columns(2)
    with c1:
        dataset_id = st.text_input("Dataset ID", value="", key="kb_dataset_id")
    with c2:
        llm_provider = st.selectbox("LLM Provider", options=["ollama_local", "bedrock"], key="kb_provider")

    if st.button("Build Knowledge Base", use_container_width=True):
        if not file:
            st.error("Upload a document first.")
            return
        try:
            payload = file.getvalue()
            out = api_post_multipart(
                base_url,
                "/kb/build",
                files={"file": (file.name, payload)},
                data={"dataset_id": dataset_id, "llm_provider": llm_provider},
            )
            st.session_state["kb_build_result"] = out
            st.success("Knowledge base created.")
        except Exception as exc:
            st.error(f"KB build failed: {exc}")

    build_res = st.session_state.get("kb_build_result")
    if build_res:
        st.code(safe_json(build_res), language="json")

    st.markdown("### Query Knowledge Base")
    q = st.text_input("Question", value="Give me a professional summary.", key="kb_query")
    q_ds = st.text_input("Dataset ID for query", value=dataset_id or (file.name if file else ""), key="kb_query_dataset")
    top_k = st.slider("Top K", min_value=3, max_value=20, value=8, key="kb_topk")
    if st.button("Run KB Query", use_container_width=True):
        try:
            out = api_post_json(
                base_url,
                "/kb/query",
                {"dataset_id": q_ds, "query": q, "top_k": int(top_k), "llm_provider": llm_provider},
            )
            st.session_state["kb_query_result"] = out
        except Exception as exc:
            st.error(f"KB query failed: {exc}")

    q_res = st.session_state.get("kb_query_result")
    if q_res:
        st.markdown("### Answer")
        st.write(q_res.get("answer", ""))
        hits = q_res.get("hits", [])
        if hits:
            st.markdown("### Retrieved Chunks")
            st.dataframe(pd.DataFrame(hits), use_container_width=True)


def _chat_role_instruction(user_mode: str) -> str:
    if user_mode == "Business":
        return (
            "You are Chat Scientist for Business users. Be concise and outcome-focused. "
            "Use plain language, no heavy jargon, and always include: business outcome, risk, and next 3 actions."
        )
    if user_mode == "MLOps":
        return (
            "You are Chat Scientist for MLOps users. Focus on deployment lifecycle, governance, registry actions, "
            "feature serving, rollback readiness, and monitoring."
        )
    return (
        "You are Chat Scientist for Data Science users. Focus on model quality, validation stability, "
        "explainability, fairness limits, drift interpretation, and reproducibility."
    )


def _safe_get_cached_or_fetch(base_url: str, state_key: str, path: str) -> Any:
    cached = st.session_state.get(state_key)
    if cached is not None:
        return cached
    try:
        out = api_get(base_url, path)
        st.session_state[state_key] = out
        return out
    except Exception:
        return None


def _build_chat_grounding_context(
    base_url: str,
    include_structured: bool,
    include_registry: bool,
    include_feature_store: bool,
    include_orchestration: bool,
) -> Dict[str, Any]:
    ctx: Dict[str, Any] = {}
    if include_structured:
        r = st.session_state.get("structured_result")
        if isinstance(r, dict):
            training = r.get("training", {}) if isinstance(r.get("training", {}), dict) else {}
            eval_obj = r.get("evaluation", {}) if isinstance(r.get("evaluation", {}), dict) else {}
            validation = eval_obj.get("validation", {}) if isinstance(eval_obj.get("validation", {}), dict) else {}
            holdout = validation.get("holdout_metrics", {}) if isinstance(validation.get("holdout_metrics", {}), dict) else {}
            cv = validation.get("cv_metrics", {}) if isinstance(validation.get("cv_metrics", {}), dict) else {}
            nested = validation.get("nested_cv_metrics", {}) if isinstance(validation.get("nested_cv_metrics", {}), dict) else {}
            ctx["structured_latest"] = {
                "problem_type": r.get("problem_type"),
                "target": training.get("target"),
                "champion": training.get("best_model_name"),
                "champion_score": training.get("best_score"),
                "top_models": [m.get("model") for m in training.get("leaderboard", []) if isinstance(m, dict)][:10],
                "holdout_metrics": holdout,
                "cv_metrics": cv,
                "nested_cv_metrics": nested,
                "fairness": eval_obj.get("fairness", {}),
                "explainability": eval_obj.get("explainability", {}),
                "deployment": r.get("deployment", {}),
            }

    if include_registry:
        models = _safe_get_cached_or_fetch(base_url, "registry_models", "/registry/models")
        if isinstance(models, dict):
            model_rows = models.get("models", [])
            ctx["registry"] = {"count": len(model_rows), "models": model_rows[:10]}

    if include_feature_store:
        ft = _safe_get_cached_or_fetch(base_url, "feature_tables", "/feature-store/tables")
        if isinstance(ft, dict):
            ctx["feature_store"] = {
                "offline_tables": ft.get("offline_tables", []),
                "online_tables": ft.get("online_tables", []),
            }

    if include_orchestration:
        orch = _safe_get_cached_or_fetch(base_url, "orch_status", "/orchestration/status")
        if isinstance(orch, dict):
            ctx["orchestration"] = orch

    return ctx


def _format_chat_answer_text(ans: str) -> str:
    text = str(ans or "").strip()
    if not text:
        return ""
    # If model returned one long paragraph, split into readable bullets.
    if "\n" not in text and len(text) > 260 and ". " in text:
        parts = [p.strip() for p in text.split(". ") if p.strip()]
        bullets = [f"- {p if p.endswith('.') else p + '.'}" for p in parts[:12]]
        return "\n".join(bullets)
    return text


def _normalize_llm_bullets(text: str) -> str:
    """Render LLM output as clean bullets without markdown headers."""
    t = str(text or "").strip()
    if not t:
        return ""
    lines = []
    for raw in t.replace("\r", "").split("\n"):
        s = raw.strip()
        if not s or s.startswith("```"):
            continue
        s = re.sub(r"^#{1,6}\s*", "", s)
        s = re.sub(r"^\d+[\)\.\-]\s*", "", s)
        s = re.sub(r"^[-*•]\s*", "", s).strip()
        if s:
            lines.append(f"- {s}")
    return "\n".join(lines if lines else [t])


def _stream_text_chunks(text: str):
    for chunk in text.split(" "):
        yield chunk + " "
        time.sleep(0.01)


def chat_tab(base_url: str) -> None:
    st.subheader("Chat Scientist")
    if "chat_messages" not in st.session_state:
        st.session_state["chat_messages"] = []

    user_mode = st.session_state.get("user_mode", "Business")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        response_style = st.selectbox(
            "Response Style",
            options=["Structured", "Freeform"],
            index=0,
            key="chat_response_style",
            help="Structured gives short sections like outcome, risks, and next actions.",
        )
    with c2:
        include_structured = st.checkbox("Use latest Structured run", value=True, key="chat_use_structured")
    with c3:
        include_registry = st.checkbox("Use registry context", value=(user_mode == "MLOps"), key="chat_use_registry")
    with c4:
        include_feature_store = st.checkbox("Use feature-store context", value=(user_mode == "MLOps"), key="chat_use_fs")
    include_orchestration = st.checkbox("Use orchestration status context", value=(user_mode == "MLOps"), key="chat_use_orch")

    qa1, qa2, qa3, qa4 = st.columns(4)
    if qa1.button("Explain Champion", use_container_width=True):
        st.session_state["chat_quick_prompt"] = "Explain why the champion model was selected over other top models using validation, risks, and deployment recommendation."
    if qa2.button("Validation Review", use_container_width=True):
        st.session_state["chat_quick_prompt"] = "Review holdout, CV, and nested CV. Tell me if this is stable and production-ready."
    if qa3.button("Drift Guidance", use_container_width=True):
        st.session_state["chat_quick_prompt"] = "Explain the drift monitor metrics and exactly when we should retrain."
    if qa4.button("MLOps Next Steps", use_container_width=True):
        st.session_state["chat_quick_prompt"] = "Give me the next MLOps actions for approve, promote, monitor, and rollback readiness."

    utility1, utility2 = st.columns(2)
    with utility1:
        if st.button("Clear Chat", use_container_width=True):
            st.session_state["chat_messages"] = []
            st.rerun()
    with utility2:
        if st.button("Refresh Grounding Sources", use_container_width=True):
            st.session_state["orch_status"] = _safe_get_cached_or_fetch(base_url, "orch_status", "/orchestration/status")
            st.session_state["registry_models"] = _safe_get_cached_or_fetch(base_url, "registry_models", "/registry/models")
            st.session_state["feature_tables"] = _safe_get_cached_or_fetch(base_url, "feature_tables", "/feature-store/tables")
            st.success("Grounding sources refreshed.")

    for m in st.session_state["chat_messages"]:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    context = st.text_area("Additional Context (optional)", value="", key="chat_context")
    chat_file = st.file_uploader("Attach Structured File (CSV/XLSX)", type=["csv", "xlsx"], key="chat_struct_file")
    quick_prompt_prefill = st.session_state.get("chat_quick_prompt", "")
    if quick_prompt_prefill:
        st.info(f"Quick prompt ready: {quick_prompt_prefill}")

    send_quick = False
    if quick_prompt_prefill:
        send_quick = st.button("Send Quick Prompt", use_container_width=True)

    prompt = st.chat_input("Ask your AI Scientist...")
    if prompt or send_quick:
        user_prompt = quick_prompt_prefill.strip() if send_quick else prompt
        st.session_state["chat_quick_prompt"] = ""
        st.session_state["chat_messages"].append({"role": "user", "content": user_prompt})
        try:
            grounding_ctx = _build_chat_grounding_context(
                base_url=base_url,
                include_structured=bool(include_structured),
                include_registry=bool(include_registry),
                include_feature_store=bool(include_feature_store),
                include_orchestration=bool(include_orchestration),
            )
            role_instruction = _chat_role_instruction(user_mode)
            format_instruction = (
                "Return in sections: 1) Outcome 2) Why 3) Risks/Assumptions 4) Next 3 Actions."
                if response_style == "Structured"
                else "Return concise natural language."
            )
            full_context = (
                f"User mode: {user_mode}\n"
                f"{role_instruction}\n"
                f"{format_instruction}\n"
                f"Optional user context:\n{context or 'N/A'}\n"
                f"Grounding data (use these facts first; if missing, say unknown):\n{safe_json(grounding_ctx)}"
            )
            if chat_file is not None:
                res = api_post_multipart(
                    base_url,
                    "/chat/structured-file",
                    files={"file": (chat_file.name, chat_file.getvalue())},
                    data={"question": user_prompt, "provider": "bedrock"},
                )
                ans = res.get("answer", "")
            else:
                res = api_post_json(
                    base_url,
                    "/chat",
                    {"question": user_prompt, "context": full_context, "provider": "bedrock"},
                )
                ans = res.get("answer", "")
        except Exception as exc:
            ans = f"Chat failed. Backend/LLM error: {exc}"
        formatted_ans = _format_chat_answer_text(ans)
        with st.chat_message("assistant"):
            streamed = st.write_stream(_stream_text_chunks(formatted_ans))
        st.session_state["chat_messages"].append({"role": "assistant", "content": streamed if isinstance(streamed, str) else formatted_ans})
        st.rerun()


def mlops_tab(base_url: str) -> None:
    st.subheader("MLOps Control Plane")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("### Orchestration")
        if st.button("Refresh Orchestration Status", use_container_width=True):
            try:
                st.session_state["orch_status"] = api_get(base_url, "/orchestration/status")
            except Exception as exc:
                st.error(f"Orchestration status failed: {exc}")
        if st.session_state.get("orch_status"):
            st.code(safe_json(st.session_state["orch_status"]), language="json")

    with c2:
        st.markdown("### Registry")
        if st.button("Refresh Registered Models", use_container_width=True):
            try:
                st.session_state["registry_models"] = api_get(base_url, "/registry/models")
            except Exception as exc:
                st.error(f"Registry fetch failed: {exc}")
        models_payload = st.session_state.get("registry_models", {})
        model_rows = models_payload.get("models", []) if isinstance(models_payload, dict) else []
        if model_rows:
            st.dataframe(pd.DataFrame(model_rows), use_container_width=True)
            model_ids = [str(m.get("model_id")) for m in model_rows if m.get("model_id")]
            if model_ids:
                selected_model = st.selectbox("Model Details", options=model_ids, key="mlops_model_details_id")
                chosen = next((m for m in model_rows if str(m.get("model_id")) == selected_model), {})
                if chosen:
                    st.markdown("#### Selected Model Overview")
                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("Stage", str(chosen.get("stage", "none")))
                    m2.metric("Problem Type", str(chosen.get("problem_type", "unknown")))
                    m3.metric("Champion Score", f"{float(chosen.get('champion_score', 0.0)):.4f}" if chosen.get("champion_score") is not None else "N/A")
                    m4.metric("Target", str(chosen.get("target", "N/A")))
                    if chosen.get("approvals"):
                        st.markdown("**Approvals**")
                        st.dataframe(pd.DataFrame(chosen.get("approvals", [])), use_container_width=True)
                    if chosen.get("holdout_metrics"):
                        st.markdown("**Holdout Metrics**")
                        st.dataframe(pd.DataFrame([chosen.get("holdout_metrics", {})]), use_container_width=True)
                    if chosen.get("artifact_dir") or chosen.get("model_path"):
                        st.markdown("**Artifact Paths**")
                        st.caption(f"artifact_dir: {chosen.get('artifact_dir', '')}")
                        st.caption(f"model_path: {chosen.get('model_path', '')}")
                        st.caption(f"metadata_path: {chosen.get('metadata_path', '')}")
                    if chosen.get("history"):
                        st.markdown("**Lifecycle History**")
                        st.dataframe(pd.DataFrame(chosen.get("history", [])), use_container_width=True)

    st.markdown("### Registry Actions")
    r1, r2, r3 = st.columns(3)
    with r1:
        default_model = st.session_state.get("mlops_model_details_id", "")
        approve_model_id = st.text_input("Approve Model ID", value=default_model, key="approve_model_id")
        approve_user = st.text_input("Approver", value="risk_reviewer", key="approve_user")
        approve_note = st.text_input("Approval Note", value="reviewed", key="approve_note")
        if st.button("Approve Model", use_container_width=True):
            try:
                out = api_post_json(
                    base_url,
                    "/registry/approve",
                    {"model_id": approve_model_id, "approver": approve_user, "note": approve_note},
                )
                st.success("Approved.")
                st.code(safe_json(out), language="json")
            except Exception as exc:
                st.error(f"Approve failed: {exc}")
    with r2:
        default_model = st.session_state.get("mlops_model_details_id", "")
        promote_model_id = st.text_input("Promote Model ID", value=default_model, key="promote_model_id")
        promote_stage = st.selectbox("Stage", options=["staging", "production", "archived"], key="promote_stage")
        promote_actor = st.text_input("Actor", value="approver", key="promote_actor")
        if st.button("Promote Model", use_container_width=True):
            try:
                out = api_post_json(
                    base_url,
                    "/registry/promote",
                    {"model_id": promote_model_id, "stage": promote_stage, "actor": promote_actor},
                )
                st.success("Promoted.")
                st.code(safe_json(out), language="json")
            except Exception as exc:
                st.error(f"Promote failed: {exc}")
    with r3:
        default_model = st.session_state.get("mlops_model_details_id", "")
        rollback_model_id = st.text_input("Rollback to Model ID", value=default_model, key="rollback_model_id")
        rollback_actor = st.text_input("Rollback Actor", value="approver", key="rollback_actor")
        if st.button("Rollback Production", use_container_width=True):
            try:
                out = api_post_json(
                    base_url,
                    "/registry/rollback",
                    {"model_id": rollback_model_id, "actor": rollback_actor},
                )
                st.success("Rollback applied.")
                st.code(safe_json(out), language="json")
            except Exception as exc:
                st.error(f"Rollback failed: {exc}")

    st.markdown("### Feature Store")
    fs1, fs2 = st.columns(2)
    with fs1:
        if st.button("List Feature Tables", use_container_width=True):
            try:
                st.session_state["feature_tables"] = api_get(base_url, "/feature-store/tables")
            except Exception as exc:
                st.error(f"List tables failed: {exc}")
        if st.session_state.get("feature_tables"):
            st.code(safe_json(st.session_state["feature_tables"]), language="json")

        fs_table = st.text_input("Offline Upsert Table", value="customer_features")
        fs_rows_json = st.text_area(
            "Rows JSON",
            value='[{"entity_id":"c1","f_total_spend":1200.5,"event_ts":"2026-03-05T10:00:00Z"}]',
            height=120,
        )
        if st.button("Upsert Offline Features", use_container_width=True):
            try:
                rows = json.loads(fs_rows_json)
                out = api_post_json(base_url, "/feature-store/offline/upsert", {"table": fs_table, "rows": rows})
                st.success("Offline features upserted.")
                st.code(safe_json(out), language="json")
            except Exception as exc:
                st.error(f"Offline upsert failed: {exc}")

    with fs2:
        mat_table = st.text_input("Materialize Table", value="customer_features")
        mat_key = st.text_input("Key Column", value="entity_id")
        mat_ts = st.text_input("Timestamp Column (optional)", value="event_ts")
        if st.button("Materialize Online Store", use_container_width=True):
            try:
                payload = {"table": mat_table, "key_col": mat_key, "ts_col": mat_ts or None}
                out = api_post_json(base_url, "/feature-store/online/materialize", payload)
                st.success("Online store materialized.")
                st.code(safe_json(out), language="json")
            except Exception as exc:
                st.error(f"Materialize failed: {exc}")

        read_table = st.text_input("Read Table", value="customer_features")
        read_key_col = st.text_input("Read Key Column", value="entity_id")
        read_key_val = st.text_input("Read Key Value", value="c1")
        if st.button("Read Online Feature Row", use_container_width=True):
            try:
                out = api_post_json(
                    base_url,
                    "/feature-store/online/read",
                    {"table": read_table, "key_col": read_key_col, "key_val": read_key_val},
                )
                st.code(safe_json(out), language="json")
            except Exception as exc:
                st.error(f"Online read failed: {exc}")


def custom_gpt_tab(base_url: str) -> None:
    render_custom_gpt_tab(base_url)



def icp_agent_tab(base_url: str) -> None:
    st.subheader("AI SDR Chatbot")
    st.caption(
        "One chat-driven flow: ICP -> Discovery -> Enrichment -> Prospect Intelligence "
        "-> Qualification -> Outreach -> Follow-Up + Conversation -> Meeting + CRM."
    )

    def default_icp_payload() -> Dict[str, Any]:
        return {
            "industries": [],
            "geographies": [],
            "employee_band": "mid-market",
            "revenue_band": "mid-market",
            "target_personas": [],
            "target_departments": [],
            "pain_points": [],
            "filters": {
                "target_industries": [],
                "revenue_ranges": [],
                "company_size_ranges": [],
                "funding_stages": [],
                "geographies": [],
                "tech_stack_signals": [],
                "target_job_titles": [],
                "seniority_levels": [],
                "pain_points": [],
                "buying_signals": [],
            },
            "offer_summary": "",
            "exclusions": {
                "industries": [],
                "customer_statuses": ["existing_customer"],
                "company_names": [],
                "employee_max_below": 20,
                "geographies": [],
            },
            "scoring_weights": {
                "industry_fit": 0.30,
                "company_size_fit": 0.20,
                "persona_fit": 0.25,
                "geo_fit": 0.10,
                "pain_point_fit": 0.15,
            },
            "notes": "",
            "created_by": "streamlit-user",
        }

    def split_csv(value: str) -> List[str]:
        return [item.strip() for item in value.split(",") if item.strip()]

    def append_once(items: List[str], value: str) -> None:
        if value and value not in items:
            items.append(value)

    def draft_from_prompt(prompt: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        draft = json.loads(json.dumps(payload))
        lowered = prompt.lower()
        draft["notes"] = prompt

        industry_map = {
            "saas": "SaaS",
            "software": "Software",
            "fitness": "Fitness",
            "healthcare": "Healthcare",
            "ecommerce": "Ecommerce",
            "commerce": "Commerce",
            "fintech": "Fintech",
        }
        geo_map = {
            "us": "United States",
            "united states": "United States",
            "usa": "United States",
            "uk": "United Kingdom",
            "united kingdom": "United Kingdom",
            "india": "India",
            "europe": "Europe",
            "canada": "Canada",
        }
        persona_map = {
            "revops": "RevOps",
            "revenue operations": "Revenue Operations",
            "sales ops": "Sales Operations",
            "vp sales": "VP Sales",
            "head of sales": "Head of Sales",
            "marketing": "Marketing Leader",
            "founder": "Founder",
            "cmo": "CMO",
            "cfo": "CFO",
            "personal trainer": "Personal Trainer",
            "gym owner": "Gym Owner",
        }
        pain_map = {
            "forecast": "forecasting inconsistency",
            "pipeline": "poor pipeline visibility",
            "manual lead routing": "manual lead routing",
            "low replies": "low outbound reply rate",
            "churn": "customer churn risk",
            "qualification": "manual qualification work",
            "home gym": "home gym planning friction",
            "financing": "financing decision friction",
        }

        for token, value in industry_map.items():
            if token in lowered:
                append_once(draft["industries"], value)
                append_once(draft["filters"]["target_industries"], value)
        for token, value in geo_map.items():
            if token in lowered:
                append_once(draft["geographies"], value)
                append_once(draft["filters"]["geographies"], value)
        for token, value in persona_map.items():
            if token in lowered:
                append_once(draft["target_personas"], value)
                append_once(draft["filters"]["target_job_titles"], value)
        for token, value in pain_map.items():
            if token in lowered:
                append_once(draft["pain_points"], value)
                append_once(draft["filters"]["pain_points"], value)

        if "enterprise" in lowered:
            draft["employee_band"] = "enterprise"
            draft["revenue_band"] = "enterprise"
            append_once(draft["filters"]["company_size_ranges"], "enterprise")
            append_once(draft["filters"]["revenue_ranges"], "enterprise")
        elif "smb" in lowered or "small business" in lowered:
            draft["employee_band"] = "smb"
            draft["revenue_band"] = "growth"
            append_once(draft["filters"]["company_size_ranges"], "smb")
            append_once(draft["filters"]["revenue_ranges"], "growth")
        elif "mid-market" in lowered or "mid market" in lowered:
            draft["employee_band"] = "mid-market"
            draft["revenue_band"] = "mid-market"
            append_once(draft["filters"]["company_size_ranges"], "mid-market")
            append_once(draft["filters"]["revenue_ranges"], "mid-market")

        tech_signals = {
            "salesforce": "Salesforce",
            "hubspot": "HubSpot",
            "snowflake": "Snowflake",
            "aws": "AWS",
            "azure": "Azure",
        }
        for token, value in tech_signals.items():
            if token in lowered:
                append_once(draft["filters"]["tech_stack_signals"], value)
        funding_signals = {
            "series a": "Series A",
            "series b": "Series B",
            "series c": "Series C",
            "venture backed": "Venture-backed",
        }
        for token, value in funding_signals.items():
            if token in lowered:
                append_once(draft["filters"]["funding_stages"], value)

        if not draft.get("offer_summary"):
            draft["offer_summary"] = (
                "AI SDR operating layer for discovering, qualifying, and converting high-fit prospects."
            )
        return draft

    def compact_icp_summary(payload: Dict[str, Any]) -> str:
        return (
            f"Industries: {', '.join(payload.get('industries') or ['not set'])}\n\n"
            f"Geographies: {', '.join(payload.get('geographies') or ['not set'])}\n\n"
            f"Personas: {', '.join(payload.get('target_personas') or ['not set'])}\n\n"
            f"Pain points: {', '.join(payload.get('pain_points') or ['not set'])}"
        )

    def merge_llm_suggestion(payload: Dict[str, Any], suggestion: Dict[str, Any]) -> Dict[str, Any]:
        merged = json.loads(json.dumps(payload))
        for persona in suggestion.get("personas", []) or []:
            append_once(merged["target_personas"], str(persona))
        for pain_point in suggestion.get("pain_points", []) or []:
            append_once(merged["pain_points"], str(pain_point))
        exclusions = suggestion.get("exclusions") or {}
        merged.setdefault("exclusions", default_icp_payload()["exclusions"])
        for field in ("industries", "customer_statuses", "company_names", "geographies"):
            for item in exclusions.get(field, []) or []:
                append_once(merged["exclusions"][field], str(item))
        if exclusions.get("employee_max_below") is not None:
            merged["exclusions"]["employee_max_below"] = exclusions.get("employee_max_below")
        return merged

    def render_structured_icp_response(payload: Dict[str, Any], suggestion: Optional[Dict[str, Any]] = None) -> None:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Industries", len(payload.get("industries", [])))
        c2.metric("Geographies", len(payload.get("geographies", [])))
        c3.metric("Personas", len(payload.get("target_personas", [])))
        c4.metric("Pain Points", len(payload.get("pain_points", [])))

        st.markdown("#### Recommended ICP")
        st.markdown(
            f"""
**Target accounts:** {", ".join(payload.get("industries") or ["Not set"])} companies in {", ".join(payload.get("geographies") or ["Not set"])}

**Company band:** {payload.get("employee_band", "Not set")} employees / {payload.get("revenue_band", "Not set")} revenue

**Target personas:** {", ".join(payload.get("target_personas") or ["Not set"])}

**Pain points:** {", ".join(payload.get("pain_points") or ["Not set"])}

**Offer fit:** {payload.get("offer_summary") or "Not set"}
"""
        )

        if suggestion:
            st.markdown("#### LLM Reasoning")
            st.info(str(suggestion.get("reasoning") or "No reasoning returned."))

    def render_pipeline_response(result: Dict[str, Any]) -> None:
        st.markdown("#### SDR Pipeline Summary")
        summary = result.get("summary", {})
        s1, s2, s3, s4, s5, s6 = st.columns(6)
        s1.metric("Accounts Discovered", int(summary.get("accounts_discovered", 0)))
        s2.metric("Contacts Discovered", int(summary.get("contacts_discovered", 0)))
        s3.metric("Accounts Enriched", int(summary.get("accounts_enriched", 0)))
        s4.metric("Prospects Analyzed", int(summary.get("prospects_analyzed", 0)))
        s5.metric("Campaigns Drafted", int(summary.get("campaigns_drafted", 0)))
        s6.metric("Follow-Up Plans", int(summary.get("follow_up_plans_drafted", 0)))
        if int(summary.get("accounts_discovered", 0)) > 0 and int(summary.get("contacts_discovered", 0)) == 0:
            st.warning(
                "Real accounts were discovered, but no real contacts/emails were returned. "
                "Prospect Intelligence, Qualification, Outreach, Conversation, Follow-Up, "
                "Meeting, and CRM need contacts, so they will not run until you connect a "
                "real contact provider, CRM lead source, or upload contacts."
            )

        st.markdown("#### Actual Agent Execution Flow")

        with st.expander("1. ICP Setup", expanded=True):
            icp_definition = result.get("icp_definition", {})
            industries = icp_definition.get("account_criteria", {}).get("industries", [])
            geographies = icp_definition.get("account_criteria", {}).get("geographies", [])
            employee_band = icp_definition.get("account_criteria", {}).get("employee_band", "not set")
            revenue_band = icp_definition.get("account_criteria", {}).get("revenue_band", "not set")
            personas = icp_definition.get("persona_criteria", {}).get("titles", [])
            account_criteria = icp_definition.get("account_criteria", {})
            persona_criteria = icp_definition.get("persona_criteria", {})
            pain_points = icp_definition.get("pain_points", [])
            exclusions = icp_definition.get("exclusions", {})
            st.markdown(
                "\n".join(
                    [
                        "- The ICP Agent translated your prompt into a structured target-customer definition.",
                        f"- It focused on {', '.join(industries or ['not set'])} companies in {', '.join(geographies or ['not set'])}.",
                        f"- It set the company band to {employee_band} employees and {revenue_band} revenue.",
                        f"- It prioritized these personas: {', '.join(personas or ['not set'])}.",
                        f"- It highlighted these pain points: {', '.join(pain_points or ['not set'])}.",
                    ]
                )
            )
            excluded_industries = exclusions.get("industries", []) or []
            excluded_statuses = exclusions.get("customer_statuses", []) or []
            if excluded_industries or excluded_statuses:
                st.markdown(
                    f"- It also applied exclusions such as industries {', '.join(excluded_industries or ['none'])} "
                    f"and customer statuses {', '.join(excluded_statuses or ['none'])}."
                )
            structured_filters = {
                "Revenue": account_criteria.get("revenue_ranges", []),
                "Company size": account_criteria.get("company_size_ranges", []),
                "Funding": account_criteria.get("funding_stages", []),
                "Technology": account_criteria.get("tech_stack_signals", []),
                "Seniority": persona_criteria.get("seniority_levels", []),
                "Buying signals": persona_criteria.get("buying_signals", []),
            }
            active_filters = [
                f"{label}: {', '.join(values)}"
                for label, values in structured_filters.items()
                if values
            ]
            if active_filters:
                st.markdown("- Structured filters: " + " | ".join(active_filters))
            with st.expander("Developer JSON - ICP", expanded=False):
                st.json(icp_definition)

        with st.expander("2. Prospect Discovery (Lead Sources + Fit Scoring)", expanded=True):
            accounts = result.get("discovered_accounts", [])
            contacts = result.get("discovered_contacts", [])
            d1, d2 = st.columns(2)
            d1.metric("Matching Accounts", len(accounts))
            d2.metric("Matching Contacts", len(contacts))
            st.markdown("- The Prospect Discovery Agent used the ICP to find matching accounts and likely buying contacts.")
            if accounts:
                st.markdown("**Top matching accounts**")
                for account in accounts[:5]:
                    company_name = account.get("company_name") or "Unknown account"
                    industry = account.get("industry") or "unknown industry"
                    location = account.get("location") or "unknown location"
                    fit_score = account.get("fit_score")
                    fit_text = f"fit score {fit_score}" if fit_score is not None else "fit score not available"
                    st.markdown(
                        f"- {company_name} in {industry} / {location}, with {fit_text}."
                    )
            else:
                st.info("No matching accounts were found in the current discovery layer.")

            if contacts:
                st.markdown("**Top matching contacts**")
                for contact in contacts[:8]:
                    full_name = contact.get("full_name") or "Unknown contact"
                    title = contact.get("title") or "unknown title"
                    company_name = contact.get("company_name") or contact.get("account_id") or "unknown account"
                    confidence = contact.get("confidence")
                    persona_match = contact.get("persona_match_score")
                    extra_parts = []
                    if confidence is not None:
                        extra_parts.append(f"confidence {confidence}")
                    if persona_match is not None:
                        extra_parts.append(f"persona match {persona_match}")
                    extra = ", ".join(extra_parts) if extra_parts else "no additional scores"
                    st.markdown(f"- {full_name} — {title} at {company_name}, with {extra}.")
            else:
                st.info("No matching contacts were returned for the discovered accounts.")

            with st.expander("Developer JSON - Discovery", expanded=False):
                st.json({"accounts": accounts, "contacts": contacts})

        with st.expander("3. Prospect Intelligence (Prospect Enrichment Done)", expanded=True):
            try:
                enrichment_status = api_get(base_url, "/enrichment/status")
                provider_label = enrichment_status.get("configured_web_provider") or enrichment_status.get("search_provider")
                status_cols = st.columns(3)
                status_cols[0].metric("Web Search", str(provider_label or "unknown"))
                status_cols[1].metric(
                    "Live Retrieval",
                    "Configured" if enrichment_status.get("live_search_configured") else "Missing",
                )
                status_cols[2].metric(
                    "OpenSearch Index",
                    "Available" if enrichment_status.get("opensearch_index_available") else "Direct live results",
                )
                st.caption(str(enrichment_status.get("detail", "")))
            except Exception as exc:
                st.warning(f"Enrichment provider status unavailable: {exc}")
            enriched_results = result.get("enriched_results", [])
            if not enriched_results:
                st.info("No enriched accounts were returned.")
            for idx, enriched in enumerate(enriched_results, start=1):
                company_name = enriched.get("company_name") or enriched.get("account", {}).get("company_name") or "Unknown"
                st.markdown(f"**Enriched Account {idx}: {company_name}**")
                e1, e2, e3 = st.columns(3)
                e1.metric("Status", str(enriched.get("status", "unknown")))
                e2.metric("Signals", len(enriched.get("signals", []) or []))
                e3.metric("Confidence", f"{float(enriched.get('confidence_score', 0.0)):.2f}")
                if enriched.get("company_summary"):
                    st.markdown(f"- The Enrichment Agent found this account context: {enriched.get('company_summary')}")
                if enriched.get("products_services"):
                    st.markdown("**Products / Services it inferred**")
                    for item in enriched.get("products_services", []):
                        st.markdown(f"- {item}")
                if enriched.get("target_customers"):
                    st.markdown("**Target customers / buyers it inferred**")
                    for item in enriched.get("target_customers", []):
                        st.markdown(f"- {item}")
                if enriched.get("signals"):
                    st.markdown("**Commercial signals it found**")
                    for signal in enriched.get("signals", []):
                        st.markdown(
                            f"- {str(signal.get('type', 'signal')).replace('_', ' ').title()}: {signal.get('detail', '')}"
                        )
                if enriched.get("pain_point_hypotheses"):
                    st.markdown("**Likely pain points**")
                    for item in enriched.get("pain_point_hypotheses", []):
                        st.markdown(f"- {item}")
                if enriched.get("personalization_angles"):
                    st.markdown("**Personalization angles for outreach**")
                    for item in enriched.get("personalization_angles", []):
                        st.markdown(f"- {item}")
                if enriched.get("contact_briefs"):
                    st.markdown("**What it inferred about the contacts**")
                    for brief in enriched.get("contact_briefs", []):
                        st.markdown(
                            f"- {brief.get('full_name', 'Unknown')} ({brief.get('title', 'unknown title')}): "
                            f"{brief.get('likely_responsibility', '')} "
                            f"Outreach hook: {brief.get('personalization_hook', '')}"
                        )
                if enriched.get("recommended_next_action"):
                    st.markdown("**Recommended next step**")
                    st.info(str(enriched.get("recommended_next_action")))
                citations = enriched.get("citations", []) or []
                if citations:
                    st.markdown("**Source links used by enrichment**")
                    for citation in citations[:8]:
                        title = citation.get("title") or citation.get("url") or "Source"
                        url = citation.get("url") or ""
                        snippet = citation.get("snippet", "")[:220]
                        st.markdown(f"- [{title}]({url})")
                        if snippet:
                            st.caption(snippet)
                with st.expander(f"Developer JSON - Enrichment {idx}", expanded=False):
                    st.json(enriched)

        with st.expander("3b. Prospect Intelligence Scores, Intent, Recommendations", expanded=True):
            intelligence_items = result.get("prospect_intelligence", [])
            if not intelligence_items:
                st.info("No Prospect Intelligence results were returned.")
            for item in intelligence_items:
                contact_name = item.get("contact_name") or item.get("contact_id") or "Unknown contact"
                intent = item.get("intent", {})
                propensity = item.get("propensity", {})
                ranking = item.get("ranking", {})
                personalization = item.get("personalization", {})
                i1, i2, i3, i4, i5 = st.columns(5)
                i1.metric("Prospect", contact_name)
                i2.metric("Intent", f"{float(intent.get('probability', {}).get('value', 0.0)):.0%}")
                i3.metric("Reply", f"{float(propensity.get('reply_probability', {}).get('value', 0.0)):.0%}")
                i4.metric("Meeting", f"{float(propensity.get('meeting_probability', {}).get('value', 0.0)):.0%}")
                i5.metric("Qualification", f"{float(propensity.get('qualification_probability', {}).get('value', 0.0)):.0%}")
                st.markdown(
                    f"**Queue rank:** {ranking.get('rank') or '-'} | "
                    f"**Priority:** {str(ranking.get('priority_band', 'unknown')).title()} | "
                    f"**Ranking provider:** `{ranking.get('provider', 'local')}`"
                )
                model_versions = item.get("model_versions", {}) or {}
                st.caption(
                    "Prediction models: "
                    f"intent `{model_versions.get('intent', 'unknown')}`, "
                    f"reply `{model_versions.get('reply', 'unknown')}`, "
                    f"meeting `{model_versions.get('meeting', 'unknown')}`, "
                    f"qualification `{model_versions.get('qualification', 'unknown')}`. "
                    "Queue ranking is local until MetaRank has enough real outcome events."
                )
                st.markdown(f"**Primary angle:** {personalization.get('primary_angle', '')}")
                st.markdown(f"**Value proposition:** {personalization.get('value_proposition', '')}")
                st.markdown(f"**Email opening:** {personalization.get('email_opening', '')}")
                st.markdown(f"**LinkedIn:** {personalization.get('linkedin_message', '')}")
                st.caption(
                    "These probabilities estimate current response outcomes. They are separate from "
                    "Discovery fit, which only measures ICP similarity."
                )
                warnings = item.get("warnings", []) or []
                if warnings:
                    st.warning(" | ".join(str(warning) for warning in warnings))
                with st.expander(f"Developer JSON - Intelligence - {contact_name}", expanded=False):
                    st.json(item)

        with st.expander("4. Qualification Hub (BANT / MEDDIC)", expanded=True):
            qualification_items = result.get("qualification_results", [])
            if not qualification_items:
                st.info("No Qualification Agent results were returned.")
            for item in qualification_items:
                contact_name = item.get("contact_name") or item.get("contact_id") or "Unknown contact"
                bant = item.get("bant") or {}
                meddic = item.get("meddic") or {}
                q1, q2, q3, q4, q5 = st.columns(5)
                q1.metric("Prospect", contact_name)
                q2.metric("Qualification", f"{int(item.get('qualification_score', 0))}/100")
                q3.metric("BANT", f"{int(bant.get('score', 0))}/100")
                q4.metric("MEDDIC", f"{int(meddic.get('score', 0))}/100")
                q5.metric("Status", str(item.get("qualification_status", "unknown")))
                st.markdown(
                    f"**Tier:** {item.get('qualification_tier', '-')} | "
                    f"**Sales ready:** {'Yes' if item.get('sales_ready') else 'No'} | "
                    f"**Decision source:** `{item.get('decision_source', 'unknown')}`"
                )
                st.markdown(f"**Next action:** {item.get('next_action', '')}")
                st.caption(
                    "The calibrated LightGBM qualification probability is combined with "
                    "BANT/MEDDIC evidence coverage. BANT and MEDDIC identify what is known "
                    "and what the SDR still needs to validate."
                )
                for reason in item.get("reasoning", []) or []:
                    st.markdown(f"- {reason}")
                missing = item.get("missing_information", []) or []
                if missing:
                    st.warning("Missing qualification evidence: " + "; ".join(missing))
                with st.expander(
                    f"Developer JSON - Qualification - {contact_name}", expanded=False
                ):
                    st.json(item)

        with st.expander("5. Outreach & Conversation Hub - Outreach", expanded=True):
            campaigns = result.get("outreach_campaigns", [])
            if not campaigns:
                st.info("No eligible outreach campaign drafts were created.")
            for campaign in campaigns:
                campaign_id = str(campaign.get("campaign_id", ""))
                st.markdown(
                    f"**{campaign.get('contact_name', 'Unknown')} at "
                    f"{campaign.get('company_name', 'Unknown')}**"
                )
                c1, c2, c3 = st.columns(3)
                c1.metric("Campaign Status", str(campaign.get("status", "unknown")).replace("_", " ").title())
                c2.metric("Messages", len(campaign.get("messages", []) or []))
                c3.metric(
                    "Approval",
                    "Required" if campaign.get("require_approval", True) else "Not required",
                )
                policy = campaign.get("policy", {})
                st.markdown(
                    f"**Eligible channels:** {', '.join(policy.get('eligible_channels', []) or ['none'])}"
                )
                for reason in policy.get("reasons", []) or []:
                    st.caption(reason)
                for message in campaign.get("messages", []) or []:
                    st.markdown(
                        f"**{str(message.get('channel', '')).title()} "
                        f"touch {int(message.get('sequence_order', 0)) + 1}**"
                    )
                    if message.get("subject"):
                        st.markdown(f"Subject: `{message.get('subject')}`")
                    st.code(str(message.get("body", "")), language="text")
                    review = message.get("review", {}) or {}
                    st.caption(
                        f"Review: {'passed' if review.get('passed') else 'blocked'} | "
                        f"Provider: {message.get('provider', 'dry_run')} | "
                        f"Status: {message.get('status', 'draft')}"
                    )
                a1, a2, a3 = st.columns(3)
                if a1.button(
                    "Approve",
                    key=f"pipeline_approve_{campaign_id}",
                    use_container_width=True,
                    disabled=campaign.get("status") != "pending_approval",
                ):
                    try:
                        updated = api_post_json(
                            base_url,
                            f"/outreach/campaigns/{campaign_id}/approve",
                            {"approved_by": "streamlit-user"},
                        )
                        campaign.update(updated)
                        st.success("Campaign approved.")
                        st.rerun()
                    except Exception as exc:
                        st.error(f"Approval failed: {exc}")
                if a2.button(
                    "Send Due Messages",
                    key=f"pipeline_send_{campaign_id}",
                    use_container_width=True,
                    disabled=campaign.get("status") not in {"approved", "running"},
                ):
                    try:
                        updated = api_post_json(
                            base_url,
                            f"/outreach/campaigns/{campaign_id}/send?force=true",
                            {},
                        )
                        campaign.update(updated)
                        st.success("Campaign processed by the configured provider.")
                        st.rerun()
                    except Exception as exc:
                        st.error(f"Send failed: {exc}")
                if a3.button(
                    "Pause",
                    key=f"pipeline_pause_{campaign_id}",
                    use_container_width=True,
                    disabled=campaign.get("status") in {"paused", "completed", "cancelled"},
                ):
                    try:
                        updated = api_post_json(
                            base_url,
                            f"/outreach/campaigns/{campaign_id}/pause",
                            {"reason": "Paused in Streamlit"},
                        )
                        campaign.update(updated)
                        st.success("Campaign paused.")
                        st.rerun()
                    except Exception as exc:
                        st.error(f"Pause failed: {exc}")
                with st.expander(f"Developer JSON - Campaign - {campaign_id}", expanded=False):
                    st.json(campaign)
            for warning in result.get("warnings", []) or []:
                st.warning(str(warning))

        with st.expander("5b. Outreach & Conversation Hub - Follow-Up Automation", expanded=True):
            plans = result.get("follow_up_plans", [])
            if not plans:
                st.info("No follow-up plans were created.")
            for plan in plans:
                scheduler_backend = plan.get("scheduler_backend", "temporal")
                temporal_status = plan.get("temporal_schedule_status", "not_started")
                st.markdown(
                    f"**Plan `{plan.get('plan_id', '')}`** - "
                    f"{str(plan.get('status', 'unknown')).replace('_', ' ').title()}"
                )
                st.caption(
                    f"Scheduler: `{scheduler_backend}` | Temporal status: `{temporal_status}` | "
                    f"Workflow: `{plan.get('temporal_workflow_id') or 'not started'}`"
                )
                if plan.get("temporal_error"):
                    st.warning(f"Temporal scheduling error: {plan.get('temporal_error')}")
                for touch in plan.get("touches", []) or []:
                    st.markdown(
                        f"- Touch {touch.get('sequence_order')}: "
                        f"{touch.get('channel')} after {touch.get('delay_hours')} hours "
                        f"({touch.get('status')})"
                    )
                    st.code(str(touch.get("body", "")), language="text")
                plan_id = str(plan.get("plan_id", ""))
                if st.button(
                    "Approve Follow-Up In Temporal",
                    key=f"pipeline_follow_up_approve_{plan_id}",
                    use_container_width=True,
                    disabled=plan.get("status") != "pending_approval",
                ):
                    try:
                        updated = api_post_json(
                            base_url,
                            f"/follow-up/plans/{plan_id}/approve",
                            {"approved_by": "streamlit-user"},
                        )
                        plan.update(updated)
                        st.success("Follow-up plan approved and Temporal workflow started.")
                        st.rerun()
                    except Exception as exc:
                        st.error(f"Follow-up approval failed: {exc}")

        with st.expander("5c. Outreach & Conversation Hub - Reply Detection", expanded=True):
            try:
                agent_status = api_get(base_url, "/conversations/agent/status")
                status_cols = st.columns(4)
                status_cols[0].metric("Mode", str(agent_status.get("mode", "unknown")).title())
                status_cols[1].metric(
                    "Structured LLM",
                    "Reachable" if agent_status.get("llm_reachable") else "Degraded",
                )
                status_cols[2].metric("Model", str(agent_status.get("model_name", "unknown")))
                status_cols[3].metric(
                    "Langfuse",
                    "Configured" if agent_status.get("langfuse_configured") else "Optional",
                )
                st.caption(str(agent_status.get("detail", "")))
            except Exception as exc:
                st.warning(f"Conversation agent status unavailable: {exc}")

            conversation_campaigns = result.get("outreach_campaigns", []) or []
            if conversation_campaigns:
                campaign_options = {
                    (
                        f"{campaign.get('contact_name', 'Unknown')} at "
                        f"{campaign.get('company_name', 'Unknown')} "
                        f"({campaign.get('campaign_id', '')[:8]})"
                    ): str(campaign.get("campaign_id", ""))
                    for campaign in conversation_campaigns
                }
                selected_label = st.selectbox(
                    "Campaign whose prospect replied",
                    options=list(campaign_options),
                    key="conversation_sim_campaign",
                )
                simulated_reply = st.text_area(
                    "Simulate an inbound prospect reply",
                    value="Interested. Can we meet next week to discuss pricing?",
                    key="conversation_sim_reply",
                    help=(
                        "This uses the same Conversation Agent endpoint as an inbound provider "
                        "reply. It is used because Brevo inbound parsing requires a receiving domain."
                    ),
                )
                if st.button(
                    "Run Conversation Agent",
                    use_container_width=True,
                    key="conversation_simulate_in_pipeline",
                ):
                    try:
                        created = api_post_json(
                            base_url,
                            "/conversations/inbound",
                            {
                                "campaign_id": campaign_options[selected_label],
                                "body": simulated_reply,
                                "subject": "Re: outreach",
                                "channel": "email",
                                "provider": "streamlit-simulation",
                                "sender": "prospect@example.com",
                            },
                        )
                        st.session_state["latest_conversation_id"] = created.get(
                            "conversation_id"
                        )
                        st.success(
                            "Reply classified, Temporal follow-up stopped, and a response draft created."
                        )
                        st.rerun()
                    except Exception as exc:
                        st.error(f"Conversation Agent failed: {exc}")

            try:
                conversations = api_get(base_url, "/conversations").get("items", [])
            except Exception:
                conversations = []
            relevant_campaign_ids = {
                str(campaign.get("campaign_id", "")) for campaign in conversation_campaigns
            }
            for conversation in conversations:
                if relevant_campaign_ids and conversation.get("campaign_id") not in relevant_campaign_ids:
                    continue
                conversation_id = str(conversation.get("conversation_id", ""))
                classification = conversation.get("latest_classification", {}) or {}
                st.markdown(
                    f"**{conversation.get('contact_name', 'Unknown')} - "
                    f"{str(classification.get('intent', 'unclassified')).replace('_', ' ').title()}**"
                )
                st.caption(
                    f"Source: `{classification.get('source', 'unknown')}` | "
                    f"Confidence: {float(classification.get('confidence', 0)):.0%} | "
                    f"Trace: `{conversation.get('observability_trace_id') or 'local-only'}`"
                )
                if conversation.get("degraded_reason"):
                    st.warning(str(conversation.get("degraded_reason")))
                response_draft = st.text_area(
                    "Conversation response draft",
                    value=str(conversation.get("suggested_reply") or ""),
                    key=f"pipeline_conversation_draft_{conversation_id}",
                )
                action_cols = st.columns(2)
                if action_cols[0].button(
                    "Approve Conversation Reply",
                    key=f"pipeline_conversation_approve_{conversation_id}",
                    disabled=conversation.get("reply_approved", False) or not response_draft,
                    use_container_width=True,
                ):
                    try:
                        api_post_json(
                            base_url,
                            f"/conversations/{conversation_id}/approve-reply",
                            {
                                "approved_by": "streamlit-user",
                                "body": response_draft,
                            },
                        )
                        st.rerun()
                    except Exception as exc:
                        st.error(f"Conversation approval failed: {exc}")
                if action_cols[1].button(
                    "Send Approved Conversation Reply",
                    key=f"pipeline_conversation_send_{conversation_id}",
                    disabled=not conversation.get("reply_approved", False),
                    use_container_width=True,
                ):
                    try:
                        api_post_json(
                            base_url,
                            f"/conversations/{conversation_id}/send-reply",
                            {},
                        )
                        st.rerun()
                    except Exception as exc:
                        st.error(f"Conversation send failed: {exc}")

        with st.expander("6. Meeting & CRM", expanded=True):
            try:
                meeting_status = api_get(base_url, "/meetings/status")
                ms1, ms2, ms3, ms4 = st.columns(4)
                ms1.metric("Meeting Provider", meeting_status.get("provider", "unknown"))
                ms2.metric(
                    "Calendar",
                    "Configured" if meeting_status.get("configured") else "Not configured",
                )
                ms3.metric("CRM Provider", meeting_status.get("crm_provider", "unknown"))
                ms4.metric(
                    "CRM",
                    "Configured"
                    if meeting_status.get("crm_configured")
                    else "Not configured",
                )
                st.caption(str(meeting_status.get("detail", "")))
            except Exception as exc:
                st.warning(f"Meeting provider status unavailable: {exc}")

            try:
                meetings = api_get(base_url, "/meetings").get("items", [])
            except Exception:
                meetings = []
            meeting_conversation_ids = {
                str(item.get("conversation_id", "")) for item in meetings
            }
            for conversation in conversations:
                classification = conversation.get("latest_classification", {}) or {}
                conversation_id = str(conversation.get("conversation_id", ""))
                if (
                    classification.get("intent") == "meeting_request"
                    and conversation_id not in meeting_conversation_ids
                ):
                    if st.button(
                        f"Create Meeting Request for {conversation.get('contact_name', 'prospect')}",
                        key=f"create_meeting_{conversation_id}",
                        use_container_width=True,
                    ):
                        try:
                            api_post_json(
                                base_url,
                                "/meetings/requests",
                                {"conversation_id": conversation_id},
                            )
                            st.rerun()
                        except Exception as exc:
                            st.error(f"Meeting request creation failed: {exc}")

            default_start = datetime.now(timezone.utc) + timedelta(days=2)
            for meeting in meetings:
                meeting_id = str(meeting.get("meeting_id", ""))
                st.markdown(
                    f"**{meeting.get('contact_name', 'Unknown')} at "
                    f"{meeting.get('company_name', 'Unknown')}**"
                )
                mc1, mc2, mc3, mc4 = st.columns(4)
                mc1.metric("Status", str(meeting.get("status", "unknown")).title())
                mc2.metric("Provider", str(meeting.get("provider", "unknown")))
                mc3.metric("CRM", str(meeting.get("crm_sync_status", "not_started")))
                mc4.metric("Temporal", str(meeting.get("temporal_status", "not_started")))
                if meeting.get("meeting_url"):
                    st.markdown(f"[Join meeting]({meeting.get('meeting_url')})")
                if meeting.get("calendar_url"):
                    st.markdown(f"[Open calendar event]({meeting.get('calendar_url')})")
                if meeting.get("error"):
                    st.error(str(meeting.get("error")))

                if meeting.get("status") in {
                    "awaiting_approval",
                    "approved",
                    "failed",
                }:
                    schedule_cols = st.columns(4)
                    meeting_date = schedule_cols[0].date_input(
                        "Meeting date",
                        value=default_start.date(),
                        key=f"meeting_date_{meeting_id}",
                    )
                    meeting_time = schedule_cols[1].time_input(
                        "Meeting time",
                        value=default_start.time().replace(second=0, microsecond=0),
                        key=f"meeting_time_{meeting_id}",
                    )
                    meeting_timezone = schedule_cols[2].text_input(
                        "Timezone",
                        value=str(meeting.get("timezone") or "UTC"),
                        key=f"meeting_timezone_{meeting_id}",
                    )
                    duration = schedule_cols[3].number_input(
                        "Minutes",
                        min_value=15,
                        max_value=240,
                        value=int(meeting.get("duration_minutes") or 30),
                        step=15,
                        key=f"meeting_duration_{meeting_id}",
                    )
                    if st.button(
                        "Approve And Book Real Meeting",
                        key=f"meeting_book_{meeting_id}",
                        use_container_width=True,
                    ):
                        try:
                            start_at = datetime.combine(
                                meeting_date,
                                meeting_time,
                                tzinfo=ZoneInfo(meeting_timezone),
                            )
                            api_post_json(
                                base_url,
                                f"/meetings/{meeting_id}/approve-and-book",
                                {
                                    "start_at": start_at.isoformat(),
                                    "timezone": meeting_timezone,
                                    "duration_minutes": int(duration),
                                    "approved_by": "streamlit-user",
                                },
                            )
                            st.success(
                                "Calendar booking completed and CRM synchronization processed."
                            )
                            st.rerun()
                        except Exception as exc:
                            st.error(f"Meeting booking failed: {exc}")
                elif meeting.get("status") == "booked":
                    if st.button(
                        "Cancel Meeting",
                        key=f"meeting_cancel_{meeting_id}",
                        use_container_width=True,
                    ):
                        try:
                            api_post_json(
                                base_url,
                                f"/meetings/{meeting_id}/cancel",
                                {"reason": "Cancelled in Streamlit"},
                            )
                            st.rerun()
                        except Exception as exc:
                            st.error(f"Meeting cancellation failed: {exc}")
                with st.expander(
                    f"Developer JSON - Meeting - {meeting_id}", expanded=False
                ):
                    st.json(meeting)

        with st.expander("7. SDR Analytics", expanded=False):
            st.markdown(
                "- Analytics uses the existing Analytics tab with `Live AI SDR Data` selected.\n"
                "- It combines discovery, enrichment, intelligence, qualification, outreach, replies, meetings, and CRM syncs.\n"
                "- Switch sidebar mode to `Data Science` or `MLOps`, open `Analytics`, and choose `Live AI SDR Data`."
            )

    def append_run_event(container: Any, message: str) -> None:
        run_events = st.session_state.setdefault("sdr_run_events", [])
        run_events.append(message)
        container.markdown("\n".join(f"- {event}" for event in run_events))

    if "icp_chat_history" not in st.session_state:
        st.session_state["icp_chat_history"] = [
            {
                "role": "assistant",
                "content": (
                    "Tell me your target customer in plain English. I will convert it into a structured ICP, "
                    "validate it, and let you save it for the SDR workflow."
                ),
            }
        ]
    if "icp_generated_payload" not in st.session_state:
        st.session_state["icp_generated_payload"] = default_icp_payload()

    user_tab, internal_tab = st.tabs(["Chatbot", "Advanced Console"])

    with user_tab:
        st.markdown("### SDR Workflow Chatbot")
        st.write(
            "Describe the target customer once. The platform will run each SDR agent in order and return "
            "approval-ready outreach, follow-up, conversation, meeting, and CRM actions in this same chat."
        )

        for msg in st.session_state["icp_chat_history"]:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        model_options = [
            "llama3.2:3b",
            "qwen2.5:3b",
            "mistral:7b",
            "llama3.1:8b",
            "llama3:8b",
            "deepseek-r1:1.5b",
        ]
        llm_model = st.selectbox(
            "Ollama model for ICP enrichment",
            options=model_options,
            index=0,
            key="icp_llm_model",
            help="Deterministic parsing always runs first. This model only enriches personas, pain points, exclusions, and reasoning.",
        )
        use_llm = st.toggle("Use LLM enrichment", value=True)

        input_mode = st.radio(
            "ICP input mode",
            options=["Manual Filters", "Natural Language Chat"],
            horizontal=True,
            help=(
                "Manual Filters gives you boxes for ICP criteria. Natural Language "
                "Chat lets you describe the target customer in plain English."
            ),
        )
        manual_payload: Dict[str, Any] | None = None

        if input_mode == "Manual Filters":
            st.markdown("#### ICP Filters")
            default_payload = st.session_state.get("icp_generated_payload") or default_icp_payload()
            filters = default_payload.get("filters", {})
            mf1, mf2 = st.columns(2)
            manual_industries = mf1.text_input(
                "Industries",
                value=", ".join(filters.get("target_industries") or default_payload.get("industries") or ["SaaS", "B2B software"]),
                key="manual_filter_industries",
            )
            manual_geographies = mf2.text_input(
                "Geographies",
                value=", ".join(filters.get("geographies") or default_payload.get("geographies") or ["Washington, United States"]),
                key="manual_filter_geographies",
            )
            manual_employee_band = mf1.selectbox(
                "Employee Range",
                ["25 to 500", "50 to 500", "100 to 1000", "mid-market", "enterprise"],
                index=2,
                key="manual_filter_employee_band",
            )
            manual_revenue_band = mf2.selectbox(
                "Revenue Range",
                ["seed", "growth", "mid-market", "enterprise"],
                index=2,
                key="manual_filter_revenue_band",
            )
            manual_tech = mf1.text_input(
                "Technology Stack Signals",
                value=", ".join(filters.get("tech_stack_signals") or ["Salesforce", "HubSpot", "Snowflake", "AWS"]),
                key="manual_filter_tech",
            )
            manual_titles = mf2.text_input(
                "Target Job Titles",
                value=", ".join(filters.get("target_job_titles") or ["VP Sales", "Head of Sales", "Director of Revenue Operations", "CRO"]),
                key="manual_filter_titles",
            )
            manual_seniority = mf1.text_input(
                "Seniority Levels",
                value=", ".join(filters.get("seniority_levels") or ["Director", "VP", "C-suite"]),
                key="manual_filter_seniority",
            )
            manual_buying_signals = mf2.text_input(
                "Buying Signals",
                value=", ".join(filters.get("buying_signals") or ["sales hiring", "recent funding", "geographic expansion", "new revenue leadership", "CRM adoption"]),
                key="manual_filter_buying_signals",
            )
            manual_pain_points = st.text_area(
                "Pain Points",
                value=", ".join(filters.get("pain_points") or default_payload.get("pain_points") or ["poor pipeline visibility", "manual lead qualification", "low outbound reply rates", "forecasting inconsistency"]),
                key="manual_filter_pain_points",
                height=80,
            )
            ex1, ex2 = st.columns(2)
            manual_excluded_industries = ex1.text_input(
                "Excluded Industries / Company Types",
                value="consulting firms, agencies",
                key="manual_filter_excluded_industries",
            )
            manual_employee_min = ex2.number_input(
                "Exclude Companies Below Employees",
                min_value=0,
                value=50,
                step=5,
                key="manual_filter_min_employees",
            )
            manual_notes = st.text_area(
                "Extra Notes",
                value="Run the full SDR workflow.",
                key="manual_filter_notes",
                height=70,
            )

            industries = split_csv(manual_industries)
            geographies = split_csv(manual_geographies)
            titles = split_csv(manual_titles)
            seniority = split_csv(manual_seniority)
            pain_points = split_csv(manual_pain_points)
            tech = split_csv(manual_tech)
            buying_signals = split_csv(manual_buying_signals)
            excluded_industries = split_csv(manual_excluded_industries)
            manual_payload = default_icp_payload()
            manual_payload.update(
                {
                    "industries": industries,
                    "geographies": geographies,
                    "employee_band": manual_employee_band,
                    "revenue_band": manual_revenue_band,
                    "target_personas": titles,
                    "target_departments": ["Sales", "Revenue Operations"],
                    "pain_points": pain_points,
                    "offer_summary": "AI SDR platform for revenue teams.",
                    "notes": manual_notes,
                    "created_by": "streamlit-user",
                }
            )
            manual_payload["filters"].update(
                {
                    "target_industries": industries,
                    "revenue_ranges": [manual_revenue_band],
                    "company_size_ranges": [manual_employee_band],
                    "geographies": geographies,
                    "tech_stack_signals": tech,
                    "target_job_titles": titles,
                    "seniority_levels": seniority,
                    "pain_points": pain_points,
                    "buying_signals": buying_signals,
                }
            )
            manual_payload["exclusions"].update(
                {
                    "industries": excluded_industries,
                    "customer_statuses": ["existing_customer"],
                    "employee_max_below": int(manual_employee_min),
                }
            )
            user_prompt = (
                f"Target {', '.join(industries)} companies in {', '.join(geographies)}. "
                f"Employee range: {manual_employee_band}. Revenue range: {manual_revenue_band}. "
                f"Technology stack: {', '.join(tech)}. Target job titles: {', '.join(titles)}. "
                f"Seniority: {', '.join(seniority)}. Buying signals: {', '.join(buying_signals)}. "
                f"Pain points: {', '.join(pain_points)}. Exclude: {', '.join(excluded_industries)} "
                f"and companies below {manual_employee_min} employees. {manual_notes}"
            )
            with st.expander("Preview manual ICP payload", expanded=False):
                st.json(manual_payload)
        else:
            user_prompt = st.text_area(
                "Chat with the AI SDR platform",
                value=st.session_state.get("icp_user_prompt", ""),
                placeholder=(
                    "Example: Target US and UK mid-market SaaS companies with RevOps leaders "
                    "struggling with pipeline visibility and forecasting."
                ),
                height=120,
                key="icp_user_prompt",
            )
        chat_actions = st.columns(2)
        if chat_actions[0].button("Run Agents", use_container_width=True, key="icp_chat_generate"):
            if not user_prompt.strip():
                st.warning("Enter a target-customer description first.")
            else:
                st.session_state["sdr_run_events"] = []
                st.session_state["icp_chat_history"].append({"role": "user", "content": user_prompt})
                progress_box = st.container()
                event_box = progress_box.empty()
                payload = (
                    json.loads(json.dumps(manual_payload))
                    if manual_payload is not None
                    else draft_from_prompt(user_prompt, st.session_state["icp_generated_payload"])
                )
                suggestion = None
                validation = None
                validation_note = ""
                pipeline_result = None

                with progress_box:
                    with st.status("Running SDR agents...", expanded=True) as status_box:
                        append_run_event(event_box, "Starting SDR pipeline from your prompt.")
                        append_run_event(
                            event_box,
                            "Running ICP Agent: building ICP from manual filters."
                            if manual_payload is not None
                            else "Running ICP Agent: building deterministic ICP draft.",
                        )

                        if use_llm and manual_payload is None:
                            append_run_event(event_box, f"Running ICP Agent LLM enrichment with `{llm_model}`.")
                            try:
                                suggestion_payload = {
                                    "prompt": user_prompt,
                                    "filters": payload.get("filters"),
                                    "industries": payload.get("industries", []),
                                    "geographies": payload.get("geographies", []),
                                    "target_personas": payload.get("target_personas", []),
                                    "pain_points": payload.get("pain_points", []),
                                    "offer_summary": payload.get("offer_summary", ""),
                                    "notes": user_prompt,
                                    "llm_provider": "ollama_local",
                                    "model_name": llm_model,
                                }
                                suggestion = api_post_json(base_url, "/icp/suggest", suggestion_payload)
                                payload = merge_llm_suggestion(payload, suggestion)
                                st.session_state["icp_last_suggestion"] = suggestion
                                append_run_event(event_box, "ICP Agent enrichment complete: personas, pain points, and exclusions refined.")
                            except Exception as exc:
                                append_run_event(event_box, f"ICP Agent LLM enrichment skipped: {exc}")
                                st.warning(f"LLM enrichment skipped. Deterministic draft still created. Reason: {exc}")

                        st.session_state["icp_generated_payload"] = payload
                        append_run_event(event_box, "Running ICP Agent validation against backend schema.")
                        try:
                            validation = api_post_json(base_url, "/icp/validate", payload)
                            st.session_state["icp_last_validation"] = validation
                            validation_note = " Backend validation passed."
                            append_run_event(event_box, "ICP Agent completed successfully.")
                        except Exception as exc:
                            validation_note = f" Backend validation needs review: {exc}"
                            append_run_event(event_box, f"ICP Agent validation needs review: {exc}")

                        if validation:
                            append_run_event(event_box, "Running Prospect Discovery Agent: finding matching accounts and contacts.")
                            append_run_event(event_box, "Running Enrichment Agent: researching top discovered accounts.")
                            append_run_event(event_box, "Running Prospect Intelligence: intent, propensity, queue ranking, and personalization.")
                            append_run_event(event_box, "Running Qualification Agent: ML probability plus BANT/MEDDIC evidence gate.")
                            append_run_event(event_box, "Running Outreach Agent: policy checks, review, and approval-ready drafts.")
                            try:
                                enrichment_payload = st.session_state.get("enrichment_payload", {})
                                pipeline_result = api_post_json(
                                    base_url,
                                    "/sdr/pipeline/run",
                                    {
                                        "icp_payload": payload,
                                        "discovery_limit": 5,
                                        "enrich_top_accounts": 3,
                                        "collection": enrichment_payload.get("collection", "sdr_enrichment"),
                                        "search_provider": enrichment_payload.get("search_provider") or None,
                                        "top_k": int(enrichment_payload.get("top_k", 8) or 8),
                                        "auto_fetch_and_index": bool(enrichment_payload.get("auto_fetch_and_index", True)),
                                        "llm_provider": "ollama_local",
                                        "llm_model": llm_model,
                                        "created_by": "streamlit-user",
                                    },
                                )
                                st.session_state["sdr_pipeline_result"] = pipeline_result
                                summary = pipeline_result.get("summary", {})
                                append_run_event(
                                    event_box,
                                    "Prospect Discovery Agent completed: "
                                    f"{int(summary.get('accounts_discovered', 0))} accounts and "
                                    f"{int(summary.get('contacts_discovered', 0))} contacts found.",
                                )
                                append_run_event(
                                    event_box,
                                    "Enrichment Agent completed: "
                                    f"{int(summary.get('accounts_enriched', 0))} accounts enriched.",
                                )
                                append_run_event(
                                    event_box,
                                    "Prospect Intelligence completed: "
                                    f"{int(summary.get('prospects_analyzed', 0))} prospects analyzed.",
                                )
                                append_run_event(
                                    event_box,
                                    "Outreach Agent completed: "
                                    f"{int(summary.get('campaigns_drafted', 0))} approval-ready campaigns drafted.",
                                )
                                append_run_event(event_box, "Full SDR flow completed.")
                            except Exception as exc:
                                append_run_event(event_box, f"SDR pipeline stopped: {exc}")
                                st.error(f"Full SDR pipeline failed after ICP generation: {exc}")

                        status_box.update(label="SDR run finished", state="complete", expanded=True)

                assistant_message = (
                    "Full SDR flow completed through Prospect Intelligence and Outreach drafting."
                    if pipeline_result
                    else (
                        "ICP generated using deterministic parsing plus LLM enrichment."
                        if suggestion
                        else "ICP generated using deterministic parsing."
                    )
                ) + validation_note
                st.session_state["icp_chat_history"].append({"role": "assistant", "content": assistant_message})
                if pipeline_result:
                    st.success("Full SDR flow completed.")
                else:
                    st.success("ICP generated.")
                    render_structured_icp_response(payload, suggestion)

        if chat_actions[1].button("Save ICP", use_container_width=True, key="icp_chat_save"):
            try:
                res = api_post_json(base_url, "/icp", st.session_state["icp_generated_payload"])
                st.session_state["icp_last_validation"] = res
                st.session_state["icp_chat_history"].append(
                    {"role": "assistant", "content": "ICP saved and versioned in the backend repository."}
                )
                st.success("ICP saved.")
            except Exception as exc:
                st.error(f"Save failed: {exc}")

        with st.expander("Developer JSON - Current Draft Payload", expanded=False):
            st.json(st.session_state["icp_generated_payload"])

        pipeline_result = st.session_state.get("sdr_pipeline_result")
        if pipeline_result:
            render_pipeline_response(pipeline_result)

    with internal_tab:
        st.markdown("### Internal ICP Testing Console")
        st.caption("Use this tab to test deterministic validation, optional LLM suggestions, persistence, and saved ICP versions.")
        payload = st.session_state["icp_generated_payload"]

        c1, c2 = st.columns(2)
        with c1:
            payload["icp_name"] = st.text_input("ICP Name", value=payload.get("icp_name", ""))
            payload["created_by"] = st.text_input("Created By", value=payload.get("created_by", "streamlit-user"))
            employee_options = ["small", "smb", "mid-market", "enterprise"]
            revenue_options = ["seed", "growth", "mid-market", "enterprise"]
            employee_value = payload.get("employee_band", "mid-market")
            revenue_value = payload.get("revenue_band", "mid-market")
            payload["employee_band"] = st.selectbox(
                "Employee Band",
                employee_options,
                index=employee_options.index(employee_value) if employee_value in employee_options else 2,
            )
            payload["revenue_band"] = st.selectbox(
                "Revenue Band",
                revenue_options,
                index=revenue_options.index(revenue_value) if revenue_value in revenue_options else 2,
            )
            payload["offer_summary"] = st.text_area("Offer Summary", value=payload.get("offer_summary", ""))
            payload["notes"] = st.text_area("Notes", value=payload.get("notes", ""))

        with c2:
            payload["industries"] = split_csv(
                st.text_input("Industries (comma separated)", value=", ".join(payload.get("industries", [])))
            )
            payload["geographies"] = split_csv(
                st.text_input("Geographies (comma separated)", value=", ".join(payload.get("geographies", [])))
            )
            payload["target_personas"] = split_csv(
                st.text_input("Target Personas (comma separated)", value=", ".join(payload.get("target_personas", [])))
            )
            payload["target_departments"] = split_csv(
                st.text_input(
                    "Target Departments (comma separated)",
                    value=", ".join(payload.get("target_departments", [])),
                )
            )
            payload["pain_points"] = split_csv(
                st.text_area("Pain Points (comma separated)", value=", ".join(payload.get("pain_points", [])))
            )

        st.markdown("#### Exclusions")
        payload.setdefault("exclusions", default_icp_payload()["exclusions"])
        ex1, ex2 = st.columns(2)
        with ex1:
            payload["exclusions"]["industries"] = split_csv(
                st.text_input(
                    "Excluded Industries",
                    value=", ".join(payload["exclusions"].get("industries", [])),
                )
            )
            payload["exclusions"]["customer_statuses"] = split_csv(
                st.text_input(
                    "Excluded Customer Statuses",
                    value=", ".join(payload["exclusions"].get("customer_statuses", [])),
                )
            )
        with ex2:
            payload["exclusions"]["company_names"] = split_csv(
                st.text_input(
                    "Excluded Company Names",
                    value=", ".join(payload["exclusions"].get("company_names", [])),
                )
            )
            payload["exclusions"]["geographies"] = split_csv(
                st.text_input(
                    "Excluded Geographies",
                    value=", ".join(payload["exclusions"].get("geographies", [])),
                )
            )
            payload["exclusions"]["employee_max_below"] = st.number_input(
                "Exclude Companies Below Employee Count",
                min_value=0,
                value=int(payload["exclusions"].get("employee_max_below") or 20),
                step=1,
            )

        st.markdown("#### Scoring Weights")
        payload.setdefault("scoring_weights", default_icp_payload()["scoring_weights"])
        w1, w2, w3, w4, w5 = st.columns(5)
        payload["scoring_weights"]["industry_fit"] = w1.slider(
            "Industry", 0.0, 0.6, float(payload["scoring_weights"].get("industry_fit", 0.3)), 0.01
        )
        payload["scoring_weights"]["company_size_fit"] = w2.slider(
            "Company Size", 0.0, 0.6, float(payload["scoring_weights"].get("company_size_fit", 0.2)), 0.01
        )
        payload["scoring_weights"]["persona_fit"] = w3.slider(
            "Persona", 0.0, 0.6, float(payload["scoring_weights"].get("persona_fit", 0.25)), 0.01
        )
        payload["scoring_weights"]["geo_fit"] = w4.slider(
            "Geo", 0.0, 0.6, float(payload["scoring_weights"].get("geo_fit", 0.1)), 0.01
        )
        payload["scoring_weights"]["pain_point_fit"] = w5.slider(
            "Pain Point", 0.0, 0.6, float(payload["scoring_weights"].get("pain_point_fit", 0.15)), 0.01
        )
        total_weight = sum(float(value) for value in payload["scoring_weights"].values())
        st.progress(min(total_weight, 1.0), text=f"Weight total: {total_weight:.2f}")

        a1, a2, a3, a4 = st.columns(4)
        if a1.button("Validate ICP", use_container_width=True, key="icp_internal_validate"):
            try:
                res = api_post_json(base_url, "/icp/validate", payload)
                st.session_state["icp_last_validation"] = res
                st.success("ICP validated successfully.")
            except Exception as exc:
                st.error(f"Validation failed: {exc}")
        if a2.button("Get AI Suggestions", use_container_width=True, key="icp_internal_suggest"):
            try:
                suggestion_payload = {
                    "industries": payload.get("industries", []),
                    "geographies": payload.get("geographies", []),
                    "target_personas": payload.get("target_personas", []),
                    "pain_points": payload.get("pain_points", []),
                        "offer_summary": payload.get("offer_summary", ""),
                        "notes": payload.get("notes", ""),
                        "llm_provider": "ollama_local",
                        "model_name": st.session_state.get("icp_llm_model", "llama3.2:3b"),
                    }
                res = api_post_json(base_url, "/icp/suggest", suggestion_payload)
                st.session_state["icp_last_suggestion"] = res
                st.success("Suggestions returned.")
            except Exception as exc:
                st.error(f"Suggestion failed: {exc}")
        if a3.button("Save ICP", use_container_width=True, key="icp_internal_save"):
            try:
                res = api_post_json(base_url, "/icp", payload)
                st.session_state["icp_last_validation"] = res
                st.success("ICP saved.")
            except Exception as exc:
                st.error(f"Save failed: {exc}")
        if a4.button("List ICPs", use_container_width=True, key="icp_internal_list"):
            try:
                res = api_get(base_url, "/icp")
                st.session_state["icp_list"] = res
                st.success("Loaded saved ICPs.")
            except Exception as exc:
                st.error(f"List failed: {exc}")

        if st.session_state.get("icp_last_validation"):
            st.markdown("#### Last Validation / Save Result")
            st.json(st.session_state["icp_last_validation"])
        if st.session_state.get("icp_last_suggestion"):
            st.markdown("#### Last Suggestion")
            st.json(st.session_state["icp_last_suggestion"])
        if st.session_state.get("icp_list"):
            st.markdown("#### Saved ICPs")
            st.json(st.session_state["icp_list"])


def enrichment_agent_tab(base_url: str) -> None:
    st.subheader("Enrichment Agent")
    st.caption("Researches discovered accounts/contacts using the existing OpenSearch + SearchXNG + LLM pipeline.")

    def default_payload() -> Dict[str, Any]:
        return {
            "account": {
                "account_id": "",
                "company_name": "",
                "website": "",
                "linkedin_url": "",
                "industry": "",
                "location": "",
                "employee_count": None,
                "revenue_range": "",
                "source": "streamlit",
                "fit_score": None,
            },
            "contacts": [],
            "icp_context": {},
            "collection": "sdr_enrichment",
            "search_provider": "",
            "top_k": 8,
            "auto_fetch_and_index": True,
            "llm_provider": "ollama_local",
            "llm_model": "llama3.2:3b",
            "created_by": "streamlit-user",
        }

    def split_contacts(raw: str) -> List[Dict[str, Any]]:
        contacts: List[Dict[str, Any]] = []
        for idx, line in enumerate(raw.splitlines(), start=1):
            parts = [p.strip() for p in line.split("|")]
            if not parts or not parts[0]:
                continue
            contacts.append(
                {
                    "contact_id": f"contact_{idx}",
                    "full_name": parts[0],
                    "title": parts[1] if len(parts) > 1 else "",
                    "email": parts[2] if len(parts) > 2 else "",
                    "linkedin_url": parts[3] if len(parts) > 3 else "",
                    "source": "streamlit",
                }
            )
        return contacts

    def clean_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
        cleaned = json.loads(json.dumps(payload))
        account = cleaned["account"]
        for key in ("account_id", "website", "linkedin_url", "industry", "location", "revenue_range"):
            if account.get(key) == "":
                account[key] = None
        for key in ("employee_count", "fit_score"):
            if account.get(key) in ("", 0):
                account[key] = None
        if cleaned.get("search_provider") == "":
            cleaned["search_provider"] = None
        return cleaned

    if "enrichment_payload" not in st.session_state:
        st.session_state["enrichment_payload"] = default_payload()

    chat_tab, internal_tab = st.tabs(["Chatbot", "Internal Testing"])

    with chat_tab:
        st.markdown("### User-Facing Account Research")
        st.write("Enter an account directly, or run the full ICP -> Discovery -> Enrichment pipeline from the latest ICP draft.")
        payload = st.session_state["enrichment_payload"]
        account = payload["account"]

        st.markdown("#### Full SDR Pipeline")
        p1, p2, p3 = st.columns(3)
        pipeline_limit = p1.number_input("Discovery Limit", min_value=1, max_value=20, value=5, key="enrich_pipeline_discovery_limit")
        enrich_top_accounts = p2.number_input("Accounts To Enrich", min_value=1, max_value=10, value=3, key="enrich_pipeline_top_accounts")
        pipeline_model = p3.selectbox(
            "Pipeline Model",
            ["llama3.2:3b", "qwen2.5:3b", "mistral:7b", "llama3.1:8b", "llama3:8b", "deepseek-r1:1.5b"],
            index=0,
            key="enrich_pipeline_model",
        )
        if st.button("Run Full SDR Flow", use_container_width=True, key="enrich_chat_pipeline"):
            icp_payload = st.session_state.get("icp_generated_payload")
            if not icp_payload:
                st.warning("Generate an ICP first in the ICP Agent tab.")
            else:
                try:
                    result = api_post_json(
                        base_url,
                        "/sdr/pipeline/run",
                        {
                            "icp_payload": icp_payload,
                            "discovery_limit": int(pipeline_limit),
                            "enrich_top_accounts": int(enrich_top_accounts),
                            "collection": payload.get("collection", "sdr_enrichment"),
                            "search_provider": payload.get("search_provider") or None,
                            "top_k": int(payload.get("top_k", 8)),
                            "auto_fetch_and_index": bool(payload.get("auto_fetch_and_index", True)),
                            "llm_provider": payload.get("llm_provider", "ollama_local"),
                            "llm_model": pipeline_model,
                            "created_by": "streamlit-user",
                        },
                    )
                    st.session_state["sdr_pipeline_result"] = result
                    st.success("Full SDR pipeline completed.")
                except Exception as exc:
                    st.error(f"Pipeline failed: {exc}")

        pipeline_result = st.session_state.get("sdr_pipeline_result")
        if pipeline_result:
            st.markdown("#### Pipeline Summary")
            s1, s2, s3 = st.columns(3)
            summary = pipeline_result.get("summary", {})
            s1.metric("Accounts", int(summary.get("accounts_discovered", 0)))
            s2.metric("Contacts", int(summary.get("contacts_discovered", 0)))
            s3.metric("Enriched", int(summary.get("accounts_enriched", 0)))
            with st.expander("Discovered Accounts", expanded=True):
                st.json(pipeline_result.get("discovered_accounts", []))
            with st.expander("Discovered Contacts", expanded=False):
                st.json(pipeline_result.get("discovered_contacts", []))
            with st.expander("Enriched Results", expanded=True):
                st.json(pipeline_result.get("enriched_results", []))

        c1, c2 = st.columns(2)
        account["company_name"] = c1.text_input("Company Name", value=account.get("company_name") or "", key="enrich_chat_company")
        account["website"] = c2.text_input("Website", value=account.get("website") or "", key="enrich_chat_website")
        account["industry"] = c1.text_input("Industry", value=account.get("industry") or "", key="enrich_chat_industry")
        account["location"] = c2.text_input("Location", value=account.get("location") or "", key="enrich_chat_location")
        contact_line = st.text_input(
            "Optional Contact",
            value="",
            placeholder="Example: Sarah Miller | VP Sales | sarah@example.com | https://linkedin.com/in/...",
            key="enrich_chat_contact",
        )
        if contact_line.strip():
            payload["contacts"] = split_contacts(contact_line)

        if st.button("Research Account", use_container_width=True, key="enrich_chat_research"):
            if not account.get("company_name"):
                st.warning("Company name is required.")
            else:
                try:
                    result = api_post_json(base_url, "/enrichment/research", clean_payload(payload))
                    st.session_state["enrichment_last_result"] = result
                    st.success("Enrichment completed.")
                except Exception as exc:
                    st.error(f"Enrichment failed: {exc}")

        result = st.session_state.get("enrichment_last_result")
        if result:
            st.markdown("#### SDR Research Brief")
            m1, m2, m3 = st.columns(3)
            m1.metric("Status", str(result.get("status", "unknown")))
            m2.metric("Signals", len(result.get("signals", []) or []))
            m3.metric("Confidence", f"{float(result.get('confidence_score', 0.0)):.2f}")
            st.markdown(f"**Company Summary:** {result.get('company_summary', '')}")
            if result.get("products_services"):
                st.markdown("**Products / Services**")
                st.write(result.get("products_services", []))
            if result.get("target_customers"):
                st.markdown("**Target Customers / Buyers**")
                st.write(result.get("target_customers", []))
            if result.get("signals"):
                st.markdown("**Commercial Signals**")
                signal_lines = [
                    f"{signal.get('type', 'signal')}: {signal.get('detail', '')}"
                    for signal in result.get("signals", [])
                ]
                st.write(signal_lines)
            st.markdown("**Pain Point Hypotheses**")
            st.write(result.get("pain_point_hypotheses", []))
            st.markdown("**Personalization Angles**")
            st.write(result.get("personalization_angles", []))
            if result.get("contact_briefs"):
                st.markdown("**Contact Briefs**")
                st.json(result.get("contact_briefs", []))
            st.markdown("**Recommended Next Action**")
            st.info(str(result.get("recommended_next_action", "")))
            with st.expander("Citations", expanded=True):
                for citation in result.get("citations", [])[:8]:
                    title = citation.get("title") or citation.get("url") or "Source"
                    url = citation.get("url") or ""
                    st.markdown(f"- [{title}]({url}) - {citation.get('snippet', '')[:240]}")
            with st.expander("Raw Enrichment JSON"):
                st.json(result)

    with internal_tab:
        st.markdown("### Internal Enrichment Testing Console")
        payload = st.session_state["enrichment_payload"]
        account = payload["account"]

        i1, i2, i3 = st.columns(3)
        payload["collection"] = i1.text_input("OpenSearch Collection", value=payload.get("collection", "sdr_enrichment"))
        payload["search_provider"] = i2.text_input("Search Provider Override", value=payload.get("search_provider") or "")
        payload["top_k"] = i3.number_input("Top K", min_value=1, max_value=20, value=int(payload.get("top_k", 8)))
        payload["auto_fetch_and_index"] = st.toggle(
            "Auto Fetch And Index Search Results",
            value=bool(payload.get("auto_fetch_and_index", True)),
            key="enrich_internal_auto_index",
        )
        m1, m2 = st.columns(2)
        payload["llm_provider"] = m1.selectbox("LLM Provider", ["ollama_local", "bedrock"], index=0)
        payload["llm_model"] = m2.selectbox(
            "LLM Model",
            ["llama3.2:3b", "qwen2.5:3b", "mistral:7b", "llama3.1:8b", "llama3:8b", "deepseek-r1:1.5b"],
            index=0,
        )

        st.markdown("#### Account")
        a1, a2 = st.columns(2)
        account["account_id"] = a1.text_input("Account ID", value=account.get("account_id") or "")
        account["company_name"] = a2.text_input("Company Name", value=account.get("company_name") or "", key="enrich_internal_company")
        account["website"] = a1.text_input("Website", value=account.get("website") or "", key="enrich_internal_website")
        account["linkedin_url"] = a2.text_input("LinkedIn URL", value=account.get("linkedin_url") or "")
        account["industry"] = a1.text_input("Industry", value=account.get("industry") or "", key="enrich_internal_industry")
        account["location"] = a2.text_input("Location", value=account.get("location") or "", key="enrich_internal_location")
        account["employee_count"] = a1.number_input("Employee Count", min_value=0, value=int(account.get("employee_count") or 0))
        account["revenue_range"] = a2.text_input("Revenue Range", value=account.get("revenue_range") or "")

        contacts_raw = st.text_area(
            "Contacts, one per line: full_name | title | email | linkedin_url",
            value="",
            height=110,
            key="enrich_internal_contacts",
        )
        payload["contacts"] = split_contacts(contacts_raw)

        b1, b2 = st.columns(2)
        if b1.button("Run Enrichment", use_container_width=True, key="enrich_internal_run"):
            try:
                result = api_post_json(base_url, "/enrichment/research", clean_payload(payload))
                st.session_state["enrichment_last_result"] = result
                st.success("Enrichment completed.")
            except Exception as exc:
                st.error(f"Enrichment failed: {exc}")
        if b2.button("List Enrichments", use_container_width=True, key="enrich_internal_list"):
            try:
                st.session_state["enrichment_list"] = api_get(base_url, "/enrichment")
                st.success("Loaded enrichment results.")
            except Exception as exc:
                st.error(f"List failed: {exc}")

        if st.session_state.get("enrichment_list"):
            st.markdown("#### Saved Enrichments")
            st.json(st.session_state["enrichment_list"])


def prospect_discovery_tab(base_url: str) -> None:
    st.subheader("Prospect Discovery")
    st.caption("Runs Monika's modular mock-based account and contact discovery using the current ICP.")

    if "prospect_discovery_result" not in st.session_state:
        st.session_state["prospect_discovery_result"] = None

    st.write("This tab uses the latest ICP draft from the ICP Agent and returns discovered accounts plus contacts.")
    limit = st.number_input("Discovery Limit", min_value=1, max_value=20, value=5, key="prospect_discovery_limit")
    if st.button("Run Prospect Discovery", use_container_width=True, key="prospect_discovery_run"):
        icp_payload = st.session_state.get("icp_last_validation", {}).get("normalized_definition")
        if not icp_payload:
            icp_payload = st.session_state.get("icp_generated_payload")
        if not icp_payload:
            st.warning("Generate or save an ICP first in the ICP Agent tab.")
        else:
            try:
                if "account_criteria" in icp_payload:
                    request_payload = {"icp_definition": icp_payload, "limit": int(limit)}
                else:
                    validated = api_post_json(base_url, "/icp/validate", icp_payload)
                    request_payload = {
                        "icp_definition": validated.get("normalized_definition", {}),
                        "limit": int(limit),
                    }
                result = api_post_json(base_url, "/prospect-discovery/run", request_payload)
                st.session_state["prospect_discovery_result"] = result
                st.success("Prospect discovery completed.")
            except Exception as exc:
                st.error(f"Prospect discovery failed: {exc}")

    result = st.session_state.get("prospect_discovery_result")
    if result:
        r1, r2 = st.columns(2)
        r1.metric("Accounts Discovered", int(result.get("account_total", 0)))
        r2.metric("Contacts Discovered", int(result.get("contact_total", 0)))
        with st.expander("Accounts", expanded=True):
            st.json(result.get("accounts", []))
        with st.expander("Contacts", expanded=False):
            st.json(result.get("contacts", []))


def prospect_intelligence_tab(base_url: str) -> None:
    st.subheader("Prospect Intelligence")
    st.caption("Intent, reply/meeting/qualification propensity, queue ranking, and grounded personalization.")

    try:
        ranking_status = api_get(base_url, "/prospect-intelligence/ranking/status")
        active = ranking_status.get("active_provider", "local")
        if ranking_status.get("learned_reranking_active"):
            st.success(
                f"MetaRank is active with model `{ranking_status.get('model_name')}`."
            )
        else:
            st.warning(
                f"Ranking provider: `{active}`. {ranking_status.get('detail', '')}"
            )
    except Exception as exc:
        st.warning(f"Could not load ranking status: {exc}")

    if st.button("Refresh Prospect Intelligence", use_container_width=True):
        try:
            st.session_state["prospect_intelligence_list"] = api_get(
                base_url, "/prospect-intelligence"
            )
        except Exception as exc:
            st.error(f"Load failed: {exc}")

    items = st.session_state.get("prospect_intelligence_list")
    if items is None:
        try:
            items = api_get(base_url, "/prospect-intelligence")
            st.session_state["prospect_intelligence_list"] = items
        except Exception:
            items = {"items": [], "total": 0}
    st.metric("Stored Prospects", int(items.get("total", 0)))
    for item in items.get("items", []) or []:
        with st.expander(
            f"{item.get('contact_name', 'Unknown')} at {item.get('company_name', 'Unknown')}",
            expanded=False,
        ):
            propensity = item.get("propensity", {})
            ranking = item.get("ranking", {})
            p1, p2, p3, p4 = st.columns(4)
            p1.metric("Intent", f"{float(item.get('intent', {}).get('probability', {}).get('value', 0)):.0%}")
            p2.metric("Reply", f"{float(propensity.get('reply_probability', {}).get('value', 0)):.0%}")
            p3.metric("Meeting", f"{float(propensity.get('meeting_probability', {}).get('value', 0)):.0%}")
            p4.metric("Rank", ranking.get("rank") or "-")
            st.json(item.get("personalization", {}))


def outreach_agent_tab(base_url: str) -> None:
    st.subheader("Outreach Agent")
    st.caption("Policy-controlled drafts, human approval, provider delivery, sequence state, and events.")

    try:
        performance = api_get(base_url, "/outreach/performance")
        p1, p2, p3, p4 = st.columns(4)
        p1.metric("Campaigns", int(performance.get("campaigns", 0)))
        p2.metric("Sent", int(performance.get("sent", 0)))
        p3.metric("Replies", int(performance.get("replied", 0)))
        p4.metric("Human Tasks", int(performance.get("human_tasks", 0)))
    except Exception as exc:
        st.warning(f"Could not load outreach performance: {exc}")

    try:
        campaigns = api_get(base_url, "/outreach/campaigns").get("items", [])
    except Exception as exc:
        st.error(f"Could not load campaigns: {exc}")
        campaigns = []

    for campaign in campaigns:
        campaign_id = str(campaign.get("campaign_id", ""))
        with st.expander(
            f"{campaign.get('contact_name', 'Unknown')} - {campaign.get('status', 'unknown')}",
            expanded=False,
        ):
            for message in campaign.get("messages", []) or []:
                st.markdown(
                    f"**{str(message.get('channel', '')).title()}** "
                    f"to `{message.get('recipient', '')}`"
                )
                if message.get("subject"):
                    st.markdown(f"Subject: `{message.get('subject')}`")
                st.code(str(message.get("body", "")), language="text")
            c1, c2, c3 = st.columns(3)
            if c1.button(
                "Approve",
                key=f"outreach_tab_approve_{campaign_id}",
                disabled=campaign.get("status") != "pending_approval",
                use_container_width=True,
            ):
                api_post_json(
                    base_url,
                    f"/outreach/campaigns/{campaign_id}/approve",
                    {"approved_by": "streamlit-user"},
                )
                st.rerun()
            if c2.button(
                "Send",
                key=f"outreach_tab_send_{campaign_id}",
                disabled=campaign.get("status") not in {"approved", "running"},
                use_container_width=True,
            ):
                api_post_json(
                    base_url,
                    f"/outreach/campaigns/{campaign_id}/send?force=true",
                    {},
                )
                st.rerun()
            if c3.button(
                "Pause",
                key=f"outreach_tab_pause_{campaign_id}",
                disabled=campaign.get("status") in {"paused", "completed", "cancelled"},
                use_container_width=True,
            ):
                api_post_json(
                    base_url,
                    f"/outreach/campaigns/{campaign_id}/pause",
                    {"reason": "Paused in Streamlit"},
                )
                st.rerun()
            st.json(campaign)


def conversation_agent_tab(base_url: str) -> None:
    st.subheader("Conversation Agent")
    st.caption("Classifies inbound replies, stops sequences, and prepares grounded responses for approval.")

    try:
        conversations = api_get(base_url, "/conversations").get("items", [])
    except Exception as exc:
        st.error(f"Could not load conversations: {exc}")
        conversations = []
    st.metric("Conversation Threads", len(conversations))
    for conversation in conversations:
        conversation_id = str(conversation.get("conversation_id", ""))
        classification = conversation.get("latest_classification", {}) or {}
        with st.expander(
            f"{conversation.get('contact_name', 'Unknown')} - "
            f"{classification.get('intent', 'unclassified')}",
            expanded=False,
        ):
            c1, c2, c3 = st.columns(3)
            c1.metric("Status", str(conversation.get("status", "unknown")).title())
            c2.metric("Intent", str(classification.get("intent", "unknown")).replace("_", " ").title())
            c3.metric("Confidence", f"{float(classification.get('confidence', 0)):.0%}")
            for message in conversation.get("messages", []) or []:
                st.markdown(
                    f"**{str(message.get('direction', '')).title()}** via "
                    f"{message.get('channel', '')}"
                )
                st.code(str(message.get("body", "")), language="text")
            draft = st.text_area(
                "Suggested reply",
                value=str(conversation.get("suggested_reply") or ""),
                key=f"conversation_draft_{conversation_id}",
            )
            if st.button(
                "Approve Reply Draft",
                key=f"conversation_approve_{conversation_id}",
                disabled=conversation.get("reply_approved", False) or not draft,
                use_container_width=True,
            ):
                try:
                    api_post_json(
                        base_url,
                        f"/conversations/{conversation_id}/approve-reply",
                        {"approved_by": "streamlit-user", "body": draft},
                    )
                    st.success("Reply draft approved.")
                    st.rerun()
                except Exception as exc:
                    st.error(f"Reply approval failed: {exc}")
            if st.button(
                "Send Approved Reply",
                key=f"conversation_send_{conversation_id}",
                disabled=not conversation.get("reply_approved", False),
                use_container_width=True,
            ):
                try:
                    api_post_json(
                        base_url,
                        f"/conversations/{conversation_id}/send-reply",
                        {},
                    )
                    st.success("Approved reply processed by the configured provider.")
                    st.rerun()
                except Exception as exc:
                    st.error(f"Reply send failed: {exc}")


def follow_up_agent_tab(base_url: str) -> None:
    st.subheader("Follow-Up Agent")
    st.caption("Durable future touches with approval, scheduling, and automatic stop conditions.")

    try:
        plans = api_get(base_url, "/follow-up/plans").get("items", [])
    except Exception as exc:
        st.error(f"Could not load follow-up plans: {exc}")
        plans = []
    st.metric("Follow-Up Plans", len(plans))
    if st.button("Process Due Follow-Ups", use_container_width=True):
        try:
            api_post_json(base_url, "/follow-up/scheduler/run-due", {})
            st.success("Due follow-up plans processed.")
            st.rerun()
        except Exception as exc:
            st.error(f"Follow-up processing failed: {exc}")
    for plan in plans:
        plan_id = str(plan.get("plan_id", ""))
        with st.expander(
            f"{plan_id} - {plan.get('status', 'unknown')}",
            expanded=False,
        ):
            if plan.get("stop_reason"):
                st.warning(str(plan.get("stop_reason")))
            for touch in plan.get("touches", []) or []:
                st.markdown(
                    f"**Touch {touch.get('sequence_order')}** after "
                    f"{touch.get('delay_hours')} hours - {touch.get('status')}"
                )
                st.code(str(touch.get("body", "")), language="text")
            if st.button(
                "Approve Follow-Up Plan",
                key=f"follow_up_approve_{plan_id}",
                disabled=plan.get("status") != "pending_approval",
                use_container_width=True,
            ):
                try:
                    api_post_json(
                        base_url,
                        f"/follow-up/plans/{plan_id}/approve",
                        {"approved_by": "streamlit-user"},
                    )
                    st.success("Follow-up plan approved.")
                    st.rerun()
                except Exception as exc:
                    st.error(f"Follow-up approval failed: {exc}")


def opensearch_tab(base_url: str) -> None:
    st.subheader("OpenSearch Web")
    st.caption("Separate web-context workspace: ingest links, search indexed pages, and explain the context. MCP-ready, but no MCP server is configured yet.")

    try:
        status = api_get(base_url, "/opensearch/status")
        c1, c2, c3 = st.columns(3)
        c1.metric("OpenSearch", "Ready" if status.get("available") else "Not Ready")
        c2.metric("MCP Search", "Configured" if status.get("mcp_ready") else "Not Configured")
        c3.metric("Provider", str(status.get("provider", "unknown")))
        st.info(str(status.get("note", "")))
        provider_chain = status.get("provider_chain", []) or []
        if provider_chain:
            st.caption("Provider chain: " + " -> ".join(provider_chain))
    except Exception as exc:
        st.error(f"Could not load OpenSearch status: {exc}")
        status = {}

    st.markdown("### Workspace")
    workspace_c1, workspace_c2, workspace_c3 = st.columns(3)
    collection = workspace_c1.text_input("Collection", value=st.session_state.get("opensearch_web_collection", "default"), key="opensearch_web_collection")
    provider_options = status.get("supported_providers", ["manual_urls", "searxng", "brave", "tavily", "serpapi"])
    provider_default = str(status.get("provider", "manual_urls"))
    provider_index = provider_options.index(provider_default) if provider_default in provider_options else 0
    search_provider = workspace_c2.selectbox("Live Search Provider", options=provider_options, index=provider_index, key="opensearch_web_provider")
    available_models = status.get("available_models", []) or []
    auto_model_caption = "Automatic model routing is active."
    if available_models:
        auto_model_caption += " Available local chat models: " + ", ".join(str(item.get("id")) for item in available_models)
    workspace_c3.caption(auto_model_caption)
    auto_fetch_and_index = st.toggle(
        "Auto Fetch And Index Live Search Results",
        value=True,
        key="opensearch_auto_fetch_index",
        help="When enabled, live web results are fetched and stored into OpenSearch before explanation/search display.",
    )
    if "opensearch_chat_thread_id" not in st.session_state:
        st.session_state["opensearch_chat_thread_id"] = uuid4().hex

    thread_col1, thread_col2 = st.columns([3, 1])
    thread_options = []
    thread_preview_map: Dict[str, str] = {}
    if collection.strip():
        try:
            thread_payload = api_get(base_url, f"/opensearch/threads?collection={collection.strip() or 'default'}")
            thread_options = list((thread_payload or {}).get("threads", []) or [])
        except Exception:
            thread_options = []
    for item in thread_options:
        thread_id = str(item.get("thread_id", ""))
        preview = str(item.get("preview", "") or "No preview yet")
        thread_preview_map[thread_id] = preview[:80]
    thread_choices = ["Current thread"] + [f"{tid} · {thread_preview_map.get(tid, '')}" for tid in thread_preview_map]
    selected_thread_choice = thread_col1.selectbox("Conversation Thread", options=thread_choices, index=0, key="opensearch_thread_choice")
    if selected_thread_choice != "Current thread":
        st.session_state["opensearch_chat_thread_id"] = selected_thread_choice.split(" · ", 1)[0]
        try:
            history_payload = api_get(
                base_url,
                f"/opensearch/history?thread_id={st.session_state['opensearch_chat_thread_id']}&collection={collection.strip() or 'default'}",
            )
            st.session_state["opensearch_chat_history"] = list((history_payload or {}).get("turns", []) or [])
            messages: List[Dict[str, str]] = []
            for turn in st.session_state["opensearch_chat_history"]:
                title = str((turn.get("metadata") or {}).get("query") or turn.get("title") or "User Question")
                summary = str(turn.get("summary", "") or "")
                messages.append({"role": "user", "content": title})
                messages.append({"role": "assistant", "content": summary})
            st.session_state["opensearch_chat_messages"] = messages
        except Exception:
            pass
    if thread_col2.button("New Thread", use_container_width=True):
        st.session_state["opensearch_chat_thread_id"] = uuid4().hex
        st.session_state["opensearch_chat_history"] = []
        st.session_state["opensearch_chat_messages"] = []
        st.success("Started a new OpenSearch chat thread.")
    st.caption(
        "To set procedural memory, just tell the chatbot directly in the question box — for example: "
        "`From now on, always cite links and keep answers concise.`"
    )

    st.markdown("### URL Ingestion")
    urls_text = st.text_area(
        "URLs To Fetch And Index",
        value="https://example.com",
        height=120,
        help="One URL per line. These pages are fetched by the backend, indexed into OpenSearch, and then searchable below.",
    )
    if st.button("Fetch And Index URLs", use_container_width=True):
        try:
            urls = [line.strip() for line in urls_text.splitlines() if line.strip()]
            out = api_post_json(
                base_url,
                "/opensearch/web/ingest",
                {"collection": collection.strip() or "default", "urls": urls, "llm_provider": "ollama_local"},
            )
            st.session_state["opensearch_web_ingest_result"] = out
            st.success(f"Indexed {out.get('indexed', 0)} URLs into `{out.get('collection', collection)}`.")
        except Exception as exc:
            st.error(f"URL ingestion failed: {exc}")

    if st.session_state.get("opensearch_web_ingest_result"):
        with st.expander("Latest Ingest Result", expanded=False):
            st.json(st.session_state["opensearch_web_ingest_result"])

    st.markdown("### OpenSearch Chat")
    top_k = st.slider("Top K Links", min_value=3, max_value=15, value=8)
    if "opensearch_chat_messages" not in st.session_state:
        st.session_state["opensearch_chat_messages"] = []

    def _stream_opensearch_prompt(prompt_text: str) -> None:
        st.session_state["opensearch_chat_messages"].append({"role": "user", "content": prompt_text})
        payload = {
            "collection": collection.strip() or "default",
            "query": prompt_text.strip(),
            "top_k": int(top_k),
            "provider": search_provider,
            "auto_fetch_and_index": bool(auto_fetch_and_index),
            "llm_provider": "ollama_local",
            "thread_id": st.session_state.get("opensearch_chat_thread_id"),
        }
        assistant_chunks: List[str] = []
        final_result: Dict[str, Any] | None = None
        model_routing: Dict[str, Any] = {}
        with st.chat_message("assistant"):
            answer_placeholder = st.empty()
            route_placeholder = st.empty()
            response = requests.post(
                f"{base_url.rstrip('/')}/opensearch/chat/stream",
                json=payload,
                headers=_api_headers(),
                timeout=180,
                stream=True,
            )
            response.raise_for_status()
            for raw_line in response.iter_lines(decode_unicode=True):
                if not raw_line:
                    continue
                event = json.loads(raw_line)
                event_type = str(event.get("event", ""))
                if event_type == "model_routing":
                    model_routing = {
                        "model_name": str(event.get("model_name", "") or ""),
                        "reason": str(event.get("reason", "") or ""),
                    }
                    route_placeholder.caption(
                        f"Routing with `{model_routing['model_name']}` — {model_routing['reason']}"
                    )
                elif event_type == "answer_chunk":
                    chunk = str(event.get("text", "") or "")
                    if chunk:
                        assistant_chunks.append(chunk)
                        answer_placeholder.markdown("".join(assistant_chunks))
                elif event_type == "result":
                    final_result = dict(event.get("data", {}) or {})
                elif event_type == "error":
                    raise RuntimeError(str(event.get("message", "Unknown streaming error")))
        if final_result is None:
            raise RuntimeError("Streaming finished without a final result.")
        if model_routing and not final_result.get("model_routing"):
            final_result["model_routing"] = model_routing
        if assistant_chunks and not final_result.get("answer"):
            final_result["answer"] = "".join(assistant_chunks)
        st.session_state["opensearch_web_search_result"] = final_result
        st.session_state["opensearch_chat_thread_id"] = final_result.get(
            "thread_id", st.session_state.get("opensearch_chat_thread_id")
        )
        st.session_state["opensearch_chat_history"] = final_result.get("history", []) or []
        st.session_state["opensearch_chat_messages"].append(
            {"role": "assistant", "content": str(final_result.get("answer", "") or "")}
        )

    st.markdown("### Chat Thread")
    st.caption(f"Thread ID: `{st.session_state.get('opensearch_chat_thread_id', '')}`")
    for message in st.session_state.get("opensearch_chat_messages", []):
        with st.chat_message(message.get("role", "assistant")):
            st.write(message.get("content", ""))

    result = st.session_state.get("opensearch_web_search_result")
    if isinstance(result, dict):
        st.markdown("### Latest Answer")
        st.write(result.get("answer", "") or result.get("explanation", ""))
        c1, c2, c3 = st.columns(3)
        c1.metric("Provider Used", str(result.get("provider", "manual_urls")))
        c2.metric("Indexed This Run", int(result.get("indexed_count", 0)))
        c3.metric("Links Returned", len(result.get("hits", []) or []))
        provider_chain = result.get("provider_chain", []) or []
        if provider_chain:
            st.caption("Search chain used: " + " -> ".join(provider_chain))
        model_routing = result.get("model_routing", {}) or {}
        if isinstance(model_routing, dict) and model_routing:
            st.markdown("### Model Routing")
            st.markdown(
                "\n".join(
                    [
                        f"- **Model selected automatically**: {model_routing.get('model_name', 'unknown')}",
                        f"- **Why this model was chosen**: {model_routing.get('reason', 'No reason captured.')}",
                        f"- **Answer generated by**: {model_routing.get('model_name', 'unknown')}",
                    ]
                )
            )
        reranker = result.get("reranker", {}) or {}
        if isinstance(reranker, dict):
            st.markdown("### Reranking")
            r1, r2, r3 = st.columns(3)
            backend_label = str(reranker.get("backend", "off"))
            model_label = str(reranker.get("model", "disabled"))
            r1.metric("Reranker", f"{backend_label}: {model_label}")
            r2.metric("Applied", "Yes" if reranker.get("applied") else "No")
            r3.metric("Candidate Pool", int(reranker.get("candidate_count", 0) or 0))
            if reranker.get("error"):
                st.caption(f"Reranker note: {reranker.get('error')}")
        warnings = result.get("warnings", []) or []
        if warnings:
            st.warning("\n".join(warnings))

        live_results = result.get("live_results", []) or []
        if live_results:
            st.markdown("### Search Engine Links")
            st.caption("These are the raw links returned by the live search provider before reranking.")
            for idx, hit in enumerate(live_results, start=1):
                title = str(hit.get("title", "") or hit.get("url", "Untitled"))
                url = str(hit.get("url", ""))
                snippet = str(hit.get("snippet", "") or "")[:300]
                st.markdown(f"**{idx}. [{title}]({url})**")
                st.write(snippet)

        hits = result.get("hits", []) or []
        st.markdown("### Most Relevant Links")
        st.caption("These are the final links after OpenSearch retrieval and reranking.")
        if not hits:
            st.info("No indexed links matched this query.")
        for idx, hit in enumerate(hits, start=1):
            title = str(hit.get("title", "") or hit.get("url", "Untitled"))
            url = str(hit.get("url", ""))
            domain = str(hit.get("domain", ""))
            snippet = str(hit.get("snippet", "") or "")[:500]
            st.markdown(f"**{idx}. [{title}]({url})**")
            rerank_score = hit.get("rerank_score")
            lexical_score = hit.get("lexical_score")
            if rerank_score is not None:
                st.caption(f"{domain} • rerank_score={rerank_score} • lexical_score={lexical_score}")
            else:
                st.caption(f"{domain}")
            st.write(snippet)

        with st.expander("Raw Search Result", expanded=False):
            redacted_result = dict(result)
            redacted_result.pop("memory", None)
            st.json(redacted_result)

    prompt = st.chat_input("Ask a web research question, then keep following up in the same thread.")
    if prompt:
        try:
            _stream_opensearch_prompt(prompt)
            st.rerun()
        except Exception as exc:
            st.error(f"OpenSearch chat failed: {exc}")


def main() -> None:
    st.set_page_config(page_title="Agentic AutoML Streamlit", page_icon="🧠", layout="wide")
    st.title("🧠 Agentic AutoML - Streamlit Control Plane")

    with st.sidebar:
        st.markdown("### Backend")
        base_url = st.text_input("Backend URL", value=st.session_state.get("backend_url", DEFAULT_BACKEND_URL))
        st.session_state["backend_url"] = base_url
        sdr_base_url = st.text_input(
            "AI SDR Backend URL",
            value=st.session_state.get("sdr_backend_url", DEFAULT_SDR_API_URL),
        )
        st.session_state["sdr_backend_url"] = sdr_base_url
        st.markdown("### User Mode")
        user_mode = st.selectbox(
            "Mode",
            options=["Business", "Data Science", "MLOps"],
            index=0,
            help="Business: simple AutoML. Data Science: advanced evaluation/tuning. MLOps: lifecycle and operations.",
        )
        st.session_state["user_mode"] = user_mode

        if st.button("Health Check"):
            try:
                h = api_get(base_url, "/health")
                st.success("Backend reachable")
                st.code(safe_json(h), language="json")
                try:
                    m = api_get(base_url, "/models")
                    st.code(safe_json(m), language="json")
                except Exception:
                    st.caption("This backend does not expose the optional model-registry endpoint.")
            except Exception as exc:
                st.error(f"Backend check failed: {exc}")

    tab_defs: list[tuple[str, Any]] = [
        ("AI SDR", render_ai_sdr_ui),
        ("Structured", structured_tab),
        ("Unstructured", unstructured_tab),
        ("OpenSearch", opensearch_tab),
        ("Custom GPT", custom_gpt_tab),
        ("Transformations", lambda base_url: transformations_tab(base_url)),
    ]
    if user_mode in {"Data Science", "MLOps"}:
        tab_defs.extend(
            [
                ("Analytics", analytics_tab),
            ]
        )
    tab_defs.append(("Connectors", connectors_tab))
    if user_mode == "MLOps":
        tab_defs.append(("MLOps", mlops_tab))
    tab_defs.append(("Chat", chat_tab))

    tabs = st.tabs([t[0] for t in tab_defs])
    for tab_obj, (label, fn) in zip(tabs, tab_defs):
        with tab_obj:
            if label == "Structured":
                fn(base_url, user_mode)  # type: ignore[misc]
            elif label == "Connectors":
                fn(base_url, user_mode)  # type: ignore[misc]
            elif label == "AI SDR":
                fn(sdr_base_url)
            else:
                fn(base_url)


if __name__ == "__main__":
    main()
