from __future__ import annotations

import json
from typing import Any, Dict

import requests
import streamlit as st
import streamlit.components.v1 as components

from custom_gpt.auth_context import apply_token, clear_auth_context, get_auth_context, set_auth_context
from custom_gpt.client import build_headers
from custom_gpt.types import AuthContext, HeaderConfig


def _auth_card(ctx: AuthContext) -> None:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Auth Status", "Authenticated" if ctx.authenticated else "Anonymous")
    c2.metric("User", ctx.user_id or "N/A")
    c3.metric("Tenant", ctx.tenant_id or "N/A")
    c4.metric("Role", ctx.role or "N/A")
    if ctx.scopes:
        st.caption("Scopes: " + ", ".join(ctx.scopes))


def _auth_form() -> HeaderConfig:
    ctx = get_auth_context()
    with st.expander("Auth Context", expanded=True):
        token_input = st.text_area("JWT Token", value=ctx.token, height=90, key="custom_gpt_token")
        c1, c2 = st.columns(2)
        with c1:
            tenant = st.text_input("Tenant ID", value=ctx.tenant_id, key="custom_gpt_tenant")
            role = st.text_input("Role", value=ctx.role, key="custom_gpt_role")
        with c2:
            user_id = st.text_input("User ID", value=ctx.user_id, key="custom_gpt_user_id")
            scopes = st.text_input("Scopes (space separated)", value=" ".join(ctx.scopes), key="custom_gpt_scopes")

        apply_role_headers = st.toggle("Apply Role Headers", value=True, key="custom_gpt_apply_role_headers")
        include_api_key = st.toggle("Include X-API-Key", value=False, key="custom_gpt_include_api_key")
        api_key = st.text_input("API Key (optional)", value="", type="password", key="custom_gpt_api_key") if include_api_key else None

        b1, b2, b3 = st.columns(3)
        with b1:
            if st.button("Apply JWT", use_container_width=True):
                if token_input.strip():
                    apply_token(token_input.strip())
                    st.success("JWT applied.")
                else:
                    st.warning("Token is empty.")
        with b2:
            if st.button("Save Manual Context", use_container_width=True):
                set_auth_context(
                    AuthContext(
                        token=token_input.strip(),
                        user_id=user_id.strip(),
                        tenant_id=tenant.strip(),
                        role=role.strip() or "viewer",
                        scopes=[s for s in scopes.split() if s],
                        authenticated=bool(token_input.strip() or user_id.strip()),
                        auth_source="manual",
                    )
                )
                st.success("Context saved.")
        with b3:
            if st.button("Logout / Clear Auth", use_container_width=True):
                clear_auth_context()
                st.success("Auth context cleared.")

    return HeaderConfig(apply_role_headers=apply_role_headers, include_api_key=include_api_key, api_key=api_key)


def _debug_headers_panel(headers: Dict[str, str]) -> None:
    safe_headers = {k: ("***" if "authorization" in k.lower() or "api-key" in k.lower() else v) for k, v in headers.items()}
    st.markdown("### Request Headers Preview")
    st.code(str(safe_headers))


def _ping_backend(base_url: str, headers: Dict[str, str]) -> Dict[str, Any]:
    url = f"{base_url.rstrip('/')}/health"
    resp = requests.get(url, headers=headers, timeout=60)
    detail: Any = resp.text
    try:
        detail = resp.json()
    except Exception:
        pass
    return {"status_code": resp.status_code, "ok": resp.ok, "detail": detail}


def _fetch_connectors(base_url: str, headers: Dict[str, str]) -> list[dict]:
    url = f"{base_url.rstrip('/')}/connectors"
    resp = requests.get(url, headers=headers, timeout=60)
    resp.raise_for_status()
    return resp.json()


def _check_connector_access(base_url: str, headers: Dict[str, str], connector: str, source_scope: Dict[str, Any]) -> Dict[str, Any]:
    url = f"{base_url.rstrip('/')}/connectors/access/check"
    resp = requests.post(url, json={"connector": connector, "source_scope": source_scope}, headers=headers, timeout=60)
    detail: Any = resp.text
    try:
        detail = resp.json()
    except Exception:
        pass
    if not resp.ok:
        raise requests.HTTPError(f"{resp.status_code} {resp.reason}: {detail}")
    return detail


def _classify_intent(base_url: str, headers: Dict[str, str], payload: Dict[str, Any]) -> Dict[str, Any]:
    url = f"{base_url.rstrip('/')}/custom-gpt/intent"
    resp = requests.post(url, json=payload, headers=headers, timeout=90)
    detail: Any = resp.text
    try:
        detail = resp.json()
    except Exception:
        pass
    if not resp.ok:
        raise requests.HTTPError(f"{resp.status_code} {resp.reason}: {detail}")
    return detail


def _run_custom_gpt_query(base_url: str, headers: Dict[str, str], payload: Dict[str, Any]) -> Dict[str, Any]:
    url = f"{base_url.rstrip('/')}/custom-gpt/query"
    resp = requests.post(url, json=payload, headers=headers, timeout=120)
    detail: Any = resp.text
    try:
        detail = resp.json()
    except Exception:
        pass
    if not resp.ok:
        raise requests.HTTPError(f"{resp.status_code} {resp.reason}: {detail}")
    return detail


def _get_custom_gpt_metrics(base_url: str, headers: Dict[str, str], limit: int = 200) -> Dict[str, Any]:
    url = f"{base_url.rstrip('/')}/custom-gpt/metrics?limit={int(limit)}"
    resp = requests.get(url, headers=headers, timeout=60)
    detail: Any = resp.text
    try:
        detail = resp.json()
    except Exception:
        pass
    if not resp.ok:
        raise requests.HTTPError(f"{resp.status_code} {resp.reason}: {detail}")
    return detail


def _run_custom_gpt_eval(base_url: str, headers: Dict[str, str], payload: Dict[str, Any]) -> Dict[str, Any]:
    url = f"{base_url.rstrip('/')}/custom-gpt/evals/run"
    resp = requests.post(url, json=payload, headers=headers, timeout=300)
    detail: Any = resp.text
    try:
        detail = resp.json()
    except Exception:
        pass
    if not resp.ok:
        raise requests.HTTPError(f"{resp.status_code} {resp.reason}: {detail}")
    return detail


def _build_unstructured_kb(base_url: str, headers: Dict[str, str], file_name: str, payload: bytes, form_data: Dict[str, Any]) -> Dict[str, Any]:
    url = f"{base_url.rstrip('/')}/kb/build"
    resp = requests.post(url, files={"file": (file_name, payload)}, data=form_data, headers=headers, timeout=300)
    detail: Any = resp.text
    try:
        detail = resp.json()
    except Exception:
        pass
    if not resp.ok:
        raise requests.HTTPError(f"{resp.status_code} {resp.reason}: {detail}")
    return detail


def _kg_build_from_connector(base_url: str, headers: Dict[str, str], payload: Dict[str, Any]) -> Dict[str, Any]:
    url = f"{base_url.rstrip('/')}/kg/build-from-connector"
    resp = requests.post(url, json=payload, headers=headers, timeout=300)
    detail: Any = resp.text
    try:
        detail = resp.json()
    except Exception:
        pass
    if not resp.ok:
        raise requests.HTTPError(f"{resp.status_code} {resp.reason}: {detail}")
    return detail


def _kg_entities_search(base_url: str, headers: Dict[str, str], payload: Dict[str, Any]) -> Dict[str, Any]:
    url = f"{base_url.rstrip('/')}/kg/entities/search"
    resp = requests.post(url, json=payload, headers=headers, timeout=120)
    detail: Any = resp.text
    try:
        detail = resp.json()
    except Exception:
        pass
    if not resp.ok:
        raise requests.HTTPError(f"{resp.status_code} {resp.reason}: {detail}")
    return detail


def _kg_customer360(base_url: str, headers: Dict[str, str], payload: Dict[str, Any]) -> Dict[str, Any]:
    url = f"{base_url.rstrip('/')}/kg/customer360"
    resp = requests.post(url, json=payload, headers=headers, timeout=120)
    detail: Any = resp.text
    try:
        detail = resp.json()
    except Exception:
        pass
    if not resp.ok:
        raise requests.HTTPError(f"{resp.status_code} {resp.reason}: {detail}")
    return detail


def _kg_path(base_url: str, headers: Dict[str, str], payload: Dict[str, Any]) -> Dict[str, Any]:
    url = f"{base_url.rstrip('/')}/kg/path"
    resp = requests.post(url, json=payload, headers=headers, timeout=120)
    detail: Any = resp.text
    try:
        detail = resp.json()
    except Exception:
        pass
    if not resp.ok:
        raise requests.HTTPError(f"{resp.status_code} {resp.reason}: {detail}")
    return detail


def _kg_root_cause(base_url: str, headers: Dict[str, str], payload: Dict[str, Any]) -> Dict[str, Any]:
    url = f"{base_url.rstrip('/')}/kg/root-cause"
    resp = requests.post(url, json=payload, headers=headers, timeout=120)
    detail: Any = resp.text
    try:
        detail = resp.json()
    except Exception:
        pass
    if not resp.ok:
        raise requests.HTTPError(f"{resp.status_code} {resp.reason}: {detail}")
    return detail


def _kg_analytics(base_url: str, headers: Dict[str, str], payload: Dict[str, Any]) -> Dict[str, Any]:
    url = f"{base_url.rstrip('/')}/kg/analytics"
    resp = requests.post(url, json=payload, headers=headers, timeout=120)
    detail: Any = resp.text
    try:
        detail = resp.json()
    except Exception:
        pass
    if not resp.ok:
        raise requests.HTTPError(f"{resp.status_code} {resp.reason}: {detail}")
    return detail


def _kg_subgraph(base_url: str, headers: Dict[str, str], payload: Dict[str, Any]) -> Dict[str, Any]:
    url = f"{base_url.rstrip('/')}/kg/subgraph"
    resp = requests.post(url, json=payload, headers=headers, timeout=120)
    detail: Any = resp.text
    try:
        detail = resp.json()
    except Exception:
        pass
    if not resp.ok:
        raise requests.HTTPError(f"{resp.status_code} {resp.reason}: {detail}")
    return detail


def _kg_cypher(base_url: str, headers: Dict[str, str], payload: Dict[str, Any]) -> Dict[str, Any]:
    url = f"{base_url.rstrip('/')}/kg/cypher"
    resp = requests.post(url, json=payload, headers=headers, timeout=180)
    detail: Any = resp.text
    try:
        detail = resp.json()
    except Exception:
        pass
    if not resp.ok:
        raise requests.HTTPError(f"{resp.status_code} {resp.reason}: {detail}")
    return detail


def _render_cytoscape(
    subgraph: Dict[str, Any],
    layout: str = "cose",
    height: int = 620,
    max_nodes: int | None = None,
    max_edges: int | None = None,
) -> None:
    raw_nodes = subgraph.get("nodes", []) if isinstance(subgraph, dict) else []
    raw_edges = subgraph.get("edges", []) if isinstance(subgraph, dict) else []
    nodes = []
    for node in raw_nodes:
        if not isinstance(node, dict):
            continue
        node_id = str(node.get("id") or "").strip()
        if not node_id or node_id.lower() == "none":
            continue
        normalized = dict(node)
        normalized["id"] = node_id
        normalized["name"] = str(node.get("name") or node_id)
        normalized["label"] = str(node.get("label") or "Entity")
        nodes.append(normalized)
    valid_ids = {str(node.get("id")) for node in nodes}
    edges = []
    for edge in raw_edges:
        if not isinstance(edge, dict):
            continue
        src = str(edge.get("source") or "").strip()
        dst = str(edge.get("target") or "").strip()
        if not src or not dst or src.lower() == "none" or dst.lower() == "none":
            continue
        if src not in valid_ids or dst not in valid_ids:
            continue
        edges.append({**edge, "source": src, "target": dst, "type": str(edge.get("type") or "RELATED_TO")})
    if not nodes:
        st.info("No graph nodes to visualize yet.")
        return

    if max_nodes and max_nodes > 0:
        nodes = nodes[:max_nodes]
    kept_ids = {str(node.get("id")) for node in nodes}
    edges = [edge for edge in edges if str(edge.get("source")) in kept_ids and str(edge.get("target")) in kept_ids]
    if max_edges and max_edges > 0:
        edges = edges[:max_edges]

    elements = []
    degree_map: Dict[str, int] = {}
    for edge in edges:
        src = str(edge.get("source"))
        dst = str(edge.get("target"))
        degree_map[src] = degree_map.get(src, 0) + 1
        degree_map[dst] = degree_map.get(dst, 0) + 1
    for node in nodes:
        label = str(node.get("name") or node.get("id") or "")
        node_label = str(node.get("label") or "Entity")
        elements.append(
            {
                "data": {
                    "id": str(node.get("id")),
                    "label": label,
                    "type": node_label,
                    "degree": int(degree_map.get(str(node.get("id")), 0)),
                    "raw": node,
                }
            }
        )
    for idx, edge in enumerate(edges):
        elements.append(
            {
                "data": {
                    "id": f"e{idx}",
                    "source": str(edge.get("source")),
                    "target": str(edge.get("target")),
                    "label": str(edge.get("type") or "RELATED_TO"),
                    "raw": edge,
                }
            }
        )

    html = f"""
    <style>
      .kg-shell {{
        display: grid;
        grid-template-columns: minmax(0, 1fr) 280px;
        gap: 14px;
        align-items: start;
        font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
      }}
      .kg-stage {{
        border: 1px solid #cbd5e1;
        border-radius: 14px;
        overflow: hidden;
        background:
          radial-gradient(circle at top left, rgba(14,165,233,0.08), transparent 32%),
          linear-gradient(180deg, #f8fbff 0%, #eef6ff 100%);
        box-shadow: 0 10px 28px rgba(15, 23, 42, 0.08);
      }}
      .kg-toolbar {{
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
        padding: 10px 12px;
        border-bottom: 1px solid #dbeafe;
        background: rgba(255,255,255,0.82);
        backdrop-filter: blur(6px);
      }}
      .kg-btn {{
        border: 1px solid #bfdbfe;
        background: white;
        color: #1d4ed8;
        border-radius: 999px;
        padding: 6px 10px;
        font-size: 12px;
        cursor: pointer;
      }}
      .kg-btn:hover {{
        background: #eff6ff;
      }}
      .kg-stat {{
        margin-left: auto;
        display: flex;
        gap: 8px;
      }}
      .kg-pill {{
        background: #0f172a;
        color: white;
        border-radius: 999px;
        padding: 6px 10px;
        font-size: 12px;
      }}
      #cy {{
        width: 100%;
        height: {height}px;
      }}
      .kg-side {{
        border: 1px solid #cbd5e1;
        border-radius: 14px;
        background: white;
        box-shadow: 0 8px 24px rgba(15, 23, 42, 0.06);
        overflow: hidden;
      }}
      .kg-side-header {{
        padding: 12px 14px;
        border-bottom: 1px solid #e2e8f0;
        background: linear-gradient(180deg, #f8fafc 0%, #f1f5f9 100%);
      }}
      .kg-side-header h4 {{
        margin: 0;
        font-size: 14px;
        color: #0f172a;
      }}
      .kg-side-header p {{
        margin: 4px 0 0 0;
        font-size: 12px;
        color: #475569;
      }}
      .kg-inspector {{
        padding: 12px 14px;
        font-size: 12px;
        color: #0f172a;
      }}
      .kg-inspector pre {{
        white-space: pre-wrap;
        word-break: break-word;
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 10px;
        max-height: 420px;
        overflow: auto;
        margin: 8px 0 0 0;
      }}
      .kg-legend {{
        padding: 0 14px 14px 14px;
        display: grid;
        gap: 8px;
      }}
      .kg-legend-item {{
        display: flex;
        align-items: center;
        gap: 8px;
        font-size: 12px;
        color: #334155;
      }}
      .kg-swatch {{
        width: 12px;
        height: 12px;
        border-radius: 999px;
        border: 1px solid rgba(15,23,42,0.2);
      }}
    </style>
    <div class="kg-shell">
      <div class="kg-stage">
        <div class="kg-toolbar">
          <button class="kg-btn" data-layout="cose">Organic</button>
          <button class="kg-btn" data-layout="breadthfirst">Hierarchy</button>
          <button class="kg-btn" data-layout="concentric">Concentric</button>
          <button class="kg-btn" data-layout="circle">Circle</button>
          <button class="kg-btn" id="fit-btn">Fit</button>
          <button class="kg-btn" id="reset-btn">Reset Highlights</button>
          <div class="kg-stat">
            <span class="kg-pill">Nodes: {len(nodes)}</span>
            <span class="kg-pill">Edges: {len(edges)}</span>
          </div>
        </div>
        <div id="cy"></div>
      </div>
      <div class="kg-side">
        <div class="kg-side-header">
          <h4>Inspector</h4>
          <p>Click any node or edge to inspect it, similar to Neo4j Browser.</p>
        </div>
        <div class="kg-inspector">
          <div id="inspector-title"><strong>No selection</strong></div>
          <div id="inspector-meta">Tap a node or edge to see details.</div>
          <pre id="inspector-json">Nothing selected yet.</pre>
        </div>
        <div class="kg-side-header">
          <h4>Legend</h4>
          <p>Common entity colors used in the graph.</p>
        </div>
        <div class="kg-legend">
          <div class="kg-legend-item"><span class="kg-swatch" style="background:#1d4ed8"></span> Customer / Account</div>
          <div class="kg-legend-item"><span class="kg-swatch" style="background:#ea580c"></span> Order / Shipment</div>
          <div class="kg-legend-item"><span class="kg-swatch" style="background:#7c3aed"></span> Company / Conversation / Document</div>
          <div class="kg-legend-item"><span class="kg-swatch" style="background:#dc2626"></span> Issue / Ticket / Risk</div>
          <div class="kg-legend-item"><span class="kg-swatch" style="background:#0f766e"></span> Other entity types</div>
        </div>
      </div>
    </div>
    <script src="https://unpkg.com/cytoscape@3.28.1/dist/cytoscape.min.js"></script>
    <script>
      const elements = {json.dumps(elements)};
      const cy = cytoscape({{
        container: document.getElementById('cy'),
        elements: elements,
        layout: {{ name: {json.dumps(layout)}, animate: true, padding: 30, spacingFactor: 1.15 }},
        style: [
          {{
            selector: 'node',
            style: {{
              'label': 'data(label)',
              'text-wrap': 'wrap',
              'text-max-width': 140,
              'font-size': 11,
              'background-color': '#0f766e',
              'color': '#0f172a',
              'text-valign': 'bottom',
              'text-margin-y': -10,
              'border-width': 2.5,
              'border-color': '#134e4a',
              'width': 'mapData(degree, 0, 8, 28, 52)',
              'height': 'mapData(degree, 0, 8, 28, 52)',
              'overlay-padding': 6,
              'text-background-color': '#ffffff',
              'text-background-opacity': 0.85,
              'text-background-padding': 2,
              'text-background-shape': 'roundrectangle'
            }}
          }},
          {{
            selector: 'edge',
            style: {{
              'curve-style': 'bezier',
              'width': 2,
              'line-color': '#94a3b8',
              'target-arrow-color': '#94a3b8',
              'target-arrow-shape': 'triangle',
              'label': 'data(label)',
              'font-size': 9,
              'text-background-color': '#ffffff',
              'text-background-opacity': 0.8,
              'text-background-padding': 2,
              'arrow-scale': 1.1
            }}
          }},
          {{
            selector: 'node[type = "Customer"], node[type = "Account"]',
            style: {{ 'background-color': '#1d4ed8', 'border-color': '#1e3a8a', 'color': '#1e293b' }}
          }},
          {{
            selector: 'node[type = "Order"]',
            style: {{ 'background-color': '#ea580c', 'border-color': '#9a3412' }}
          }},
          {{
            selector: 'node[type = "Issue"], node[type = "Ticket"]',
            style: {{ 'background-color': '#dc2626', 'border-color': '#7f1d1d' }}
          }},
          {{
            selector: 'node[type = "Conversation"], node[type = "Document"]',
            style: {{ 'background-color': '#7c3aed', 'border-color': '#4c1d95' }}
          }},
          {{
            selector: ':selected',
            style: {{ 'border-width': 5, 'border-color': '#f59e0b', 'line-color': '#f59e0b', 'target-arrow-color': '#f59e0b', 'z-index': 999 }}
          }},
          {{
            selector: '.faded',
            style: {{ 'opacity': 0.16 }}
          }},
          {{
            selector: '.highlighted',
            style: {{ 'opacity': 1, 'line-color': '#2563eb', 'target-arrow-color': '#2563eb', 'border-color': '#2563eb' }}
          }}
        ]
      }});
      cy.fit(undefined, 40);

      const inspectorTitle = document.getElementById('inspector-title');
      const inspectorMeta = document.getElementById('inspector-meta');
      const inspectorJson = document.getElementById('inspector-json');

      function setInspector(title, meta, raw) {{
        inspectorTitle.innerHTML = '<strong>' + title + '</strong>';
        inspectorMeta.textContent = meta;
        inspectorJson.textContent = JSON.stringify(raw, null, 2);
      }}

      function clearHighlights() {{
        cy.elements().removeClass('faded');
        cy.elements().removeClass('highlighted');
      }}

      function highlightNeighborhood(node) {{
        clearHighlights();
        cy.elements().addClass('faded');
        node.removeClass('faded').addClass('highlighted');
        const neighborhood = node.closedNeighborhood();
        neighborhood.removeClass('faded').addClass('highlighted');
      }}

      cy.on('tap', 'node', function(evt) {{
        const node = evt.target;
        highlightNeighborhood(node);
        setInspector(
          node.data('label'),
          'Node • ' + node.data('type') + ' • Degree ' + node.data('degree'),
          node.data('raw') || node.data()
        );
      }});

      cy.on('tap', 'edge', function(evt) {{
        const edge = evt.target;
        clearHighlights();
        cy.elements().addClass('faded');
        edge.removeClass('faded').addClass('highlighted');
        edge.connectedNodes().removeClass('faded').addClass('highlighted');
        setInspector(
          edge.data('label'),
          'Edge • ' + edge.data('source') + ' → ' + edge.data('target'),
          edge.data('raw') || edge.data()
        );
      }});

      cy.on('tap', function(evt) {{
        if (evt.target === cy) {{
          clearHighlights();
          setInspector('No selection', 'Tap a node or edge to see details.', 'Nothing selected yet.');
        }}
      }});

      document.querySelectorAll('.kg-btn[data-layout]').forEach((btn) => {{
        btn.addEventListener('click', () => {{
          const name = btn.getAttribute('data-layout');
          cy.layout({{
            name,
            animate: true,
            padding: 30,
            spacingFactor: 1.15
          }}).run();
        }});
      }});

      document.getElementById('fit-btn').addEventListener('click', () => cy.fit(undefined, 40));
      document.getElementById('reset-btn').addEventListener('click', () => {{
        clearHighlights();
        setInspector('No selection', 'Tap a node or edge to see details.', 'Nothing selected yet.');
        cy.fit(undefined, 40);
      }});
    </script>
    """
    components.html(html, height=height + 12, scrolling=False)


def _best_graph_seed(result: Dict[str, Any]) -> str:
    debug = result.get("debug", {}) if isinstance(result.get("debug"), dict) else {}
    execution = result.get("execution", {}) if isinstance(result.get("execution"), dict) else {}

    entity_search = debug.get("entity_search", {})
    entities = entity_search.get("entities", []) if isinstance(entity_search, dict) else []
    if entities:
        return str(entities[0].get("name", "") or "")

    graph_debug = debug.get("graph", {}) if isinstance(debug.get("graph"), dict) else {}
    search = graph_debug.get("search", {}) if isinstance(graph_debug.get("search"), dict) else {}
    graph_entities = search.get("entities", []) if isinstance(search, dict) else []
    if graph_entities:
        return str(graph_entities[0].get("name", "") or "")

    anchor = execution.get("anchor") if isinstance(execution, dict) else None
    if anchor:
        return str(anchor)
    return ""


def _render_key_value_block(data: Dict[str, Any], title: str | None = None) -> None:
    if not isinstance(data, dict) or not data:
        return
    if title:
        st.markdown(f"**{title}**")
    lines = []
    for key, value in data.items():
        if isinstance(value, (dict, list)):
            continue
        label = str(key).replace("_", " ").title()
        lines.append(f"- `{label}`: {value}")
    if lines:
        st.markdown("\n".join(lines))


def _render_list_block(items: list[Any], title: str) -> None:
    if not items:
        return
    st.markdown(f"**{title}**")
    lines = []
    for item in items:
        lines.append(f"- {item}")
    st.markdown("\n".join(lines))


def _render_intent_summary(intent: Dict[str, Any]) -> None:
    st.markdown("### Intent Route (Auto)")
    c1, c2, c3 = st.columns(3)
    c1.metric("Route", str(intent.get("route", "N/A")))
    c2.metric("Confidence", f"{float(intent.get('confidence', 0.0)):.2f}")
    c3.metric("Graph Expansion", "Yes" if intent.get("use_graph_expansion") else "No")
    st.markdown(f"**Reason**: {intent.get('reason', 'No reason provided.')}")
    source_flags = []
    if intent.get("use_structured"):
        source_flags.append("Structured")
    if intent.get("use_docs"):
        source_flags.append("Documents")
    if intent.get("use_kg"):
        source_flags.append("Knowledge Graph")
    if source_flags:
        st.markdown(f"**Selected Sources**: {', '.join(source_flags)}")
    if intent.get("suggested_next_step"):
        st.markdown(f"**Next Step**: {intent.get('suggested_next_step')}")


def _render_structured_execution_help() -> None:
    st.markdown("### Structured Execution Options")
    st.caption("These options control where a structured answer comes from and how aggressively we push the request into SQL-style execution.")
    st.markdown(
        "\n".join(
            [
                "- **Use Current Structured Preview Data**: answers from the dataframe currently loaded in the Structured tab.",
                "- **Use Latest Structured Pipeline Result**: answers from the most recent structured pipeline run, such as model outputs, summary tables, or metrics.",
                "- **Use Connector Source for Execution**: runs the structured request against the selected connector, such as CRM or OMS.",
                "- **Force SQL Route**: pushes the structured path toward direct SQL-style lookup instead of a lighter summary path.",
                "- **Include Debug Payload**: keeps the developer-style execution details available in the debug section below the answer.",
            ]
        )
    )


def _render_monitoring_summary(metrics: Dict[str, Any]) -> None:
    latency = metrics.get("latency_ms", {}) if isinstance(metrics.get("latency_ms"), dict) else {}
    confidence = metrics.get("confidence", {}) if isinstance(metrics.get("confidence"), dict) else {}
    citations = metrics.get("citations", {}) if isinstance(metrics.get("citations"), dict) else {}
    routes = metrics.get("routes", {}) if isinstance(metrics.get("routes"), dict) else {}
    actions = metrics.get("actions", {}) if isinstance(metrics.get("actions"), dict) else {}
    sla = metrics.get("sla", {}) if isinstance(metrics.get("sla"), dict) else {}
    drift = metrics.get("drift", {}) if isinstance(metrics.get("drift"), dict) else {}

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Trace Count", str(metrics.get("trace_count", 0)))
    m2.metric("Avg Latency", str(latency.get("avg", 0.0)))
    m3.metric("P95 Latency", str(latency.get("p95", 0.0)))
    m4.metric("Low Confidence Rate", str(confidence.get("low_confidence_rate", 0.0)))

    st.markdown("**Production Summary**")
    st.markdown(
        "\n".join(
            [
                f"- Average confidence: `{confidence.get('avg', 0.0)}`",
                f"- Average citation count: `{citations.get('avg', 0.0)}`",
                f"- No-citation rate: `{citations.get('no_citation_rate', 0.0)}`",
                f"- Action-request rate: `{actions.get('action_requested_rate', 0.0)}`",
                f"- Approval-required rate: `{actions.get('approval_required_rate', 0.0)}`",
                f"- Error rate: `{metrics.get('error_rate', 0.0)}`",
            ]
        )
    )
    _render_key_value_block(routes, "Route Distribution")
    _render_key_value_block(sla, "SLA Snapshot")
    _render_key_value_block(drift, "Drift Signals")


def _render_eval_summary(eval_run: Dict[str, Any]) -> None:
    m1, m2, m3 = st.columns(3)
    m1.metric("Eval Status", str(eval_run.get("status", "N/A")).upper())
    m2.metric("Trace Count", str(eval_run.get("trace_count", 0)))
    m3.metric("Average Score", str(eval_run.get("avg_score", 0.0)))
    _render_key_value_block(eval_run.get("metric_summary", {}), "Metric Summary")
    metrics = eval_run.get("metrics", [])
    if not metrics:
        return
    st.markdown("**Recent Trace Eval Results**")
    for item in metrics:
        if not isinstance(item, dict):
            continue
        title = f"Trace `{item.get('trace_id', 'N/A')}`"
        route = item.get("route", "unknown")
        latency_ms = item.get("latency_ms", "N/A")
        citations = item.get("citation_count", 0)
        confidence = item.get("confidence", 0.0)
        with st.expander(title, expanded=False):
            st.markdown(
                "\n".join(
                    [
                        f"- **Route**: {route}",
                        f"- **Latency (ms)**: {latency_ms}",
                        f"- **Citation Count**: {citations}",
                        f"- **Confidence**: {confidence}",
                    ]
                )
            )
            deepeval = item.get("deepeval", {}) if isinstance(item.get("deepeval"), dict) else {}
            if deepeval:
                st.markdown(f"**DeepEval Status**: {deepeval.get('status', 'unknown')}")
                for metric in deepeval.get("metrics", []):
                    if not isinstance(metric, dict):
                        continue
                    st.markdown(
                        f"- `{metric.get('metric', 'metric')}`: score `{metric.get('score', 0)}` · status `{metric.get('status', 'unknown')}`"
                    )
                    if metric.get("reason"):
                        st.caption(metric.get("reason"))


def _render_answer_explanation(
    result: Dict[str, Any],
    *,
    selected_connector: str,
    use_current_preview: bool,
    use_latest_run: bool,
    use_connector_source: bool,
    force_sql: bool,
    include_debug: bool,
) -> None:
    st.markdown("### How This Answer Was Produced")
    route = str(result.get("route", "unknown") or "unknown")
    execution = result.get("execution", {}) if isinstance(result.get("execution"), dict) else {}
    debug = result.get("debug", {}) if isinstance(result.get("debug"), dict) else {}
    source_name = execution.get("connector") or (selected_connector if use_connector_source else None)
    data_source = debug.get("data_source") or execution.get("data_source")
    if not source_name and use_current_preview:
        source_name = "Structured preview data"
    if not source_name and use_latest_run:
        source_name = "Latest structured pipeline result"
    if not source_name:
        source_name = "Auto-selected by the current route"
    sql_used = result.get("sql")
    source_rows = execution.get("source_rows") or debug.get("rows_in_source")
    result_rows = execution.get("row_count") or result.get("row_count")
    sql_strategy = execution.get("sql_strategy") or debug.get("sql_strategy")
    bullets = [
        f"- **Primary route**: {route}",
        f"- **Main source used**: {source_name}",
        f"- **Underlying data source**: {data_source or 'Not exposed for this route'}",
        f"- **Structured preview data enabled**: {'Yes' if use_current_preview else 'No'}",
        f"- **Latest pipeline result enabled**: {'Yes' if use_latest_run else 'No'}",
        f"- **Connector execution enabled**: {'Yes' if use_connector_source else 'No'}",
        f"- **Force SQL route enabled**: {'Yes' if force_sql else 'No'}",
        f"- **Debug payload kept**: {'Yes' if include_debug else 'No'}",
    ]
    if source_rows is not None:
        bullets.append(f"- **Rows available in the source before answering**: {source_rows}")
    if result_rows is not None:
        bullets.append(f"- **Rows returned for the answer**: {result_rows}")
    if sql_strategy:
        bullets.append(f"- **SQL generation strategy**: {sql_strategy}")
    if result.get("sql"):
        bullets.append("- **SQL execution**: A concrete SQL query was generated and is shown below.")
    elif route == "structured":
        bullets.append("- **SQL execution**: No explicit SQL query was surfaced for this answer. The system used the structured execution path and summarized the returned result.")
    elif route == "hybrid":
        bullets.append("- **Hybrid behavior**: The answer combined structured facts with graph or document context before generating the final response.")
    st.markdown("\n".join(bullets))
    if sql_used:
        st.markdown("**SQL Query Used**")
        st.code(str(sql_used), language="sql")


def render_custom_gpt_tab(base_url: str) -> None:
    st.subheader("Custom GPT")
    st.caption("Enterprise auth context, role display, and header-ready request flow.")

    ctx = get_auth_context()
    _auth_card(ctx)
    header_cfg = _auth_form()
    headers = build_headers(header_cfg)

    _debug_headers_panel(headers)

    if st.button("Test Backend with Current Auth Headers", use_container_width=True):
        try:
            out = _ping_backend(base_url, headers)
            if out["ok"]:
                st.success("Backend reachable with current headers.")
            else:
                st.warning("Backend responded with non-200 status.")
            st.json(out)
        except Exception as exc:
            st.error(f"Ping failed: {exc}")

    st.markdown("### Connector Access Gateway")
    if st.button("Refresh Accessible Connectors", use_container_width=True):
        try:
            st.session_state["custom_gpt_connectors_catalog"] = _fetch_connectors(base_url, headers)
            st.success("Connector catalog refreshed.")
        except Exception as exc:
            st.error(f"Connector catalog fetch failed: {exc}")

    connector_catalog = st.session_state.get("custom_gpt_connectors_catalog", [])
    allowed_connectors = [c for c in connector_catalog if isinstance(c, dict) and c.get("allowed")]
    connector_options = [c.get("name", "") for c in allowed_connectors] or ["PostgreSQL"]
    selected_connector = st.selectbox("Connector", options=connector_options, key="custom_gpt_connector_name")
    connector_key_map = {str(c.get("name", "")): str(c.get("key", "")) for c in allowed_connectors if isinstance(c, dict)}
    selected_connector_key = connector_key_map.get(selected_connector, selected_connector.lower().replace(" ", "_"))
    source_scope: Dict[str, Any] = {}
    source_scope_text = st.session_state.get("custom_gpt_connector_scope", '{"table":"customer_churn_realistic","query":"SELECT * FROM customer_churn_realistic LIMIT 10"}')
    if selected_connector_key == "crm":
        st.caption("CRM is wired to the marketing team APIs. It can read customer profile, CRM lead, Customer360, journey, campaign history, and analytics endpoints directly with the configured `x-api-key`.")
        crm_c1, crm_c2, crm_c3 = st.columns(3)
        crm_view = crm_c1.selectbox(
            "CRM View",
            options=[
                "customer_360",
                "crm_leads",
                "lookup",
                "customer_profile",
                "identity",
                "live_events",
                "audience_preset",
                "customer360",
                "active_journey",
                "campaign_history",
                "channel_history",
                "journeys",
                "journey_graph",
                "journey_validate",
                "journey_templates",
                "journey_template",
                "journey_runtime",
                "journey_runtime_lead",
                "journey_state",
                "journey_active",
                "journey_history",
                "journey_distribution",
                "journey_node_distribution",
                "analytics_ai_decisions",
                "analytics_channels",
                "analytics_campaigns",
                "profile",
                "attributes",
                "intelligence",
                "engagement",
                "commercial",
            ],
            index=0,
            key="custom_gpt_crm_view",
        )
        crm_company = crm_c2.text_input("Company Filter", value="", key="custom_gpt_crm_company")
        crm_segment = crm_c3.text_input("Segment Filter", value="", key="custom_gpt_crm_segment")
        crm_c4, crm_c5, crm_c6, crm_c7, crm_c8 = st.columns(5)
        crm_customer_id = crm_c4.text_input("Lead / Customer ID Filter", value="", key="custom_gpt_crm_customer_id")
        crm_email = crm_c5.text_input("Email Filter", value="", key="custom_gpt_crm_email")
        crm_journey_name = crm_c6.text_input("Journey Name Filter", value="", key="custom_gpt_crm_journey_name")
        crm_template_name = crm_c7.text_input("Template Name Filter", value="", key="custom_gpt_crm_template_name")
        crm_preset_name = crm_c8.selectbox(
            "Audience Preset",
            options=["", "top_churn", "high_intent", "high_clv", "highly_engaged", "dormant"],
            index=0,
            key="custom_gpt_crm_preset_name",
        )
        crm_limit = st.number_input("CRM Limit", min_value=10, max_value=1000, value=200, step=10, key="custom_gpt_crm_limit")
        source_scope = {"view": crm_view, "limit": int(crm_limit)}
        if crm_company.strip():
            source_scope["company"] = crm_company.strip()
        if crm_segment.strip():
            source_scope["segment"] = crm_segment.strip()
        if crm_customer_id.strip():
            source_scope["customer_id"] = crm_customer_id.strip()
            source_scope["lead_id"] = crm_customer_id.strip()
        if crm_email.strip():
            source_scope["email"] = crm_email.strip()
        if crm_journey_name.strip():
            source_scope["journey_name"] = crm_journey_name.strip()
        if crm_template_name.strip():
            source_scope["template_name"] = crm_template_name.strip()
        if crm_preset_name.strip():
            source_scope["preset"] = crm_preset_name.strip()
        source_scope_text = json.dumps(source_scope, indent=2)
        with st.expander("CRM Scope Preview", expanded=False):
            st.code(source_scope_text, language="json")
    elif selected_connector_key == "oms":
        st.caption("OMS can be filtered directly here too, so you don't need raw source JSON for the common cases.")
        oms_c1, oms_c2, oms_c3, oms_c4 = st.columns(4)
        oms_view = oms_c1.selectbox(
            "OMS View",
            options=["orders", "shipments", "returns", "refunds", "notifications"],
            index=0,
            key="custom_gpt_oms_view",
        )
        oms_order_number = oms_c2.text_input("Order Number Filter", value="", key="custom_gpt_oms_order_number")
        oms_status = oms_c3.text_input("Status Filter", value="", key="custom_gpt_oms_status")
        oms_limit = oms_c4.number_input("OMS Limit", min_value=10, max_value=1000, value=200, step=10, key="custom_gpt_oms_limit")
        oms_customer_email = st.text_input("OMS Customer Email Filter", value="", key="custom_gpt_oms_customer_email")
        source_scope = {"view": oms_view, "limit": int(oms_limit)}
        if oms_order_number.strip():
            source_scope["order_number"] = oms_order_number.strip()
        if oms_status.strip():
            source_scope["status"] = oms_status.strip()
        if oms_customer_email.strip():
            source_scope["customer_email"] = oms_customer_email.strip()
        source_scope_text = json.dumps(source_scope, indent=2)
        with st.expander("OMS Scope Preview", expanded=False):
            st.code(source_scope_text, language="json")
    else:
        source_scope_text = st.text_area(
            "Source Scope JSON",
            value=source_scope_text,
            height=90,
            key="custom_gpt_connector_scope",
        )
        source_scope = json.loads(source_scope_text) if source_scope_text.strip() else {}
    if st.button("Validate Connector Access for Custom GPT", use_container_width=True):
        try:
            access = _check_connector_access(base_url, headers, selected_connector, source_scope)
            st.session_state["custom_gpt_connector_access"] = access
            if access.get("allowed"):
                st.success("Connector gateway approved this request.")
            else:
                st.warning("Connector gateway denied this request.")
            st.json(access)
        except Exception as exc:
            st.error(f"Connector gateway check failed: {exc}")

    st.markdown("### Query Surface (Skeleton)")
    query = st.text_area(
        "Query",
        placeholder="Ask: Why is retention down in Washington?",
        height=120,
        key="custom_gpt_query_box",
    )
    s1, s2, s3 = st.columns(3)
    use_structured = s1.checkbox("Use Structured Source", value=True, key="custom_gpt_src_structured")
    use_docs = s2.checkbox("Use Docs Source", value=True, key="custom_gpt_src_docs")
    use_kg = s3.checkbox("Use Knowledge Graph Source", value=True, key="custom_gpt_src_kg")
    action_mode = st.toggle("Enable Action Mode", value=False, key="custom_gpt_action_toggle")

    st.markdown("### Unstructured Dataset")
    unstructured_dataset_id = st.text_input("Dataset ID", value=st.session_state.get("custom_gpt_unstructured_dataset_id", ""), key="custom_gpt_unstructured_dataset_id")
    parser_provider = st.selectbox("Parser Provider", options=["docling", "legacy"], index=0, key="custom_gpt_parser_provider")
    unstructured_upload = st.file_uploader(
        "Upload Unstructured File",
        type=["pdf", "docx", "txt", "md", "html", "csv", "json"],
        key="custom_gpt_unstructured_upload",
    )
    if st.button("Build Unstructured KB", use_container_width=True):
        try:
            if unstructured_upload is None:
                st.error("Upload a file first.")
            else:
                dataset_name = (unstructured_dataset_id or unstructured_upload.name).strip()
                out = _build_unstructured_kb(
                    base_url,
                    headers,
                    unstructured_upload.name,
                    unstructured_upload.getvalue(),
                    {
                        "dataset_id": dataset_name,
                        "llm_provider": "ollama_local",
                        "parser_provider": parser_provider,
                        "fast_mode": "false",
                        "chunk_cap": "200",
                        "skip_ner": "false",
                        "skip_pii": "false",
                        "skip_enrichment": "false",
                        "enrichment_batch_size": "8",
                        "enrichment_workers": "2",
                        "embedding_batch_size": "32",
                        "embedding_workers": "2",
                    },
                )
                st.session_state["custom_gpt_unstructured_build"] = out
                st.success(f"Built unstructured KB `{dataset_name}`.")
        except Exception as exc:
            st.error(f"KB build failed: {exc}")
    if isinstance(st.session_state.get("custom_gpt_unstructured_build"), dict):
        with st.expander("Latest KB Build", expanded=False):
            st.json(st.session_state["custom_gpt_unstructured_build"])

    st.markdown("### Knowledge Graph")
    graph_dataset_id = st.text_input(
        "Graph Dataset ID",
        value=st.session_state.get("custom_gpt_graph_dataset_id", "enterprise_graph"),
        key="custom_gpt_graph_dataset_id",
        help="Use one stable graph dataset id when building CRM and OMS into the same graph so entities can connect across systems.",
    )
    st.caption("Build the universal KG once from CRM or OMS connector data. After that, just ask in the main query box and Custom GPT will route to KG or hybrid automatically and render the graph for you.")
    graph_c1, graph_c2, graph_c3 = st.columns(3)
    graph_hops = graph_c1.selectbox("Graph Hops", options=[1, 2, 3], index=1, key="custom_gpt_graph_hops")
    graph_max_nodes = graph_c2.slider("Max Nodes", min_value=10, max_value=150, value=40, step=5, key="custom_gpt_graph_max_nodes")
    graph_max_edges = graph_c3.slider("Max Edges", min_value=10, max_value=200, value=60, step=5, key="custom_gpt_graph_max_edges")
    if st.button("Build KG From Selected Connector", use_container_width=True):
        try:
            parsed_scope = dict(source_scope)
            connector_query = parsed_scope.pop("query", None)
            connector_table = parsed_scope.pop("table", None)
            connector_limit = int(parsed_scope.pop("limit", 500))
            out = _kg_build_from_connector(
                base_url,
                headers,
                {
                    "dataset_id": graph_dataset_id.strip() or "enterprise_graph",
                    "connector": selected_connector,
                    "config": parsed_scope,
                    "query": connector_query,
                    "table": connector_table,
                    "limit": connector_limit,
                },
            )
            st.session_state["custom_gpt_kg_build"] = out
            st.success(f"Built KG dataset `{out.get('dataset_id')}` from `{out.get('connector')}`.")
        except Exception as exc:
            st.error(f"KG build failed: {exc}")

    for label, key in [
        ("Latest KG Build", "custom_gpt_kg_build"),
        ("KG Analytics", "custom_gpt_kg_analytics"),
    ]:
        if isinstance(st.session_state.get(key), dict):
            with st.expander(label, expanded=False):
                st.json(st.session_state[key])

    with st.expander("Cypher Workbench", expanded=False):
        st.caption("Run safe read-only Cypher against the current Neo4j graph dataset and render the result as a graph, table, or raw response.")
        cypher_query = st.text_area(
            "Cypher Query",
            value=st.session_state.get(
                "custom_gpt_kg_cypher_text",
                "MATCH (n:GraphMeta) RETURN n LIMIT 25",
            ),
            height=140,
            key="custom_gpt_kg_cypher_text",
        )
        cypher_limit = st.number_input("Cypher Row Limit", min_value=1, max_value=200, value=25, step=5, key="custom_gpt_kg_cypher_limit")
        if st.button("Run Cypher", use_container_width=True):
            try:
                out = _kg_cypher(
                    base_url,
                    headers,
                    {
                        "dataset_id": graph_dataset_id.strip() or "enterprise_graph",
                        "cypher": cypher_query,
                        "limit": int(cypher_limit),
                    },
                )
                st.session_state["custom_gpt_kg_cypher_result"] = out
                st.success("Cypher query completed.")
            except Exception as exc:
                st.error(f"Cypher workbench failed: {exc}")
        cypher_result = st.session_state.get("custom_gpt_kg_cypher_result")
        if isinstance(cypher_result, dict):
            tabs = st.tabs(["Graph", "Table", "Raw"])
            with tabs[0]:
                summary = cypher_result.get("summary", {}) if isinstance(cypher_result.get("summary"), dict) else {}
                t1, t2, t3 = st.columns(3)
                t1.metric("Rows", str(summary.get("row_count", 0)))
                t2.metric("Nodes", str(summary.get("node_count", 0)))
                t3.metric("Edges", str(summary.get("edge_count", 0)))
                if cypher_result.get("nodes"):
                    _render_cytoscape(
                        {"nodes": cypher_result.get("nodes", []), "edges": cypher_result.get("edges", [])},
                        layout="cose",
                        max_nodes=int(graph_max_nodes),
                        max_edges=int(graph_max_edges),
                        height=540,
                    )
                else:
                    st.info("This Cypher result did not return graph-shaped data to visualize.")
            with tabs[1]:
                rows = cypher_result.get("rows", [])
                if rows:
                    st.dataframe(rows, use_container_width=True)
                else:
                    st.info("No rows returned.")
            with tabs[2]:
                st.json(cypher_result)

    auto_intent_payload = {
        "query": query,
        "use_structured": use_structured,
        "use_docs": use_docs,
        "use_kg": use_kg,
        "action_mode": action_mode,
        "connector": selected_connector if selected_connector else None,
        "dataset_id": unstructured_dataset_id.strip() or None,
        "graph_dataset_id": graph_dataset_id.strip() or None,
        "llm_provider": "ollama_local",
    }
    auto_intent_key = json.dumps(auto_intent_payload, sort_keys=True)
    last_auto_intent_key = st.session_state.get("custom_gpt_intent_key")
    if query.strip() and auto_intent_key != last_auto_intent_key:
        try:
            intent = _classify_intent(base_url, headers, auto_intent_payload)
            st.session_state["custom_gpt_intent"] = intent
            st.session_state["custom_gpt_intent_key"] = auto_intent_key
        except Exception as exc:
            st.session_state["custom_gpt_intent_error"] = str(exc)
    elif not query.strip():
        st.session_state.pop("custom_gpt_intent", None)
        st.session_state.pop("custom_gpt_intent_key", None)
        st.session_state.pop("custom_gpt_intent_error", None)

    intent_error = st.session_state.get("custom_gpt_intent_error")
    if intent_error and query.strip():
        st.warning(f"Intent routing fallback in effect: {intent_error}")
    if st.session_state.get("custom_gpt_intent"):
        _render_intent_summary(st.session_state["custom_gpt_intent"])

    _render_structured_execution_help()
    c1, c2 = st.columns(2)
    use_current_preview = c1.toggle(
        "Use Current Structured Preview Data",
        value=isinstance(st.session_state.get("structured_df"), object) and st.session_state.get("structured_df") is not None,
        key="custom_gpt_use_structured_preview",
    )
    use_latest_run = c2.toggle(
        "Use Latest Structured Pipeline Result",
        value=isinstance(st.session_state.get("structured_result"), dict),
        key="custom_gpt_use_latest_structured_result",
    )
    use_connector_source = st.toggle("Use Connector Source for Execution", value=False, key="custom_gpt_use_connector_source")
    force_sql = st.toggle("Force SQL Route", value=False, key="custom_gpt_force_sql")
    include_debug = st.toggle("Include Debug Payload", value=True, key="custom_gpt_include_debug")
    st.caption("For connector-backed structured runs, we still support connector query, table, and limit under the hood, but the main UI now keeps the common cases readable.")

    if st.button("Run Custom GPT Query", use_container_width=True):
        try:
            intent = st.session_state.get("custom_gpt_intent", {}) if isinstance(st.session_state.get("custom_gpt_intent"), dict) else {}
            parsed_scope = dict(source_scope)
            connector_query = parsed_scope.pop("query", None)
            connector_table = parsed_scope.pop("table", None)
            connector_limit = int(parsed_scope.pop("limit", 500))
            dataframe_payload = None
            structured_df = st.session_state.get("structured_df")
            if use_current_preview and hasattr(structured_df, "to_dict"):
                dataframe_payload = structured_df.replace({None: None}).to_dict(orient="records")
            latest_structured_result = st.session_state.get("structured_result") if use_latest_run else None
            payload = {
                "query": query,
                "route": intent.get("route"),
                "use_structured": use_structured,
                "use_docs": use_docs,
                "use_kg": use_kg,
                "action_mode": action_mode,
                "connector": selected_connector if use_connector_source and selected_connector else None,
                "connector_config": parsed_scope if use_connector_source else {},
                "connector_query": connector_query if use_connector_source else None,
                "connector_table": connector_table if use_connector_source else None,
                "connector_limit": connector_limit if use_connector_source else 500,
                "dataset_id": unstructured_dataset_id.strip() or None,
                "graph_dataset_id": graph_dataset_id.strip() or None,
                "top_k": 8,
                "dataframe": dataframe_payload,
                "latest_structured_result": latest_structured_result,
                "llm_provider": "ollama_local",
                "force_sql": force_sql,
                "include_debug": include_debug,
            }
            result = _run_custom_gpt_query(base_url, headers, payload)
            st.session_state["custom_gpt_query_result"] = result
            st.session_state.pop("custom_gpt_kg_graph_error", None)
            route = str(result.get("route", "") or "")
            active_graph_dataset = graph_dataset_id.strip() or None
            if route in {"knowledge_graph", "hybrid"} and active_graph_dataset:
                seed = _best_graph_seed(result)
                if seed:
                    try:
                        subgraph = _kg_subgraph(
                            base_url,
                            headers,
                            {
                                "dataset_id": active_graph_dataset,
                                "seed_entity": seed,
                                "hops": int(graph_hops),
                                "limit": max(int(graph_max_nodes), int(graph_max_edges), 40),
                            },
                        )
                        st.session_state["custom_gpt_kg_subgraph"] = subgraph
                        st.session_state["custom_gpt_kg_auto_seed"] = seed
                    except Exception as graph_exc:
                        st.session_state["custom_gpt_kg_subgraph"] = None
                        st.session_state["custom_gpt_kg_graph_error"] = str(graph_exc)
                else:
                    st.session_state["custom_gpt_kg_subgraph"] = None
                    st.session_state["custom_gpt_kg_graph_error"] = "No strong graph entity could be extracted from this question."
            else:
                st.session_state["custom_gpt_kg_subgraph"] = None
                st.session_state.pop("custom_gpt_kg_auto_seed", None)
            st.success(f"Custom GPT query completed via route `{result.get('route', 'unknown')}`.")
        except Exception as exc:
            st.error(f"Custom GPT query failed: {exc}")

    result = st.session_state.get("custom_gpt_query_result")
    if isinstance(result, dict):
        st.markdown("### Answer")
        st.write(result.get("answer", ""))
        c1, c2, c3 = st.columns(3)
        c1.metric("Route", str(result.get("route", "N/A")))
        c2.metric("Mode", str(result.get("mode", "N/A")))
        c3.metric("Confidence", f"{float(result.get('confidence', 0.0)):.2f}")
        _render_answer_explanation(
            result,
            selected_connector=selected_connector,
            use_current_preview=bool(use_current_preview),
            use_latest_run=bool(use_latest_run),
            use_connector_source=bool(use_connector_source),
            force_sql=bool(force_sql),
            include_debug=bool(include_debug),
        )

        governance = result.get("governance", {}) if isinstance(result.get("governance"), dict) else {}
        monitoring = result.get("monitoring", {}) if isinstance(result.get("monitoring"), dict) else {}
        audit = result.get("audit", {}) if isinstance(result.get("audit"), dict) else {}
        if governance or monitoring or audit:
            st.markdown("### Policy, Audit, And Monitoring")
            g1, g2, g3, g4 = st.columns(4)
            g1.metric("Action Requested", "Yes" if governance.get("action_requested") else "No")
            g2.metric("Risk Level", str(governance.get("risk_level", "none")).upper())
            g3.metric("Approval", "Required" if governance.get("requires_human_approval") else "Not Required")
            g4.metric("Latency (ms)", str(monitoring.get("latency_ms", "N/A")))
            if audit.get("trace_id"):
                st.caption(f"Audit Trace ID: `{audit.get('trace_id')}`")

        rows = result.get("rows", [])
        if rows:
            st.markdown("### Result Rows")
            st.dataframe(rows, use_container_width=True)

        sql = result.get("sql")
        if sql:
            st.markdown("### SQL Used")
            st.code(sql, language="sql")

        execution = result.get("execution", {}) if isinstance(result.get("execution"), dict) else {}
        if execution:
            st.markdown("### Execution Summary")
            connector_name = execution.get("connector") or result.get("connector")
            execution_lines = []
            if connector_name:
                execution_lines.append(f"- **Source**: {connector_name}")
            if execution.get("dataset_id"):
                execution_lines.append(f"- **Dataset**: {execution.get('dataset_id')}")
            if execution.get("graph_dataset_id"):
                execution_lines.append(f"- **Graph Dataset**: {execution.get('graph_dataset_id')}")
            if execution.get("anchor"):
                execution_lines.append(f"- **Primary Graph Entity**: {execution.get('anchor')}")
            if execution.get("entities"):
                execution_lines.append(f"- **Matched Graph Entities**: {len(execution.get('entities', []))}")
            if execution_lines:
                st.markdown("\n".join(execution_lines))

        chart_spec = result.get("chart_spec")
        if chart_spec:
            st.markdown("### Suggested Chart")
            st.json(chart_spec)

        citations_text = "No citations yet."
        citations = result.get("citations", [])
        if citations:
            citations_text = "\n".join(
                f"- [{c.get('type', 'source')}] {c.get('title', c.get('source', 'Source'))}: {c.get('detail', '')}"
                for c in citations
            )
        st.markdown("### Citation Panel")
        st.text_area("Citations", value=citations_text, height=140, key="custom_gpt_citations_panel")

        if str(result.get("route", "") or "") in {"knowledge_graph", "hybrid"}:
            st.markdown("### Knowledge Graph")
            auto_seed = st.session_state.get("custom_gpt_kg_auto_seed") or _best_graph_seed(result)
            if auto_seed:
                st.caption(f"Auto-rendered graph neighborhood for `{auto_seed}` based on the entities extracted from your question.")
            graph_error = st.session_state.get("custom_gpt_kg_graph_error")
            graph_subgraph = st.session_state.get("custom_gpt_kg_subgraph")
            if isinstance(graph_subgraph, dict) and graph_subgraph.get("nodes"):
                node_count = len(graph_subgraph.get("nodes", []) or [])
                edge_count = len(graph_subgraph.get("edges", []) or [])
                g1, g2, g3 = st.columns(3)
                g1.metric("Nodes", str(min(node_count, int(graph_max_nodes))))
                g2.metric("Edges", str(min(edge_count, int(graph_max_edges))))
                g3.metric("Graph Hops", str(graph_hops))
                _render_cytoscape(
                    graph_subgraph,
                    layout="cose",
                    max_nodes=int(graph_max_nodes),
                    max_edges=int(graph_max_edges),
                )
            elif graph_error:
                st.info(f"Graph rendering skipped: {graph_error}")
            else:
                st.info("No graph neighborhood was available for this answer yet.")

        with st.expander("Execution + Debug", expanded=False):
            reasoning = result.get("reasoning")
            if reasoning:
                st.markdown(f"**Reasoning**: {reasoning}")
            _render_list_block(result.get("warnings", []), "Warnings")
            _render_list_block(result.get("sources_used", []), "Sources Used")
            _render_key_value_block(result.get("capabilities", {}), "Capabilities Used")
            _render_key_value_block(result.get("execution", {}), "Execution Details")
            debug = result.get("debug", {})
            if isinstance(debug, dict) and debug:
                with st.expander("Raw Debug Payload", expanded=False):
                    st.json(debug)

        with st.expander("Offline Eval + Production Metrics", expanded=False):
            eval_c1, eval_c2 = st.columns(2)
            with eval_c1:
                if st.button("Refresh Production Metrics", use_container_width=True):
                    try:
                        st.session_state["custom_gpt_prod_metrics"] = _get_custom_gpt_metrics(base_url, headers, limit=200)
                        st.success("Production metrics refreshed.")
                    except Exception as exc:
                        st.error(f"Metrics refresh failed: {exc}")
            with eval_c2:
                if st.button("Run DeepEval On Recent Traces", use_container_width=True):
                    try:
                        st.session_state["custom_gpt_eval_run"] = _run_custom_gpt_eval(
                            base_url,
                            headers,
                            {"limit": 15, "model_name": "ollama_local", "metrics": ["faithfulness", "answer_relevancy"]},
                        )
                        st.success("Custom GPT eval completed.")
                    except Exception as exc:
                        st.error(f"Eval run failed: {exc}")
            if isinstance(st.session_state.get("custom_gpt_prod_metrics"), dict):
                _render_monitoring_summary(st.session_state["custom_gpt_prod_metrics"])
            if isinstance(st.session_state.get("custom_gpt_eval_run"), dict):
                st.markdown("### DeepEval")
                _render_eval_summary(st.session_state["custom_gpt_eval_run"])
    else:
        st.markdown("### Citation Panel")
        st.text_area("Citations", value="No citations yet.", height=140, key="custom_gpt_citations_panel")
