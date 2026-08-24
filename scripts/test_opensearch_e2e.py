#!/usr/bin/env python3
"""End-to-end smoke tests for the OpenSearch tab (backend APIs).

Usage:
  cd /path/to/Agentic-auto-ml
  .venv/bin/python scripts/test_opensearch_e2e.py

Optional env:
  BACKEND_URL=http://127.0.0.1:8000
  OPENSEARCH_URL=http://127.0.0.1:9200
  COLLECTION=e2e_test
"""
from __future__ import annotations

import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

try:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
    load_dotenv(ROOT / "backend" / ".env", override=True)
except Exception:
    pass

BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000").rstrip("/")
OPENSEARCH_URL = os.getenv("OPENSEARCH_URL", "http://127.0.0.1:9200").rstrip("/")
COLLECTION = os.getenv("COLLECTION", "e2e_test")
SEARCH_QUERY = "music streaming trends 2025"
INGEST_URL = "https://www.siriusxmmedia.com/insights/streaming-industry-trends-whats-in-the-forecast-for-2025"

passed = 0
failed = 0
skipped = 0


def ok(name: str, detail: str = "") -> None:
    global passed
    passed += 1
    suffix = f" — {detail}" if detail else ""
    print(f"  PASS  {name}{suffix}")


def fail(name: str, detail: str) -> None:
    global failed
    failed += 1
    print(f"  FAIL  {name}: {detail}")


def skip(name: str, detail: str) -> None:
    global skipped
    skipped += 1
    print(f"  SKIP  {name}: {detail}")


def request_json(
    method: str,
    path: str,
    body: dict[str, Any] | None = None,
    timeout: int = 120,
) -> tuple[int, Any]:
    url = f"{BACKEND_URL}{path}"
    data = None
    headers = {"Accept": "application/json"}
    if body is not None:
        data = json.dumps(body).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read().decode("utf-8")
        return resp.status, json.loads(raw) if raw else None


def ping(url: str, timeout: int = 5) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return 200 <= resp.status < 300
    except Exception:
        return False


def section(title: str) -> None:
    print(f"\n== {title} ==")


def test_prerequisites() -> bool:
    section("0. Prerequisites")
    ready = True

    if ping(f"{BACKEND_URL}/health/ping"):
        ok("Backend reachable", BACKEND_URL)
    else:
        fail("Backend reachable", f"Cannot reach {BACKEND_URL}/health/ping")
        ready = False

    if ping(OPENSEARCH_URL):
        ok("OpenSearch cluster reachable", OPENSEARCH_URL)
    else:
        fail("OpenSearch cluster reachable", f"Cannot reach {OPENSEARCH_URL}")
        ready = False

    tavily_key = os.getenv("TAVILY_API_KEY", "").strip()
    if tavily_key:
        ok("TAVILY_API_KEY set", f"length={len(tavily_key)}")
    else:
        fail("TAVILY_API_KEY set", "Add TAVILY_API_KEY to .env and restart backend")
        ready = False

    if not ping(f"{BACKEND_URL.replace(':8000', ':11434')}/api/tags", timeout=3):
        skip("Ollama reachable", "Optional for LLM answers; chat may still return links without it")
    else:
        ok("Ollama reachable", "LLM answers should stream")

    return ready


def test_status() -> None:
    section("1. GET /opensearch/status")
    try:
        status, data = request_json("GET", "/opensearch/status")
    except Exception as exc:
        fail("status endpoint", str(exc))
        return

    if status != 200:
        fail("status endpoint", f"HTTP {status}")
        return
    ok("status endpoint", "HTTP 200")

    if data.get("available"):
        ok("OpenSearch available")
    else:
        fail("OpenSearch available", "available=false — check OPENSEARCH_URL and restart backend")

    providers = data.get("available_providers") or []
    if "tavily" in providers:
        ok("Tavily provider configured", "tavily in available_providers")
    else:
        fail("Tavily provider configured", f"available_providers={providers}")

    if data.get("mcp_ready"):
        ok("MCP ready")
    else:
        fail("MCP ready", "mcp_ready=false")


def test_live_search() -> None:
    section("2. POST /opensearch/web/search (Tavily live)")
    payload = {
        "query": SEARCH_QUERY,
        "collection": COLLECTION,
        "provider": "tavily",
        "top_k": 5,
        "auto_fetch_and_index": False,
        "llm_provider": "ollama_local",
    }
    try:
        status, data = request_json("POST", "/opensearch/web/search", payload, timeout=90)
    except Exception as exc:
        fail("web search", str(exc))
        return

    if status != 200:
        fail("web search", f"HTTP {status}")
        return
    ok("web search", "HTTP 200")

    warnings = data.get("warnings") or []
    bad = [w for w in warnings if "tavily" in str(w).lower() and "not configured" in str(w).lower()]
    if bad:
        fail("no Tavily config warnings", "; ".join(bad))
    else:
        ok("no Tavily config warnings")

    hits = data.get("hits") or []
    live = data.get("live_results") or []
    links = hits or live
    if len(links) >= 1:
        ok("search returned links", f"count={len(links)} first={links[0].get('url', '')[:72]}")
    else:
        fail("search returned links", f"hits={len(hits)} live_results={len(live)} warnings={warnings}")

    if data.get("provider") == "tavily":
        ok("provider is tavily")
    else:
        fail("provider is tavily", f"provider={data.get('provider')}")


def test_ingest() -> None:
    section("3. POST /opensearch/web/ingest (manual URL)")
    payload = {"urls": [INGEST_URL], "collection": COLLECTION, "llm_provider": "ollama_local"}
    try:
        status, data = request_json("POST", "/opensearch/web/ingest", payload, timeout=90)
    except Exception as exc:
        fail("web ingest", str(exc))
        return

    if status != 200:
        fail("web ingest", f"HTTP {status}")
        return

    indexed = int(data.get("indexed", 0) or 0)
    if indexed >= 1:
        ok("URL indexed", f"indexed={indexed}")
    else:
        fail("URL indexed", f"indexed={indexed} warnings={data.get('warnings')}")


def test_chat_sync() -> str | None:
    section("4. POST /opensearch/chat (sync)")
    payload = {
        "query": SEARCH_QUERY,
        "collection": COLLECTION,
        "provider": "tavily",
        "top_k": 5,
        "auto_fetch_and_index": True,
        "llm_provider": "ollama_local",
    }
    try:
        status, data = request_json("POST", "/opensearch/chat", payload, timeout=180)
    except Exception as exc:
        fail("chat sync", str(exc))
        return None

    if status != 200:
        fail("chat sync", f"HTTP {status}")
        return None
    ok("chat sync", "HTTP 200")

    thread_id = data.get("thread_id")
    if thread_id:
        ok("thread_id returned", str(thread_id)[:16])
    else:
        fail("thread_id returned", "missing thread_id")

    warnings = data.get("warnings") or []
    if any("not configured" in str(w).lower() for w in warnings):
        fail("chat has no provider warnings", str(warnings))
    else:
        ok("chat has no provider warnings")

    hits = data.get("hits") or []
    live = data.get("live_results") or []
    if len(hits) + len(live) >= 1:
        ok("chat grounded with links", f"hits={len(hits)} live={len(live)}")
    else:
        fail("chat grounded with links", "Model will hallucinate without links")

    answer = str(data.get("answer") or "").strip()
    if answer:
        ok("chat answer non-empty", f"chars={len(answer)}")
    else:
        fail("chat answer non-empty", "empty answer")

    generic_markers = [
        "no specific data or predictions",
        "it would be best to consult recent industry reports",
    ]
    if any(marker in answer.lower() for marker in generic_markers) and not hits and not live:
        fail("answer not generic fallback", "Looks like LLM guessed without web results")
    else:
        ok("answer not generic fallback")

    return str(thread_id) if thread_id else None


def test_chat_stream() -> str | None:
    section("5. POST /opensearch/chat/stream (NDJSON)")
    payload = {
        "query": "What is Spotify's market position in streaming?",
        "collection": COLLECTION,
        "provider": "tavily",
        "top_k": 4,
        "auto_fetch_and_index": True,
        "llm_provider": "ollama_local",
    }
    url = f"{BACKEND_URL}/opensearch/chat/stream"
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "Accept": "application/x-ndjson"},
        method="POST",
    )
    thread_id: str | None = None
    chunks: list[str] = []
    saw_routing = False
    result_payload: dict[str, Any] | None = None
    error_message: str | None = None

    try:
        with urllib.request.urlopen(req, timeout=180) as resp:
            for raw_line in resp:
                line = raw_line.decode("utf-8").strip()
                if not line:
                    continue
                event = json.loads(line)
                event_type = event.get("event") or event.get("type")
                if event_type == "model_routing":
                    saw_routing = True
                elif event_type == "answer_chunk":
                    chunks.append(str(event.get("text") or ""))
                elif event_type == "result":
                    data = event.get("data") or event
                    if isinstance(data, dict):
                        result_payload = data
                        thread_id = str(data.get("thread_id") or thread_id or "")
                elif event_type == "error":
                    error_message = str(event.get("message") or event)
    except Exception as exc:
        fail("chat stream", str(exc))
        return None

    if error_message:
        fail("chat stream no error event", error_message)
        return None
    ok("chat stream no error event")

    if saw_routing:
        ok("stream model_routing event")
    else:
        fail("stream model_routing event", "not received")

    if chunks:
        ok("stream answer chunks", f"chunks={len(chunks)}")
    else:
        skip("stream answer chunks", "Ollama may be down; result event still checked")

    if result_payload:
        ok("stream result event")
        hits = result_payload.get("hits") or []
        live = result_payload.get("live_results") or []
        if len(hits) + len(live) >= 1:
            ok("stream result has links", f"hits={len(hits)} live={len(live)}")
        else:
            fail("stream result has links", str(result_payload.get("warnings")))
        thread_id = str(result_payload.get("thread_id") or thread_id or "") or None
    else:
        fail("stream result event", "missing")

    return thread_id


def test_threads_and_history(thread_id: str | None) -> None:
    section("6. GET /opensearch/threads + /opensearch/history")
    try:
        status, data = request_json("GET", f"/opensearch/threads?collection={COLLECTION}")
    except Exception as exc:
        fail("threads list", str(exc))
        return

    if status == 200:
        threads = data.get("threads") or []
        ok("threads list", f"count={len(threads)}")
    else:
        fail("threads list", f"HTTP {status}")

    if not thread_id:
        skip("thread history", "no thread_id from chat tests")
        return

    try:
        status, data = request_json(
            "GET",
            f"/opensearch/history?thread_id={thread_id}&collection={COLLECTION}",
        )
    except Exception as exc:
        fail("thread history", str(exc))
        return

    if status != 200:
        fail("thread history", f"HTTP {status}")
        return

    turns = data.get("turns") or []
    messages = data.get("messages") or []
    if turns or messages:
        ok("thread history has content", f"turns={len(turns)} messages={len(messages)}")
    else:
        fail("thread history has content", "empty history after chat")


def test_mcp() -> None:
    section("7. MCP tools bridge")
    try:
        status, data = request_json("GET", "/opensearch/mcp/tools")
    except Exception as exc:
        fail("mcp tools list", str(exc))
        return

    if status == 200 and data.get("tools"):
        ok("mcp tools list", f"count={len(data['tools'])}")
    else:
        fail("mcp tools list", f"HTTP {status} tools={data.get('tools')}")

    payload = {
        "tool": "web_search",
        "arguments": {
            "query": SEARCH_QUERY,
            "collection": COLLECTION,
            "top_k": 3,
            "provider": "tavily",
            "auto_fetch_and_index": False,
        },
    }
    try:
        status, data = request_json("POST", "/opensearch/mcp/call", payload, timeout=90)
    except Exception as exc:
        fail("mcp web_search", str(exc))
        return

    if status == 200 and data.get("ok"):
        result = data.get("result") or {}
        results = result.get("results") or result.get("live_results") or result.get("hits") or []
        if results:
            ok("mcp web_search", f"results={len(results)}")
        else:
            fail("mcp web_search", f"empty result: {result}")
    else:
        fail("mcp web_search", str(data))


def main() -> int:
    print("OpenSearch tab — end-to-end API test")
    print(f"Backend:    {BACKEND_URL}")
    print(f"OpenSearch: {OPENSEARCH_URL}")
    print(f"Collection: {COLLECTION}")

    if not test_prerequisites():
        print("\nFix prerequisites before continuing.")
        return 1

    test_status()
    test_live_search()
    test_ingest()
    thread_sync = test_chat_sync()
    thread_stream = test_chat_stream()
    test_threads_and_history(thread_sync or thread_stream)
    test_mcp()

    section("Summary")
    total = passed + failed + skipped
    print(f"  {passed} passed, {failed} failed, {skipped} skipped (of {total} checks)")
    if failed:
        print("\nSome checks failed. Common fixes:")
        print("  1. Restart backend: ./scripts/run_backend_local.sh")
        print("  2. Ensure .env has TAVILY_API_KEY and OPENSEARCH_URL")
        print("  3. Start OpenSearch Docker on :9200")
        print("  4. Start Ollama for LLM streaming answers")
        return 1

    print("\nAll critical checks passed. Run the UI checklist below in the browser.")
    print_ui_checklist()
    return 0


def print_ui_checklist() -> None:
    print(
        """
--- Manual UI checklist (OpenSearch tab) ---

1. Open http://127.0.0.1:5173 → OpenSearch tab
2. Sidebar status: healthy + MCP ready banner visible
3. Provider dropdown: select "tavily"
4. Collection: "default" or "e2e_test"
5. Auto-fetch: ON
6. Ask: "What are the latest trends in music streaming in 2025?"
   EXPECT: streaming answer + source links (not generic bullet list)
7. Links panel shows URLs from Tavily (siriusxm, explodingtopics, etc.)
8. Paste a URL in sidebar → Index URLs → "Indexed N document(s)"
9. Thread dropdown: prior conversation appears after chat
10. Switch thread → history reloads user + assistant messages

If API tests pass but UI shows blank links, hard-refresh the frontend (Cmd+Shift+R).
"""
    )


if __name__ == "__main__":
    raise SystemExit(main())
