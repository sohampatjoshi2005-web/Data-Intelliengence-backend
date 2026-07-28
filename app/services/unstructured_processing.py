from __future__ import annotations

import re
from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any, Dict, List

from app.core.llm_clients import LLMRouter

try:
    from markitdown import MarkItDown
except Exception:
    MarkItDown = None

try:
    from presidio_analyzer import AnalyzerEngine
    from presidio_analyzer.nlp_engine import NlpEngineProvider
    from presidio_anonymizer import AnonymizerEngine
except Exception:
    AnalyzerEngine = None
    NlpEngineProvider = None
    AnonymizerEngine = None


_PII_ANALYZER = None
_PII_ANONYMIZER = None


@dataclass
class EnrichedChunk:
    chunk_id: str
    text: str
    summary: str
    questions: List[str]
    headings: List[str]
    entities: List[List[str]]
    relationships: List[List[str]]
    sentiment: float


def convert_to_markdown(file_path: str) -> str:
    if MarkItDown is None:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    return MarkItDown().convert(file_path).text_content


def anonymize_text(text: str) -> str:
    global _PII_ANALYZER, _PII_ANONYMIZER
    if AnalyzerEngine is None or AnonymizerEngine is None:
        return text
    try:
        if _PII_ANALYZER is None or _PII_ANONYMIZER is None:
            if NlpEngineProvider is not None:
                provider = NlpEngineProvider(
                    nlp_configuration={
                        "nlp_engine_name": "spacy",
                        "models": [{"lang_code": "en", "model_name": "en_core_web_sm"}],
                    }
                )
                nlp_engine = provider.create_engine()
                _PII_ANALYZER = AnalyzerEngine(nlp_engine=nlp_engine, supported_languages=["en"])
            else:
                _PII_ANALYZER = AnalyzerEngine()
            _PII_ANONYMIZER = AnonymizerEngine()

        findings = _PII_ANALYZER.analyze(text=text, language="en")
        return _PII_ANONYMIZER.anonymize(text=text, analyzer_results=findings).text
    except Exception:
        return text


def extract_headings(markdown: str) -> List[str]:
    headings = re.findall(r"^#{1,6}\s+(.+)$", markdown, flags=re.MULTILINE)
    return headings[:200]


def chunk_text(markdown: str, chunk_size: int = 1200, overlap: int = 200) -> List[str]:
    text = markdown.strip()
    if not text:
        return []
    chunks = []
    step = max(chunk_size - overlap, 1)
    for i in range(0, len(text), step):
        chunks.append(text[i : i + chunk_size])
    return chunks


def _extract_with_llm(router: LLMRouter, provider: str, text_chunk: str) -> Dict[str, Any]:
    prompt = f"""
Extract structured metadata from text for RAG enrichment.
Return JSON with keys:
entities: [[name, type]]
relationships: [[subject, predicate, object]]
summary: short summary
questions: [q1, q2, q3]
sentiment: number -1 to 1
TEXT:\n{text_chunk[:2200]}
"""
    response = router.complete(prompt, provider=provider)

    # Keep robust even when provider returns non-JSON.
    try:
        import json

        parsed = json.loads(response)
        return {
            "entities": parsed.get("entities", []),
            "relationships": parsed.get("relationships", []),
            "summary": parsed.get("summary", ""),
            "questions": parsed.get("questions", []),
            "sentiment": float(parsed.get("sentiment", 0.0)),
        }
    except Exception:
        return {
            "entities": [],
            "relationships": [],
            "summary": text_chunk[:220].replace("\n", " "),
            "questions": [
                "What is the core claim in this chunk?",
                "Which entities are most important here?",
            ],
            "sentiment": 0.0,
        }


def enrich_chunks(markdown: str, provider: str = "openai") -> List[EnrichedChunk]:
    headings = extract_headings(markdown)
    max_chunks = int(os.getenv("UNSTRUCTURED_MAX_CHUNKS", "30"))
    chunks = chunk_text(markdown)[:max_chunks]
    router = LLMRouter()
    out: List[EnrichedChunk] = []

    for idx, chunk in enumerate(chunks):
        meta = _extract_with_llm(router, provider, chunk)
        out.append(
            EnrichedChunk(
                chunk_id=f"chunk_{idx}",
                text=chunk,
                summary=meta.get("summary", ""),
                questions=meta.get("questions", []),
                headings=headings,
                entities=meta.get("entities", []),
                relationships=meta.get("relationships", []),
                sentiment=float(meta.get("sentiment", 0.0)),
            )
        )
    return out


def save_temp_upload(filename: str, payload: bytes, tmp_dir: str = "/tmp") -> str:
    out = Path(tmp_dir) / filename
    out.write_bytes(payload)
    return str(out)
