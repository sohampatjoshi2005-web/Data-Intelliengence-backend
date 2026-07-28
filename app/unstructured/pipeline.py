from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List

from pydantic import BaseModel, Field

from app.core.config import settings
from app.core.llm_clients import LLMRouter
from app.unstructured.schemas import KBChunk, KBState

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

try:
    from langdetect import detect
except Exception:
    detect = None

try:
    import spacy
except Exception:
    spacy = None

try:
    from markdown_it import MarkdownIt
except Exception:
    MarkdownIt = None

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except Exception:
    RecursiveCharacterTextSplitter = None


_NER_NLP = None
_SENT_NLP = None
_PII_ANALYZER = None
_PII_ANONYMIZER = None


def _get_ner_nlp():
    global _NER_NLP
    if spacy is None:
        return None
    if _NER_NLP is not None:
        return _NER_NLP
    try:
        _NER_NLP = spacy.load("en_core_web_sm")
        return _NER_NLP
    except Exception:
        return None


def _get_sent_nlp():
    global _SENT_NLP
    if spacy is None:
        return None
    if _SENT_NLP is not None:
        return _SENT_NLP
    try:
        nlp = spacy.blank("en")
        if "sentencizer" not in nlp.pipe_names:
            nlp.add_pipe("sentencizer")
        _SENT_NLP = nlp
        return _SENT_NLP
    except Exception:
        return None


def _get_pii_engines():
    global _PII_ANALYZER, _PII_ANONYMIZER
    if AnalyzerEngine is None or AnonymizerEngine is None:
        return None, None
    if _PII_ANALYZER is not None and _PII_ANONYMIZER is not None:
        return _PII_ANALYZER, _PII_ANONYMIZER

    try:
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
        return _PII_ANALYZER, _PII_ANONYMIZER
    except Exception:
        return None, None


class EnrichmentPayload(BaseModel):
    metadata: Dict[str, Any] = Field(default_factory=dict)
    tags: List[str] = Field(default_factory=list)
    questions: List[str] = Field(default_factory=list)
    summary: str = ""
    keywords: List[str] = Field(default_factory=list)
    tod_items: List[str] = Field(default_factory=list)
    sentiment: float = 0.0


class EnrichmentBatchPayload(BaseModel):
    items: List[EnrichmentPayload] = Field(default_factory=list)


def _parse_enrichment_payload(raw: str) -> EnrichmentPayload:
    try:
        return EnrichmentPayload.model_validate_json(raw)
    except Exception:
        try:
            return EnrichmentPayload.model_validate(json.loads(raw))
        except Exception:
            return EnrichmentPayload()


def _state_bool(state: KBState, key: str, default: bool = False) -> bool:
    val = state.get(key, default)
    if isinstance(val, bool):
        return val
    if isinstance(val, str):
        return val.strip().lower() in {"1", "true", "yes", "on"}
    return bool(val)


def _state_int(state: KBState, key: str, default: int) -> int:
    try:
        return int(state.get(key, default))
    except Exception:
        return int(default)


def data_acquisition(state: KBState) -> KBState:
    file_path = state.get("file_path")
    if not file_path:
        raise ValueError("file_path is required")
    data = Path(file_path).read_text(encoding="utf-8", errors="ignore")
    # PostgreSQL text/json columns reject NUL bytes.
    data = data.replace("\x00", " ")
    state["raw_text"] = data
    return state


def markdownizer(state: KBState) -> KBState:
    file_path = state["file_path"]
    if MarkItDown is None:
        state["markdown_text"] = state.get("raw_text", "")
        state.setdefault("warnings", []).append("markitdown unavailable; used raw text fallback")
        return state
    try:
        md = MarkItDown().convert(file_path).text_content
        state["markdown_text"] = md
    except Exception as exc:
        state["markdown_text"] = state.get("raw_text", "")
        state.setdefault("warnings", []).append(f"markitdown failed; fallback to raw text ({exc})")
    return state


def language_detection(state: KBState) -> KBState:
    text = state.get("markdown_text", "")[:4000]
    if not text:
        state["language"] = "unknown"
        return state
    if detect is None:
        state["language"] = "en"
        state.setdefault("warnings", []).append("langdetect unavailable; defaulted language=en")
        return state
    try:
        state["language"] = detect(text)
    except Exception:
        state["language"] = "en"
    return state


def data_standardization(state: KBState) -> KBState:
    text = state.get("markdown_text", "")
    text = text.replace("\x00", " ")
    text = text.replace("\u00a0", " ")
    text = text.replace("<URL>", " ")

    # Normalize spacing line-by-line to preserve document structure.
    normalized_lines = [" ".join(line.split()) for line in text.splitlines()]
    state["standardized_text"] = "\n".join(normalized_lines).strip()
    return state


def feature_extraction(state: KBState) -> KBState:
    text = state.get("standardized_text", "")
    metadata = {
        "char_count": len(text),
        "word_count": len(text.split()),
        "line_count": text.count("\n") + 1 if text else 0,
        "language": state.get("language", "unknown"),
    }
    state.setdefault("warnings", [])
    state.setdefault("chunks", [])
    state.setdefault("headings", [])
    # store in state warnings/persistence slots later
    state["persistence"] = {"document_features": metadata}
    return state


def structure_extraction(state: KBState) -> KBState:
    text = state.get("standardized_text", "")
    markdown_text = state.get("markdown_text", "")

    headings: List[str] = []
    if MarkdownIt is not None:
        try:
            tokens = MarkdownIt().parse(markdown_text)
            for i, token in enumerate(tokens):
                if token.type == "heading_open" and i + 1 < len(tokens):
                    inline = tokens[i + 1]
                    if inline.type == "inline":
                        heading = (inline.content or "").strip()
                        if heading:
                            headings.append(heading)
        except Exception:
            headings = []

    if not headings:
        # Heuristic fallback: title-like short lines with mostly alphabetic tokens.
        fallback: List[str] = []
        for line in state.get("raw_text", "").splitlines():
            candidate = " ".join(line.split()).strip()
            if len(candidate) < 6 or len(candidate) > 80:
                continue
            if not candidate[0].isalpha() or not candidate[0].isupper():
                continue
            alpha_chars = sum(1 for c in candidate if c.isalpha())
            if alpha_chars / max(len(candidate), 1) < 0.6:
                continue
            fallback.append(candidate)
        headings = fallback

    entities: List[str] = []
    skip_ner = _state_bool(state, "skip_ner", settings.kb_skip_ner) or _state_bool(state, "fast_mode", settings.kb_fast_mode)
    if not skip_ner:
        nlp = _get_ner_nlp()
        if nlp is not None:
            try:
                # Keep NER bounded for performance.
                doc = nlp(text[:20000])
                entities = sorted({ent.text.strip() for ent in doc.ents if ent.text.strip()})[:300]
            except Exception:
                entities = []
    else:
        state.setdefault("warnings", []).append("NER skipped by performance flags")
    state["headings"] = headings[:200]
    state["entities"] = entities
    return state


def redact_processing(state: KBState) -> KBState:
    text = state.get("standardized_text", "")
    skip_pii = _state_bool(state, "skip_pii", settings.kb_skip_pii)
    if _state_bool(state, "fast_mode", settings.kb_fast_mode) and len(text) > settings.kb_pii_max_chars:
        skip_pii = True
    if skip_pii:
        state["redacted_text"] = text
        state.setdefault("warnings", []).append("PII redaction skipped by performance flags")
        return state
    analyzer, anonymizer = _get_pii_engines()
    if analyzer is None or anonymizer is None:
        state["redacted_text"] = text
        state.setdefault("warnings", []).append("presidio unavailable; PII redaction skipped")
        return state
    try:
        findings = analyzer.analyze(text=text, language="en")
        state["redacted_text"] = anonymizer.anonymize(text=text, analyzer_results=findings).text
    except Exception as exc:
        state["redacted_text"] = text
        state.setdefault("warnings", []).append(f"presidio failure; redaction skipped ({exc})")
    return state


def chunking(state: KBState) -> KBState:
    text = state.get("redacted_text", "")
    if not text.strip():
        state["chunks_text"] = []
        return state

    # Primary path: token-aware recursive splitter (best for embeddings/RAG).
    if RecursiveCharacterTextSplitter is not None:
        try:
            chunk_size = _state_int(state, "chunk_size_tokens", settings.kb_chunk_size_tokens)
            chunk_overlap = _state_int(state, "chunk_overlap_tokens", settings.kb_chunk_overlap_tokens)
            chunk_cap = _state_int(state, "chunk_cap", settings.kb_chunk_cap if _state_bool(state, "fast_mode", settings.kb_fast_mode) else 400)
            splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
                encoding_name="cl100k_base",
                chunk_size=max(chunk_size, 200),
                chunk_overlap=max(min(chunk_overlap, chunk_size - 1), 0),
                separators=["\n\n", "\n", ". ", " ", ""],
            )
            chunks = [c.strip() for c in splitter.split_text(text) if c and c.strip()]
            state["chunks_text"] = chunks[: max(chunk_cap, 1)]
            return state
        except Exception:
            pass

    # Fallback: sentence-aware spaCy chunking.
    chunks: List[str] = []
    max_chars = 1200
    overlap_sentences = 2

    sentences: List[str] = []
    nlp = _get_sent_nlp()
    if nlp is not None:
        try:
            doc = nlp(text)
            sentences = [s.text.strip() for s in doc.sents if s.text and s.text.strip()]
        except Exception:
            sentences = []

    if sentences:
        current: List[str] = []
        current_len = 0
        i = 0
        while i < len(sentences):
            sentence = sentences[i]
            add_len = len(sentence) + (1 if current else 0)
            if current and current_len + add_len > max_chars:
                chunks.append(" ".join(current).strip())
                overlap = current[-overlap_sentences:] if overlap_sentences > 0 else []
                # Drop overlap when it would prevent the next sentence from ever fitting.
                overlap_len = len(" ".join(overlap))
                if overlap and overlap_len + len(sentence) + 1 > max_chars:
                    overlap = []
                current = overlap.copy()
                current_len = len(" ".join(current))
                continue
            current.append(sentence)
            current_len += add_len
            i += 1
        if current:
            chunks.append(" ".join(current).strip())
    else:
        # Fallback only when sentence segmentation is unavailable.
        size = 1200
        overlap = 200
        step = max(size - overlap, 1)
        chunks = [text[i : i + size] for i in range(0, len(text), step) if text[i : i + size].strip()]

    chunk_cap = _state_int(state, "chunk_cap", settings.kb_chunk_cap if _state_bool(state, "fast_mode", settings.kb_fast_mode) else 400)
    state["chunks_text"] = chunks[: max(chunk_cap, 1)]
    return state


def contextualization(state: KBState) -> KBState:
    chunks = state.get("chunks_text", [])
    headings = state.get("headings", [])
    out: List[str] = []
    for i, chunk in enumerate(chunks):
        prev_txt = chunks[i - 1][:220] if i > 0 else ""
        next_txt = chunks[i + 1][:220] if i + 1 < len(chunks) else ""
        heading = headings[min(i, len(headings) - 1)] if headings else ""
        ctx = f"heading: {heading}\nprev: {prev_txt}\nchunk: {chunk}\nnext: {next_txt}"
        out.append(ctx.strip())
    state["contextual_chunks"] = out
    return state


def enrichment(state: KBState) -> KBState:
    router = LLMRouter()
    provider = state.get("llm_provider", "bedrock")
    chunks = state.get("contextual_chunks", [])
    out: List[KBChunk] = []
    skip_enrichment = _state_bool(state, "skip_enrichment", settings.kb_skip_enrichment)
    if skip_enrichment:
        for i, ctx in enumerate(chunks):
            src = state.get("chunks_text", [""])[i] if i < len(state.get("chunks_text", [])) else ""
            out.append(
                KBChunk(
                    chunk_id=f"chunk_{i}",
                    chunk_text=src,
                    contextual_text=ctx,
                    summary=src[:240],
                    tags=[],
                    questions=[],
                    keywords=[],
                    tod_items=[],
                    metadata={"source_type": "unstructured", "intent": "fast_mode"},
                    sentiment=0.0,
                    embedding=[],
                )
            )
        state.setdefault("warnings", []).append("Enrichment skipped by performance flags")
        state["chunks"] = out
        return state

    batch_size = max(_state_int(state, "enrichment_batch_size", settings.kb_enrich_batch_size), 1)
    workers = max(_state_int(state, "enrichment_workers", settings.kb_enrich_workers), 1)

    def _enrich_batch(start_idx: int, batch: List[str]) -> List[tuple[int, EnrichmentPayload]]:
        numbered = "\n\n".join([f"ITEM {i + 1}:\n{txt[:2500]}" for i, txt in enumerate(batch)])
        prompt = (
            "Return strict JSON only with shape: {\"items\": [ ... ]}.\n"
            "Each item must include keys: metadata, tags, questions, summary, keywords, tod_items, sentiment.\n"
            "metadata must include source_type and intent.\n"
            f"TEXT ITEMS:\n{numbered}"
        )
        raw = router.complete(prompt, provider=provider)

        payload_items: List[EnrichmentPayload] = []
        try:
            parsed = EnrichmentBatchPayload.model_validate_json(raw)
            payload_items = parsed.items
        except Exception:
            try:
                parsed = EnrichmentBatchPayload.model_validate(json.loads(raw))
                payload_items = parsed.items
            except Exception:
                payload_items = []

        if not payload_items:
            payload_items = [EnrichmentPayload() for _ in batch]
        if len(payload_items) < len(batch):
            payload_items.extend([EnrichmentPayload() for _ in range(len(batch) - len(payload_items))])
        payload_items = payload_items[: len(batch)]
        return [(start_idx + i, payload_items[i]) for i in range(len(batch))]

    indexed_context = list(enumerate(chunks))
    batch_specs: List[tuple[int, List[str]]] = []
    for i in range(0, len(indexed_context), batch_size):
        part = indexed_context[i : i + batch_size]
        batch_specs.append((part[0][0], [ctx for _, ctx in part]))

    results: Dict[int, EnrichmentPayload] = {}
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(_enrich_batch, start, batch) for start, batch in batch_specs]
        for fut in as_completed(futures):
            try:
                for idx, payload in fut.result():
                    results[idx] = payload
            except Exception:
                continue

    for i, ctx in enumerate(chunks):
        parsed = results.get(i, EnrichmentPayload())
        out.append(
            KBChunk(
                chunk_id=f"chunk_{i}",
                chunk_text=state.get("chunks_text", [""])[i] if i < len(state.get("chunks_text", [])) else "",
                contextual_text=ctx,
                summary=str(parsed.summary)[:800],
                tags=[str(x) for x in parsed.tags][:25],
                questions=[str(x) for x in parsed.questions][:20],
                keywords=[str(x) for x in parsed.keywords][:30],
                tod_items=[str(x) for x in parsed.tod_items][:20],
                metadata=parsed.metadata if isinstance(parsed.metadata, dict) else {},
                sentiment=float(parsed.sentiment),
                embedding=[],
            )
        )

    state["chunks"] = out
    return state


def gen_embeddings(state: KBState) -> KBState:
    from app.services.bedrock_embeddings import bedrock_embed_texts

    chunks = state.get("chunks", [])
    if not chunks:
        state["chunks"] = []
        return state

    batch_size = max(_state_int(state, "embedding_batch_size", settings.kb_embed_batch_size), 1)
    workers = max(_state_int(state, "embedding_workers", settings.kb_embed_workers), 1)

    inputs: List[str] = []
    for ch in chunks:
        emb_input = " ".join(
            [
                ch.chunk_text,
                ch.summary,
                " ".join(ch.tags),
                " ".join(ch.keywords),
                " ".join(ch.questions),
            ]
        ).strip()[:6000]
        inputs.append(emb_input)

    def _embed_batch(start_idx: int, batch_inputs: List[str]) -> List[tuple[int, List[float]]]:
        try:
            embeddings = bedrock_embed_texts(batch_inputs)
            out = []
            for offset, emb in enumerate(embeddings):
                out.append((start_idx + offset, [float(x) for x in emb]))
            return out
        except Exception:
            return [(start_idx + offset, []) for offset in range(len(batch_inputs))]

    batch_specs: List[tuple[int, List[str]]] = []
    for i in range(0, len(inputs), batch_size):
        batch_specs.append((i, inputs[i : i + batch_size]))

    vectors: Dict[int, List[float]] = {}
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(_embed_batch, start, batch_inputs) for start, batch_inputs in batch_specs]
        for fut in as_completed(futures):
            for idx, vec in fut.result():
                vectors[idx] = vec

    for i, ch in enumerate(chunks):
        ch.embedding = vectors.get(i, [])
    state["chunks"] = chunks
    return state
