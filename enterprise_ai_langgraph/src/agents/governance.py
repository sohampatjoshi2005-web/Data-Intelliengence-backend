from __future__ import annotations

from presidio_analyzer import AnalyzerEngine


def governance_agent(text: str, user_name: str, user_phone: str, analyzer: AnalyzerEngine) -> str:
    safe_text = text.replace(user_name, "[NAME]").replace(user_phone, "[PHONE]")
    results = analyzer.analyze(text=safe_text, entities=["EMAIL_ADDRESS", "LOCATION", "CRYPTO"], language="en")
    for res in sorted(results, key=lambda x: x.start, reverse=True):
        safe_text = safe_text[: res.start] + f"[{res.entity_type}]" + safe_text[res.end :]
    return safe_text
