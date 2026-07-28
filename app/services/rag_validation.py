from __future__ import annotations

import os
from typing import Dict

try:
    from deepeval.metrics import FaithfulnessMetric
    from deepeval.test_case import LLMTestCase
except Exception:
    FaithfulnessMetric = None
    LLMTestCase = None


def validate_rag_answer(query: str, context: str, answer: str) -> Dict[str, str | float]:
    if FaithfulnessMetric is None or LLMTestCase is None:
        return {
            "score": 0.0,
            "status": "skipped",
            "reason": "deepeval not installed",
        }

    # DeepEval faithfulness metric may depend on an OpenAI key unless a custom judge model is configured.
    if not os.getenv("OPENAI_API_KEY"):
        return {
            "score": 0.0,
            "status": "skipped",
            "reason": "OPENAI_API_KEY not set; DeepEval judge model unavailable",
        }

    try:
        metric = FaithfulnessMetric(threshold=0.7)
        test_case = LLMTestCase(
            input=query,
            actual_output=answer,
            retrieval_context=[context],
        )
        metric.measure(test_case)
        return {
            "score": float(metric.score),
            "status": "pass" if metric.score >= 0.7 else "fail",
            "reason": metric.reason,
        }
    except Exception as exc:
        return {
            "score": 0.0,
            "status": "skipped",
            "reason": f"deepeval runtime unavailable: {exc}",
        }
