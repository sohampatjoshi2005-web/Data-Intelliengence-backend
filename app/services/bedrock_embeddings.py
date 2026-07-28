from __future__ import annotations

import json
from typing import List

import boto3

from app.core.config import settings


def bedrock_embed_texts(texts: List[str]) -> List[List[float]]:
    model_id = settings.bedrock_embed_model_id
    if not model_id:
        raise RuntimeError("BEDROCK_EMBED_MODEL_ID is not set.")

    client = boto3.client("bedrock-runtime", region_name=settings.bedrock_region)
    out: List[List[float]] = []
    for text in texts:
        payload = {"inputText": text}
        resp = client.invoke_model(
            modelId=model_id,
            body=json.dumps(payload).encode("utf-8"),
            contentType="application/json",
            accept="application/json",
        )
        raw = resp["body"].read().decode("utf-8")
        data = json.loads(raw)
        emb = data.get("embedding") or data.get("embeddings") or []
        out.append([float(x) for x in emb])
    return out
