from __future__ import annotations

import hashlib
import json
import pickle
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

import pandas as pd

try:
    import joblib
except Exception:
    joblib = None


def _hash_dataset(df: pd.DataFrame) -> str:
    digest = hashlib.sha256()
    digest.update(str(df.shape).encode("utf-8"))
    digest.update("|".join([str(c) for c in df.columns]).encode("utf-8"))
    sample = df.head(min(200, len(df))).to_csv(index=False)
    digest.update(sample.encode("utf-8", errors="ignore"))
    return digest.hexdigest()[:16]


def package_model(
    *,
    model: Any,
    dataset_name: str,
    problem_type: str,
    target: str,
    metrics: Dict[str, Any],
    base_dir: str = "artifacts/models",
) -> Dict[str, Any]:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    safe_name = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in dataset_name)
    model_id = f"{safe_name}_{problem_type}_{ts}"

    out_dir = Path(base_dir) / model_id
    out_dir.mkdir(parents=True, exist_ok=True)

    model_path = out_dir / "model.joblib"
    if joblib is not None:
        joblib.dump(model, model_path)
        serializer = "joblib"
    else:
        model_path = out_dir / "model.pkl"
        with model_path.open("wb") as f:
            pickle.dump(model, f)
        serializer = "pickle"

    metadata = {
        "model_id": model_id,
        "dataset_name": dataset_name,
        "problem_type": problem_type,
        "target": target,
        "created_at_utc": ts,
        "serializer": serializer,
        "metrics": metrics,
    }
    meta_path = out_dir / "metadata.json"
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    card_path = out_dir / "model_card.md"
    card_path.write_text(
        "\n".join(
            [
                f"# Model Card: {model_id}",
                "",
                f"- Dataset: {dataset_name}",
                f"- Problem Type: {problem_type}",
                f"- Target: {target}",
                f"- Created At (UTC): {ts}",
                "",
                "## Key Metrics",
                "```json",
                json.dumps(metrics, indent=2),
                "```",
            ]
        ),
        encoding="utf-8",
    )

    return {
        "model_id": model_id,
        "artifact_dir": str(out_dir),
        "model_path": str(model_path),
        "metadata_path": str(meta_path),
        "model_card_path": str(card_path),
    }
