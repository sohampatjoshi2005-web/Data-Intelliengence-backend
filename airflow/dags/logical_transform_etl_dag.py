from __future__ import annotations

import io
import json
import os
from datetime import datetime
from pathlib import Path

import pandas as pd
from airflow import DAG
from airflow.operators.python import PythonOperator


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _artifacts_dir() -> Path:
    path = _repo_root() / "artifacts" / "etl_outputs"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _extract_dataframe(recipe: dict) -> pd.DataFrame:
    source_type = recipe.get("source_type")
    source = recipe.get("source_config", {})

    if source_type == "upload_file":
        file_path = source.get("file_path")
        if not file_path:
            raise ValueError("source_config.file_path is required for upload_file source_type")
        p = Path(file_path)
        if not p.exists():
            raise FileNotFoundError(f"Source file not found: {file_path}")
        ext = p.suffix.lower()
        if ext == ".csv":
            return pd.read_csv(p)
        if ext in {".xls", ".xlsx"}:
            return pd.read_excel(p)
        if ext == ".json":
            return pd.read_json(p)
        if ext == ".tsv":
            return pd.read_csv(p, sep="\t")
        raise ValueError(f"Unsupported source file extension: {ext}")

    if source_type == "postgres_table":
        from sqlalchemy import create_engine

        host = source.get("host", "127.0.0.1")
        port = int(source.get("port", 5432))
        database = source.get("database", "kbdb")
        username = source.get("username", "postgres")
        password = source.get("password", "")
        table = source.get("table")
        query = source.get("query")
        limit = int(source.get("limit", 10000))
        dsn = source.get("dsn")
        if dsn:
            engine = create_engine(dsn)
        else:
            engine = create_engine(f"postgresql+psycopg2://{username}:{password}@{host}:{port}/{database}")
        sql = query or f"SELECT * FROM {table} LIMIT {limit}"
        return pd.read_sql(sql, engine)

    raise ValueError(f"Unsupported source_type: {source_type}")


def _apply_transforms(df: pd.DataFrame, recipe: dict) -> dict:
    import sys

    sys.path.append(str(_repo_root() / "backend"))
    from app.services.transformations import run_logical_transformation

    payload = io.StringIO()
    df.to_csv(payload, index=False)
    out = run_logical_transformation(
        file_name="source.csv",
        payload=payload.getvalue().encode("utf-8"),
        operations=recipe.get("operations", []),
        right_file_name=None,
        right_payload=None,
    )
    return out


def _load_target(out: dict, recipe: dict, run_id: str) -> dict:
    target = recipe.get("target_config", {}) or {}
    target_type = target.get("type", "csv")
    b64 = out.get("content_base64", "")
    raw = io.BytesIO()
    raw.write(__import__("base64").b64decode(b64.encode("utf-8")))
    raw.seek(0)
    transformed_df = pd.read_csv(raw)

    if target_type == "postgres":
        from sqlalchemy import create_engine

        host = target.get("host", "127.0.0.1")
        port = int(target.get("port", 5432))
        database = target.get("database", "kbdb")
        username = target.get("username", "postgres")
        password = target.get("password", "")
        table = target.get("table", "etl_output")
        mode = target.get("mode", "replace")
        dsn = target.get("dsn")
        if dsn:
            engine = create_engine(dsn)
        else:
            engine = create_engine(f"postgresql+psycopg2://{username}:{password}@{host}:{port}/{database}")
        if_exists = "replace" if mode == "replace" else "append"
        transformed_df.to_sql(table, engine, if_exists=if_exists, index=False)
        return {"target_type": "postgres", "table": table, "rows": int(len(transformed_df))}

    out_path = Path(target.get("file_path") or str(_artifacts_dir() / f"{run_id}_output.csv"))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    transformed_df.to_csv(out_path, index=False)
    return {"target_type": "csv", "file_path": str(out_path), "rows": int(len(transformed_df))}


def run_logical_etl(**context) -> None:
    conf = context.get("dag_run").conf or {}
    run_id = conf.get("run_id") or context.get("run_id") or "manual"
    recipe = conf.get("recipe") or {}
    if not recipe:
        raise ValueError("Missing recipe in dag_run conf")

    df = _extract_dataframe(recipe)
    result = _apply_transforms(df, recipe)
    target_meta = _load_target(result, recipe, run_id)
    summary_path = _artifacts_dir() / f"{run_id}_summary.json"
    summary = {
        "run_id": run_id,
        "row_count": result.get("row_count", 0),
        "warnings": result.get("warnings", []),
        "report": result.get("report", {}),
        "target": target_meta,
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[logical_transform_etl] Completed run: {json.dumps(summary)}", flush=True)


with DAG(
    dag_id="logical_transform_etl",
    description="Execute logical transformation ETL recipe from Streamlit/backend trigger",
    start_date=datetime(2026, 1, 1),
    schedule_interval=None,
    catchup=False,
    max_active_runs=4,
) as dag:
    run_pipeline = PythonOperator(
        task_id="run_logical_etl",
        python_callable=run_logical_etl,
        provide_context=True,
    )
