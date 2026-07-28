"""
Celery tasks for async job execution (Phase 2 Implementation)

This module handles long-running operations asynchronously:
- Structured ML pipeline orchestration
- Knowledge base building
- Analytics processing

Expected improvements:
- Client doesn't hang waiting for results
- Can process multiple jobs concurrently
- Real-time progress updates via WebSocket
- Better resource utilization
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict

from celery import Celery, Task
from celery.result import AsyncResult
import pandas as pd

from app.core.config import settings
from app.services.data_loader import load_dataframe_from_upload
from app.services.preprocessing import apply_preprocessing
from app.graph.state import AgentState
from app.graph.workflow import build_workflow, run_sequential_fallback

# Configure Celery
celery_app = Celery(
    "automl_worker",
    broker=settings.celery_broker_url or "redis://localhost:6379/0",
    backend=settings.celery_backend_url or "redis://localhost:6379/0",
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_time_limit=30 * 60,  # 30-minute hard limit
    task_soft_time_limit=25 * 60,  # 25-minute soft limit
    worker_max_tasks_per_child=100,
    result_expires=86400,  # Results expire after 24 hours
)

logger = logging.getLogger(__name__)


class CallbackTask(Task):
    """Task with WebSocket callback support"""
    
    def on_success(self, retval, task_id, args, kwargs):
        """Success callback"""
        logger.info(f"Task {task_id} completed successfully")
        
    def on_failure(self, exc, task_id, args, kwargs, einfo):
        """Failure callback"""
        logger.error(f"Task {task_id} failed: {exc}")


@celery_app.task(bind=True, base=CallbackTask, name="tasks.orchestrate_pipeline")
def orchestrate_pipeline(
    self,
    filename: str,
    payload_base64: str,
    business_problem: str = "",
    target_column: str = "",
    model_family: str = "",
    fixed_model: str = "",
    llm_provider: str = "bedrock",
    preprocess_config: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """
    Async ML pipeline orchestration (Phase 2)
    
    Changes vs synchronous version:
    - Returns immediately with job_id to client
    - Runs in background worker process
    - Client can poll /job/{job_id} for status
    - Can scale to multiple workers on separate machines
    
    Improvement: Client waits 0s initially instead of 5+ minutes
    """
    try:
        import base64
        import tempfile
        from pathlib import Path
        from app.core.llm_clients import LLMRouter
        from app.core.database import get_collection
        from app.schemas import OrchestrateResponse
        
        self.update_state(state="PROGRESS", meta={"status": "Loading data...", "progress": 5})
        
        # Decode and load data
        payload_bytes = base64.b64decode(payload_base64)
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(filename).suffix) as tmp:
            tmp.write(payload_bytes)
            tmp_path = tmp.name
        
        df = load_dataframe_from_upload(filename, payload_bytes)
        
        self.update_state(state="PROGRESS", meta={"status": "Preprocessing...", "progress": 10})
        
        # Preprocess
        df = apply_preprocessing(df, target_column=target_column or None, config=preprocess_config)
        
        self.update_state(state="PROGRESS", meta={"status": "Building ML pipeline...", "progress": 20})
        
        # Run workflow
        workflow = build_workflow()
        llm_provider_resolved = llm_provider if llm_provider not in ("", "bedrock", "ollama_local") else settings.structured_llm_provider
        
        state: AgentState = {
            "business_problem": business_problem,
            "user_prompt": business_problem,
            "dataset_name": filename,
            "dataframe": df,
            "target_column": target_column or None,
            "model_family": model_family or None,
            "fixed_model": fixed_model or None,
            "llm_provider": llm_provider_resolved,
            "warnings": [],
        }
        
        self.update_state(state="PROGRESS", meta={"status": "Running agents...", "progress": 30})
        
        final_state = workflow.invoke(state) if workflow else run_sequential_fallback(state)
        
        self.update_state(state="PROGRESS", meta={"status": "Finalizing results...", "progress": 90})
        
        # Process results
        training = dict(final_state.get("training", {}))
        training.pop("model_object", None)
        training.pop("primary_model_object", None)
        
        result = {
            "dataset_name": filename,
            "target_column": target_column or None,
            "model_family": model_family or None,
            "task": final_state.get("task"),
            "problem_type": final_state.get("problem_type"),
            "data_insights": final_state.get("data_insights", {}),
            "pipeline": final_state.get("pipeline", {}),
            "training": training,
            "deployment": final_state.get("deployment", {}),
            "warnings": final_state.get("warnings", []),
        }
        
        # Store in database for later retrieval
        try:
            collection = get_collection("orchestration_results")
            collection.insert_one({
                "job_id": self.request.id,
                "status": "completed",
                "result": result,
            })
        except Exception as e:
            logger.warning(f"Failed to store result: {e}")
        
        self.update_state(state="SUCCESS", meta={"status": "Complete", "progress": 100})
        
        return result
        
    except Exception as exc:
        logger.error(f"Pipeline failed: {exc}", exc_info=True)
        self.update_state(state="FAILURE", meta={"status": f"Failed: {str(exc)}", "error": str(exc)})
        raise


@celery_app.task(bind=True, base=CallbackTask, name="tasks.build_kb")
def build_kb_task(
    self,
    filename: str,
    payload_base64: str,
    dataset_id: str,
    llm_provider: str = "bedrock",
    fast_mode: bool = False,
    chunk_cap: int = 200,
    skip_ner: bool = False,
    skip_pii: bool = False,
    skip_enrichment: bool = False,
    enrichment_batch_size: int = 8,
    enrichment_workers: int = 2,
    embedding_batch_size: int = 32,
    embedding_workers: int = 2,
) -> Dict[str, Any]:
    """
    Async knowledge base building (Phase 2)
    
    Benefits:
    - Non-blocking KB ingestion (200+ chunks)
    - Real-time progress updates
    - Can cancel long-running builds
    """
    try:
        import base64
        import tempfile
        from pathlib import Path
        from app.unstructured.service import KnowledgeBaseService
        from app.unstructured.schemas import KBState
        from app.unstructured.workflow import build_kb_workflow, run_kb_sequential_fallback
        
        self.update_state(state="PROGRESS", meta={"status": "Loading document...", "progress": 5})
        
        payload_bytes = base64.b64decode(payload_base64)
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(filename).suffix) as tmp:
            tmp.write(payload_bytes)
            tmp_path = tmp.name
        
        kb_service = KnowledgeBaseService()
        
        self.update_state(state="PROGRESS", meta={"status": "Building chunks...", "progress": 15})
        
        state: KBState = {
            "dataset_id": dataset_id or filename,
            "file_path": tmp_path,
            "llm_provider": llm_provider,
            "warnings": [],
            "fast_mode": fast_mode,
            "chunk_cap": chunk_cap,
            "chunk_size_tokens": settings.kb_chunk_size_tokens,
            "chunk_overlap_tokens": settings.kb_chunk_overlap_tokens,
            "skip_ner": skip_ner,
            "skip_pii": skip_pii,
            "skip_enrichment": skip_enrichment,
            "enrichment_batch_size": enrichment_batch_size,
            "enrichment_workers": enrichment_workers,
            "embedding_batch_size": embedding_batch_size,
            "embedding_workers": embedding_workers,
        }
        
        workflow = build_kb_workflow()
        def _progress(progress: int, message: str) -> None:
            self.update_state(state="PROGRESS", meta={"status": message, "progress": progress})

        result = kb_service.build_from_bytes(
            filename=filename,
            payload=payload_bytes,
            dataset_id=dataset_id,
            llm_provider=llm_provider,
            fast_mode=fast_mode,
            chunk_cap=chunk_cap,
            skip_ner=skip_ner,
            skip_pii=skip_pii,
            skip_enrichment=skip_enrichment,
            enrichment_batch_size=enrichment_batch_size,
            enrichment_workers=enrichment_workers,
            embedding_batch_size=embedding_batch_size,
            embedding_workers=embedding_workers,
            progress_callback=_progress,
        )

        self.update_state(state="SUCCESS", meta={"status": "Complete", "progress": 100})

        return result
        
    except Exception as exc:
        logger.error(f"KB build failed: {exc}", exc_info=True)
        self.update_state(state="FAILURE", meta={"status": f"Failed: {str(exc)}"})
        raise


def get_job_status(job_id: str) -> Dict[str, Any]:
    """Get status of async job"""
    result = AsyncResult(job_id, app=celery_app)
    
    if result.state == "PENDING":
        return {"status": "pending", "progress": 0}
    elif result.state == "PROGRESS":
        return {"status": "running", **result.info}
    elif result.state == "SUCCESS":
        return {"status": "completed", "result": result.result}
    elif result.state == "FAILURE":
        return {"status": "failed", "error": str(result.info)}
    else:
        return {"status": result.state.lower()}


def cancel_job(job_id: str) -> bool:
    """Cancel async job"""
    try:
        celery_app.control.revoke(job_id, terminate=True)
        return True
    except Exception:
        return False
