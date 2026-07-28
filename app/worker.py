from celery import Celery
import pandas as pd
import json
from backend.app.main import _run_structured_pipeline, _structured_provider

celery_app = Celery(
    "automl_worker",
    broker="redis://localhost:6379/0",
    backend="redis://localhost:6379/0"
)

# Crucial: memory leak prevention. Respawn PyCaret workers after every 5 jobs.
celery_app.conf.worker_max_tasks_per_child = 5
celery_app.conf.task_track_started = True

@celery_app.task(bind=True)
def run_heavy_ml_pipeline(self, df_json, dataset_name, business_problem, target_column, model_family, fixed_model, llm_provider, preprocess_config):
    """
    Runs PyCaret/LightGBM correctly in the background, freeing up the FastAPI event loop.
    """
    df = pd.read_json(df_json)
    
    preprocess_payload = json.loads(preprocess_config) if preprocess_config else None

    # We execute this completely outside of the web server with CPU limits
    result = _run_structured_pipeline(
        df=df,
        dataset_name=dataset_name,
        business_problem=business_problem,
        target_column=target_column,
        model_family=model_family,
        fixed_model=fixed_model,
        llm_provider=_structured_provider(llm_provider),
        preprocess_config=preprocess_payload
    )
    return result
