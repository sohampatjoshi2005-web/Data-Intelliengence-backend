from __future__ import annotations

import json

from app.agents.base import BaseAgent
from app.graph.state import AgentState
from app.services.model_registry import LocalModelRegistry
from app.services.model_packaging import package_model
from app.services.online_monitoring import build_monitoring_spec


class DeploymentAgent(BaseAgent):
    name = "deployment"

    def run(self, state: AgentState) -> AgentState:
        training = state.get("training", {})
        model_name = training.get("best_model_name", "model")
        model_obj = training.get("model_object")
        df = state.get("dataframe")
        target = training.get("target")
        problem = state.get("problem_type", "classification")
        holdout_metrics = state.get("evaluation", {}).get("validation", {}).get("holdout_metrics", {})
        champion_score = float(training.get("best_score", 0.0))

        api_stub = f"""
from fastapi import FastAPI
import joblib

app = FastAPI()
model = joblib.load('artifacts/{model_name}.joblib')

@app.post('/predict')
def predict(payload: dict):
    # map payload to dataframe and call model.predict
    return {{'prediction': 'TODO'}}
""".strip()

        package_info = {
            "status": "skipped",
            "reason": "model/data unavailable",
        }
        if model_obj is not None and df is not None and target and target in df.columns:
            try:
                package_info = package_model(
                    model=model_obj,
                    dataset_name=state.get("dataset_name", "dataset"),
                    problem_type=problem,
                    target=target,
                    metrics=holdout_metrics,
                )
            except Exception as exc:
                package_info = {
                    "status": "failed",
                    "reason": str(exc),
                }
        registry_info = {"status": "skipped"}
        if package_info.get("model_id"):
            try:
                registry = LocalModelRegistry()
                registry_info = registry.register(
                    {
                        "model_id": package_info.get("model_id"),
                        "dataset_name": state.get("dataset_name", ""),
                        "problem_type": problem,
                        "target": target,
                        "champion_model_name": model_name,
                        "champion_score": champion_score,
                        "holdout_metrics": holdout_metrics,
                        "validation": state.get("evaluation", {}).get("validation", {}),
                        "fairness": state.get("evaluation", {}).get("fairness", {}),
                        "explainability": state.get("evaluation", {}).get("explainability", {}),
                        "artifact_dir": package_info.get("artifact_dir", ""),
                        "model_path": package_info.get("model_path", ""),
                        "metadata_path": package_info.get("metadata_path", ""),
                    }
                )
            except Exception as exc:
                registry_info = {"status": "failed", "reason": str(exc)}

        monitoring = build_monitoring_spec(problem_type=problem, champion_score=champion_score)
        governance = {
            "project": state.get("dataset_name", "default_project"),
            "approval_required_for_production": True,
            "roles": {
                "data_scientist": ["build", "evaluate", "propose_deploy"],
                "ml_engineer": ["package", "serve", "monitor"],
                "risk_reviewer": ["fairness_review", "compliance_review"],
                "approver": ["promote_to_production", "rollback"],
            },
            "gates": [
                "validation metrics above threshold",
                "fairness disparity reviewed",
                "data leakage checks pass",
                "signed approval by approver role",
            ],
        }

        state["deployment"] = {
            "artifact_path": f"artifacts/{model_name}.joblib",
            "packaging": package_info,
            "model_registry": {
                "backend": "mlflow_or_local",
                "model_name": model_name,
                "version": package_info.get("model_id", "pending"),
                "metadata": package_info.get("metadata_path", ""),
                "registration": registry_info,
            },
            "prediction_api_stub": api_stub,
            "drift_monitoring_plan": monitoring,
            "governance": governance,
            "release_checklist": {
                "validated": bool(holdout_metrics),
                "packaged": bool(package_info.get("model_path")),
                "governance_ready": True,
            },
            "deployment_manifest": json.dumps(
                {
                    "model": model_name,
                    "problem_type": problem,
                    "dataset": state.get("dataset_name", ""),
                    "monitoring": monitoring,
                    "governance": governance,
                }
            ),
        }
        return state
