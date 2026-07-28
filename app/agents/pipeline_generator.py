from __future__ import annotations

from app.agents.base import BaseAgent
from app.graph.state import AgentState


DEFAULT_MODELS = {
    "classification": [
        "xgboost",
        "lightgbm",
        "catboost",
        "random_forest",
        "extra_trees",
        "gradient_boosting",
        "adaboost",
        "logistic_regression",
        "svc_rbf",
        "knn",
    ],
    "regression": [
        "xgboost_regressor",
        "lightgbm_regressor",
        "catboost_regressor",
        "random_forest_regressor",
        "extra_trees_regressor",
        "gradient_boosting_regressor",
        "linear_regression",
        "elasticnet",
        "ridge",
        "knn_regressor",
    ],
    "time_series": ["prophet", "arima", "lstm", "xgboost_ts", "random_forest_ts"],
    "ranking": ["xgboost_ranker", "lightgbm_ranker", "catboost_ranker", "random_forest", "logistic_regression"],
    "clustering": ["kmeans", "dbscan", "gaussian_mixture", "hierarchical", "birch"],
}

FAMILY_MAP = {
    "classification": {
        "tree_based": ["random_forest", "extra_trees"],
        "boosting": ["xgboost", "lightgbm", "catboost", "gradient_boosting", "adaboost"],
        "linear": ["logistic_regression"],
        "kernel": ["svc_rbf"],
        "instance_based": ["knn"],
    },
    "regression": {
        "tree_based": ["random_forest_regressor", "extra_trees_regressor"],
        "boosting": ["xgboost_regressor", "lightgbm_regressor", "catboost_regressor", "gradient_boosting_regressor"],
        "linear": ["linear_regression", "elasticnet", "ridge"],
        "kernel": ["svr_rbf"],
        "instance_based": ["knn_regressor"],
    },
}


class PipelineGeneratorAgent(BaseAgent):
    name = "pipeline_generator"

    def run(self, state: AgentState) -> AgentState:
        problem = state.get("problem_type", "classification")
        family = state.get("model_family")
        fixed_model = state.get("fixed_model")
        profile = state.get("data_profile", {})
        ts_candidates = profile.get("time_series_candidates", [])

        models = list(DEFAULT_MODELS.get(problem, DEFAULT_MODELS["classification"]))
        family_key = (family or "").strip().lower()
        if family_key and family_key != "auto":
            scoped = FAMILY_MAP.get(problem, {}).get(family_key, [])
            if scoped:
                models = [m for m in models if m in scoped] or models
        if fixed_model:
            models = [fixed_model]

        state["pipeline"] = {
            "problem_type": problem,
            "data_steps": [
                "schema_validation",
                "missing_value_handling",
                "categorical_encoding",
                "scaling",
                "leakage_guard",
            ],
            "validation_strategy": "time_aware_split" if ts_candidates or problem == "time_series" else "stratified_or_random_split",
            "candidate_models_top10": models[:10],
            "auto_model_decision": "enabled" if not fixed_model else "disabled",
            "fixed_model": fixed_model,
            "model_family": family or "auto",
        }
        return state
