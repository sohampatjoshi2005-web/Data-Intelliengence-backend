from __future__ import annotations

from datetime import datetime, timezone

from app.agents.base import BaseAgent
from app.graph.state import AgentState
from app.services.experiment_tracker import ExperimentTracker
from app.services.fairness import fairness_report
from app.services.model_explainability import explain_model
from app.services.model_validation import evaluate_model


class EvaluatorAgent(BaseAgent):
    name = "evaluator"

    def run(self, state: AgentState) -> AgentState:
        training = state.get("training", {})
        leaderboard = training.get("leaderboard", [])
        top = leaderboard[0] if leaderboard else {}
        df = state.get("dataframe")
        target = training.get("target")
        problem = state.get("problem_type", "classification")
        champion = training.get("model_object")

        validation = {}
        explainability = {}
        fairness = {}
        eval_warnings: list[str] = []
        if df is not None and target and target in df.columns and champion is not None:
            try:
                v = evaluate_model(df=df, target=target, problem=problem, model=champion)
                validation = {
                    "holdout_metrics": v.holdout_metrics,
                    "cv_metrics": v.cv_metrics,
                    "nested_cv_metrics": v.nested_cv_metrics,
                    "threshold_tuning": v.threshold_tuning,
                    "calibration": v.calibration,
                }
            except Exception as exc:
                eval_warnings.append(f"validation_failed: {exc}")
            try:
                explainability = explain_model(df=df, target=target, problem=problem, model=champion, top_n=15)
            except Exception as exc:
                eval_warnings.append(f"explainability_failed: {exc}")
            try:
                fairness = fairness_report(df=df, target=target, problem=problem, model=champion)
            except Exception as exc:
                eval_warnings.append(f"fairness_failed: {exc}")
        else:
            eval_warnings.append("evaluation artifacts skipped: dataframe/target/model unavailable")

        tracker = ExperimentTracker()
        tracking = tracker.log_run(
            run_name=f"{state.get('dataset_name', 'dataset')}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}",
            dataset_name=state.get("dataset_name", "unknown"),
            business_problem=state.get("business_problem", ""),
            problem_type=problem,
            params={
                "pipeline": state.get("pipeline", {}),
                "training": {
                    "backend": training.get("automl_backend", "unknown"),
                    "best_model_name": training.get("best_model_name", "unknown"),
                },
            },
            metrics={
                "training_best_score": float(training.get("best_score", 0.0)),
                "validation": validation,
            },
            artifacts={},
        )

        state["evaluation"] = {
            "top_10_models": leaderboard,
            "champion": top,
            "model_selection_strategy": "Top 5 ranked by primary metric",
            "llm_studio_backlog": {
                "fine_tuning": "planned",
                "experiment_compare": "planned",
                "feedback_loops": "planned",
                "secure_deploy": "planned",
            },
            "monitoring": {
                "drift": "ADWIN + performance delta thresholds",
                "data_quality": "null-rate and schema checks",
                "retraining_trigger": "metric drop > 5% over rolling window",
            },
            "validation": validation,
            "explainability": explainability,
            "fairness": fairness,
            "experiment_tracking": tracking,
            "warnings": eval_warnings,
        }
        return state
