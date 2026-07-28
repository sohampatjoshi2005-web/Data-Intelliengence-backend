from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class MonitoringSpec:
    primary_metric_floor: float
    drift_detector: str
    retrain_trigger: str
    schema_checks: list[str]


def build_monitoring_spec(problem_type: str, champion_score: float) -> Dict[str, Any]:
    floor = max(0.0, float(champion_score) - 0.05)
    metric = "f1_weighted" if problem_type == "classification" else "r2"
    spec = MonitoringSpec(
        primary_metric_floor=floor,
        drift_detector="ADWIN",
        retrain_trigger=f"retrain if rolling {metric} < {floor:.4f} for 3 windows",
        schema_checks=["column presence", "dtype checks", "null-rate shift < 20%"],
    )
    return {
        "problem_type": problem_type,
        "primary_metric": metric,
        "spec": {
            "primary_metric_floor": spec.primary_metric_floor,
            "drift_detector": spec.drift_detector,
            "retrain_trigger": spec.retrain_trigger,
            "schema_checks": spec.schema_checks,
        },
    }
