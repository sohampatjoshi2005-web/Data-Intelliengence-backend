import pandas as pd

from app.agents import analytics_code_generation as codegen
from app.services.analytics_helpers import build_reasoning_fallback


def test_predictive_outcome_query_detection_matches_species_forecast_language():
    assert codegen._is_predictive_outcome_query("Forecast potential outcomes for species")
    assert not codegen._is_predictive_outcome_query("Summarize missing values by column")


def test_predictive_species_query_uses_structured_fallback(monkeypatch):
    monkeypatch.setattr(codegen, "chat_complete", lambda **_: "")

    agent = codegen.AnalyticsCodeGenerationAgent()
    state = {
        "dataframe": pd.DataFrame(
            {
                "sepal_length": [5.1, 4.9, 7.0, 6.4],
                "sepal_width": [3.5, 3.0, 3.2, 3.2],
                "species": ["setosa", "setosa", "versicolor", "versicolor"],
            }
        ),
        "analytics_query": "Forecast potential outcomes for species",
        "analytics_chat_context": "",
        "analytics_should_plot": False,
        "analytics_force_sql": False,
    }

    out = agent.run(state)

    assert "model_metrics" in out["analytics_code"]
    assert "conclusions" in out["analytics_code"]
    assert "summary" not in out["analytics_code"]


def test_reasoning_fallback_clarifies_visual_pack_is_separate():
    text = build_reasoning_fallback(
        "Summarize the dataset",
        {
            "ok": True,
            "result": {"type": "object", "value": {"summary": "basic profile"}},
            "plot_base64": None,
        },
    )

    assert "Query-specific plot generated=False" in text
    assert "Dataset-level visuals are generated separately" in text
