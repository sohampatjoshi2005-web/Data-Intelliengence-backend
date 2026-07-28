from __future__ import annotations

from typing import Any, Dict, List, Optional, TypedDict

import pandas as pd


class AgentState(TypedDict, total=False):
    business_problem: str
    user_prompt: str
    dataset_name: str
    dataframe: pd.DataFrame
    target_column: Optional[str]
    model_family: Optional[str]
    fixed_model: Optional[str]
    llm_provider: str
    step_overrides: Dict[str, Any]

    data_profile: Dict[str, Any]
    problem_type: str
    pipeline: Dict[str, Any]
    training: Dict[str, Any]
    evaluation: Dict[str, Any]
    deployment: Dict[str, Any]
    warnings: List[str]
    # Analytics query workflow
    analytics_query: str
    analytics_chat_context: str
    analytics_should_plot: bool
    analytics_code: str
    analytics_execution: Dict[str, Any]
    analytics_reasoning: str
    analytics_insights: str
