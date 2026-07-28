from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel


class OrchestrateResponse(BaseModel):
    data_profile: Dict[str, Any]
    problem_type: str
    pipeline: Dict[str, Any]
    training: Dict[str, Any]
    evaluation: Dict[str, Any]
    deployment: Dict[str, Any]
    warnings: List[str]


class ChatRequest(BaseModel):
    question: str
    context: Optional[str] = None
    provider: str = "bedrock"


class ChatResponse(BaseModel):
    answer: str


class AnalyticsQueryRequest(BaseModel):
    dataframe: List[Dict[str, Any]]
    query: str
    chat_context: Optional[str] = None
    llm_provider: str = "bedrock"
    force_sql: bool = False


class AnalyticsNLToSQLRequest(BaseModel):
    dataframe: List[Dict[str, Any]]
    query: str
    llm_provider: str = "bedrock"


class AnalyticsNLToSQLResponse(BaseModel):
    sql: str


class AnalyticsSQLExplainRequest(BaseModel):
    sql: str
    llm_provider: str = "bedrock"


class AnalyticsSQLExplainResponse(BaseModel):
    explanation: str


class AnalyticsQueryResponse(BaseModel):
    should_plot: bool
    code: str
    execution: Dict[str, Any]
    reasoning: str


class AnalyticsSQLRequest(BaseModel):
    dataframe: List[Dict[str, Any]]
    sql: str


class AnalyticsSQLResponse(BaseModel):
    rows: List[Dict[str, Any]]
    columns: List[str]
    row_count: int


class UnstructuredAnalyzeResponse(BaseModel):
    file_name: str
    file_type: str
    text_preview: str
    text_length: int
    entities: List[Dict[str, Any]]
    entity_counts: List[Dict[str, Any]]
    warnings: List[str] = []
    duration: Optional[float] = None
    caption: Optional[str] = None
    caption_model: Optional[str] = None


class AnalyticsInsightsRequest(BaseModel):
    dataframe: List[Dict[str, Any]]
    llm_provider: str = "bedrock"


class AnalyticsInsightsResponse(BaseModel):
    insights: str


class AnalyticsVisualsResponse(BaseModel):
    visuals: List[Dict[str, Any]]


class ConnectorTestRequest(BaseModel):
    connector: str
    config: Dict[str, Any]


class ConnectorTestResponse(BaseModel):
    ok: bool
    connector: str
    message: str
    status_code: Optional[int] = None


class ConnectorLoadRequest(BaseModel):
    connector: str
    config: Dict[str, Any]
    query: Optional[str] = None
    table: Optional[str] = None
    limit: int = 1000


class ConnectorLoadResponse(BaseModel):
    connector: str
    rows: int
    columns: List[str]
    preview: List[Dict[str, Any]]


class ConnectorOrchestrateRequest(BaseModel):
    connector: str
    config: Dict[str, Any]
    query: Optional[str] = None
    table: Optional[str] = None
    limit: int = 1000
    dataset_name: Optional[str] = None
    business_problem: str = ""
    target_column: str = ""
    model_family: str = ""
    fixed_model: str = ""
    llm_provider: str = "bedrock"
    preprocess_config: Optional[Dict[str, Any]] = None


class KBBuildResponse(BaseModel):
    dataset_id: str
    language: str
    chunk_count: int
    stored_chunks: int
    headings_count: int
    entities_count: int
    warnings: List[str]
    performance: Optional[Dict[str, Any]] = None
    progress: Optional[List[Dict[str, Any]]] = None
    pipeline_steps: Optional[List[Dict[str, Any]]] = None


class KBQueryRequest(BaseModel):
    dataset_id: str
    query: str
    top_k: int = 8
    llm_provider: str = "bedrock"


class KBQueryResponse(BaseModel):
    dataset_id: str
    query: str
    hits: List[Dict[str, Any]]
    answer: str


class KGBuildRequest(BaseModel):
    dataset_id: str


class KGBuildResponse(BaseModel):
    dataset_id: str
    graph_name: str
    chunks: int
    entities: int
    edges: int


class KGQueryRequest(BaseModel):
    dataset_id: str
    entity: str
    limit: int = 20
    question: Optional[str] = None


class KGQueryResponse(BaseModel):
    dataset_id: str
    entity: str
    graph_name: str
    neighbors: List[Dict[str, Any]]
    answer: Optional[str] = None


class KGSubgraphRequest(BaseModel):
    dataset_id: str
    seed_entity: str
    hops: int = 1
    limit: int = 100


class KGSubgraphResponse(BaseModel):
    dataset_id: str
    seed_entity: str
    graph_name: str
    nodes: List[Dict[str, Any]]
    edges: List[Dict[str, Any]]


class RegistryPromoteRequest(BaseModel):
    model_id: str
    stage: str
    actor: str = "system"


class RegistryApproveRequest(BaseModel):
    model_id: str
    approver: str
    note: str = ""


class RegistryRollbackRequest(BaseModel):
    model_id: str
    actor: str = "system"


class FeatureStoreUpsertRequest(BaseModel):
    table: str
    rows: List[Dict[str, Any]]


class FeatureStoreMaterializeRequest(BaseModel):
    table: str
    key_col: str
    ts_col: Optional[str] = None


class FeatureStoreReadRequest(BaseModel):
    table: str
    key_col: str
    key_val: Any


class DeepEvalRequest(BaseModel):
    input_text: str
    actual_output: str
    context: Optional[str] = None
    model_name: Optional[str] = None
    metrics: Optional[List[str]] = None


class StructuredPreviewResponse(BaseModel):
    columns: List[str]
    rows: List[Dict[str, Any]]
    shape: Dict[str, int]


class StructuredTuneRequest(BaseModel):
    rows: List[Dict[str, Any]]
    target: str
    task: str
    model_key: str
    n_trials: int = 20


class StructuredTuneResponse(BaseModel):
    baseline_score: float
    tuned_score: float
    improvement_abs: float
    improvement_pct: float
    best_params: Dict[str, Any]
    trial_scores: Optional[List[float]] = None
    best_scores: Optional[List[float]] = None


class StructuredPredictTrainRequest(BaseModel):
    rows: List[Dict[str, Any]]
    target: str
    task: str
    model_key: str


class StructuredPredictTrainResponse(BaseModel):
    model_id: str
    task: str
    feature_columns: List[str]


class StructuredPredictRequest(BaseModel):
    model_id: str
    rows: List[Dict[str, Any]]
    return_proba: bool = True


class StructuredPredictResponse(BaseModel):
    predictions: List[Any]
    probabilities: Optional[List[List[float]]] = None


class StructuredOnlineStartRequest(BaseModel):
    task: str


class StructuredOnlineStartResponse(BaseModel):
    stream_id: str
    task: str


class StructuredOnlineBatchRequest(BaseModel):
    stream_id: str
    rows: List[Dict[str, Any]]
    target: str
    max_rows: Optional[int] = None


class StructuredOnlineBatchResponse(BaseModel):
    processed: int
    drift_hits: int
    drift_events: int
    accuracy: float
    running_variance: Optional[float] = None
    history: List[int]
    accuracy_history: List[float]
    variance_history: List[float]


class StructuredOnlineStatusResponse(BaseModel):
    drift_events: int
    accuracy: float
    history: List[int]
    accuracy_history: List[float]
    variance_history: List[float]


class StructuredOnlineStopResponse(BaseModel):
    stream_id: str
    ok: bool


class TransformationRequest(BaseModel):
    transform_type: str


class TransformationResponse(BaseModel):
    file_name: str
    media_type: str
    content_base64: str


class LogicalTransformResponse(BaseModel):
    file_name: str
    media_type: str
    content_base64: str
    columns: List[str] = []
    row_count: int = 0
    preview_rows: List[Dict[str, Any]] = []
    warnings: List[str] = []
    report: Dict[str, Any] = {}


class EdgeQuakeQueryRequest(BaseModel):
    base_url: str
    query: str
    mode: str = "hybrid"


class EdgeQuakeQueryResponse(BaseModel):
    data: Dict[str, Any]


class EdgeQuakeUploadResponse(BaseModel):
    data: Dict[str, Any]


class StructuredExplainRequest(BaseModel):
    result: Dict[str, Any]
    business_problem: str = ""
    drift_context: Optional[Dict[str, Any]] = None
    response_style: str = "technical"
    llm_provider: str = "bedrock"


class StructuredExplainResponse(BaseModel):
    explanation: str


class StructuredBusinessSummaryRequest(BaseModel):
    result: Dict[str, Any]
    business_problem: str = ""


class StructuredBusinessSummaryResponse(BaseModel):
    summary: str


class AnalyticsDashboardRequest(BaseModel):
    dataframe: List[Dict[str, Any]]


class AnalyticsDashboardResponse(BaseModel):
    numeric_summary: List[Dict[str, Any]]
    categorical_summary: List[Dict[str, Any]]
    correlations: List[Dict[str, Any]]


class StructuredExplainabilityRequest(BaseModel):
    rows: List[Dict[str, Any]]
    target: str
    task: str
    model_key: str
    top_n: int = 15


class StructuredExplainabilityResponse(BaseModel):
    explainability: Dict[str, Any]


class StructuredTuneFileResponse(BaseModel):
    baseline_score: float
    tuned_score: float
    improvement_abs: float
    improvement_pct: float
    best_params: Dict[str, Any]
    trial_scores: Optional[List[float]] = None
    best_scores: Optional[List[float]] = None
    file_name: str
    rows_processed: int


class StructuredPredictFileResponse(BaseModel):
    predictions: List[Any]
    probabilities: Optional[List[List[float]]] = None
    file_name: str
    rows_processed: int


class StructuredOnlineBatchFileResponse(BaseModel):
    processed: int
    drift_hits: int
    drift_events: int
    accuracy: float
    running_variance: Optional[float] = None
    history: List[int]
    accuracy_history: List[float]
    variance_history: List[float]
    file_name: str
