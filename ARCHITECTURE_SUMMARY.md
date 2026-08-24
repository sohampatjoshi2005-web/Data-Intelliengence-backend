# Automl-Backend: Comprehensive Architecture Summary

**Last Updated:** April 2026  
**Framework:** FastAPI + Python 3.9+  
**Architecture Pattern:** Multi-agent LangGraph workflows with modular services  
**Production Status:** ✅ Production Ready

---

## Table of Contents

1. [System Overview](#system-overview)
2. [API Endpoints (Complete Reference)](#api-endpoints-complete-reference)
3. [Core Architecture & Components](#core-architecture--components)
4. [Data Pipelines & Workflows](#data-pipelines--workflows)
5. [Database & Data Models](#database--data-models)
6. [External Service Integrations](#external-service-integrations)
7. [LLM & AI Model Integrations](#llm--ai-model-integrations)
8. [Structured AutoML Pipeline](#structured-automl-pipeline)
9. [Unstructured Knowledge Base (RAG)](#unstructured-knowledge-base-rag)
10. [Analytics Engine](#analytics-engine)
11. [Knowledge Graph (Apache AGE)](#knowledge-graph-apache-age)
12. [MLOps & Governance](#mlops--governance)
13. [Data Connectors](#data-connectors)
14. [Authentication & Authorization](#authentication--authorization)
15. [Configuration & Environment Variables](#configuration--environment-variables)

---

## System Overview

### Purpose
Multi-tenant enterprise AutoML platform supporting:
- **Structured AutoML**: Tabular data → trained & deployed ML models
- **Unstructured Knowledge Base**: Document ingestion → grounded Q&A
- **Knowledge Graph**: Entity relationship extraction & graph exploration
- **Analytics Engine**: Natural language → SQL/Python analysis with AI reasoning

### Tech Stack

**Backend Core:**
- Framework: FastAPI (async HTTP)
- Workflow Orchestration: LangGraph (agent-based state graphs)
- Process Distribution: Ray, MLflow (optional)
- Async Processing: Uvicorn + async/await

**Data & Storage:**
- Primary DB: MongoDB Atlas (user, model metadata, KB documents)
- Vector Store: PostgreSQL + pgvector (KB embeddings)
- Knowledge Graph: PostgreSQL + Apache AGE extension (entity/relationship graphs)
- Caching: Redis (optional)
- Memory DB: DuckDB (in-memory analytics)

**ML & Data:**
- Auto-ML: PyCaret, scikit-learn, XGBoost, LightGBM, CatBoost
- Hyperparameter Tuning: Optuna
- Feature Engineering: scikit-learn pipelines
- Online Monitoring: River (drift detection: Hoeffding trees, ADWIN)
- Explainability: SHAP, Permutation importance
- NLP: spaCy, transformers, LangChain
- Document Processing: MarkItDown, PyTesseract, Unstructured, Tika, pdf2image

**LLM Providers:**
- AWS Bedrock (Claude 3.5 Sonnet, Titan embeddings) — **Production**
- Ollama Local (Mistral, LLaMA, etc.) — **Development**
- LangFuse (observability, optional)
- DeepEval (LLM output evaluation)

**External Integrations:**
- Connectors: BigQuery, Snowflake, MySQL, PostgreSQL, ElasticSearch
- Search: Elasticsearch, BM25 (keyword search)
- Ranking: FlashRank (semantic re-ranking)
- Vision: BLIP (image/video captioning)
- OCR: Tesseract (PDF text extraction)

---

## API Endpoints (Complete Reference)

### Base URL
```
http://localhost:8000
```

### Authentication
- Default: Disabled (`API_AUTH_ENABLED=false`)
- When enabled: Header-based RBAC with roles (viewer, data_scientist, ml_engineer, risk_reviewer, approver)

---

### 1. Health & Configuration

| Method | Endpoint | Purpose | Auth |
|--------|----------|---------|------|
| GET | `/health` | Backend status & provider availability | None |
| GET | `/models` | Model configuration, defaults, and orchestration status | None |
| GET | `/ws/health` | WebSocket health monitoring (streaming) | None |

**Response (GET /health):**
```json
{
  "status": "ok",
  "providers": ["bedrock", "ollama_local"],
  "langgraph_structured": true,
  "langgraph_analytics": true
}
```

---

### 2. Authentication & User Management

| Method | Endpoint | Purpose | Auth |
|--------|----------|---------|------|
| POST | `/auth/register` | Create new user (first user is admin) | None |
| POST | `/auth/login` | Login and get JWT token | None |
| GET | `/admin/users` | List all users (paginated) | risk_reviewer |
| POST | `/admin/users/{user_id}/approve` | Approve/deny user access | risk_reviewer |
| PATCH | `/admin/users/{user_id}/role` | Change user role | risk_reviewer |
| DELETE | `/admin/users/{user_id}` | Delete user | risk_reviewer |
| GET | `/admin/stats` | Admin dashboard stats | risk_reviewer |

**User Registration (POST /auth/register):**
```json
{
  "email": "user@example.com",
  "password": "secure_password",
  "role": "data scientist"
}
```

---

### 3. Structured AutoML

#### 3.1 Data Upload & Preview

| Method | Endpoint | Purpose | Auth |
|--------|----------|---------|------|
| POST | `/structured/preview` | Preview file structure | viewer |

**Request (multipart):**
- `file` (CSV, Excel, JSON, Parquet, etc.)

**Response:**
```json
{
  "columns": ["column1", "column2", ...],
  "rows": [{...}, {...}],
  "shape": {"rows": 1000, "columns": 5}
}
```

---

#### 3.2 Model Training & Orchestration

| Method | Endpoint | Purpose | Auth |
|--------|----------|---------|------|
| POST | `/orchestrate` | Full AutoML pipeline (upload → train → evaluate) | None |
| POST | `/orchestrate-from-connector` | AutoML from data connector | data_scientist |

**Request (POST /orchestrate, multipart):**
```
file: <CSV/XLSX file>
business_problem: "Predict customer churn"
target_column: "churn"
model_family: "" (empty = auto-detect)
fixed_model: "" (or "xgboost", "lightgbm", etc.)
llm_provider: "bedrock" | "ollama_local"
preprocess_config: {"drop_columns": ["id"], ...}
```

**Response (OrchestrateResponse):**
```json
{
  "data_profile": {
    "rows": 1000, "columns": 5, "missing_pct": {...}, 
    "numeric_cols": [...], "categorical_cols": [...]
  },
  "problem_type": "classification",
  "pipeline": {
    "preprocessing": {...},
    "feature_engineering": {...}
  },
  "training": {
    "best_model_name": "xgboost",
    "best_score": 0.92,
    "leaderboard": [{...}, {...}],
    "best_model_object": {...}
  },
  "evaluation": {
    "validation": {
      "holdout_metrics": {...},
      "cv_metrics": {...},
      "nested_cv_metrics": {...}
    },
    "explainability": {
      "feature_importance": {...},
      "shap_values": [...]
    },
    "fairness": {
      "demographic_parity": {...}
    }
  },
  "deployment": {
    "model_card": {...},
    "monitoring_plan": {...},
    "runtime_config": {...}
  },
  "warnings": [...]
}
```

---

#### 3.3 Hyperparameter Tuning

| Method | Endpoint | Purpose | Auth |
|--------|----------|---------|------|
| POST | `/structured/tune` | Run Optuna hyperparameter search | data_scientist |

**Request:**
```json
{
  "rows": [{...}, {...}],
  "target": "target_column",
  "task": "classification" | "regression",
  "model_key": "xgboost",
  "n_trials": 20
}
```

**Response:**
```json
{
  "baseline_score": 0.85,
  "tuned_score": 0.92,
  "improvement_abs": 0.07,
  "improvement_pct": 8.2,
  "best_params": {"n_estimators": 150, "max_depth": 10},
  "trial_scores": [0.80, 0.85, 0.89, ...],
  "best_scores": [0.80, 0.85, 0.89, ...]
}
```

---

#### 3.4 Model Training & Prediction

| Method | Endpoint | Purpose | Auth |
|--------|----------|---------|------|
| POST | `/structured/predict/train` | Train a single model (returns model_id) | data_scientist |
| POST | `/structured/predict` | Predict using trained model | viewer |

**POST /structured/predict/train Request:**
```json
{
  "rows": [{...}],
  "target": "target_col",
  "task": "classification",
  "model_key": "random_forest"
}
```

**Response:** `{"model_id": "uuid-xxx", "task": "classification", "feature_columns": [...]}`

**POST /structured/predict Request:**
```json
{
  "model_id": "uuid-xxx",
  "rows": [{...}, {...}],
  "return_proba": true
}
```

**Response:** `{"predictions": [1, 0, 1], "probabilities": [[0.1, 0.9], ...]}`

---

#### 3.5 Online Monitoring & Drift Detection

| Method | Endpoint | Purpose | Auth |
|--------|----------|---------|------|
| POST | `/structured/online/start` | Start streaming monitor | data_scientist |
| POST | `/structured/online/batch` | Process batch of predictions | data_scientist |
| GET | `/structured/online/status` | Get current drift metrics | viewer |
| POST | `/structured/online/stop` | Stop monitor | data_scientist |

**Stream Monitor Features (River):**
- **Hoeffding Trees**: Incremental learning for classification
- **ADWIN**: Adaptive Windowing for drift detection
- **Running Variance**: Statistical monitoring
- **Accuracy Tracking**: Continuous performance metrics

**POST /structured/online/batch Response:**
```json
{
  "processed": 100,
  "drift_hits": 5,
  "drift_events": 2,
  "accuracy": 0.88,
  "running_variance": 0.042,
  "history": [0.90, 0.89, 0.88],
  "accuracy_history": [0.90, 0.89, 0.88],
  "variance_history": [0.045, 0.043, 0.042]
}
```

---

#### 3.6 Model Explanation & Summarization

| Method | Endpoint | Purpose | Auth |
|--------|----------|---------|------|
| POST | `/structured/explain` | LLM-generated explanation of results | viewer |
| POST | `/structured/business-summary` | Deterministic business-friendly summary | viewer |
| POST | `/structured/explainability` | SHAP/permutation importance | data_scientist |

**POST /structured/explain Request:**
```json
{
  "result": {...},
  "business_problem": "Predict customer churn",
  "drift_context": {...},
  "response_style": "technical",
  "llm_provider": "bedrock"
}
```

**Response:**
```json
{
  "explanation": "The XGBoost model achieved 92% accuracy by focusing on customer lifetime value... [4 sections: problem solving, champion selection, drift interpretation, next steps]"
}
```

---

### 4. Data Connectors

| Method | Endpoint | Purpose | Auth |
|--------|----------|---------|------|
| GET | `/connectors` | List all available connectors | None |
| POST | `/connectors/test` | Test connector configuration | data_scientist |
| POST | `/connectors/load` | Load data from connector | data_scientist |

**Supported Connectors:**
- BigQuery
- Snowflake
- MySQL
- PostgreSQL
- ElasticSearch
- Pandas DataFrame (via export)

**POST /connectors/test Request:**
```json
{
  "connector": "snowflake",
  "config": {
    "account": "xy12345",
    "user": "account",
    "password": "secret",
    "database": "ANALYTICS",
    "warehouse": "COMPUTE"
  }
}
```

**Response:** `{"ok": true, "connector": "snowflake", "message": "Successfully connected"}`

**POST /connectors/load Request:**
```json
{
  "connector": "bigquery",
  "config": {"project_id": "my-project"},
  "table": "dataset.table_name",
  "limit": 10000
}
```

**Response:**
```json
{
  "connector": "bigquery",
  "rows": 10000,
  "columns": ["col1", "col2", ...],
  "preview": [{...}, {...}]
}
```

---

### 5. Analytics Engine

#### 5.1 Query Analytics

| Method | Endpoint | Purpose | Auth |
|--------|----------|---------|------|
| POST | `/analytics/query` | Multi-agent analytics (query → code → execute → reason) | None |
| POST | `/analytics/query-file` | Analytics on uploaded file | None |
| POST | `/analytics/run-file-stream` | Streaming analytics (reasoning, insights, visuals) | None |

**POST /analytics/query Request:**
```json
{
  "dataframe": [{col1: value1, col2: value2}, ...],
  "query": "What is the average petal length by species?",
  "chat_context": "",
  "llm_provider": "bedrock",
  "force_sql": false
}
```

**Response (AnalyticsQueryResponse):**
```json
{
  "should_plot": true,
  "code": "SELECT species, AVG(petal_length) FROM data GROUP BY species;",
  "execution": {
    "ok": true,
    "result": [{species: "setosa", avg_length: 5.2}, ...],
    "plot_base64": "iVBORw0K..."
  },
  "reasoning": "Computed average petal length grouped by iris species..."
}
```

**Workflow:**
1. **Query Understanding** (LLM): Parse intent, identify semantics
2. **Code Generation** (LLM): Generate SQL or Python
3. **Execution**: Run with DuckDB (in-memory)
4. **Visualization** (Plotly): Auto-generate plot
5. **Reasoning** (LLM): Explain results

---

#### 5.2 Natural Language → SQL

| Method | Endpoint | Purpose | Auth |
|--------|----------|---------|------|
| POST | `/analytics/nl-to-sql` | Convert natural language to SQL | None |
| POST | `/analytics/sql-explain` | Explain SQL in plain English | None |
| POST | `/analytics/run-sql` | Execute custom SQL | None |

**POST /analytics/nl-to-sql Request:**
```json
{
  "dataframe": [{...}],
  "query": "How many customers spent over $100?",
  "llm_provider": "bedrock"
}
```

**Response:** `{"sql": "SELECT COUNT(*) FROM data WHERE spend > 100;"}`

---

#### 5.3 Insights & Dashboard

| Method | Endpoint | Purpose | Auth |
|--------|----------|---------|------|
| POST | `/analytics/insights` | Generate AI-powered dataset insights | None |
| POST | `/analytics/insights-file` | Insights from uploaded file | None |
| POST | `/analytics/dashboard` | Summary statistics (numeric/categorical/correlations) | None |
| POST | `/analytics/visuals-file` | Auto-generate visual pack | None |

**POST /analytics/dashboard Response:**
```json
{
  "numeric_summary": [
    {"column": "age", "mean": 35.5, "std": 10.2, "min": 18, "max": 80},
    ...
  ],
  "categorical_summary": [
    {"column": "gender", "unique": 2, "top": "M", "freq": 600},
    ...
  ],
  "correlations": [
    {"var1": "age", "var2": "income", "coefficient": 0.72},
    ...
  ]
}
```

---

### 6. Knowledge Base (Unstructured RAG)

#### 6.1 Knowledge Base Build

| Method | Endpoint | Purpose | Auth |
|--------|----------|---------|------|
| POST | `/kb/build` | Ingest document → extract knowledge → embed & persist | None |

**Request (multipart):**
```
file: <PDF, DOCX, MD, TXT, JSON>
dataset_id: "financial_docs_2024"
llm_provider: "bedrock"
fast_mode: false
chunk_cap: 200
skip_ner: false
skip_pii: false
skip_enrichment: false
enrichment_batch_size: 8
enrichment_workers: 2
embedding_batch_size: 32
embedding_workers: 2
```

**KB Build Pipeline (14 Steps):**

1. **Data Acquisition**: File → bytes
2. **Markdownization**: Convert to standardized markdown (PDF→text, table extraction, formatting)
3. **Language Detection**: Identify language (langdetect)
4. **Data Standardization**: Clean text, normalize whitespace
5. **Feature Extraction**: Extract numbers, dates, URLs
6. **Structure Extraction**: Headings, lists, tables
7. **Redaction (PII Removal)**: Names, emails, SSN, credit cards (Presidio)
8. **Chunking**: Token-based (800 tokens default, 150 overlap) or semantic
9. **Contextualization**: Add heading/section context to chunks
10. **Enrichment (Parallel)**:
    - **Metadata Extraction**: Title, author, creation date
    - **Tag Extraction**: Auto-tagging (LLM)
    - **Question Extraction**: Generate anticipated questions (LLM)
    - **Summarization**: Chunk summaries (LLM)
    - **Keyword Extraction**: Top keywords per chunk
    - **Task Extraction**: Tasks/TODOs mentioned
11. **Embedding Generation**: Token-based (Titan/Ollama embeddings)
12. **Persistence**:
    - **Metadata Store**: MongoDB (chunk metadata, summaries, tags)
    - **Vector Index**: PostgreSQL + pgvector (embeddings, similarity search)
    - **Keyword Index**: BM25 ranking (inverted index for keyword search)

**Response (KBBuildResponse):**
```json
{
  "dataset_id": "financial_docs_2024",
  "language": "en",
  "chunk_count": 250,
  "stored_chunks": 245,
  "headings_count": 45,
  "entities_count": 800,
  "warnings": ["OCR not available for page 3"],
  "performance": {
    "fast_mode": false,
    "chunk_cap": 200,
    "skip_ner": false,
    "skip_pii": false,
    "skip_enrichment": false,
    "enrichment_batch_size": 8,
    "enrichment_workers": 2,
    "embedding_batch_size": 32,
    "embedding_workers": 2
  }
}
```

---

#### 6.2 Knowledge Base Query

| Method | Endpoint | Purpose | Auth |
|--------|----------|---------|------|
| POST | `/kb/query` | Retrieve grounded answer from KB | None |

**Request:**
```json
{
  "dataset_id": "financial_docs_2024",
  "query": "What are the Q4 revenue projections?",
  "top_k": 8,
  "llm_provider": "bedrock"
}
```

**Query Workflow (Hybrid Retrieval):**
1. **Vector Search**: Embed query → pgvector similarity search (top 16)
2. **BM25 Search**: Keyword matching (top 16)
3. **Reciprocal Rank Fusion**: Merge by combined score (final top 8)
4. **Context Compilation**: Concatenate snippets
5. **LLM Answer Generation**: Context + question → answer
6. **Citation Tracking**: Return source chunks with scores

**Response (KBQueryResponse):**
```json
{
  "dataset_id": "financial_docs_2024",
  "query": "What are the Q4 revenue projections?",
  "hits": [
    {
      "chunk_id": "chunk-001",
      "summary": "Q4 revenue forecast section",
      "snippet": "Q4 2024 projected revenue of $5.2M, up 15% YoY...",
      "bm25_score": 0.85,
      "vector_score": 0.92,
      "fused_score": 0.88,
      "bm25_rank": 1,
      "vector_rank": 1
    },
    ...
  ],
  "answer": "Based on the financial documentation, Q4 2024 revenue is projected at $5.2M, representing a 15% year-over-year increase..."
}
```

---

#### 6.3 Unstructured File Analysis

| Method | Endpoint | Purpose | Auth |
|--------|----------|---------|------|
| POST | `/unstructured/analyze` | Analyze file without KB build (preview) | None |

**Response (UnstructuredAnalyzeResponse):**
```json
{
  "file_name": "document.pdf",
  "file_type": "pdf",
  "text_preview": "First 500 chars of extracted text...",
  "text_length": 15000,
  "entities": [
    {"text": "Apple Inc.", "label": "ORG", "start": 100, "end": 110},
    {"text": "2024-01-15", "label": "DATE", "start": 200, "end": 210}
  ],
  "entity_counts": [
    {"label": "ORG", "count": 25},
    {"label": "PERSON", "count": 12}
  ],
  "warnings": ["OCR may be needed for page 5"],
  "duration": 2.3,
  "caption": "Graph showing quarterly revenue trends"
}
```

---

### 7. Knowledge Graph (Apache AGE)

#### 7.1 Knowledge Graph Build

| Method | Endpoint | Purpose | Auth |
|--------|----------|---------|------|
| POST | `/kg/build` | Extract entities & relationships from KB → build graph | None |

**Request:**
```json
{
  "dataset_id": "financial_docs_2024"
}
```

**KG Build Process:**
1. Retrieve chunks from KB
2. Extract named entities (spaCy, LLM)
3. Extract entity relationships (LLM)
4. Resolve entity coreference (LLM)
5. Build property graph in Apache AGE (PostgreSQL)
6. Index for fast traversal

**Response (KGBuildResponse):**
```json
{
  "dataset_id": "financial_docs_2024",
  "graph_name": "kg_financial_docs_2024",
  "chunks": 250,
  "entities": 800,
  "edges": 1200
}
```

---

#### 7.2 Knowledge Graph Query

| Method | Endpoint | Purpose | Auth |
|--------|----------|---------|------|
| POST | `/kg/query` | Query entity neighbors + optional LLM reasoning | None |
| POST | `/kg/subgraph` | Get subgraph around seed entity | None |

**POST /kg/query Request:**
```json
{
  "dataset_id": "financial_docs_2024",
  "entity": "Apple Inc.",
  "limit": 20,
  "question": "What partnerships does Apple have?"
}
```

**Response (KGQueryResponse):**
```json
{
  "dataset_id": "financial_docs_2024",
  "entity": "Apple Inc.",
  "graph_name": "kg_financial_docs_2024",
  "neighbors": [
    {
      "node_type": "COMPANY",
      "entity": "Microsoft Corp",
      "relationship": "PARTNER_WITH",
      "distance": 1
    },
    {
      "node_type": "PERSON",
      "entity": "Tim Cook",
      "relationship": "LEADS",
      "distance": 1
    }
  ],
  "answer": "Based on the knowledge graph, Apple has partnerships with Microsoft Corporation..."
}
```

**POST /kg/subgraph:**
```json
{
  "dataset_id": "financial_docs_2024",
  "seed_entity": "Apple Inc.",
  "hops": 2,
  "limit": 100
}
```

**Response (KGSubgraphResponse):**
```json
{
  "dataset_id": "financial_docs_2024",
  "seed_entity": "Apple Inc.",
  "graph_name": "kg_financial_docs_2024",
  "nodes": [
    {"id": "node-1", "type": "COMPANY", "label": "Apple Inc.", "color": "green"},
    {"id": "node-2", "type": "PERSON", "label": "Tim Cook", "color": "green"}
  ],
  "edges": [
    {"source": "node-1", "target": "node-2", "type": "LEADS"}
  ]
}
```

---

### 8. Chat & Conversational Analysis

| Method | Endpoint | Purpose | Auth |
|--------|----------|---------|------|
| POST | `/chat` | Stream conversational responses (SSE) | None |
| POST | `/chat/structured-file` | Answer questions about tabular data | None |

**POST /chat Request:**
```json
{
  "question": "What are best practices for RandomForest tuning?",
  "context": "In context of ML classification...",
  "provider": "bedrock"
}
```

**Response:** Server-Sent Events (SSE) streaming text chunks

---

### 9. MLOps & Model Registry

#### 9.1 Model Registry

| Method | Endpoint | Purpose | Auth |
|--------|----------|---------|------|
| GET | `/registry/models` | List all registered models & versions | viewer |
| POST | `/registry/approve` | Approve model for promotion | risk_reviewer |
| POST | `/registry/promote` | Promote model to staging/production | approver |
| POST | `/registry/rollback` | Rollback production to prior model | approver |

**Model Lifecycle:**
- **Candidate**: Newly trained model
- **Staging**: Approved for testing
- **Production**: Live serving
- **Archived**: Retired

**POST /registry/promote Request:**
```json
{
  "model_id": "model-uuid-123",
  "stage": "production",
  "actor": "ml_engineer_john"
}
```

**Response:**
```json
{
  "ok": true,
  "model_id": "model-uuid-123",
  "stage": "production",
  "promoted_at": "2024-04-15T10:30:00Z",
  "history": [
    {"event": "approve", "approver": "risk_reviewer_jane", "at": "2024-04-14T..."},
    {"event": "promote", "stage": "staging", "actor": "ml_engineer_john", "at": "2024-04-15T..."}
  ]
}
```

---

#### 9.2 Feature Store

| Method | Endpoint | Purpose | Auth |
|--------|----------|---------|------|
| GET | `/feature-store/tables` | List feature tables | viewer |
| POST | `/feature-store/offline/upsert` | Insert/update features offline | data_scientist |
| POST | `/feature-store/online/materialize` | Materialize online cache from offline | ml_engineer |
| POST | `/feature-store/online/read` | Read online feature by key | viewer |

**Feature Store Pattern:**
- **Offline**: Batch computation, materialized in CSV/Parquet
- **Online**: Low-latency serving cache (in-memory or Redis)

---

#### 9.3 Orchestration Status

| Method | Endpoint | Purpose | Auth |
|--------|----------|---------|------|
| GET | `/orchestration/status` | Check orchestration mode & backend | None |

**Response:**
```json
{
  "mode": "local" | "distributed",
  "backend": "local" | "ray" | "dask",
  "status": "ready" | "busy",
  "workers": 4,
  "pending_tasks": 2
}
```

---

### 10. Model Evaluation & Assessment

| Method | Endpoint | Purpose | Auth |
|--------|----------|---------|------|
| GET | `/evaluation-tools` | List evaluation metrics & tools | None |
| POST | `/deepeval/run` | Run DeepEval metrics on LLM outputs | None |

**DeepEval Metrics:**
- `faithfulness`: Does output follow context?
- `answer_relevancy`: Is output relevant to question?
- `hallucination`: Is output factually accurate?
- `bias`: Is output biased?

**POST /deepeval/run Request:**
```json
{
  "input_text": "What is the Q4 revenue?",
  "actual_output": "Q4 2024 revenue is $5.2M...",
  "context": "Financial report excerpt: Q4 2024 revenue of $5.2M, up 15% YoY.",
  "model_name": "claude-3-5-sonnet",
  "metrics": ["faithfulness", "answer_relevancy"]
}
```

**Response:**
```json
{
  "status": "ok",
  "metrics": [
    {
      "metric": "faithfulness",
      "score": 0.95,
      "pass": true,
      "reason": "Output is well-grounded in provided context"
    },
    {
      "metric": "answer_relevancy",
      "score": 0.98,
      "pass": true,
      "reason": "Output directly addresses the question"
    }
  ]
}
```

---

### 11. File Transformation

| Method | Endpoint | Purpose | Auth |
|--------|----------|---------|------|
| POST | `/transform/run` | Format conversion (CSV↔Excel, etc.) | viewer |

---

## Core Architecture & Components

### Directory Structure

```
backend/
├── app/
│   ├── agents/              # LangGraph agents (state machine nodes)
│   │   ├── data_understanding.py
│   │   ├── problem_classification.py
│   │   ├── pipeline_generator.py
│   │   ├── model_training.py
│   │   ├── evaluator.py
│   │   ├── deployment.py
│   │   ├── analytics_query_understanding.py
│   │   ├── analytics_code_generation.py
│   │   ├── analytics_execution.py
│   │   ├── analytics_reasoning.py
│   │   ├── analytics_insights.py
│   │   ├── unstructured_ingestion.py
│   │   ├── unstructured_enrichment.py
│   │   └── unstructured_embedding.py
│   │
│   ├── core/                # Configuration & clients
│   │   ├── config.py        # Settings (env vars)
│   │   ├── database.py      # MongoDB connection
│   │   └── llm_clients.py   # Bedrock & Ollama clients
│   │
│   ├── graph/               # LangGraph workflow assembly
│   │   ├── state.py         # TypedDict state definitions
│   │   └── workflow.py      # Graph compilation & execution
│   │
│   ├── models/              # Data models
│   │   └── user.py          # User, auth models
│   │
│   ├── schemas.py           # Pydantic request/response schemas
│   │
│   ├── services/            # Business logic & I/O
│   │   ├── structured_runtime.py      # Optuna tuning, online monitoring (River)
│   │   ├── model_registry.py          # Model versioning & lifecycle
│   │   ├── feature_store.py           # Offline/online feature management
│   │   ├── connector_runtime.py       # Data connector execution
│   │   ├── connectors.py              # List available connectors
│   │   ├── analytics_dashboard.py     # Summary stats for data
│   │   ├── analytics_visuals.py       # Auto-visualization
│   │   ├── analytics_llm.py           # Analytics-specific LLM calls
│   │   ├── deepeval_runner.py         # LLM output evaluation
│   │   ├── preprocessing.py           # Feature engineering pipeline
│   │   ├── model_explainability.py    # SHAP, permutation importance
│   │   ├── data_loader.py             # Load CSV/Excel/Parquet
│   │   ├── auth_service.py            # JWT token creation/validation
│   │   ├── authz.py                   # Role-based access control
│   │   ├── bed_roi_embeddings.py      # Bedrock embeddings API
│   │   ├── email_service.py           # SMTP notifications
│   │   ├── distributed_orchestrator.py # Ray/Dask coordination
│   │   ├── dspy_runtime.py            # DSPy NL→SQL
│   │   └── ...
│   │
│   ├── unstructured/        # KB, KG, RAG pipelines
│   │   ├── service.py       # KnowledgeBaseService (build, query)
│   │   ├── age_graph.py     # AGEGraphService (KG operations)
│   │   ├── workflow.py      # KB pipeline (ingestion → enrichment → embedding)
│   │   ├── storage.py       # All oyLikeKBStore (MongoDB + pgvector + BM25)
│   │   ├── schemas.py       # KBState, chunk types
│   │   └── ...
│   │
│   └── main.py              # FastAPI app, route handlers
│
├── requirements.txt         # Python dependencies
├── Dockerfile              # Container image
└── app.py                  # (Root) Streamlit UI
```

---

### Key Service Classes

#### **LLMRouter** (`core/llm_clients.py`)
```python
class LLMRouter:
    def complete(prompt, provider="bedrock") -> str
    def stream_complete(prompt, provider, max_tokens) -> Iterator[str]
    def available_providers() -> List[str]
```

Routs LLM calls to Bedrock (prod) or Ollama (dev) based on config/request.

---

#### **KnowledgeBaseService** (`unstructured/service.py`)
```python
class KnowledgeBaseService:
    def build_from_bytes(filename, payload, dataset_id, ...) -> Dict
    def query(dataset_id, query, top_k, llm_provider) -> Dict
```

**Responsibilities:**
- Trigger KB pipeline (ingestion → enrichment → embedding)
- Manage persistence (MongoDB + pgvector + BM25)
- Execute hybrid retrieval (vector + BM25)
- Generate grounded answers

---

#### **AGEGraphService** (`unstructured/age_graph.py`)
```python
class AGEGraphService:
    def build_for_dataset(dataset_id) -> GraphBuildResult
    def query_neighbors(dataset_id, entity, limit) -> Dict
    def subgraph(dataset_id, seed_entity, hops, limit) -> Dict
```

**Responsibilities:**
- Extract entities & relationships from KB chunks
- Build property graph in PostgreSQL + AGE
- Query by entity or hops

---

#### **LocalModelRegistry** (`services/model_registry.py`)
```python
class LocalModelRegistry:
    def register(model_record) -> Dict
    def list_models() -> List[Dict]
    def approve(model_id, approver, note) -> Dict
    def promote(model_id, stage, actor) -> Dict
    def rollback_to(model_id, actor) -> Dict
```

**Responsibilities:**
- Metadata storage (JSON file or MongoDB)
- Approval workflow tracking
- Stage transitions (candidate → staging → production)
- Rollback history

---

#### **LocalFeatureStore** (`services/feature_store.py`)
```python
class LocalFeatureStore:
    def list_tables() -> Dict
    def upsert_offline(table, rows) -> Dict
    def materialize_online(table, key_col, ts_col) -> Dict
    def read_online(table, key_col, key_val) -> Dict
```

---

### Agent-Based Workflow (LangGraph)

#### **Structured AutoML Workflow**
```
DataUnderstanding → ProblemClassification → PipelineGeneration → 
ModelTraining → Evaluation → Deployment
```

Each node is a LangGraph Agent:

```python
class DataUnderstandingAgent:
    def run(state: AgentState) -> AgentState:
        # Profiling: missing values, dtypes, distributions
        # Output: state["data_profile"]
        
class ProblemClassificationAgent:
    def run(state) -> AgentState:
        # Detect task: classification / regression / time-series
        # Output: state["problem_type"]

class PipelineGeneratorAgent:
    def run(state) -> AgentState:
        # Feature engineering, preprocessing
        # Output: state["pipeline"]

class ModelTrainingAgent:
    def run(state) -> AgentState:
        # PyCaret auto-training, leaderboard
        # Output: state["training"] (best_model, scores, leaderboard)

class EvaluatorAgent:
    def run(state) -> AgentState:
        # Validation: holdout, CV, nested CV, SHAP, fairness
        # Output: state["evaluation"]

class DeploymentAgent:
    def run(state) -> AgentState:
        # Model card, monitoring plan, registry metadata
        # Output: state["deployment"]
```

---

#### **Analytics Workflow**
```
AnalyticsQueryUnderstanding → AnalyticsCodeGeneration → 
AnalyticsExecution → AnalyticsReasoning
```

**Agents:**

1. **AnalyticsQueryUnderstandingAgent**: Parse intent, identify dimensions/metrics
2. **AnalyticsCodeGenerationAgent**: Generate SQL or Python code
3. **AnalyticsExecutionAgent**: Execute against DuckDB + generate plots
4. **AnalyticsReasoningAgent**: Reason about results, output insights

---

#### **Unstructured KB Workflow** (14-step pipeline)
```
1. Ingestion → 2. Markdownization → 3. Language Detection → 
4. Standardization → 5. Feature Extraction → 6. Structure Extraction → 
7. Redaction → 8. Chunking → 9. Contextualization → 
10. Enrichment (parallel: metadata, tags, questions, summaries, keywords, tasks) → 
11. Embedding → 12. Persistence (MongoDB + pgvector + BM25) → 
13. Graph Extraction (KG build) → 14. Complete
```

---

## Data Pipelines & Workflows

### Structured AutoML Pipeline

**Input:** CSV file + business problem + target column  
**Output:** Trained model + evaluation metrics + deployment metadata

**Steps:**

1. **Data Loading**: File → Pandas DataFrame (handle missing, dtypes)
2. **Preprocessing**: Advanced transformations
   - Drop columns (manual selection)
   - Drop duplicates
   - Missing value handling (mean/median/mode/forward-fill)
   - Outlier capping (IQR-based)
   - Low-variance feature removal
   - Date parsing & part extraction (year, month, day, dayofweek)
3. **Data Profiling**: Rows, columns, missing %, correlations, distributions
4. **Problem Detection**: Task type (classification/regression) inferred from target
5. **Model Candidate Selection**: Filter by family if specified
6. **Auto-Training**: PyCaret or sklearn fallback
   - Train multiple algorithms
   - Generate leaderboard (top 10 models)
7. **Hyperparameter Tuning**: Optuna (configurable n_trials)
8. **Validation**: 
   - Holdout split (80/20 or configured)
   - Cross-validation (5-fold default)
   - Nested cross-validation (for fair hyperparameter evaluation)
9. **Explainability**:
   - Feature importance (permutation)
   - SHAP values (if enabled)
10. **Fairness Diagnostics**: Group-based metrics (demographic parity, etc.)
11. **Packaging**: Model artifact + metadata
12. **Registry**: Automatic model registration with metadata
13. **Deployment**: Runtime config, monitoring plan, model card generation

---

### Unstructured KB Pipeline (Detailed)

**Input:** PDF/DOCX/TXT/JSON file  
**Output:** Persisted knowledge base (MongoDB + pgvector + BM25)

**Step-by-Step:**

1. **Data Acquisition**
   - Read file bytes
   - Temp file creation

2. **Markdownization**
   - Convert to markdown for parse-ability
   - Extract tables from PDFs/Excel
   - Preserve structure (headings, lists)
   - Use: MarkItDown, Tika, pdfplumber

3. **Language Detection**
   - Identify document language
   - Use: langdetect

4. **Data Standardization**
   - Trim whitespace
   - Normalize line breaks
   - Remove control characters

5. **Feature Extraction**
   - Identify numbers, dates, URLs, emails
   - Store as features

6. **Structure Extraction**
   - Extract heading hierarchy
   - Identify list items, tables
   - Build document outline

7. **Redaction (PII Removal)**
   - Detect: Names, emails, phones, SSNs, credit cards
   - Redact or mask (optional, configurable)
   - Use: Presidio

8. **Chunking**
   - Token-based (800 tokens default, 150 overlap)
   - Semantic chunking option (sentence-transformers)
   - Hierarchical chunking option (preserve structure)
   - Ensure chunk boundaries don't split sentences

9. **Contextualization**
   - Add parent heading to each chunk
   - Add previous/next section references
   - Preserve hierarchy info

10. **Enrichment** (Parallel, configurable workers)
    - **Metadata Extraction**: Title, author, keywords, creation date
    - **Tag Extraction**: Auto-categorize chunks (LLM)
    - **Question Extraction**: Generate Q&A pairs (LLM)
    - **Summarization**: Brief summary per chunk (LLM)
    - **Keyword Extraction**: Top 5-10 keywords per chunk
    - **Task/TOD Extraction**: Extract tasks mentioned

11. **Embedding Generation**
    - Embed chunk text + summary
    - Use: Bedrock Titan embed or Ollama embeddings
    - Batch processing (configurable batch size & workers)

12. **Persistence** (3-layer)
    - **MongoDB**: Chunk metadata (summary, tags, questions, keywords, tasks)
    - **pgvector**: Embeddings (for dense vector similarity search)
    - **BM25 Index**: Keyword inverted index (for sparse keyword search)

13. **Storage Schema**
    - `documents` collection: dataset_id, source_name, language, feature_json
    - `chunks` collection: dataset_id, chunk_id, chunk_text, summary, embedding, metadata, tags, questions, keywords, tasks

14. **Warnings**: Log issues (OCR failed, encoding errors, etc.)

---

### Analytics Pipeline

**Input:** Natural language query + dataframe (or file)  
**Output:** SQL code + executed result + plot + reasoning

**Steps:**

1. **Query Understanding** (LLM Agent)
   - Parse intent (filter, aggregate, group, sort)
   - Identify dimensions & metrics
   - Detect special case patterns (built-in handler for iris dataset checks)

2. **Code Generation** (LLM Agent)
   - Generate SQL (DuckDB syntax)
   - Validate syntax
   - Handle edge cases

3. **Execution** (DuckDB in-memory)
   - Register dataframe as table "data"
   - Execute SQL
   - Handle errors (NULL handling, type mismatches)

4. **Visualization** (Plotly)
   - Infer plot type (scatter, histogram, box, bar, etc.)
   - Generate interactive plot
   - Encode as base64

5. **Reasoning** (LLM Agent)
   - Summarize findings
   - Highlight key insights
   - Contextualize against business problem

---

### Knowledge Graph Build Pipeline

**Input:** KB chunks + entities (from KB build)  
**Output:** Property graph in Apache AGE (PostgreSQL)

**Steps:**

1. **Entity Extraction**: Extract from KB chunks (spaCy + LLM)
2. **Relationship Extraction**: Find relationships between entities (LLM)
3. **Coreference Resolution**: Merge duplicate entities (LLM-assisted)
4. **Graph Insertion**: Create nodes/edges in AGE
   - Node types: Dataset, Chunk, Entity, Document
   - Edge types: MENTIONED_IN, RELATED_TO, PART_OF, etc.
5. **Indexing**: Create indices for fast traversal

---

## Database & Data Models

### Primary Database: MongoDB (User, Metadata)

**Collections:**

#### 1. `users`
```json
{
  "_id": "uuid",
  "email": "user@example.com",
  "hashed_password": "...",
  "role": "admin" | "data scientist" | "business/analyst",
  "is_approved": true,
  "created_at": "2024-04-15T10:00:00Z",
  "last_login": "2024-04-15T14:30:00Z"
}
```

#### 2. `documents` (KB)
```json
{
  "_id": "uuid",
  "dataset_id": "financial_docs_2024",
  "source_name": "annual_report.pdf",
  "language": "en",
  "file_size": 2500000,
  "encoding": "utf-8",
  "feature_json": {
    "title": "2024 Annual Report",
    "author": "Company",
    "created_date": "2024-01-15"
  },
  "created_at": "2024-04-15T10:00:00Z"
}
```

#### 3. `chunks` (KB)
```json
{
  "_id": "uuid",
  "dataset_id": "financial_docs_2024",
  "chunk_id": "chunk-001",
  "source_document_id": "doc-uuid",
  "chunk_text": "Q4 2024 revenue is projected at...",
  "summary": "Revenue forecast for Q4 2024",
  "metadata": {
    "heading": "Financial Projections",
    "section": "2.4",
    "page": 15
  },
  "tags": ["financial", "forecast", "q4"],
  "questions": ["What is Q4 revenue?", "What are the projections?"],
  "keywords": ["revenue", "forecast", "projection", "2024", "q4"],
  "tasks": ["Review Q4 projections"],
  "stored_at": "2024-04-15T10:00:00Z"
}
```

#### 4. `models` (Registry)
```json
{
  "_id": "model-uuid-123",
  "dataset_name": "customer_churn",
  "problem_type": "classification",
  "created_at": "2024-04-15T10:00:00Z",
  "created_by": "data_scientist_john",
  "stage": "production",
  "champion_score": 0.92,
  "target_column": "churn",
  "artifact_path": "/models/model-uuid-123.pkl",
  "feature_columns": ["age", "balance", "tenure"],
  "holdout_metrics": {
    "accuracy": 0.91,
    "f1_weighted": 0.90,
    "precision": 0.92
  },
  "cv_metrics": {
    "mean": 0.90,
    "std": 0.02
  },
  "nested_cv_metrics": {
    "outer_mean": 0.89
  },
  "approvals": [
    {"approver": "risk_reviewer_jane", "approved_at": "2024-04-14T..."}
  ],
  "history": [
    {"event": "approve", "approver": "risk_reviewer_jane", "at": "2024-04-14T..."},
    {"event": "promote", "stage": "production", "actor": "approver_admin", "at": "2024-04-15T..."}
  ]
}
```

---

### Vector Database: PostgreSQL + pgvector (KB Embeddings)

**Schema:**

```sql
CREATE TABLE kb_embeddings (
  chunk_id UUID PRIMARY KEY,
  dataset_id VARCHAR(255),
  embedding VECTOR(768),  -- Embedding dimension
  chunk_text TEXT,
  summary TEXT,
  metadata JSONB
);

CREATE INDEX ON kb_embeddings USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);
```

**Query:**
```sql
SELECT * FROM kb_embeddings
ORDER BY embedding <-> '[0.1, 0.2, ...]'::vector
LIMIT 10;
```

---

### Knowledge Graph: PostgreSQL + Apache AGE

**Schema (AGE):**

```sql
CREATE EXTENSION IF NOT EXISTS age;
LOAD 'age';
CREATE SCHEMA IF NOT EXISTS kb_graph;
SET search_path = kb_graph;

SELECT * FROM ag_graph_create('kb_financial_2024', true);

-- Create nodes
MATCH (d:Dataset {dataset_id: 'financial_2024'})
CREATE (d)-[:CONTAINS]->(c:Chunk {chunk_id: 'chunk-001'})
CREATE (c)-[:MENTIONS]->(e:Entity {name: 'Apple Inc.', type: 'COMPANY'})
CREATE (e)-[:RELATED_TO]->(e2:Entity {name: 'Tim Cook', type: 'PERSON'})
```

**Node Types:**
- Dataset
- Chunk
- Entity (ORG, PERSON, LOCATION, PRODUCT, etc.)

**Edge Types:**
- CONTAINS (Dataset → Chunk)
- MENTIONS (Chunk → Entity)
- RELATED_TO (Entity → Entity)
- PART_OF (Entity → Entity, hierarchical)
- LEADS (PERSON → organization)
- FOUNDED (PERSON → COMPANY)

---

### Feature Store (Local)

**Tables:**

```
offline:
  customer_features:
    | customer_id | age | score | updated_at |
    | 001         | 35  | 0.85  | 2024-04-15 |

online:
  (Redis or in-memory cache)
  customer_features:
    001 → {age: 35, score: 0.85} (materialized at 2024-04-15T15:00:00Z)
```

---

## External Service Integrations

### 1. Data Connectors

**Supported:**

| Connector | Status | Purpose |
|-----------|--------|---------|
| BigQuery | ✅ Prod | Google Cloud data warehouse |
| Snowflake | ✅ Prod | Cloud data platform |
| MySQL | ✅ Prod | Relational database |
| PostgreSQL | ✅ Prod | Relational database |
| ElasticSearch | ✅ Prod | Search & analytics |
| Pandas (CSV/Parquet) | ✅ Prod | Local files |

**Connector Runtime** (`services/connector_runtime.py`):
```python
def load_dataframe_from_connector(
    connector: str,      # "bigquery", "snowflake", ...
    config: Dict,        # Connection credentials
    query: Optional[str],# SQL query or
    table: Optional[str],# table name
    limit: int = 1000
) -> pd.DataFrame:
    # Execute connection
    # Run query or SELECT * FROM table LIMIT limit
    # Return DataFrame
```

**Config Examples:**

BigQuery:
```json
{
  "project_id": "my-project",
  "credentials": {...}
}
```

Snowflake:
```json
{
  "account": "xy12345",
  "user": "account",
  "password": "secret",
  "database": "ANALYTICS",
  "warehouse": "COMPUTE"
}
```

---

### 2. Search Services

#### Elasticsearch
- Used for KB keyword search (alternative to BM25)
- Index: `kb_{dataset_id}`
- Query: Hybrid BM25 + vector cosine similarity

#### BM25 (In-Memory)
- Default keyword index for KB
- Library: `rank-bm25`
- Used for fast keyword retrieval before vector search

---

### 3. Email Service

**Configuration:**
```env
EMAIL_ENABLED=true
SMTP_SERVER=smtp.resend.com
SMTP_PORT=587
SMTP_USERNAME=resend
SMTP_PASSWORD=<api-key>
EMAIL_FROM=onboarding@superhumanlythoughts.com
```

**Use Cases:**
- User registration notifications
- Approval notifications
- Model promotion alerts

---

### 4. LLM Provider Integrations

#### AWS Bedrock (Production)

**Setup:**
```bash
# Check available models
aws bedrock list-foundation-models --region us-east-2

# Test invocation
aws bedrock-runtime invoke-model \
  --model-id anthropic.claude-3-5-sonnet-20241022-v2:0 \
  --region us-east-2 \
  --body '{"prompt":"Hello"}' \
  response.json
```

**Models:**
- **Chat**: `anthropic.claude-3-5-sonnet-20241022-v2:0`
- **Embeddings**: `amazon.titan-embed-text-v2:0`

**Client** (`core/llm_clients.py`):
```python
class BedrockClient:
    def invoke(prompt, model_id, max_tokens, temperature) -> str
    def stream_invoke(prompt, model_id, max_tokens) -> Iterator[str]
```

---

#### Ollama Local (Development)

**Setup:**
```bash
ollama serve
ollama pull mistral:latest
ollama pull bge-small-en-v1.5
```

**Models:**
- **Chat**: mistral:latest, llama3.2:3b, etc.
- **Embeddings**: nomic-embed-text, bge-small-en-v1.5

**Endpoint:** `http://localhost:11434/v1`

**Client** (`core/llm_clients.py`):
```python
class OllamaClient:
    def complete(prompt, model, max_tokens) -> str
    def stream_complete(prompt, model, max_tokens) -> Iterator[str]
    def embed(texts) -> List[List[float]]
```

---

### 5. Observability

#### LangFuse (Optional)
- Monitor LLM calls
- Track latency, tokens, cost
- Debug agent workflows

**Configuration:**
```env
LANGFUSE_PUBLIC_KEY=<key>
LANGFUSE_SECRET_KEY=<key>
LANGFUSE_HOST=https://cloud.langfuse.com
```

#### MLflow (Optional)
- Experiment tracking
- Model registry
- Hyperparameter logging

---

### 6. Embedding Services

#### Bedrock Titan Embeddings
```python
def bedrock_embed_texts(texts: List[str]) -> List[List[float]]:
    # Embed list of texts
    # Returns: List of 1024-dim vectors
```

#### Ollama Embeddings
```python
def ollama_embed_texts(texts, model="bge-small-en-v1.5"):
    # Embed list of texts
    # Returns: List of 384-dim vectors (bge-small)
```

---

## LLM & AI Model Integrations

### LLMRouter: Unified Interface

```python
class LLMRouter:
    def complete(prompt: str, provider: str = "bedrock") -> str:
        """Synchronous completion"""
        
    def stream_complete(
        prompt: str, 
        provider: str, 
        max_tokens: int
    ) -> Iterator[str]:
        """Streaming completion"""
        
    def available_providers() -> List[str]:
        """Returns: ['bedrock', 'ollama_local']"""
```

### Use Cases by Workflow

| Workflow | LLM Usage |
|----------|-----------|
| **Structured AutoML** | Problem classification, pipeline reasoning, explainability reports |
| **Unstructured KB** | Enrichment (tags, questions, summaries, keywords), question answering |
| **Analytics** | Query understanding, code generation, reasoning |
| **Knowledge Graph** | Entity extraction, relationship inference, exploration reasoning |
| **Chat** | Conversational analysis, context generation |
| **Explainability** | Feature attribution explanation, business impact translation |

### Model-Specific Configurations

#### Claude 3.5 Sonnet (Bedrock)
```
Max tokens: 2048
Temperature: 0.3  (lower = more deterministic)
Region: us-east-2
Cost: ~$0.003 / 1K input tokens
```

#### Mistral (Ollama)
```
Max tokens: 2048
Base URL: http://localhost:11434/v1
Free (local inference)
~8B parameters
```

---

## Structured AutoML Pipeline (Deep Dive)

### Phase 1: Data Understanding
**Agent:** `DataUnderstandingAgent`  
**Input:** Dataframe  
**Output:** `state["data_profile"]`

**Profile includes:**
```json
{
  "rows": 1000,
  "columns": 5,
  "numeric_cols": ["age", "income"],
  "categorical_cols": ["gender", "region"],
  "missing_values": {
    "age": 0.02,
    "income": 0.05
  },
  "duplicate_rows": 15,
  "dtypes": {
    "age": "int64",
    "income": "float64"
  },
  "summary_stats": {
    "age": {"mean": 35.5, "std": 10.2, "min": 18, "max": 80}
  },
  "distribu...": {...}
}
```

---

### Phase 2: Problem Classification
**Agent:** `ProblemClassificationAgent`  
**Input:** Target column + data profile  
**Output:** `state["problem_type"]`

**Detection Logic:**
```python
def classify_problem(df, target_column):
    # Check target dtype & cardinality
    if target.dtype in ['int64', 'float64'] and n_unique > 20:
        return "regression"
    elif target.dtype == 'object' or n_unique <= 20:
        return "classification"
    else:
        return "regression"
```

---

### Phase 3: Pipeline Generation
**Agent:** `PipelineGeneratorAgent`  
**Input:** Data profile + problem type  
**Output:** `state["pipeline"]`

**Standard Pipeline:**
```python
Pipeline([
    ('preprocessing', ColumnTransformer([
        ('numeric', Pipeline([
            SimpleImputer(strategy='median'),
            StandardScaler()
        ]), numeric_cols),
        ('categorical', Pipeline([
            SimpleImputer(strategy='most_frequent'),
            OneHotEncoder(handle_unknown='ignore')
        ]), categorical_cols)
    ])),
    ('model', XGBClassifier(...))
])
```

---

### Phase 4: Model Training
**Agent:** `ModelTrainingAgent`  
**Input:** Preprocessed data + problem type  
**Output:** `state["training"]`

**Training Logic:**
```python
def train_models(X_train, y_train, problem_type):
    candidates = {
        'random_forest': RandomForestClassifier(...),
        'xgboost': XGBClassifier(...),
        'lightgbm': LGBMClassifier(...),
        'logistic_regression': LogisticRegression(...)
    }
    
    leaderboard = []
    for name, model in candidates.items():
        train_pipe = Pipeline([
            ('preprocessing', preprocessor),
            ('model', model)
        ])
        score = cross_val_score(
            train_pipe, X_train, y_train, 
            cv=5, scoring=scoring_metric
        ).mean()
        leaderboard.append({
            'model': name,
            'score': score
        })
    
    leaderboard.sort(key=lambda x: x['score'], reverse=True)
    return {
        'best_model_name': leaderboard[0]['model'],
        'best_score': leaderboard[0]['score'],
        'leaderboard': leaderboard
    }
```

---

### Phase 5: Evaluation
**Agent:** `EvaluatorAgent`  
**Input:** Trained model + full data  
**Output:** `state["evaluation"]`

**Evaluation Metrics:**

**Holdout (80/20 split):**
```
Accuracy, F1, Precision, Recall, ROC-AUC
```

**Cross-Validation (5-fold):**
```
Mean CV score, std, fold scores
```

**Nested Cross-Validation:**
```
Outer CV: 5 folds
Inner CV: 3 folds (for hyperparameter tuning)
Unbiased estimate of final score
```

**Explainability:**
```json
{
  "feature_importance": {
    "age": 0.35,
    "income": 0.25,
    "tenure": 0.20
  },
  "shap_values": [...]  // If enabled
}
```

**Fairness (Group-based):**
```json
{
  "demographic_parity": {
    "gender_M": {"accuracy": 0.92},
    "gender_F": {"accuracy": 0.90},
    "disparity": 0.02
  }
}
```

---

### Phase 6: Deployment
**Agent:** `DeploymentAgent`  
**Input:** Evaluation results + model  
**Output:** `state["deployment"]` + registry entry

**Model Card:**
```json
{
  "model_name": "xgboost_churn_v1",
  "version": "1.0",
  "created_date": "2024-04-15",
  "task": "classification",
  "metrics": {
    "holdout_accuracy": 0.92,
    "holdout_f1": 0.90
  },
  "intended_use": "Predict customer churn",
  "limitations": "Trained on 2024 Q1-Q2 data",
  "ethical_considerations": "Monitor for demographic bias"
}
```

**Monitoring Plan:**
```json
{
  "monitoring_metrics": [
    "prediction_accuracy",
    "feature_drift",
    "data_drift",
    "prediction_distribution"
  ],
  "alert_thresholds": {
    "accuracy_drop": 0.05,
    "drift_threshold": 0.1
  },
  "monitoring_frequency": "daily"
}
```

**Registry Entry:**
```json
{
  "model_id": "model-uuid-123",
  "dataset_name": "customer_churn",
  "stage": "candidate",
  "champion_score": 0.92,
  "created_by": "system",
  "artifact_path": "/models/model-uuid-123.pkl",
  "feature_columns": [...],
  "monitoring_plan": {...}
}
```

---

## Unstructured Knowledge Base (RAG)

### RAG v2 Configuration

**Settings:**
```python
rag_v2_enabled=true
rag_query_expansions=3  # Generate 3 query variations
rag_reranker="flashrank"  # Semantic re-ranking
rag_rerank_top_k=10  # Re-rank top 10 candidates
rag_retrieval_pool=40  # Initial pool size
rag_graph_hops=1  # KG traversal depth
rag_context_compress=true  # Compress context
rag_semantic_chunking=true  # Use semantic boundaries
rag_hierarchical_chunking=true  # Preserve hierarchy
```

### Query Expansion
**Input:** "What is Q4 revenue?"  
**Expanded queries (LLM-generated):**
1. "What are the Q4 2024 revenue figures?"
2. "How much revenue did the company make in Q4?"
3. "Q4 financial results and revenue"

**Benefit:** Improve recall by searching multiple reformulations

### Hybrid Retrieval

```
1. BM25 Search (keyword, sparse)
   ↓
   Top 40 candidates
   ↓
2. Vector Search (semantic, dense)
   ↓
   Top 40 candidates
   ↓
3. Merge & Re-rank (FlashRank)
   ↓
   Top 10 final hits
   ↓
4. Context Compression (remove redundancy)
   ↓
   Deduplicated context
```

### Graph-Augmented Retrieval

**KG Hops:**
```
Query Entity (in KB) 
  ↓ (1 hop)
Related Entities
  ↓ (fetch chunks mentioning them)
Related Chunks

Result: Original + related content combined
```

---

## Analytics Engine

### Query Types Handled

#### 1. Distribution & Class Means (Built-in Handler)
**Pattern Detection:** "distribution" + "mean" + ("class"|"variety")

**Execute:** Compute class distribution + group means  
**Output:** Deterministic SQL + table results (no ML needed)

---

#### 2. Simple Distribution (Built-in Handler)
**Pattern Detection:** "distribution" + ("numeric"|"histogram"|"first numeric")

**Execute:** Compute descriptive stats (mean, std, quartiles)  
**Output:** Summary + SQL preview

---

#### 3. Pairwise Separation (Built-in Handler)
**Pattern Detection:** "separates" + multiple markers

**Execute:** Effect size (standardized mean difference) per feature pair  
**Output:** Best separating feature + threshold

---

#### 4. General Queries (Workflow)
**Any other query:**
```
Query Understanding → Code Generation → Execution → Reasoning
```

### Code Generation Strategy

**If DSPy available:**
```python
sql = dspy_nl_to_sql(query, columns, table_name="data")
```

**Else (LLM):**
```python
prompt = f"You are a SQL generator. Table: data. Columns: {cols}. Query: '{query}'. Return only SQL."
sql = llm_router.complete(prompt, provider)
```

### Execution Engine (DuckDB)

```python
import duckdb

con = duckdb.connect(":memory:")
con.register("data", df)
result = con.execute(sql).df()
```

**Advantages:**
- In-memory, no network latency
- SQL standard (most LLMs generate valid DuckDB SQL)
- Handles NULL, type casting, aggregations

---

## Knowledge Graph (Apache AGE)

### GraphGraph Model (Property Graph)

**Node Types:**
```
Dataset  (root)
  ├── Chunk (text snippets)
  │   └── Entity (persons, organizations, locations, etc.)
  └── (relationships between entities)
```

**Edge Types:**
```
CONTAINS → Chunk contains Entity mention
MENTIONS → Entity mentioned in Chunk
RELATED_TO → Entity related to Entity (co-occurrence, inferred)
PART_OF → Hierarchical (e.g., Company.Department)
LEADS → Person leads Organization
FOUNDED → Person founded Company
```

### Query Examples

**Get all companies mentioned in dataset:**
```agraphql
MATCH (d:Dataset)-[]->(e:Entity {type:'COMPANY'})
RETURN DISTINCT e.name
```

**Get entity's neighbors (1 hop):**
```agraphql
MATCH (e:Entity {name:'Apple Inc.'})-[r]->(n)
RETURN n, r
LIMIT 20
```

**Subgraph (2 hops):**
```agraphql
MATCH p = (seed:Entity {name:'Apple Inc.'})-[*1..2]-(*)
RETURN nodes(p), relationships(p)
LIMIT 100
```

---

## MLOps & Governance

### Model Lifecycle

```
Candidate (just trained)
    ↓
  [APPROVE] (risk_reviewer reviews)
    ↓
Staging (approved, ready for testing)
    ↓
  [PROMOTE] (approver promotes to production)
    ↓
Production (live serving)
    ↓
  [ROLLBACK] (in case of issues)
    ↓
Previous Production or Archived
```

### Approval Workflow

**Step 1: Risk Review**
- Risk reviewer examines model metrics
- Checks fairness, explainability, monitoring plan
- Approves or denies

**Step 2: Promotion**
- Approver (higher role) promotes to staging/production
- Triggers monitoring activation
- Logs event + timestamp

**Step 3: Rollback (if needed)**
- Approver rolls back to prior production model
- Reactivates previous monitoring
- Logs incident

---

### Feature Store Pattern

**Offline** (Batch):
```
Historical data → Compute features → Store (CSV/Parquet)
```

**Online** (Real-time):
```
Offline table → Materialize snapshot → Cache (Redis/In-memory)
```

**Read path:**
```
Model prediction request → Fetch online features by key → Join with request data → Predict
```

---

## Data Connectors

### Connector Interface

```python
class Connector:
    def test(config: Dict) -> Dict:
        # Check connectivity
        # Return: {ok: bool, message: str}
        
    def load(
        config: Dict,
        query: Optional[str],
        table: Optional[str],
        limit: int
    ) -> pd.DataFrame:
        # Connect
        # Execute query or SELECT * FROM table LIMIT limit
        # Return DataFrame
```

### Supported Connectors

#### BigQuery
```json
{
  "type": "bigquery",
  "config": {
    "project_id": "my-project",
    "credentials": {...}
  }
}
```

#### Snowflake
```json
{
  "type": "snowflake",
  "config": {
    "account": "xy12345",
    "user": "account",
    "password": "secret",
    "warehouse": "COMPUTE",
    "database": "ANALYTICS"
  }
}
```

#### MySQL/PostgreSQL
```json
{
  "type": "postgresql",
  "config": {
    "host": "localhost",
    "port": 5432,
    "database": "analytics",
    "user": "user",
    "password": "pass"
  }
}
```

---

## Authentication & Authorization

### JWT-based (When Enabled)

**Environment:**
```env
API_AUTH_ENABLED=true
JWT_SECRET_KEY=<your-secret>
ACCESS_TOKEN_EXPIRE_MINUTES=1440
```

**Token Structure:**
```json
{
  "sub": "user@example.com",
  "role": "data scientist",
  "exp": 1713189600
}
```

### Role-Based Access Control (RBAC)

**Roles:**
- `admin`: Full access, user management, approvals
- `data scientist`: Dataset upload, model training, tuning
- `ml_engineer`: Feature store materialization
- `risk_reviewer`: Model approvals
- `approver`: Model promotion to production
- `viewer` / `business/analyst`: Read-only access

**Endpoint Protection:**
```python
@app.post("/structured/tune")
def structured_tune(
    req: StructuredTuneRequest,
    _: None = Depends(_require_data_scientist)  # ← Auth check
):
    ...
```

**HTTP Headers (when enabled):**
```
x-api-key: <token or api-key>
x-role: <role>
```

---

## Configuration & Environment Variables

### Core Settings

**File:** `backend/app/core/config.py`

```python
@dataclass(frozen=True)
class Settings:
    # App
    app_name: str = "Superhumanly Thoughts"
    
    # Ollama (Development)
    ollama_base_url: str = "http://localhost:11434/v1"
    ollama_model: str = "llama3.2:3b"
    ollama_embed_model: str = "nomic-embed-text"
    
    # Bedrock (Production)
    bedrock_region: str = "us-east-2"
    bedrock_model_id: str = "anthropic.claude-3-5-sonnet-20241022-v2:0"
    bedrock_embed_model_id: str = "amazon.titan-embed-text-v2:0"
    
    # Knowledge Base / KB
    kb_pg_dsn: str = "postgresql+psycopg2://user:pass@localhost:5432/kbdb"
    kb_age_dsn: str = "postgresql+psycopg2://user:pass@localhost:55432/kbdb"
    kb_age_graph_name: str = "kb_graph"
    kb_fast_mode: bool = False
    kb_chunk_cap: int = 200
    kb_chunk_size_tokens: int = 800
    kb_chunk_overlap_tokens: int = 150
    kb_skip_ner: bool = False
    kb_skip_pii: bool = False
    kb_skip_enrichment: bool = False
    kb_enrich_batch_size: int = 8
    kb_enrich_workers: int = 2
    kb_embed_batch_size: int = 32
    kb_embed_workers: int = 2
    
    # RAG v2
    rag_v2_enabled: bool = True
    rag_query_expansions: int = 3
    rag_reranker: str = "flashrank"
    rag_rerank_top_k: int = 10
    rag_retrieval_pool: int = 40
    rag_graph_hops: int = 1
    rag_context_compress: bool = True
    rag_semantic_chunking: bool = True
    rag_hierarchical_chunking: bool = True
    
    # Database
    mongodb_uri: str = ""
    
    # Auth (if enabled)
    api_auth_enabled: bool = False
    jwt_secret_key: str = "prod-secret-change-me"
    access_token_expire_minutes: int = 1440
    
    # CORS
    cors_origins: str = "*"
    
    # Email (if enabled)
    email_enabled: bool = False
    smtp_server: str = "smtp.resend.com"
    smtp_port: int = 587
    email_from: str = "onboarding@superhumanlythoughts.com"
    
    # Distributed
    distributed_enabled: bool = False
    distributed_backend: str = "auto"  # ray, dask, auto
```

---

### Environment Variable Examples

**Development (.env):**
```bash
OLLAMA_OPENAI_BASE_URL=http://localhost:11434/v1
OLLAMA_OPENAI_API_KEY=ollama
OLLAMA_CHAT_MODEL=mistral:latest
OLLAMA_EMBED_MODEL=bge-small-en-v1.5

KB_PG_DSN=postgresql+psycopg2://postgres:password@localhost:5432/kbdb
KB_AGE_DSN=postgresql+psycopg2://postgres:password@localhost:55432/kbdb

API_AUTH_ENABLED=false
```

**Production (.env.production):**
```bash
STRUCTURED_LLM_PROVIDER=bedrock
BEDROCK_REGION=us-east-2
BEDROCK_MODEL_ID=anthropic.claude-3-5-sonnet-20241022-v2:0
BEDROCK_EMBED_MODEL_ID=amazon.titan-embed-text-v2:0

MONGODB_URI=mongodb+srv://user:pass@cluster.mongodb.net/db

KB_PG_DSN=postgresql+psycopg2://user:pass@aws-db.rds.amazonaws.com:5432/kbdb

API_AUTH_ENABLED=true
JWT_SECRET_KEY=<strong-random-secret>

CORS_ORIGINS=https://app.superhumanlythoughts.com,https://api.superhumanlythoughts.com

EMAIL_ENABLED=true
SMTP_PASSWORD=<resend-api-key>

LANGFUSE_PUBLIC_KEY=<key>
LANGFUSE_SECRET_KEY=<key>
```

---

## Summary: Key Statistics

| Aspect | Count |
|--------|-------|
| API Endpoints | 31+ |
| Database Collections (MongoDB) | 4+ |
| Agent Modules | 10+ |
| Supported Connectors | 6+ |
| ML Models Trained (per run) | 4-10 |
| Enrichment Tasks (parallel) | 6 |
| KB Pipeline Steps | 14 |
| Evaluation Metrics per Model | 15+ |
| Environment Variables | 40+ |

---

## Quick Start

### 1. Set up environment
```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure (choose one):

**Development (Ollama):**
```bash
export OLLAMA_OPENAI_BASE_URL="http://localhost:11434/v1"
export OLLAMA_CHAT_MODEL="mistral:latest"
export KB_PG_DSN="postgresql+psycopg2://user:pass@localhost:5432/kbdb"
```

**Production (Bedrock):**
```bash
export BEDROCK_REGION="us-east-2"
export BEDROCK_MODEL_ID="anthropic.claude-3-5-sonnet-20241022-v2:0"
export KB_PG_DSN="postgresql+psycopg2://user:pass@aws-rds.amazonaws.com:5432/kbdb"
```

### 3. Run backend
```bash
python -m uvicorn app.main:app --reload --port 8000
```

### 4. Test health
```bash
curl http://localhost:8000/health
```

---

## Production Deployment

**AWS EC2 Instance:** `t3.medium` (4GB RAM, Ubuntu 22.04)  
**Process Manager:** Gunicorn or systemd  
**Reverse Proxy:** Nginx  
**Service File:** `agentic-ai-backend.service` (included)

See `PRODUCTION_BACKEND_GUIDE.md` for full deployment steps.

---

**End of Architecture Summary**

This comprehensive document covers:
- ✅ All 31+ API endpoints with request/response examples
- ✅ Core architecture (LangGraph workflows, modular services)
- ✅ Data pipelines (structured AutoML, unstructured KB, analytics, KG)
- ✅ Database schemas (MongoDB, pgvector, Apache AGE)
- ✅ External integrations (Bedrock, Ollama, connectors, email)
- ✅ LLM & AI model integrations
- ✅ MLOps & governance framework
- ✅ Configuration & environment variables
