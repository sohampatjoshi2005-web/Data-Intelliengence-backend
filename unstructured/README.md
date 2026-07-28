# Unstructured RAG Streamlit App

Standalone Streamlit UI for unstructured document ingestion, grounded Q&A, and Knowledge Graph exploration.

## What This App Covers

1. Build Knowledge Base from documents (`pdf`, `docx`, `md`, `txt`, `json`)
2. Query KB with grounded retrieval (`/kb/query`)
3. Build and explore KG on Apache AGE (`/kg/build`, `/kg/query`, `/kg/subgraph`)

Main app file:
- `unstructured_app.py`

Detailed documentation:
- `UNSTRUCTURED_RAG_DOCUMENTATION.md`

## Prerequisites

- Python virtual environment with project dependencies
- Backend running on `http://127.0.0.1:8000`
- Ollama running locally
- Postgres/AGE configured (for KG features)

## Run (3 terminals)

### Terminal 1: Ollama
```bash
ollama serve
```

### Terminal 2: Backend
```bash
cd /Users/sathya/Documents/ml/backend
source ../.venv/bin/activate
pip install -r requirements.txt

export OLLAMA_OPENAI_BASE_URL="http://127.0.0.1:11434/v1"
export OLLAMA_OPENAI_API_KEY="ollama"
export OLLAMA_CHAT_MODEL="llama3.1:8b"
export OLLAMA_EMBED_MODEL="nomic-embed-text:latest"
export KB_PG_DSN="postgresql+psycopg2://postgres:mysecretpassword@127.0.0.1:5432/kbdb"
export KB_AGE_DSN="postgresql+psycopg2://postgres:mysecretpassword@127.0.0.1:55432/kbdb"
export KB_AGE_GRAPH_NAME="kb_graph"
export API_AUTH_ENABLED="false"
export DISTRIBUTED_ENABLED="false"
export DISTRIBUTED_BACKEND="auto"

python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

### Terminal 3: Streamlit (Unstructured App)
```bash
cd /Users/sathya/Documents/ml
source .venv/bin/activate
streamlit run unstructured_app.py
```

Open:
- Streamlit: `http://localhost:8501`
- Backend health: `http://127.0.0.1:8000/health`

## Core UI Flows

### 1) Build Knowledge Base
- Upload document
- Set optional `dataset_id`
- Select `llm_provider`
- Adjust build-performance knobs (chunk cap, NER/PII/enrichment controls)
- Click `Build KB`

### 2) Query Knowledge Base
- Provide dataset id + question + top-k
- Choose `Simple` or `Debug` view
- Click `Run Query`

### 3) Knowledge Graph (AGE)
- Build KG from dataset
- Query by entity
- Render subgraph in multiple view modes:
  - overview
  - entity-centric
  - community
  - layered

## APIs Used

- `GET /health`
- `GET /models`
- `POST /kb/build` (multipart)
- `POST /kb/query`
- `POST /kg/build`
- `POST /kg/query`
- `POST /kg/subgraph`

## Suggested Separate Repo Structure

```text
unstructured-rag-app/
├── unstructured_app.py
├── UNSTRUCTURED_RAG_DOCUMENTATION.md
├── README.md
└── .gitignore
```

`.gitignore`:
```gitignore
.venv/
__pycache__/
*.pyc
.DS_Store
.env
.env.*
```

## Push To Separate GitHub Repo

```bash
mkdir -p /Users/sathya/Documents/unstructured-rag-app
cp /Users/sathya/Documents/ml/unstructured_app.py /Users/sathya/Documents/unstructured-rag-app/
cp /Users/sathya/Documents/ml/UNSTRUCTURED_RAG_DOCUMENTATION.md /Users/sathya/Documents/unstructured-rag-app/
cp /Users/sathya/Documents/ml/unstructured/README.md /Users/sathya/Documents/unstructured-rag-app/README.md

cat > /Users/sathya/Documents/unstructured-rag-app/.gitignore << 'EOF'
.venv/
__pycache__/
*.pyc
.DS_Store
.env
.env.*
EOF

cd /Users/sathya/Documents/unstructured-rag-app
git init
git add .
git commit -m "Initial commit: Unstructured RAG Streamlit app"
git branch -M main
git remote add origin https://github.com/SATHYAGITH368/unstructured-rag-app.git
git push -u origin main
```
