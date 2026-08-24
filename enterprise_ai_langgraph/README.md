# Enterprise AI LangGraph (Modular)

This folder contains a modular version of your Streamlit app with a LangGraph-based agent orchestration pipeline.

## Structure

- `app.py`: Streamlit entrypoint
- `src/config.py`: environment-based settings
- `src/clients.py`: cached external clients (Brevo, OpenAI/Ollama, Presidio, Chroma, Whisper)
- `src/database.py`: SQLite init + interaction logging
- `src/services/`: email + ticket services
- `src/agents/`: governance and LLM agents
- `src/models/state.py`: shared LangGraph state schema
- `src/graph/workflow.py`: LangGraph workflow wiring all agents
- `src/ui/`: one module per menu page

## LangGraph pipeline

`sanitize -> classify -> ticket -> context -> kb -> strategy -> draft`

## Run

```bash
cd /Users/sathya/Documents/ml/enterprise_ai_langgraph
pip install -r requirements.txt
streamlit run app.py
```

## Environment variables

Set these before running:

- `BREVO_API_KEY`
- `SENDER_EMAIL`
- `OLLAMA_OPENAI_BASE_URL` (default: `http://localhost:11434/v1`)
- `OLLAMA_MODEL_NAME` (default: `llama3.1:8b`)
- `OPENAI_API_KEY` (default: `ollama`)
- `APP_DB_PATH` (default: `enterprise_ai_system.db`)
- `CHROMA_PATH` (default: `./startup_kb_vector`)
- `KB_COLLECTION_NAME` (default: `kb_storage`)
- `EMBEDDING_MODEL` (default: `all-MiniLM-L6-v2`)
- `WHISPER_MODEL_NAME` (default: `base`)
