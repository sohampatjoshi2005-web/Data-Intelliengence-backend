#!/bin/zsh
# Start AutoML backend on :8000 using the project venv (not conda/homebrew python).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT/backend"

if [[ -f "$ROOT/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$ROOT/.env"
  set +a
fi

if [[ ! -x "$ROOT/.venv/bin/python" ]]; then
  echo "Missing venv at $ROOT/.venv"
  echo "Create it: cd $ROOT && python3.11 -m venv .venv && .venv/bin/pip install -r backend/requirements.txt"
  exit 1
fi

export API_AUTH_ENABLED=false
export OLLAMA_OPENAI_BASE_URL="${OLLAMA_OPENAI_BASE_URL:-http://127.0.0.1:11434/v1}"
export OLLAMA_CHAT_MODEL="${OLLAMA_CHAT_MODEL:-llama3.2:3b}"
export STRUCTURED_LLM_PROVIDER="${STRUCTURED_LLM_PROVIDER:-ollama_local}"
export CORS_ORIGINS="${CORS_ORIGINS:-http://localhost:5173,http://127.0.0.1:5173,http://localhost:8501,http://127.0.0.1:8501}"
export OPENSEARCH_URL="${OPENSEARCH_URL:-http://127.0.0.1:9200}"
export WEB_SEARCH_PROVIDER="${WEB_SEARCH_PROVIDER:-tavily}"

echo "Starting backend: $ROOT/backend"
echo "Python: $ROOT/.venv/bin/python ($($ROOT/.venv/bin/python --version))"
echo "Wait ~15-20s on first startup..."
exec "$ROOT/.venv/bin/python" -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
