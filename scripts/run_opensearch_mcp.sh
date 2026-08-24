#!/bin/zsh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT/backend"

if [[ ! -x "$ROOT/.venv/bin/python" ]]; then
  echo "Missing venv at $ROOT/.venv"
  exit 1
fi

export PYTHONPATH="$ROOT/backend${PYTHONPATH:+:$PYTHONPATH}"
exec "$ROOT/.venv/bin/python" -m app.mcp.server
