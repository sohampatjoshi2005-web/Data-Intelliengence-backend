#!/bin/zsh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
exec "$ROOT/.venv/bin/python" "$ROOT/scripts/test_opensearch_e2e.py" "$@"
