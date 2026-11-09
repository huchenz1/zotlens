#!/usr/bin/env bash
# ZotLens launcher
#   ./run.sh            start the web app  ->  http://127.0.0.1:8765
#   ./run.sh check      verify embedding / rerank / chat backends (no index is written)
#   ./run.sh index      build or incrementally update the index (reuses unchanged embeddings)
#   ./run.sh eval [N]   retrieval evaluation with auto-generated questions -> data/eval_results.md
#   ./run.sh mcp        MCP server over stdio (Claude Desktop / Cursor / WorkBuddy)
#   ./run.sh mcp-http   MCP server over streamable HTTP -> http://127.0.0.1:8766/mcp
#   ./run.sh status     index + server status
#
# Config comes from .env (gitignored) or the environment. The `cloud` profile needs only
# OPENROUTER_API_KEY; the `local` profile needs the two Qwen3 models under ~/models/zotlens.
# Set ZOTLENS_PY to use another interpreter.
set -euo pipefail
cd "$(dirname "$0")"
if [[ -n "${ZOTLENS_PY:-}" ]]; then
  PY="$ZOTLENS_PY"
elif [[ -x ".venv/bin/python" ]]; then
  PY=".venv/bin/python"
else
  PY="python3"
fi
export TOKENIZERS_PARALLELISM=false PYTHONWARNINGS=ignore

case "${1:-serve}" in
  serve)    exec "$PY" -u -m zotlens.server ;;
  check)    exec "$PY" -m zotlens.preflight ;;
  index)    exec "$PY" -u -m zotlens.build_index ;;
  eval)     exec "$PY" -u -m zotlens.evaluate --n "${2:-100}" ${3:-} ;;
  mcp)      exec "$PY" -m zotlens.mcp_server ;;
  mcp-http) exec "$PY" -m zotlens.mcp_server --http ;;
  status)
    cat data/status.json 2>/dev/null; echo
    curl -s --noproxy '*' -m 3 "http://127.0.0.1:${ZOTLENS_PORT:-8765}/api/status" | head -c 400 \
      || echo "server not running"; echo ;;
  *) sed -n '2,14p' "$0" ;;
esac
