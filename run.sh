#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Activate venv if exists
if [ -f "${SCRIPT_DIR}/venv/bin/activate" ]; then
    source "${SCRIPT_DIR}/venv/bin/activate"
fi

export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH:-}"
export SIGNAL_VIEWER_DEBUG=true

echo "Starting Engineering Signal Viewer..."
echo "Open http://127.0.0.1:8050 in your browser"
echo ""
python3 -m signal_viewer.server.app "$@"
