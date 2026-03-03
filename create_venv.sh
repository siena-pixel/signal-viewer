#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${SCRIPT_DIR}/venv"
PYTHON="${PYTHON:-python3}"

echo "=== Engineering Signal Viewer — Environment Setup ==="
echo ""

# Check Python version
PYVER=$($PYTHON --version 2>&1 | cut -d' ' -f2)
echo "[1/4] Python version: $PYVER"

# Create venv
echo "[2/4] Creating virtual environment at ${VENV_DIR}..."
$PYTHON -m venv "$VENV_DIR"

# Activate
source "${VENV_DIR}/bin/activate"

# Upgrade pip
echo "[3/4] Upgrading pip..."
pip install --upgrade pip setuptools wheel --quiet

# Install package with all optional deps + dev tools
echo "[4/4] Installing signal-viewer with all dependencies..."
pip install -e "${SCRIPT_DIR}[full,dev]"

# Create data directory if it doesn't exist
mkdir -p "${SCRIPT_DIR}/data"

echo ""
echo "=== Setup complete! ==="
echo ""
echo "Activate the environment:"
echo "  source ${VENV_DIR}/bin/activate"
echo ""
echo "Place HDF5 data in:"
echo "  ${SCRIPT_DIR}/data/"
echo ""
echo "Run the app:"
echo "  signal-viewer"
echo "  # or: python -m signal_viewer.server.app"
echo ""
echo "Run tests:"
echo "  pytest tests/"
echo ""
echo "Open in browser: http://127.0.0.1:8050"
