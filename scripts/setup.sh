#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-.venv}"

"$PYTHON_BIN" -m venv "$VENV_DIR"
"$VENV_DIR/bin/python" -m pip install --upgrade pip
"$VENV_DIR/bin/python" -m pip install -r requirements-dev.txt
"$VENV_DIR/bin/python" -m pip install -e .

echo "MY-AI environment ready. Activate with: source $VENV_DIR/bin/activate"
echo "Run tests with: pytest"
