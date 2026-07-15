#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO="$BASE_DIR/hifics"
PYTHON="${PYTHON:-/opt/homebrew/bin/python3.11}"

mkdir -p "$BASE_DIR/reports" "$BASE_DIR/runs" "$BASE_DIR/logs" "$BASE_DIR/artifacts"
if [[ ! -x "$REPO/.venv/bin/python" ]]; then
  "$PYTHON" -m venv "$REPO/.venv"
fi
source "$REPO/.venv/bin/activate"
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r "$REPO/requirements-macos.txt"
python -m pip check
python - <<'PY'
import platform, sys, torch
print('Python:', sys.version)
print('Platform:', platform.platform())
print('Machine:', platform.machine())
print('PyTorch:', torch.__version__)
print('MPS built:', torch.backends.mps.is_built())
print('MPS available:', torch.backends.mps.is_available())
print('CUDA available:', torch.cuda.is_available())
PY
python -m pip freeze > "$BASE_DIR/reports/pip_freeze.txt"
echo "Environment ready: $REPO/.venv"

