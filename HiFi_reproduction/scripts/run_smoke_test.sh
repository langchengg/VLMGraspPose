#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO="$BASE_DIR/hifics"
MANIFEST="${1:-$REPO/datasets/ocidvlg_smoke_dataset/train/ocid_vlg_train.json}"
source "$REPO/.venv/bin/activate"
cd "$REPO"
python -c 'import torch; print("Selected device: mps; available:", torch.backends.mps.is_available())'
python tools/mac_smoke_test.py --manifest "$MANIFEST" --device mps \
  2>&1 | tee "$BASE_DIR/logs/mac_smoke_test.log"
echo "Smoke report: $BASE_DIR/reports/mac_smoke_test.json"

