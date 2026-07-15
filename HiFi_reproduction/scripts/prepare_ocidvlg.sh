#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO="$BASE_DIR/hifics"
source "$REPO/.venv/bin/activate"
cd "$REPO"
echo "Selected device: dataset preparation is CPU I/O only"
python datasets/prepare_ocidvlg.py \
  --source-root datasets/OCID-VLG \
  --output-root datasets/ocidvlg_final_dataset \
  --splits train test "$@" 2>&1 | tee "$BASE_DIR/logs/ocidvlg_preparation.log"
echo "Prepared dataset: $REPO/datasets/ocidvlg_final_dataset"

