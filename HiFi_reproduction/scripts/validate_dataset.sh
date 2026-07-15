#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO="$BASE_DIR/hifics"
source "$REPO/.venv/bin/activate"
cd "$REPO"
echo "Selected device: validation is CPU I/O only"
python datasets/prepare_ocidvlg.py \
  --source-root datasets/OCID-VLG \
  --output-root datasets/ocidvlg_final_dataset \
  --splits train test --validate-only "$@" 2>&1 | tee "$BASE_DIR/logs/ocidvlg_validation.log"
echo "Validation log: $BASE_DIR/logs/ocidvlg_validation.log"

