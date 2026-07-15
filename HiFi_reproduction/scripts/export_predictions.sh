#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO="$BASE_DIR/hifics"
RUN_DIR="${1:?usage: export_predictions.sh RUN_DIR CHECKPOINT METRICS_CSV [OUTPUT_DIR]}"
CHECKPOINT="${2:?missing checkpoint}"
METRICS_CSV="${3:?missing per-sample metrics CSV}"
OUTPUT_DIR="${4:-$RUN_DIR/figures}"
TEST_JSON="${5:-$RUN_DIR/ocid_vlg_test.json}"
SOURCE_ROOT="${6:-$BASE_DIR/../crog_reproduction/OCID-VLG}"

absolute_path() {
  case "$1" in
    /*) printf '%s\n' "$1" ;;
    *) printf '%s/%s\n' "$PWD" "$1" ;;
  esac
}

RUN_DIR="$(absolute_path "$RUN_DIR")"
CHECKPOINT="$(absolute_path "$CHECKPOINT")"
METRICS_CSV="$(absolute_path "$METRICS_CSV")"
OUTPUT_DIR="$(absolute_path "$OUTPUT_DIR")"
TEST_JSON="$(absolute_path "$TEST_JSON")"
SOURCE_ROOT="$(absolute_path "$SOURCE_ROOT")"
mkdir -p "$OUTPUT_DIR"
RUN_DIR="$(cd "$RUN_DIR" && pwd -P)"
CHECKPOINT="$(cd "$(dirname "$CHECKPOINT")" && pwd -P)/$(basename "$CHECKPOINT")"
METRICS_CSV="$(cd "$(dirname "$METRICS_CSV")" && pwd -P)/$(basename "$METRICS_CSV")"
OUTPUT_DIR="$(cd "$OUTPUT_DIR" && pwd -P)"
TEST_JSON="$(cd "$(dirname "$TEST_JSON")" && pwd -P)/$(basename "$TEST_JSON")"
SOURCE_ROOT="$(cd "$SOURCE_ROOT" && pwd -P)"
LOG="$OUTPUT_DIR/export_predictions.log"

source "$REPO/.venv/bin/activate"
cd "$REPO"
echo "Selected device: mps"
env -u PYTORCH_ENABLE_MPS_FALLBACK python tools/export_ocidvlg_predictions.py --device mps \
  --run-dir "$RUN_DIR" --checkpoint "$CHECKPOINT" \
  --test-json "$TEST_JSON" --metrics-csv "$METRICS_CSV" \
  --source-root "$SOURCE_ROOT" --output-dir "$OUTPUT_DIR" \
  2>&1 | tee "$LOG"
echo "Full predictions: $RUN_DIR/predictions"
echo "Prediction figures: $OUTPUT_DIR"
echo "Export log: $LOG"
