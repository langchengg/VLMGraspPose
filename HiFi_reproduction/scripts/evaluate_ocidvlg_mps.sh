#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO="$BASE_DIR/hifics"
RUN_DIR="${1:?usage: evaluate_ocidvlg_mps.sh RUN_DIR CHECKPOINT [OUTPUT_DIR]}"
CHECKPOINT="${2:?usage: evaluate_ocidvlg_mps.sh RUN_DIR CHECKPOINT [OUTPUT_DIR]}"
OUTPUT_DIR="${3:-$RUN_DIR/evaluation}"

absolute_path() {
  case "$1" in
    /*) printf '%s\n' "$1" ;;
    *) printf '%s/%s\n' "$PWD" "$1" ;;
  esac
}

RUN_DIR="$(absolute_path "$RUN_DIR")"
CHECKPOINT="$(absolute_path "$CHECKPOINT")"
OUTPUT_DIR="$(absolute_path "$OUTPUT_DIR")"
mkdir -p "$OUTPUT_DIR"
RUN_DIR="$(cd "$RUN_DIR" && pwd -P)"
CHECKPOINT="$(cd "$(dirname "$CHECKPOINT")" && pwd -P)/$(basename "$CHECKPOINT")"
OUTPUT_DIR="$(cd "$OUTPUT_DIR" && pwd -P)"
TEST_MANIFEST="$RUN_DIR/ocid_vlg_test.json"
if [[ ! -f "$TEST_MANIFEST" ]]; then
  echo "Frozen test manifest not found: $TEST_MANIFEST" >&2
  exit 1
fi

source "$REPO/.venv/bin/activate"
cd "$REPO"
echo "Selected device: mps"
env -u PYTORCH_ENABLE_MPS_FALLBACK python score.py config_macos_ocidvlg.yaml 1 1 \
  --device mps --run-dir "$RUN_DIR" --checkpoint "$CHECKPOINT" \
  --test-json "$TEST_MANIFEST" --output-dir "$OUTPUT_DIR" \
  2>&1 | tee "$OUTPUT_DIR/evaluation.log"
echo "Evaluation outputs: $OUTPUT_DIR"
