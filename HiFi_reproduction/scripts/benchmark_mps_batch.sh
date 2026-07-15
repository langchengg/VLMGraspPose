#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO="$BASE_DIR/hifics"
source "$REPO/.venv/bin/activate"
cd "$REPO"
echo "Selected device: mps"
python tools/benchmark_mps_batch.py --device mps --sizes 1 2 4 8 16 --iterations 3 \
  --output "$BASE_DIR/reports/mps_batch_benchmark.csv" "$@" \
  2>&1 | tee "$BASE_DIR/logs/mps_batch_benchmark.log"
echo "Benchmark CSV: $BASE_DIR/reports/mps_batch_benchmark.csv"

