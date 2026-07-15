#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO="$BASE_DIR/hifics"
echo "Git status:"
git -C "$REPO" status --short
echo "Active training processes:"
ps ax -o pid=,command= | grep -Ei '[p]ython.*training.py' || true
echo "Active screen sessions:"
screen -ls 2>/dev/null || true
LATEST_CHECKPOINT="$(find "$BASE_DIR/runs" -name latest.pth -type f -exec ls -t {} + 2>/dev/null | head -1 || true)"
echo "Latest checkpoint: ${LATEST_CHECKPOINT:-none}"
LATEST_LOG="$BASE_DIR/logs/hifics_ocidvlg_training.log"
if [[ -f "$LATEST_LOG" ]]; then
  echo "Latest log timestamp: $(stat -f '%Sm' "$LATEST_LOG")"
  echo "Latest training update:"
  grep -E 'update [0-9]+/' "$LATEST_LOG" | tail -1 || true
fi
echo "Disk space:"
df -h "$BASE_DIR"
echo "Most recent evaluation metrics:"
LATEST_METRICS="$(find "$BASE_DIR/runs" -name evaluation_metrics.json -type f -exec ls -t {} + 2>/dev/null | head -1 || true)"
if [[ -n "$LATEST_METRICS" ]]; then
  echo "$LATEST_METRICS"
  cat "$LATEST_METRICS"
else
  echo "none"
fi
