#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO="$BASE_DIR/hifics"
RUN_DIR="${1:?usage: resume_training.sh RUN_DIR CHECKPOINT}"
CHECKPOINT="${2:?missing checkpoint}"
SESSION="hifics_ocidvlg"
LOG="$BASE_DIR/logs/hifics_ocidvlg_training.log"
source "$REPO/.venv/bin/activate"
if [[ ! -f "$CHECKPOINT" ]]; then
  echo "Checkpoint not found: $CHECKPOINT" >&2
  exit 1
fi
python -c 'import sys,torch; assert torch.backends.mps.is_available(), "MPS unavailable"; payload=torch.load(sys.argv[1],map_location="cpu",weights_only=False); assert "model_state" in payload and "metadata" in payload, "not a full-state checkpoint"' "$CHECKPOINT"
if ps ax -o command= | grep -Eiq '[p]ython.*training.py' || screen -ls 2>/dev/null | grep -q "[.]$SESSION"; then
  echo "Refusing duplicate: HiFi training is already active" >&2
  exit 1
fi
echo "Selected device: mps"
screen -dmS "$SESSION" zsh -lc "set -o pipefail; cd '$REPO' && caffeinate -i '$REPO/.venv/bin/python' training.py config_macos_ocidvlg.yaml 1 --device mps --run-dir '$RUN_DIR' --resume '$CHECKPOINT' 2>&1 | tee -a '$LOG' '$RUN_DIR/training.log'"
STARTED=false
for _ in {1..10}; do
  if screen -ls 2>/dev/null | grep -q "[.]$SESSION" || ps ax -o command= | grep -Fq -- "--run-dir $RUN_DIR --resume $CHECKPOINT"; then
    STARTED=true
    break
  fi
  sleep 1
done
if [[ "$STARTED" != true ]]; then
  echo "Resume process exited during startup; inspect $LOG" >&2
  tail -n 40 "$LOG" >&2 || true
  exit 1
fi
echo "Resumed in screen session: $SESSION"
echo "Run directory: $RUN_DIR"
echo "Log: $LOG"
