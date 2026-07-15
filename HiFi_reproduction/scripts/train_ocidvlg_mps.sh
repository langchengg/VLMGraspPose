#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO="$BASE_DIR/hifics"
CONFIG="${1:-config_macos_ocidvlg.yaml}"
RUN_ID="hifics_ocidvlg_$(date +%Y%m%d_%H%M%S)"
RUN_DIR="${2:-$BASE_DIR/runs/$RUN_ID}"
SESSION="hifics_ocidvlg"
LOG="$BASE_DIR/logs/hifics_ocidvlg_training.log"
source "$REPO/.venv/bin/activate"
python -c 'import sys,torch,yaml; assert torch.backends.mps.is_available(), "MPS unavailable"; raw=yaml.safe_load(open(sys.argv[1])); assert len(raw["individual_configurations"]) > 1' "$REPO/experiments/$CONFIG"
if ps ax -o command= | grep -Eiq '[p]ython.*training.py'; then
  echo "Refusing duplicate: an existing HiFi training process is active" >&2
  exit 1
fi
if screen -ls 2>/dev/null | grep -q "[.]$SESSION"; then
  echo "Refusing duplicate: screen session $SESSION is active" >&2
  exit 1
fi
mkdir -p "$RUN_DIR"
TRAIN_MANIFEST="$REPO/datasets/ocidvlg_final_dataset/train/ocid_vlg_train.json"
TEST_MANIFEST="$REPO/datasets/ocidvlg_final_dataset/test/ocid_vlg_test.json"
if [[ ! -f "$TRAIN_MANIFEST" || ! -f "$TEST_MANIFEST" ]]; then
  echo "Prepared train/test manifests are required before full training" >&2
  exit 1
fi
cp "$REPO/experiments/$CONFIG" "$RUN_DIR/$CONFIG"
cp "$TRAIN_MANIFEST" "$RUN_DIR/ocid_vlg_train.json"
cp "$TEST_MANIFEST" "$RUN_DIR/ocid_vlg_test.json"
cp "$BASE_DIR/reports/macos_code_changes.md" "$RUN_DIR/code_change_summary.md"
git -C "$REPO" rev-parse HEAD > "$RUN_DIR/git_commit.txt"
python -c 'import torch; print("Selected device: mps; available:", torch.backends.mps.is_available())' | tee "$RUN_DIR/device.txt"
python -m pip freeze > "$RUN_DIR/environment.txt"
python -c 'import json,sys,yaml; raw=yaml.safe_load(open(sys.argv[3])); cfg={**raw["configuration"],**raw["individual_configurations"][1]}; physical=int(cfg["batch_size"]); accumulation=int(cfg.get("accumulation_steps",1)); print(json.dumps({"train_manifest":sys.argv[1],"test_manifest":sys.argv[2],"train_count":len(json.load(open(sys.argv[1]))),"test_count":len(json.load(open(sys.argv[2]))),"physical_batch_size":physical,"accumulation_steps":accumulation,"effective_batch_size":physical*accumulation,"seed":int(cfg.get("seed",42))}, indent=2))' "$TRAIN_MANIFEST" "$TEST_MANIFEST" "$REPO/experiments/$CONFIG" > "$RUN_DIR/dataset_and_batch_metadata.json"
shasum -a 256 "$TRAIN_MANIFEST" "$TEST_MANIFEST" > "$RUN_DIR/manifest_sha256.txt"
date -u +%FT%TZ > "$RUN_DIR/start_time_utc.txt"
printf '%s\n' 'Measured 100-update rate projects about 3.4 hours; operational estimate 3.5-5 hours including startup, I/O, and checkpoints.' > "$RUN_DIR/estimated_finish_note.txt"
screen -dmS "$SESSION" zsh -lc "set -o pipefail; cd '$REPO' && caffeinate -i '$REPO/.venv/bin/python' training.py '$CONFIG' 1 --device mps --run-dir '$RUN_DIR' 2>&1 | tee -a '$LOG' '$RUN_DIR/training.log'"
STARTED=false
for _ in {1..10}; do
  if screen -ls 2>/dev/null | grep -q "[.]$SESSION" || ps ax -o command= | grep -Fq "training.py $CONFIG 1 --device mps --run-dir $RUN_DIR"; then
    STARTED=true
    break
  fi
  sleep 1
done
if [[ "$STARTED" != true ]]; then
  echo "Training process exited during startup; inspect $LOG" >&2
  tail -n 40 "$LOG" >&2 || true
  exit 1
fi
echo "Training launched in screen session: $SESSION"
echo "Run directory: $RUN_DIR"
echo "Training log: $LOG"
