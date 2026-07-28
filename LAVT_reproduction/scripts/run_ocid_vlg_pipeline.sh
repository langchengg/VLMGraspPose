#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

.venv/bin/python scripts/discover_data.py \
  --version unique
eval "$(.venv/bin/python scripts/paths_to_env.py)"
.venv/bin/python scripts/audit_ocid_vlg.py \
  --train-manifest "$TRAIN_MANIFEST" \
  --val-manifest "$VAL_MANIFEST" \
  --test-manifest "$TEST_MANIFEST" \
  --tokenizer pretrained_weights/bert-base-uncased \
  --max-tokens 20 \
  --image-size 480 \
  --seed 42 \
  --json-output outputs/dataset_audit.json \
  --token-output outputs/token_length_audit.json \
  --markdown-output docs/OCID_VLG_DATASET_AUDIT.md \
  --visualization-dir outputs/audit_visualizations
.venv/bin/python -m pytest -q 2>&1 | tee outputs/tests.log
scripts/run_overfit_test.sh
scripts/run_smoke_test.sh
scripts/train_ocid_vlg.sh
RUN_DIR="$(ls -dt outputs/ocid_vlg/*_lavt_base_unique_full | head -1)"
scripts/evaluate_ocid_vlg.sh \
  "$RUN_DIR/checkpoints/checkpoint_best_miou.pth" "$RUN_DIR"
scripts/compare_with_hifics.sh "$RUN_DIR"
