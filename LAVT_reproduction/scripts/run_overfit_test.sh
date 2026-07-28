#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
eval "$(.venv/bin/python scripts/paths_to_env.py)"
export PYTORCH_ENABLE_MPS_FALLBACK=1
.venv/bin/python train_ocid_vlg.py \
  --config configs/ocid_vlg_lavt_base_mps.yaml \
  --ocid_root "$OCID_ROOT" --ocid_api_root "$OCID_API_ROOT" \
  --train_manifest "$TRAIN_MANIFEST" --val_manifest "$TRAIN_MANIFEST" \
  --validation_split train \
  --pretrained_swin_weights pretrained_weights/swin_base_patch4_window12_384_22k.pth \
  --ck_bert pretrained_weights/bert-base-uncased \
  --bert_tokenizer pretrained_weights/bert-base-uncased \
  --limit_train_samples 8 --limit_val_samples 8 \
  --img_size 128 \
  --grad_accum_steps 1 --effective_batch_size 1 \
  --epochs 30 --output_root outputs/overfit_test \
  --run_name overfit8 "$@"

RUN_DIR="$(ls -dt outputs/overfit_test/*_overfit8 | head -1)"
.venv/bin/python scripts/check_overfit.py "$RUN_DIR"
