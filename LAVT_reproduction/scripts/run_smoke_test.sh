#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
eval "$(.venv/bin/python scripts/paths_to_env.py)"
export PYTORCH_ENABLE_MPS_FALLBACK=1

.venv/bin/python train_ocid_vlg.py \
  --config configs/ocid_vlg_lavt_base_mps.yaml \
  --ocid_root "$OCID_ROOT" --ocid_api_root "$OCID_API_ROOT" \
  --train_manifest "$TRAIN_MANIFEST" --val_manifest "$VAL_MANIFEST" \
  --pretrained_swin_weights pretrained_weights/swin_base_patch4_window12_384_22k.pth \
  --ck_bert pretrained_weights/bert-base-uncased \
  --bert_tokenizer pretrained_weights/bert-base-uncased \
  --limit_train_samples 4 --limit_val_samples 2 \
  --grad_accum_steps 1 --effective_batch_size 1 \
  --epochs 2 --stop_after_epochs 1 --run_name smoke_resume

RUN_DIR="$(ls -dt outputs/ocid_vlg/*_smoke_resume | head -1)"
.venv/bin/python train_ocid_vlg.py \
  --config configs/ocid_vlg_lavt_base_mps.yaml \
  --ocid_root "$OCID_ROOT" --ocid_api_root "$OCID_API_ROOT" \
  --train_manifest "$TRAIN_MANIFEST" --val_manifest "$VAL_MANIFEST" \
  --ck_bert pretrained_weights/bert-base-uncased \
  --bert_tokenizer pretrained_weights/bert-base-uncased \
  --limit_train_samples 4 --limit_val_samples 2 \
  --grad_accum_steps 1 --effective_batch_size 1 \
  --epochs 2 --run_name smoke_resume \
  --resume "$RUN_DIR/checkpoints/checkpoint_last.pth" \
  --resolved_run_dir "$RUN_DIR"

.venv/bin/python evaluate_ocid_vlg.py \
  --config configs/ocid_vlg_lavt_base_mps.yaml \
  --ocid_root "$OCID_ROOT" --ocid_api_root "$OCID_API_ROOT" \
  --test_manifest "$TEST_MANIFEST" \
  --ck_bert pretrained_weights/bert-base-uncased \
  --bert_tokenizer pretrained_weights/bert-base-uncased \
  --limit_test_samples 2 \
  --resume "$RUN_DIR/checkpoints/checkpoint_last.pth" \
  --resolved_run_dir "$RUN_DIR"
