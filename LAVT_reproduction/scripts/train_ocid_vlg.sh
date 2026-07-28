#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
eval "$(.venv/bin/python scripts/paths_to_env.py)"
export PYTORCH_ENABLE_MPS_FALLBACK=1
exec .venv/bin/python train_ocid_vlg.py \
  --config configs/ocid_vlg_lavt_base_mps.yaml \
  --ocid_root "$OCID_ROOT" --ocid_api_root "$OCID_API_ROOT" \
  --train_manifest "$TRAIN_MANIFEST" --val_manifest "$VAL_MANIFEST" \
  --pretrained_swin_weights pretrained_weights/swin_base_patch4_window12_384_22k.pth \
  --ck_bert pretrained_weights/bert-base-uncased \
  --bert_tokenizer pretrained_weights/bert-base-uncased \
  --run_name lavt_base_unique_full "$@"

