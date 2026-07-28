#!/usr/bin/env bash
set -euo pipefail
if [[ $# -lt 1 ]]; then
  echo "usage: $0 CHECKPOINT [RUN_DIR] [extra evaluate args...]" >&2
  exit 2
fi
cd "$(dirname "$0")/.."
CHECKPOINT="$1"
shift
RUN_DIR="${1:-$(dirname "$(dirname "$CHECKPOINT")")}"
if [[ $# -gt 0 ]]; then shift; fi
eval "$(.venv/bin/python scripts/paths_to_env.py)"
export PYTORCH_ENABLE_MPS_FALLBACK=1
exec .venv/bin/python evaluate_ocid_vlg.py \
  --config configs/ocid_vlg_lavt_base_mps.yaml \
  --ocid_root "$OCID_ROOT" --ocid_api_root "$OCID_API_ROOT" \
  --test_manifest "$TEST_MANIFEST" \
  --ck_bert pretrained_weights/bert-base-uncased \
  --bert_tokenizer pretrained_weights/bert-base-uncased \
  --resume "$CHECKPOINT" --resolved_run_dir "$RUN_DIR" "$@"

