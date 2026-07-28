#!/usr/bin/env bash
set -euo pipefail
if [[ $# -lt 1 ]]; then
  echo "usage: $0 LAVT_RUN_DIR [extra comparison args...]" >&2
  exit 2
fi
cd "$(dirname "$0")/.."
RUN_DIR="$1"
shift
eval "$(.venv/bin/python scripts/paths_to_env.py)"
exec .venv/bin/python compare_with_hifics.py \
  --lavt-manifest "$RUN_DIR/predictions_manifest.jsonl" \
  --hifics-export-manifest "$HIFICS_EXPORT_MANIFEST" \
  --test-manifest "$TEST_MANIFEST" \
  --output-dir "$RUN_DIR" "$@"

