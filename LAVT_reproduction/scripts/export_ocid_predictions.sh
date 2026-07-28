#!/usr/bin/env bash
set -euo pipefail
# Evaluation always writes probability arrays, binary masks, metadata, and a
# verified predictions manifest; this wrapper keeps the requested entry point.
exec "$(dirname "$0")/evaluate_ocid_vlg.sh" "$@"

