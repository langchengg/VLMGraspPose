#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
IMAGE="${SAM3_DOCKER_IMAGE:-vlmgrasp/sam3-refinement:1.0.0}"
MODEL_CACHE="${SAM3_MODEL_CACHE:-${REPO_ROOT}/models/sam3-huggingface}"

mkdir -p "${MODEL_CACHE}"
exec docker run --rm --gpus all \
  --shm-size=8g \
  -e HF_TOKEN \
  -v "${REPO_ROOT}/..:/workspace" \
  -v "${MODEL_CACHE}:/models/huggingface" \
  -w /workspace/HiFi_reproduction \
  "${IMAGE}" "$@"

