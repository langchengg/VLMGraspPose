#!/usr/bin/env bash
set -euo pipefail

GQCNN_ROOT="${GQCNN_ROOT:-/opt/gqcnn}"
MODEL_ROOT="${MODEL_ROOT:-/models}"
ARCHIVE="${MODEL_ROOT}/model_zoo.zip"
mkdir -p "${MODEL_ROOT}"

echo "Attempting the official v1.3.0 repository download script first"
(cd "${GQCNN_ROOT}" && ./scripts/downloads/models/download_models.sh) || true
if [[ -f "${GQCNN_ROOT}/models/GQCNN-2.1/config.json" ]]; then
  cp -R "${GQCNN_ROOT}/models/GQCNN-2.1" "${MODEL_ROOT}/GQCNN-2.1"
  echo "Official repository script produced GQCNN-2.1"
else
  echo "Official repository script failed (the historical Box URLs currently return HTTP 404)" >&2
fi

if [[ -d "${MODEL_ROOT}/GQCNN-2.1" ]]; then
  echo "GQCNN-2.1 already available at ${MODEL_ROOT}/GQCNN-2.1"
  exit 0
fi

echo "Downloading the current official Dex-Net model zoo fallback"
curl -fL --retry 3 --retry-delay 2 \
  -o "${ARCHIVE}" \
  'https://drive.usercontent.google.com/download?id=1fbC0sGtVEUmAy7WPT_J-50IuIInMR9oO&export=download&confirm=t'

rm -rf "${MODEL_ROOT}/model_zoo_extract"
mkdir -p "${MODEL_ROOT}/model_zoo_extract"
unzip -q "${ARCHIVE}" 'model_zoo/GQCNN-2.1.zip' -d "${MODEL_ROOT}/model_zoo_extract"
unzip -q "${MODEL_ROOT}/model_zoo_extract/model_zoo/GQCNN-2.1.zip" \
  -d "${MODEL_ROOT}/model_zoo_extract/model_zoo/GQCNN-2.1"

MODEL_DIR="$(find "${MODEL_ROOT}/model_zoo_extract/model_zoo/GQCNN-2.1" -type f -name config.json -print -quit | xargs dirname)"
if [[ -z "${MODEL_DIR}" || ! -f "${MODEL_DIR}/config.json" ]]; then
  echo "Could not locate the official GQCNN-2.1 config.json after extraction" >&2
  exit 1
fi
mv "${MODEL_DIR}" "${MODEL_ROOT}/GQCNN-2.1"
test -f "${MODEL_ROOT}/GQCNN-2.1/config.json"
echo "GQCNN-2.1 ready at ${MODEL_ROOT}/GQCNN-2.1"
