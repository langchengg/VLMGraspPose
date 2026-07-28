#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

clone_at_commit() {
  local name="$1"
  local url="$2"
  local destination="$3"
  local commit="$4"
  local absolute_destination="${ROOT}/${destination}"

  if [[ -e "${absolute_destination}" ]]; then
    printf 'skip %-20s existing path: %s\n' "${name}" "${destination}"
    return
  fi

  mkdir -p "$(dirname "${absolute_destination}")"
  git clone --no-checkout "${url}" "${absolute_destination}"
  git -C "${absolute_destination}" fetch --depth=1 origin "${commit}"
  git -C "${absolute_destination}" checkout --detach "${commit}"
  printf 'ready %-19s %s\n' "${name}" "${commit}"
}

clone_at_commit \
  "HiFi-CS" \
  "https://github.com/vineet2104/hifics.git" \
  "HiFi_reproduction/hifics" \
  "4be6b3be7ce79fae481fb51616adfa2b803f07a0"

clone_at_commit \
  "GQ-CNN" \
  "https://github.com/BerkeleyAutomation/gqcnn.git" \
  "HiFi_reproduction/third_party/gqcnn-official" \
  "499a609fe9dfb074bdfb6c4e6e33667ea50f4c21"

clone_at_commit \
  "GraspNet API" \
  "https://github.com/graspnet/graspnetAPI.git" \
  "legacy/external_graspnet/graspnetAPI" \
  "bd6783c3effdebd895abfba8b96dc22a42ec3b5a"

clone_at_commit \
  "GraspNet baseline" \
  "https://github.com/graspnet/graspnet-baseline.git" \
  "legacy/external_graspnet/graspnet-baseline" \
  "280c215129f759ed8649cb4e89fc5dfee55f4f80"

clone_at_commit \
  "VL-Grasp" \
  "https://github.com/luyh20/VL-Grasp.git" \
  "ranking_baseline/VL-Grasp" \
  "dd6bd6d7b4045b8b72df7d4bebb6ff4a1344076f"
