# SAM 3 coarse-to-fine pipeline audit

Audit date: 2026-07-16. This document records discovered local artifacts; it does not
claim that SAM 3 inference ran.

## Source of truth and protected outputs

The frozen HiFi-CS prediction run is
`runs/hifics_ocidvlg_20260711_112921`. The prediction-only Dex-Net input root is
`runs/hifics_ocidvlg_20260711_112921/anygrasp_input_predicted_mask`, backed by its
`manifest.jsonl`. Ground-truth masks live separately under the run's `predictions`
directories and are evaluation-only.

The following roots are protected against mutation:

- `runs/hifics_ocidvlg_20260711_112921/predictions`
- `runs/hifics_ocidvlg_20260711_112921/anygrasp_input_predicted_mask`
- `outputs/dexnet_candidates_ten_samples`
- `outputs/gqcnn_original_ranking_evaluation`

`outputs/sam3_protected_baseline_hashes.json` records SHA-256 values for 130,847 files.
Run `python scripts/audit_sam3_protected_outputs.py verify` after the experiment.

## Actual data contract

Each frozen HiFi bundle contains `color.png`, `depth.png`, `target_mask.png`,
`target_probability.npy`, `language.txt`, `intrinsics.json`, `metadata.json`, and
`checksums.sha256`.

| Signal | Shape | Type / units | Convention |
|---|---:|---|---|
| RGB | 480 × 640 × 3 | uint8 | RGB |
| predicted mask | 480 × 640 | uint8 | foreground 255, background 0 |
| predicted probability | 480 × 640 | float32 | finite foreground probability |
| depth | 480 × 640 | uint16 | millimetres; unchanged full scene |

The stored foreground probability is `1 - sigmoid(logit)`. The binary prediction equals
`probability >= 0.15000000000000002` for all ten samples. The preserved probability map
must not be reconstructed from the binary mask. SAM prompt cleanup removes only connected
components below the configured minimum; it does not force a largest-component choice.

The ten-sample RGB and depth provenance both resolve to the same OCID frame:
`ARID10/floor/top/non-fruits/seq09,result_2018-08-27-16-13-28.png`. The language queries
and target masks differ by question. `sample_index` and `question_index` diverge after the
first five rows, so `sample_id` is the only join key.

| sample_id | sample index | question index | coarse area (px) |
|---|---:|---:|---:|
| q0000000_b32eb3299dcd3ae9 | 0 | 0 | 3,481 |
| q0000001_a9a5f9b502546016 | 1 | 1 | 2,016 |
| q0000002_65b99b4d1aaf2b7b | 2 | 2 | 7,200 |
| q0000003_c9f21176e1f0d767 | 3 | 3 | 21,544 |
| q0000004_23c60d130c4f6a9e | 4 | 4 | 1,773 |
| q0000006_479d6565d0c9dc86 | 6 | 6 | 5,062 |
| q0000012_3c71c3e0ca6b64a8 | 12 | 12 | 1,455 |
| q0000014_365bf2150cb2b2fb | 14 | 14 | 2,028 |
| q0000020_70edb52adfcfaa63 | 20 | 20 | 4,605 |
| q0000021_ab3537b86eafece1 | 21 | 21 | 2,198 |

## No-GT refinement boundary

`outputs/sam3_refinement_inputs` contains exactly four files per sample: RGB, coarse
mask, preserved probability, and allowlisted metadata. It contains no GT mask, GT box,
answer instance, grasp annotation, evaluation IoU, or visualization that overlays GT.
Depth provenance is recorded but depth is not a SAM input. Optional depth consistency may
only read the unchanged depth at inference time when explicitly enabled; it is disabled in
the frozen default configuration.

GT is first opened by `scripts/evaluate_sam3_mask_refinement.py`, after selection metadata
and selected masks already exist.

## Existing downstream implementation

- Candidate generation: `scripts/run_hifics_dexnet_candidates.py`
- Candidate configuration: `configs/dexnet_candidates.yaml`
- Frozen generation settings: 256 requests, top 30, seed 42
- GQ-CNN fixed-candidate scorer: `scripts/score_existing_dexnet_candidates.py`
- Model: official GQCNN-2.1 in the existing linux/amd64 TensorFlow 1.15 Docker runtime
- Geometric ranker: `scripts/rank_existing_dexnet_candidates.py`
- Frozen geometric configuration: `configs/dexnet_geometric_ranker.yaml`
- Shared predicate: `configs/dexnet_grasp_consistency.yaml`

The predicate is 2D consistency with OCID-VLG planar grasp annotations: angle difference
modulo pi at most 30 degrees and grasp-rectangle IoU at least 0.25 against any target
annotation, with Top-5 cutoff 5. Depth is not part of this predicate. It is not physical
grasp success.

The SAM adapter creates a new ordinary-file bundle root rather than symlinking or modifying
the old root. RGB, full depth, intrinsics, language, sampler, seed, filtering, and NMS remain
identical; only `target_mask.png` and its matching probability change.

## Environment and official model audit

The host is macOS 26.3.2 arm64 on Apple M5 Pro (16-core Metal GPU). PyTorch reports MPS
available and CUDA unavailable; `nvidia-smi` and `nvcc` are absent. Therefore full SAM 3
inference is prohibited on this host.

The logged-in Hugging Face account is `langcheng`. Official metadata resolves
`facebook/sam3` to commit `3c879f39826c281e95690f02c7821c4de09afae7`, but an authenticated
config download returns HTTP 403 because this account is not on the gated repository's
authorized list. The credential file permission was tightened from 0644 to 0600 without
reading the secret.

The official Transformers checkpoint is `model.safetensors` (3,439,938,512 bytes,
SHA-256 `6d06f0a5f84e435071fe6603e61d0b4cc7b40e0d39d487cfd4d67d8cc11cc14a`).
The separate native `sam3.pt` checkpoint has a different hash and is deliberately excluded
from the Transformers download allowlist.

Official references checked:

- <https://github.com/facebookresearch/sam3>
- <https://huggingface.co/facebook/sam3>
- <https://huggingface.co/docs/transformers/model_doc/sam3_tracker>

