# Repeated-FiLM retained-stage reuse audit

Audit time: 2026-07-29T19:44:06Z.

The retained test pipeline
`runs/modular_hierfilm_standard_dexnet_gqcnn_20260728_094528` is a complete
7,675-sample repeated-FiLM chain, not a checkpoint-only fragment. It occupies
about 26.85 GiB and is now protected by `.DO_NOT_PRUNE`.

## Reuse decision

| Stage | Status | Decision |
|---|---|---|
| test manifest | `COMPLETE_AND_REUSABLE` | reference frozen 7,675 rows |
| predicted masks | `COMPLETE_AND_REUSABLE` | reference 7,675 masks; one is empty |
| soft probabilities | `COMPLETE_AND_REUSABLE` | reference lossless float32 arrays |
| RGB/depth/intrinsics | `COMPLETE_AND_REUSABLE` | reference each of 325 scenes once |
| Dex-Net raw | `COMPLETE_AND_REUSABLE` | reuse 1,466,046-row Parquet |
| mask-validated | `PARTIAL_BUT_RESUMABLE` | losslessly convert 1,449,011 retained JSON candidates to compact Parquet |
| NMS | `COMPLETE_AND_REUSABLE` | reuse 187,077-row Parquet |
| GQ-CNN q | `PARTIAL_BUT_RESUMABLE` | project 187,077 finite q-values to a clean inference-only table |
| candidate identity | `COMPLETE_AND_REUSABLE` | NMS and q keys/geometry match exactly |
| existing evaluator | `COMPLETE_AND_REUSABLE` | post-hoc baseline check only |
| raw/mask Oracle funnel | `MISSING_REGENERATE` | re-evaluate retained candidates without model inference |
| train/val downstream chain | `MISSING_REGENERATE` | compact repeated-FiLM generation required |
| local Qwen3-VL 4B model | `COMPLETE_AND_REUSABLE` | pin and re-audit locally |
| VLM results/cache | `MISSING_REGENERATE` | run afresh in the new lineage |

Therefore test HiFi, Dex-Net and GQ-CNN inference will not be rerun. Train and
validation predictions/candidates/q-values must be regenerated.

## Integrity

- raw candidates: 1,466,046 rows, SHA
  `13e4a92178cb8b8319050a556ead9e62314f18e29a3c9385ddeec83c8ffb9782`;
- mask-valid candidates: 1,449,011 rows;
- NMS candidates: 187,077 rows, SHA
  `00a2cd9ce1d3d5007b0a25ff4c448497772e9b9944c7d46129213d63b34eb8cd`;
- GQ-CNN scores: 187,077/187,077 finite, SHA
  `7094c770fee08208ca18ffc629e81b2b678d4d8ea3d4c785d46b7f8dff948a50`;
- duplicate candidate keys: zero;
- NMS/score key sets: exactly equal;
- NMS/score geometry: exactly equal;
- candidate or scoring execution failures: zero.

The retained baseline materializes Top-1 2,934/7,675, Top-5 4,655/7,675,
Top-10 5,201/7,675, Oracle 5,799/7,675 and MRR 0.483547657. These are audit
anchors, not copied final results; the new run independently recomputes them.

## GT-leakage boundary

Several retained files contain post-hoc ground-truth fields:

- `input_manifest.csv` contains GT mask/object/grasp metadata;
- mask metadata contains GT mask IoU;
- old bundles include a ground-truth mask;
- score/evaluation Parquets include candidate success, GT IoU and GT angle.

They must never be consumed wholesale by feature extraction, reranker
inference, VLM inputs, or a safe-switch gate. The new run creates a physically
separate deployment projection containing only identity, predicted-mask/RGB-D
references, candidate geometry and q-values. GT is joined only after inference
features are frozen and only for training/evaluation.

## Storage decision

The old pipeline already duplicates RGB, depth and candidate arrays per query.
The new run will not copy that layout. It will:

1. reference test predictions and scene assets read-only;
2. store each train/val scene reference once;
3. use lossless compressed probabilities;
4. use ZSTD Parquet for candidate/feature tables;
5. generate VLM images on demand;
6. write temporary files only under the new run's `tmp/`.

Machine-readable inventory:

`runs/modular_reranking_repeatedfilm_v1_20260729_203147/manifests/reuse_inventory.json`
