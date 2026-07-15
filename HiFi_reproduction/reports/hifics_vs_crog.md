# Indicative comparison — test splits differ

This comparison is descriptive, not a fair model ranking. The completed HiFi-CS run covers 7,675 OCID-VLG `unique/test` expressions. The completed CROG aggregate covers 17,749 `multiple/test` expressions.

## Sample relationship

- Pairing key: `question_index` (not HiFi's local `num` field)
- Common samples: 7,675
- HiFi-only samples: 0
- CROG-only samples: 10,074
- The common records match on image, query, answer, target, box, and grasps.

## Shared visual-grounding metrics

| Metric | CROG full 17,749 | CROG paired subset 7,675 | HiFi 7,675 |
|---|---:|---:|---:|
| IoU | 79.02 | 79.22 | 76.20 |
| P@50 | 95.51 | 95.49 | 86.57 |
| P@60 | 93.26 | 93.52 | 82.53 |
| P@70 | 85.51 | 86.55 | 76.60 |
| P@80 | 63.56 | 65.21 | 64.44 |
| P@90 | 16.53 | 15.77 | 34.78 |

J@1 and J@Any are excluded because they are not the same HiFi visual-grounding metrics.

## Why the aggregate is not fair

The paired CROG subset fixes the sample-population difference only. It does not align metric postprocessing:

- CROG uses a 416 letterbox, sigmoid prediction inverse-warped cubically to 640×480, then strict probability `> 0.35`.
- HiFi uses a direct 352×352 stretch, canonical foreground probability `>= 0.15`, and evaluates at model resolution.
- CROG treats empty/empty IoU as 0 through its smoothed denominator; HiFi defines it as 1.
- Both implementations use strict `IoU > X` for P@X, but their mask construction and threshold inclusivity differ.

Therefore sample-set comparability is **yes only for the paired 7,675 subset**, metric-protocol comparability is **no**, and neither the full-aggregate nor paired numeric difference is a fair causal comparison of model quality. A fair result requires rescoring both checkpoints on the common samples through one shared resize, threshold, foreground, and empty-mask implementation. Neither model was retrained for this note.

## CROG provenance

- Per-sample predictions: `../crog_reproduction/CROG/failure_analysis/predictions/test_predictions.jsonl` (`c8bfbd6f75528212aa03e3ef6cfd561767fe2589bb264feda60610e593b1a25c`)
- Prediction metadata: `../crog_reproduction/CROG/failure_analysis/predictions/test_predictions.meta.json` (`25f22359802b1b6c7bf148c3aad9c548fc5e24eaf845e9ed5fcfdb4725270b87`)
- Checkpoint: `../crog_reproduction/CROG/exp/OCID-VLG_multiple_mac/CROG_mac_mps_official_params_50epoch_bs8/best_jindex_model.pth` (`ac1304da520fbd3f6998dea88ba1e63b39b596b775856b39e5360a29576f1ddf`)
