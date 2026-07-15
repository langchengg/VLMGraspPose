# HiFi-CS versus CROG split comparison

Stable sample IDs in the four accompanying files are composite `image_filename<TAB>question_index` values. `question_index` alone is reused across images and is not a globally unique key.

| Protocol | Train expressions | Test expressions | Train images | Test images | Train/test sample overlap |
|---|---:|---:|---:|---:|---:|
| HiFi released preparation (`unique`) | 26,295 | 7,675 | 1,104 | 325 | 0 |
| Completed CROG run (`multiple`) | 63,221 | 17,749 | 1,201 | 344 | 0 |

## Cross-protocol overlap

- Exact HiFi-test / CROG-test composite sample overlap: **7,675**
- HiFi-test-only samples: **0**
- CROG-test-only samples: **10,074**
- Therefore the HiFi `unique/test` set is an exact sample subset of CROG `multiple/test`.
- HiFi test sequence paths: 111; CROG test sequence paths: 115.
- HiFi train/test sequence paths overlap even though image/sample IDs do not. This is the dataset's published split behavior and should not be described as unseen-scene evaluation.

## Comparison validity

The already completed aggregate CROG test metrics cover all 17,749 `multiple/test` expressions, while the primary released HiFi evaluation covers 7,675 `unique/test` expressions. Comparing those aggregate numbers is **indicative, not fair**, because the sample sets differ.

A fair comparison is feasible without retraining: evaluate both checkpoints on the exact 7,675-sample `unique/test` subset, with the same binary masks, resizing, empty-mask handling, foreground convention, and IoU thresholds. No test sample may be added to training. Until such paired evaluation is executed, CROG and HiFi results must remain labelled protocol-different.

The paper states a 70/30 OCID-VLG split, but neither its exact manifests nor checkpoint are published. The released HiFi preparation uses the dataset's `unique` train and test files from a 70/10/20 protocol while ignoring validation. Paper comparison is therefore approximate even when local code matches the repository.
