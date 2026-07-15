# Official HiFi-CS repository reproduction on macOS/MPS

## Current status

Training and the formal 7,675-sample test evaluation are complete. The run reached 20,000 optimizer updates normally, and the final checkpoint was evaluated on Apple MPS in FP32 with CPU fallback disabled. All 7,675 original-resolution prediction bundles, the qualitative/failure review, all 7,675 predicted-mask AnyGrasp input bundles, the 20-sample geometry-verified subset, and the checksummed Kaggle archive are also complete. No AnyGrasp inference or full-DoF grasp-candidate generation was performed.

## Fixed protocol

- Upstream commit: `4be6b3be7ce79fae481fb51616adfa2b803f07a0`
- Hardware: Apple M5 Pro, 24 GB unified memory
- Python/PyTorch: 3.11.15 / 2.12.1
- Device: MPS float32; CPU fallback disabled
- Dataset: OCID-VLG `unique`, 26,295 train and 7,675 test
- Model: released `CLIPDensePredT`, CLIP ViT-B/16, blocks `[1,3,5,7,9]`, reduce dimension 64
- Parameters: 151,732,162 total; 2,111,425 `requires_grad=True`; CLIP frozen
- Batch: physical 16, accumulation 1, effective 16
- Optimizer/scheduler/loss: released AdamW / cosine / BCE-with-logits
- Target: 20,000 optimizer updates

## Scientific limitations

This is not an exact paper reproduction. The paper says hierarchical FiLM, Adam, roughly 6M trainable parameters, and a 70/30 split; the released default uses single-block FiLM (`extended_film=false`), AdamW, 2.11M trainable parameters, and published `unique` train/test manifests from a 70/10/20 protocol with validation omitted. The released evaluator also thresholds raw logits despite computing sigmoid; local evaluation corrects that defect and records both configured background and canonical foreground thresholds.

## Formal test result

| Metric | Result |
|---|---:|
| Samples | 7,675 |
| mean IoU | 76.20 |
| median IoU | 86.29 |
| IoU standard deviation | 24.28 |
| minimum / maximum IoU | 0.00 / 97.06 |
| P@50 | 86.57 |
| P@60 | 82.53 |
| P@70 | 76.60 |
| P@80 | 64.44 |
| P@90 | 34.78 |

- Complete evaluation-loop runtime: 221.84 seconds
- Synchronized model-inference total: 195.23 seconds
- Mean / median batch-normalized inference time per sample: 0.02544 / 0.02541 seconds
- Checkpoint SHA-256: `436a54ecc159a36664f55f762463c54fc9b082f44205cee8020bed59fb5280d0`
- Test manifest SHA-256: `915e002bf31f044419db7140bc1145b8fcc45f9a6b35259637d923c6d4610409`
- Prediction threshold: canonical foreground probability `>= 0.15`
- P@ convention: strict `IoU > X`

## Paper comparison

The paper reports IoU 88.26 and P@50/60/70/80/90 of 92.68/92.13/91.53/89.69/83.21. Signed local-minus-paper differences are -12.06/-6.11/-9.60/-14.93/-25.25/-48.43 percentage points. These are approximate reference deltas, not exact-reproduction deltas, because the exact paper checkpoint, manifests, threshold behavior, and full recipe are unpublished and the paper/repository settings differ materially.

The float32 P@ numerators over 7,675 samples are 6,644 / 6,334 / 5,879 / 4,946 / 2,669. These raw counts are the reproducibility contract for P@50 through P@90; see `final_metric_validation.md` for the exact-boundary note.

Primary sources checked: the [HiFi-CS paper](https://arxiv.org/pdf/2409.10419v2), the [commit-pinned released configuration](https://github.com/vineet2104/hifics/blob/4be6b3be7ce79fae481fb51616adfa2b803f07a0/experiments/config.yaml), and the [official OCID-VLG split documentation](https://github.com/gtziafas/OCID-VLG#versions). No external code was copied.

## Downstream export status

- Prediction export: 7,675/7,675 complete; original-resolution float32 probabilities and uint8 binary masks are present.
- AnyGrasp input export: 7,675 ready, 0 blocked; predicted masks only; no oracle directory was created.
- Intrinsics/depth: effective pinhole fits and a 1,000 units-per-metre depth scale were verified against organized PCD correspondences. These are not factory calibration values.
- Verified subset: 20/20 pass bundle and geometric validation; human visual review passes 13 and marks 7 low/over-segmented predictions unsafe for target-specific grasping.
- Kaggle archive: `artifacts/hifi_anygrasp_inputs_hifics_ocidvlg_20260711_112921.tar.gz`, SHA-256 `118d868daddf0dba7e1de199f115099630c80bfeff1d855a11826365d7a2d3f8`.
