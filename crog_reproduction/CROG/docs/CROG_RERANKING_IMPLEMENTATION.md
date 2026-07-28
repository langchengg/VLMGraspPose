# CROG frozen-candidate post-hoc reranking

## Summary

本实现只研究一个问题：在 CROG 原始 Top-5 4-DoF 候选集合完全不变时，新的候选级排序能否选出更有效的 Top-1。CROG checkpoint、forward、peak threshold、`min_distance`、候选数、中心、角度、宽度和 fixed height 均未修改。

完整 17,749-sample test 证明 100-sample 的表面正向结果不能泛化。`rule_2d_equal` 从 83.2047% 降至 81.0186%（-2.186 pp，600 recovered / 988 harmful，McNemar `p=1.67e-22`，frame-bootstrap 95% CI `[-3.696,-0.636]` pp）。固定权重 `rule_fixed_v1` 为 82.9455%（-0.259 pp，620/666，`p=0.210`，CI 跨 0）。没有任何预声明 ranker 显示可支持的正向提升；所有负结果和 harmful flips 均保留。

## Start-state audit

- Initial branch: `mac-mps-single-device`
- Initial commit: `1eeee85de1fe6bffdc66c9ed9a622028ea04578e`
- Initial dirty worktree was preserved. Existing tracked changes included `.gitignore`, `engine/crog_engine.py`, `utils/grasp_eval.py`, and `utils/misc.py`; the Mac port and `failure_analysis/` were user-owned untracked work.
- No on-disk `AGENTS.md` exists under `VLMGraspPose`; the task-provided repository rules were applied.
- Runtime: `.venv/bin/python`, Python 3.12.12, PyTorch 2.12.1, scikit-image 0.26.0, OpenCV 4.11.0, NumPy 1.26.4, SciPy 1.17.1.
- MPS is built, available, and passed a real tensor execution probe. Conda base is a different PyTorch environment and was not used.
- Checkpoint: `best_jindex_model.pth`, 1,766,269,277 bytes, SHA256 `ac1304da520fbd3f6998dea88ba1e63b39b596b775856b39e5360a29576f1ddf`.
- Immutable old regression artifact: `failure_analysis/predictions/test_predictions.jsonl`, SHA256 `c8bfbd6f75528212aa03e3ef6cfd561767fe2589bb264feda60610e593b1a25c`.
- Streamed old full-test baseline: 17,749 samples, J@1 `14768/17749 = 83.2047%`, J@Any `16129/17749 = 90.8727%`, headroom `7.6680` pp.
- Modification-time maintained test suite: 30/30 passed. Full-repository pytest was already blocked during collection by `test_diff_refer_types.py` importing nonexistent `engine.engine`.

## Verified CROG data flow

Source of truth is `engine/crog_engine.py`, `utils/grasp_eval.py`, and `utils/dataset.py`.

1. The network returns logits for instance mask, quality and width, plus raw `sin(2θ)` and `cos(2θ)` maps.
2. Instance mask, quality and width each receive exactly one sigmoid. Sin/cos do not.
3. Maps are bicubically resized to the network input resolution with `align_corners=True`, then inverse-warped with OpenCV `INTER_CUBIC` to the original image size.
4. The predicted mask is a probability map until strict threshold `P_M > 0.35`, after which it is binary. The exporter retains aggregate probability features and optional binary RLE.
5. Candidate peaks are extracted from the inverse-warped, already-sigmoid quality map. The CROG inference path does not apply the Gaussian smoothing used in the separate SSG path.
6. Quality is not explicitly multiplied by the predicted mask before peak extraction.
7. `peak_local_max` coordinates are `(row,column)`; grasp output is `(x=column,y=row,width,height,angle)`.
8. `θ = 0.5 atan2(sin2θ,cos2θ)`. It is stored in radians and degrees, uses 180-degree parallel-jaw symmetry, and is passed as `-θ` to OpenCV because image y increases downwards.
9. Predicted width is the width-map value times 100 px; it is the long rectangle/opening axis. Fixed rectangle height remains 20 px. At `θ=0`, opening is horizontal; the image-space opening vector is `(cosθ,-sinθ)`.
10. OCID depth is uint16 millimetres on disk and is converted to metres by `/1000`. Depth zero is invalid.
11. PCD files are binary organized `640×480` records with fields `x y z rgba label`. Local checks showed `z` is metres, invalid z is NaN, RGB/PCD/depth are row-major pixel aligned, and valid z differs from `depth/1000` by less than 1 mm. Only copied XYZ can leave the parser; `label` is never an inference feature. Test PCD coverage is 344/344 unique frames; train has eight missing frames, handled by neutral fallback.

The official evaluator is intentionally unchanged. It clips GT width to `[0,100]`, replaces GT height with 20, uses a strict IoU `>0.25`, and applies the legacy CROG angle gate `|pred-gt|≤30 OR |pred+gt|≤30`. That gate is **not** the general 180-degree symmetric distance formula. On the immutable full prediction artifact, replacing only this gate by `abs(((pred-gt+90) mod 180)-90)<30` changes 127 J@1 states and 28 J@Any states; corrected counts are 14,641/17,749 (82.4892%) and 16,101/17,749 (90.7150%). Main metrics retain the legacy gate for exact 83.2047/90.8727 comparability; corrected-angle results are sensitivity analysis only. The evaluator's `skimage.draw.polygon` x/y use also has a confirmed right-side clipping defect for x≥480.

## Frozen candidate interface

`utils.grasp_eval.detect_grasp_candidates()` calls the unchanged:

```text
peak_local_max(Q, min_distance=2, threshold_abs=0.4, num_peaks=5)
```

No resampling, deduplication, padding, hard rejection, NMS change or candidate mutation is performed. Zero peaks yields `no_grasp`; fewer than five remain fewer than five. `detect_grasps()` delegates to the new interface and retains its original return values.

Each candidate contains `candidate_id`, `legacy_rank`, independent `q_rank`, `(row,col)`, `(cx,cy)`, angle in radians/degrees, width/height, OpenCV polygon, `q_raw`, legacy grasp tuple and SHA256 geometry checksum. `q_raw` is exactly `Q[row,col]` from the map passed to `peak_local_max`. Every ranker validates the declared checksum against actual geometry before and after sorting and asserts identical id/checksum/geometry mappings.

scikit-image 0.26.0 sorts peaks by descending intensity with a stable tie order. Official CROG used 0.20.0, where equal-valued tie order was not stable. Runtime version, `legacy_rank` and diagnostic `q_rank` are therefore all exported; legacy order is never overwritten.

## Physical schema and leakage boundary

Every inference run writes a new directory and refuses silent overwrite:

- `features.jsonl`: only inference-time information, predicted-mask RLE, frozen candidates, per-candidate aggregates, reliabilities and missing reasons.
- `labels.jsonl`: `sample_id`, candidate key/checksum, official-evaluator candidate validity, GT grasp count, and old-vs-recomputed regression flags.
- `predictions.jsonl`: backward-compatible combined view retaining every old field plus schema/candidate metadata. Rankers never consume it.
- `commit_journal.jsonl`: a sample is committed only after feature, label and combined records are flushed. Resume validates exact ordered prefixes, truncates only an uncommitted tail (including a final half-line), and rejects corruption inside the committed prefix.
- `metadata.json`: schema, source commit, dirty/untracked behavior-source hash, checkpoint path/SHA256, independent regression-artifact SHA256, dataset root/split/count, effective batch/workers, K/threshold/distance, mask threshold, output/config/calibration hashes, creation time, runtime versions and cache fingerprint.

Evaluation, rebuilt-label and MLP outputs have separate run manifests hashing every input and behavior parameter. `--resume` accepts only an exact fingerprint match. Test export automatically joins the immutable old prediction JSONL by `sample_id`; old J@1/J@Any are never taken from the new forward itself, and any mismatch is recorded then fails the command unless explicitly overridden. Calibration JSON must declare `provenance.source_split="train"`; tuned weights must declare `provenance.source_split="val"`.

`INFERENCE_FEATURE_ALLOWLIST` is the only MLP/rule feature gateway. Rankers cannot ingest an entire record. Candidate id, rank, sample/image id, filenames, absolute x/y, GT mask/grasp/object/instance, evaluator validity, IoU, errors, success flags, and PCD label are excluded. Tests add/delete/change GT-derived fields and verify candidate scores are bit-identical. Feature records are recursively rejected if a key contains `gt`, `label`, `success`, `error`, `iou`, `validity`, or `failure_category`.

## Feature definitions

Every potentially unavailable feature stores `(value,reliability,missing_reason)`. Rule scoring uses:

```text
E(f,r) = r*f + (1-r)*0.5
```

Thus unavailable geometry contributes the same neutral constant to all candidates.

### Quality

- `Q = clip(q_raw,0,1)`.
- `q_patch_mean`: mean in a radius-2 centre patch.
- `q_prominence = q_raw - median(Q in a 3..7 px ring)`. It is raw diagnostic/MLP input and is standardized only by train-fitted preprocessing.

### Predicted-mask consistency

- `CenterProb`: `[1,2,1;2,4,2;1,2,1]` weighted 3×3 probability mean.
- `SoftCoverage`: mean predicted probability over the visible grasp rectangle.
- `BinaryCoverage`: fraction of visible rectangle with `P_M>0.35`.
- `ImageSupport`: visible discrete rectangle pixels divided by the same rectangle rasterized without image clipping; it scales coverage reliability.
- `SignedDistance = EDT(binary)-EDT(not-binary)` and `CenterMargin = clip(0.5 + distance/(2*max(2,0.25h)),0,1)`.
- `MaskConsistency = 0.30E(CenterProb)+0.45E(SoftCoverage)+0.25E(CenterMargin)`.

### Mask-span width compatibility

Five narrow scanlines at perpendicular offsets `[-2,-1,0,1,2]` follow the verified opening axis. Holes of at most 2 px are filled. A scanline is valid only if it contains the centre and finds both object boundaries before the image boundary. Empty mask, centre outside all valid spans, width below 3 px, boundary truncation or shape mismatch marks width missing.

```text
w_object = d_minus + d_plus
WidthRatio = w_pred/(w_object+eps)
WidthSymmetry = 1-|d_minus-d_plus|/(d_minus+d_plus+eps)
WidthCompatibility = min(w_pred,w_object)/(max(w_pred,w_object)+eps)
```

The optional calibrated Gaussian in log-ratio space is produced only from train-positive candidates and saved with the MLP artifact. It is named mask-span width compatibility, not hardware aperture feasibility.

### Angle-map consistency

Inside the existing grasp rectangle, each pixel contributes the normalized vector `(cos2θ,sin2θ)` with weight `P_M*Q`. `AngleConcentration` is the norm of the weighted mean, `CenterAlignment=(1+dot(candidate_vector,normalized_mean))/2`, and `AngleConsistency=0.5*(concentration+alignment)`. Too few weighted pixels produces reliability zero.

### Depth geometry

Centre depth is a median over valid depth in the centre 5×5/predicted-mask intersection. `DepthMAD=1.4826*median(|z-median(z)|)`. Left/right inner contact bands produce median contact depths and absolute difference. The normalized thresholds are available only when fitted from train-positive candidates:

```text
tau_variance = max(P90(train-positive DepthMAD), 0.005 m)
tau_balance = max(P90(train-positive contact difference), 0.005 m)
DepthGeometry = 0.5E(exp(-DepthMAD/tau_variance)) +
                0.5E(exp(-difference/tau_balance))
```

Without that train calibration, the normalized depth score has reliability zero; raw depth diagnostics remain available to the MLP.

### Relative 2.5D obstacle proxy

The target reference depth is the median valid depth in `R∩predicted_mask`. The mask is dilated by 2 px. An obstacle is valid depth outside that dilation with depth no farther than `z_ref+10 mm`. Left/right jaw bands use thickness ratio 0.15. Collision is obstacle pixels over valid jaw-band pixels; clearance is nearest obstacle pixel distance divided by `h/2` and clipped to `[0,1]`.

```text
Safety2p5D = 0.70*(1-Collision2p5D) + 0.30*Clearance2p5D
```

The repository does not define gripper dimensions, approach frame/direction, swept volume, maximum opening or safe clearance. Metric 3D collision is therefore explicitly unavailable (`reliability=0`); no physical parameters are invented.

## Rankers and MLP

- `legacy`: unchanged CROG order.
- `q_only`: `Q`; must reproduce `q_rank`.
- `rule_2d_equal`: equal groups `Q`, mask consistency and uncalibrated mask-span width compatibility.
- `rule_fixed_v1`: weights `0.45 Q + 0.25 Mask + 0.10 Width + 0.05 Angle + 0.05 Depth + 0.10 Safety`.
- Ablations: Q only; Q+Mask; Q+Mask+Width; Q+Mask+Width+Angle; Q+Mask+Width+Depth; full fixed rule.
- `rule_val_tuned`: accepts only an explicit non-negative validation-derived weight JSON; no test tuning is implemented.
- MLP: `Linear(d,32)-ReLU-Dropout(0.1)-Linear(32,16)-ReLU-Linear(16,1)`.

Tie-break is score descending, actual `q_raw` descending, legacy rank ascending, candidate id ascending. The MLP uses the requested positive-listwise loss plus `0.2` class-balanced BCE. Frame-group split, imputer, scaler, calibration and early stopping are train-only; official test records are rejected by training code. Samples with no positive/all positive candidates are counted. The current checkpoint itself was trained on the full CROG train split, so MLP prediction stacking is in-sample; out-of-fold CROG predictions would be required for a strong learned-ranker claim.

## Validation results

### Real-data smoke and integration

The first real sample's binary rectangle coverages were `0.354, 0.318, 0.169, 0, 0.013`, matching the requested approximate coordinate/RLE pattern. All first-sample q values came from their exact peak coordinates. Historical-vs-current MPS forward rectangles can differ around `1e-5` px/deg because of cross-run floating-point drift; within one forward, legacy output and structured candidates are exactly equal.

10-test smoke: legacy/q-only/fixed rule all J@1 80%, Oracle@5 80%; angle equal-group ablation fell to 60% with two harmful flips.

100-test integration:

| Ranker | J@1 | Δ pp | Oracle@5 | Changed | Recovered | Harmful | McNemar p | Frame bootstrap Δ 95% CI |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| legacy | 75% | 0 | 83% | 0 | 0 | 0 | 1.000 | [0.0,0.0] pp |
| q_only | 75% | 0 | 83% | 0 | 0 | 0 | 1.000 | [0.0,0.0] pp |
| Q+Mask | 73% | -2 | 83% | 52 | 7 | 9 | 0.804 | [-8.0,0.0] pp |
| rule_2d_equal | 78% | +3 | 83% | 71 | 8 | 5 | 0.581 | [-8.0,+6.7] pp |
| Q+Mask+Width+Angle | 59% | -16 | 83% | 69 | 8 | 24 | 0.007 | [-20.0,-14.7] pp |
| Q+Mask+Width+Depth | 78% | +3 | 83% | 71 | 8 | 5 | 0.581 | [-8.0,+6.7] pp |
| rule_fixed_v1 | 75% | 0 | 83% | 71 | 4 | 4 | 1.000 | [-8.0,+2.7] pp |
| MLP smoke (200 train expressions/3 frames) | 70% | -5 | 83% | 59 | 7 | 12 | 0.359 | [-8.0,-4.0] pp |

The depth ablation equals the 2D rule because no train-derived normalized depth calibration was injected into that test cache. MLP is an engineering smoke only, not a statistically defensible model result.

Complete 17,749-test evaluation (all share Oracle@5 `90.8727%`):

| Ranker | J@1 | Δ pp | Changed | Recovered | Harmful | McNemar p | Frame bootstrap Δ 95% CI |
|---|---:|---:|---:|---:|---:|---:|---:|
| legacy | 83.2047% | 0 | 0 | 0 | 0 | 1.000 | [0,0] pp |
| q_only | 83.2047% | 0 | 0 | 0 | 0 | 1.000 | [0,0] pp |
| Q+Mask | 83.2723% | +0.0676 | 9,949 | 552 | 540 | 0.739 | [-1.093,+1.307] pp |
| rule_2d_equal | 81.0186% | -2.1860 | 11,527 | 600 | 988 | 1.67e-22 | [-3.696,-0.636] pp |
| Q+Mask+Width+Angle | 76.3311% | -6.8736 | 12,699 | 629 | 1,849 | 2.88e-138 | [-8.600,-5.290] pp |
| Q+Mask+Width+Depth | 81.0186% | -2.1860 | 11,527 | 600 | 988 | 1.67e-22 | [-3.696,-0.636] pp |
| rule_fixed_v1 | 82.9455% | -0.2592 | 10,821 | 620 | 666 | 0.210 | [-1.466,+0.996] pp |
| MLP engineering smoke | 83.0357% | -0.1690 | 9,278 | 455 | 485 | 0.344 | [-1.128,+0.894] pp |

The full cache contains 17,749 unique ordered sample ids and 88,745 candidates. Every sample has exactly five candidates; geometry checksum, exact `q_raw` feature, q-rank, journal alignment and independent old/new J@1/J@Any checks all have zero mismatches. Export wall time was 1:09:04 on MPS with batch size 16.

For `rule_fixed_v1`, recovered ids were `40,67,87,92`; harmful ids were `14,21,59,61`. For `rule_2d_equal`, recovered ids were `40,63,67,86,87,92,97,99`; harmful ids were `14,21,32,59,61`. Every sample, candidate feature, score decomposition, validity under an `evaluation` namespace, missing reason, and RGB/mask visualization index is in the corresponding `per_sample.jsonl`; `case_index.json` lists all sample ids by outcome/diagnostic category.

## Commands and artifacts

See `failure_analysis/reranking/README.md` for complete commands. Primary verified runs:

```bash
.venv/bin/python -m failure_analysis.reranking.cli build-features \
  --split test --limit 10 --device mps --batch-size 2 --workers 0 \
  --output failure_analysis/reranking_outputs/smoke_test_10 --overwrite

.venv/bin/python -m failure_analysis.reranking.cli build-features \
  --split test --limit 100 --device mps --batch-size 8 --workers 0 \
  --output failure_analysis/reranking_outputs/integration_test_100 --overwrite

.venv/bin/python -m failure_analysis.reranking.cli evaluate \
  --features failure_analysis/reranking_outputs/integration_test_100/features.jsonl \
  --labels failure_analysis/reranking_outputs/integration_test_100/labels.jsonl \
  --ranker rule_fixed_v1 \
  --output failure_analysis/reranking_outputs/integration_test_100/eval_rule_fixed_v1 \
  --overwrite

.venv/bin/python -m failure_analysis.reranking.cli build-features \
  --split test --device mps --batch-size 16 --workers 0 \
  --output failure_analysis/reranking_outputs/full_test_17749_v1
```

Generated caches are gitignored. No old prediction, checkpoint, dataset or report was overwritten.

## External resources and reuse decision

- CROG official MIT repository at commit `1eeee85`: https://github.com/HilbertXu/CROG
- CROG paper: https://proceedings.mlr.press/v229/tziafas23a.html
- scikit-image `peak_local_max` documentation/source: https://scikit-image.org/docs/stable/api/skimage.feature.html#skimage.feature.peak_local_max and https://github.com/scikit-image/scikit-image/blob/v0.20.0/skimage/feature/peak.py
- Jacquard paper/evaluation definition: https://arxiv.org/abs/1803.11469
- OCID official organization/units: https://www.acin.tuwien.ac.at/vision-for-robotics/software-tools/object-clutter-indoor-dataset/
- OCID-VLG reference repository: https://github.com/gtziafas/OCID-VLG

The local MIT CROG implementation, evaluator, OpenCV geometry, dataset loader and custom RLE were reused directly. No external source code, model or new dependency was copied. OCID-VLG's repository/data license is unclear, so it was used only as a layout/API reference.

## Remaining risks and incomplete validation

- Official evaluator right-side x/y clipping is retained for comparability and can under-score candidates with x≥480.
- The official legacy angle gate is not a general 180-degree symmetric error. Corrected-angle sensitivity changes 127 old-full J@1 states and 28 J@Any states; main labels intentionally remain legacy-compatible.
- The commit journal protects normal process interruption. Records are flushed but not `fsync`ed per sample, so a sudden power loss or OS crash is outside the durability guarantee.
- scikit-image tie behavior differs between official 0.20 and local 0.26; version/provenance is recorded and legacy order is preserved.
- Fixed heuristic weights are predeclared baselines, not optimized weights. The 100-sample result is too small and statistically inconclusive.
- Relative 2.5D safety is a local proxy, not a real collision guarantee. Metric 3D remains disabled until all physical gripper and approach parameters are known.
- Learned-ranker conclusions require much larger frame-grouped data and ideally out-of-fold CROG predictions.
