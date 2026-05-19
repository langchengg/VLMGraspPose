# Architecture Audit Report

Audit date: 2026-05-17  
Repository root: `/Users/delaynomore/Downloads/VLMGraspPose/target_aware_graspnet`

## 1. Executive Summary

Overall status: **PARTIAL**

The project is more than a scaffold. It contains an executable CPU/Mac-compatible pipeline for OCID-VLG and an auxiliary OCID-Grasp fallback path. A real OCID-VLG sample was run end-to-end and produced target masks, target point clouds, grasp candidates, ranked grasps, Top-1 output, visualizations, and evaluation reports.

However, I do **not** classify the architecture as fully complete because several components are prototype-level:

- OCID-VLG is the primary dataset path and works on real data.
- OCID-Grasp fallback works on a smoke sample with generated class commands.
- The geometric sampler is implemented, but some configured sample counts are not used and side/normal candidates are approximate.
- Feature association is connected and non-constant, but target overlap, collision, and 2D grasp evaluation are lightweight proxies.
- Visualization is headless and saves images, but RGB visualization does not draw Top-K grasp candidates.
- Evaluation supports proxy and OCID 2D center-hit metrics, but not full grasp rectangle IoU/angle matching.
- The generic CLI names requested in the audit prompt are not present; the project uses dataset-specific OCID scripts instead.

Final recommendation: **ready for small prototype experiments and debugging on OCID-VLG, but needs fixes before claiming full experimental completeness or paper-grade evaluation.**

## 2. Repository Structure Check

Actual structure:

```text
target_aware_graspnet/
├── README.md
├── requirements.txt
├── configs/
├── src/
│   ├── dataset/
│   ├── target/
│   ├── pointcloud/
│   ├── grasp_sampler/
│   ├── association/
│   ├── scoring/
│   ├── evaluation/
│   ├── visualization/
│   └── utils/
├── scripts/
├── tests/
└── outputs/
```

Status: **COMPLETE for structure**

The folders contain meaningful Python files. No empty Python implementation files were found under `src/`. A few files are intentionally thin re-exports, for example `src/grasp_sampler/grasp_candidate.py`.

## 3. Dataset Support Status

| Area | Status | Evidence | Notes |
|---|---|---|---|
| OCID-VLG loader | COMPLETE | `src/dataset/ocid_vlg_loader.py::OCIDVLGIndexBuilder`, `OCIDVLGLoader` | Reads expressions JSON, RGB, depth, target label, target index, bbox, instance mask, grasp rectangles, fallback intrinsics. |
| OCID-VLG processing unit | COMPLETE | `OCIDVLGSample` in `src/utils/data_types.py` | One sample is `(image_id, sentence, target_label, target_bbox, target_mask)`, not image-only. |
| OCID-VLG language command | COMPLETE | `OCIDVLGIndexBuilder._sample_from_row` | Uses `question` / `sentence` directly as `command`; no object-id pseudo-command when language is present. |
| OCID-Grasp fallback | PARTIAL | `OCIDGraspIndexBuilder` | Builds class-generated commands and reads boxes/grasps. It has a smoke-tested runner but no rich split management. |
| OCID-Grasp disambiguation | COMPLETE for implemented fallback | `OCIDGraspIndexBuilder._commands_for_rows` | Generates left/right/center commands when duplicate class names occur in one image. |
| GraspNet as primary | COMPLETE avoidance | `configs/ocid_vlg.yaml`, README, OCID scripts | OCID-VLG is configured primary. GraspNet code remains as legacy, not the primary language-grasping path. |
| Object-language mapping | PARTIAL | `src/target/object_language_mapping.py` | Complete for GraspNet-style label maps; OCID-VLG does not need it because it uses real sentences. |
| Multi-object per image | COMPLETE for OCID-VLG rows | `run_ocid_split.py` smoke processed same image with two language-target samples | `summary.csv` shows two rows for the same RGB-D frame with different target ids/commands. |

## 4. Pipeline Status Table

| Pipeline Step | File | Class/Function | Input | Output | Status | Notes |
|---|---|---|---|---|---|---|
| Load OCID-VLG index | `src/dataset/ocid_vlg_loader.py` | `OCIDVLGIndexBuilder.build` | `refer_split`, `split`, dataset root | list of `OCIDVLGSample` | COMPLETE | Reads real OCID-VLG expressions JSON. |
| Load OCID-Grasp fallback | `src/dataset/ocid_vlg_loader.py` | `OCIDGraspIndexBuilder.build` | dataset root | list of generated language-target samples | PARTIAL | Works, but no split/subset taxonomy beyond `max_samples`. |
| Load sample tensors | `src/dataset/ocid_vlg_loader.py` | `OCIDVLGLoader.load_sample` | `OCIDVLGSample` | RGB, depth in meters, intrinsics, `TargetRegion`, grasp rectangles | COMPLETE | Smoke-tested. |
| Target selection / grounding supervision | `src/dataset/ocid_vlg_loader.py` | `_load_target_mask`, `TargetRegion` construction | bbox, instance index, instance mask | target label, bbox, mask, command | COMPLETE | Uses dataset target region, not Florence/GraspNet names. |
| RGB-D to scene point cloud | `src/pointcloud/rgbd_to_pointcloud.py` | `rgbd_to_pointcloud` | RGB, depth, K | Open3D scene point cloud | COMPLETE | CPU Open3D implementation. |
| Target point cloud extraction | `src/pointcloud/target_extraction.py` | `extract_target_pointcloud_from_mask`, `crop_pointcloud_by_bbox` | RGB, depth, mask/bbox, K | target Open3D point cloud | COMPLETE | Mask path preferred; bbox fallback exists. |
| Point cloud preprocessing | `src/pointcloud/processor.py`, `preprocessing.py` | `PointCloudProcessor.process`, `preprocess_target_pcd` | scene/target point clouds | clean target pcd | PARTIAL | Voxel/statistical outlier removal exists; radius outlier function exists but is not called by default. |
| Table plane segmentation | `src/pointcloud/plane_segmentation.py` | `segment_table_plane` | scene pcd | plane equation, inliers | PARTIAL | RANSAC plane is computed and used for scoring/sampler clearance, but plane points are not removed from target pcd. |
| Normal estimation | `src/pointcloud/normal_estimation.py` | `estimate_normals`, `orient_normals_towards_camera` | clean target pcd | pcd normals | COMPLETE | Connected before sampling. |
| AABB / OBB estimation | `src/pointcloud/bbox_estimation.py` | `compute_aabb`, `compute_obb` | clean target pcd | AABB/OBB | COMPLETE | Works on smoke sample. Edge cases for near-degenerate clouds are not specially handled. |
| Top-down sampler | `src/grasp_sampler/top_down_sampler.py` | `sample_top_down_grasps` | `PointCloudRepresentation` | `GraspCandidate` list | COMPLETE | Produces normalized top-down candidates in smoke output. |
| BBox-aligned sampler | `src/grasp_sampler/bbox_aligned_sampler.py` | `sample_bbox_aligned_grasps` | target OBB | OBB-axis-aligned candidates | COMPLETE | Connected and smoke output includes this type. |
| Side sampler | `src/grasp_sampler/side_grasp_sampler.py` | `sample_side_grasps` | target OBB, table plane | side candidates | PARTIAL | Implemented, but only does simple table clearance and does not fully check blocked approach paths. |
| Normal-based sampler | `src/grasp_sampler/normal_based_sampler.py` | `sample_normal_based_grasps` | target pcd normals | normal-pair candidates | PARTIAL | Implemented approximate normal-pair logic; not a full analytical antipodal planner. |
| Candidate validation | `src/grasp_sampler/geometric_sampler.py` | `_valid` | candidate | bool | PARTIAL | Checks finite position/orientation and width range; does not explicitly check quaternion norm, approach norm, or score range. Smoke candidates passed these checks externally. |
| Feature extraction | `src/association/feature_extractor.py` and helpers | `CandidateFeatureExtractor.extract_one` | candidates, target, pcd, depth, K | `CandidateFeatureVector` | PARTIAL | Features are computed from data, but target overlap/collision are lightweight approximations. |
| Rule-based re-ranker | `src/scoring/rule_based_scorer.py` | `RuleBasedScorer.score`, `top_k` | candidates + features | sorted `ScoredGrasp` list | COMPLETE | Formula matches requirement and weights are configurable. |
| Save per-sample outputs | `src/main.py` | `_save_outputs`, `_best_grasp_json` | `FrameResult`, data, pcr | JSON, PNG, PLY outputs | COMPLETE | Verified all expected files for OCID-VLG smoke sample. |
| OCID-VLG single runner | `scripts/run_ocid_one.py` | `main` | CLI args | one processed sample | COMPLETE | Smoke-tested successfully. |
| OCID-VLG batch runner | `scripts/run_ocid_split.py` | `main` | CLI args | multiple processed samples + summary | COMPLETE | Smoke-tested with 2 language-target samples. |
| OCID-Grasp runner | `scripts/run_ocid_grasp.py` | `main` | CLI args | fallback processed samples | PARTIAL | Smoke-tested one sample, but less configurable than OCID-VLG runner. |
| Proxy evaluation | `src/evaluation/evaluator.py` | `evaluate_records(mode="proxy")` | best_grasp JSONs | metric rows | COMPLETE for proxy | Uses geometric thresholds. |
| OCID 2D evaluation | `src/evaluation/evaluator.py`, `metrics.py` | `mode="ocid_2d"` | projected grasp centers, GT rectangles | center-hit rates | PARTIAL | Checks projected grasp center inside GT rectangle; no rectangle IoU/angle metric. |
| Report generation | `src/evaluation/report_generator.py` | `generate_reports` | output root | metrics/runtime/failure CSVs | PARTIAL | Works after small fix; grouping for OCID is by output directory levels, not true dataset split/scene taxonomy. |
| RGB visualization | `src/visualization/visualize_rgb.py` | `save_rgb_overlay` | RGB, target, best grasp | `visualization_rgb.png` | PARTIAL | Saves command/mask/bbox; does not draw Top-K or projected grasp rectangles. |
| 3D visualization | `src/visualization/visualize_pointcloud.py` | `save_pointcloud_figure` | pcd + grasps | `visualization_3d.png` | COMPLETE for headless prototype | Uses Matplotlib Agg, no GUI required. |
| Paper figure export | `scripts/make_paper_figures.py` | `main` | output root | paper_figures files | PARTIAL | Works after small fix; mostly copies existing figures and adds score bar, not a full multi-panel qualitative figure generator. |

## 5. Core Data Structures Check

Status: **COMPLETE with minor type-hint issue**

Implemented in `src/utils/data_types.py`:

- `GraspNetSample`
- `OCIDVLGSample`
- `TargetRegion`
- `PointCloudRepresentation`
- `GraspCandidate`
- `CandidateFeatureVector`
- `ScoredGrasp`
- `FrameResult`

JSON serialization:

- `_to_jsonable` converts NumPy arrays, NumPy scalars, `Path`, lists, tuples, dicts, and nested `to_json()` objects.
- Open3D objects are not serialized directly; `PointCloudRepresentation.to_json()` emits metadata and point counts instead.
- Smoke output JSON files were successfully written and re-read.

Minor issue:

- `FrameResult.sample` is type-hinted as `GraspNetSample`, but OCID uses `OCIDVLGSample` at runtime. This works at runtime but should be changed to a union or protocol for type clarity.

## 6. Point Cloud Module Check

Status: **PARTIAL**

What works:

- RGB loading: `dataset/camera_loader.py::load_rgb`
- Depth loading: `load_depth` converts raw PNG depth to meters via `depth_raw / depth_scale`
- OCID-VLG config uses `depth_scale: 1000.0`
- Intrinsics: fallback matrix from `configs/ocid_vlg.yaml`
- RGB-D to Open3D point cloud: `pointcloud/rgbd_to_pointcloud.py::rgbd_to_pointcloud`
- Mask-based target point cloud extraction: `target_extraction.py::extract_target_pointcloud_from_mask`
- BBox fallback: `crop_pointcloud_by_bbox`
- Target PLY saving: `rgbd_to_pointcloud.py::save_pointcloud`
- Voxel downsampling and statistical outlier removal are connected.
- Plane segmentation, normal estimation, AABB and OBB computation are connected.

Issues / assumptions:

- OCID uses fallback intrinsics, not per-frame intrinsics.
- RGB-depth-mask resolution mismatch is not handled; code assumes aligned same-size arrays.
- Radius outlier removal and plane removal functions exist but are not part of the default processor path.
- Table plane is segmented from scene pcd but not removed from target pcd.
- Coordinate convention is the Open3D camera frame; robot/world transforms are not implemented.

Smoke evidence:

- OCID-VLG target point cloud had **308** points in `target_pointcloud.ply`.
- OCID-Grasp fallback target point cloud had **4186** points.

## 7. Grasp Sampler Check

Status: **PARTIAL**

Implemented candidate types:

- Top-down: `sample_top_down_grasps`
- BBox-aligned: `sample_bbox_aligned_grasps`
- Side: `sample_side_grasps`
- Normal-based: `sample_normal_based_grasps`

Smoke evidence on OCID-VLG sample:

- Candidate count: **15**
- Ranked Top-K count: **5**
- Candidate types produced: `bbox_aligned`, `normal_based`, `top_down`
- Width range: `0.03924457409033789` to `0.1`
- Quaternion norm range: `0.9999999999999998` to `1.0`
- Approach vector norm range: `0.9999999999999999` to `1.0`
- Closing direction norm range: `0.9999999999999999` to `1.0`
- Ranked scores: `[0.7603507568758718, 0.7268602562056975, 0.6658764670417046, 0.6465141738023481, 0.6326191740124448]`

Issues:

- `top_down_samples`, `side_samples`, and `bbox_samples` config values are not actually used to control sample count.
- Candidate validation only checks finite arrays and width range.
- Side sampler has a simple table clearance check, not full approach-path collision checking.
- Normal-based sampler is approximate; it should be described as a prototype, not a full analytic grasp planner.

## 8. Candidate-Target Feature Association Check

Status: **PARTIAL**

Implemented features:

- `target_overlap`: projected grasp center plus local mask patch score
- `center_alignment`: exponential distance score
- `distance_to_target_center`: Euclidean 3D distance
- `gripper_width_match`: compares width with OBB dimension
- `approach_direction_score`: top-down or horizontal-side heuristic
- `depth_stability`: local depth variance score
- `collision_penalty`: table and nearby non-target scene point approximation
- `boundary_penalty`: distance transform over target mask
- `initial_geometric_score`: copied from candidate
- `grounding_score`: copied from target

This is connected and not hard-coded to constants, but the features are proxy heuristics. `target_overlap` is not a true gripper-region overlap; `collision_penalty` is not a full gripper mesh/path collision check.

## 9. Re-Ranking Module Check

Status: **COMPLETE**

Implemented in `src/scoring/rule_based_scorer.py`.

Formula used:

```text
final_score =
0.20 * initial_geometric_score
+ 0.25 * target_overlap
+ 0.15 * center_alignment
+ 0.10 * gripper_width_match
+ 0.10 * depth_stability
+ 0.10 * approach_direction_score
- 0.07 * collision_penalty
- 0.03 * boundary_penalty
```

Weights are configurable in `configs/scoring.yaml`. Scores are clamped to `[0, 1]`, sorted descending, and ranks are assigned.

Unit test evidence:

- `tests/test_rule_based_scorer.py` passed.

Smoke evidence:

- Top-K scores were finite, non-identical, and sorted descending.

## 10. Output Format Check

Status: **COMPLETE for smoke output, PARTIAL for target identity in path**

OCID-VLG smoke output generated all required per-sample files:

```text
target_mask.png
target_pointcloud.ply
grasp_candidates.json
ranked_grasps.json
best_grasp.json
score_breakdown.json
visualization_rgb.png
visualization_3d.png
```

`best_grasp.json` includes:

- split
- scene_id
- camera
- frame_id
- image_id
- target_id
- target_label
- command
- target_bbox
- best grasp position
- projected 2D grasp center
- orientation quaternion
- approach vector
- closing direction
- gripper width
- grasp type
- final score
- feature breakdown
- GT grasp rectangles
- Top-K fallback candidates
- runtime

Path issue:

- OCID-VLG output path is unique per language expression:
  `outputs/.../ocid_vlg/{refer_split}/{split}/{image_id}/`
- The `image_id` includes an expression index, so multiple targets from one image do not overwrite each other.
- However, the path does **not** explicitly include `target_003` or target label. This is acceptable for uniqueness but weaker for auditability.

## 11. Evaluation Module Check

Status: **PARTIAL**

Supported:

- Proxy evaluation:
  - Top-1 / Top-3 / Top-5 proxy valid rate
  - collision-free proxy rate
  - mean target overlap
  - mean center distance
  - mean final score
  - runtime per frame
- Annotation-style 3D matching skeleton:
  - Requires `annotation_valid_grasps` or `valid_annotated_grasps` inside `best_grasp.json`
  - Current OCID pipeline does not populate those fields.
- OCID 2D mode:
  - Projects 3D grasp center to image coordinates.
  - Reports whether center falls inside any GT grasp rectangle.

Missing / partial:

- No full 2D grasp rectangle IoU and angle threshold metric.
- No `metrics_by_dataset.csv`.
- `metrics_by_scene.csv` grouping is based on output directory levels; for OCID-VLG smoke it reports `scene_id=multiple`, not the true sequence path.
- Failure rate is computed from discovered `best_grasp.json` records, so it does not fully include failed samples in the denominator unless additional bookkeeping is added.

Small fix applied during audit:

- `src/evaluation/report_generator.py` now writes an empty `failure_cases.csv` with standard headers.

## 12. Visualization Check

Status: **PARTIAL**

Works:

- `visualization_rgb.png` saves RGB image with command text, target mask overlay, and bbox.
- `visualization_3d.png` saves target point cloud and grasp direction quivers using Matplotlib Agg.
- `make_paper_figures.py` now exports a score breakdown bar chart and copies figures.

Missing:

- RGB visualization does not draw projected Top-K grasp candidates or highlight Top-1 grasp rectangle.
- Paper figure script is not a true multi-panel qualitative figure generator.

Small fix applied during audit:

- `scripts/make_paper_figures.py` no longer crashes when `failure_cases.csv` is empty.
- The script imports `_common` before Matplotlib, so the local writable Matplotlib cache config is applied earlier.

## 13. CLI and Script Check

Status: **PARTIAL**

Actual working OCID commands:

```bash
python scripts/run_ocid_one.py \
  --dataset-root ../data/raw/OCID-VLG \
  --refer-split multiple \
  --split test \
  --index 0 \
  --output-root outputs/audit_ocid_vlg \
  --top-k 5 \
  --overwrite
```

```bash
python scripts/run_ocid_split.py \
  --dataset-root ../data/raw/OCID-VLG \
  --refer-split multiple \
  --split test \
  --max-samples 2 \
  --output-root outputs/audit_ocid_vlg_split \
  --top-k 3 \
  --overwrite
```

```bash
python scripts/run_ocid_grasp.py \
  --dataset-root ../data/raw/OCID-VLG \
  --max-samples 1 \
  --output-root outputs/audit_ocid_grasp \
  --top-k 5 \
  --overwrite
```

```bash
python scripts/evaluate_outputs.py \
  --output-root outputs/audit_ocid_vlg \
  --mode ocid_2d
```

```bash
python scripts/make_paper_figures.py \
  --output-root outputs/audit_ocid_vlg \
  --num-success 1 \
  --num-failure 1
```

Missing requested generic scripts:

- No `scripts/run_one_sample.py`
- No generic `scripts/run_dataset.py`

The actual interface is dataset-specific: `run_ocid_one.py`, `run_ocid_split.py`, and `run_ocid_grasp.py`.

## 14. Smoke Test Results

### Unit tests

Command:

```bash
python -m pytest tests
```

Result:

```text
11 passed in 3.19s
```

### Compile check

Command:

```bash
python -m compileall -q src scripts tests
```

Result: exit code `0`.

### OCID-VLG real sample smoke test

Command:

```bash
python scripts/run_ocid_one.py --dataset-root ../data/raw/OCID-VLG --refer-split multiple --split test --index 0 --output-root outputs/audit_ocid_vlg --top-k 5 --overwrite
```

Result:

```text
ARID10__floor__top__non-fruits__seq09__result_2018-08-27-16-13-28__000000: success
command: Grasp the flashlight
target: flashlight_1 index=2 bbox=[259, 381, 344, 408]
gt_grasps: 7
final_score: 0.7604
```

Verification:

- Missing expected files: `[]`
- Target point cloud points: `308`
- Grasp candidates: `15`
- Ranked grasps: `5`
- Final score finite: `True`
- Top-K fallback candidates: `4`
- `best_grasp.json` required keys present: `True`

### OCID-VLG batch smoke test

Command:

```bash
python scripts/run_ocid_split.py --dataset-root ../data/raw/OCID-VLG --refer-split multiple --split test --max-samples 2 --output-root outputs/audit_ocid_vlg_split --top-k 3 --overwrite
```

Result:

```text
processed_units: 2
```

The generated `summary.csv` has two rows for the same RGB-D frame with different targets/commands:

- target `2`, command `Grasp the flashlight`
- target `9`, command `The can food`

This confirms the processing unit is language-target sample, not just image frame.

### OCID-Grasp fallback smoke test

Command:

```bash
python scripts/run_ocid_grasp.py --dataset-root ../data/raw/OCID-VLG --max-samples 1 --output-root outputs/audit_ocid_grasp --top-k 5 --overwrite
```

Result:

- Processed units: `1`
- Command: `pick the cereal box`
- Target label: `cereal_box_3`
- Score: `0.8097557983947881`
- Target point cloud points: `4186`

### Evaluation smoke test

Command:

```bash
python scripts/evaluate_outputs.py --output-root outputs/audit_ocid_vlg --mode ocid_2d
```

Result files:

- `outputs/audit_ocid_vlg/metrics_by_split.csv`
- `outputs/audit_ocid_vlg/metrics_by_scene.csv`
- `outputs/audit_ocid_vlg/runtime_report.csv`
- `outputs/audit_ocid_vlg/failure_cases.csv`

Single-sample metric row:

- processed frames: `1`
- mean final score: `0.7603507568758718`
- Top-1 2D grasp-center hit rate: `1.0`
- mean runtime per frame: `0.19799020899517927`

### Paper figure export smoke test

Before fix:

- `scripts/make_paper_figures.py` failed with `pandas.errors.EmptyDataError: No columns to parse from file` when `failure_cases.csv` existed but was empty.
- It also emitted Matplotlib cache warnings because Matplotlib was imported before `_common` set `MPLCONFIGDIR`.

Fix applied:

- `scripts/make_paper_figures.py`: import `_common` before Matplotlib and tolerate empty failure CSVs.
- `src/evaluation/report_generator.py`: write `failure_cases.csv` with standard headers even when there are no failures.

After fix command:

```bash
python scripts/make_paper_figures.py --output-root outputs/audit_ocid_vlg --num-success 1 --num-failure 1
```

After fix result:

```text
paper_figures: outputs/audit_ocid_vlg/paper_figures
```

## 15. Mac Compatibility Check

Status: **SAFE for core pipeline**

`requirements.txt`:

```text
numpy
opencv-python
open3d
scipy
pyyaml
pandas
tqdm
matplotlib
pillow
```

Search command:

```bash
rg -n "cuda|\\.cuda\\(|torch\\.cuda|MinkowskiEngine|spconv|cupy|Isaac|pointnet2_ops|pytorch3d|GraspNet baseline|graspnet-baseline" target_aware_graspnet
```

Findings:

- Only README mentions CUDA-only libraries as intentionally avoided.
- No core imports or calls to CUDA, `torch.cuda`, MinkowskiEngine, spconv, Isaac Sim, cupy, pointnet2 ops, or CUDA-only PyTorch3D were found.

## 16. Missing or Broken Items

| File | Function | Issue | Fix Needed |
|---|---|---|---|
| `src/evaluation/evaluator.py` | `evaluate_records(mode="ocid_2d")` | Only center-inside-rectangle metric, not full 2D grasp rectangle IoU/angle metric. | Implement rectangle prediction projection and standard grasp rectangle matching. |
| `src/evaluation/split_evaluator.py` | `evaluate_by_split`, `evaluate_by_scene` | OCID output grouping does not map cleanly to true `split` and `scene_id`; no `metrics_by_dataset.csv`. | Group by fields inside `best_grasp.json`, not directory levels. |
| `src/visualization/visualize_rgb.py` | `save_rgb_overlay` | Does not render Top-K / Top-1 projected grasp candidates. | Draw projected centers or grasp rectangles from ranked grasps. |
| `src/grasp_sampler/geometric_sampler.py` | `_valid` | Candidate validation incomplete. | Add quaternion norm, vector norm, positive finite score checks. |
| `src/grasp_sampler/top_down_sampler.py` and related samplers | sampler functions | Configured sample count fields are mostly unused. | Respect `top_down_samples`, `bbox_samples`, `side_samples`. |
| `src/pointcloud/processor.py` | `process` | Radius outlier removal and plane removal are not connected. | Add optional config flags and call these functions. |
| `src/utils/data_types.py` | `FrameResult` | `sample` type hint excludes `OCIDVLGSample`. | Use `GraspNetSample | OCIDVLGSample` or a shared protocol. |
| `scripts/` | missing generic runners | No `run_one_sample.py` or `run_dataset.py`. | Add generic dispatch scripts if a unified CLI is required. |
| `outputs` path design | OCID output path | OCID path is unique but does not explicitly show `target_XXX`. | Add target id/label to directory name for easier inspection. |

## 17. Final Recommendation

The project is a **working Mac-compatible prototype** for OCID-VLG target-aware RGB-D grasping. It can process real OCID-VLG language-conditioned samples end-to-end and can process OCID-Grasp fallback samples with generated commands.

It is **not yet a fully complete research pipeline** in the strict sense because evaluation and visualization remain proxy-level, and some geometric modules are simplified. Before full experiments, prioritize:

1. Implement true OCID 2D grasp rectangle matching with angle/IoU thresholds.
2. Improve output grouping/reporting by dataset, refer split, split, and true sequence.
3. Draw Top-K grasp predictions in RGB visualizations.
4. Tighten candidate validation and connect optional point cloud preprocessing flags.
5. Add generic `run_dataset.py` / `run_one_sample.py` CLI wrappers if the public interface should be dataset-agnostic.

## 18. Post-Audit Fixes Applied

After the audit, the following requested gaps were implemented and verified:

- Standard OCID 2D grasp rectangle evaluation:
  - Added predicted `best_grasp_rectangle_2d` and `top_k_grasp_rectangles_2d` to `best_grasp.json`.
  - Added rotated rectangle IoU and 180-degree symmetric angle matching.
  - Added `top1_2d_rectangle_match_rate`, `top3_2d_rectangle_match_rate`, and `top5_2d_rectangle_match_rate`.
- Top-K RGB visualization:
  - `visualization_rgb.png` now draws fallback Top-K grasp rectangles and highlights Top-1.
- True dataset/split/scene metrics grouping:
  - Reports now group by fields inside `best_grasp.json`, not output directory depth.
  - Added `metrics_by_dataset.csv`.

Verification after these fixes:

```bash
python -m pytest tests
# 16 passed

python scripts/run_ocid_one.py --dataset-root ../data/raw/OCID-VLG --refer-split multiple --split test --index 0 --output-root outputs/fix_ocid_vlg --top-k 5 --overwrite
# success

python scripts/evaluate_outputs.py --output-root outputs/fix_ocid_vlg --mode ocid_2d
# generated metrics_by_dataset.csv, metrics_by_split.csv, metrics_by_scene.csv
```

The project remains classified as **PARTIAL** rather than fully complete because generic dataset CLIs, full robot execution, and more rigorous 3D collision/execution validation are still outside the current implementation.

## 19. Florence-2 and MLP Scoring Update

The pipeline now includes:

- `target/florence2_grounder.py`: optional Florence-2 phrase grounding wrapper using Hugging Face `transformers`.
- `scoring/mlp_scorer.py`: CPU-only MLP scoring head.
- `scoring/factory.py`: scorer factory selecting `mlp` or `rule_based` from config.

Default ranking is now configured as:

```yaml
scoring:
  method: mlp
```

The default MLP uses a rule-based initialization so the pipeline remains executable without a trained checkpoint. To use a learned MLP, provide a checkpoint path under `scoring.mlp.checkpoint_path`.

Florence-2 is optional at runtime:

```bash
python scripts/run_ocid_one.py \
  --dataset-root ../data/raw/OCID-VLG \
  --refer-split multiple \
  --split test \
  --index 0 \
  --output-root outputs/ocid_florence_debug \
  --top-k 5 \
  --target-grounder florence2 \
  --florence-model-id microsoft/Florence-2-base-ft \
  --overwrite
```

If Florence-2 is unavailable and `fallback_to_annotation: true`, the pipeline falls back to the OCID target annotation and records the Florence error in target metadata.

Verification:

```bash
python -m pytest tests
# 19 passed

python scripts/run_ocid_one.py --dataset-root ../data/raw/OCID-VLG --refer-split multiple --split test --index 0 --output-root outputs/mlp_smoke --top-k 5 --overwrite
# success, ranked_grasps.json reports scorer=mlp
```
