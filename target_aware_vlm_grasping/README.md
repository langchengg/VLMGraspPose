# Target-Aware VLM RGB-D Grasping

Mac-compatible Python project for language-guided target-aware RGB-D grasping on OCID-VLG and OCID-Grasp.

The active default VLM path is now:

```text
Florence-2-large-ft bbox grounding
→ SAM bbox-prompted mask refinement
→ depth/table cleanup
→ target point cloud
→ Open3D geometric grasp sampler
→ semantic-geometric re-ranker
→ Top-1 / Top-K grasps
```

Oracle mode is still available and should be used to validate the grasping pipeline independently of VLM grounding quality.

## Architecture

```text
Text command + RGB image + depth image + camera intrinsics
→ Target grounding
   - oracle: dataset bbox / mask
   - vlm: Florence-2-large-ft + SAM
→ target bbox + refined target mask
→ target RGB-D point cloud extraction
→ table/floor plane removal inside target region
→ Open3D RGB-D geometric grasp sampler
→ candidate-target semantic-geometric feature extraction
→ rule-based scorer or optional MLP scorer
→ ranked Top-K grasps
→ JSON outputs, visualizations, metrics
```

## Modules

- `src/dataset/`: OCID-VLG language-conditioned samples and OCID-Grasp fallback samples.
- `src/target/`: oracle grounder, Florence-2 VLM grounder, command parsing, relation-aware bbox selection, SAM mask refinement.
- `src/pointcloud/`: RGB-D to Open3D point cloud, target extraction, depth fallback refinement, table plane segmentation, normals, AABB/OBB.
- `src/grasp_sampler/`: CPU geometric candidates: top-down, bbox-aligned, side, normal-based.
- `src/association/`: target overlap, center alignment, width match, depth stability, approach score, collision and boundary penalties.
- `src/scoring/`: rule-based scorer and optional CPU NumPy MLP scoring head.
- `src/evaluation/`: proxy metrics, grounding metrics, OCID 2D grasp rectangle metrics.
- `src/visualization/`: RGB overlay with command/bbox/mask/Top-K grasps and 3D point cloud figures.

## Installation

Core CPU dependencies:

```bash
cd target_aware_vlm_grasping
python -m pip install -r requirements.txt
```

Optional VLM dependencies for Florence-2-large-ft + SAM:

```bash
python -m pip install -r requirements-vlm.txt
```

The core oracle/geometric pipeline does not require CUDA, MinkowskiEngine, spconv, PointNet++ custom ops, Isaac Sim, AnyGrasp, or the GraspNet baseline.

## Local Models

Expected local model paths:

```text
models/vlm/florence2-large-ft/
models/vlm/sam/sam_vit_b_01ec64.pth
```

The previous Florence-2 base weights may remain under:

```text
models/vlm/florence2/
```

but the default config no longer uses them.

Default VLM config:

```yaml
# configs/target_grounding.yaml
vlm_backend: florence2_sam
florence2:
  model_id: models/vlm/florence2-large-ft
  sam_enabled: true
  sam_required: true
  sam_checkpoint: models/vlm/sam/sam_vit_b_01ec64.pth
  sam_model_type: vit_b
```

## Dataset Setup

OCID-VLG default:

```text
data/OCID-VLG
```

OCID-Grasp default:

```text
data/OCID-Grasp
```

OCID-VLG samples are treated as language-conditioned target samples:

```text
(image_id, sentence, target_label, target_bbox, target_mask, grasp_rectangles)
```

One RGB-D image can therefore produce multiple target-conditioned samples.

## Run One Sample

Oracle mode:

```bash
python scripts/run_one_sample.py \
  --dataset ocid_vlg \
  --dataset-root data/OCID-VLG \
  --index 0 \
  --target-source oracle \
  --scorer rule_based \
  --output-root outputs/debug_oracle \
  --top-k 5 \
  --overwrite
```

VLM mode with Florence-2-large-ft + SAM:

```bash
python scripts/run_one_sample.py \
  --dataset ocid_vlg \
  --dataset-root data/OCID-VLG \
  --index 0 \
  --target-source vlm \
  --vlm-backend florence2_sam \
  --scorer rule_based \
  --output-root outputs/debug_vlm_large_sam \
  --top-k 5 \
  --overwrite
```

## Run Sampled Splits

Example: 200 samples for a split:

```bash
python scripts/run_dataset.py \
  --dataset ocid_vlg \
  --dataset-root data/OCID-VLG \
  --refer-split multiple \
  --split test \
  --target-source vlm \
  --vlm-backend florence2_sam \
  --scorer rule_based \
  --output-root outputs/vlm_large_sam_split_200 \
  --top-k 5 \
  --max-samples 200 \
  --overwrite
```

All OCID-VLG refer/split combinations can be run with:

```bash
for refer in multiple novel-classes novel-instances unique; do
  for split in train val test; do
    python scripts/run_dataset.py \
      --dataset ocid_vlg \
      --dataset-root data/OCID-VLG \
      --refer-split "$refer" \
      --split "$split" \
      --target-source vlm \
      --vlm-backend florence2_sam \
      --scorer rule_based \
      --output-root outputs/vlm_large_sam_split_200 \
      --top-k 5 \
      --max-samples 200 \
      --overwrite
  done
done
```

## Scoring

The rule-based scorer is active by default:

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

Weights are configured in `configs/scoring.yaml`.

Optional MLP scoring:

```bash
python scripts/run_one_sample.py \
  --dataset ocid_vlg \
  --dataset-root data/OCID-VLG \
  --index 0 \
  --target-source oracle \
  --scorer mlp \
  --output-root outputs/mlp_debug \
  --overwrite
```

If no MLP checkpoint is provided, the system falls back to a rule-initialized CPU MLP / rule-like behavior.

## Evaluation

Proxy evaluation:

```bash
python scripts/evaluate_outputs.py --output-root outputs/vlm_large_sam_split_200 --mode proxy
```

OCID 2D grasp rectangle evaluation:

```bash
python scripts/evaluate_outputs.py --output-root outputs/vlm_large_sam_split_200 --mode ocid_2d
```

Generated reports include:

- `metrics_by_dataset.csv`
- `metrics_by_split.csv`
- `metrics_by_scene.csv`
- `metrics_by_target_source.csv`
- `metrics_by_scorer.csv`
- `runtime_report.csv`
- `failure_cases.csv`

## Output Files

Each processed language-conditioned sample writes:

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

- dataset name, split, sample id / image id
- command / sentence
- target label and target id
- target source: `oracle` or `vlm`
- GT bbox and predicted bbox where available
- grounding metadata, including multi-query agreement and SAM metadata
- best grasp pose, orientation quaternion, approach vector, closing direction, width, type, score
- feature breakdown
- Top-K fallback candidates
- runtime

## Tests

```bash
python -m compileall src
python -m pytest tests
```

The test suite includes a synthetic RGB-D smoke test for non-empty point clouds, grasp candidate generation, scoring, output JSON, and RGB visualization.

## Known Limitations

- SAM improves mask quality inside a predicted bbox, but it cannot fix a semantically wrong Florence-2 bbox.
- Florence-2-large-ft is stronger than the previous base model, but cluttered tabletop referring expressions should still be evaluated with bbox/mask IoU.
- CPU SAM is slower than bbox-only mode, especially on full split runs.
- The geometric sampler is a CPU prototype, not a learned 6-DoF grasp detector.
- OCID 2D grasp rectangle evaluation uses projected grasp rectangles and is an approximation.
