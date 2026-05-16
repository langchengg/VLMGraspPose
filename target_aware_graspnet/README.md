# Target-Aware GraspNet

Mac-compatible prototype for language-guided target-aware RGB-D grasping on the GraspNet dataset.

The pipeline is:

```text
Text command + RGB-D image + camera intrinsics
-> target selection / object-language mapping
-> target point cloud extraction
-> Open3D RGB-D geometric grasp sampler
-> candidate-target semantic-geometric features
-> rule-based target-conditioned re-ranking
-> Top-1 / Top-K grasp pose outputs
-> proxy metrics and qualitative figures
```

This version intentionally avoids CUDA-only dependencies. It does not use the official GraspNet baseline, MinkowskiEngine, spconv, PointNet++ custom ops, or Isaac Sim.

## Install

```bash
cd /Users/delaynomore/Downloads/VLMGraspPose/target_aware_graspnet
python -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
```

## Dataset

Expected dataset root:

```text
/Users/delaynomore/Downloads/VLMGraspPose/data/raw/graspnet
```

Expected scene layout:

```text
data/raw/graspnet/scenes/scene_0000/realsense/rgb/0000.png
data/raw/graspnet/scenes/scene_0000/realsense/depth/0000.png
data/raw/graspnet/scenes/scene_0000/realsense/label/0000.png
data/raw/graspnet/scenes/scene_0000/realsense/camK.npy
```

You can change paths in `configs/dataset.yaml` or pass `--dataset-root` / `--output-root`.

## Splits

```text
train:        scene_0000 - scene_0089
val:          scene_0090 - scene_0099
test_seen:    scene_0100 - scene_0129
test_similar: scene_0130 - scene_0159
test_novel:   scene_0160 - scene_0189
```

## Object-Language Mapping

GraspNet frames contain multiple objects and do not provide natural-language commands. This project builds an explicit mapping:

```text
(scene_id, camera, frame_id, target_id, command)
```

Examples:

```text
frame_0000 + "pick object_003" -> target_id 3
frame_0000 + "pick the left mug" -> target_id 1
```

Mapping files are saved to:

```text
outputs/mappings/object_language_mapping.csv
outputs/mappings/object_language_mapping.json
```

Quick debug defaults to one target per frame. Full split runs default to all visible targets per frame.

## Commands

Check indexing:

```bash
python scripts/build_index.py \
  --dataset-root ../data/raw/graspnet \
  --camera realsense \
  --split test_seen \
  --max-scenes 1 \
  --max-frames 5
```

Run one frame, largest visible target:

```bash
python scripts/run_one_frame.py \
  --dataset-root ../data/raw/graspnet \
  --scene-id scene_0100 \
  --camera realsense \
  --frame-id 0000 \
  --output-root outputs/debug \
  --top-k 5
```

Run all visible targets in one frame:

```bash
python scripts/run_one_frame.py \
  --dataset-root ../data/raw/graspnet \
  --scene-id scene_0100 \
  --camera realsense \
  --frame-id 0000 \
  --all-targets-per-frame \
  --output-root outputs/debug \
  --top-k 5
```

Run a small split test:

```bash
python scripts/run_split.py \
  --dataset-root ../data/raw/graspnet \
  --split test_seen \
  --camera realsense \
  --max-scenes 1 \
  --max-frames 10 \
  --one-target-per-frame \
  --output-root outputs \
  --top-k 5
```

Run a full split:

```bash
python scripts/run_split.py \
  --dataset-root ../data/raw/graspnet \
  --split train \
  --camera realsense \
  --output-root outputs \
  --top-k 5
```

Run all splits:

```bash
python scripts/run_all.py \
  --dataset-root ../data/raw/graspnet \
  --camera realsense \
  --output-root outputs \
  --top-k 5
```

Evaluate proxy metrics:

```bash
python scripts/evaluate_outputs.py --output-root outputs --mode proxy
```

Export paper figures:

```bash
python scripts/make_paper_figures.py \
  --output-root outputs \
  --num-success 12 \
  --num-failure 6
```

## Per-Target Output

Each processing unit writes to:

```text
outputs/{split}/{scene_id}/{camera}/{frame_id}/target_{target_id}/
```

Files:

```text
target_mask.png
target_pointcloud.ply
grasp_candidates.json
ranked_grasps.json
best_grasp.json
visualization_rgb.png
visualization_3d.png
score_breakdown.json
```

`best_grasp.json` contains split, scene, camera, frame, target id, command, bbox, position, quaternion, approach vector, closing direction, gripper width, grasp type, final score, feature breakdown, and Top-K fallbacks.

## Global Outputs

```text
outputs/summary.csv
outputs/metrics_by_split.csv
outputs/metrics_by_scene.csv
outputs/failure_cases.csv
outputs/runtime_report.csv
outputs/paper_figures/
```

## Known Limitations

- First version uses label images / target IDs, not Florence-2 grounding.
- Category names are disabled by default because GraspNet label values must be matched to a trusted dataset-specific label table. Without a trusted table, commands fall back to pseudo-language such as `pick object_003`. To enable real names, set `target_mapping.category_labels_path` and `target_mapping.category_labels_trusted: true` in `configs/dataset.yaml`.
- The sampler is geometric and CPU-only. It is suitable for an offline Mac prototype, not a learned GraspNet baseline replacement.
- Evaluation defaults to proxy validity. Full annotation matching against official 6D grasp labels is scaffolded but not fully implemented.
- Depth scale, intrinsics, mask alignment, and coordinate conventions should be checked visually on a small subset before running all scenes.
