# Target-Aware OCID-VLG

Mac-compatible prototype for language-guided target-aware RGB-D grasping. OCID-VLG is the primary dataset. GraspNet support is kept only as a legacy/fallback path because GraspNet does not provide reliable object category names or natural-language referring expressions.

The pipeline is:

```text
Text command + RGB-D image + camera intrinsics
-> Florence-2 target grounding or dataset target supervision
-> target point cloud extraction
-> Open3D RGB-D geometric grasp sampler
-> candidate-target semantic-geometric features
-> MLP scoring head target-conditioned re-ranking
-> Top-1 / Top-K grasp pose outputs
-> OCID 2D rectangle metrics and qualitative figures
```

This version intentionally avoids CUDA-only dependencies. It does not use the official GraspNet baseline, MinkowskiEngine, spconv, PointNet++ custom ops, or Isaac Sim.

## Install

```bash
cd /Users/delaynomore/Downloads/VLMGraspPose/target_aware_graspnet
python -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
```

Florence-2 grounding is optional because it requires local model dependencies and weights:

```bash
python -m pip install -r requirements-florence.txt
```

## Primary Dataset: OCID-VLG

Expected dataset root:

```text
/Users/delaynomore/Downloads/VLMGraspPose/data/raw/OCID-VLG
```

Expected OCID-VLG fields:

```text
refer/{unique,multiple,novel-classes,novel-instances}/{train,val,test}_expressions.json
ARID*/.../rgb/*.png
ARID*/.../depth/*.png
ARID*/.../seg_mask_instances_combi/*.png
grasps from expression JSON or Grasps_per_instance
```

Each OCID-VLG processing unit is:

```text
(image_id, sentence, target_label, target_bbox, target_mask)
```

The `sentence` / `question` field is used directly as the command. Pseudo object-id commands are not generated unless language is missing.

## OCID Commands

Run one OCID-VLG language target:

```bash
python scripts/run_ocid_one.py \
  --dataset-root ../data/raw/OCID-VLG \
  --refer-split multiple \
  --split test \
  --index 0 \
  --output-root outputs/ocid_debug \
  --top-k 5 \
  --overwrite
```

Run with Florence-2 target grounding instead of dataset-provided target boxes/masks:

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

Run an OCID-VLG split:

```bash
python scripts/run_ocid_split.py \
  --dataset-root ../data/raw/OCID-VLG \
  --refer-split multiple \
  --split test \
  --max-samples 20 \
  --output-root outputs/ocid_debug \
  --top-k 5
```

Evaluate against 2D grasp rectangles:

```bash
python scripts/evaluate_outputs.py \
  --output-root outputs/ocid_debug \
  --mode ocid_2d
```

Run OCID-Grasp fallback samples without referring expressions:

```bash
python scripts/run_ocid_grasp.py \
  --dataset-root ../data/raw/OCID-VLG \
  --max-samples 20 \
  --output-root outputs/ocid_grasp_debug \
  --top-k 5
```

## Legacy GraspNet Support

The old GraspNet scripts still exist:

```text
scripts/run_one_frame.py
scripts/run_split.py
scripts/run_all.py
```

Use them only for non-language proxy experiments.

Export paper figures:

```bash
python scripts/make_paper_figures.py \
  --output-root outputs \
  --num-success 12 \
  --num-failure 6
```

## Per-Target Output

Each OCID-VLG processing unit writes to:

```text
outputs/ocid_vlg/{refer_split}/{split}/{image_id}/
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

`best_grasp.json` contains split, image id, scene path, frame id, target id, target label, sentence command, bbox, 3D pose, projected 2D grasp center, projected 2D grasp rectangles, GT grasp rectangles, final score, feature breakdown, and Top-K fallbacks.

## Global Outputs

```text
outputs/summary.csv
outputs/metrics_by_dataset.csv
outputs/metrics_by_split.csv
outputs/metrics_by_scene.csv
outputs/failure_cases.csv
outputs/runtime_report.csv
outputs/paper_figures/
```

## Known Limitations

- OCID-VLG uses dataset-provided target boxes/masks by default for reproducible offline benchmarking. Use `--target-grounder florence2` to run Florence-2 phrase grounding when local weights are installed.
- The MLP scoring head defaults to a rule-based initialization. Train or load a checkpoint before treating it as a learned ranker.
- OCID-Grasp fallback generates class commands from `catalog.csv` only when natural language is missing.
- The sampler is geometric and CPU-only. It is suitable for an offline Mac prototype, not a learned GraspNet baseline replacement.
- OCID 2D evaluation includes projected center hit rate and rectangle IoU/angle matching. It is still 2D evaluation, not full robot execution validation.
- Depth scale, intrinsics, mask alignment, and coordinate conventions should be checked visually on a small subset before running all scenes.
