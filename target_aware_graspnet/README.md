# Target-Aware OCID-VLG

Mac-compatible prototype for language-guided target-aware RGB-D grasping. OCID-VLG is the primary dataset. GraspNet support is kept only as a legacy/fallback path because GraspNet does not provide reliable object category names or natural-language referring expressions.

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

`best_grasp.json` contains split, image id, scene path, frame id, target id, target label, sentence command, bbox, 3D pose, projected 2D grasp center, GT grasp rectangles, final score, feature breakdown, and Top-K fallbacks.

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

- OCID-VLG uses dataset-provided target boxes/masks as grounding supervision; Florence-2 grounding is not used in the default offline benchmark.
- OCID-Grasp fallback generates class commands from `catalog.csv` only when natural language is missing.
- The sampler is geometric and CPU-only. It is suitable for an offline Mac prototype, not a learned GraspNet baseline replacement.
- OCID 2D evaluation currently checks whether projected grasp centers fall inside GT grasp rectangles; this is a lightweight proxy, not full rectangle angle/IoU matching.
- Depth scale, intrinsics, mask alignment, and coordinate conventions should be checked visually on a small subset before running all scenes.
