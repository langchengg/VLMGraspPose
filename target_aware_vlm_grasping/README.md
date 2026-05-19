# Target-Aware VLM RGB-D Grasping

Mac-compatible Python project for language-guided target-aware RGB-D grasping.

Primary dataset: OCID-VLG.  
Auxiliary dataset: OCID-Grasp.  
Legacy reference code, if retained, is archived outside the active project and is not part of the active pipeline.

## Architecture

```text
Text command + RGB image + depth image + camera intrinsics
→ Target grounding
   - oracle mode: use dataset bbox / mask
   - VLM mode: optional Florence-2 / other grounding backend
→ target bbox / mask
→ target point cloud extraction
→ Open3D RGB-D geometric grasp sampler
→ candidate-target semantic-geometric feature extraction
→ rule-based or optional MLP re-ranker
→ Top-1 / Top-K grasp output
→ evaluation and visualization
```

## What Each Module Does

- `src/dataset/`: loads OCID-VLG language-conditioned samples and OCID-Grasp fallback samples.
- `src/target/`: target grounding abstraction. `OracleTargetGrounder` needs no VLM. `VLMTargetGrounder` lazy-loads optional backends.
- `src/pointcloud/`: RGB-D to Open3D point cloud, target extraction, filtering, table plane segmentation, normals, AABB/OBB.
- `src/grasp_sampler/`: CPU geometric grasp candidates: top-down, bbox-aligned, side, normal-based.
- `src/association/`: features per candidate: overlap, center alignment, width match, depth stability, approach score, collision and boundary penalties.
- `src/scoring/`: rule-based scorer and optional NumPy MLP scoring head.
- `src/evaluation/`: proxy metrics, OCID 2D grasp rectangle metrics, grounding metrics, grouped reports.
- `src/visualization/`: RGB overlay, Top-K projected grasps, 3D point cloud plot, export helpers.

## Installation

```bash
cd target_aware_vlm_grasping
python -m venv .venv
.venv/bin/pip install -r requirements.txt
```

Core dependencies are CPU/Mac compatible. Optional VLM dependencies are separate:

```bash
.venv/bin/pip install -r requirements-vlm.txt
```

The core pipeline does not require CUDA, MinkowskiEngine, spconv, PointNet++ custom ops, Isaac Sim, AnyGrasp, or any CUDA-only learned grasp baseline.

## Dataset Setup

OCID-VLG is expected by default at:

```text
data/OCID-VLG
```

The loader expects OCID-VLG samples with:

- RGB image
- depth image
- sentence / referring expression
- target label
- target bbox
- target mask if available
- 2D grasp rectangles if available
- fallback camera intrinsics from config

OCID-Grasp can be used as auxiliary data. If language is missing, commands are generated from class labels, for example:

```text
pick the cup
pick the left cup
pick the right bottle
```

OCID-Grasp is expected by default at:

```text
data/OCID-Grasp
```

## Run One Sample

Oracle mode uses dataset target bbox / mask:

```bash
python scripts/run_one_sample.py \
  --dataset ocid_vlg \
  --dataset-root data/OCID-VLG \
  --index 0 \
  --target-source oracle \
  --scorer rule_based \
  --output-root outputs/debug \
  --top-k 5 \
  --overwrite
```

VLM mode uses text + RGB to predict the target region. The default Florence-2 config points to the project-local weights:

```text
models/vlm/florence2
```

Backends are optional:

```bash
python scripts/run_one_sample.py \
  --dataset ocid_vlg \
  --dataset-root data/OCID-VLG \
  --index 0 \
  --target-source vlm \
  --vlm-backend florence2 \
  --scorer rule_based \
  --output-root outputs/debug_vlm \
  --top-k 5
```

If the selected VLM backend is not installed or the local model path is missing, the script fails clearly and recommends oracle mode. It does not import VLM packages during oracle mode. The local Florence-2 backend is executable, but target boxes should be evaluated against OCID ground truth before using VLM mode for final metrics.

## Run Dataset

```bash
python scripts/run_dataset.py \
  --dataset ocid_vlg \
  --dataset-root data/OCID-VLG \
  --target-source oracle \
  --scorer rule_based \
  --max-samples 20 \
  --output-root outputs/ocid_vlg \
  --top-k 5 \
  --resume
```

OCID-Grasp fallback:

```bash
python scripts/run_dataset.py \
  --dataset ocid_grasp \
  --dataset-root data/OCID-Grasp \
  --target-source oracle \
  --scorer rule_based \
  --max-samples 20 \
  --output-root outputs/ocid_grasp
```

MLP scoring head:

```bash
python scripts/run_one_sample.py \
  --dataset ocid_vlg \
  --dataset-root data/OCID-VLG \
  --index 0 \
  --target-source oracle \
  --scorer mlp \
  --output-root outputs/mlp_debug
```

If an MLP checkpoint is missing, the project falls back to a rule-initialized CPU MLP / rule behavior instead of making training mandatory.

## Evaluation

Proxy evaluation:

```bash
python scripts/evaluate_outputs.py --output-root outputs/ocid_vlg --mode proxy
```

OCID 2D grasp rectangle evaluation:

```bash
python scripts/evaluate_outputs.py --output-root outputs/ocid_vlg --mode ocid_2d
```

Generated reports:

- `metrics_by_dataset.csv`
- `metrics_by_split.csv`
- `metrics_by_scene.csv`
- `metrics_by_target_source.csv`
- `metrics_by_scorer.csv`
- `runtime_report.csv`
- `failure_cases.csv`

## Output Files

Each processed language-conditioned target sample writes:

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

The output path includes the language-conditioned sample id, so multiple targets from the same RGB-D image do not overwrite each other.

`best_grasp.json` includes:

- dataset name
- sample id / image id
- command / sentence
- target label and target id
- target source: `oracle` or `vlm`
- GT and predicted bbox fields where available
- best grasp pose, quaternion, approach vector, closing direction, width, type, score
- feature breakdown
- Top-K fallback candidates
- runtime

## Tests

```bash
python -m pytest tests
```

The test suite includes a full synthetic RGB-D smoke test that verifies non-empty point clouds, candidate generation, scoring, `best_grasp.json`, and RGB visualization output.

## Known Limitations

- VLM mode is executable with the local Florence-2 directory at `models/vlm/florence2`; optional Python packages are still required. Grounding accuracy is not guaranteed and should be validated with bbox/mask IoU.
- SAM refinement is not bundled in the Mac core path.
- The geometric sampler is a CPU prototype, not a learned 6-DoF grasp detector.
- 3D grasp quality is evaluated with proxy metrics unless richer annotations are available.
- OCID 2D grasp rectangle evaluation uses projected grasp rectangles and is an approximation.
