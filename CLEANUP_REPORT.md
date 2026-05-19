# Cleanup Report

## Active Project

The only active project is:

```text
target_aware_vlm_grasping/
```

Root-level `legacy/` contains archived reference code only.

## Moved

| Source | Destination | Result |
|---|---|---|
| `data/raw/OCID-VLG/` | `target_aware_vlm_grasping/data/OCID-VLG/` | moved, about 12G |
| `data/raw/OCID_grasp/` | `target_aware_vlm_grasping/data/OCID-Grasp/` | moved, about 12G |
| `models/florence2_base_ft/` | `target_aware_vlm_grasping/models/vlm/florence2/` | moved, about 887M |
| `target_aware_vlm_grasping/legacy/graspnet_optional/` | `legacy/unused_graspnet_optional/` | archived after active-reference check |
| `vis/` | `legacy/root_visualization_helpers/` | archived after active-reference check |

Created:

- `target_aware_vlm_grasping/data/README.md`
- `target_aware_vlm_grasping/models/README.md`
- `target_aware_vlm_grasping/.gitignore`
- placeholder `.gitkeep` files for lightweight tracked directories

## Deleted / Cleaned

Deleted reproducible or noisy paths:

- `target_aware_vlm_grasping/outputs/*` from previous runs
- project cache folders: `.cache/`, `.hf_cache/`, `.matplotlib_cache/`, `.pytest_cache/`
- old root placeholders: `data/metadata`, `data/splits`, `data/raw`
- old reproducible roots: `derived/`, `results/`
- empty/obsolete `external/`
- old output/cache folders under `legacy/target_aware_graspnet/`
- `.DS_Store`, `__pycache__/`, and `*.pyc` outside protected Git internals

Before deleting `data/metadata`, `data/splits`, `derived`, and `results`, the active project was searched with:

```bash
rg -n "data/metadata|data/splits|derived/|results/" target_aware_vlm_grasping --glob '!outputs/**' --glob '!legacy/**'
```

No active references were found.

## Updated Paths

Config defaults now point to project-local paths:

```yaml
datasets:
  ocid_vlg:
    root: data/OCID-VLG
  ocid_grasp:
    root: data/OCID-Grasp
output_root: outputs
```

Other updated paths:

- `configs/ocid_vlg.yaml`: `root: data/OCID-VLG`
- `configs/default.yaml`: dataset root `data/OCID-VLG`, Florence-2 path `models/vlm/florence2`
- `configs/target_grounding.yaml`: `model_id: models/vlm/florence2`
- `configs/mlp.yaml`: default checkpoint directory `models/reranker/mlp`
- README examples use `data/OCID-VLG`, `data/OCID-Grasp`, and `models/vlm/florence2`

`scripts/run_one_sample.py` and `scripts/run_dataset.py` still accept `--dataset-root`, but it is now optional:

- `ocid_vlg` default: `data/OCID-VLG`
- `ocid_grasp` default: `data/OCID-Grasp`

## Code Fixes During Cleanup

- Extended `OCIDGraspIndexBuilder` to support the actual moved OCID-Grasp layout with `rgb/`, `depth/`, `label/`, `seg_mask_instances_combi/`, and `Annotations_per_class/`.
- Updated tests to use project-local dataset paths.
- Updated the default logger name from the old project name to `target_aware_vlm_grasping`.

## Git Tracking Safety

Both root `.gitignore` and `target_aware_vlm_grasping/.gitignore` now ignore:

- `data/`
- `models/`
- outputs/logs/cache folders
- `.pt`, `.pth`, `.ckpt`, `.safetensors`, `.bin`, `.onnx`, `.pkl`, `.joblib`

The required tracking check was run:

```bash
git ls-files | rg "target_aware_vlm_grasping/(data|models)|\\.pt|\\.pth|\\.ckpt|\\.safetensors|\\.bin|\\.onnx|\\.pkl|\\.joblib"
```

Result: no tracked large data/model files matched, so no `git rm --cached` was required.

## Final Active Tree

```text
target_aware_vlm_grasping/
├── configs/
├── data/
│   ├── OCID-VLG/
│   └── OCID-Grasp/
├── models/
│   ├── vlm/
│   │   └── florence2/
│   └── reranker/
├── outputs/
├── scripts/
├── src/
└── tests/
```

Archived root-level folders:

```text
legacy/
├── external_graspnet/
├── root_step_project/
├── root_visualization_helpers/
├── target_aware_graspnet/
└── unused_graspnet_optional/
```

## Verification Results

| Check | Result |
|---|---|
| `find . -maxdepth 3 -type d` | ran; final active and legacy trees verified |
| `python -m compileall target_aware_vlm_grasping/src` | passed |
| `python scripts/run_one_sample.py --help` | passed; `--dataset-root` is optional |
| `pytest tests` | passed, 20 tests |
| `pytest tests/test_smoke_pipeline.py` | passed, 1 test |
| OCID-VLG oracle smoke | passed |
| OCID-VLG VLM Florence-2 smoke | passed |
| path search for old absolute/raw/model paths | passed, no active matches outside ignored outputs/legacy |
| large tracked file check | passed, no matching tracked files |

Oracle smoke command:

```bash
python scripts/run_one_sample.py \
  --dataset ocid_vlg \
  --dataset-root data/OCID-VLG \
  --index 0 \
  --target-source oracle \
  --scorer rule_based \
  --output-root outputs/ocid_vlg_debug \
  --top-k 5 \
  --overwrite
```

Result:

- status: success
- target: `flashlight_1`
- bbox: `[259, 381, 344, 408]`
- final score: `0.7591`
- output files generated: `target_mask.png`, `target_pointcloud.ply`, `grasp_candidates.json`, `ranked_grasps.json`, `best_grasp.json`, `score_breakdown.json`, `visualization_rgb.png`, `visualization_3d.png`

VLM smoke command:

```bash
python scripts/run_one_sample.py \
  --dataset ocid_vlg \
  --dataset-root data/OCID-VLG \
  --index 0 \
  --target-source vlm \
  --vlm-backend florence2 \
  --scorer rule_based \
  --output-root outputs/vlm_debug \
  --top-k 5 \
  --overwrite
```

Result:

- status: success
- local Florence-2 loaded from `models/vlm/florence2`
- predicted bbox: `[208, 321, 249, 364]`
- GT bbox: `[259, 381, 344, 408]`
- final score: `0.7630`
- output files generated: same required per-sample output set

## Remaining Issues

- Florence-2 VLM mode is executable, but its first-sample bbox is visibly different from the OCID-VLG GT bbox. This is a model/prompt/backend quality issue, not a runtime blocker. Use oracle mode for controlled grasp experiments and evaluate VLM grounding with bbox/mask IoU before reporting final VLM results.
- A `.DS_Store` inside `.git/` could not be removed by normal cleanup due filesystem permissions. It is not part of the active project tree.
- `git status` still shows many tracked deletions from old generated outputs and archived-code moves because the cleanup intentionally removed reproducible artifacts and moved old reference folders. Data/model contents are ignored and not tracked.
