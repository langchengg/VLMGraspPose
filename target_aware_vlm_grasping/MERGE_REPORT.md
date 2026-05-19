# Merge Report

## Active Project

`target_aware_vlm_grasping/`

This is now the active project path. It contains the executable OCID-VLG / OCID-Grasp language-conditioned RGB-D grasping pipeline.

## Old Codebases Found

1. Root step-based project:
   - old folders: `src/`, `scripts/`, `tests/`
   - old files: root `README.md`, `requirements.txt`, `config.py`
   - assumptions: GraspNet-oriented staged scripts, old Florence/GraspNet naming, CUDA device checks in old grounding code

2. Previous `target_aware_graspnet/` project:
   - stronger modular implementation
   - already contained OCID-VLG loader, OCID-Grasp fallback, Open3D sampler, feature extraction, scorer, evaluation, visualization
   - still had GraspNet naming and legacy GraspNet scripts

## What Was Kept

- OCID-VLG loader and OCID-Grasp fallback loader
- Open3D RGB-D point cloud processing
- geometric grasp sampler: top-down, bbox-aligned, side, normal-based
- semantic-geometric feature extraction
- rule-based scorer
- optional CPU NumPy MLP scorer
- OCID 2D rectangle metrics and grouped reports
- headless matplotlib visualizations
- Florence-2 optional target grounding wrapper

## What Was Changed

- Introduced unified `DatasetSample` dataclass.
- Added target grounder abstraction:
  - `OracleTargetGrounder`
  - `VLMTargetGrounder`
  - optional backend factory
- Replaced old GraspNet-oriented main runner with `run_dataset_sample`.
- Added new CLIs:
  - `scripts/run_one_sample.py`
  - `scripts/run_dataset.py`
  - `scripts/run_oracle_mode.py`
  - `scripts/run_vlm_mode.py`
  - `scripts/train_mlp_reranker.py`
- Added missing structure modules:
  - `target/language_mapping.py`
  - `pointcloud/pointcloud_processor.py`
  - `association/projection.py`
  - `evaluation/grounding_evaluator.py`
  - `evaluation/proxy_evaluator.py`
  - `evaluation/grasp_evaluator.py`
  - `scoring/train_mlp.py`
- Added tests for synthetic smoke pipeline, dataset loaders, mask utilities, RGB-D point cloud, candidate validation, JSON serialization, and language mapping.

## What Was Archived

- Root old project was moved to `legacy/root_step_project/`.
- Previous `target_aware_graspnet/` project was moved to `legacy/target_aware_graspnet/`.
- GraspNet-specific files copied into the active project during migration were moved to:
  - `target_aware_vlm_grasping/legacy/graspnet_optional/`

## Current Verification

- `python -m pytest tests`: 20 passed.
- Real OCID-VLG oracle CLI smoke test: passed.
- Real OCID-VLG MLP scorer smoke test: passed.
- Proxy report generation on smoke output: passed.
- VLM unavailable-backend test: failed gracefully with a clear runtime error and recommendation to use oracle mode or install optional backend.
- Core dependency scan found no CUDA-only runtime dependency in active source.

## Unresolved Issues

- VLM mode requires optional weights/packages and was not run with real Florence-2 weights during this merge.
- SAM refinement is not included in the core Mac path.
- 2D grasp rectangle evaluation is an approximation of projected 3D candidates, not full physical grasp execution.
