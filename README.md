# VLMGraspPose

VLMGraspPose is a research codebase for language-guided robotic grasping and referring grasp synthesis. The active project compares visual-grounding and grasp-selection baselines against a target-aware grasp ranking method for RGB-D scenes.

The core idea is to keep perception and grasp proposal generation fixed, then change only the final candidate selection rule. Given an RGB-D observation, a natural-language referring expression, and a top-k grasp candidate pool from a grasp detector or geometric sampler, the project ranks each candidate with target-awareness, semantic consistency, geometry, clearance, and collision signals.

## Motivation

Language-guided grasping is hard because a good grasp is not only a high-confidence grasp. It also has to belong to the referred target, avoid nearby clutter, and remain physically plausible for the gripper. End-to-end methods can learn joint visual-language-action behavior, while modular systems are easier to inspect and ablate. This repository focuses on controlled comparison between those choices.

## Method Overview

For a grasp candidate `g_i`, the intended target-aware ranking rule is:

```text
S(g_i) = α·Q(g_i) + β·M(g_i) + γ·Sem(g_i) + δ·Clear(g_i) − λ·Coll(g_i)
```

Where:

- `Q(g_i)`: original grasp detector confidence.
- `M(g_i)`: target mask or object-overlap support.
- `Sem(g_i)`: language-target semantic consistency.
- `Clear(g_i)`: local gripper approach clearance.
- `Coll(g_i)`: collision risk with nearby scene geometry.

The active implementation maps this conceptual formula to concrete features in `target_aware_vlm_grasping/src/association/` and `target_aware_vlm_grasping/src/scoring/`: initial geometric score, target overlap, center alignment, gripper-width match, depth stability, approach-direction score, collision penalty, and boundary penalty.

Current active pipeline:

```text
RGB image + depth image + camera intrinsics + text command
-> target grounding
   - oracle dataset bbox/mask, or
   - Florence-2 + optional SAM mask refinement
-> target RGB-D point-cloud extraction
-> table/floor cleanup and local geometry features
-> top-k grasp candidates from geometric sampler or detector output
-> target-aware reranking
-> top-1 and top-k grasp outputs
-> JSON records, visualizations, and evaluation reports
```

## Controlled Baseline Comparison

The intended comparison keeps the visual grounding output, point-cloud preprocessing, grasp detector, and top-k candidate pool fixed whenever possible.

1. **CROG-style end-to-end baseline**
   - CROG can be treated as an end-to-end or joint referring grasping baseline when the full model and dataset setup are available.
   - If full CROG grasp inference is not runnable locally, use it as a visual-grounding-only baseline and state that limitation.

2. **HiFi-CS-style modular baseline**
   - RGB + text -> target mask -> depth or point cloud -> grasp detector -> confidence-only top-1 grasp.
   - This isolates open-vocabulary visual grounding quality before final grasp selection.

3. **VL-Grasp-style modular baseline**
   - RGB + text -> bbox/mask -> point-cloud filter -> 6-DoF grasp candidate generation -> confidence-only selection.
   - The local checkout has experimental reranking integration notes, but the external VL-Grasp repository is not vendored here.

4. **Confidence-only selection**
   - Selects the top grasp by the detector's original score `Q(g_i)`.
   - This is the primary selection-rule baseline for controlled ablation.

5. **Proposed target-aware ranking**
   - Uses the same target estimate, point cloud, detector, and top-k candidate pool as the modular baselines.
   - Replaces only the final candidate selection rule with the target-aware score above.

## Repository Structure

The current local checkout contains these relevant areas:

```text
.
├── target_aware_vlm_grasping/      # Active implementation, configs, scripts, tests
├── HiFi_reproduction/              # HiFi/SAM3 + Dex-Net/GQ-CNN/VGN integration
├── LAVT_reproduction/              # Isolated GPLv3 LAVT reproduction source
├── crog_reproduction/CROG/         # MIT CROG source and reranking experiments
├── src/unified_reranking/           # Shared leakage-safe reranking pipeline
├── tools/unified_reranking/         # Unified pipeline command-line entrypoints
├── tests/unified_reranking/         # Unified reranking contract and pipeline tests
├── src/d1_reranking/                # Retrospective Dex-Net/GQ-CNN reranking extension
├── tools/d1_reranking/              # D1 planning, execution, and audit entrypoints
├── tests/d1_reranking/              # D1 contracts and lifecycle tests
├── src/gtmask_counterfactual/       # Post-formal GT-mask oracle diagnostic
├── tools/gtmask_counterfactual/     # Counterfactual protocol and replay entrypoints
├── tests/gtmask_counterfactual/     # Counterfactual mapping and reporting tests
├── src/graspnet6d/                 # GraspNet + frozen VGN 6-DoF experiment route
├── configs/graspnet6d/             # Portable 6-DoF profiles, schemas, and dependency lock
├── tests/graspnet6d/               # GraspNet 6-DoF contracts and workflow tests
├── src/robustness_suite/           # Post-hoc robustness and sensitivity analyses
├── src/robustness_completion/      # Duplicate and runtime completion analyses
├── reranking/                       # Modular experiment-matrix implementation
├── experiments/                     # Controlled cross-baseline evaluations
├── legacy/                         # Archived implementations and GraspNet adapters
├── ranking_baseline/               # Local VL-Grasp clone-time dependency
├── scripts/                        # Fixed-version external repository bootstrap
├── graphify-out/                   # Generated graph output; ignored for Git upload
├── docs/THIRD_PARTY_NOTICES.md     # Upstream SHAs and license boundaries
├── docs/CLEANUP_REPORT.md          # Prior cleanup and active-tree notes
├── LICENSE                         # Repository license
├── README.md                       # This file
└── .gitignore                      # Upload safety rules
```

Only lightweight source code, configs, scripts, tests, and documentation are committed. Local datasets, model weights, downloaded paper PDFs, generated outputs, caches, and license-restricted third-party clones are intentionally excluded.

## Installation

Use Python 3.10+ for the active pipeline.

```bash
cd target_aware_vlm_grasping
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Optional VLM grounding dependencies:

```bash
python -m pip install -r requirements-vlm.txt
```

The macOS dependency set for the GGCNN/GR-ConvNet 4-DoF adapters is recorded in
`HiFi_reproduction/requirements-grasp4dof-macos.txt`. Other optional baseline
environments remain separated under `HiFi_reproduction/requirements-*.txt` and
`crog_reproduction/CROG/requirements_mac.txt` to avoid combining incompatible
research stacks into one environment.

The active oracle/geometric pipeline is designed to run on macOS CPU/MPS without CUDA custom ops. Full 6-DoF detector baselines such as GraspNet, FGC-GraspNet, and VL-Grasp may require Linux, CUDA, and compiled extensions.

## Dataset Policy

Datasets are not included in this repository.

Do not commit:

- OCID-VLG, OCID-Grasp, RoboRefIt, GraspNet, GraspNet-1Billion, or other dataset folders.
- Model weights or checkpoints such as `.pth`, `.pt`, `.ckpt`, `.safetensors`, `.onnx`, or `.bin`.
- Raw arrays or experiment data such as `.npy`, `.npz`, `.pkl`, `.h5`, or `.hdf5`.
- Generated outputs, logs, visualizations, caches, virtual environments, or downloaded third-party paper PDFs.

Expected local dataset layout for the active project:

```text
target_aware_vlm_grasping/
├── data/
│   ├── OCID-VLG/
│   ├── OCID-Grasp/
│   └── 001_chips_can/              # Optional local single-object RGB-D data
├── models/
│   ├── vlm/
│   │   ├── florence2/
│   │   ├── florence2-large-ft/
│   │   └── sam/
│   └── reranker/
└── outputs/                        # Generated locally; not committed
```

The tracked placeholders `target_aware_vlm_grasping/data/README.md` and `target_aware_vlm_grasping/models/README.md` document local expected layouts without committing the data or weights.

## Running the Active Pipeline

Run one OCID-VLG sample with oracle target grounding:

```bash
cd target_aware_vlm_grasping
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

Run one sample with VLM target grounding:

```bash
python scripts/run_one_sample.py \
  --dataset ocid_vlg \
  --dataset-root data/OCID-VLG \
  --index 0 \
  --target-source vlm \
  --vlm-backend florence2_sam \
  --scorer rule_based \
  --output-root outputs/debug_vlm \
  --top-k 5 \
  --overwrite
```

Run a sampled split:

```bash
python scripts/run_dataset.py \
  --dataset ocid_vlg \
  --dataset-root data/OCID-VLG \
  --refer-split multiple \
  --split test \
  --target-source oracle \
  --scorer rule_based \
  --output-root outputs/oracle_split \
  --top-k 5 \
  --max-samples 200 \
  --overwrite
```

Evaluate generated outputs:

```bash
python scripts/evaluate_outputs.py --output-root outputs/oracle_split --mode proxy
python scripts/evaluate_outputs.py --output-root outputs/oracle_split --mode ocid_2d
```

Create a rule-initialized MLP reranker checkpoint:

```bash
python scripts/train_mlp_reranker.py --output outputs/checkpoints/mlp_rule_initialized.npz
```

Additional reproducibility entrypoints are grouped by experiment family:

| Experiment family | Primary entrypoint | Configuration / dependency source |
| --- | --- | --- |
| Unified cross-backend reranking | `tools/unified_reranking/run_all.py` | `src/unified_reranking/`, `tests/unified_reranking/` |
| Retrospective D1 reranking extension | `tools/d1_reranking/bootstrap.py` | `src/d1_reranking/`, `tests/d1_reranking/` |
| Post-formal GT-mask counterfactual diagnostic | `tools/gtmask_counterfactual/bootstrap.py` | `src/gtmask_counterfactual/`, `tests/gtmask_counterfactual/` |
| Post-formal case analysis | `tools/unified_reranking/run_case_analysis.py` | `src/unified_reranking/case_analysis.py`, `case_visuals.py` |
| Post-lock R1-R6 Test comparison | `tools/unified_reranking/run_postlock_r1_r6_test_comparison.py` | Frozen unified-reranking run artifacts |
| GraspNet + frozen VGN 6-DoF route | `python -m graspnet6d.cli` | `configs/graspnet6d/`, `docs/DATA_LICENSES.md`, `docs/MANUAL_DOWNLOAD_REQUIRED.md` |
| Robustness and sensitivity suite | `python -m robustness_suite.cli` | `src/robustness_suite/`, `tests/robustness_suite/` |
| Robustness completion analyses | `python -m robustness_completion.cli` | `src/robustness_completion/`, `tests/robustness_completion/` |
| Controlled CROG/HiFi/G1/C1 comparison without reranking | `experiments/fair_crog_hifics_g1_c1_no_rerank/prepare_run.py` | Package-local scripts and tests |
| 4-DoF analytic, GGCNN, and GR-ConvNet validation | `HiFi_reproduction/tools/grasp4dof/run_method.py` | `HiFi_reproduction/configs/grasp4dof_validation/`, `requirements-grasp4dof-macos.txt` |
| SAM3 proposal-bank evaluation | `HiFi_reproduction/scripts/run_sam3_p90_experiment.py` | `HiFi_reproduction/configs/sam3_proposal_bank_p90_v1/` |
| Selective SAM3 visual grounding | `HiFi_reproduction/tools/selective_sam3_vg/run_formal_inference.py` | `HiFi_reproduction/configs/selective_sam3_vg_validation.yaml` |
| HiFi modular reranking | `HiFi_reproduction/tools/modular_reranking/train_rerankers.py` | `HiFi_reproduction/configs/modular_reranking_v1.yaml` |
| CROG VLM-safe reranking | `crog_reproduction/CROG/scripts/run_vlm_safe_rerank.py` | `crog_reproduction/CROG/prompts/` and `.env.example` |

Machine-locked experiment manifests that contain absolute local paths are kept
outside Git. Recreate them from the portable configs and scripts above after
placing datasets and checkpoints at local paths.

## Baselines

### CROG

CROG is a referring grasp synthesis baseline for language-guided robot grasping. A source-only snapshot is included under `crog_reproduction/CROG/` at upstream commit `1eeee85de1fe6bffdc66c9ed9a622028ea04578e`, together with the local Mac/MPS and reranking work. Datasets, checkpoints, generated outputs, and two upstream entrypoints containing an embedded external-service credential are excluded.

Use `train_crog_mac.py` for the included single-device training path.

Follow the upstream dataset and CUDA/DDP instructions for full training. The local Mac path can run preprocessing, inspection, visualization, and limited single-device experiments, but exact full training is better suited to an NVIDIA CUDA server.

### HiFi-CS

HiFi-CS is treated as a visual-grounding baseline:

```text
RGB + text -> target mask -> depth/point cloud -> grasp detector -> confidence-only top-1 grasp
```

HiFi-CS itself is not vendored because the upstream repository does not declare a repository-wide license. The project-owned adapters, scripts, tests, and SAM3 integration remain under `HiFi_reproduction/`. Recreate the exact upstream clone with:

```bash
bash scripts/fetch_external_repositories.sh
```

### VL-Grasp

VL-Grasp is treated as a modular 6-DoF grasping baseline:

```text
RGB + text -> bbox/mask -> point-cloud filter -> 6-DoF candidates -> confidence-only selection
```

The local `ranking_baseline/VL-Grasp/` checkout contains experimental reranking work, but the full external baseline repo, RoboRefIt files, GraspNet files, compiled extensions, and generated outputs are intentionally excluded from Git.

The same bootstrap script recreates VL-Grasp and other clone-time dependencies at their audited commits without overwriting an existing local checkout.

### LAVT

`LAVT_reproduction/` contains an isolated GPLv3 source snapshot at upstream commit `1da0af9f21b637c0cae9ea1363d2dd9b40e19628`, plus the OCID-VLG adaptation, configs, scripts, and tests. Its GPLv3 license applies to that subtree; the root MIT license does not relicense it.

### VGN

`HiFi_reproduction/third_party/vgn/` contains the BSD-3-Clause VGN `corl2020` source snapshot at commit `d7af0622433f52ae88ebe81533f12b46b33e951a`. Local VGN adapters, runners, requirements, and tests are in `HiFi_reproduction/`.

### GGCNN and GR-ConvNet

`HiFi_reproduction/third_party_src/ggcnn/` contains the BSD-3-Clause GG-CNN
source snapshot at commit `0c50aa7600e8a30d44c5c85cebd6e3394a81f30e`.
`HiFi_reproduction/third_party_src/grconvnet/` contains the BSD-3-Clause
robotic-grasping/GR-ConvNet source snapshot at commit
`bdd49367f8619be94123fb3187c2f8ad5100ef46`. Their upstream license files are
retained. Downloaded checkpoints and upstream `trained-models/` files are not
committed.

### Confidence-Only Selection

This baseline selects the candidate with the largest original detector confidence. In the active geometric path, this corresponds to using the initial geometric score without target-aware penalties or semantic/geometric support terms.

### Proposed Ranking

The proposed method keeps the same candidate pool and reranks candidates with target overlap, semantic consistency, clearance, and collision-aware geometry. This makes the ablation focused on selection quality rather than changes in detection or grounding.

## Evaluation Metrics

Use metrics appropriate to the available labels and runtime:

- Visual grounding: IoU, P@50, P@75, P@90.
- Target-hit rate: whether the selected grasp center or footprint belongs to the referred target.
- Collision-free rate: proxy collision-free rate from local geometry or a simulator/robot checker when available.
- Clearance score: local gripper approach clearance around the selected candidate.
- Top-1 grasp success proxy: whether the selected top-1 satisfies target overlap, alignment, collision, depth, and width thresholds.
- OCID 2D grasp proxy: top-k grasp center hit rate and top-k rectangle match rate.
- Simulation or real grasp success: report only when a simulator or robot execution setup is actually available.

The active evaluator writes grouped CSV reports such as `metrics_by_dataset.csv`, `metrics_by_split.csv`, `metrics_by_target_source.csv`, `metrics_by_scorer.csv`, `runtime_report.csv`, and `failure_cases.csv`.

## Reproducibility Notes

- Use oracle grounding to test grasp generation and ranking independently of VLM grounding quality.
- Use VLM grounding runs to evaluate the full language-to-grasp pipeline.
- Keep output roots separate by experiment condition.
- Record dataset split, target source, scorer, top-k, random seed, model paths, and hardware.
- Do not report local one-sample smoke runs as final benchmark results.
- For controlled comparisons, verify that all methods use the same images, target expressions, point-cloud preprocessing, and top-k candidate pool.

## Limitations

- Reranking cannot create a good grasp if the top-k candidate pool contains no feasible grasp.
- Target-aware ranking quality depends on grounding quality and local geometry quality.
- Collision and clearance in the active pipeline are proxy features, not a full robot collision checker.
- Full CUDA-based grasp detectors may require Linux, NVIDIA GPUs, and compiled custom extensions.
- macOS can run README-level setup, preprocessing, visualization, oracle mode, VLM grounding, and ranking logic, but heavy 6-DoF inference or full baseline training may need a remote GPU server.
- Third-party source is included only where redistribution terms were verified and preserved. Restricted, mixed-license, or unlicensed upstream repositories remain fixed-version clone-time dependencies.

## Related Work and Citations

Please cite the relevant upstream work when using these baselines or datasets:

- CROG: Language-guided Robot Grasping: CLIP-based Referring Grasp Synthesis in Clutter, https://github.com/HilbertXu/CROG
- VL-Grasp: A 6-DoF Interactive Grasp Policy for Language-Oriented Objects in Cluttered Indoor Scenes, https://github.com/luyh20/VL-Grasp
- RoboRefIt dataset, https://luyh20.github.io/RoboRefIt.github.io/
- HiFi-CS: Towards Open Vocabulary Visual Grounding For Robotic Grasping Using Vision-Language Models, https://github.com/vineet2104/hifics
- Vision-Language-Action for target-oriented grasping reference baseline, https://github.com/xukechun/Vision-Language-Grasping

Downloaded third-party paper PDFs should stay outside Git. Add BibTeX entries here when final paper references are selected.

## License

This repository is released under the MIT License. See `LICENSE`.

Third-party baselines, datasets, and model checkpoints keep their own licenses and terms. See `docs/THIRD_PARTY_NOTICES.md` for exact upstream commits, included components, and clone-time dependencies.
