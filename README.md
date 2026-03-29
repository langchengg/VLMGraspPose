# VLMGraspPose

**Semantic-Guided 6-DoF Robotic Grasping with Vision-Language Models**

A 5-stage pipeline that uses Vision-Language Models (VLMs) to ground target objects in cluttered scenes and generate semantically-aware 6-DoF grasp poses for parallel-jaw grippers.

<p align="center">
  <img src="docs/pipeline_overview.png" alt="Pipeline Overview" width="800"/>
</p>

## Overview

Given an RGB-D scene and a natural language query (e.g., *"grasp the strawberry"*), the pipeline:

1. **Stage 1 — Target Grounding**: Localise the target object via VLM (Florence-2) or GT labels
2. **Stage 2 — Grasp Generation**: Sample 6-DoF antipodal grasp candidates on the target region
3. **Stage 3 — Feature Extraction**: Compute per-candidate feature vectors (grasp quality, target alignment, etc.)
4. **Stage 4 — Scoring & Ranking**: Score candidates using rule-based, logistic regression, or MLP models
5. **Stage 5 — Selection**: Select the top-K grasps and output final 6-DoF poses

### Output Format

Each grasp pose is a **6-DoF parallel-jaw gripper pose** in camera frame:

| Field | Format | Description |
|-------|--------|-------------|
| `position` | `[x, y, z]` | Grasp centre in camera frame (metres) |
| `orientation` | `[qx, qy, qz, qw]` | Quaternion — x-axis: closing direction, z-axis: approach |
| `width` | `float` | Gripper opening width (metres) |

---

## Project Structure

```
VLMGraspPose/
├── config.py                  # Central configuration (paths, hyperparameters)
├── requirements.txt           # Python dependencies
├── scripts/
│   ├── download_data.py       # Download GraspNet dataset from Google Drive
│   └── download_weights.py    # Download pre-trained model weights
├── data/
│   ├── dataset.py             # Scene loader & sample generator
│   ├── point_cloud.py         # Depth → point cloud utilities
│   └── preprocess.py          # Generate JSONL sample index
├── stage1/
│   ├── grounding.py           # GroundTruthGrounder + VLMGrounder (Florence-2)
│   └── postprocess_bbox.py    # Bbox post-processing & persistence
├── stage2/
│   ├── grasp_generator.py     # AntipodalGraspSampler + GraspCandidate dataclass
│   └── roi_sampler.py         # Target-region local grasp generation
├── stage3/
│   └── feature_extractor.py   # 5-dim (core) or 9-dim (extended) feature vectors
├── stage4/
│   ├── rule_scorer.py         # Weighted-sum baseline scorer
│   ├── logistic_scorer.py     # Logistic Regression scorer
│   ├── mlp_scorer.py          # MLP scorer (PyTorch)
│   └── label_generator.py     # Pseudo-label generation for training
├── stage5/
│   └── select_best_grasp.py   # Top-K selection & JSON output
├── experiments/
│   ├── run_pipeline.py        # End-to-end pipeline runner
│   ├── eval.py                # Evaluation metrics (Hit@1, Hit@5, etc.)
│   └── train_ranker.py        # Train logistic / MLP scorers
└── vis/
    ├── grasp_drawing.py       # Gripper drawing primitives
    ├── vis_2d.py              # 2D overlay on RGB images
    ├── vis_3d.py              # 3D point cloud + grasp pose visualisation
    └── compare_gt.py          # Top-1 grasp vs GT comparison & metrics
```

---

## Quick Start

### 1. Environment Setup

```bash
# Clone the repository
git clone https://github.com/langchengg/VLMGraspPose.git
cd VLMGraspPose

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Linux/macOS
# venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
```

**Requirements**: Python ≥ 3.9, PyTorch ≥ 2.0

### 2. Download Dataset

This project uses the [GraspNet-1Billion](https://graspnet.net/) dataset. We provide a download script that automatically fetches data from Google Drive:

```bash
# Quick start — download test_seen split only (~7 GB)
python scripts/download_data.py --test-seen

# Download training data (~30 GB)
python scripts/download_data.py --train

# Download all test splits (seen + similar + novel)
python scripts/download_data.py --test-all

# Download everything including grasp labels and 3D models (~60 GB)
python scripts/download_data.py --all
```

The script will download, extract, and organise scenes into the correct directory structure:

```
VLMGraspPose/
├── test_seen/              # scenes 0100-0129 (auto-created)
│   ├── scene_0100/
│   │   ├── kinect/
│   │   │   ├── rgb/        # 0000.png – 0255.png
│   │   │   ├── depth/      # 0000.png – 0255.png
│   │   │   └── label/      # 0000.png – 0255.png
│   │   ├── annotations/
│   │   └── meta/
│   └── ...
├── train/                  # scenes 0000-0099 (if downloaded)
│   └── ...
```

> **Note**: Files are hosted on Google Drive. If download is slow, you can also download manually from [graspnet.net/datasets.html](https://graspnet.net/datasets.html) and place scenes in the corresponding split directories.

### 3. Download Model Weights

```bash
# Download all model weights (Florence-2 + Grounding DINO + GraspNet baseline)
python scripts/download_weights.py --all

# Or download individually:
python scripts/download_weights.py --florence2       # ~450 MB
python scripts/download_weights.py --grounding-dino  # ~700 MB
python scripts/download_weights.py --graspnet        # ~200 MB
```

Weights are saved to `models/` and automatically excluded from version control.

> **Note**: Florence-2 requires a GPU (CUDA) for stable inference. It may segfault on macOS/CPU due to `trust_remote_code` model internals.

### 4. Preprocess Data

Generate a JSONL index for efficient sample iteration:

```bash
python -m data.preprocess --split test_seen
```

This creates `processed/test_seen.jsonl` with all valid `(scene, view, object)` triples.

### 5. Run the Pipeline

```bash
# Quick demo (5 samples)
python -m experiments.run_pipeline --max-samples 5

# Full test_seen evaluation
python -m experiments.run_pipeline --split test_seen --scorer rule

# With extended 9-dim features
python -m experiments.run_pipeline --split test_seen --scorer rule --extended
```

**Options**:
| Flag | Description | Default |
|------|-------------|---------|
| `--split` | Data split to process | `test_seen` |
| `--scorer` | Scoring method: `rule`, `logistic`, `mlp` | `rule` |
| `--max-samples` | Limit number of samples | all |
| `--extended` | Use 9-dim extended features | 5-dim core |
| `--view-stride` | Process every N-th view | 16 |

Results are saved to `results/`.

### 6. Evaluate

```bash
# Evaluate the default rule-based results
python -m experiments.eval

# Evaluate a specific results file
python -m experiments.eval --results results/pipeline_summary_test_seen_rule.json

# Compare multiple scorers side-by-side
python -m experiments.eval --compare
```

**Metrics**:
- Target Hit@1 / Hit@5
- Average Top-1 Score
- Average Latency per Sample

---

## Visualisation

### 2D — RGB Overlay with BBox + Grasp Candidates

```bash
# Visualise a single sample with top-K grasps
python -m vis.vis_2d --sample scene_0100_0000_012_strawberry

# Show ALL candidates (not just top-K)
python -m vis.vis_2d --sample scene_0100_0000_012_strawberry --show-all
```

### 3D — Point Cloud + Gripper Poses

```bash
# Static PNG output (Matplotlib)
python -m vis.vis_3d --sample scene_0100_0000_012_strawberry

# Interactive viewer (requires Open3D)
python -m vis.vis_3d --sample scene_0100_0000_012_strawberry --backend open3d
```

### GT Comparison — Top-1 Grasp vs Ground Truth

```bash
# Single sample comparison
python -m vis.compare_gt --sample scene_0100_0000_012_strawberry --draw

# Batch comparison report
python -m vis.compare_gt --scorer rule --draw

# Batch report (no figures, just metrics)
python -m vis.compare_gt --scorer rule --max-samples 100
```

All visualisation outputs are saved to `vis_output/`.

---

## Training (Optional)

To train the learning-based scorers (Logistic Regression or MLP), you need training data:

### 1. Prepare Training Data

```bash
# Download training scenes (0000-0099)
python scripts/download_data.py --train
```

Then add the training split to `config.py`:

```python
DATA_DIRS = {
    "test_seen": PROJECT_ROOT / "test_seen",
    "train": PROJECT_ROOT / "train",
}
```

### 2. Generate Training Features & Labels

```bash
# Preprocess training data
python -m data.preprocess --split train

# Run pipeline on training split to generate features
python -m experiments.run_pipeline --split train --scorer rule

# Generate pseudo-labels (will be created automatically by the pipeline)
```

### 3. Train Scorers

```bash
# Train Logistic Regression scorer
python -m experiments.train_ranker --mode pseudo --scorer logistic

# Train MLP scorer
python -m experiments.train_ranker --mode pseudo --scorer mlp
```

### 4. Evaluate with Trained Scorers

```bash
# Run pipeline with trained MLP scorer
python -m experiments.run_pipeline --split test_seen --scorer mlp

# Compare all scorers
python -m experiments.eval --compare
```

> ⚠️ **Important**: Only train on `train` split data. Never train on `test_seen` — it's for evaluation only.

---

## Configuration

All hyperparameters are centralised in [`config.py`](config.py):

| Parameter | Default | Description |
|-----------|---------|-------------|
| `VIEW_STRIDE` | 16 | Process every N-th view per scene |
| `GRASP_TOP_K` | 50 | Max candidates per target |
| `GRASP_MIN_WIDTH` | 0.02 m | Minimum gripper opening |
| `GRASP_MAX_WIDTH` | 0.10 m | Maximum gripper opening |
| `VOXEL_SIZE` | 0.005 m | Point cloud downsampling resolution |
| `FEATURE_DIM_CORE` | 5 | Core feature dimensions (f1–f5) |
| `FEATURE_DIM_EXTENDED` | 9 | Extended feature dimensions (f1–f9) |
| `MLP_HIDDEN_DIMS` | [64, 32] | MLP scorer architecture |
| `MLP_LR` | 1e-3 | MLP learning rate |
| `MLP_EPOCHS` | 50 | MLP training epochs |

### Feature Descriptions

| Feature | Name | Description |
|---------|------|-------------|
| f1 | Grasp Score | Raw antipodal quality score |
| f2 | In-Target | Whether grasp centre is inside target region (0/1) |
| f3 | Distance | Normalised distance to target centre |
| f4 | IoU | Grasp footprint IoU with target bbox |
| f5 | VLM Confidence | VLM detection confidence |
| f6 | Depth Consistency | Grasp depth vs target depth alignment |
| f7 | Collision Risk | Nearby scene points within gripper volume |
| f8 | Boundary Distance | Distance from grasp centre to mask boundary |
| f9 | Normal Alignment | Grasp axis vs surface normal alignment |

---

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{vlmgrasppose2026,
    title={VLMGraspPose: Semantic-Guided 6-DoF Robotic Grasping with Vision-Language Models},
    year={2026},
    url={https://github.com/<your-username>/VLMGraspPose}
}
```

## Acknowledgements

- [GraspNet-1Billion](https://graspnet.net/) for the benchmark dataset
- [Florence-2](https://huggingface.co/microsoft/Florence-2-base) by Microsoft for open-vocabulary detection
- [Grounding DINO](https://github.com/IDEA-Research/GroundingDINO) by IDEA Research

## License

MIT License — see [LICENSE](LICENSE) for details.
