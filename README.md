# VLMGraspPose

**Semantic-Guided 6-DoF Robotic Grasping with Vision-Language Models**

This repository contains a 5-stage pipeline for semantic-guided 6-DoF robotic grasping.

## 🚀 Quick Start: Local Reproduction

Follow these steps to reproduce the entire pipeline locally, from data preparation to final evaluation and visualisation.

### 1. Environment Setup

```bash
git clone https://github.com/langchengg/VLMGraspPose.git
cd VLMGraspPose

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```
*(Requires: Python ≥ 3.9, PyTorch ≥ 2.0)*

### 2. Download Data & Models
Download the required dataset split (`test_seen`) and pre-trained model weights (for VLM grounding).

```bash
# Download GraspNet test_seen data (~7 GB)
python scripts/download_data.py --test-seen

# Download Florence-2 and other model weights (Required for VLM modes)
python scripts/download_weights.py --all

# Preprocess the data index
python -m data.preprocess --split test_seen
```

### 3. Run the Pipeline
Execute the grasping pipeline with different grounders and scorers. Results are saved to `results/`.

```bash
# Baseline (Ground Truth + Rule-based Scorer)
python -m experiments.run_pipeline --split test_seen --grounder gt --scorer rule

# VLM Open-Vocabulary Mode (Requires GPU)
python -m experiments.run_pipeline --split test_seen --grounder vlm --scorer rule

# VLM Phrase Grounding Mode (Requires GPU)
python -m experiments.run_pipeline --split test_seen --grounder phrase --scorer rule
```

### 4. Evaluate Metrics
Compare the performance of different pipeline configurations (Hit@1, Hit@5, latency, errors).

```bash
# Compare metrics across all run results
python -m experiments.eval --compare

# Compute Grasp vs Ground Truth deviation metrics (Position/Angular errors)
python -m vis.compare_gt --scorer rule --max-samples 100
```

### 5. Visualisation
Generate 2D overlays, 3D point clouds, and Grasp vs GT comparison panels. Outputs are saved to `vis_output/`.

```bash
# Define a sample ID for visualization
export SAMPLE="scene_0100_0000_012_strawberry"

# 2D RGB Overlay with BBox + Grasp Candidates
python -m vis.vis_2d --sample $SAMPLE --scorer rule

# 3D Point Cloud with Gripper Poses (Static PNG)
python -m vis.vis_3d --sample $SAMPLE --scorer rule

# Ground Truth Comparison Panel
python -m vis.compare_gt --sample $SAMPLE --scorer rule --draw
```

---

## 🛠️ Advanced: Train Custom Scorers
If you want to train the learning-based scorers (Logistic Regression or MLP):

```bash
# 1. Download and preprocess training data (~30 GB)
python scripts/download_data.py --train
python -m data.preprocess --split train

# 2. Generate training features
# NOTE: Ensure '"train": PROJECT_ROOT / "train"' is uncommented in config.py
python -m experiments.run_pipeline --split train --grounder gt --scorer rule

# 3. Train the scorers
python -m experiments.train_ranker --mode pseudo --scorer logistic
python -m experiments.train_ranker --mode pseudo --scorer mlp

# 4. Evaluate with the trained MLP scorer
python -m experiments.run_pipeline --split test_seen --grounder gt --scorer mlp
```
