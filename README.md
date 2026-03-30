# VLMGraspPose

A Vision-Language Model (VLM) guided robotic grasp pose generation pipeline, evaluated on the GraspNet-1Billion dataset. This project takes an RGB-D image and a natural language text query (e.g., "pick the mug") and outputs a precise 6-DoF grasp pose for a robotic gripper.

## Architecture

The pipeline strictly follows a 6-block semantic grasping architecture:
1. **Input**: Text Target Description + Scene RGB-D Image
2. **Branch A (Target Grounding)**: Uses a lightweight VLM (Florence-2) for open-vocabulary detection or phrase grounding (with a Ground-Truth oracle option).
3. **Branch B (Grasp Proposal)**: Generates geometry-based grasp candidates (Antipodal geometric sampling) from the point cloud.
4. **Candidate-Target Association**: Computes a semantic-geometric feature vector (5-dim core or 9-dim extended) for each grasp candidate relative to the VLM target region.
5. **Re-Ranking Module**: Scores candidates using a defined ranking strategy (Rule-based, Logistic Regression, or MLP).
6. **Final Selection**: Outputs the Top-K 6-DoF grasp poses (position, orientation, grip width).

---

## 🚀 Quick Start (Google Colab)

The easiest way to reproduce the results and generate paper-ready figures is using Google Colab (Requires Colab Pro + T4 GPU).

### 1. Full Pipeline (Training + Evaluation)
Copy the entire contents of `colab_full_pipeline.py` into a single Colab cell and run it. 
This script performs the end-to-end workflow automatically (~5-7 hours):
- Downloads the GraspNet dataset splits and Florence-2 model weights.
- Generates features on the `train` split and trains the learning-based scorers (Logistic, MLP).
- Evaluates the pipeline on the `test_seen` split across a matrix of Grounder × Scorer combinations.
- Computes deviation metrics (Position/Angular errors) against Ground Truth.
- Generates 2D and 3D visualization figures.

### 2. Disk Recovery (If Colab Runs Out of Space)
Due to the large size of the dataset (~37GB), Colab disk space may fill up during training, causing the final evaluation step to fail.
If this happens, **do not reset the runtime**. Simply copy the contents of `colab_recovery.py` into a **new Colab cell** and run it.
- It safely deletes bulky training data and intermediate outputs (since the models are already trained).
- It resumes and completes the lightweight evaluation and visualization steps on the `test_seen` split.

---

## 💻 Local CLI Usage

If you have downloaded the dataset locally, you can run the pipeline directly:

```bash
# Baseline: Ground Truth oracle + Rule-based scorer
python -m experiments.run_pipeline --split test_seen --grounder gt --scorer rule

# VLM Experiment: Florence-2 (Open-Vocabulary) + MLP scorer
python -m experiments.run_pipeline --split test_seen --grounder vlm --scorer mlp

# VLM Experiment: Florence-2 (Phrase Grounding) + MLP scorer + Extended Features
python -m experiments.run_pipeline --split test_seen --grounder phrase --scorer mlp --extended
```

Evaluate generated results and compare against Ground Truth:
```bash
python -m experiments.eval --compare
python -m vis.compare_gt --scorer mlp --max-samples 50
```

---

## 📊 Outputs & Paper Assets

All results are automatically saved to your mounted Google Drive (or local directory) and are structured for direct inclusion in academic papers.

* **Quantitative Metrics (`results/`)**
  * `*.metrics.json`: Aggregated Hit@1, Hit@5, average scores, and latency.
  * `scene_*.json`: Contains the exact **Top-K 6-DoF grasp poses** for each object:
    * `position`: `[x, y, z]` coordinates of the gripper center.
    * `orientation`: `[qx, qy, qz, qw]` quaternion.
    * `width`: Gripper opening width.
* **Visualizations (`vis_output/`)**
  * `*_2d.png`: RGB image overlays showing target bounding boxes and projected 2D grasp skeletons.
  * `*_3d.png`: Point cloud renderings with 3D gripper poses.
  * `*_compare.png`: Side-by-side Ground Truth vs. Prediction error analysis panels.
* **Trained Models (`models/`)**
  * Saved weights for the `LogisticScorer` and `MLPScorer`.
