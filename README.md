# VLMGraspPose

**Semantic-Guided 6-DoF Robotic Grasping with Vision-Language Models**

A 12-step research pipeline for text-conditioned target grasp selection using
GraspNet-1Billion and Florence-2.

**Thesis claim:** *A lightweight semantic-geometric reranker improves
target-object grasp selection over a fixed generic grasp detector.*

---

## Architecture

```
Input (RGB + Depth + Text Query)
  │
  ├── Stage 1: Florence-2 grounds the target → bbox + optional mask
  ├── Stage 2: Fixed grasp detector on full-scene point cloud → top-K candidates
  ├── Stage 3: Each candidate → 9-dim semantic-geometric feature vector
  └── Stage 4: Lightweight reranker scores candidates
  │
  Output: Ranked target-aware grasps + top-1 grasp pose
```

## Data Requirements

| File | Required | Size |
|------|----------|------|
| `train_1.zip` – `train_4.zip` | Yes (training) | ~30 GB |
| `test_seen.zip`, `test_similar.zip`, `test_novel.zip` | Yes | ~7 GB |
| `grasp_label.zip` | Yes | ~2 GB |
| `collision_label.zip` | Yes | ~1 GB |
| `models.zip` | Yes | ~1 GB |
| `dex_models.zip` | Optional (faster eval) | ~1 GB |

Source: [graspnet.net](https://graspnet.net/datasets.html)

---

## Quick Start

### 1. Environment Setup

```bash
git clone https://github.com/langchengg/VLMGraspPose.git
cd VLMGraspPose

python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Download Data & Models

```bash
# Download GraspNet data to data/raw/graspnet/
python scripts/download_data.py --all

# Download Florence-2-large-ft and GraspNet checkpoint
python scripts/download_weights.py --all
```

### 3. Run the 12-Step Pipeline

```bash
# Step 1: Build view-level index
python scripts/step01_build_index.py

# Step 2: Generate text queries
python scripts/step02_create_queries.py

# Step 3: Build oracle (GT) target annotations
python scripts/step03_oracle_targets.py

# Step 4: Run Florence-2 grounding (requires GPU)
python scripts/step04_florence_grounding.py --splits test_seen

# Step 5: Convert depth to point clouds
python scripts/step05_depth_to_pcd.py

# Step 6: Generate grasp candidates (full-scene)
python scripts/step06_grasp_candidates.py

# Step 7: Build target-aware training labels (train/val only)
python scripts/step07_build_labels.py --splits train val

# Step 8: Extract candidate features
python scripts/step08_extract_features.py

# Step 9: Train reranker
python scripts/step09_train_reranker.py --model mlp

# Step 10: Full inference on test sets
python scripts/step10_inference.py --splits test_seen test_similar test_novel

# Step 11: Evaluate
python scripts/step11_evaluate.py
```

---

## Project Layout

```
project/
├── config.py                    # Central configuration
├── src/                         # Core library
│   ├── data_utils.py            #   Scene loading, image/depth/label loaders
│   ├── point_cloud.py           #   3D geometry utilities
│   ├── grounding.py             #   GT + Florence-2 grounders
│   ├── grasp_detector.py        #   GraspNet detector wrapper
│   ├── feature_extractor.py     #   9-dim candidate features
│   ├── label_builder.py         #   Target-aware label generation
│   ├── reranker.py              #   Rule / Logistic / MLP / Pairwise rerankers
│   └── evaluation.py            #   Metrics (AP, Success@K, breakdowns)
├── scripts/                     # CLI step scripts
│   ├── step01–step11            #   Pipeline steps
│   ├── download_data.py         #   GraspNet data download
│   └── download_weights.py      #   Model weight download
├── data/
│   ├── raw/graspnet/            #   Official GraspNet layout (gitignored)
│   ├── splits/                  #   View-level JSONL indexes
│   └── metadata/                #   Object names, query templates
├── derived/                     #   Pipeline intermediate outputs
├── models/                      #   Downloaded/trained weights
├── results/                     #   Predictions and metrics
└── vis/                         #   Visualization utilities
```

---

## Reranker Variants

| Model | Description | Training |
|-------|-------------|----------|
| `detector` | Raw detector score (no reranking) | None |
| `rule` | Deterministic weighted-sum | None |
| `logistic` | Logistic regression | Fast |
| `mlp` | 2-layer MLP (main model) | ~1 min |
| `pairwise` | Pairwise MLP (strongest) | ~5 min |

---

## Evaluation Metrics

- **Target Success@1 / @5** — is the grasped object the target?
- **Target-ranking Precision@K / AP** — how well does the reranker place target grasps at the top? (uses `is_on_target`, not physical grasp quality)
- **Per-split breakdowns** — seen / similar / novel objects
- **Ablation** — oracle vs predicted grounding

---

## License

MIT
