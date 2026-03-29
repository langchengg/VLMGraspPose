"""
stage4/label_generator.py — Generate training labels for candidates
====================================================================
Two modes:
  --mode pseudo   Rule-based pseudo labels (for demo / sanity-check)
  --mode gt       Official GraspNet labels (for formal experiments)

IMPORTANT: Only use data from train_* splits for label generation.
           Never label test_seen data for training — that is data leakage.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config


def generate_pseudo_labels(
    features: np.ndarray,
    candidates: list,
    target_mask: Optional[np.ndarray],
    intrinsics: np.ndarray,
    grasp_score_thresh: float = config.LABEL_GRASP_SCORE_THRESH,
    collision_thresh: float = config.LABEL_COLLISION_THRESH,
) -> np.ndarray:
    """Generate rule-based pseudo labels.

    Positive (1) if ALL of:
      • candidate centre projects inside target mask/bbox  (f2 == 1)
      • raw grasp score ≥ threshold                        (f1 ≥ thresh)

    Otherwise negative (0).

    Parameters
    ----------
    features : (N, >=5) — columns [f1, f2, f3, f4, f5, ...]
    """
    N = len(features)
    labels = np.zeros(N, dtype=np.int32)

    for i in range(N):
        f1 = features[i, 0]  # grasp score
        f2 = features[i, 1]  # in target region

        is_positive = (f2 >= 0.5) and (f1 >= grasp_score_thresh)

        # If extended features available, also check collision risk
        if features.shape[1] >= 7:
            f7 = features[i, 6]   # collision risk
            is_positive = is_positive and (f7 < collision_thresh)

        labels[i] = 1 if is_positive else 0

    return labels


def save_ranking_data(
    sample_id: str,
    features: np.ndarray,
    labels: np.ndarray,
    candidate_ids: List[int],
    split: str = "train",
    output_dir: Path = config.RANKING_DATA_DIR,
    label_type: str = "pseudo",
) -> Path:
    """Append labelled samples to ranking data JSONL."""
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{split}_rank.jsonl"

    with open(out_path, "a") as f:
        for i, cid in enumerate(candidate_ids):
            record = {
                "sample_id": sample_id,
                "candidate_id": cid,
                "features": features[i].tolist(),
                "label": int(labels[i]),
                "label_type": label_type,
            }
            f.write(json.dumps(record) + "\n")

    return out_path


def load_ranking_data(
    split: str = "train",
    input_dir: Path = config.RANKING_DATA_DIR,
) -> Tuple[np.ndarray, np.ndarray]:
    """Load ranking data JSONL → (X, y) arrays."""
    path = input_dir / f"{split}_rank.jsonl"
    if not path.exists():
        return np.zeros((0, config.FEATURE_DIM_CORE)), np.zeros(0, dtype=np.int32)

    X_list, y_list = [], []
    with open(path) as f:
        for line in f:
            record = json.loads(line)
            X_list.append(record["features"])
            y_list.append(record["label"])

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int32)
    return X, y
