"""
scripts/step09_train_reranker.py — Train the reranker
=======================================================
Step 9: Train models that turn generic grasp candidates into
language-conditioned target grasps.

Usage:
    python scripts/step09_train_reranker.py --model logistic
    python scripts/step09_train_reranker.py --model mlp
    python scripts/step09_train_reranker.py --model pairwise
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from src.reranker import (
    LogisticReranker, MLPReranker, PairwiseMLPReranker,
)


def _find_feature_file(split: str, grounding: str = None) -> Path:
    """Find the feature parquet for a split.

    step08 now writes:
      - oracle:    {split}_oracle_features.parquet
      - predicted: {split}_predicted_{task}_features.parquet  (new)
                   {split}_predicted_features.parquet          (old)

    Args:
        split: split name (train, val, ...)
        grounding: explicit grounding mode ('oracle' or 'predicted').
                   If None, auto-detect with preference: predicted > oracle.
    """
    if grounding == "oracle":
        path = config.RANK_FEATURES_DIR / f"{split}_oracle_features.parquet"
        if path.exists():
            return path
        print(f"  [WARN] Requested oracle features not found: {path.name}")
        return None

    if grounding == "predicted":
        # Try task-specific new naming first (phrase > seg)
        for task in ["phrase", "seg"]:
            path = config.RANK_FEATURES_DIR / f"{split}_predicted_{task}_features.parquet"
            if path.exists():
                return path
        # Fallback to old predicted naming
        path = config.RANK_FEATURES_DIR / f"{split}_predicted_features.parquet"
        if path.exists():
            return path
        print(f"  [WARN] Requested predicted features not found in {config.RANK_FEATURES_DIR}")
        return None

    # Auto-detect: prefer predicted (matches default inference grounder=phrase)
    for pattern in [
        f"{split}_predicted_phrase_features.parquet",
        f"{split}_predicted_seg_features.parquet",
        f"{split}_predicted_features.parquet",
        f"{split}_oracle_features.parquet",
        f"{split}_features.parquet",  # legacy
    ]:
        path = config.RANK_FEATURES_DIR / pattern
        if path.exists():
            return path
    return None


def load_train_val_data(grounding: str = None):
    """Load features and labels from parquet files.

    Args:
        grounding: 'oracle' or 'predicted'. If None, auto-detect.

    Returns:
        X_train, y_train, sample_ids_train, X_val, y_val
        sample_ids_train comes from the MERGED table (features ∩ labels),
        guaranteeing exact row-correspondence with X_train/y_train.
    """
    # Training data
    feat_path = _find_feature_file("train", grounding)
    label_path = config.RANK_LABELS_DIR / "train_labels.parquet"

    if feat_path is None or not label_path.exists():
        print("[ERROR] Training data not found.")
        print(f"  Features dir: {config.RANK_FEATURES_DIR}")
        print(f"  Labels:       {label_path}")
        print("  Run step07 and step08 first.")
        return None, None, None, None, None

    print(f"  Loading features: {feat_path.name}")
    feat_df = pd.read_parquet(feat_path)
    label_df = pd.read_parquet(label_path)

    # Merge on sample_id + candidate_id
    merged = feat_df.merge(
        label_df[["sample_id", "candidate_id", "label"]],
        on=["sample_id", "candidate_id"],
        how="inner",
    )

    feature_cols = config.FEATURE_NAMES
    X_train = merged[feature_cols].values.astype(np.float32)
    y_train = merged["label"].values.astype(np.int32)
    sample_ids_train = merged["sample_id"].values

    # Validation data
    val_feat_path = _find_feature_file("val", grounding)
    val_label = config.RANK_LABELS_DIR / "val_labels.parquet"

    X_val, y_val = None, None
    if val_feat_path is not None and val_label.exists():
        print(f"  Loading val features: {val_feat_path.name}")
        vf = pd.read_parquet(val_feat_path)
        vl = pd.read_parquet(val_label)
        vm = vf.merge(
            vl[["sample_id", "candidate_id", "label"]],
            on=["sample_id", "candidate_id"],
            how="inner",
        )
        X_val = vm[feature_cols].values.astype(np.float32)
        y_val = vm["label"].values.astype(np.int32)

    return X_train, y_train, sample_ids_train, X_val, y_val


def train_reranker(model_name: str = "logistic", grounding: str = None):
    """Train a reranker model.

    Args:
        grounding: 'oracle' or 'predicted'. Determines which feature set
                   to train on. Should match the grounder used at inference
                   (default: phrase → predicted) to avoid train/test
                   distribution mismatch.
    """
    print(f"{'=' * 60}")
    print(f"  Training reranker: {model_name}")
    print(f"{'=' * 60}")

    X_train, y_train, sample_ids_train, X_val, y_val = load_train_val_data(grounding)
    if X_train is None:
        return

    # Warn about potential train/test mismatch
    train_file = _find_feature_file("train", grounding)
    if train_file and "_oracle_" in train_file.name:
        print("  [WARN] Training on oracle features. Default inference uses")
        print("         grounder=phrase (predicted features). f5/f6/f9 may")
        print("         have different distributions at test time. Consider:")
        print("           --grounding predicted")
        print("         or run step10 with --grounder gt.")

    n_pos = int(y_train.sum())
    n_neg = len(y_train) - n_pos
    print(f"  Train: {len(X_train)} samples ({n_pos} pos, {n_neg} neg)")
    if X_val is not None:
        print(f"  Val:   {len(X_val)} samples ({int(y_val.sum())} pos)")

    if model_name == "logistic":
        reranker = LogisticReranker()
        reranker.train(X_train, y_train)
        reranker.save()
        print(f"  Model saved → {config.RERANKER_LOGREG_PATH}")

    elif model_name == "mlp":
        reranker = MLPReranker(feature_dim=len(config.FEATURE_NAMES))
        reranker.train(X_train, y_train)
        reranker.save()
        print(f"  Model saved → {config.RERANKER_MLP_PATH}")

    elif model_name == "pairwise":
        # sample_ids_train comes from the MERGED table, guaranteeing
        # exact row-correspondence with X_train/y_train.
        reranker = PairwiseMLPReranker(feature_dim=len(config.FEATURE_NAMES))
        reranker.train(X_train, y_train, sample_ids=sample_ids_train)
        reranker.save()
        print(f"  Model saved → {config.MODELS_DIR / 'reranker_pairwise.pt'}")

    else:
        print(f"[ERROR] Unknown model: {model_name}")
        return

    # Validate
    if X_val is not None:
        scores = reranker.score(X_val)
        preds = (scores > 0.5).astype(int)
        acc = float((preds == y_val).mean())
        print(f"  Val accuracy: {acc:.4f}")



def main():
    parser = argparse.ArgumentParser(
        description="Step 9: Train the reranker"
    )
    parser.add_argument(
        "--model", type=str, default="logistic",
        choices=["logistic", "mlp", "pairwise"],
    )
    parser.add_argument(
        "--grounding", type=str, default=None,
        choices=["oracle", "predicted"],
        help="Which feature set to train on. Should match the grounder "
             "used at inference. If not set, auto-detects (prefers predicted).",
    )
    args = parser.parse_args()
    train_reranker(model_name=args.model, grounding=args.grounding)


if __name__ == "__main__":
    main()
