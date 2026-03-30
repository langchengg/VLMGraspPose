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


def load_train_val_data():
    """Load features and labels from parquet files."""
    # Training data
    feat_path = config.RANK_FEATURES_DIR / "train_features.parquet"
    label_path = config.RANK_LABELS_DIR / "train_labels.parquet"

    if not feat_path.exists() or not label_path.exists():
        print("[ERROR] Training data not found.")
        print(f"  Features: {feat_path}")
        print(f"  Labels:   {label_path}")
        print("  Run step07 and step08 first.")
        return None, None, None, None

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

    # Validation data
    val_feat = config.RANK_FEATURES_DIR / "val_features.parquet"
    val_label = config.RANK_LABELS_DIR / "val_labels.parquet"

    X_val, y_val = None, None
    if val_feat.exists() and val_label.exists():
        vf = pd.read_parquet(val_feat)
        vl = pd.read_parquet(val_label)
        vm = vf.merge(
            vl[["sample_id", "candidate_id", "label"]],
            on=["sample_id", "candidate_id"],
            how="inner",
        )
        X_val = vm[feature_cols].values.astype(np.float32)
        y_val = vm["label"].values.astype(np.int32)

    return X_train, y_train, X_val, y_val


def train_reranker(model_name: str = "logistic"):
    """Train a reranker model."""
    print(f"{'=' * 60}")
    print(f"  Training reranker: {model_name}")
    print(f"{'=' * 60}")

    X_train, y_train, X_val, y_val = load_train_val_data()
    if X_train is None:
        return

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
        # Load sample_ids for query-wise pair construction
        sample_ids = _load_sample_ids("train")
        reranker = PairwiseMLPReranker(feature_dim=len(config.FEATURE_NAMES))
        reranker.train(X_train, y_train, sample_ids=sample_ids)
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


def _load_sample_ids(split: str):
    """Load sample_ids from the training feature parquet for pair grouping."""
    import pandas as pd
    # Try grounding-aware filename first
    for grounding in ["oracle", "predicted"]:
        path = config.RANK_FEATURES_DIR / f"{split}_{grounding}_features.parquet"
        if path.exists():
            df = pd.read_parquet(path, columns=["sample_id"])
            return df["sample_id"].values
    # Fallback to old naming
    path = config.RANK_FEATURES_DIR / f"{split}_features.parquet"
    if path.exists():
        df = pd.read_parquet(path, columns=["sample_id"])
        return df["sample_id"].values
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Step 9: Train the reranker"
    )
    parser.add_argument(
        "--model", type=str, default="logistic",
        choices=["logistic", "mlp", "pairwise"],
    )
    args = parser.parse_args()
    train_reranker(model_name=args.model)


if __name__ == "__main__":
    main()
