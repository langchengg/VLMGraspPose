"""
scripts/step09_train_reranker.py — Train the reranker
=======================================================
Step 9: Train models that turn generic grasp candidates into
language-conditioned target grasps.

Usage:
    python scripts/step09_train_reranker.py
    python scripts/step09_train_reranker.py --grounding predicted --detector geometric
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from src.reranker import MLPReranker


def _find_feature_file(split: str, grounding: str = None,
                        detector: str = config.DEFAULT_DETECTOR) -> Path:
    """Find the feature parquet for a split.

    step08 now writes (newest → oldest naming):
      - {split}_{grounding}_{detector}_features.parquet              (current)
      - {split}_predicted_{task}_{detector}_features.parquet          (current, predicted)
      - {split}_predicted_{task}_features.parquet                     (legacy v2)
      - {split}_{grounding}_features.parquet                         (legacy v1)
      - {split}_features.parquet                                     (legacy v0)

    Args:
        split: split name (train, val, ...)
        grounding: 'oracle' or 'predicted'. If None, auto-detect.
        detector: detector type to match (geometric).
    """
    if grounding == "oracle":
        # Current naming first, then legacy
        for pattern in [
            f"{split}_oracle_{detector}_features.parquet",
            f"{split}_oracle_features.parquet",  # legacy
        ]:
            path = config.RANK_FEATURES_DIR / pattern
            if path.exists():
                return path
        print(f"  [WARN] Requested oracle features not found for detector={detector}")
        return None

    if grounding == "predicted":
        # Current naming (task × detector), then legacy
        for task in ["seg", "phrase"]:
            path = config.RANK_FEATURES_DIR / f"{split}_predicted_{task}_{detector}_features.parquet"
            if path.exists():
                return path
        # Legacy naming (without detector)
        for task in ["seg", "phrase"]:
            path = config.RANK_FEATURES_DIR / f"{split}_predicted_{task}_features.parquet"
            if path.exists():
                print(f"  [WARN] Using legacy feature file (no detector tag): {path.name}")
                return path
        path = config.RANK_FEATURES_DIR / f"{split}_predicted_features.parquet"
        if path.exists():
            return path
        print(f"  [WARN] Requested predicted features not found for detector={detector}")
        return None

    # Auto-detect: prefer predicted-seg with matching detector, then fallbacks
    for pattern in [
        f"{split}_predicted_seg_{detector}_features.parquet",
        f"{split}_predicted_phrase_{detector}_features.parquet",
        f"{split}_oracle_{detector}_features.parquet",
        # Legacy (no detector tag)
        f"{split}_predicted_seg_features.parquet",
        f"{split}_predicted_phrase_features.parquet",
        f"{split}_predicted_features.parquet",
        f"{split}_oracle_features.parquet",
        f"{split}_features.parquet",
    ]:
        path = config.RANK_FEATURES_DIR / pattern
        if path.exists():
            return path
    return None


def _find_label_file(split: str, detector: str = config.DEFAULT_DETECTOR) -> Path:
    """Find the label parquet for a split.

    Searches detector-tagged filename first, then legacy.
    """
    # Current: {split}_{detector}_labels.parquet
    path = config.RANK_LABELS_DIR / f"{split}_{detector}_labels.parquet"
    if path.exists():
        return path
    # Legacy: {split}_labels.parquet
    path = config.RANK_LABELS_DIR / f"{split}_labels.parquet"
    if path.exists():
        print(f"  [WARN] Using legacy label file (no detector tag): {path.name}")
        return path
    return None


def load_train_val_data(grounding: str = None, detector: str = config.DEFAULT_DETECTOR):
    """Load features and labels from parquet files.

    Args:
        grounding: 'oracle' or 'predicted'. If None, auto-detect.
        detector: detector type whose labels/features to load.

    Returns:
        X_train, y_train, sample_ids_train, X_val, y_val
        sample_ids_train comes from the MERGED table (features ∩ labels),
        guaranteeing exact row-correspondence with X_train/y_train.
    """
    # Training data
    feat_path = _find_feature_file("train", grounding, detector)
    label_path = _find_label_file("train", detector)

    if feat_path is None or label_path is None:
        print("[ERROR] Training data not found.")
        print(f"  Features dir: {config.RANK_FEATURES_DIR}")
        print(f"  Labels dir:   {config.RANK_LABELS_DIR}")
        print(f"  Detector:     {detector}")
        print("  Run step07 and step08 first with matching --detector.")
        return None, None, None, None, None

    print(f"  Loading features: {feat_path.name}")
    print(f"  Loading labels:   {label_path.name}")
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
    val_feat_path = _find_feature_file("val", grounding, detector)
    val_label_path = _find_label_file("val", detector)

    X_val, y_val = None, None
    if val_feat_path is not None and val_label_path is not None:
        print(f"  Loading val features: {val_feat_path.name}")
        vf = pd.read_parquet(val_feat_path)
        vl = pd.read_parquet(val_label_path)
        vm = vf.merge(
            vl[["sample_id", "candidate_id", "label"]],
            on=["sample_id", "candidate_id"],
            how="inner",
        )
        X_val = vm[feature_cols].values.astype(np.float32)
        y_val = vm["label"].values.astype(np.int32)

    return X_train, y_train, sample_ids_train, X_val, y_val


def _model_save_path(model_name: str, detector: str, grounding: str) -> Path:
    """Generate a unique model save path including detector and grounding tags.

    e.g. models/reranker_mlp_geometric_predicted.pt
    """
    grounding_tag = grounding or "auto"
    if model_name == "mlp":
        return config.MODELS_DIR / f"reranker_mlp_{detector}_{grounding_tag}.pt"
    return config.MODELS_DIR / f"reranker_{model_name}_{detector}_{grounding_tag}.pt"


def train_reranker(model_name: str = config.DEFAULT_RERANKER, grounding: str = None,
                   detector: str = config.DEFAULT_DETECTOR):
    """Train a reranker model.

    Args:
        grounding: 'oracle' or 'predicted'. Determines which feature set
                   to train on. Should match the grounder used at inference
                   (default: seg → predicted) to avoid train/test
                   distribution mismatch.
        detector: detector type whose features/labels to use. Must match
                  step06/step07/step08.
    """
    print(f"{'=' * 60}")
    print(f"  Training reranker: {model_name}")
    print(f"  Detector: {detector}  |  Grounding: {grounding or 'auto-detect'}")
    print(f"{'=' * 60}")

    X_train, y_train, sample_ids_train, X_val, y_val = load_train_val_data(
        grounding, detector,
    )
    if X_train is None:
        return

    # Warn about potential train/test mismatch
    train_file = _find_feature_file("train", grounding, detector)
    if train_file and "_oracle_" in train_file.name:
        print("  [WARN] Training on oracle features. Default inference uses")
        print("         grounder=seg (predicted features). Target geometry may")
        print("         have a different distribution at test time. Consider:")
        print("           --grounding predicted")
        print("         or run step10 with --grounder gt.")

    n_pos = int(y_train.sum())
    n_neg = len(y_train) - n_pos
    print(f"  Train: {len(X_train)} samples ({n_pos} pos, {n_neg} neg)")
    if X_val is not None:
        print(f"  Val:   {len(X_val)} samples ({int(y_val.sum())} pos)")

    save_path = _model_save_path(model_name, detector, grounding)

    if model_name == "mlp":
        reranker = MLPReranker(feature_dim=len(config.FEATURE_NAMES))
        reranker.train(X_train, y_train)
        reranker.save(save_path)
        print(f"  Model saved → {save_path}")

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
        "--model", type=str, default=config.DEFAULT_RERANKER,
        choices=["mlp"],
    )
    parser.add_argument(
        "--grounding", type=str, default=None,
        choices=["oracle", "predicted"],
        help="Which feature set to train on. Should match the grounder "
             "used at inference. If not set, auto-detects (prefers predicted).",
    )
    parser.add_argument(
        "--detector", type=str, default=config.DEFAULT_DETECTOR,
        choices=["geometric"],
        help="Which detector's features/labels to use (default: geometric).",
    )
    args = parser.parse_args()
    train_reranker(
        model_name=args.model,
        grounding=args.grounding,
        detector=args.detector,
    )


if __name__ == "__main__":
    main()
