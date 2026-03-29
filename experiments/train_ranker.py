"""
experiments/train_ranker.py — Train scoring models
====================================================
Usage:
    python -m experiments.train_ranker --mode pseudo --scorer logistic
    python -m experiments.train_ranker --mode pseudo --scorer mlp
    python -m experiments.train_ranker --mode gt     --scorer mlp

IMPORTANT: Only train on data from train_* splits.
           test_seen is for evaluation only.
"""

import argparse
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from stage4.label_generator import load_ranking_data


def train_scorer(mode: str = "pseudo", scorer_type: str = "logistic"):
    print(f"{'='*60}")
    print(f"Training {scorer_type} scorer (mode={mode})")
    print(f"{'='*60}")

    # Load training data
    X_train, y_train = load_ranking_data("train")
    X_val, y_val = load_ranking_data("val")

    if len(X_train) == 0:
        print("[ERROR] No training data found in ranking_data/train_rank.jsonl")
        print("        Make sure to run the pipeline on train_* data first,")
        print("        then generate labels with label_generator.py")
        return

    print(f"  Train: {len(X_train)} samples, {y_train.sum()} positive")
    if len(X_val) > 0:
        print(f"  Val:   {len(X_val)} samples, {y_val.sum()} positive")

    if scorer_type == "logistic":
        from stage4.logistic_scorer import LogisticScorer
        scorer = LogisticScorer()
        scorer.train(X_train, y_train)
        scorer.save()
        print(f"  Model saved → {config.MODELS_DIR / 'scorer_logreg.pkl'}")

    elif scorer_type == "mlp":
        from stage4.mlp_scorer import MLPScorer
        feature_dim = X_train.shape[1]
        scorer = MLPScorer(feature_dim=feature_dim)
        scorer.train(X_train, y_train)
        scorer.save()
        print(f"  Model saved → {config.MODELS_DIR / 'scorer_mlp.pt'}")

    else:
        print(f"[ERROR] Unknown scorer type: {scorer_type}")
        return

    # Evaluate on validation set if available
    if len(X_val) > 0:
        scores = scorer.score(X_val)
        preds = (scores > 0.5).astype(int)
        acc = (preds == y_val).mean()
        print(f"  Val accuracy: {acc:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Train grasp scorer")
    parser.add_argument("--mode", type=str, default="pseudo",
                        choices=["pseudo", "gt"])
    parser.add_argument("--scorer", type=str, default="logistic",
                        choices=["logistic", "mlp"])
    args = parser.parse_args()

    train_scorer(mode=args.mode, scorer_type=args.scorer)


if __name__ == "__main__":
    main()
