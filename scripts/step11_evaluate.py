"""
scripts/step11_evaluate.py — Evaluate the system
===================================================
Step 11: Compute metrics, breakdowns, ablations.

Usage:
    python scripts/step11_evaluate.py --grounder phrase --reranker rule
    python scripts/step11_evaluate.py --ablation --reranker rule
"""

import argparse
import json
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from src.evaluation import (
    full_evaluation,
    ablation_grounding,
)


def find_prediction_files(
    splits: list,
    grounder: str = None,
    reranker: str = None,
) -> list:
    """Find prediction files matching the given grounder/reranker.

    File naming convention:
        predictions_{split}_{grounder}_{reranker}.json
    """
    found_files = []
    for split in splits:
        # Try exact match first
        if grounder and reranker:
            path = config.RESULTS_DIR / f"predictions_{split}_{grounder}_{reranker}.json"
            if path.exists():
                found_files.append(path)
                continue

        # Fallback: search for any matching pattern
        pattern = f"predictions_{split}_*.json"
        candidates = sorted(config.RESULTS_DIR.glob(pattern))
        if grounder:
            candidates = [c for c in candidates if f"_{grounder}_" in c.name]
        if reranker:
            candidates = [c for c in candidates if c.name.endswith(f"_{reranker}.json")]
        found_files.extend(candidates)

    return found_files


def evaluate(
    splits: list = None,
    grounder: str = None,
    reranker: str = None,
):
    """Run evaluation on test prediction files."""
    if splits is None:
        splits = config.TEST_SPLITS

    pred_files = find_prediction_files(splits, grounder, reranker)

    if not pred_files:
        print("[ERROR] No prediction files found.")
        print(f"  Looked in: {config.RESULTS_DIR}")
        if grounder:
            print(f"  Grounder filter: {grounder}")
        if reranker:
            print(f"  Reranker filter: {reranker}")
        print("  Run step10 first.")
        return

    all_predictions = []
    for pf in pred_files:
        with open(pf) as f:
            preds = json.load(f)
        all_predictions.extend(preds)
        print(f"  Loaded {len(preds)} predictions from {pf.name}")

    if not all_predictions:
        print("[ERROR] No predictions found.")
        return

    # Build output name
    tag = ""
    if grounder:
        tag += f"_{grounder}"
    if reranker:
        tag += f"_{reranker}"
    out_path = config.RESULTS_DIR / f"evaluation_report{tag}.json"
    full_evaluation(all_predictions, output_path=out_path)


def run_ablation(reranker: str = "rule"):
    """Compare oracle (GT) vs predicted (Florence-2) grounding.

    Looks for:
        predictions_{split}_gt_{reranker}.json       (oracle)
        predictions_{split}_phrase_{reranker}.json   (predicted)
    """
    oracle_preds = []
    predicted_preds = []

    for split in config.TEST_SPLITS:
        # Oracle predictions (grounder=gt)
        oracle_path = config.RESULTS_DIR / f"predictions_{split}_gt_{reranker}.json"
        if oracle_path.exists():
            with open(oracle_path) as f:
                oracle_preds.extend(json.load(f))

        # Predicted predictions (grounder=phrase)
        pred_path = config.RESULTS_DIR / f"predictions_{split}_phrase_{reranker}.json"
        if pred_path.exists():
            with open(pred_path) as f:
                predicted_preds.extend(json.load(f))

        # Also try grounder=seg
        seg_path = config.RESULTS_DIR / f"predictions_{split}_seg_{reranker}.json"
        if seg_path.exists():
            with open(seg_path) as f:
                predicted_preds.extend(json.load(f))

    if not oracle_preds and not predicted_preds:
        print("[ERROR] No predictions found for ablation.")
        print(f"  Expected files like: predictions_*_gt_{reranker}.json")
        print(f"                   and: predictions_*_phrase_{reranker}.json")
        print("  Run step10 with both --grounder gt and --grounder phrase")
        return

    if oracle_preds and predicted_preds:
        result = ablation_grounding(oracle_preds, predicted_preds)

        print(f"\n{'═' * 60}")
        print(f"  Grounding Ablation  (reranker={reranker})")
        print(f"{'═' * 60}")
        print(f"  Oracle S@1:    {result['oracle_grounding']['target_success_at_1']:.4f}")
        print(f"  Predicted S@1: {result['predicted_grounding']['target_success_at_1']:.4f}")
        print(f"  Δ S@1:         {result['delta_success_at_1']:.4f}")
        print(f"  Oracle S@5:    {result['oracle_grounding']['target_success_at_5']:.4f}")
        print(f"  Predicted S@5: {result['predicted_grounding']['target_success_at_5']:.4f}")
        print(f"  Δ S@5:         {result['delta_success_at_5']:.4f}")
        print(f"{'═' * 60}")

        # Save
        out_path = config.RESULTS_DIR / f"ablation_grounding_{reranker}.json"
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"  Saved → {out_path}")
    else:
        print("[WARN] Need both oracle and predicted predictions for ablation")
        if oracle_preds:
            print(f"  Oracle:    {len(oracle_preds)} predictions")
            full_evaluation(oracle_preds,
                            config.RESULTS_DIR / f"eval_oracle_{reranker}.json")
        if predicted_preds:
            print(f"  Predicted: {len(predicted_preds)} predictions")
            full_evaluation(predicted_preds,
                            config.RESULTS_DIR / f"eval_predicted_{reranker}.json")


def main():
    parser = argparse.ArgumentParser(
        description="Step 11: Evaluate the system"
    )
    parser.add_argument("--splits", nargs="+", default=None)
    parser.add_argument("--grounder", type=str, default=None,
                        help="Filter by grounder (gt, phrase, seg)")
    parser.add_argument("--reranker", type=str, default=None,
                        help="Filter by reranker (detector, rule, logistic, mlp, pairwise)")
    parser.add_argument("--ablation", action="store_true",
                        help="Run oracle vs predicted grounding ablation")
    args = parser.parse_args()

    if args.ablation:
        run_ablation(reranker=args.reranker or "rule")
    else:
        evaluate(
            splits=args.splits,
            grounder=args.grounder,
            reranker=args.reranker,
        )


if __name__ == "__main__":
    main()
