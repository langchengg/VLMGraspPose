"""
scripts/step11_evaluate.py — Evaluate the system
===================================================
Step 11: Compute metrics, breakdowns, ablations.

Usage:
    python scripts/step11_evaluate.py --splits test_seen test_similar test_novel
    python scripts/step11_evaluate.py --ablation
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


def evaluate(splits: list = None):
    """Run evaluation on test prediction files."""
    if splits is None:
        splits = config.TEST_SPLITS

    all_predictions = []

    for split in splits:
        pred_path = config.RESULTS_DIR / f"predictions_{split}.json"
        if not pred_path.exists():
            print(f"  [SKIP] {pred_path} not found (run step10 first)")
            continue

        with open(pred_path) as f:
            preds = json.load(f)
        all_predictions.extend(preds)
        print(f"  Loaded {len(preds)} predictions from {split}")

    if not all_predictions:
        print("[ERROR] No predictions found. Run step10 first.")
        return

    # Full evaluation with per-split breakdown
    out_path = config.RESULTS_DIR / "evaluation_report.json"
    full_evaluation(all_predictions, output_path=out_path)


def run_ablation():
    """Compare oracle (GT) vs predicted (Florence-2) grounding."""
    # Look for prediction files with different grounders
    oracle_preds = []
    predicted_preds = []

    for split in config.TEST_SPLITS:
        # Oracle predictions (grounder=gt)
        oracle_path = config.RESULTS_DIR / f"predictions_{split}_oracle.json"
        if oracle_path.exists():
            with open(oracle_path) as f:
                oracle_preds.extend(json.load(f))

        # Predicted predictions (grounder=phrase/seg)
        pred_path = config.RESULTS_DIR / f"predictions_{split}.json"
        if pred_path.exists():
            with open(pred_path) as f:
                preds = json.load(f)
                # Separate by grounder type
                for p in preds:
                    if p.get("grounder") == "gt":
                        oracle_preds.append(p)
                    else:
                        predicted_preds.append(p)

    if not oracle_preds and not predicted_preds:
        print("[ERROR] No predictions found for ablation.")
        print("  Run step10 with both --grounder gt and --grounder phrase")
        return

    if oracle_preds and predicted_preds:
        result = ablation_grounding(oracle_preds, predicted_preds)

        print(f"\n{'═' * 60}")
        print(f"  Grounding Ablation")
        print(f"{'═' * 60}")
        print(f"  Oracle S@1:    {result['oracle_grounding']['target_success_at_1']:.4f}")
        print(f"  Predicted S@1: {result['predicted_grounding']['target_success_at_1']:.4f}")
        print(f"  Δ S@1:         {result['delta_success_at_1']:.4f}")
        print(f"  Oracle S@5:    {result['oracle_grounding']['target_success_at_5']:.4f}")
        print(f"  Predicted S@5: {result['predicted_grounding']['target_success_at_5']:.4f}")
        print(f"  Δ S@5:         {result['delta_success_at_5']:.4f}")
        print(f"{'═' * 60}")

        # Save
        out_path = config.RESULTS_DIR / "ablation_grounding.json"
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"  Saved → {out_path}")
    else:
        print("[WARN] Need both oracle and predicted predictions for ablation")
        if oracle_preds:
            print(f"  Oracle:    {len(oracle_preds)} predictions")
            full_evaluation(oracle_preds,
                            config.RESULTS_DIR / "eval_oracle.json")
        if predicted_preds:
            print(f"  Predicted: {len(predicted_preds)} predictions")
            full_evaluation(predicted_preds,
                            config.RESULTS_DIR / "eval_predicted.json")


def main():
    parser = argparse.ArgumentParser(
        description="Step 11: Evaluate the system"
    )
    parser.add_argument("--splits", nargs="+", default=None)
    parser.add_argument("--ablation", action="store_true",
                        help="Run oracle vs predicted grounding ablation")
    args = parser.parse_args()

    if args.ablation:
        run_ablation()
    else:
        evaluate(splits=args.splits)


if __name__ == "__main__":
    main()
