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
    detector: str = None,
) -> list:
    """Find prediction files matching the given grounder/reranker/detector.

    File naming conventions (searches both):
        New:    predictions_{split}_{grounder}_{reranker}_{detector}.json
        Legacy: predictions_{split}_{grounder}_{reranker}.json

    Returns a list of (path, grounder, reranker) tuples.
    """
    found_files = []
    for split in splits:
        # Try exact match first (new format with detector tag)
        if grounder and reranker:
            if detector:
                path = config.RESULTS_DIR / f"predictions_{split}_{grounder}_{reranker}_{detector}.json"
                if path.exists():
                    found_files.append((path, grounder, reranker))
                    continue
            # Legacy format (no detector tag)
            path = config.RESULTS_DIR / f"predictions_{split}_{grounder}_{reranker}.json"
            if path.exists():
                found_files.append((path, grounder, reranker))
                continue

        # Fallback: search for any matching pattern
        pattern = f"predictions_{split}_*.json"
        candidates = sorted(config.RESULTS_DIR.glob(pattern))
        if grounder:
            candidates = [c for c in candidates if f"_{grounder}_" in c.name]
        if reranker:
            # Match reranker as the 2nd-to-last or last segment before .json
            candidates = [c for c in candidates
                          if f"_{reranker}_" in c.name
                          or c.name.endswith(f"_{reranker}.json")]
        if detector:
            candidates = [c for c in candidates if f"_{detector}.json" in c.name]

        for c in candidates:
            # Parse grounder and reranker from filename
            # New:    predictions_{split}_{grounder}_{reranker}_{detector}.json
            # Legacy: predictions_{split}_{grounder}_{reranker}.json
            stem = c.stem
            rest = stem.replace(f"predictions_{split}_", "")
            parts = rest.split("_")
            if len(parts) >= 2:
                # grounder is first part, reranker is second
                found_files.append((c, parts[0], parts[1]))
            else:
                found_files.append((c, "unknown", "unknown"))

    return found_files


def evaluate(
    splits: list = None,
    grounder: str = None,
    reranker: str = None,
    detector: str = None,
):
    """Run evaluation on test prediction files.

    When no grounder/reranker is specified and multiple configs exist,
    evaluates each configuration separately to avoid mixing results.
    """
    if splits is None:
        splits = config.TEST_SPLITS

    file_entries = find_prediction_files(splits, grounder, reranker, detector)

    if not file_entries:
        print("[ERROR] No prediction files found.")
        print(f"  Looked in: {config.RESULTS_DIR}")
        if grounder:
            print(f"  Grounder filter: {grounder}")
        if reranker:
            print(f"  Reranker filter: {reranker}")
        print("  Run step10 first.")
        return

    # Group by (grounder, reranker) config
    from collections import defaultdict
    config_groups = defaultdict(list)
    for path, g, r in file_entries:
        config_groups[(g, r)].append(path)

    if len(config_groups) > 1 and not (grounder and reranker):
        print(f"  Found {len(config_groups)} configurations — evaluating each separately.")

    for (g, r), paths in sorted(config_groups.items()):
        all_predictions = []
        for pf in paths:
            with open(pf) as f:
                preds = json.load(f)
            all_predictions.extend(preds)
            print(f"  Loaded {len(preds)} predictions from {pf.name}")

        if not all_predictions:
            continue

        tag = f"_{g}_{r}"
        out_path = config.RESULTS_DIR / f"evaluation_report{tag}.json"
        print(f"  → Evaluating: grounder={g}, reranker={r}, N={len(all_predictions)}")
        full_evaluation(all_predictions, output_path=out_path)


def run_ablation(reranker: str = config.DEFAULT_RERANKER):
    """Compare oracle (GT) vs predicted (Florence-2) grounding.

    Evaluates each predicted grounder (phrase, seg) separately
    to avoid mixing their results.
    """
    oracle_preds = []
    predicted_by_grounder = {}  # {"phrase": [...], "seg": [...]}

    for split in config.TEST_SPLITS:
        # Oracle predictions (grounder=gt) — search new + legacy filenames
        for pattern in [
            f"predictions_{split}_gt_{reranker}_*.json",  # new (with detector)
            f"predictions_{split}_gt_{reranker}.json",     # legacy
        ]:
            for oracle_path in sorted(config.RESULTS_DIR.glob(pattern)):
                with open(oracle_path) as f:
                    oracle_preds.extend(json.load(f))
                break  # take first match per pattern group

        # Predicted predictions — load each grounder separately
        for grounder in ["phrase", "seg"]:
            for pattern in [
                f"predictions_{split}_{grounder}_{reranker}_*.json",
                f"predictions_{split}_{grounder}_{reranker}.json",
            ]:
                for pred_path in sorted(config.RESULTS_DIR.glob(pattern)):
                    with open(pred_path) as f:
                        preds = json.load(f)
                    predicted_by_grounder.setdefault(grounder, []).extend(preds)
                    break

    if not oracle_preds and not predicted_by_grounder:
        print("[ERROR] No predictions found for ablation.")
        print(f"  Expected files like: predictions_*_gt_{reranker}_*.json")
        print(f"                   and: predictions_*_phrase_{reranker}_*.json")
        print("  Run step10 with both --grounder gt and --grounder phrase")
        return

    # Ablation for each predicted grounder vs oracle
    for grounder, preds in predicted_by_grounder.items():
        if oracle_preds and preds:
            result = ablation_grounding(oracle_preds, preds)

            n_paired = result["paired_samples"]
            n_oracle_only = result["oracle_only_samples"]
            n_pred_only = result["predicted_only_samples"]

            print(f"\n{'═' * 60}")
            print(f"  Grounding Ablation: GT vs {grounder}  (reranker={reranker})")
            print(f"{'═' * 60}")
            print(f"  Paired samples:  {n_paired}")
            if n_oracle_only or n_pred_only:
                print(f"    oracle-only:   {n_oracle_only}")
                print(f"    {grounder}-only: {n_pred_only}")
            oracle_fail = result['paired_oracle'].get('failure_rate', 0)
            pred_fail = result['paired_predicted'].get('failure_rate', 0)
            if oracle_fail or pred_fail:
                print(f"  Failure rates:   oracle={oracle_fail:.1%}  {grounder}={pred_fail:.1%}")
            print(f"  Oracle S@1:    {result['oracle_grounding']['target_success_at_1']:.4f}")
            print(f"  {grounder:8s} S@1: {result['predicted_grounding']['target_success_at_1']:.4f}")
            print(f"  Δ S@1:         {result['delta_success_at_1']:.4f}")
            print(f"  Oracle S@5:    {result['oracle_grounding']['target_success_at_5']:.4f}")
            print(f"  {grounder:8s} S@5: {result['predicted_grounding']['target_success_at_5']:.4f}")
            print(f"  Δ S@5:         {result['delta_success_at_5']:.4f}")
            print(f"{'═' * 60}")

            # Save per-grounder ablation
            out_path = config.RESULTS_DIR / f"ablation_{grounder}_vs_gt_{reranker}.json"
            with open(out_path, "w") as f:
                json.dump(result, f, indent=2)
            print(f"  Saved → {out_path}")
        else:
            # Only one side available — just run full eval
            if preds:
                print(f"\n  {grounder}: {len(preds)} predictions (no oracle for comparison)")
                full_evaluation(preds,
                                config.RESULTS_DIR / f"eval_{grounder}_{reranker}.json")

    if oracle_preds and not predicted_by_grounder:
        print(f"\n  Oracle: {len(oracle_preds)} predictions (no predicted for comparison)")
        full_evaluation(oracle_preds,
                        config.RESULTS_DIR / f"eval_oracle_{reranker}.json")


def main():
    parser = argparse.ArgumentParser(
        description="Step 11: Evaluate the system"
    )
    parser.add_argument("--splits", nargs="+", default=None)
    parser.add_argument("--grounder", type=str, default=None,
                        help="Filter by grounder (gt, phrase, seg)")
    parser.add_argument("--reranker", type=str, default=None,
                        help="Filter by reranker (detector, rule, mlp)")
    parser.add_argument("--detector", type=str, default=None,
                        help="Filter by detector (geometric)")
    parser.add_argument("--ablation", action="store_true",
                        help="Run oracle vs predicted grounding ablation")
    args = parser.parse_args()

    if args.ablation:
        run_ablation(reranker=args.reranker or config.DEFAULT_RERANKER)
    else:
        evaluate(
            splits=args.splits,
            grounder=args.grounder,
            reranker=args.reranker,
            detector=args.detector,
        )


if __name__ == "__main__":
    main()
