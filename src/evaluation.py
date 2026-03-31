"""
src/evaluation.py — Evaluation metrics (Step 11)
===================================================
Rewritten from experiments/eval.py.

Metrics:
  1. Target Success@1 / Success@5 — is the target object grasped?
  2. Target-ranking Precision@K / AP — quality of the reranker's
     target-object ordering (using `is_on_target` from GT labels,
     NOT physical grasp quality like force-closure or collision-free)
  3. Per-split breakdowns: seen / similar / novel
  4. Ablation: oracle grounding vs predicted grounding
  5. Latency statistics
"""

import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config


# ═════════════════════════════════════════════════════════════════════
#  Target-aware metrics
# ═════════════════════════════════════════════════════════════════════

def target_success_at_k(
    predictions: List[dict],
    k: int = 1,
) -> float:
    """Compute Target Success@K.

    A prediction is successful if any of the top-K ranked grasps is
    associated with the target object.

    Denominator = len(predictions), which includes failure records
    (ranked_grasps=[]).  Failed queries count as 0, preventing
    systemic upward bias from hard-sample evaporation.

    Each prediction dict should have:
        'ranked_grasps': [{..., 'is_on_target': bool}, ...]
    """
    if not predictions:
        return 0.0

    successes = 0
    for pred in predictions:
        grasps = pred.get("ranked_grasps", [])
        top_k = grasps[:k]
        if any(g.get("is_on_target", False) for g in top_k):
            successes += 1

    return successes / len(predictions)


def compute_target_metrics(predictions: List[dict]) -> dict:
    """Compute all target-aware metrics.

    Reports failure counts so the user can see how many queries
    had no grasps generated (and thus counted as 0).
    """
    num_failed = sum(
        1 for p in predictions
        if not p.get("ranked_grasps")
    )
    return {
        "target_success_at_1": target_success_at_k(predictions, k=1),
        "target_success_at_5": target_success_at_k(predictions, k=5),
        "num_samples": len(predictions),
        "num_failed": num_failed,
        "failure_rate": round(num_failed / max(len(predictions), 1), 4),
    }


# ═════════════════════════════════════════════════════════════════════
#  Target-ranking quality metrics
#
#  NOTE: These are NOT physical grasp quality metrics (e.g. force
#  closure, GraspNet μ-AP).  They measure how well the reranker
#  places target-object grasps at the top of its ranking.
#
#  "Relevant" = grasp is on the target object (is_on_target == True).
# ═════════════════════════════════════════════════════════════════════

def precision_at_k(
    predictions: List[dict],
    k: int = 1,
) -> float:
    """Target-ranking Precision@K.

    Measures: of the top-K grasps, what fraction are on the target object?

    This evaluates the reranker's ability to prioritise target-associated
    grasps, NOT physical grasp quality (collision, stability, etc.).

    Uses the `is_on_target` GT annotation (set by step10 using GT label),
    NOT the reranker's own score — that would be self-referential.
    """
    if not predictions:
        return 0.0

    precisions = []
    for pred in predictions:
        grasps = pred.get("ranked_grasps", [])
        top_k = grasps[:k]
        if not top_k:
            precisions.append(0.0)
            continue
        good = sum(
            1 for g in top_k
            if g.get("is_on_target", False)
        )
        precisions.append(good / len(top_k))

    return float(np.mean(precisions))


def average_precision(
    predictions: List[dict],
) -> float:
    """Target-ranking Average Precision.

    Measures how well the reranker places target-object grasps before
    non-target grasps.  Relevance = is_on_target (from GT labels).

    This is NOT equivalent to GraspNet AP which evaluates grasp quality.
    """
    if not predictions:
        return 0.0

    aps = []
    for pred in predictions:
        grasps = pred.get("ranked_grasps", [])
        if not grasps:
            aps.append(0.0)
            continue

        relevant = 0
        precision_sum = 0.0
        for rank, g in enumerate(grasps, 1):
            is_relevant = g.get("is_on_target", False)
            if is_relevant:
                relevant += 1
                precision_sum += relevant / rank

        ap = precision_sum / max(relevant, 1)
        aps.append(ap)

    return float(np.mean(aps))


# ═════════════════════════════════════════════════════════════════════
#  Per-split breakdown
# ═════════════════════════════════════════════════════════════════════

def evaluate_by_split(
    predictions: List[dict],
) -> Dict[str, dict]:
    """Compute metrics separately for each split."""
    by_split = {}
    for pred in predictions:
        split = pred.get("split", "unknown")
        if split not in by_split:
            by_split[split] = []
        by_split[split].append(pred)

    results = {}
    for split, preds in sorted(by_split.items()):
        results[split] = {
            **compute_target_metrics(preds),
            "precision_at_1": precision_at_k(preds, k=1),
            "precision_at_5": precision_at_k(preds, k=5),
            "average_precision": average_precision(preds),
        }

    return results


# ═════════════════════════════════════════════════════════════════════
#  Oracle vs Predicted grounding ablation
# ═════════════════════════════════════════════════════════════════════

def ablation_grounding(
    oracle_predictions: List[dict],
    predicted_predictions: List[dict],
) -> dict:
    """Compare oracle (GT) grounding vs predicted (Florence-2) grounding.

    Performs a PAIRED comparison on the intersection of sample_ids
    to ensure both methods are evaluated on the exact same queries.
    This prevents bias from hard-sample evaporation (where one method
    fails on hard samples the other succeeds on, or vice versa).
    """
    # Build sample_id → prediction maps
    oracle_map = {p["sample_id"]: p for p in oracle_predictions}
    pred_map = {p["sample_id"]: p for p in predicted_predictions}

    # Paired evaluation: only samples present in BOTH
    common_ids = sorted(set(oracle_map.keys()) & set(pred_map.keys()))
    oracle_paired = [oracle_map[sid] for sid in common_ids]
    pred_paired = [pred_map[sid] for sid in common_ids]

    oracle_only = len(oracle_map) - len(common_ids)
    pred_only = len(pred_map) - len(common_ids)

    paired_oracle_metrics = compute_target_metrics(oracle_paired)
    paired_pred_metrics = compute_target_metrics(pred_paired)

    # Also compute unpaired (all available) for reference
    unpaired_oracle_metrics = compute_target_metrics(oracle_predictions)
    unpaired_pred_metrics = compute_target_metrics(predicted_predictions)

    return {
        # Primary result: paired comparison (fair)
        "paired_oracle": paired_oracle_metrics,
        "paired_predicted": paired_pred_metrics,
        "paired_samples": len(common_ids),
        "delta_success_at_1": (
            paired_oracle_metrics["target_success_at_1"]
            - paired_pred_metrics["target_success_at_1"]
        ),
        "delta_success_at_5": (
            paired_oracle_metrics["target_success_at_5"]
            - paired_pred_metrics["target_success_at_5"]
        ),
        # For reference: unpaired (may differ in sample count)
        "unpaired_oracle": unpaired_oracle_metrics,
        "unpaired_predicted": unpaired_pred_metrics,
        "oracle_only_samples": oracle_only,
        "predicted_only_samples": pred_only,
        # Backward-compat aliases
        "oracle_grounding": paired_oracle_metrics,
        "predicted_grounding": paired_pred_metrics,
    }


# ═════════════════════════════════════════════════════════════════════
#  Latency statistics
# ═════════════════════════════════════════════════════════════════════

def latency_stats(predictions: List[dict]) -> dict:
    """Compute latency statistics."""
    latencies = [
        p.get("latency", 0) for p in predictions if "latency" in p
    ]
    if not latencies:
        return {"avg_latency": 0, "p50_latency": 0, "p95_latency": 0}

    return {
        "avg_latency": float(np.mean(latencies)),
        "p50_latency": float(np.median(latencies)),
        "p95_latency": float(np.percentile(latencies, 95)),
        "total_latency": float(np.sum(latencies)),
    }


# ═════════════════════════════════════════════════════════════════════
#  Full evaluation report
# ═════════════════════════════════════════════════════════════════════

def full_evaluation(
    predictions: List[dict],
    output_path: Optional[Path] = None,
) -> dict:
    """Run complete evaluation and optionally save results."""
    report = {
        "overall": {
            **compute_target_metrics(predictions),
            "precision_at_1": precision_at_k(predictions, k=1),
            "precision_at_5": precision_at_k(predictions, k=5),
            "average_precision": average_precision(predictions),
            **latency_stats(predictions),
        },
        "by_split": evaluate_by_split(predictions),
    }

    # Pretty print
    print(f"\n{'═' * 60}")
    print(f"  Evaluation Report")
    print(f"{'═' * 60}")
    overall = report["overall"]
    print(f"  Samples:           {overall['num_samples']}")
    print(f"  Target Success@1:  {overall['target_success_at_1']:.4f}")
    print(f"  Target Success@5:  {overall['target_success_at_5']:.4f}")
    print(f"  Precision@1:       {overall['precision_at_1']:.4f}")
    print(f"  Precision@5:       {overall['precision_at_5']:.4f}")
    print(f"  AP:                {overall['average_precision']:.4f}")
    print(f"  Avg Latency:       {overall.get('avg_latency', 0):.3f}s")

    if report["by_split"]:
        print(f"\n  {'Split':<15} {'S@1':<8} {'S@5':<8} {'P@1':<8} {'AP':<8}")
        print(f"  {'─' * 47}")
        for split, m in report["by_split"].items():
            print(f"  {split:<15} {m['target_success_at_1']:<8.4f} "
                  f"{m['target_success_at_5']:<8.4f} "
                  f"{m['precision_at_1']:<8.4f} "
                  f"{m['average_precision']:<8.4f}")
    print(f"{'═' * 60}")

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(report, f, indent=2)
        print(f"  Saved → {output_path}")

    return report
