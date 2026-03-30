"""
src/evaluation.py — Evaluation metrics (Step 11)
===================================================
Rewritten from experiments/eval.py.

Metrics:
  1. Target Success@1 / Success@5 — is the target object grasped?
  2. GraspNet-standard AP / Precision@k
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
    """Compute all target-aware metrics."""
    return {
        "target_success_at_1": target_success_at_k(predictions, k=1),
        "target_success_at_5": target_success_at_k(predictions, k=5),
        "num_samples": len(predictions),
    }


# ═════════════════════════════════════════════════════════════════════
#  Grasp quality metrics (GraspNet-standard)
# ═════════════════════════════════════════════════════════════════════

def precision_at_k(
    predictions: List[dict],
    k: int = 1,
    score_threshold: float = 0.3,
) -> float:
    """Precision@K: fraction of top-K grasps with score above threshold."""
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
            if g.get("rerank_score", 0) > score_threshold
        )
        precisions.append(good / len(top_k))

    return float(np.mean(precisions))


def average_precision(
    predictions: List[dict],
    score_threshold: float = 0.3,
) -> float:
    """Simplified AP over all predictions."""
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
            is_good = g.get("rerank_score", 0) > score_threshold
            if is_good:
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
    """Compare oracle (GT) grounding vs predicted (Florence-2) grounding."""
    oracle_metrics = compute_target_metrics(oracle_predictions)
    pred_metrics = compute_target_metrics(predicted_predictions)

    return {
        "oracle_grounding": oracle_metrics,
        "predicted_grounding": pred_metrics,
        "delta_success_at_1": (
            oracle_metrics["target_success_at_1"]
            - pred_metrics["target_success_at_1"]
        ),
        "delta_success_at_5": (
            oracle_metrics["target_success_at_5"]
            - pred_metrics["target_success_at_5"]
        ),
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
