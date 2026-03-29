"""
stage5/select_best_grasp.py — Final Grasp Selection
=====================================================
Sort candidates by score, return top-1 / top-5.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from stage2.grasp_generator import GraspCandidate


def select_best_grasp(
    candidates: List[GraspCandidate],
    scores: np.ndarray,
    top_k: int = 5,
) -> List[Dict]:
    """Re-rank candidates by *scores* and return top-K.

    Parameters
    ----------
    candidates : list of GraspCandidate
    scores : (N,) final scores (higher = better)
    top_k : how many to return

    Returns
    -------
    list of dicts with keys:
        position, orientation, width, final_score, candidate_id, rank
    """
    if len(candidates) == 0:
        return []

    order = np.argsort(-scores)
    results = []

    for rank, idx in enumerate(order[:top_k]):
        c = candidates[idx]
        results.append({
            "rank": rank + 1,
            "candidate_id": c.candidate_id,
            "position": c.position,
            "orientation": c.orientation,
            "width": c.width,
            "final_score": float(scores[idx]),
            "source": c.source,
        })

    return results


def save_selection(
    sample_id: str,
    results: List[Dict],
    scorer_name: str = "rule",
    output_dir: Path = config.PROJECT_ROOT / "results",
) -> Path:
    """Save selection results."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{sample_id}_{scorer_name}.json"

    with open(path, "w") as f:
        json.dump({
            "sample_id": sample_id,
            "scorer": scorer_name,
            "selections": results,
        }, f, indent=2)

    return path
