"""
stage4/rule_scorer.py — Rule-Based Weighted Scoring (Baseline)
===============================================================
S = α*f1 + β*f2 + γ*(1−f3) + δ*f4 + ε*f5

No training needed.  This is the first scorer to verify the pipeline.
"""

from typing import Dict, List

import numpy as np

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config


class RuleScorer:
    """Deterministic weighted-sum scorer."""

    def __init__(self, weights: Dict[str, float] = None):
        w = weights or config.RULE_WEIGHTS
        self.alpha = w["f1_grasp_score"]
        self.beta  = w["f2_in_target"]
        self.gamma = w["f3_distance"]
        self.delta = w["f4_iou"]
        self.eps   = w["f5_vlm_conf"]

    def score(self, features: np.ndarray) -> np.ndarray:
        """Score candidates.

        Parameters
        ----------
        features : (N, >=5) array where columns are [f1, f2, f3, f4, f5, ...]

        Returns
        -------
        scores : (N,) float array, higher is better.
        """
        f1 = features[:, 0]
        f2 = features[:, 1]
        f3 = features[:, 2]
        f4 = features[:, 3]
        f5 = features[:, 4]

        s = (self.alpha * f1
             + self.beta * f2
             + self.gamma * (1.0 - f3)
             + self.delta * f4
             + self.eps * f5)

        return s

    def rank(self, features: np.ndarray) -> np.ndarray:
        """Return candidate indices sorted by descending score."""
        scores = self.score(features)
        return np.argsort(-scores)
