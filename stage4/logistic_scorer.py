"""
stage4/logistic_scorer.py — Logistic Regression Scorer
=======================================================
Interface-ready.  Training requires labelled data from train_* splits.
Do NOT train on test_seen — that would be data leakage.
"""

import pickle
from pathlib import Path
from typing import Optional

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config


class LogisticScorer:
    """Thin wrapper around sklearn LogisticRegression."""

    def __init__(self, model_path: Optional[Path] = None):
        self._model = None
        if model_path and model_path.exists():
            self.load(model_path)

    @property
    def is_trained(self) -> bool:
        return self._model is not None

    def train(self, X: np.ndarray, y: np.ndarray):
        """Fit logistic regression.

        Parameters
        ----------
        X : (N, feature_dim)
        y : (N,) binary labels
        """
        from sklearn.linear_model import LogisticRegression

        self._model = LogisticRegression(max_iter=1000, class_weight="balanced")
        self._model.fit(X, y)

    def score(self, features: np.ndarray) -> np.ndarray:
        """Return P(y=1 | x) for each candidate.

        Returns zeros if model is not trained.
        """
        if not self.is_trained:
            return np.zeros(len(features))
        return self._model.predict_proba(features)[:, 1]

    def rank(self, features: np.ndarray) -> np.ndarray:
        scores = self.score(features)
        return np.argsort(-scores)

    def save(self, path: Path = config.MODELS_DIR / "scorer_logreg.pkl"):
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self._model, f)

    def load(self, path: Path = config.MODELS_DIR / "scorer_logreg.pkl"):
        with open(path, "rb") as f:
            self._model = pickle.load(f)
