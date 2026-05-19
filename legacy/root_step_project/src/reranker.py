"""
src/reranker.py — Target-conditioned semantic-geometric re-ranker
==================================================================

All rerankers share the Reranker ABC with score() and rank() methods.

Models:
  • DetectorBaseline   — initial geometric score only
  • RuleReranker       — deterministic target-conditioned weighted score
  • MLPReranker        — PyTorch MLP scoring head used by the project
"""

import abc
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config


# ═════════════════════════════════════════════════════════════════════
#  Base class
# ═════════════════════════════════════════════════════════════════════

class Reranker(abc.ABC):
    """Abstract reranker interface."""

    @abc.abstractmethod
    def score(self, features: np.ndarray) -> np.ndarray:
        """Score candidates. Returns (N,) float, higher = better."""
        ...

    def rank(self, features: np.ndarray) -> np.ndarray:
        """Return candidate indices sorted by descending score."""
        scores = self.score(features)
        return np.argsort(-scores)

    def select_top_k(
        self, features: np.ndarray, candidates: list, k: int = 5,
    ) -> List[dict]:
        """Re-rank candidates and return top-K as dicts."""
        scores = self.score(features)
        order = np.argsort(-scores)
        results = []
        for rank_pos, idx in enumerate(order[:k]):
            c = candidates[idx]
            results.append({
                "rank": rank_pos + 1,
                "candidate_id": c.candidate_id,
                "position": c.position,
                "rotation": c.rotation,
                "approach_vector": c.approach_vector,
                "closing_direction": c.closing_direction,
                "width": c.width,
                "grasp_type": c.grasp_type,
                "initial_geometric_score": c.detector_score,
                "final_score": float(scores[idx]),
                "feature_vector": features[idx].astype(float).tolist(),
            })
        return results


# ═════════════════════════════════════════════════════════════════════
#  Baseline 1: Detector score only
# ═════════════════════════════════════════════════════════════════════

class DetectorBaseline(Reranker):
    """Use raw detector score only (no reranking)."""

    def score(self, features: np.ndarray) -> np.ndarray:
        return features[:, 8]  # initial_geometric_score


# ═════════════════════════════════════════════════════════════════════
#  Baseline 2: Rule-based weighted score
# ═════════════════════════════════════════════════════════════════════

class RuleReranker(Reranker):
    """Deterministic weighted-sum scorer."""

    def __init__(self, weights: Optional[Dict[str, float]] = None):
        w = weights or config.RULE_WEIGHTS
        self.weights = w

    def score(self, features: np.ndarray) -> np.ndarray:
        """Score using weighted sum of features.

        features columns follow config.FEATURE_NAMES:
        [target_overlap, center_alignment, distance_to_target_center,
         gripper_width_match, approach_direction_score, depth_stability,
         collision_penalty, boundary_penalty, initial_geometric_score,
         grounding_score]
        """
        target_overlap = features[:, 0]
        center_alignment = features[:, 1]
        gripper_width_match = features[:, 3]
        approach_direction_score = features[:, 4]
        depth_stability = features[:, 5]
        collision_penalty = features[:, 6]
        boundary_penalty = features[:, 7]
        initial_geometric_score = features[:, 8]
        grounding_score = features[:, 9]

        w = self.weights
        s = (
            w.get("initial_geometric_score", 0.20) * initial_geometric_score
            + w.get("target_overlap", 0.25) * target_overlap
            + w.get("center_alignment", 0.15) * center_alignment
            + w.get("gripper_width_match", 0.10) * gripper_width_match
            + w.get("depth_stability", 0.10) * depth_stability
            + w.get("approach_direction_score", 0.10) * approach_direction_score
            + w.get("grounding_score", 0.05) * grounding_score
            - w.get("collision_penalty", 0.07) * collision_penalty
            - w.get("boundary_penalty", 0.03) * boundary_penalty
        )
        return s


# ═════════════════════════════════════════════════════════════════════
#  Main model: MLP Reranker
# ═════════════════════════════════════════════════════════════════════

class MLPReranker(Reranker):
    """PyTorch MLP scoring head.

    Architecture: feature_dim → 64 → 32 → 1 (sigmoid)
    """

    def __init__(
        self,
        feature_dim: int = config.FEATURE_DIM,
        hidden_dims: Optional[list] = None,
        model_path: Optional[Path] = None,
    ):
        self.feature_dim = feature_dim
        self.hidden_dims = hidden_dims or config.MLP_HIDDEN_DIMS
        self._model = None
        self._device = "cpu"

        if model_path and model_path.exists():
            self.load(model_path)

    @property
    def is_trained(self) -> bool:
        return self._model is not None

    def _build_model(self):
        import torch.nn as nn

        layers = []
        in_dim = self.feature_dim
        for h in self.hidden_dims:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        layers.append(nn.Sigmoid())
        return nn.Sequential(*layers)

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = config.MLP_EPOCHS,
        lr: float = config.MLP_LR,
        batch_size: int = config.MLP_BATCH_SIZE,
    ):
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset

        self._model = self._build_model().to(self._device)

        dataset = TensorDataset(
            torch.tensor(X, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32).unsqueeze(1),
        )
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        criterion = nn.BCELoss()
        optimiser = torch.optim.Adam(self._model.parameters(), lr=lr)

        self._model.train()
        for epoch in range(epochs):
            total_loss = 0
            for xb, yb in loader:
                xb, yb = xb.to(self._device), yb.to(self._device)
                pred = self._model(xb)
                loss = criterion(pred, yb)
                optimiser.zero_grad()
                loss.backward()
                optimiser.step()
                total_loss += loss.item() * len(xb)

            if (epoch + 1) % 10 == 0:
                avg = total_loss / len(dataset)
                print(f"  [MLP] Epoch {epoch + 1}/{epochs}  loss={avg:.4f}")

    def score(self, features: np.ndarray) -> np.ndarray:
        if not self.is_trained:
            return np.zeros(len(features))
        import torch
        self._model.eval()
        with torch.no_grad():
            x = torch.tensor(features, dtype=torch.float32).to(self._device)
            out = self._model(x).squeeze(-1).cpu().numpy()
        return out

    def save(self, path: Path = config.RERANKER_MLP_PATH):
        import torch
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self._model.state_dict(), str(path))
        meta_path = path.with_suffix(".json")
        with open(meta_path, "w") as f:
            json.dump({
                "feature_dim": self.feature_dim,
                "hidden_dims": self.hidden_dims,
            }, f)

    def load(self, path: Path = config.RERANKER_MLP_PATH):
        import torch
        meta_path = path.with_suffix(".json")
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
                self.feature_dim = meta["feature_dim"]
                self.hidden_dims = meta["hidden_dims"]
        self._model = self._build_model().to(self._device)
        self._model.load_state_dict(
            torch.load(str(path), map_location=self._device)
        )
        self._model.eval()


# ═════════════════════════════════════════════════════════════════════
#  Factory
# ═════════════════════════════════════════════════════════════════════

def get_reranker(name: str = "rule", **kwargs) -> Reranker:
    """Factory to get a reranker by name.

    Pass model_path=<Path> to load a specific checkpoint.
    Pass model_path=None to create an untrained reranker (no file loaded).
    Omit model_path to use the default fallback path from config.
    """
    if name == "detector":
        return DetectorBaseline()
    elif name == "rule":
        return RuleReranker(**{k: v for k, v in kwargs.items()
                               if k != "model_path"})
    elif name == "mlp":
        mp = kwargs.get("model_path", config.RERANKER_MLP_PATH)
        return MLPReranker(model_path=mp)
    else:
        raise ValueError(
            f"Unknown reranker: {name}. "
            f"Choose from: detector, rule, mlp"
        )
