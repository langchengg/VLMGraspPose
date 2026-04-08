"""
src/reranker.py — Grasp candidate re-ranking models (Step 9)
=============================================================
Consolidated from stage4/rule_scorer.py, logistic_scorer.py, mlp_scorer.py.

All rerankers share the Reranker ABC with score() and rank() methods.

Models:
  • DetectorBaseline   — raw detector score only (no reranking)
  • RuleReranker       — deterministic weighted-sum
  • LogisticReranker   — sklearn LogisticRegression
  • MLPReranker        — PyTorch MLP scoring head
  • PairwiseMLPReranker — pairwise MLP (strongest thesis variant)
"""

import abc
import json
import pickle
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
                "width": c.width,
                "grasp_score": c.detector_score,
                "rerank_score": float(scores[idx]),
            })
        return results


# ═════════════════════════════════════════════════════════════════════
#  Baseline 1: Detector score only
# ═════════════════════════════════════════════════════════════════════

class DetectorBaseline(Reranker):
    """Use raw detector score only (no reranking)."""

    def score(self, features: np.ndarray) -> np.ndarray:
        return features[:, 0]  # f1 = detector_score


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

        features columns: [detector_score, dist_target_3d, proj_dist_2d,
                           proj_overlap, target_points_ratio,
                           nontarget_points_ratio, collision_risk,
                           depth_consistency, florence_conf]
        """
        f1 = features[:, 0]   # detector_score
        f2 = features[:, 1]   # dist_target_3d (lower = better)
        f4 = features[:, 3]   # proj_overlap
        f5 = features[:, 4]   # target_points_ratio
        f7 = features[:, 6]   # collision_risk (lower = better)
        f9 = features[:, 8]   # florence_conf

        w = self.weights
        s = (w.get("detector_score", 0.25) * f1
             + w.get("dist_target_3d", 0.15) * (1.0 - f2)
             + w.get("proj_overlap", 0.20) * f4
             + w.get("target_points_ratio", 0.20) * f5
             + w.get("collision_risk", 0.10) * (1.0 - f7)
             + w.get("florence_conf", 0.10) * f9)
        return s


# ═════════════════════════════════════════════════════════════════════
#  Baseline 3: Logistic Regression
# ═════════════════════════════════════════════════════════════════════

class LogisticReranker(Reranker):
    """Sklearn LogisticRegression reranker."""

    def __init__(self, model_path: Optional[Path] = None):
        self._model = None
        if model_path and model_path.exists():
            self.load(model_path)

    @property
    def is_trained(self) -> bool:
        return self._model is not None

    def train(self, X: np.ndarray, y: np.ndarray):
        from sklearn.linear_model import LogisticRegression
        self._model = LogisticRegression(
            max_iter=1000, class_weight="balanced"
        )
        self._model.fit(X, y)

    def score(self, features: np.ndarray) -> np.ndarray:
        if not self.is_trained:
            return np.zeros(len(features))
        return self._model.predict_proba(features)[:, 1]

    def save(self, path: Path = config.RERANKER_LOGREG_PATH):
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self._model, f)

    def load(self, path: Path = config.RERANKER_LOGREG_PATH):
        with open(path, "rb") as f:
            self._model = pickle.load(f)


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
#  Strongest variant: Pairwise MLP Reranker
# ═════════════════════════════════════════════════════════════════════

class PairwiseMLPReranker(Reranker):
    """Pairwise MLP: learns which of two candidates is better.

    At inference, aggregate pairwise scores to produce pointwise ranking.
    """

    def __init__(
        self,
        feature_dim: int = config.FEATURE_DIM,
        hidden_dims: Optional[list] = None,
        model_path: Optional[Path] = None,
    ):
        self.feature_dim = feature_dim
        self.hidden_dims = hidden_dims or [64, 32]
        self._model = None
        self._device = "cpu"

        if model_path and model_path.exists():
            self.load(model_path)

    @property
    def is_trained(self) -> bool:
        return self._model is not None

    def _build_model(self):
        import torch.nn as nn
        in_dim = self.feature_dim * 2  # concatenated pair
        layers = []
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
        sample_ids: Optional[np.ndarray] = None,
        epochs: int = config.MLP_EPOCHS,
        lr: float = config.MLP_LR,
        batch_size: int = config.MLP_BATCH_SIZE,
        max_pairs_per_query: int = 50,
    ):
        """Train on pairwise comparisons.

        Generates pairs WITHIN each query/sample group:
        (positive, negative) → label = 1 (positive is better).

        If sample_ids is provided, pairs are only formed between
        candidates from the same query. Otherwise falls back to
        global pairing (less correct but still functional).
        """
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset

        rng = np.random.RandomState(42)
        pairs_feat, pair_labels = [], []

        if sample_ids is not None:
            # ── Query-wise pairing (correct) ─────────────────────────
            unique_ids = np.unique(sample_ids)
            for sid in unique_ids:
                mask = sample_ids == sid
                X_q = X[mask]
                y_q = y[mask]

                pos_idx = np.where(y_q == 1)[0]
                neg_idx = np.where(y_q == 0)[0]

                if len(pos_idx) == 0 or len(neg_idx) == 0:
                    continue

                n_pairs = min(
                    len(pos_idx) * len(neg_idx),
                    max_pairs_per_query,
                )
                for _ in range(n_pairs):
                    pi = rng.choice(pos_idx)
                    ni = rng.choice(neg_idx)
                    pairs_feat.append(np.concatenate([X_q[pi], X_q[ni]]))
                    pair_labels.append(1.0)
                    pairs_feat.append(np.concatenate([X_q[ni], X_q[pi]]))
                    pair_labels.append(0.0)
        else:
            # ── Global pairing (fallback) ────────────────────────────
            pos_idx = np.where(y == 1)[0]
            neg_idx = np.where(y == 0)[0]

            if len(pos_idx) == 0 or len(neg_idx) == 0:
                print("[PairwiseMLP] Need both positive and negative samples")
                return

            n_pairs = min(len(pos_idx) * len(neg_idx),
                          max_pairs_per_query * 100)
            for _ in range(n_pairs):
                pi = rng.choice(pos_idx)
                ni = rng.choice(neg_idx)
                pairs_feat.append(np.concatenate([X[pi], X[ni]]))
                pair_labels.append(1.0)
                pairs_feat.append(np.concatenate([X[ni], X[pi]]))
                pair_labels.append(0.0)

        if not pairs_feat:
            print("[PairwiseMLP] No pairs generated")
            return

        X_pairs = np.array(pairs_feat, dtype=np.float32)
        y_pairs = np.array(pair_labels, dtype=np.float32)
        print(f"  [PairwiseMLP] {len(X_pairs)} training pairs")

        self._model = self._build_model().to(self._device)

        dataset = TensorDataset(
            torch.tensor(X_pairs),
            torch.tensor(y_pairs).unsqueeze(1),
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
                print(f"  [PairwiseMLP] Epoch {epoch + 1}/{epochs}  "
                      f"loss={avg:.4f}")

    def score(self, features: np.ndarray) -> np.ndarray:
        """Aggregate pairwise scores to produce pointwise ranking."""
        if not self.is_trained:
            return np.zeros(len(features))

        import torch

        N = len(features)
        if N <= 1:
            return np.ones(N)

        self._model.eval()
        scores = np.zeros(N)

        with torch.no_grad():
            for i in range(N):
                win_count = 0
                for j in range(N):
                    if i == j:
                        continue
                    pair = np.concatenate([features[i], features[j]])
                    x = torch.tensor(
                        pair, dtype=torch.float32
                    ).unsqueeze(0).to(self._device)
                    prob = self._model(x).item()
                    win_count += prob
                scores[i] = win_count / max(N - 1, 1)

        return scores

    def save(self, path: Path = config.MODELS_DIR / "reranker_pairwise.pt"):
        import torch
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self._model.state_dict(), str(path))
        meta_path = path.with_suffix(".json")
        with open(meta_path, "w") as f:
            json.dump({
                "feature_dim": self.feature_dim,
                "hidden_dims": self.hidden_dims,
            }, f)

    def load(self, path: Path = config.MODELS_DIR / "reranker_pairwise.pt"):
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
    elif name == "logistic":
        mp = kwargs.get("model_path", config.RERANKER_LOGREG_PATH)
        return LogisticReranker(model_path=mp)
    elif name == "mlp":
        mp = kwargs.get("model_path", config.RERANKER_MLP_PATH)
        return MLPReranker(model_path=mp)
    elif name == "pairwise":
        mp = kwargs.get("model_path",
                         config.MODELS_DIR / "reranker_pairwise.pt")
        return PairwiseMLPReranker(model_path=mp)
    else:
        raise ValueError(
            f"Unknown reranker: {name}. "
            f"Choose from: detector, rule, logistic, mlp, pairwise"
        )

