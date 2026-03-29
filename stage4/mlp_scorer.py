"""
stage4/mlp_scorer.py — MLP Scoring Head
=========================================
Architecture: feature_dim → 64 → 32 → 1 (sigmoid)
Loss: BCE
Interface-ready.  Training requires labelled data from train_* splits.
"""

from pathlib import Path
from typing import Optional

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config


class MLPScorer:
    """PyTorch MLP scoring head."""

    def __init__(
        self,
        feature_dim: int = config.FEATURE_DIM_CORE,
        hidden_dims: list = None,
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
        import torch
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
        """Train the MLP on feature vectors and binary labels.

        Parameters
        ----------
        X : (N, feature_dim)
        y : (N,) binary labels
        """
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
                print(f"  [MLP] Epoch {epoch+1}/{epochs}  loss={avg:.4f}")

    def score(self, features: np.ndarray) -> np.ndarray:
        """Return sigmoid score for each candidate.

        Returns zeros if model is not trained.
        """
        if not self.is_trained:
            return np.zeros(len(features))

        import torch
        self._model.eval()
        with torch.no_grad():
            x = torch.tensor(features, dtype=torch.float32).to(self._device)
            out = self._model(x).squeeze(-1).cpu().numpy()
        return out

    def rank(self, features: np.ndarray) -> np.ndarray:
        scores = self.score(features)
        return np.argsort(-scores)

    def save(self, path: Path = config.MODELS_DIR / "scorer_mlp.pt"):
        import torch
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self._model.state_dict(), str(path))
        # Also save architecture info
        meta_path = path.with_suffix(".json")
        import json
        with open(meta_path, "w") as f:
            json.dump({
                "feature_dim": self.feature_dim,
                "hidden_dims": self.hidden_dims,
            }, f)

    def load(self, path: Path = config.MODELS_DIR / "scorer_mlp.pt"):
        import torch
        import json
        meta_path = path.with_suffix(".json")
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
                self.feature_dim = meta["feature_dim"]
                self.hidden_dims = meta["hidden_dims"]

        self._model = self._build_model().to(self._device)
        self._model.load_state_dict(torch.load(str(path), map_location=self._device))
        self._model.eval()
