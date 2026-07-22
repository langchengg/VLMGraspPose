"""Optional official GQ-CNN scoring for an already frozen candidate set."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .dexnet_adapter import ensure_official_gqcnn_path, gqcnn_runtime


MODEL_ALIASES = ("GQCNN-2.0", "GQCNN-2.1", "GQCNN-4.0-PJ")


class GQCNNScoringUnavailable(RuntimeError):
    """Raised when the legacy TensorFlow scoring runtime is unavailable."""


def resolve_model_directory(
    model_name: str | None,
    model_dir: Path | None,
) -> tuple[str | None, Path | None]:
    if model_dir is not None:
        path = Path(model_dir).expanduser().resolve()
        return model_name or path.name, path
    if model_name is None:
        return None, None
    if model_name not in MODEL_ALIASES:
        raise ValueError(
            f"Unknown official model {model_name!r}; use one of {MODEL_ALIASES} "
            "or pass --model-dir for a custom model"
        )
    official_models = Path(__file__).resolve().parents[2] / "third_party" / "gqcnn-official" / "models"
    return model_name, (official_models / model_name).resolve()


def score_fixed_candidates(
    state: Any,
    grasps: Sequence[Any],
    *,
    model_name: str,
    model_dir: Path,
    scoring_config: Mapping[str, Any],
    policy_config: Mapping[str, Any],
) -> np.ndarray:
    """Score every supplied grasp through the official bulk quality function.

    Values are a ranking baseline only; they are not declared calibrated
    physical success probabilities for OCID-VLG.
    """
    runtime = gqcnn_runtime()
    if not runtime["scoring_import_available"]:
        raise GQCNNScoringUnavailable(
            "Official GQ-CNN v1.3.0 scoring requires TensorFlow<=1.15, which "
            "has no Python 3.11/macOS arm64 build. Use candidate-only mode or "
            "the documented linux/amd64 legacy scoring environment."
        )
    path = Path(model_dir).expanduser().resolve()
    if not path.is_dir():
        raise FileNotFoundError(f"GQ-CNN model directory missing: {path}")
    ensure_official_gqcnn_path()
    from gqcnn.grasping.grasp_quality_function import GQCnnQualityFunction

    config = dict(scoring_config)
    config["gqcnn_model"] = str(path)
    scorer = GQCnnQualityFunction(config)
    values = np.asarray(
        scorer.quality(state, list(grasps), params=dict(policy_config)),
        dtype=np.float64,
    )
    if values.shape != (len(grasps),) or not np.all(np.isfinite(values)):
        raise ValueError(
            f"Official scorer returned invalid q-value array: {values.shape}"
        )
    return values
