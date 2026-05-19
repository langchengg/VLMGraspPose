import numpy as np


def clamp_score(score: float) -> float:
    return float(np.clip(score, 0.0, 1.0))
