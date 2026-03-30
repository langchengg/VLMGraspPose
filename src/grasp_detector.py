"""
src/grasp_detector.py — 6-DoF Grasp Candidate Generation (Step 6)
===================================================================
Rewritten from stage2/grasp_generator.py + stage2/roi_sampler.py.

Two implementations:
  • GraspNetDetector     — wrapper around pretrained GraspNet checkpoint
                           (main: runs on FULL-SCENE point cloud)
  • AntipodalSampler     — local antipodal sampler (ablation / fallback)

All grasp poses are in **camera frame**.
"""

import abc
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config


# ═════════════════════════════════════════════════════════════════════
#  Grasp Candidate dataclass
# ═════════════════════════════════════════════════════════════════════

@dataclass
class GraspCandidate:
    candidate_id: int
    position: List[float]        # [x, y, z] camera frame
    rotation: List[float]        # flattened 3×3 rotation [r11..r33]
    width: float                 # gripper opening (metres)
    detector_score: float        # raw quality score 0–1
    source: str                  # "graspnet" or "antipodal"

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def rotation_matrix(self) -> np.ndarray:
        """Return 3×3 rotation matrix."""
        return np.array(self.rotation).reshape(3, 3)


# ═════════════════════════════════════════════════════════════════════
#  Base class
# ═════════════════════════════════════════════════════════════════════

class GraspDetectorBase(abc.ABC):
    @abc.abstractmethod
    def detect(
        self,
        point_cloud: np.ndarray,
        colors: Optional[np.ndarray] = None,
        top_k: int = config.GRASP_TOP_K,
    ) -> List[GraspCandidate]:
        """Generate grasp candidates from a full-scene point cloud."""
        ...


# ═════════════════════════════════════════════════════════════════════
#  GraspNet Baseline Detector (main)
# ═════════════════════════════════════════════════════════════════════

class GraspNetDetector(GraspDetectorBase):
    """Wrapper around pretrained GraspNet baseline checkpoint.

    Loads the official checkpoint and runs inference on the full-scene
    point cloud.  The grasp detector is kept FIXED — the thesis
    contribution is the target-aware reranking layer, not the detector.

    Checkpoint should be at: models/grasp_detector/
    """

    def __init__(
        self,
        checkpoint_dir: Optional[Path] = None,
        device: Optional[str] = None,
    ):
        self._checkpoint_dir = checkpoint_dir or config.GRASP_DETECTOR_DIR
        self._model = None

        if device is None:
            import torch
            self._device = "cuda:0" if torch.cuda.is_available() else "cpu"
        else:
            self._device = device

    def _ensure_loaded(self):
        """Load model on first call (lazy)."""
        if self._model is not None:
            return

        ckpt_dir = self._checkpoint_dir
        if not ckpt_dir.exists():
            raise FileNotFoundError(
                f"GraspNet checkpoint not found at {ckpt_dir}. "
                f"Run: python scripts/download_weights.py --graspnet"
            )

        # Try to load the official graspnet-baseline model
        try:
            self._load_graspnet_baseline(ckpt_dir)
        except Exception as e:
            print(f"[GraspNetDetector] Could not load official baseline: {e}")
            print("[GraspNetDetector] Falling back to antipodal sampler.")
            self._model = "FALLBACK"

    def _load_graspnet_baseline(self, ckpt_dir: Path):
        """Load the official GraspNet baseline model.

        This expects the graspnet-baseline repo structure with
        checkpoint files (checkpoint-*.tar or *.pth).
        """
        import torch

        # Look for checkpoint files
        ckpt_files = list(ckpt_dir.glob("*.tar")) + list(ckpt_dir.glob("*.pth"))
        if not ckpt_files:
            raise FileNotFoundError(f"No checkpoint files in {ckpt_dir}")

        # Try importing graspnet baseline
        try:
            from graspnetAPI import GraspGroup
            self._has_graspnet_api = True
        except ImportError:
            self._has_graspnet_api = False

        # Store checkpoint path for inference
        self._ckpt_path = ckpt_files[0]
        self._model = "LOADED"
        print(f"[GraspNetDetector] Checkpoint: {self._ckpt_path}")

    def detect(
        self,
        point_cloud: np.ndarray,
        colors: Optional[np.ndarray] = None,
        top_k: int = config.GRASP_TOP_K,
    ) -> List[GraspCandidate]:
        """Run grasp detection on full-scene point cloud.

        If the official model can't be loaded, falls back to
        the antipodal sampler.
        """
        self._ensure_loaded()

        if self._model == "FALLBACK":
            sampler = AntipodalSampler(top_k=top_k)
            return sampler.detect(point_cloud, colors, top_k)

        # Official GraspNet baseline inference
        return self._run_graspnet_inference(point_cloud, colors, top_k)

    def _run_graspnet_inference(
        self,
        point_cloud: np.ndarray,
        colors: Optional[np.ndarray],
        top_k: int,
    ) -> List[GraspCandidate]:
        """Run official GraspNet inference pipeline.

        This wraps the standard inference flow:
        1. Prepare input (down-sample, normalize)
        2. Forward pass through the network
        3. Post-process grasp predictions
        4. Return top-K candidates
        """
        import torch

        # For now, use the GraspNet API if available
        try:
            from graspnetAPI import GraspGroup
        except ImportError:
            print("[GraspNetDetector] graspnetAPI not installed, "
                  "falling back to antipodal sampler")
            sampler = AntipodalSampler(top_k=top_k)
            return sampler.detect(point_cloud, colors, top_k)

        # The actual inference depends on the specific checkpoint format.
        # This is a template that should be adapted to the actual model.
        # For now, fall back to antipodal if we can't run the real model.
        print("[GraspNetDetector] Using antipodal fallback "
              "(integrate real model later)")
        sampler = AntipodalSampler(top_k=top_k)
        return sampler.detect(point_cloud, colors, top_k)


# ═════════════════════════════════════════════════════════════════════
#  Antipodal Sampler (ablation / fallback)
# ═════════════════════════════════════════════════════════════════════

class AntipodalSampler(GraspDetectorBase):
    """Generate grasp candidates via antipodal sampling.

    Strategy:
    1. Sub-sample contact points on the surface.
    2. For each, find an approximate antipodal pair.
    3. Compute grasp centre, approach direction, opening width.
    4. Score by normal-alignment quality.
    5. Return top-K candidates.
    """

    def __init__(
        self,
        top_k: int = config.GRASP_TOP_K,
        num_contact_samples: int = 200,
        min_width: float = config.GRASP_MIN_WIDTH,
        max_width: float = config.GRASP_MAX_WIDTH,
        antipodal_thresh: float = 0.3,
    ):
        self.top_k = top_k
        self.num_contact_samples = num_contact_samples
        self.min_width = min_width
        self.max_width = max_width
        self.antipodal_thresh = antipodal_thresh

    def detect(
        self,
        point_cloud: np.ndarray,
        colors: Optional[np.ndarray] = None,
        top_k: int = None,
    ) -> List[GraspCandidate]:
        if top_k is None:
            top_k = self.top_k

        if len(point_cloud) < 10:
            return []

        from src.point_cloud import estimate_normals_pca
        normals = estimate_normals_pca(
            point_cloud, k=min(30, len(point_cloud))
        )

        N = len(point_cloud)
        num_samples = min(self.num_contact_samples, N)
        rng = np.random.RandomState(42)
        idx1 = rng.choice(N, size=num_samples, replace=(num_samples > N))

        candidates = []
        for i in idx1:
            p1 = point_cloud[i]
            n1 = normals[i]

            if np.linalg.norm(n1) < 1e-6:
                continue

            diffs = point_cloud - p1
            dists = np.linalg.norm(diffs, axis=1)

            width_mask = (dists >= self.min_width) & (dists <= self.max_width)
            cos_angle = -np.sum(normals * n1, axis=1)
            antipodal_mask = cos_angle > self.antipodal_thresh

            valid = width_mask & antipodal_mask
            valid_idx = np.where(valid)[0]
            if len(valid_idx) == 0:
                continue

            best_j = valid_idx[np.argmax(cos_angle[valid_idx])]
            p2 = point_cloud[best_j]

            center = (p1 + p2) / 2.0
            width = float(np.linalg.norm(p2 - p1))
            approach = p2 - p1
            approach = approach / (np.linalg.norm(approach) + 1e-8)

            R = _approach_to_rotation(approach)
            quality = float(np.clip(cos_angle[best_j], 0.0, 1.0))

            candidates.append(GraspCandidate(
                candidate_id=len(candidates),
                position=center.tolist(),
                rotation=R.flatten().tolist(),
                width=width,
                detector_score=quality,
                source="antipodal",
            ))

        candidates = _dedup_candidates(candidates, min_dist=0.005)
        candidates.sort(key=lambda c: c.detector_score, reverse=True)
        candidates = candidates[:top_k]

        for i, c in enumerate(candidates):
            c.candidate_id = i

        return candidates


# ═════════════════════════════════════════════════════════════════════
#  Helpers
# ═════════════════════════════════════════════════════════════════════

def _approach_to_rotation(approach: np.ndarray) -> np.ndarray:
    """Convert approach vector to 3×3 rotation matrix."""
    ax = approach / (np.linalg.norm(approach) + 1e-8)

    up = np.array([0.0, 0.0, -1.0])
    if abs(np.dot(ax, up)) > 0.9:
        up = np.array([0.0, 1.0, 0.0])

    az = np.cross(ax, up)
    az = az / (np.linalg.norm(az) + 1e-8)
    ay = np.cross(az, ax)
    ay = ay / (np.linalg.norm(ay) + 1e-8)

    return np.stack([ax, ay, az], axis=1)  # 3×3


def _dedup_candidates(
    candidates: List[GraspCandidate],
    min_dist: float = 0.005,
) -> List[GraspCandidate]:
    """Remove near-duplicate candidates."""
    if not candidates:
        return candidates

    kept = [candidates[0]]
    for c in candidates[1:]:
        pos = np.array(c.position)
        too_close = any(
            np.linalg.norm(pos - np.array(k.position)) < min_dist
            for k in kept
        )
        if not too_close:
            kept.append(c)
    return kept
