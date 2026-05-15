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
#  GraspNet Baseline Detector
# ═════════════════════════════════════════════════════════════════════

class GraspNetDetector(GraspDetectorBase):
    """Wrapper around the official GraspNet baseline checkpoint.

    The detector is fixed: this project contributes the target-aware MLP
    reranker, not a new grasp proposal network.  The wrapper follows the
    official baseline demo contract:

      1. sample exactly ``num_point`` scene points,
      2. call ``GraspNet`` with ``end_points["point_clouds"]``,
      3. decode with ``pred_decode``,
      4. convert decoded rows through ``graspnetAPI.GraspGroup``.

    The GraspNet baseline code is not vendored.  Put it at
    ``external/graspnet-baseline`` or set ``GRASPNET_BASELINE_ROOT``.
    """

    def __init__(
        self,
        checkpoint_dir: Optional[Path] = None,
        checkpoint_path: Optional[Path] = None,
        baseline_root: Optional[Path] = None,
        device: Optional[str] = None,
        num_point: int = config.GRASPNET_NUM_POINT,
        num_view: int = config.GRASPNET_NUM_VIEW,
    ):
        self._checkpoint_dir = checkpoint_dir or config.GRASP_DETECTOR_DIR
        self._checkpoint_path = Path(checkpoint_path) if checkpoint_path else None
        self._baseline_root = Path(baseline_root) if baseline_root else None
        self.num_point = num_point
        self.num_view = num_view
        self._model = None
        self._net = None
        self._pred_decode = None
        self._GraspGroup = None

        if device is None:
            import torch
            self._device = "cuda:0" if torch.cuda.is_available() else "cpu"
        else:
            self._device = device

    def _resolve_checkpoint_path(self) -> Path:
        """Return the configured checkpoint path or discover one in the dir."""
        if self._checkpoint_path is not None:
            if self._checkpoint_path.exists():
                return self._checkpoint_path
            raise FileNotFoundError(
                f"GraspNet checkpoint not found: {self._checkpoint_path}"
            )

        preferred = config.GRASPNET_CHECKPOINT_PATH
        if preferred.exists():
            return preferred

        ckpt_dir = Path(self._checkpoint_dir)
        if not ckpt_dir.exists():
            raise FileNotFoundError(
                f"GraspNet checkpoint directory not found: {ckpt_dir}\n"
                "Run: python scripts/download_weights.py --graspnet --camera realsense"
            )

        ckpt_files = (
            list(ckpt_dir.glob("checkpoint-rs.tar"))
            + list(ckpt_dir.glob("*.tar"))
            + list(ckpt_dir.glob("*.pth"))
        )
        if not ckpt_files:
            raise FileNotFoundError(
                f"No GraspNet checkpoint files (*.tar, *.pth) in {ckpt_dir}.\n"
                "Run: python scripts/download_weights.py --graspnet --camera realsense"
            )
        return ckpt_files[0]

    def _candidate_baseline_roots(self) -> list[Path]:
        """Return baseline checkout candidates in priority order."""
        roots = []
        import os
        env_root = os.environ.get("GRASPNET_BASELINE_ROOT")
        if env_root:
            roots.append(Path(env_root))
        if self._baseline_root is not None:
            roots.append(self._baseline_root)
        roots.append(config.GRASPNET_BASELINE_ROOT)
        return roots

    def _add_baseline_to_path(self) -> Optional[Path]:
        """Expose official baseline modules if a checkout is available."""
        for root in self._candidate_baseline_roots():
            if not root.exists():
                continue
            for subdir in ["models", "dataset", "utils", "knn", "pointnet2"]:
                p = root / subdir
                if p.exists() and str(p) not in sys.path:
                    sys.path.insert(0, str(p))
            if str(root) not in sys.path:
                sys.path.insert(0, str(root))
            return root
        return None

    def _import_graspnet_baseline(self):
        """Import official baseline symbols with actionable diagnostics."""
        baseline_root = self._add_baseline_to_path()
        try:
            from graspnet import GraspNet as GraspNetModel, pred_decode
        except ImportError as e:
            checked = ", ".join(str(p) for p in self._candidate_baseline_roots())
            raise ImportError(
                "Cannot import the official GraspNet baseline modules.\n"
                "Clone https://github.com/graspnet/graspnet-baseline and either:\n"
                f"  1. place it at {config.GRASPNET_BASELINE_ROOT}, or\n"
                "  2. set GRASPNET_BASELINE_ROOT=/path/to/graspnet-baseline.\n"
                "Then install its compiled pointnet2 and knn operators.\n"
                f"Checked roots: {checked}\n"
                f"Original import error: {e}"
            )

        try:
            from graspnetAPI import GraspGroup
        except ImportError as e:
            raise ImportError(
                "Cannot import graspnetAPI. Install it from the official "
                "graspnetAPI repository before running GraspNet baseline.\n"
                f"Original import error: {e}"
            )

        self._GraspNetModel = GraspNetModel
        self._pred_decode = pred_decode
        self._GraspGroup = GraspGroup
        if baseline_root is not None:
            print(f"[GraspNetDetector] Baseline root: {baseline_root}")

    def _ensure_loaded(self):
        """Load model on first call (lazy)."""
        if self._model is not None:
            return

        import torch

        self._ckpt_path = self._resolve_checkpoint_path()
        self._import_graspnet_baseline()

        net = self._GraspNetModel(
            input_feature_dim=0,
            num_view=self.num_view,
            num_angle=12,
            num_depth=4,
            cylinder_radius=0.05,
            hmin=-0.02,
            hmax_list=[0.01, 0.02, 0.03, 0.04],
            is_training=False,
        )
        ckpt = torch.load(str(self._ckpt_path), map_location=self._device)
        state_dict = ckpt.get("model_state_dict", ckpt)
        net.load_state_dict(state_dict)
        net.to(self._device)
        net.eval()

        self._net = net
        self._model = "graspnet-baseline"
        print(f"[GraspNetDetector] Loaded checkpoint: {self._ckpt_path}")

    def _sample_points_for_network(
        self,
        point_cloud: np.ndarray,
        colors: Optional[np.ndarray] = None,
        rng: Optional[np.random.RandomState] = None,
    ) -> tuple[np.ndarray, Optional[np.ndarray]]:
        """Sample exactly ``num_point`` rows, preserving all rows if undersized."""
        points = np.asarray(point_cloud, dtype=np.float32)
        if len(points) == 0:
            raise ValueError("GraspNetDetector requires a non-empty point cloud")

        colors_arr = None
        if colors is not None:
            colors_arr = np.asarray(colors, dtype=np.float32)
            if len(colors_arr) != len(points):
                raise ValueError(
                    "colors must have the same number of rows as point_cloud"
                )

        rng = rng or np.random.RandomState(42)
        if len(points) >= self.num_point:
            idx = rng.choice(len(points), self.num_point, replace=False)
        else:
            idx_keep = np.arange(len(points))
            idx_extra = rng.choice(
                len(points),
                self.num_point - len(points),
                replace=True,
            )
            idx = np.concatenate([idx_keep, idx_extra], axis=0)

        sampled_points = points[idx].astype(np.float32, copy=False)
        sampled_colors = (
            colors_arr[idx].astype(np.float32, copy=False)
            if colors_arr is not None
            else None
        )
        return sampled_points, sampled_colors

    def detect(
        self,
        point_cloud: np.ndarray,
        colors: Optional[np.ndarray] = None,
        top_k: int = config.GRASP_TOP_K,
    ) -> List[GraspCandidate]:
        """Run grasp detection on full-scene point cloud."""
        self._ensure_loaded()

        import torch

        try:
            pc_input, color_input = self._sample_points_for_network(
                point_cloud,
                colors,
            )
        except ValueError:
            return []

        end_points = {
            "point_clouds": torch.from_numpy(pc_input[None]).to(self._device),
        }
        if color_input is not None:
            end_points["cloud_colors"] = color_input

        with torch.no_grad():
            end_points = self._net(end_points)
            grasp_preds = self._pred_decode(end_points)

        preds_np = grasp_preds[0].detach().cpu().numpy()
        gg = self._GraspGroup(preds_np)
        maybe = gg.nms()
        if maybe is not None:
            gg = maybe
        maybe = gg.sort_by_score()
        if maybe is not None:
            gg = maybe

        candidates = []
        for i, g in enumerate(gg):
            if i >= top_k:
                break
            candidates.append(GraspCandidate(
                candidate_id=i,
                position=g.translation.tolist(),
                rotation=g.rotation_matrix.flatten().tolist(),
                width=float(g.width),
                detector_score=float(np.clip(g.score, 0, 1)),
                source="graspnet",
            ))

        return candidates


class PrecomputedGraspLoader(GraspDetectorBase):
    """Load pre-computed grasp predictions from GraspNet baseline outputs.

    The official baseline saves predictions as .npy files. This class
    loads those files directly, avoiding the need for model inference.

    Expected layout: {precomputed_dir}/{scene_id}/{camera}/{frame_id:04d}.npy
    """

    def __init__(self, precomputed_dir: Optional[Path] = None):
        self._dir = precomputed_dir or config.DERIVED_DIR / "precomputed_grasps"

    def detect(
        self,
        point_cloud: np.ndarray,
        colors: Optional[np.ndarray] = None,
        top_k: int = config.GRASP_TOP_K,
    ) -> List[GraspCandidate]:
        raise NotImplementedError(
            "PrecomputedGraspLoader.detect() should not be called directly. "
            "Use load_from_file() instead."
        )

    def load_from_file(
        self,
        scene_id: int,
        camera: str,
        frame_id: int,
        top_k: int = config.GRASP_TOP_K,
    ) -> List[GraspCandidate]:
        """Load pre-computed grasps for a specific view."""
        npy_path = (
            self._dir
            / f"scene_{scene_id:04d}"
            / camera
            / f"{frame_id:04d}.npy"
        )
        if not npy_path.exists():
            return []

        try:
            from graspnetAPI import GraspGroup
        except ImportError:
            print("[PrecomputedGraspLoader] graspnetAPI not installed")
            return []

        preds = np.load(str(npy_path))
        gg = GraspGroup(preds)
        gg = gg.nms().sort_by_score()

        candidates = []
        for i, g in enumerate(gg):
            if i >= top_k:
                break
            candidates.append(GraspCandidate(
                candidate_id=i,
                position=g.translation.tolist(),
                rotation=g.rotation_matrix.flatten().tolist(),
                width=float(g.width),
                detector_score=float(np.clip(g.score, 0, 1)),
                source="graspnet_precomputed",
            ))

        return candidates


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
        max_points_for_sampling: int = config.ANTIPODAL_MAX_POINTS_FOR_SAMPLING,
    ):
        self.top_k = top_k
        self.num_contact_samples = num_contact_samples
        self.min_width = min_width
        self.max_width = max_width
        self.antipodal_thresh = antipodal_thresh
        self.max_points_for_sampling = max_points_for_sampling

    def _downsample_for_sampling(self, point_cloud: np.ndarray) -> np.ndarray:
        """Bound the point count before normal estimation and pair search."""
        if len(point_cloud) <= self.max_points_for_sampling:
            return point_cloud

        rng = np.random.RandomState(42)
        idx = rng.choice(
            len(point_cloud),
            size=self.max_points_for_sampling,
            replace=False,
        )
        return point_cloud[np.sort(idx)]

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

        point_cloud = self._downsample_for_sampling(point_cloud)

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
