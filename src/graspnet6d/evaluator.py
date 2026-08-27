"""Per-frozen-candidate adapter for the official GraspNet evaluator helpers.

The official high-level ``eval_grasp`` function applies score-dependent pose
NMS, keeps ten grasps per associated object, and then applies a global top-50
cutoff.  Those operations would change frozen candidate membership.  This
module instead invokes the same official point association, collision, Dex-Net
grasp conversion, and friction/force-closure helpers on *every* input row while
preserving input order.

Formal parity is intentionally not claimed by the implementation itself.  It
must be established on a real official scene/candidate example and recorded in
an external validation artifact before batch labels are treated as formal.
"""

from __future__ import annotations

import copy
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_GRASPNET_API_ROOT = REPOSITORY_ROOT / "legacy" / "external_graspnet" / "graspnetAPI"
GRASPNET_API_UPSTREAM = "https://github.com/graspnet/graspnetAPI"
GRASPNET_API_LOCAL_COMMIT = "bd6783c3effdebd895abfba8b96dc22a42ec3b5a"
FRICTION_DESCENDING = np.asarray([1.2, 1.0, 0.8, 0.6, 0.4, 0.2], dtype=np.float64)
FORMAL_PARITY_STATUS = "blocked_until_real_official_example_parity_is_recorded"


class EvaluatorAdapterError(RuntimeError):
    pass


@dataclass(frozen=True)
class EvaluatorParityGate:
    """Evidence required before using the adapter for batch/formal labels."""

    validated: bool
    artifact_path: str

    def validate(self) -> None:
        if not self.validated or not self.artifact_path.strip():
            raise EvaluatorAdapterError(
                "formal evaluator labeling is blocked until an official real-scene "
                "candidate parity run records association, collision, and friction agreement"
            )


@dataclass(frozen=True)
class CandidateEvaluation:
    candidate_index: int
    associated_instance_index: int
    associated_object_id: int
    target_object_id: int
    correct_target: bool
    collision: bool
    empty_grasp: bool
    friction_score: float
    valid_geometry: bool
    relevance: int

    def success_at(self, friction_threshold: float) -> bool:
        return bool(
            self.correct_target
            and self.valid_geometry
            and not self.collision
            and 0 < self.friction_score <= float(friction_threshold)
        )

    def to_record(self) -> dict[str, object]:
        record = asdict(self)
        # Canonical metric-table names. Keep the evaluator-native aliases above
        # for traceability, but downstream ranking code consumes these fields.
        record["target_match"] = self.correct_target
        record["pose_valid"] = self.valid_geometry
        record["friction_required"] = self.friction_score
        for threshold in (0.2, 0.4, 0.6, 0.8, 1.0, 1.2):
            record[f"success_mu_{threshold:.1f}"] = self.success_at(threshold)
        return record


def ensure_graspnetapi_source(
    api_root: Path | str = DEFAULT_GRASPNET_API_ROOT,
    *,
    verify_commit: bool = True,
) -> Path:
    """Put only the audited local source checkout on the import path."""

    root = Path(api_root).expanduser().resolve()
    marker = root / "graspnetAPI" / "utils" / "eval_utils.py"
    if not marker.is_file():
        raise FileNotFoundError(f"official graspnetAPI source missing: {marker}")
    if verify_commit:
        try:
            commit = subprocess.run(
                ["git", "-C", str(root), "rev-parse", "HEAD"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
            changes = subprocess.run(
                ["git", "-C", str(root), "status", "--porcelain", "--untracked-files=no"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        except (OSError, subprocess.CalledProcessError) as error:
            raise EvaluatorAdapterError(
                f"cannot verify local graspnetAPI checkout provenance at {root}"
            ) from error
        if commit != GRASPNET_API_LOCAL_COMMIT or changes:
            raise EvaluatorAdapterError(
                "local graspnetAPI checkout does not match the audited pristine commit: "
                f"commit={commit}, expected={GRASPNET_API_LOCAL_COMMIT}, changes={changes!r}"
            )
    loaded = sys.modules.get("graspnetAPI")
    loaded_file = getattr(loaded, "__file__", None) if loaded is not None else None
    if loaded_file is not None:
        try:
            Path(loaded_file).resolve().relative_to(root)
        except ValueError as error:
            raise ImportError(
                f"an incompatible graspnetAPI package is already loaded from {loaded_file}"
            ) from error
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root


def _official_eval_utils(api_root: Path | str = DEFAULT_GRASPNET_API_ROOT) -> Any:
    ensure_graspnetapi_source(api_root)
    try:
        from graspnetAPI.utils import eval_utils
    except ModuleNotFoundError as error:
        raise EvaluatorAdapterError(
            "official graspnetAPI/Dex-Net runtime dependency is missing from the "
            f"isolated environment: {error.name}; install the locked evaluator dependencies"
        ) from error

    return eval_utils


def _validate_models_and_poses(
    models: Sequence[np.ndarray], poses: Sequence[np.ndarray], object_ids: Sequence[int]
) -> tuple[list[np.ndarray], list[np.ndarray], np.ndarray]:
    if not models or len(models) != len(poses) or len(models) != len(object_ids):
        raise ValueError("models, poses, and object_ids must have the same non-zero length")
    checked_models: list[np.ndarray] = []
    checked_poses: list[np.ndarray] = []
    for index, (model, pose) in enumerate(zip(models, poses)):
        points = np.asarray(model, dtype=np.float64)
        transform = np.asarray(pose, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 3 or len(points) == 0:
            raise ValueError(f"model {index} must be a non-empty Nx3 point array")
        if not np.all(np.isfinite(points)):
            raise ValueError(f"model {index} contains NaN or Inf")
        if transform.shape != (4, 4) or not np.all(np.isfinite(transform)):
            raise ValueError(f"pose {index} must be a finite 4x4 transform")
        if not np.allclose(transform[3], [0.0, 0.0, 0.0, 1.0], atol=1e-7):
            raise ValueError(f"pose {index} has an invalid homogeneous last row")
        checked_models.append(points)
        checked_poses.append(transform)
    ids = np.asarray(object_ids, dtype=np.int64)
    return checked_models, checked_poses, ids


def associate_candidates_to_models(
    candidate_translations_camera_m: np.ndarray,
    models_object_m: Sequence[np.ndarray],
    poses_object_to_camera: Sequence[np.ndarray],
    object_ids: Sequence[int],
    *,
    api_root: Path | str = DEFAULT_GRASPNET_API_ROOT,
) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
    """Use official transforms/nearest-points to associate every candidate."""

    utils = _official_eval_utils(api_root)
    translations = np.asarray(candidate_translations_camera_m, dtype=np.float64)
    if translations.ndim != 2 or translations.shape[1] != 3:
        raise ValueError("candidate translations must have shape (N,3)")
    if not np.all(np.isfinite(translations)):
        raise ValueError("candidate translations contain NaN or Inf")
    models, poses, ids = _validate_models_and_poses(
        models_object_m, poses_object_to_camera, object_ids
    )
    transformed = [utils.transform_points(model, pose) for model, pose in zip(models, poses)]
    counts = [len(model) for model in transformed]
    scene = np.concatenate(transformed, axis=0)
    scene_instances = np.concatenate(
        [np.full(count, index, dtype=np.int64) for index, count in enumerate(counts)]
    )
    if len(translations) == 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64), transformed
    closest = utils.compute_closest_points(translations, scene)
    instances = scene_instances[closest]
    return instances, ids[instances], transformed


def _rotation_is_valid(rotation: np.ndarray, *, atol: float = 1e-4) -> bool:
    return bool(
        np.all(np.isfinite(rotation))
        and np.allclose(rotation.T @ rotation, np.eye(3), atol=atol)
        and np.isclose(np.linalg.det(rotation), 1.0, atol=atol)
    )


def validate_graspnet_rows(rows: np.ndarray, *, max_width_m: float = 0.1) -> np.ndarray:
    """Return a per-row geometry-valid mask for the official 17-value schema."""

    value = np.asarray(rows, dtype=np.float64)
    if value.ndim != 2 or value.shape[1] != 17:
        raise ValueError(f"GraspNet grasp rows must have shape (N,17), got {value.shape}")
    valid = np.all(np.isfinite(value[:, :16]), axis=1)
    valid &= (value[:, 1] >= 0) & (value[:, 1] <= max_width_m)
    valid &= value[:, 2] > 0
    valid &= value[:, 3] > 0
    for index, row in enumerate(value):
        valid[index] &= _rotation_is_valid(row[4:13].reshape(3, 3))
    return valid


def relevance_from_friction(
    friction_score: float,
    *,
    correct_target: bool,
    collision: bool,
    valid_geometry: bool = True,
) -> int:
    """Map official minimum successful friction to target-specific relevance 0..6."""

    if not correct_target or collision or not valid_geometry:
        return 0
    score = float(friction_score)
    if not np.isfinite(score) or score <= 0 or score > 1.2:
        return 0
    for threshold, relevance in ((0.2, 6), (0.4, 5), (0.6, 4), (0.8, 3), (1.0, 2), (1.2, 1)):
        if score <= threshold + 1e-9:
            return relevance
    return 0


def _force_closure_configs(config: dict[str, Any], utils: Any) -> dict[float, Any]:
    local_config = copy.deepcopy(config)
    try:
        metric = local_config["metrics"]["force_closure"]
    except (KeyError, TypeError) as error:
        raise ValueError("Dex-Net config lacks metrics.force_closure") from error
    result: dict[float, Any] = {}
    for value in FRICTION_DESCENDING:
        friction = round(float(value), 2)
        metric["friction_coef"] = friction
        result[friction] = utils.GraspQualityConfigFactory.create_config(metric)
    return result


def evaluate_frozen_candidates(
    grasp_rows: np.ndarray,
    *,
    models_object_m: Sequence[np.ndarray],
    dexnet_models: Sequence[Any],
    poses_object_to_camera: Sequence[np.ndarray],
    object_ids: Sequence[int],
    target_object_id: int,
    dexnet_config: dict[str, Any],
    table_points_camera_m: np.ndarray,
    model_voxel_size_m: float = 0.008,
    api_root: Path | str = DEFAULT_GRASPNET_API_ROOT,
    parity_gate: EvaluatorParityGate | None = None,
    validation_probe: bool = False,
) -> list[CandidateEvaluation]:
    """Evaluate all frozen rows with official low-level evaluator operations.

    Inputs are never score-sorted, NMS-filtered, top-k-filtered, or otherwise
    removed.  Model point clouds are voxel-sampled exactly as in official
    ``GraspNetEval.eval_scene`` before association and collision evaluation.
    """

    if not validation_probe:
        if parity_gate is None:
            raise EvaluatorAdapterError(
                "batch evaluation requires an explicit passed EvaluatorParityGate"
            )
        parity_gate.validate()
    utils = _official_eval_utils(api_root)
    rows = np.asarray(grasp_rows, dtype=np.float64)
    valid = validate_graspnet_rows(rows)
    if len(models_object_m) != len(dexnet_models):
        raise ValueError("one Dex-Net model is required per object instance")
    sampled_models = [utils.voxel_sample_points(np.asarray(model), model_voxel_size_m) for model in models_object_m]
    instances = np.full(len(rows), -1, dtype=np.int64)
    associated_ids = np.full(len(rows), -1, dtype=np.int64)
    finite_translation = np.all(np.isfinite(rows[:, 13:16]), axis=1)
    associable_indices = np.flatnonzero(finite_translation)
    associated_instances, ids, transformed_models = associate_candidates_to_models(
        rows[associable_indices, 13:16],
        sampled_models,
        poses_object_to_camera,
        object_ids,
        api_root=api_root,
    )
    instances[associable_indices] = associated_instances
    associated_ids[associable_indices] = ids
    number_models = len(sampled_models)
    indices_by_model = [
        np.flatnonzero((instances == index) & valid) for index in range(number_models)
    ]
    grasps_by_model = [rows[indices].copy() for indices in indices_by_model]

    scene_points = np.concatenate(transformed_models, axis=0)
    table = np.asarray(table_points_camera_m, dtype=np.float64)
    if table.ndim != 2 or table.shape[1] != 3 or not np.all(np.isfinite(table)):
        raise ValueError("table points must be a finite Nx3 array")
    if len(table) == 0:
        raise ValueError("official scene evaluation requires non-empty table geometry")
    scene_points = np.concatenate((scene_points, table), axis=0)

    collisions, empty, dex_grasps = utils.collision_detection(
        grasps_by_model,
        transformed_models,
        dexnet_models,
        poses_object_to_camera,
        scene_points,
        outlier=0.05,
        empty_thresh=10,
        return_dexgrasps=True,
    )
    force_closure = _force_closure_configs(dexnet_config, utils)
    friction = np.full(len(rows), -1.0, dtype=np.float64)
    collision = np.ones(len(rows), dtype=bool)
    empty_grasp = np.ones(len(rows), dtype=bool)
    for model_index, candidate_indices in enumerate(indices_by_model):
        model_collision = np.asarray(collisions[model_index], dtype=bool)
        model_empty = np.asarray(empty[model_index], dtype=bool)
        if len(model_collision) != len(candidate_indices):
            raise EvaluatorAdapterError("official collision helper changed candidate membership")
        collision[candidate_indices] = model_collision
        empty_grasp[candidate_indices] = model_empty
        for local_index, candidate_index in enumerate(candidate_indices):
            if not valid[candidate_index] or model_collision[local_index]:
                continue
            dex_grasp = dex_grasps[model_index][local_index]
            if dex_grasp is None:
                continue
            friction[candidate_index] = utils.get_grasp_score(
                dex_grasp,
                dexnet_models[model_index],
                FRICTION_DESCENDING,
                force_closure,
            )

    results: list[CandidateEvaluation] = []
    for index in range(len(rows)):
        correct = int(associated_ids[index]) == int(target_object_id)
        results.append(
            CandidateEvaluation(
                candidate_index=index,
                associated_instance_index=int(instances[index]),
                associated_object_id=int(associated_ids[index]),
                target_object_id=int(target_object_id),
                correct_target=correct,
                collision=bool(collision[index]),
                empty_grasp=bool(empty_grasp[index]),
                friction_score=float(friction[index]),
                valid_geometry=bool(valid[index]),
                relevance=relevance_from_friction(
                    friction[index],
                    correct_target=correct,
                    collision=bool(collision[index]),
                    valid_geometry=bool(valid[index]),
                ),
            )
        )
    if [result.candidate_index for result in results] != list(range(len(rows))):
        raise EvaluatorAdapterError("candidate ordering changed during evaluation")
    return results
