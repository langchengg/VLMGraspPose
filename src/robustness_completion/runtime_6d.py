"""Fresh-input 6-DoF runtime worker for the locked robustness child run.

This module deliberately keeps execution and parity separate.  Timed execution
starts from RGB-D/calibration files (or their explicitly preloaded bytes) and
never consumes frozen TSDF, candidate, feature, or score caches.  Frozen source
artifacts are opened only after an execution has finished, to decide whether
that route is eligible to be timed as the formal route.

The retained formal functions do not expose timing seams inside HiFi inference
or TSDF construction.  Those operations are therefore reported as combined
stages, with the missing split recorded in ``stage_contract``.  No zero or
subtracted pseudo-timings are manufactured.
"""

from __future__ import annotations

# CPU thread variables must be fixed before importing numerical libraries.
# ruff: noqa: E402

import os

# The preregistration fixes eight CPU threads when a formal count is absent.
# These variables must be set before NumPy/Torch/Open3D are imported.
for _thread_variable in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[_thread_variable] = "8"

import argparse
import hashlib
import json
import math
import resource
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Literal, Mapping, Protocol, Sequence

import numpy as np
import pandas as pd
from PIL import Image
from scipy.io import loadmat

from graspnet6d.contracts import Candidate6D
from graspnet6d.feature_extraction import RuntimeObservation, extract_candidate_features
from graspnet6d.formal_inputs import (
    _load_candidate_bundle,
    formal_feature_paths,
    load_committed_formal_feature_table,
    load_committed_oracle_mask,
    load_committed_predicted_mask,
    oracle_mask_paths,
    predicted_mask_paths,
)
from graspnet6d.geometry import CameraIntrinsics
from graspnet6d.grounding import predict_grounding
from graspnet6d.io import canonical_sha256
from graspnet6d.stages import (
    GroupManifest,
    _convert_frozen_candidates,
    load_evaluator_geometry_contract,
    load_target_language_jsonl,
)
from graspnet6d.tsdf import TSDFBuildResult, build_target_centered_tsdf
from graspnet6d.vgn import (
    ExtractionConfig,
    VGNExtractionSnapshot,
    _official_adapter,
    deterministic_candidates_from_processed,
    load_frozen_vgn,
    pose_nms,
    run_vgn,
    validate_extraction_snapshot,
    validate_vgn_input,
)
from robustness_completion.common import (
    PREREGISTRATION_SHA256,
    RUN_ID,
    atomic_frame,
    atomic_json,
    require_run_dir,
    sha256_file,
    verify_preregistration,
)


SOURCE_RUN_ID = "20260819_221819_graspnet6d_vgn_lambdamart"
SOURCE_RUN_MANIFEST_SHA256 = (
    "3e00ba91042d678d2dd6ae68b52270d4a2e0f05136ab9bb74f336be7e48674ba"
)
TARGET_MANIFEST_SHA256 = (
    "b04b5f3caa7f809a06ef77c7a18a8754c34efa1b218c295aca29085b911f3880"
)
LANGUAGE_MANIFEST_SHA256 = (
    "d11b3e16335062c89a178176ca8e29c6cfe9b6c9296748f7d57207c90221dea3"
)
SPLIT_MANIFEST_SHA256 = (
    "867f2dede7505eee0c6d13c52e0df794a843064cdb1f032aab94d59de7fdc089"
)
CONDITION_SELECTION_SHA256 = (
    "c1615cbd8ffe8017bab083e4a23f218d03c60094eb145a549848af8994615cbf"
)
VGN_CHECKPOINT_SHA256 = (
    "ba3391d0805e9c9b178cd18106866313cee808ff2b654f689663e92a814cec4b"
)
HIFI_CHECKPOINT_SHA256 = (
    "b19a649326384ba4524295cd100b22e54cb9ea615174229fc310fbd6bc898601"
)
ADAPTED_CHECKPOINT_SHA256 = (
    "44d55e627d7ad97d7fa9a3d21a197c36938e775e78befb4148b31ce56c001707"
)
CLIP_CHECKPOINT_SHA256 = (
    "5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f"
)

MISSING_FORMAL_CHECKPOINT = "MISSING_FORMAL_CHECKPOINT"
MISSING_FORMAL_RANKER_STATUS = (
    "NOT_MEASURABLE_MISSING_FORMAL_LAMBDAMART_CHECKPOINT"
)
PROFILE_GROUPS_PER_SCENE = 14
EXPECTED_TEST_SCENES = 7
EXPECTED_PROFILE_GROUPS = 98
PARITY_GROUPS = 20
WARMUP_GROUPS = 5
CPU_THREADS = 8
FEATURE_VOXEL_SIZE_M = 0.005
CPU_ATOL = 1e-8
CPU_RTOL = 1e-7

Route = Literal["oracle", "adapted"]
Method = Literal["native", "raw"]
Mode = Literal["parity", "cold", "warm-disk", "warm-preloaded"]
MemoryHook = Callable[[str, Mapping[str, Any]], None]


class Runtime6DError(RuntimeError):
    """A fail-closed worker contract violation."""


class MissingFormalCheckpointError(Runtime6DError):
    """The formal three-seed LambdaMART ensemble was not serialized."""

    code = MISSING_FORMAL_CHECKPOINT


@dataclass(frozen=True, slots=True)
class SourceContract:
    repo: Path
    run_dir: Path
    target_manifest: Path
    language_manifest: Path
    split_manifest: Path
    condition_selection: Path
    geometry_contract: Path
    adaptation_evidence: Path
    adapted_checkpoint: Path
    base_hifi_checkpoint: Path
    clip_checkpoint: Path
    vgn_checkpoint: Path
    hashes: Mapping[str, str]


@dataclass(frozen=True, slots=True)
class ProfileSelection:
    groups: tuple[GroupManifest, ...]
    parity_groups: tuple[GroupManifest, ...]
    warmup_groups: tuple[GroupManifest, ...]
    ordering_sha256: str


@dataclass(frozen=True, slots=True)
class LoadedGroup:
    group: GroupManifest
    rgb: np.ndarray | None
    instance_label: np.ndarray | None
    depth_raw: np.ndarray
    factor_depth: float
    intrinsic_matrix: np.ndarray
    intrinsics: CameraIntrinsics
    T_camera_to_table: np.ndarray
    table_normal_camera: np.ndarray
    gravity_camera: np.ndarray
    source_hashes: Mapping[str, str]


@dataclass(frozen=True, slots=True)
class LoadedModels:
    vgn: Any
    hifi: Any | None
    geometry: Any
    geometry_evidence: Mapping[str, Any]
    checkpoint_hashes: Mapping[str, str]
    checkpoint_bytes: int


@dataclass(frozen=True, slots=True)
class ExecutionResult:
    group_id: str
    condition: str
    mask_probability: np.ndarray
    mask_binary: np.ndarray
    candidates: tuple[Candidate6D, ...]
    features: pd.DataFrame
    final_candidate_id: str | None
    terminal_reason: str | None
    candidate_bytes: int
    feature_bytes: int


class RuntimeBackend(Protocol):
    """Small seam used by focused tests; production uses ``Formal6DBackend``."""

    def load_models(self, route: Route) -> LoadedModels: ...

    def load_group(self, group: GroupManifest, route: Route) -> LoadedGroup: ...

    def execute_loaded(
        self,
        loaded: LoadedGroup,
        models: LoadedModels,
        route: Route,
        *,
        instrument: bool,
        include_features: bool = True,
    ) -> tuple[ExecutionResult, list[dict[str, Any]], int]: ...

    def validate_parity(
        self, result: ExecutionResult, group: GroupManifest, route: Route
    ) -> dict[str, Any]: ...


def _stable_key(group_id: str) -> tuple[str, str]:
    value = str(group_id)
    return hashlib.sha256(value.encode("utf-8")).hexdigest(), value


def _json_object(path: Path, description: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise Runtime6DError(f"missing regular {description}: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise Runtime6DError(f"invalid {description}: {path}: {error}") from error
    if not isinstance(payload, dict):
        raise Runtime6DError(f"{description} must be a JSON object: {path}")
    return payload


def _require_hash(path: Path, expected: str, description: str) -> str:
    if path.is_symlink() or not path.is_file():
        raise Runtime6DError(f"missing regular {description}: {path}")
    observed = sha256_file(path)
    if observed != expected:
        raise Runtime6DError(
            f"{description} SHA-256 mismatch: found {observed}, expected {expected}"
        )
    return observed


def validate_source_contract(repo: Path, source_run: Path | None = None) -> SourceContract:
    """Bind the worker to the exact formal run without reading its outcomes."""

    root = repo.expanduser().resolve()
    expected_run = root / "artifacts" / "graspnet6d" / SOURCE_RUN_ID
    run_dir = expected_run if source_run is None else source_run.expanduser().resolve()
    if run_dir != expected_run:
        raise Runtime6DError(f"6D source run must be {expected_run}")

    paths = {
        "run_manifest": run_dir / "run_manifest.json",
        "target_manifest": run_dir / "manifests" / "target_groups.jsonl",
        "language_manifest": run_dir / "manifests" / "language_queries.jsonl",
        "split_manifest": run_dir / "split_manifest.json",
        "condition_selection": run_dir / "predicted_condition_selection.json",
        "geometry_contract": run_dir
        / "geometry_validation"
        / "evaluator_geometry_contract.json",
        "adaptation_evidence": run_dir
        / "grounding_adaptation"
        / "adaptation_evidence.json",
        "adapted_checkpoint": run_dir
        / "grounding_adaptation"
        / "hifics_decoder_adapted.pth",
        "base_hifi_checkpoint": root
        / "HiFi_reproduction"
        / "runs"
        / "hifics_ocidvlg_hierfilm_20260727_214615"
        / "checkpoints"
        / "best.pth",
        "clip_checkpoint": Path.home() / ".cache" / "clip" / "ViT-B-16.pt",
        "vgn_checkpoint": root
        / "HiFi_reproduction"
        / "third_party"
        / "vgn"
        / "data"
        / "models"
        / "vgn_conv.pth",
    }
    hashes = {
        "run_manifest": _require_hash(
            paths["run_manifest"], SOURCE_RUN_MANIFEST_SHA256, "source run manifest"
        ),
        "target_manifest": _require_hash(
            paths["target_manifest"], TARGET_MANIFEST_SHA256, "target manifest"
        ),
        "language_manifest": _require_hash(
            paths["language_manifest"], LANGUAGE_MANIFEST_SHA256, "language manifest"
        ),
        "split_manifest": _require_hash(
            paths["split_manifest"], SPLIT_MANIFEST_SHA256, "split manifest"
        ),
        "condition_selection": _require_hash(
            paths["condition_selection"],
            CONDITION_SELECTION_SHA256,
            "condition selection",
        ),
        "vgn_checkpoint": _require_hash(
            paths["vgn_checkpoint"], VGN_CHECKPOINT_SHA256, "VGN checkpoint"
        ),
        "base_hifi_checkpoint": _require_hash(
            paths["base_hifi_checkpoint"],
            HIFI_CHECKPOINT_SHA256,
            "base HiFi checkpoint",
        ),
        "adapted_checkpoint": _require_hash(
            paths["adapted_checkpoint"],
            ADAPTED_CHECKPOINT_SHA256,
            "adapted HiFi checkpoint",
        ),
        "clip_checkpoint": _require_hash(
            paths["clip_checkpoint"], CLIP_CHECKPOINT_SHA256, "CLIP checkpoint"
        ),
    }
    for name in ("geometry_contract", "adaptation_evidence"):
        path = paths[name]
        if path.is_symlink() or not path.is_file():
            raise Runtime6DError(f"missing regular {name}: {path}")
        hashes[name] = sha256_file(path)

    manifest = _json_object(paths["run_manifest"], "source run manifest")
    if (
        manifest.get("run_id") != SOURCE_RUN_ID
        or manifest.get("status") != "COMPLETE"
        or manifest.get("formal_results_emitted") is not True
    ):
        raise Runtime6DError("source run is not the locked completed formal run")
    selection = _json_object(paths["condition_selection"], "condition selection")
    if selection.get("selected_condition") != "hifics_adapted_mask":
        raise Runtime6DError("formal selected predicted condition is no longer adapted")

    return SourceContract(
        repo=root,
        run_dir=run_dir,
        target_manifest=paths["target_manifest"],
        language_manifest=paths["language_manifest"],
        split_manifest=paths["split_manifest"],
        condition_selection=paths["condition_selection"],
        geometry_contract=paths["geometry_contract"],
        adaptation_evidence=paths["adaptation_evidence"],
        adapted_checkpoint=paths["adapted_checkpoint"],
        base_hifi_checkpoint=paths["base_hifi_checkpoint"],
        clip_checkpoint=paths["clip_checkpoint"],
        vgn_checkpoint=paths["vgn_checkpoint"],
        hashes=hashes,
    )


def select_profile_groups(contract: SourceContract) -> ProfileSelection:
    """Select 14 metadata-only groups per formal test scene, then stable-order."""

    groups = load_target_language_jsonl(
        contract.target_manifest, contract.language_manifest
    )
    split = _json_object(contract.split_manifest, "split manifest")
    test_scenes = tuple(sorted(str(value) for value in split.get("test", [])))
    if len(test_scenes) != EXPECTED_TEST_SCENES or len(set(test_scenes)) != len(
        test_scenes
    ):
        raise Runtime6DError("formal split must contain exactly seven test scenes")
    by_scene: dict[str, list[GroupManifest]] = {scene: [] for scene in test_scenes}
    for group in groups:
        scene = str(group.target["scene_id"])
        declared_split = str(group.target.get("split", ""))
        if scene in by_scene:
            if declared_split != "test":
                raise Runtime6DError(f"test scene group has split={declared_split!r}")
            by_scene[scene].append(group)
    selected: list[GroupManifest] = []
    for scene in test_scenes:
        ordered = sorted(by_scene[scene], key=lambda item: _stable_key(item.group_id))
        if len(ordered) < PROFILE_GROUPS_PER_SCENE:
            raise Runtime6DError(
                f"scene {scene} has only {len(ordered)} groups; 14 are required"
            )
        selected.extend(ordered[:PROFILE_GROUPS_PER_SCENE])
    selected.sort(key=lambda item: _stable_key(item.group_id))
    identifiers = [item.group_id for item in selected]
    if len(identifiers) != EXPECTED_PROFILE_GROUPS or len(set(identifiers)) != len(
        identifiers
    ):
        raise Runtime6DError("6D profile subset is not exactly 98 unique groups")
    ordering_hash = canonical_sha256(identifiers)
    return ProfileSelection(
        groups=tuple(selected),
        parity_groups=tuple(selected[:PARITY_GROUPS]),
        warmup_groups=tuple(selected[:WARMUP_GROUPS]),
        ordering_sha256=ordering_hash,
    )


def assert_formal_reranker_available(contract: SourceContract) -> None:
    """Always fail closed until a formally bound ensemble manifest exists."""

    expected = contract.run_dir / "ranker" / "deployment_ensemble_manifest.json"
    if not expected.is_file():
        raise MissingFormalCheckpointError(
            f"{MISSING_FORMAL_CHECKPOINT}: formal three-seed LambdaMART ensemble "
            f"checkpoint is absent ({expected}); retraining and cached-score timing "
            "are prohibited"
        )
    raise MissingFormalCheckpointError(
        f"{MISSING_FORMAL_CHECKPOINT}: an unvalidated file appeared at {expected}; "
        "the locked source manifest does not bind a deployable ensemble"
    )


def process_memory_snapshot() -> dict[str, Any]:
    """Return process-local hooks; the orchestrator samples the process tree."""

    import psutil

    process = psutil.Process(os.getpid())
    usage = resource.getrusage(resource.RUSAGE_SELF)
    max_rss = int(usage.ru_maxrss)
    if sys.platform != "darwin":
        max_rss *= 1024
    return {
        "pid": os.getpid(),
        "perf_counter_ns": time.perf_counter_ns(),
        "rss_bytes": int(process.memory_info().rss),
        "peak_rss_bytes": max_rss,
        "device": "cpu",
        "unified_memory_peak": "not directly measurable",
        "mps_allocated_bytes": None,
    }


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(b"\0")
    digest.update(canonical_sha256(list(array.shape)).encode("ascii"))
    digest.update(b"\0")
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _rotation_distance_radians(left: np.ndarray, right: np.ndarray) -> float:
    relative = np.asarray(left, dtype=np.float64).T @ np.asarray(
        right, dtype=np.float64
    )
    cosine = float(np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0))
    return float(math.acos(cosine))


def _candidate_nbytes(candidates: Sequence[Candidate6D]) -> int:
    total = 0
    for candidate in candidates:
        for name in (
            "translation_local_m",
            "rotation_local",
            "translation_camera_m",
            "rotation_camera",
            "translation_table_m",
            "rotation_table",
        ):
            total += int(np.asarray(getattr(candidate, name)).nbytes)
    return total


def _stage_row(
    stage: str,
    elapsed_ns: int | None,
    *,
    status: str = "measured",
    combined_members: Sequence[str] = (),
) -> dict[str, Any]:
    return {
        "stage": stage,
        "elapsed_ns": elapsed_ns,
        "stage_status": status,
        "combined_members": list(combined_members),
    }


def _call_timed(function: Callable[[], Any]) -> tuple[Any, int]:
    started = time.perf_counter_ns()
    value = function()
    return value, time.perf_counter_ns() - started


class Formal6DBackend:
    """Exact raw-input adapter around the retained formal implementations."""

    def __init__(self, contract: SourceContract):
        self.contract = contract
        self.extraction = ExtractionConfig()

    def load_models(self, route: Route) -> LoadedModels:
        import torch

        torch.set_num_threads(CPU_THREADS)
        vgn_model = load_frozen_vgn(
            checkpoint=self.contract.vgn_checkpoint, device="cpu"
        )
        hifi_bundle = None
        checkpoint_hashes = {
            "vgn": self.contract.hashes["vgn_checkpoint"],
        }
        checkpoint_paths = [self.contract.vgn_checkpoint]
        if route == "adapted":
            from graspnet6d.formal_inputs import load_validated_adapted_hifi

            hifi_bundle, _ = load_validated_adapted_hifi(
                self.contract.adaptation_evidence,
                adapted_checkpoint_path=self.contract.adapted_checkpoint,
                device="cpu",
                base_checkpoint_path=self.contract.base_hifi_checkpoint,
                clip_weight_path=self.contract.clip_checkpoint,
            )
            checkpoint_hashes.update(
                {
                    "base_hifi": self.contract.hashes["base_hifi_checkpoint"],
                    "adapted_hifi": self.contract.hashes["adapted_checkpoint"],
                    "clip": self.contract.hashes["clip_checkpoint"],
                }
            )
            checkpoint_paths.extend(
                (
                    self.contract.base_hifi_checkpoint,
                    self.contract.adapted_checkpoint,
                    self.contract.clip_checkpoint,
                )
            )
        geometry, evidence = load_evaluator_geometry_contract(
            self.contract.geometry_contract, evidence_policy="formal"
        )
        return LoadedModels(
            vgn=vgn_model,
            hifi=hifi_bundle,
            geometry=geometry,
            geometry_evidence=evidence,
            checkpoint_hashes=checkpoint_hashes,
            checkpoint_bytes=sum(path.stat().st_size for path in checkpoint_paths),
        )

    @staticmethod
    def _regular_path(raw: Any, description: str) -> Path:
        path = Path(str(raw)).expanduser().resolve()
        if path.is_symlink() or not path.is_file():
            raise Runtime6DError(f"missing regular {description}: {path}")
        return path

    def load_group(self, group: GroupManifest, route: Route) -> LoadedGroup:
        target = group.target
        paths = {
            "depth": self._regular_path(target["depth_path"], "depth image"),
            "meta": self._regular_path(target["meta_path"], "frame metadata"),
            "intrinsics": self._regular_path(
                target["intrinsics_path"], "camera intrinsics"
            ),
            "camera_pose": self._regular_path(
                target["camera_pose_path"], "camera poses"
            ),
            "table_transform": self._regular_path(
                target["table_transform_path"], "table transform"
            ),
        }
        if route == "oracle":
            paths["instance_label"] = self._regular_path(
                target["instance_label_path"], "instance-label image"
            )
        else:
            paths["rgb"] = self._regular_path(target["rgb_path"], "RGB image")
        source_hashes = {name: sha256_file(path) for name, path in paths.items()}

        with Image.open(paths["depth"]) as image:
            depth = np.asarray(image).copy()
        if depth.ndim != 2 or not np.issubdtype(depth.dtype, np.integer):
            raise Runtime6DError("GraspNet depth must be a two-dimensional integer image")
        rgb = None
        label = None
        if route == "oracle":
            with Image.open(paths["instance_label"]) as image:
                label = np.asarray(image).copy()
            if label.shape != depth.shape or not np.issubdtype(label.dtype, np.integer):
                raise Runtime6DError("instance label and depth dimensions differ")
        else:
            with Image.open(paths["rgb"]) as image:
                rgb = np.asarray(image.convert("RGB")).copy()
            if rgb.shape[:2] != depth.shape:
                raise Runtime6DError("RGB and depth dimensions differ")

        meta = loadmat(paths["meta"])
        factor_values = np.asarray(meta.get("factor_depth", []), dtype=np.float64).reshape(-1)
        if (
            factor_values.size != 1
            or not np.isfinite(factor_values[0])
            or factor_values[0] <= 0
        ):
            raise Runtime6DError("factor_depth must be one positive scalar")
        factor_depth = float(factor_values[0])
        intrinsic_matrix = np.asarray(
            np.load(paths["intrinsics"], allow_pickle=False), dtype=np.float64
        )
        meta_intrinsic = np.asarray(meta.get("intrinsic_matrix", []), dtype=np.float64)
        if (
            intrinsic_matrix.shape != (3, 3)
            or meta_intrinsic.shape != (3, 3)
            or not np.allclose(intrinsic_matrix, meta_intrinsic, atol=1e-6, rtol=0)
        ):
            raise Runtime6DError("meta intrinsic_matrix disagrees with camK.npy")
        intrinsics = CameraIntrinsics(
            fx=float(intrinsic_matrix[0, 0]),
            fy=float(intrinsic_matrix[1, 1]),
            cx=float(intrinsic_matrix[0, 2]),
            cy=float(intrinsic_matrix[1, 2]),
            width=int(depth.shape[1]),
            height=int(depth.shape[0]),
        )
        camera_poses = np.asarray(
            np.load(paths["camera_pose"], allow_pickle=False), dtype=np.float64
        )
        frame_id = int(target["frame_id"])
        if (
            camera_poses.ndim != 3
            or camera_poses.shape[1:] != (4, 4)
            or not 0 <= frame_id < len(camera_poses)
        ):
            raise Runtime6DError("camera poses do not contain the requested frame")
        table = np.asarray(
            np.load(paths["table_transform"], allow_pickle=False), dtype=np.float64
        )
        if table.shape != (4, 4):
            raise Runtime6DError("table transform must be 4x4")
        camera_to_table = table @ camera_poses[frame_id]
        if not np.isfinite(camera_to_table).all() or not np.allclose(
            camera_to_table[3], [0.0, 0.0, 0.0, 1.0], atol=1e-7
        ):
            raise Runtime6DError("camera-to-table transform is invalid")
        table_normal_camera = camera_to_table[:3, :3].T @ np.array(
            [0.0, 0.0, 1.0]
        )
        return LoadedGroup(
            group=group,
            rgb=rgb,
            instance_label=label,
            depth_raw=depth,
            factor_depth=factor_depth,
            intrinsic_matrix=intrinsic_matrix,
            intrinsics=intrinsics,
            T_camera_to_table=camera_to_table,
            table_normal_camera=table_normal_camera,
            gravity_camera=-table_normal_camera,
            source_hashes=source_hashes,
        )

    @staticmethod
    def _observation(
        loaded: LoadedGroup, probability: np.ndarray, condition: str
    ) -> RuntimeObservation:
        depth_m = loaded.depth_raw.astype(np.float64) / loaded.factor_depth
        if not np.isfinite(depth_m).all() or not np.any(depth_m > 0):
            raise Runtime6DError("depth conversion produced no positive scene depth")
        rows, columns = np.nonzero(depth_m > 0)
        z = depth_m[rows, columns]
        full_scene = np.column_stack(
            (
                (columns - loaded.intrinsics.cx) * z / loaded.intrinsics.fx,
                (rows - loaded.intrinsics.cy) * z / loaded.intrinsics.fy,
                z,
            )
        ).astype(np.float64, copy=False)
        voxel_keys = np.floor(full_scene / FEATURE_VOXEL_SIZE_M).astype(np.int64)
        _, first_indices = np.unique(voxel_keys, axis=0, return_index=True)
        first_indices.sort()
        scene = full_scene[first_indices]
        provenance = {
            "factor_depth": loaded.factor_depth,
            "full_valid_depth_point_count": int(len(full_scene)),
            "scene_point_count": int(len(scene)),
            "scene_cloud_kind": "deterministic_voxel_downsample_of_full_depth",
            "scene_cloud_content_sha256": _array_sha256(scene),
            "depth_source_sha256": loaded.source_hashes["depth"],
            "intrinsics_source_sha256": loaded.source_hashes["intrinsics"],
            "downsample": {
                "algorithm": "first_row_major_depth_point_per_floor_quantized_xyz_voxel",
                "voxel_size_m": FEATURE_VOXEL_SIZE_M,
            },
            "table_normal_derivation": "R_table_camera.T @ table_positive_z",
            "gravity_derivation": "negative_table_normal_camera",
        }
        return RuntimeObservation(
            intrinsics=loaded.intrinsics,
            mask_probability=probability,
            depth_m=depth_m,
            scene_points_camera_m=scene,
            table_normal_camera=loaded.table_normal_camera,
            gravity_camera=loaded.gravity_camera,
            grounding_condition=condition,
            mask_source_sha256=_array_sha256(probability),
            depth_source_sha256=loaded.source_hashes["depth"],
            scene_points_source_sha256=canonical_sha256(provenance),
            scene_points_source_kind="deterministic_voxel_downsample_of_full_depth",
        ).validated()

    def execute_loaded(
        self,
        loaded: LoadedGroup,
        models: LoadedModels,
        route: Route,
        *,
        instrument: bool,
        include_features: bool = True,
    ) -> tuple[ExecutionResult, list[dict[str, Any]], int]:
        rows: list[dict[str, Any]] = []
        outer_start = time.perf_counter_ns()
        condition = "oracle_gt_mask" if route == "oracle" else "hifics_adapted_mask"

        if route == "oracle":
            if loaded.instance_label is None:
                raise Runtime6DError("oracle route lacks its raw instance-label input")

            def oracle_mask() -> tuple[np.ndarray, np.ndarray]:
                instance = int(loaded.group.target["target_instance_label"])
                binary = loaded.instance_label == instance
                if instance <= 0 or not binary.any():
                    raise Runtime6DError("target instance is absent from the label image")
                return binary.astype(np.float32), binary

            if instrument:
                (probability, binary), elapsed = _call_timed(oracle_mask)
                rows.append(_stage_row("gt_mask_load", elapsed))
            else:
                probability, binary = oracle_mask()
        else:
            if loaded.rgb is None or models.hifi is None:
                raise Runtime6DError("adapted route lacks RGB input or HiFi model")

            def adapted_mask() -> tuple[np.ndarray, np.ndarray]:
                prediction = predict_grounding(
                    models.hifi,
                    loaded.rgb,
                    str(loaded.group.language["query"]),
                )
                probability_value = np.asarray(
                    prediction.native_probability, dtype=np.float32
                )
                binary_value = np.asarray(prediction.native_mask, dtype=bool)
                if probability_value.shape != loaded.depth_raw.shape:
                    raise Runtime6DError("adapted mask and depth dimensions differ")
                return probability_value, binary_value

            if instrument:
                (probability, binary), elapsed = _call_timed(adapted_mask)
                rows.append(
                    _stage_row(
                        "hifics_inference_and_mask_postprocess",
                        elapsed,
                        status="combined_formal_api",
                        combined_members=("hifics_inference", "mask_postprocess"),
                    )
                )
            else:
                probability, binary = adapted_mask()

        valid_target_depth = int(np.count_nonzero(binary & (loaded.depth_raw > 0)))
        if valid_target_depth == 0:
            if route == "oracle":
                raise Runtime6DError("oracle mask contains no valid target depth")
            reason = "empty_predicted_mask" if not binary.any() else "no_valid_predicted_mask_depth"
            result = ExecutionResult(
                group_id=loaded.group.group_id,
                condition=condition,
                mask_probability=probability,
                mask_binary=binary,
                candidates=(),
                features=pd.DataFrame(),
                final_candidate_id=None,
                terminal_reason=reason,
                candidate_bytes=0,
                feature_bytes=0,
            )
            outer_ns = time.perf_counter_ns() - outer_start
            return result, rows, outer_ns

        def construct_tsdf() -> TSDFBuildResult:
            return build_target_centered_tsdf(
                loaded.depth_raw,
                binary,
                loaded.intrinsic_matrix,
                T_camera_to_table=loaded.T_camera_to_table,
                depth_scale=loaded.factor_depth,
                source_view_id=int(loaded.group.target["frame_id"]),
            )

        if instrument:
            tsdf_result, elapsed = _call_timed(construct_tsdf)
            rows.append(
                _stage_row(
                    "depth_workspace_tsdf_combined",
                    elapsed,
                    status="combined_formal_api",
                    combined_members=(
                        "depth_backprojection",
                        "workspace_construction",
                        "tsdf_integration",
                    ),
                )
            )
        else:
            tsdf_result = construct_tsdf()

        def infer_vgn() -> Any:
            return run_vgn(tsdf_result.tsdf, models.vgn, device="cpu")

        if instrument:
            raw_outputs, elapsed = _call_timed(infer_vgn)
            rows.append(_stage_row("vgn_inference", elapsed))
        else:
            raw_outputs = infer_vgn()

        def decode() -> tuple[Any, ...]:
            adapter = _official_adapter()
            quality, rotation, width = adapter.process_official(
                validate_vgn_input(tsdf_result.tsdf),
                raw_outputs.quality,
                raw_outputs.rotation_xyzw,
                raw_outputs.width_voxels,
            )
            return tuple(
                deterministic_candidates_from_processed(
                    quality,
                    rotation,
                    width,
                    group_id=loaded.group.group_id,
                    T_local_to_camera=tsdf_result.T_local_to_camera,
                    T_local_to_table=tsdf_result.T_local_to_table,
                )[: self.extraction.pre_nms_max_candidates]
            )

        if instrument:
            pre_nms, elapsed = _call_timed(decode)
            rows.append(_stage_row("candidate_decode", elapsed))
        else:
            pre_nms = decode()

        def suppress_and_convert() -> tuple[Candidate6D, ...]:
            frozen = tuple(
                pose_nms(pre_nms, self.extraction)[: self.extraction.frozen_top_k]
            )
            snapshot = validate_extraction_snapshot(
                VGNExtractionSnapshot(pre_nms, frozen),
                group_id=loaded.group.group_id,
                config=self.extraction,
            )
            converted, _ = _convert_frozen_candidates(
                snapshot.frozen_candidates,
                models.geometry,
                models.geometry_evidence,
            )
            return tuple(converted)

        if instrument:
            candidates, elapsed = _call_timed(suppress_and_convert)
            rows.append(_stage_row("pose_nms", elapsed))
        else:
            candidates = suppress_and_convert()

        def features() -> pd.DataFrame:
            observation = self._observation(loaded, probability, condition)
            return extract_candidate_features(candidates, observation)

        if include_features:
            if instrument:
                feature_frame, elapsed = _call_timed(features)
                rows.append(_stage_row("runtime_feature_extraction", elapsed))
            else:
                feature_frame = features()
        else:
            feature_frame = pd.DataFrame()
            if instrument:
                rows.append(
                    _stage_row(
                        "runtime_feature_extraction",
                        None,
                        status="not_applicable_native",
                    )
                )

        def select_native() -> str | None:
            if not candidates:
                return None
            return min(
                candidates, key=lambda item: (item.native_rank, item.candidate_id)
            ).candidate_id

        if instrument:
            final_candidate, elapsed = _call_timed(select_native)
            rows.append(_stage_row("final_selection", elapsed))
        else:
            final_candidate = select_native()
        result = ExecutionResult(
            group_id=loaded.group.group_id,
            condition=condition,
            mask_probability=probability,
            mask_binary=binary,
            candidates=candidates,
            features=feature_frame,
            final_candidate_id=final_candidate,
            terminal_reason=None,
            candidate_bytes=_candidate_nbytes(candidates),
            feature_bytes=(
                int(feature_frame.memory_usage(index=True, deep=True).sum())
                if include_features
                else 0
            ),
        )
        outer_ns = time.perf_counter_ns() - outer_start
        return result, rows, outer_ns

    def validate_parity(
        self, result: ExecutionResult, group: GroupManifest, route: Route
    ) -> dict[str, Any]:
        condition = "oracle_gt_mask" if route == "oracle" else "hifics_adapted_mask"
        group_id = group.group_id
        if route == "oracle":
            _, mask_sidecar = oracle_mask_paths(self.contract.run_dir, group_id)
            frozen_mask = load_committed_oracle_mask(
                mask_sidecar,
                expected_group_id=group_id,
                expected_instance_label_path=group.target["instance_label_path"],
                expected_target_instance_label=int(group.target["target_instance_label"]),
            )
            probability_equal = bool(
                np.array_equal(result.mask_probability, frozen_mask.probability)
            )
        else:
            _, _, mask_sidecar = predicted_mask_paths(
                self.contract.run_dir, condition, group_id
            )
            frozen_mask = load_committed_predicted_mask(
                mask_sidecar,
                expected_group_id=group_id,
                expected_condition="hifics_adapted_mask",
            )
            probability_equal = bool(
                np.allclose(
                    result.mask_probability,
                    frozen_mask.probability,
                    atol=CPU_ATOL,
                    rtol=CPU_RTOL,
                    equal_nan=True,
                )
            )
        binary_equal = bool(np.array_equal(result.mask_binary, frozen_mask.binary_mask))

        candidate_path = (
            self.contract.run_dir
            / "vgn_candidates"
            / condition
            / f"{formal_feature_paths(self.contract.run_dir, condition, group_id)[0].stem}.json"
        )
        frozen_candidates, _, _, _ = _load_candidate_bundle(candidate_path, group_id)
        actual_ids = [item.candidate_id for item in result.candidates]
        frozen_ids = [item.candidate_id for item in frozen_candidates]
        ids_equal = actual_ids == frozen_ids
        max_translation = 0.0
        max_rotation = 0.0
        max_width = 0.0
        if ids_equal:
            for actual, frozen in zip(result.candidates, frozen_candidates, strict=True):
                max_translation = max(
                    max_translation,
                    float(
                        np.linalg.norm(
                            np.asarray(actual.translation_camera_m)
                            - np.asarray(frozen.translation_camera_m)
                        )
                    ),
                )
                max_rotation = max(
                    max_rotation,
                    _rotation_distance_radians(
                        actual.rotation_camera, frozen.rotation_camera
                    ),
                )
                max_width = max(max_width, abs(actual.width_m - frozen.width_m))
        pose_equal = bool(
            ids_equal
            and max_translation <= 1e-5
            and max_rotation <= 1e-4
            and max_width <= 1e-5
        )
        top1_equal = result.final_candidate_id == (
            frozen_ids[0] if frozen_ids else None
        )

        _, feature_sidecar = formal_feature_paths(
            self.contract.run_dir, condition, group_id
        )
        frozen_features = load_committed_formal_feature_table(
            feature_sidecar,
            expected_group_id=group_id,
            expected_condition=condition,
            expected_candidate_ids=actual_ids,
        )
        numeric = frozen_features.iloc[:, 5:]
        schema_equal = list(result.features.columns) == list(numeric.columns)
        features_equal = bool(
            schema_equal
            and len(result.features) == len(numeric)
            and np.allclose(
                result.features.to_numpy(dtype=np.float64),
                numeric.to_numpy(dtype=np.float64),
                atol=CPU_ATOL,
                rtol=CPU_RTOL,
                equal_nan=True,
            )
        )
        passed = all(
            (
                probability_equal,
                binary_equal,
                ids_equal,
                pose_equal,
                top1_equal,
                features_equal,
            )
        )
        return {
            "group_id": group_id,
            "route": route,
            "condition": condition,
            "passed": passed,
            "probability_equal": probability_equal,
            "binary_mask_equal": binary_equal,
            "candidate_ids_equal": ids_equal,
            "top1_equal": top1_equal,
            "feature_schema_equal": schema_equal,
            "features_equal": features_equal,
            "candidate_count_actual": len(actual_ids),
            "candidate_count_frozen": len(frozen_ids),
            "max_translation_error_m": max_translation if ids_equal else None,
            "max_rotation_error_rad": max_rotation if ids_equal else None,
            "max_width_error_m": max_width if ids_equal else None,
        }


def _worker_root(run_dir: Path, route: Route, method: Method) -> Path:
    root = (run_dir / "runtime_full" / "workers" / "6d" / route / method).resolve()
    allowed = run_dir.resolve()
    if allowed not in root.parents:
        raise Runtime6DError("worker output escaped the child run")
    root.mkdir(parents=True, exist_ok=True)
    return root


def _stage_contract(route: Route, method: Method) -> dict[str, Any]:
    mask_required = (
        ["gt_mask_load"]
        if route == "oracle"
        else ["hifics_inference", "mask_postprocess"]
    )
    required = [
        "input_io",
        *mask_required,
        "depth_backprojection",
        "workspace_construction",
        "tsdf_integration",
        "vgn_inference",
        "candidate_decode",
        "pose_nms",
        "runtime_feature_extraction",
        "reranker",
        "gate",
        "final_selection",
    ]
    combined = {
        "depth_workspace_tsdf_combined": [
            "depth_backprojection",
            "workspace_construction",
            "tsdf_integration",
        ]
    }
    if route == "adapted":
        combined["hifics_inference_and_mask_postprocess"] = [
            "hifics_inference",
            "mask_postprocess",
        ]
    return {
        "required_stages": required,
        "measured_combined_stages": combined,
        "stage_profile_complete": False,
        "incomplete_reason": (
            "formal predict_grounding/build_target_centered_tsdf APIs do not expose "
            "non-overlapping internal timing seams; combined values are reported"
        ),
        "reranker_stage": (
            "not_applicable_native" if method == "native" else MISSING_FORMAL_CHECKPOINT
        ),
        "runtime_feature_extraction_stage": (
            "not_applicable_native" if method == "native" else "measured"
        ),
        "gate_stage": "not_applicable_native" if method == "native" else "blocked",
    }


def _selection_manifest(selection: ProfileSelection) -> dict[str, Any]:
    by_scene: dict[str, list[str]] = {}
    for group in selection.groups:
        by_scene.setdefault(str(group.target["scene_id"]), []).append(group.group_id)
    return {
        "schema_version": "runtime_6d_profile_subset_v1",
        "selection_inputs": "scene_id and SHA-256(group_id) only",
        "groups_per_scene": PROFILE_GROUPS_PER_SCENE,
        "group_count": len(selection.groups),
        "ordered_group_ids": [item.group_id for item in selection.groups],
        "ordering_sha256": selection.ordering_sha256,
        "scene_group_ids": by_scene,
        "parity_group_ids": [item.group_id for item in selection.parity_groups],
        "warmup_group_ids": [item.group_id for item in selection.warmup_groups],
    }


def _emit_memory(
    event: str,
    events: list[dict[str, Any]],
    hook: MemoryHook | None,
    **extra: Any,
) -> None:
    record = {"event": event, **process_memory_snapshot(), **extra}
    events.append(record)
    if hook is not None:
        hook(event, record)


def _write_result(path: Path, payload: Mapping[str, Any]) -> None:
    if path.suffix.lower() != ".json":
        raise Runtime6DError("--output must name a JSON file")
    atomic_json(path, dict(payload))


def _validate_output_path(output: Path, run_dir: Path, worker_root: Path) -> Path:
    path = output.expanduser().resolve()
    if run_dir.resolve() not in path.parents or worker_root.resolve() not in path.parents:
        raise Runtime6DError("--output must remain inside this route/method worker directory")
    return path


def _parity_complete(path: Path) -> bool:
    if not path.is_file():
        return False
    frame = pd.read_parquet(path)
    return bool(
        len(frame) == PARITY_GROUPS
        and frame["group_id"].nunique() == PARITY_GROUPS
        and frame["passed"].astype(bool).all()
    )


def _base_status(
    contract: SourceContract,
    selection: ProfileSelection,
    route: Route,
    method: Method,
    mode: Mode,
) -> dict[str, Any]:
    return {
        "schema_version": "runtime_6d_worker_status_v1",
        "run_id": RUN_ID,
        "source_run_id": SOURCE_RUN_ID,
        "route": route,
        "method": method,
        "mode": mode,
        "device": "cpu",
        "threads": CPU_THREADS,
        "batch_size": 1,
        "cache_disabled": True,
        "candidate_cache_used_for_timing": False,
        "feature_cache_used_for_timing": False,
        "score_cache_used_for_timing": False,
        "preregistration_sha256": PREREGISTRATION_SHA256,
        "worker_implementation_sha256": sha256_file(Path(__file__)),
        "source_hashes": dict(contract.hashes),
        "subset_ordering_sha256": selection.ordering_sha256,
        "stage_contract": _stage_contract(route, method),
        "complete_deployment": False,
        "pid": os.getpid(),
    }


def run_runtime_6d_worker(
    repo: Path,
    run_dir: Path,
    *,
    route: Route,
    method: Method = "native",
    mode: Mode = "parity",
    output: Path | None = None,
    max_samples: int | None = None,
    parity_only: bool = False,
    resume: bool = False,
    memory_hook: MemoryHook | None = None,
    backend: RuntimeBackend | None = None,
) -> dict[str, Any]:
    """Run one route/method/mode in the current fresh process.

    ``max_samples`` is diagnostic only.  Any bounded run remains incomplete;
    the locked full run uses 20 parity groups, five warmups, and 98 measured
    groups. ``parity_only`` is an alias for ``mode='parity'``.
    """

    if route not in {"oracle", "adapted"}:
        raise ValueError("route must be oracle or adapted")
    if method not in {"native", "raw"}:
        raise ValueError("method must be native or raw")
    if mode not in {"parity", "cold", "warm-disk", "warm-preloaded"}:
        raise ValueError("unsupported 6D runtime mode")
    if max_samples is not None and max_samples <= 0:
        raise ValueError("max_samples must be positive")
    if parity_only:
        mode = "parity"

    repo_path = repo.expanduser().resolve()
    locked_run = require_run_dir(repo_path, run_dir)
    verify_preregistration(locked_run)  # must precede model construction/inference
    contract = validate_source_contract(repo_path)
    selection = select_profile_groups(contract)
    worker_root = _worker_root(locked_run, route, method)
    output_path = _validate_output_path(
        output if output is not None else worker_root / f"{mode}_result.json",
        locked_run,
        worker_root,
    )
    atomic_json(worker_root / "subset_manifest.json", _selection_manifest(selection))
    status = _base_status(contract, selection, route, method, mode)
    segment_id = canonical_sha256(
        {
            "pid": os.getpid(),
            "started_perf_counter_ns": time.perf_counter_ns(),
            "route": route,
            "method": method,
            "mode": mode,
        }
    )[:20]
    status["worker_segment_id"] = segment_id
    status["resume"] = bool(resume)
    diagnostic_suffix = (
        "" if max_samples is None else f"_diagnostic_{int(max_samples)}"
    )
    memory_events: list[dict[str, Any]] = []

    if method == "raw":
        try:
            assert_formal_reranker_available(contract)
        except MissingFormalCheckpointError as error:
            status.update(
                {
                    "status": MISSING_FORMAL_RANKER_STATUS,
                    "blocker_code": error.code,
                    "blocker": str(error),
                    "complete_deployment": False,
                    "timed_samples": 0,
                }
            )
            _write_result(output_path, status)
            atomic_json(worker_root / "route_status.json", status)
            return status
        raise AssertionError("unreachable")

    implementation = backend if backend is not None else Formal6DBackend(contract)
    _emit_memory("before_model_load", memory_events, memory_hook, route=route, mode=mode)
    load_started = time.perf_counter_ns()
    models = implementation.load_models(route)
    model_load_ns = time.perf_counter_ns() - load_started
    _emit_memory("after_model_load", memory_events, memory_hook, route=route, mode=mode)
    status.update(
        {
            "model_load_ns": model_load_ns,
            "checkpoint_hashes": dict(models.checkpoint_hashes),
            "checkpoint_bytes": models.checkpoint_bytes,
        }
    )

    formal_parity_path = worker_root / "parity_results.parquet"
    parity_path = worker_root / f"parity_results{diagnostic_suffix}.parquet"
    if mode == "parity":
        parity_limit = PARITY_GROUPS if max_samples is None else min(PARITY_GROUPS, max_samples)
        parity_records: list[dict[str, Any]] = []
        for index, group in enumerate(selection.parity_groups[:parity_limit]):
            _emit_memory(
                "before_inference",
                memory_events,
                memory_hook,
                group_id=group.group_id,
                phase="parity",
            )
            loaded = implementation.load_group(group, route)
            result, _, _ = implementation.execute_loaded(
                loaded, models, route, instrument=False
            )
            # Frozen comparison artifacts are opened only after fresh execution.
            record = implementation.validate_parity(result, group, route)
            record.update(
                {
                    "parity_index": index,
                    "method": method,
                    "mode": mode,
                    "sample_id": group.group_id,
                    "device": "cpu",
                    "status": "passed" if bool(record["passed"]) else "failed",
                }
            )
            parity_records.append(record)
            _emit_memory(
                "after_inference",
                memory_events,
                memory_hook,
                group_id=group.group_id,
                phase="parity",
            )
            atomic_frame(parity_path, pd.DataFrame(parity_records))
            atomic_json(worker_root / "memory_events.json", memory_events)
            if not bool(record["passed"]):
                break
        passed = bool(parity_records) and all(bool(item["passed"]) for item in parity_records)
        complete = passed and len(parity_records) == PARITY_GROUPS
        status.update(
            {
                "status": "PARITY_COMPLETE" if complete else "PARITY_DIAGNOSTIC_OR_FAILED",
                "parity_passed": passed,
                "parity_complete": complete,
                "parity_groups_checked": len(parity_records),
                "complete_deployment": False,
                "output_paths": [str(parity_path)],
            }
        )
        _write_result(output_path, status)
        atomic_json(
            worker_root
            / ("route_status.json" if not diagnostic_suffix else f"parity{diagnostic_suffix}_status.json"),
            status,
        )
        return status

    diagnostic = max_samples is not None
    if not diagnostic and not _parity_complete(formal_parity_path):
        raise Runtime6DError(
            "20-group parity must pass before full cold/warm timing"
        )
    groups = selection.groups
    if max_samples is not None:
        groups = groups[: min(len(groups), max_samples)]

    if mode == "cold":
        group = groups[0]
        _emit_memory(
            "before_inference", memory_events, memory_hook, group_id=group.group_id, phase="cold"
        )
        started = time.perf_counter_ns()
        loaded = implementation.load_group(group, route)
        result, _, _ = implementation.execute_loaded(
            loaded, models, route, instrument=False, include_features=False
        )
        first_output_ns = time.perf_counter_ns() - started
        _emit_memory(
            "after_inference", memory_events, memory_hook, group_id=group.group_id, phase="cold"
        )
        status.update(
            {
                "status": "COMPLETE" if not diagnostic else "DIAGNOSTIC_COMPLETE",
                "group_id": group.group_id,
                "first_complete_output_ns": first_output_ns,
                "whole_elapsed_ns": first_output_ns,
                "sample_id": group.group_id,
                "cold_inner_ns": model_load_ns + first_output_ns,
                "final_candidate_id_present": result.final_candidate_id is not None,
                "candidate_count": len(result.candidates),
                "candidate_bytes": result.candidate_bytes,
                "feature_bytes": result.feature_bytes,
                "complete_deployment": not diagnostic,
            }
        )
        atomic_json(worker_root / "memory_events.json", memory_events)
        _write_result(output_path, status)
        return status

    preloaded: dict[str, LoadedGroup] = {}
    if mode == "warm-preloaded":
        for group in groups:
            preloaded[group.group_id] = implementation.load_group(group, route)

    warmups = tuple(group for group in selection.warmup_groups if group in groups)
    if len(groups) >= WARMUP_GROUPS and len(warmups) != WARMUP_GROUPS:
        raise Runtime6DError("warmup group selection is not five distinct measured groups")
    for group in warmups:
        loaded = (
            preloaded[group.group_id]
            if mode == "warm-preloaded"
            else implementation.load_group(group, route)
        )
        implementation.execute_loaded(
            loaded, models, route, instrument=False, include_features=False
        )

    whole_path = worker_root / f"{mode}{diagnostic_suffix}_whole_timings.parquet"
    stage_path = worker_root / f"{mode}{diagnostic_suffix}_stage_timings.parquet"
    if whole_path.exists() != stage_path.exists():
        raise Runtime6DError("partial timing checkpoint lacks its paired parquet")
    if whole_path.exists() and not resume:
        raise Runtime6DError(
            f"timing output already exists; pass --resume or use a new child run: {whole_path}"
        )
    whole_records = (
        pd.read_parquet(whole_path).to_dict("records") if whole_path.exists() else []
    )
    stage_records = (
        pd.read_parquet(stage_path).to_dict("records") if stage_path.exists() else []
    )
    if whole_records:
        if any(int(item.get("feature_bytes", -1)) != 0 for item in whole_records):
            raise Runtime6DError(
                "native timing checkpoint predates the no-feature deployment contract"
            )
        feature_rows = [
            item
            for item in stage_records
            if str(item.get("stage")) == "runtime_feature_extraction"
        ]
        if (
            len(feature_rows) != len(whole_records)
            or any(not pd.isna(item.get("elapsed_ns")) for item in feature_rows)
            or any(
                str(item.get("stage_status")) != "not_applicable_native"
                for item in feature_rows
            )
        ):
            raise Runtime6DError(
                "native timing checkpoint contains measured feature extraction"
            )
    completed_ids = {str(item["sample_id"]) for item in whole_records}
    if len(completed_ids) != len(whole_records):
        raise Runtime6DError("whole timing checkpoint has duplicate sample IDs")
    expected_prefix = [group.group_id for group in groups[: len(completed_ids)]]
    observed_prefix = [str(item["sample_id"]) for item in whole_records]
    if observed_prefix != expected_prefix:
        raise Runtime6DError("timing checkpoint is not the locked subset prefix")
    stage_ids = {str(item["sample_id"]) for item in stage_records}
    if stage_ids != completed_ids:
        raise Runtime6DError("whole/stage timing checkpoints cover different samples")
    pending_groups = [group for group in groups if group.group_id not in completed_ids]
    for sample_index, group in enumerate(groups):
        if group.group_id in completed_ids:
            continue
        _emit_memory(
            "before_inference",
            memory_events,
            memory_hook,
            group_id=group.group_id,
            phase=mode,
        )
        whole_started = time.perf_counter_ns()
        whole_loaded = (
            preloaded[group.group_id]
            if mode == "warm-preloaded"
            else implementation.load_group(group, route)
        )
        whole_result, _, _ = implementation.execute_loaded(
            whole_loaded,
            models,
            route,
            instrument=False,
            include_features=False,
        )
        whole_ns = time.perf_counter_ns() - whole_started

        if mode == "warm-preloaded":
            staged_loaded = preloaded[group.group_id]
            input_row = _stage_row(
                "input_io", None, status="preloaded_outside_timed_region"
            )
            instrumented_started = time.perf_counter_ns()
        else:
            instrumented_started = time.perf_counter_ns()
            staged_loaded, input_ns = _call_timed(
                lambda: implementation.load_group(group, route)
            )
            input_row = _stage_row("input_io", input_ns)
        staged_result, measured_rows, _ = implementation.execute_loaded(
            staged_loaded,
            models,
            route,
            instrument=True,
            include_features=False,
        )
        instrumented_wall_ns = time.perf_counter_ns() - instrumented_started
        if (
            whole_result.final_candidate_id != staged_result.final_candidate_id
            or [item.candidate_id for item in whole_result.candidates]
            != [item.candidate_id for item in staged_result.candidates]
        ):
            raise Runtime6DError("whole and instrumented executions disagree")
        rows_for_sample = [input_row, *measured_rows]
        measured_sum = sum(
            int(item["elapsed_ns"])
            for item in rows_for_sample
            if item["elapsed_ns"] is not None
        )
        overhead = max(0, instrumented_wall_ns - measured_sum)
        rows_for_sample.append(_stage_row("orchestration_overhead", overhead))
        rows_for_sample.extend(
            (
                _stage_row("reranker", None, status="not_applicable_native"),
                _stage_row("gate", None, status="not_applicable_native"),
            )
        )
        whole_records.append(
            {
                "route": route,
                "method": method,
                "mode": mode,
                "device": "cpu",
                "sample_index": sample_index,
                "worker_segment_id": segment_id,
                "sample_id": group.group_id,
                "group_id": group.group_id,
                "scene_id": str(group.target["scene_id"]),
                "elapsed_ns": whole_ns,
                "whole_elapsed_ns": whole_ns,
                "status": "measured",
                "candidate_count": len(whole_result.candidates),
                "candidate_bytes": whole_result.candidate_bytes,
                "feature_bytes": whole_result.feature_bytes,
                "terminal_reason": whole_result.terminal_reason,
            }
        )
        for row in rows_for_sample:
            stage_records.append(
                {
                    "route": route,
                    "method": method,
                    "mode": mode,
                    "device": "cpu",
                    "sample_index": sample_index,
                    "worker_segment_id": segment_id,
                    "sample_id": group.group_id,
                    "group_id": group.group_id,
                    "scene_id": str(group.target["scene_id"]),
                    "instrumented_wall_ns": instrumented_wall_ns,
                    "status": row["stage_status"],
                    "candidate_count": len(staged_result.candidates),
                    **row,
                }
            )
        _emit_memory(
            "after_inference",
            memory_events,
            memory_hook,
            group_id=group.group_id,
            phase=mode,
        )
        # Checkpoint every expensive sample; timing results, never model outputs.
        atomic_frame(whole_path, pd.DataFrame(whole_records))
        atomic_frame(stage_path, pd.DataFrame(stage_records))
        atomic_json(worker_root / "memory_events.json", memory_events)

    full = not diagnostic and len(whole_records) == EXPECTED_PROFILE_GROUPS
    status.update(
        {
            "status": "COMPLETE" if full else "DIAGNOSTIC_COMPLETE",
            "warmup_count": len(warmups),
            "warmup_group_ids": [group.group_id for group in warmups],
            "timed_samples": len(whole_records),
            "resumed_samples": len(completed_ids),
            "new_samples": len(pending_groups),
            "complete_deployment": full,
            "stage_profile_complete": False,
            "output_paths": [str(whole_path), str(stage_path)],
        }
    )
    _write_result(output_path, status)
    atomic_json(
        worker_root
        / ("route_status.json" if not diagnostic else f"{mode}{diagnostic_suffix}_status.json"),
        status,
    )
    return status


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--route", choices=("oracle", "adapted"), required=True)
    parser.add_argument("--method", choices=("native", "raw"), default="native")
    parser.add_argument(
        "--mode",
        choices=("parity", "cold", "warm-disk", "warm-preloaded"),
        default="parity",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--parity-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    print(
        json.dumps(
            {
                "event": "worker_started",
                "pid": os.getpid(),
                "route": arguments.route,
                "method": arguments.method,
                "mode": arguments.mode,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    try:
        result = run_runtime_6d_worker(
            arguments.repo,
            arguments.run_dir,
            route=arguments.route,
            method=arguments.method,
            mode=arguments.mode,
            output=arguments.output,
            max_samples=arguments.max_samples,
            parity_only=arguments.parity_only,
            resume=arguments.resume,
        )
    except Exception as error:
        print(
            json.dumps(
                {
                    "event": "worker_failed",
                    "pid": os.getpid(),
                    "error_type": type(error).__name__,
                    "error": str(error),
                },
                sort_keys=True,
            ),
            flush=True,
        )
        return 1
    print(json.dumps({"event": "worker_finished", **result}, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CPU_THREADS",
    "EXPECTED_PROFILE_GROUPS",
    "Formal6DBackend",
    "MISSING_FORMAL_CHECKPOINT",
    "MISSING_FORMAL_RANKER_STATUS",
    "MissingFormalCheckpointError",
    "PARITY_GROUPS",
    "ProfileSelection",
    "Runtime6DError",
    "WARMUP_GROUPS",
    "assert_formal_reranker_available",
    "process_memory_snapshot",
    "run_runtime_6d_worker",
    "select_profile_groups",
    "validate_source_contract",
]
