"""Frozen, deterministic adapter around the vendored official VGN snapshot.

Upstream VGN ``corl2020`` randomly permutes selected grasps before returning
them.  That order cannot define a reproducible native rank.  This module keeps
the official dense-network and process/local-maximum operations, then orders
local maxima by processed quality descending with voxel-index tie breaks before
experiment-level pose NMS and Top-K freezing.

VGN and GraspNet use different gripper-axis conventions.  Conversion to the
17-value GraspNet evaluator schema is intentionally blocked unless a real-data
validation artifact and all geometry semantics are supplied explicitly.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .device import benchmark_devices, choose_formal_device


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_VGN_ROOT = REPOSITORY_ROOT / "HiFi_reproduction" / "third_party" / "vgn"
DEFAULT_CHECKPOINT = DEFAULT_VGN_ROOT / "data" / "models" / "vgn_conv.pth"
VGN_UPSTREAM_URL = "https://github.com/ethz-asl/vgn"
VGN_UPSTREAM_COMMIT = "d7af0622433f52ae88ebe81533f12b46b33e951a"
EXPECTED_CHECKPOINT_SHA256 = (
    "ba3391d0805e9c9b178cd18106866313cee808ff2b654f689663e92a814cec4b"
)
INPUT_SHAPE = (1, 40, 40, 40)
VOXEL_SIZE_M = 0.30 / 40


class VGNContractError(RuntimeError):
    pass


class UnvalidatedEvaluatorGeometryError(VGNContractError):
    pass


@dataclass(frozen=True)
class VGNSourceProvenance:
    upstream_url: str
    declared_upstream_commit: str
    local_root: str
    source_tree_sha256: str
    verification: str


@dataclass(frozen=True)
class VGNRawOutputs:
    quality: np.ndarray
    rotation_xyzw: np.ndarray
    width_voxels: np.ndarray
    device: str


@dataclass(frozen=True)
class VGNCandidate:
    candidate_id: str
    group_id: str
    native_rank: int
    native_score: float
    translation_local_m: np.ndarray
    rotation_local_vgn: np.ndarray
    width_m: float
    voxel_index: tuple[int, int, int]
    translation_camera_m: np.ndarray | None = None
    rotation_camera_vgn: np.ndarray | None = None
    translation_table_m: np.ndarray | None = None
    rotation_table_vgn: np.ndarray | None = None
    gripper_frame: str = "vgn_(+Z_approach,+Y_closing)"

    def to_record(self) -> dict[str, object]:
        def optional(value: np.ndarray | None) -> object:
            return None if value is None else np.asarray(value).tolist()

        return {
            "candidate_id": self.candidate_id,
            "group_id": self.group_id,
            "native_rank": self.native_rank,
            "native_score": self.native_score,
            "translation_local_m": self.translation_local_m.tolist(),
            "rotation_local_vgn": self.rotation_local_vgn.tolist(),
            "translation_camera_m": optional(self.translation_camera_m),
            "rotation_camera_vgn": optional(self.rotation_camera_vgn),
            "translation_table_m": optional(self.translation_table_m),
            "rotation_table_vgn": optional(self.rotation_table_vgn),
            "width_m": self.width_m,
            "voxel_index": list(self.voxel_index),
            "gripper_frame": self.gripper_frame,
        }


_VGN_CANDIDATE_RECORD_FIELDS = frozenset(VGNCandidate.__dataclass_fields__)


def vgn_candidate_from_record(
    value: Any, *, expected_group_id: str | None = None
) -> VGNCandidate:
    """Strictly reconstruct one canonical :class:`VGNCandidate` record."""

    if not isinstance(value, dict):
        raise VGNContractError("VGN candidate record must be an object")
    if set(value) != _VGN_CANDIDATE_RECORD_FIELDS:
        raise VGNContractError(
            "VGN candidate record fields differ from the declared schema"
        )
    try:
        voxel = value["voxel_index"]
        if (
            not isinstance(voxel, list)
            or len(voxel) != 3
            or any(
                isinstance(item, bool) or not isinstance(item, int) for item in voxel
            )
        ):
            raise VGNContractError("VGN candidate voxel_index is invalid")

        def required_array(name: str, shape: tuple[int, ...]) -> np.ndarray:
            result = np.asarray(value[name], dtype=np.float64)
            if result.shape != shape or not np.isfinite(result).all():
                raise VGNContractError(f"VGN candidate {name} is invalid")
            return result

        def optional_array(name: str, shape: tuple[int, ...]) -> np.ndarray | None:
            return None if value[name] is None else required_array(name, shape)

        rank = value["native_rank"]
        if isinstance(rank, bool) or not isinstance(rank, int) or rank < 1:
            raise VGNContractError("VGN candidate native_rank is invalid")
        score = value["native_score"]
        width = value["width_m"]
        if isinstance(score, bool) or not isinstance(score, (int, float)):
            raise VGNContractError("VGN candidate native_score is invalid")
        if isinstance(width, bool) or not isinstance(width, (int, float)):
            raise VGNContractError("VGN candidate width_m is invalid")
        candidate = VGNCandidate(
            candidate_id=str(value["candidate_id"]),
            group_id=str(value["group_id"]),
            native_rank=rank,
            native_score=float(score),
            translation_local_m=required_array("translation_local_m", (3,)),
            rotation_local_vgn=required_array("rotation_local_vgn", (3, 3)),
            width_m=float(width),
            voxel_index=tuple(voxel),
            translation_camera_m=optional_array("translation_camera_m", (3,)),
            rotation_camera_vgn=optional_array("rotation_camera_vgn", (3, 3)),
            translation_table_m=optional_array("translation_table_m", (3,)),
            rotation_table_vgn=optional_array("rotation_table_vgn", (3, 3)),
            gripper_frame=str(value["gripper_frame"]),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise VGNContractError(f"invalid VGN candidate record: {error}") from error
    if expected_group_id is not None and candidate.group_id != expected_group_id:
        raise VGNContractError("VGN candidate record belongs to another group")
    if candidate.gripper_frame != "vgn_(+Z_approach,+Y_closing)":
        raise VGNContractError("VGN candidate record has an unexpected gripper frame")
    if not _valid_rotation(candidate.rotation_local_vgn):
        raise VGNContractError("VGN candidate record has an invalid local rotation")
    for rotation in (
        candidate.rotation_camera_vgn,
        candidate.rotation_table_vgn,
    ):
        if rotation is not None and not _valid_rotation(rotation):
            raise VGNContractError(
                "VGN candidate record has an invalid transformed rotation"
            )
    if candidate.to_record() != value:
        raise VGNContractError("VGN candidate record is not canonically serialized")
    return candidate


@dataclass(frozen=True)
class VGNExtractionSnapshot:
    """One deterministic extraction result from one dense VGN inference.

    ``pre_nms_candidates`` is the score/voxel ordered pool capped by
    ``ExtractionConfig.pre_nms_max_candidates``.  ``frozen_candidates`` is the
    immutable pose-NMS/Top-K comparison pool derived from that exact tuple.
    Keeping both is what makes Top-K sensitivity reproducible without another
    network call or a second decoding pass.
    """

    pre_nms_candidates: tuple[VGNCandidate, ...]
    frozen_candidates: tuple[VGNCandidate, ...]


@dataclass(frozen=True)
class ExtractionConfig:
    pre_nms_max_candidates: int = 100
    frozen_top_k: int = 50
    translation_threshold_m: float = 0.015
    rotation_threshold_deg: float = 15.0
    width_threshold_m: float = 0.010

    def validate(self) -> None:
        if self.pre_nms_max_candidates <= 0 or self.frozen_top_k <= 0:
            raise ValueError("candidate limits must be positive")
        if (
            min(
                self.translation_threshold_m,
                self.rotation_threshold_deg,
                self.width_threshold_m,
            )
            < 0
        ):
            raise ValueError("NMS thresholds must be non-negative")


@dataclass(frozen=True)
class EvaluatorGeometryContract:
    """Explicit gate for VGN-gripper to GraspNet-gripper conversion."""

    validated: bool
    validation_artifact: str
    R_vgn_gripper_to_graspnet_gripper: np.ndarray
    height_m: float
    depth_m: float

    def validate(self) -> None:
        if not self.validated or not self.validation_artifact.strip():
            raise UnvalidatedEvaluatorGeometryError(
                "VGN-to-GraspNet axes, height, and depth require a passed real-data "
                "visual/evaluator parity artifact before formal labeling"
            )
        rotation = np.asarray(self.R_vgn_gripper_to_graspnet_gripper, dtype=np.float64)
        if rotation.shape != (3, 3) or not _valid_rotation(rotation):
            raise ValueError("gripper-frame conversion must be a valid 3x3 rotation")
        if not np.isfinite(self.height_m) or not np.isfinite(self.depth_m):
            raise ValueError("GraspNet evaluator height/depth must be finite")
        if self.height_m <= 0 or self.depth_m <= 0:
            raise ValueError("GraspNet evaluator height/depth must be positive")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tree_sha256(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*.py")):
        digest.update(str(path.relative_to(root)).encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def ensure_vgn_source(vgn_root: Path | str = DEFAULT_VGN_ROOT) -> VGNSourceProvenance:
    """Load only the expected vendored source and describe its verification level."""

    root = Path(vgn_root).expanduser().resolve()
    source = root / "src"
    required = [
        source / "vgn" / name
        for name in ("detection.py", "networks.py", "perception.py")
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"VGN source snapshot is incomplete: {missing}")
    loaded = sys.modules.get("vgn")
    loaded_file = getattr(loaded, "__file__", None) if loaded is not None else None
    if loaded_file is not None:
        try:
            Path(loaded_file).resolve().relative_to(source)
        except ValueError as error:
            raise ImportError(
                f"an incompatible vgn package is already loaded from {loaded_file}"
            ) from error
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))
    try:
        tracked_change = subprocess.run(
            [
                "git",
                "-C",
                str(REPOSITORY_ROOT),
                "status",
                "--porcelain",
                "--untracked-files=no",
                "--",
                str(root.relative_to(REPOSITORY_ROOT)),
            ],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError, ValueError) as error:
        raise VGNContractError(
            "cannot verify the vendored VGN source against the repository index"
        ) from error
    if tracked_change:
        raise VGNContractError(
            f"vendored VGN source has tracked modifications: {tracked_change}"
        )
    return VGNSourceProvenance(
        upstream_url=VGN_UPSTREAM_URL,
        declared_upstream_commit=VGN_UPSTREAM_COMMIT,
        local_root=str(root),
        source_tree_sha256=_tree_sha256(source / "vgn"),
        verification=(
            "tracked vendored snapshot is clean against the repository index and declared "
            "in the repository README; nested Git metadata is absent, so the upstream "
            "commit was not re-derived locally"
        ),
    )


def validate_vgn_input(tsdf: np.ndarray) -> np.ndarray:
    value = np.asarray(tsdf)
    if value.shape != INPUT_SHAPE:
        raise ValueError(f"VGN input must have shape {INPUT_SHAPE}, got {value.shape}")
    if not np.issubdtype(value.dtype, np.number) or not np.all(np.isfinite(value)):
        raise ValueError("VGN input must be numeric and finite")
    return np.ascontiguousarray(value, dtype=np.float32)


def checkpoint_sha256(checkpoint: Path | str = DEFAULT_CHECKPOINT) -> str:
    path = Path(checkpoint).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"pretrained VGN checkpoint missing: {path}")
    return _sha256(path)


def load_frozen_vgn(
    checkpoint: Path | str = DEFAULT_CHECKPOINT,
    *,
    device: str = "cpu",
    vgn_root: Path | str = DEFAULT_VGN_ROOT,
    require_known_checkpoint: bool = True,
) -> Any:
    """Load the official ConvNet strictly, freeze all parameters, and use eval mode."""

    import torch

    ensure_vgn_source(vgn_root)
    path = Path(checkpoint).expanduser().resolve()
    digest = checkpoint_sha256(path)
    if require_known_checkpoint and digest != EXPECTED_CHECKPOINT_SHA256:
        raise VGNContractError(
            f"VGN checkpoint SHA-256 mismatch: found {digest}, expected {EXPECTED_CHECKPOINT_SHA256}"
        )
    from vgn.networks import get_network

    if len(path.stem.split("_")) < 2 or path.stem.split("_")[1].lower() != "conv":
        raise VGNContractError(
            f"checkpoint filename does not identify VGN ConvNet: {path.name}"
        )
    torch_device = torch.device(device)
    model = get_network("conv").to(torch_device)
    try:
        state = torch.load(path, map_location=torch_device, weights_only=True)
    except TypeError:
        state = torch.load(path, map_location=torch_device)
    model.load_state_dict(state, strict=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model._graspnet6d_checkpoint_path = str(path)
    model._graspnet6d_checkpoint_sha256 = digest
    model._graspnet6d_upstream_commit = VGN_UPSTREAM_COMMIT
    return model


def run_vgn(tsdf: np.ndarray, model: Any, *, device: str = "cpu") -> VGNRawOutputs:
    """Run frozen VGN and enforce its dense output contract."""

    import torch

    value = validate_vgn_input(tsdf)
    torch_device = torch.device(device)
    model = model.to(torch_device)
    model.eval()
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise VGNContractError(
            "VGN parameters must remain frozen during candidate generation"
        )
    tensor = torch.from_numpy(value).unsqueeze(0).to(torch_device)
    with torch.inference_mode():
        quality, rotation, width = model(tensor)
    outputs = VGNRawOutputs(
        quality=quality.detach().cpu().squeeze().numpy(),
        rotation_xyzw=rotation.detach().cpu().squeeze().numpy(),
        width_voxels=width.detach().cpu().squeeze().numpy(),
        device=torch_device.type,
    )
    expected = ((40, 40, 40), (4, 40, 40, 40), (40, 40, 40))
    observed = (
        outputs.quality.shape,
        outputs.rotation_xyzw.shape,
        outputs.width_voxels.shape,
    )
    if observed != expected:
        raise VGNContractError(
            f"unexpected VGN output shapes: {observed}, expected {expected}"
        )
    for name, output in (
        ("quality", outputs.quality),
        ("rotation", outputs.rotation_xyzw),
        ("width", outputs.width_voxels),
    ):
        if not np.all(np.isfinite(output)):
            raise VGNContractError(f"VGN {name} output contains NaN or Inf")
    return outputs


def _official_adapter() -> Any:
    # Reuse the already-audited SciPy-namespace compatibility wrapper.  It
    # mirrors upstream process/select operations without the random permutation.
    from HiFi_reproduction.src.grasping import vgn_adapter

    return vgn_adapter


def _valid_rotation(rotation: np.ndarray, *, atol: float = 1e-5) -> bool:
    value = np.asarray(rotation, dtype=np.float64)
    return bool(
        value.shape == (3, 3)
        and np.all(np.isfinite(value))
        and np.allclose(value.T @ value, np.eye(3), atol=atol)
        and np.isclose(np.linalg.det(value), 1.0, atol=atol)
    )


def _candidate_id(group_id: str, voxel_index: Sequence[int], score: float) -> str:
    payload = json.dumps(
        {
            "group_id": str(group_id),
            "voxel_index": [int(v) for v in voxel_index],
            "score": float(score),
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:24]


def _apply_transform(
    translation: np.ndarray, rotation: np.ndarray, transform: np.ndarray | None
) -> tuple[np.ndarray | None, np.ndarray | None]:
    if transform is None:
        return None, None
    value = np.asarray(transform, dtype=np.float64)
    if value.shape != (4, 4) or not np.all(np.isfinite(value)):
        raise ValueError("candidate frame transform must be a finite 4x4 matrix")
    return value[:3, :3] @ translation + value[:3, 3], value[:3, :3] @ rotation


def deterministic_candidates_from_processed(
    quality: np.ndarray,
    rotation_xyzw: np.ndarray,
    width_voxels: np.ndarray,
    *,
    group_id: str,
    voxel_size_m: float = VOXEL_SIZE_M,
    T_local_to_camera: np.ndarray | None = None,
    T_local_to_table: np.ndarray | None = None,
) -> list[VGNCandidate]:
    """Decode official local maxima and define a deterministic native ranking."""

    adapter = _official_adapter()
    decoded = adapter.select_official_candidates(
        np.asarray(quality).copy(),
        rotation_xyzw,
        width_voxels,
        voxel_size_m=voxel_size_m,
    )
    ordered = sorted(
        decoded,
        key=lambda candidate: (
            -float(candidate.vgn_quality),
            tuple(int(value) for value in candidate.voxel_index_ijk),
        ),
    )
    result: list[VGNCandidate] = []
    for native_rank, candidate in enumerate(ordered, start=1):
        translation = np.asarray(candidate.position_task_m, dtype=np.float64)
        rotation = np.asarray(candidate.rotation_task_3x3, dtype=np.float64)
        if not _valid_rotation(rotation):
            raise VGNContractError(
                f"decoded VGN candidate {candidate.voxel_index_ijk} has invalid rotation"
            )
        camera_t, camera_r = _apply_transform(translation, rotation, T_local_to_camera)
        table_t, table_r = _apply_transform(translation, rotation, T_local_to_table)
        result.append(
            VGNCandidate(
                candidate_id=_candidate_id(
                    group_id, candidate.voxel_index_ijk, candidate.vgn_quality
                ),
                group_id=str(group_id),
                native_rank=native_rank,
                native_score=float(candidate.vgn_quality),
                translation_local_m=translation,
                rotation_local_vgn=rotation,
                width_m=float(candidate.width_m),
                voxel_index=tuple(int(value) for value in candidate.voxel_index_ijk),
                translation_camera_m=camera_t,
                rotation_camera_vgn=camera_r,
                translation_table_m=table_t,
                rotation_table_vgn=table_r,
            )
        )
    return result


def rotation_distance_degrees(left: np.ndarray, right: np.ndarray) -> float:
    relative = np.asarray(left, dtype=np.float64).T @ np.asarray(
        right, dtype=np.float64
    )
    cosine = np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def pose_nms(
    candidates: Sequence[VGNCandidate], config: ExtractionConfig
) -> list[VGNCandidate]:
    """Deterministic experiment NMS (the width threshold is not official NMS)."""

    config.validate()
    ordered = sorted(candidates, key=lambda item: (item.native_rank, item.candidate_id))
    kept: list[VGNCandidate] = []
    for candidate in ordered:
        duplicate = any(
            np.linalg.norm(candidate.translation_local_m - prior.translation_local_m)
            <= config.translation_threshold_m
            and rotation_distance_degrees(
                candidate.rotation_local_vgn, prior.rotation_local_vgn
            )
            <= config.rotation_threshold_deg
            and abs(candidate.width_m - prior.width_m) <= config.width_threshold_m
            for prior in kept
        )
        if not duplicate:
            kept.append(candidate)
    return kept


def validate_extraction_snapshot(
    snapshot: VGNExtractionSnapshot,
    *,
    group_id: str,
    config: ExtractionConfig,
) -> VGNExtractionSnapshot:
    """Validate the exact pre-NMS -> NMS -> Top-K derivation."""

    config.validate()
    pre_nms = tuple(snapshot.pre_nms_candidates)
    frozen = tuple(snapshot.frozen_candidates)
    if len(pre_nms) > config.pre_nms_max_candidates:
        raise VGNContractError("pre-NMS snapshot exceeded pre_nms_max_candidates")
    if len(frozen) > config.frozen_top_k:
        raise VGNContractError("frozen candidate pool exceeded frozen_top_k")

    def validate_members(values: Sequence[VGNCandidate], description: str) -> None:
        identifiers = [str(item.candidate_id).strip() for item in values]
        if any(not identifier for identifier in identifiers):
            raise VGNContractError(f"{description} contains a blank candidate_id")
        if len(identifiers) != len(set(identifiers)):
            raise VGNContractError(f"{description} contains duplicate candidate IDs")
        if any(item.group_id != group_id for item in values):
            raise VGNContractError(f"{description} contains another group")
        for item in values:
            if not np.isfinite(float(item.native_score)):
                raise VGNContractError(f"{description} contains a non-finite score")
            if not np.isfinite(float(item.width_m)) or float(item.width_m) <= 0:
                raise VGNContractError(f"{description} contains an invalid width")
            if not _valid_rotation(item.rotation_local_vgn):
                raise VGNContractError(
                    f"{description} contains an invalid local rotation"
                )

    validate_members(pre_nms, "pre-NMS snapshot")
    validate_members(frozen, "frozen candidate pool")
    pre_ranks = [int(item.native_rank) for item in pre_nms]
    if pre_ranks != list(range(1, len(pre_nms) + 1)):
        raise VGNContractError(
            "pre-NMS candidates must have contiguous one-based deterministic ranks"
        )
    frozen_ranks = [int(item.native_rank) for item in frozen]
    if any(rank < 1 for rank in frozen_ranks) or frozen_ranks != sorted(
        set(frozen_ranks)
    ):
        raise VGNContractError(
            "frozen candidates must preserve increasing unique pre-NMS ranks"
        )
    by_id = {item.candidate_id: item for item in pre_nms}
    for item in frozen:
        source = by_id.get(item.candidate_id)
        if source is None or source.to_record() != item.to_record():
            raise VGNContractError(
                "frozen candidate is not an unchanged member of the pre-NMS snapshot"
            )
    expected = tuple(pose_nms(pre_nms, config)[: config.frozen_top_k])
    if [item.candidate_id for item in expected] != [
        item.candidate_id for item in frozen
    ]:
        raise VGNContractError(
            "frozen pool does not equal deterministic pose-NMS/Top-K of its pre-NMS snapshot"
        )
    return VGNExtractionSnapshot(pre_nms, frozen)


def extract_candidate_snapshot(
    tsdf: np.ndarray,
    raw_outputs: VGNRawOutputs,
    *,
    group_id: str,
    config: ExtractionConfig = ExtractionConfig(),
    T_local_to_camera: np.ndarray | None = None,
    T_local_to_table: np.ndarray | None = None,
) -> VGNExtractionSnapshot:
    """Decode one inference into hashable pre- and post-NMS candidate pools."""

    config.validate()
    adapter = _official_adapter()
    processed_quality, rotation, width = adapter.process_official(
        validate_vgn_input(tsdf),
        raw_outputs.quality,
        raw_outputs.rotation_xyzw,
        raw_outputs.width_voxels,
    )
    pre_nms = tuple(
        deterministic_candidates_from_processed(
            processed_quality,
            rotation,
            width,
            group_id=group_id,
            T_local_to_camera=T_local_to_camera,
            T_local_to_table=T_local_to_table,
        )[: config.pre_nms_max_candidates]
    )
    snapshot = VGNExtractionSnapshot(
        pre_nms_candidates=pre_nms,
        frozen_candidates=tuple(pose_nms(pre_nms, config)[: config.frozen_top_k]),
    )
    return validate_extraction_snapshot(snapshot, group_id=group_id, config=config)


def extract_frozen_candidates(
    tsdf: np.ndarray,
    raw_outputs: VGNRawOutputs,
    *,
    group_id: str,
    config: ExtractionConfig = ExtractionConfig(),
    T_local_to_camera: np.ndarray | None = None,
    T_local_to_table: np.ndarray | None = None,
) -> list[VGNCandidate]:
    """Official processing/local maxima -> deterministic rank -> NMS -> Top-K."""

    snapshot = extract_candidate_snapshot(
        tsdf,
        raw_outputs,
        group_id=group_id,
        config=config,
        T_local_to_camera=T_local_to_camera,
        T_local_to_table=T_local_to_table,
    )
    # Preserve native ranks; do not renumber after suppression.  This wrapper
    # remains for callers that only need the historical frozen comparison pool.
    return list(snapshot.frozen_candidates)


def candidate_to_graspnet_row(
    candidate: VGNCandidate,
    geometry: EvaluatorGeometryContract,
    *,
    score: float | None = None,
    object_id: int = -1,
) -> np.ndarray:
    """Convert a camera-frame VGN candidate only after real-data validation."""

    geometry.validate()
    if candidate.translation_camera_m is None or candidate.rotation_camera_vgn is None:
        raise ValueError("candidate lacks a camera-frame pose")
    conversion = np.asarray(
        geometry.R_vgn_gripper_to_graspnet_gripper, dtype=np.float64
    )
    rotation = candidate.rotation_camera_vgn @ conversion
    if not _valid_rotation(rotation):
        raise VGNContractError("converted GraspNet rotation is invalid")
    row = np.empty(17, dtype=np.float64)
    row[0] = candidate.native_score if score is None else float(score)
    row[1] = candidate.width_m
    row[2] = geometry.height_m
    row[3] = geometry.depth_m
    row[4:13] = rotation.reshape(-1)
    row[13:16] = candidate.translation_camera_m
    row[16] = int(object_id)
    return row


def benchmark_checkpoint(
    tsdf_tensors: Sequence[np.ndarray],
    *,
    checkpoint: Path | str = DEFAULT_CHECKPOINT,
    warmup: int = 1,
    repeats: int = 3,
) -> tuple[Any, Any, Any]:
    """Benchmark the same frozen checkpoint on CPU/MPS and return the decision."""

    factory = lambda device: load_frozen_vgn(checkpoint, device=device)  # noqa: E731
    cpu, mps = benchmark_devices(factory, tsdf_tensors, warmup=warmup, repeats=repeats)
    return cpu, mps, choose_formal_device(cpu, mps)
