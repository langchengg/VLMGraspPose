"""Strict OCID-VLG annotation and GT-mask adapters.

The GT mask source of truth is the original instance map selected by the
``answer_instance_value`` stored in HiFi-CS ``sample_metadata.json``.  The
exported ``ground_truth_mask_original_resolution.png`` is treated as a cache:
it is accepted only when it is exactly equal to that source-derived mask.

Query types are derived from OCID-VLG symbolic programs, never from natural
language keywords.  :data:`QUERY_TYPE_RULE_PAYLOAD` is JSON-serialisable and
is intended to be copied into experiment metadata/reports.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Sequence, TYPE_CHECKING

import numpy as np
from PIL import Image

if TYPE_CHECKING:  # Avoid importing the VGN pipeline merely to inspect annotations.
    from src.grasping.vgn_pipeline import ManifestSample


GT_EXPORT_FILENAME = "ground_truth_mask_original_resolution.png"

QUERY_TYPE_RULE_PAYLOAD: dict[str, Any] = {
    "version": "ocid_vlg_symbolic_v1",
    "source": "OCID-VLG expression record program node types",
    "semantic_operator_map": {
        "filter_color": "attribute",
        "relate": "relation",
        "locate": "location",
    },
    "structural_or_naming_operators": [
        "scene",
        "filter_category",
        "ground",
        "unique",
        "return",
    ],
    "rules_in_order": [
        {
            "if": "program is malformed or contains an unrecognised operator",
            "then": "unknown",
        },
        {
            "if": "no semantic operator is present",
            "then": "name",
        },
        {
            "if": "exactly one semantic dimension is present",
            "then": "attribute, relation, or location",
        },
        {
            "if": "two or more semantic dimensions are present",
            "then": "mixed",
        },
    ],
    "notes": [
        "filter_category and ground identify an object/category and do not by themselves add an attribute dimension",
        "repeated operators in the same semantic dimension do not make a query mixed",
        "natural-language keywords and template filenames are not consulted",
    ],
}

_SEMANTIC_OPERATOR_MAP = dict(QUERY_TYPE_RULE_PAYLOAD["semantic_operator_map"])
_STRUCTURAL_OPERATORS = frozenset(
    QUERY_TYPE_RULE_PAYLOAD["structural_or_naming_operators"]
)
_KNOWN_OPERATORS = frozenset(_SEMANTIC_OPERATOR_MAP) | _STRUCTURAL_OPERATORS


class GTOracleMappingError(RuntimeError):
    """A GT target mask could not be resolved without guessing."""

    def __init__(self, status: str, message: str):
        if status not in {"gt_oracle_ambiguous", "gt_oracle_unavailable"}:
            raise ValueError(f"invalid GT oracle error status: {status}")
        super().__init__(message)
        self.status = status


@dataclass(frozen=True)
class GTMaskReference:
    """Auditable reference to one uniquely resolved OCID-VLG target mask."""

    sample_id: str
    question_index: int
    scene_id: str
    query: str
    target_name: str | None
    target_category: str | None
    answer_instance_value: int
    prediction_sample_metadata_path: Path
    source_instance_mask_path: Path
    exported_gt_mask_path: Path
    width: int
    height: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "question_index": self.question_index,
            "scene_id": self.scene_id,
            "query": self.query,
            "target_name": self.target_name,
            "target_category": self.target_category,
            "answer_instance_value": self.answer_instance_value,
            "prediction_sample_metadata_path": str(
                self.prediction_sample_metadata_path
            ),
            "source_instance_mask_path": str(self.source_instance_mask_path),
            "exported_gt_mask_path": str(self.exported_gt_mask_path),
            "width": self.width,
            "height": self.height,
            "mapping_source": "source_instance_mask == answer_instance_value",
        }


@dataclass(frozen=True)
class QueryTypeAnnotation:
    """Result of the versioned symbolic query-type rule."""

    query_type: str
    program_operators: tuple[str, ...]
    semantic_dimensions: tuple[str, ...]
    unknown_operators: tuple[str, ...]
    rule_version: str = str(QUERY_TYPE_RULE_PAYLOAD["version"])

    def to_dict(self, *, include_rule_payload: bool = False) -> dict[str, Any]:
        result: dict[str, Any] = {
            "query_type": self.query_type,
            "program_operators": list(self.program_operators),
            "semantic_dimensions": list(self.semantic_dimensions),
            "unknown_operators": list(self.unknown_operators),
            "query_type_rule_version": self.rule_version,
        }
        if include_rule_payload:
            result["query_type_rule_payload"] = QUERY_TYPE_RULE_PAYLOAD
        return result


def _read_json_object(path: Path, *, label: str) -> Mapping[str, Any]:
    if not path.is_file():
        raise GTOracleMappingError(
            "gt_oracle_unavailable", f"{label} does not exist: {path}"
        )
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise GTOracleMappingError(
            "gt_oracle_unavailable", f"cannot read {label}: {path}: {error}"
        ) from error
    if not isinstance(value, Mapping):
        raise GTOracleMappingError(
            "gt_oracle_ambiguous", f"{label} must contain one JSON object: {path}"
        )
    return value


def _metadata_path(sample: "ManifestSample") -> Path:
    raw = sample.metadata.get("prediction_sample_metadata")
    if raw in (None, ""):
        raise GTOracleMappingError(
            "gt_oracle_unavailable",
            f"{sample.sample_id}: bundle metadata has no prediction_sample_metadata",
        )
    path = Path(str(raw)).expanduser()
    if not path.is_absolute():
        if sample.metadata_path is None:
            raise GTOracleMappingError(
                "gt_oracle_ambiguous",
                f"{sample.sample_id}: relative prediction_sample_metadata has no bundle metadata base",
            )
        path = sample.metadata_path.parent / path
    return path.resolve()


def _strict_integer(value: Any, *, field: str, sample_id: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise GTOracleMappingError(
            "gt_oracle_ambiguous",
            f"{sample_id}: {field} must be a JSON integer, got {value!r}",
        )
    return int(value)


def _load_2d_image(path: Path, *, label: str, sample_id: str) -> np.ndarray:
    if not path.is_file():
        raise GTOracleMappingError(
            "gt_oracle_unavailable", f"{sample_id}: missing {label}: {path}"
    )
    try:
        with Image.open(path) as image:
            array = np.asarray(image).copy()
    except (OSError, ValueError) as error:
        raise GTOracleMappingError(
            "gt_oracle_unavailable",
            f"{sample_id}: cannot read {label}: {path}: {error}",
        ) from error
    if array.ndim != 2:
        raise GTOracleMappingError(
            "gt_oracle_ambiguous",
            f"{sample_id}: {label} must be a single-channel HxW image, got {array.shape}",
        )
    return array


def target_category_from_name(target_name: Any) -> str | None:
    """Return the OCID-VLG category in ``<category>_<instance>``.

    The numeric suffix is part of the dataset's instance-name schema.  Values
    outside that schema return ``None`` instead of being heuristically split.
    """

    if not isinstance(target_name, str):
        return None
    match = re.fullmatch(r"(.+)_([0-9]+)", target_name)
    return match.group(1) if match else None


def resolve_gt_mask_reference(sample: "ManifestSample") -> GTMaskReference:
    """Resolve and fully validate one GT mask mapping for ``sample``.

    No directory search or stem matching is used.  The path chain must be:
    ``sample.metadata['prediction_sample_metadata']`` -> original instance-map
    path and integer answer ID -> exact exported GT sibling.
    """

    metadata_path = _metadata_path(sample)
    metadata = _read_json_object(metadata_path, label="prediction sample metadata")
    stable_id = metadata.get("stable_sample_id")
    if stable_id in (None, "") or str(stable_id) != str(sample.sample_id):
        raise GTOracleMappingError(
            "gt_oracle_ambiguous",
            f"{sample.sample_id}: stable_sample_id mismatch in {metadata_path}: {stable_id!r}",
        )
    raw_question_index = sample.row.get("question_index", sample.dataset_index)
    question_index = _strict_integer(
        raw_question_index,
        field="manifest question_index",
        sample_id=sample.sample_id,
    )
    expected_fields = {
        "question_index": question_index,
        "scene_id": str(sample.scene_id),
        "query": str(sample.instruction),
    }
    for field, expected in expected_fields.items():
        actual = metadata.get(field)
        if actual is None or str(actual) != str(expected):
            raise GTOracleMappingError(
                "gt_oracle_ambiguous",
                f"{sample.sample_id}: {field} mismatch in {metadata_path}: expected {expected!r}, got {actual!r}",
            )

    answer = _strict_integer(
        metadata.get("answer_instance_value"),
        field="answer_instance_value",
        sample_id=sample.sample_id,
    )
    raw_instance_path = metadata.get("source_instance_mask_path")
    if not isinstance(raw_instance_path, str) or not raw_instance_path.strip():
        raise GTOracleMappingError(
            "gt_oracle_unavailable",
            f"{sample.sample_id}: source_instance_mask_path is missing",
        )
    instance_path = Path(raw_instance_path).expanduser()
    if not instance_path.is_absolute():
        instance_path = metadata_path.parent / instance_path
    instance_path = instance_path.resolve()
    exported_path = (metadata_path.parent / GT_EXPORT_FILENAME).resolve()

    instance_map = _load_2d_image(
        instance_path, label="source instance mask", sample_id=sample.sample_id
    )
    source_gt = instance_map == answer
    if not bool(source_gt.any()):
        raise GTOracleMappingError(
            "gt_oracle_ambiguous",
            f"{sample.sample_id}: answer instance {answer} is absent from {instance_path}",
        )
    exported = _load_2d_image(
        exported_path, label="exported original-resolution GT mask", sample_id=sample.sample_id
    )
    exported_bool = exported != 0
    if exported.shape != instance_map.shape or not np.array_equal(exported_bool, source_gt):
        raise GTOracleMappingError(
            "gt_oracle_ambiguous",
            f"{sample.sample_id}: exported GT mask does not exactly equal source instance mask == {answer}",
        )

    source_rgb = metadata.get("source_rgb_path")
    if isinstance(source_rgb, str) and source_rgb:
        rgb_path = Path(source_rgb).expanduser()
        if not rgb_path.is_absolute():
            rgb_path = metadata_path.parent / rgb_path
        rgb = _load_2d_or_rgb_shape(rgb_path.resolve(), sample.sample_id)
        if rgb[:2] != instance_map.shape:
            raise GTOracleMappingError(
                "gt_oracle_ambiguous",
                f"{sample.sample_id}: source RGB shape {rgb[:2]} differs from GT shape {instance_map.shape}",
            )

    target_name_raw = metadata.get("target_name")
    target_name = str(target_name_raw) if target_name_raw not in (None, "") else None
    height, width = instance_map.shape
    return GTMaskReference(
        sample_id=str(sample.sample_id),
        question_index=question_index,
        scene_id=str(sample.scene_id),
        query=str(sample.instruction),
        target_name=target_name,
        target_category=target_category_from_name(target_name),
        answer_instance_value=answer,
        prediction_sample_metadata_path=metadata_path,
        source_instance_mask_path=instance_path,
        exported_gt_mask_path=exported_path,
        width=int(width),
        height=int(height),
    )


def _load_2d_or_rgb_shape(path: Path, sample_id: str) -> tuple[int, ...]:
    if not path.is_file():
        raise GTOracleMappingError(
            "gt_oracle_unavailable", f"{sample_id}: source RGB is missing: {path}"
    )
    try:
        with Image.open(path) as image:
            shape = np.asarray(image).shape
    except (OSError, ValueError) as error:
        raise GTOracleMappingError(
            "gt_oracle_unavailable",
            f"{sample_id}: cannot read source RGB {path}: {error}",
        ) from error
    if len(shape) not in {2, 3}:
        raise GTOracleMappingError(
            "gt_oracle_ambiguous", f"{sample_id}: invalid source RGB shape {shape}"
        )
    return shape


def resolve_gt_oracle_mapping(sample: "ManifestSample") -> Path:
    """Return the unique, source-validated original-resolution GT mask path."""

    return resolve_gt_mask_reference(sample).exported_gt_mask_path


def load_gt_mask(sample_or_reference: "ManifestSample | GTMaskReference") -> np.ndarray:
    """Load a validated GT target mask as an HxW boolean array."""

    reference = (
        sample_or_reference
        if isinstance(sample_or_reference, GTMaskReference)
        else resolve_gt_mask_reference(sample_or_reference)
    )
    exported = _load_2d_image(
        reference.exported_gt_mask_path,
        label="exported original-resolution GT mask",
        sample_id=reference.sample_id,
    )
    instance_map = _load_2d_image(
        reference.source_instance_mask_path,
        label="source instance mask",
        sample_id=reference.sample_id,
    )
    expected = instance_map == reference.answer_instance_value
    if exported.shape != expected.shape or not np.array_equal(exported != 0, expected):
        raise GTOracleMappingError(
            "gt_oracle_ambiguous",
            f"{reference.sample_id}: GT mask changed after mapping was resolved",
        )
    return expected


def build_gt_oracle_sample(sample: "ManifestSample") -> "ManifestSample":
    """Clone a pipeline sample with only its mask/provenance changed to GT."""

    gt_path = resolve_gt_oracle_mapping(sample)
    # load_sample_arrays verifies the value stored under this historical key.
    from src.grasping.vgn_pipeline import sha256_file

    metadata = dict(sample.metadata)
    metadata.update(
        {
            "mask_source": "ground_truth_mask_oracle",
            "prediction_mask_sha256": sha256_file(gt_path),
            "gt_oracle_source_sample_id": sample.sample_id,
        }
    )
    return replace(sample, mask_path=gt_path, metadata=metadata)


def mask_iou(predicted_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    """Compute binary-mask IoU; two empty masks have IoU 1 by convention."""

    predicted = _as_bool_mask(predicted_mask, "predicted_mask")
    target = _as_bool_mask(gt_mask, "gt_mask")
    if predicted.shape != target.shape:
        raise ValueError(
            f"predicted/GT mask shape mismatch: {predicted.shape} != {target.shape}"
        )
    intersection = int(np.count_nonzero(predicted & target))
    union = int(np.count_nonzero(predicted | target))
    return 1.0 if union == 0 else float(intersection / union)


def _as_bool_mask(mask: np.ndarray, label: str) -> np.ndarray:
    value = np.asarray(mask)
    if value.ndim == 3 and value.shape[2] == 1:
        value = value[..., 0]
    if value.ndim != 2:
        raise ValueError(f"{label} must be HxW, got {value.shape}")
    if np.issubdtype(value.dtype, np.floating) and not np.all(np.isfinite(value)):
        raise ValueError(f"{label} contains NaN or Inf")
    return value != 0


def iou_precision_indicators(iou: float) -> dict[str, int]:
    """Return per-sample indicators used to aggregate P@50 through P@90."""

    value = float(iou)
    if not np.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"IoU must be finite in [0, 1], got {iou!r}")
    return {
        f"p_at_{threshold}": int(value >= threshold / 100.0)
        for threshold in (50, 60, 70, 80, 90)
    }


def evaluate_predicted_mask(sample: "ManifestSample") -> dict[str, Any]:
    """Evaluate the sample's predicted mask against its uniquely mapped GT."""

    reference = resolve_gt_mask_reference(sample)
    predicted = _load_2d_image(
        Path(sample.mask_path), label="predicted target mask", sample_id=sample.sample_id
    )
    gt_mask = load_gt_mask(reference)
    value = mask_iou(predicted, gt_mask)
    return {
        "gt_oracle_available": True,
        "gt_mask_path": str(reference.exported_gt_mask_path),
        "gt_answer_instance_value": reference.answer_instance_value,
        "target_name": reference.target_name,
        "target_category": reference.target_category,
        "pred_mask_area_px": int(np.count_nonzero(predicted)),
        "gt_mask_area_px": int(np.count_nonzero(gt_mask)),
        "mask_iou": value,
        **iou_precision_indicators(value),
    }


def derive_query_type(program: Any) -> QueryTypeAnnotation:
    """Classify one OCID-VLG symbolic program with the documented rule."""

    if not isinstance(program, Sequence) or isinstance(program, (str, bytes)):
        return QueryTypeAnnotation("unknown", (), (), ("<malformed_program>",))
    operators: list[str] = []
    malformed = False
    for node in program:
        if not isinstance(node, Mapping) or not isinstance(node.get("type"), str):
            malformed = True
            continue
        operators.append(str(node["type"]))
    if malformed or not operators:
        unknown = tuple(sorted(set(op for op in operators if op not in _KNOWN_OPERATORS)))
        return QueryTypeAnnotation(
            "unknown",
            tuple(operators),
            (),
            unknown + (("<malformed_node>",) if malformed else ("<empty_program>",)),
        )
    unknown = tuple(sorted(set(operators) - _KNOWN_OPERATORS))
    dimensions = tuple(
        sorted({_SEMANTIC_OPERATOR_MAP[op] for op in operators if op in _SEMANTIC_OPERATOR_MAP})
    )
    if unknown:
        query_type = "unknown"
    elif not dimensions:
        query_type = "name"
    elif len(dimensions) == 1:
        query_type = dimensions[0]
    else:
        query_type = "mixed"
    return QueryTypeAnnotation(
        query_type=query_type,
        program_operators=tuple(operators),
        semantic_dimensions=dimensions,
        unknown_operators=unknown,
    )


def annotate_expression(expression: Mapping[str, Any]) -> dict[str, Any]:
    """Return query-type/category fields for one OCID-VLG expression record."""

    query = derive_query_type(expression.get("program"))
    target_name = expression.get("target")
    return {
        **query.to_dict(include_rule_payload=True),
        "target_name": target_name if isinstance(target_name, str) else None,
        "target_category": target_category_from_name(target_name),
    }


def load_expression_index(path: Path | str) -> dict[int, Mapping[str, Any]]:
    """Load ``test_expressions.json`` and reject duplicate question IDs."""

    source = Path(path).expanduser().resolve()
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read OCID-VLG expressions {source}: {error}") from error
    records = payload.get("data") if isinstance(payload, Mapping) else None
    if not isinstance(records, list):
        raise ValueError(f"OCID-VLG expressions must contain a data list: {source}")
    index: dict[int, Mapping[str, Any]] = {}
    for offset, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise ValueError(f"expression at offset {offset} is not an object")
        value = record.get("question_index")
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"invalid question_index at offset {offset}: {value!r}")
        if value in index:
            raise ValueError(f"duplicate question_index in expressions: {value}")
        index[value] = record
    return index


def expression_for_sample(
    sample: "ManifestSample", expression_index: Mapping[int, Mapping[str, Any]]
) -> Mapping[str, Any]:
    """Return the uniquely aligned expression and reject any provenance drift."""

    raw_question_index = sample.row.get("question_index", sample.dataset_index)
    if isinstance(raw_question_index, bool) or not isinstance(
        raw_question_index, (int, np.integer)
    ):
        raise ValueError(
            f"invalid manifest question_index for {sample.sample_id}: {raw_question_index!r}"
        )
    question_index = int(raw_question_index)
    record = expression_index.get(question_index)
    if record is None:
        raise ValueError(f"no expression for question_index {question_index}")
    checks = {
        "question_index": question_index,
        "image_filename": str(sample.scene_id),
        "question": str(sample.instruction),
    }
    for field, expected in checks.items():
        if record.get(field) != expected:
            raise ValueError(
                f"expression {field} mismatch for {sample.sample_id}: expected {expected!r}, got {record.get(field)!r}"
            )
    return record


def projected_uv_inside_mask(
    projected_uv: Sequence[float], mask: np.ndarray
) -> bool:
    """Test a floating projection using VGN's nearest-pixel convention."""

    target = _as_bool_mask(mask, "mask")
    uv = np.asarray(projected_uv, dtype=np.float64)
    if uv.shape != (2,) or not np.all(np.isfinite(uv)):
        return False
    height, width = target.shape
    if not (0.0 <= uv[0] < width and 0.0 <= uv[1] < height):
        return False
    u, v = np.rint(uv).astype(int)
    u = min(max(int(u), 0), width - 1)
    v = min(max(int(v), 0), height - 1)
    return bool(target[v, u])


def nearest_gt_point_distance_m(
    position_camera_m: Sequence[float], gt_points_camera_m: np.ndarray
) -> float:
    """Return Euclidean distance to the nearest valid GT target 3D point."""

    position = np.asarray(position_camera_m, dtype=np.float64)
    points = np.asarray(gt_points_camera_m, dtype=np.float64)
    if position.shape != (3,) or not np.all(np.isfinite(position)):
        raise ValueError("position_camera_m must be a finite 3-vector")
    if points.ndim != 2 or points.shape[1] != 3 or points.shape[0] == 0:
        raise ValueError("gt_points_camera_m must be a non-empty Nx3 array")
    if not np.all(np.isfinite(points)):
        raise ValueError("gt_points_camera_m must be finite")
    return float(np.sqrt(np.min(np.sum(np.square(points - position), axis=1))))


def _load_intrinsics(sample: "ManifestSample", shape: tuple[int, int]) -> dict[str, float]:
    if sample.intrinsics_path is None:
        raise GTOracleMappingError(
            "gt_oracle_unavailable",
            f"{sample.sample_id}: intrinsics are required for GT 3D diagnostics",
        )
    payload = _read_json_object(Path(sample.intrinsics_path), label="camera intrinsics")
    required = ("width", "height", "fx", "fy", "cx", "cy")
    if any(payload.get(key) is None for key in required):
        raise GTOracleMappingError(
            "gt_oracle_unavailable",
            f"{sample.sample_id}: camera intrinsics lack one of {required}",
        )
    try:
        values = {key: float(payload[key]) for key in required}
    except (TypeError, ValueError) as error:
        raise GTOracleMappingError(
            "gt_oracle_ambiguous", f"{sample.sample_id}: camera intrinsics are non-numeric"
        ) from error
    if int(values["width"]) != shape[1] or int(values["height"]) != shape[0]:
        raise GTOracleMappingError(
            "gt_oracle_ambiguous",
            f"{sample.sample_id}: intrinsics resolution {(int(values['height']), int(values['width']))} differs from depth/GT {shape}",
        )
    if values["fx"] <= 0 or values["fy"] <= 0:
        raise GTOracleMappingError(
            "gt_oracle_ambiguous", f"{sample.sample_id}: focal lengths must be positive"
        )
    return values


def annotate_top1_with_gt(
    sample: "ManifestSample",
    top1_payload: Mapping[str, Any],
    *,
    depth_unit: str = "mm",
    depth_scale: float = 1000.0,
) -> dict[str, Any]:
    """Compute GT target-consistency diagnostics for one selected VGN grasp.

    ``top1_payload`` may be the complete ``top1.json`` object or its nested
    ``candidate``.  The returned quantities are diagnostics, not success rates.
    """

    candidate = top1_payload.get("candidate", top1_payload)
    if not isinstance(candidate, Mapping):
        raise ValueError("top1 payload candidate must be an object")
    position = np.asarray(candidate.get("position_camera_m"), dtype=np.float64)
    uv = np.asarray(candidate.get("projected_uv"), dtype=np.float64)
    if position.shape != (3,) or not np.all(np.isfinite(position)):
        raise ValueError("top1 candidate lacks a finite position_camera_m")
    if uv.shape != (2,) or not np.all(np.isfinite(uv)):
        raise ValueError("top1 candidate lacks a finite projected_uv")

    gt_mask = load_gt_mask(sample)
    depth_raw = _load_2d_image(
        Path(sample.depth_path), label="depth image", sample_id=sample.sample_id
    ).astype(np.float64)
    if depth_raw.shape != gt_mask.shape:
        raise GTOracleMappingError(
            "gt_oracle_ambiguous",
            f"{sample.sample_id}: depth shape {depth_raw.shape} differs from GT {gt_mask.shape}",
        )
    if depth_unit == "mm":
        if not np.isfinite(depth_scale) or depth_scale <= 0:
            raise ValueError("depth_scale must be finite and positive")
        depth_m = depth_raw / float(depth_scale)
    elif depth_unit == "m":
        depth_m = depth_raw
    else:
        raise ValueError("depth_unit must be explicitly 'm' or 'mm'")

    intrinsics = _load_intrinsics(sample, gt_mask.shape)
    valid = gt_mask & np.isfinite(depth_m) & (depth_m > 0)
    v, u = np.nonzero(valid)
    if u.size == 0:
        raise GTOracleMappingError(
            "gt_oracle_unavailable",
            f"{sample.sample_id}: GT target has no valid metric depth points",
        )
    z = depth_m[v, u]
    points = np.column_stack(
        (
            (u - intrinsics["cx"]) * z / intrinsics["fx"],
            (v - intrinsics["cy"]) * z / intrinsics["fy"],
            z,
        )
    )

    projected_difference: float | None = None
    height, width = depth_m.shape
    if 0.0 <= uv[0] < width and 0.0 <= uv[1] < height:
        pixel_u, pixel_v = np.rint(uv).astype(int)
        pixel_u = min(max(int(pixel_u), 0), width - 1)
        pixel_v = min(max(int(pixel_v), 0), height - 1)
        observed = float(depth_m[pixel_v, pixel_u])
        if np.isfinite(observed) and observed > 0:
            projected_difference = float(position[2] - observed)

    nearest = nearest_gt_point_distance_m(position, points)
    return {
        "target_consistency_metric": True,
        "top1_inside_gt_target_mask": projected_uv_inside_mask(uv, gt_mask),
        "top1_nearest_gt_target_point_distance_m": nearest,
        "top1_projected_depth_difference_m": projected_difference,
        "top1_projected_depth_error_m": (
            abs(projected_difference) if projected_difference is not None else None
        ),
        "gt_target_valid_depth_points": int(points.shape[0]),
    }


__all__ = [
    "GTMaskReference",
    "GTOracleMappingError",
    "QUERY_TYPE_RULE_PAYLOAD",
    "QueryTypeAnnotation",
    "annotate_expression",
    "annotate_top1_with_gt",
    "build_gt_oracle_sample",
    "derive_query_type",
    "evaluate_predicted_mask",
    "expression_for_sample",
    "iou_precision_indicators",
    "load_expression_index",
    "load_gt_mask",
    "mask_iou",
    "nearest_gt_point_distance_m",
    "projected_uv_inside_mask",
    "resolve_gt_mask_reference",
    "resolve_gt_oracle_mapping",
    "target_category_from_name",
]
