"""GT-free manifest contract for the repeated-FiLM 4-DoF benchmark.

Deployment manifests contain only information available to an offline grasp
backend at inference time.  OCID-VLG grasp rectangles and prepared target masks
are deliberately materialized in separate label tables.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, MutableMapping, Sequence

import pyarrow as pa
import pyarrow.parquet as pq


SCHEMA_VERSION = 1
SPLITS = ("train", "val", "test")
OUTPUT_SPLIT_NAMES = {"train": "train", "val": "validation", "test": "test"}

DEPLOYMENT_COLUMNS = (
    "schema_version",
    "sample_id",
    "sample_index",
    "scene_id",
    "expression_index",
    "question_index",
    "source_rgb_path",
    "source_rgb_sha256",
    "source_depth_path",
    "source_depth_sha256",
    "rgbd_pair_sha256",
    "language",
    "language_sha256",
    "predicted_mask_path",
    "predicted_mask_sha256",
    "predicted_probability_path",
    "predicted_probability_sha256",
    "intrinsics_path",
    "intrinsics_sha256",
    "intrinsics_status",
    "intrinsics_provenance",
    "split",
    "checkpoint_path",
    "checkpoint_sha256",
    "config_path",
    "config_sha256",
    "frozen_manifest_path",
    "frozen_manifest_sha256",
    "compact_manifest_path",
    "compact_manifest_sha256",
    "inference_contract_sha256",
)

LABEL_COLUMNS = (
    "schema_version",
    "sample_id",
    "sample_index",
    "scene_id",
    "question_index",
    "split",
    "prepared_gt_mask_path",
    "prepared_gt_mask_sha256",
    "gt_grasp_rectangles",
    "gt_grasp_count",
    "target_object_id",
    "official_annotations_path",
    "official_annotations_sha256",
)

_FORBIDDEN_EXACT = {
    "answer",
    "bbox",
    "box",
    "candidate_correctness",
    "correct_candidate_id",
    "correctness",
    "gt_bbox",
    "gt_grasp",
    "gt_grasps",
    "gt_mask",
    "gt_mask_path",
    "instance_id",
    "iou",
    "mask_iou",
    "mask_path",
    "official_gt_grasp_count",
    "official_instance_mask_path",
    "prepared_gt_mask_path",
    "target_instance_gt_id",
    "target_object_id",
}
_FORBIDDEN_FRAGMENTS = (
    "angle_error",
    "candidate_correct",
    "correct_candidate",
    "grasp_rectangle",
    "ground_truth",
)


class DataContractError(ValueError):
    """Raised when an input would violate the frozen benchmark contract."""


def sha256_file(path: str | Path, block_size: int = 1024 * 1024) -> str:
    """Return a streaming SHA-256 digest for an existing regular file."""

    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file() or resolved.stat().st_size <= 0:
        raise FileNotFoundError(f"missing or empty file: {resolved}")
    digest = hashlib.sha256()
    with resolved.open("rb") as stream:
        for block in iter(lambda: stream.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_sample_id(scene_id: str, question_index: int) -> str:
    """Return the repository-wide identity for a scene/expression pair."""

    index = int(question_index)
    identity = f"{scene_id}\t{index}".encode("utf-8")
    suffix = hashlib.sha256(identity).hexdigest()[:16]
    return f"q{index:07d}_{suffix}"


def stable_pair_hash(rgb_sha256: str, depth_sha256: str) -> str:
    """Bind one RGB digest and one depth digest without path dependence."""

    return hashlib.sha256(
        f"{rgb_sha256}\0{depth_sha256}".encode("ascii")
    ).hexdigest()


def language_hash(language: str) -> str:
    """Hash exact UTF-8 language, preserving case and whitespace."""

    return hashlib.sha256(str(language).encode("utf-8")).hexdigest()


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_json_rows(path: str | Path, *, data_key: str | None = None) -> list[dict[str, Any]]:
    """Read a JSON list/dict payload or JSONL stream into mapping rows."""

    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    if resolved.suffix == ".jsonl":
        rows: list[dict[str, Any]] = []
        with resolved.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise DataContractError(
                        f"{resolved}:{line_number}: expected JSON object"
                    )
                rows.append(value)
        return rows
    payload = _read_json(resolved)
    if data_key is not None:
        if not isinstance(payload, dict) or data_key not in payload:
            raise DataContractError(f"{resolved}: missing JSON key {data_key!r}")
        payload = payload[data_key]
    if not isinstance(payload, list) or any(not isinstance(row, dict) for row in payload):
        raise DataContractError(f"{resolved}: expected a list of JSON objects")
    return list(payload)


def assert_gt_free_columns(columns: Iterable[str]) -> None:
    """Reject label, oracle, evaluator, or correctness fields from deployment."""

    forbidden: list[str] = []
    for column in columns:
        lowered = str(column).lower()
        if (
            lowered in _FORBIDDEN_EXACT
            or lowered.startswith("gt_")
            or lowered.endswith("_gt")
            or any(fragment in lowered for fragment in _FORBIDDEN_FRAGMENTS)
        ):
            forbidden.append(str(column))
    if forbidden:
        raise DataContractError(
            f"deployment manifest contains GT/evaluation columns: {sorted(forbidden)}"
        )


def _require_sha256(value: Any, *, field: str) -> str:
    digest = str(value).lower()
    if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
        raise DataContractError(f"invalid SHA-256 in {field}: {value!r}")
    return digest


def _require_file(path: str | Path, *, label: str) -> Path:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file() or resolved.stat().st_size <= 0:
        raise FileNotFoundError(f"missing or empty {label}: {resolved}")
    return resolved


def _resolve_prepared_path(hifics_root: Path, raw_path: Any, *, label: str) -> Path:
    candidate = Path(str(raw_path).replace("\\", "/")).expanduser()
    if not candidate.is_absolute():
        candidate = hifics_root / candidate
    return _require_file(candidate, label=label)


def _field(row: Mapping[str, Any], names: Sequence[str], *, label: str) -> Any:
    for name in names:
        value = row.get(name)
        if value is not None and value != "":
            return value
    raise DataContractError(f"compact row is missing {label}: tried {list(names)}")


def _checked_declared_file(
    row: Mapping[str, Any],
    *,
    path_fields: Sequence[str],
    hash_fields: Sequence[str],
    label: str,
    hash_cache: MutableMapping[Path, str],
) -> tuple[Path, str]:
    path = _require_file(_field(row, path_fields, label=f"{label} path"), label=label)
    declared = _require_sha256(
        _field(row, hash_fields, label=f"{label} SHA-256"),
        field=f"{label} SHA-256",
    )
    actual = hash_cache.get(path)
    if actual is None:
        actual = sha256_file(path)
        hash_cache[path] = actual
    if declared != actual:
        raise DataContractError(
            f"{label} SHA-256 mismatch for {path}: declared={declared}, actual={actual}"
        )
    return path, actual


def _validate_grasps(value: Any, *, sample_id: str) -> list[list[list[float]]]:
    if not isinstance(value, list) or not value:
        raise DataContractError(f"{sample_id}: GT grasp rectangle list is empty")
    result: list[list[list[float]]] = []
    for rectangle_index, rectangle in enumerate(value):
        if not isinstance(rectangle, list) or len(rectangle) != 4:
            raise DataContractError(
                f"{sample_id}: grasp {rectangle_index} is not four corners"
            )
        converted: list[list[float]] = []
        for point in rectangle:
            if not isinstance(point, list) or len(point) != 2:
                raise DataContractError(
                    f"{sample_id}: grasp {rectangle_index} has malformed point"
                )
            xy = [float(point[0]), float(point[1])]
            if not all(math.isfinite(number) for number in xy):
                raise DataContractError(
                    f"{sample_id}: grasp {rectangle_index} is non-finite"
                )
            converted.append(xy)
        result.append(converted)
    return result


def _identity_map(
    rows: Sequence[Mapping[str, Any]],
    *,
    scene_field: str,
    question_field: str,
    source: str,
) -> dict[str, Mapping[str, Any]]:
    result: dict[str, Mapping[str, Any]] = {}
    for position, row in enumerate(rows):
        try:
            sample_id = stable_sample_id(
                str(row[scene_field]), int(row[question_field])
            )
        except (KeyError, TypeError, ValueError) as error:
            raise DataContractError(f"{source}[{position}] has invalid identity") from error
        if sample_id in result:
            raise DataContractError(f"{source} has duplicate sample ID {sample_id}")
        result[sample_id] = row
    return result


def _intrinsics_reference(
    compact: Mapping[str, Any],
    *,
    hash_cache: MutableMapping[Path, str],
) -> tuple[str | None, str | None, str, str]:
    explicit = next(
        (
            compact.get(name)
            for name in ("intrinsics_path", "camera_intrinsics_path")
            if compact.get(name)
        ),
        None,
    )
    if explicit is None and compact.get("frozen_source_metadata_path"):
        sibling = Path(str(compact["frozen_source_metadata_path"])).expanduser().resolve().parent
        candidate = sibling / "intrinsics.json"
        if candidate.is_file():
            explicit = candidate
    if explicit is not None:
        path = _require_file(explicit, label="camera intrinsics")
        digest = hash_cache.get(path)
        if digest is None:
            digest = sha256_file(path)
            hash_cache[path] = digest
        declared = compact.get("intrinsics_sha256") or compact.get(
            "camera_intrinsics_sha256"
        )
        if declared is not None and _require_sha256(
            declared, field="intrinsics_sha256"
        ) != digest:
            raise DataContractError(f"camera intrinsics SHA-256 mismatch: {path}")
        provenance = _canonical_json({"kind": "explicit_file", "path": str(path)})
        return str(path), digest, "explicit_file", provenance

    pcd_path = compact.get("source_pcd_path")
    pcd_sha = compact.get("source_pcd_sha256")
    if pcd_path and pcd_sha:
        pcd = _require_file(pcd_path, label="organized PCD intrinsics provenance")
        declared = _require_sha256(pcd_sha, field="source_pcd_sha256")
        actual = hash_cache.get(pcd)
        if actual is None:
            actual = sha256_file(pcd)
            hash_cache[pcd] = actual
        if actual != declared:
            raise DataContractError(f"organized PCD SHA-256 mismatch: {pcd}")
        provenance = _canonical_json(
            {
                "kind": "derived_from_organized_pcd",
                "source_pcd_path": str(pcd),
                "source_pcd_sha256": actual,
                "status": "derivation_required",
            }
        )
        return None, None, "derived_from_organized_pcd", provenance

    provenance = _canonical_json(
        {"kind": "unavailable", "status": "explicitly_missing"}
    )
    return None, None, "unavailable", provenance


def build_split_records(
    *,
    split: str,
    frozen_manifest_path: str | Path,
    compact_manifest_path: str | Path,
    annotations_path: str | Path,
    hifics_root: str | Path,
    checkpoint_path: str | Path,
    config_path: str | Path,
    hash_cache: MutableMapping[Path, str] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Validate and join one frozen split into deployment and label records."""

    if split not in SPLITS:
        raise DataContractError(f"unsupported split: {split!r}")
    cache: MutableMapping[Path, str] = hash_cache if hash_cache is not None else {}
    frozen_path = _require_file(frozen_manifest_path, label=f"{split} frozen manifest")
    compact_path = _require_file(compact_manifest_path, label=f"{split} compact manifest")
    official_path = _require_file(annotations_path, label=f"{split} annotations")
    checkpoint = _require_file(checkpoint_path, label="repeated-FiLM checkpoint")
    config = _require_file(config_path, label="repeated-FiLM config")
    hifics = Path(hifics_root).expanduser().resolve()

    frozen_sha = sha256_file(frozen_path)
    compact_sha = sha256_file(compact_path)
    annotations_sha = sha256_file(official_path)
    checkpoint_sha = sha256_file(checkpoint)
    config_sha = sha256_file(config)
    frozen_rows = read_json_rows(frozen_path)
    compact_rows = read_json_rows(compact_path)
    official_rows = read_json_rows(official_path, data_key="data")
    frozen_by_id = _identity_map(
        frozen_rows,
        scene_field="scene_id",
        question_field="question_index",
        source=f"{split} frozen manifest",
    )
    compact_by_id = _identity_map(
        compact_rows,
        scene_field="scene_id",
        question_field="question_index",
        source=f"{split} compact manifest",
    )
    official_by_id = _identity_map(
        official_rows,
        scene_field="image_filename",
        question_field="question_index",
        source=f"{split} official annotations",
    )
    identity_sets = (set(frozen_by_id), set(compact_by_id), set(official_by_id))
    if not (identity_sets[0] == identity_sets[1] == identity_sets[2]):
        raise DataContractError(
            f"{split} incomplete one-to-one coverage: "
            f"frozen={len(identity_sets[0])}, compact={len(identity_sets[1])}, "
            f"annotations={len(identity_sets[2])}"
        )

    deployment: list[dict[str, Any]] = []
    labels: list[dict[str, Any]] = []
    for position, frozen in enumerate(frozen_rows):
        question_index = int(frozen["question_index"])
        scene_id = str(frozen["scene_id"])
        sample_id = stable_sample_id(scene_id, question_index)
        compact = compact_by_id[sample_id]
        official = official_by_id[sample_id]
        language = str(frozen["text"])
        if (
            str(compact.get("sample_id")) != sample_id
            or str(compact.get("query")) != language
            or str(official.get("question")) != language
            or str(official.get("split")) != split
            or str(compact.get("split")) != split
        ):
            raise DataContractError(f"{split}/{sample_id}: identity or language mismatch")
        sample_index = int(frozen.get("num", position))
        if int(compact.get("sample_index", -1)) != sample_index:
            raise DataContractError(f"{split}/{sample_id}: sample index mismatch")
        if compact.get("ready") is not True:
            raise DataContractError(f"{split}/{sample_id}: compact prediction not ready")
        if compact.get("gt_artifacts_exported") is not False:
            raise DataContractError(
                f"{split}/{sample_id}: compact GT-free status is not explicit"
            )
        compact_manifest_sha = _require_sha256(
            compact.get("manifest_sha256"), field="compact manifest_sha256"
        )
        if compact_manifest_sha != frozen_sha:
            raise DataContractError(f"{split}/{sample_id}: frozen manifest lineage mismatch")
        if _require_sha256(
            compact.get("checkpoint_sha256"), field="compact checkpoint_sha256"
        ) != checkpoint_sha:
            raise DataContractError(f"{split}/{sample_id}: checkpoint lineage mismatch")
        if compact.get("config_sha256") is not None and _require_sha256(
            compact["config_sha256"], field="compact config_sha256"
        ) != config_sha:
            raise DataContractError(f"{split}/{sample_id}: config lineage mismatch")

        rgb_path, rgb_sha = _checked_declared_file(
            compact,
            path_fields=("source_rgb_path", "source_rgb"),
            hash_fields=("source_rgb_sha256", "rgb_sha256"),
            label=f"{split}/{sample_id} source RGB",
            hash_cache=cache,
        )
        depth_path, depth_sha = _checked_declared_file(
            compact,
            path_fields=("source_depth_path", "source_depth"),
            hash_fields=("source_depth_sha256", "depth_sha256"),
            label=f"{split}/{sample_id} source depth",
            hash_cache=cache,
        )
        predicted_mask_path, predicted_mask_sha = _checked_declared_file(
            compact,
            path_fields=("native_mask_path", "predicted_mask_path"),
            hash_fields=("native_mask_sha256", "predicted_mask_sha256"),
            label=f"{split}/{sample_id} predicted mask",
            hash_cache=cache,
        )
        probability_path, probability_sha = _checked_declared_file(
            compact,
            path_fields=("probability_path", "predicted_probability_path"),
            hash_fields=("probability_sha256", "predicted_probability_sha256"),
            label=f"{split}/{sample_id} predicted probability",
            hash_cache=cache,
        )
        intrinsics_path, intrinsics_sha, intrinsics_status, intrinsics_provenance = (
            _intrinsics_reference(compact, hash_cache=cache)
        )
        inference_contract_sha = _require_sha256(
            compact.get("inference_contract_sha256"),
            field="inference_contract_sha256",
        )
        deployment.append(
            {
                "schema_version": SCHEMA_VERSION,
                "sample_id": sample_id,
                "sample_index": sample_index,
                "scene_id": scene_id,
                "expression_index": question_index,
                "question_index": question_index,
                "source_rgb_path": str(rgb_path),
                "source_rgb_sha256": rgb_sha,
                "source_depth_path": str(depth_path),
                "source_depth_sha256": depth_sha,
                "rgbd_pair_sha256": stable_pair_hash(rgb_sha, depth_sha),
                "language": language,
                "language_sha256": language_hash(language),
                "predicted_mask_path": str(predicted_mask_path),
                "predicted_mask_sha256": predicted_mask_sha,
                "predicted_probability_path": str(probability_path),
                "predicted_probability_sha256": probability_sha,
                "intrinsics_path": intrinsics_path,
                "intrinsics_sha256": intrinsics_sha,
                "intrinsics_status": intrinsics_status,
                "intrinsics_provenance": intrinsics_provenance,
                "split": split,
                "checkpoint_path": str(checkpoint),
                "checkpoint_sha256": checkpoint_sha,
                "config_path": str(config),
                "config_sha256": config_sha,
                "frozen_manifest_path": str(frozen_path),
                "frozen_manifest_sha256": frozen_sha,
                "compact_manifest_path": str(compact_path),
                "compact_manifest_sha256": compact_sha,
                "inference_contract_sha256": inference_contract_sha,
            }
        )

        prepared_mask = _resolve_prepared_path(
            hifics,
            frozen["mask_path"],
            label=f"{split}/{sample_id} prepared GT mask",
        )
        prepared_mask_sha = cache.get(prepared_mask)
        if prepared_mask_sha is None:
            prepared_mask_sha = sha256_file(prepared_mask)
            cache[prepared_mask] = prepared_mask_sha
        grasps = _validate_grasps(official.get("grasps"), sample_id=sample_id)
        target = official.get("answer")
        if isinstance(target, bool) or not isinstance(target, int):
            raise DataContractError(f"{split}/{sample_id}: invalid target object ID")
        labels.append(
            {
                "schema_version": SCHEMA_VERSION,
                "sample_id": sample_id,
                "sample_index": sample_index,
                "scene_id": scene_id,
                "question_index": question_index,
                "split": split,
                "prepared_gt_mask_path": str(prepared_mask),
                "prepared_gt_mask_sha256": prepared_mask_sha,
                "gt_grasp_rectangles": grasps,
                "gt_grasp_count": len(grasps),
                "target_object_id": int(target),
                "official_annotations_path": str(official_path),
                "official_annotations_sha256": annotations_sha,
            }
        )

    assert_deployment_records(deployment, expected_split=split)
    if len(labels) != len(deployment):
        raise DataContractError(f"{split}: label/deployment coverage mismatch")
    return deployment, labels


def assert_deployment_records(
    rows: Sequence[Mapping[str, Any]], *, expected_split: str | None = None
) -> None:
    """Validate the exact GT-free schema and row-level deterministic hashes."""

    assert_gt_free_columns(DEPLOYMENT_COLUMNS)
    seen: set[str] = set()
    for index, row in enumerate(rows):
        if set(row) != set(DEPLOYMENT_COLUMNS):
            extra = sorted(set(row) - set(DEPLOYMENT_COLUMNS))
            missing = sorted(set(DEPLOYMENT_COLUMNS) - set(row))
            raise DataContractError(
                f"deployment row {index} schema mismatch: extra={extra}, missing={missing}"
            )
        assert_gt_free_columns(row)
        sample_id = str(row["sample_id"])
        if sample_id in seen:
            raise DataContractError(f"duplicate deployment sample ID: {sample_id}")
        seen.add(sample_id)
        if sample_id != stable_sample_id(
            str(row["scene_id"]), int(row["question_index"])
        ):
            raise DataContractError(f"unstable deployment sample ID: {sample_id}")
        if int(row["expression_index"]) != int(row["question_index"]):
            raise DataContractError(f"{sample_id}: expression/question index mismatch")
        if expected_split is not None and str(row["split"]) != expected_split:
            raise DataContractError(f"{sample_id}: wrong split {row['split']!r}")
        if row["language_sha256"] != language_hash(str(row["language"])):
            raise DataContractError(f"{sample_id}: language hash mismatch")
        if row["rgbd_pair_sha256"] != stable_pair_hash(
            str(row["source_rgb_sha256"]), str(row["source_depth_sha256"])
        ):
            raise DataContractError(f"{sample_id}: RGB-D pair hash mismatch")


def validate_split_isolation(
    deployment_by_split: Mapping[str, Sequence[Mapping[str, Any]]]
) -> dict[str, dict[str, int]]:
    """Require row, scene, RGB, depth, and RGB-D isolation across splits."""

    if set(deployment_by_split) != set(SPLITS):
        raise DataContractError(f"deployment splits must be exactly {list(SPLITS)}")
    keys = (
        "sample_id",
        "scene_id",
        "source_rgb_sha256",
        "source_depth_sha256",
        "rgbd_pair_sha256",
    )
    values: dict[str, dict[str, set[str]]] = {}
    for split in SPLITS:
        assert_deployment_records(deployment_by_split[split], expected_split=split)
        values[split] = {
            key: {str(row[key]) for row in deployment_by_split[split]} for key in keys
        }
    audit: dict[str, dict[str, int]] = {}
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        pair = f"{left}__{right}"
        audit[pair] = {
            key: len(values[left][key] & values[right][key]) for key in keys
        }
    violations = {pair: row for pair, row in audit.items() if any(row.values())}
    if violations:
        raise DataContractError(f"split leakage detected: {violations}")
    return audit


def deployment_schema() -> pa.Schema:
    fields = [
        pa.field("schema_version", pa.int16(), nullable=False),
        pa.field("sample_id", pa.string(), nullable=False),
        pa.field("sample_index", pa.int64(), nullable=False),
        pa.field("scene_id", pa.string(), nullable=False),
        pa.field("expression_index", pa.int64(), nullable=False),
        pa.field("question_index", pa.int64(), nullable=False),
        pa.field("source_rgb_path", pa.string(), nullable=False),
        pa.field("source_rgb_sha256", pa.string(), nullable=False),
        pa.field("source_depth_path", pa.string(), nullable=False),
        pa.field("source_depth_sha256", pa.string(), nullable=False),
        pa.field("rgbd_pair_sha256", pa.string(), nullable=False),
        pa.field("language", pa.string(), nullable=False),
        pa.field("language_sha256", pa.string(), nullable=False),
        pa.field("predicted_mask_path", pa.string(), nullable=False),
        pa.field("predicted_mask_sha256", pa.string(), nullable=False),
        pa.field("predicted_probability_path", pa.string(), nullable=False),
        pa.field("predicted_probability_sha256", pa.string(), nullable=False),
        pa.field("intrinsics_path", pa.string()),
        pa.field("intrinsics_sha256", pa.string()),
        pa.field("intrinsics_status", pa.string(), nullable=False),
        pa.field("intrinsics_provenance", pa.string(), nullable=False),
        pa.field("split", pa.string(), nullable=False),
        pa.field("checkpoint_path", pa.string(), nullable=False),
        pa.field("checkpoint_sha256", pa.string(), nullable=False),
        pa.field("config_path", pa.string(), nullable=False),
        pa.field("config_sha256", pa.string(), nullable=False),
        pa.field("frozen_manifest_path", pa.string(), nullable=False),
        pa.field("frozen_manifest_sha256", pa.string(), nullable=False),
        pa.field("compact_manifest_path", pa.string(), nullable=False),
        pa.field("compact_manifest_sha256", pa.string(), nullable=False),
        pa.field("inference_contract_sha256", pa.string(), nullable=False),
    ]
    schema = pa.schema(fields, metadata={b"contract": b"grasp4dof_gt_free_v1"})
    if tuple(schema.names) != DEPLOYMENT_COLUMNS:
        raise AssertionError("deployment Arrow schema drift")
    assert_gt_free_columns(schema.names)
    return schema


def labels_schema() -> pa.Schema:
    point = pa.list_(pa.float64(), list_size=2)
    rectangle = pa.list_(point, list_size=4)
    fields = [
        pa.field("schema_version", pa.int16(), nullable=False),
        pa.field("sample_id", pa.string(), nullable=False),
        pa.field("sample_index", pa.int64(), nullable=False),
        pa.field("scene_id", pa.string(), nullable=False),
        pa.field("question_index", pa.int64(), nullable=False),
        pa.field("split", pa.string(), nullable=False),
        pa.field("prepared_gt_mask_path", pa.string(), nullable=False),
        pa.field("prepared_gt_mask_sha256", pa.string(), nullable=False),
        pa.field("gt_grasp_rectangles", pa.list_(rectangle), nullable=False),
        pa.field("gt_grasp_count", pa.int32(), nullable=False),
        pa.field("target_object_id", pa.int32(), nullable=False),
        pa.field("official_annotations_path", pa.string(), nullable=False),
        pa.field("official_annotations_sha256", pa.string(), nullable=False),
    ]
    schema = pa.schema(fields, metadata={b"contract": b"grasp4dof_labels_v1"})
    if tuple(schema.names) != LABEL_COLUMNS:
        raise AssertionError("label Arrow schema drift")
    return schema


def write_parquet_bundle(
    *,
    output_dir: str | Path,
    deployment_by_split: Mapping[str, Sequence[Mapping[str, Any]]],
    labels_by_split: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, str]:
    """Write six immutable ZSTD Parquets after global split validation."""

    if set(labels_by_split) != set(SPLITS):
        raise DataContractError(f"label splits must be exactly {list(SPLITS)}")
    validate_split_isolation(deployment_by_split)
    for split in SPLITS:
        deployment_ids = [str(row["sample_id"]) for row in deployment_by_split[split]]
        label_ids = [str(row["sample_id"]) for row in labels_by_split[split]]
        if deployment_ids != label_ids:
            raise DataContractError(f"{split}: deployment/label row coverage mismatch")
        for row in labels_by_split[split]:
            if set(row) != set(LABEL_COLUMNS):
                raise DataContractError(f"{split}: label row schema mismatch")

    destination = Path(output_dir).expanduser().resolve()
    if destination.exists() and destination.is_symlink():
        raise DataContractError(f"refusing symlink output directory: {destination}")
    destination.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, Path] = {}
    for split in SPLITS:
        name = OUTPUT_SPLIT_NAMES[split]
        outputs[f"{split}_samples"] = destination / f"{name}_samples.parquet"
        outputs[f"{split}_labels"] = destination / f"{name}_labels.parquet"
    existing = [str(path) for path in outputs.values() if path.exists()]
    if existing:
        raise FileExistsError(f"refusing to overwrite manifest Parquets: {existing}")

    for split in SPLITS:
        sample_key = f"{split}_samples"
        label_key = f"{split}_labels"
        sample_table = pa.Table.from_pylist(
            list(deployment_by_split[split]), schema=deployment_schema()
        )
        label_table = pa.Table.from_pylist(
            list(labels_by_split[split]), schema=labels_schema()
        )
        pq.write_table(sample_table, outputs[sample_key], compression="zstd")
        pq.write_table(label_table, outputs[label_key], compression="zstd")
    return {key: str(path) for key, path in outputs.items()}
