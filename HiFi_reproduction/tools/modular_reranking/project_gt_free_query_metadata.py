#!/usr/bin/env python3
"""Project OCID-VLG expressions into an exact-schema, GT-free query artifact.

This is the only re-ranking stage allowed to open ``*_expressions.json``.
Downstream feature extraction consumes the five-field projection and verifies
that it is bound to the same frozen prediction and split manifests.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.experiments.ocid_annotations import (  # noqa: E402
    annotate_expression,
    load_expression_index,
)
from src.grasping.reranking_v1.identity import (  # noqa: E402
    sha256_file,
    stable_sample_id,
)


QUERY_METADATA_SCHEMA_VERSION = 1
QUERY_METADATA_ARTIFACT_TYPE = "gt_free_query_metadata_v1"
QUERY_METADATA_FIELDS = (
    "sample_id",
    "question_index",
    "scene_id",
    "query",
    "query_type",
)
FORBIDDEN_QUERY_METADATA_FIELDS = (
    "answer",
    "box",
    "candidate_gt_angle_error",
    "candidate_gt_iou",
    "candidate_labels",
    "candidate_positive",
    "concept_map",
    "grasps",
    "ground_truth_mask",
    "gt_mask",
    "gt_mask_path",
    "mask_path",
    "program",
    "target",
    "target_mask",
    "target_mask_path",
)
QUERY_METADATA_MANIFEST_FIELDS = (
    "schema_version",
    "artifact_type",
    "status",
    "split",
    "fields",
    "exact_schema_required",
    "ground_truth_allowed",
    "forbidden_fields",
    "row_count",
    "query_metadata_identity_sha256",
    "source_annotations_path",
    "source_annotations_sha256",
    "source_annotations_may_contain_ground_truth",
    "source_annotations_consumed_only_by_projection",
    "feature_extractor_must_not_read_source_annotations",
    "source_prediction_manifest_path",
    "source_prediction_manifest_sha256",
    "source_prediction_identity_sha256",
    "source_frozen_manifest_path",
    "source_frozen_manifest_sha256",
    "query_metadata_path",
    "query_metadata_sha256",
)
PREDICTION_IDENTITY_FIELDS = (
    "sample_index",
    "sample_id",
    "question_index",
    "scene_id",
    "query",
    "split",
    "manifest_path",
    "manifest_sha256",
)
PREDICTION_FORBIDDEN_FIELDS = frozenset(
    {
        "answer",
        "box",
        "candidate_gt_angle_error",
        "candidate_gt_iou",
        "candidate_labels",
        "candidate_positive",
        "grasps",
        "ground_truth_mask",
        "gt_mask",
        "gt_mask_path",
        "target",
        "target_mask",
        "target_mask_path",
    }
)


def _canonical_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                raise ValueError(f"blank JSONL row at {path}:{line_number}")
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"invalid JSONL row at {path}:{line_number}: {error}"
                ) from error
            if not isinstance(value, dict):
                raise ValueError(
                    f"JSONL row is not an object at {path}:{line_number}"
                )
            rows.append(value)
    if not rows:
        raise ValueError(f"JSONL artifact is empty: {path}")
    return rows


def _active_run_root(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    start = resolved if resolved.is_dir() else resolved.parent
    for candidate in (start, *start.parents):
        if (candidate / ".RUN_ACTIVE").is_file():
            return candidate
    raise ValueError(f"path is not inside a marked active run: {resolved}")


def validate_tmp_scope(output_root: Path, tmp_root: Path) -> tuple[Path, Path, Path]:
    """Bind output and atomic temporary files to one marked current run."""

    output = output_root.expanduser().resolve()
    temporary = tmp_root.expanduser().resolve()
    output_run = _active_run_root(output.parent)
    temporary_run = _active_run_root(temporary)
    if output_run != temporary_run:
        raise ValueError("output-root and tmp-root belong to different active runs")
    configured_tmp = output_run / "tmp"
    if temporary != configured_tmp and configured_tmp not in temporary.parents:
        raise ValueError(f"tmp-root must be at or below {configured_tmp}")
    if output == output_run or output_run not in output.parents:
        raise ValueError(f"output-root must be below active run {output_run}")
    temporary.mkdir(parents=True, exist_ok=True)
    return output, temporary, output_run


def _temporary_path(path: Path, tmp_root: Path) -> Path:
    temporary = (
        tmp_root
        / "atomic_query_metadata"
        / f"{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    )
    temporary.parent.mkdir(parents=True, exist_ok=True)
    if tmp_root != temporary.parent and tmp_root not in temporary.parents:
        raise RuntimeError("atomic temporary escaped --tmp-root")
    return temporary


def atomic_text(path: Path, value: str, *, tmp_root: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = _temporary_path(path, tmp_root)
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def atomic_json(path: Path, value: Any, *, tmp_root: Path) -> None:
    atomic_text(
        path,
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n",
        tmp_root=tmp_root,
    )


@dataclass(frozen=True)
class PredictionBundle:
    root: Path
    manifest_path: Path
    manifest_sha256: str
    identity_sha256: str
    rows: tuple[dict[str, Any], ...]
    by_sample_id: dict[str, dict[str, Any]]
    frozen_manifest_path: Path
    frozen_manifest_sha256: str


def _prediction_identity(rows: Sequence[Mapping[str, Any]]) -> str:
    return _canonical_hash(
        [
            {field: row[field] for field in PREDICTION_IDENTITY_FIELDS}
            for row in rows
        ]
    )


def load_verified_prediction_bundle(
    prediction_root: Path | str, *, split: str
) -> PredictionBundle:
    """Verify JSONL/sidecar/frozen-manifest identity without reading GT labels."""

    root = Path(prediction_root).expanduser().resolve()
    manifest_path = root / "manifest.jsonl"
    summary_path = root / "summary.json"
    if not manifest_path.is_file() or not summary_path.is_file():
        raise FileNotFoundError(
            f"compact prediction manifest/summary missing under {root}"
        )
    rows = _read_jsonl(manifest_path)
    by_sample_id: dict[str, dict[str, Any]] = {}
    question_indices: set[int] = set()
    frozen_identities: set[tuple[str, str]] = set()
    for offset, row in enumerate(rows):
        missing = set(PREDICTION_IDENTITY_FIELDS) - set(row)
        if missing:
            raise ValueError(
                f"prediction row {offset} missing identity fields: {sorted(missing)}"
            )
        forbidden = sorted(PREDICTION_FORBIDDEN_FIELDS & set(row))
        if forbidden:
            raise ValueError(
                f"prediction row {offset} contains forbidden GT fields: {forbidden}"
            )
        if row.get("ready") is not True:
            raise ValueError(f"prediction row is not ready: {offset}")
        if row.get("gt_artifacts_exported") is not False:
            raise ValueError(
                f"prediction row does not prove GT-free export: {offset}"
            )
        if row["split"] != split:
            raise ValueError(f"prediction split mismatch at row {offset}")
        if isinstance(row["question_index"], bool):
            raise ValueError(f"invalid prediction question_index at row {offset}")
        question_index = int(row["question_index"])
        scene_id = str(row["scene_id"])
        sample_id = str(row["sample_id"])
        query = str(row["query"])
        if not scene_id or not query:
            raise ValueError(f"empty prediction scene/query at row {offset}")
        if sample_id != stable_sample_id(scene_id, question_index):
            raise ValueError(f"prediction sample identity mismatch: {sample_id}")
        if int(row["sample_index"]) != offset:
            raise ValueError(
                f"prediction sample_index is not canonical at row {offset}"
            )
        if sample_id in by_sample_id:
            raise ValueError(f"duplicate prediction sample_id: {sample_id}")
        if question_index in question_indices:
            raise ValueError(
                f"duplicate prediction question_index: {question_index}"
            )
        by_sample_id[sample_id] = row
        question_indices.add(question_index)
        frozen_identities.add(
            (str(row["manifest_path"]), str(row["manifest_sha256"]))
        )
    if len(frozen_identities) != 1:
        raise ValueError(
            f"prediction rows have mixed frozen manifests: {frozen_identities}"
        )
    frozen_path_text, frozen_sha256 = next(iter(frozen_identities))
    frozen_path = Path(frozen_path_text).expanduser().resolve()
    if not frozen_path.is_file() or sha256_file(frozen_path) != frozen_sha256:
        raise ValueError("frozen split manifest identity mismatch")

    row_files = sorted((root / "rows").glob("*.json"))
    row_file_ids = {path.stem for path in row_files}
    if row_file_ids != set(by_sample_id):
        missing = sorted(set(by_sample_id) - row_file_ids)[:5]
        extra = sorted(row_file_ids - set(by_sample_id))[:5]
        raise ValueError(
            f"prediction sidecar universe mismatch: missing={missing}, extra={extra}"
        )
    for path in row_files:
        sidecar = json.loads(path.read_text(encoding="utf-8"))
        if sidecar != by_sample_id[path.stem]:
            raise ValueError(f"prediction sidecar/manifest mismatch: {path.stem}")

    manifest_sha256 = sha256_file(manifest_path)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    expected_summary = {
        "status": "COMPLETED",
        "split": split,
        "samples": len(rows),
        "output_manifest_sha256": manifest_sha256,
    }
    for field, expected in expected_summary.items():
        if summary.get(field) != expected:
            raise ValueError(
                f"prediction summary {field} mismatch: "
                f"{summary.get(field)!r} != {expected!r}"
            )
    summary_frozen_sha256 = summary.get(
        "frozen_manifest_sha256", summary.get("manifest_sha256")
    )
    if summary_frozen_sha256 != frozen_sha256:
        raise ValueError("prediction summary frozen manifest digest mismatch")
    if Path(str(summary.get("output_manifest", ""))).expanduser().resolve() != (
        manifest_path
    ):
        raise ValueError("prediction summary output_manifest path mismatch")
    return PredictionBundle(
        root=root,
        manifest_path=manifest_path,
        manifest_sha256=manifest_sha256,
        identity_sha256=_prediction_identity(rows),
        rows=tuple(rows),
        by_sample_id=by_sample_id,
        frozen_manifest_path=frozen_path,
        frozen_manifest_sha256=frozen_sha256,
    )


def _load_frozen_identity_rows(bundle: PredictionBundle) -> dict[int, dict[str, Any]]:
    payload = json.loads(bundle.frozen_manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("frozen split manifest must contain a list")
    output: dict[int, dict[str, Any]] = {}
    for offset, value in enumerate(payload):
        if not isinstance(value, dict):
            raise ValueError(f"frozen split row is not an object: {offset}")
        required = {"question_index", "scene_id", "text"}
        if not required <= set(value):
            raise ValueError(f"frozen split row missing identity: {offset}")
        question_index = int(value["question_index"])
        if question_index in output:
            raise ValueError(
                f"duplicate frozen split question_index: {question_index}"
            )
        output[question_index] = value
    return output


def _validate_query_row(row: Mapping[str, Any], *, offset: int) -> dict[str, Any]:
    keys = set(row)
    expected = set(QUERY_METADATA_FIELDS)
    if keys != expected:
        forbidden = sorted(keys & set(FORBIDDEN_QUERY_METADATA_FIELDS))
        extra = sorted(keys - expected)
        missing = sorted(expected - keys)
        raise ValueError(
            "query metadata must use the exact GT-free schema: "
            f"row={offset}, forbidden={forbidden}, extra={extra}, missing={missing}"
        )
    if isinstance(row["question_index"], bool):
        raise ValueError(f"invalid query metadata question_index at row {offset}")
    normalized = {
        "sample_id": str(row["sample_id"]),
        "question_index": int(row["question_index"]),
        "scene_id": str(row["scene_id"]),
        "query": str(row["query"]),
        "query_type": str(row["query_type"]),
    }
    if (
        not normalized["scene_id"]
        or not normalized["query"]
        or not normalized["query_type"]
    ):
        raise ValueError(f"query metadata contains an empty value at row {offset}")
    if normalized["sample_id"] != stable_sample_id(
        normalized["scene_id"], normalized["question_index"]
    ):
        raise ValueError(
            f"query metadata stable identity mismatch: {normalized['sample_id']}"
        )
    return normalized


def load_query_metadata_bundle(
    query_metadata_path: Path | str,
    *,
    prediction_bundle: PredictionBundle,
    split: str,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """Load an exact five-field projection and bind it to predictions."""

    path = Path(query_metadata_path).expanduser().resolve()
    manifest_path = path.parent / "query_metadata_manifest.json"
    if not path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError(
            f"query metadata artifact/manifest missing: {path}, {manifest_path}"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if set(manifest) != set(QUERY_METADATA_MANIFEST_FIELDS):
        forbidden = sorted(
            set(manifest) & set(FORBIDDEN_QUERY_METADATA_FIELDS)
        )
        extra = sorted(set(manifest) - set(QUERY_METADATA_MANIFEST_FIELDS))
        missing = sorted(set(QUERY_METADATA_MANIFEST_FIELDS) - set(manifest))
        raise ValueError(
            "query metadata manifest must use the exact safe schema: "
            f"forbidden={forbidden}, extra={extra}, missing={missing}"
        )
    expected_manifest_values = {
        "schema_version": QUERY_METADATA_SCHEMA_VERSION,
        "artifact_type": QUERY_METADATA_ARTIFACT_TYPE,
        "status": "COMPLETED",
        "split": split,
        "fields": list(QUERY_METADATA_FIELDS),
        "exact_schema_required": True,
        "ground_truth_allowed": False,
        "query_metadata_sha256": sha256_file(path),
        "source_prediction_manifest_sha256": prediction_bundle.manifest_sha256,
        "source_prediction_identity_sha256": prediction_bundle.identity_sha256,
        "source_frozen_manifest_sha256": (
            prediction_bundle.frozen_manifest_sha256
        ),
    }
    for field, expected in expected_manifest_values.items():
        if manifest.get(field) != expected:
            raise ValueError(
                f"query metadata manifest {field} mismatch: "
                f"{manifest.get(field)!r} != {expected!r}"
            )
    path_fields = {
        "query_metadata_path": path,
        "source_prediction_manifest_path": prediction_bundle.manifest_path,
        "source_frozen_manifest_path": prediction_bundle.frozen_manifest_path,
    }
    for field, expected in path_fields.items():
        observed = Path(str(manifest.get(field, ""))).expanduser().resolve()
        if observed != expected:
            raise ValueError(
                f"query metadata manifest {field} path mismatch: "
                f"{observed} != {expected}"
            )
    if manifest.get("forbidden_fields") != list(FORBIDDEN_QUERY_METADATA_FIELDS):
        raise ValueError("query metadata manifest forbidden-field policy mismatch")

    rows = _read_jsonl(path)
    index: dict[str, dict[str, Any]] = {}
    question_indices: set[int] = set()
    normalized_rows: list[dict[str, Any]] = []
    for offset, row in enumerate(rows):
        normalized = _validate_query_row(row, offset=offset)
        sample_id = normalized["sample_id"]
        question_index = normalized["question_index"]
        if sample_id in index:
            raise ValueError(f"duplicate query metadata sample_id: {sample_id}")
        if question_index in question_indices:
            raise ValueError(
                f"duplicate query metadata question_index: {question_index}"
            )
        index[sample_id] = normalized
        question_indices.add(question_index)
        normalized_rows.append(normalized)
    if int(manifest.get("row_count", -1)) != len(normalized_rows):
        raise ValueError("query metadata manifest row_count mismatch")
    identity_sha256 = _canonical_hash(normalized_rows)
    if manifest.get("query_metadata_identity_sha256") != identity_sha256:
        raise ValueError("query metadata identity digest mismatch")
    if set(index) != set(prediction_bundle.by_sample_id):
        missing = sorted(set(prediction_bundle.by_sample_id) - set(index))[:5]
        extra = sorted(set(index) - set(prediction_bundle.by_sample_id))[:5]
        raise ValueError(
            f"query/prediction universe mismatch: missing={missing}, extra={extra}"
        )
    for sample_id, query_row in index.items():
        prediction = prediction_bundle.by_sample_id[sample_id]
        expected = {
            "question_index": int(prediction["question_index"]),
            "scene_id": str(prediction["scene_id"]),
            "query": str(prediction["query"]),
        }
        for field, value in expected.items():
            if query_row[field] != value:
                raise ValueError(
                    f"query/prediction {field} mismatch: {sample_id}"
                )
    return index, {
        "path": path,
        "sha256": expected_manifest_values["query_metadata_sha256"],
        "identity_sha256": identity_sha256,
        "manifest_path": manifest_path,
        "manifest_sha256": sha256_file(manifest_path),
        "source_annotations_path": manifest.get("source_annotations_path"),
        "source_annotations_sha256": manifest.get("source_annotations_sha256"),
    }


def project_query_metadata(
    *,
    annotations_path: Path,
    prediction_bundle: PredictionBundle,
    split: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Create rows in prediction-manifest order after three-way identity checks."""

    expressions = load_expression_index(annotations_path)
    raw_payload = json.loads(annotations_path.read_text(encoding="utf-8"))
    if raw_payload.get("info", {}).get("split") != split:
        raise ValueError("official expression split mismatch")
    frozen_rows = _load_frozen_identity_rows(prediction_bundle)
    if set(expressions) != {
        int(row["question_index"]) for row in prediction_bundle.rows
    }:
        raise ValueError("official expression/prediction question universe mismatch")
    if set(frozen_rows) != set(expressions):
        raise ValueError("frozen split/expression question universe mismatch")

    projected: list[dict[str, Any]] = []
    for offset, prediction in enumerate(prediction_bundle.rows):
        question_index = int(prediction["question_index"])
        scene_id = str(prediction["scene_id"])
        query = str(prediction["query"])
        expression = expressions[question_index]
        expected_expression = {
            "question_index": question_index,
            "image_filename": scene_id,
            "question": query,
            "split": split,
        }
        for field, expected in expected_expression.items():
            if expression.get(field) != expected:
                raise ValueError(
                    f"official expression {field} mismatch at row {offset}"
                )
        frozen = frozen_rows[question_index]
        expected_frozen = {"scene_id": scene_id, "text": query}
        for field, expected in expected_frozen.items():
            if frozen.get(field) != expected:
                raise ValueError(f"frozen split {field} mismatch at row {offset}")
        query_type = str(annotate_expression(expression)["query_type"])
        row = _validate_query_row(
            {
                "sample_id": str(prediction["sample_id"]),
                "question_index": question_index,
                "scene_id": scene_id,
                "query": query,
                "query_type": query_type,
            },
            offset=offset,
        )
        projected.append(row)
    annotations_sha256 = sha256_file(annotations_path)
    return projected, {
        "schema_version": QUERY_METADATA_SCHEMA_VERSION,
        "artifact_type": QUERY_METADATA_ARTIFACT_TYPE,
        "status": "COMPLETED",
        "split": split,
        "fields": list(QUERY_METADATA_FIELDS),
        "exact_schema_required": True,
        "ground_truth_allowed": False,
        "forbidden_fields": list(FORBIDDEN_QUERY_METADATA_FIELDS),
        "row_count": len(projected),
        "query_metadata_identity_sha256": _canonical_hash(projected),
        "source_annotations_path": str(annotations_path),
        "source_annotations_sha256": annotations_sha256,
        "source_annotations_may_contain_ground_truth": True,
        "source_annotations_consumed_only_by_projection": True,
        "feature_extractor_must_not_read_source_annotations": True,
        "source_prediction_manifest_path": str(
            prediction_bundle.manifest_path
        ),
        "source_prediction_manifest_sha256": (
            prediction_bundle.manifest_sha256
        ),
        "source_prediction_identity_sha256": (
            prediction_bundle.identity_sha256
        ),
        "source_frozen_manifest_path": str(
            prediction_bundle.frozen_manifest_path
        ),
        "source_frozen_manifest_sha256": (
            prediction_bundle.frozen_manifest_sha256
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--prediction-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--tmp-root", type=Path, required=True)
    parser.add_argument(
        "--split",
        choices=("train", "development", "val", "test"),
        required=True,
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_root, tmp_root, _ = validate_tmp_scope(
        args.output_root, args.tmp_root
    )
    if output_root.exists():
        raise FileExistsError(
            f"query metadata output already exists: {output_root}"
        )
    annotations_path = args.annotations.expanduser().resolve()
    if not annotations_path.is_file():
        raise FileNotFoundError(f"official expressions missing: {annotations_path}")
    prediction_bundle = load_verified_prediction_bundle(
        args.prediction_root, split=args.split
    )
    rows, manifest = project_query_metadata(
        annotations_path=annotations_path,
        prediction_bundle=prediction_bundle,
        split=args.split,
    )
    output_root.mkdir(parents=True)
    query_metadata_path = output_root / "query_metadata.jsonl"
    atomic_text(
        query_metadata_path,
        "".join(
            json.dumps(
                row,
                sort_keys=True,
                ensure_ascii=False,
                allow_nan=False,
            )
            + "\n"
            for row in rows
        ),
        tmp_root=tmp_root,
    )
    manifest = {
        **manifest,
        "query_metadata_path": str(query_metadata_path),
        "query_metadata_sha256": sha256_file(query_metadata_path),
    }
    atomic_json(
        output_root / "query_metadata_manifest.json",
        manifest,
        tmp_root=tmp_root,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
