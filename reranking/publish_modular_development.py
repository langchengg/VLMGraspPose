"""Publish and independently verify a Modular development feature table.

The historical feature extractor deliberately omitted some raw pose columns.
This publisher restores only immutable candidate geometry by exact
``(sample_id, candidate_id)`` joins against finalized post-NMS Parquets.  Its
companion manifest commits the output, every input, exact candidate/query key
coverage, inferred split/source coverage, and the geometry recovery protocol.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any, Sequence

import numpy as np
import pandas as pd


class PublishError(ValueError):
    """Raised when publication content or provenance cannot be verified."""


SCHEMA_VERSION = 3
KEYS = ("sample_id", "candidate_id")
GEOMETRY = {
    "center_u_px": "x_px",
    "center_v_px": "y_px",
    "center_depth_m": "z_m",
    "angle_rad": "angle_rad",
    "width_m": "width_m",
    "width_px": "width_px",
}
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
RESERVED_COLUMNS = {"__source_path", "__row_order"}


@dataclass(frozen=True)
class _InputTable:
    kind: str
    path: Path
    sha256: str
    frame: pd.DataFrame
    companion: dict[str, Any]


@dataclass(frozen=True)
class _UniverseInput:
    kind: str
    path: Path
    sha256: str
    frame: pd.DataFrame


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _strict_file(raw: str | os.PathLike[str], *, name: str) -> Path:
    unresolved = Path(raw).expanduser()
    if unresolved.is_symlink():
        raise PublishError(f"{name} is missing or symlinked: {unresolved}")
    path = unresolved.resolve()
    if not path.is_file():
        raise PublishError(f"{name} is missing or symlinked: {path}")
    return path


def _read_json(path: Path, *, name: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise PublishError(f"cannot read {name}: {path}") from error
    if not isinstance(value, dict):
        raise PublishError(f"{name} must contain a JSON object: {path}")
    return value


def _normalise_keys(frame: pd.DataFrame, *, name: str) -> pd.DataFrame:
    missing = sorted(set(KEYS) - set(frame.columns))
    if missing:
        raise PublishError(f"{name} lacks keys: {missing}")
    result = frame.copy()
    for column in KEYS:
        if result[column].isna().any():
            raise PublishError(f"{name} contains null {column}")
        result[column] = result[column].astype(str)
        if result[column].str.strip().eq("").any():
            raise PublishError(f"{name} contains empty {column}")
    if result.duplicated(list(KEYS)).any():
        raise PublishError(f"{name} contains duplicate candidate keys")
    return result


def _companion_manifest(
    path: Path, *, kind: str, data_sha256: str, rows: int
) -> dict[str, Any]:
    if kind == "feature":
        candidate = path.parent / "dataset_manifest.json"
        manifest_kind = "feature_dataset_manifest"
    else:
        candidate = path.parent / "run_config.json"
        manifest_kind = "finalized_candidate_run_config"
    if not candidate.exists():
        raise PublishError(f"{kind} companion manifest is missing: {candidate}")
    candidate = _strict_file(candidate, name=f"{kind} companion manifest")
    manifest = _read_json(candidate, name=f"{kind} companion manifest")
    declared_split: str | None = None
    declared_rows: int | None = None
    declared_sha256: str | None = None
    declared_samples: int | None = None
    if kind == "feature":
        if manifest.get("status") != "COMPLETED":
            raise PublishError(
                f"feature companion manifest is not COMPLETED: {candidate}"
            )
        declared_split = manifest.get("split")
        declared_rows = manifest.get("candidate_count")
        declared_sha256 = manifest.get("per_candidate_sha256")
        declared_samples = manifest.get("sample_count")
    else:
        if manifest.get("status") != "COMPLETED":
            raise PublishError(
                f"candidate companion manifest is not COMPLETED: {candidate}"
            )
        protocol = manifest.get("protocol_family_identity", {})
        if isinstance(protocol, dict):
            declared_split = protocol.get("split")
        stage = manifest.get("candidate_stage_artifacts", {}).get("nms", {})
        if isinstance(stage, dict):
            declared_rows = stage.get("rows")
            declared_sha256 = stage.get("sha256")
            declared_path = stage.get("path")
            if declared_path and Path(declared_path).resolve() != path:
                raise PublishError(
                    f"candidate companion manifest points at another NMS table: {path}"
                )
        counts = manifest.get("counts", {})
        if isinstance(counts, dict):
            declared_samples = counts.get("samples")
    if declared_rows is None or int(declared_rows) != rows:
        raise PublishError(f"{kind} companion manifest row count disagrees: {path}")
    if declared_sha256 is None:
        raise PublishError(f"{kind} companion manifest omits SHA-256: {path}")
    declared_sha256 = str(declared_sha256).lower()
    if not SHA256_PATTERN.fullmatch(declared_sha256):
        raise PublishError(f"{kind} companion manifest has invalid SHA-256: {path}")
    if declared_sha256 != data_sha256:
        raise PublishError(f"{kind} companion manifest SHA-256 disagrees: {path}")
    if declared_samples is None or int(declared_samples) < 0:
        raise PublishError(f"{kind} companion manifest has negative sample count")
    if declared_split is not None:
        declared_split = str(declared_split).strip()
        if not declared_split:
            raise PublishError(f"{kind} companion manifest has empty split")
    source_fields = {
        key: manifest.get(key)
        for key in (
            "prediction_manifest_sha256",
            "frozen_split_manifest_sha256",
            "query_metadata_sha256",
            "candidate_protocol_family_identity_sha256",
            "scorer_model_identity_sha256",
            "labels_source_sha256",
        )
        if manifest.get(key) is not None
    }
    return {
        "available": True,
        "kind": manifest_kind,
        "status": str(manifest["status"]),
        "path": str(candidate),
        "sha256": _sha256(candidate),
        "declared_split": declared_split,
        "declared_rows": None if declared_rows is None else int(declared_rows),
        "declared_samples": (
            None if declared_samples is None else int(declared_samples)
        ),
        "declared_per_sample_sha256": (
            manifest.get("per_sample_sha256") if kind == "feature" else None
        ),
        "source_fields": source_fields,
    }


def _load_inputs(
    paths: Sequence[str | os.PathLike[str]], *, kind: str
) -> tuple[_InputTable, ...]:
    if not paths:
        raise PublishError(f"at least one {kind} input is required")
    tables: list[_InputTable] = []
    resolved_paths: set[Path] = set()
    for raw in paths:
        path = _strict_file(raw, name=f"{kind} input")
        if path in resolved_paths:
            raise PublishError(f"duplicate {kind} input path: {path}")
        resolved_paths.add(path)
        sha256 = _sha256(path)
        try:
            frame = pd.read_parquet(path)
        except Exception as error:
            raise PublishError(f"cannot read {kind} Parquet: {path}") from error
        if frame.empty:
            raise PublishError(f"{kind} input is empty: {path}")
        reserved = sorted(RESERVED_COLUMNS & set(frame.columns))
        if reserved:
            raise PublishError(f"{kind} input uses reserved columns: {reserved}")
        frame = _normalise_keys(frame, name=f"{kind} input {path}")
        companion = _companion_manifest(
            path, kind=kind, data_sha256=sha256, rows=len(frame)
        )
        tables.append(
            _InputTable(
                kind=kind,
                path=path,
                sha256=sha256,
                frame=frame,
                companion=companion,
            )
        )
    return tuple(tables)


def _concat(tables: Sequence[_InputTable]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for table in tables:
        frame = table.frame.copy()
        frame["__source_path"] = str(table.path)
        frames.append(frame)
    result = pd.concat(frames, ignore_index=True)
    return _normalise_keys(result, name=f"combined {tables[0].kind} inputs")


def _canonical_scalar(value: Any) -> Any:
    if value is None or pd.isna(value):
        return None
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        numeric = float(value)
        if not np.isfinite(numeric):
            raise PublishError("cannot digest a non-finite numeric value")
        return {"float64_hex": numeric.hex()}
    return str(value)


def _rows_sha256(
    frame: pd.DataFrame, columns: Sequence[str], *, sort_by: Sequence[str]
) -> str:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise PublishError(f"digest columns missing: {missing}")
    ordered = frame.sort_values(list(sort_by), kind="mergesort")
    records = [
        [_canonical_scalar(value) for value in row]
        for row in ordered.loc[:, list(columns)].itertuples(index=False, name=None)
    ]
    return _json_sha256({"columns": list(columns), "rows": records})


def _key_sha256(frame: pd.DataFrame) -> str:
    return _rows_sha256(frame, KEYS, sort_by=KEYS)


def _query_coverage(frame: pd.DataFrame) -> dict[str, Any]:
    queries = pd.DataFrame(
        {"sample_id": sorted(frame["sample_id"].astype(str).unique().tolist())}
    )
    return {
        "count": int(len(queries)),
        "ids_sha256": _rows_sha256(
            queries, ("sample_id",), sort_by=("sample_id",)
        ),
    }


def _read_tabular(path: Path, *, name: str) -> pd.DataFrame:
    try:
        if path.suffix.lower() == ".parquet":
            return pd.read_parquet(path)
        if path.suffix.lower() == ".csv":
            return pd.read_csv(path)
        if path.suffix.lower() in {".jsonl", ".ndjson"}:
            return pd.read_json(path, lines=True)
    except Exception as error:
        raise PublishError(f"cannot read {name}: {path}") from error
    raise PublishError(f"unsupported {name} format: {path}")


def _load_universe_inputs(
    paths: Sequence[str | os.PathLike[str]], *, kind: str
) -> tuple[_UniverseInput, ...]:
    if not paths:
        raise PublishError(f"at least one {kind} input is required")
    inputs: list[_UniverseInput] = []
    seen: set[Path] = set()
    for raw in paths:
        path = _strict_file(raw, name=f"{kind} input")
        if path in seen:
            raise PublishError(f"duplicate {kind} input path: {path}")
        seen.add(path)
        frame = _read_tabular(path, name=kind)
        if frame.empty:
            raise PublishError(f"{kind} input is empty: {path}")
        if "sample_id" not in frame:
            raise PublishError(f"{kind} input lacks sample_id: {path}")
        frame = frame.copy()
        if frame["sample_id"].isna().any():
            raise PublishError(f"{kind} input contains null sample_id: {path}")
        frame["sample_id"] = frame["sample_id"].astype(str).str.strip()
        if frame["sample_id"].eq("").any():
            raise PublishError(f"{kind} input contains empty sample_id: {path}")
        if frame["sample_id"].duplicated().any():
            raise PublishError(f"{kind} input contains duplicate sample_id: {path}")
        inputs.append(
            _UniverseInput(
                kind=kind,
                path=path,
                sha256=_sha256(path),
                frame=frame,
            )
        )
    return tuple(inputs)


def _normalise_query_universe(
    inputs: Sequence[_UniverseInput],
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for item in inputs:
        missing = sorted({"split", "scene_id"} - set(item.frame.columns))
        if missing:
            raise PublishError(f"query universe lacks fields {missing}: {item.path}")
        frame = item.frame[["sample_id", "split", "scene_id"]].copy()
        for column in ("split", "scene_id"):
            if frame[column].isna().any():
                raise PublishError(
                    f"query universe contains null {column}: {item.path}"
                )
            frame[column] = frame[column].astype(str).str.strip()
            if frame[column].eq("").any():
                raise PublishError(
                    f"query universe contains empty {column}: {item.path}"
                )
        splits = set(frame["split"])
        if len(splits) != 1 or not splits <= {"train", "val"}:
            raise PublishError(
                f"each query universe input must identify exactly one train/val split: {item.path}"
            )
        frames.append(frame)
    universe = pd.concat(frames, ignore_index=True)
    if universe["sample_id"].duplicated().any():
        raise PublishError("query universe contains duplicate sample_id across inputs")
    if set(universe["split"]) != {"train", "val"}:
        raise PublishError("development query universe must contain train and val")
    return universe.sort_values("sample_id", kind="mergesort").reset_index(drop=True)


def _normalise_candidate_count_evidence(
    inputs: Sequence[_UniverseInput], universe: pd.DataFrame
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for item in inputs:
        if "candidate_count" not in item.frame:
            raise PublishError(
                f"candidate-count evidence lacks candidate_count: {item.path}"
            )
        frame = item.frame[["sample_id", "candidate_count"]].copy()
        numeric = pd.to_numeric(frame["candidate_count"], errors="coerce")
        if numeric.isna().any() or not np.equal(numeric, np.floor(numeric)).all():
            raise PublishError(
                f"candidate-count evidence is not integral: {item.path}"
            )
        frame["candidate_count"] = numeric.astype(np.int64)
        if frame["candidate_count"].lt(0).any():
            raise PublishError(
                f"candidate-count evidence is negative: {item.path}"
            )
        frames.append(frame)
    evidence = pd.concat(frames, ignore_index=True)
    if evidence["sample_id"].duplicated().any():
        raise PublishError(
            "candidate-count evidence contains duplicate sample_id across inputs"
        )
    universe_ids = set(universe["sample_id"])
    evidence_ids = set(evidence["sample_id"])
    if evidence_ids != universe_ids:
        missing = sorted(universe_ids - evidence_ids)[:5]
        extra = sorted(evidence_ids - universe_ids)[:5]
        raise PublishError(
            "candidate-count evidence does not exactly cover the query universe: "
            f"missing={missing}, extra={extra}"
        )
    return evidence.sort_values("sample_id", kind="mergesort").reset_index(drop=True)


def _universe_evidence(
    *,
    query_inputs: Sequence[_UniverseInput],
    count_inputs: Sequence[_UniverseInput],
    universe: pd.DataFrame,
    counts: pd.DataFrame,
) -> dict[str, Any]:
    def record(item: _UniverseInput, columns: Sequence[str]) -> dict[str, Any]:
        return {
            "path": str(item.path),
            "sha256": item.sha256,
            "rows": int(len(item.frame)),
            "query_ids_sha256": _rows_sha256(
                item.frame, ("sample_id",), sort_by=("sample_id",)
            ),
            "rows_sha256": _rows_sha256(
                item.frame, columns, sort_by=("sample_id",)
            ),
        }

    combined = universe.merge(counts, on="sample_id", validate="one_to_one")
    return {
        "query_universe_inputs": [
            record(item, ("sample_id", "split", "scene_id"))
            for item in query_inputs
        ],
        "candidate_count_evidence_inputs": [
            record(item, ("sample_id", "candidate_count"))
            for item in count_inputs
        ],
        "query_count": int(len(combined)),
        "queries_with_candidates": int(combined["candidate_count"].gt(0).sum()),
        "zero_candidate_queries": int(combined["candidate_count"].eq(0).sum()),
        "candidate_count": int(combined["candidate_count"].sum()),
        "split_query_counts": {
            str(split): int(len(group))
            for split, group in combined.groupby("split", sort=True)
        },
        "query_identity_sha256": _rows_sha256(
            combined,
            ("sample_id", "split", "scene_id", "candidate_count"),
            sort_by=("sample_id",),
        ),
        "coverage_rule": (
            "each query has exactly candidate_count candidate rows; zero is "
            "accepted only when candidate_count=0 is explicitly committed"
        ),
    }


def _assignment(
    frame: pd.DataFrame, column: str, *, source: str
) -> tuple[dict[str, str], dict[str, Any] | None]:
    if column not in frame.columns:
        return {}, None
    if frame[column].isna().any():
        raise PublishError(f"{source} contains null {column}")
    values = frame[column].astype(str).str.strip()
    if values.eq("").any():
        raise PublishError(f"{source} contains empty {column}")
    mapping_frame = pd.DataFrame(
        {"sample_id": frame["sample_id"].astype(str), column: values}
    ).drop_duplicates()
    counts = mapping_frame.groupby("sample_id", sort=True)[column].nunique()
    if counts.gt(1).any():
        bad = sorted(counts.index[counts.gt(1)].tolist())[:5]
        raise PublishError(f"queries map to multiple {column} values in {source}: {bad}")
    mapping = dict(
        mapping_frame.sort_values("sample_id", kind="mergesort").itertuples(
            index=False, name=None
        )
    )
    coverage = {
        "query_count": len(mapping),
        "value_count": len(set(mapping.values())),
        "values": sorted(set(mapping.values())) if column == "split" else None,
        "assignment_sha256": _json_sha256(sorted(mapping.items())),
    }
    return mapping, coverage


def _split_assignment(table: _InputTable) -> tuple[dict[str, str], dict[str, Any]]:
    mapping, frame_coverage = _assignment(
        table.frame, "split", source=str(table.path)
    )
    declared_split = table.companion.get("declared_split")
    if declared_split is not None:
        if mapping and set(mapping.values()) != {declared_split}:
            raise PublishError(
                f"split column disagrees with companion manifest: {table.path}"
            )
        if not mapping:
            mapping = {
                query_id: declared_split
                for query_id in table.frame["sample_id"].astype(str).unique()
            }
    return mapping, {
        "available": bool(mapping),
        "evidence": (
            "column:split"
            if frame_coverage is not None
            else (
                "companion_manifest:declared_split"
                if declared_split is not None
                else None
            )
        ),
        "values": sorted(set(mapping.values())),
        "query_count": len(mapping),
        "assignment_sha256": _json_sha256(sorted(mapping.items())),
    }


def _table_evidence(table: _InputTable) -> dict[str, Any]:
    _, split_coverage = _split_assignment(table)
    _, scene_coverage = _assignment(
        table.frame, "scene_id", source=str(table.path)
    )
    schema = [[str(column), str(dtype)] for column, dtype in table.frame.dtypes.items()]
    return {
        "path": str(table.path),
        "sha256": table.sha256,
        "rows": int(len(table.frame)),
        "queries": int(table.frame["sample_id"].nunique()),
        "candidate_keys_sha256": _key_sha256(table.frame),
        "query_ids_sha256": _query_coverage(table.frame)["ids_sha256"],
        "schema_sha256": _json_sha256(schema),
        "split_coverage": split_coverage,
        "scene_source_coverage": (
            {"available": False}
            if scene_coverage is None
            else {"available": True, **scene_coverage}
        ),
        "companion_manifest": table.companion,
    }


def _combine_assignments(
    tables: Sequence[_InputTable],
    *,
    kind: str,
) -> tuple[dict[str, str], list[str]]:
    combined: dict[str, str] = {}
    evidence: list[str] = []
    for table in tables:
        if kind == "split":
            mapping, details = _split_assignment(table)
            if details["evidence"] is not None:
                evidence.append(f"{table.kind}:{table.path}:{details['evidence']}")
        else:
            mapping, details = _assignment(
                table.frame, "scene_id", source=str(table.path)
            )
            if details is not None:
                evidence.append(f"{table.kind}:{table.path}:column:scene_id")
        for query_id, value in mapping.items():
            previous = combined.setdefault(query_id, value)
            if previous != value:
                raise PublishError(
                    f"query {query_id} has conflicting {kind} coverage: {previous}/{value}"
                )
    return combined, evidence


def _publication_coverage(
    feature_tables: Sequence[_InputTable],
    candidate_tables: Sequence[_InputTable],
    features: pd.DataFrame,
    universe: pd.DataFrame,
    counts: pd.DataFrame,
    universe_evidence: dict[str, Any],
) -> dict[str, Any]:
    tables = [*feature_tables, *candidate_tables]
    queries = sorted(universe["sample_id"].astype(str).tolist())
    query_set = set(queries)
    split_mapping, split_evidence = _combine_assignments(tables, kind="split")
    scene_mapping, scene_evidence = _combine_assignments(tables, kind="scene")
    extra_split = sorted(set(split_mapping) - query_set)
    extra_scene = sorted(set(scene_mapping) - query_set)
    if extra_split or extra_scene:
        raise PublishError("coverage contains queries outside the published candidate keys")

    authoritative_split = dict(
        universe[["sample_id", "split"]].itertuples(index=False, name=None)
    )
    authoritative_scene = dict(
        universe[["sample_id", "scene_id"]].itertuples(index=False, name=None)
    )
    for name, observed, authoritative in (
        ("split", split_mapping, authoritative_split),
        ("scene_id", scene_mapping, authoritative_scene),
    ):
        conflicts = sorted(
            query
            for query, value in observed.items()
            if authoritative.get(query) != value
        )
        if conflicts:
            raise PublishError(
                f"candidate rows disagree with query-universe {name}: {conflicts[:5]}"
            )

    def coverage(
        mapping: dict[str, str], evidence: list[str], *, expose_values: bool
    ) -> dict[str, Any]:
        assigned = sorted((query, mapping[query]) for query in query_set & set(mapping))
        values = sorted({value for _, value in assigned})
        return {
            "available": bool(assigned),
            "complete": len(assigned) == len(queries),
            "assigned_query_count": len(assigned),
            "unassigned_query_count": len(queries) - len(assigned),
            "values": values if expose_values else None,
            "value_count": len(values),
            "assignment_sha256": _json_sha256(assigned),
            "evidence": sorted(evidence),
        }

    return {
        "query": {
            "count": len(queries),
            "ids_sha256": _rows_sha256(
                universe, ("sample_id",), sort_by=("sample_id",)
            ),
            "queries_with_candidates": int(counts["candidate_count"].gt(0).sum()),
            "zero_candidate_queries": int(counts["candidate_count"].eq(0).sum()),
            "all_feature_candidate_keys_covered_once": True,
            "all_universe_queries_accounted_for": True,
        },
        "split": {
            **coverage(authoritative_split, split_evidence, expose_values=True),
            "evidence": sorted(
                str(item["path"])
                for item in universe_evidence["query_universe_inputs"]
            ),
        },
        "scene_source": {
            **coverage(authoritative_scene, scene_evidence, expose_values=False),
            "evidence": sorted(
                str(item["path"])
                for item in universe_evidence["query_universe_inputs"]
            ),
        },
        "input_source": {
            "feature_input_count": len(feature_tables),
            "candidate_input_count": len(candidate_tables),
            "all_input_files_sha256_committed": True,
            "feature_paths_sha256": _json_sha256(
                sorted(str(table.path) for table in feature_tables)
            ),
            "candidate_paths_sha256": _json_sha256(
                sorted(str(table.path) for table in candidate_tables)
            ),
            "companion_manifest_count": sum(
                int(bool(table.companion.get("available"))) for table in tables
            ),
        },
    }


def _validate_universe_candidate_coverage(
    *,
    feature_tables: Sequence[_InputTable],
    candidate_tables: Sequence[_InputTable],
    features: pd.DataFrame,
    candidates: pd.DataFrame,
    count_inputs: Sequence[_UniverseInput],
    universe: pd.DataFrame,
    counts: pd.DataFrame,
) -> None:
    expected_count_paths = {
        table.path.parent / "per_sample.parquet" for table in feature_tables
    }
    observed_count_paths = {item.path for item in count_inputs}
    if observed_count_paths != expected_count_paths:
        raise PublishError(
            "candidate-count evidence must be each feature input's sibling "
            "per_sample.parquet"
        )
    by_path = {item.path: item for item in count_inputs}
    for table in feature_tables:
        evidence = by_path[table.path.parent / "per_sample.parquet"]
        declared = table.companion.get("declared_per_sample_sha256")
        if not isinstance(declared, str) or not SHA256_PATTERN.fullmatch(
            declared.lower()
        ):
            raise PublishError(
                f"feature companion manifest omits valid per-sample SHA-256: {table.path}"
            )
        if evidence.sha256 != declared.lower():
            raise PublishError(
                f"candidate-count evidence SHA-256 disagrees with feature companion: {evidence.path}"
            )

    query_count = len(universe)
    if sum(int(table.companion["declared_samples"]) for table in feature_tables) != query_count:
        raise PublishError("feature companion sample counts do not equal query universe")
    if sum(int(table.companion["declared_samples"]) for table in candidate_tables) != query_count:
        raise PublishError("candidate companion sample counts do not equal query universe")

    expected = counts.set_index("sample_id")["candidate_count"].astype(np.int64)
    observed = features.groupby("sample_id", sort=True).size().astype(np.int64)
    observed = observed.reindex(expected.index, fill_value=0)
    if not observed.equals(expected):
        mismatches = sorted(expected.index[observed.ne(expected)].tolist())[:5]
        raise PublishError(
            "candidate rows do not match per-query candidate-count evidence: "
            f"{mismatches}"
        )
    candidate_observed = candidates.groupby("sample_id", sort=True).size().astype(
        np.int64
    )
    candidate_observed = candidate_observed.reindex(expected.index, fill_value=0)
    if not candidate_observed.equals(expected):
        mismatches = sorted(
            expected.index[candidate_observed.ne(expected)].tolist()
        )[:5]
        raise PublishError(
            "NMS candidate rows do not match per-query candidate-count evidence: "
            f"{mismatches}"
        )


def _build_joined(features: pd.DataFrame, candidates: pd.DataFrame) -> pd.DataFrame:
    missing_geometry = sorted(set(GEOMETRY) - set(candidates.columns))
    if missing_geometry:
        raise PublishError(f"candidate inputs lack geometry: {missing_geometry}")
    feature_keys = pd.MultiIndex.from_frame(features[list(KEYS)])
    candidate_keys = pd.MultiIndex.from_frame(candidates[list(KEYS)])
    if set(feature_keys) != set(candidate_keys):
        missing_features = sorted(set(candidate_keys) - set(feature_keys))[:5]
        missing_candidates = sorted(set(feature_keys) - set(candidate_keys))[:5]
        raise PublishError(
            "feature/candidate key sets differ: "
            f"missing_features={missing_features}, missing_candidates={missing_candidates}"
        )
    working = features.copy()
    working["__row_order"] = np.arange(len(working), dtype=np.int64)
    projection = candidates[[*KEYS, *GEOMETRY]].rename(columns=GEOMETRY)
    joined = working.merge(
        projection,
        on=list(KEYS),
        how="left",
        validate="one_to_one",
        suffixes=("", "__candidate"),
        sort=False,
    )
    for target in GEOMETRY.values():
        candidate_column = f"{target}__candidate" if target in features.columns else target
        candidate_values = pd.to_numeric(
            joined[candidate_column], errors="coerce"
        ).to_numpy(np.float64)
        if not np.isfinite(candidate_values).all():
            raise PublishError(f"candidate geometry {target} is non-finite")
        if target in features.columns:
            existing = pd.to_numeric(joined[target], errors="coerce").to_numpy(
                np.float64
            )
            finite = np.isfinite(existing)
            if finite.any() and not np.array_equal(
                existing[finite].astype(np.float32),
                candidate_values[finite].astype(np.float32),
            ):
                raise PublishError(f"existing geometry disagrees for {target}")
        joined[target] = candidate_values
        if candidate_column != target:
            joined = joined.drop(columns=candidate_column)
    joined = joined.sort_values("__row_order", kind="mergesort").drop(
        columns=["__row_order", "__source_path"]
    )
    if len(joined) != len(features):
        raise PublishError("geometry join changed row count")
    return joined


def _geometry_protocol(output_frame: pd.DataFrame) -> dict[str, Any]:
    geometry_columns = tuple(GEOMETRY.values())
    return {
        "source": "finalized post-NMS candidate Parquet",
        "source_to_output_columns": GEOMETRY,
        "join_keys": list(KEYS),
        "join_cardinality": "one_to_one",
        "candidate_key_relation": "exact set equality",
        "output_row_order": "feature input concatenation order",
        "candidate_geometry_requirement": "finite float64",
        "existing_geometry_equality": "exact after canonical float32 round-trip",
        "labels_or_scores_modified": False,
        "output_geometry_sha256": _rows_sha256(
            output_frame,
            (*KEYS, *geometry_columns),
            sort_by=KEYS,
        ),
    }


def _manifest_payload(
    *,
    output: Path,
    output_frame: pd.DataFrame,
    feature_tables: Sequence[_InputTable],
    candidate_tables: Sequence[_InputTable],
    features: pd.DataFrame,
    query_inputs: Sequence[_UniverseInput],
    count_inputs: Sequence[_UniverseInput],
    universe: pd.DataFrame,
    counts: pd.DataFrame,
) -> dict[str, Any]:
    feature_evidence = [_table_evidence(table) for table in feature_tables]
    candidate_evidence = [_table_evidence(table) for table in candidate_tables]
    universe_sources = _universe_evidence(
        query_inputs=query_inputs,
        count_inputs=count_inputs,
        universe=universe,
        counts=counts,
    )
    coverage = _publication_coverage(
        feature_tables,
        candidate_tables,
        features,
        universe,
        counts,
        universe_sources,
    )
    geometry_protocol = _geometry_protocol(output_frame)
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": "modular_development_geometry_complete_features",
        "status": "COMPLETE",
        "output": str(output),
        "output_sha256": _sha256(output),
        "rows": int(len(output_frame)),
        "queries": int(len(universe)),
        "queries_with_candidates": int(output_frame["sample_id"].nunique()),
        "output_candidate_keys_sha256": _key_sha256(output_frame),
        "output_query_ids_sha256": _query_coverage(output_frame)["ids_sha256"],
        "feature_inputs": feature_evidence,
        "candidate_inputs": candidate_evidence,
        "development_query_universe": universe_sources,
        "coverage": coverage,
        "geometry_recovery_protocol": geometry_protocol,
        # Compatibility fields retained for existing report/test readers.
        "join": "strict one-to-one on sample_id+candidate_id",
        "existing_geometry_equality": geometry_protocol[
            "existing_geometry_equality"
        ],
        "geometry_columns": sorted(set(GEOMETRY.values())),
        "labels_or_scores_modified": False,
    }


def _write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def publish_modular_development(
    feature_paths: Sequence[str | os.PathLike[str]],
    nms_candidate_paths: Sequence[str | os.PathLike[str]],
    output_path: str | os.PathLike[str],
    *,
    query_universe_paths: Sequence[str | os.PathLike[str]],
    candidate_count_evidence_paths: Sequence[str | os.PathLike[str]],
) -> dict[str, Any]:
    """Restore geometry and write a deterministic provenance companion manifest."""

    feature_tables = _load_inputs(feature_paths, kind="feature")
    candidate_tables = _load_inputs(nms_candidate_paths, kind="candidate")
    query_inputs = _load_universe_inputs(
        query_universe_paths, kind="query universe"
    )
    count_inputs = _load_universe_inputs(
        candidate_count_evidence_paths, kind="candidate-count evidence"
    )
    output = Path(output_path).expanduser().resolve()
    input_paths = {
        item.path
        for item in (
            *feature_tables,
            *candidate_tables,
            *query_inputs,
            *count_inputs,
        )
    }
    if output in input_paths:
        raise PublishError("output must not overwrite an input Parquet")
    features = _concat(feature_tables)
    candidates = _concat(candidate_tables)
    universe = _normalise_query_universe(query_inputs)
    counts = _normalise_candidate_count_evidence(count_inputs, universe)
    _validate_universe_candidate_coverage(
        feature_tables=feature_tables,
        candidate_tables=candidate_tables,
        features=features,
        candidates=candidates,
        count_inputs=count_inputs,
        universe=universe,
        counts=counts,
    )
    joined = _build_joined(features, candidates)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    joined.to_parquet(temporary, index=False, compression="zstd")
    os.replace(temporary, output)
    payload = _manifest_payload(
        output=output,
        output_frame=joined,
        feature_tables=feature_tables,
        candidate_tables=candidate_tables,
        features=features,
        query_inputs=query_inputs,
        count_inputs=count_inputs,
        universe=universe,
        counts=counts,
    )
    manifest = {
        **payload,
        "manifest_payload_sha256": _json_sha256(payload),
    }
    _write_json_atomic(output.with_name(output.name + ".manifest.json"), manifest)
    return manifest


def verify_modular_development_publication(
    output_path: str | os.PathLike[str],
    manifest_path: str | os.PathLike[str] | None = None,
    *,
    expected_query_universe_paths: Sequence[str | os.PathLike[str]] | None = None,
    expected_candidate_count_evidence_paths: Sequence[
        str | os.PathLike[str]
    ]
    | None = None,
) -> dict[str, Any]:
    """Recompute a publication from committed inputs and reject any divergence."""

    output = _strict_file(output_path, name="published output")
    manifest_file = _strict_file(
        manifest_path or output.with_name(output.name + ".manifest.json"),
        name="publication companion manifest",
    )
    manifest = _read_json(manifest_file, name="publication companion manifest")
    recorded_digest = manifest.get("manifest_payload_sha256")
    payload = {
        key: value for key, value in manifest.items() if key != "manifest_payload_sha256"
    }
    if not isinstance(recorded_digest, str) or recorded_digest != _json_sha256(payload):
        raise PublishError("publication manifest payload digest mismatch")
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise PublishError("unsupported publication manifest schema")
    if manifest.get("status") != "COMPLETE":
        raise PublishError("publication manifest is not COMPLETE")
    if Path(str(manifest.get("output", ""))).resolve() != output:
        raise PublishError("publication manifest identifies another output")
    feature_records = manifest.get("feature_inputs")
    candidate_records = manifest.get("candidate_inputs")
    if not isinstance(feature_records, list) or not feature_records:
        raise PublishError("publication manifest omits feature input sources")
    if not isinstance(candidate_records, list) or not candidate_records:
        raise PublishError("publication manifest omits candidate input sources")
    universe_record = manifest.get("development_query_universe")
    if not isinstance(universe_record, dict):
        raise PublishError("publication manifest omits development query universe")
    query_records = universe_record.get("query_universe_inputs")
    count_records = universe_record.get("candidate_count_evidence_inputs")
    if not isinstance(query_records, list) or not query_records:
        raise PublishError("publication manifest omits query-universe inputs")
    if not isinstance(count_records, list) or not count_records:
        raise PublishError("publication manifest omits candidate-count evidence inputs")
    try:
        feature_paths = [record["path"] for record in feature_records]
        candidate_paths = [record["path"] for record in candidate_records]
        query_paths = [record["path"] for record in query_records]
        count_paths = [record["path"] for record in count_records]
    except (KeyError, TypeError) as error:
        raise PublishError("publication manifest has malformed input sources") from error
    feature_tables = _load_inputs(feature_paths, kind="feature")
    candidate_tables = _load_inputs(candidate_paths, kind="candidate")
    query_inputs = _load_universe_inputs(query_paths, kind="query universe")
    count_inputs = _load_universe_inputs(
        count_paths, kind="candidate-count evidence"
    )
    for expected, observed, name in (
        (expected_query_universe_paths, query_paths, "query universe"),
        (
            expected_candidate_count_evidence_paths,
            count_paths,
            "candidate-count evidence",
        ),
    ):
        if expected is not None and [
            str(Path(path).expanduser().resolve()) for path in expected
        ] != [str(Path(path).resolve()) for path in observed]:
            raise PublishError(
                f"publication manifest identifies unexpected {name} inputs"
            )
    features = _concat(feature_tables)
    candidates = _concat(candidate_tables)
    universe = _normalise_query_universe(query_inputs)
    counts = _normalise_candidate_count_evidence(count_inputs, universe)
    _validate_universe_candidate_coverage(
        feature_tables=feature_tables,
        candidate_tables=candidate_tables,
        features=features,
        candidates=candidates,
        count_inputs=count_inputs,
        universe=universe,
        counts=counts,
    )
    expected_output = _build_joined(features, candidates)
    try:
        observed_output = pd.read_parquet(output)
    except Exception as error:
        raise PublishError(f"cannot read published output: {output}") from error
    observed_output = _normalise_keys(observed_output, name="published output")
    try:
        pd.testing.assert_frame_equal(
            observed_output.reset_index(drop=True),
            expected_output.reset_index(drop=True),
            check_dtype=True,
            check_exact=True,
        )
    except AssertionError as error:
        raise PublishError("published output is not an exact reconstruction") from error
    expected_payload = _manifest_payload(
        output=output,
        output_frame=observed_output,
        feature_tables=feature_tables,
        candidate_tables=candidate_tables,
        features=features,
        query_inputs=query_inputs,
        count_inputs=count_inputs,
        universe=universe,
        counts=counts,
    )
    if payload != expected_payload:
        raise PublishError(
            "publication manifest provenance/coverage differs from recomputed evidence"
        )
    return manifest


def _parse(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature", action="append", default=[])
    parser.add_argument("--nms-candidates", action="append", default=[])
    parser.add_argument("--query-universe", action="append", default=[])
    parser.add_argument(
        "--candidate-count-evidence", action="append", default=[]
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="verify --output and its companion manifest without publishing",
    )
    parser.add_argument("--manifest", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse(argv)
    if args.verify_only:
        manifest = verify_modular_development_publication(
            args.output,
            args.manifest,
            expected_query_universe_paths=args.query_universe or None,
            expected_candidate_count_evidence_paths=(
                args.candidate_count_evidence or None
            ),
        )
    else:
        manifest = publish_modular_development(
            args.feature,
            args.nms_candidates,
            args.output,
            query_universe_paths=args.query_universe,
            candidate_count_evidence_paths=args.candidate_count_evidence,
        )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
