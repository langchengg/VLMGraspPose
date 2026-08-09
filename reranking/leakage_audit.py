"""Entity-level development/test and model-fit leakage evidence.

The audit deliberately derives identities from source metadata and concrete fit
rows.  Human-readable declarations such as ``fit_scope`` are not accepted as
evidence that a model avoided the retrospective test set.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Any

import pandas as pd


SCHEMA_VERSION = 1
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
IDENTITY_COLUMNS = (
    "dataset",
    "split",
    "query_id",
    "scene_id",
    "frame_id",
    "group_identity_sha256",
    "rgb_sha256",
    "depth_sha256",
)
OVERLAP_FIELDS = (
    ("query_id", "query_id"),
    ("scene_id", "scene_id"),
    ("frame_identity", "group_identity_sha256"),
    ("rgb_sha256", "rgb_sha256"),
    ("depth_sha256", "depth_sha256"),
)

_ALIASES: dict[str, tuple[str, ...]] = {
    "query_id": ("query_id", "sample_id", "source_sample_id"),
    "scene_id": ("scene_id", "source_scene_id"),
    "frame_id": ("frame_id", "source_frame_id"),
    "rgb_sha256": ("rgb_sha256", "source_rgb_sha256", "native_rgb_sha256"),
    "depth_sha256": (
        "depth_sha256",
        "source_depth_sha256",
        "native_depth_sha256",
    ),
    "rgb_path": (
        "source_rgb_path",
        "native_rgb_path",
        "rgb_path",
        "processed_rgb_path",
        "image_path",
    ),
    "depth_path": (
        "source_depth_path",
        "native_depth_path",
        "depth_path",
        "processed_depth_path",
    ),
}


class LeakageAuditError(ValueError):
    """Raised when source identity evidence is malformed or incomplete."""


@dataclass(frozen=True)
class LeakageAuditResult:
    """In-memory evidence and deterministic summary for one audit."""

    summary: Mapping[str, Any]
    identity_rows: pd.DataFrame
    identity_overlaps: pd.DataFrame
    fold_fit_identities: pd.DataFrame
    fold_fit_overlaps: pd.DataFrame
    fit_resolution_errors: pd.DataFrame

    @property
    def passed(self) -> bool:
        return bool(self.summary["passed"])

    def write_bundle(self, output_dir: str | Path) -> tuple[Path, ...]:
        """Write canonical JSON/JSONL evidence plus a file checksum manifest."""

        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        artifacts: list[Path] = []
        tables = {
            "identity_rows.jsonl": self.identity_rows,
            "identity_overlaps.jsonl": self.identity_overlaps,
            "fold_fit_identities.jsonl": self.fold_fit_identities,
            "fold_fit_overlaps.jsonl": self.fold_fit_overlaps,
            "fit_resolution_errors.jsonl": self.fit_resolution_errors,
        }
        for name, frame in tables.items():
            path = output / name
            _write_jsonl(path, frame)
            artifacts.append(path)
        summary_path = output / "summary.json"
        _write_json(summary_path, self.summary)
        artifacts.append(summary_path)
        markdown_path = output / "LEAKAGE_AUDIT.md"
        markdown_path.write_text(_render_markdown(self), encoding="utf-8")
        artifacts.append(markdown_path)
        checksum_path = output / "artifacts.sha256"
        checksum_path.write_text(
            "".join(f"{_file_sha256(path)}  {path.name}\n" for path in artifacts),
            encoding="utf-8",
        )
        artifacts.append(checksum_path)
        return tuple(artifacts)


def verify_leakage_audit_bundle(
    output_dir: str | Path, *, require_fit_evidence: bool = True
) -> LeakageAuditResult:
    """Independently verify checksums, canonical digests, and overlap details."""

    output = Path(output_dir)
    table_names = (
        "identity_rows.jsonl",
        "identity_overlaps.jsonl",
        "fold_fit_identities.jsonl",
        "fold_fit_overlaps.jsonl",
        "fit_resolution_errors.jsonl",
    )
    required_names = {*table_names, "summary.json", "LEAKAGE_AUDIT.md"}
    checksum_path = output / "artifacts.sha256"
    if not checksum_path.is_file():
        raise LeakageAuditError(f"audit checksum manifest is missing: {checksum_path}")
    declared: dict[str, str] = {}
    for line_number, line in enumerate(
        checksum_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        parts = line.split("  ", maxsplit=1)
        if len(parts) != 2 or not SHA256_PATTERN.fullmatch(parts[0]):
            raise LeakageAuditError(f"malformed checksum line {line_number}")
        name = parts[1]
        if name in declared or Path(name).name != name:
            raise LeakageAuditError(f"unsafe or duplicate checksum artifact: {name}")
        declared[name] = parts[0]
    if set(declared) != required_names:
        raise LeakageAuditError(
            f"checksum artifact set mismatch: {sorted(set(declared) ^ required_names)}"
        )
    for name, expected in declared.items():
        path = output / name
        if not path.is_file() or _file_sha256(path) != expected:
            raise LeakageAuditError(f"audit artifact checksum mismatch: {name}")

    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    digest = summary.pop("audit_digest_sha256", None)
    if digest != _json_sha256(summary):
        raise LeakageAuditError("summary audit_digest_sha256 mismatch")
    summary["audit_digest_sha256"] = digest
    frames = {name: _read_jsonl(output / name) for name in table_names}
    identity_rows = frames["identity_rows.jsonl"]
    if "audit_role" not in identity_rows:
        raise LeakageAuditError("identity rows do not identify development/test audit role")
    roles = set(identity_rows["audit_role"].astype(str))
    if roles != {"development", "test"}:
        raise LeakageAuditError(f"unexpected identity audit roles: {sorted(roles)}")
    development = _validate_identity_rows(
        identity_rows.loc[identity_rows["audit_role"].eq("development")],
        name="development",
    )
    test = _validate_identity_rows(
        identity_rows.loc[identity_rows["audit_role"].eq("test")], name="test"
    )
    stored_overlaps = frames["identity_overlaps.jsonl"]
    recomputed_overlaps = _identity_overlaps(development, test)
    if frame_sha256(stored_overlaps) != frame_sha256(recomputed_overlaps):
        raise LeakageAuditError("stored identity overlap details are not reproducible")
    fit_rows = frames["fold_fit_identities.jsonl"]
    stored_fit_overlaps = frames["fold_fit_overlaps.jsonl"]
    recomputed_fit_overlaps = _fit_overlaps(fit_rows, test)
    if frame_sha256(stored_fit_overlaps) != frame_sha256(recomputed_fit_overlaps):
        raise LeakageAuditError("stored fold-fit overlap details are not reproducible")
    fit_errors = frames["fit_resolution_errors.jsonl"]
    evidence = summary.get("evidence_digests", {})
    expected_digests = {
        "identity_rows_sha256": frame_sha256(identity_rows),
        "identity_overlaps_sha256": frame_sha256(stored_overlaps),
        "fold_fit_identities_sha256": frame_sha256(fit_rows),
        "fold_fit_overlaps_sha256": frame_sha256(stored_fit_overlaps),
        "fit_resolution_errors_sha256": frame_sha256(fit_errors),
    }
    if evidence != expected_digests:
        raise LeakageAuditError("summary evidence digests do not match audit tables")
    recomputed_pass = bool(
        stored_overlaps.empty and stored_fit_overlaps.empty and fit_errors.empty
    )
    if bool(summary.get("passed")) != recomputed_pass:
        raise LeakageAuditError("summary pass state disagrees with evidence")
    if int(summary.get("development_query_count", -1)) != len(development):
        raise LeakageAuditError("development query count mismatch")
    if int(summary.get("test_query_count", -1)) != len(test):
        raise LeakageAuditError("test query count mismatch")
    if require_fit_evidence and not bool(summary.get("fit_membership_audited")):
        raise LeakageAuditError("bundle has no complete concrete fold-fit evidence")
    result = LeakageAuditResult(
        summary=summary,
        identity_rows=identity_rows,
        identity_overlaps=stored_overlaps,
        fold_fit_identities=fit_rows,
        fold_fit_overlaps=stored_fit_overlaps,
        fit_resolution_errors=fit_errors,
    )
    recorded_markdown = (output / "LEAKAGE_AUDIT.md").read_text(encoding="utf-8")
    if recorded_markdown != _render_markdown(result):
        raise LeakageAuditError("LEAKAGE_AUDIT.md is not reproducible from evidence")
    return result


def _load_source(source: pd.DataFrame | str | Path) -> tuple[pd.DataFrame, Path | None]:
    if isinstance(source, pd.DataFrame):
        return source.copy(), None
    path = Path(source).resolve()
    if not path.is_file():
        raise LeakageAuditError(f"source metadata does not exist: {path}")
    suffixes = tuple(path.suffixes)
    if suffixes[-1:] == (".parquet",):
        frame = pd.read_parquet(path)
    elif suffixes[-1:] == (".csv",):
        frame = pd.read_csv(path)
    elif suffixes[-1:] in ((".jsonl",), (".ndjson",)):
        frame = pd.read_json(path, lines=True, convert_dates=False)
    elif suffixes[-1:] == (".json",):
        frame = pd.read_json(path, convert_dates=False)
    else:
        raise LeakageAuditError(f"unsupported source metadata format: {path}")
    return frame, path.parent


def _column(
    frame: pd.DataFrame,
    logical_name: str,
    overrides: Mapping[str, str] | None,
    *,
    required: bool,
) -> str | None:
    if overrides and logical_name in overrides:
        value = str(overrides[logical_name])
        if value not in frame.columns:
            raise LeakageAuditError(
                f"identity column override {logical_name}={value!r} is absent"
            )
        return value
    for candidate in _ALIASES[logical_name]:
        if candidate in frame.columns:
            return candidate
    if required:
        raise LeakageAuditError(
            f"source metadata has no {logical_name} column; tried {_ALIASES[logical_name]}"
        )
    return None


def _text(value: Any, *, column: str, row: int) -> str:
    if pd.isna(value):
        raise LeakageAuditError(f"{column} is null at source row {row}")
    result = str(value).strip()
    if not result:
        raise LeakageAuditError(f"{column} is empty at source row {row}")
    return result


def _sha256(value: Any, *, column: str, row: int) -> str | None:
    if value is None or pd.isna(value) or not str(value).strip():
        return None
    result = str(value).strip().lower()
    if not SHA256_PATTERN.fullmatch(result):
        raise LeakageAuditError(f"{column} is not a SHA-256 digest at source row {row}")
    return result


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_path(value: Any, *, base_dir: Path | None, column: str, row: int) -> Path:
    text = _text(value, column=column, row=row)
    path = Path(text).expanduser()
    if not path.is_absolute():
        if base_dir is None:
            raise LeakageAuditError(
                f"relative {column} requires base_dir at source row {row}: {text}"
            )
        path = base_dir / path
    path = path.resolve()
    if not path.is_file():
        raise LeakageAuditError(f"{column} does not exist at source row {row}: {path}")
    return path


def _json_sha256(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _normalise_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _normalise_json(item) for key, item in sorted(value.items())}
    if isinstance(value, (list, tuple, set)):
        items = [_normalise_json(item) for item in value]
        return sorted(items, key=lambda item: json.dumps(item, sort_keys=True))
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item"):
        return value.item()
    if pd.isna(value):
        return None
    return value


def frame_sha256(frame: pd.DataFrame, columns: Sequence[str] | None = None) -> str:
    """Hash a table as sorted canonical JSON records, independent of row order."""

    selected = tuple(columns or sorted(map(str, frame.columns)))
    missing = sorted(set(selected) - set(frame.columns))
    if missing:
        raise LeakageAuditError(f"digest columns missing: {missing}")
    records = [
        {column: _normalise_json(row[column]) for column in selected}
        for row in frame.loc[:, selected].to_dict(orient="records")
    ]
    records.sort(key=lambda row: json.dumps(row, sort_keys=True, ensure_ascii=False))
    return _json_sha256(records)


def build_identity_rows(
    source: pd.DataFrame | str | Path,
    *,
    dataset: str,
    split: str,
    columns: Mapping[str, str] | None = None,
    base_dir: str | Path | None = None,
    verify_declared_hashes: bool = False,
) -> pd.DataFrame:
    """Materialize one canonical source identity row per query.

    SHA-256 values are read from source metadata when present.  When absent,
    the corresponding source file is hashed.  If ``verify_declared_hashes`` is
    true and a path is available, declared and recomputed hashes must agree.
    Candidate-level repeated rows are collapsed only when all identities agree.
    """

    frame, inferred_base = _load_source(source)
    if frame.empty:
        raise LeakageAuditError("source metadata is empty")
    dataset_value = str(dataset).strip()
    split_value = str(split).strip()
    if not dataset_value or not split_value:
        raise LeakageAuditError("dataset and split must be non-empty")
    root = Path(base_dir).resolve() if base_dir is not None else inferred_base
    query_col = _column(frame, "query_id", columns, required=True)
    scene_col = _column(frame, "scene_id", columns, required=True)
    frame_col = _column(frame, "frame_id", columns, required=False)
    rgb_sha_col = _column(frame, "rgb_sha256", columns, required=False)
    depth_sha_col = _column(frame, "depth_sha256", columns, required=False)
    rgb_path_col = _column(frame, "rgb_path", columns, required=False)
    depth_path_col = _column(frame, "depth_path", columns, required=False)
    if rgb_sha_col is None and rgb_path_col is None:
        raise LeakageAuditError("source metadata has neither RGB SHA-256 nor RGB path")
    if depth_sha_col is None and depth_path_col is None:
        raise LeakageAuditError("source metadata has neither depth SHA-256 nor depth path")

    hash_cache: dict[Path, str] = {}

    def cached_file_hash(path: Path) -> str:
        value = hash_cache.get(path)
        if value is None:
            value = _file_sha256(path)
            hash_cache[path] = value
        return value

    def media_hash(
        row_data: pd.Series,
        row_number: int,
        *,
        sha_col: str | None,
        path_col: str | None,
        media: str,
    ) -> str:
        declared = (
            _sha256(row_data[sha_col], column=sha_col, row=row_number)
            if sha_col is not None
            else None
        )
        has_path = (
            path_col is not None
            and not pd.isna(row_data[path_col])
            and bool(str(row_data[path_col]).strip())
        )
        computed: str | None = None
        if declared is None or (verify_declared_hashes and has_path):
            if not has_path:
                raise LeakageAuditError(
                    f"{media} SHA-256 is missing and no source path is available at row {row_number}"
                )
            path = _resolve_path(
                row_data[path_col], base_dir=root, column=str(path_col), row=row_number
            )
            computed = cached_file_hash(path)
        if declared is not None and computed is not None and declared != computed:
            raise LeakageAuditError(
                f"declared {media} SHA-256 disagrees with file at source row {row_number}"
            )
        return str(declared or computed)

    rows: list[dict[str, Any]] = []
    for row_number, row in frame.reset_index(drop=True).iterrows():
        query_id = _text(row[query_col], column=str(query_col), row=row_number)
        scene_id = _text(row[scene_col], column=str(scene_col), row=row_number)
        frame_id = (
            _text(row[frame_col], column=str(frame_col), row=row_number)
            if frame_col is not None
            else scene_id
        )
        identity = {
            "dataset": dataset_value,
            "split": split_value,
            "query_id": query_id,
            "scene_id": scene_id,
            "frame_id": frame_id,
            "group_identity_sha256": _json_sha256([scene_id, frame_id]),
            "rgb_sha256": media_hash(
                row,
                row_number,
                sha_col=rgb_sha_col,
                path_col=rgb_path_col,
                media="RGB",
            ),
            "depth_sha256": media_hash(
                row,
                row_number,
                sha_col=depth_sha_col,
                path_col=depth_path_col,
                media="depth",
            ),
        }
        identity["identity_row_sha256"] = _json_sha256(identity)
        rows.append(identity)
    result = pd.DataFrame(rows)
    conflicts = result.groupby(["dataset", "query_id"], sort=True).agg(
        identity_count=("identity_row_sha256", "nunique")
    )
    bad = conflicts.index[conflicts["identity_count"].ne(1)].tolist()
    if bad:
        raise LeakageAuditError(f"query IDs map to conflicting source identities: {bad[:5]}")
    return (
        result.drop_duplicates(["dataset", "query_id"])
        .sort_values(["dataset", "query_id"], kind="mergesort")
        .reset_index(drop=True)
    )


def _validate_identity_rows(frame: pd.DataFrame, *, name: str) -> pd.DataFrame:
    required = set(IDENTITY_COLUMNS) | {"identity_row_sha256"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise LeakageAuditError(f"{name} identity rows missing columns: {missing}")
    result = frame.loc[:, [*IDENTITY_COLUMNS, "identity_row_sha256"]].copy()
    if result.empty:
        raise LeakageAuditError(f"{name} identity rows are empty")
    for column in ("dataset", "split", "query_id", "scene_id", "frame_id"):
        if result[column].isna().any() or result[column].astype(str).str.strip().eq("").any():
            raise LeakageAuditError(f"{name} identity column {column} is null or empty")
        result[column] = result[column].astype(str)
    for column in (
        "group_identity_sha256",
        "rgb_sha256",
        "depth_sha256",
        "identity_row_sha256",
    ):
        if not result[column].astype(str).str.fullmatch(SHA256_PATTERN).all():
            raise LeakageAuditError(f"{name} identity column {column} is not SHA-256")
    if result.duplicated(["dataset", "query_id"]).any():
        raise LeakageAuditError(f"{name} contains duplicate dataset/query identities")
    expected_group = [
        _json_sha256([scene, frame_id])
        for scene, frame_id in zip(result["scene_id"], result["frame_id"], strict=True)
    ]
    if expected_group != result["group_identity_sha256"].tolist():
        raise LeakageAuditError(f"{name} group identities are not derived from scene/frame")
    for row in result.to_dict(orient="records"):
        recorded = row.pop("identity_row_sha256")
        if _json_sha256(row) != recorded:
            raise LeakageAuditError(f"{name} contains a stale identity_row_sha256")
    return result.sort_values(["dataset", "query_id"], kind="mergesort").reset_index(
        drop=True
    )


def _identity_overlaps(development: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    datasets = sorted(set(development["dataset"]) | set(test["dataset"]))
    for dataset in datasets:
        dev = development.loc[development["dataset"].eq(dataset)]
        held_out = test.loc[test["dataset"].eq(dataset)]
        for identity_type, column in OVERLAP_FIELDS:
            shared = sorted(set(dev[column]) & set(held_out[column]))
            for value in shared:
                dev_queries = sorted(dev.loc[dev[column].eq(value), "query_id"].tolist())
                test_queries = sorted(
                    held_out.loc[held_out[column].eq(value), "query_id"].tolist()
                )
                rows.append(
                    {
                        "dataset": dataset,
                        "identity_type": identity_type,
                        "identity_value": value,
                        "development_query_ids": dev_queries,
                        "test_query_ids": test_queries,
                        "development_query_count": len(dev_queries),
                        "test_query_count": len(test_queries),
                    }
                )
    columns = (
        "dataset",
        "identity_type",
        "identity_value",
        "development_query_ids",
        "test_query_ids",
        "development_query_count",
        "test_query_count",
    )
    return pd.DataFrame(rows, columns=columns)


def _fit_query_frame(value: pd.DataFrame | Iterable[str]) -> pd.DataFrame:
    if isinstance(value, pd.DataFrame):
        if value.empty:
            return pd.DataFrame(columns=["query_id"])
        query_col = next(
            (name for name in _ALIASES["query_id"] if name in value.columns), None
        )
        if query_col is None:
            raise LeakageAuditError(
                "concrete fold fit rows must contain query_id/sample_id, not only fit_scope"
            )
        columns = [query_col]
        for name in (
            "dataset",
            "scene_id",
            "frame_id",
            "group_identity_sha256",
            "group_hash",
        ):
            if name in value.columns:
                columns.append(name)
        result = value.loc[:, columns].copy().rename(columns={query_col: "query_id"})
        result["actual_fit_row_count"] = result.groupby("query_id")["query_id"].transform(
            "size"
        )
        return result.drop_duplicates().reset_index(drop=True)
    if isinstance(value, (str, bytes)):
        raise LeakageAuditError("fold fit evidence must contain query IDs, not a scope string")
    return pd.DataFrame({"query_id": [str(item) for item in value]}).assign(
        actual_fit_row_count=1
    )


def _materialize_fit_rows(
    development: pd.DataFrame,
    test: pd.DataFrame,
    fit_partitions: Mapping[Any, pd.DataFrame | Iterable[str]],
    expected_partitions: Iterable[Any] | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    fit_rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    partition_names = {_normalise_partition(value) for value in fit_partitions}
    if expected_partitions is not None:
        expected = {_normalise_partition(value) for value in expected_partitions}
        for value in sorted(expected - partition_names):
            errors.append({"fit_partition": value, "error": "missing_fit_partition"})
        for value in sorted(partition_names - expected):
            errors.append({"fit_partition": value, "error": "unexpected_fit_partition"})
    combined = pd.concat(
        [development.assign(identity_source="development"), test.assign(identity_source="test")],
        ignore_index=True,
    )
    by_key = {
        (str(row.dataset), str(row.query_id)): row
        for row in test.assign(identity_source="test").itertuples()
    }
    by_key.update(
        {
            (str(row.dataset), str(row.query_id)): row
            for row in development.assign(identity_source="development").itertuples()
        }
    )
    datasets_by_query = (
        combined.groupby("query_id", sort=True)["dataset"].agg(lambda values: sorted(set(values)))
    ).to_dict()
    dev_keys = set(zip(development["dataset"], development["query_id"], strict=True))

    for partition, value in sorted(
        fit_partitions.items(), key=lambda item: _normalise_partition(item[0])
    ):
        partition_name = _normalise_partition(partition)
        actual = _fit_query_frame(value)
        if actual.empty:
            errors.append({"fit_partition": partition_name, "error": "empty_fit_partition"})
            continue
        actual["query_id"] = actual["query_id"].astype(str)
        for query_id, query_rows in actual.groupby("query_id", sort=True):
            if "dataset" in query_rows:
                dataset_values = sorted(set(query_rows["dataset"].astype(str)))
            else:
                dataset_values = datasets_by_query.get(str(query_id), [])
            if len(dataset_values) != 1:
                errors.append(
                    {
                        "fit_partition": partition_name,
                        "query_id": str(query_id),
                        "error": "unknown_or_ambiguous_query_dataset",
                        "candidate_datasets": dataset_values,
                    }
                )
                continue
            dataset = dataset_values[0]
            identity = by_key.get((dataset, str(query_id)))
            if identity is None:
                errors.append(
                    {
                        "fit_partition": partition_name,
                        "dataset": dataset,
                        "query_id": str(query_id),
                        "error": "fit_query_absent_from_development_and_test_identities",
                    }
                )
                continue
            expected_group = str(identity.group_identity_sha256)
            evidence_source = "development_identity_join"
            observed_groups: set[str] = set()
            if {"scene_id", "frame_id"} <= set(query_rows.columns):
                observed_groups = {
                    _json_sha256([str(scene), str(frame_id)])
                    for scene, frame_id in zip(
                        query_rows["scene_id"], query_rows["frame_id"], strict=True
                    )
                }
                evidence_source = "actual_fit_scene_frame"
            elif "group_identity_sha256" in query_rows:
                observed_groups = set(query_rows["group_identity_sha256"].astype(str))
                evidence_source = "actual_fit_group_identity_sha256"
            elif "group_hash" in query_rows:
                observed_groups = set(query_rows["group_hash"].astype(str))
                evidence_source = "actual_fit_group_hash"
            observed_groups.discard("")
            if len(observed_groups) > 1:
                errors.append(
                    {
                        "fit_partition": partition_name,
                        "dataset": dataset,
                        "query_id": str(query_id),
                        "error": "fit_query_maps_to_multiple_groups",
                        "observed_group_identities": sorted(observed_groups),
                    }
                )
            observed_group = min(observed_groups) if observed_groups else expected_group
            if observed_group != expected_group:
                errors.append(
                    {
                        "fit_partition": partition_name,
                        "dataset": dataset,
                        "query_id": str(query_id),
                        "error": "fit_group_disagrees_with_source_identity",
                        "expected_group_identity_sha256": expected_group,
                        "observed_group_identity_sha256": observed_group,
                    }
                )
            fit_rows.append(
                {
                    "fit_partition": partition_name,
                    "dataset": dataset,
                    "query_id": str(query_id),
                    "group_identity_sha256": observed_group,
                    "identity_source": str(identity.identity_source),
                    "evidence_source": evidence_source,
                    "actual_fit_row_count": int(query_rows["actual_fit_row_count"].max()),
                    "is_development_identity": (dataset, str(query_id)) in dev_keys,
                }
            )
    fit_columns = (
        "fit_partition",
        "dataset",
        "query_id",
        "group_identity_sha256",
        "identity_source",
        "evidence_source",
        "actual_fit_row_count",
        "is_development_identity",
    )
    return (
        pd.DataFrame(fit_rows, columns=fit_columns),
        pd.DataFrame(errors),
    )


def _normalise_partition(value: Any) -> str:
    if isinstance(value, tuple):
        return json.dumps([_normalise_json(item) for item in value], separators=(",", ":"))
    return str(value)


def _fit_overlaps(fit_rows: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    columns = (
        "fit_partition",
        "dataset",
        "identity_type",
        "identity_value",
        "fit_query_ids",
        "test_query_ids",
    )
    rows: list[dict[str, Any]] = []
    if fit_rows.empty:
        return pd.DataFrame(columns=columns)
    for partition, partition_rows in fit_rows.groupby("fit_partition", sort=True):
        for dataset, actual in partition_rows.groupby("dataset", sort=True):
            held_out = test.loc[test["dataset"].eq(dataset)]
            for identity_type, fit_column, test_column in (
                ("query_id", "query_id", "query_id"),
                (
                    "group_identity",
                    "group_identity_sha256",
                    "group_identity_sha256",
                ),
            ):
                for value in sorted(set(actual[fit_column]) & set(held_out[test_column])):
                    rows.append(
                        {
                            "fit_partition": partition,
                            "dataset": dataset,
                            "identity_type": identity_type,
                            "identity_value": value,
                            "fit_query_ids": sorted(
                                actual.loc[actual[fit_column].eq(value), "query_id"].tolist()
                            ),
                            "test_query_ids": sorted(
                                held_out.loc[
                                    held_out[test_column].eq(value), "query_id"
                                ].tolist()
                            ),
                        }
                    )
    return pd.DataFrame(rows, columns=columns)


def audit_development_test_identities(
    development_identities: pd.DataFrame,
    test_identities: pd.DataFrame,
    *,
    fit_partitions: Mapping[Any, pd.DataFrame | Iterable[str]] | None = None,
    expected_fit_partitions: Iterable[Any] | None = None,
    require_fit_evidence: bool = False,
) -> LeakageAuditResult:
    """Audit source entities and concrete per-fold fit membership.

    Overlap is scoped by dataset so separately trained routes do not contaminate
    each other's namespace.  A fit partition may be an actual candidate/query
    frame or an explicit iterable of the query IDs used by the fitting call.
    Scope labels and precomputed membership digests are intentionally ignored.
    """

    development = _validate_identity_rows(development_identities, name="development")
    test = _validate_identity_rows(test_identities, name="test")
    overlaps = _identity_overlaps(development, test)
    fit_evidence_requested = bool(
        require_fit_evidence
        or fit_partitions is not None
        or expected_fit_partitions is not None
    )
    if fit_partitions is None and expected_fit_partitions is None:
        fit_rows = pd.DataFrame(
            columns=(
                "fit_partition",
                "dataset",
                "query_id",
                "group_identity_sha256",
                "identity_source",
                "evidence_source",
                "actual_fit_row_count",
                "is_development_identity",
            )
        )
        fit_errors = (
            pd.DataFrame(
                [{"fit_partition": None, "error": "fit_evidence_not_provided"}]
            )
            if require_fit_evidence
            else pd.DataFrame()
        )
    else:
        fit_rows, fit_errors = _materialize_fit_rows(
            development, test, fit_partitions or {}, expected_fit_partitions
        )
        if fit_partitions == {} and fit_errors.empty:
            fit_errors = pd.DataFrame(
                [{"fit_partition": None, "error": "no_fit_partitions"}]
            )
    fit_overlaps = _fit_overlaps(fit_rows, test)
    combined = pd.concat(
        [
            development.assign(audit_role="development"),
            test.assign(audit_role="test"),
        ],
        ignore_index=True,
    )
    identity_overlap_counts = {
        name: int(overlaps["identity_type"].eq(name).sum())
        for name, _ in OVERLAP_FIELDS
    }
    fit_overlap_counts = {
        name: int(fit_overlaps["identity_type"].eq(name).sum())
        for name in ("query_id", "group_identity")
    }
    passed = bool(overlaps.empty and fit_overlaps.empty and fit_errors.empty)
    evidence_digests = {
        "identity_rows_sha256": frame_sha256(combined),
        "identity_overlaps_sha256": frame_sha256(overlaps),
        "fold_fit_identities_sha256": frame_sha256(fit_rows),
        "fold_fit_overlaps_sha256": frame_sha256(fit_overlaps),
        "fit_resolution_errors_sha256": frame_sha256(fit_errors),
    }
    summary_without_digest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "passed": passed,
        "overlap_scope": ["dataset"],
        "development_query_count": int(len(development)),
        "test_query_count": int(len(test)),
        "development_dataset_counts": {
            str(key): int(value)
            for key, value in development["dataset"].value_counts().sort_index().items()
        },
        "test_dataset_counts": {
            str(key): int(value)
            for key, value in test["dataset"].value_counts().sort_index().items()
        },
        "identity_overlap_counts": identity_overlap_counts,
        "fit_partition_count": int(fit_rows["fit_partition"].nunique())
        if not fit_rows.empty
        else 0,
        "fit_query_identity_count": int(len(fit_rows)),
        "fit_overlap_counts": fit_overlap_counts,
        "fit_resolution_error_count": int(len(fit_errors)),
        "fit_membership_audited": bool(
            fit_evidence_requested and not fit_rows.empty and fit_errors.empty
        ),
        "fit_evidence_required": bool(require_fit_evidence),
        "fit_evidence_requirement": "concrete_query_rows_or_query_id_membership",
        "fit_scope_strings_trusted": False,
        "evidence_digests": evidence_digests,
    }
    summary = {
        **summary_without_digest,
        "audit_digest_sha256": _json_sha256(summary_without_digest),
    }
    return LeakageAuditResult(
        summary=summary,
        identity_rows=combined,
        identity_overlaps=overlaps,
        fold_fit_identities=fit_rows,
        fold_fit_overlaps=fit_overlaps,
        fit_resolution_errors=fit_errors,
    )


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def _sorted_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    records = [_normalise_json(row) for row in frame.to_dict(orient="records")]
    records.sort(key=lambda row: json.dumps(row, sort_keys=True, ensure_ascii=False))
    return records


def _markdown_cell(value: Any) -> str:
    normalised = _normalise_json(value)
    if normalised is None:
        return "—"
    if isinstance(normalised, (dict, list)):
        text = json.dumps(
            normalised,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    else:
        text = str(normalised)
    return text.replace("\\", "\\\\").replace("|", "\\|").replace("\n", " ")


def _render_markdown(result: LeakageAuditResult) -> str:
    """Render the human audit solely from checksum-committed evidence."""

    summary = result.summary
    passed = bool(summary["passed"])
    lines = [
        "# Leakage Audit",
        "",
        f"**Overall result: {'PASS' if passed else 'FAIL'}**",
        "",
        "This report checks concrete source and fit identities. `fit_scope` strings are not trusted as evidence.",
        "",
        "## Development and retrospective test coverage",
        "",
        "| Audit role | Dataset | Queries |",
        "|---|---|---:|",
    ]
    development_counts = dict(summary.get("development_dataset_counts", {}))
    test_counts = dict(summary.get("test_dataset_counts", {}))
    for role, counts in (
        ("development", development_counts),
        ("test", test_counts),
    ):
        for dataset in sorted(counts):
            lines.append(
                f"| {role} | {_markdown_cell(dataset)} | {int(counts[dataset])} |"
            )
    lines.extend(
        [
            f"| **development total** | **all** | **{int(summary['development_query_count'])}** |",
            f"| **test total** | **all** | **{int(summary['test_query_count'])}** |",
            "",
            "## Development/test source-entity overlap",
            "",
            "| Identity type | Overlap count | Status | Full evidence |",
            "|---|---:|---|---|",
        ]
    )
    identity_counts = dict(summary.get("identity_overlap_counts", {}))
    for identity_type, _ in OVERLAP_FIELDS:
        count = int(identity_counts.get(identity_type, -1))
        lines.append(
            f"| `{identity_type}` | {count} | {'PASS' if count == 0 else 'FAIL'} | `identity_overlaps.jsonl` |"
        )

    fit_error_count = int(summary.get("fit_resolution_error_count", 0))
    fit_membership_audited = bool(summary.get("fit_membership_audited"))
    if fit_membership_audited:
        fit_completeness = "PASS"
    elif fit_error_count:
        fit_completeness = "FAIL"
    else:
        fit_completeness = "NOT_AUDITED"
    lines.extend(
        [
            "",
            "## Concrete fold-fit membership",
            "",
            f"- Fit partition completeness: **{fit_completeness}**",
            f"- Observed fit partitions: **{int(summary.get('fit_partition_count', 0))}**",
            f"- Concrete fit query identities: **{int(summary.get('fit_query_identity_count', 0))}**",
            f"- Fit resolution/coverage errors: **{fit_error_count}**",
            f"- Fit evidence required by caller: **{str(bool(summary.get('fit_evidence_required'))).lower()}**",
            "",
            "| Fit partition | Fit queries | Fit groups | Fit/test overlaps | Resolution errors | Status |",
            "|---|---:|---:|---:|---:|---|",
        ]
    )
    fit_rows = result.fold_fit_identities
    fit_overlaps = result.fold_fit_overlaps
    fit_errors = result.fit_resolution_errors
    partitions: set[str] = set()
    for frame in (fit_rows, fit_overlaps, fit_errors):
        if "fit_partition" in frame:
            partitions.update(
                str(value)
                for value in frame["fit_partition"].dropna().astype(str).tolist()
            )
    if not partitions:
        lines.append("| — | 0 | 0 | 0 | 0 | NOT_AUDITED |")
    else:
        for partition in sorted(partitions):
            selected_fit = (
                fit_rows.loc[fit_rows["fit_partition"].astype(str).eq(partition)]
                if "fit_partition" in fit_rows
                else fit_rows.iloc[0:0]
            )
            selected_overlaps = (
                fit_overlaps.loc[
                    fit_overlaps["fit_partition"].astype(str).eq(partition)
                ]
                if "fit_partition" in fit_overlaps
                else fit_overlaps.iloc[0:0]
            )
            selected_errors = (
                fit_errors.loc[fit_errors["fit_partition"].astype(str).eq(partition)]
                if "fit_partition" in fit_errors
                else fit_errors.iloc[0:0]
            )
            group_count = (
                int(selected_fit["group_identity_sha256"].nunique())
                if "group_identity_sha256" in selected_fit
                else 0
            )
            partition_passed = bool(
                not selected_fit.empty
                and selected_overlaps.empty
                and selected_errors.empty
            )
            lines.append(
                "| "
                + " | ".join(
                    (
                        _markdown_cell(partition),
                        str(len(selected_fit)),
                        str(group_count),
                        str(len(selected_overlaps)),
                        str(len(selected_errors)),
                        "PASS" if partition_passed else "FAIL",
                    )
                )
                + " |"
            )

    fit_counts = dict(summary.get("fit_overlap_counts", {}))
    lines.extend(
        [
            "",
            "## Fit/test entity overlap",
            "",
            "| Identity type | Overlap count | Status | Full evidence |",
            "|---|---:|---|---|",
        ]
    )
    for identity_type in ("query_id", "group_identity"):
        count = int(fit_counts.get(identity_type, -1))
        lines.append(
            f"| `{identity_type}` | {count} | {'PASS' if count == 0 else 'FAIL'} | `fold_fit_overlaps.jsonl` |"
        )

    lines.extend(
        [
            "",
            "## Failure details",
            "",
            "Full canonical records are retained in the referenced JSONL files. The line numbers below refer to their deterministic sorted order.",
            "",
            "### Development/test source overlaps",
            "",
        ]
    )
    identity_overlap_records = _sorted_records(result.identity_overlaps)
    if not identity_overlap_records:
        lines.append("None.")
    else:
        lines.extend(
            [
                "| JSONL line | Dataset | Type | Identity | Development queries | Test queries |",
                "|---:|---|---|---|---|---|",
            ]
        )
        for line_number, row in enumerate(identity_overlap_records, start=1):
            lines.append(
                "| "
                + " | ".join(
                    (
                        str(line_number),
                        _markdown_cell(row.get("dataset")),
                        _markdown_cell(row.get("identity_type")),
                        _markdown_cell(row.get("identity_value")),
                        _markdown_cell(row.get("development_query_ids")),
                        _markdown_cell(row.get("test_query_ids")),
                    )
                )
                + " |"
            )

    lines.extend(["", "### Fold-fit/test overlaps", ""])
    fit_overlap_records = _sorted_records(result.fold_fit_overlaps)
    if not fit_overlap_records:
        lines.append("None.")
    else:
        lines.extend(
            [
                "| JSONL line | Partition | Dataset | Type | Identity | Fit queries | Test queries |",
                "|---:|---|---|---|---|---|---|",
            ]
        )
        for line_number, row in enumerate(fit_overlap_records, start=1):
            lines.append(
                "| "
                + " | ".join(
                    (
                        str(line_number),
                        _markdown_cell(row.get("fit_partition")),
                        _markdown_cell(row.get("dataset")),
                        _markdown_cell(row.get("identity_type")),
                        _markdown_cell(row.get("identity_value")),
                        _markdown_cell(row.get("fit_query_ids")),
                        _markdown_cell(row.get("test_query_ids")),
                    )
                )
                + " |"
            )

    lines.extend(["", "### Fit resolution and partition coverage errors", ""])
    fit_error_records = _sorted_records(result.fit_resolution_errors)
    if not fit_error_records:
        lines.append("None.")
    else:
        lines.extend(
            [
                "| JSONL line | Partition | Query | Error | Record |",
                "|---:|---|---|---|---|",
            ]
        )
        for line_number, row in enumerate(fit_error_records, start=1):
            lines.append(
                "| "
                + " | ".join(
                    (
                        str(line_number),
                        _markdown_cell(row.get("fit_partition")),
                        _markdown_cell(row.get("query_id")),
                        _markdown_cell(row.get("error")),
                        _markdown_cell(row),
                    )
                )
                + " |"
            )

    lines.extend(
        [
            "",
            "## Evidence digests",
            "",
            "| Evidence | SHA-256 |",
            "|---|---|",
        ]
    )
    for name, digest in sorted(dict(summary.get("evidence_digests", {})).items()):
        lines.append(f"| `{name}` | `{_markdown_cell(digest)}` |")
    lines.extend(
        [
            f"| `audit_digest_sha256` | `{_markdown_cell(summary.get('audit_digest_sha256'))}` |",
            "",
        ]
    )
    return "\n".join(lines)


def _write_jsonl(path: Path, frame: pd.DataFrame) -> None:
    records = _sorted_records(frame)
    with path.open("w", encoding="utf-8") as handle:
        for row in records:
            handle.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                )
                + "\n"
            )


def _read_jsonl(path: Path) -> pd.DataFrame:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return pd.DataFrame(rows)
