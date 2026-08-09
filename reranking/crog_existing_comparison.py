"""Fail-closed R10 comparison of already-frozen CROG re-rankers.

This module does not train or invoke legacy inference commands.  It discovers
completed existing runs, admits only methods that pass the repository's R10
eligibility audit, loads a hash-bound frozen ranking artifact, requires an
exact join to the caller-supplied CROG Top-5 pool, and independently evaluates
the ranking with :mod:`reranking.evaluate`.

The output contract deliberately distinguishes discovery/provenance from a
comparison.  An eligible method is counted as compared only after normalized
candidate-level predictions, per-query outcomes, an evaluator result, and
their hashed provenance have all been materialized.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from reranking import evaluate as evaluator_module
from reranking.data_contracts import streaming_sha256
from reranking.evaluate import (
    EvaluationError,
    assert_same_candidate_pool,
    compare_rankings,
    evaluate_rankings,
    validate_rank_permutations,
)
from reranking.existing_run_discovery import discover_existing_runs


_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_COMPLETE_STATUSES = frozenset(
    {"complete", "completed", "locked", "pass", "passed", "success", "verified"}
)
_PREDICTION_SUFFIXES = frozenset({".json", ".jsonl", ".csv", ".parquet"})
_PREDICTION_WORDS = ("prediction", "ranking", "method_rankings")


class CrogExistingComparisonError(ValueError):
    """Raised when the frozen CROG comparison contract is violated."""


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(child) for child in value]
    if isinstance(value, np.ndarray):
        return [_jsonable(child) for child in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if pd.isna(value) if not isinstance(value, (dict, list, tuple)) else False:
        return None
    return value


def _write_json(path: Path, value: Any) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_jsonable(value), indent=2, sort_keys=True, ensure_ascii=False)
        + "\n",
        encoding="utf-8",
    )
    return _descriptor(path)


def _write_jsonl(path: Path, frame: pd.DataFrame) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for row in frame.to_dict(orient="records"):
            stream.write(
                json.dumps(_jsonable(row), sort_keys=True, ensure_ascii=False) + "\n"
            )
    return _descriptor(path)


def _descriptor(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "sha256": streaming_sha256(path),
        "size_bytes": int(path.stat().st_size),
    }


def _canonical_frame_sha256(frame: pd.DataFrame, columns: Sequence[str]) -> str:
    selected = frame.loc[:, list(columns)].copy()
    selected = selected.sort_values(list(columns), kind="mergesort").reset_index(
        drop=True
    )
    payload = selected.to_json(
        orient="records", lines=True, force_ascii=False, double_precision=15
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _safe_method_directory(method_id: str) -> str:
    prefix = re.sub(r"[^a-zA-Z0-9_.-]+", "_", method_id).strip("._")
    digest = hashlib.sha256(method_id.encode("utf-8")).hexdigest()[:12]
    return f"{prefix[:80] or 'method'}-{digest}"


def _normalise_pool(candidate_pool: pd.DataFrame, *, top_k: int) -> pd.DataFrame:
    required = {"query_id", "candidate_id", "label", "scene_id", "frame_id"}
    missing = sorted(required - set(candidate_pool.columns))
    if missing:
        raise CrogExistingComparisonError(
            f"CROG candidate pool missing columns: {missing}"
        )
    pool = candidate_pool.copy()
    if "candidate_identity_sha256" not in pool.columns:
        if "candidate_checksum" not in pool.columns:
            raise CrogExistingComparisonError(
                "CROG candidate pool lacks candidate checksum identity"
            )
        pool["candidate_identity_sha256"] = pool["candidate_checksum"]
    identities = pool["candidate_identity_sha256"].astype(str).str.lower()
    if not identities.map(lambda value: bool(_SHA256.fullmatch(value))).all():
        raise CrogExistingComparisonError(
            "CROG candidate identity values must be SHA-256 digests"
        )
    pool["candidate_identity_sha256"] = identities
    pool["query_id"] = pool["query_id"].astype(str)
    pool["candidate_id"] = pool["candidate_id"].astype(str)
    if pool.duplicated(["query_id", "candidate_id"]).any():
        raise CrogExistingComparisonError("CROG candidate pool has duplicate keys")
    counts = pool.groupby("query_id", sort=False).size()
    if counts.empty or not bool(counts.eq(int(top_k)).all()):
        examples = counts[counts.ne(int(top_k))].head(5).to_dict()
        raise CrogExistingComparisonError(
            f"CROG frozen pool is not exactly Top-{top_k} per query: {examples}"
        )
    candidate_numbers = pool["candidate_id"].str.extract(r"^candidate_(\d+)$")[0]
    if candidate_numbers.isna().any():
        raise CrogExistingComparisonError(
            "CROG candidate IDs must use the frozen candidate_<index> identity"
        )
    expected = set(range(int(top_k)))
    for query_id, values in candidate_numbers.astype(int).groupby(pool["query_id"]):
        if set(values.tolist()) != expected:
            raise CrogExistingComparisonError(
                f"{query_id} candidate IDs are not candidate_0..candidate_{top_k - 1}"
            )
    return pool


def _load_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    if path.suffix.lower() == ".jsonl":
        return pd.read_json(path, lines=True)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return pd.DataFrame(payload)
    if isinstance(payload, Mapping) and isinstance(payload.get("predictions"), list):
        return pd.DataFrame(payload["predictions"])
    raise CrogExistingComparisonError(f"unsupported prediction JSON: {path}")


def _require_prediction_identities(frame: pd.DataFrame) -> pd.DataFrame:
    """Require an unambiguous SHA-256 identity for every frozen candidate."""

    result = frame.copy()
    if "candidate_identity_sha256" not in result.columns:
        if "candidate_checksum" not in result.columns:
            raise CrogExistingComparisonError(
                "frozen predictions lack per-candidate checksum identity"
            )
        result["candidate_identity_sha256"] = result["candidate_checksum"]
    identities = result["candidate_identity_sha256"].astype(str).str.lower()
    if not identities.map(lambda value: bool(_SHA256.fullmatch(value))).all():
        raise CrogExistingComparisonError(
            "every frozen prediction requires a SHA-256 candidate identity"
        )
    if "candidate_checksum" in result.columns:
        legacy_identities = result["candidate_checksum"].astype(str).str.lower()
        if not legacy_identities.map(
            lambda value: bool(_SHA256.fullmatch(value))
        ).all() or not legacy_identities.equals(identities):
            raise CrogExistingComparisonError(
                "frozen candidate checksum identity columns disagree"
            )
    result["candidate_identity_sha256"] = identities
    return result


def _normalise_frozen_predictions(
    path: Path,
    *,
    method_id: str,
) -> pd.DataFrame:
    source = _load_table(path)
    if source.empty:
        raise CrogExistingComparisonError("frozen prediction artifact is empty")
    query_column = "query_id" if "query_id" in source.columns else "sample_id"
    if query_column not in source.columns:
        raise CrogExistingComparisonError("frozen predictions lack query/sample IDs")
    if "method" in source.columns:
        declared = set(source["method"].dropna().astype(str))
        if declared and declared != {method_id}:
            raise CrogExistingComparisonError(
                f"prediction method identity mismatch: {sorted(declared)} != {method_id}"
            )
    if "candidate_id" in source.columns and (
        "score" in source.columns or "rank" in source.columns
    ):
        result = source.rename(columns={query_column: "query_id"}).copy()
        if "rank" not in result.columns:
            result["score"] = pd.to_numeric(result["score"], errors="coerce")
            result = result.sort_values(
                ["query_id", "score", "candidate_id"],
                ascending=[True, False, True],
                kind="mergesort",
            )
            result["rank"] = result.groupby("query_id", sort=False).cumcount() + 1
        if "score" not in result.columns:
            result["rank"] = pd.to_numeric(result["rank"], errors="coerce")
            result["score"] = -result["rank"].astype(float)
        columns = ["query_id", "candidate_id", "score", "rank"]
        for identity in ("candidate_identity_sha256", "candidate_checksum"):
            if identity in result.columns:
                columns.append(identity)
        result = result[columns].copy()
        if (
            "candidate_identity_sha256" not in result.columns
            and "candidate_checksum" in result.columns
        ):
            result["candidate_identity_sha256"] = result["candidate_checksum"]
        return _require_prediction_identities(result)

    if "candidate_order" not in source.columns:
        raise CrogExistingComparisonError(
            "frozen query-level predictions lack candidate_order"
        )
    rows: list[dict[str, Any]] = []
    for record in source.to_dict(orient="records"):
        query_id = str(record[query_column])
        order = record.get("candidate_order")
        if not isinstance(order, (list, tuple)) or not order:
            raise CrogExistingComparisonError(
                f"{query_id} has no complete frozen candidate_order"
            )
        candidate_ids = [str(value) for value in order]
        if len(set(candidate_ids)) != len(candidate_ids):
            raise CrogExistingComparisonError(
                f"{query_id} candidate_order contains duplicates"
            )
        checksum_by_id: dict[str, str] = {}
        checksum_ids = record.get("candidate_checksum_ids")
        checksums = record.get("candidate_checksums")
        if not isinstance(checksum_ids, (list, tuple)) or not isinstance(
            checksums, (list, tuple)
        ):
            raise CrogExistingComparisonError(
                f"{query_id} lacks per-candidate checksum identities"
            )
        if len(checksum_ids) != len(checksums):
            raise CrogExistingComparisonError(
                f"{query_id} candidate checksum arrays differ in length"
            )
        normalised_checksum_ids = [str(value) for value in checksum_ids]
        if len(set(normalised_checksum_ids)) != len(normalised_checksum_ids):
            raise CrogExistingComparisonError(
                f"{query_id} candidate checksum IDs contain duplicates"
            )
        if set(normalised_checksum_ids) != set(candidate_ids):
            raise CrogExistingComparisonError(
                f"{query_id} candidate checksums do not cover its frozen order"
            )
        checksum_by_id = {
            candidate_id: str(checksum)
            for candidate_id, checksum in zip(
                normalised_checksum_ids, checksums, strict=True
            )
        }
        for rank, candidate_id in enumerate(candidate_ids, start=1):
            row: dict[str, Any] = {
                "query_id": query_id,
                "candidate_id": candidate_id,
                # The frozen order is authoritative after any legacy gate or
                # stability policy.  A monotone synthetic score preserves it.
                "score": float(len(candidate_ids) - rank + 1),
                "rank": rank,
            }
            if candidate_id in checksum_by_id:
                row["candidate_identity_sha256"] = checksum_by_id[candidate_id]
            rows.append(row)
    return _require_prediction_identities(pd.DataFrame(rows))


def _contains_symlink(path: Path) -> bool:
    absolute = Path(os.path.abspath(path))
    return any(candidate.is_symlink() for candidate in (absolute, *absolute.parents))


def _resolve_descriptor(source: Path, descriptor: Mapping[str, Any]) -> dict[str, Any]:
    declared_path = descriptor.get("path")
    expected = str(descriptor.get("sha256", "")).lower()
    path = Path(str(declared_path)).expanduser() if declared_path else None
    if path is not None and not path.is_absolute():
        path = source.parent / path
    path = Path(os.path.abspath(path)) if path is not None else None
    symlink = bool(path is not None and _contains_symlink(path))
    regular = bool(path is not None and path.is_file() and not symlink)
    actual = streaming_sha256(path) if regular and path is not None else None
    passed = bool(_SHA256.fullmatch(expected) and regular and actual == expected)
    return {
        "source_document": str(source),
        "path": str(path) if path is not None else None,
        "expected_sha256": expected or None,
        "actual_sha256": actual,
        "size_bytes": int(path.stat().st_size) if regular and path is not None else None,
        "regular_file": regular,
        "symlink": symlink,
        "passed": passed,
        "issue": (
            None
            if passed
            else "prediction_descriptor_path_or_sha256_invalid"
        ),
    }


def _walk(value: Any, trail: str = "$") -> Sequence[tuple[str, Any]]:
    output: list[tuple[str, Any]] = [(trail, value)]
    if isinstance(value, Mapping):
        for key, child in value.items():
            output.extend(_walk(child, f"{trail}.{key}"))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            output.extend(_walk(child, f"{trail}[{index}]"))
    return output


def _prediction_descriptors(
    method: Mapping[str, Any], *, scope: str
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return passing and rejected hash-bound ranking descriptors for a method."""

    run_root = Path(str(method["run_root"]))
    method_id = str(method["method_id"])
    passing: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for source in sorted(run_root.rglob("*.json"), key=lambda value: value.as_posix()):
        if _contains_symlink(source) or not source.is_file():
            continue
        try:
            payload = json.loads(source.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            continue
        if not isinstance(payload, Mapping):
            continue
        status = str(payload.get("status", "")).strip().lower()
        document_is_bound = status in _COMPLETE_STATUSES or any(
            token in source.name.lower() for token in ("manifest", "lock")
        )
        if not document_is_bound:
            continue
        for trail, child in _walk(payload):
            if not (
                isinstance(child, Mapping)
                and isinstance(child.get("path"), str)
                and "sha256" in child
            ):
                continue
            declared = str(child.get("path", ""))
            declared_path = Path(declared).expanduser()
            if not declared_path.is_absolute():
                declared_path = source.parent / declared_path
            declared_path = Path(os.path.abspath(declared_path))
            try:
                local_declared = declared_path.relative_to(run_root).as_posix()
            except ValueError:
                # Keep external artifacts attributable by their filename, but
                # never let ancestor directory names (for example a pytest
                # temp directory containing "test") establish split scope.
                local_declared = declared_path.name
            semantic = f"{source.name} {trail} {local_declared}".lower()
            suffix = Path(declared).suffix.lower()
            if suffix not in _PREDICTION_SUFFIXES or not any(
                word in semantic for word in _PREDICTION_WORDS
            ):
                continue
            if "oof" in semantic:
                continue
            normal_method = re.sub(r"[^a-z0-9]+", "_", method_id.lower()).strip("_")
            normal_semantic = re.sub(r"[^a-z0-9]+", "_", semantic).strip("_")
            if normal_method not in normal_semantic:
                continue
            scope_exact = scope.lower() in semantic
            # Validation rankings are never silently substituted for test.
            if scope.lower() == "test" and "validation" in semantic and not scope_exact:
                continue
            audited = _resolve_descriptor(source, child)
            audited.update(
                {
                    "manifest_location": trail,
                    "scope_exact": scope_exact,
                    "method_id": method_id,
                    "priority": int(scope_exact) * 4
                    + int("method_rankings" in trail.lower()) * 2
                    + int("prediction" in declared.lower()),
                }
            )
            (passing if audited["passed"] else rejected).append(audited)
    unique: dict[tuple[str, str], dict[str, Any]] = {}
    for record in passing:
        unique[(str(record["source_document"]), str(record["path"]))] = record
    ordered = sorted(
        unique.values(),
        key=lambda item: (-int(item["priority"]), str(item["path"])),
    )
    return ordered, rejected


def _select_prediction_descriptor(
    method: Mapping[str, Any], *, scope: str
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    passing, rejected = _prediction_descriptors(method, scope=scope)
    if not passing:
        return None, {
            "code": "frozen_prediction_missing_or_hash_invalid",
            "message": "no completed, method-bound, hash-valid frozen ranking was found",
            "rejected_descriptors": rejected,
        }
    best_priority = int(passing[0]["priority"])
    best = [item for item in passing if int(item["priority"]) == best_priority]
    paths = {str(item["path"]) for item in best}
    if len(paths) != 1:
        return None, {
            "code": "frozen_prediction_descriptor_ambiguous",
            "message": "multiple equally strong frozen ranking descriptors disagree",
            "candidates": best,
        }
    return best[0], {"accepted_candidates": best, "rejected_descriptors": rejected}


def _identity_columns(pool: pd.DataFrame, predictions: pd.DataFrame) -> tuple[str, ...]:
    for column in ("candidate_identity_sha256", "candidate_checksum"):
        if column in pool.columns and column in predictions.columns:
            return (column,)
    return ()


def _comparison_payload(result: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "reference_metrics": result["reference"]["metrics"],
        "challenger_metrics": result["challenger"]["metrics"],
        "switch_metrics": result["switch_metrics"],
        "statistics": result["statistics"],
        "oracle_invariant": result["oracle_invariant"],
    }


def validate_crog_existing_comparison_report(report: Mapping[str, Any]) -> bool:
    """Reject reports that claim comparisons using provenance alone."""

    def require_artifact(value: Any, role: str) -> None:
        if not isinstance(value, Mapping):
            raise CrogExistingComparisonError(f"{role} descriptor is malformed")
        path_value = value.get("path")
        expected = str(value.get("sha256", "")).lower()
        size = value.get("size_bytes")
        path = Path(str(path_value)) if path_value else None
        if (
            path is None
            or _contains_symlink(path)
            or not path.is_file()
            or not _SHA256.fullmatch(expected)
            or not isinstance(size, int)
            or int(size) != int(path.stat().st_size)
            or streaming_sha256(path) != expected
        ):
            raise CrogExistingComparisonError(
                f"{role} artifact is missing or hash-invalid"
            )

    comparisons = report.get("comparisons")
    if not isinstance(comparisons, list):
        raise CrogExistingComparisonError("comparison report lacks comparisons")
    report_artifacts = report.get("artifacts")
    if not isinstance(report_artifacts, Mapping):
        raise CrogExistingComparisonError("comparison report lacks audit artifacts")
    for role in ("discovery", "exclusions", "reference_predictions"):
        require_artifact(report_artifacts.get(role), role)
    if int(report.get("comparison_count", -1)) != len(comparisons):
        raise CrogExistingComparisonError("comparison_count is inconsistent")
    exclusions = report.get("exclusions")
    if not isinstance(exclusions, list) or int(report.get("excluded_count", -1)) != len(
        exclusions
    ):
        raise CrogExistingComparisonError("exclusion evidence is inconsistent")
    compared_ids: set[str] = set()
    for comparison in comparisons:
        if not isinstance(comparison, Mapping):
            raise CrogExistingComparisonError("comparison row is malformed")
        method_id = str(comparison.get("method_id", ""))
        evidence = comparison.get("comparison_evidence")
        payload = comparison.get("comparison")
        if not method_id or not isinstance(evidence, Mapping) or not isinstance(
            payload, Mapping
        ):
            raise CrogExistingComparisonError(
                "method provenance without evaluator output is not a comparison"
            )
        required_artifacts = {"predictions", "per_query", "comparison", "provenance"}
        artifacts = evidence.get("artifacts")
        if not isinstance(artifacts, Mapping) or not required_artifacts.issubset(
            artifacts
        ):
            raise CrogExistingComparisonError(
                "comparison lacks materialized machine-readable evidence"
            )
        for role in sorted(required_artifacts):
            require_artifact(artifacts[role], role)
        require_artifact(evidence.get("evaluator"), "evaluator")
        if not evidence.get("exact_candidate_join") or not evidence.get(
            "independent_recomputed"
        ):
            raise CrogExistingComparisonError(
                "comparison lacks exact join or independent recomputation"
            )
        challenger = payload.get("challenger_metrics")
        if not isinstance(challenger, Mapping) or "j_at_1" not in challenger:
            raise CrogExistingComparisonError("comparison lacks challenger metrics")
        if method_id in compared_ids:
            raise CrogExistingComparisonError("method compared more than once")
        compared_ids.add(method_id)
        identity_sha256 = str(evidence.get("candidate_pool_identity_sha256", ""))
        if not _SHA256.fullmatch(identity_sha256):
            raise CrogExistingComparisonError(
                "comparison lacks candidate-pool identity hash"
            )
    eligible_count = int(report.get("eligible_count", -1))
    if eligible_count > len(comparisons) and report.get("status") == "complete":
        raise CrogExistingComparisonError(
            "eligible methods remain without comparisons"
        )
    if eligible_count == 0 and report.get("status") != "complete_no_eligible":
        raise CrogExistingComparisonError("zero-eligible report status is invalid")
    if eligible_count == 0 and not exclusions:
        raise CrogExistingComparisonError("zero-eligible report lacks exclusion evidence")
    return True


def run_crog_existing_comparisons(
    *,
    roots: Sequence[str | os.PathLike[str]],
    candidate_pool: pd.DataFrame,
    reference_predictions: pd.DataFrame,
    query_universe: pd.DataFrame,
    output_dir: str | os.PathLike[str],
    discovery_report: Mapping[str, Any] | None = None,
    scope: str = "test",
    top_k: int = 5,
    bootstrap_iterations: int = 10_000,
    bootstrap_seed: int = 20260801,
) -> dict[str, Any]:
    """Discover, independently recompute, and persist R10 CROG comparisons.

    ``candidate_pool`` and ``reference_predictions`` are supplied by the
    current formal run.  This keeps the module independent of Modular paths and
    guarantees every admitted legacy method is compared on exactly that same
    frozen CROG candidate identity.  Existing checkpoints are audited but are
    never loaded or trained here.  A training-stage ``discovery_report`` may be
    reused after lock so the expensive read-only scan is not repeated.
    """

    if scope not in {"test", "validation"}:
        raise CrogExistingComparisonError("scope must be test or validation")
    if int(top_k) != 5:
        raise CrogExistingComparisonError("R10 CROG comparison requires frozen Top-5")
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    pool = _normalise_pool(candidate_pool, top_k=top_k)
    assert_same_candidate_pool(pool, reference_predictions)
    reference_identity_columns = _identity_columns(pool, reference_predictions)
    reference_result = evaluate_rankings(
        pool,
        reference_predictions,
        query_universe=query_universe,
        identity_columns=reference_identity_columns,
    )
    evaluator_path = Path(evaluator_module.__file__).resolve()
    evaluator = _descriptor(evaluator_path)
    pool_identity_columns = ["query_id", "candidate_id"]
    for column in ("candidate_identity_sha256", "candidate_checksum"):
        if column in pool.columns:
            pool_identity_columns.append(column)
            break
    pool_identity_sha256 = _canonical_frame_sha256(pool, pool_identity_columns)
    reference_identity_sha256 = _canonical_frame_sha256(
        reference_result["ranked_candidates"],
        ["query_id", "candidate_id", "score", "rank"],
    )
    reference_artifact = _write_jsonl(
        output / "reference_predictions.jsonl",
        reference_result["ranked_candidates"][
            ["query_id", "candidate_id", "score", "rank"]
        ],
    )
    discovery = (
        discover_existing_runs(roots)
        if discovery_report is None
        else json.loads(json.dumps(_jsonable(discovery_report)))
    )
    if discovery.get("kind") != "existing_reranker_r10_discovery":
        raise CrogExistingComparisonError("discovery report kind is invalid")
    discovery_artifact = _write_json(output / "discovery.json", discovery)
    eligible = list(discovery["eligible_methods"])
    methods_by_key = {
        (item["run_root"], item["method_id"]): item for item in discovery["methods"]
    }
    comparisons: list[dict[str, Any]] = []
    exclusions: list[dict[str, Any]] = [
        {
            "run_id": item["run_id"],
            "run_root": item["run_root"],
            "method_id": item["method_id"],
            "eligibility_checks": item["checks"],
            "gaps": item["gaps"],
            "stage": "eligibility",
        }
        for item in discovery["methods"]
        if not item["eligible"]
    ]
    if not eligible and not exclusions:
        exclusions.append(
            {
                "run_id": None,
                "run_root": None,
                "method_id": None,
                "stage": "eligibility",
                "exclusion": {
                    "code": "no_existing_methods_discovered",
                    "message": "no content-declared existing CROG reranker was discovered",
                    "roots": discovery["roots"],
                    "scan": discovery["scan"],
                },
            }
        )
    for eligible_row in eligible:
        key = (eligible_row["run_root"], eligible_row["method_id"])
        method = methods_by_key[key]
        descriptor, descriptor_audit = _select_prediction_descriptor(
            method, scope=scope
        )
        if descriptor is None:
            exclusions.append(
                {
                    **eligible_row,
                    "stage": "prediction_materialization",
                    "eligibility_checks": method["checks"],
                    "exclusion": descriptor_audit,
                }
            )
            continue
        try:
            predictions = _normalise_frozen_predictions(
                Path(str(descriptor["path"])),
                method_id=str(method["method_id"]),
            )
            identity_columns = ("candidate_identity_sha256",)
            assert_same_candidate_pool(pool, predictions)
            identity = identity_columns[0]
            check = pool[["query_id", "candidate_id", identity]].merge(
                predictions[["query_id", "candidate_id", identity]],
                on=["query_id", "candidate_id"],
                how="inner",
                validate="one_to_one",
                suffixes=("_pool", "_prediction"),
            )
            if not check[f"{identity}_pool"].astype(str).equals(
                check[f"{identity}_prediction"].astype(str)
            ):
                raise CrogExistingComparisonError(
                    f"frozen prediction changed {identity}"
                )
            validate_rank_permutations(predictions)
            result = compare_rankings(
                pool,
                reference_predictions,
                predictions,
                query_universe=query_universe,
                identity_columns=identity_columns,
                bootstrap_iterations=bootstrap_iterations,
                bootstrap_seed=bootstrap_seed,
            )
        except (CrogExistingComparisonError, EvaluationError, OSError, ValueError) as error:
            exclusions.append(
                {
                    **eligible_row,
                    "stage": "independent_recomputation",
                    "eligibility_checks": method["checks"],
                    "prediction_descriptor": descriptor,
                    "exclusion": {
                        "code": "frozen_prediction_recomputation_failed",
                        "message": f"{type(error).__name__}: {error}",
                    },
                }
            )
            continue

        method_dir = output / "methods" / _safe_method_directory(str(method["method_id"]))
        prediction_artifact = _write_jsonl(
            method_dir / "predictions.jsonl",
            predictions.sort_values(["query_id", "rank"], kind="mergesort"),
        )
        per_query_artifact = _write_jsonl(
            method_dir / "per_query.jsonl",
            result["challenger"]["per_query"].sort_values(
                "query_id", kind="mergesort"
            ),
        )
        comparison_payload = _comparison_payload(result)
        comparison_artifact = _write_json(
            method_dir / "comparison.json", comparison_payload
        )
        provenance = {
            "schema_version": 1,
            "kind": "crog_existing_r10_comparison_provenance",
            "method_id": method["method_id"],
            "run_id": method["run_id"],
            "run_root": method["run_root"],
            "scope": scope,
            "trained_by_this_run": False,
            "checkpoint_loaded_by_this_module": False,
            "source_prediction": descriptor,
            "prediction_descriptor_audit": descriptor_audit,
            "eligibility_checks": method["checks"],
            "candidate_pool_identity_columns": pool_identity_columns,
            "candidate_pool_identity_sha256": pool_identity_sha256,
            "reference_predictions_identity_sha256": reference_identity_sha256,
            "candidate_count": int(len(pool)),
            "query_count": int(query_universe["query_id"].nunique()),
            "exact_candidate_join": True,
            "evaluator": evaluator,
            "independent_recomputed": True,
        }
        provenance_artifact = _write_json(method_dir / "provenance.json", provenance)
        comparisons.append(
            {
                "method_id": method["method_id"],
                "run_id": method["run_id"],
                "run_root": method["run_root"],
                "comparison": comparison_payload,
                "comparison_evidence": {
                    "exact_candidate_join": True,
                    "independent_recomputed": True,
                    "candidate_pool_identity_sha256": pool_identity_sha256,
                    "evaluator": evaluator,
                    "artifacts": {
                        "predictions": prediction_artifact,
                        "per_query": per_query_artifact,
                        "comparison": comparison_artifact,
                        "provenance": provenance_artifact,
                    },
                },
            }
        )
    exclusions_artifact = _write_json(output / "exclusions.json", exclusions)
    status = (
        "complete_no_eligible"
        if not eligible
        else "complete"
        if len(comparisons) == len(eligible)
        else "incomplete"
    )
    report: dict[str, Any] = {
        "schema_version": 1,
        "kind": "crog_existing_r10_comparison",
        "dataset": "CROG",
        "scope": scope,
        "status": status,
        "top_k": int(top_k),
        "eligible_count": len(eligible),
        "comparison_count": len(comparisons),
        "excluded_count": len(exclusions),
        "eligible_methods": eligible,
        "comparisons": comparisons,
        "exclusions": exclusions,
        "candidate_pool_identity_columns": pool_identity_columns,
        "candidate_pool_identity_sha256": pool_identity_sha256,
        "reference_predictions_identity_sha256": reference_identity_sha256,
        "reference_metrics": reference_result["metrics"],
        "evaluator": evaluator,
        "artifacts": {
            "discovery": discovery_artifact,
            "exclusions": exclusions_artifact,
            "reference_predictions": reference_artifact,
        },
    }
    validate_crog_existing_comparison_report(report)
    _write_json(output / "comparison_report.json", report)
    return report


__all__ = [
    "CrogExistingComparisonError",
    "run_crog_existing_comparisons",
    "validate_crog_existing_comparison_report",
]
