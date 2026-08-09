"""Repository-local provenance and candidate identity contracts.

This module contains no artifact writers.  Paths are derived from this file's
location rather than the process working directory, and hashing is streamed so
large canonical artifacts are never loaded wholly into memory.
"""

from __future__ import annotations

import hashlib
import os
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
HIFI_ROOT = REPO_ROOT / "HiFi_reproduction"
CROG_ROOT = REPO_ROOT / "crog_reproduction" / "CROG"

MODULAR_CANONICAL_RUN = (
    HIFI_ROOT
    / "runs"
    / "modular_hierfilm_standard_dexnet_gqcnn_20260728_094528"
)
MODULAR_INCOMPLETE_RERANK_RUN = (
    HIFI_ROOT
    / "runs"
    / "modular_reranking_repeatedfilm_v1_20260729_203147"
)
CROG_CANONICAL_RUN = (
    CROG_ROOT
    / "failure_analysis"
    / "reranking_outputs"
    / "v3_fullchain_20260801T114301+0100"
)
CROG_CANONICAL_CANDIDATES = (
    CROG_ROOT
    / "failure_analysis"
    / "reranking_outputs"
    / "full_test_17749_v1"
    / "features.jsonl"
)
HIFI_CANONICAL_LABELS = (
    HIFI_ROOT / "src" / "grasping" / "reranking_v1" / "labels.py"
)

CANONICAL_PATHS: dict[str, Path] = {
    "repo_root": REPO_ROOT,
    "hifi_root": HIFI_ROOT,
    "crog_root": CROG_ROOT,
    "modular_run": MODULAR_CANONICAL_RUN,
    "modular_incomplete_rerank_run": MODULAR_INCOMPLETE_RERANK_RUN,
    "crog_run": CROG_CANONICAL_RUN,
    "crog_candidates": CROG_CANONICAL_CANDIDATES,
    "hifi_label_evaluator": HIFI_CANONICAL_LABELS,
}


# These are observed hashes of immutable canonical inputs/results.  Relative
# paths make the contract portable when the repository is relocated.
KNOWN_HASHES: dict[str, str] = {
    "HiFi_reproduction/src/grasping/reranking_v1/labels.py": (
        "e2df8fba3b7bf60676b30f81627fea84a7b82b69c82d729e6b0de59c8b8be983"
    ),
    "HiFi_reproduction/runs/modular_hierfilm_standard_dexnet_gqcnn_20260728_094528/final_output_manifest.json": (
        "938d1e0298f90016ef9977e5e6ba5c8f615d7b7712093c8cd126141143fea911"
    ),
    "HiFi_reproduction/runs/modular_hierfilm_standard_dexnet_gqcnn_20260728_094528/input_manifest.csv": (
        "86cafdddb8996758ff05fdf669a9dbce48620ada8336238ecf20ea862a0b4ae6"
    ),
    "HiFi_reproduction/runs/modular_hierfilm_standard_dexnet_gqcnn_20260728_094528/sample_status.jsonl": (
        "71c077d794bf87ea08d37603e8ad26c6a0ae74d799ff7c6759dcf623a32f085c"
    ),
    "HiFi_reproduction/runs/modular_hierfilm_standard_dexnet_gqcnn_20260728_094528/candidates/dexnet_raw_candidates.parquet": (
        "13e4a92178cb8b8319050a556ead9e62314f18e29a3c9385ddeec83c8ffb9782"
    ),
    "HiFi_reproduction/runs/modular_hierfilm_standard_dexnet_gqcnn_20260728_094528/candidates/dexnet_nms_candidates.parquet": (
        "00a2cd9ce1d3d5007b0a25ff4c448497772e9b9944c7d46129213d63b34eb8cd"
    ),
    "HiFi_reproduction/runs/modular_hierfilm_standard_dexnet_gqcnn_20260728_094528/scores/gqcnn_per_candidate.parquet": (
        "7094c770fee08208ca18ffc629e81b2b678d4d8ea3d4c785d46b7f8dff948a50"
    ),
    "HiFi_reproduction/runs/modular_hierfilm_standard_dexnet_gqcnn_20260728_094528/evaluation/hierfilm_gqcnn_per_candidate.parquet": (
        "95cc1144ce9bad44278255aa1eec33fa1c3f8941b48a6af18a3a8bb564e3c04a"
    ),
    "crog_reproduction/CROG/utils/grasp_metrics.py": (
        "5d39ef6258d385a6a2384b1c5412d407498f148bc2cf6eff89bbf43dbee7b06d"
    ),
    "crog_reproduction/CROG/failure_analysis/reranking_outputs/full_test_17749_v1/features.jsonl": (
        "04f467d669a4cfb2ccafca61e9dee18ebaa18c093133798533a2761221f652aa"
    ),
    "crog_reproduction/CROG/failure_analysis/reranking_outputs/full_test_17749_v1/metadata.json": (
        "a7cf3065d6d8f3773c828f99701284896f6a0b3a84308ff7fa5892c58a1e452e"
    ),
    "crog_reproduction/CROG/failure_analysis/reranking_outputs/v3_fullchain_20260801T114301+0100/frozen_experiment_manifest.json": (
        "e5ea054211060d7e6faa5a75be6018cb512d8fdd1ee782b0156730ab32bd7a9c"
    ),
    "crog_reproduction/CROG/failure_analysis/reranking_outputs/v3_fullchain_20260801T114301+0100/formal/test/TEST_RUN_COMPLETE.json": (
        "c7095f019037f7ad976c33b6222d2fa2ff95e5a655a261a13f89a715514ebecf"
    ),
}


@dataclass(frozen=True)
class CanonicalArtifact:
    """One repository-relative artifact with an expected SHA-256 digest."""

    name: str
    relative_path: str
    sha256: str

    @property
    def path(self) -> Path:
        return REPO_ROOT / self.relative_path

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["path"] = str(self.path)
        return result


CANONICAL_ARTIFACTS: tuple[CanonicalArtifact, ...] = tuple(
    CanonicalArtifact(
        name=Path(relative_path).name,
        relative_path=relative_path,
        sha256=sha256,
    )
    for relative_path, sha256 in KNOWN_HASHES.items()
)


def streaming_sha256(
    path: str | os.PathLike[str],
    *,
    chunk_size: int = 1024 * 1024,
    reject_symlinks: bool = True,
) -> str:
    """Return a SHA-256 digest while reading at most ``chunk_size`` bytes."""

    source = Path(path)
    chunk_size = int(chunk_size)
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    if reject_symlinks and source.is_symlink():
        raise ValueError(f"refusing to hash symlink: {source}")
    if not source.is_file():
        raise FileNotFoundError(f"not a regular file: {source}")
    digest = hashlib.sha256()
    with source.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def file_inventory(
    root: str | os.PathLike[str],
    *,
    hash_files: bool = False,
    include_hidden: bool = True,
) -> list[dict[str, Any]]:
    """Return a deterministic recursive file/symlink inventory."""

    base = Path(root)
    if not base.is_dir():
        raise NotADirectoryError(base)
    records: list[dict[str, Any]] = []
    for path in sorted(base.rglob("*"), key=lambda value: value.as_posix()):
        relative = path.relative_to(base)
        if not include_hidden and any(part.startswith(".") for part in relative.parts):
            continue
        if path.is_symlink():
            records.append(
                {
                    "relative_path": relative.as_posix(),
                    "type": "symlink",
                    "size_bytes": None,
                    "sha256": None,
                    "symlink_target": os.readlink(path),
                }
            )
        elif path.is_file():
            records.append(
                {
                    "relative_path": relative.as_posix(),
                    "type": "file",
                    "size_bytes": int(path.stat().st_size),
                    "sha256": streaming_sha256(path) if hash_files else None,
                    "symlink_target": None,
                }
            )
    return records


def _records(values: Any, name: str) -> list[Mapping[str, Any]]:
    if hasattr(values, "to_dict"):
        try:
            converted = values.to_dict(orient="records")
        except TypeError:
            converted = None
        if converted is not None:
            values = converted
    if isinstance(values, Mapping):
        raise TypeError(f"{name} must be a table or iterable of records")
    result = list(values)
    if not all(isinstance(value, Mapping) for value in result):
        raise TypeError(f"{name} must contain mapping records")
    return result


def candidate_key_sequence(
    records: Any,
    *,
    query_col: str = "sample_id",
    candidate_id_col: str = "candidate_id",
    name: str = "candidate records",
) -> list[tuple[str, str]]:
    """Extract validated candidate keys without changing source row order."""

    keys: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for index, record in enumerate(_records(records, name)):
        if query_col not in record or candidate_id_col not in record:
            raise ValueError(
                f"{name} row {index} lacks {query_col}/{candidate_id_col}"
            )
        raw_query = record[query_col]
        raw_candidate = record[candidate_id_col]
        if raw_query is None or raw_candidate is None:
            raise ValueError(f"{name} row {index} has a null candidate key")
        key = (str(raw_query), str(raw_candidate))
        if not key[0] or not key[1]:
            raise ValueError(f"{name} row {index} has an empty candidate key")
        if key in seen:
            raise ValueError(f"{name} contains duplicate candidate key {key}")
        seen.add(key)
        keys.append(key)
    return keys


def assert_candidate_keys_and_source_order(
    reference_records: Any,
    observed_records: Any,
    *,
    query_col: str = "sample_id",
    candidate_id_col: str = "candidate_id",
    require_query_order: bool = False,
) -> dict[str, Any]:
    """Require exact key sets and identical within-query candidate order.

    Query blocks may be reordered as a whole by default.  Set
    ``require_query_order`` to require byte-table-like global row order too.
    """

    reference = candidate_key_sequence(
        reference_records,
        query_col=query_col,
        candidate_id_col=candidate_id_col,
        name="reference candidates",
    )
    observed = candidate_key_sequence(
        observed_records,
        query_col=query_col,
        candidate_id_col=candidate_id_col,
        name="observed candidates",
    )

    def query_blocks(keys: list[tuple[str, str]], name: str) -> list[str]:
        order: list[str] = []
        completed: set[str] = set()
        current: str | None = None
        for query_id, _ in keys:
            if query_id == current:
                continue
            if query_id in completed:
                raise ValueError(f"{name} query block is non-contiguous: {query_id}")
            if current is not None:
                completed.add(current)
            current = query_id
            order.append(query_id)
        return order

    reference_query_order = query_blocks(reference, "reference candidates")
    observed_query_order = query_blocks(observed, "observed candidates")
    expected_set, observed_set = set(reference), set(observed)
    if expected_set != observed_set:
        missing = sorted(expected_set - observed_set)[:5]
        extra = sorted(observed_set - expected_set)[:5]
        raise ValueError(f"candidate key set changed: missing={missing}, extra={extra}")
    expected_by_query: dict[str, list[str]] = defaultdict(list)
    observed_by_query: dict[str, list[str]] = defaultdict(list)
    for query_id, candidate_id in reference:
        expected_by_query[query_id].append(candidate_id)
    for query_id, candidate_id in observed:
        observed_by_query[query_id].append(candidate_id)
    for query_id in expected_by_query:
        if expected_by_query[query_id] != observed_by_query[query_id]:
            raise ValueError(f"candidate source order changed for query {query_id}")
    if require_query_order and reference_query_order != observed_query_order:
        raise ValueError("global query/source order changed")
    return {
        "candidate_count": len(reference),
        "query_count": len(expected_by_query),
        "key_set_equal": True,
        "source_order_equal_within_query": True,
        "global_order_equal": reference == observed,
    }


validate_candidate_identity = assert_candidate_keys_and_source_order


__all__ = [
    "CANONICAL_ARTIFACTS",
    "CANONICAL_PATHS",
    "CROG_CANONICAL_CANDIDATES",
    "CROG_CANONICAL_RUN",
    "CROG_ROOT",
    "CanonicalArtifact",
    "HIFI_CANONICAL_LABELS",
    "HIFI_ROOT",
    "KNOWN_HASHES",
    "MODULAR_CANONICAL_RUN",
    "MODULAR_INCOMPLETE_RERANK_RUN",
    "REPO_ROOT",
    "assert_candidate_keys_and_source_order",
    "candidate_key_sequence",
    "file_inventory",
    "streaming_sha256",
    "validate_candidate_identity",
]
