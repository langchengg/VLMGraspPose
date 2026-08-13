"""Audited source-byte adapter for the completed D1 primary matrix.

The primary plan froze the pre-fix ``datasets.py`` byte stream.  P10 needed a
dtype-insensitive equality check for numerically identical native ranks.  This
adapter proves that the Git blob still contains the exact frozen bytes, that
the live file differs by only that one comparison, and that rebuilding the
entire primary plan then substituting the frozen code record reproduces the
immutable 360-job plan exactly.
"""

from __future__ import annotations

import hashlib
import copy
from pathlib import Path
import subprocess
from typing import Any, Mapping

from unified_reranking.artifacts import (
    verified_artifact_path,
    verify_artifact_records_recursive,
)
from unified_reranking.hashing import atomic_json, canonical_sha256

from .execution import artifact_record, load_content_manifest
from .plan import (
    PRIMARY_PLAN_POINTER_RELATIVE,
    PRIMARY_PLAN_REGISTRY_RELATIVE,
    primary_plan,
)


ADAPTER_RELATIVE = Path("configs/d1_primary_source_adapter.json")
ADAPTER_HISTORY_RELATIVE = Path("configs/d1_primary_source_adapter_history")
DATASETS_RELATIVE = Path("src/unified_reranking/datasets.py")
OLD_SNIPPET = """            if feature_rank.isna().any() or not feature_rank.equals(label_rank):
"""
NEW_SNIPPET = """            if feature_rank.isna().any() or not np.array_equal(
                feature_rank.to_numpy(dtype=np.float64),
                label_rank.to_numpy(dtype=np.float64),
            ):
"""


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _git(*args: str) -> bytes:
    completed = subprocess.run(
        ["git", "-C", str(_repo_root()), *args],
        check=True,
        capture_output=True,
    )
    return completed.stdout


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def verify_exact_rank_dtype_patch(old_bytes: bytes, current_bytes: bytes) -> None:
    """Prove the live source equals the frozen source plus one exact patch."""

    old_text = old_bytes.decode("utf-8")
    current_text = current_bytes.decode("utf-8")
    if old_text.count(OLD_SNIPPET) != 1:
        raise RuntimeError("frozen datasets.py rank comparison is not unique")
    if old_text.replace(OLD_SNIPPET, NEW_SNIPPET) != current_text:
        raise RuntimeError("live datasets.py differs beyond the rank-dtype patch")


def _frozen_dataset_record(plan: Mapping[str, Any]) -> dict[str, Any]:
    code = plan.get("sources", {}).get("code")  # type: ignore[union-attr]
    if not isinstance(code, list):
        raise RuntimeError("D1 primary plan code inventory differs")
    records = [
        dict(record)
        for record in code
        if isinstance(record, Mapping)
        and Path(str(record.get("path", ""))).resolve()
        == (_repo_root() / DATASETS_RELATIVE).resolve()
    ]
    if len(records) != 1:
        raise RuntimeError("D1 primary plan datasets.py record is not unique")
    return records[0]


def _load_pointer_and_plan(
    root: Path,
) -> tuple[Path, dict[str, Any], Path, dict[str, Any]]:
    pointer_path = root / PRIMARY_PLAN_POINTER_RELATIVE
    pointer = load_content_manifest(
        pointer_path,
        name="D1 primary-plan pointer",
        statuses=("PLANNED_POINTER",),
    )
    plan_path = verified_artifact_path(
        pointer["active_plan"], name="D1 active primary plan"
    )
    if plan_path.parent != (root / PRIMARY_PLAN_REGISTRY_RELATIVE).resolve():
        raise RuntimeError("D1 active primary plan is outside its registry")
    plan = load_content_manifest(
        plan_path, name="D1 primary matrix plan", statuses=("PLANNED",)
    )
    return pointer_path, pointer, plan_path, plan


def _rebuild_with_frozen_record(
    root: Path, plan: Mapping[str, Any], frozen_record: Mapping[str, Any]
) -> None:
    sources = plan.get("sources")
    if not isinstance(sources, Mapping):
        raise RuntimeError("D1 primary plan sources differ")
    environment = sources.get("environment")
    if not isinstance(environment, Mapping):
        raise RuntimeError("D1 primary plan environment differs")
    executable = verified_artifact_path(
        environment["python_executable"], name="D1 primary Python executable"
    )
    code = sources.get("code")
    if not isinstance(code, list):
        raise RuntimeError("D1 primary plan code inventory differs")
    tool_paths = tuple(Path(str(record["path"])).resolve() for record in code)
    expected = primary_plan(
        tool_paths=tool_paths,
        run_dir=root,
        python_path=executable,
    )
    expected_code = expected["sources"]["code"]
    matches = [
        index
        for index, record in enumerate(expected_code)
        if Path(str(record["path"])).resolve()
        == (_repo_root() / DATASETS_RELATIVE).resolve()
    ]
    if len(matches) != 1:
        raise RuntimeError("rebuilt D1 primary datasets.py record is not unique")
    expected_code[matches[0]] = dict(frozen_record)
    expected["source_signature_sha256"] = canonical_sha256(expected["sources"])
    unsigned = dict(expected)
    unsigned.pop("content_sha256", None)
    expected["content_sha256"] = canonical_sha256(unsigned)
    if dict(plan) != expected:
        raise RuntimeError(
            "D1 primary plan differs beyond the exact datasets.py source record"
        )


def _adapter_tool() -> Path:
    return _repo_root() / "tools/d1_reranking/prepare_primary_source_adapter.py"


def _adapter_value(
    root: Path,
    *,
    pointer_path: Path,
    plan_path: Path,
    plan: Mapping[str, Any],
    git_commit: str,
    git_blob_oid: str,
) -> dict[str, Any]:
    frozen_record = _frozen_dataset_record(plan)
    live_path = (_repo_root() / DATASETS_RELATIVE).resolve()
    old_bytes = _git("cat-file", "blob", git_blob_oid)
    current_bytes = live_path.read_bytes()
    if _sha256_bytes(old_bytes) != frozen_record.get("sha256"):
        raise RuntimeError("Git datasets.py blob does not match the primary plan")
    verify_exact_rank_dtype_patch(old_bytes, current_bytes)
    _rebuild_with_frozen_record(root, plan, frozen_record)
    sources = {
        "primary_plan_pointer": artifact_record(pointer_path),
        "primary_plan": artifact_record(plan_path),
        "live_datasets": artifact_record(live_path),
        "adapter_module": artifact_record(Path(__file__).resolve()),
        "adapter_tool": artifact_record(_adapter_tool()),
    }
    value: dict[str, Any] = {
        "schema_version": 1,
        "status": "COMPLETE",
        "candidate_test_labels_read": False,
        "test_inputs_referenced": False,
        "scientific_values_changed": False,
        "compatibility_scope": "native_rank_numeric_dtype_equality_only",
        "frozen_source": {
            "path": str(live_path),
            "sha256": frozen_record["sha256"],
            "git_commit": git_commit,
            "git_blob_oid": git_blob_oid,
        },
        "live_source": artifact_record(live_path),
        "plan_rebuild_after_frozen_record_substitution": "EXACT",
        "source_signature_sha256": canonical_sha256(sources),
        "sources": sources,
    }
    value["content_sha256"] = canonical_sha256(value)
    return value


def prepare_primary_source_adapter(
    run_dir: str | Path, *, resume: bool, refresh: bool = False
) -> dict[str, Any]:
    root = Path(run_dir).expanduser().resolve()
    pointer_path, _pointer, plan_path, plan = _load_pointer_and_plan(root)
    git_commit = _git("rev-parse", "HEAD").decode().strip()
    git_blob_oid = (
        _git("rev-parse", f"{git_commit}:{DATASETS_RELATIVE}").decode().strip()
    )
    value = _adapter_value(
        root,
        pointer_path=pointer_path,
        plan_path=plan_path,
        plan=plan,
        git_commit=git_commit,
        git_blob_oid=git_blob_oid,
    )
    path = root / ADAPTER_RELATIVE
    if path.exists():
        existing = load_content_manifest(
            path, name="D1 primary source adapter", statuses=("COMPLETE",)
        )
        if resume and existing == value:
            return existing
        if not refresh:
            raise RuntimeError("D1 primary source adapter exists and differs")
        history_path = (
            root / ADAPTER_HISTORY_RELATIVE / f"{existing['content_sha256']}.json"
        )
        if history_path.exists():
            archived = load_content_manifest(
                history_path,
                name="archived D1 primary source adapter",
                statuses=("COMPLETE",),
            )
            if archived != existing:
                raise RuntimeError("archived D1 primary source adapter differs")
        else:
            atomic_json(history_path, existing)
    atomic_json(path, value)
    return value


def load_primary_plan_with_source_adapter(
    run_dir: str | Path,
) -> tuple[Path, dict[str, Any], dict[str, Any], dict[str, Any]]:
    root = Path(run_dir).expanduser().resolve()
    pointer_path, _pointer, plan_path, plan = _load_pointer_and_plan(root)
    adapter_path = root / ADAPTER_RELATIVE
    adapter = load_content_manifest(
        adapter_path, name="D1 primary source adapter", statuses=("COMPLETE",)
    )
    frozen = adapter.get("frozen_source")
    if not isinstance(frozen, Mapping):
        raise RuntimeError("D1 primary adapter frozen source differs")
    expected = _adapter_value(
        root,
        pointer_path=pointer_path,
        plan_path=plan_path,
        plan=plan,
        git_commit=str(frozen.get("git_commit", "")),
        git_blob_oid=str(frozen.get("git_blob_oid", "")),
    )
    if adapter != expected:
        raise RuntimeError("D1 primary source adapter contract differs")
    return plan_path, plan, artifact_record(adapter_path), adapter


def verify_primary_cell_records_with_source_adapter(
    cell: Mapping[str, Any],
    *,
    adapter: Mapping[str, Any],
    name: str,
) -> None:
    """Verify a frozen primary cell against the one audited source patch.

    Completed primary cells legitimately bind the pre-fix ``datasets.py``
    bytes in two identical training-code inventories.  The adapter proves the
    live file is exactly that frozen Git blob plus the numeric dtype equality
    fix.  This helper substitutes the already-verified live record only in
    those two exact inventory locations, then applies the normal recursive
    artifact verifier to every source and output record.
    """

    frozen = adapter.get("frozen_source")
    live = adapter.get("live_source")
    if not isinstance(frozen, Mapping) or not isinstance(live, Mapping):
        raise RuntimeError("D1 primary source adapter records differ")
    frozen_path = str(frozen.get("path", ""))
    frozen_sha = str(frozen.get("sha256", ""))
    if not frozen_path or not frozen_sha:
        raise RuntimeError("D1 primary frozen source record differs")

    verification_value = copy.deepcopy(dict(cell))
    observed_records: list[dict[str, Any]] = []
    for location in (
        ("configuration", "sources", "training_code"),
        ("sources", "training_code"),
    ):
        node: Any = verification_value
        for key in location:
            if not isinstance(node, dict) or key not in node:
                raise RuntimeError(
                    f"{name} misses frozen training-code inventory at "
                    + ".".join(location)
                )
            node = node[key]
        if not isinstance(node, list):
            raise RuntimeError(
                f"{name} training-code inventory is not a list at " + ".".join(location)
            )
        matches = [
            index
            for index, record in enumerate(node)
            if isinstance(record, Mapping)
            and str(record.get("path", "")) == frozen_path
        ]
        if len(matches) != 1:
            raise RuntimeError(
                f"{name} frozen datasets.py record is not unique at "
                + ".".join(location)
            )
        index = matches[0]
        record = node[index]
        if (
            record.get("sha256") != frozen_sha
            or not isinstance(record.get("bytes"), int)
            or int(record["bytes"]) <= 0
        ):
            raise RuntimeError(f"{name} frozen datasets.py record differs")
        observed_records.append(dict(record))
        node[index] = dict(live)
    if observed_records[0] != observed_records[1]:
        raise RuntimeError(f"{name} frozen datasets.py records disagree")
    verify_artifact_records_recursive(
        verification_value,
        name=name,
        require_at_least_one=True,
    )


def verify_records_with_primary_source_adapter(
    value: Mapping[str, Any],
    *,
    adapter: Mapping[str, Any],
    name: str,
) -> int:
    """Recursively verify an artifact that binds the audited frozen source.

    Unlike the primary-cell helper, downstream applications may bind
    ``datasets.py`` once and omit a byte count.  Every occurrence must still
    have the exact frozen path and SHA; all other records use the ordinary
    recursive verifier without adaptation.
    """

    frozen = adapter.get("frozen_source")
    live = adapter.get("live_source")
    if not isinstance(frozen, Mapping) or not isinstance(live, Mapping):
        raise RuntimeError("D1 primary source adapter records differ")
    frozen_path = str(frozen.get("path", ""))
    frozen_sha = str(frozen.get("sha256", ""))
    verification_value = copy.deepcopy(dict(value))
    adapted = 0

    def visit(node: Any) -> None:
        nonlocal adapted
        if isinstance(node, dict):
            if str(node.get("path", "")) == frozen_path and "sha256" in node:
                if node.get("sha256") != frozen_sha:
                    raise RuntimeError(f"{name} frozen datasets.py record differs")
                node.clear()
                node.update(dict(live))
                adapted += 1
                return
            for child in node.values():
                visit(child)
        elif isinstance(node, list):
            for child in node:
                visit(child)

    visit(verification_value)
    if adapted <= 0:
        raise RuntimeError(f"{name} contains no audited frozen datasets.py record")
    verify_artifact_records_recursive(
        verification_value,
        name=name,
        require_at_least_one=True,
    )
    return adapted


__all__ = [
    "ADAPTER_RELATIVE",
    "ADAPTER_HISTORY_RELATIVE",
    "load_primary_plan_with_source_adapter",
    "prepare_primary_source_adapter",
    "verify_primary_cell_records_with_source_adapter",
    "verify_records_with_primary_source_adapter",
    "verify_exact_rank_dtype_patch",
]
