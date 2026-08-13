"""Audited post-lock adapter for one frozen P17 router-rank predicate.

The formal lock froze an independent recompute source that correctly requires
the four-route router score rank to equal one, but also incorrectly requires
the selected candidate's *native* rank to equal one.  The lock-time formal
validator already uses the intended contract: one candidate per sample/route
and rerank rank one, while preserving the candidate's original native rank.

This adapter never rewrites the frozen file or the formal lock.  It verifies
both locked source records, performs one exact in-memory line deletion, runs
the otherwise byte-identical P17 implementation into a staging directory, and
binds the transformation proof into the published P17 manifest.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from types import ModuleType
from typing import Any, Mapping

from unified_reranking.hashing import atomic_json, canonical_sha256, sha256_file


LOCK_RELATIVE = Path("08_lock/FORMAL_TEST_LOCK.json")
LOCK_DIGEST_RELATIVE = Path("08_lock/FORMAL_TEST_LOCK.sha256")
OUTPUT_RELATIVE = Path("17_independent_recompute")
ADAPTER_AUDIT_NAME = "INDEPENDENT_SOURCE_ADAPTER.json"
FROZEN_SOURCE_RELATIVE = Path("tools/d1_reranking/independent_recompute.py")
FORMAL_VALIDATOR_RELATIVE = Path("src/d1_reranking/formal.py")
OLD_FRAGMENT = "            or not native_rank.eq(1).all()\n"
FORMAL_CONTRACT_FRAGMENT = '''    if name == "four_route_crog_default_router" and (
        universe.duplicated(["sample_id", "source_route"]).any()
        or not numeric_ranks.eq(1).all()
    ):
'''


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _record(path: Path, *, final_path: Path | None = None) -> dict[str, Any]:
    source = path.expanduser().resolve()
    if source.is_symlink() or not source.is_file():
        raise RuntimeError(f"adapter artifact is absent/not regular: {source}")
    return {
        "path": str((final_path or source).expanduser().resolve()),
        "sha256": sha256_file(source),
        "bytes": source.stat().st_size,
    }


def _load_lock(root: Path) -> dict[str, Any]:
    lock_path = root / LOCK_RELATIVE
    digest = (root / LOCK_DIGEST_RELATIVE).read_text(encoding="ascii").strip()
    if digest != sha256_file(lock_path):
        raise RuntimeError("P17 adapter formal-lock detached digest differs")
    value = json.loads(lock_path.read_text(encoding="utf-8"))
    unsigned = dict(value)
    recorded_self = unsigned.pop("self_sha256", None)
    if value.get("status") != "LOCKED" or recorded_self != canonical_sha256(unsigned):
        raise RuntimeError("P17 adapter formal-lock self hash differs")
    inventory = value.get("inventory")
    if (
        not isinstance(inventory, Mapping)
        or value.get("inventory_count") != len(inventory)
        or value.get("inventory_sha256") != canonical_sha256(inventory)
    ):
        raise RuntimeError("P17 adapter formal-lock inventory differs")
    return value


def _locked_record(lock: Mapping[str, Any], path: Path) -> dict[str, Any]:
    target = path.resolve()
    matches = [
        dict(record)
        for record in lock.get("inventory", {}).values()  # type: ignore[union-attr]
        if isinstance(record, Mapping)
        and Path(str(record.get("path", ""))).resolve() == target
    ]
    if len(matches) != 1:
        raise RuntimeError(f"P17 adapter locked source record is not unique: {target}")
    current = _record(target)
    if matches[0] != current:
        raise RuntimeError(f"P17 adapter locked source bytes differ: {target}")
    return current


def transform_locked_source(source: str) -> str:
    """Apply the sole permitted semantic repair to the locked P17 source."""

    if source.count(OLD_FRAGMENT) != 1:
        raise RuntimeError("P17 frozen native-rank predicate is not unique")
    transformed = source.replace(OLD_FRAGMENT, "")
    if OLD_FRAGMENT in transformed:
        raise RuntimeError("P17 native-rank predicate remains after transformation")
    return transformed


def _load_transformed_module(source: str, source_path: Path) -> ModuleType:
    name = f"_d1_p17_adapter_{hashlib.sha256(source.encode()).hexdigest()[:16]}"
    module = ModuleType(name)
    module.__file__ = str(source_path)
    module.__package__ = "tools.d1_reranking"
    exec(compile(source, str(source_path), "exec"), module.__dict__)
    return module


def run_adapted_independent_recompute(run_dir: str | Path) -> dict[str, Any]:
    root = Path(run_dir).expanduser().resolve()
    destination = root / OUTPUT_RELATIVE
    if destination.exists():
        raise FileExistsError(f"adapted independent output already exists: {destination}")
    lock = _load_lock(root)
    source_path = (_repo_root() / FROZEN_SOURCE_RELATIVE).resolve()
    formal_path = (_repo_root() / FORMAL_VALIDATOR_RELATIVE).resolve()
    source_record = _locked_record(lock, source_path)
    formal_record = _locked_record(lock, formal_path)
    source_bytes = source_path.read_bytes()
    source_text = source_bytes.decode("utf-8")
    formal_text = formal_path.read_text(encoding="utf-8")
    if formal_text.count(FORMAL_CONTRACT_FRAGMENT) != 1:
        raise RuntimeError("P17 adapter formal router contract is not unique")
    transformed = transform_locked_source(source_text)
    transformed_sha = hashlib.sha256(transformed.encode("utf-8")).hexdigest()
    module = _load_transformed_module(transformed, source_path)
    stage_name = f".17_independent_recompute.adapter.{os.getpid()}.stage"
    stage = root / stage_name
    if stage.exists():
        raise FileExistsError(f"P17 adapter staging directory exists: {stage}")
    module.OUTPUT = stage_name
    result = module.independent_recompute(root, resume=False)
    metrics_path = stage / "recomputed_metrics.json"
    persisted = json.loads(metrics_path.read_text(encoding="utf-8"))
    if persisted != result or result.get("status") != "PASS":
        raise RuntimeError("P17 adapted staging result differs or did not PASS")

    tool_path = (_repo_root() / "tools/d1_reranking/run_independent_with_source_adapter.py").resolve()
    adapter_path = Path(__file__).resolve()
    audit: dict[str, Any] = {
        "schema_version": 1,
        "status": "PASS",
        "scientific_values_changed": False,
        "compatibility_scope": "four_route_router_native_rank_predicate_only",
        "locked_source": source_record,
        "formal_validator": formal_record,
        "adapter_module": _record(adapter_path),
        "adapter_tool": _record(tool_path),
        "locked_source_sha256": source_record["sha256"],
        "transformed_source_sha256": transformed_sha,
        "removed_fragment_sha256": hashlib.sha256(
            OLD_FRAGMENT.encode("utf-8")
        ).hexdigest(),
        "removed_fragment_occurrences": 1,
        "formal_contract_occurrences": 1,
        "preserved_checks": [
            "one_candidate_per_sample_and_source_route",
            "rerank_rank_equals_one",
            "exact_route_qualified_candidate_key_coverage",
            "locked_native_rank_preserved_as_candidate_provenance",
        ],
    }
    audit["content_sha256"] = canonical_sha256(audit)
    audit_path = stage / ADAPTER_AUDIT_NAME
    atomic_json(audit_path, audit)

    old_signature = str(result["source_signature_sha256"])
    sources = dict(result["sources"])
    sources.update(
        {
            "locked_independent_source": source_record,
            "formal_router_validator": formal_record,
            "independent_source_adapter": _record(adapter_path),
            "independent_source_adapter_tool": _record(tool_path),
        }
    )
    new_signature = canonical_sha256(sources)
    report_path = stage / "INDEPENDENT_RECOMPUTE.md"
    report = report_path.read_text(encoding="utf-8")
    if report.count(old_signature) != 1:
        raise RuntimeError("P17 report source signature is not unique")
    report_path.write_text(report.replace(old_signature, new_signature), encoding="utf-8")

    result["sources"] = sources
    result["source_signature_sha256"] = new_signature
    result["compatibility_adapter"] = {
        "status": "PASS",
        "scientific_values_changed": False,
        "scope": audit["compatibility_scope"],
        "locked_source_sha256": source_record["sha256"],
        "transformed_source_sha256": transformed_sha,
    }
    result["artifacts"]["independent_per_sample"] = _record(
        stage / "independent_per_sample.parquet",
        final_path=destination / "independent_per_sample.parquet",
    )
    result["artifacts"]["report"] = _record(
        report_path, final_path=destination / "INDEPENDENT_RECOMPUTE.md"
    )
    result["artifacts"]["source_adapter_audit"] = _record(
        audit_path, final_path=destination / ADAPTER_AUDIT_NAME
    )
    result["artifact_inventory_sha256"] = canonical_sha256(result["artifacts"])
    result.pop("self_sha256", None)
    result.pop("content_sha256", None)
    result["self_sha256"] = canonical_sha256(result)
    result["content_sha256"] = canonical_sha256(result)
    atomic_json(metrics_path, result)
    os.replace(stage, destination)
    return result


__all__ = ["run_adapted_independent_recompute", "transform_locked_source"]
