"""Deterministic Validation-only evidence tables for the D1 P13 lock."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pandas as pd

from unified_reranking.artifacts import (
    verified_artifact_path,
    verify_artifact_records_recursive,
)
from unified_reranking.hashing import atomic_json, canonical_sha256, sha256_file

from .ablation_schema_adapter import ADAPTER_RELATIVE
from .ablation_selection import SELECTION_RELATIVE
from .execution import load_content_manifest
from .four_route import validation_router_union_summary
from .io import atomic_csv
from .provenance import load_source_closure


MANIFEST_RELATIVE_PATH = "configs/d1_validation_evidence_tables.json"
OUTPUT_ROOT = "07_validation/postformal_evidence"
TABLE_NAMES = (
    "d1_provenance_comparison.csv",
    "d1_r0_r7_validation.csv",
    "d1_k_sensitivity.csv",
    "d1_evidence_tracks.csv",
    "d1_feature_ablation.csv",
    "d1_gate_transitions.csv",
    "four_route_router.csv",
    "top20_union.csv",
    "complete_four_route_comparison.csv",
)
TABLE_SCHEMAS = {
    "d1_provenance_comparison.csv": (
        "snapshot",
        "run_path",
        "physical_candidate_pool",
        "candidate_rows",
        "paired_samples",
        "no_output_samples",
        "train_validation_pools",
        "current_evaluator_labels",
        "canonical_use",
        "decision",
        "metric_status",
        "reason",
        "evidence_root",
    ),
    "d1_r0_r7_validation.csv": (
        "method",
        "role",
        "status",
        "validation_sample_count",
        "source_manifest_sha256",
        "validation_mrr_at_5",
        "validation_ndcg_at_5",
        "delta_vs_r0",
        "validation_j_at_1",
    ),
    "d1_k_sensitivity.csv": (
        "scenario_id",
        "pool",
        "track",
        "max_candidates",
        "role",
        "method",
        "validation_j_at_1",
        "validation_mrr_at_5",
        "validation_ndcg_at_5",
        "oof_j_at_1",
        "oof_mrr_at_5",
        "oof_ndcg_at_5",
    ),
    "d1_evidence_tracks.csv": (
        "track",
        "method",
        "status",
        "sample_count",
        "j_at_1",
        "delta_vs_r0",
        "gate_decision",
    ),
    "d1_feature_ablation.csv": (
        "variant_id",
        "kind",
        "target_family",
        "status",
        "feature_count",
        "reference_variant_id",
        "reason",
        "validation_j_at_1_mean",
        "validation_j_at_1_std",
        "delta_vs_full_t4_mean",
        "delta_vs_full_t4_std",
        "per_seed_json",
    ),
    "d1_gate_transitions.csv": (
        "transition",
        "count",
        "sample_count",
        "rate",
        "switch_state",
    ),
    **{
        name: (
            "section",
            "system",
            "route",
            "metric",
            "value",
            "sample_count",
            "development_split",
        )
        for name in (
            "four_route_router.csv",
            "top20_union.csv",
            "complete_four_route_comparison.csv",
        )
    },
}
FIXED_SOURCES = {
    "r0_r1": "07_validation/r0_r1_selection.json",
    "primary": "07_validation/selected_primary_ungated.json",
    "gate": "07_validation/gate/d1/gate_selection.json",
    "k": "11_k_sensitivity/selection_manifest.json",
    "ablation": str(SELECTION_RELATIVE),
    "ablation_adapter": str(ADAPTER_RELATIVE),
    "p12": "13_four_route_extension/validation_router_union_manifest.json",
}


def _record(path: str | Path) -> dict[str, Any]:
    source = Path(path).expanduser().resolve()
    if source.is_symlink() or not source.is_file():
        raise RuntimeError(f"D1 Validation evidence source is absent: {source}")
    return {
        "path": str(source),
        "sha256": sha256_file(source),
        "bytes": source.stat().st_size,
    }


def _manifest(root: Path, name: str) -> dict[str, Any]:
    value = load_content_manifest(
        root / FIXED_SOURCES[name],
        name=f"D1 Validation evidence {name}",
        statuses=("COMPLETE",),
    )
    if value.get("candidate_test_labels_read") is not False:
        raise PermissionError(f"D1 Validation evidence {name} is not label-isolated")
    verify_artifact_records_recursive(
        value, name=f"D1 Validation evidence {name}", require_at_least_one=True
    )
    return value


def _artifact(manifest: Mapping[str, Any], key: str, *, name: str) -> Path:
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, Mapping) or not isinstance(
        artifacts.get(key), Mapping
    ):
        raise RuntimeError(f"D1 Validation evidence artifact is absent: {name}/{key}")
    return verified_artifact_path(artifacts[key], name=f"{name}/{key}")


def _metric(metrics: Mapping[str, Any], key: str) -> float:
    value = metrics.get(key)
    if not isinstance(value, (int, float)):
        return float("nan")
    return float(value)


def _r0_r7(
    r0_r1: Mapping[str, Any], primary: Mapping[str, Any], gate: Mapping[str, Any]
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for index in (0, 1):
        metrics = r0_r1.get(f"r{index}_validation_metrics")
        if not isinstance(metrics, Mapping):
            raise RuntimeError(f"D1 R{index} Validation metrics are absent")
        rows.append(
            {
                "method": f"R{index}",
                "role": "baseline",
                "validation_mrr_at_5": _metric(metrics, "mrr_at_5"),
                "validation_ndcg_at_5": _metric(metrics, "ndcg_at_5"),
                "validation_j_at_1": _metric(metrics, "j_at_1"),
            }
        )
    winners = primary.get("method_winners")
    if not isinstance(winners, Mapping) or set(winners) != {
        f"R{i}" for i in range(2, 7)
    }:
        raise RuntimeError("D1 R2-R6 Validation winner inventory differs")
    for index in range(2, 7):
        row = winners[f"R{index}"]
        if not isinstance(row, Mapping):
            raise RuntimeError(f"D1 R{index} Validation winner is absent")
        rows.append(
            {
                "method": f"R{index}",
                "role": "selected_primary"
                if primary.get("selected_method") == f"R{index}"
                else "screened",
                "validation_mrr_at_5": _metric(row, "validation_mrr_at_5"),
                "validation_ndcg_at_5": _metric(row, "validation_ndcg_at_5"),
                "validation_j_at_1": _metric(row, "validation_j_at_1"),
            }
        )
    gate_metrics = gate.get("validation_metrics")
    if not isinstance(gate_metrics, Mapping):
        raise RuntimeError("D1 R7 Validation metrics are absent")
    rows.append(
        {
            "method": "R7",
            "role": "expected_gain_gate",
            "validation_mrr_at_5": float("nan"),
            "validation_ndcg_at_5": float("nan"),
            "validation_j_at_1": _metric(gate_metrics, "gated_j_at_1"),
        }
    )
    result = pd.DataFrame(rows)
    if result["validation_j_at_1"].isna().any() or list(result["method"]) != [
        f"R{i}" for i in range(8)
    ]:
        raise RuntimeError("D1 R0-R7 Validation table is incomplete")
    return result


def _same_frame(left: pd.DataFrame, right: pd.DataFrame, *, name: str) -> None:
    try:
        pd.testing.assert_frame_equal(
            left.reset_index(drop=True), right.reset_index(drop=True), check_dtype=False
        )
    except AssertionError as error:
        raise RuntimeError(f"D1 {name} semantic replay differs") from error


def _gate_transitions(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"native_correct", "challenger_correct", "switch"}
    if frame.empty or required.difference(frame.columns):
        raise RuntimeError("D1 gate Validation decision schema differs")
    native = frame["native_correct"].astype(bool)
    challenger = frame["challenger_correct"].astype(bool)
    switch = frame["switch"].astype(bool)
    masks = {
        "recovered": (~native & challenger & switch, "SWITCH"),
        "harmful": (native & ~challenger & switch, "SWITCH"),
        "missed_recoverable": (~native & challenger & ~switch, "STAY"),
        "prevented_harmful": (native & ~challenger & ~switch, "STAY"),
        "wrong_to_wrong": (~native & ~challenger, "MIXED"),
        "correct_to_correct": (native & challenger, "MIXED"),
    }
    denominator = len(frame)
    return pd.DataFrame(
        [
            {
                "transition": name,
                "count": int(mask.sum()),
                "sample_count": denominator,
                "rate": float(mask.mean()),
                "switch_state": state,
            }
            for name, (mask, state) in masks.items()
        ]
    )


def _replay_sources(
    root: Path, closure_path: Path, closure: Mapping[str, Any]
) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    """Freshly replay all Validation evidence from its fixed source manifests."""

    manifests = {name: _manifest(root, name) for name in FIXED_SOURCES}
    r0_r7 = _r0_r7(manifests["r0_r1"], manifests["primary"], manifests["gate"])
    gate_decisions = pd.read_parquet(
        _artifact(manifests["gate"], "validation_decisions", name="gate")
    )
    r0_decisions = pd.read_parquet(
        _artifact(manifests["r0_r1"], "r0_validation_decisions", name="R0/R1"),
        columns=["sample_id"],
    )
    primary_decisions = pd.read_parquet(
        _artifact(
            manifests["primary"], "selected_validation_decisions", name="primary"
        ),
        columns=["sample_id"],
    )
    validation_count = len(gate_decisions)
    denominator = set(gate_decisions["sample_id"].astype(str))
    if (
        validation_count <= 0
        or len(denominator) != validation_count
        or set(r0_decisions["sample_id"].astype(str)) != denominator
        or set(primary_decisions["sample_id"].astype(str)) != denominator
    ):
        raise RuntimeError("D1 Validation evidence denominator differs")
    r0_r7["status"] = "SELECTED_OR_REPORTED"
    r0_r7["validation_sample_count"] = validation_count
    source_hashes = {
        "R0": sha256_file(root / FIXED_SOURCES["r0_r1"]),
        "R1": sha256_file(root / FIXED_SOURCES["r0_r1"]),
        **{f"R{i}": sha256_file(root / FIXED_SOURCES["primary"]) for i in range(2, 7)},
        "R7": sha256_file(root / FIXED_SOURCES["gate"]),
    }
    r0_r7["source_manifest_sha256"] = r0_r7["method"].map(source_hashes)
    r0_value = float(r0_r7.loc[r0_r7["method"].eq("R0"), "validation_j_at_1"].iloc[0])
    r0_r7["delta_vs_r0"] = r0_r7["validation_j_at_1"] - r0_value
    r0_r7 = r0_r7.loc[:, TABLE_SCHEMAS["d1_r0_r7_validation.csv"]]

    p12 = manifests["p12"]
    p12_samples = pd.read_parquet(_artifact(p12, "router_decisions", name="P12"))
    p12_union = pd.read_parquet(_artifact(p12, "top20_union", name="P12"))
    p12_summary = validation_router_union_summary(p12_samples, p12_union)
    _same_frame(
        pd.read_csv(_artifact(p12, "prelock_table", name="P12")),
        p12_summary,
        name="P12 Validation summary",
    )

    registry = (
        closure.get("artifacts", {}).get("run_registry")
        if isinstance(closure.get("artifacts"), Mapping)
        else None
    )
    if not isinstance(registry, Mapping):
        registry_path = closure_path.parent / "D1_RUN_REGISTRY.csv"
    else:
        registry_path = verified_artifact_path(registry, name="D1 provenance registry")
    tables = {
        "d1_provenance_comparison.csv": pd.read_csv(registry_path),
        "d1_r0_r7_validation.csv": r0_r7,
        "d1_k_sensitivity.csv": pd.read_csv(
            _artifact(manifests["k"], "comparison_table", name="K")
        ),
        "d1_evidence_tracks.csv": pd.read_csv(
            _artifact(manifests["ablation"], "evidence_track_table", name="ablation")
        ),
        "d1_feature_ablation.csv": pd.read_csv(
            _artifact(manifests["ablation"], "feature_ablation_table", name="ablation")
        ),
        "d1_gate_transitions.csv": _gate_transitions(gate_decisions),
        "four_route_router.csv": p12_summary.loc[
            p12_summary["section"].str.startswith("router")
        ].reset_index(drop=True),
        "top20_union.csv": p12_summary.loc[
            p12_summary["section"].str.startswith("union")
        ].reset_index(drop=True),
        "complete_four_route_comparison.csv": p12_summary,
    }
    if set(tables) != set(TABLE_NAMES) or any(frame.empty for frame in tables.values()):
        raise RuntimeError("D1 Validation evidence table inventory is incomplete")
    for name, frame in tables.items():
        if tuple(frame.columns) != TABLE_SCHEMAS[name]:
            raise RuntimeError(f"D1 Validation evidence schema differs: {name}")
    sources = {
        "source_closure": _record(closure_path),
        "provenance_registry": _record(registry_path),
        **{name: _record(root / path) for name, path in FIXED_SOURCES.items()},
    }
    return tables, sources


def assemble_validation_evidence_tables(
    run_dir: str | Path, *, resume: bool = False
) -> dict[str, Any]:
    root = Path(run_dir).expanduser().resolve()
    destination = root / MANIFEST_RELATIVE_PATH
    if any(
        (root / value).exists()
        for value in (
            "08_lock/FORMAL_TEST_LOCK.json",
            "FINAL_RUN_LOCK.json",
            "COMPLETE",
        )
    ):
        raise PermissionError(
            "D1 Validation evidence must be assembled before formal lock"
        )
    if destination.exists():
        if not resume:
            raise FileExistsError(
                f"D1 Validation evidence already exists: {destination}"
            )
        return load_validation_evidence_tables(root)[1]

    closure_path, closure = load_source_closure(root)
    tables, sources = _replay_sources(root, closure_path, closure)
    output = root / OUTPUT_ROOT
    artifacts = {
        name: _record(atomic_csv(tables[name], output / name)) for name in TABLE_NAMES
    }
    r0_path = root / "07_validation/tables/r0_r7_full.csv"
    atomic_csv(tables["d1_r0_r7_validation.csv"], r0_path)
    result: dict[str, Any] = {
        "schema_version": 1,
        "status": "COMPLETE",
        "development_split": "Validation",
        "candidate_test_labels_read": False,
        "selection_used_test_metrics": False,
        "fixed_source_paths": FIXED_SOURCES,
        "table_names": list(TABLE_NAMES),
        "sources": sources,
        "source_signature_sha256": canonical_sha256(sources),
        "artifacts": {"tables": artifacts, "r0_r7_full": _record(r0_path)},
    }
    result["content_sha256"] = canonical_sha256(result)
    atomic_json(destination, result)
    return result


def load_validation_evidence_tables(run_dir: str | Path) -> tuple[Path, dict[str, Any]]:
    root = Path(run_dir).expanduser().resolve()
    path = root / MANIFEST_RELATIVE_PATH
    value = load_content_manifest(
        path, name="D1 Validation evidence tables", statuses=("COMPLETE",)
    )
    tables = (
        value.get("artifacts", {}).get("tables")
        if isinstance(value.get("artifacts"), Mapping)
        else None
    )
    if (
        value.get("candidate_test_labels_read") is not False
        or value.get("selection_used_test_metrics") is not False
        or value.get("development_split") != "Validation"
        or value.get("fixed_source_paths") != FIXED_SOURCES
        or value.get("table_names") != list(TABLE_NAMES)
        or value.get("source_signature_sha256")
        != canonical_sha256(value.get("sources"))
        or not isinstance(tables, Mapping)
        or set(tables) != set(TABLE_NAMES)
    ):
        raise RuntimeError("D1 Validation evidence table contract differs")
    closure_path, closure = load_source_closure(root)
    replayed, current_sources = _replay_sources(root, closure_path, closure)
    if value.get("sources") != current_sources:
        raise RuntimeError("D1 Validation evidence source records differ")
    for name in TABLE_NAMES:
        table_path = verified_artifact_path(
            tables[name], name=f"D1 Validation table {name}"
        )
        frame = pd.read_csv(table_path)
        if (
            table_path != (root / OUTPUT_ROOT / name).resolve()
            or frame.empty
            or tuple(frame.columns) != TABLE_SCHEMAS[name]
        ):
            raise RuntimeError(f"D1 Validation evidence table differs: {name}")
        _same_frame(frame, replayed[name], name=f"Validation evidence {name}")
    r0_record = value.get("artifacts", {}).get("r0_r7_full")
    if not isinstance(r0_record, Mapping):
        raise RuntimeError("D1 r0_r7_full artifact record is absent")
    r0_path = verified_artifact_path(r0_record, name="D1 r0_r7_full")
    if r0_path != (root / "07_validation/tables/r0_r7_full.csv").resolve():
        raise RuntimeError("D1 r0_r7_full path differs")
    _same_frame(
        pd.read_csv(r0_path),
        replayed["d1_r0_r7_validation.csv"],
        name="r0_r7_full",
    )
    verify_artifact_records_recursive(
        value, name="D1 Validation evidence", require_at_least_one=True
    )
    return path, value
