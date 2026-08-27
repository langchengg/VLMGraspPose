"""Outcome-blind C1 pilot selection and independent retrospective acceptance.

The pilot is selected before the protocol lock from frozen *predicted-branch*
outcomes plus P2 mapping covariates.  No GT-counterfactual outcome is an input.
The selected source view is content addressed and is therefore suitable for
binding into the immutable protocol declaration.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

from .g1_c1_adapter import LABEL_PROJECTION, verify_g1_c1_source_adapter
from .io import (
    artifact_record,
    atomic_copy,
    atomic_json,
    atomic_parquet,
    canonical_sha256,
)


PILOT_SIZE = 200
PILOT_SALT = "gtmask-c1-pilot-v1"
MANIFEST_NAME = "C1_PILOT_SOURCE_ADAPTER.json"
PREDICTED_CLASSES = (
    "predicted_success",
    "predicted_ranking_limited",
    "predicted_no_positive",
    "predicted_no_output",
)


class C1PilotError(RuntimeError):
    """The C1 audit pilot differs from its deterministic pre-lock contract."""


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _object(path: str | Path, *, label: str) -> dict[str, Any]:
    source = Path(path).expanduser().resolve()
    if source.is_symlink() or not source.is_file():
        raise C1PilotError(f"{label} is absent or unsafe: {source}")
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise C1PilotError(f"cannot parse {label}: {source}") from error
    if not isinstance(value, dict):
        raise C1PilotError(f"{label} must contain one JSON object")
    return value


def _self_hashed(path: str | Path, *, label: str) -> dict[str, Any]:
    value = _object(path, label=label)
    unsigned = dict(value)
    recorded = unsigned.pop("content_sha256", None)
    if recorded != canonical_sha256(unsigned):
        raise C1PilotError(f"{label} content hash differs")
    return value


def _verified_record(record: Mapping[str, Any], *, label: str) -> Path:
    if not isinstance(record, Mapping):
        raise C1PilotError(f"{label} is not an artifact record")
    path = Path(str(record.get("path", ""))).expanduser().resolve()
    if dict(record) != artifact_record(path):
        raise C1PilotError(f"{label} artifact differs")
    return path


def _ids(frame: pd.DataFrame, *, label: str) -> list[str]:
    if "sample_id" not in frame or frame["sample_id"].isna().any():
        raise C1PilotError(f"{label} lacks sample IDs")
    values = frame["sample_id"].astype(str).tolist()
    if any(not value for value in values) or len(values) != len(set(values)):
        raise C1PilotError(f"{label} sample IDs are empty or duplicated")
    return values


def _predicted_taxonomy_path(baseline_manifest_path: Path) -> Path:
    baseline = _self_hashed(baseline_manifest_path, label="baseline replay manifest")
    reconciliation_path = _verified_record(
        baseline.get("derived_reconciliation", {}),
        label="baseline derived reconciliation",
    )
    reconciliation = _self_hashed(
        reconciliation_path, label="baseline derived reconciliation"
    )
    source = reconciliation.get("source_artifacts", {}).get(
        "unified_full_pool_taxonomy"
    )
    return _verified_record(source, label="locked predicted full-pool taxonomy")


def _target_quartiles(frame: pd.DataFrame) -> dict[str, str]:
    """Assign four deterministic rank quartiles, breaking size ties by SHA."""

    if frame["foreground_fraction"].isna().any():
        raise C1PilotError("evaluable pilot rows lack target-size covariates")
    work = frame[["sample_id", "foreground_fraction"]].copy()
    work["_tie"] = work["sample_id"].astype(str).map(
        lambda value: _sha(f"{PILOT_SALT}|quartile|{value}")
    )
    work = work.sort_values(
        ["foreground_fraction", "_tie", "sample_id"], kind="mergesort"
    ).reset_index(drop=True)
    count = len(work)
    if count < 4:
        raise C1PilotError("target-size quartiles require at least four samples")
    return {
        str(row.sample_id): f"Q{min(4, index * 4 // count + 1)}"
        for index, row in enumerate(work.itertuples(index=False))
    }


def _predicted_class(row: Any) -> str:
    if int(row.candidate_count_all) == 0:
        return "predicted_no_output"
    if bool(row.native_correct):
        return "predicted_success"
    if not bool(row.full_pool_positive):
        return "predicted_no_positive"
    return "predicted_ranking_limited"


def _balanced_selection(frame: pd.DataFrame, *, size: int) -> pd.DataFrame:
    """Round-robin joint strata; SHA-256 resolves every ordering tie."""

    if size <= 0 or len(frame) < size:
        raise C1PilotError("pilot size is invalid for the executable population")
    strata: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in frame.to_dict("records"):
        key = (
            str(row["query_type"]),
            str(row["target_size_quartile"]),
            str(row["predicted_outcome_class"]),
        )
        strata[key].append(row)
    ordered: list[tuple[str, tuple[str, str, str], list[dict[str, Any]]]] = []
    for key, rows in strata.items():
        rows.sort(key=lambda row: (str(row["selection_sha256"]), row["sample_id"]))
        ordered.append((_sha(f"{PILOT_SALT}|stratum|{'|'.join(key)}"), key, rows))
    ordered.sort(key=lambda item: (item[0], item[1]))
    selected: list[dict[str, Any]] = []
    round_index = 0
    while len(selected) < size:
        progressed = False
        for _, _, rows in ordered:
            if round_index < len(rows) and len(selected) < size:
                selected.append(rows[round_index])
                progressed = True
        if not progressed:
            break
        round_index += 1
    if len(selected) != size:
        raise C1PilotError("joint-stratum round robin did not fill the pilot")
    result = pd.DataFrame(selected).reset_index(drop=True)
    result.insert(0, "selection_rank", range(1, len(result) + 1))
    return result


def build_c1_pilot_source_adapter(
    *,
    run_dir: str | Path,
    full_adapter_manifest: str | Path,
    sample_manifest_path: str | Path,
    registry_path: str | Path,
    baseline_manifest_path: str | Path,
    pilot_size: int = PILOT_SIZE,
    resume: bool = False,
) -> Path:
    """Build the pre-lock, content-addressed 200-ID C1 audit source view."""

    root = Path(run_dir).expanduser().resolve()
    full_path = Path(full_adapter_manifest).expanduser().resolve()
    full = verify_g1_c1_source_adapter(full_path)
    sample_path = Path(sample_manifest_path).expanduser().resolve()
    registry = Path(registry_path).expanduser().resolve()
    baseline_path = Path(baseline_manifest_path).expanduser().resolve()
    predicted_path = _predicted_taxonomy_path(baseline_path)
    source_records = {
        "full_execution_adapter": artifact_record(full_path),
        "sample_manifest": artifact_record(sample_path),
        "gt_mapping_registry": artifact_record(registry),
        "baseline_replay": artifact_record(baseline_path),
        "predicted_full_pool_taxonomy": artifact_record(predicted_path),
    }

    deployment = pd.read_parquet(full["test_samples"]["path"])
    labels = pd.read_parquet(full["test_labels_projection"]["path"])
    samples = pd.read_parquet(sample_path, columns=["sample_id", "query_type"])
    mapping = pd.read_parquet(
        registry,
        columns=[
            "sample_id",
            "mapping_status",
            "pixel_qa_status",
            "bulk_gt_pixels_read",
            "foreground_fraction",
        ],
    )
    predicted = pd.read_parquet(
        predicted_path,
        columns=[
            "sample_id",
            "route",
            "candidate_count_all",
            "native_correct",
            "full_pool_positive",
        ],
    )
    predicted = predicted.loc[
        predicted["route"].astype(str).str.lower().eq("c1")
    ].drop(columns="route")
    for name, frame in (
        ("full adapter deployment", deployment),
        ("full adapter labels", labels),
        ("counterfactual manifest", samples),
        ("P2 registry", mapping),
        ("predicted full-pool outcomes", predicted),
    ):
        _ids(frame, label=name)
    executable = set(deployment["sample_id"].astype(str))
    population = (
        samples.merge(mapping, on="sample_id", validate="one_to_one")
        .merge(predicted, on="sample_id", validate="one_to_one")
        .loc[lambda value: value["sample_id"].astype(str).isin(executable)]
        .copy()
    )
    if set(population["sample_id"].astype(str)) != executable:
        raise C1PilotError("pilot covariates do not exactly cover executable IDs")
    if not (
        population["mapping_status"].eq("PASS")
        & population["pixel_qa_status"].eq("P2_MAPPING_QA_PASS")
        & population["bulk_gt_pixels_read"].astype(bool)
    ).all():
        raise C1PilotError("pilot population includes a non-executable P2 row")
    quartiles = _target_quartiles(population)
    population["target_size_quartile"] = population["sample_id"].map(quartiles)
    population["predicted_outcome_class"] = [
        _predicted_class(row) for row in population.itertuples(index=False)
    ]
    population["selection_sha256"] = population["sample_id"].astype(str).map(
        lambda value: _sha(f"{PILOT_SALT}|sample|{value}")
    )
    selection = _balanced_selection(population, size=pilot_size)
    selected_ids = selection["sample_id"].astype(str).tolist()
    for required in PREDICTED_CLASSES:
        if required not in set(selection["predicted_outcome_class"]):
            raise C1PilotError(f"pilot cannot cover predicted class: {required}")
    if set(selection["target_size_quartile"]) != {"Q1", "Q2", "Q3", "Q4"}:
        raise C1PilotError("pilot does not cover all target-size quartiles")
    if set(selection["query_type"]) != set(population["query_type"]):
        raise C1PilotError("pilot does not cover every query type")

    identity = {
        "schema_version": 1,
        "selection_rule": "joint_stratum_round_robin_sha256_v1",
        "selection_salt": PILOT_SALT,
        "pilot_size": pilot_size,
        "ordered_selected_ids_sha256": canonical_sha256(selected_ids),
        "source_artifacts": source_records,
    }
    destination = root / "05_gtmask_inputs/c1_pilot_source_adapter" / canonical_sha256(
        identity
    )[:24]
    manifest_path = destination / MANIFEST_NAME
    if manifest_path.exists():
        if not resume:
            raise FileExistsError(f"C1 pilot adapter exists; pass --resume: {manifest_path}")
        verify_c1_pilot_source_adapter(manifest_path, expected_size=pilot_size)
        return manifest_path

    selected_set = set(selected_ids)
    deployment_output = deployment.loc[
        deployment["sample_id"].astype(str).isin(selected_set)
    ].copy()
    source_order = deployment_output["sample_id"].astype(str).tolist()
    labels_by_id = labels.set_index(labels["sample_id"].astype(str), drop=False)
    label_output = labels_by_id.loc[source_order].reset_index(drop=True)
    sample_output = atomic_parquet(
        deployment_output, destination / "manifests/test_samples.parquet"
    )
    label_path = atomic_parquet(
        label_output.loc[:, list(LABEL_PROJECTION)],
        destination / "manifests/test_labels.parquet",
    )
    config_source = Path(full["selected_configs"]["c1"]["path"])
    config_output = atomic_copy(
        config_source, destination / "selected_configs/C1.json"
    )
    selection_columns = [
        "selection_rank",
        "sample_id",
        "query_type",
        "target_size_quartile",
        "predicted_outcome_class",
        "selection_sha256",
    ]
    selection_path = atomic_parquet(
        selection.loc[:, selection_columns], destination / "pilot_selection.parquet"
    )
    manifest: dict[str, Any] = {
        **identity,
        "status": "COMPLETE",
        "scientific_role": "pre-lock outcome-blind C1 retrospective audit pilot",
        "counterfactual_outcomes_read": 0,
        "gt_candidate_generation_performed": False,
        "source_population_count": len(population),
        "test_samples": artifact_record(sample_output),
        "test_labels_projection": artifact_record(label_path),
        "selected_config": artifact_record(config_output),
        "selection": artifact_record(selection_path),
        "source_ordered_ids_sha256": canonical_sha256(source_order),
        "query_type_counts": dict(
            sorted(Counter(map(str, selection["query_type"])).items())
        ),
        "target_size_quartile_counts": dict(
            sorted(Counter(map(str, selection["target_size_quartile"])).items())
        ),
        "predicted_outcome_counts": dict(
            sorted(Counter(map(str, selection["predicted_outcome_class"])).items())
        ),
        "label_projection_columns": list(LABEL_PROJECTION),
    }
    manifest["content_sha256"] = canonical_sha256(manifest)
    atomic_json(manifest_path, manifest)
    verify_c1_pilot_source_adapter(manifest_path, expected_size=pilot_size)
    return manifest_path


def verify_c1_pilot_source_adapter(
    manifest_path: str | Path, *, expected_size: int = PILOT_SIZE
) -> dict[str, Any]:
    """Recompute pilot membership and all source-byte bindings."""

    path = Path(manifest_path).expanduser().resolve()
    value = _self_hashed(path, label="C1 pilot source adapter")
    if (
        value.get("status") != "COMPLETE"
        or int(value.get("pilot_size", -1)) != expected_size
        or value.get("selection_rule") != "joint_stratum_round_robin_sha256_v1"
        or value.get("selection_salt") != PILOT_SALT
        or value.get("counterfactual_outcomes_read") != 0
        or value.get("gt_candidate_generation_performed") is not False
        or value.get("label_projection_columns") != list(LABEL_PROJECTION)
    ):
        raise C1PilotError("C1 pilot declaration differs")
    records = value.get("source_artifacts")
    if not isinstance(records, Mapping):
        raise C1PilotError("C1 pilot source inventory is malformed")
    for name, record in {
        **dict(records),
        "test_samples": value.get("test_samples"),
        "test_labels_projection": value.get("test_labels_projection"),
        "selected_config": value.get("selected_config"),
        "selection": value.get("selection"),
    }.items():
        _verified_record(record, label=f"C1 pilot {name}")
    full_path = Path(str(records["full_execution_adapter"]["path"]))
    full = verify_g1_c1_source_adapter(full_path)
    pilot_config = Path(str(value["selected_config"]["path"])).resolve()
    full_config = Path(str(full["selected_configs"]["c1"]["path"])).resolve()
    if (
        pilot_config != path.parent / "selected_configs/C1.json"
        or value["selected_config"]["sha256"]
        != full["selected_configs"]["c1"]["sha256"]
        or pilot_config.read_bytes() != full_config.read_bytes()
    ):
        raise C1PilotError("C1 pilot selected config differs from full adapter")
    selection = pd.read_parquet(value["selection"]["path"])
    samples = pd.read_parquet(value["test_samples"]["path"])
    labels = pd.read_parquet(value["test_labels_projection"]["path"])
    selected_ids = _ids(selection, label="C1 pilot selection")
    sample_ids = _ids(samples, label="C1 pilot samples")
    label_ids = _ids(labels, label="C1 pilot labels")
    if (
        len(selected_ids) != expected_size
        or set(selected_ids) != set(sample_ids)
        or sample_ids != label_ids
        or list(labels.columns) != list(LABEL_PROJECTION)
        or selection["selection_rank"].astype(int).tolist()
        != list(range(1, expected_size + 1))
        or value.get("ordered_selected_ids_sha256")
        != canonical_sha256(selected_ids)
        or value.get("source_ordered_ids_sha256") != canonical_sha256(sample_ids)
        or set(selection["target_size_quartile"]) != {"Q1", "Q2", "Q3", "Q4"}
        or set(PREDICTED_CLASSES).difference(selection["predicted_outcome_class"])
    ):
        raise C1PilotError("C1 pilot membership/coverage differs")
    expected_identity = {
        "schema_version": 1,
        "selection_rule": value["selection_rule"],
        "selection_salt": value["selection_salt"],
        "pilot_size": expected_size,
        "ordered_selected_ids_sha256": canonical_sha256(selected_ids),
        "source_artifacts": dict(records),
    }
    if path.parent.name != canonical_sha256(expected_identity)[:24]:
        raise C1PilotError("C1 pilot content-addressed directory differs")
    return value


__all__ = [
    "C1PilotError",
    "MANIFEST_NAME",
    "PILOT_SALT",
    "PILOT_SIZE",
    "PREDICTED_CLASSES",
    "build_c1_pilot_source_adapter",
    "verify_c1_pilot_source_adapter",
]
