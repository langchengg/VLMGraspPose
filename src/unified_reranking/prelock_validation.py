"""Semantic verification for development evidence frozen by P11.

These checks deliberately operate only on Train/Validation artifacts and
label-free Test application manifests.  They have no candidate-Test-label
reader and are shared by prelock assembly and formal-lock creation.
"""

from __future__ import annotations

import json
import math
import pickle
import re
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .artifacts import verify_artifact_records_recursive, verified_artifact_path
from .ablation import ablation_feature_sets
from .hashing import canonical_sha256, sha256_file
from .metrics import compare_selections, evaluate_order_only, select_order_only
from .gate import (
    ConservativeTransitionModel,
    GateEvidence,
    GateOperatingPoint,
    OOFTransitionData,
    gate_switch_mask,
    select_gate_operating_point,
)
from .route_router import (
    CROGDefaultTransitionRouter,
    RouterEvidence,
    RouterOperatingPoint,
    route_decisions,
    route_utilities,
    select_router_operating_point,
)
from .telemetry import TELEMETRY_FIELDS, resolved_track_extraction_latency


ROUTES = ("crog", "g1", "c1")
FORMAL_SEEDS = (42, 123, 2026)
PRIMARY_TRACK = "T2_matched_common"
HEADROOM_THRESHOLD = 0.0025
FEATURE_BENCHMARK_COMPONENTS = (
    "common/crog",
    "rgb/crog",
    "T1_native/crog",
    "T2_matched_common/crog",
    "common/g1",
    "rgb/g1",
    "T1_native/g1",
    "T2_matched_common/g1",
    "common/c1",
    "rgb/c1",
    "T1_native/c1",
    "T2_matched_common/c1",
    "backend_maps/g1",
    "backend_maps/c1",
)


def _load(path: Path, name: str, statuses: Sequence[str]) -> dict[str, Any]:
    path = path.resolve()
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{name} is not a regular file: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("status") not in set(statuses):
        raise RuntimeError(f"{name} has invalid status")
    return value


def _record(path: Path) -> dict[str, str]:
    path = path.resolve()
    return {"path": str(path), "sha256": sha256_file(path)}


def _same_record(observed: Any, expected: Mapping[str, Any], name: str) -> None:
    if not isinstance(observed, Mapping):
        raise ValueError(f"{name} is not an artifact record")
    normalized = {
        "path": str(Path(str(observed.get("path", ""))).resolve()),
        "sha256": observed.get("sha256"),
    }
    if normalized != dict(expected):
        raise ValueError(f"{name} binding mismatch")
    verified_artifact_path(observed, name=name)


def _finite(value: Any, name: str, *, minimum: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} is not numeric") from error
    if not math.isfinite(number) or number < minimum:
        raise ValueError(f"{name} must be finite and >= {minimum}")
    return number


def _validate_telemetry(cell: Mapping[str, Any], name: str) -> None:
    telemetry = cell.get("telemetry")
    if not isinstance(telemetry, Mapping):
        raise ValueError(f"{name} lacks complete telemetry")
    if set(telemetry.get("measurement_protocols", {})) != set(TELEMETRY_FIELDS):
        raise ValueError(f"{name} telemetry measurement protocols are incomplete")
    not_applicable = set(map(str, telemetry.get("not_applicable_fields", [])))
    if not not_applicable.issubset(TELEMETRY_FIELDS):
        raise ValueError(f"{name} telemetry has unknown not-applicable fields")
    for field in TELEMETRY_FIELDS:
        value = telemetry.get(field)
        if field in not_applicable:
            if value is not None:
                raise ValueError(f"{name} telemetry {field} is inconsistently applicable")
        else:
            number = _finite(value, f"{name} telemetry {field}")
            if field == "missing_feature_rate" and number > 1.0:
                raise ValueError(f"{name} missing_feature_rate must be <= 1")
        if cell.get(field) != value:
            raise ValueError(f"{name} flattened telemetry differs for {field}")


def _validate_cell(record: Mapping[str, Any], name: str) -> tuple[dict[str, Any], Path]:
    path = verified_artifact_path(record, name=name)
    cell = _load(path, name, ("COMPLETE",))
    verify_artifact_records_recursive(
        {"sources": cell.get("sources"), "artifacts": cell.get("artifacts")},
        name=name,
        require_at_least_one=True,
    )
    _validate_telemetry(cell, name)
    metrics = cell.get("metrics")
    if not isinstance(metrics, Mapping):
        raise ValueError(f"{name} lacks metrics")
    for metric in ("j_at_1", "mrr_at_5"):
        _finite(metrics.get(metric), f"{name} {metric}")
    return cell, path


def _command_options(command: Any) -> dict[str, str]:
    if not isinstance(command, list) or "tools.unified_reranking.train_matrix_cell" not in command:
        raise ValueError("encoder plan contains a non-matrix command")
    options: dict[str, str] = {}
    index = command.index("tools.unified_reranking.train_matrix_cell") + 1
    while index < len(command):
        token = str(command[index])
        if not token.startswith("--") or index + 1 >= len(command):
            raise ValueError("encoder plan command has invalid option syntax")
        key = token[2:].replace("-", "_")
        if key in options:
            raise ValueError("encoder plan command repeats an option")
        options[key] = str(command[index + 1])
        index += 2
    return options


def _configuration_matches_command(
    configuration: Mapping[str, Any], command: Any
) -> bool:
    options = _command_options(command)
    required = {"run_dir", "route", "track", "encoder", "loss", "seed", "mode"}
    if not required.issubset(options):
        return False
    for key, raw in options.items():
        if key == "run_dir":
            continue
        config_key = "held_fold" if key == "fold" else key
        observed = configuration.get(config_key)
        if isinstance(observed, bool):
            expected: Any = raw.lower() in {"true", "1"}
        elif isinstance(observed, int):
            expected = int(raw)
        elif isinstance(observed, float):
            expected = float(raw)
        else:
            expected = raw
        if observed != expected:
            return False
    if "fold" not in options and configuration.get("held_fold") is not None:
        return False
    return True


def validate_encoder_execution(
    path: Path, *, expected_selection: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    """Verify one complete encoder-phase execution and every output cell."""

    execution = _load(path, "encoder latest execution", ("COMPLETE",))
    if execution.get("phase") != "encoder":
        raise ValueError("encoder latest execution has the wrong phase")
    plan_record = execution.get("plan")
    if not isinstance(plan_record, Mapping):
        raise ValueError("encoder latest execution lacks its plan record")
    plan_path = verified_artifact_path(plan_record, name="encoder execution plan")
    plan = _load(plan_path, "encoder execution plan", ("PLANNED",))
    if plan.get("phase") != "encoder":
        raise ValueError("encoder execution plan has the wrong phase")
    if execution.get("selection") != plan.get("selection"):
        raise ValueError("encoder execution and plan selection bindings differ")
    selection = execution.get("selection")
    if not isinstance(selection, Mapping):
        raise ValueError("encoder execution lacks a selection binding")
    verified_artifact_path(selection, name="encoder loss selection")
    if expected_selection is not None:
        _same_record(selection, expected_selection, "encoder loss selection")
    jobs = plan.get("jobs")
    results = execution.get("results")
    outputs = execution.get("output_manifests")
    job_count = int(execution.get("job_count", -1))
    if (
        not isinstance(jobs, list)
        or not isinstance(results, list)
        or not isinstance(outputs, list)
        or job_count <= 0
        or int(plan.get("job_count", -1)) != job_count
        or len(jobs) != job_count
        or len(results) != job_count
        or len(outputs) != job_count
    ):
        raise ValueError("encoder execution does not exactly cover its planned jobs")
    selection_payload = json.loads(
        Path(str(selection["path"])).read_text(encoding="utf-8")
    )
    choices = selection_payload.get("selections", selection_payload)
    expected_keys = {
        f"{route}/{track}"
        for route in map(str, plan.get("routes", []))
        for track in map(str, plan.get("tracks", []))
    }
    if not isinstance(choices, Mapping) or set(choices) != expected_keys:
        raise ValueError("encoder loss selections do not exactly cover plan routes/tracks")
    expected_configurations: set[tuple[Any, ...]] = set()
    for key in sorted(expected_keys):
        route, track = key.split("/", 1)
        choice = choices[key]
        if not isinstance(choice, Mapping):
            raise ValueError("encoder phase requires one locked choice per route/track")
        loss = str(choice.get("loss", ""))
        parameters = dict(choice.get("parameters", {}))
        for encoder in ("mlp", "deepsets", "set_transformer", "gnn"):
            blocks = (1, 2) if encoder == "set_transformer" else (
                int(parameters.get("num_attention_blocks", 2)),
            )
            for block_count in blocks:
                budget = {**parameters, "num_attention_blocks": block_count}
                budget_key = canonical_sha256(budget)
                for seed in FORMAL_SEEDS:
                    for mode, folds in (("oof", range(5)), ("validation", (None,))):
                        for fold in folds:
                            expected_configurations.add(
                                (
                                    route,
                                    track,
                                    encoder,
                                    loss,
                                    seed,
                                    mode,
                                    fold,
                                    budget_key,
                                )
                            )
    observed_configurations: set[tuple[Any, ...]] = set()
    for job in jobs:
        options = _command_options(job.get("command"))
        budget = {
            key: value
            for key, value in options.items()
            if key
            not in {
                "run_dir",
                "route",
                "track",
                "encoder",
                "loss",
                "seed",
                "mode",
                "fold",
            }
        }
        # Command strings are the producer's canonical serialization.  Convert
        # them to the selected parameter types before comparing the budget.
        choice_budget = dict(choices[f"{options['route']}/{options['track']}"].get("parameters", {}))
        typed_budget: dict[str, Any] = {}
        for key, raw in budget.items():
            reference = choice_budget.get(key)
            if key == "num_attention_blocks":
                typed_budget[key] = int(raw)
            elif isinstance(reference, int):
                typed_budget[key] = int(raw)
            elif isinstance(reference, float):
                typed_budget[key] = float(raw)
            else:
                typed_budget[key] = raw
        observed_configurations.add(
            (
                options["route"],
                options["track"],
                options["encoder"],
                options["loss"],
                int(options["seed"]),
                options["mode"],
                None if "fold" not in options else int(options["fold"]),
                canonical_sha256(typed_budget),
            )
        )
    if observed_configurations != expected_configurations or len(jobs) != len(
        expected_configurations
    ):
        raise ValueError("encoder plan does not exactly cover the selected encoder grid")
    job_ids = [str(job.get("identifier", "")) for job in jobs if isinstance(job, Mapping)]
    result_ids = [
        str(result.get("identifier", ""))
        for result in results
        if isinstance(result, Mapping)
    ]
    if (
        len(job_ids) != job_count
        or len(set(job_ids)) != job_count
        or sorted(job_ids) != sorted(result_ids)
        or any(int(result.get("returncode", -1)) != 0 for result in results)
    ):
        raise ValueError("encoder execution result inventory differs from the plan")
    result_outputs = [result.get("output_manifest") for result in results]
    if result_outputs != outputs or len(
        {str(record.get("path", "")) for record in outputs if isinstance(record, Mapping)}
    ) != job_count:
        raise ValueError("encoder output-manifest inventory is not exact")
    cells = []
    for index, (result, record) in enumerate(zip(results, outputs, strict=True)):
        cell = _validate_cell(record, f"encoder cell {index}")[0]
        job = next(job for job in jobs if job["identifier"] == result["identifier"])
        command = job.get("command")
        if canonical_sha256(tuple(command))[:16] != str(job["identifier"]):
            raise ValueError("encoder plan job identifier differs from its command")
        if not _configuration_matches_command(cell.get("configuration", {}), command):
            raise ValueError("encoder output configuration differs from its planned command")
        cells.append(cell)
    plan_routes = set(map(str, plan.get("routes", [])))
    plan_tracks = set(map(str, plan.get("tracks", [])))
    for cell in cells:
        config = cell.get("configuration", {})
        if (
            config.get("route") not in plan_routes
            or config.get("track") not in plan_tracks
            or config.get("encoder") not in {"mlp", "deepsets", "set_transformer", "gnn"}
            or config.get("mode") not in {"oof", "validation"}
        ):
            raise ValueError("encoder cell configuration is outside the frozen plan")
    return execution


def validate_feature_ablation_manifest(
    path: Path, *, expected_selection: Mapping[str, Any]
) -> dict[str, Any]:
    """Verify the two ablation CSVs and their complete cell/telemetry inventory."""

    path = path.resolve()
    run_dir = path.parents[2]
    manifest = _load(path, "feature ablation manifest", ("COMPLETE",))
    selection_path = verified_artifact_path(
        expected_selection, name="feature ablation primary selection"
    )
    selection = _load(
        selection_path, "feature ablation primary selection", ("VALIDATION_LOCKED",)
    )
    selections = selection.get("selections")
    if not isinstance(selections, Mapping) or set(selections) != set(ROUTES):
        raise ValueError("feature ablation primary selection is incomplete")
    from tools.unified_reranking import run_validation_feature_ablations as producer

    current_tool_record = _record(Path(producer.__file__))
    benchmark_path = (
        run_dir / "07_validation/telemetry/feature_extraction_benchmark.json"
    )
    validate_feature_extraction_benchmark(benchmark_path)
    benchmark_record = _record(benchmark_path)
    if (
        manifest.get("analysis") != "validation_feature_family_ablation"
        or manifest.get("split") != "validation"
        or manifest.get("candidate_test_labels_read") is not False
    ):
        raise ValueError("feature ablation is not a Validation-only label-free contract")
    if manifest.get("selection_sha256") != expected_selection.get("sha256"):
        raise ValueError("feature ablation selection hash mismatch")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, Mapping) or set(artifacts) != {
        "cumulative",
        "leave_one_family_out",
        "cell_manifests",
    }:
        raise ValueError("feature ablation artifact inventory is not exact")
    cumulative_path = verified_artifact_path(
        artifacts["cumulative"], name="cumulative feature ablation CSV"
    )
    leave_one_path = verified_artifact_path(
        artifacts["leave_one_family_out"], name="leave-one-family-out ablation CSV"
    )
    expected_rows = manifest.get("expected_rows_by_route")
    manifest_sources = manifest.get("sources")
    if not isinstance(manifest_sources, Mapping) or set(manifest_sources) != {
        "routes",
        "tool",
    }:
        raise ValueError("feature ablation source inventory is not exact")
    _same_record(
        manifest_sources.get("tool"), current_tool_record, "feature ablation producer"
    )
    route_sources = manifest_sources.get("routes")
    if not isinstance(expected_rows, Mapping) or set(expected_rows) != set(ROUTES):
        raise ValueError("feature ablation expected-row inventory is incomplete")
    if not isinstance(route_sources, Mapping) or set(route_sources) != set(ROUTES):
        raise ValueError("feature ablation route-source inventory is incomplete")
    all_records = artifacts["cell_manifests"]
    if not isinstance(all_records, list):
        raise ValueError("feature ablation cell manifest inventory is invalid")
    record_by_path: dict[str, Mapping[str, Any]] = {}
    cell_by_path: dict[str, Mapping[str, Any]] = {}
    for index, record in enumerate(all_records):
        _cell, cell_path = _validate_cell(record, f"feature ablation cell {index}")
        _validate_cell_extraction_binding(
            _cell,
            run_dir=run_dir,
            expected_t3_record=benchmark_record,
            name=f"feature ablation cell {index}",
        )
        _replay_cell_metrics(run_dir, _cell, f"feature ablation cell {index}")
        if str(cell_path) in record_by_path:
            raise ValueError("feature ablation cell manifest inventory has duplicates")
        record_by_path[str(cell_path)] = record
        cell_by_path[str(cell_path)] = _cell
    seen_paths: set[str] = set()
    for ablation_type, csv_path in (
        ("cumulative", cumulative_path),
        ("leave_one_family_out", leave_one_path),
    ):
        frame = pd.read_csv(csv_path)
        required = {
            "route",
            "track",
            "ablation",
            "ablation_type",
            "family",
            "family_order",
            "seed_count",
            "seeds",
            "selected_encoder",
            "selected_loss",
            "fixed_budget_json",
            "fixed_budget_sha256",
            "selected_source_signature_sha256",
            "included_families_json",
            "included_features_json",
            "included_feature_count",
            "excluded_feature_count",
            "j_at_1",
            "j_at_1_std",
            "mrr_at_5",
            "mrr_at_5_std",
            "delta_j_at_1",
            "cell_manifest_paths_json",
            "cell_manifest_sha256s_json",
            "candidate_test_labels_read",
            "split",
        }
        if frame.empty or required.difference(frame.columns):
            raise ValueError(f"{ablation_type} ablation CSV schema is incomplete")
        for route in ROUTES:
            rows = frame.loc[frame["route"].astype(str).eq(route)]
            source_columns = route_sources[route].get("feature_columns")
            if not isinstance(source_columns, list):
                raise ValueError("feature ablation source lacks its feature schema")
            cumulative_specs, leave_specs = ablation_feature_sets(source_columns)
            independently_expected = {
                "cumulative": cumulative_specs,
                "leave_one_family_out": leave_specs,
            }[ablation_type]
            if int(expected_rows[route][ablation_type]) != len(
                independently_expected
            ):
                raise ValueError("feature ablation expected rows differ from frozen schema")
            if len(rows) != int(expected_rows[route][ablation_type]):
                raise ValueError(f"{route} {ablation_type} ablation row count mismatch")
            source_contract = route_sources[route]
            if not isinstance(source_contract, Mapping) or set(source_contract) != {
                "selection",
                "ensemble",
                "selected_cells",
                "feature_columns",
                "feature_schema_sha256",
                "tool",
            }:
                raise ValueError("feature ablation route-source contract is not exact")
            verify_artifact_records_recursive(
                source_contract,
                name=f"{route} ablation source contract",
                require_at_least_one=True,
            )
            choice = selections[route]
            expected_ensemble_record = {
                "path": str(choice.get("validation_manifest", "")),
                "sha256": str(choice.get("validation_manifest_sha256", "")),
            }
            _same_record(
                source_contract.get("selection"),
                _record(selection_path),
                f"{route} ablation primary selection",
            )
            _same_record(
                source_contract.get("ensemble"),
                {
                    "path": str(Path(expected_ensemble_record["path"]).resolve()),
                    "sha256": expected_ensemble_record["sha256"],
                },
                f"{route} ablation selected ensemble",
            )
            _same_record(
                source_contract.get("tool"),
                current_tool_record,
                f"{route} ablation producer",
            )
            ensemble_path = verified_artifact_path(
                expected_ensemble_record, name=f"{route} ablation selected ensemble"
            )
            ensemble = _load(
                ensemble_path, f"{route} ablation selected ensemble", ("COMPLETE",)
            )
            selected_records = source_contract.get("selected_cells")
            ensemble_records = ensemble.get("sources", {}).get("matrix_manifests")
            if (
                not isinstance(selected_records, list)
                or not isinstance(ensemble_records, list)
                or selected_records != ensemble_records
                or len(selected_records) != len(FORMAL_SEEDS)
            ):
                raise ValueError("feature ablation selected-cell lineage is incomplete")
            baseline_by_seed: dict[int, float] = {}
            expected_budget: dict[str, Any] | None = None
            selected_feature_columns = list(map(str, source_columns))
            for seed, record in zip(FORMAL_SEEDS, selected_records, strict=True):
                selected_cell, _ = _validate_cell(
                    record, f"{route} ablation selected seed {seed}"
                )
                _validate_cell_extraction_binding(
                    selected_cell,
                    run_dir=run_dir,
                    expected_t3_record=benchmark_record,
                    name=f"{route} ablation selected seed {seed}",
                )
                _replay_cell_metrics(
                    run_dir, selected_cell, f"{route} ablation selected seed {seed}"
                )
                configuration = selected_cell.get("configuration", {})
                if (
                    int(configuration.get("seed", -1)) != seed
                    or configuration.get("route") != route
                    or configuration.get("mode") != "validation"
                    or configuration.get("track") != choice.get("primary_track")
                    or configuration.get("encoder") != choice.get("encoder")
                    or configuration.get("loss") != choice.get("loss")
                    or list(map(str, selected_cell.get("feature_columns", [])))
                    != selected_feature_columns
                ):
                    raise ValueError("feature ablation selected cell differs from locked choice")
                budget = {
                    field: configuration.get(field, producer._DEFAULTS[field])
                    for field in producer._BUDGET_FIELDS
                }
                if expected_budget is None:
                    expected_budget = budget
                elif budget != expected_budget:
                    raise ValueError("feature ablation selected cells use different budgets")
                baseline_by_seed[seed] = float(selected_cell["metrics"]["j_at_1"])
            if list(map(str, source_contract.get("feature_columns", []))) != selected_feature_columns:
                raise ValueError("feature ablation source feature schema mismatch")
            if source_contract.get("feature_schema_sha256") != canonical_sha256(
                tuple(selected_feature_columns)
            ):
                raise ValueError("feature ablation feature-schema hash mismatch")
            signature = canonical_sha256(source_contract)
            if set(rows["family"].astype(str)) != {
                str(spec["family"]) for spec in independently_expected
            }:
                raise ValueError("feature ablation families differ from frozen schema")
            for row in rows.itertuples(index=False):
                if (
                    str(row.ablation_type) != ablation_type
                    or int(row.seed_count) != len(FORMAL_SEEDS)
                    or json.loads(str(row.seeds)) != list(FORMAL_SEEDS)
                    or str(row.selected_source_signature_sha256) != signature
                    or bool(row.candidate_test_labels_read)
                    or str(row.split) != "validation"
                ):
                    raise ValueError("feature ablation row contract mismatch")
                for metric in (
                    "j_at_1",
                    "j_at_1_std",
                    "mrr_at_5",
                    "mrr_at_5_std",
                    "delta_j_at_1",
                ):
                    if not math.isfinite(float(getattr(row, metric))):
                        raise ValueError("feature ablation CSV contains non-finite metrics")
                paths = list(map(str, json.loads(str(row.cell_manifest_paths_json))))
                digests = list(map(str, json.loads(str(row.cell_manifest_sha256s_json))))
                if len(paths) != len(FORMAL_SEEDS) or len(digests) != len(paths):
                    raise ValueError("feature ablation row lacks the formal seed cells")
                matching_specs = [
                    spec
                    for spec in independently_expected
                    if str(spec["family"]) == str(row.family)
                ]
                if len(matching_specs) != 1:
                    raise ValueError("feature ablation row has no unique frozen specification")
                spec = matching_specs[0]
                included_features = list(map(str, spec["included_features"]))
                expected_budget = expected_budget or {}
                expected_strings = {
                    "track": str(choice.get("primary_track")),
                    "ablation": f"{ablation_type}:{spec['family']}",
                    "selected_encoder": str(choice.get("encoder")),
                    "selected_loss": str(choice.get("loss")),
                    "fixed_budget_json": json.dumps(
                        expected_budget, sort_keys=True, separators=(",", ":")
                    ),
                    "fixed_budget_sha256": canonical_sha256(expected_budget),
                    "included_families_json": json.dumps(
                        spec["included_families"], separators=(",", ":")
                    ),
                    "included_features_json": json.dumps(
                        included_features, separators=(",", ":")
                    ),
                }
                for field, expected in expected_strings.items():
                    if str(getattr(row, field)) != expected:
                        raise ValueError(f"feature ablation row {field} mismatch")
                if (
                    int(row.family_order) != int(spec["family_order"])
                    or int(row.included_feature_count) != len(included_features)
                    or int(row.excluded_feature_count)
                    != len(selected_feature_columns) - len(included_features)
                ):
                    raise ValueError("feature ablation row feature counts mismatch")
                replayed_j: list[float] = []
                replayed_mrr: list[float] = []
                replayed_baseline: list[float] = []
                for seed, child_path, digest in zip(
                    FORMAL_SEEDS, paths, digests, strict=True
                ):
                    resolved = str(Path(child_path).resolve())
                    record = record_by_path.get(resolved)
                    if record is None or record.get("sha256") != digest:
                        raise ValueError("feature ablation CSV cell binding mismatch")
                    child = cell_by_path[resolved]
                    configuration = child.get("configuration", {})
                    analysis_contract = configuration.get("analysis_contract")
                    if (
                        int(configuration.get("seed", -1)) != seed
                        or configuration.get("route") != route
                        or configuration.get("mode") != "validation"
                        or configuration.get("track") != choice.get("primary_track")
                        or configuration.get("encoder") != choice.get("encoder")
                        or configuration.get("loss") != choice.get("loss")
                        or list(map(str, child.get("feature_columns", [])))
                        != included_features
                        or not isinstance(analysis_contract, Mapping)
                        or analysis_contract.get("analysis")
                        != "validation_feature_family_ablation"
                        or analysis_contract.get("split") != "validation"
                        or analysis_contract.get("ablation_type") != ablation_type
                        or analysis_contract.get("family") != spec["family"]
                        or analysis_contract.get("source_signature_sha256") != signature
                        or analysis_contract.get("candidate_test_labels_read") is not False
                    ):
                        raise ValueError("feature ablation child contract mismatch")
                    child_budget = {
                        field: configuration.get(field, producer._DEFAULTS[field])
                        for field in producer._BUDGET_FIELDS
                    }
                    if child_budget != expected_budget:
                        raise ValueError("feature ablation child budget mismatch")
                    replayed_j.append(float(child["metrics"]["j_at_1"]))
                    replayed_mrr.append(float(child["metrics"]["mrr_at_5"]))
                    replayed_baseline.append(baseline_by_seed[seed])
                    seen_paths.add(resolved)
                expected_metrics = {
                    "j_at_1": float(np.mean(replayed_j)),
                    "j_at_1_std": float(np.std(replayed_j, ddof=1)),
                    "mrr_at_5": float(np.mean(replayed_mrr)),
                    "mrr_at_5_std": float(np.std(replayed_mrr, ddof=1)),
                    "delta_j_at_1": float(
                        np.mean(np.asarray(replayed_j) - np.asarray(replayed_baseline))
                    ),
                }
                for metric, expected in expected_metrics.items():
                    if not math.isclose(
                        float(getattr(row, metric)),
                        expected,
                        rel_tol=0.0,
                        abs_tol=1e-12,
                    ):
                        raise ValueError(f"feature ablation aggregate {metric} mismatch")
    if seen_paths != set(record_by_path):
        raise ValueError("feature ablation cell inventory has unreferenced manifests")
    expected_cell_count = sum(
        int(expected_rows[route][kind]) * len(FORMAL_SEEDS)
        for route in ROUTES
        for kind in ("cumulative", "leave_one_family_out")
    )
    if len(record_by_path) != expected_cell_count:
        raise ValueError("feature ablation cell inventory is incomplete")
    content = dict(manifest)
    declared_content_hash = content.pop("content_sha256", None)
    if declared_content_hash != canonical_sha256(content):
        raise ValueError("feature ablation content hash mismatch")
    return manifest


def validate_feature_extraction_benchmark(path: Path) -> dict[str, Any]:
    """Verify the frozen Validation-only T1/T2/T3/component telemetry bundle."""

    benchmark = _load(path, "feature extraction benchmark", ("COMPLETE",))
    expected_keys = {
        "status",
        "schema_version",
        "analysis",
        "source_signature_sha256",
        "configuration",
        "command",
        "feature_extraction_elapsed_seconds",
        "feature_extraction_latency_ms",
        "feature_extraction_latency_ms_per_sample",
        "peak_memory_mb",
        "measurement_protocol",
        "component_measurements",
        "candidate_test_labels_read",
        "sources",
        "artifacts",
        "content_sha256",
    }
    if set(benchmark) != expected_keys:
        raise ValueError("feature extraction benchmark top-level inventory is not exact")
    unsigned = dict(benchmark)
    declared_content_hash = unsigned.pop("content_sha256", None)
    if declared_content_hash != canonical_sha256(unsigned):
        raise ValueError("feature extraction benchmark content hash mismatch")
    if (
        benchmark.get("schema_version") != 1
        or benchmark.get("analysis") != "validation_feature_extraction_runtime"
        or benchmark.get("candidate_test_labels_read") is not False
        or not str(benchmark.get("measurement_protocol", ""))
    ):
        raise ValueError("feature extraction benchmark is not Validation-only")
    configuration = benchmark.get("configuration")
    expected_configuration_keys = {
        "split",
        "sample_limit",
        "tag",
        "device",
        "batch_size",
        "chunk_size",
        "sample_identity_sha256",
        "candidate_rows_by_route",
        "total_candidate_rows",
        "candidate_test_labels_read",
        "component_inventory",
    }
    if not isinstance(configuration, Mapping) or set(configuration) != expected_configuration_keys:
        raise ValueError("feature extraction benchmark configuration is not exact")
    route_rows = configuration.get("candidate_rows_by_route")
    if (
        configuration.get("split") != "validation"
        or configuration.get("sample_limit") != 128
        or configuration.get("tag") != "latency_benchmark_128"
        or configuration.get("device") not in {"cpu", "mps"}
        or not isinstance(configuration.get("batch_size"), int)
        or int(configuration["batch_size"]) <= 0
        or not isinstance(configuration.get("chunk_size"), int)
        or int(configuration["chunk_size"]) <= 0
        or re.fullmatch(
            r"[0-9a-f]{64}", str(configuration.get("sample_identity_sha256", ""))
        )
        is None
        or not isinstance(route_rows, Mapping)
        or set(route_rows) != set(ROUTES)
        or any(not isinstance(value, int) or value <= 0 for value in route_rows.values())
        or configuration.get("total_candidate_rows") != sum(route_rows.values())
        or configuration.get("candidate_test_labels_read") is not False
        or list(configuration.get("component_inventory", []))
        != list(FEATURE_BENCHMARK_COMPONENTS)
    ):
        raise ValueError("feature extraction benchmark frozen configuration mismatch")
    sources = benchmark.get("sources")
    if not isinstance(sources, Mapping) or set(sources) != {
        "paired_validation",
        "candidate_pools",
        "extractor_tool",
        "benchmark_tool",
        "component_manifests",
    }:
        raise ValueError("feature extraction benchmark source inventory is not exact")
    candidate_pools = sources.get("candidate_pools")
    components = sources.get("component_manifests")
    if (
        not isinstance(candidate_pools, Mapping)
        or set(candidate_pools) != set(ROUTES)
        or not isinstance(components, Mapping)
        or set(components) != set(FEATURE_BENCHMARK_COMPONENTS)
    ):
        raise ValueError("feature extraction benchmark nested source inventory is not exact")
    verify_artifact_records_recursive(
        sources,
        name="feature extraction benchmark sources",
        require_at_least_one=True,
    )
    run_dir = path.resolve().parents[2]
    paired_path = run_dir / "01_manifests/paired_validation.parquet"
    _same_record(
        sources["paired_validation"],
        _record(paired_path),
        "feature extraction paired Validation source",
    )
    sample_ids = pd.read_parquet(paired_path, columns=["sample_id"])[
        "sample_id"
    ].astype(str)
    if (
        sample_ids.empty
        or sample_ids.duplicated().any()
        or configuration["sample_identity_sha256"]
        != canonical_sha256(sample_ids.iloc[:128].tolist())
    ):
        raise ValueError("feature extraction benchmark sample identity mismatch")
    for route in ROUTES:
        candidate_path = (
            run_dir / "02_candidates" / f"{route}_validation_top5.parquet"
        )
        _same_record(
            candidate_pools[route],
            _record(candidate_path),
            f"feature extraction {route} candidate pool",
        )
        candidates = pd.read_parquet(
            candidate_path, columns=["sample_id", "candidate_id"]
        )
        candidates[["sample_id", "candidate_id"]] = candidates[
            ["sample_id", "candidate_id"]
        ].astype(str)
        if candidates[["sample_id", "candidate_id"]].duplicated().any():
            raise ValueError(
                f"feature extraction {route} candidate pool has duplicate keys"
            )
        benchmark_sample_ids = set(sample_ids.iloc[:128])
        candidate_rows = int(
            candidates["sample_id"].isin(benchmark_sample_ids).sum()
        )
        if route_rows[route] != candidate_rows:
            raise ValueError(
                f"feature extraction {route} candidate-row count mismatch"
            )
    repository_root = Path(__file__).resolve().parents[2]
    _same_record(
        sources["extractor_tool"],
        _record(
            repository_root
            / "tools/unified_reranking/extract_tri_backend_dense_features.py"
        ),
        "feature extraction implementation tool",
    )
    _same_record(
        sources["benchmark_tool"],
        _record(
            repository_root
            / "tools/unified_reranking/benchmark_feature_extraction_latency.py"
        ),
        "feature extraction benchmark tool",
    )
    if benchmark.get("source_signature_sha256") != canonical_sha256(
        {"configuration": configuration, "sources": sources}
    ):
        raise ValueError("feature extraction benchmark source signature mismatch")
    component_payloads: dict[str, dict[str, Any]] = {}
    for name, record in components.items():
        component, route = name.split("/", 1)
        if component in {"T1_native", "T2_matched_common"}:
            expected_component_path = (
                run_dir
                / "03_features/tracks"
                / component
                / f"{route}_validation/feature_manifest.json"
            )
        else:
            expected_component_path = (
                run_dir
                / "03_features"
                / component
                / f"{route}_validation/feature_manifest.json"
            )
        _same_record(
            record,
            _record(expected_component_path),
            f"feature extraction component {name}",
        )
        component_path = verified_artifact_path(
            record, name=f"feature extraction component {name}"
        )
        component_manifest = _load(
            component_path, f"feature extraction component {name}", ("COMPLETE",)
        )
        verify_artifact_records_recursive(
            {
                "sources": component_manifest.get("sources"),
                "artifacts": component_manifest.get("artifacts"),
                "artifact": component_manifest.get("artifact"),
            },
            name=f"feature extraction component {name}",
            require_at_least_one=True,
        )
        component_payloads[name] = component_manifest
    measurements = benchmark.get("component_measurements")
    if not isinstance(measurements, list) or len(measurements) != len(
        FEATURE_BENCHMARK_COMPONENTS
    ):
        raise ValueError("feature extraction benchmark component rows are incomplete")
    for expected_name, row in zip(
        FEATURE_BENCHMARK_COMPONENTS, measurements, strict=True
    ):
        if not isinstance(row, Mapping) or set(row) != {
            "name",
            "route",
            "component_or_track",
            "measurement_scope",
            "candidate_rows",
            "feature_extraction_latency_ms",
            "peak_memory_mb",
            "manifest",
        }:
            raise ValueError("feature extraction component row schema is not exact")
        route, component = (
            expected_name.split("/", 1)[::-1]
            if expected_name.startswith("backend_maps/")
            else (expected_name.rsplit("/", 1)[1], expected_name.rsplit("/", 1)[0])
        )
        component_manifest = component_payloads[expected_name]
        row_field = "candidate_features" if component == "common" else "candidate_rows"
        expected_peak_memory = component_manifest.get(
            "feature_extraction_peak_memory_mb",
            component_manifest.get("peak_memory_mb"),
        )
        if (
            row.get("name") != expected_name
            or row.get("route") != route
            or row.get("component_or_track") != component
            or row.get("measurement_scope")
            != "full_validation_persisted_extraction"
            or not isinstance(row.get("candidate_rows"), int)
            or int(row["candidate_rows"]) <= 0
            or not math.isfinite(float(row.get("feature_extraction_latency_ms", math.nan)))
            or float(row["feature_extraction_latency_ms"]) < 0
            or not math.isfinite(float(row.get("peak_memory_mb", math.nan)))
            or float(row["peak_memory_mb"]) <= 0
            or row.get("manifest") != components[expected_name]
            or int(row["candidate_rows"]) != int(component_manifest.get(row_field, -1))
            or float(row["feature_extraction_latency_ms"])
            != float(component_manifest.get("feature_extraction_latency_ms", math.nan))
            or float(row["peak_memory_mb"]) != float(expected_peak_memory)
        ):
            raise ValueError(f"feature extraction component measurement mismatch: {expected_name}")
    artifacts = benchmark.get("artifacts")
    if not isinstance(artifacts, Mapping) or set(artifacts) != {"tagged_output_manifest"}:
        raise ValueError("feature extraction benchmark artifact inventory is not exact")
    tagged_path = verified_artifact_path(
        artifacts["tagged_output_manifest"], name="tagged T3 benchmark output"
    )
    expected_tagged_path = (
        run_dir
        / "03_features/tri_backend_dense/validation_latency_benchmark_128/manifest.json"
    )
    _same_record(
        artifacts["tagged_output_manifest"],
        _record(expected_tagged_path),
        "tagged T3 benchmark output",
    )
    tagged = _load(tagged_path, "tagged T3 benchmark output", ("COMPLETE",))
    if (
        tagged.get("split") != "validation"
        or tagged.get("tag") != "latency_benchmark_128"
        or tagged.get("candidate_test_labels_read") is not None
        or not isinstance(tagged.get("artifacts"), Mapping)
        or set(tagged["artifacts"]) != set(ROUTES)
    ):
        raise ValueError("tagged T3 benchmark output contract mismatch")
    verify_artifact_records_recursive(
        tagged["artifacts"], name="tagged T3 benchmark outputs", require_at_least_one=True
    )
    elapsed = float(benchmark.get("feature_extraction_elapsed_seconds", math.nan))
    latency = float(benchmark.get("feature_extraction_latency_ms", math.nan))
    per_sample = float(
        benchmark.get("feature_extraction_latency_ms_per_sample", math.nan)
    )
    peak = float(benchmark.get("peak_memory_mb", math.nan))
    if (
        not all(math.isfinite(value) and value > 0 for value in (elapsed, latency, per_sample, peak))
        or not math.isclose(
            latency,
            elapsed * 1000.0 / int(configuration["total_candidate_rows"]),
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        or not math.isclose(
            per_sample,
            elapsed * 1000.0 / int(configuration["sample_limit"]),
            rel_tol=0.0,
            abs_tol=1e-12,
        )
    ):
        raise ValueError("feature extraction benchmark timing formula mismatch")
    command = benchmark.get("command")
    if not isinstance(command, list) or len(command) != 19 or command[1:3] != [
        "-m",
        "tools.unified_reranking.extract_tri_backend_dense_features",
    ]:
        raise ValueError("feature extraction benchmark command is not exact")
    flags = dict(zip(command[3::2], command[4::2], strict=True))
    if set(flags) != {
        "--run-dir",
        "--fair-test-source",
        "--split",
        "--device",
        "--batch-size",
        "--chunk-size",
        "--limit",
        "--tag",
    } or flags != {
        "--run-dir": str(path.resolve().parents[2]),
        "--fair-test-source": flags.get("--fair-test-source"),
        "--split": "validation",
        "--device": str(configuration["device"]),
        "--batch-size": str(configuration["batch_size"]),
        "--chunk-size": str(configuration["chunk_size"]),
        "--limit": "128",
        "--tag": "latency_benchmark_128",
    } or not str(flags.get("--fair-test-source", "")):
        raise ValueError("feature extraction benchmark command differs from configuration")
    return benchmark


def _validate_cell_extraction_binding(
    cell: Mapping[str, Any],
    *,
    run_dir: Path,
    expected_t3_record: Mapping[str, Any],
    name: str,
) -> None:
    configuration = cell.get("configuration")
    sources = cell.get("sources")
    if (
        not isinstance(configuration, Mapping)
        or not isinstance(sources, Mapping)
    ):
        raise ValueError(f"{name} is not a matrix/rule cell")
    source_identity = configuration.get("source_identity")
    if not isinstance(source_identity, Mapping):
        raise ValueError(f"{name} lacks source identity")
    track = str(configuration.get("track", ""))
    expected_benchmark = expected_t3_record if track == "T3_tri_backend" else None
    expected_sha = None if expected_benchmark is None else expected_benchmark["sha256"]

    train_manifest_path = verified_artifact_path(
        sources.get("train_feature_manifest", {}),
        name=f"{name} Train feature manifest",
    )
    train_manifest = _load(
        train_manifest_path, f"{name} Train feature manifest", ("COMPLETE",)
    )
    train_latency, train_benchmark = resolved_track_extraction_latency(
        run_dir, track, train_manifest
    )
    if train_benchmark != expected_benchmark:
        raise ValueError(f"{name} Train benchmark differs from the locked benchmark")
    if sources.get("train_feature_extraction_benchmark") != train_benchmark:
        raise ValueError(f"{name} Train feature benchmark binding mismatch")
    if source_identity.get("train_feature_extraction_benchmark_sha256") != expected_sha:
        raise ValueError(f"{name} Train benchmark identity mismatch")
    mode = str(configuration.get("mode", ""))
    rule_cell = "method" in configuration
    prediction_identity_key = (
        "prediction_feature_extraction_benchmark_sha256"
        if rule_cell
        else "validation_feature_extraction_benchmark_sha256"
    )
    prediction_source_key = (
        "prediction_feature_extraction_benchmark"
        if rule_cell
        else "validation_feature_extraction_benchmark"
    )
    if mode == "validation" or rule_cell:
        if source_identity.get(prediction_identity_key) != expected_sha:
            raise ValueError(f"{name} prediction benchmark identity mismatch")
    if mode == "oof":
        if not rule_cell and sources.get("validation_feature_extraction_benchmark") is not None:
            raise ValueError(f"{name} OOF cell has a Validation benchmark source")
    elif mode != "validation":
        raise ValueError(f"{name} has unsupported matrix mode")
    feature_manifest_key = (
        "prediction_feature_manifest"
        if rule_cell
        else "validation_feature_manifest"
        if mode == "validation"
        else "train_feature_manifest"
    )
    feature_manifest_path = verified_artifact_path(
        sources.get(feature_manifest_key, {}),
        name=f"{name} feature manifest",
    )
    feature_manifest = _load(
        feature_manifest_path, f"{name} feature manifest", ("COMPLETE",)
    )
    prediction_latency, prediction_benchmark = resolved_track_extraction_latency(
        run_dir, track, feature_manifest
    )
    if prediction_benchmark != expected_benchmark:
        raise ValueError(f"{name} prediction benchmark differs from the locked benchmark")
    if (mode == "validation" or rule_cell) and sources.get(
        prediction_source_key
    ) != prediction_benchmark:
        raise ValueError(f"{name} prediction benchmark binding mismatch")
    expected_latency_ms = train_latency if mode == "oof" else prediction_latency
    observed_latency = _finite(
        cell.get("feature_extraction_latency_ms"), f"{name} extraction latency"
    )
    if not math.isclose(
        observed_latency, expected_latency_ms, rel_tol=0.0, abs_tol=1e-12
    ):
        raise ValueError(f"{name} extraction latency differs from benchmark")


def _replay_cell_metrics(run_dir: Path, cell: Mapping[str, Any], name: str) -> None:
    """Recompute one development cell's metrics/decisions from frozen sources."""

    configuration = cell.get("configuration")
    sources = cell.get("sources")
    artifacts = cell.get("artifacts")
    if not all(isinstance(value, Mapping) for value in (configuration, sources, artifacts)):
        raise ValueError(f"{name} contract is incomplete")
    route = str(configuration.get("route", ""))
    mode = str(configuration.get("mode", ""))
    rule_cell = "method" in configuration
    if route not in ROUTES or mode not in {"oof", "validation"}:
        raise ValueError(f"{name} route/mode is invalid")
    split = "train" if mode == "oof" else "validation"
    label_key = (
        "prediction_labels"
        if rule_cell
        else "train_labels"
        if mode == "oof"
        else "validation_labels"
    )
    label_path = run_dir / f"03_features/candidate_labels_{route}_{split}_top5.parquet"
    _same_record(sources.get(label_key), _record(label_path), f"{name} labels")
    candidate_path = run_dir / f"02_candidates/{route}_{split}_top5.parquet"
    candidates = pd.read_parquet(
        candidate_path, columns=["sample_id", "candidate_id", "native_rank"]
    )
    labels = pd.read_parquet(
        label_path, columns=["sample_id", "candidate_id", "candidate_success"]
    )
    if mode == "validation":
        denominator_path = run_dir / "01_manifests/paired_validation.parquet"
        _same_record(
            sources.get("validation_denominator"),
            _record(denominator_path),
            f"{name} denominator",
        )
        denominator = pd.read_parquet(
            denominator_path, columns=["sample_id"]
        )["sample_id"].astype(str).tolist()
    else:
        fold_path = run_dir / "04_splits/fold_assignments.parquet"
        _same_record(sources.get("folds"), _record(fold_path), f"{name} folds")
        held_fold = int(configuration.get("held_fold", -1))
        folds = pd.read_parquet(fold_path, columns=["sample_id", "fold"])
        denominator = folds.loc[
            folds["fold"].astype(int).eq(held_fold), "sample_id"
        ].astype(str).tolist()
        candidates = candidates.loc[
            candidates["sample_id"].astype(str).isin(set(denominator))
        ].copy()
        labels = labels.loc[
            labels["sample_id"].astype(str).isin(set(denominator))
        ].copy()
    prediction_path = verified_artifact_path(
        artifacts.get("predictions"), name=f"{name} predictions"
    )
    predictions = pd.read_parquet(prediction_path)[
        ["sample_id", "candidate_id", "score"]
    ]
    keys = ["sample_id", "candidate_id"]
    expected_keys = set(map(tuple, candidates[keys].astype(str).to_numpy()))
    observed_keys = set(map(tuple, predictions[keys].astype(str).to_numpy()))
    if (
        observed_keys != expected_keys
        or predictions[keys].duplicated().any()
        or labels[keys].duplicated().any()
    ):
        raise ValueError(f"{name} predictions/labels do not exactly cover candidates")
    evaluation = candidates.merge(labels, on=keys, validate="one_to_one").merge(
        predictions, on=keys, validate="one_to_one"
    )
    metrics, decisions = evaluate_order_only(
        denominator, evaluation, score_column="score"
    )
    if canonical_sha256(metrics) != canonical_sha256(cell.get("metrics")):
        raise ValueError(f"{name} metrics differ from replayed predictions")
    decision_path = verified_artifact_path(
        artifacts.get("decisions"), name=f"{name} decisions"
    )
    stored = pd.read_parquet(decision_path)
    pd.testing.assert_frame_equal(
        stored.sort_values("sample_id", kind="mergesort").reset_index(drop=True),
        decisions[stored.columns]
        .sort_values("sample_id", kind="mergesort")
        .reset_index(drop=True),
        check_dtype=False,
        check_exact=True,
    )


def validate_matrix_phase_execution(
    run_dir: Path,
    phase: str,
    *,
    expected_selection: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Replay one exact screen/selected/encoder job universe and its cells."""

    from argparse import Namespace

    from .matrix_phase import load_matrix_phase_cells
    from tools.unified_reranking import train_matrix

    run_dir = run_dir.resolve()
    if phase not in {"screen", "selected", "encoder"}:
        raise ValueError("unsupported matrix phase")
    execution_path = (
        run_dir / "05_models/matrix_plans" / f"{phase}_latest_execution.json"
    )
    execution = _load(
        execution_path, f"matrix {phase} latest execution", ("COMPLETE",)
    )
    plan_path = verified_artifact_path(
        execution.get("plan"), name=f"matrix {phase} execution plan"
    )
    plan = _load(plan_path, f"matrix {phase} execution plan", ("PLANNED",))
    if (
        plan.get("phase") != phase
        or plan.get("routes") != list(ROUTES)
        or plan.get("tracks")
        != ["T1_native", "T2_matched_common", "T3_tri_backend"]
    ):
        raise ValueError(f"matrix {phase} plan route/track universe is not exact")
    _same_record(
        plan.get("planner_tool"),
        _record(Path(train_matrix.__file__)),
        f"matrix {phase} planner tool",
    )
    selection_path: Path | None = None
    if phase == "screen":
        if expected_selection is not None:
            raise ValueError("matrix screen cannot have a selection input")
    else:
        if expected_selection is None:
            raise ValueError(f"matrix {phase} requires its selection record")
        _same_record(
            plan.get("selection"), expected_selection, f"matrix {phase} selection"
        )
        selection_path = Path(str(expected_selection["path"])).resolve()
    expected_jobs = train_matrix.build_jobs(
        Namespace(
            run_dir=run_dir,
            phase=phase,
            routes=list(ROUTES),
            tracks=["T1_native", "T2_matched_common", "T3_tri_backend"],
            selection_json=selection_path,
        )
    )
    expected_inventory = [
        {"identifier": job.identifier, "command": list(job.command)}
        for job in expected_jobs
    ]
    if plan.get("jobs") != expected_inventory or int(
        plan.get("job_count", -1)
    ) != len(expected_inventory):
        raise ValueError(f"matrix {phase} plan differs from regenerated job universe")
    cells = load_matrix_phase_cells(
        run_dir,
        phase,
        expected_selection=selection_path,
    )
    if len(cells) != len(expected_inventory):
        raise ValueError(f"matrix {phase} cell inventory is incomplete")
    benchmark_path = (
        run_dir / "07_validation/telemetry/feature_extraction_benchmark.json"
    )
    benchmark = validate_feature_extraction_benchmark(benchmark_path)
    benchmark_record = _record(benchmark_path)
    for index, (cell, _path) in enumerate(cells):
        _validate_telemetry(cell, f"matrix {phase} cell {index}")
        _validate_cell_extraction_binding(
            cell,
            run_dir=run_dir,
            expected_t3_record=benchmark_record,
            name=f"matrix {phase} cell {index}",
        )
        _replay_cell_metrics(run_dir, cell, f"matrix {phase} cell {index}")
    return execution


def validate_screen_selection(path: Path) -> dict[str, Any]:
    """Replay the complete seed-42 screen table and both frozen selections."""

    from .matrix_phase import load_matrix_phase_cells
    from tools.unified_reranking import select_validation_screen as selector

    path = path.resolve()
    run_dir = path.parents[1]
    manifest = _load(path, "screen selection manifest", ("COMPLETE",))
    unsigned = dict(manifest)
    recorded_content = unsigned.pop("content_sha256", None)
    if recorded_content != canonical_sha256(unsigned):
        raise ValueError("screen selection manifest content hash mismatch")
    if set(manifest.get("sources", {})) != {"screen_execution", "selector_tool"}:
        raise ValueError("screen selection source inventory is not exact")
    if set(manifest.get("artifacts", {})) != {
        "screen_table",
        "screen_winners",
        "finalists",
        "encoder_loss_selections",
    }:
        raise ValueError("screen selection artifact inventory is not exact")
    execution_path = run_dir / "05_models/matrix_plans/screen_latest_execution.json"
    _same_record(
        manifest["sources"]["screen_execution"],
        _record(execution_path),
        "screen selection execution",
    )
    _same_record(
        manifest["sources"]["selector_tool"],
        _record(Path(selector.__file__)),
        "screen selector tool",
    )
    validate_matrix_phase_execution(run_dir, "screen")
    cells = load_matrix_phase_cells(run_dir, "screen")
    cells_by_path = {str(cell_path): cell for cell, cell_path in cells}
    table_path = verified_artifact_path(
        manifest["artifacts"]["screen_table"], name="screen trial table"
    )
    winners_path = verified_artifact_path(
        manifest["artifacts"]["screen_winners"], name="screen winner table"
    )
    table = pd.read_csv(table_path)
    if len(table) != len(cells) or table["manifest_path"].duplicated().any():
        raise ValueError("screen trial table does not exactly cover phase cells")
    native_cache: dict[str, tuple[dict[str, Any], pd.DataFrame]] = {}
    for row in table.itertuples(index=False):
        cell_path = str(Path(str(row.manifest_path)).resolve())
        cell = cells_by_path.get(cell_path)
        if cell is None or str(row.manifest_sha256) != sha256_file(Path(cell_path)):
            raise ValueError("screen table references an off-execution cell")
        configuration = cell.get("configuration")
        if not isinstance(configuration, dict) or json.loads(
            str(row.configuration_json)
        ) != configuration:
            raise ValueError("screen table configuration differs from its cell")
        route = str(configuration["route"])
        if route not in native_cache:
            native_cache[route] = selector._native_decisions(run_dir, route)
        native_metrics, native_decisions = native_cache[route]
        recomputed_metrics, decisions = selector._recompute_validation_cell(
            run_dir, cell, Path(cell_path)
        )
        comparison = compare_selections(
            native_decisions,
            decisions,
            oracle_at_5=float(native_metrics["oracle_at_5"]),
        )
        rule_cell = "method" in configuration
        expected = {
            "route": route,
            "track": str(configuration["track"]),
            "method_code": selector.method_code(configuration, rule=rule_cell),
            "cell_key": str(cell["cell_key"]),
            "j_at_1": float(recomputed_metrics["j_at_1"]),
            "mrr_at_5": float(recomputed_metrics["mrr_at_5"]),
            "ndcg_at_5": float(recomputed_metrics["ndcg_at_5"]),
            **comparison,
        }
        for field, expected_value in expected.items():
            observed = getattr(row, field)
            if isinstance(expected_value, (int, float)) and not isinstance(
                expected_value, bool
            ):
                if not math.isclose(
                    float(observed), float(expected_value), rel_tol=0.0, abs_tol=1e-12
                ):
                    raise ValueError(f"screen table {field} differs from replay")
            elif str(observed) != str(expected_value):
                raise ValueError(f"screen table {field} differs from replay")
    sort_columns = [
        "route",
        "track",
        "method_code",
        "j_at_1",
        "harmful",
        "switch_rate",
        "mrr_at_5",
        "cell_key",
    ]
    expected_order = table.sort_values(
        sort_columns,
        ascending=[True, True, True, False, True, True, False, True],
        kind="mergesort",
    )["cell_key"].astype(str).tolist()
    if table["cell_key"].astype(str).tolist() != expected_order:
        raise ValueError("screen table order differs from the locked tie-break")
    expected_best = ~table.duplicated(["route", "track", "method_code"])
    if not table["best_within_method"].astype(bool).equals(expected_best):
        raise ValueError("screen within-method winners differ from replay")
    winners = pd.read_csv(winners_path)
    expected_winners = table.loc[expected_best].reset_index(drop=True)
    pd.testing.assert_frame_equal(
        winners.reset_index(drop=True), expected_winners, check_dtype=False
    )

    finalist_path = verified_artifact_path(
        manifest["artifacts"]["finalists"], name="screen finalists"
    )
    encoder_path = verified_artifact_path(
        manifest["artifacts"]["encoder_loss_selections"],
        name="encoder loss selections",
    )
    finalist_payload = _load(
        finalist_path, "screen finalists", ("VALIDATION_SCREEN_LOCKED",)
    )
    encoder_payload = _load(
        encoder_path,
        "encoder loss selections",
        ("CONTROLLED_LOSS_LOCKED_FOR_ENCODER_COMPARISON",),
    )
    for payload, name in (
        (finalist_payload, "screen finalists"),
        (encoder_payload, "encoder loss selections"),
    ):
        _same_record(
            payload.get("screen_execution"),
            _record(execution_path),
            f"{name} execution",
        )
        _same_record(
            payload.get("screen_table"), _record(table_path), f"{name} table"
        )
        _same_record(
            payload.get("selector_tool"),
            _record(Path(selector.__file__)),
            f"{name} selector",
        )
    finalist_codes = {
        "R4_mlp_ranknet",
        "R6_lambdamart",
        "R7_mlp_jacquard_margin",
    }
    neural_codes = {
        "R3_mlp_bce",
        "R4_mlp_ranknet",
        "R5_mlp_listwise",
        "R7_mlp_jacquard_margin",
    }
    expected_finalists: dict[str, list[dict[str, Any]]] = {}
    expected_encoder: dict[str, dict[str, Any]] = {}
    for (route, track), group in expected_winners.groupby(
        ["route", "track"], sort=True
    ):
        key = f"{route}/{track}"
        chosen = group.loc[group["method_code"].isin(finalist_codes)].sort_values(
            "method_code"
        )
        if set(chosen["method_code"]) != finalist_codes:
            raise ValueError(f"screen replay lacks finalist families for {key}")
        expected_finalists[key] = []
        for row in chosen.itertuples(index=False):
            cell_path = Path(str(row.manifest_path)).resolve()
            entry = selector._selection_entry(
                cells_by_path[str(cell_path)]["configuration"], cell_path
            )
            entry["method_code"] = str(row.method_code)
            entry["screen_cell_key"] = str(row.cell_key)
            expected_finalists[key].append(entry)
        neural = group.loc[group["method_code"].isin(neural_codes)].sort_values(
            ["j_at_1", "harmful", "switch_rate", "mrr_at_5", "method_code"],
            ascending=[False, True, True, False, True],
            kind="mergesort",
        )
        best = neural.iloc[0]
        best_path = Path(str(best["manifest_path"])).resolve()
        entry = selector._selection_entry(
            cells_by_path[str(best_path)]["configuration"], best_path
        )
        entry["method_code"] = str(best["method_code"])
        entry["screen_cell_key"] = str(best["cell_key"])
        expected_encoder[key] = entry
    if canonical_sha256(finalist_payload.get("selections")) != canonical_sha256(
        expected_finalists
    ):
        raise ValueError("screen finalists differ from replayed winners")
    if canonical_sha256(encoder_payload.get("selections")) != canonical_sha256(
        expected_encoder
    ):
        raise ValueError("encoder loss selections differ from replayed winners")
    return manifest


def _replay_scalar_ensemble(
    run_dir: Path,
    ensemble: Mapping[str, Any],
    *,
    name: str,
) -> set[tuple[str, str]]:
    """Rebuild one three-seed ensemble from its exact cell predictions."""

    from tools.unified_reranking import select_primary_rankers as primary

    identity = ensemble.get("identity")
    sources = ensemble.get("sources")
    artifacts = ensemble.get("artifacts")
    if not all(isinstance(value, Mapping) for value in (identity, sources, artifacts)):
        raise ValueError(f"{name} ensemble contract is incomplete")
    route = str(identity.get("route", ""))
    track = str(identity.get("track", ""))
    split = str(ensemble.get("split", ""))
    if route not in ROUTES or split not in {"train", "validation"}:
        raise ValueError(f"{name} ensemble identity is invalid")
    candidate_path = verified_artifact_path(
        sources.get("candidates"), name=f"{name} candidates"
    )
    label_path = verified_artifact_path(sources.get("labels"), name=f"{name} labels")
    expected_candidate_path = (
        run_dir / f"02_candidates/{route}_{split}_top5.parquet"
    )
    expected_label_path = (
        run_dir / f"03_features/candidate_labels_{route}_{split}_top5.parquet"
    )
    _same_record(sources.get("candidates"), _record(expected_candidate_path), f"{name} candidates")
    _same_record(sources.get("labels"), _record(expected_label_path), f"{name} labels")
    candidates = pd.read_parquet(candidate_path)
    labels = pd.read_parquet(label_path)[
        ["sample_id", "candidate_id", "candidate_success"]
    ]
    keys = ["sample_id", "candidate_id"]
    expected_keys = set(map(tuple, candidates[keys].astype(str).to_numpy()))
    matrix_records = sources.get("matrix_manifests")
    expected_cell_count = len(FORMAL_SEEDS) * (5 if split == "train" else 1)
    if not isinstance(matrix_records, list) or len(matrix_records) != expected_cell_count:
        raise ValueError(f"{name} matrix-cell inventory is incomplete")
    predictions = candidates[[*keys, "native_rank"]].copy()
    used_records: set[tuple[str, str]] = set()
    for seed in FORMAL_SEEDS:
        pieces: list[pd.DataFrame] = []
        expected_folds = set(range(5)) if split == "train" else {None}
        observed_folds: set[int | None] = set()
        for index, record in enumerate(matrix_records):
            cell, cell_path = _validate_cell(record, f"{name} cell {index}")
            configuration = cell.get("configuration", {})
            if int(configuration.get("seed", -1)) != seed:
                continue
            fold = configuration.get("held_fold")
            if (
                configuration.get("route") != route
                or configuration.get("track") != track
                or configuration.get("encoder") != identity.get("encoder")
                or configuration.get("loss") != identity.get("loss")
                or configuration.get("mode")
                != ("oof" if split == "train" else "validation")
                or fold not in expected_folds
                or fold in observed_folds
            ):
                raise ValueError(f"{name} cell configuration differs from ensemble")
            observed_folds.add(fold)
            prediction_path = verified_artifact_path(
                cell.get("artifacts", {}).get("predictions"),
                name=f"{name} cell predictions",
            )
            pieces.append(pd.read_parquet(prediction_path)[[*keys, "score"]])
            used_records.add((str(cell_path), sha256_file(cell_path)))
        if observed_folds != expected_folds:
            raise ValueError(f"{name} seed-{seed} fold inventory is incomplete")
        seed_predictions = pd.concat(pieces, ignore_index=True)
        observed_keys = set(map(tuple, seed_predictions[keys].astype(str).to_numpy()))
        if (
            observed_keys != expected_keys
            or seed_predictions[keys].duplicated().any()
        ):
            raise ValueError(f"{name} seed-{seed} predictions do not exactly cover candidates")
        predictions = predictions.merge(
            seed_predictions.rename(columns={"score": f"score_seed_{seed}"}),
            on=keys,
            validate="one_to_one",
        )
    score_columns = [f"score_seed_{seed}" for seed in FORMAL_SEEDS]
    predictions["ensemble_score"] = predictions[score_columns].mean(axis=1)
    stored_prediction_path = verified_artifact_path(
        artifacts.get("predictions"), name=f"{name} ensemble predictions"
    )
    stored_predictions = pd.read_parquet(stored_prediction_path)
    expected_columns = [*keys, "native_rank", *score_columns, "ensemble_score"]
    pd.testing.assert_frame_equal(
        stored_predictions[expected_columns]
        .sort_values(keys, kind="mergesort")
        .reset_index(drop=True),
        predictions[expected_columns]
        .sort_values(keys, kind="mergesort")
        .reset_index(drop=True),
        check_dtype=False,
        check_exact=True,
    )
    denominator = pd.read_parquet(
        run_dir / f"01_manifests/paired_{split}.parquet", columns=["sample_id"]
    )["sample_id"].astype(str).tolist()
    evaluation = predictions.merge(labels, on=keys, validate="one_to_one")
    metrics, decisions = evaluate_order_only(
        denominator, evaluation, score_column="ensemble_score"
    )
    decisions = primary._augment_decisions(candidates, predictions, decisions)
    if canonical_sha256(metrics) != canonical_sha256(ensemble.get("metrics")):
        raise ValueError(f"{name} ensemble metrics differ from cell replay")
    stored_decision_path = verified_artifact_path(
        artifacts.get("decisions"), name=f"{name} ensemble decisions"
    )
    stored_decisions = pd.read_parquet(stored_decision_path)
    pd.testing.assert_frame_equal(
        stored_decisions.sort_values("sample_id", kind="mergesort").reset_index(drop=True),
        decisions[stored_decisions.columns]
        .sort_values("sample_id", kind="mergesort")
        .reset_index(drop=True),
        check_dtype=False,
        check_exact=True,
    )
    if len(used_records) != expected_cell_count:
        raise ValueError(f"{name} cell records are duplicated")
    return used_records


def validate_scalar_selection(path: Path) -> dict[str, Any]:
    """Recompute the scalar winner from all hash-bound Validation trial rows."""

    path = path.resolve()
    run_dir = path.parents[1]
    selection = _load(path, "primary scalar selection", ("VALIDATION_LOCKED",))
    validate_screen_selection(
        run_dir / "07_validation/screen_selection_manifest.json"
    )
    finalist_path = run_dir / "05_models/screen_finalists.json"
    selected_execution_path = (
        run_dir / "05_models/matrix_plans/selected_latest_execution.json"
    )
    from tools.unified_reranking import select_primary_rankers as primary_selector

    _same_record(
        selection.get("screen_finalists"),
        _record(finalist_path),
        "primary scalar screen finalists",
    )
    _same_record(
        selection.get("selected_execution"),
        _record(selected_execution_path),
        "primary scalar selected execution",
    )
    _same_record(
        selection.get("selector_tool"),
        _record(Path(primary_selector.__file__)),
        "primary scalar selector tool",
    )
    validate_matrix_phase_execution(
        run_dir,
        "selected",
        expected_selection=_record(finalist_path),
    )
    selected_execution = _load(
        selected_execution_path, "selected matrix execution", ("COMPLETE",)
    )
    selected_output_records = {
        (str(Path(str(record["path"])).resolve()), str(record["sha256"]))
        for record in selected_execution.get("output_manifests", [])
        if isinstance(record, Mapping)
    }
    benchmark_path = (
        run_dir / "07_validation/telemetry/feature_extraction_benchmark.json"
    )
    benchmark = validate_feature_extraction_benchmark(benchmark_path)
    benchmark_record = _record(benchmark_path)
    if (
        selection.get("primary_track") != PRIMARY_TRACK
        or set(selection.get("selections", {})) != set(ROUTES)
        or list(selection.get("seeds", [])) != list(FORMAL_SEEDS)
    ):
        raise ValueError("primary scalar selection contract is incomplete")
    table_path = verified_artifact_path(selection.get("table", {}), name="scalar finalist table")
    table = pd.read_csv(table_path)
    required = {
        "route",
        "track",
        "method_code",
        "encoder",
        "loss",
        "ensemble_id",
        "j_at_1",
        "mrr_at_5",
        "delta_j_at_1",
        "recovered",
        "harmful",
        "switch_rate",
        "validation_manifest",
        "validation_manifest_sha256",
        "oof_manifest",
        "oof_manifest_sha256",
    }
    if table.empty or required.difference(table.columns):
        raise ValueError("scalar finalist table schema is incomplete")
    if len(table) != len(ROUTES) * 3 * 3:
        raise ValueError("scalar finalist table does not cover the exact finalist grid")
    finalist_payload = _load(
        finalist_path, "screen finalists", ("VALIDATION_SCREEN_LOCKED",)
    )
    expected_trials = {
        (str(route), str(track), str(choice["method_code"]))
        for key, choices in finalist_payload.get("selections", {}).items()
        for route, track in (str(key).split("/", 1),)
        for choice in choices
    }
    observed_trials = set(
        table[["route", "track", "method_code"]].astype(str).itertuples(
            index=False, name=None
        )
    )
    if observed_trials != expected_trials:
        raise ValueError("scalar finalist table differs from screen finalists")
    if table.duplicated(["route", "track", "method_code", "encoder", "loss"]).any():
        raise ValueError("scalar finalist table contains duplicate trials")
    used_selected_cells: set[tuple[str, str]] = set()
    for row in table.itertuples(index=False):
        validation_record = {
            "path": str(row.validation_manifest),
            "sha256": str(row.validation_manifest_sha256),
        }
        oof_record = {"path": str(row.oof_manifest), "sha256": str(row.oof_manifest_sha256)}
        validation_path = verified_artifact_path(validation_record, name="scalar Validation trial")
        oof_path = verified_artifact_path(oof_record, name="scalar OOF trial")
        validation = _load(validation_path, "scalar Validation trial", ("COMPLETE",))
        oof = _load(oof_path, "scalar OOF trial", ("COMPLETE",))
        used_selected_cells.update(
            _replay_scalar_ensemble(
                run_dir, validation, name=f"scalar {row.route}/{row.track} Validation"
            )
        )
        used_selected_cells.update(
            _replay_scalar_ensemble(
                run_dir, oof, name=f"scalar {row.route}/{row.track} OOF"
            )
        )
        identity = validation.get("identity", {})
        if identity != oof.get("identity") or any(
            str(identity.get(field, "")) != str(getattr(row, field))
            for field in ("route", "track", "method_code", "encoder", "loss")
        ) or str(validation.get("ensemble_id", "")) != str(row.ensemble_id):
            raise ValueError("scalar trial identity differs from bound ensembles")
        verify_artifact_records_recursive(
            {"sources": validation.get("sources"), "artifacts": validation.get("artifacts")},
            name="scalar Validation trial",
            require_at_least_one=True,
        )
        sources = validation.get("sources", {})
        artifacts = validation.get("artifacts", {})
        candidate_path = verified_artifact_path(
            sources.get("candidates", {}), name="scalar Validation candidates"
        )
        label_path = verified_artifact_path(
            sources.get("labels", {}), name="scalar Validation labels"
        )
        prediction_path = verified_artifact_path(
            artifacts.get("predictions", {}), name="scalar Validation predictions"
        )
        decision_path = verified_artifact_path(
            artifacts.get("decisions", {}), name="scalar Validation decisions"
        )
        candidates = pd.read_parquet(candidate_path)[
            ["sample_id", "candidate_id", "native_rank"]
        ]
        labels = pd.read_parquet(label_path)[
            ["sample_id", "candidate_id", "candidate_success"]
        ]
        predictions = pd.read_parquet(prediction_path)
        keys = ["sample_id", "candidate_id"]
        evaluation = candidates.merge(labels, on=keys, validate="one_to_one").merge(
            predictions[keys + ["ensemble_score"]], on=keys, validate="one_to_one"
        )
        if len(evaluation) != len(candidates) or len(evaluation) != len(labels):
            raise ValueError("scalar Validation predictions/labels do not exactly cover candidates")
        denominator = pd.read_parquet(
            run_dir / "01_manifests" / "paired_validation.parquet",
            columns=["sample_id"],
        )["sample_id"].astype(str).tolist()
        recomputed_metrics, recomputed_decisions = evaluate_order_only(
            denominator, evaluation, score_column="ensemble_score"
        )
        native_input = evaluation.copy()
        native_input["native_control_score"] = -pd.to_numeric(
            native_input["native_rank"], errors="raise"
        ).astype(float)
        native_metrics, native_decisions = evaluate_order_only(
            denominator, native_input, score_column="native_control_score"
        )
        comparison = compare_selections(
            native_decisions,
            recomputed_decisions,
            oracle_at_5=float(native_metrics["oracle_at_5"]),
        )
        stored_decisions = pd.read_parquet(decision_path)
        decision_keys = stored_decisions[["sample_id", "selected_candidate_id"]].copy()
        expected_keys = recomputed_decisions[["sample_id", "selected_candidate_id"]].copy()
        if not decision_keys.equals(expected_keys):
            raise ValueError("scalar Validation decisions differ from recomputed predictions")
        for metric in ("j_at_1", "mrr_at_5"):
            observed = _finite(recomputed_metrics.get(metric), f"scalar trial {metric}")
            if not math.isclose(observed, float(getattr(row, metric)), rel_tol=0.0, abs_tol=1e-12) or not math.isclose(
                observed,
                float(validation.get("metrics", {}).get(metric)),
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                raise ValueError("scalar trial metrics differ from recomputed Validation predictions")
        for field in ("delta_j_at_1", "recovered", "harmful", "switch_rate"):
            if not math.isclose(
                float(comparison[field]),
                float(getattr(row, field)),
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                raise ValueError("scalar trial comparison differs from recomputed Validation predictions")
        verify_artifact_records_recursive(
            {"sources": oof.get("sources"), "artifacts": oof.get("artifacts")},
            name="scalar OOF trial",
            require_at_least_one=True,
        )
        if str(identity.get("track")) == "T3_tri_backend":
            for ensemble_name, ensemble in (
                ("Validation", validation),
                ("OOF", oof),
            ):
                matrix_records = ensemble.get("sources", {}).get(
                    "matrix_manifests"
                )
                if not isinstance(matrix_records, list) or not matrix_records:
                    raise ValueError(
                        f"scalar T3 {ensemble_name} trial lacks matrix cells"
                    )
                for index, record in enumerate(matrix_records):
                    cell, _ = _validate_cell(
                        record, f"scalar T3 {ensemble_name} cell {index}"
                    )
                    _validate_cell_extraction_binding(
                        cell,
                        run_dir=run_dir,
                        expected_t3_record=benchmark_record,
                        name=f"scalar T3 {ensemble_name} cell {index}",
                    )
    if used_selected_cells != selected_output_records:
        raise ValueError(
            "scalar ensemble cells do not exactly equal selected-phase outputs"
        )
    ranked = table.sort_values(
        ["route", "track", "j_at_1", "harmful", "switch_rate", "mrr_at_5", "method_code"],
        ascending=[True, True, False, True, True, False, True],
        kind="mergesort",
    ).drop_duplicates(["route", "track"])
    primary = ranked.loc[ranked["track"].eq(PRIMARY_TRACK)].set_index("route")
    if set(primary.index) != set(ROUTES):
        raise ValueError("scalar finalist table lacks one T2 winner per route")
    for route in ROUTES:
        row = primary.loc[route]
        chosen = selection["selections"][route]
        for field in ("method_code", "encoder", "loss", "ensemble_id"):
            if str(chosen.get(field, "")) != str(row[field]):
                raise ValueError(f"{route} scalar selected winner is not the best bound trial")
        for field in (
            "validation_manifest",
            "validation_manifest_sha256",
            "oof_manifest",
            "oof_manifest_sha256",
        ):
            if str(chosen.get(field, "")) != str(row[field]):
                raise ValueError(f"{route} scalar selected trial binding mismatch")
        recorded_metrics = chosen.get("selection_metrics", {})
        for field in (
            "j_at_1",
            "mrr_at_5",
            "delta_j_at_1",
            "recovered",
            "harmful",
            "switch_rate",
        ):
            if field not in recorded_metrics or not math.isclose(
                float(recorded_metrics[field]), float(row[field]), rel_tol=0.0, abs_tol=1e-12
            ):
                raise ValueError(f"{route} scalar selection metric mismatch: {field}")
    return selection


def _headroom_split(run_dir: Path, split: str) -> tuple[dict[str, Any], pd.DataFrame]:
    denominator = pd.read_parquet(
        run_dir / "01_manifests" / f"paired_{split}.parquet",
        columns=["sample_id", "scene_id"],
    )
    denominator["sample_id"] = denominator["sample_id"].astype(str)
    if denominator.empty or denominator["sample_id"].duplicated().any():
        raise ValueError(f"union {split} denominator is invalid")
    per_sample = denominator.copy()
    union_parts: list[pd.DataFrame] = []
    rates: dict[str, float] = {}
    counts: dict[str, int] = {}
    for route in ROUTES:
        candidates = pd.read_parquet(
            run_dir / "02_candidates" / f"{route}_{split}_top5.parquet",
            columns=["sample_id", "candidate_id", "native_rank"],
        )
        labels = pd.read_parquet(
            run_dir / "03_features" / f"candidate_labels_{route}_{split}_top5.parquet",
            columns=["sample_id", "candidate_id", "candidate_success"],
        )
        keys = ["sample_id", "candidate_id"]
        joined = candidates.merge(labels, on=keys, validate="one_to_one")
        if len(joined) != len(candidates) or len(joined) != len(labels):
            raise ValueError(f"union {route}/{split} label coverage mismatch")
        joined["route_candidate_id"] = route.upper() + ":" + joined["candidate_id"].astype(str)
        union_parts.append(joined)
        route_summary = joined.groupby("sample_id", sort=False).agg(
            **{
                f"{route}_candidate_count": ("candidate_id", "size"),
                f"{route}_oracle": ("candidate_success", "max"),
            }
        ).reset_index()
        per_sample = per_sample.merge(route_summary, on="sample_id", how="left", validate="one_to_one")
        per_sample[f"{route}_candidate_count"] = per_sample[f"{route}_candidate_count"].fillna(0).astype(int)
        per_sample[f"{route}_oracle"] = per_sample[f"{route}_oracle"].fillna(False).astype(bool)
        counts[route] = int(per_sample[f"{route}_oracle"].sum())
        rates[route] = float(per_sample[f"{route}_oracle"].mean())
    union = pd.concat(union_parts, ignore_index=True)
    union_summary = union.groupby("sample_id", sort=False).agg(
        union_candidate_count=("route_candidate_id", "size"),
        union_oracle=("candidate_success", "max"),
    ).reset_index()
    per_sample = per_sample.merge(union_summary, on="sample_id", how="left", validate="one_to_one")
    per_sample["union_candidate_count"] = per_sample["union_candidate_count"].fillna(0).astype(int)
    per_sample["union_oracle"] = per_sample["union_oracle"].fillna(False).astype(bool)
    if (per_sample["union_candidate_count"] > 15).any():
        raise ValueError("union pool exceeds Top-15")
    best = min(ROUTES, key=lambda route: (-rates[route], ROUTES.index(route)))
    union_rate = float(per_sample["union_oracle"].mean())
    gain = union_rate - rates[best]
    return {
        "split": split,
        "sample_count": len(per_sample),
        "candidate_count": len(union),
        "maximum_candidates_per_sample": int(per_sample["union_candidate_count"].max()),
        "route_oracle_successes": counts,
        "route_oracle_rates": rates,
        "route_oracle_at_5": rates,
        "best_single_route": best.upper(),
        "best_single_route_oracle": rates[best],
        "union_oracle_successes": int(per_sample["union_oracle"].sum()),
        "union_oracle": union_rate,
        "union_oracle_at_all": union_rate,
        "union_oracle_at_15": union_rate,
        "union_gain_over_best_single": gain,
        "union_gain_over_best_single_pp": 100.0 * gain,
        "primary_union_deduplication": "NONE",
    }, per_sample


def validate_union_headroom(run_dir: Path, path: Path) -> dict[str, Any]:
    """Recompute both development union summaries and the threshold decision."""

    manifest = _load(path, "union headroom", ("COMPLETE",))
    if manifest.get("test_access") != "NONE":
        raise PermissionError("union headroom is not development-only")
    sources = manifest.get("sources")
    configuration = manifest.get("configuration")
    if not isinstance(sources, Mapping) or not isinstance(configuration, Mapping):
        raise ValueError("union headroom lacks source/configuration bindings")
    verify_artifact_records_recursive(sources, name="union headroom sources", require_at_least_one=True)
    if manifest.get("signature_sha256") != canonical_sha256(
        {"configuration": configuration, "sources": sources}
    ):
        raise ValueError("union headroom signature mismatch")
    if float(configuration.get("validation_headroom_threshold", -1)) != HEADROOM_THRESHOLD:
        raise ValueError("union headroom threshold differs from the frozen contract")
    summaries = manifest.get("summaries")
    artifacts = manifest.get("artifacts")
    if not isinstance(summaries, Mapping) or set(summaries) != {"train", "validation"}:
        raise ValueError("union headroom summary inventory is incomplete")
    if not isinstance(artifacts, Mapping) or set(artifacts) != {
        "train_per_sample",
        "validation_per_sample",
        "decision",
    }:
        raise ValueError("union headroom artifact inventory is incomplete")
    for split in ("train", "validation"):
        recomputed, per_sample = _headroom_split(run_dir, split)
        if summaries[split] != recomputed:
            raise ValueError(f"union headroom {split} summary mismatch")
        stored = pd.read_parquet(
            verified_artifact_path(artifacts[f"{split}_per_sample"], name=f"union {split} per-sample")
        )
        pd.testing.assert_frame_equal(
            stored.reset_index(drop=True), per_sample.reset_index(drop=True), check_dtype=True
        )
    gain = float(summaries["validation"]["union_gain_over_best_single"])
    decision = "UNION_HEADROOM_AVAILABLE" if gain >= HEADROOM_THRESHOLD else "NO_UNION_HEADROOM"
    if manifest.get("decision") != decision:
        raise ValueError("union headroom decision differs from recomputed Validation gain")
    decision_payload = json.loads(
        verified_artifact_path(
            artifacts["decision"], name="union headroom decision artifact"
        ).read_text(encoding="utf-8")
    )
    if (
        decision_payload.get("decision") != decision
        or float(decision_payload.get("observed_validation_gain", float("nan")))
        != gain
        or float(decision_payload.get("threshold", float("nan")))
        != HEADROOM_THRESHOLD
        or decision_payload.get("test_labels_read") is not False
    ):
        raise ValueError("union headroom decision artifact mismatch")
    return manifest


def _replay_union_ensemble(
    run_dir: Path,
    ensemble: Mapping[str, Any],
    *,
    encoder: str,
    split: str,
) -> tuple[set[tuple[str, str]], set[str]]:
    from tools.unified_reranking import train_union_rankers as producer

    features, columns, feature_path, label_path = producer._load_development(
        run_dir, split
    )
    _train, train_columns, train_feature_path, train_label_path = (
        producer._load_development(run_dir, "train")
    )
    if columns != train_columns:
        raise ValueError("union Train/prediction feature schemas differ")
    fold_path = run_dir / "04_splits/fold_assignments.parquet"
    implementation_record = _record(Path(producer.__file__))
    identity = ensemble.get("identity")
    sources = ensemble.get("sources")
    artifacts = ensemble.get("artifacts")
    if (
        not isinstance(identity, Mapping)
        or identity.get("encoder") != encoder
        or identity.get("split") != split
        or not isinstance(sources, Mapping)
        or not isinstance(artifacts, Mapping)
    ):
        raise ValueError(f"union {encoder}/{split} ensemble identity is invalid")
    denominator_path = run_dir / f"01_manifests/paired_{split}.parquet"
    _same_record(sources.get("features"), _record(feature_path), "union features")
    _same_record(sources.get("labels"), _record(label_path), "union labels")
    _same_record(
        sources.get("denominator"), _record(denominator_path), "union denominator"
    )
    cell_records = sources.get("cells")
    expected_count = len(FORMAL_SEEDS) * (5 if split == "train" else 1)
    if not isinstance(cell_records, list) or len(cell_records) != expected_count:
        raise ValueError(f"union {encoder}/{split} cell inventory is incomplete")
    keys = ["sample_id", "candidate_id"]
    predictions = features[
        [
            *keys,
            "native_rank",
            "candidate_success",
            "base_logit",
            "source_route",
            "source_candidate_id",
            "candidate_geometry_sha256",
        ]
    ].copy()
    expected_keys = set(map(tuple, predictions[keys].astype(str).to_numpy()))
    used: set[tuple[str, str]] = set()
    configurations: set[str] = set()
    for seed in FORMAL_SEEDS:
        pieces: list[pd.DataFrame] = []
        expected_folds = set(range(5)) if split == "train" else {None}
        observed_folds: set[int | None] = set()
        for index, record in enumerate(cell_records):
            cell_path = verified_artifact_path(
                record, name=f"union {encoder}/{split} cell {index}"
            )
            cell = _load(
                cell_path,
                f"union {encoder}/{split} cell {index}",
                ("COMPLETE",),
            )
            configuration = cell.get("configuration")
            if not isinstance(configuration, Mapping):
                raise ValueError("union cell lacks configuration")
            if int(configuration.get("seed", -1)) != seed:
                continue
            fold = configuration.get("held_fold")
            if (
                configuration.get("encoder") != encoder
                or configuration.get("mode")
                != ("oof" if split == "train" else "validation")
                or fold not in expected_folds
                or fold in observed_folds
            ):
                raise ValueError("union cell configuration differs from ensemble")
            expected_configuration = {
                "pool": "primary_union_top15_no_dedup",
                "track": producer.TRACK,
                "encoder": encoder,
                "loss": (
                    "lambdarank"
                    if encoder == "lambdamart"
                    else producer.FORMAL_UNION_BUDGET.deepsets_loss
                ),
                "seed": seed,
                "mode": "oof" if split == "train" else "validation",
                "held_fold": fold,
                "early_stop_fold": (int(fold) + 1) % 5 if fold is not None else 0,
                "deterministic_cpu": True,
                "route_candidate_identity_preserved": True,
                "budget": asdict(producer.FORMAL_UNION_BUDGET),
                "sources": {
                    "train_features": _record(train_feature_path),
                    "train_labels": _record(train_label_path),
                    "prediction_features": _record(feature_path),
                    "prediction_labels": _record(label_path),
                    "folds": _record(fold_path),
                    "implementation_tool": implementation_record,
                },
            }
            if dict(configuration) != expected_configuration:
                raise ValueError("union cell differs from the canonical training contract")
            expected_cell_id = canonical_sha256(expected_configuration)[:16]
            if (
                cell.get("cell_id") != expected_cell_id
                or cell_path.parent.name != expected_cell_id
                or list(map(str, cell.get("feature_columns", []))) != list(columns)
                or cell.get("feature_schema_sha256") != canonical_sha256(columns)
                or cell.get("test_access") != "NONE"
            ):
                raise ValueError("union cell identity/schema contract mismatch")
            observed_folds.add(fold)
            verify_artifact_records_recursive(
                configuration.get("sources"),
                name=f"union {encoder}/{split} cell sources",
                require_at_least_one=True,
            )
            verify_artifact_records_recursive(
                cell.get("artifacts"),
                name=f"union {encoder}/{split} cell artifacts",
                require_at_least_one=True,
            )
            prediction_path = verified_artifact_path(
                cell["artifacts"]["predictions"], name="union cell predictions"
            )
            pieces.append(pd.read_parquet(prediction_path)[[*keys, "score"]])
            used.add((str(cell_path), sha256_file(cell_path)))
            configurations.add(
                canonical_sha256(
                    {
                        key: configuration[key]
                        for key in (
                            "encoder",
                            "seed",
                            "mode",
                            "held_fold",
                            "early_stop_fold",
                            "budget",
                        )
                    }
                )
            )
        if observed_folds != expected_folds:
            raise ValueError(f"union {encoder}/{split} seed fold inventory is incomplete")
        seed_frame = pd.concat(pieces, ignore_index=True)
        if (
            set(map(tuple, seed_frame[keys].astype(str).to_numpy())) != expected_keys
            or seed_frame[keys].duplicated().any()
        ):
            raise ValueError("union seed predictions do not exactly cover candidates")
        predictions = predictions.merge(
            seed_frame.rename(columns={"score": f"score_seed_{seed}"}),
            on=keys,
            validate="one_to_one",
        )
    score_columns = [f"score_seed_{seed}" for seed in FORMAL_SEEDS]
    predictions["ensemble_score"] = predictions[score_columns].mean(axis=1)
    stored_predictions = pd.read_parquet(
        verified_artifact_path(
            artifacts.get("predictions"), name="union ensemble predictions"
        )
    )
    pd.testing.assert_frame_equal(
        stored_predictions.sort_values(keys, kind="mergesort").reset_index(drop=True),
        predictions[stored_predictions.columns]
        .sort_values(keys, kind="mergesort")
        .reset_index(drop=True),
        check_dtype=False,
        check_exact=True,
    )
    denominator = pd.read_parquet(
        denominator_path, columns=["sample_id"]
    )["sample_id"].astype(str).tolist()
    metrics, decisions = evaluate_order_only(
        denominator, predictions, score_column="ensemble_score", max_k=15
    )
    baseline_metrics, baseline = evaluate_order_only(
        denominator, predictions, score_column="base_logit", max_k=15
    )
    comparison = compare_selections(
        baseline, decisions, oracle_at_5=float(baseline_metrics["oracle_at_5"])
    )
    metrics["oracle_at_15"] = metrics.pop("oracle_at_5")
    metrics["mrr_at_15"] = metrics.pop("mrr_at_5")
    baseline_metrics["oracle_at_15"] = baseline_metrics.pop("oracle_at_5")
    baseline_metrics["mrr_at_15"] = baseline_metrics.pop("mrr_at_5")
    comparison["headroom_recovery_at_15"] = comparison.pop(
        "headroom_recovery_at_5"
    )
    for declared, recomputed, label in (
        (ensemble.get("metrics"), metrics, "metrics"),
        (
            ensemble.get("calibrated_union_baseline_metrics"),
            baseline_metrics,
            "baseline metrics",
        ),
        (
            ensemble.get("comparison_to_calibrated_union_baseline"),
            comparison,
            "comparison",
        ),
    ):
        if canonical_sha256(declared) != canonical_sha256(recomputed):
            raise ValueError(f"union {encoder}/{split} {label} differs from replay")
    stored_decisions = pd.read_parquet(
        verified_artifact_path(artifacts.get("decisions"), name="union decisions")
    )
    pd.testing.assert_frame_equal(
        stored_decisions.sort_values("sample_id", kind="mergesort").reset_index(drop=True),
        decisions[stored_decisions.columns]
        .sort_values("sample_id", kind="mergesort")
        .reset_index(drop=True),
        check_dtype=False,
        check_exact=True,
    )
    if len(used) != expected_count:
        raise ValueError("union cell records are duplicated")
    return used, configurations


def validate_union_selection(path: Path) -> dict[str, Any]:
    """Verify all bound union trials and recompute the selected encoder."""

    path = path.resolve()
    run_dir = path.parents[2]
    selection = _load(path, "union ranker selection", ("VALIDATION_LOCKED",))
    from tools.unified_reranking import train_union_rankers as producer

    frozen_plan_path = run_dir / "configs/union_ranker_frozen_plan.json"
    _same_record(
        selection.get("formal_plan"), _record(frozen_plan_path), "union formal plan"
    )
    _same_record(
        selection.get("selection_tool"),
        _record(Path(producer.__file__)),
        "union selection tool",
    )
    frozen_plan = _load(frozen_plan_path, "union formal plan", ("FROZEN",))
    unsigned_plan = dict(frozen_plan)
    recorded_plan_hash = unsigned_plan.pop("content_sha256", None)
    if recorded_plan_hash != canonical_sha256(unsigned_plan):
        raise ValueError("union formal plan content hash mismatch")
    expected_plan = list(producer.formal_plan(producer.FORMAL_UNION_BUDGET))
    if (
        frozen_plan.get("cells") != expected_plan
        or int(frozen_plan.get("cell_count", -1)) != len(expected_plan)
        or frozen_plan.get("fixed_budget") != asdict(producer.FORMAL_UNION_BUDGET)
    ):
        raise ValueError("union formal plan differs from the fixed grid")
    _same_record(
        frozen_plan.get("sources", {}).get("implementation_tool"),
        _record(Path(producer.__file__)),
        "union formal-plan tool",
    )
    headroom_path = run_dir / "07_validation/union_headroom/manifest.json"
    _same_record(
        frozen_plan.get("sources", {}).get("headroom"),
        _record(headroom_path),
        "union formal-plan headroom",
    )
    headroom = validate_union_headroom(run_dir, headroom_path)
    if (
        frozen_plan.get("decision") != "UNION_HEADROOM_AVAILABLE"
        or headroom.get("decision") != "UNION_HEADROOM_AVAILABLE"
    ):
        raise ValueError("union formal plan lacks mandatory positive headroom")
    expected_configuration_hashes = {
        canonical_sha256(
            {
                key: cell[key]
                for key in (
                    "encoder",
                    "seed",
                    "mode",
                    "held_fold",
                    "early_stop_fold",
                    "budget",
                )
            }
        )
        for cell in expected_plan
    }
    trials = selection.get("trials")
    trial_ensembles = selection.get("trial_ensembles")
    if not isinstance(trials, list) or not trials or not isinstance(trial_ensembles, Mapping):
        raise ValueError("union selection lacks hash-bound trial ensembles")
    if set(trial_ensembles) != {str(row.get("encoder")) for row in trials}:
        raise ValueError("union trial ensemble inventory differs from trials")
    rows: list[dict[str, Any]] = []
    observed_configurations: set[str] = set()
    all_used_cells: set[tuple[str, str]] = set()
    for row in trials:
        encoder = str(row.get("encoder", ""))
        records = trial_ensembles[encoder]
        if not isinstance(records, Mapping) or set(records) != {"validation", "oof"}:
            raise ValueError("union trial lacks Validation/OOF ensemble records")
        validation_path = verified_artifact_path(records["validation"], name=f"{encoder} union Validation ensemble")
        oof_path = verified_artifact_path(records["oof"], name=f"{encoder} union OOF ensemble")
        validation = _load(validation_path, f"{encoder} union Validation ensemble", ("COMPLETE",))
        oof = _load(oof_path, f"{encoder} union OOF ensemble", ("COMPLETE",))
        validation_used, validation_configs = _replay_union_ensemble(
            run_dir, validation, encoder=encoder, split="validation"
        )
        oof_used, oof_configs = _replay_union_ensemble(
            run_dir, oof, encoder=encoder, split="train"
        )
        all_used_cells.update(validation_used)
        all_used_cells.update(oof_used)
        observed_configurations.update(validation_configs)
        observed_configurations.update(oof_configs)
        if validation.get("identity", {}).get("encoder") != encoder or oof.get("identity", {}).get("encoder") != encoder:
            raise ValueError("union trial ensemble identity mismatch")
        verify_artifact_records_recursive(validation, name=f"{encoder} union Validation ensemble", require_at_least_one=True)
        verify_artifact_records_recursive(oof, name=f"{encoder} union OOF ensemble", require_at_least_one=True)
        prediction_path = verified_artifact_path(
            validation.get("artifacts", {}).get("predictions", {}),
            name=f"{encoder} union Validation predictions",
        )
        decision_path = verified_artifact_path(
            validation.get("artifacts", {}).get("decisions", {}),
            name=f"{encoder} union Validation decisions",
        )
        denominator_path = verified_artifact_path(
            validation.get("sources", {}).get("denominator", {}),
            name=f"{encoder} union Validation denominator",
        )
        predictions = pd.read_parquet(prediction_path)
        required_columns = {
            "sample_id",
            "candidate_id",
            "native_rank",
            "candidate_success",
            "ensemble_score",
            "base_logit",
        }
        if required_columns.difference(predictions.columns) or predictions.duplicated(
            ["sample_id", "candidate_id"]
        ).any():
            raise ValueError("union Validation predictions have an invalid evaluation schema")
        denominator = pd.read_parquet(denominator_path, columns=["sample_id"])[
            "sample_id"
        ].astype(str).tolist()
        metrics, decisions = evaluate_order_only(
            denominator, predictions, score_column="ensemble_score", max_k=15
        )
        baseline_metrics, baseline = evaluate_order_only(
            denominator, predictions, score_column="base_logit", max_k=15
        )
        comparison = compare_selections(
            baseline, decisions, oracle_at_5=float(baseline_metrics["oracle_at_5"])
        )
        metrics["oracle_at_15"] = metrics.pop("oracle_at_5")
        metrics["mrr_at_15"] = metrics.pop("mrr_at_5")
        comparison["headroom_recovery_at_15"] = comparison.pop(
            "headroom_recovery_at_5"
        )
        stored_decisions = pd.read_parquet(decision_path)
        if not stored_decisions[["sample_id", "selected_candidate_id"]].equals(
            decisions[["sample_id", "selected_candidate_id"]]
        ):
            raise ValueError("union Validation decisions differ from recomputed predictions")
        for field, observed in metrics.items():
            declared = validation.get("metrics", {}).get(field)
            if isinstance(observed, (int, float)) and (
                declared is None
                or not math.isclose(
                    float(observed), float(declared), rel_tol=0.0, abs_tol=1e-12
                )
            ):
                raise ValueError("union Validation metrics differ from recomputed predictions")
        declared_comparison = validation.get(
            "comparison_to_calibrated_union_baseline", {}
        )
        for field, observed in comparison.items():
            if isinstance(observed, (int, float)) and (
                field not in declared_comparison
                or not math.isclose(
                    float(observed),
                    float(declared_comparison[field]),
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
            ):
                raise ValueError("union comparison differs from recomputed Validation predictions")
        observed = {
            "encoder": encoder,
            "validation_j_at_1": metrics.get("j_at_1"),
            "harmful": comparison.get("harmful"),
            "switch_rate": comparison.get("switch_rate"),
        }
        if observed != row:
            raise ValueError("union trial summary differs from its Validation ensemble")
        for field in ("validation_j_at_1", "harmful", "switch_rate"):
            _finite(row.get(field), f"union trial {field}")
        rows.append(dict(row))
    if (
        observed_configurations != expected_configuration_hashes
        or len(all_used_cells) != len(expected_plan)
    ):
        raise ValueError("union ensemble cells differ from the frozen formal plan")
    order = {"lambdamart": 0, "deepsets": 1}
    if set(order) != {row["encoder"] for row in rows}:
        raise ValueError("union selection must compare the two predeclared encoders")
    best = max(
        rows,
        key=lambda row: (
            row["validation_j_at_1"],
            -row["harmful"],
            -row["switch_rate"],
            -order[row["encoder"]],
        ),
    )
    if selection.get("selected_encoder") != best["encoder"]:
        raise ValueError("selected union winner is not the best bound Validation trial")
    selected_records = trial_ensembles[best["encoder"]]
    if selection.get("selected_validation_manifest") != selected_records["validation"] or selection.get("selected_oof_manifest") != selected_records["oof"]:
        raise ValueError("selected union manifest bindings differ from the winning trial")
    selected_validation = _load(
        verified_artifact_path(
            selected_records["validation"], name="selected union Validation ensemble"
        ),
        "selected union Validation ensemble",
        ("COMPLETE",),
    )
    selected_oof = _load(
        verified_artifact_path(
            selected_records["oof"], name="selected union OOF ensemble"
        ),
        "selected union OOF ensemble",
        ("COMPLETE",),
    )
    if (
        canonical_sha256(selection.get("selected_validation_ensemble"))
        != canonical_sha256(selected_validation)
        or canonical_sha256(selection.get("selected_oof_ensemble"))
        != canonical_sha256(selected_oof)
    ):
        raise ValueError("embedded selected union ensembles differ from locked manifests")
    return selection


_GATE_COLUMN_DEFAULTS = {
    "sample_id": "sample_id",
    "scene_id": "scene_id",
    "provenance": "prediction_source",
    "oof_fold": "oof_fold",
    "native_correct": "native_correct",
    "challenger_correct": "challenger_correct",
    "native_candidate_id": "native_candidate_id",
    "challenger_candidate_id": "challenger_candidate_id",
    "score_margin": "score_margin",
    "challenger_reliability": "challenger_reliability",
    "perturbation_stability": "perturbation_stability",
    "seed_challenger_votes": "seed_challenger_votes",
    "candidate_id_unchanged": "candidate_id_unchanged",
    "geometry_hash_unchanged": "geometry_hash_unchanged",
    "challenger_exists": "challenger_exists",
}


def _gate_grid(path: Path) -> tuple[GateOperatingPoint, ...]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if "operating_points" in payload:
        return tuple(GateOperatingPoint(**dict(row)) for row in payload["operating_points"])
    return tuple(
        GateOperatingPoint(lam, utility, margin, reliability, stability)
        for lam in payload["lambda_harm"]
        for utility in payload["utility_thresholds"]
        for margin in payload["score_margin_thresholds"]
        for reliability in payload["reliability_thresholds"]
        for stability in payload["stability_thresholds"]
    )


def _selection_content_signature(
    manifest: Mapping[str, Any], *, name: str
) -> None:
    unsigned = dict(manifest)
    declared = unsigned.pop("content_sha256", None)
    if declared != canonical_sha256(unsigned):
        raise ValueError(f"{name} content hash mismatch")
    if manifest.get("signature_sha256") != canonical_sha256(
        {
            "configuration": manifest.get("configuration"),
            "sources": manifest.get("sources"),
        }
    ):
        raise ValueError(f"{name} source signature mismatch")


def _assert_frame_equal(left: pd.DataFrame, right: pd.DataFrame, name: str) -> None:
    try:
        pd.testing.assert_frame_equal(
            left.reset_index(drop=True),
            right.reset_index(drop=True),
            check_dtype=True,
            check_exact=False,
            rtol=1e-12,
            atol=1e-12,
        )
    except AssertionError as error:
        raise ValueError(f"{name} differs from independent recomputation") from error


def validate_gate_input_manifest(path: Path) -> dict[str, Any]:
    """Rebuild gate Train-OOF/Validation rows from the selected ensembles."""

    from tools.unified_reranking import prepare_gate_inputs as producer

    path = path.resolve()
    run_dir = path.parents[3]
    manifest = _load(path, "gate input manifest", ("COMPLETE",))
    route = str(manifest.get("route", ""))
    if (
        route not in ROUTES
        or manifest.get("test_access") != "NONE"
        or tuple(manifest.get("feature_columns", ()))
        != tuple(producer.FEATURE_COLUMNS)
    ):
        raise ValueError("gate input manifest contract is invalid")
    sources = manifest.get("sources")
    artifacts = manifest.get("artifacts")
    if not isinstance(sources, Mapping) or set(sources) != {
        "selection",
        "oof_ensemble",
        "validation_ensemble",
    }:
        raise ValueError("gate input source inventory is not exact")
    if not isinstance(artifacts, Mapping) or set(artifacts) != {
        "train_oof",
        "validation",
    }:
        raise ValueError("gate input artifact inventory is not exact")
    selection_path = run_dir / "07_validation/selected_primary_ungated.json"
    _same_record(sources["selection"], _record(selection_path), "gate input selection")
    selection = _load(selection_path, "primary scalar selection", ("VALIDATION_LOCKED",))
    selected = selection.get("selections", {}).get(route)
    if not isinstance(selected, Mapping):
        raise ValueError("gate input route is absent from the primary selection")
    expected_oof = {
        "path": str(Path(str(selected["oof_manifest"])).resolve()),
        "sha256": str(selected["oof_manifest_sha256"]),
    }
    expected_validation = {
        "path": str(Path(str(selected["validation_manifest"])).resolve()),
        "sha256": str(selected["validation_manifest_sha256"]),
    }
    _same_record(sources["oof_ensemble"], expected_oof, "gate input OOF ensemble")
    _same_record(
        sources["validation_ensemble"],
        expected_validation,
        "gate input Validation ensemble",
    )
    oof_manifest = _load(
        verified_artifact_path(expected_oof, name="gate input OOF ensemble"),
        "gate input OOF ensemble",
        ("COMPLETE",),
    )
    validation_manifest = _load(
        verified_artifact_path(
            expected_validation, name="gate input Validation ensemble"
        ),
        "gate input Validation ensemble",
        ("COMPLETE",),
    )
    expected_train = producer._build_split(
        run_dir, route, "train", oof_manifest
    )
    expected_validation_frame = producer._build_split(
        run_dir, route, "validation", validation_manifest
    )
    observed_train = pd.read_parquet(
        verified_artifact_path(artifacts["train_oof"], name="gate Train OOF input")
    )
    observed_validation = pd.read_parquet(
        verified_artifact_path(
            artifacts["validation"], name="gate Validation input"
        )
    )
    _assert_frame_equal(observed_train, expected_train, "gate Train OOF input")
    _assert_frame_equal(
        observed_validation, expected_validation_frame, "gate Validation input"
    )
    return manifest


def validate_gate_selection(path: Path) -> dict[str, Any]:
    """Refit and reselect one conservative gate from its bound dev inputs."""

    manifest = _load(path, "gate selection", ("COMPLETE",))
    if manifest.get("test_access") != "NONE":
        raise PermissionError("gate selection is not development-only")
    _selection_content_signature(manifest, name="gate selection")
    sources = manifest.get("sources")
    artifacts = manifest.get("artifacts")
    if not isinstance(sources, Mapping) or set(sources) != {
        "input_manifest",
        "train_oof",
        "validation",
        "predeclared_grid",
        "selection_code",
    }:
        raise ValueError("gate selection source inventory is not exact")
    if not isinstance(artifacts, Mapping) or set(artifacts) != {
        "transition_model",
        "validation_decisions",
        "validation_trials",
    }:
        raise ValueError("gate selection artifact inventory is not exact")
    verify_artifact_records_recursive(sources, name="gate selection sources", require_at_least_one=True)
    verify_artifact_records_recursive(artifacts, name="gate selection artifacts", require_at_least_one=True)
    config = manifest.get("configuration")
    if not isinstance(config, Mapping):
        raise ValueError("gate selection lacks configuration")
    input_manifest_path = verified_artifact_path(
        sources["input_manifest"], name="gate input manifest"
    )
    input_manifest = validate_gate_input_manifest(input_manifest_path)
    if (
        input_manifest.get("route") != config.get("route")
        or input_manifest.get("artifacts", {}).get("train_oof") != sources["train_oof"]
        or input_manifest.get("artifacts", {}).get("validation")
        != sources["validation"]
    ):
        raise ValueError("gate selection inputs differ from their producer manifest")
    columns = dict(_GATE_COLUMN_DEFAULTS)
    columns.update(dict(config.get("columns", {})))
    features = tuple(map(str, config.get("feature_columns", [])))
    grid_path = verified_artifact_path(
        sources["predeclared_grid"], name="gate predeclared grid"
    )
    points = _gate_grid(grid_path)
    if config.get("operating_points") != [asdict(point) for point in points]:
        raise ValueError("gate configuration differs from its predeclared grid")
    if not features or not points:
        raise ValueError("gate selection lacks features or operating points")
    oof = pd.read_parquet(
        verified_artifact_path(sources["train_oof"], name="gate Train OOF")
    )
    validation = pd.read_parquet(
        verified_artifact_path(sources["validation"], name="gate Validation")
    )
    if set(oof[columns["provenance"]].astype(str).str.lower()) != {"train_oof"} or set(
        validation[columns["provenance"]].astype(str).str.lower()
    ) != {"validation"}:
        raise ValueError("gate selection input provenance mismatch")
    model = ConservativeTransitionModel(seed=int(config["model_seed"])).fit(
        OOFTransitionData(
            features=oof.loc[:, features].to_numpy(float),
            feature_names=features,
            native_correct=oof[columns["native_correct"]].to_numpy(),
            challenger_correct=oof[columns["challenger_correct"]].to_numpy(),
            scene_ids=oof[columns["scene_id"]].to_numpy(),
            oof_fold_ids=oof[columns["oof_fold"]].to_numpy(),
            prediction_source="train_oof",
        )
    )
    matrix = validation.loc[:, features].to_numpy(float)
    recover, harm = model.predict_probabilities(matrix)
    evidence = GateEvidence(
        score_margin=validation[columns["score_margin"]].to_numpy(),
        challenger_reliability=validation[columns["challenger_reliability"]].to_numpy(),
        perturbation_stability=validation[columns["perturbation_stability"]].to_numpy(),
        seed_challenger_votes=validation[columns["seed_challenger_votes"]].to_numpy(),
        candidate_id_unchanged=validation[columns["candidate_id_unchanged"]].to_numpy(),
        geometry_hash_unchanged=validation[columns["geometry_hash_unchanged"]].to_numpy(),
        challenger_exists=validation[columns["challenger_exists"]].to_numpy(),
    )
    result = select_gate_operating_point(
        recover,
        harm,
        evidence,
        validation[columns["native_correct"]].to_numpy(),
        validation[columns["challenger_correct"]].to_numpy(),
        validation[columns["scene_id"]].to_numpy(),
        points,
        bootstrap_iterations=int(config["bootstrap_iterations"]),
        bootstrap_seed=int(config["bootstrap_seed"]),
    )
    if result.selected_operating_point is None:
        switches = np.zeros(len(validation), dtype=bool)
        utility = np.full(len(validation), np.nan)
    else:
        switches = gate_switch_mask(recover, harm, evidence, result.selected_operating_point)
        utility = recover - result.selected_operating_point.lambda_harm * harm
    native_correct = validation[columns["native_correct"]].astype(bool).to_numpy()
    challenger_correct = validation[columns["challenger_correct"]].astype(bool).to_numpy()
    native_ids = validation[columns["native_candidate_id"]].fillna("").astype(str).to_numpy()
    challenger_ids = validation[columns["challenger_candidate_id"]].fillna("").astype(str).to_numpy()
    decisions = pd.DataFrame(
        {
            "sample_id": validation[columns["sample_id"]].astype(str),
            "scene_id": validation[columns["scene_id"]].astype(str),
            "probability_recover": recover,
            "probability_harm": harm,
            "utility": utility,
            "switch": switches,
            "native_candidate_id": native_ids,
            "challenger_candidate_id": challenger_ids,
            "selected_candidate_id": np.where(switches, challenger_ids, native_ids),
            "native_correct": native_correct,
            "challenger_correct": challenger_correct,
            "selected_correct": np.where(switches, challenger_correct, native_correct),
            "transition": np.select(
                [~native_correct & challenger_correct, native_correct & ~challenger_correct],
                ["recovered_if_switched", "harmful_if_switched"],
                default="outcome_unchanged",
            ),
        }
    )
    trials = pd.DataFrame(
        [
            {
                **asdict(trial.operating_point),
                "bootstrap_lower_bound": trial.bootstrap_lower_bound,
                "mean_delta": trial.mean_delta,
                "recovered": trial.recovered,
                "harmful": trial.harmful,
                "switch_count": trial.switch_count,
                "switch_rate": trial.switch_rate,
            }
            for trial in result.trials
        ]
    )
    _assert_frame_equal(
        pd.read_parquet(verified_artifact_path(artifacts["validation_decisions"], name="gate decisions")),
        decisions,
        "gate Validation decisions",
    )
    _assert_frame_equal(
        pd.read_parquet(verified_artifact_path(artifacts["validation_trials"], name="gate trials")),
        trials,
        "gate Validation trials",
    )
    with verified_artifact_path(artifacts["transition_model"], name="gate model").open("rb") as stream:
        stored_model = pickle.load(stream)
    if not isinstance(stored_model, ConservativeTransitionModel):
        raise ValueError("gate model artifact has an unexpected type")
    stored_probabilities = stored_model.predict_probabilities(matrix)
    if (
        not np.allclose(stored_probabilities[0], recover, rtol=0.0, atol=1e-12)
        or not np.allclose(stored_probabilities[1], harm, rtol=0.0, atol=1e-12)
        or stored_model.artifact() != model.artifact()
        or manifest.get("transition_model") != model.artifact()
        or manifest.get("selection") != result.artifact()
        or manifest.get("decision") != result.status
    ):
        raise ValueError("gate model/selection differs from independent recomputation")
    return manifest


_ROUTER_COLUMN_DEFAULTS = {
    "sample_id": "sample_id",
    "scene_id": "scene_id",
    "provenance": "prediction_source",
    "oof_fold": "oof_fold",
    "crog_correct": "crog_correct",
    "g1_correct": "g1_correct",
    "c1_correct": "c1_correct",
    "crog_candidate_id": "crog_candidate_id",
    "g1_candidate_id": "g1_candidate_id",
    "c1_candidate_id": "c1_candidate_id",
    "g1_margin": "g1_margin",
    "c1_margin": "c1_margin",
    "g1_reliability": "g1_reliability",
    "c1_reliability": "c1_reliability",
    "g1_stability": "g1_stability",
    "c1_stability": "c1_stability",
    "g1_candidate_exists": "g1_candidate_exists",
    "c1_candidate_exists": "c1_candidate_exists",
}


def _router_grid(path: Path) -> tuple[RouterOperatingPoint, ...]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if "operating_points" in payload:
        return tuple(RouterOperatingPoint(**dict(row)) for row in payload["operating_points"])
    return tuple(
        RouterOperatingPoint(lam, utility, margin, reliability, stability)
        for lam in payload["lambda_router"]
        for utility in payload["utility_thresholds"]
        for margin in payload["margin_thresholds"]
        for reliability in payload["reliability_thresholds"]
        for stability in payload["stability_thresholds"]
    )


def validate_router_input_manifest(path: Path) -> dict[str, Any]:
    """Rebuild paired router rows from the exact gate/ranker lineage."""

    from tools.unified_reranking import prepare_route_router_inputs as producer

    path = path.resolve()
    run_dir = path.parents[2]
    manifest = _load(path, "route-router input manifest", ("COMPLETE",))
    configuration = manifest.get("configuration")
    sources = manifest.get("sources")
    artifacts = manifest.get("artifacts")
    if (
        manifest.get("test_access") != "NONE"
        or not isinstance(configuration, Mapping)
        or not isinstance(sources, Mapping)
        or not isinstance(artifacts, Mapping)
        or set(artifacts) != {"train_oof", "validation"}
        or manifest.get("signature_sha256")
        != canonical_sha256({"configuration": configuration, "sources": sources})
    ):
        raise ValueError("route-router input manifest contract is invalid")
    verify_artifact_records_recursive(
        sources, name="route-router input sources", require_at_least_one=True
    )
    gate_values: dict[str, dict[str, Any]] = {}
    expected_source_names = {"implementation_tool", "implementation_primitives"}
    for route in ROUTES:
        gate_path = run_dir / f"08_lock/gates/{route}/gate_selection.json"
        gate_record = sources.get(f"{route}_gate_selection")
        _same_record(gate_record, _record(gate_path), f"router input {route} gate")
        gate_values[route] = validate_gate_selection(gate_path)
        expected_source_names.add(f"{route}_gate_selection")
        for split in ("train", "validation"):
            expected = {
                f"{route}_{split}_gate_input": run_dir
                / f"07_validation/gate_inputs/{route}/"
                / ("train_oof.parquet" if split == "train" else "validation.parquet"),
                f"{route}_{split}_candidates": run_dir
                / f"02_candidates/{route}_{split}_top5.parquet",
                f"{route}_{split}_features": run_dir
                / f"03_features/tracks/T2_matched_common/{route}_{split}/candidate_features.parquet",
            }
            for name, expected_path in expected.items():
                expected_source_names.add(name)
                _same_record(
                    sources.get(name), _record(expected_path), f"router input {name}"
                )
    if set(sources) != expected_source_names:
        raise ValueError("route-router input source inventory is not exact")
    _same_record(
        sources["implementation_tool"],
        _record(Path(producer.__file__)),
        "route-router input implementation",
    )
    _same_record(
        sources["implementation_primitives"],
        _record(Path(producer.ROOT) / "src/unified_reranking/cross_route_inputs.py"),
        "route-router input primitives",
    )
    expected_audits: dict[str, Any] = {}
    for split, artifact_key in (("train", "train_oof"), ("validation", "validation")):
        by_route: dict[str, pd.DataFrame] = {}
        for route in ROUTES:
            by_route[route], audits = producer._development_gated_route(
                run_dir, route, split, gate_values[route]
            )
            if audits:
                expected_audits[route] = list(audits)
        expected_frame = producer._finalize_router_columns(
            producer.add_router_features(by_route)
        )
        observed = pd.read_parquet(
            verified_artifact_path(
                artifacts[artifact_key], name=f"router {artifact_key} input"
            )
        )
        _assert_frame_equal(observed, expected_frame, f"router {artifact_key} input")
    if manifest.get("outer_gate_cross_fit_audit") != expected_audits:
        raise ValueError("route-router outer cross-fit audit differs from replay")
    expected_points = {
        route: producer.operating_point_audit(
            producer.gate_operating_point_from_manifest(gate_values[route])
        )
        for route in ROUTES
    }
    if manifest.get("validation_gate_operating_points") != expected_points:
        raise ValueError("route-router gate operating points differ from replay")
    return manifest


def validate_router_selection(path: Path) -> dict[str, Any]:
    """Refit and reselect the CROG-default router from its bound dev inputs."""

    manifest = _load(path, "route-router selection", ("COMPLETE",))
    if manifest.get("test_access") != "NONE":
        raise PermissionError("route-router selection is not development-only")
    _selection_content_signature(manifest, name="route-router selection")
    sources = manifest.get("sources")
    artifacts = manifest.get("artifacts")
    if not isinstance(sources, Mapping) or set(sources) != {
        "input_manifest",
        "paired_train_oof",
        "paired_validation",
        "predeclared_grid",
        "selection_code",
    }:
        raise ValueError("route-router source inventory is not exact")
    if not isinstance(artifacts, Mapping) or set(artifacts) != {
        "transition_models",
        "validation_decisions",
        "validation_trials",
    }:
        raise ValueError("route-router artifact inventory is not exact")
    verify_artifact_records_recursive(sources, name="route-router sources", require_at_least_one=True)
    verify_artifact_records_recursive(artifacts, name="route-router artifacts", require_at_least_one=True)
    config = manifest.get("configuration")
    if not isinstance(config, Mapping) or config.get("default_route") != "CROG" or list(
        config.get("tie_break", [])
    ) != ["G1", "C1"]:
        raise ValueError("route-router configuration is not CROG-default")
    input_manifest_path = verified_artifact_path(
        sources["input_manifest"], name="route-router input manifest"
    )
    input_manifest = validate_router_input_manifest(input_manifest_path)
    if (
        input_manifest.get("artifacts", {}).get("train_oof")
        != sources["paired_train_oof"]
        or input_manifest.get("artifacts", {}).get("validation")
        != sources["paired_validation"]
    ):
        raise ValueError("route-router selection inputs differ from their manifest")
    columns = dict(_ROUTER_COLUMN_DEFAULTS)
    columns.update(dict(config.get("columns", {})))
    feature_columns = {
        route: tuple(map(str, config.get("feature_columns", {}).get(route, [])))
        for route in ("G1", "C1")
    }
    grid_path = verified_artifact_path(
        sources["predeclared_grid"], name="route-router predeclared grid"
    )
    points = _router_grid(grid_path)
    if config.get("operating_points") != [asdict(point) for point in points]:
        raise ValueError("route-router configuration differs from its predeclared grid")
    if any(not values for values in feature_columns.values()) or not points:
        raise ValueError("route-router lacks features or operating points")
    oof = pd.read_parquet(
        verified_artifact_path(sources["paired_train_oof"], name="router Train OOF")
    )
    validation = pd.read_parquet(
        verified_artifact_path(sources["paired_validation"], name="router Validation")
    )
    if set(oof[columns["provenance"]].astype(str).str.lower()) != {"train_oof"} or set(
        validation[columns["provenance"]].astype(str).str.lower()
    ) != {"validation"}:
        raise ValueError("route-router input provenance mismatch")

    def transition_data(route: str, outcome: str) -> OOFTransitionData:
        return OOFTransitionData(
            features=oof.loc[:, feature_columns[route]].to_numpy(float),
            feature_names=feature_columns[route],
            native_correct=oof[columns["crog_correct"]].to_numpy(),
            challenger_correct=oof[columns[outcome]].to_numpy(),
            scene_ids=oof[columns["scene_id"]].to_numpy(),
            oof_fold_ids=oof[columns["oof_fold"]].to_numpy(),
            prediction_source="train_oof",
        )

    router = CROGDefaultTransitionRouter(seed=int(config["model_seed"])).fit(
        transition_data("G1", "g1_correct"),
        transition_data("C1", "c1_correct"),
    )
    validation_features = {
        route: validation.loc[:, names].to_numpy(float)
        for route, names in feature_columns.items()
    }
    probabilities = router.predict_probabilities(validation_features)
    evidence = {
        "G1": RouterEvidence(
            route_margin=validation[columns["g1_margin"]].to_numpy(),
            reliability=validation[columns["g1_reliability"]].to_numpy(),
            perturbation_stability=validation[columns["g1_stability"]].to_numpy(),
            candidate_exists=validation[columns["g1_candidate_exists"]].to_numpy(),
        ),
        "C1": RouterEvidence(
            route_margin=validation[columns["c1_margin"]].to_numpy(),
            reliability=validation[columns["c1_reliability"]].to_numpy(),
            perturbation_stability=validation[columns["c1_stability"]].to_numpy(),
            candidate_exists=validation[columns["c1_candidate_exists"]].to_numpy(),
        ),
    }
    correct = {
        route: validation[columns[f"{route.lower()}_correct"]].to_numpy()
        for route in ("CROG", "G1", "C1")
    }
    result = select_router_operating_point(
        probabilities,
        evidence,
        correct,
        validation[columns["scene_id"]].to_numpy(),
        points,
        tie_break=("G1", "C1"),
        bootstrap_iterations=int(config["bootstrap_iterations"]),
        bootstrap_seed=int(config["bootstrap_seed"]),
    )
    if result.selected_operating_point is None:
        selected_routes = np.full(len(validation), "CROG", dtype=object)
        utilities = {route: np.full(len(validation), np.nan) for route in ("G1", "C1")}
    else:
        selected_routes = route_decisions(
            probabilities, evidence, result.selected_operating_point, tie_break=("G1", "C1")
        )
        utilities = route_utilities(
            probabilities, lambda_router=result.selected_operating_point.lambda_router
        )
    candidate_ids = {
        route: validation[columns[f"{route.lower()}_candidate_id"]].fillna("").astype(str).to_numpy()
        for route in ("CROG", "G1", "C1")
    }
    correct_bool = {route: np.asarray(values, dtype=bool) for route, values in correct.items()}
    indexes = np.arange(len(validation))
    selected_candidate = np.asarray(
        [candidate_ids[str(route)][index] for index, route in zip(indexes, selected_routes, strict=True)],
        dtype=object,
    )
    selected_correct = np.asarray(
        [correct_bool[str(route)][index] for index, route in zip(indexes, selected_routes, strict=True)],
        dtype=bool,
    )
    decisions = pd.DataFrame(
        {
            "sample_id": validation[columns["sample_id"]].astype(str),
            "scene_id": validation[columns["scene_id"]].astype(str),
            "g1_probability_recover": probabilities["G1"][0],
            "g1_probability_harm": probabilities["G1"][1],
            "g1_utility": utilities["G1"],
            "c1_probability_recover": probabilities["C1"][0],
            "c1_probability_harm": probabilities["C1"][1],
            "c1_utility": utilities["C1"],
            "selected_route": selected_routes,
            "switched_from_crog": selected_routes != "CROG",
            "crog_candidate_id": candidate_ids["CROG"],
            "g1_candidate_id": candidate_ids["G1"],
            "c1_candidate_id": candidate_ids["C1"],
            "selected_candidate_id": selected_candidate,
            "crog_correct": correct_bool["CROG"],
            "g1_correct": correct_bool["G1"],
            "c1_correct": correct_bool["C1"],
            "selected_correct": selected_correct,
        }
    )
    trials = pd.DataFrame(
        [
            {
                **asdict(trial.operating_point),
                "bootstrap_lower_bound": trial.bootstrap_lower_bound,
                "mean_delta": trial.mean_delta,
                "recovered": trial.recovered,
                "harmful": trial.harmful,
                "switch_count": trial.switch_count,
                "switch_rate": trial.switch_rate,
                "g1_switches": trial.g1_switches,
                "c1_switches": trial.c1_switches,
            }
            for trial in result.trials
        ]
    )
    _assert_frame_equal(
        pd.read_parquet(verified_artifact_path(artifacts["validation_decisions"], name="router decisions")),
        decisions,
        "route-router Validation decisions",
    )
    _assert_frame_equal(
        pd.read_parquet(verified_artifact_path(artifacts["validation_trials"], name="router trials")),
        trials,
        "route-router Validation trials",
    )
    with verified_artifact_path(artifacts["transition_models"], name="router models").open("rb") as stream:
        stored_router = pickle.load(stream)
    if not isinstance(stored_router, CROGDefaultTransitionRouter):
        raise ValueError("route-router model artifact has an unexpected type")
    stored_probabilities = stored_router.predict_probabilities(validation_features)
    if any(
        not np.allclose(stored_probabilities[route][index], probabilities[route][index], rtol=0.0, atol=1e-12)
        for route in ("G1", "C1")
        for index in (0, 1)
    ) or stored_router.artifact() != router.artifact() or manifest.get(
        "transition_models"
    ) != router.artifact() or manifest.get("selection") != result.artifact() or manifest.get(
        "decision"
    ) != result.status:
        raise ValueError("route-router model/selection differs from independent recomputation")
    return manifest


def validate_application_signature(
    manifest: Mapping[str, Any], *, kind: str
) -> None:
    """Recompute the producer-defined label-free Test application signature."""

    sources = manifest.get("sources")
    if not isinstance(sources, Mapping):
        raise ValueError(f"{kind} Test application lacks source bindings")
    verify_artifact_records_recursive(sources, name=f"{kind} Test application sources", require_at_least_one=True)
    if kind == "ranker":
        payload = {"configuration": manifest.get("configuration"), "sources": sources}
    elif kind == "gate":
        payload = {
            "route": manifest.get("route"),
            "feature_columns": manifest.get("feature_columns"),
            "operating_point": manifest.get("selected_operating_point"),
            "sources": sources,
        }
    elif kind in {"router", "union"}:
        payload = {"configuration": manifest.get("configuration"), "sources": sources}
    else:
        raise ValueError(f"unknown Test application kind: {kind}")
    if manifest.get("signature_sha256") != canonical_sha256(payload):
        raise ValueError(f"{kind} Test application signature mismatch")


def validate_gate_application_outputs(
    path: Path,
    selection: Mapping[str, Any],
) -> dict[str, Any]:
    """Recompute a label-free Test gate's inputs and decisions."""

    from tools.unified_reranking.apply_locked_test_gates import (
        FEATURE_COLUMNS,
        build_label_free_test_gate_inputs,
    )

    application = _load(path, "gate Test application", ("COMPLETE",))
    route = str(application.get("route", "")).lower()
    denominator_path = verified_artifact_path(
        application.get("sources", {}).get("sample_denominator", {}),
        name="gate Test denominator",
    )
    run_dir = denominator_path.parent.parent
    recomputed_inputs = build_label_free_test_gate_inputs(run_dir, route)
    stored_inputs = pd.read_parquet(
        verified_artifact_path(
            application.get("artifacts", {}).get("inputs", {}),
            name="gate Test inputs",
        )
    )
    _assert_frame_equal(stored_inputs, recomputed_inputs, "gate Test inputs")
    model_path = verified_artifact_path(
        selection.get("artifacts", {}).get("transition_model", {}),
        name="gate transition model",
    )
    with model_path.open("rb") as stream:
        model = pickle.load(stream)
    if not isinstance(model, ConservativeTransitionModel):
        raise ValueError("gate Test application model has an unexpected type")
    matrix = recomputed_inputs.loc[:, FEATURE_COLUMNS].to_numpy(float)
    recover, harm = model.predict_probabilities(matrix)
    point_payload = selection.get("selection", {}).get("selected_operating_point")
    if point_payload is None:
        switches = np.zeros(len(recomputed_inputs), dtype=bool)
        utility = np.full(len(recomputed_inputs), np.nan)
    else:
        point = GateOperatingPoint(**dict(point_payload))
        evidence = GateEvidence(
            score_margin=recomputed_inputs["score_margin"].to_numpy(),
            challenger_reliability=recomputed_inputs[
                "challenger_reliability"
            ].to_numpy(),
            perturbation_stability=recomputed_inputs[
                "perturbation_stability"
            ].to_numpy(),
            seed_challenger_votes=recomputed_inputs[
                "seed_challenger_votes"
            ].to_numpy(),
            candidate_id_unchanged=recomputed_inputs[
                "candidate_id_unchanged"
            ].to_numpy(),
            geometry_hash_unchanged=recomputed_inputs[
                "geometry_hash_unchanged"
            ].to_numpy(),
            challenger_exists=recomputed_inputs["challenger_exists"].to_numpy(),
        )
        switches = gate_switch_mask(recover, harm, evidence, point)
        utility = recover - point.lambda_harm * harm
    native = recomputed_inputs["native_candidate_id"].astype(str).to_numpy()
    challenger = recomputed_inputs["challenger_candidate_id"].astype(str).to_numpy()
    decisions = pd.DataFrame(
        {
            "sample_id": recomputed_inputs["sample_id"].astype(str),
            "prediction_source": "test_label_free",
            "probability_recover": recover,
            "probability_harm": harm,
            "utility": utility,
            "switch": switches,
            "native_candidate_id": native,
            "challenger_candidate_id": challenger,
            "selected_candidate_id": np.where(switches, challenger, native),
        }
    )
    stored_decisions = pd.read_parquet(
        verified_artifact_path(
            application.get("artifacts", {}).get("decisions", {}),
            name="gate Test decisions",
        )
    )
    _assert_frame_equal(stored_decisions, decisions, "gate Test decisions")
    return application


def validate_router_application_outputs(
    path: Path,
    selection: Mapping[str, Any],
) -> dict[str, Any]:
    """Recompute a label-free Test CROG-default router decision table."""

    application = _load(path, "router Test application", ("COMPLETE",))
    sources = application.get("sources", {})
    input_path = verified_artifact_path(
        sources.get("test_router_inputs", {}), name="router Test inputs"
    )
    frame = pd.read_parquet(input_path)
    if set(frame["prediction_source"].astype(str)) != {"test_label_free"}:
        raise ValueError("router Test input provenance mismatch")
    model_path = verified_artifact_path(
        sources.get("router_models", {}), name="router Test model"
    )
    with model_path.open("rb") as stream:
        router = pickle.load(stream)
    if not isinstance(router, CROGDefaultTransitionRouter):
        raise ValueError("router Test application model has an unexpected type")
    configuration = selection.get("configuration", {})
    feature_columns = {
        route: tuple(map(str, configuration.get("feature_columns", {}).get(route, [])))
        for route in ("G1", "C1")
    }
    probabilities = router.predict_probabilities(
        {
            route: frame.loc[:, columns].to_numpy(float)
            for route, columns in feature_columns.items()
        }
    )
    evidence = {
        "G1": RouterEvidence(
            route_margin=frame["g1_margin"].to_numpy(),
            reliability=frame["g1_reliability"].to_numpy(),
            perturbation_stability=frame["g1_stability"].to_numpy(),
            candidate_exists=frame["g1_candidate_exists"].to_numpy(),
        ),
        "C1": RouterEvidence(
            route_margin=frame["c1_margin"].to_numpy(),
            reliability=frame["c1_reliability"].to_numpy(),
            perturbation_stability=frame["c1_stability"].to_numpy(),
            candidate_exists=frame["c1_candidate_exists"].to_numpy(),
        ),
    }
    point_payload = selection.get("selection", {}).get("selected_operating_point")
    if selection.get("decision") == "NO_GO_CROG" and point_payload is None:
        selected_routes = np.full(len(frame), "CROG", dtype=object)
        utilities = {route: np.zeros(len(frame), dtype=float) for route in ("G1", "C1")}
    elif selection.get("decision") == "GO" and isinstance(point_payload, Mapping):
        point = RouterOperatingPoint(**dict(point_payload))
        selected_routes = route_decisions(
            probabilities, evidence, point, tie_break=("G1", "C1")
        )
        utilities = route_utilities(
            probabilities, lambda_router=point.lambda_router
        )
    else:
        raise ValueError("router Test application has no valid locked decision")
    route_ids = {
        route: frame[f"{route.lower()}_candidate_id"].fillna("").astype(str).to_numpy()
        for route in ("CROG", "G1", "C1")
    }
    route_hashes = {
        route: frame[f"{route.lower()}_selected_candidate_geometry_sha256"]
        .fillna("")
        .astype(str)
        .to_numpy()
        for route in ("CROG", "G1", "C1")
    }
    indexes = np.arange(len(frame))
    selected_ids = np.asarray(
        [
            route_ids[str(route)][index]
            for index, route in zip(indexes, selected_routes, strict=True)
        ],
        dtype=object,
    )
    selected_hashes = np.asarray(
        [
            route_hashes[str(route)][index]
            for index, route in zip(indexes, selected_routes, strict=True)
        ],
        dtype=object,
    )
    decisions = pd.DataFrame(
        {
            "sample_id": frame["sample_id"].astype(str),
            "scene_id": frame["scene_id"].astype(str),
            "prediction_source": "test_label_free",
            "g1_probability_recover": probabilities["G1"][0],
            "g1_probability_harm": probabilities["G1"][1],
            "g1_utility": utilities["G1"],
            "c1_probability_recover": probabilities["C1"][0],
            "c1_probability_harm": probabilities["C1"][1],
            "c1_utility": utilities["C1"],
            "selected_route": selected_routes,
            "switched_from_crog": selected_routes != "CROG",
            "crog_candidate_id": route_ids["CROG"],
            "g1_candidate_id": route_ids["G1"],
            "c1_candidate_id": route_ids["C1"],
            "selected_candidate_id": selected_ids,
            "selected_candidate_geometry_sha256": selected_hashes,
        }
    )
    stored = pd.read_parquet(
        verified_artifact_path(
            application.get("artifacts", {}).get("decisions", {}),
            name="router Test decisions",
        )
    )
    _assert_frame_equal(stored, decisions, "router Test decisions")
    return application


def validate_union_application_outputs(
    path: Path,
    selection: Mapping[str, Any],
) -> dict[str, Any]:
    """Recompute all label-free Top-15 union scores and decisions."""

    from tools.unified_reranking.apply_locked_union_ranker import _predict_cell

    application = _load(path, "union Test application", ("COMPLETE",))
    sources = application.get("sources", {})
    features = pd.read_parquet(
        verified_artifact_path(sources.get("test_features", {}), name="union Test features")
    )
    predictions = features[
        [
            "sample_id",
            "candidate_id",
            "native_rank",
            "source_route",
            "source_candidate_id",
            "candidate_geometry_sha256",
        ]
    ].copy()
    cells: dict[int, Mapping[str, Any]] = {}
    for record in dict(sources.get("cells", {})).values():
        cell_path = verified_artifact_path(record, name="union Test selected cell")
        cell = _load(cell_path, "union Test selected cell", ("COMPLETE",))
        seed = int(cell.get("configuration", {}).get("seed", -1))
        if seed in cells:
            raise ValueError("union Test application contains duplicate seed cells")
        cells[seed] = cell
    if set(cells) != set(FORMAL_SEEDS):
        raise ValueError("union Test application lacks the three selected seed cells")
    for seed in FORMAL_SEEDS:
        scores = _predict_cell(dict(cells[seed]), features).rename(
            columns={"score": f"score_seed_{seed}"}
        )
        predictions = predictions.merge(
            scores,
            on=["sample_id", "candidate_id"],
            validate="one_to_one",
        )
    score_columns = [f"score_seed_{seed}" for seed in FORMAL_SEEDS]
    predictions["ensemble_score"] = predictions[score_columns].mean(axis=1)
    denominator = pd.read_parquet(
        verified_artifact_path(sources.get("denominator", {}), name="union Test denominator"),
        columns=["sample_id"],
    )["sample_id"].astype(str).tolist()
    decisions = select_order_only(
        denominator, predictions, score_column="ensemble_score"
    )
    selected = predictions[
        [
            "sample_id",
            "candidate_id",
            "source_route",
            "source_candidate_id",
            "candidate_geometry_sha256",
        ]
    ].rename(columns={"candidate_id": "selected_candidate_id"})
    decisions = decisions.merge(
        selected,
        on=["sample_id", "selected_candidate_id"],
        how="left",
        validate="one_to_one",
    )
    decisions["prediction_source"] = "test_label_free"
    stored_predictions = pd.read_parquet(
        verified_artifact_path(
            application.get("artifacts", {}).get("predictions", {}),
            name="union Test predictions",
        )
    )
    stored_decisions = pd.read_parquet(
        verified_artifact_path(
            application.get("artifacts", {}).get("decisions", {}),
            name="union Test decisions",
        )
    )
    _assert_frame_equal(stored_predictions, predictions, "union Test predictions")
    _assert_frame_equal(stored_decisions, decisions, "union Test decisions")
    if (
        application.get("sample_count") != len(decisions)
        or application.get("candidate_rows") != len(predictions)
        or application.get("configuration", {}).get("encoder")
        != selection.get("selected_encoder")
    ):
        raise ValueError("union Test application summary differs from recomputation")
    return application


__all__ = [
    "validate_application_signature",
    "validate_encoder_execution",
    "validate_feature_ablation_manifest",
    "validate_gate_application_outputs",
    "validate_gate_selection",
    "validate_router_application_outputs",
    "validate_router_selection",
    "validate_scalar_selection",
    "validate_union_application_outputs",
    "validate_union_headroom",
    "validate_union_selection",
]
