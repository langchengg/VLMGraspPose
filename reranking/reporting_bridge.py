"""Bridge matrix candidate predictions into the strict reporting contract.

The training matrix stores one score per frozen candidate.  Reporting consumes
one auditable row per query.  This module performs that conversion by ID join,
creates one isolated reporting bundle per route/pool, and then aggregates only
machine-generated artifacts into the formal run root.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from reranking.data_contracts import MODULAR_CANONICAL_RUN, streaming_sha256
from reranking.evaluate import evaluate_rankings, validate_oracle_invariance
from reranking.independent_evaluator import independent_j_at_1
from reranking.matrix import (
    DatasetArtifact,
    MatrixError,
    _baseline_predictions,
    _discover_datasets,
    _load_joined,
    _manifest_rows,
    _query_universe,
)
from reranking.matrix_reporting import reliability_bins, summarize_outcomes
from reranking.report import REPORT_NAMES, run_reporting_stage
from reranking.visualize import FIGURE_NAMES, build_galleries, build_visualizations


REPORTING_STAGES = ("statistics", "visualize", "report")

GALLERY_FEATURE_COLUMNS = (
    "p_center",
    "p_contact_min",
    "p_contact_imbalance",
    "grasp_axis_mask_support",
    "mask_entropy_local",
    "normalized_width_mismatch",
    "candidate_uniqueness",
    "valid_depth_support",
    "approach_clearance",
    "collision_proxy_total",
    "border_clearance_px",
    "axis_valid_ratio",
)


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, (float, np.floating)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(child) for child in value]
    return str(value)


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(_json_safe(value), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def _atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def _read_candidate_predictions(path: str | os.PathLike[str]) -> pd.DataFrame:
    source = Path(path)
    if source.is_symlink() or not source.is_file():
        raise MatrixError(f"candidate prediction artifact missing: {source}")
    frame = pd.read_parquet(source)
    required = {"query_id", "candidate_id", "score"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise MatrixError(f"candidate prediction artifact lacks {missing}: {source}")
    return frame[["query_id", "candidate_id", "score"]].copy()


def candidate_predictions_to_query_outcomes(
    artifact: DatasetArtifact,
    frame: pd.DataFrame,
    reference: pd.DataFrame,
    baseline_predictions: pd.DataFrame,
    selected_predictions: pd.DataFrame,
    *,
    experiment_id: str,
    gate_id: str = "G0",
) -> pd.DataFrame:
    """Create the explicit reporting outcome contract from a frozen pool."""

    universe = _query_universe(artifact, reference)
    baseline = evaluate_rankings(
        reference, baseline_predictions, query_universe=universe
    )["per_query"]
    selected = evaluate_rankings(
        reference, selected_predictions, query_universe=universe
    )["per_query"]
    validate_oracle_invariance(baseline, selected)
    selected_projection = selected[
        ["query_id", "j_at_1", "top_candidate_id", "oracle"]
    ].rename(
        columns={
            "j_at_1": "selected_correct",
            "top_candidate_id": "selected_candidate_id",
            "oracle": "selected_oracle",
        }
    )
    outcome = baseline.rename(
        columns={
            "query_id": "sample_id",
            "j_at_1": "baseline_correct",
            "top_candidate_id": "baseline_candidate_id",
            "positive_candidate_count": "positive_count",
            "oracle": "baseline_oracle",
        }
    ).merge(
        selected_projection.rename(columns={"query_id": "sample_id"}),
        on="sample_id",
        how="inner",
        validate="one_to_one",
    )
    if len(outcome) != len(universe):
        raise MatrixError("query outcome conversion changed the registered universe")
    if not outcome["baseline_oracle"].astype(int).equals(
        outcome["selected_oracle"].astype(int)
    ):
        raise MatrixError("reranking changed the frozen-pool oracle")
    outcome["oracle"] = outcome["baseline_oracle"].astype(np.int8)
    outcome["first_positive_rank"] = (
        pd.to_numeric(outcome["first_positive_rank"], errors="coerce")
        .fillna(0)
        .astype(np.int64)
    )
    for column in (
        "candidate_count",
        "positive_count",
        "baseline_correct",
        "selected_correct",
    ):
        outcome[column] = pd.to_numeric(outcome[column], errors="raise").astype(
            np.int64
        )
    outcome["baseline_candidate_id"] = outcome["baseline_candidate_id"].fillna("")
    outcome["selected_candidate_id"] = outcome["selected_candidate_id"].fillna("")
    q_lookup = {
        (str(query), str(candidate)): float(q)
        for query, candidate, q in frame[
            ["query_id", "candidate_id", "q_raw"]
        ].itertuples(index=False, name=None)
    }
    outcome["baseline_q"] = [
        q_lookup.get((str(query), str(candidate)), np.nan)
        for query, candidate in outcome[
            ["sample_id", "baseline_candidate_id"]
        ].itertuples(index=False, name=None)
    ]
    outcome["selected_q"] = [
        q_lookup.get((str(query), str(candidate)), np.nan)
        for query, candidate in outcome[
            ["sample_id", "selected_candidate_id"]
        ].itertuples(index=False, name=None)
    ]
    outcome["q_drop"] = outcome["baseline_q"] - outcome["selected_q"]
    scored = selected_predictions[["query_id", "candidate_id", "score"]].copy()
    scored["query_id"] = scored["query_id"].astype(str)
    scored["candidate_id"] = scored["candidate_id"].astype(str)
    scored["score"] = pd.to_numeric(scored["score"], errors="raise")
    scored = scored.sort_values(
        ["query_id", "score", "candidate_id"],
        ascending=[True, False, True],
        kind="mergesort",
    )
    scored["rerank_rank"] = scored.groupby("query_id", sort=False).cumcount() + 1
    visual = frame.merge(
        scored,
        on=["query_id", "candidate_id"],
        how="left",
        validate="one_to_one",
    )
    if visual["score"].isna().any():
        raise MatrixError("gallery candidate pool lacks reranker scores")
    payload_by_query: dict[str, str] = {}
    feature_delta_by_query: dict[str, str] = {}
    for query_id, group in visual.groupby("query_id", sort=False):
        rows: list[dict[str, Any]] = []
        for candidate in group.sort_values(
            ["original_rank", "candidate_id"], kind="mergesort"
        ).to_dict(orient="records"):
            record = {
                "candidate_id": str(candidate["candidate_id"]),
                "q": float(candidate["q_raw"]),
                "original_rank": int(candidate["original_rank"]),
                "rerank_score": float(candidate["score"]),
                "rerank_rank": int(candidate["rerank_rank"]),
                "correct": bool(candidate["label"]),
                "x_px": float(candidate.get("x_px", 0.0)),
                "y_px": float(candidate.get("y_px", 0.0)),
                "z_m": float(candidate.get("z_m", 0.0)),
                "angle_rad": float(candidate.get("angle_rad", 0.0)),
                "width_px": float(candidate.get("width_px", 0.0)),
                "height_px": float(
                    candidate.get(
                        "height_px", candidate.get("rectangle_height_px", 20.0)
                    )
                ),
                "features": {},
            }
            for name in GALLERY_FEATURE_COLUMNS:
                if name not in candidate:
                    continue
                try:
                    value = float(candidate[name])
                except (TypeError, ValueError):
                    continue
                if np.isfinite(value):
                    record["features"][name] = value
            rows.append(record)
        payload_by_query[str(query_id)] = json.dumps(
            rows, sort_keys=True, separators=(",", ":"), allow_nan=False
        )
        by_id = {row["candidate_id"]: row for row in rows}
        baseline_id = str(
            outcome.loc[
                outcome["sample_id"].astype(str).eq(str(query_id)),
                "baseline_candidate_id",
            ].iloc[0]
        )
        selected_id = str(
            outcome.loc[
                outcome["sample_id"].astype(str).eq(str(query_id)),
                "selected_candidate_id",
            ].iloc[0]
        )
        before = by_id.get(baseline_id, {}).get("features", {})
        after = by_id.get(selected_id, {}).get("features", {})
        deltas = [
            {
                "feature": name,
                "baseline": float(before[name]),
                "selected": float(after[name]),
                "delta": float(after[name] - before[name]),
            }
            for name in sorted(set(before) & set(after))
        ]
        deltas.sort(key=lambda row: (-abs(row["delta"]), row["feature"]))
        feature_delta_by_query[str(query_id)] = json.dumps(
            deltas[:8], sort_keys=True, separators=(",", ":"), allow_nan=False
        )
    outcome["candidate_pool_json"] = outcome["sample_id"].astype(str).map(
        payload_by_query
    ).fillna("[]")
    outcome["feature_delta_json"] = outcome["sample_id"].astype(str).map(
        feature_delta_by_query
    ).fillna("[]")
    outcome["gate_id"] = str(gate_id)
    outcome["switch_applied"] = outcome["baseline_candidate_id"].astype(str).ne(
        outcome["selected_candidate_id"].astype(str)
    )

    metadata_columns = tuple(
        column
        for column in (
            "image_path",
            "depth_path",
            "language_instruction",
            "predicted_mask_path",
            "gt_mask_path",
        )
        if column in frame.columns
    )
    if metadata_columns:
        assets = (
            frame[["query_id", *metadata_columns]]
            .drop_duplicates("query_id")
            .rename(columns={"query_id": "sample_id"})
        )
        outcome = outcome.merge(assets, on="sample_id", how="left", validate="one_to_one")
    if artifact.route == "modular" and artifact.split == "test":
        manifest_path = MODULAR_CANONICAL_RUN / "input_manifest.csv"
        if manifest_path.is_file():
            manifest = pd.read_csv(manifest_path)
            projection = {"native_rgb_path": "image_path", "native_depth_path": "depth_path", "query": "language_instruction", "gt_mask_path": "gt_mask_path"}
            available = [name for name in projection if name in manifest.columns]
            assets = manifest[["sample_id", *available]].rename(columns=projection)
            assets["sample_id"] = assets["sample_id"].astype(str)
            assets["predicted_mask_path"] = [
                str(MODULAR_CANONICAL_RUN / "masks" / "hierfilm" / "bundles" / sample_id / "target_mask.png")
                for sample_id in assets["sample_id"]
            ]
            outcome = outcome.merge(
                assets,
                on="sample_id",
                how="left",
                validate="one_to_one",
                suffixes=("", "__canonical"),
            )
            for destination in (
                "image_path",
                "depth_path",
                "language_instruction",
                "predicted_mask_path",
                "gt_mask_path",
            ):
                canonical = f"{destination}__canonical"
                if canonical not in outcome:
                    continue
                if destination not in outcome:
                    outcome[destination] = outcome[canonical]
                else:
                    missing = outcome[destination].isna() | outcome[
                        destination
                    ].astype(str).str.strip().eq("")
                    outcome.loc[missing, destination] = outcome.loc[
                        missing, canonical
                    ]
                outcome = outcome.drop(columns=[canonical])
    outcome["failure_stage"] = np.select(
        [
            outcome["candidate_count"].eq(0),
            outcome["positive_count"].eq(0),
            outcome["first_positive_rank"].gt(5),
            outcome["baseline_correct"].eq(1) & outcome["selected_correct"].eq(0),
            outcome["baseline_correct"].eq(0) & outcome["oracle"].eq(1),
        ],
        ["F3", "F4", "F5", "F7", "F6"],
        default="NONE",
    )
    outcome["experiment_id"] = str(experiment_id)
    outcome["dataset"] = artifact.key
    columns = [
        "sample_id",
        "scene_id",
        "frame_id",
        "experiment_id",
        "dataset",
        "baseline_correct",
        "selected_correct",
        "oracle",
        "baseline_oracle",
        "selected_oracle",
        "baseline_candidate_id",
        "selected_candidate_id",
        "candidate_count",
        "positive_count",
        "first_positive_rank",
        "baseline_q",
        "selected_q",
        "q_drop",
        "candidate_pool_json",
        "feature_delta_json",
        "gate_id",
        "switch_applied",
        "failure_stage",
    ]
    columns.extend(
        column
        for column in (
            "image_path",
            "depth_path",
            "language_instruction",
            "predicted_mask_path",
            "gt_mask_path",
        )
        if column in outcome.columns
    )
    return outcome[columns].sort_values("sample_id", kind="mergesort").reset_index(
        drop=True
    )


def _test_manifests(output: Path, dataset: str) -> list[dict[str, Any]]:
    return [
        row
        for row in _manifest_rows(output)
        if row.get("status") == "COMPLETE"
        and row.get("dataset") == dataset
        and row.get("stage") in {"test-primary", "test-post-lock"}
        # Exclude only the separately emitted comparator.  If validation locks
        # the q-only method itself, its LOCKED_PRIMARY_TEST artifact remains a
        # legitimate primary prediction and must reach the reporting stage.
        and row.get("prediction_designation")
        != "LOCKED_PRIMARY_TEST_BASELINE"
    ]


def _validation_rows(output: Path, dataset: str) -> list[dict[str, Any]]:
    path = output / "metrics" / "all_validation_results.json"
    if not path.is_file():
        raise MatrixError(f"validation results missing: {path}")
    rows = json.loads(path.read_text(encoding="utf-8")).get("results", [])
    selected = [dict(row) for row in rows if row.get("dataset") == dataset]
    if not selected:
        raise MatrixError(f"validation registry has no rows for {dataset}")
    return selected


def _prepare_reporting_view(
    output: Path,
    artifact: DatasetArtifact,
    selected: Mapping[str, Any],
) -> tuple[Path, str, list[Path]]:
    view = output / "reporting_inputs" / artifact.key
    (view / "manifests").mkdir(parents=True, exist_ok=True)
    (view / "metrics").mkdir(parents=True, exist_ok=True)
    (view / "predictions").mkdir(parents=True, exist_ok=True)
    frame, reference, _, _ = _load_joined(artifact)
    baseline = _baseline_predictions(frame)
    manifests = _test_manifests(output, artifact.key)
    if not manifests:
        raise MatrixError(f"no test predictions registered for {artifact.key}")
    primary_matches = [
        row
        for row in manifests
        if row.get("stage") == "test-primary"
        and row.get("method") == selected.get("method")
        and row.get("gate") == selected.get("gate")
    ]
    if len(primary_matches) != 1:
        raise MatrixError(
            f"expected one locked-primary test manifest for {artifact.key}, "
            f"found {len(primary_matches)}"
        )
    primary_id = str(primary_matches[0]["experiment_id"])
    artifacts: list[Path] = []
    registry_rows: list[dict[str, Any]] = []
    for manifest in sorted(manifests, key=lambda row: str(row["experiment_id"])):
        experiment_id = str(manifest["experiment_id"])
        candidate_predictions = _read_candidate_predictions(manifest["prediction_path"])
        outcomes = candidate_predictions_to_query_outcomes(
            artifact,
            frame,
            reference,
            baseline,
            candidate_predictions,
            experiment_id=experiment_id,
            gate_id=str(manifest.get("gate", "G0")),
        )
        outcome_path = view / "predictions" / f"{experiment_id}.parquet"
        temporary = outcome_path.with_name(f".{outcome_path.name}.{os.getpid()}.tmp")
        outcomes.to_parquet(temporary, index=False, compression="zstd")
        os.replace(temporary, outcome_path)
        artifacts.append(outcome_path)
        registry_rows.append(
            {
                "experiment_id": experiment_id,
                "dataset": artifact.key,
                "route": artifact.route,
                "pool": artifact.pool,
                "method": manifest["method"],
                "gate": manifest.get("gate", "G0"),
                "stage": manifest["stage"],
                "designation": manifest.get("prediction_designation", ""),
                "prediction_path": str(outcome_path.resolve()),
            }
        )
    lock_path = view / "manifests" / "PRIMARY_METHOD_LOCK.json"
    _atomic_json(
        lock_path,
        {
            "locked_before_test": True,
            "test_used_for_selection": False,
            "primary_experiment_id": primary_id,
            "baseline_experiment_id": "r0_q_baseline",
            "dataset": artifact.key,
            "source_lock": str(
                (output / "manifests" / "PRIMARY_METHOD_LOCK.json").resolve()
            ),
        },
    )
    validation_path = view / "metrics" / "validation_registry.json"
    test_path = view / "metrics" / "test_registry.json"
    _atomic_json(validation_path, {"results": _validation_rows(output, artifact.key)})
    _atomic_json(test_path, {"results": registry_rows})
    artifacts.extend((lock_path, validation_path, test_path))
    return view, primary_id, artifacts


def _copy_table_bundles(
    output: Path, bundles: Sequence[tuple[str, Path]]
) -> list[Path]:
    outputs: list[Path] = []
    for relative in (
        "statistics/mcnemar_results.csv",
        "statistics/bootstrap_intervals.csv",
        "statistics/holm_corrected_results.csv",
        "metrics/multi_seed_summary.csv",
    ):
        parts = []
        for dataset, bundle in bundles:
            frame = pd.read_csv(bundle / relative)
            frame.insert(0, "dataset", dataset)
            parts.append(frame)
        destination = output / relative
        _atomic_csv(destination, pd.concat(parts, ignore_index=True))
        outputs.append(destination)
    summaries = {
        dataset: json.loads(
            (bundle / "metrics" / "primary_summary.json").read_text(encoding="utf-8")
        )
        for dataset, bundle in bundles
    }
    primary_path = output / "metrics" / "primary_summary.json"
    existing = (
        json.loads(primary_path.read_text(encoding="utf-8"))
        if primary_path.is_file()
        else {}
    )
    existing["reporting_by_dataset"] = summaries
    existing["two_dimensional_consistency_only"] = True
    _atomic_json(primary_path, existing)
    outputs.append(primary_path)
    return outputs


def _ablation_tables(output: Path) -> list[Path]:
    path = output / "metrics" / "all_validation_results.json"
    rows = json.loads(path.read_text(encoding="utf-8"))["results"]
    frame = pd.DataFrame(rows)
    method = frame.get("method", pd.Series("", index=frame.index)).astype(str)
    gate = frame.get("gate", pd.Series("", index=frame.index)).astype(str)
    tables = {
        "feature_ablation.csv": frame.loc[
            method.str.contains("feature|baseline_only|plus_|minus_|mask_input")
        ],
        "loss_ablation.csv": frame.loc[
            method.str.contains("bce|ranknet|listwise")
        ],
        "encoder_ablation.csv": frame.loc[
            method.str.contains("linear|mlp|deepsets|gnn|transformer")
        ],
        "gate_ablation.csv": frame.loc[gate.ne("")],
        "pool_ablation.csv": frame.loc[
            frame.get("pool", pd.Series("", index=frame.index)).astype(str).ne("")
        ],
    }
    pool_status = output / "data" / "candidate_pool_status.json"
    if pool_status.is_file():
        availability = pd.DataFrame(
            json.loads(pool_status.read_text(encoding="utf-8")).get("pools", [])
        )
        if len(availability):
            availability["record_type"] = "pool_availability"
            tables["pool_ablation.csv"] = pd.concat(
                [tables["pool_ablation.csv"], availability],
                ignore_index=True,
                sort=False,
            )
    outputs = []
    for name, table in tables.items():
        destination = output / "metrics" / name
        _atomic_csv(destination, table)
        outputs.append(destination)
    return outputs


def _run_statistics(output: Path) -> list[Path]:
    lock_path = output / "manifests" / "PRIMARY_METHOD_LOCK.json"
    if not lock_path.is_file():
        raise MatrixError("primary lock missing before reporting")
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    selected = {str(row["dataset"]): row for row in lock.get("primaries", [])}
    bundles: list[tuple[str, Path]] = []
    outputs: list[Path] = []
    for artifact in _discover_datasets(output, "test"):
        if artifact.key not in selected:
            raise MatrixError(f"locked primary missing for {artifact.key}")
        view, _, view_artifacts = _prepare_reporting_view(
            output, artifact, selected[artifact.key]
        )
        outputs.extend(view_artifacts)
        bundle = output / "reporting_bundles" / artifact.key
        manifest = bundle / "reporting_manifest.json"
        if not manifest.is_file():
            if bundle.exists():
                raise MatrixError(
                    f"incomplete reporting bundle exists; preserve for audit: {bundle}"
                )
            run_reporting_stage(view, bundle)
        bundles.append((artifact.key, bundle))
        outputs.append(manifest)
    outputs.extend(_copy_table_bundles(output, bundles))
    outputs.extend(_ablation_tables(output))
    bundle_index = output / "statistics" / "reporting_bundle_index.json"
    _atomic_json(
        bundle_index,
        {
            "bundles": [
                {
                    "dataset": dataset,
                    "path": str(bundle.resolve()),
                    "manifest_sha256": streaming_sha256(
                        bundle / "reporting_manifest.json"
                    ),
                }
                for dataset, bundle in bundles
            ],
            "test_used_for_primary_selection": False,
            "two_dimensional_consistency_only": True,
        },
    )
    outputs.append(bundle_index)
    return outputs


def _aggregate_outcomes(output: Path) -> pd.DataFrame:
    rows = []
    lock = json.loads(
        (output / "manifests" / "PRIMARY_METHOD_LOCK.json").read_text(
            encoding="utf-8"
        )
    )
    for selected in lock["primaries"]:
        dataset = str(selected["dataset"])
        view = output / "reporting_inputs" / dataset
        view_lock = json.loads(
            (view / "manifests" / "PRIMARY_METHOD_LOCK.json").read_text(
                encoding="utf-8"
            )
        )
        primary_id = str(view_lock["primary_experiment_id"])
        frame = pd.read_parquet(view / "predictions" / f"{primary_id}.parquet")
        frame["sample_id"] = dataset + "/" + frame["sample_id"].astype(str)
        frame["scene_id"] = dataset + "/" + frame["scene_id"].astype(str)
        rows.append(frame)
    if not rows:
        raise MatrixError("no primary outcome artifacts found")
    return pd.concat(rows, ignore_index=True)


def _run_visualize(output: Path) -> list[Path]:
    bundle_index = output / "statistics" / "reporting_bundle_index.json"
    if not bundle_index.is_file():
        raise MatrixError("statistics must complete before visualization")
    validation = pd.read_csv(output / "metrics" / "all_validation_results.csv")
    outcomes = _aggregate_outcomes(output)
    bootstrap_rows = pd.read_csv(
        output / "statistics" / "bootstrap_intervals.csv"
    ).to_dict(orient="records")
    summary = summarize_outcomes(outcomes)
    (output / "figures").mkdir(parents=True, exist_ok=True)
    (output / "galleries").mkdir(parents=True, exist_ok=True)
    aggregate = output / "reporting_aggregate"
    figures_dir = aggregate / "figures"
    galleries_dir = aggregate / "galleries"
    if not figures_dir.exists():
        build_visualizations(
            figures_dir,
            validation_registry=validation,
            primary_outcomes=outcomes,
            reliability=reliability_bins(outcomes),
            bootstrap_rows=bootstrap_rows,
            primary_summary=summary,
        )
    if not galleries_dir.exists():
        build_galleries(galleries_dir, outcomes)
    outputs: list[Path] = []
    for name in FIGURE_NAMES:
        for suffix in ("png", "pdf"):
            source = figures_dir / f"{name}.{suffix}"
            destination = output / "figures" / source.name
            shutil.copy2(source, destination)
            outputs.append(destination)
    for source in galleries_dir.rglob("*"):
        if not source.is_file():
            continue
        destination = output / "galleries" / source.relative_to(galleries_dir)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        outputs.append(destination)
    return outputs


def _finalize_sanity_audit(output: Path) -> tuple[Path, Path]:
    json_path = output / "audit" / "SANITY_AUDIT.json"
    md_path = output / "audit" / "SANITY_AUDIT.md"
    if not json_path.is_file():
        raise MatrixError("development sanity audit is missing")
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    by_dataset = {
        str(item["dataset"]): item for item in payload.get("datasets", [])
    }
    lock = json.loads(
        (output / "manifests" / "PRIMARY_METHOD_LOCK.json").read_text(
            encoding="utf-8"
        )
    )
    selected = {str(item["dataset"]): item for item in lock["primaries"]}
    manifests = _manifest_rows(output)
    for artifact in _discover_datasets(output, "test"):
        audit = by_dataset.get(artifact.key)
        if audit is None:
            raise MatrixError(f"sanity audit lacks dataset {artifact.key}")
        primary = selected[artifact.key]
        matches = [
            item
            for item in manifests
            if item.get("status") == "COMPLETE"
            and item.get("stage") == "test-primary"
            and item.get("dataset") == artifact.key
            and item.get("method") == primary["method"]
            and item.get("gate") == primary["gate"]
            and item.get("prediction_designation") == "LOCKED_PRIMARY_TEST"
        ]
        if len(matches) != 1:
            raise MatrixError(
                f"independent evaluator expected one locked primary: {artifact.key}"
            )
        manifest = matches[0]
        frame, reference, _, _ = _load_joined(artifact)
        universe = _query_universe(artifact, reference)
        predictions = _read_candidate_predictions(manifest["prediction_path"])
        independent = independent_j_at_1(reference, predictions, universe)
        recorded = manifest.get("metrics", {})
        matched = bool(
            independent["j_at_1_count"] == recorded.get("j_at_1_count")
            and abs(
                independent["j_at_1"] - float(recorded.get("j_at_1", np.nan))
            )
            <= 1e-15
            and independent["oracle_count"] == recorded.get("oracle_count")
        )
        if not matched:
            raise MatrixError(
                f"independent primary metric recomputation disagrees: {artifact.key}"
            )
        audit["independent_test_evaluator"] = {
            "status": "PASS",
            "experiment_id": manifest["experiment_id"],
            "independent": independent,
            "recorded": {
                key: recorded.get(key)
                for key in ("j_at_1_count", "j_at_1", "oracle_count", "oracle")
            },
            "exact_match": True,
        }

        development_queries_path = (
            output / "data" / f"split_query_assignments_{artifact.key}.parquet"
        )
        if not development_queries_path.is_file():
            raise MatrixError(
                f"development split assignments missing: {development_queries_path}"
            )
        development_ids = set(
            pd.read_parquet(development_queries_path)["query_id"].astype(str)
        )
        test_ids = set(universe["query_id"].astype(str))
        overlap = sorted(development_ids & test_ids)
        fit_scope_failures = [
            item["experiment_id"]
            for item in manifests
            if item.get("status") == "COMPLETE"
            and item.get("stage") == "train"
            and item.get("dataset") == artifact.key
            and item.get("fit_scope")
            not in {"development_train_fold_only", "none_fixed_method"}
        ]
        if overlap or fit_scope_failures:
            raise MatrixError(f"test-fit scanner failed for {artifact.key}")
        audit["test_fit_scanner"] = {
            "passed": True,
            "development_query_count": len(development_ids),
            "test_query_count": len(test_ids),
            "query_id_overlap": overlap,
            "fit_scope_failures": fit_scope_failures,
        }

        checks = []
        for query_id in sorted(test_ids)[:20]:
            feature_ids = set(
                frame.loc[
                    frame["query_id"].astype(str).eq(query_id), "candidate_id"
                ].astype(str)
            )
            label_ids = set(
                reference.loc[
                    reference["query_id"].astype(str).eq(query_id), "candidate_id"
                ].astype(str)
            )
            prediction_ids = set(
                predictions.loc[
                    predictions["query_id"].astype(str).eq(query_id),
                    "candidate_id",
                ].astype(str)
            )
            checks.append(
                {
                    "query_id": query_id,
                    "candidate_count": len(feature_ids),
                    "feature_label_prediction_id_match": (
                        feature_ids == label_ids == prediction_ids
                    ),
                }
            )
        if not all(row["feature_label_prediction_id_match"] for row in checks):
            raise MatrixError(f"test candidate ID cohort failed: {artifact.key}")
        audit["candidate_id_join_test_cohort"] = {
            "requested_samples": 20,
            "checked_samples": len(checks),
            "rows": checks,
            "passed": True,
        }

    gallery_summary_path = output / "galleries" / "gallery_summary.json"
    if not gallery_summary_path.is_file():
        raise MatrixError("visual sanity audit requires gallery_summary.json")
    gallery_summary = json.loads(gallery_summary_path.read_text(encoding="utf-8"))
    required_categories = (
        "recovered",
        "harmful",
        "unchanged",
        "bothwrong",
        "empty",
        "no-positive",
        "rank>5",
    )
    visual = {
        "status": "COMPLETE_WITH_EXPLICIT_SHORTFALLS",
        "required_categories": list(required_categories),
        "categories": {
            name: gallery_summary.get(name, {}) for name in required_categories
        },
        "note": (
            "Deterministic cases were selected for every eligible requested category; "
            "zero-eligible and unavailable-image shortfalls remain explicit."
        ),
    }
    for audit in by_dataset.values():
        audit["visual_inspection"] = visual
    payload["stage"] = "post_test_complete"
    payload["finalized_at"] = pd.Timestamp.now(tz="UTC").isoformat()
    payload["all_mandatory_checks_executed"] = True
    _atomic_json(json_path, payload)

    lines = [
        "# Sanity audit",
        "",
        "Status: all mandatory checks executed. Numeric pass/fail results and "
        "all explicit qualitative shortfalls are preserved in `SANITY_AUDIT.json`.",
        "",
    ]
    for dataset, audit in sorted(by_dataset.items()):
        order_pass = all(
            row.get("equivalent")
            for row in audit["candidate_order_permutation"].values()
        )
        lines.extend(
            [
                f"## {dataset}",
                "",
                f"- Candidate-order permutation: {'PASS' if order_pass else 'FAIL'}",
                f"- GT leakage scanner: {'PASS' if audit['gt_leakage_scanner']['passed'] else 'FAIL'}",
                f"- Test-fit scanner: {'PASS' if audit['test_fit_scanner']['passed'] else 'FAIL'}",
                f"- Candidate geometry hash: {'PASS' if audit['candidate_geometry_invariance']['passed'] else 'FAIL'}",
                f"- Oracle@5 invariance: {'PASS' if audit['oracle_at_5_invariance']['passed'] else 'FAIL'}",
                f"- Independent evaluator: {audit['independent_test_evaluator']['status']}",
                f"- Test candidate ID checks: {audit['candidate_id_join_test_cohort']['checked_samples']}",
                f"- Visual inspection: {audit['visual_inspection']['status']}",
                "",
            ]
        )
    _atomic_text(md_path, "\n".join(lines))
    return json_path, md_path


def _run_report(output: Path) -> list[Path]:
    index = json.loads(
        (output / "statistics" / "reporting_bundle_index.json").read_text(
            encoding="utf-8"
        )
    )
    outputs: list[Path] = list(_finalize_sanity_audit(output))
    (output / "reports").mkdir(parents=True, exist_ok=True)
    for report_name in REPORT_NAMES:
        sections = [
            "# Combined two-route reranking evidence",
            "",
            "Every success value below is frozen 2D annotation consistency, not physical grasp success.",
        ]
        for row in index["bundles"]:
            dataset = str(row["dataset"])
            source = Path(row["path"]) / "reports" / report_name
            sections.extend(
                [
                    "",
                    f"## Dataset: {dataset}",
                    "",
                    source.read_text(encoding="utf-8"),
                ]
            )
        destination = output / "reports" / report_name
        _atomic_text(destination, "\n".join(sections).rstrip() + "\n")
        outputs.append(destination)
    checksum_path = output / "checksums.sha256"
    checksum_roots = (
        "audit",
        "configs",
        "data",
        "features",
        "figures",
        "galleries",
        "manifests",
        "metrics",
        "predictions",
        "reports",
        "statistics",
    )
    files = sorted(
        path
        for name in checksum_roots
        for path in (output / name).rglob("*")
        if path.is_file() and not path.is_symlink()
    )
    lines = [
        f"{streaming_sha256(path)}  {path.relative_to(output).as_posix()}"
        for path in files
    ]
    _atomic_text(checksum_path, "\n".join(lines) + "\n")
    outputs.append(checksum_path)
    return outputs


def run_reporting_bridge(stage: str, output: str | os.PathLike[str]) -> list[str]:
    if stage not in REPORTING_STAGES:
        raise ValueError(stage)
    root = Path(output).resolve()
    if stage == "statistics":
        paths = _run_statistics(root)
    elif stage == "visualize":
        paths = _run_visualize(root)
    else:
        paths = _run_report(root)
    missing = [str(path) for path in paths if not path.is_file() or path.stat().st_size == 0]
    if missing:
        raise MatrixError(f"reporting stage {stage} has missing/empty artifacts: {missing}")
    return [str(path.resolve()) for path in paths]


__all__ = [
    "REPORTING_STAGES",
    "candidate_predictions_to_query_outcomes",
    "run_reporting_bridge",
]
