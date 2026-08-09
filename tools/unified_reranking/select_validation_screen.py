"""Summarise seed-42 Validation screening and freeze matched-budget finalists."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from unified_reranking.artifacts import (
    verified_artifact_path,
    verify_artifact_records_recursive,
)
from unified_reranking.hashing import atomic_json, canonical_sha256, sha256_file
from unified_reranking.ledger import ledger_stage
from unified_reranking.metrics import compare_selections, evaluate_order_only
from unified_reranking.matrix_phase import load_matrix_phase_cells


PARAMETER_NAMES = (
    "learning_rate",
    "weight_decay",
    "alpha",
    "temperature",
    "beta",
    "epochs",
    "patience",
    "batch_size",
    "num_leaves",
    "tree_learning_rate",
    "n_estimators",
    "num_attention_blocks",
)


def method_code(configuration: dict[str, object], *, rule: bool) -> str:
    if rule:
        return f"R1_{configuration['method']}"
    encoder, loss = str(configuration["encoder"]), str(configuration["loss"])
    if encoder == "linear":
        return f"R2_linear_{loss}"
    if encoder == "lambdamart":
        return "R6_lambdamart"
    if encoder == "mlp":
        return {
            "bce": "R3_mlp_bce",
            "ranknet": "R4_mlp_ranknet",
            "listwise": "R5_mlp_listwise",
            "jacquard_margin_ranknet": "R7_mlp_jacquard_margin",
        }[loss]
    return f"secondary_{encoder}_{loss}"


def _atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def _native_decisions(run_dir: Path, route: str) -> tuple[dict[str, object], pd.DataFrame]:
    labels = pd.read_parquet(
        run_dir / "03_features" / f"candidate_labels_{route}_validation_top5.parquet"
    )
    candidates = pd.read_parquet(
        run_dir / "02_candidates" / f"{route}_validation_top5.parquet",
        columns=["sample_id", "candidate_id", "native_rank"],
    )
    evaluation = candidates.merge(
        labels[["sample_id", "candidate_id", "candidate_success"]],
        on=["sample_id", "candidate_id"],
        validate="one_to_one",
    )
    evaluation["native_control_score"] = -evaluation["native_rank"].astype(float)
    denominator = pd.read_parquet(
        run_dir / "01_manifests" / "paired_validation.parquet", columns=["sample_id"]
    )["sample_id"].astype(str).tolist()
    return evaluate_order_only(denominator, evaluation, score_column="native_control_score")


def _assert_reported_metrics(
    reported: object, recomputed: dict[str, object], manifest_path: Path
) -> None:
    if not isinstance(reported, dict) or canonical_sha256(reported) != canonical_sha256(
        recomputed
    ):
        raise RuntimeError(
            f"Validation cell metrics do not match verified predictions: {manifest_path}"
        )


def _recompute_validation_cell(
    run_dir: Path, value: dict[str, object], manifest_path: Path
) -> tuple[dict[str, object], pd.DataFrame]:
    configuration = value["configuration"]
    route = str(configuration["route"])
    candidate_path = (
        run_dir / "02_candidates" / f"{route}_validation_top5.parquet"
    )
    label_path = (
        run_dir
        / "03_features"
        / f"candidate_labels_{route}_validation_top5.parquet"
    )
    denominator_path = run_dir / "01_manifests" / "paired_validation.parquet"
    candidates = pd.read_parquet(
        candidate_path, columns=["sample_id", "candidate_id", "native_rank"]
    )
    labels = pd.read_parquet(
        label_path, columns=["sample_id", "candidate_id", "candidate_success"]
    )
    prediction_path = verified_artifact_path(
        value["artifacts"]["predictions"],
        name=f"Validation screen predictions {manifest_path}",
    )
    predictions = pd.read_parquet(prediction_path)
    evaluation = candidates.merge(
        labels,
        on=["sample_id", "candidate_id"],
        validate="one_to_one",
    ).merge(
        predictions[["sample_id", "candidate_id", "score"]],
        on=["sample_id", "candidate_id"],
        validate="one_to_one",
    )
    denominator = pd.read_parquet(
        denominator_path, columns=["sample_id"]
    )["sample_id"].astype(str).tolist()
    metrics, decisions = evaluate_order_only(
        denominator, evaluation, score_column="score"
    )
    _assert_reported_metrics(value.get("metrics"), metrics, manifest_path)
    return metrics, decisions


def _verified_current_cell(
    value: dict[str, object], path: Path, *, is_rule: bool
) -> bool:
    """Return whether a COMPLETE cell is bound to current sources.

    Source drift makes a historical cell ineligible.  Output drift is treated
    as corruption and fails closed instead of silently changing a selection.
    """

    configuration = value.get("configuration")
    if not isinstance(configuration, dict) or not isinstance(
        configuration.get("source_identity"), dict
    ):
        return False
    try:
        verify_artifact_records_recursive(
            value.get("sources", {}),
            name=f"matrix screen sources {path}",
            require_at_least_one=True,
        )
    except RuntimeError:
        return False
    verify_artifact_records_recursive(
        value.get("artifacts", {}),
        name=f"matrix screen outputs {path}",
        require_at_least_one=True,
    )
    cell_key = str(value.get("cell_key", ""))
    if not cell_key or cell_key != canonical_sha256(configuration)[:16]:
        raise RuntimeError(f"cell key/configuration binding mismatch: {path}")
    if path.parent.name != cell_key:
        raise RuntimeError(f"cell directory/key binding mismatch: {path}")
    columns = tuple(map(str, value.get("feature_columns", ())))
    if not columns or value.get("feature_schema_sha256") != canonical_sha256(columns):
        raise RuntimeError(f"cell feature schema binding mismatch: {path}")
    expected_encoder = "rule" if is_rule else configuration.get("encoder")
    if is_rule and configuration.get("method") is None:
        raise RuntimeError(f"rule cell misses method identity: {path}")
    if expected_encoder is None:
        raise RuntimeError(f"matrix cell misses encoder identity: {path}")
    return True


def _scan(run_dir: Path) -> list[tuple[dict[str, object], Path, bool]]:
    values: list[tuple[dict[str, object], Path, bool]] = []
    for value, path in load_matrix_phase_cells(run_dir, "screen"):
        configuration = value.get("configuration", {})
        is_rule = configuration.get("encoder") == "rule" or "method" in configuration
        if (
            configuration.get("mode") != "validation"
            or int(configuration.get("seed", -1)) != 42
            or not _verified_current_cell(value, path, is_rule=is_rule)
        ):
            raise RuntimeError(f"matrix screen contains an ineligible cell: {path}")
        values.append((value, path, is_rule))
    return values


def _selection_entry(
    configuration: dict[str, object], manifest_path: Path
) -> dict[str, object]:
    return {
        "encoder": configuration["encoder"],
        "loss": configuration["loss"],
        "parameters": {
            name: configuration[name] for name in PARAMETER_NAMES if name in configuration
        },
        "screen_cell_key": configuration.get("cell_key"),
        "screen_source_identity": configuration["source_identity"],
        "screen_manifest": str(manifest_path.resolve()),
        "screen_manifest_sha256": sha256_file(manifest_path),
    }


def run(run_dir: Path) -> dict[str, object]:
    screen_execution_path = (
        run_dir / "05_models/matrix_plans/screen_latest_execution.json"
    )
    manifests = _scan(run_dir)
    if not manifests:
        raise RuntimeError("no completed seed-42 Validation screen cells found")
    native_cache: dict[str, tuple[dict[str, object], pd.DataFrame]] = {}
    rows: list[dict[str, object]] = []
    manifest_by_cell: dict[str, tuple[dict[str, object], Path]] = {}
    for value, path, is_rule in manifests:
        configuration = value["configuration"]
        route, track = str(configuration["route"]), str(configuration["track"])
        if route not in native_cache:
            native_cache[route] = _native_decisions(run_dir, route)
        native_metrics, native = native_cache[route]
        metrics, decisions = _recompute_validation_cell(run_dir, value, path)
        comparison = compare_selections(
            native,
            decisions,
            oracle_at_5=float(native_metrics["oracle_at_5"]),
        )
        code = method_code(configuration, rule=is_rule)
        cell_key = str(value["cell_key"])
        manifest_by_cell[cell_key] = (value, path)
        rows.append(
            {
                "route": route,
                "track": track,
                "method_code": code,
                "cell_key": cell_key,
                "j_at_1": float(metrics["j_at_1"]),
                "mrr_at_5": float(metrics["mrr_at_5"]),
                "ndcg_at_5": float(metrics["ndcg_at_5"]),
                **comparison,
                "configuration_json": json.dumps(configuration, sort_keys=True),
                "manifest_path": str(path.resolve()),
                "manifest_sha256": sha256_file(path),
            }
        )
    table = pd.DataFrame(rows)
    table = table.sort_values(
        ["route", "track", "method_code", "j_at_1", "harmful", "switch_rate", "mrr_at_5", "cell_key"],
        ascending=[True, True, True, False, True, True, False, True],
        kind="mergesort",
    ).reset_index(drop=True)
    table["best_within_method"] = ~table.duplicated(["route", "track", "method_code"])
    winners = table.loc[table["best_within_method"]].copy()
    finalists: dict[str, list[dict[str, object]]] = {}
    encoder_losses: dict[str, dict[str, object]] = {}
    finalist_codes = {"R4_mlp_ranknet", "R6_lambdamart", "R7_mlp_jacquard_margin"}
    neural_codes = {"R3_mlp_bce", "R4_mlp_ranknet", "R5_mlp_listwise", "R7_mlp_jacquard_margin"}
    for (route, track), group in winners.groupby(["route", "track"], sort=True):
        key = f"{route}/{track}"
        chosen_finalists = group.loc[group["method_code"].isin(finalist_codes)].copy()
        if set(chosen_finalists["method_code"]) != finalist_codes:
            missing = sorted(finalist_codes.difference(chosen_finalists["method_code"]))
            raise RuntimeError(f"screen misses primary finalist families for {key}: {missing}")
        finalists[key] = []
        for row in chosen_finalists.sort_values("method_code").itertuples(index=False):
            cell, cell_path = manifest_by_cell[row.cell_key]
            configuration = cell["configuration"]
            entry = _selection_entry(configuration, cell_path)
            entry["method_code"] = row.method_code
            entry["screen_cell_key"] = row.cell_key
            finalists[key].append(entry)
        neural = group.loc[group["method_code"].isin(neural_codes)].sort_values(
            ["j_at_1", "harmful", "switch_rate", "mrr_at_5", "method_code"],
            ascending=[False, True, True, False, True],
            kind="mergesort",
        )
        if neural.empty:
            raise RuntimeError(f"screen misses controlled MLP losses for {key}")
        best_neural = neural.iloc[0]
        cell, cell_path = manifest_by_cell[str(best_neural["cell_key"])]
        configuration = cell["configuration"]
        encoder_losses[key] = _selection_entry(configuration, cell_path)
        encoder_losses[key]["method_code"] = str(best_neural["method_code"])
        encoder_losses[key]["screen_cell_key"] = str(best_neural["cell_key"])

    output = run_dir / "07_validation" / "tables"
    table_path = output / "screen_all_trials.csv"
    winner_path = output / "screen_best_within_method.csv"
    _atomic_csv(table_path, table)
    _atomic_csv(winner_path, winners)
    finalist_path = run_dir / "05_models" / "screen_finalists.json"
    encoder_path = run_dir / "05_models" / "encoder_loss_selections.json"
    atomic_json(
        finalist_path,
        {
            "status": "VALIDATION_SCREEN_LOCKED",
            "selection_rule": "max J@1; then fewer harmful, lower switch rate, higher MRR@5, stable method/cell key",
            "eligible_primary_families": sorted(finalist_codes),
            "selections": finalists,
            "screen_execution": {
                "path": str(screen_execution_path.resolve()),
                "sha256": sha256_file(screen_execution_path),
            },
            "screen_table": {
                "path": str(table_path.resolve()),
                "sha256": sha256_file(table_path),
            },
            "screen_table_sha256": sha256_file(table_path),
            "selector_tool": {
                "path": str(Path(__file__).resolve()),
                "sha256": sha256_file(Path(__file__)),
            },
        },
    )
    atomic_json(
        encoder_path,
        {
            "status": "CONTROLLED_LOSS_LOCKED_FOR_ENCODER_COMPARISON",
            "selection_rule": "best MLP loss under the same four-trial Validation budget",
            "selections": encoder_losses,
            "screen_execution": {
                "path": str(screen_execution_path.resolve()),
                "sha256": sha256_file(screen_execution_path),
            },
            "screen_table": {
                "path": str(table_path.resolve()),
                "sha256": sha256_file(table_path),
            },
            "screen_table_sha256": sha256_file(table_path),
            "selector_tool": {
                "path": str(Path(__file__).resolve()),
                "sha256": sha256_file(Path(__file__)),
            },
        },
    )
    result = {
        "status": "COMPLETE",
        "screen_cells": len(table),
        "best_within_method": len(winners),
        "screen_table": str(table_path.resolve()),
        "screen_table_sha256": sha256_file(table_path),
        "finalists": str(finalist_path.resolve()),
        "finalists_sha256": sha256_file(finalist_path),
        "encoder_loss_selections": str(encoder_path.resolve()),
        "encoder_loss_selections_sha256": sha256_file(encoder_path),
        "sources": {
            "screen_execution": {
                "path": str(screen_execution_path.resolve()),
                "sha256": sha256_file(screen_execution_path),
            },
            "selector_tool": {
                "path": str(Path(__file__).resolve()),
                "sha256": sha256_file(Path(__file__)),
            },
        },
        "artifacts": {
            "screen_table": {
                "path": str(table_path.resolve()),
                "sha256": sha256_file(table_path),
            },
            "screen_winners": {
                "path": str(winner_path.resolve()),
                "sha256": sha256_file(winner_path),
            },
            "finalists": {
                "path": str(finalist_path.resolve()),
                "sha256": sha256_file(finalist_path),
            },
            "encoder_loss_selections": {
                "path": str(encoder_path.resolve()),
                "sha256": sha256_file(encoder_path),
            },
        },
    }
    result["content_sha256"] = canonical_sha256(result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    artifact = run_dir / "07_validation" / "screen_selection_manifest.json"
    with ledger_stage(
        run_dir / "run_ledger.sqlite",
        stage="P7",
        substage="select_validation_screen",
        command=" ".join(map(str, sys.argv)),
    ) as state:
        result = run(run_dir)
        atomic_json(artifact, result)
        state["artifact_path"] = str(artifact)
        state["artifact_sha256"] = sha256_file(artifact)
    print(json.dumps(result, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
