"""Build three-seed OOF/Validation ensembles and lock scalar ranker winners.

The primary evidence track is predeclared as T2 matched-common.  Candidate
rankers are restricted to the R4/R6/R7 finalists frozen by the matched-budget
seed-42 screen.  This command never reads Test labels or Test predictions.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from unified_reranking.artifacts import verified_artifact_path
from unified_reranking.hashing import atomic_json, canonical_sha256, sha256_file
from unified_reranking.ledger import ledger_stage
from unified_reranking.metrics import (
    compare_selections,
    evaluate_order_only,
    rank_by_score,
)
from unified_reranking.matrix_phase import load_matrix_phase_cells
from unified_reranking.training import FORMAL_SEEDS
from tools.unified_reranking.select_validation_screen import _verified_current_cell


ROUTES = ("crog", "g1", "c1")
TRACKS = ("T1_native", "T2_matched_common", "T3_tri_backend")
PRIMARY_TRACK = "T2_matched_common"
ELIGIBLE_PRIMARY_METHODS = frozenset(
    {"R4_mlp_ranknet", "R6_lambdamart", "R7_mlp_jacquard_margin"}
)


def _atomic_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    frame.to_parquet(temporary, index=False, compression="zstd")
    os.replace(temporary, path)


def _atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def _configuration_matches(
    configuration: dict[str, Any],
    choice: dict[str, Any],
    *,
    seed: int,
    mode: str,
    fold: int | None,
) -> bool:
    if (
        configuration.get("encoder") != choice.get("encoder")
        or configuration.get("loss") != choice.get("loss")
        or int(configuration.get("seed", -1)) != int(seed)
        or configuration.get("mode") != mode
        or configuration.get("held_fold") != fold
    ):
        return False
    if not all(
        configuration.get(name) == value
        for name, value in choice.get("parameters", {}).items()
    ):
        return False
    screen_source = choice.get("screen_source_identity")
    cell_source = configuration.get("source_identity")
    if not isinstance(screen_source, dict) or not isinstance(cell_source, dict):
        return False
    required = {
        "train_features_sha256",
        "train_feature_manifest_sha256",
        "train_feature_columns",
        "train_feature_schema_sha256",
        "train_labels_sha256",
        "folds_sha256",
        "selected_feature_columns",
        "selected_feature_schema_sha256",
        "tool_sha256",
        "training_code_sha256",
    }
    if "train_relations_sha256" in screen_source:
        required.add("train_relations_sha256")
    if configuration.get("track") == "T3_tri_backend":
        required.add("train_feature_extraction_benchmark_sha256")
        if mode == "validation":
            required.add("validation_feature_extraction_benchmark_sha256")
    if not required.issubset(screen_source) or not required.issubset(cell_source):
        return False
    return all(screen_source[key] == cell_source[key] for key in required)


def _scan_cells(run_dir: Path) -> list[tuple[dict[str, Any], Path]]:
    selection = run_dir / "05_models" / "screen_finalists.json"
    result = load_matrix_phase_cells(
        run_dir, "selected", expected_selection=selection
    )
    for value, path in result:
        configuration = value.get("configuration", {})
        if (
            configuration.get("mode") not in {"validation", "oof"}
            or not _verified_current_cell(value, path, is_rule=False)
        ):
            raise RuntimeError(f"selected matrix phase contains an invalid cell: {path}")
    return result


def _find_cell(
    cells: Iterable[tuple[dict[str, Any], Path]],
    *,
    route: str,
    track: str,
    choice: dict[str, Any],
    seed: int,
    mode: str,
    fold: int | None,
) -> tuple[dict[str, Any], Path]:
    matches = [
        (value, path)
        for value, path in cells
        if value.get("configuration", {}).get("route") == route
        and value.get("configuration", {}).get("track") == track
        and _configuration_matches(
            value["configuration"], choice, seed=seed, mode=mode, fold=fold
        )
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"expected exactly one matrix cell for {route}/{track}/{choice.get('method_code')}/"
            f"seed={seed}/mode={mode}/fold={fold}; found {len(matches)}"
        )
    return matches[0]


def _sample_universe(run_dir: Path, split: str) -> list[str]:
    return (
        pd.read_parquet(
            run_dir / "01_manifests" / f"paired_{split}.parquet",
            columns=["sample_id"],
        )["sample_id"]
        .astype(str)
        .tolist()
    )


def _native_decisions(
    run_dir: Path, route: str, split: str
) -> tuple[dict[str, Any], pd.DataFrame]:
    candidates = pd.read_parquet(
        run_dir / "02_candidates" / f"{route}_{split}_top5.parquet",
        columns=["sample_id", "candidate_id", "native_rank"],
    )
    labels = pd.read_parquet(
        run_dir / "03_features" / f"candidate_labels_{route}_{split}_top5.parquet",
        columns=["sample_id", "candidate_id", "candidate_success"],
    )
    evaluation = candidates.merge(
        labels, on=["sample_id", "candidate_id"], validate="one_to_one"
    )
    evaluation["native_control_score"] = -evaluation["native_rank"].astype(float)
    return evaluate_order_only(
        _sample_universe(run_dir, split),
        evaluation,
        score_column="native_control_score",
    )


def _augment_decisions(
    candidates: pd.DataFrame,
    predictions: pd.DataFrame,
    decisions: pd.DataFrame,
) -> pd.DataFrame:
    """Attach label-free margins, seed votes, and immutable geometry identity."""

    score_columns = [f"score_seed_{seed}" for seed in FORMAL_SEEDS]
    ranked = rank_by_score(
        candidates[
            ["sample_id", "candidate_id", "candidate_geometry_sha256"]
        ].merge(predictions, on=["sample_id", "candidate_id"], validate="one_to_one"),
        score_column="ensemble_score",
    )
    rows: list[dict[str, Any]] = []
    for sample_id, group in ranked.groupby("sample_id", sort=False):
        ordered = group.sort_values("rerank_rank", kind="mergesort")
        selected = ordered.iloc[0]
        votes = 0
        for column in score_columns:
            seed_winner = group.sort_values(
                [column, "native_rank", "candidate_id"],
                ascending=[False, True, True],
                kind="mergesort",
            ).iloc[0]
            votes += int(str(seed_winner["candidate_id"]) == str(selected["candidate_id"]))
        native = group.sort_values(
            ["native_rank", "candidate_id"], kind="mergesort"
        ).iloc[0]
        # The gate transition is explicitly challenger a versus native b.  A
        # Top1-versus-runner-up margin is different whenever native ranks 3+.
        margin = float(selected["ensemble_score"] - native["ensemble_score"])
        rows.append(
            {
                "sample_id": str(sample_id),
                "native_candidate_id": str(native["candidate_id"]),
                "native_geometry_sha256": str(native["candidate_geometry_sha256"]),
                "selected_geometry_sha256": str(selected["candidate_geometry_sha256"]),
                "ensemble_score_margin": margin,
                "seed_challenger_votes": votes,
                "challenger_exists": str(selected["candidate_id"])
                != str(native["candidate_id"]),
            }
        )
    evidence = pd.DataFrame(rows)
    output = decisions.merge(evidence, on="sample_id", how="left", validate="one_to_one")
    if output["ensemble_score_margin"].isna().any():
        # No-output rows have no candidate and therefore cannot switch.
        output["ensemble_score_margin"] = output["ensemble_score_margin"].fillna(0.0)
        output["seed_challenger_votes"] = output["seed_challenger_votes"].fillna(0).astype(int)
        output["challenger_exists"] = output["challenger_exists"].fillna(False).astype(bool)
    return output


def _build_ensemble(
    run_dir: Path,
    cells: list[tuple[dict[str, Any], Path]],
    *,
    route: str,
    track: str,
    choice: dict[str, Any],
    split: str,
) -> dict[str, Any]:
    if split not in {"train", "validation"}:
        raise PermissionError("ranker selection ensembles are development-only")
    candidate_path = run_dir / "02_candidates" / f"{route}_{split}_top5.parquet"
    label_path = run_dir / "03_features" / f"candidate_labels_{route}_{split}_top5.parquet"
    candidates = pd.read_parquet(candidate_path)
    labels = pd.read_parquet(
        label_path, columns=["sample_id", "candidate_id", "candidate_success"]
    )
    predictions = candidates[["sample_id", "candidate_id", "native_rank"]].copy()
    source_manifests: list[dict[str, str]] = []
    for seed in FORMAL_SEEDS:
        pieces = []
        folds: Iterable[int | None] = range(5) if split == "train" else (None,)
        mode = "oof" if split == "train" else "validation"
        for fold in folds:
            value, manifest_path = _find_cell(
                cells,
                route=route,
                track=track,
                choice=choice,
                seed=seed,
                mode=mode,
                fold=fold,
            )
            prediction_path = verified_artifact_path(
                value["artifacts"]["predictions"],
                name=f"{route}/{track}/{mode}/seed-{seed}/fold-{fold} predictions",
            )
            pieces.append(pd.read_parquet(prediction_path))
            source_manifests.append(
                {"path": str(manifest_path.resolve()), "sha256": sha256_file(manifest_path)}
            )
        seed_predictions = pd.concat(pieces, ignore_index=True)
        if seed_predictions[["sample_id", "candidate_id"]].duplicated().any():
            raise RuntimeError("seed prediction ensemble contains duplicate candidate keys")
        predictions = predictions.merge(
            seed_predictions.rename(columns={"score": f"score_seed_{seed}"}),
            on=["sample_id", "candidate_id"],
            how="left",
            validate="one_to_one",
        )
    score_columns = [f"score_seed_{seed}" for seed in FORMAL_SEEDS]
    if predictions[score_columns].isna().any().any():
        raise RuntimeError("three-seed predictions do not exactly cover frozen candidates")
    predictions["ensemble_score"] = predictions[score_columns].mean(axis=1)
    evaluation = predictions.merge(
        labels, on=["sample_id", "candidate_id"], validate="one_to_one"
    )
    metrics, decisions = evaluate_order_only(
        _sample_universe(run_dir, split), evaluation, score_column="ensemble_score"
    )
    decisions = _augment_decisions(candidates, predictions, decisions)
    identity = {
        "route": route,
        "track": track,
        "method_code": choice["method_code"],
        "encoder": choice["encoder"],
        "loss": choice["loss"],
        "parameters": choice.get("parameters", {}),
        "seeds": list(FORMAL_SEEDS),
    }
    ensemble_id = canonical_sha256(identity)[:16]
    root = run_dir / ("06_oof" if split == "train" else "07_validation") / "ensembles" / ensemble_id
    prediction_path = root / "per_candidate_scores.parquet"
    decision_path = root / "per_sample_decisions.parquet"
    _atomic_parquet(prediction_path, predictions)
    _atomic_parquet(decision_path, decisions)
    result = {
        "status": "COMPLETE",
        "split": split,
        "prediction_source": "train_oof" if split == "train" else "validation",
        "identity": identity,
        "ensemble_id": ensemble_id,
        "metrics": metrics,
        "sources": {
            "candidates": {"path": str(candidate_path.resolve()), "sha256": sha256_file(candidate_path)},
            "labels": {"path": str(label_path.resolve()), "sha256": sha256_file(label_path)},
            "matrix_manifests": source_manifests,
        },
        "artifacts": {
            "predictions": {"path": str(prediction_path.resolve()), "sha256": sha256_file(prediction_path)},
            "decisions": {"path": str(decision_path.resolve()), "sha256": sha256_file(decision_path)},
        },
    }
    atomic_json(root / "manifest.json", result)
    return result


def run(run_dir: Path) -> dict[str, Any]:
    finalist_path = run_dir / "05_models" / "screen_finalists.json"
    selected_execution_path = (
        run_dir / "05_models/matrix_plans/selected_latest_execution.json"
    )
    finalist_payload = json.loads(finalist_path.read_text(encoding="utf-8"))
    if finalist_payload.get("status") != "VALIDATION_SCREEN_LOCKED":
        raise RuntimeError("screen finalists are not Validation-locked")
    selections = finalist_payload["selections"]
    for key, choices in selections.items():
        for choice in choices if isinstance(choices, list) else [choices]:
            verified_artifact_path(
                {
                    "path": choice.get("screen_manifest"),
                    "sha256": choice.get("screen_manifest_sha256"),
                },
                name=f"{key} screen selection manifest",
            )
    cells = _scan_cells(run_dir)
    rows: list[dict[str, Any]] = []
    ensemble_manifests: dict[str, dict[str, Any]] = {}
    for route in ROUTES:
        for track in TRACKS:
            key = f"{route}/{track}"
            choices = selections.get(key)
            if not choices:
                raise RuntimeError(f"screen finalists miss {key}")
            for choice in choices:
                if choice.get("method_code") not in ELIGIBLE_PRIMARY_METHODS:
                    raise RuntimeError(f"ineligible finalist in {key}: {choice.get('method_code')}")
                validation = _build_ensemble(
                    run_dir,
                    cells,
                    route=route,
                    track=track,
                    choice=choice,
                    split="validation",
                )
                oof = _build_ensemble(
                    run_dir,
                    cells,
                    route=route,
                    track=track,
                    choice=choice,
                    split="train",
                )
                native_metrics, native = _native_decisions(run_dir, route, "validation")
                decisions = pd.read_parquet(
                    verified_artifact_path(
                        validation["artifacts"]["decisions"],
                        name=f"{route}/{track} Validation ensemble decisions",
                    )
                )
                comparison = compare_selections(
                    native, decisions, oracle_at_5=float(native_metrics["oracle_at_5"])
                )
                manifest_key = f"{key}/{choice['method_code']}"
                ensemble_manifests[manifest_key] = {
                    "validation": str(
                        (
                            Path(validation["artifacts"]["decisions"]["path"]).parent
                            / "manifest.json"
                        ).resolve()
                    ),
                    "oof": str(
                        (
                            Path(oof["artifacts"]["decisions"]["path"]).parent
                            / "manifest.json"
                        ).resolve()
                    ),
                }
                validation_manifest_path = Path(
                    ensemble_manifests[manifest_key]["validation"]
                )
                oof_manifest_path = Path(ensemble_manifests[manifest_key]["oof"])
                rows.append(
                    {
                        "route": route,
                        "track": track,
                        "method_code": choice["method_code"],
                        "encoder": choice["encoder"],
                        "loss": choice["loss"],
                        "ensemble_id": validation["ensemble_id"],
                        "j_at_1": validation["metrics"]["j_at_1"],
                        "mrr_at_5": validation["metrics"]["mrr_at_5"],
                        **comparison,
                        "validation_manifest": ensemble_manifests[manifest_key]["validation"],
                        "validation_manifest_sha256": sha256_file(
                            validation_manifest_path
                        ),
                        "oof_manifest": ensemble_manifests[manifest_key]["oof"],
                        "oof_manifest_sha256": sha256_file(oof_manifest_path),
                    }
                )
    table = pd.DataFrame(rows).sort_values(
        ["route", "track", "j_at_1", "harmful", "switch_rate", "mrr_at_5", "method_code"],
        ascending=[True, True, False, True, True, False, True],
        kind="mergesort",
    )
    table["selected_within_track"] = ~table.duplicated(["route", "track"])
    primary = table.loc[
        table["selected_within_track"] & table["track"].eq(PRIMARY_TRACK)
    ].copy()
    if set(primary["route"]) != set(ROUTES) or len(primary) != len(ROUTES):
        raise RuntimeError("primary T2 scalar selection did not produce exactly one ranker per route")
    output = run_dir / "07_validation" / "tables"
    table_path = output / "three_seed_scalar_finalists.csv"
    primary_path = output / "selected_primary_ungated.csv"
    _atomic_csv(table_path, table)
    _atomic_csv(primary_path, primary)
    selected = {
        row.route: {
            "primary_track": PRIMARY_TRACK,
            "method_code": row.method_code,
            "encoder": row.encoder,
            "loss": row.loss,
            "ensemble_id": row.ensemble_id,
            "validation_manifest": row.validation_manifest,
            "validation_manifest_sha256": row.validation_manifest_sha256,
            "oof_manifest": row.oof_manifest,
            "oof_manifest_sha256": row.oof_manifest_sha256,
            "selection_metrics": {
                "j_at_1": row.j_at_1,
                "delta_j_at_1": row.delta_j_at_1,
                "recovered": row.recovered,
                "harmful": row.harmful,
                "switch_rate": row.switch_rate,
                "mrr_at_5": row.mrr_at_5,
            },
        }
        for row in primary.itertuples(index=False)
    }
    selection_path = run_dir / "07_validation" / "selected_primary_ungated.json"
    atomic_json(
        selection_path,
        {
            "status": "VALIDATION_LOCKED",
            "primary_track": PRIMARY_TRACK,
            "eligible_methods": sorted(ELIGIBLE_PRIMARY_METHODS),
            "selection_rule": "max three-seed ensemble Validation J@1; then fewer harmful, lower switch rate, higher MRR@5, stable method code",
            "seeds": list(FORMAL_SEEDS),
            "selections": selected,
            "table": {"path": str(table_path.resolve()), "sha256": sha256_file(table_path)},
            "screen_finalists": {
                "path": str(finalist_path.resolve()),
                "sha256": sha256_file(finalist_path),
            },
            "selected_execution": {
                "path": str(selected_execution_path.resolve()),
                "sha256": sha256_file(selected_execution_path),
            },
            "selector_tool": {
                "path": str(Path(__file__).resolve()),
                "sha256": sha256_file(Path(__file__)),
            },
        },
    )
    return {
        "status": "COMPLETE",
        "selection": str(selection_path.resolve()),
        "selection_sha256": sha256_file(selection_path),
        "primary_table": str(primary_path.resolve()),
        "primary_table_sha256": sha256_file(primary_path),
        "ensemble_count": len(table),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    with ledger_stage(
        run_dir / "run_ledger.sqlite",
        stage="P7_P11",
        substage="select_three_seed_primary_ungated_rankers",
        evidence_track=PRIMARY_TRACK,
        command=" ".join(map(str, sys.argv)),
    ) as state:
        result = run(run_dir)
        artifact = Path(result["selection"])
        state["artifact_path"] = str(artifact)
        state["artifact_sha256"] = sha256_file(artifact)
    print(json.dumps(result, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
