"""Executable, content-addressed P12 router/Top20 Validation producers.

All model selection in this module is restricted to Train OOF and Validation.
The Test application entry point inspects label-free schemas before row access.
"""

from __future__ import annotations

# ruff: noqa: E402 -- native thread limits must be set before numeric imports

import itertools
import os
import pickle
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

# The frozen CPU protocol is single-threaded.  Set native thread limits before
# NumPy, LightGBM, or Torch initialize their runtimes; this also prevents the
# macOS OpenMP conflict observed when the union trains both encoder families.
for _name in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_name, "1")

import numpy as np
import pandas as pd

# Import LightGBM before Torch.  On the frozen macOS experiment runtime this
# avoids the known OpenMP runtime-order conflict while the actual ranker stays
# behind the audited unified wrapper.
try:
    import lightgbm as _lightgbm  # noqa: F401
except ModuleNotFoundError:  # pragma: no cover - optional outside frozen runtime
    _lightgbm = None
import torch

from unified_reranking.artifacts import (
    verified_artifact_path,
    verify_artifact_records_recursive,
)
from unified_reranking.datasets import (
    FoldPreprocessor,
    build_inference_query_arrays,
    build_query_arrays,
)
from unified_reranking.hashing import atomic_json, canonical_sha256, sha256_file
from unified_reranking.metrics import (
    compare_selections,
    evaluate_order_only,
    select_order_only,
)
from unified_reranking.models import DeepSetsResidualScorer, LightGBMLambdaRank
from unified_reranking.route_router import RouterEvidence, RouterOperatingPoint
from unified_reranking.training import (
    FORMAL_SEEDS,
    NeuralTrainingConfig,
    fit_neural_ranker,
    predict_neural_ranker,
    set_deterministic_cpu,
)

from .contracts import assert_label_free_parquet_schema
from .execution import load_content_manifest
from .four_route import (
    ALTERNATIVE_ROUTES,
    DEFAULT_ROUTE_TIE_BREAK,
    FOUR_ROUTES,
    FourRouteCROGDefaultTransitionRouter,
    build_top20_union,
    four_route_decisions,
    select_four_route_operating_point,
    validate_top20_union_frame,
)
from .io import atomic_parquet


ROUTER_LAMBDAS = (1.0, 2.0, 4.0)
ROUTER_UTILITY_THRESHOLDS = (0.0, 0.05, 0.1)
ROUTER_MARGIN_THRESHOLDS = (0.0, 0.05, 0.1)
ROUTER_RELIABILITY_THRESHOLDS = (0.5, 0.75)
ROUTER_STABILITY_THRESHOLDS = (0.5, 0.75)
ROUTER_BOOTSTRAP_SEED = 20260808
ROUTER_BOOTSTRAP_ITERATIONS = 10_000
UNION_ENCODERS = ("lambdamart", "deepsets")
UNION_FOLDS = tuple(range(5))


def _record(path: str | Path) -> dict[str, Any]:
    source = Path(path).expanduser().resolve()
    if source.is_symlink() or not source.is_file():
        raise FileNotFoundError(f"P12 artifact is not a regular file: {source}")
    return {
        "path": str(source),
        "sha256": sha256_file(source),
        "bytes": source.stat().st_size,
    }


def _atomic_pickle(value: object, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as stream:
        pickle.dump(value, stream, protocol=pickle.HIGHEST_PROTOCOL)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, destination)
    return destination


def _atomic_torch(value: object, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    torch.save(value, temporary)
    os.replace(temporary, destination)
    return destination


def _resume(
    path: Path, *, signature: str, statuses: tuple[str, ...]
) -> dict[str, Any] | None:
    if not path.exists():
        return None
    value = load_content_manifest(path, name=f"P12 {path.name}", statuses=statuses)
    if value.get("signature_sha256") != signature:
        raise RuntimeError(f"immutable P12 artifact signature differs: {path}")
    verify_artifact_records_recursive(
        value.get("artifacts"),
        name=f"P12 {path.name} artifacts",
        require_at_least_one=True,
    )
    return value


def four_route_router_grid() -> tuple[RouterOperatingPoint, ...]:
    points = tuple(
        RouterOperatingPoint(*values)
        for values in itertools.product(
            ROUTER_LAMBDAS,
            ROUTER_UTILITY_THRESHOLDS,
            ROUTER_MARGIN_THRESHOLDS,
            ROUTER_RELIABILITY_THRESHOLDS,
            ROUTER_STABILITY_THRESHOLDS,
        )
    )
    if len(points) != 108:
        raise AssertionError("P12 router grid must contain exactly 108 points")
    return points


def write_top20_union_split(
    *,
    split: str,
    route_top5_paths: Mapping[str, str | Path],
    denominator_path: str | Path,
    output_dir: str | Path,
    source_paths: Mapping[str, str | Path],
    resume: bool,
) -> dict[str, Any]:
    """Materialize one exact 4xTop5 union with a current-source manifest."""

    split_name = str(split).lower()
    normalized = {
        str(route).upper(): Path(path).resolve()
        for route, path in route_top5_paths.items()
    }
    if set(normalized) != set(FOUR_ROUTES):
        raise ValueError("P12 Top20 split requires CROG/G1/C1/D1 Top5 paths")
    denominator_source = Path(denominator_path).resolve()
    if split_name == "test":
        for route, path in normalized.items():
            assert_label_free_parquet_schema(path, name=f"P12 {route} Test Top5")
        assert_label_free_parquet_schema(
            denominator_source, name="P12 Test denominator"
        )
    sources = {
        "route_top5": {route: _record(path) for route, path in normalized.items()},
        "denominator": _record(denominator_source),
        **{name: _record(path) for name, path in sorted(source_paths.items())},
    }
    configuration = {
        "split": split_name,
        "routes": list(FOUR_ROUTES),
        "source_pool": "exact_top5",
        "maximum_candidates": 20,
        "cross_route_deduplication": "NONE",
        "route_qualified_candidate_ids": True,
        "candidate_test_labels_read": False,
    }
    signature = canonical_sha256({"configuration": configuration, "sources": sources})
    output = Path(output_dir).resolve()
    marker = output / "manifest.json"
    existing = _resume(marker, signature=signature, statuses=("COMPLETE",))
    if existing is not None:
        if resume:
            return existing
        raise FileExistsError("P12 Top20 split already exists")
    frames = {route: pd.read_parquet(path) for route, path in normalized.items()}
    denominator = pd.read_parquet(denominator_source, columns=["sample_id"])
    union, audit = build_top20_union(frames, denominator, split=split_name)
    feature_path = atomic_parquet(union, output / "candidate_features.parquet")
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "status": "COMPLETE",
        "signature_sha256": signature,
        "configuration": configuration,
        "audit": audit,
        "sources": sources,
        "artifacts": {"candidate_features": _record(feature_path)},
        "candidate_test_labels_read": False,
    }
    manifest["content_sha256"] = canonical_sha256(manifest)
    atomic_json(marker, manifest)
    return manifest


def _router_evidence(frame: pd.DataFrame) -> dict[str, RouterEvidence]:
    return {
        route: RouterEvidence(
            route_margin=frame[f"{route.lower()}_margin"].to_numpy(),
            reliability=frame[f"{route.lower()}_reliability"].to_numpy(),
            perturbation_stability=frame[f"{route.lower()}_stability"].to_numpy(),
            candidate_exists=frame[f"{route.lower()}_candidate_exists"].to_numpy(),
        )
        for route in ALTERNATIVE_ROUTES
    }


def _router_utilities(
    probabilities: Mapping[str, tuple[np.ndarray, np.ndarray]], lambda_router: float
) -> dict[str, np.ndarray]:
    return {
        route: probabilities[route][0] - float(lambda_router) * probabilities[route][1]
        for route in ALTERNATIVE_ROUTES
    }


def _router_trial_frame(result: Any) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                **asdict(trial.operating_point),
                "bootstrap_lower_bound": trial.bootstrap_lower_bound,
                "mean_delta": trial.mean_delta,
                "recovered": trial.recovered,
                "harmful": trial.harmful,
                "switch_count": trial.switch_count,
                "switch_rate": trial.switch_rate,
                **{
                    f"{route.lower()}_switches": count
                    for route, count in trial.route_switches
                },
            }
            for trial in result.trials
        ]
    )


def run_four_route_router_validation(
    *,
    train_oof_path: str | Path,
    validation_path: str | Path,
    feature_columns: Mapping[str, Sequence[str]],
    output_dir: str | Path,
    source_paths: Mapping[str, str | Path],
    model_seed: int = ROUTER_BOOTSTRAP_SEED,
    bootstrap_iterations: int = ROUTER_BOOTSTRAP_ITERATIONS,
    resume: bool,
) -> dict[str, Any]:
    """Fit OOF transitions and lock one of the exact 108 Validation trials."""

    train_path = Path(train_oof_path).expanduser().resolve()
    validation_source = Path(validation_path).expanduser().resolve()
    columns = {
        route: tuple(map(str, feature_columns[route])) for route in ALTERNATIVE_ROUTES
    }
    if set(feature_columns) != set(ALTERNATIVE_ROUTES) or any(
        not value for value in columns.values()
    ):
        raise ValueError("P12 router feature schemas must cover G1/C1/D1")
    points = four_route_router_grid()
    sources = {
        "train_oof": _record(train_path),
        "validation": _record(validation_source),
        **{name: _record(path) for name, path in sorted(source_paths.items())},
    }
    configuration = {
        "default_route": "CROG",
        "alternative_routes": list(ALTERNATIVE_ROUTES),
        "tie_break": list(DEFAULT_ROUTE_TIE_BREAK),
        "feature_columns": {
            route: list(columns[route]) for route in ALTERNATIVE_ROUTES
        },
        "model_seed": int(model_seed),
        "bootstrap_iterations": int(bootstrap_iterations),
        "bootstrap_seed": ROUTER_BOOTSTRAP_SEED,
        "operating_points": [asdict(point) for point in points],
        "candidate_test_labels_read": False,
    }
    signature = canonical_sha256({"configuration": configuration, "sources": sources})
    output = Path(output_dir).expanduser().resolve()
    marker = output / "router_selection_manifest.json"
    existing = _resume(marker, signature=signature, statuses=("COMPLETE",))
    if existing is not None:
        if resume:
            return existing
        raise FileExistsError("P12 router selection already exists")
    train = pd.read_parquet(train_path)
    validation = pd.read_parquet(validation_source)
    if set(train["prediction_source"].astype(str)) != {"train_oof"} or set(
        validation["prediction_source"].astype(str)
    ) != {"validation"}:
        raise ValueError("P12 router input prediction provenance differs")
    shared = {
        "sample_id",
        "scene_id",
        "crog_correct",
        "g1_correct",
        "c1_correct",
        "d1_correct",
    }
    required_train = {
        *shared,
        "oof_fold",
        *itertools.chain.from_iterable(columns.values()),
    }
    required_validation = {
        *shared,
        *itertools.chain.from_iterable(columns.values()),
        *{
            f"{route.lower()}_{suffix}"
            for route in ALTERNATIVE_ROUTES
            for suffix in ("margin", "reliability", "stability", "candidate_exists")
        },
        *{
            f"{route.lower()}_{suffix}"
            for route in FOUR_ROUTES
            for suffix in ("candidate_id", "candidate_geometry_sha256")
        },
    }
    for frame, required, name in (
        (train, required_train, "Train OOF"),
        (validation, required_validation, "Validation"),
    ):
        missing = sorted(required.difference(frame.columns))
        if missing or frame.empty or frame["sample_id"].astype(str).duplicated().any():
            raise ValueError(f"P12 router {name} contract differs; missing={missing}")

    def transition(route: str) -> Any:
        from unified_reranking.gate import OOFTransitionData

        return OOFTransitionData(
            features=train.loc[:, columns[route]].to_numpy(float),
            feature_names=columns[route],
            native_correct=train["crog_correct"].to_numpy(),
            challenger_correct=train[f"{route.lower()}_correct"].to_numpy(),
            scene_ids=train["scene_id"].to_numpy(),
            oof_fold_ids=train["oof_fold"].to_numpy(),
            prediction_source="train_oof",
        )

    router = FourRouteCROGDefaultTransitionRouter(seed=int(model_seed)).fit(
        {route: transition(route) for route in ALTERNATIVE_ROUTES}
    )
    probabilities = router.predict_probabilities(
        {
            route: validation.loc[:, columns[route]].to_numpy(float)
            for route in ALTERNATIVE_ROUTES
        }
    )
    evidence = _router_evidence(validation)
    correct = {
        route: validation[f"{route.lower()}_correct"].to_numpy()
        for route in FOUR_ROUTES
    }
    result = select_four_route_operating_point(
        probabilities,
        evidence,
        correct,
        validation["scene_id"].to_numpy(),
        points,
        bootstrap_iterations=int(bootstrap_iterations),
        bootstrap_seed=ROUTER_BOOTSTRAP_SEED,
    )
    if result.selected_operating_point is None:
        selected_routes = np.full(len(validation), "CROG", dtype=object)
        utilities = {
            route: np.zeros(len(validation), dtype=float)
            for route in ALTERNATIVE_ROUTES
        }
    else:
        selected_routes = four_route_decisions(
            probabilities, evidence, result.selected_operating_point
        )
        utilities = _router_utilities(
            probabilities, result.selected_operating_point.lambda_router
        )
    route_ids = {
        route: validation[f"{route.lower()}_candidate_id"]
        .fillna("")
        .astype(str)
        .to_numpy()
        for route in FOUR_ROUTES
    }
    route_hashes = {
        route: validation[f"{route.lower()}_candidate_geometry_sha256"]
        .fillna("")
        .astype(str)
        .to_numpy()
        for route in FOUR_ROUTES
    }
    indexes = np.arange(len(validation))
    selected_ids = np.asarray(
        [
            route_ids[str(route)][index]
            for index, route in zip(indexes, selected_routes, strict=True)
        ]
    )
    selected_hashes = np.asarray(
        [
            route_hashes[str(route)][index]
            for index, route in zip(indexes, selected_routes, strict=True)
        ]
    )
    selected_correct = np.asarray(
        [
            bool(correct[str(route)][index])
            for index, route in zip(indexes, selected_routes, strict=True)
        ]
    )
    decisions: dict[str, Any] = {
        "sample_id": validation["sample_id"].astype(str),
        "scene_id": validation["scene_id"].astype(str),
        "prediction_source": "validation",
        "selected_route": selected_routes,
        "switched_from_crog": selected_routes != "CROG",
        "selected_candidate_id": selected_ids,
        "selected_candidate_geometry_sha256": selected_hashes,
        "selected_correct": selected_correct,
    }
    for route in FOUR_ROUTES:
        decisions[f"{route.lower()}_candidate_id"] = route_ids[route]
        decisions[f"{route.lower()}_correct"] = np.asarray(correct[route], dtype=bool)
    for route in ALTERNATIVE_ROUTES:
        decisions[f"{route.lower()}_probability_recover"] = probabilities[route][0]
        decisions[f"{route.lower()}_probability_harm"] = probabilities[route][1]
        decisions[f"{route.lower()}_utility"] = utilities[route]
    decision_frame = pd.DataFrame(decisions)
    model_path = _atomic_pickle(router, output / "transition_models.pkl")
    decision_path = atomic_parquet(
        decision_frame, output / "validation_decisions.parquet"
    )
    trial_path = atomic_parquet(
        _router_trial_frame(result), output / "validation_trials.parquet"
    )
    artifacts = {
        "transition_models": _record(model_path),
        "validation_decisions": _record(decision_path),
        "validation_trials": _record(trial_path),
    }
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "status": "COMPLETE",
        "decision": result.status,
        "signature_sha256": signature,
        "configuration": configuration,
        "transition_models": router.artifact(),
        "selection": result.artifact(),
        "sources": sources,
        "artifacts": artifacts,
        "candidate_test_labels_read": False,
        "selection_used_test_metrics": False,
    }
    manifest["content_sha256"] = canonical_sha256(manifest)
    atomic_json(marker, manifest)
    return manifest


@dataclass(frozen=True)
class Top20UnionBudget:
    lambdamart_num_leaves: int = 31
    lambdamart_learning_rate: float = 0.05
    lambdamart_n_estimators: int = 200
    deepsets_learning_rate: float = 3e-4
    deepsets_weight_decay: float = 1e-4
    deepsets_alpha: float = 0.5
    deepsets_epochs: int = 100
    deepsets_patience: int = 10
    deepsets_batch_size: int = 512

    def validate(self) -> None:
        values = np.asarray(list(asdict(self).values()), dtype=float)
        if not np.isfinite(values).all() or (values <= 0).any():
            raise ValueError("P12 Top20 union budget must be finite and positive")


FORMAL_TOP20_BUDGET = Top20UnionBudget()


def top20_union_plan(
    budget: Top20UnionBudget = FORMAL_TOP20_BUDGET,
) -> tuple[dict[str, Any], ...]:
    budget.validate()
    return tuple(
        {
            "encoder": encoder,
            "seed": seed,
            "mode": mode,
            "held_fold": fold,
            "early_stop_fold": (fold + 1) % 5 if fold is not None else 0,
            "budget": asdict(budget),
        }
        for encoder in UNION_ENCODERS
        for seed in FORMAL_SEEDS
        for mode, folds in (("oof", UNION_FOLDS), ("validation", (None,)))
        for fold in folds
    )


def _flat(arrays: Any) -> tuple[np.ndarray, np.ndarray, list[str]]:
    valid = ~arrays.padding_mask.numpy()
    query_ids = [
        sample_id
        for sample_id, candidate_ids in zip(
            arrays.sample_ids, arrays.candidate_ids, strict=True
        )
        for _ in candidate_ids
    ]
    return (
        arrays.features.numpy()[valid],
        arrays.labels.numpy()[valid].astype(np.int32),
        query_ids,
    )


def _prediction_rows(arrays: Any, scores: np.ndarray) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    cursor = 0
    for sample_id, candidate_ids in zip(
        arrays.sample_ids, arrays.candidate_ids, strict=True
    ):
        for candidate_id in candidate_ids:
            rows.append(
                {
                    "sample_id": sample_id,
                    "candidate_id": candidate_id,
                    "score": float(scores[cursor]),
                }
            )
            cursor += 1
    if cursor != len(scores):
        raise RuntimeError("P12 union scores do not cover candidate tensors")
    return pd.DataFrame(rows)


def _load_union_frame(
    path: Path, *, split: str, columns: Sequence[str]
) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    required = {
        "sample_id",
        "candidate_id",
        "native_rank",
        "base_logit",
        "candidate_success",
        "jacquard_margin",
        *columns,
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"P12 {split} Top20 training frame misses columns: {missing}")
    validate_top20_union_frame(
        frame, tuple(dict.fromkeys(frame["sample_id"].astype(str)))
    )
    return frame


def _train_union_cell(
    *,
    train: pd.DataFrame,
    validation: pd.DataFrame,
    folds: pd.DataFrame,
    train_denominator: Sequence[str],
    validation_denominator: Sequence[str],
    feature_columns: tuple[str, ...],
    configuration: Mapping[str, Any],
    sources: Mapping[str, Any],
    output_root: Path,
) -> dict[str, Any]:
    config = dict(configuration)
    encoder = str(config["encoder"])
    seed = int(config["seed"])
    held_fold = config["held_fold"]
    early_fold = int(config["early_stop_fold"])
    budget = Top20UnionBudget(**dict(config["budget"]))
    merged = train.merge(folds, on="sample_id", validate="many_to_one")
    fit = merged.loc[
        (merged["fold"] != early_fold)
        & ((merged["fold"] != held_fold) if held_fold is not None else True)
    ]
    early = merged.loc[merged["fold"] == early_fold]
    if held_fold is None:
        predict = validation
        denominator = list(map(str, validation_denominator))
    else:
        predict = merged.loc[merged["fold"] == held_fold]
        denominator = (
            folds.loc[folds["fold"].eq(held_fold), "sample_id"].astype(str).tolist()
        )
    preprocessor = FoldPreprocessor.fit(fit, feature_columns)
    fit_arrays = build_query_arrays(fit, preprocessor=preprocessor, max_candidates=20)
    early_arrays = build_query_arrays(
        early, preprocessor=preprocessor, max_candidates=20
    )
    predict_arrays = build_query_arrays(
        predict, preprocessor=preprocessor, max_candidates=20
    )
    cell_signature = canonical_sha256({"configuration": config, "sources": sources})
    root = output_root / canonical_sha256(config)[:16]
    marker = root / "manifest.json"
    resumed = _resume(marker, signature=cell_signature, statuses=("COMPLETE",))
    if resumed is not None:
        return resumed
    if encoder == "lambdamart":
        fit_x, fit_y, fit_q = _flat(fit_arrays)
        early_x, early_y, early_q = _flat(early_arrays)
        predict_x, _, _ = _flat(predict_arrays)
        model: Any = LightGBMLambdaRank(
            seed=seed,
            num_leaves=budget.lambdamart_num_leaves,
            learning_rate=budget.lambdamart_learning_rate,
            n_estimators=budget.lambdamart_n_estimators,
        ).fit(fit_x, fit_y, fit_q, eval_set=(early_x, early_y, early_q))
        predictions = _prediction_rows(predict_arrays, model.predict(predict_x))
        model_path = _atomic_pickle(model, root / "model.pkl")
        training = model.artifact()
    elif encoder == "deepsets":
        set_deterministic_cpu(seed)
        model = DeepSetsResidualScorer(
            len(feature_columns), alpha=budget.deepsets_alpha
        )
        trained = fit_neural_ranker(
            model,
            fit_arrays,
            early_arrays,
            config=NeuralTrainingConfig(
                loss="listwise",
                learning_rate=budget.deepsets_learning_rate,
                weight_decay=budget.deepsets_weight_decay,
                alpha=budget.deepsets_alpha,
                epochs=budget.deepsets_epochs,
                patience=budget.deepsets_patience,
                batch_size=budget.deepsets_batch_size,
                seed=seed,
            ),
        )
        model.load_state_dict(trained.state_dict)
        predictions = predict_neural_ranker(model, predict_arrays)
        model_path = _atomic_torch(
            {
                "state_dict": trained.state_dict,
                "input_dim": len(feature_columns),
                "alpha": budget.deepsets_alpha,
                "encoder": "deepsets",
            },
            root / "model.pt",
        )
        training = {
            "best_epoch": trained.best_epoch,
            "best_validation_loss": trained.best_validation_loss,
            "epochs_ran": trained.epochs_ran,
        }
    else:
        raise ValueError(f"unknown P12 union encoder: {encoder}")
    evaluation = predict[
        ["sample_id", "candidate_id", "native_rank", "candidate_success"]
    ].merge(predictions, on=["sample_id", "candidate_id"], validate="one_to_one")
    metrics, decisions = evaluate_order_only(
        denominator, evaluation, score_column="score", max_k=20
    )
    prediction_path = atomic_parquet(predictions, root / "candidate_scores.parquet")
    decision_path = atomic_parquet(decisions, root / "per_sample_decisions.parquet")
    artifacts = {
        "model": _record(model_path),
        "candidate_scores": _record(prediction_path),
        "per_sample_decisions": _record(decision_path),
    }
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "status": "COMPLETE",
        "signature_sha256": cell_signature,
        "configuration": config,
        "feature_columns": list(feature_columns),
        "preprocessor": preprocessor.artifact(),
        "training": training,
        "metrics": metrics,
        "sources": dict(sources),
        "artifacts": artifacts,
        "candidate_test_labels_read": False,
    }
    manifest["content_sha256"] = canonical_sha256(manifest)
    atomic_json(marker, manifest)
    return manifest


def _ensemble_union(
    *,
    encoder: str,
    split: str,
    cells: Sequence[dict[str, Any]],
    frame: pd.DataFrame,
    denominator: Sequence[str],
    output_dir: Path,
) -> dict[str, Any]:
    selected = [
        cell
        for cell in cells
        if cell["configuration"]["encoder"] == encoder
        and cell["configuration"]["mode"]
        == ("oof" if split == "train_oof" else "validation")
    ]
    expected = len(FORMAL_SEEDS) * (5 if split == "train_oof" else 1)
    if len(selected) != expected:
        raise RuntimeError("P12 union ensemble cell universe differs")
    scores = frame[
        ["sample_id", "candidate_id", "native_rank", "candidate_success"]
    ].copy()
    cell_records: list[dict[str, Any]] = []
    for seed in FORMAL_SEEDS:
        seed_cells = [
            cell for cell in selected if int(cell["configuration"]["seed"]) == seed
        ]
        parts = [
            pd.read_parquet(
                verified_artifact_path(
                    cell["artifacts"]["candidate_scores"], name="P12 union cell scores"
                )
            )
            for cell in seed_cells
        ]
        combined = pd.concat(parts, ignore_index=True)
        if combined.duplicated(["sample_id", "candidate_id"]).any():
            raise RuntimeError("P12 union seed ensemble duplicates candidate scores")
        scores = scores.merge(
            combined.rename(columns={"score": f"score_seed_{seed}"}),
            on=["sample_id", "candidate_id"],
            validate="one_to_one",
        )
        cell_records.extend(
            _record(Path(cell["artifacts"]["model"]["path"]).parent / "manifest.json")
            for cell in seed_cells
        )
    seed_columns = [f"score_seed_{seed}" for seed in FORMAL_SEEDS]
    scores["score"] = scores[seed_columns].mean(axis=1)
    metrics, decisions = evaluate_order_only(
        denominator, scores, score_column="score", max_k=20
    )
    baseline_metrics, baseline_decisions = evaluate_order_only(
        denominator, frame, score_column="base_logit", max_k=20
    )
    comparison = compare_selections(
        baseline_decisions,
        decisions,
        oracle_at_5=float(baseline_metrics["oracle_at_5"]),
    )
    signature = canonical_sha256(
        {
            "encoder": encoder,
            "split": split,
            "cells": cell_records,
            "seeds": list(FORMAL_SEEDS),
        }
    )
    marker = output_dir / f"{encoder}_{split}" / "manifest.json"
    resumed = _resume(marker, signature=signature, statuses=("COMPLETE",))
    if resumed is not None:
        return resumed
    score_path = atomic_parquet(scores, marker.parent / "candidate_scores.parquet")
    decision_path = atomic_parquet(
        decisions, marker.parent / "per_sample_decisions.parquet"
    )
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "status": "COMPLETE",
        "signature_sha256": signature,
        "identity": {"encoder": encoder, "split": split, "seeds": list(FORMAL_SEEDS)},
        "metrics": metrics,
        "baseline_metrics": baseline_metrics,
        "comparison_to_native_union_baseline": comparison,
        "sources": {"cells": cell_records},
        "artifacts": {
            "candidate_scores": _record(score_path),
            "per_sample_decisions": _record(decision_path),
        },
        "candidate_test_labels_read": False,
    }
    manifest["content_sha256"] = canonical_sha256(manifest)
    atomic_json(marker, manifest)
    return manifest


def run_top20_union_validation(
    *,
    train_path: str | Path,
    validation_path: str | Path,
    folds_path: str | Path,
    train_denominator_path: str | Path,
    validation_denominator_path: str | Path,
    feature_columns: Sequence[str],
    output_dir: str | Path,
    source_paths: Mapping[str, str | Path],
    budget: Top20UnionBudget = FORMAL_TOP20_BUDGET,
    resume: bool,
) -> dict[str, Any]:
    """Execute the exact three-seed OOF/Validation Top20 union plan."""

    budget.validate()
    columns = tuple(map(str, feature_columns))
    train_source = Path(train_path).resolve()
    validation_source = Path(validation_path).resolve()
    fold_source = Path(folds_path).resolve()
    train_denominator_source = Path(train_denominator_path).resolve()
    validation_denominator_source = Path(validation_denominator_path).resolve()
    sources = {
        "train_top20": _record(train_source),
        "validation_top20": _record(validation_source),
        "folds": _record(fold_source),
        "train_denominator": _record(train_denominator_source),
        "validation_denominator": _record(validation_denominator_source),
        **{name: _record(path) for name, path in sorted(source_paths.items())},
    }
    cells_plan = top20_union_plan(budget)
    plan: dict[str, Any] = {
        "schema_version": 1,
        "status": "PLANNED",
        "encoders": list(UNION_ENCODERS),
        "deepsets_availability": "AVAILABLE_EXISTING_STABLE_IMPLEMENTATION",
        "seeds": list(FORMAL_SEEDS),
        "folds": list(UNION_FOLDS),
        "cell_count": len(cells_plan),
        "cells": list(cells_plan),
        "feature_columns": list(columns),
        "budget": asdict(budget),
        "sources": sources,
        "candidate_test_labels_read": False,
    }
    plan["content_sha256"] = canonical_sha256(plan)
    output = Path(output_dir).expanduser().resolve()
    plan_path = output / "union_plan.json"
    if plan_path.exists():
        existing_plan = load_content_manifest(
            plan_path, name="P12 union plan", statuses=("PLANNED", "COMPLETE")
        )
        if existing_plan != plan:
            raise RuntimeError("immutable P12 union plan differs")
    else:
        atomic_json(plan_path, plan)
    train = _load_union_frame(train_source, split="Train", columns=columns)
    validation = _load_union_frame(
        validation_source, split="Validation", columns=columns
    )
    folds = pd.read_parquet(fold_source, columns=["sample_id", "fold"])
    folds["sample_id"] = folds["sample_id"].astype(str)
    if (
        set(pd.to_numeric(folds["fold"])) != set(UNION_FOLDS)
        or folds["sample_id"].duplicated().any()
    ):
        raise ValueError("P12 union fold contract differs")
    train_denominator = (
        pd.read_parquet(train_denominator_source, columns=["sample_id"])["sample_id"]
        .astype(str)
        .tolist()
    )
    validation_denominator = (
        pd.read_parquet(validation_denominator_source, columns=["sample_id"])[
            "sample_id"
        ]
        .astype(str)
        .tolist()
    )
    cell_sources = {**sources, "plan": _record(plan_path)}
    cells = [
        _train_union_cell(
            train=train,
            validation=validation,
            folds=folds,
            train_denominator=train_denominator,
            validation_denominator=validation_denominator,
            feature_columns=columns,
            configuration=config,
            sources=cell_sources,
            output_root=output / "cells",
        )
        for config in cells_plan
    ]
    ensembles = {
        encoder: {
            "train_oof": _ensemble_union(
                encoder=encoder,
                split="train_oof",
                cells=cells,
                frame=train,
                denominator=train_denominator,
                output_dir=output / "ensembles",
            ),
            "validation": _ensemble_union(
                encoder=encoder,
                split="validation",
                cells=cells,
                frame=validation,
                denominator=validation_denominator,
                output_dir=output / "ensembles",
            ),
        }
        for encoder in UNION_ENCODERS
    }
    tie_order = {encoder: index for index, encoder in enumerate(UNION_ENCODERS)}
    trials = [
        {
            "encoder": encoder,
            "validation_j_at_1": ensembles[encoder]["validation"]["metrics"]["j_at_1"],
            "harmful": ensembles[encoder]["validation"][
                "comparison_to_native_union_baseline"
            ]["harmful"],
            "switch_rate": ensembles[encoder]["validation"][
                "comparison_to_native_union_baseline"
            ]["switch_rate"],
        }
        for encoder in UNION_ENCODERS
    ]
    selected = max(
        trials,
        key=lambda row: (
            row["validation_j_at_1"],
            -row["harmful"],
            -row["switch_rate"],
            -tie_order[str(row["encoder"])],
        ),
    )
    encoder = str(selected["encoder"])
    selection_sources = {
        "plan": _record(plan_path),
        "selected_oof_ensemble": _record(
            output / "ensembles" / f"{encoder}_train_oof" / "manifest.json"
        ),
        "selected_validation_ensemble": _record(
            output / "ensembles" / f"{encoder}_validation" / "manifest.json"
        ),
        "selected_validation_cells": [
            _record(Path(cell["artifacts"]["model"]["path"]).parent / "manifest.json")
            for cell in cells
            if cell["configuration"]["encoder"] == encoder
            and cell["configuration"]["mode"] == "validation"
        ],
    }
    selection_signature = canonical_sha256(
        {"trials": trials, "sources": selection_sources}
    )
    selection_path = output / "selected_union_ranker.json"
    existing = _resume(
        selection_path, signature=selection_signature, statuses=("VALIDATION_LOCKED",)
    )
    if existing is not None:
        if resume:
            return existing
        raise FileExistsError("P12 union selection already exists")
    selection: dict[str, Any] = {
        "schema_version": 1,
        "status": "VALIDATION_LOCKED",
        "signature_sha256": selection_signature,
        "selected_encoder": encoder,
        "seeds": list(FORMAL_SEEDS),
        "feature_columns": list(columns),
        "budget": asdict(budget),
        "trials": trials,
        "selection_order": [
            "validation_j_at_1",
            "fewer_harmful_vs_native_union",
            "lower_switch_rate",
            "lambdamart_before_deepsets_tie_break",
        ],
        "sources": selection_sources,
        "artifacts": {
            "selected_validation_scores": ensembles[encoder]["validation"]["artifacts"][
                "candidate_scores"
            ],
            "selected_validation_decisions": ensembles[encoder]["validation"][
                "artifacts"
            ]["per_sample_decisions"],
        },
        "candidate_test_labels_read": False,
        "selection_used_test_metrics": False,
    }
    selection["content_sha256"] = canonical_sha256(selection)
    atomic_json(selection_path, selection)
    return selection


def finalize_p12_validation(
    *,
    run_dir: str | Path,
    router_selection_path: str | Path,
    router_validation_input_path: str | Path,
    union_selection_path: str | Path,
    validation_top20_path: str | Path,
    source_paths: Mapping[str, str | Path],
    resume: bool,
) -> dict[str, Any]:
    """Publish the P13 table from hash-bound router and union outputs."""

    from .four_route_plan import write_validation_router_union_artifacts

    router_manifest = load_content_manifest(
        router_selection_path, name="P12 router selection", statuses=("COMPLETE",)
    )
    union_manifest = load_content_manifest(
        union_selection_path,
        name="P12 union selection",
        statuses=("VALIDATION_LOCKED",),
    )
    if union_manifest.get("selected_encoder") not in UNION_ENCODERS:
        raise RuntimeError("P12 union selection has no supported locked encoder")
    router_decisions = pd.read_parquet(
        verified_artifact_path(
            router_manifest["artifacts"]["validation_decisions"],
            name="P12 router Validation decisions",
        )
    )
    router_input = pd.read_parquet(router_validation_input_path)
    required = {
        "sample_id",
        "scene_id",
        "crog_correct",
        "g1_correct",
        "c1_correct",
        "d1_correct",
        "three_route_router_correct",
        "existing_top15_oracle",
    }
    missing = sorted(required.difference(router_input.columns))
    if missing:
        raise ValueError(f"P12 Validation reporting input misses columns: {missing}")
    samples = (
        router_input.loc[:, list(required)]
        .merge(
            router_decisions[["sample_id", "selected_route"]],
            on="sample_id",
            validate="one_to_one",
        )
        .rename(columns={"selected_route": "four_route_decision"})
    )
    samples["prediction_source"] = "validation"
    top20 = pd.read_parquet(validation_top20_path)
    bound_sources = {
        "router_selection": Path(router_selection_path),
        "router_validation_inputs": Path(router_validation_input_path),
        "union_selection": Path(union_selection_path),
        "validation_top20": Path(validation_top20_path),
        **{name: Path(path) for name, path in source_paths.items()},
    }
    return write_validation_router_union_artifacts(
        run_dir,
        samples=samples,
        top20_union=top20,
        router_selection=router_manifest["selection"],
        source_paths=bound_sources,
        resume=resume,
    )


def _load_union_model(cell: Mapping[str, Any]) -> tuple[str, Any, FoldPreprocessor]:
    encoder = str(cell["configuration"]["encoder"])
    preprocessor = FoldPreprocessor.from_artifact(dict(cell["preprocessor"]))
    model_path = verified_artifact_path(
        cell["artifacts"]["model"], name="P12 union model"
    )
    if encoder == "lambdamart":
        # The frozen training artifact is a legacy wrapper pickle.  Recover
        # only its audited native LightGBM model string; never execute pickle
        # constructors on the Validation or label-free Test replay path.
        from tools.unified_reranking.apply_locked_matrix_cell import (
            _load_native_lightgbm_ranker,
        )

        model = _load_native_lightgbm_ranker(model_path)
    else:
        payload = torch.load(model_path, map_location="cpu", weights_only=True)
        model = DeepSetsResidualScorer(
            int(payload["input_dim"]), alpha=float(payload["alpha"])
        )
        model.load_state_dict(payload["state_dict"])
    return encoder, model, preprocessor


def _predict_union_cell(
    cell: Mapping[str, Any], features: pd.DataFrame
) -> pd.DataFrame:
    encoder, model, preprocessor = _load_union_model(cell)
    arrays = (
        build_query_arrays(features, preprocessor=preprocessor, max_candidates=20)
        if {"candidate_success", "jacquard_margin"}.issubset(features.columns)
        else build_inference_query_arrays(
            features, preprocessor=preprocessor, max_candidates=20
        )
    )
    if encoder == "lambdamart":
        valid = ~arrays.padding_mask.numpy()
        return _prediction_rows(arrays, model.predict(arrays.features.numpy()[valid]))
    set_deterministic_cpu(int(cell["configuration"]["seed"]))
    return predict_neural_ranker(model, arrays)


def _normalized_artifacts(
    *,
    denominator: Sequence[str],
    universe: pd.DataFrame,
    scored: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    score_source = (
        "native_score"
        if "native_score" in universe.columns
        else "native_score_raw"
        if "native_score_raw" in universe.columns
        else ""
    )
    geometry_columns = ["cx_px", "cy_px", "theta_deg", "width_px", "height_px"]
    missing_geometry = sorted(
        {"candidate_geometry_sha256", *geometry_columns}.difference(universe.columns)
    )
    if not score_source or missing_geometry:
        raise ValueError(
            f"P12 normalized universe lacks native score/geometry: {missing_geometry}"
        )
    universe_columns = [
        "source_route",
        "sample_id",
        "candidate_id",
        "candidate_geometry_sha256",
        "native_rank",
        score_source,
        *geometry_columns,
    ]
    normalized_universe = (
        universe.loc[:, universe_columns]
        .copy()
        .rename(columns={score_source: "native_score"})
    )
    numeric = normalized_universe[
        ["native_score", "native_rank", *geometry_columns]
    ].apply(pd.to_numeric, errors="coerce")
    if (
        not np.isfinite(numeric.to_numpy(float)).all()
        or (numeric[["width_px", "height_px"]] <= 0).any().any()
    ):
        raise ValueError("P12 normalized universe contains invalid native geometry")
    normalized_universe[["native_score", "native_rank", *geometry_columns]] = numeric
    normalized_scores = scored.loc[
        :, ["source_route", "sample_id", "candidate_id", "native_rank", "score"]
    ].copy()
    normalized_scores["rank"] = (
        normalized_scores.sort_values(
            ["sample_id", "score", "candidate_id"],
            ascending=[True, False, True],
            kind="mergesort",
        )
        .groupby("sample_id", sort=False)
        .cumcount()
        .add(1)
        .reindex(normalized_scores.index)
    )
    selected = select_order_only(denominator, normalized_scores, score_column="score")
    identity = normalized_scores[["sample_id", "candidate_id", "source_route"]].rename(
        columns={
            "candidate_id": "selected_candidate_id",
            "source_route": "selected_source_route",
        }
    )
    decisions = selected.merge(
        identity,
        on=["sample_id", "selected_candidate_id"],
        how="left",
        validate="one_to_one",
    )[["sample_id", "selected_source_route", "selected_candidate_id"]]
    decisions[["selected_source_route", "selected_candidate_id"]] = decisions[
        ["selected_source_route", "selected_candidate_id"]
    ].fillna("")
    return normalized_universe, normalized_scores, decisions


def apply_locked_four_route_test(
    *,
    router_selection_path: str | Path,
    router_test_input_path: str | Path,
    union_selection_path: str | Path,
    union_test_feature_path: str | Path,
    denominator_path: str | Path,
    output_dir: str | Path,
    resume: bool,
    require_formal_budget: bool = True,
) -> dict[str, dict[str, Any]]:
    """Apply Validation locks to label-free Test and emit formal-normalized systems."""

    # Rebuild and independently replay both development-only locks before even
    # inspecting a Test schema.  The transition model returned here is the
    # freshly refitted object; the stored pickle is byte-verified but never
    # deserialized by this Test application path.
    from .four_route_validation import (
        rebuild_validated_four_route_router,
        validate_top20_union_selection,
    )

    router_selection, router = rebuild_validated_four_route_router(
        Path(router_selection_path).expanduser().resolve()
    )
    union_selection = validate_top20_union_selection(
        Path(union_selection_path).expanduser().resolve(),
        require_formal_budget=require_formal_budget,
    )
    router_input_path = Path(router_test_input_path).resolve()
    union_feature_path = Path(union_test_feature_path).resolve()
    denominator_source = Path(denominator_path).resolve()
    assert_label_free_parquet_schema(router_input_path, name="P12 router Test inputs")
    assert_label_free_parquet_schema(union_feature_path, name="P12 union Test features")
    assert_label_free_parquet_schema(denominator_source, name="P12 Test denominator")
    if (
        router_selection.get("candidate_test_labels_read") is not False
        or union_selection.get("candidate_test_labels_read") is not False
    ):
        raise PermissionError(
            "P12 Test application requires label-free Validation locks"
        )
    denominator = (
        pd.read_parquet(denominator_source, columns=["sample_id"])["sample_id"]
        .astype(str)
        .tolist()
    )
    output = Path(output_dir).resolve()

    router_frame = pd.read_parquet(router_input_path)
    if set(router_frame["prediction_source"].astype(str)) != {"test_label_free"}:
        raise PermissionError("P12 router Test inputs have non-label-free provenance")
    config = router_selection["configuration"]
    router_geometry_fields = (
        "native_score",
        "native_rank",
        "cx_px",
        "cy_px",
        "theta_deg",
        "width_px",
        "height_px",
    )
    missing_router_geometry = sorted(
        {
            f"{route.lower()}_{field}"
            for route in FOUR_ROUTES
            for field in router_geometry_fields
        }.difference(router_frame.columns)
    )
    if missing_router_geometry:
        raise ValueError(
            f"P12 router Test input lacks formal geometry: {missing_router_geometry}"
        )
    feature_columns = {
        route: tuple(config["feature_columns"][route]) for route in ALTERNATIVE_ROUTES
    }
    model_path = verified_artifact_path(
        router_selection["artifacts"]["transition_models"], name="P12 router model"
    )
    probabilities = router.predict_probabilities(
        {
            route: router_frame.loc[:, feature_columns[route]].to_numpy(float)
            for route in ALTERNATIVE_ROUTES
        }
    )
    evidence = _router_evidence(router_frame)
    point_payload = router_selection["selection"]["selected_operating_point"]
    if point_payload is None:
        selected_routes = np.full(len(router_frame), "CROG", dtype=object)
        utilities = {route: np.zeros(len(router_frame)) for route in ALTERNATIVE_ROUTES}
    else:
        point = RouterOperatingPoint(**point_payload)
        selected_routes = four_route_decisions(probabilities, evidence, point)
        utilities = _router_utilities(probabilities, point.lambda_router)
    router_universe_rows: list[dict[str, Any]] = []
    router_scores_rows: list[dict[str, Any]] = []
    router_decisions_rows: list[dict[str, str]] = []
    for index, row in enumerate(router_frame.itertuples(index=False)):
        sample_id = str(getattr(row, "sample_id"))
        chosen_route = str(selected_routes[index])
        chosen_id = ""
        for route in FOUR_ROUTES:
            candidate_id = str(getattr(row, f"{route.lower()}_candidate_id") or "")
            geometry = str(
                getattr(row, f"{route.lower()}_candidate_geometry_sha256") or ""
            )
            if not candidate_id:
                continue
            native_score = float(getattr(row, f"{route.lower()}_native_score"))
            native_rank = int(getattr(row, f"{route.lower()}_native_rank"))
            geometry_values = {
                field: float(getattr(row, f"{route.lower()}_{field}"))
                for field in ("cx_px", "cy_px", "theta_deg", "width_px", "height_px")
            }
            if (
                not geometry
                or not np.isfinite(
                    [native_score, native_rank, *geometry_values.values()]
                ).all()
                or native_rank <= 0
                or geometry_values["width_px"] <= 0
                or geometry_values["height_px"] <= 0
            ):
                raise ValueError(
                    f"P12 router Test {route} candidate geometry is invalid"
                )
            score = 0.0 if route == "CROG" else float(utilities[route][index])
            router_universe_rows.append(
                {
                    "source_route": route,
                    "sample_id": sample_id,
                    "candidate_id": candidate_id,
                    "candidate_geometry_sha256": geometry,
                    "native_rank": native_rank,
                    "native_score": native_score,
                    **geometry_values,
                }
            )
            router_scores_rows.append(
                {
                    "source_route": route,
                    "sample_id": sample_id,
                    "candidate_id": candidate_id,
                    "score": score,
                    "rank": 1,
                }
            )
            if route == chosen_route:
                chosen_id = candidate_id
        router_decisions_rows.append(
            {
                "sample_id": sample_id,
                "selected_source_route": chosen_route if chosen_id else "",
                "selected_candidate_id": chosen_id,
            }
        )
    router_universe = pd.DataFrame(router_universe_rows)
    router_scores = pd.DataFrame(router_scores_rows)
    router_decisions = pd.DataFrame(router_decisions_rows)

    union_features = pd.read_parquet(union_feature_path)
    validate_top20_union_frame(union_features, denominator)
    cells: dict[int, Mapping[str, Any]] = {}
    for record in union_selection["sources"]["selected_validation_cells"]:
        cell_path = verified_artifact_path(record, name="P12 selected union cell")
        cell = load_content_manifest(
            cell_path, name="P12 selected union cell", statuses=("COMPLETE",)
        )
        cells[int(cell["configuration"]["seed"])] = cell
    if set(cells) != set(FORMAL_SEEDS):
        raise RuntimeError("P12 Test union requires exactly three selected seeds")
    union_scored = union_features.copy()
    for seed in FORMAL_SEEDS:
        predictions = _predict_union_cell(cells[seed], union_features).rename(
            columns={"score": f"score_seed_{seed}"}
        )
        union_scored = union_scored.merge(
            predictions, on=["sample_id", "candidate_id"], validate="one_to_one"
        )
    union_scored["score"] = union_scored[
        [f"score_seed_{seed}" for seed in FORMAL_SEEDS]
    ].mean(axis=1)
    union_universe, union_scores, union_decisions = _normalized_artifacts(
        denominator=denominator,
        universe=union_scored,
        scored=union_scored,
    )

    results: dict[str, dict[str, Any]] = {}
    for name, selection_path, universe, scores, decisions in (
        (
            "four_route_crog_default_router",
            Path(router_selection_path).resolve(),
            router_universe,
            router_scores,
            router_decisions,
        ),
        (
            "top20_union",
            Path(union_selection_path).resolve(),
            union_universe,
            union_scores,
            union_decisions,
        ),
    ):
        system_output = output / name
        sources = {
            "selection": _record(selection_path),
            "test_inputs": _record(
                router_input_path
                if name == "four_route_crog_default_router"
                else union_feature_path
            ),
            "denominator": _record(denominator_source),
            "selected_models": (
                _record(model_path)
                if name == "four_route_crog_default_router"
                else [
                    _record(Path(cell["artifacts"]["model"]["path"]))
                    for cell in cells.values()
                ]
            ),
        }
        configuration = {
            "system": name,
            "routes": list(FOUR_ROUTES),
            "prediction_source": "test_label_free",
            "candidate_test_labels_read": False,
        }
        signature = canonical_sha256(
            {"configuration": configuration, "sources": sources}
        )
        marker = system_output / "manifest.json"
        existing = _resume(marker, signature=signature, statuses=("COMPLETE",))
        if existing is not None:
            if not resume:
                raise FileExistsError(f"P12 Test system exists: {name}")
            results[name] = existing
            continue
        universe_path = atomic_parquet(
            universe, system_output / "candidate_universe.parquet"
        )
        score_path = atomic_parquet(scores, system_output / "candidate_scores.parquet")
        decision_path = atomic_parquet(
            decisions, system_output / "per_sample_decisions.parquet"
        )
        manifest: dict[str, Any] = {
            "schema_version": 1,
            "status": "COMPLETE",
            "signature_sha256": signature,
            "configuration": configuration,
            "sources": sources,
            "artifacts": {
                "candidate_universe": _record(universe_path),
                "candidate_scores": _record(score_path),
                "per_sample_decisions": _record(decision_path),
            },
            "candidate_test_labels_read": False,
            "selection_used_test_metrics": False,
        }
        manifest["content_sha256"] = canonical_sha256(manifest)
        atomic_json(marker, manifest)
        results[name] = manifest
    return results


__all__ = [
    "FORMAL_TOP20_BUDGET",
    "ROUTER_BOOTSTRAP_ITERATIONS",
    "Top20UnionBudget",
    "apply_locked_four_route_test",
    "four_route_router_grid",
    "finalize_p12_validation",
    "run_four_route_router_validation",
    "run_top20_union_validation",
    "top20_union_plan",
    "write_top20_union_split",
]
