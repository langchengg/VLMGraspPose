#!/usr/bin/env python3
"""Run validation-time cross-backend oracle, router, and union experiments."""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression


PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPOSITORY_ROOT = PROJECT_ROOT.parent
for path in (str(REPOSITORY_ROOT), str(PROJECT_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from src.grasping.g1_c1_safe_rerank.artifacts import atomic_json, atomic_parquet  # noqa: E402
from src.grasping.g1_c1_safe_rerank.evaluation import (  # noqa: E402
    evaluate_selected_ids,
    oracle_metrics,
    stable_bootstrap_seed,
)
from src.grasping.g1_c1_safe_rerank.pools import (  # noqa: E402
    build_deduplicated_union,
    build_raw_union,
    pool_manifest,
)
from tools.g1_c1_rerank.run_local_matrix import (  # noqa: E402
    FEATURE_FAMILIES,
    SEEDS,
    _attach_calibration,
    _columns,
    _fit_model,
    _load_joined,
    _universe,
)


ROUTER_FEATURES = (
    "g1_top1_probability",
    "c1_top1_probability",
    "probability_delta_c1_minus_g1",
    "g1_margin",
    "c1_margin",
    "g1_candidate_count",
    "c1_candidate_count",
    "g1_mask_reliability",
    "c1_mask_reliability",
    "top1_center_distance_normalized",
    "top1_angle_difference_normalized",
    "g1_cross_agreement",
    "c1_cross_agreement",
    "g1_nonempty",
    "c1_nonempty",
)
ROUTER_OUTCOMES = ("G1-only", "C1-only", "both-correct", "both-wrong")


def _parse(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-run", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("mps", "cpu"), default="mps")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args(argv)


def _periodic(left: float, right: float) -> float:
    return abs((float(left) - float(right) + 90.0) % 180.0 - 90.0)


def _top_views(frame: pd.DataFrame, prefix: str) -> pd.DataFrame:
    ordered = frame.sort_values(["sample_id", "original_rank", "stable_candidate_id"], kind="mergesort")
    top = ordered.groupby("sample_id", sort=False).head(1).copy()
    second = ordered.loc[ordered["original_rank"].eq(2), ["sample_id", "original_score"]].rename(columns={"original_score": "second_score"})
    top = top.merge(second, on="sample_id", how="left", validate="one_to_one")
    top["score_margin"] = top["original_score"] - top["second_score"].fillna(top["original_score"])
    columns = {
        "stable_candidate_id": f"{prefix}_candidate_id",
        "candidate_correct": f"{prefix}_correct",
        "source_score_calibrated": f"{prefix}_top1_probability",
        "score_margin": f"{prefix}_margin",
        "candidate_count_normalized": f"{prefix}_candidate_count",
        "feature_reliability_score": f"{prefix}_mask_reliability",
        "cross_backend_agreement_score": f"{prefix}_cross_agreement",
        "center_x": f"{prefix}_center_x",
        "center_y": f"{prefix}_center_y",
        "angle_deg": f"{prefix}_angle_deg",
    }
    available = ["sample_id", "scene_id", *[name for name in columns if name in top.columns]]
    return top[available].rename(columns=columns)


def _router_frame(g1: pd.DataFrame, c1: pd.DataFrame, universe: pd.DataFrame, *, include_labels: bool) -> pd.DataFrame:
    left = _top_views(g1, "g1")
    right = _top_views(c1, "c1")
    frame = universe[["sample_id", "scene_id"]].merge(left.drop(columns=["scene_id"]), on="sample_id", how="left", validate="one_to_one").merge(
        right.drop(columns=["scene_id"]), on="sample_id", how="left", validate="one_to_one"
    )
    for backend in ("g1", "c1"):
        frame[f"{backend}_nonempty"] = frame[f"{backend}_candidate_id"].notna().astype(float)
        for column in (
            f"{backend}_top1_probability",
            f"{backend}_margin",
            f"{backend}_candidate_count",
            f"{backend}_mask_reliability",
            f"{backend}_cross_agreement",
            f"{backend}_center_x",
            f"{backend}_center_y",
            f"{backend}_angle_deg",
        ):
            frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(0.0)
    frame["probability_delta_c1_minus_g1"] = frame["c1_top1_probability"] - frame["g1_top1_probability"]
    frame["top1_center_distance_normalized"] = np.hypot(
        frame["g1_center_x"] - frame["c1_center_x"],
        frame["g1_center_y"] - frame["c1_center_y"],
    ) / 800.0
    frame["top1_angle_difference_normalized"] = [
        _periodic(left, right) / 90.0 for left, right in zip(frame["g1_angle_deg"], frame["c1_angle_deg"])
    ]
    if include_labels:
        frame["g1_correct"] = frame["g1_correct"].fillna(False).astype(bool)
        frame["c1_correct"] = frame["c1_correct"].fillna(False).astype(bool)
        frame["outcome"] = np.select(
            [
                frame["g1_correct"] & ~frame["c1_correct"],
                ~frame["g1_correct"] & frame["c1_correct"],
                frame["g1_correct"] & frame["c1_correct"],
            ],
            ["G1-only", "C1-only", "both-correct"],
            default="both-wrong",
        )
    return frame


@dataclass
class RouterModel:
    kind: str
    seed: int = 17
    median: np.ndarray | None = None
    mean: np.ndarray | None = None
    scale: np.ndarray | None = None
    model: Any = None

    def _matrix(self, frame: pd.DataFrame, *, fit: bool) -> np.ndarray:
        values = frame.loc[:, ROUTER_FEATURES].apply(
            pd.to_numeric, errors="coerce"
        ).to_numpy(dtype=float, copy=True)
        values[~np.isfinite(values)] = np.nan
        if fit:
            safe_values = values.copy()
            safe_values[:, np.isnan(safe_values).all(axis=0)] = 0.0
            self.median = np.nanmedian(safe_values, axis=0)
            self.median = np.where(np.isfinite(self.median), self.median, 0.0)
            filled = np.where(np.isfinite(values), values, self.median)
            self.mean = filled.mean(axis=0)
            std = filled.std(axis=0)
            self.scale = np.where(std > 1e-12, std, 1.0)
        if self.median is None or self.mean is None or self.scale is None:
            raise RuntimeError("router preprocessor not fitted")
        return (np.where(np.isfinite(values), values, self.median) - self.mean) / self.scale

    def fit(self, frame: pd.DataFrame) -> "RouterModel":
        x = self._matrix(frame, fit=True)
        if self.kind == "multinomial_logistic":
            self.model = LogisticRegression(C=0.1, class_weight="balanced", solver="lbfgs", max_iter=3000, random_state=self.seed)
        elif self.kind == "gradient_boosted":
            self.model = HistGradientBoostingClassifier(max_iter=120, learning_rate=0.05, max_leaf_nodes=15, l2_regularization=1e-3, random_state=self.seed)
        else:
            raise ValueError(self.kind)
        self.model.fit(x, frame["outcome"].astype(str))
        return self

    def predict(self, frame: pd.DataFrame) -> pd.DataFrame:
        probability = self.model.predict_proba(self._matrix(frame, fit=False))
        output = frame.copy()
        lookup = {str(name): index for index, name in enumerate(self.model.classes_)}
        for name in ROUTER_OUTCOMES:
            output[f"p_{name.lower().replace('-', '_')}"] = probability[:, lookup[name]] if name in lookup else 0.0
        return output


def _router_sweep(predicted: pd.DataFrame, *, kind: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    scenes = np.asarray(sorted(predicted["scene_id"].astype(str).unique()))
    scene_map = {value: index for index, value in enumerate(scenes)}
    sample_scene = predicted["scene_id"].astype(str).map(scene_map).to_numpy(dtype=int)
    scene_sizes = np.bincount(sample_scene, minlength=len(scenes))
    rng = np.random.default_rng(stable_bootstrap_seed(20260806, f"router/{kind}"))
    draws = rng.integers(0, len(scenes), size=(10_000, len(scenes)))
    draw_counts = np.zeros((10_000, len(scenes)), dtype=np.int16)
    np.add.at(draw_counts, (np.repeat(np.arange(10_000), len(scenes)), draws.ravel()), 1)
    denominator = draw_counts @ scene_sizes
    rows = [{"lambda_h": 3.0, "tau": np.inf, "switch_count": 0, "recovered": 0, "harmful": 0, "net": 0, "delta_j_at_1": 0.0, "bootstrap_lower": 0.0, "bootstrap_upper": 0.0, "operating_point": "never_switch"}]
    for lambda_h in (1.0, 1.5, 2.0, 3.0):
        utility = predicted["p_c1_only"].to_numpy(dtype=float) - lambda_h * predicted["p_g1_only"].to_numpy(dtype=float)
        for tau in np.linspace(-0.05, 0.45, 11):
            switch = (utility > tau) & predicted["c1_nonempty"].astype(bool).to_numpy()
            recovered_mask = switch & ~predicted["g1_correct"].to_numpy(dtype=bool) & predicted["c1_correct"].to_numpy(dtype=bool)
            harmful_mask = switch & predicted["g1_correct"].to_numpy(dtype=bool) & ~predicted["c1_correct"].to_numpy(dtype=bool)
            delta = recovered_mask.astype(np.int8) - harmful_mask.astype(np.int8)
            scene_net = np.bincount(sample_scene, weights=delta, minlength=len(scenes))
            distribution = (draw_counts @ scene_net) / np.maximum(denominator, 1)
            recovered, harmful = int(recovered_mask.sum()), int(harmful_mask.sum())
            rows.append({"lambda_h": lambda_h, "tau": float(tau), "switch_count": int(switch.sum()), "recovered": recovered, "harmful": harmful, "net": recovered - harmful, "delta_j_at_1": float(delta.mean()), "bootstrap_lower": float(np.quantile(distribution, .025)), "bootstrap_upper": float(np.quantile(distribution, .975)), "operating_point": "candidate"})
    sweep = pd.DataFrame(rows)
    selected = sweep.sort_values(["bootstrap_lower", "delta_j_at_1", "harmful", "switch_count"], ascending=[False, False, True, True], kind="mergesort").iloc[0].to_dict()
    return sweep, {"safe_lcb": selected, "deploy": bool(float(selected["bootstrap_lower"]) > 0.0)}


def _union_frames(args: argparse.Namespace, split: str, pool: str, *, deduplicate: bool) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    joined: dict[str, pd.DataFrame] = {}
    candidate_only: dict[str, pd.DataFrame] = {}
    for backend in ("g1", "c1"):
        frame = _attach_calibration(args.run_dir, _load_joined(args.run_dir, backend, split, pool), backend, split)
        frame["backend_g1"] = float(backend == "g1")
        joined[backend] = frame
        candidate_only[backend] = pd.read_parquet(args.run_dir / "data" / f"frozen_{backend}_{split}_candidates.parquet")
        if pool == "top5":
            candidate_only[backend] = candidate_only[backend].loc[candidate_only[backend]["original_rank"].le(5)].copy()
        candidate_only[backend] = candidate_only[backend].merge(
            frame[["sample_id", "stable_candidate_id", "source_score_calibrated"]],
            on=["sample_id", "stable_candidate_id"], validate="one_to_one"
        )
    raw = build_raw_union(candidate_only["g1"], candidate_only["c1"])
    raw["source_score_calibrated"] = raw["source_score_calibrated"].astype(float)
    union = build_deduplicated_union(raw) if deduplicate else raw.copy()
    if not deduplicate:
        union = union.sort_values(["sample_id", "source_score_calibrated", "stable_candidate_id"], ascending=[True, False, True], kind="mergesort").reset_index(drop=True)
        union["pool_rank"] = union.groupby("sample_id", sort=False).cumcount() + 1
        union["pool_score"] = union["source_score_calibrated"]
    keep = set(zip(union["sample_id"].astype(str), union["stable_candidate_id"].astype(str)))
    features = pd.concat([joined["g1"], joined["c1"]], ignore_index=True)
    feature_keys = list(zip(features["sample_id"].astype(str), features["stable_candidate_id"].astype(str)))
    features = features.loc[[key in keep for key in feature_keys]].copy().reset_index(drop=True)
    pool_rank = union[["sample_id", "stable_candidate_id", "pool_rank", "pool_score"]]
    features = features.drop(columns=["pool_rank", "pool_score"], errors="ignore").merge(pool_rank, on=["sample_id", "stable_candidate_id"], validate="one_to_one")
    labels = features[["sample_id", "stable_candidate_id", "candidate_correct"]].copy()
    return features, labels, union


def _evaluate_union(frame: pd.DataFrame, labels: pd.DataFrame, universe: pd.DataFrame, scores: np.ndarray, method: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    scored = frame[["sample_id", "stable_candidate_id", "backend", "original_rank"]].copy()
    scored["score"] = scores
    selected = scored.sort_values(["sample_id", "score", "stable_candidate_id"], ascending=[True, False, True], kind="mergesort").groupby("sample_id", sort=False).head(1)
    baseline = scored.loc[scored["backend"].astype(str).eq("G1") & scored["original_rank"].eq(1)]
    selections = baseline[["sample_id", "stable_candidate_id"]].rename(columns={"stable_candidate_id": "baseline_candidate_id"}).merge(
        selected[["sample_id", "stable_candidate_id"]].rename(columns={"stable_candidate_id": "selected_candidate_id"}),
        on="sample_id", how="outer", validate="one_to_one"
    )
    # Always-G1 is explicitly incorrect when G1 emitted no candidate; never
    # inflate the reference by backfilling it from the union challenger.
    selections["baseline_candidate_id"] = selections["baseline_candidate_id"].fillna("")
    selections["selected_candidate_id"] = selections["selected_candidate_id"].fillna(selections["baseline_candidate_id"])
    selections["fallback"] = False
    return evaluate_selected_ids(selections, labels, universe, method=method)


def run(args: argparse.Namespace) -> None:
    router_artifact: dict[str, Any] = {"fit_scope": "Train labels; thresholds official validation only", "models": {}}
    for split in ("train", "validation"):
        universe = _universe(args.base_run, split)
        frames = {}
        for backend in ("g1", "c1"):
            frames[backend] = _attach_calibration(args.run_dir, _load_joined(args.run_dir, backend, split, "allnms"), backend, split)
        router = _router_frame(frames["g1"], frames["c1"], universe, include_labels=True)
        atomic_parquet(args.run_dir / "07_validation" / "cross_backend" / f"router_{split}.parquet", router)
    train_router = pd.read_parquet(args.run_dir / "07_validation" / "cross_backend" / "router_train.parquet")
    validation_router = pd.read_parquet(args.run_dir / "07_validation" / "cross_backend" / "router_validation.parquet")
    router_rows = []
    for kind in ("multinomial_logistic", "gradient_boosted"):
        model = RouterModel(kind=kind).fit(train_router)
        predicted = model.predict(validation_router)
        sweep, selection = _router_sweep(predicted, kind=kind)
        root = args.run_dir / "07_validation" / "cross_backend" / "router" / kind
        root.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, root / "model.joblib", compress=3)
        atomic_parquet(root / "predicted.parquet", predicted)
        sweep.to_csv(root / "sweep.csv", index=False)
        atomic_json(root / "selection.json", selection)
        router_artifact["models"][kind] = selection
        router_rows.append({"method": kind, **selection["safe_lcb"], "deploy": selection["deploy"]})
    best_router = max(router_artifact["models"], key=lambda kind: (router_artifact["models"][kind]["safe_lcb"]["bootstrap_lower"], router_artifact["models"][kind]["safe_lcb"]["delta_j_at_1"], -router_artifact["models"][kind]["safe_lcb"]["harmful"]))
    router_artifact["selected"] = {"kind": best_router, **router_artifact["models"][best_router]}
    atomic_json(args.run_dir / "07_validation" / "cross_backend" / "ROUTER_SELECTION.json", router_artifact)
    pd.DataFrame(router_rows).to_csv(args.run_dir / "07_validation" / "cross_backend" / "ROUTER_RESULTS.csv", index=False)

    validation_g1 = _attach_calibration(args.run_dir, _load_joined(args.run_dir, "g1", "validation", "allnms"), "g1", "validation")
    validation_c1 = _attach_calibration(args.run_dir, _load_joined(args.run_dir, "c1", "validation", "allnms"), "c1", "validation")
    universe_validation = _universe(args.base_run, "validation")
    top = _router_frame(validation_g1, validation_c1, universe_validation, include_labels=True)
    oracle_record = {
        "samples": len(top),
        "g1_correct_c1_correct": int((top["g1_correct"] & top["c1_correct"]).sum()),
        "g1_correct_c1_wrong": int((top["g1_correct"] & ~top["c1_correct"]).sum()),
        "g1_wrong_c1_correct": int((~top["g1_correct"] & top["c1_correct"]).sum()),
        "both_wrong": int((~top["g1_correct"] & ~top["c1_correct"]).sum()),
    }
    union_records: list[dict[str, Any]] = []
    for pool in ("top5", "allnms"):
        validation_matrix = pd.read_csv(
            args.run_dir / "07_validation" / "VALIDATION_MATRIX.csv"
        )
        loss_rows = validation_matrix.loc[
            validation_matrix["seed"].astype(str).eq("ensemble")
            & validation_matrix["pool"].astype(str).eq(pool)
            & validation_matrix["method"].astype(str).isin(
                {"r3_mlp_bce", "r4_mlp_ranknet", "r5_mlp_listwise"}
            )
        ]
        selected_loss_method = str(
            loss_rows.groupby("method", as_index=False)["j_at_1"]
            .mean()
            .sort_values(["j_at_1", "method"], ascending=[False, True])
            .iloc[0]["method"]
        )
        union_objective = {
            "r3_mlp_bce": "bce",
            "r4_mlp_ranknet": "ranknet",
            "r5_mlp_listwise": "listwise",
        }[selected_loss_method]
        union_temperature = 1.0
        if union_objective == "listwise":
            temperature_cv = pd.read_csv(
                args.run_dir / "07_validation" / "R5_TEMPERATURE_TRAIN_CV.csv"
            )
            temperature_cv = temperature_cv.loc[
                temperature_cv["pool"].astype(str).eq(pool)
            ]
            aggregate = temperature_cv.groupby("temperature", as_index=False).agg(
                net=("net", "sum"),
                harmful=("harmful", "sum"),
                switch_count=("switch_count", "sum"),
                total=("total", "sum"),
            )
            aggregate["delta_j_at_1"] = aggregate["net"] / aggregate["total"]
            aggregate["switch_rate"] = aggregate["switch_count"] / aggregate["total"]
            union_temperature = float(
                aggregate.sort_values(
                    ["delta_j_at_1", "harmful", "switch_rate", "temperature"],
                    ascending=[False, True, True, True],
                    kind="mergesort",
                ).iloc[0]["temperature"]
            )
        atomic_json(
            args.run_dir / "07_validation" / "cross_backend" / f"LOSS_SELECTION_{pool}.json",
            {
                "status": "LOCKED_FROM_OFFICIAL_VALIDATION_AND_TRAIN_CV",
                "pool": pool,
                "selected_backend_average_method": selected_loss_method,
                "objective": union_objective,
                "temperature": union_temperature,
                "test_labels_accessed": False,
            },
        )
        for deduplicate, union_name in ((False, "union_concat"), (True, "union_nms")):
            train, _, train_pool = _union_frames(args, "train", pool, deduplicate=deduplicate)
            validation, validation_labels, validation_pool = _union_frames(args, "validation", pool, deduplicate=deduplicate)
            root = args.run_dir / "07_validation" / "cross_backend" / union_name / pool
            atomic_parquet(root / "train_pool.parquet", train_pool)
            atomic_parquet(root / "validation_pool.parquet", validation_pool)
            atomic_json(root / "pool_manifest.json", {"train": pool_manifest(train_pool), "validation": pool_manifest(validation_pool)})
            oracle = oracle_metrics(validation_pool, validation_labels, universe_validation)
            union_records.append({"track": union_name, "pool": pool, "method": "oracle", **oracle})
            columns = (*_columns(train, through="F7"), "backend_g1")
            for method, kind in (("union_mlp", "mlp_bce"), ("union_deepsets", "deepsets"), ("union_gnn", "gnn")):
                score_frames = []
                for seed in SEEDS:
                    checkpoint = args.run_dir / "checkpoints" / "cross_backend" / union_name / pool / f"{method}_seed{seed}.pt"
                    model = _fit_model(
                        kind,
                        columns,
                        train,
                        seed=seed,
                        device=args.device,
                        checkpoint=checkpoint,
                        objective=union_objective,
                        temperature=union_temperature,
                    )
                    scores = model.predict_scores(validation)
                    scored = validation[["sample_id", "stable_candidate_id", "candidate_identity_sha256"]].copy()
                    scored["reranker_score"] = scores
                    atomic_parquet(root / method / f"scores_seed{seed}.parquet", scored)
                    model_path = args.run_dir / "05_models" / "cross_backend" / union_name / pool / method / f"seed{seed}.joblib"
                    model_path.parent.mkdir(parents=True, exist_ok=True)
                    joblib.dump(model, model_path, compress=3)
                    score_frames.append(scored.rename(columns={"reranker_score": f"score_seed{seed}"}))
                ensemble = score_frames[0]
                for scored in score_frames[1:]:
                    ensemble = ensemble.merge(scored.drop(columns=["candidate_identity_sha256"]), on=["sample_id", "stable_candidate_id"], validate="one_to_one")
                ensemble["reranker_score"] = ensemble[[f"score_seed{seed}" for seed in SEEDS]].mean(axis=1)
                aligned = validation[["sample_id", "stable_candidate_id"]].merge(ensemble[["sample_id", "stable_candidate_id", "reranker_score"]], on=["sample_id", "stable_candidate_id"], validate="one_to_one")["reranker_score"].to_numpy(dtype=float)
                outcomes, metric = _evaluate_union(validation, validation_labels, universe_validation, aligned, method)
                atomic_parquet(root / method / "scores_ensemble.parquet", ensemble)
                atomic_parquet(root / method / "outcomes_ensemble.parquet", outcomes)
                metric.update({"objective": union_objective, "temperature": union_temperature})
                atomic_json(root / method / "metrics_ensemble.json", metric)
                union_records.append({"track": union_name, "pool": pool, "method": method, **metric, **{f"oracle_{key}": value for key, value in oracle.items() if key.startswith("oracle")}})
    pd.DataFrame([oracle_record]).to_csv(args.run_dir / "07_validation" / "cross_backend" / "CROSS_BACKEND_TOP1.csv", index=False)
    pd.DataFrame(union_records).to_csv(args.run_dir / "07_validation" / "cross_backend" / "CROSS_BACKEND_ORACLE_AND_UNION.csv", index=False)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse(argv)
    args.base_run = args.base_run.expanduser().resolve()
    args.run_dir = args.run_dir.expanduser().resolve()
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
