"""Leakage-safe D1 expected-gain gate input construction."""

from __future__ import annotations

import numpy as np
import pandas as pd

from unified_reranking.gate import SAFE_GATE_FEATURE_COLUMNS
from unified_reranking.metrics import rank_by_score


_FEATURE_COLUMNS = (
    "calibrated_native_probability",
    "native_score_raw",
    "overall_feature_reliability",
    "peak_retention_rate",
    "perturbed_valid_fraction",
    "mask_reliability",
)


def _assert_exact_feature_membership(
    features: pd.DataFrame, candidates: pd.DataFrame
) -> None:
    keys = ["sample_id", "candidate_id"]
    for name, frame in (("features", features), ("candidates", candidates)):
        missing = sorted(set(keys).difference(frame.columns))
        if missing:
            raise ValueError(f"D1 gate {name} miss candidate keys: {missing}")
        if frame[keys].isna().any().any() or frame.duplicated(keys).any():
            raise ValueError(f"D1 gate {name} contain invalid candidate keys")
    expected = set(map(tuple, candidates[keys].astype(str).to_numpy()))
    observed = set(map(tuple, features[keys].astype(str).to_numpy()))
    if expected != observed or len(features) != len(candidates):
        raise RuntimeError(
            "D1 gate feature/candidate membership differs: "
            f"missing={len(expected - observed)} extra={len(observed - expected)}"
        )


def _selected_features(
    features: pd.DataFrame, selected: pd.DataFrame, *, prefix: str
) -> pd.DataFrame:
    required = {"sample_id", "candidate_id", *_FEATURE_COLUMNS}
    missing = sorted(required.difference(features.columns))
    if missing:
        raise ValueError(f"D1 gate feature table misses columns: {missing}")
    source = features.loc[:, ["sample_id", "candidate_id", *_FEATURE_COLUMNS]].copy()
    source["perturbation_stability"] = source[
        ["peak_retention_rate", "perturbed_valid_fraction"]
    ].min(axis=1)
    source = source.drop(columns=["peak_retention_rate", "perturbed_valid_fraction"])
    source = source.rename(
        columns={
            "candidate_id": f"{prefix}_candidate_id",
            "calibrated_native_probability": f"{prefix}_calibrated_probability",
            "native_score_raw": f"{prefix}_native_score",
            "overall_feature_reliability": f"{prefix}_overall_reliability",
            "perturbation_stability": f"{prefix}_perturbation_stability",
            "mask_reliability": f"{prefix}_mask_reliability",
        }
    )
    return selected.merge(
        source,
        on=["sample_id", f"{prefix}_candidate_id"],
        how="left",
        validate="one_to_one",
    )


def _challenger_summary(predictions: pd.DataFrame) -> pd.DataFrame:
    required = {
        "sample_id",
        "candidate_id",
        "native_rank",
        "candidate_geometry_sha256",
        "ensemble_score",
        "score_seed_42",
        "score_seed_123",
        "score_seed_2026",
    }
    missing = sorted(required.difference(predictions.columns))
    if missing:
        raise ValueError(f"D1 gate challenger predictions miss columns: {missing}")
    ranked = rank_by_score(predictions, score_column="ensemble_score")
    rows = []
    for sample_id, group in ranked.groupby("sample_id", sort=False):
        ordered = group.sort_values("rerank_rank", kind="mergesort")
        top = ordered.iloc[0]
        second_score = (
            float(ordered.iloc[1]["ensemble_score"])
            if len(ordered) > 1
            else float(top["ensemble_score"])
        )
        votes = 0
        for seed in (42, 123, 2026):
            seed_ranked = rank_by_score(group, score_column=f"score_seed_{seed}")
            votes += int(
                str(seed_ranked.iloc[0]["candidate_id"]) == str(top["candidate_id"])
            )
        rows.append(
            {
                "sample_id": str(sample_id),
                "challenger_candidate_id": str(top["candidate_id"]),
                "challenger_geometry_sha256": str(top["candidate_geometry_sha256"]),
                "ranker_score_margin": float(top["ensemble_score"]) - second_score,
                "seed_challenger_votes": votes,
            }
        )
    return pd.DataFrame(rows)


def build_gate_input_frame(
    *,
    paired: pd.DataFrame,
    native_predictions: pd.DataFrame,
    native_decisions: pd.DataFrame,
    challenger_predictions: pd.DataFrame,
    challenger_decisions: pd.DataFrame,
    candidate_features: pd.DataFrame,
    candidates: pd.DataFrame,
    prediction_source: str,
    folds: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Construct one row per paired sample without exposing labels as features."""

    if prediction_source not in {"train_oof", "validation"}:
        raise ValueError("D1 gate prediction source must be train_oof or validation")
    _assert_exact_feature_membership(candidate_features, candidates)
    denominator = paired.loc[:, ["sample_id", "scene_id"]].copy()
    if denominator["sample_id"].duplicated().any():
        raise ValueError("D1 gate paired denominator contains duplicate samples")
    contract_columns = [
        "sample_id",
        "candidate_id",
        "native_rank",
        "candidate_identity_sha256",
        "candidate_geometry_sha256",
    ]
    missing = sorted(set(contract_columns).difference(candidates.columns))
    if missing:
        raise ValueError(f"D1 gate candidates miss contract columns: {missing}")
    contract = candidates.loc[:, contract_columns].copy()
    if contract.duplicated(["sample_id", "candidate_id"]).any():
        raise ValueError("D1 gate candidates contain duplicate keys")
    for name, predictions in (
        ("native", native_predictions),
        ("challenger", challenger_predictions),
    ):
        missing = sorted(set(contract_columns).difference(predictions.columns))
        if missing:
            raise ValueError(
                f"D1 gate {name} predictions miss contract columns: {missing}"
            )
        observed = predictions.loc[:, contract_columns].copy()
        expected = contract.copy()
        observed[["sample_id", "candidate_id"]] = observed[
            ["sample_id", "candidate_id"]
        ].astype(str)
        expected[["sample_id", "candidate_id"]] = expected[
            ["sample_id", "candidate_id"]
        ].astype(str)
        # Parquet may preserve the canonical pool rank as int32 while derived
        # prediction tables materialize it as int64.  Rank width is a storage
        # detail, not part of candidate identity; compare its exact integer
        # value after normalizing both sides to the same lossless dtype.
        observed["native_rank"] = pd.to_numeric(
            observed["native_rank"], errors="raise"
        ).astype(np.int64)
        expected["native_rank"] = pd.to_numeric(
            expected["native_rank"], errors="raise"
        ).astype(np.int64)
        observed = observed.sort_values(
            ["sample_id", "candidate_id"], kind="mergesort"
        ).reset_index(drop=True)
        expected = expected.sort_values(
            ["sample_id", "candidate_id"], kind="mergesort"
        ).reset_index(drop=True)
        if len(observed) != len(expected) or not observed.equals(expected):
            raise RuntimeError(
                f"D1 gate {name} predictions differ from canonical candidates"
            )
    native_top = native_predictions.rename(
        columns={
            "candidate_id": "native_candidate_id",
            "candidate_geometry_sha256": "native_geometry_sha256",
        }
    )
    native_top = native_top.loc[
        native_top["native_rank"].eq(1),
        [
            "sample_id",
            "native_candidate_id",
            "native_geometry_sha256",
        ],
    ]
    native_outcomes = native_decisions.loc[
        :, ["sample_id", "selected_candidate_id", "selected_correct"]
    ].rename(
        columns={
            "selected_candidate_id": "native_decision_candidate_id",
            "selected_correct": "native_correct",
        }
    )
    challenger = _challenger_summary(challenger_predictions)
    challenger_outcomes = challenger_decisions.loc[
        :, ["sample_id", "selected_candidate_id", "selected_correct"]
    ].rename(
        columns={
            "selected_candidate_id": "challenger_decision_candidate_id",
            "selected_correct": "challenger_correct",
        }
    )
    result = denominator.merge(
        native_top, on="sample_id", how="left", validate="one_to_one"
    )
    result = result.merge(
        native_outcomes, on="sample_id", how="left", validate="one_to_one"
    )
    result = result.merge(challenger, on="sample_id", how="left", validate="one_to_one")
    result = result.merge(
        challenger_outcomes, on="sample_id", how="left", validate="one_to_one"
    )
    for decision, expected, name in (
        ("native_decision_candidate_id", "native_candidate_id", "native"),
        ("challenger_decision_candidate_id", "challenger_candidate_id", "challenger"),
    ):
        left = result[decision].fillna("").astype(str)
        right = result[expected].fillna("").astype(str)
        if not left.equals(right):
            raise RuntimeError(f"D1 gate {name} decisions differ from predictions")
    result["native_correct"] = result["native_correct"].fillna(False).astype(bool)
    result["challenger_correct"] = (
        result["challenger_correct"].fillna(False).astype(bool)
    )
    result["challenger_exists"] = (
        result["challenger_candidate_id"].notna()
        & result["native_candidate_id"].notna()
        & (
            result["challenger_candidate_id"].astype(str)
            != result["native_candidate_id"].astype(str)
        )
    )
    locked_challenger = contract.loc[
        :, ["sample_id", "candidate_id", "candidate_geometry_sha256"]
    ].rename(
        columns={
            "candidate_id": "challenger_candidate_id",
            "candidate_geometry_sha256": "locked_challenger_geometry_sha256",
        }
    )
    result = result.merge(
        locked_challenger,
        on=["sample_id", "challenger_candidate_id"],
        how="left",
        validate="one_to_one",
    )
    result["candidate_id_unchanged"] = (
        result["challenger_candidate_id"].notna()
        & result["locked_challenger_geometry_sha256"].notna()
    )
    result["geometry_hash_unchanged"] = (
        result["challenger_geometry_sha256"].fillna("").astype(str)
        == result["locked_challenger_geometry_sha256"].fillna("").astype(str)
    ) & result["candidate_id_unchanged"]
    result = _selected_features(candidate_features, result, prefix="native")
    result = _selected_features(candidate_features, result, prefix="challenger")
    scalar = [
        "native_calibrated_probability",
        "native_native_score",
        "native_overall_reliability",
        "native_perturbation_stability",
        "native_mask_reliability",
        "challenger_calibrated_probability",
        "challenger_native_score",
        "challenger_overall_reliability",
        "challenger_perturbation_stability",
        "challenger_mask_reliability",
        "ranker_score_margin",
        "seed_challenger_votes",
    ]
    result[scalar] = result[scalar].fillna(0.0)
    result["calibrated_probability_delta"] = (
        result["challenger_calibrated_probability"]
        - result["native_calibrated_probability"]
    )
    result["native_score_delta"] = (
        result["challenger_native_score"] - result["native_native_score"]
    )
    result["overall_reliability_delta"] = (
        result["challenger_overall_reliability"] - result["native_overall_reliability"]
    )
    result["perturbation_stability_delta"] = (
        result["challenger_perturbation_stability"]
        - result["native_perturbation_stability"]
    )
    result["mask_reliability_delta"] = (
        result["challenger_mask_reliability"] - result["native_mask_reliability"]
    )
    result["challenger_exists_numeric"] = result["challenger_exists"].astype(float)
    result["score_margin"] = result["ranker_score_margin"]
    result["challenger_reliability"] = result["challenger_overall_reliability"].clip(
        0.0, 1.0
    )
    result["perturbation_stability"] = result["challenger_perturbation_stability"].clip(
        0.0, 1.0
    )
    result["prediction_source"] = prediction_source
    if prediction_source == "train_oof":
        if folds is None:
            raise ValueError("D1 Train gate inputs require fold assignments")
        result = result.merge(
            folds[["sample_id", "fold"]].rename(columns={"fold": "oof_fold"}),
            on="sample_id",
            how="left",
            validate="one_to_one",
        )
        if result["oof_fold"].isna().any():
            raise RuntimeError("D1 Train gate inputs lack OOF folds")
    output_columns = [
        "sample_id",
        "scene_id",
        "prediction_source",
        *(["oof_fold"] if prediction_source == "train_oof" else []),
        "native_correct",
        "challenger_correct",
        "native_candidate_id",
        "challenger_candidate_id",
        "native_geometry_sha256",
        "challenger_geometry_sha256",
        "score_margin",
        "challenger_reliability",
        "perturbation_stability",
        "seed_challenger_votes",
        "candidate_id_unchanged",
        "geometry_hash_unchanged",
        "challenger_exists",
        *SAFE_GATE_FEATURE_COLUMNS,
    ]
    output = result.loc[:, list(dict.fromkeys(output_columns))].copy()
    output["native_candidate_id"] = output["native_candidate_id"].fillna("")
    output["challenger_candidate_id"] = output["challenger_candidate_id"].fillna("")
    if not np.isfinite(output.loc[:, SAFE_GATE_FEATURE_COLUMNS].to_numpy(float)).all():
        raise RuntimeError("D1 gate inference features are non-finite")
    return output


def build_label_free_test_gate_input_frame(
    *,
    paired: pd.DataFrame,
    ranker_decisions: pd.DataFrame,
    candidate_features: pd.DataFrame,
    candidates: pd.DataFrame,
) -> pd.DataFrame:
    """Rebuild the exact gate schema on Test without consulting outcomes."""

    forbidden = {
        "candidate_success",
        "native_correct",
        "challenger_correct",
        "selected_correct",
        "jacquard_margin",
        "first_positive_rank",
        "diagnostic_iou",
        "diagnostic_angle_error_deg",
    }
    leaked = sorted(forbidden.intersection(ranker_decisions.columns))
    if leaked:
        raise PermissionError(
            f"D1 label-free Test ranker decisions contain outcomes: {leaked}"
        )
    required_decisions = {
        "sample_id",
        "native_candidate_id",
        "native_identity_sha256",
        "native_geometry_sha256",
        "selected_candidate_id",
        "selected_identity_sha256",
        "selected_geometry_sha256",
        "ensemble_score_margin",
        "seed_challenger_votes",
        "candidate_count",
        "challenger_exists",
    }
    missing = sorted(required_decisions.difference(ranker_decisions.columns))
    if missing:
        raise ValueError(f"D1 label-free Test ranker decisions miss columns: {missing}")
    _assert_exact_feature_membership(candidate_features, candidates)
    denominator_columns = ["sample_id"] + (
        ["scene_id"] if "scene_id" in paired.columns else []
    )
    denominator = paired.loc[:, denominator_columns].copy()
    denominator["sample_id"] = denominator["sample_id"].astype(str)
    if denominator["sample_id"].duplicated().any():
        raise ValueError("D1 Test paired denominator contains duplicate samples")
    decisions = ranker_decisions.copy()
    decisions["sample_id"] = decisions["sample_id"].astype(str)
    if decisions["sample_id"].duplicated().any():
        raise ValueError("D1 Test ranker decisions contain duplicate samples")
    result = denominator.merge(
        decisions, on="sample_id", how="left", validate="one_to_one"
    )
    if result["candidate_count"].isna().any():
        raise RuntimeError("D1 Test ranker decisions do not preserve the denominator")

    candidate_required = {
        "sample_id",
        "candidate_id",
        "native_rank",
        "candidate_identity_sha256",
        "candidate_geometry_sha256",
    }
    missing = sorted(candidate_required.difference(candidates.columns))
    if missing:
        raise ValueError(f"D1 Test candidates miss gate columns: {missing}")
    locked = candidates.loc[:, sorted(candidate_required)].copy()
    if locked.duplicated(["sample_id", "candidate_id"]).any():
        raise ValueError("D1 Test candidates contain duplicate keys")
    expected_counts = candidates.groupby("sample_id").size().astype(int)
    declared_counts = result["candidate_count"].astype(int)
    recomputed_counts = result["sample_id"].map(expected_counts).fillna(0).astype(int)
    if not declared_counts.equals(recomputed_counts):
        raise RuntimeError("D1 Test ranker candidate counts differ from Top5")
    native = candidates.loc[
        candidates["native_rank"].eq(1),
        ["sample_id", "candidate_id", "candidate_geometry_sha256"],
    ].rename(
        columns={
            "candidate_id": "locked_native_candidate_id",
            "candidate_geometry_sha256": "locked_native_geometry_sha256",
        }
    )
    native_identity = candidates.loc[
        candidates["native_rank"].eq(1),
        ["sample_id", "candidate_identity_sha256"],
    ].rename(columns={"candidate_identity_sha256": "locked_native_identity_sha256"})
    native = native.merge(native_identity, on="sample_id", validate="one_to_one")
    result = result.merge(native, on="sample_id", how="left", validate="one_to_one")
    if not result["native_candidate_id"].fillna("").astype(str).equals(
        result["locked_native_candidate_id"].fillna("").astype(str)
    ) or not result["native_geometry_sha256"].fillna("").astype(str).equals(
        result["locked_native_geometry_sha256"].fillna("").astype(str)
    ):
        raise RuntimeError("D1 Test ranker native identity differs from Top5")
    if (
        not result["native_identity_sha256"]
        .fillna("")
        .astype(str)
        .equals(result["locked_native_identity_sha256"].fillna("").astype(str))
    ):
        raise RuntimeError("D1 Test ranker native full identity differs from Top5")
    locked = candidates.loc[
        :,
        [
            "sample_id",
            "candidate_id",
            "candidate_identity_sha256",
            "candidate_geometry_sha256",
        ],
    ].rename(
        columns={
            "candidate_id": "challenger_candidate_id",
            "candidate_identity_sha256": "locked_challenger_identity_sha256",
            "candidate_geometry_sha256": "locked_challenger_geometry_sha256",
        }
    )
    result["challenger_candidate_id"] = result["selected_candidate_id"]
    result["challenger_identity_sha256"] = result["selected_identity_sha256"]
    result["challenger_geometry_sha256"] = result["selected_geometry_sha256"]
    result = result.merge(
        locked,
        on=["sample_id", "challenger_candidate_id"],
        how="left",
        validate="one_to_one",
    )
    has_selection = result["challenger_candidate_id"].notna()
    result["candidate_id_unchanged"] = (
        has_selection & result["locked_challenger_geometry_sha256"].notna()
    )
    result["geometry_hash_unchanged"] = (
        result["challenger_geometry_sha256"].fillna("").astype(str)
        == result["locked_challenger_geometry_sha256"].fillna("").astype(str)
    ) & result["candidate_id_unchanged"]
    if (
        not result["challenger_identity_sha256"]
        .fillna("")
        .astype(str)
        .equals(result["locked_challenger_identity_sha256"].fillna("").astype(str))
    ):
        raise RuntimeError("D1 Test challenger full identity differs from Top5")
    declared_exists = result["challenger_exists"].fillna(False).astype(bool)
    recomputed_exists = (
        result["challenger_candidate_id"].notna()
        & result["native_candidate_id"].notna()
        & (
            result["challenger_candidate_id"].astype(str)
            != result["native_candidate_id"].astype(str)
        )
    )
    if not declared_exists.equals(recomputed_exists):
        raise RuntimeError("D1 Test challenger-exists flag differs from candidate IDs")
    result["challenger_exists"] = recomputed_exists
    bearing = result["candidate_count"].gt(0)
    identity_columns = [
        "native_candidate_id",
        "native_identity_sha256",
        "native_geometry_sha256",
        "challenger_candidate_id",
        "challenger_identity_sha256",
        "challenger_geometry_sha256",
    ]
    if result.loc[bearing, identity_columns].isna().any().any():
        raise RuntimeError("D1 candidate-bearing Test row lacks gate identity")

    features = candidate_features.copy()
    for prefix, candidate_column in (
        ("native", "native_candidate_id"),
        ("challenger", "challenger_candidate_id"),
    ):
        selected = result.loc[:, ["sample_id", candidate_column]].copy()
        selected = selected.rename(columns={candidate_column: f"{prefix}_candidate_id"})
        result = result.drop(columns=[candidate_column])
        result = result.merge(
            _selected_features(features, selected, prefix=prefix),
            on="sample_id",
            how="left",
            validate="one_to_one",
        )

    scalar = [
        "native_calibrated_probability",
        "native_native_score",
        "native_overall_reliability",
        "native_perturbation_stability",
        "native_mask_reliability",
        "challenger_calibrated_probability",
        "challenger_native_score",
        "challenger_overall_reliability",
        "challenger_perturbation_stability",
        "challenger_mask_reliability",
    ]
    result[scalar] = result[scalar].fillna(0.0)
    result["ranker_score_margin"] = result["ensemble_score_margin"].fillna(0.0)
    result["calibrated_probability_delta"] = (
        result["challenger_calibrated_probability"]
        - result["native_calibrated_probability"]
    )
    result["native_score_delta"] = (
        result["challenger_native_score"] - result["native_native_score"]
    )
    result["overall_reliability_delta"] = (
        result["challenger_overall_reliability"] - result["native_overall_reliability"]
    )
    result["perturbation_stability_delta"] = (
        result["challenger_perturbation_stability"]
        - result["native_perturbation_stability"]
    )
    result["mask_reliability_delta"] = (
        result["challenger_mask_reliability"] - result["native_mask_reliability"]
    )
    result["challenger_exists_numeric"] = result["challenger_exists"].astype(float)
    result["score_margin"] = result["ranker_score_margin"]
    result["challenger_reliability"] = result["challenger_overall_reliability"].clip(
        0.0, 1.0
    )
    result["perturbation_stability"] = result["challenger_perturbation_stability"].clip(
        0.0, 1.0
    )
    result["prediction_source"] = "test_label_free"
    for column in ("native_candidate_id", "challenger_candidate_id"):
        result[column] = result[column].fillna("").astype(str)
    output_columns = [
        "sample_id",
        *(["scene_id"] if "scene_id" in result.columns else []),
        "prediction_source",
        "native_candidate_id",
        "challenger_candidate_id",
        "native_identity_sha256",
        "challenger_identity_sha256",
        "native_geometry_sha256",
        "challenger_geometry_sha256",
        "score_margin",
        "challenger_reliability",
        "perturbation_stability",
        "seed_challenger_votes",
        "candidate_id_unchanged",
        "geometry_hash_unchanged",
        "challenger_exists",
        "candidate_count",
        *SAFE_GATE_FEATURE_COLUMNS,
    ]
    output = result.loc[:, list(dict.fromkeys(output_columns))].copy()
    if not np.isfinite(output.loc[:, SAFE_GATE_FEATURE_COLUMNS].to_numpy(float)).all():
        raise RuntimeError("D1 label-free Test gate features are non-finite")
    return output
