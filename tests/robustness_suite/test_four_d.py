from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from robustness_suite.four_d import (
    THRESHOLD_GRID,
    _apply_duplicate_filter,
    _assert_feature_candidate_geometry,
    _common_filtered_sample_ids,
    _fit_preprocessor_on_train_only,
    _formal_regression_expected,
    _k1_no_reranking_row,
    _list_features_for_prefix,
    _load_valid_seed_artifact,
    _ranker_hyperparameters,
    _route_sources,
    _sha256_file,
    assert_threshold_monotonicity,
    load_frozen_canonical_evaluator,
    native_prefix,
    run_duplicate_exclusion,
    sequence_id_from_scene,
    threshold_success,
)


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_4d_same_gt_matching() -> None:
    # GT-A clears IoU only and GT-B clears angle only.  Combining those two
    # different GTs must not create a success.
    assert not threshold_success(
        [0.40, 0.10],
        [45.0, 5.0],
        iou_threshold=0.25,
        angle_threshold_deg=30.0,
    )
    assert threshold_success(
        [0.40, 0.30],
        [45.0, 5.0],
        iou_threshold=0.25,
        angle_threshold_deg=30.0,
    )


def test_4d_angle_periodicity() -> None:
    evaluator, _path, _digest = load_frozen_canonical_evaluator(REPO_ROOT)
    assert evaluator.periodic_angle_error_deg(89.0, -89.0) == pytest.approx(2.0)
    assert evaluator.periodic_angle_error_deg(10.0, -170.0) == pytest.approx(0.0)


def test_threshold_grid_complete() -> None:
    assert len(THRESHOLD_GRID) == 9
    assert len(set(THRESHOLD_GRID)) == 9
    assert {cell[0] for cell in THRESHOLD_GRID} == {0.20, 0.25, 0.30}
    assert {cell[1] for cell in THRESHOLD_GRID} == {20, 30, 40}


def test_4d_primary_threshold_reproduces_locked_metrics() -> None:
    assert _formal_regression_expected(REPO_ROOT, "crog") == {
        "denominator": 7675,
        "native": 6848,
        "raw": 7087,
        "reranked": 7089,
        "oracle": 7219,
    }
    assert _formal_regression_expected(REPO_ROOT, "g1") == {
        "denominator": 7675,
        "native": 3647,
        "raw": 4339,
        "reranked": 4347,
        "oracle": 4469,
    }
    assert _formal_regression_expected(REPO_ROOT, "c1") == {
        "denominator": 7675,
        "native": 3363,
        "raw": 4318,
        "reranked": 4317,
        "oracle": 4512,
    }
    assert _formal_regression_expected(REPO_ROOT, "d1") == {
        "denominator": 7675,
        "native": 2525,
        "raw": 3954,
        "reranked": 3954,
        "oracle": 4527,
    }


def test_threshold_boundary_operators_are_strict_iou_and_inclusive_angle() -> None:
    assert not threshold_success(
        [0.25], [30.0], iou_threshold=0.25, angle_threshold_deg=30.0
    )
    assert threshold_success(
        [0.250001], [30.0], iou_threshold=0.25, angle_threshold_deg=30.0
    )


def test_threshold_monotonicity() -> None:
    rows = []
    for iou, angle in THRESHOLD_GRID:
        # Looser angle and lower IoU can only add successes.
        value = 100 - int(round(100 * iou)) + angle
        rows.append(
            {
                "route": "CROG",
                "iou_threshold": iou,
                "angle_threshold_deg": angle,
                "native_success_num": value,
                "reranked_success_num": value + 1,
                "oracle_num": value + 2,
            }
        )
    assert_threshold_monotonicity(pd.DataFrame(rows))


def test_threshold_monotonicity_rejects_bug() -> None:
    frame = pd.DataFrame(
        [
            {
                "route": "CROG",
                "iou_threshold": 0.20,
                "angle_threshold_deg": 30,
                "native_success_num": 10,
            },
            {
                "route": "CROG",
                "iou_threshold": 0.25,
                "angle_threshold_deg": 30,
                "native_success_num": 11,
            },
        ]
    )
    with pytest.raises(RuntimeError, match="monotonicity"):
        assert_threshold_monotonicity(frame)


def _candidate_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sample_id": ["a", "a", "a", "b", "b"],
            "candidate_id": ["a1", "a2", "a3", "b1", "b2"],
            "native_rank": [1, 2, 3, 1, 2],
            "native_score": [3.0, 2.0, 1.0, 2.0, 1.0],
        }
    )


def test_topk_is_native_prefix() -> None:
    result = native_prefix(_candidate_frame(), 2)
    assert result.groupby("sample_id")["candidate_id"].apply(list).to_dict() == {
        "a": ["a1", "a2"],
        "b": ["b1", "b2"],
    }


def test_topk_candidate_ids_unique() -> None:
    frame = _candidate_frame()
    frame.loc[2, "candidate_id"] = "a2"
    with pytest.raises(ValueError, match="not unique"):
        native_prefix(frame, 3)


def test_topk_no_candidate_padding() -> None:
    result = native_prefix(_candidate_frame(), 5)
    assert len(result.loc[result["sample_id"].eq("a")]) == 3
    assert len(result.loc[result["sample_id"].eq("b")]) == 2


def test_topk_k1_has_no_fake_reranking() -> None:
    row = _k1_no_reranking_row({"route": "CROG", "k": "1", "native_num": 7})
    assert row["native_num"] == 7
    assert row["gate_status"] == "NOT_APPLICABLE_K1"
    for column in (
        "raw_reranked_num",
        "raw_reranked_j1",
        "gated_num",
        "gated_j1",
        "recovered",
        "harmful",
    ):
        assert pd.isna(row[column])


def test_topk_retraining_uses_train_val_only() -> None:
    train = pd.DataFrame({"sample_id": ["train-a", "train-b"], "feature": [1, 2]})

    class SpyPreprocessor:
        seen: pd.DataFrame | None = None
        seen_columns: tuple[str, ...] | None = None

        @classmethod
        def fit(cls, frame: pd.DataFrame, columns: tuple[str, ...]) -> object:
            cls.seen = frame.copy()
            cls.seen_columns = tuple(columns)
            return object()

    _fit_preprocessor_on_train_only(SpyPreprocessor, train, ("feature",))
    assert SpyPreprocessor.seen is not None
    assert SpyPreprocessor.seen["sample_id"].tolist() == ["train-a", "train-b"]
    assert SpyPreprocessor.seen_columns == ("feature",)


def test_topk_prefix_list_features_match_frozen_extractor() -> None:
    from unified_reranking.feature_extractors.common import _list_features

    frame = pd.DataFrame(
        {
            "sample_id": ["a", "a", "a", "b"],
            "candidate_id": ["a2", "a1", "a3", "b1"],
            "native_rank": [2, 1, 3, 1],
            "native_score_raw": [0.2, 0.8, -0.1, 2.0],
        }
    )
    expected = _list_features(frame)
    observed = _list_features_for_prefix(frame)
    pd.testing.assert_frame_equal(observed, expected, check_exact=True)


def test_topk_uses_exact_locked_lambdamart_hyperparameters() -> None:
    for route in ("crog", "g1", "c1"):
        assert _ranker_hyperparameters(REPO_ROOT, route) == {
            "num_leaves": 63,
            "learning_rate": 0.05,
            "n_estimators": 400,
        }
    assert _ranker_hyperparameters(REPO_ROOT, "d1") == {
        "num_leaves": 31,
        "learning_rate": 0.05,
        "n_estimators": 200,
    }
    assert _route_sources(REPO_ROOT)["d1"].retrospective is True


def test_topk_resume_binds_seed_to_contract_and_outputs(tmp_path: Path) -> None:
    cell_dir = tmp_path / "models/crog/k3"
    cell_dir.mkdir(parents=True)
    seed = 20260815
    score_frame = pd.DataFrame(
        {
            "sample_id": ["a", "a"],
            "candidate_id": ["a1", "a2"],
            "native_rank": [1, 2],
        }
    )
    scores = score_frame.assign(score=[0.2, 0.8])
    score_path = cell_dir / f"seed_{seed}_scores.parquet"
    model_path = cell_dir / f"seed_{seed}_model.txt"
    manifest_path = cell_dir / f"seed_{seed}.json"
    scores.to_parquet(score_path, index=False)
    model_path.write_text("frozen-model", encoding="utf-8")
    hyperparameters = {
        "num_leaves": 63,
        "learning_rate": 0.05,
        "n_estimators": 400,
    }
    manifest_path.write_text(
        json.dumps(
            {
                "status": "COMPLETE",
                "seed": seed,
                "k": 3,
                "route": "CROG",
                "hyperparameters": hyperparameters,
                "preprocessing_fit_split": "train",
                "ranker_runtime_ms_per_group": 0.1,
                "candidate_ids_frozen": True,
                "candidate_geometry_unchanged": True,
            }
        ),
        encoding="utf-8",
    )
    contract = {
        "source_sha256": {"source": "abc"},
        "config_sha256": "def",
        "preprocessor_sha256": "ghi",
        "candidate_keys_sha256": "jkl",
        "cell_signature_sha256": "mno",
    }
    loaded = _load_valid_seed_artifact(
        seed=seed,
        route="crog",
        k=3,
        cell_dir=cell_dir,
        score_frame=score_frame,
        hyperparameters=hyperparameters,
        contract=contract,
    )
    assert loaded is not None
    upgraded = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert upgraded["score_sha256"] == _sha256_file(score_path)
    assert upgraded["model_sha256"] == _sha256_file(model_path)
    assert upgraded["cell_signature_sha256"] == "mno"
    scores.assign(score=[0.9, 0.1]).to_parquet(score_path, index=False)
    assert (
        _load_valid_seed_artifact(
            seed=seed,
            route="crog",
            k=3,
            cell_dir=cell_dir,
            score_frame=score_frame,
            hyperparameters=hyperparameters,
            contract=contract,
        )
        is None
    )


def test_topk_rejects_noncontiguous_native_ranks() -> None:
    frame = _candidate_frame()
    frame.loc[2, "native_rank"] = 4
    with pytest.raises(ValueError, match="not contiguous"):
        native_prefix(frame, 3)


def test_topk_candidate_geometry_frozen() -> None:
    candidates = _candidate_frame().assign(
        cx_px=[1.0, 2.0, 3.0, 4.0, 5.0],
        cy_px=1.0,
        theta_deg=0.0,
        width_px=20.0,
        height_px=10.0,
    )
    features = candidates.copy()
    _assert_feature_candidate_geometry(candidates, features)
    features.loc[0, "theta_deg"] = 1.0
    with pytest.raises(RuntimeError, match="geometry differs"):
        _assert_feature_candidate_geometry(candidates, features)


def test_sequence_id_uses_sequence_not_tuple() -> None:
    scene = "ARID10/floor/top/non-fruits/seq09,result_2018-08-27.png"
    assert sequence_id_from_scene(scene) == "ARID10/floor/top/non-fruits/seq09"


def _paired_predictions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "route": ["CROG", "CROG", "CROG", "G1", "G1", "G1"],
            "sample_id": ["a", "b", "c", "b", "c", "d"],
            "native_correct": [True, False, True, False, True, False],
            "reranked_correct": [True, True, False, True, True, False],
        }
    )


def test_duplicate_exclusion_deterministic(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    run_dir = repo_root / "artifacts/robustness_suite/run"
    run_dir.mkdir(parents=True)
    (run_dir / "SOURCE_AUDIT.md").write_text(
        "Near duplicates: repository search found no train/test near-duplicate map.\n"
        "This subexperiment is fail-closed.\n",
        encoding="utf-8",
    )
    first = run_duplicate_exclusion(repo_root, run_dir, resume=False)
    first_bytes = (
        run_dir / "4d_near_duplicate_excluded/source_duplicate_map.json"
    ).read_bytes()
    second = run_duplicate_exclusion(repo_root, run_dir, resume=False)
    second_bytes = (
        run_dir / "4d_near_duplicate_excluded/source_duplicate_map.json"
    ).read_bytes()
    assert first == second
    assert first_bytes == second_bytes
    assert json.loads(first_bytes)["status"] == "FAIL_CLOSED"


def test_4d_outputs_require_isolated_robustness_run_dir(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="isolated child"):
        run_duplicate_exclusion(REPO_ROOT, tmp_path / "not-a-run", resume=False)


def test_no_unflagged_tuple_removed() -> None:
    marked, filtered = _apply_duplicate_filter(_paired_predictions(), {"c"})
    assert set(marked.loc[marked["excluded"], "sample_id"]) == {"c"}
    assert set(filtered["sample_id"]) == {"a", "b", "d"}


def test_filtered_predictions_identical_to_source() -> None:
    source = _paired_predictions()
    _marked, filtered = _apply_duplicate_filter(source, {"c"})
    expected = source.loc[source["sample_id"].ne("c")].reset_index(drop=True)
    pd.testing.assert_frame_equal(
        filtered[list(source.columns)].reset_index(drop=True), expected
    )


def test_common_intersection_consistent() -> None:
    paired = _paired_predictions()
    assert _common_filtered_sample_ids(paired, {"c"}) == ("b",)
