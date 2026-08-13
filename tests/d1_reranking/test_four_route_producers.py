from __future__ import annotations

import hashlib
import json
from pathlib import Path
import pickle

import pandas as pd
import pytest

from unified_reranking.hashing import canonical_sha256, sha256_file

from d1_reranking.four_route import FOUR_ROUTES, build_top20_union
from d1_reranking.four_route_producer import (
    Top20UnionBudget,
    apply_locked_four_route_test,
    run_four_route_router_validation,
    run_top20_union_validation,
)
from d1_reranking.four_route_t4 import write_t4_features
from d1_reranking.four_route_validation import (
    validate_four_route_router_selection,
    validate_p12_test_application,
    validate_t4_manifest,
    validate_top20_union_selection,
)


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _write(path: Path, value: str = "fixed") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")
    return path


def _refresh_manifest_record(
    manifest_path: Path, *, section: str, key: str, artifact_path: Path
) -> None:
    value = json.loads(manifest_path.read_text(encoding="utf-8"))
    value[section][key] = {
        "path": str(artifact_path.resolve()),
        "sha256": sha256_file(artifact_path),
        "bytes": artifact_path.stat().st_size,
    }
    if section == "sources":
        value["signature_sha256"] = canonical_sha256(
            {"configuration": value["configuration"], "sources": value["sources"]}
        )
    value.pop("content_sha256")
    value["content_sha256"] = canonical_sha256(value)
    manifest_path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")


def _router_inputs(
    root: Path, *, no_go: bool = False
) -> tuple[Path, Path, dict[str, tuple[str, ...]]]:
    feature_columns = {
        route: (f"{route.lower()}_signal", f"{route.lower()}_candidate_exists_feature")
        for route in ("G1", "C1", "D1")
    }
    train_rows = []
    patterns = ((False, True), (True, False), (False, False), (True, True))
    for fold in range(5):
        for pattern_index, (crog, challenger) in enumerate(patterns):
            row: dict[str, object] = {
                "sample_id": f"train-{fold}-{pattern_index}",
                "scene_id": f"train-scene-{fold}-{pattern_index}",
                "oof_fold": fold,
                "prediction_source": "train_oof",
                "crog_correct": crog,
            }
            for route_index, route in enumerate(("G1", "C1", "D1")):
                row[f"{route.lower()}_correct"] = challenger
                row[f"{route.lower()}_signal"] = float(
                    pattern_index + route_index * 0.1
                )
                row[f"{route.lower()}_candidate_exists_feature"] = 1.0
            train_rows.append(row)
    validation_rows = []
    for index in range(8):
        crog = bool(index % 2)
        row = {
            "sample_id": f"validation-{index}",
            "scene_id": f"validation-scene-{index}",
            "prediction_source": "validation",
            "crog_correct": crog,
            "three_route_router_correct": crog,
            "existing_top15_oracle": crog,
        }
        for route_index, route in enumerate(FOUR_ROUTES):
            correct = (
                crog
                if no_go or route == "CROG"
                else bool((index + route_index) % 3 == 0)
            )
            row[f"{route.lower()}_correct"] = correct
            row[f"{route.lower()}_candidate_id"] = "duplicate-raw-id"
            row[f"{route.lower()}_candidate_geometry_sha256"] = _sha(f"{route}-{index}")
        for route_index, route in enumerate(("G1", "C1", "D1")):
            row[f"{route.lower()}_signal"] = float(index % 4 + route_index * 0.1)
            row[f"{route.lower()}_candidate_exists_feature"] = 1.0
            row[f"{route.lower()}_margin"] = 1.0
            row[f"{route.lower()}_reliability"] = 1.0
            row[f"{route.lower()}_stability"] = 1.0
            row[f"{route.lower()}_candidate_exists"] = True
        validation_rows.append(row)
    train_path = root / "router_train.parquet"
    validation_path = root / "router_validation.parquet"
    pd.DataFrame(train_rows).to_parquet(train_path, index=False)
    pd.DataFrame(validation_rows).to_parquet(validation_path, index=False)
    return train_path, validation_path, feature_columns


def test_router_producer_replays_108_trials_and_detects_tampering(
    tmp_path: Path,
) -> None:
    train, validation, features = _router_inputs(tmp_path)
    code = _write(tmp_path / "router_code.py")
    output = tmp_path / "router"
    manifest = run_four_route_router_validation(
        train_oof_path=train,
        validation_path=validation,
        feature_columns=features,
        output_dir=output,
        source_paths={"producer": code},
        resume=False,
    )
    assert len(manifest["selection"]["trials"]) == 108
    assert validate_four_route_router_selection(
        output / "router_selection_manifest.json"
    )["decision"] in {"GO", "NO_GO_CROG"}

    trials = Path(manifest["artifacts"]["validation_trials"]["path"])
    frame = pd.read_parquet(trials)
    frame.loc[0, "harmful"] += 1
    frame.to_parquet(trials, index=False)
    _refresh_manifest_record(
        output / "router_selection_manifest.json",
        section="artifacts",
        key="validation_trials",
        artifact_path=trials,
    )
    with pytest.raises(RuntimeError, match="differs"):
        validate_four_route_router_selection(output / "router_selection_manifest.json")


@pytest.mark.parametrize("target", ["transition_models", "validation"])
def test_router_replay_rejects_manifest_rehashed_model_or_input(
    tmp_path: Path, target: str
) -> None:
    train, validation, features = _router_inputs(tmp_path)
    output = tmp_path / target
    manifest = run_four_route_router_validation(
        train_oof_path=train,
        validation_path=validation,
        feature_columns=features,
        output_dir=output,
        source_paths={"producer": _write(tmp_path / f"{target}_code.py")},
        resume=False,
    )
    manifest_path = output / "router_selection_manifest.json"
    if target == "transition_models":
        artifact_path = Path(manifest["artifacts"][target]["path"])
        artifact_path.write_bytes(artifact_path.read_bytes() + b"tampered")
        section = "artifacts"
    else:
        artifact_path = validation
        frame = pd.read_parquet(artifact_path)
        frame.loc[0, "g1_signal"] = float(frame.loc[0, "g1_signal"]) + 0.25
        frame.to_parquet(artifact_path, index=False)
        section = "sources"
    _refresh_manifest_record(
        manifest_path, section=section, key=target, artifact_path=artifact_path
    )
    with pytest.raises(RuntimeError, match="differs"):
        validate_four_route_router_selection(manifest_path)


def test_router_producer_preserves_explicit_no_go(tmp_path: Path) -> None:
    train, validation, features = _router_inputs(tmp_path, no_go=True)
    manifest = run_four_route_router_validation(
        train_oof_path=train,
        validation_path=validation,
        feature_columns=features,
        output_dir=tmp_path / "router_no_go",
        source_paths={"producer": _write(tmp_path / "router_no_go_code.py")},
        resume=False,
    )
    assert manifest["decision"] == "NO_GO_CROG"
    decisions = pd.read_parquet(manifest["artifacts"]["validation_decisions"]["path"])
    assert set(decisions["selected_route"]) == {"CROG"}


def _candidate_frame(route: str, samples: list[str], *, labels: bool) -> pd.DataFrame:
    rows = []
    for sample_index, sample_id in enumerate(samples):
        positive_rank = sample_index % 5 + 1
        for rank in range(1, 6):
            row = {
                "sample_id": sample_id,
                "route": route,
                "candidate_id": f"raw-{rank}",
                "native_rank": rank,
                "native_score": float(6 - rank) / 6.0,
                "base_logit": float(6 - rank) / 6.0,
                "candidate_identity_sha256": _sha(
                    f"identity-{route}-{sample_id}-{rank}"
                ),
                "candidate_geometry_sha256": _sha(
                    f"geometry-{route}-{sample_id}-{rank}"
                ),
                "cx_px": float(10 * rank + FOUR_ROUTES.index(route)),
                "cy_px": float(5 * rank + sample_index),
                "theta_deg": float(10 * rank),
                "width_px": float(20 + rank),
                "height_px": 20.0,
                "feature_a": float(rank),
                "feature_b": float(sample_index % 3),
            }
            if labels:
                row["candidate_success"] = (
                    rank == positive_rank and route == FOUR_ROUTES[sample_index % 4]
                )
                row["jacquard_margin"] = 1.0 if row["candidate_success"] else -1.0
            rows.append(row)
    return pd.DataFrame(rows)


def _union_frame(samples: list[str], *, labels: bool) -> pd.DataFrame:
    frames = {
        route: _candidate_frame(route, samples, labels=labels) for route in FOUR_ROUTES
    }
    union, _audit = build_top20_union(
        frames, samples, split="validation" if labels else "test"
    )
    return union


@pytest.fixture(scope="module")
def trained_union(tmp_path_factory: pytest.TempPathFactory) -> dict[str, object]:
    root = tmp_path_factory.mktemp("p12_union")
    train_samples = [f"train-{index}" for index in range(20)]
    validation_samples = [f"validation-{index}" for index in range(8)]
    train = _union_frame(train_samples, labels=True)
    validation = _union_frame(validation_samples, labels=True)
    train_path = root / "train.parquet"
    validation_path = root / "validation.parquet"
    train.to_parquet(train_path, index=False)
    validation.to_parquet(validation_path, index=False)
    folds = pd.DataFrame(
        {"sample_id": train_samples, "fold": [index % 5 for index in range(20)]}
    )
    folds_path = root / "folds.parquet"
    folds.to_parquet(folds_path, index=False)
    train_denominator = root / "train_denominator.parquet"
    validation_denominator = root / "validation_denominator.parquet"
    pd.DataFrame({"sample_id": train_samples}).to_parquet(
        train_denominator, index=False
    )
    pd.DataFrame({"sample_id": validation_samples}).to_parquet(
        validation_denominator, index=False
    )
    budget = Top20UnionBudget(
        lambdamart_num_leaves=3,
        lambdamart_learning_rate=0.1,
        lambdamart_n_estimators=2,
        deepsets_learning_rate=1e-3,
        deepsets_weight_decay=1e-4,
        deepsets_alpha=0.1,
        deepsets_epochs=1,
        deepsets_patience=1,
        deepsets_batch_size=32,
    )
    output = root / "union"
    selection = run_top20_union_validation(
        train_path=train_path,
        validation_path=validation_path,
        folds_path=folds_path,
        train_denominator_path=train_denominator,
        validation_denominator_path=validation_denominator,
        feature_columns=(
            "feature_a",
            "feature_b",
            "union_route_crog",
            "union_route_g1",
            "union_route_c1",
            "union_route_d1",
        ),
        output_dir=output,
        source_paths={"producer": _write(root / "union_code.py")},
        budget=budget,
        resume=False,
    )
    return {
        "root": root,
        "output": output,
        "selection": selection,
        "validation": validation,
        "validation_denominator": validation_denominator,
    }


def test_union_producer_executes_exact_36_cell_two_encoder_plan(
    trained_union: dict[str, object],
) -> None:
    output = Path(trained_union["output"])
    selection = trained_union["selection"]
    assert len(list((output / "cells").glob("*/manifest.json"))) == 36
    assert {row["encoder"] for row in selection["trials"]} == {
        "lambdamart",
        "deepsets",
    }
    replay = validate_top20_union_selection(
        output / "selected_union_ranker.json", require_formal_budget=False
    )
    assert replay["selected_encoder"] == selection["selected_encoder"]


def _t4_candidate(route: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sample_id": ["sample-1", "sample-1"],
            "candidate_id": ["duplicate-raw", "route-second"],
            "native_rank": [1, 2],
            "native_score": [0.8, 0.4],
            "candidate_identity_sha256": [_sha(f"{route}-i1"), _sha(f"{route}-i2")],
            "candidate_geometry_sha256": [_sha(f"{route}-g1"), _sha(f"{route}-g2")],
            "cx_px": [10.0 + FOUR_ROUTES.index(route), 50.0],
            "cy_px": [20.0, 60.0],
            "theta_deg": [10.0, 40.0],
            "width_px": [30.0, 25.0],
            "height_px": [20.0, 20.0],
        }
    )


def _t3_manifest(root: Path, d1_path: Path, *, split: str = "test") -> Path:
    frame = pd.read_parquet(d1_path)
    frame["calibrated_native_probability"] = [0.8, 0.4]
    frame["base_logit"] = [1.4, -0.4]
    frame["overall_feature_reliability"] = [0.9, 0.7]
    frame["peak_retention_rate"] = [0.95, 0.75]
    frame["perturbed_valid_fraction"] = [0.9, 0.7]
    frame["mask_reliability"] = [0.85, 0.65]
    columns = [
        "calibrated_native_probability",
        "base_logit",
        "overall_feature_reliability",
        "peak_retention_rate",
        "perturbed_valid_fraction",
        "mask_reliability",
    ]
    feature_path = root / "candidate_features.parquet"
    feature_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(feature_path, index=False)
    record = {
        "path": str(feature_path.resolve()),
        "sha256": sha256_file(feature_path),
        "bytes": feature_path.stat().st_size,
    }
    value: dict[str, object] = {
        "schema_version": 1,
        "status": "COMPLETE",
        "configuration": {
            "route": "D1",
            "split": split,
            "pool": "top5",
            "track": "T3_route_rich",
        },
        "model_feature_columns": columns,
        "model_feature_schema_sha256": canonical_sha256(columns),
        "candidate_test_labels_read": False,
        "artifacts": {"candidate_features": record},
    }
    value["content_sha256"] = canonical_sha256(value)
    manifest_path = root / "manifest.json"
    manifest_path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")
    return manifest_path


def test_t4_builder_is_hash_bound_replayable_and_geometry_complete(
    tmp_path: Path,
) -> None:
    paths = {}
    for route in FOUR_ROUTES:
        path = tmp_path / f"{route}.parquet"
        _t4_candidate(route).to_parquet(path, index=False)
        paths[route] = path
    output = tmp_path / "t4"
    t3_manifest = _t3_manifest(tmp_path / "t3", paths["D1"])
    manifest = write_t4_features(
        split="test",
        d1_top5_path=paths["D1"],
        t3_manifest_path=t3_manifest,
        peer_top5_paths={route: paths[route] for route in ("CROG", "G1", "C1")},
        output_dir=output,
        source_paths={"builder": _write(tmp_path / "t4_code.py")},
        resume=False,
    )
    features = pd.read_parquet(manifest["artifacts"]["candidate_features"]["path"])
    assert {
        "calibrated_native_probability",
        "t4_crog_rotated_iou",
        "t4_g1_mutual_nearest",
        "t4_c1_native_score",
        "t4_agreement_mean",
        "cx_px",
        "theta_deg",
    }.issubset(features.columns)
    assert manifest["model_feature_columns"][:6] == [
        "calibrated_native_probability",
        "base_logit",
        "overall_feature_reliability",
        "peak_retention_rate",
        "perturbed_valid_fraction",
        "mask_reliability",
    ]
    validate_t4_manifest(output / "manifest.json", split="test")


def test_t4_test_schema_fails_before_pandas_row_access(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = {}
    for route in FOUR_ROUTES:
        frame = _t4_candidate(route)
        if route == "D1":
            frame["candidate_success"] = False
        path = tmp_path / f"{route}.parquet"
        frame.to_parquet(path, index=False)
        paths[route] = path

    t3_manifest = _t3_manifest(tmp_path / "bad_t3", paths["CROG"])

    def forbidden(*_args: object, **_kwargs: object) -> pd.DataFrame:
        raise AssertionError("pandas row access occurred before schema rejection")

    monkeypatch.setattr(pd, "read_parquet", forbidden)
    with pytest.raises(PermissionError, match="forbidden Test columns"):
        write_t4_features(
            split="test",
            d1_top5_path=paths["D1"],
            t3_manifest_path=t3_manifest,
            peer_top5_paths={route: paths[route] for route in ("CROG", "G1", "C1")},
            output_dir=tmp_path / "bad_t4",
            source_paths={"builder": _write(tmp_path / "bad_t4_code.py")},
            resume=False,
        )


def test_t4_widening_rejects_t3_candidate_key_drift(tmp_path: Path) -> None:
    paths = {}
    for route in FOUR_ROUTES:
        path = tmp_path / f"{route}.parquet"
        _t4_candidate(route).to_parquet(path, index=False)
        paths[route] = path
    t3_manifest = _t3_manifest(tmp_path / "t3_drift", paths["D1"])
    t3_path = tmp_path / "t3_drift/candidate_features.parquet"
    t3 = pd.read_parquet(t3_path)
    t3.loc[0, "candidate_id"] = "drifted-candidate"
    t3.to_parquet(t3_path, index=False)
    _refresh_manifest_record(
        t3_manifest,
        section="artifacts",
        key="candidate_features",
        artifact_path=t3_path,
    )

    with pytest.raises(RuntimeError, match="candidate membership differs"):
        write_t4_features(
            split="test",
            d1_top5_path=paths["D1"],
            t3_manifest_path=t3_manifest,
            peer_top5_paths={route: paths[route] for route in ("CROG", "G1", "C1")},
            output_dir=tmp_path / "bad_t4_drift",
            source_paths={"builder": _write(tmp_path / "t4_code.py")},
            resume=False,
        )


def test_locked_test_apply_emits_formal_normalized_geometry(
    trained_union: dict[str, object],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    router_train, router_validation, router_features = _router_inputs(tmp_path)
    router_output = tmp_path / "router"
    run_four_route_router_validation(
        train_oof_path=router_train,
        validation_path=router_validation,
        feature_columns=router_features,
        output_dir=router_output,
        source_paths={"producer": _write(tmp_path / "router_code.py")},
        resume=False,
    )
    router_test = pd.read_parquet(router_validation).drop(
        columns=[
            "crog_correct",
            "g1_correct",
            "c1_correct",
            "d1_correct",
            "three_route_router_correct",
            "existing_top15_oracle",
        ]
    )
    router_test["prediction_source"] = "test_label_free"
    for route_index, route in enumerate(FOUR_ROUTES):
        prefix = route.lower()
        router_test[f"{prefix}_native_score"] = 0.5
        router_test[f"{prefix}_native_rank"] = 1
        router_test[f"{prefix}_cx_px"] = 10.0 + route_index
        router_test[f"{prefix}_cy_px"] = 20.0
        router_test[f"{prefix}_theta_deg"] = 15.0
        router_test[f"{prefix}_width_px"] = 30.0
        router_test[f"{prefix}_height_px"] = 20.0
    router_test_path = tmp_path / "router_test.parquet"
    router_test.to_parquet(router_test_path, index=False)

    validation = trained_union["validation"]
    union_test = validation.drop(columns=["candidate_success", "jacquard_margin"])
    union_test_path = tmp_path / "union_test.parquet"
    union_test.to_parquet(union_test_path, index=False)
    denominator = tmp_path / "denominator.parquet"
    pd.DataFrame({"sample_id": sorted(union_test["sample_id"].unique())}).to_parquet(
        denominator, index=False
    )

    def forbidden_pickle_load(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("P12 Test replay must not call pickle.load")

    monkeypatch.setattr(pickle, "load", forbidden_pickle_load)
    results = apply_locked_four_route_test(
        router_selection_path=router_output / "router_selection_manifest.json",
        router_test_input_path=router_test_path,
        union_selection_path=Path(trained_union["output"])
        / "selected_union_ranker.json",
        union_test_feature_path=union_test_path,
        denominator_path=denominator,
        output_dir=tmp_path / "test_apply",
        resume=False,
        require_formal_budget=False,
    )
    for system, manifest in results.items():
        universe = pd.read_parquet(manifest["artifacts"]["candidate_universe"]["path"])
        assert {
            "native_score",
            "cx_px",
            "cy_px",
            "theta_deg",
            "width_px",
            "height_px",
        }.issubset(universe.columns)
        validate_p12_test_application(
            tmp_path / "test_apply" / system / "manifest.json",
            system=system,
            require_formal_budget=False,
        )
    union_manifest_path = tmp_path / "test_apply/top20_union/manifest.json"
    union_manifest = json.loads(union_manifest_path.read_text(encoding="utf-8"))
    decision_path = Path(union_manifest["artifacts"]["per_sample_decisions"]["path"])
    decisions = pd.read_parquet(decision_path)
    decisions.loc[0, "selected_candidate_id"] = "D1:tampered"
    decisions.to_parquet(decision_path, index=False)
    _refresh_manifest_record(
        union_manifest_path,
        section="artifacts",
        key="per_sample_decisions",
        artifact_path=decision_path,
    )
    with pytest.raises(RuntimeError, match="semantic replay"):
        validate_p12_test_application(
            union_manifest_path,
            system="top20_union",
            require_formal_budget=False,
        )


def test_locked_test_apply_rejects_supervision_before_row_access(
    trained_union: dict[str, object], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    router_train, router_validation, router_features = _router_inputs(tmp_path)
    router_output = tmp_path / "router"
    run_four_route_router_validation(
        train_oof_path=router_train,
        validation_path=router_validation,
        feature_columns=router_features,
        output_dir=router_output,
        source_paths={"producer": _write(tmp_path / "router_code.py")},
        resume=False,
    )
    from d1_reranking import four_route_validation

    router_selection, rebuilt_router = (
        four_route_validation.rebuild_validated_four_route_router(
            router_output / "router_selection_manifest.json"
        )
    )
    monkeypatch.setattr(
        four_route_validation,
        "rebuild_validated_four_route_router",
        lambda _path: (router_selection, rebuilt_router),
    )
    monkeypatch.setattr(
        four_route_validation,
        "validate_top20_union_selection",
        lambda _path, *, require_formal_budget: trained_union["selection"],
    )
    bad_union = trained_union["validation"].copy()
    bad_union_path = tmp_path / "bad_union_test.parquet"
    bad_union.to_parquet(bad_union_path, index=False)
    safe_router = tmp_path / "safe_router.parquet"
    pd.DataFrame({"sample_id": ["x"]}).to_parquet(safe_router, index=False)
    denominator = tmp_path / "denominator.parquet"
    pd.DataFrame({"sample_id": ["x"]}).to_parquet(denominator, index=False)

    def forbidden(*_args: object, **_kwargs: object) -> pd.DataFrame:
        raise AssertionError("row access occurred before Test schema rejection")

    monkeypatch.setattr(pd, "read_parquet", forbidden)
    with pytest.raises(PermissionError, match="forbidden Test columns"):
        apply_locked_four_route_test(
            router_selection_path=router_output / "router_selection_manifest.json",
            router_test_input_path=safe_router,
            union_selection_path=Path(trained_union["output"])
            / "selected_union_ranker.json",
            union_test_feature_path=bad_union_path,
            denominator_path=denominator,
            output_dir=tmp_path / "bad_apply",
            resume=False,
            require_formal_budget=False,
        )
