from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from d1_reranking.four_route import (
    FOUR_ROUTES,
    FourRouteRouterSelection,
    build_top20_union,
    four_route_decisions,
    select_four_route_operating_point,
    validate_top20_union_frame,
)
from d1_reranking.four_route_plan import (
    D1_SOURCE_NAMES,
    THREE_ROUTE_SOURCE_NAMES,
    write_four_route_plan,
    write_validation_router_union_artifacts,
)
from d1_reranking.four_route_execution import load_four_route_execution_plan
from unified_reranking.hashing import canonical_sha256, sha256_file
from unified_reranking.route_router import RouterEvidence, RouterOperatingPoint


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _record(path: Path) -> dict[str, object]:
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
    }


def _content_manifest(path: Path, artifact: Path) -> Path:
    value: dict[str, object] = {
        "schema_version": 1,
        "status": "COMPLETE",
        "candidate_test_labels_read": False,
        "artifacts": {"predictions": _record(artifact)},
    }
    value["content_sha256"] = canonical_sha256(value)
    return _write(path, json.dumps(value, sort_keys=True))


def _top5(route: str, *, success_at: int | None = None) -> pd.DataFrame:
    rows = []
    for rank in range(1, 6):
        rows.append(
            {
                "sample_id": "sample-1",
                "route": route,
                "candidate_id": f"raw-{rank}",
                "native_rank": rank,
                "candidate_identity_sha256": f"identity-{route}-{rank}",
                "candidate_geometry_sha256": f"geometry-{rank}",
                **(
                    {"candidate_success": rank == success_at}
                    if success_at is not None
                    else {}
                ),
            }
        )
    return pd.DataFrame(rows)


def _evidence(
    margin: list[float], *, exists: list[bool] | None = None
) -> RouterEvidence:
    length = len(margin)
    return RouterEvidence(
        route_margin=margin,
        reliability=np.ones(length),
        perturbation_stability=np.ones(length),
        candidate_exists=np.ones(length, dtype=bool) if exists is None else exists,
    )


def _point() -> RouterOperatingPoint:
    return RouterOperatingPoint(
        lambda_router=1.0,
        utility_threshold=0.0,
        margin_threshold=0.0,
        reliability_threshold=0.0,
        stability_threshold=0.0,
    )


def test_top20_preserves_exact_route_membership_and_duplicate_raw_ids() -> None:
    inputs = {route: _top5(route) for route in FOUR_ROUTES}
    output, audit = build_top20_union(inputs, ["sample-1"], split="test")

    assert len(output) == 20
    assert output["candidate_id"].nunique() == 20
    assert output["native_rank"].tolist() == list(range(1, 21))
    assert output.iloc[:4]["candidate_id"].tolist() == [
        "CROG:raw-1",
        "G1:raw-1",
        "C1:raw-1",
        "D1:raw-1",
    ]
    assert audit["full_top20_samples"] == 1
    assert audit["raw_id_duplicate_rows_across_routes_retained"] == 20
    assert audit["geometry_duplicate_rows_across_routes_retained"] == 20
    assert audit["candidate_test_labels_read"] is False

    tampered = output.copy()
    tampered.loc[0, "candidate_id"] = "WRONG:raw-1"
    with pytest.raises(ValueError, match="route-qualified"):
        validate_top20_union_frame(tampered, ["sample-1"])

    inputs["D1"] = inputs["D1"].assign(candidate_success=False)
    with pytest.raises(PermissionError, match="supervision"):
        build_top20_union(inputs, ["sample-1"], split="test")


def test_top20_rejects_duplicate_raw_identity_within_route() -> None:
    inputs = {route: _top5(route) for route in FOUR_ROUTES}
    inputs["D1"].loc[1, "candidate_id"] = inputs["D1"].loc[0, "candidate_id"]
    with pytest.raises(ValueError, match="duplicate raw"):
        build_top20_union(inputs, ["sample-1"], split="validation")


def test_four_route_router_defaults_to_crog_and_uses_declared_tie_break() -> None:
    probabilities = {
        "G1": (np.array([0.8, 0.4, 0.9]), np.array([0.1, 0.1, 0.1])),
        "C1": (np.array([0.8, 0.5, 0.8]), np.array([0.1, 0.1, 0.1])),
        "D1": (np.array([0.7, 0.9, 0.9]), np.array([0.1, 0.1, 0.1])),
    }
    evidence = {
        "G1": _evidence([1.0, 1.0, 1.0], exists=[True, True, False]),
        "C1": _evidence([1.0, 1.0, 1.0], exists=[True, True, False]),
        "D1": _evidence([1.0, 1.0, 1.0], exists=[True, True, False]),
    }
    decisions = four_route_decisions(probabilities, evidence, _point())
    assert decisions.tolist() == ["G1", "D1", "CROG"]

    route_correct = {
        "CROG": [True, False, True],
        "G1": [True, False, True],
        "C1": [True, False, True],
        "D1": [True, False, True],
    }
    selection = select_four_route_operating_point(
        probabilities,
        evidence,
        route_correct,
        ["scene-1", "scene-2", "scene-3"],
        [_point()],
        bootstrap_iterations=25,
    )
    assert selection.status == "NO_GO_CROG"
    assert selection.selected_operating_point is None
    assert selection.artifact()["candidate_test_labels_read"] is False


def _completed_three_route_run(root: Path) -> dict[str, Path]:
    sources = {
        name: _write(root / "outputs" / f"{name}.bin", f"frozen-{name}")
        for name in THREE_ROUTE_SOURCE_NAMES
    }
    inventory = [_record(path) for path in sources.values()]
    lock: dict[str, object] = {
        "schema_version": 1,
        "status": "COMPLETE",
        "inventory": inventory,
        "inventory_count": len(inventory),
        "inventory_content_sha256": canonical_sha256(inventory),
    }
    lock["self_sha256"] = canonical_sha256(lock)
    lock_path = _write(root / "FINAL_RUN_LOCK.json", json.dumps(lock, sort_keys=True))
    digest = sha256_file(lock_path)
    _write(root / "COMPLETE", f"COMPLETE\nFINAL_RUN_LOCK.json sha256={digest}\n")
    return sources


def _d1_sources(root: Path) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for name in D1_SOURCE_NAMES:
        path = root / "sources" / f"{name}.bin"
        if name == "test_top5":
            path = root / "sources" / "test_top5.parquet"
            path.parent.mkdir(parents=True, exist_ok=True)
            _top5("D1").to_parquet(path, index=False)
        elif name in {"test_ranker_application", "test_gate_application"}:
            artifact = _write(root / "sources" / f"{name}_predictions.bin", "safe")
            path = _content_manifest(root / "sources" / f"{name}.json", artifact)
        elif name.endswith("_t3_manifest"):
            split = name.removesuffix("_t3_manifest")
            feature_path = root / "sources" / f"{split}_t3_features.parquet"
            feature_path.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(
                {
                    "sample_id": ["sample-1"],
                    "candidate_id": ["candidate-1"],
                    "native_score_raw": [0.8],
                }
            ).to_parquet(feature_path, index=False)
            value: dict[str, object] = {
                "schema_version": 1,
                "status": "COMPLETE",
                "configuration": {
                    "route": "D1",
                    "split": split,
                    "pool": "top5",
                    "track": "T3_route_rich",
                },
                "model_feature_columns": ["native_score_raw"],
                "model_feature_schema_sha256": canonical_sha256(("native_score_raw",)),
                "candidate_test_labels_read": False,
                "artifacts": {"candidate_features": _record(feature_path)},
            }
            value["content_sha256"] = canonical_sha256(value)
            path = _write(
                root / "sources" / f"{name}.json",
                json.dumps(value, sort_keys=True),
            )
        else:
            _write(path, f"d1-{name}")
        result[name] = path
    return result


def test_plan_binds_completed_inventory_and_rejects_resume_drift(
    tmp_path: Path,
) -> None:
    three_root = tmp_path / "completed-three"
    d1_root = tmp_path / "d1"
    three_sources = _completed_three_route_run(three_root)
    d1_sources = _d1_sources(d1_root)
    lock_sha256 = sha256_file(three_root / "FINAL_RUN_LOCK.json")
    completed_snapshot = {
        path: sha256_file(path) for path in three_root.rglob("*") if path.is_file()
    }

    plan = write_four_route_plan(
        d1_run_dir=d1_root,
        completed_three_route_run=three_root,
        three_route_sources=three_sources,
        d1_sources=d1_sources,
        code_paths=(Path(__file__),),
        resume=False,
        expected_final_lock_sha256=lock_sha256,
    )
    assert plan["completed_three_route_run_mode"] == "STRICT_READ_ONLY"
    assert plan["router_contract"]["default_route"] == "CROG"
    assert plan["top20_union_contract"]["maximum_candidates"] == 20
    assert plan["candidate_test_labels_read"] is False
    assert completed_snapshot == {
        path: sha256_file(path) for path in three_root.rglob("*") if path.is_file()
    }

    resumed = write_four_route_plan(
        d1_run_dir=d1_root,
        completed_three_route_run=three_root,
        three_route_sources=three_sources,
        d1_sources=d1_sources,
        code_paths=(Path(__file__),),
        resume=True,
        expected_final_lock_sha256=lock_sha256,
    )
    assert resumed == plan

    _write(d1_sources["validation_r7_predictions"], "drift")
    with pytest.raises(RuntimeError, match="exists and differs"):
        write_four_route_plan(
            d1_run_dir=d1_root,
            completed_three_route_run=three_root,
            three_route_sources=three_sources,
            d1_sources=d1_sources,
            code_paths=(Path(__file__),),
            resume=True,
            expected_final_lock_sha256=lock_sha256,
        )


def test_plan_rejects_completed_source_not_bound_by_inventory(tmp_path: Path) -> None:
    three_root = tmp_path / "completed-three"
    d1_root = tmp_path / "d1"
    three_sources = _completed_three_route_run(three_root)
    d1_sources = _d1_sources(d1_root)
    lock_sha256 = sha256_file(three_root / "FINAL_RUN_LOCK.json")
    _write(three_sources["validation_gated_top5"], "post-lock drift")
    with pytest.raises(RuntimeError, match="not frozen by final lock"):
        write_four_route_plan(
            d1_run_dir=d1_root,
            completed_three_route_run=three_root,
            three_route_sources=three_sources,
            d1_sources=d1_sources,
            code_paths=(Path(__file__),),
            resume=False,
            expected_final_lock_sha256=lock_sha256,
        )


def test_plan_expands_exact_validation_producer_spec_without_test(
    tmp_path: Path,
) -> None:
    three_root = tmp_path / "completed-three"
    d1_root = tmp_path / "d1"
    three_sources = _completed_three_route_run(three_root)
    d1_sources = _d1_sources(d1_root)

    def development(name: str) -> str:
        return str(_write(d1_root / "producer_inputs" / f"{name}.bin", name))

    spec: dict[str, object] = {
        "schema_version": 1,
        "status": "VALIDATION_PRODUCER_DECLARED",
        "router": {
            "train_oof": development("router_train_oof"),
            "validation": development("router_validation"),
            "feature_columns": {
                route: [f"{route.lower()}_feature"] for route in ("G1", "C1", "D1")
            },
        },
        "union": {
            "train_top20": development("union_train_top20"),
            "validation_top20": development("union_validation_top20"),
            "folds": development("folds"),
            "train_denominator": development("train_denominator"),
            "validation_denominator": development("validation_denominator"),
            "feature_columns": ["feature_a"],
        },
        "t4": {
            "train": {
                "d1_top5": development("train_d1_top5"),
                "t3_manifest": str(d1_sources["train_t3_manifest"]),
                "peer_top5": {
                    route: str(three_sources["train_oof_predictions"])
                    for route in ("CROG", "G1", "C1")
                },
            },
            "validation": {
                "d1_top5": str(d1_sources["validation_top5"]),
                "t3_manifest": str(d1_sources["validation_t3_manifest"]),
                "peer_top5": {
                    route: str(three_sources["validation_gated_top5"])
                    for route in ("CROG", "G1", "C1")
                },
            },
        },
        "candidate_test_labels_read": False,
        "selection_used_test_metrics": False,
    }
    spec["content_sha256"] = canonical_sha256(spec)
    spec_path = _write(
        d1_root / "configs/p12_validation_producer_spec.json",
        json.dumps(spec, sort_keys=True),
    )
    plan = write_four_route_plan(
        d1_run_dir=d1_root,
        completed_three_route_run=three_root,
        three_route_sources=three_sources,
        d1_sources=d1_sources,
        producer_spec_path=spec_path,
        code_paths=(Path(__file__),),
        resume=False,
        expected_final_lock_sha256=sha256_file(three_root / "FINAL_RUN_LOCK.json"),
    )
    _path, loaded = load_four_route_execution_plan(d1_root)
    producer = loaded["sources"]["validation_producer"]
    assert plan["validation_producer_ready"] is True
    assert producer["job_count"] == 5
    assert set(producer["t4"]) == {"train", "validation"}
    assert producer["test_inputs_referenced"] is False


def test_plan_rejects_test_application_label_access(tmp_path: Path) -> None:
    three_root = tmp_path / "completed-three"
    d1_root = tmp_path / "d1"
    three_sources = _completed_three_route_run(three_root)
    d1_sources = _d1_sources(d1_root)
    application = d1_sources["test_gate_application"]
    value = json.loads(application.read_text(encoding="utf-8"))
    value.pop("content_sha256")
    value["candidate_test_labels_read"] = True
    value["content_sha256"] = canonical_sha256(value)
    _write(application, json.dumps(value, sort_keys=True))

    with pytest.raises(PermissionError, match="candidate_test_labels_read=false"):
        write_four_route_plan(
            d1_run_dir=d1_root,
            completed_three_route_run=three_root,
            three_route_sources=three_sources,
            d1_sources=d1_sources,
            code_paths=(Path(__file__),),
            resume=False,
            expected_final_lock_sha256=sha256_file(three_root / "FINAL_RUN_LOCK.json"),
        )


def test_validation_writer_is_prelock_ready_and_content_addressed(
    tmp_path: Path,
) -> None:
    union_inputs = {
        "CROG": _top5("CROG", success_at=0),
        "G1": _top5("G1", success_at=0),
        "C1": _top5("C1", success_at=0),
        "D1": _top5("D1", success_at=1),
    }
    union, _audit = build_top20_union(union_inputs, ["sample-1"], split="validation")
    samples = pd.DataFrame(
        {
            "sample_id": ["sample-1"],
            "scene_id": ["scene-1"],
            "prediction_source": ["validation"],
            "crog_correct": [False],
            "g1_correct": [False],
            "c1_correct": [False],
            "d1_correct": [True],
            "three_route_router_correct": [False],
            "four_route_decision": ["D1"],
            "existing_top15_oracle": [False],
        }
    )
    selection = FourRouteRouterSelection(
        status="GO",
        selected_operating_point=_point(),
        trials=(),
        bootstrap_iterations=25,
    )
    source = _write(tmp_path / "input" / "validation_predictions.bin", "fixed")
    run = tmp_path / "run"
    manifest = write_validation_router_union_artifacts(
        run,
        samples=samples,
        top20_union=union,
        router_selection=selection,
        source_paths={"validation_predictions": source},
        resume=False,
    )
    table = pd.read_csv(run / "13_four_route_extension" / "validation_router_union.csv")
    assert set(FOUR_ROUTES).issubset(set(table["route"]))
    assert table.loc[table["metric"] == "oracle_delta_vs_top15", "value"].iloc[0] == 1.0
    assert manifest["candidate_test_labels_read"] is False
    unsigned = dict(manifest)
    assert unsigned.pop("content_sha256") == canonical_sha256(unsigned)

    resumed = write_validation_router_union_artifacts(
        run,
        samples=samples,
        top20_union=union,
        router_selection=selection,
        source_paths={"validation_predictions": source},
        resume=True,
    )
    assert resumed == manifest
    _write(source, "drift")
    with pytest.raises(RuntimeError, match="exist and differ"):
        write_validation_router_union_artifacts(
            run,
            samples=samples,
            top20_union=union,
            router_selection=selection,
            source_paths={"validation_predictions": source},
            resume=True,
        )
