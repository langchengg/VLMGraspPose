"""Fixed label-free Test inputs for the P12 four-route applications."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from unified_reranking.artifacts import (
    verified_artifact_path,
    verify_artifact_records_recursive,
)
from unified_reranking.gate import SAFE_GATE_FEATURE_COLUMNS
from unified_reranking.hashing import atomic_json, canonical_sha256, sha256_file

from .contracts import assert_label_free_parquet_schema
from .execution import load_content_manifest
from .four_route import FOUR_ROUTES, build_top20_union, validate_top20_union_frame
from .four_route_t4 import GEOMETRY_COLUMNS, write_t4_features
from .four_route_validation import (
    validate_four_route_router_selection,
    validate_four_route_source_plan,
    validate_p12_test_application,
    validate_t4_manifest,
    validate_top20_union_selection,
)
from .io import atomic_parquet


THREE_ROUTES = ("CROG", "G1", "C1")
TEST_INPUTS_RELATIVE = Path("13_four_route_extension/test_inputs")


def _record(path: str | Path) -> dict[str, Any]:
    source = Path(path).expanduser().resolve()
    if source.is_symlink() or not source.is_file():
        raise FileNotFoundError(f"P12 Test source is not regular: {source}")
    return {
        "path": str(source),
        "sha256": sha256_file(source),
        "bytes": source.stat().st_size,
    }


def _ordered_sample_ids(path: Path) -> pd.Series:
    assert_label_free_parquet_schema(path, name="P12 Test denominator")
    values = pd.read_parquet(path, columns=["sample_id"])["sample_id"].astype(str)
    if values.empty or values.duplicated().any() or values.eq("").any():
        raise RuntimeError("P12 Test denominator sample IDs differ")
    return values.reset_index(drop=True)


def _assert_sample_order(frame: pd.DataFrame, expected: pd.Series, *, name: str) -> None:
    observed = frame["sample_id"].astype(str).reset_index(drop=True)
    if observed.duplicated().any() or not observed.equals(expected):
        raise RuntimeError(f"{name} sample order/coverage differs")


def _canonical_candidates(path: Path, *, route: str) -> pd.DataFrame:
    assert_label_free_parquet_schema(path, name=f"P12 {route} Test Top5")
    frame = pd.read_parquet(path)
    score_column = "native_score" if "native_score" in frame else "native_score_raw"
    required = {
        "sample_id",
        "candidate_id",
        "native_rank",
        score_column,
        "candidate_geometry_sha256",
        *GEOMETRY_COLUMNS,
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"P12 {route} Test Top5 misses columns: {missing}")
    columns = [
        "sample_id",
        "candidate_id",
        "native_rank",
        score_column,
        "candidate_geometry_sha256",
        *GEOMETRY_COLUMNS,
    ]
    if "candidate_identity_sha256" in frame.columns:
        columns.append("candidate_identity_sha256")
    work = frame.loc[:, columns].copy().rename(columns={score_column: "native_score"})
    work[["sample_id", "candidate_id", "candidate_geometry_sha256"]] = work[
        ["sample_id", "candidate_id", "candidate_geometry_sha256"]
    ].astype(str)
    if work.duplicated(["sample_id", "candidate_id"]).any():
        raise RuntimeError(f"P12 {route} Test Top5 contains duplicate candidate keys")
    numeric = work[["native_rank", "native_score", *GEOMETRY_COLUMNS]].apply(
        pd.to_numeric, errors="coerce"
    )
    if (
        not np.isfinite(numeric.to_numpy(float)).all()
        or not np.equal(numeric["native_rank"], np.floor(numeric["native_rank"])).all()
        or ((numeric["native_rank"] < 1) | (numeric["native_rank"] > 5)).any()
        or (numeric[["width_px", "height_px"]] <= 0).any().any()
        or work["candidate_geometry_sha256"].eq("").any()
    ):
        raise RuntimeError(f"P12 {route} Test Top5 numeric/geometry contract differs")
    work[["native_rank", "native_score", *GEOMETRY_COLUMNS]] = numeric
    work["native_rank"] = work["native_rank"].astype(int)
    if "candidate_identity_sha256" not in work:
        work["candidate_identity_sha256"] = [
            canonical_sha256(
                {
                    "identity_contract": "p12_extension_route_raw_id_geometry_v1",
                    "route": route,
                    "sample_id": sample_id,
                    "candidate_id": candidate_id,
                    "candidate_geometry_sha256": geometry,
                }
            )
            for sample_id, candidate_id, geometry in work[
                ["sample_id", "candidate_id", "candidate_geometry_sha256"]
            ].itertuples(index=False, name=None)
        ]
    else:
        work["candidate_identity_sha256"] = work[
            "candidate_identity_sha256"
        ].astype(str)
    if work["candidate_identity_sha256"].eq("").any():
        raise RuntimeError(f"P12 {route} Test candidate identity differs")
    return work


def _attach_selected_contract(
    frame: pd.DataFrame,
    *,
    candidates: pd.DataFrame,
    route: str,
    selected_ids: pd.Series,
    selected_geometry: pd.Series,
) -> pd.DataFrame:
    prefix = route.lower()
    selected = pd.DataFrame(
        {
            "sample_id": frame["sample_id"].astype(str),
            "candidate_id": selected_ids.fillna("").astype(str),
            "declared_geometry": selected_geometry.fillna("").astype(str),
        }
    )
    bearing = selected["candidate_id"].ne("")
    matched = selected.loc[bearing].merge(
        candidates,
        on=["sample_id", "candidate_id"],
        how="left",
        validate="one_to_one",
    )
    if len(matched) != int(bearing.sum()) or matched["native_rank"].isna().any():
        raise RuntimeError(f"P12 {route} selected Test candidate is outside Top5")
    if not matched["declared_geometry"].equals(
        matched["candidate_geometry_sha256"].astype(str)
    ):
        raise RuntimeError(f"P12 {route} selected Test geometry differs from Top5")
    contract_columns = [
        "sample_id",
        "candidate_id",
        "candidate_geometry_sha256",
        "native_score",
        "native_rank",
        *GEOMETRY_COLUMNS,
    ]
    renamed = matched.loc[:, contract_columns].rename(
        columns={
            "candidate_id": f"{prefix}_candidate_id_contract",
            "candidate_geometry_sha256": f"{prefix}_candidate_geometry_sha256",
            **{
                column: f"{prefix}_{column}"
                for column in ("native_score", "native_rank", *GEOMETRY_COLUMNS)
            },
        }
    )
    result = frame.merge(renamed, on="sample_id", how="left", validate="one_to_one")
    contract_id = f"{prefix}_candidate_id_contract"
    observed = result[contract_id].fillna("").astype(str)
    if not observed.equals(selected["candidate_id"]):
        raise RuntimeError(f"P12 {route} selected Test candidate alignment differs")
    result = result.drop(columns=[contract_id])
    result[f"{prefix}_candidate_geometry_sha256"] = result[
        f"{prefix}_candidate_geometry_sha256"
    ].fillna("")
    return result


def build_router_test_frame(
    *,
    three_route_frame: pd.DataFrame,
    d1_gate_inputs: pd.DataFrame,
    d1_gate_decisions: pd.DataFrame,
    candidates_by_route: Mapping[str, pd.DataFrame],
    denominator_ids: pd.Series,
    feature_columns: Mapping[str, Sequence[str]],
) -> pd.DataFrame:
    """Assemble one exact label-free router Test frame."""

    if set(candidates_by_route) != set(FOUR_ROUTES):
        raise ValueError("P12 router candidates must contain all four routes")
    for name, frame in (
        ("three-route router Test", three_route_frame),
        ("D1 gate Test inputs", d1_gate_inputs),
        ("D1 gate Test decisions", d1_gate_decisions),
    ):
        _assert_sample_order(frame, denominator_ids, name=name)
        if set(frame["prediction_source"].astype(str)) != {"test_label_free"}:
            raise PermissionError(f"{name} provenance differs")
    result = three_route_frame.copy()
    for column in SAFE_GATE_FEATURE_COLUMNS:
        result[f"d1_{column}"] = pd.to_numeric(
            d1_gate_inputs[column], errors="raise"
        )
    counts = pd.to_numeric(d1_gate_decisions["candidate_count"], errors="raise").astype(
        int
    )
    result["d1_candidate_exists"] = counts.gt(0)
    result["d1_candidate_id"] = (
        d1_gate_decisions["selected_candidate_id"].fillna("").astype(str)
    )
    result["d1_margin"] = pd.to_numeric(
        d1_gate_inputs["score_margin"], errors="raise"
    )
    result["d1_reliability"] = pd.to_numeric(
        d1_gate_inputs["challenger_reliability"], errors="raise"
    )
    result["d1_stability"] = pd.to_numeric(
        d1_gate_inputs["perturbation_stability"], errors="raise"
    )
    for route in THREE_ROUTES:
        prefix = route.lower()
        result = _attach_selected_contract(
            result,
            candidates=candidates_by_route[route],
            route=route,
            selected_ids=result[f"{prefix}_candidate_id"],
            selected_geometry=result[
                f"{prefix}_selected_candidate_geometry_sha256"
            ],
        )
    result = _attach_selected_contract(
        result,
        candidates=candidates_by_route["D1"],
        route="D1",
        selected_ids=result["d1_candidate_id"],
        selected_geometry=d1_gate_decisions["selected_geometry_sha256"],
    )
    for route, columns in feature_columns.items():
        missing = sorted(set(map(str, columns)).difference(result.columns))
        if missing:
            raise RuntimeError(f"P12 {route} router Test features are absent: {missing}")
        matrix = result.loc[:, list(map(str, columns))].apply(
            pd.to_numeric, errors="coerce"
        )
        if not np.isfinite(matrix.to_numpy(float)).all():
            raise RuntimeError(f"P12 {route} router Test features are non-finite")
    result["prediction_source"] = "test_label_free"
    return result


def _route_union_frame(
    *,
    candidates: pd.DataFrame,
    features: pd.DataFrame,
    route: str,
    feature_columns: Sequence[str],
) -> pd.DataFrame:
    keys = ["sample_id", "candidate_id", "native_rank"]
    feature_work = features.copy()
    if "source_route" in feature_work.columns:
        feature_work = feature_work.loc[
            feature_work["source_route"].astype(str).str.upper().eq(route)
        ].copy()
        feature_work["candidate_id"] = feature_work["source_candidate_id"].astype(str)
        feature_work["native_rank"] = pd.to_numeric(
            feature_work["route_native_rank"], errors="raise"
        ).astype(int)
    if feature_work.duplicated(keys[:2]).any():
        raise RuntimeError(f"P12 {route} Test feature keys are duplicated")
    expected_keys = set(map(tuple, candidates[keys].to_numpy()))
    observed_keys = set(map(tuple, feature_work[keys].to_numpy()))
    if expected_keys != observed_keys or len(candidates) != len(feature_work):
        raise RuntimeError(f"P12 {route} Test feature membership differs from Top5")
    frozen = {
        "native_rank",
        "native_score",
        "candidate_identity_sha256",
        "candidate_geometry_sha256",
        *GEOMETRY_COLUMNS,
    }
    payload = [
        column
        for column in map(str, feature_columns)
        if column not in frozen
        and column not in keys
        and not column.startswith("union_route_")
    ]
    missing = sorted(set(payload).difference(feature_work.columns))
    if missing:
        raise RuntimeError(f"P12 {route} Test union features are absent: {missing}")
    result = candidates.merge(
        feature_work.loc[:, [*keys, *payload]],
        on=keys,
        validate="one_to_one",
        sort=False,
    )
    score = pd.to_numeric(feature_work["native_score_raw"], errors="raise")
    aligned = candidates[keys].merge(
        feature_work[[*keys, "native_score_raw"]], on=keys, validate="one_to_one"
    )
    if not np.array_equal(
        candidates["native_score"].to_numpy(float),
        pd.to_numeric(aligned["native_score_raw"], errors="raise").to_numpy(float),
    ) or not np.isfinite(score.to_numpy(float)).all():
        raise RuntimeError(f"P12 {route} Test native scores differ")
    result["native_score_raw"] = result["native_score"].astype(float)
    result["route"] = route
    return result


def build_union_test_frame(
    *,
    three_route_union: pd.DataFrame,
    d1_features: pd.DataFrame,
    candidates_by_route: Mapping[str, pd.DataFrame],
    denominator_ids: pd.Series,
    feature_columns: Sequence[str],
) -> pd.DataFrame:
    route_frames = {
        route: _route_union_frame(
            candidates=candidates_by_route[route],
            features=three_route_union if route in THREE_ROUTES else d1_features,
            route=route,
            feature_columns=feature_columns,
        )
        for route in FOUR_ROUTES
    }
    union, _audit = build_top20_union(
        route_frames, denominator_ids.tolist(), split="test"
    )
    validate_top20_union_frame(union, denominator_ids.tolist())
    missing = sorted(set(map(str, feature_columns)).difference(union.columns))
    if missing:
        raise RuntimeError(f"P12 Test Top20 model features are absent: {missing}")
    matrix = union.loc[:, list(map(str, feature_columns))].apply(
        pd.to_numeric, errors="coerce"
    )
    if np.isinf(matrix.to_numpy(float)).any():
        raise RuntimeError("P12 Test Top20 model features contain infinity")
    return union


def _same_frame(observed: pd.DataFrame, expected: pd.DataFrame, *, name: str) -> None:
    try:
        pd.testing.assert_frame_equal(
            observed.reset_index(drop=True),
            expected.reset_index(drop=True),
            check_dtype=True,
            check_exact=True,
        )
    except AssertionError as error:
        raise RuntimeError(f"{name} differs from deterministic replay") from error


def assemble_four_route_test_inputs(
    *, d1_run_dir: str | Path, three_route_run: str | Path, resume: bool
) -> dict[str, Any]:
    root = Path(d1_run_dir).expanduser().resolve()
    three_root = Path(three_route_run).expanduser().resolve()
    plan_path = root / "configs/d1_four_route_extension_plan.json"
    router_selection_path = (
        root / "13_four_route_extension/router/router_selection_manifest.json"
    )
    union_selection_path = (
        root / "13_four_route_extension/union/selected_union_ranker.json"
    )
    plan = validate_four_route_source_plan(plan_path)
    from .four_route_execution import validate_complete_execution

    execution = validate_complete_execution(root)
    router_selection = validate_four_route_router_selection(router_selection_path)
    union_selection = validate_top20_union_selection(union_selection_path)
    denominator_path = root / "01_manifests/d1_paired_manifest.parquet"
    denominator_ids = _ordered_sample_ids(denominator_path)
    three_router_path = (
        three_root / "08_lock/route_router_inputs/test_label_free.parquet"
    )
    d1_gate_manifest_path = root / "08_lock/label_free_test_gates/d1/manifest.json"
    d1_gate_manifest = load_content_manifest(
        d1_gate_manifest_path, name="D1 Test gate application", statuses=("COMPLETE",)
    )
    verify_artifact_records_recursive(
        {
            "sources": d1_gate_manifest.get("sources"),
            "artifacts": d1_gate_manifest.get("artifacts"),
        },
        name="D1 Test gate application",
        require_at_least_one=True,
    )
    d1_gate_inputs_path = verified_artifact_path(
        d1_gate_manifest["artifacts"]["inputs"],
        name="D1 gate Test inputs",
    )
    d1_gate_decisions_path = verified_artifact_path(
        d1_gate_manifest["artifacts"]["decisions"],
        name="D1 gate Test decisions",
    )
    three_union_path = (
        three_root
        / "03_features/tracks/T5_cross_route/union_test/candidate_features.parquet"
    )
    d1_t2_manifest_path = root / "03_features/test/top5/T2_matched_common/manifest.json"
    d1_t2_manifest = load_content_manifest(
        d1_t2_manifest_path, name="D1 Test Top5 T2", statuses=("COMPLETE",)
    )
    d1_t2_path = verified_artifact_path(
        d1_t2_manifest["artifacts"]["candidate_features"],
        name="D1 Test Top5 T2 features",
    )
    source_paths = [
        denominator_path,
        three_router_path,
        d1_gate_inputs_path,
        d1_gate_decisions_path,
        three_union_path,
        d1_t2_path,
    ]
    for path in source_paths:
        assert_label_free_parquet_schema(path, name=f"P12 Test source {path.name}")
    candidate_paths = {
        **{
            route: three_root / f"02_candidates/{route.lower()}_test_top5.parquet"
            for route in THREE_ROUTES
        },
        "D1": root / "02_candidates/d1_test_top5.parquet",
    }
    candidates = {
        route: _canonical_candidates(path, route=route)
        for route, path in candidate_paths.items()
    }
    router_frame = build_router_test_frame(
        three_route_frame=pd.read_parquet(three_router_path),
        d1_gate_inputs=pd.read_parquet(d1_gate_inputs_path),
        d1_gate_decisions=pd.read_parquet(d1_gate_decisions_path),
        candidates_by_route=candidates,
        denominator_ids=denominator_ids,
        feature_columns=router_selection["configuration"]["feature_columns"],
    )
    union_frame = build_union_test_frame(
        three_route_union=pd.read_parquet(three_union_path),
        d1_features=pd.read_parquet(d1_t2_path),
        candidates_by_route=candidates,
        denominator_ids=denominator_ids,
        feature_columns=union_selection["feature_columns"],
    )
    destination = root / TEST_INPUTS_RELATIVE
    router_path = atomic_parquet(
        router_frame, destination / "router/test_label_free.parquet"
    )
    union_path = atomic_parquet(union_frame, destination / "union/test_top20.parquet")
    t4_manifest = write_t4_features(
        split="test",
        d1_top5_path=candidate_paths["D1"],
        t3_manifest_path=root / "03_features/test/top5/T3_route_rich/manifest.json",
        peer_top5_paths={route: candidate_paths[route] for route in THREE_ROUTES},
        output_dir=root / "03_features/test/top5/T4_four_route_consensus",
        source_paths={
            "p12_plan": plan_path,
            "p12_validation_execution": Path(execution["event"]["path"]),
            "p12_router_selection": router_selection_path,
            "p12_union_selection": union_selection_path,
            "p12_test_router_inputs": router_path,
            "p12_test_union_inputs": union_path,
            "producer": Path(__file__),
        },
        resume=resume,
    )
    sources = {
        "p12_plan": _record(plan_path),
        "p12_validation_execution": _record(Path(execution["event"]["path"])),
        "router_selection": _record(router_selection_path),
        "union_selection": _record(union_selection_path),
        "denominator": _record(denominator_path),
        "three_route_router_test": _record(three_router_path),
        "d1_gate_manifest": _record(d1_gate_manifest_path),
        "d1_gate_inputs": _record(d1_gate_inputs_path),
        "d1_gate_decisions": _record(d1_gate_decisions_path),
        "three_route_union_test": _record(three_union_path),
        "d1_t2_manifest": _record(d1_t2_manifest_path),
        "d1_t2_features": _record(d1_t2_path),
        "candidate_top5": {
            route: _record(path) for route, path in candidate_paths.items()
        },
        "producer": _record(Path(__file__)),
    }
    artifacts = {
        "router_test_input": _record(router_path),
        "union_test_top20": _record(union_path),
        "t4_test_manifest": _record(
            root / "03_features/test/top5/T4_four_route_consensus/manifest.json"
        ),
    }
    configuration = {
        "split": "test",
        "routes": list(FOUR_ROUTES),
        "pool": "exact_four_route_top20_no_dedup",
        "prediction_source": "test_label_free",
        "candidate_test_labels_read": False,
        "plan_content_sha256": plan["content_sha256"],
        "t4_content_sha256": t4_manifest["content_sha256"],
    }
    signature = canonical_sha256({"configuration": configuration, "sources": sources})
    result: dict[str, Any] = {
        "schema_version": 1,
        "status": "COMPLETE",
        "configuration": configuration,
        "signature_sha256": signature,
        "sources": sources,
        "artifacts": artifacts,
        "router_rows": int(len(router_frame)),
        "union_rows": int(len(union_frame)),
        "candidate_test_labels_read": False,
        "selection_used_test_metrics": False,
    }
    result["content_sha256"] = canonical_sha256(result)
    manifest_path = destination / "manifest.json"
    if manifest_path.exists():
        existing = load_content_manifest(
            manifest_path, name="P12 Test inputs", statuses=("COMPLETE",)
        )
        verify_artifact_records_recursive(
            {"sources": existing["sources"], "artifacts": existing["artifacts"]},
            name="P12 Test inputs",
            require_at_least_one=True,
        )
        if not resume or existing != result:
            raise RuntimeError("immutable P12 Test inputs exist and differ")
        return existing
    atomic_json(manifest_path, result)
    return result


def validate_four_route_test_inputs(
    run_dir: str | Path, *, require_applications: bool
) -> dict[str, Any]:
    """Replay the fixed Test inputs and bind them to downstream applications."""

    root = Path(run_dir).expanduser().resolve()
    manifest_path = root / TEST_INPUTS_RELATIVE / "manifest.json"
    manifest = load_content_manifest(
        manifest_path, name="P12 Test inputs", statuses=("COMPLETE",)
    )
    verify_artifact_records_recursive(
        {"sources": manifest["sources"], "artifacts": manifest["artifacts"]},
        name="P12 Test inputs",
        require_at_least_one=True,
    )
    plan_path = root / "configs/d1_four_route_extension_plan.json"
    plan = validate_four_route_source_plan(plan_path)
    from .four_route_execution import validate_complete_execution

    execution = validate_complete_execution(root)
    router_selection_path = (
        root / "13_four_route_extension/router/router_selection_manifest.json"
    )
    union_selection_path = (
        root / "13_four_route_extension/union/selected_union_ranker.json"
    )
    router_selection = validate_four_route_router_selection(router_selection_path)
    union_selection = validate_top20_union_selection(union_selection_path)
    three_root = Path(
        plan["sources"]["completed_three_route_final_lock"]["path"]
    ).resolve().parent
    denominator_path = root / "01_manifests/d1_paired_manifest.parquet"
    denominator_ids = _ordered_sample_ids(denominator_path)
    three_router_path = (
        three_root / "08_lock/route_router_inputs/test_label_free.parquet"
    )
    three_union_path = (
        three_root
        / "03_features/tracks/T5_cross_route/union_test/candidate_features.parquet"
    )
    d1_gate_manifest_path = root / "08_lock/label_free_test_gates/d1/manifest.json"
    d1_gate_manifest = load_content_manifest(
        d1_gate_manifest_path, name="D1 Test gate application", statuses=("COMPLETE",)
    )
    d1_gate_inputs_path = verified_artifact_path(
        d1_gate_manifest["artifacts"]["inputs"],
        name="D1 gate Test inputs",
    )
    d1_gate_decisions_path = verified_artifact_path(
        d1_gate_manifest["artifacts"]["decisions"],
        name="D1 gate Test decisions",
    )
    d1_t2_manifest_path = root / "03_features/test/top5/T2_matched_common/manifest.json"
    d1_t2_manifest = load_content_manifest(
        d1_t2_manifest_path, name="D1 Test Top5 T2", statuses=("COMPLETE",)
    )
    d1_t2_path = verified_artifact_path(
        d1_t2_manifest["artifacts"]["candidate_features"],
        name="D1 Test Top5 T2 features",
    )
    candidate_paths = {
        **{
            route: three_root / f"02_candidates/{route.lower()}_test_top5.parquet"
            for route in THREE_ROUTES
        },
        "D1": root / "02_candidates/d1_test_top5.parquet",
    }
    expected_sources = {
        "p12_plan": _record(plan_path),
        "p12_validation_execution": _record(Path(execution["event"]["path"])),
        "router_selection": _record(router_selection_path),
        "union_selection": _record(union_selection_path),
        "denominator": _record(denominator_path),
        "three_route_router_test": _record(three_router_path),
        "d1_gate_manifest": _record(d1_gate_manifest_path),
        "d1_gate_inputs": _record(d1_gate_inputs_path),
        "d1_gate_decisions": _record(d1_gate_decisions_path),
        "three_route_union_test": _record(three_union_path),
        "d1_t2_manifest": _record(d1_t2_manifest_path),
        "d1_t2_features": _record(d1_t2_path),
        "candidate_top5": {
            route: _record(path) for route, path in candidate_paths.items()
        },
        "producer": _record(Path(__file__)),
    }
    if manifest["sources"] != expected_sources:
        raise RuntimeError("P12 Test input source inventory differs")
    candidates = {
        route: _canonical_candidates(path, route=route)
        for route, path in candidate_paths.items()
    }
    expected_router = build_router_test_frame(
        three_route_frame=pd.read_parquet(three_router_path),
        d1_gate_inputs=pd.read_parquet(d1_gate_inputs_path),
        d1_gate_decisions=pd.read_parquet(d1_gate_decisions_path),
        candidates_by_route=candidates,
        denominator_ids=denominator_ids,
        feature_columns=router_selection["configuration"]["feature_columns"],
    )
    expected_union = build_union_test_frame(
        three_route_union=pd.read_parquet(three_union_path),
        d1_features=pd.read_parquet(d1_t2_path),
        candidates_by_route=candidates,
        denominator_ids=denominator_ids,
        feature_columns=union_selection["feature_columns"],
    )
    router_path = verified_artifact_path(
        manifest["artifacts"]["router_test_input"], name="P12 router Test input"
    )
    union_path = verified_artifact_path(
        manifest["artifacts"]["union_test_top20"], name="P12 Top20 Test input"
    )
    _same_frame(
        pd.read_parquet(router_path), expected_router, name="P12 router Test input"
    )
    _same_frame(
        pd.read_parquet(union_path), expected_union, name="P12 Top20 Test input"
    )
    t4_manifest_path = verified_artifact_path(
        manifest["artifacts"]["t4_test_manifest"], name="P12 Test T4 manifest"
    )
    validate_t4_manifest(t4_manifest_path, split="test")
    if manifest.get("router_rows") != len(expected_router) or manifest.get(
        "union_rows"
    ) != len(expected_union):
        raise RuntimeError("P12 Test input row counts differ")
    records: dict[str, Any] = {
        "manifest": _record(manifest_path),
        "router": _record(router_path),
        "union": _record(union_path),
        "t4": _record(t4_manifest_path),
    }
    if require_applications:
        application_paths = {
            "four_route_crog_default_router": root
            / "08_lock/formal_inputs/four_route_crog_default_router/manifest.json",
            "top20_union": root / "08_lock/formal_inputs/top20_union/manifest.json",
        }
        for system, application_path in application_paths.items():
            application = validate_p12_test_application(
                application_path, system=system
            )
            expected_input = (
                manifest["artifacts"]["router_test_input"]
                if system == "four_route_crog_default_router"
                else manifest["artifacts"]["union_test_top20"]
            )
            if (
                application["sources"].get("test_inputs") != expected_input
                or application["sources"].get("denominator")
                != manifest["sources"]["denominator"]
            ):
                raise RuntimeError(
                    f"P12 {system} application does not bind fixed Test inputs"
                )
            records[f"application_{system}"] = _record(application_path)
    return records


__all__ = [
    "TEST_INPUTS_RELATIVE",
    "assemble_four_route_test_inputs",
    "build_router_test_frame",
    "build_union_test_frame",
    "validate_four_route_test_inputs",
]
