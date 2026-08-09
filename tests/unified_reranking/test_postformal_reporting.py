from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from unified_reranking.postformal_reporting import (
    FIGURE_NAMES,
    TABLE_NAMES,
    FormalBundle,
    _access_log_check,
    _current_source_hash_checks,
    _feature_leakage_check,
    _native_score_source_check,
    classify_failures,
    deployment_decision,
    deterministic_case_selection,
    formal_results_table,
    hash_inventory,
    normalize_candidate_labels,
    shared_hifi_mask_check,
)
from unified_reranking.hashing import canonical_sha256, sha256_file
from unified_reranking.ledger import initialize_ledger, ledger_stage
from tools.unified_reranking.build_postformal_artifacts import (
    _validation_cells,
    build_figures,
    build_postlock_bridge,
    build_tables,
    required_analysis_registry,
    verify_final_readiness,
    write_tables,
)


ROUTES = ("crog", "g1", "c1")


def _write_valid_ledger(run: Path) -> None:
    ledger = initialize_ledger(run / "run_ledger.sqlite")
    for stage, substage in (
        ("P15", "independent_recompute"),
        ("POSTFORMAL", "tables_figures_galleries_reports_and_prefinal_integrity"),
    ):
        with ledger_stage(
            ledger,
            stage=stage,
            substage=substage,
            command=f"python -m synthetic.{substage}",
        ):
            pass


def _artifact_record(path: Path) -> dict[str, str]:
    return {"path": str(path.resolve()), "sha256": sha256_file(path)}


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")


def _write_encoder_phase(run: Path, *, wrong_selection: bool = False) -> Path:
    columns = ["safe_feature"]
    schema = canonical_sha256(columns)
    screen = run / "05_models" / "screen_reference" / "manifest.json"
    _write_json(screen, {"feature_columns": columns})
    selection = run / "05_models" / "encoder_loss_selections.json"
    choice = {
        "encoder": "mlp",
        "loss": "ranknet",
        "parameters": {"num_attention_blocks": 2},
        "screen_manifest": str(screen.resolve()),
        "screen_manifest_sha256": sha256_file(screen),
    }
    _write_json(
        selection,
        {
            "status": "CONTROLLED_LOSS_LOCKED_FOR_ENCODER_COMPARISON",
            "selections": {"crog/T2_matched_common": choice},
        },
    )
    source = run / "synthetic" / "source.bin"
    output = run / "synthetic" / "output.bin"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_bytes(b"source")
    output.write_bytes(b"output")
    feature_manifest = (
        run
        / "03_features/tracks/T2_matched_common/crog_validation/feature_manifest.json"
    )
    _write_json(
        feature_manifest,
        {
            "status": "COMPLETE",
            "route": "crog",
            "split": "validation",
            "track": "T2_matched_common",
            "feature_extraction_latency_ms": 1.5,
        },
    )
    configuration = {
        "route": "crog",
        "track": "T2_matched_common",
        "encoder": "mlp",
        "loss": "ranknet",
        "seed": 42,
        "mode": "validation",
        "num_attention_blocks": 2,
        "source_identity": {"selected_feature_schema_sha256": schema},
    }
    cell_key = canonical_sha256(configuration)[:16]
    cell = run / "07_validation" / "matrix_cells" / cell_key / "manifest.json"
    _write_json(
        cell,
        {
            "status": "COMPLETE",
            "cell_key": cell_key,
            "configuration": configuration,
            "feature_columns": columns,
            "feature_schema_sha256": schema,
            "metrics": {"j_at_1": 0.5, "mrr_at_5": 0.5},
            "parameter_count": 3,
            "ranker_latency_ms": 0.2,
            "feature_latency_ms": 0.4,
            "feature_extraction_latency_ms": 1.5,
            "peak_memory_mb": 10.0,
            "missing_feature_rate": 0.0,
            "sources": {
                "input": _artifact_record(source),
                "validation_feature_manifest": _artifact_record(feature_manifest),
                "validation_feature_extraction_benchmark": None,
            },
            "artifacts": {"output": _artifact_record(output)},
        },
    )
    command = [
        "python",
        "-m",
        "tools.unified_reranking.train_matrix_cell",
        "--run-dir",
        str(run.resolve()),
        "--route",
        "crog",
        "--track",
        "T2_matched_common",
        "--encoder",
        "mlp",
        "--loss",
        "ranknet",
        "--seed",
        "42",
        "--mode",
        "validation",
        "--num-attention-blocks",
        "2",
    ]
    identifier = canonical_sha256(tuple(map(str, command)))[:16]
    selection_record = _artifact_record(selection)
    used_selection = selection_record
    if wrong_selection:
        other = run / "05_models" / "wrong_encoder_loss_selections.json"
        _write_json(other, {"status": "WRONG", "selections": {}})
        used_selection = _artifact_record(other)
    planner = (
        Path(__file__).resolve().parents[2]
        / "src"
        / "unified_reranking"
        / "matrix_phase.py"
    )
    plan = run / "05_models" / "matrix_plans" / "encoder_synthetic.json"
    _write_json(
        plan,
        {
            "status": "PLANNED",
            "phase": "encoder",
            "job_count": 1,
            "selection": used_selection,
            "planner_tool": _artifact_record(planner),
            "jobs": [{"identifier": identifier, "command": command}],
        },
    )
    record = _artifact_record(cell)
    result = {
        "identifier": identifier,
        "returncode": 0,
        "output_manifest": record,
    }
    _write_json(
        run / "05_models" / "matrix_plans" / "encoder_latest_execution.json",
        {
            "status": "COMPLETE",
            "phase": "encoder",
            "plan": _artifact_record(plan),
            "selection": used_selection,
            "job_count": 1,
            "results": [result],
            "output_manifests": [record],
        },
    )
    return cell


def test_candidate_labels_follow_locked_method_variant_normalization() -> None:
    pools = {
        route: pd.DataFrame(
            {
                "sample_id": ["s1", "s1"],
                "candidate_id": ["a", "b"],
                "native_rank": [1, 2],
                "candidate_geometry_sha256": ["ga", "gb"],
            }
        )
        for route in ROUTES
    }
    rows = []
    for route, variant in zip(ROUTES, ("crog_native", "g1", "c1"), strict=True):
        for candidate, success in (("a", 0), ("b", 1)):
            rows.append(
                {
                    "method": route.upper(),
                    "variant": variant,
                    "sample_id": "s1",
                    "candidate_id": candidate,
                    "candidate_success": success,
                    "diagnostic_iou": 0.5,
                }
            )
    rows.append(
        {
            "method": "G1",
            "variant": "unlocked_secondary",
            "sample_id": "s1",
            "candidate_id": "outside",
            "candidate_success": 1,
            "diagnostic_iou": 1.0,
        }
    )
    labels = normalize_candidate_labels(
        pd.DataFrame(rows),
        label_manifest={
            "normalization": {
                "route_column": "method",
                "variant_column": "variant",
                "include_variants": ["crog_native", "g1", "c1"],
            }
        },
        sample_ids={"s1"},
        candidate_pools=pools,
    )
    assert set(labels["route"]) == set(ROUTES)
    assert set(labels["candidate_id"]) == {"a", "b"}
    assert len(labels) == 6
    assert "diagnostic_iou" in labels


def test_access_log_allows_opaque_prelock_hash_but_not_row_open(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import unified_reranking.postformal_reporting as reporting

    run = tmp_path
    log = run / "09_formal_test" / "test_access.log"
    log.parent.mkdir(parents=True)
    labels = run / "locked_labels.parquet"
    bridge = run / "locked_bridge_gt.parquet"
    labels.write_bytes(b"labels")
    bridge.write_bytes(b"bridge")
    label_manifest = run / "candidate_label_manifest.json"
    _write_json(
        label_manifest,
        {
            "candidate_labels_path": str(labels.resolve()),
            "candidate_labels_sha256": sha256_file(labels),
        },
    )
    plan = run / "formal_plan.json"
    _write_json(
        plan,
        {
            "candidate_label_manifest": str(label_manifest.resolve()),
            "test_bridge_contract": {
                "historical_ground_truth": _artifact_record(bridge)
            },
        },
    )
    formal_manifest = log.parent / "formal_test_manifest.json"
    _write_json(formal_manifest, {"content_sha256": "f" * 64})
    lock = {
        "locked_files": {
            "formal_evaluation_plan": _artifact_record(plan),
        }
    }
    execution = {
        "artifacts": {
            "formal_test_manifest": _artifact_record(formal_manifest),
        }
    }
    monkeypatch.setattr(reporting, "verify_formal_test_lock", lambda _run: lock)
    monkeypatch.setattr(
        reporting,
        "_load_execution_bound_formal_manifest",
        lambda _run, _lock: (execution, {"content_sha256": "f" * 64}, {}),
    )
    stage_manifest = run / "label_free_stage.json"
    _write_json(stage_manifest, {"status": "COMPLETE"})
    stage_spec = {
        "event": "prelock_label_free_test_stage",
        "stage": "synthetic_label_free_stage",
        "route": None,
        "application_id": None,
        "path": str(stage_manifest.resolve()),
        "sha256": sha256_file(stage_manifest),
    }
    monkeypatch.setattr(
        reporting, "_access_manifest_specifications", lambda _run: [stage_spec]
    )
    records = [
        {
            "event": "prelock_candidate_test_label_hash_only",
            "candidate_labels_opened_as_table": False,
            "allowed_access": "opaque_byte_hash_only",
        },
        {"event": "candidate_test_label_access_denied"},
        {
            "event": "prelock_label_free_test_stage",
            "stage": "synthetic_label_free_stage",
            "output_manifest": str(stage_manifest.resolve()),
            "output_manifest_sha256": sha256_file(stage_manifest),
            "candidate_labels_opened_as_table": False,
        },
        {"event": "formal_test_exclusive_claim_created"},
        {"event": "candidate_test_label_access_authorized"},
        {
            "event": "candidate_test_labels_read_once",
            "path": str(labels.resolve()),
            "sha256": sha256_file(labels),
            "row_count": 1,
        },
        {
            "event": "formal_bridge_test_ground_truth_read_once",
            "path": str(bridge.resolve()),
            "sha256": sha256_file(bridge),
            "row_count": 1,
        },
        {
            "event": "formal_test_execution_finalized",
            "execution_count": 1,
            "manifest_sha256": sha256_file(formal_manifest),
        },
        {
            "event": "postclaim_independent_candidate_labels_read",
            "source_path": str(labels.resolve()),
            "source_sha256": sha256_file(labels),
            "rows": 1,
        },
        {
            "event": "postclaim_independent_bridge_ground_truth_read",
            "source_path": str(bridge.resolve()),
            "source_sha256": sha256_file(bridge),
            "rows": 1,
        },
        {
            "event": "postclaim_postformal_candidate_labels_read",
            "source_path": str(labels.resolve()),
            "source_sha256": sha256_file(labels),
            "rows": 1,
        },
    ]
    log.write_text("".join(json.dumps(row) + "\n" for row in records), encoding="utf-8")
    assert _access_log_check(run)[0] is True
    without_stage = [
        row for row in records if row.get("stage") != "synthetic_label_free_stage"
    ]
    log.write_text(
        "".join(json.dumps(row) + "\n" for row in without_stage), encoding="utf-8"
    )
    ok, details = _access_log_check(run)
    assert ok is False
    assert any("missing-bound-preclaim-stage" in value for value in details)
    records.insert(
        1,
        {
            "event": "candidate_test_label_rows_opened",
            "candidate_labels_opened_as_table": True,
        },
    )
    log.write_text("".join(json.dumps(row) + "\n" for row in records), encoding="utf-8")
    assert _access_log_check(run)[0] is False


def test_shared_hifi_mask_check_rejects_divergent_route_context(tmp_path: Path) -> None:
    run = tmp_path
    masks = run / "masks"
    masks.mkdir()
    first = masks / "first.png"
    second = masks / "second.png"
    first.write_bytes(b"same-mask-bytes")
    second.write_bytes(b"different-mask-bytes")
    paired = pd.DataFrame(
        {
            "sample_id": ["s1"],
            "predicted_mask_path": [str(first)],
            "predicted_mask_sha256": [sha256_file(first)],
        }
    )
    paired_path = run / "01_manifests" / "paired_test.parquet"
    paired_path.parent.mkdir(parents=True)
    paired.to_parquet(paired_path, index=False)
    for route in ("g1", "c1"):
        common_dir = run / "03_features" / "common" / f"{route}_test"
        track_dir = (
            run / "03_features" / "tracks" / "T2_matched_common" / f"{route}_test"
        )
        common_dir.mkdir(parents=True)
        track_dir.mkdir(parents=True)
        feature = common_dir / "candidate_features.parquet"
        pd.DataFrame(
            {"sample_id": ["s1"], "candidate_id": ["a"], "x": [1.0]}
        ).to_parquet(feature, index=False)
        context = paired.copy()
        if route == "c1":
            context["predicted_mask_path"] = str(second)
            context["predicted_mask_sha256"] = sha256_file(second)
        context_path = common_dir / "sample_context.parquet"
        context.to_parquet(context_path, index=False)
        common_manifest = {
            "status": "COMPLETE",
            "route": route,
            "split": "test",
            "paired_manifest_sha256": sha256_file(paired_path),
            "artifacts": {
                "candidate_features": {
                    "path": str(feature),
                    "sha256": sha256_file(feature),
                },
                "sample_context": {
                    "path": str(context_path),
                    "sha256": sha256_file(context_path),
                },
            },
        }
        (common_dir / "feature_manifest.json").write_text(
            json.dumps(common_manifest), encoding="utf-8"
        )
        track_manifest = {
            "status": "COMPLETE",
            "sources": [{"path": str(feature), "sha256": sha256_file(feature)}],
        }
        (track_dir / "feature_manifest.json").write_text(
            json.dumps(track_manifest), encoding="utf-8"
        )
    ok, errors = shared_hifi_mask_check(run)
    assert ok is False
    assert any(
        "c1" in error.lower() or "divergent" in error.lower() for error in errors
    )

    # Binding C1 back to the same context and bytes must pass.
    c1_context = run / "03_features/common/c1_test/sample_context.parquet"
    paired.to_parquet(c1_context, index=False)
    c1_manifest_path = run / "03_features/common/c1_test/feature_manifest.json"
    c1_manifest = json.loads(c1_manifest_path.read_text(encoding="utf-8"))
    c1_manifest["artifacts"]["sample_context"]["sha256"] = sha256_file(c1_context)
    c1_manifest_path.write_text(json.dumps(c1_manifest), encoding="utf-8")
    assert shared_hifi_mask_check(run) == (True, [])


def test_source_hash_inventory_requires_all_fourteen_exact_names(
    tmp_path: Path,
) -> None:
    from unified_reranking.postformal_reporting import EXPECTED_SOURCE_RUN_HASH_NAMES

    records = {}
    for name in EXPECTED_SOURCE_RUN_HASH_NAMES:
        source = tmp_path / "sources" / f"{name}.bin"
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_bytes(name.encode("utf-8"))
        records[name] = {
            **_artifact_record(source),
            "bytes": source.stat().st_size,
        }
    inventory = tmp_path / "00_audit" / "source_run_hashes.json"
    _write_json(inventory, records)
    assert _current_source_hash_checks(tmp_path)[0] is True
    _write_json(inventory, {"paired_manifest": records["paired_manifest"]})
    ok, rows = _current_source_hash_checks(tmp_path)
    assert ok is False
    assert sum(row["status"] == "MISSING" for row in rows) == 13


def test_feature_leakage_uses_canonical_forbidden_contract(tmp_path: Path) -> None:
    source = tmp_path / "03_features" / "safe_source.bin"
    artifact = tmp_path / "03_features" / "safe_output.bin"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_bytes(b"source")
    artifact.write_bytes(b"artifact")
    records = {}
    for route in ROUTES:
        manifest = (
            tmp_path
            / "03_features"
            / "tracks"
            / "T2_matched_common"
            / f"{route}_test"
            / "feature_manifest.json"
        )
        columns = ["safe_feature"]
        _write_json(
            manifest,
            {
                "status": "COMPLETE",
                "model_feature_columns": columns,
                "model_feature_schema_sha256": canonical_sha256(columns),
                "labels_physically_separate": True,
                "sources": {"input": _artifact_record(source)},
                "artifacts": {"output": _artifact_record(artifact)},
            },
        )
        records[f"{route}_feature_manifest_record"] = _artifact_record(manifest)
    prelock = tmp_path / "08_lock" / "prelock_assembly_manifest.json"
    _write_json(
        prelock,
        {"status": "COMPLETE", "stage": "P11_PRELOCK", "sources": records},
    )
    assert _feature_leakage_check(tmp_path) == (True, [])

    crog = Path(records["crog_feature_manifest_record"]["path"])
    payload = json.loads(crog.read_text(encoding="utf-8"))
    payload["model_feature_columns"] = ["scene_graph_embedding"]
    payload["model_feature_schema_sha256"] = canonical_sha256(
        payload["model_feature_columns"]
    )
    _write_json(crog, payload)
    records["crog_feature_manifest_record"] = _artifact_record(crog)
    _write_json(
        prelock,
        {"status": "COMPLETE", "stage": "P11_PRELOCK", "sources": records},
    )
    ok, violations = _feature_leakage_check(tmp_path)
    assert ok is False
    assert any("scene_graph_embedding" in value for value in violations)


def _synthetic_bundle(tmp_path: Path) -> FormalBundle:
    samples = pd.DataFrame(
        {
            "sample_id": ["s0", "s1", "s2", "s3"],
            "scene_id": ["z0", "z1", "z2", "z3"],
            "frame_id": ["f0", "f1", "f2", "f3"],
        }
    )
    systems = {}
    per_sample = []
    rankings = []
    per_candidate_scores = []
    labels = []
    pools = {}
    metrics = {}
    for route in ROUTES:
        names = {kind: f"{route}_{kind}" for kind in ("native", "ungated", "gated")}
        for kind, name in names.items():
            systems[name] = {"name": name, "route": route, "kind": kind}
            metrics[name] = {"sample_count": 4, "j_at_1": 0.25, "j_at_5": 0.5}
        pool_rows = []
        # s0: no output; s1: no positive; s2: positive challenger; s3: native positive.
        for sample_id in ("s1", "s2", "s3"):
            for rank, candidate in enumerate(("a", "b"), start=1):
                success = (sample_id == "s2" and candidate == "b") or (
                    sample_id == "s3" and candidate == "a"
                )
                pool_rows.append(
                    {
                        "sample_id": sample_id,
                        "candidate_id": candidate,
                        "native_rank": rank,
                        "native_score": 1.0 / rank,
                        "candidate_geometry_sha256": f"{route}-{sample_id}-{candidate}",
                    }
                )
                labels.append(
                    {
                        "route": route,
                        "sample_id": sample_id,
                        "candidate_id": candidate,
                        "candidate_success": success,
                    }
                )
                rankings.append(
                    {
                        "system_name": names["native"],
                        "sample_id": sample_id,
                        "candidate_id": candidate,
                        "rank": rank,
                    }
                )
        pools[route] = pd.DataFrame(pool_rows)
        selections = {
            "native": {"s0": "", "s1": "a", "s2": "a", "s3": "a"},
            "ungated": {"s0": "", "s1": "a", "s2": "b", "s3": "b"},
            "gated": {"s0": "", "s1": "a", "s2": "a", "s3": "a"},
        }
        success_lookup = {
            (row["sample_id"], row["candidate_id"]): bool(row["candidate_success"])
            for row in labels
            if row["route"] == route
        }
        for kind, by_sample in selections.items():
            for sample_id, candidate in by_sample.items():
                per_sample.append(
                    {
                        "system_name": names[kind],
                        "system_kind": kind,
                        "sample_id": sample_id,
                        "selected_route": route,
                        "selected_candidate_id": candidate,
                        "selected_correct": success_lookup.get(
                            (sample_id, candidate), False
                        ),
                    }
                )
                local = [row for row in pool_rows if row["sample_id"] == sample_id]
                ordered = sorted(
                    local,
                    key=lambda row: (
                        row["candidate_id"] != candidate,
                        row["native_rank"],
                        row["candidate_id"],
                    ),
                )
                for formal_rank, row in enumerate(ordered, start=1):
                    per_candidate_scores.append(
                        {
                            **row,
                            "system_name": names[kind],
                            "system_kind": kind,
                            "route": route,
                            "rank": formal_rank,
                            "formal_score": -float(formal_rank),
                            "candidate_success": success_lookup[
                                (sample_id, row["candidate_id"])
                            ],
                        }
                    )
    score_frame = pd.DataFrame(per_candidate_scores)
    score_path = tmp_path / "09_formal_test" / "per_candidate_scores.parquet"
    score_path.parent.mkdir(parents=True, exist_ok=True)
    score_frame.to_parquet(score_path, index=False)
    bridge_frame = score_frame.loc[
        score_frame["system_kind"].eq("native")
        & score_frame["route"].isin(["g1", "c1"])
    ].copy()
    bridge_frame["candidate_pool_contract"] = "fair_gaussian"
    bridge_frame["source_candidate_id"] = bridge_frame["candidate_id"].astype(str)
    bridge_frame["raw_candidate_id"] = bridge_frame["candidate_id"].astype(str)
    bridge_frame["candidate_id"] = (
        bridge_frame["route"].str.upper()
        + "::fair_gaussian::"
        + bridge_frame["candidate_id"].astype(str)
    )
    bridge_frame["fair_native_selector_score"] = pd.to_numeric(
        bridge_frame["native_score"], errors="raise"
    )
    bridge_frame["historical_selector_score"] = (
        bridge_frame["fair_native_selector_score"] * 0.8
    )
    bridge_frame["evaluator_sha256"] = "a" * 64
    bridge_frame = bridge_frame[
        [
            "route",
            "candidate_pool_contract",
            "sample_id",
            "candidate_id",
            "source_candidate_id",
            "raw_candidate_id",
            "native_rank",
            "candidate_geometry_sha256",
            "fair_native_selector_score",
            "historical_selector_score",
            "candidate_success",
            "evaluator_sha256",
        ]
    ]
    bridge_path = tmp_path / "09_formal_test" / "bridge_per_candidate_scores.parquet"
    bridge_frame.to_parquet(bridge_path, index=False)
    return FormalBundle(
        run_dir=tmp_path,
        manifest={
            "test_bridge": {"evaluator": {"sha256": "a" * 64}},
            "artifacts": {
                "per_candidate_scores": {
                    "path": str(score_path.resolve()),
                    "sha256": sha256_file(score_path),
                },
                "bridge_per_candidate_scores": {
                    "path": str(bridge_path.resolve()),
                    "sha256": sha256_file(bridge_path),
                },
            },
        },
        plan={},
        systems=systems,
        metrics=metrics,
        statistics={},
        sample_manifest=samples,
        per_sample=pd.DataFrame(per_sample),
        rankings=pd.DataFrame(rankings),
        per_candidate_scores=score_frame,
        bridge_per_candidate_scores=bridge_frame,
        labels=pd.DataFrame(labels),
        candidate_pools=pools,
    )


def test_native_score_check_rejects_semantic_drift_with_rebased_hash(
    tmp_path: Path,
) -> None:
    from unified_reranking.postformal_reporting import EXPECTED_SOURCE_RUN_HASH_NAMES

    bundle = _synthetic_bundle(tmp_path)
    records = {}
    for name in EXPECTED_SOURCE_RUN_HASH_NAMES:
        path = tmp_path / "audited" / f"{name}.bin"
        path.parent.mkdir(parents=True, exist_ok=True)
        if name in {"crog_candidates", "g1_candidates", "c1_candidates"}:
            route = name.split("_", maxsplit=1)[0]
            path = path.with_suffix(".parquet")
            bundle.candidate_pools[route].to_parquet(path, index=False)
        else:
            path.write_bytes(name.encode("utf-8"))
        records[name] = _artifact_record(path)
    inventory = tmp_path / "00_audit" / "source_run_hashes.json"
    _write_json(inventory, records)
    assert _native_score_source_check(bundle) == (True, [])

    g1_path = Path(records["g1_candidates"]["path"])
    drifted = pd.read_parquet(g1_path)
    drifted.loc[0, "native_score"] += 0.125
    drifted.to_parquet(g1_path, index=False)
    records["g1_candidates"] = _artifact_record(g1_path)
    _write_json(inventory, records)
    ok, errors = _native_score_source_check(bundle)
    assert ok is False
    assert "g1: audited native scores differ" in errors


def test_failure_taxonomy_covers_e0_e1_and_gate_counterfactuals(tmp_path: Path) -> None:
    taxonomy = classify_failures(_synthetic_bundle(tmp_path))
    for route in ROUTES:
        frame = taxonomy.loc[taxonomy["route"].eq(route)].set_index("sample_id")
        assert bool(frame.at["s0", "E0"])
        assert bool(frame.at["s1", "E1"])
        assert bool(frame.at["s2", "E9"])
        assert bool(frame.at["s3", "E10"])
        assert frame["primary_category"].notna().all()


def test_case_selection_and_deployment_rule_are_deterministic() -> None:
    first = deterministic_case_selection(["c", "a", "b", "a"], "recovered", 2)
    second = deterministic_case_selection(["b", "c", "a"], "recovered", 2)
    assert first == second and len(first) == 2
    row = {
        "delta_j_at_1": 0.01,
        "scene_ci95_lower": 0.001,
        "holm_p": 0.04,
    }
    assert (
        deployment_decision(row, independent_pass=True, integrity_pass=True)[0] == "GO"
    )
    row["scene_ci95_lower"] = -0.001
    assert (
        deployment_decision(row, independent_pass=True, integrity_pass=True)[0]
        == "CAUTION"
    )
    row["delta_j_at_1"] = 0.0
    assert (
        deployment_decision(row, independent_pass=True, integrity_pass=True)[0]
        == "NO-GO"
    )


@pytest.mark.parametrize("decision", ["NO_GO_NATIVE", "NO_GO_CROG"])
def test_deployment_decision_normalizes_locked_no_go_enums(decision: str) -> None:
    row = {"delta_j_at_1": 0.1, "scene_ci95_lower": 0.01, "holm_p": 0.001}
    status, reason = deployment_decision(
        row,
        independent_pass=True,
        integrity_pass=True,
        validation_gate_decision=decision,
    )
    assert status == "NO-GO"
    assert "Validation gate" in reason


def test_union_metrics_preserve_full_top15_reporting_schema(tmp_path: Path) -> None:
    bundle = _synthetic_bundle(tmp_path)
    systems = dict(bundle.systems)
    metrics = dict(bundle.metrics)
    systems["union_concat"] = {
        "name": "union_concat",
        "kind": "union",
        "route": "cross_route",
    }
    union_metrics = {
        "sample_count": 4,
        **{f"j_at_{k}": k / 20 for k in range(1, 16)},
        "oracle_at_15": 0.75,
        "oracle_all": 0.8,
        "mrr_at_15": 0.42,
        "ndcg_at_15": 0.57,
        "headroom_recovery_at_15": 0.25,
    }
    metrics["union_concat"] = union_metrics
    extended = FormalBundle(
        **{
            **bundle.__dict__,
            "systems": systems,
            "metrics": metrics,
        }
    )
    row = formal_results_table(extended).set_index("system_name").loc["union_concat"]
    for k in range(1, 16):
        assert row[f"j_at_{k}"] == union_metrics[f"j_at_{k}"]
    assert row["oracle_at_15"] == 0.75
    assert row["mrr_at_15"] == 0.42
    assert row["ndcg_at_15"] == 0.57
    assert row["headroom_recovery_at_15"] == 0.25


def test_postlock_bridge_uses_only_consumed_formal_score_label_bundle(
    tmp_path: Path,
    monkeypatch,
) -> None:
    bundle = _synthetic_bundle(tmp_path)
    original_read_parquet = pd.read_parquet

    def guarded_read(path, *args, **kwargs):
        if "candidate_test" in str(path) or "labels" in str(path):
            raise AssertionError("post-lock bridge reopened a source label table")
        return original_read_parquet(path, *args, **kwargs)

    monkeypatch.setattr(pd, "read_parquet", guarded_read)
    manifest = build_postlock_bridge(bundle)
    assert manifest["status"] == "FAILED_INCOMPLETE_FORMAL_BUNDLE"
    assert manifest["source_candidate_test_label_rows_reopened"] is False
    table = pd.read_csv(tmp_path / "11_attribution_bridge/bridge_postlock_test.csv")
    required = table.loc[table["bridge_cell_required"]]
    assert len(required) == 8
    assert set(required["route"]) == {"g1", "c1"}
    available = required.loc[required["candidate_pool_contract"].eq("fair_gaussian")]
    assert len(available) == 4
    assert available["status"].eq("AVAILABLE_FROM_FORMAL_BUNDLE").all()
    assert available["j_at_1"].eq(0.25).all()
    missing = required.drop(available.index)
    assert missing["status"].eq("MISSING_FORMAL_BRIDGE_INPUT").all()
    assert not table["feeds_selection"].any()
    assert not table["source_label_rows_reopened"].any()


def test_postlock_bridge_requires_all_eight_finite_cells(tmp_path: Path) -> None:
    bundle = _synthetic_bundle(tmp_path)
    scores = bundle.bridge_per_candidate_scores.copy()
    historical_parts = []
    for route in ("g1", "c1"):
        part = scores.loc[scores["route"].eq(route)].copy()
        part["candidate_pool_contract"] = "historical_nms"
        part["candidate_id"] = (
            part["route"].str.upper()
            + "::historical_nms::"
            + part["raw_candidate_id"].astype(str)
        )
        part["fair_native_selector_score"] = 1.0 / pd.to_numeric(
            part["native_rank"], errors="raise"
        )
        part["historical_selector_score"] = part["fair_native_selector_score"] * 0.6
        historical_parts.append(part)
    scores = pd.concat([scores, *historical_parts], ignore_index=True, sort=False)
    score_path = Path(
        bundle.manifest["artifacts"]["bridge_per_candidate_scores"]["path"]
    )
    scores.to_parquet(score_path, index=False)
    manifest_payload = {
        **bundle.manifest,
        "artifacts": {
            **bundle.manifest["artifacts"],
            "bridge_per_candidate_scores": {
                "path": str(score_path.resolve()),
                "sha256": sha256_file(score_path),
            },
        },
    }
    enriched = FormalBundle(
        **{
            **bundle.__dict__,
            "manifest": manifest_payload,
            "bridge_per_candidate_scores": scores,
        }
    )
    manifest = build_postlock_bridge(enriched)
    assert manifest["status"] == "COMPLETE"
    table = pd.read_csv(tmp_path / "11_attribution_bridge/bridge_postlock_test.csv")
    assert len(table) == 8
    assert table["status"].eq("AVAILABLE_FROM_FORMAL_BUNDLE").all()
    for column in (
        "j_at_1",
        "j_at_5",
        "oracle_at_5",
        "mrr_at_5",
        "ndcg_at_1",
        "ndcg_at_5",
        "oracle_all",
    ):
        assert pd.to_numeric(table[column], errors="coerce").notna().all()


def test_all_named_tables_and_vector_figures_are_generated_from_frames(
    tmp_path: Path,
) -> None:
    bundle = _synthetic_bundle(tmp_path)
    independent = tmp_path / "15_independent_recompute" / "INDEPENDENT_RECOMPUTE.json"
    independent.parent.mkdir(parents=True)
    independent.write_text(
        json.dumps({"status": "PASS", "formal_test_execution_count": 1}),
        encoding="utf-8",
    )
    taxonomy = classify_failures(bundle)
    taxonomy_path = (
        tmp_path / "13_failure_galleries" / "failure_taxonomy_per_sample.parquet"
    )
    taxonomy_path.parent.mkdir(parents=True)
    taxonomy.to_parquet(taxonomy_path, index=False)
    tables = build_tables(bundle, taxonomy, integrity_pass=True)
    assert set(TABLE_NAMES).issubset(tables)
    manifest = write_tables(tmp_path, tables)
    assert manifest["manual_numeric_entries"] == 0
    figures = build_figures(tmp_path, tables)
    assert set(figures["figures"]) == set(FIGURE_NAMES)
    for record in figures["figures"].values():
        assert Path(record["pdf"]).is_file()
        assert Path(record["svg"]).is_file()
        assert Path(record["caption_path"]).is_file()


def test_required_analysis_registry_fails_closed_on_empty_ablations(
    tmp_path: Path,
) -> None:
    bundle = _synthetic_bundle(tmp_path)
    independent = tmp_path / "15_independent_recompute" / "INDEPENDENT_RECOMPUTE.json"
    independent.parent.mkdir(parents=True)
    independent.write_text(
        json.dumps({"status": "PASS", "formal_test_execution_count": 1}),
        encoding="utf-8",
    )
    tables = build_tables(bundle, classify_failures(bundle), integrity_pass=True)
    assert tables["feature_ablation.csv"].empty
    registry = required_analysis_registry(tmp_path, tables)
    assert registry["status"] == "FAIL"
    assert registry["checks"]["cumulative_feature_ablation"]["status"] == "FAIL"
    assert registry["checks"]["leave_one_family_out_ablation"]["status"] == "FAIL"


def test_validation_cells_use_only_latest_phase_output_inventory(
    tmp_path: Path,
) -> None:
    current = _write_encoder_phase(tmp_path)
    stale_configuration = {
        "route": "crog",
        "track": "T2_matched_common",
        "encoder": "mlp",
        "loss": "ranknet",
        "seed": 42,
        "mode": "validation",
        "num_attention_blocks": 2,
        "source_identity": {
            "selected_feature_schema_sha256": canonical_sha256(["safe_feature"])
        },
    }
    stale_key = canonical_sha256({**stale_configuration, "stale": True})[:16]
    stale = tmp_path / "07_validation" / "matrix_cells" / stale_key / "manifest.json"
    stale.parent.mkdir(parents=True, exist_ok=True)
    stale.write_text(
        '{"status":"COMPLETE","metrics":{"j_at_1":0.99}}\n', encoding="utf-8"
    )

    cells = _validation_cells(tmp_path)
    assert len(cells) == 1
    assert cells.iloc[0]["manifest_path"] == str(current.resolve())
    assert cells.iloc[0]["j_at_1"] == pytest.approx(0.5)
    assert cells.iloc[0]["feature_extraction_latency_ms"] == pytest.approx(1.5)
    assert cells.iloc[0]["cell_load_preprocess_latency_ms"] == pytest.approx(0.4)


def test_validation_cells_reject_encoder_execution_wrong_selection(
    tmp_path: Path,
) -> None:
    _write_encoder_phase(tmp_path, wrong_selection=True)
    with pytest.raises(RuntimeError, match="wrong selection artifact"):
        _validation_cells(tmp_path)


def test_final_readiness_recomputes_inventory_and_marker_binding(
    tmp_path: Path,
) -> None:
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "status": "COMPLETE",
                "test_label_state": "FORMAL_TEST_COMPLETE",
                "formal_test_execution_count": 1,
            }
        ),
        encoding="utf-8",
    )
    _write_valid_ledger(tmp_path)
    artifact = tmp_path / "tables" / "result.csv"
    artifact.parent.mkdir(parents=True)
    artifact.write_text("value\n1\n", encoding="utf-8")
    inventory = hash_inventory(tmp_path)
    lock = {
        "schema_version": 1,
        "status": "COMPLETE",
        "inventory": inventory,
    }
    lock["self_sha256"] = canonical_sha256(lock)
    lock_path = tmp_path / "FINAL_RUN_LOCK.json"
    lock_path.write_text(
        json.dumps(lock, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    lock_sha = sha256_file(lock_path)
    (tmp_path / "FINAL_RUN_SHA256.txt").write_text(
        f"{lock_sha}  FINAL_RUN_LOCK.json\n", encoding="utf-8"
    )
    (tmp_path / "COMPLETE").write_text(
        f"COMPLETE\nFINAL_RUN_LOCK.json sha256={lock_sha}\n", encoding="utf-8"
    )
    assert verify_final_readiness(tmp_path)["ready"] is True
    (tmp_path / "COMPLETE").write_text(
        "COMPLETE\nFINAL_RUN_LOCK.json sha256=wrong\n", encoding="utf-8"
    )
    import tools.unified_reranking.build_postformal_artifacts as module

    with pytest.raises(PermissionError, match="invalid SHA file or terminal marker"):
        module.run(tmp_path)
    (tmp_path / "COMPLETE").write_text(
        f"COMPLETE\nFINAL_RUN_LOCK.json sha256={lock_sha}\n", encoding="utf-8"
    )
    artifact.write_text("value\n2\n", encoding="utf-8")
    readiness = verify_final_readiness(tmp_path)
    assert readiness["ready"] is False
    assert readiness["inventory_ok"] is False
    (tmp_path / "FAILED").write_text("FAILED\n", encoding="utf-8")
    assert verify_final_readiness(tmp_path)["ready"] is False


@pytest.mark.parametrize(
    "tampered",
    ["README_REPRODUCE.md", "manifest.json", "run_ledger.sqlite", "commands.log"],
)
def test_root_control_artifact_tampering_fails_readiness(
    tmp_path: Path, tampered: str
) -> None:
    run = tmp_path / tampered.replace(".", "_")
    run.mkdir()
    root_files = {
        "manifest.json": (
            b'{"status":"COMPLETE","test_label_state":"FORMAL_TEST_COMPLETE",'
            b'"formal_test_execution_count":1}\n'
        ),
        "README_REPRODUCE.md": b"# reproduce\n",
        "environment.txt": b"python=synthetic\n",
        "git_state.txt": b"commit=synthetic\n",
    }
    for name, payload in root_files.items():
        (run / name).write_bytes(payload)
    _write_valid_ledger(run)
    root_files["run_ledger.sqlite"] = (run / "run_ledger.sqlite").read_bytes()
    root_files["commands.log"] = (run / "commands.log").read_bytes()
    inventory = hash_inventory(run)
    assert {row["relative_path"] for row in inventory} == set(root_files)
    lock = {"schema_version": 1, "status": "COMPLETE", "inventory": inventory}
    lock["self_sha256"] = canonical_sha256(lock)
    lock_path = run / "FINAL_RUN_LOCK.json"
    lock_path.write_text(json.dumps(lock, sort_keys=True) + "\n", encoding="utf-8")
    lock_sha = sha256_file(lock_path)
    (run / "FINAL_RUN_SHA256.txt").write_text(
        f"{lock_sha}  FINAL_RUN_LOCK.json\n", encoding="utf-8"
    )
    (run / "COMPLETE").write_text(
        f"COMPLETE\nFINAL_RUN_LOCK.json sha256={lock_sha}\n", encoding="utf-8"
    )
    assert verify_final_readiness(run)["ready"] is True
    with (run / tampered).open("ab") as stream:
        stream.write(b"tamper")
    readiness = verify_final_readiness(run)
    assert readiness["ready"] is False
    assert readiness["inventory_ok"] is False


def _write_synthetic_final_lock(run: Path) -> None:
    inventory = hash_inventory(run)
    lock = {"schema_version": 1, "status": "COMPLETE", "inventory": inventory}
    lock["self_sha256"] = canonical_sha256(lock)
    lock_path = run / "FINAL_RUN_LOCK.json"
    lock_path.write_text(json.dumps(lock, sort_keys=True) + "\n", encoding="utf-8")
    digest = sha256_file(lock_path)
    (run / "FINAL_RUN_SHA256.txt").write_text(
        f"{digest}  FINAL_RUN_LOCK.json\n", encoding="utf-8"
    )
    (run / "COMPLETE").write_text(
        f"COMPLETE\nFINAL_RUN_LOCK.json sha256={digest}\n", encoding="utf-8"
    )


def test_final_readiness_rejects_file_added_under_locked_prefix(tmp_path: Path) -> None:
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "status": "COMPLETE",
                "test_label_state": "FORMAL_TEST_COMPLETE",
                "formal_test_execution_count": 1,
            }
        ),
        encoding="utf-8",
    )
    _write_valid_ledger(tmp_path)
    table = tmp_path / "tables/result.csv"
    table.parent.mkdir(parents=True)
    table.write_text("value\n1\n", encoding="utf-8")
    _write_synthetic_final_lock(tmp_path)
    assert verify_final_readiness(tmp_path)["ready"] is True

    added = tmp_path / "tables/added_after_lock.txt"
    added.write_text("unlocked addition\n", encoding="utf-8")
    readiness = verify_final_readiness(tmp_path)
    assert readiness["ready"] is False
    assert readiness["fresh_inventory_exact_match"] is False
    assert any(
        "added-after-lock:tables/added_after_lock.txt" == item
        for item in readiness["inventory_errors"]
    )


def test_final_readiness_rejects_missing_commands_log(tmp_path: Path) -> None:
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "status": "COMPLETE",
                "test_label_state": "FORMAL_TEST_COMPLETE",
                "formal_test_execution_count": 1,
            }
        ),
        encoding="utf-8",
    )
    table = tmp_path / "tables/result.csv"
    table.parent.mkdir(parents=True)
    table.write_text("value\n1\n", encoding="utf-8")
    _write_synthetic_final_lock(tmp_path)
    readiness = verify_final_readiness(tmp_path)
    assert readiness["ready"] is False
    assert readiness["commands_log_ok"] is False


def test_finalize_updates_root_manifest_before_inventory_and_binds_alias(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import tools.unified_reranking.build_postformal_artifacts as module

    bundle = _synthetic_bundle(tmp_path)
    monkeypatch.setattr(module, "TABLE_NAMES", ())
    monkeypatch.setattr(module, "FIGURE_NAMES", ())
    monkeypatch.setattr(module, "REPORT_NAMES", ())
    monkeypatch.setattr(module, "ROUTES", ())
    monkeypatch.setattr(module, "ROUTE_GALLERY_QUOTAS", {})
    monkeypatch.setattr(module, "CROSS_ROUTE_GALLERY_QUOTAS", {})
    required = (
        "README_REPRODUCE.md",
        "run_ledger.sqlite",
        "commands.log",
        "environment.txt",
        "git_state.txt",
        "08_lock/PRIMARY_METHOD_DECLARATION.md",
        "08_lock/FORMAL_TEST_LOCK.json",
        "09_formal_test/per_candidate_scores.parquet",
        "09_formal_test/bridge_per_candidate_scores.parquet",
        "09_formal_test/per_sample_decisions.parquet",
        "07_validation/bridge_train_validation.csv",
        "11_attribution_bridge/BRIDGE_DESIGN.md",
        "11_attribution_bridge/bridge_train_validation.csv",
        "11_attribution_bridge/bridge_postlock_test.csv",
        "15_independent_recompute/INDEPENDENT_RECOMPUTE.json",
        "15_independent_recompute/INDEPENDENT_RECOMPUTE.md",
        "tables/TABLE_GENERATION_MANIFEST.json",
        "12_figures/FIGURE_MANIFEST.json",
        "13_failure_galleries/GALLERY_MANIFEST.json",
        "14_reports/REPORT_MANIFEST.json",
        "14_reports/REQUIRED_ANALYSIS_REGISTRY.json",
        "tables/failure_strata.csv",
        "tables/failure_bottleneck_summary.csv",
    )
    for relative in required:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        if relative not in {"run_ledger.sqlite", "commands.log"}:
            path.write_text("{}\n", encoding="utf-8")
    _write_valid_ledger(tmp_path)
    root_manifest = {
        "status": "FORMAL_TEST_COMPLETE",
        "test_label_state": "FORMAL_TEST_COMPLETE",
        "formal_test_execution_count": 1,
        "preserved_field": "kept",
    }
    (tmp_path / "manifest.json").write_text(json.dumps(root_manifest), encoding="utf-8")
    postlock_manifest = (
        tmp_path / "11_attribution_bridge/BRIDGE_POSTLOCK_TEST_MANIFEST.json"
    )
    postlock_manifest.write_text(
        json.dumps({"status": "COMPLETE", "content_sha256": "b" * 64}),
        encoding="utf-8",
    )
    baseline = tmp_path / "14_reports/BASELINE_GAP_ATTRIBUTION.md"
    alias = tmp_path / "11_attribution_bridge/BASELINE_GAP_ATTRIBUTION.md"
    baseline.write_text("# same bytes\n", encoding="utf-8")
    alias.write_bytes(baseline.read_bytes())
    report_manifest = {
        "status": "COMPLETE",
        "manual_numeric_entries": 0,
        "reports": {},
        "prompt_aliases": {
            "baseline_gap_attribution": {
                "path": str(alias.resolve()),
                "sha256": sha256_file(alias),
            }
        },
    }
    result = module.finalize_run(
        bundle,
        {"status": "PASS", "checks": {}},
        {"status": "COMPLETE", "manual_numeric_entries": 0, "tables": {}},
        {"status": "COMPLETE", "figures": {}},
        {"status": "COMPLETE", "manual_selection": False, "summaries": []},
        report_manifest,
        {"status": "PASS", "content_sha256": "a" * 64},
    )
    assert result["status"] == "COMPLETE"
    finalized_root = json.loads(
        (tmp_path / "manifest.json").read_text(encoding="utf-8")
    )
    assert finalized_root["status"] == "COMPLETE"
    assert finalized_root["test_label_state"] == "FORMAL_TEST_COMPLETE"
    assert finalized_root["formal_test_execution_count"] == 1
    assert finalized_root["preserved_field"] == "kept"
    assert verify_final_readiness(tmp_path)["ready"] is True
    command_rows = [
        json.loads(line)
        for line in (tmp_path / "commands.log").read_text(encoding="utf-8").splitlines()
    ]
    assert {row["stage"] for row in command_rows}.issuperset({"P15", "POSTFORMAL"})

    finalized_root["status"] = "FAILED"
    (tmp_path / "manifest.json").write_text(
        json.dumps(finalized_root), encoding="utf-8"
    )
    readiness = verify_final_readiness(tmp_path)
    assert readiness["ready"] is False
    assert readiness["root_manifest_agrees_with_lock"] is False


def test_top_level_run_wires_required_analysis_into_reports_and_finalizer(
    tmp_path: Path,
    monkeypatch,
) -> None:
    import tools.unified_reranking.build_postformal_artifacts as module

    bundle = _synthetic_bundle(tmp_path)
    taxonomy = classify_failures(bundle)
    tables = {name: pd.DataFrame() for name in TABLE_NAMES}
    registry = {"status": "FAIL", "content_sha256": "a" * 64, "checks": {}}
    monkeypatch.setattr(module, "load_formal_bundle", lambda _path: bundle)
    monkeypatch.setattr(module, "classify_failures", lambda _bundle: taxonomy)
    monkeypatch.setattr(
        module,
        "core_integrity_checks",
        lambda _bundle, _taxonomy: {"status": "PASS", "checks": {}},
    )
    monkeypatch.setattr(module, "build_tables", lambda *_args: tables)
    monkeypatch.setattr(module, "write_tables", lambda *_args: {"status": "COMPLETE"})
    monkeypatch.setattr(module, "required_analysis_registry", lambda *_args: registry)
    monkeypatch.setattr(
        module, "build_galleries", lambda *_args: {"status": "COMPLETE"}
    )
    monkeypatch.setattr(module, "build_figures", lambda *_args: {"status": "COMPLETE"})
    seen = {}

    def reports(*args):
        seen["reports_registry"] = args[-1]
        manifest = tmp_path / "14_reports" / "REPORT_MANIFEST.json"
        manifest.parent.mkdir(parents=True, exist_ok=True)
        manifest.write_text('{"status":"COMPLETE"}\n', encoding="utf-8")
        return {"status": "COMPLETE"}

    def finalize(*args):
        seen["final_registry"] = args[-1]
        return {"status": "FAILED"}

    monkeypatch.setattr(module, "build_reports", reports)
    monkeypatch.setattr(module, "finalize_run", finalize)
    result = module.run(tmp_path)
    assert result["status"] == "FAILED"
    assert seen == {"reports_registry": registry, "final_registry": registry}
