from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import pandas as pd
import pytest

from src.grasping.reranking_v1.evaluation import (
    EvaluationError,
    evaluate_predictions,
    exact_mcnemar,
    holm_adjust,
    scene_grouped_bootstrap,
    verify_recomputed_metrics,
    vlm_results_to_predictions,
    vlm_runtime_metrics,
    write_evaluation_bundle,
)
from tools.modular_reranking.select_primary_method import (
    main as select_primary_main,
)
from tools.modular_reranking.evaluate_rerankers import (
    _canonical_json_sha256,
    _validate_formal_vlm_execution_evidence,
    _validate_formal_vlm_summary_runtime,
    _required_formal_apply_paths,
    parse_args as parse_evaluator_args,
    run as run_evaluator,
)
from src.grasping.reranking_v1.local_vlm import (
    SESSION_AUDIT_POLICY_VERSION,
    stable_session_contract,
    stable_session_contract_sha256,
)
from src.grasping.reranking_v1.identity import sha256_file
from src.grasping.reranking_v1.artifact_contract import identity_payload
from src.grasping.reranking_v1.method_namespace import (
    FULL_NMS_BASELINE,
    GQCNN_TOP5_BASELINE,
)
from tools.modular_reranking import join_formal_evaluation_labels


def _candidates() -> pd.DataFrame:
    rows = []
    # Baseline J@1 outcomes: s1=True, s2=False, s3=True, s4=False.
    labels = {
        "s1": (True, False),
        "s2": (False, True),
        "s3": (True, False),
        "s4": (False, False),
    }
    scenes = {"s1": "a", "s2": "a", "s3": "b", "s4": "b"}
    for sample_id, positives in labels.items():
        for offset, positive in enumerate(positives, start=1):
            rows.append(
                {
                    "sample_id": sample_id,
                    "scene_id": scenes[sample_id],
                    "candidate_id": f"{sample_id}_g{offset}",
                    "candidate_identity_sha256": f"{sample_id}-{offset}",
                    "original_gqcnn_rank": offset,
                    "q_raw": 1.0 - 0.25 * offset,
                    "query_type": "name" if sample_id in {"s1", "s2"} else "relation",
                    "p_axis_mean": 0.2 * offset,
                    "width_m": 0.02 * offset,
                    "candidate_positive": positive,
                }
            )
    return pd.DataFrame(rows)


def _predictions(candidates: pd.DataFrame) -> pd.DataFrame:
    # s1 is harmed, s2 is recovered, s3 remains correct, s4 remains wrong.
    top_order = {
        "s1": ["s1_g2", "s1_g1"],
        "s2": ["s2_g2", "s2_g1"],
        "s3": ["s3_g1", "s3_g2"],
        "s4": ["s4_g2", "s4_g1"],
    }
    identity = candidates.set_index(
        ["sample_id", "candidate_id"]
    )["candidate_identity_sha256"]
    rows = []
    for sample_id, ordered in top_order.items():
        for rank, candidate_id in enumerate(ordered, start=1):
            rows.append(
                {
                    "sample_id": sample_id,
                    "candidate_id": candidate_id,
                    "candidate_identity_sha256": identity.loc[
                        sample_id, candidate_id
                    ],
                    "method": "repeatedfilm_residual_mlp",
                    "protocol": "full_nms",
                    "rank": rank,
                }
            )
    return pd.DataFrame(rows)


def test_formal_label_join_requires_completion_before_reading_labels(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    called = False

    def reject_incomplete(
        lock_path: Path, *, stage: str, manifest_path: Path
    ) -> dict:
        nonlocal called
        called = True
        assert lock_path == (tmp_path / "lock.json").resolve()
        assert stage == "TEST"
        assert manifest_path == (tmp_path / "formal.json").resolve()
        raise FileNotFoundError("formal TEST completion ledger is missing")

    monkeypatch.setattr(
        join_formal_evaluation_labels,
        "verify_completed_formal_stage",
        reject_incomplete,
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "join_formal_evaluation_labels.py",
            "--inference-per-candidate",
            str(tmp_path / "test_candidates.parquet"),
            "--labels-parquet",
            str(tmp_path / "strict_labels.parquet"),
            "--formal-inference-manifest",
            str(tmp_path / "formal.json"),
            "--experiment-lock",
            str(tmp_path / "lock.json"),
            "--output-root",
            str(tmp_path / "joined"),
        ],
    )
    with pytest.raises(
        FileNotFoundError, match="completion ledger is missing"
    ):
        join_formal_evaluation_labels.main()
    assert called


def test_metrics_outcomes_and_denominators_are_recomputed_from_labels() -> None:
    candidates = _candidates()
    samples = pd.DataFrame(
        [
            *candidates[["sample_id", "scene_id"]]
            .drop_duplicates()
            .to_dict("records"),
            {"sample_id": "empty", "scene_id": "c"},
        ]
    )
    result = evaluate_predictions(
        candidates,
        _predictions(candidates),
        sample_universe=samples,
        bootstrap_replicates=50,
        seed=9,
    )
    metrics = result.per_method_metrics.set_index("method")

    baseline = metrics.loc[FULL_NMS_BASELINE]
    assert baseline["sample_count_all"] == 5
    assert baseline["sample_count_nonempty"] == 4
    assert baseline["j_at_1_count"] == 2
    assert baseline["j_at_1_all"] == pytest.approx(2 / 5)
    assert baseline["j_at_1_nonempty"] == pytest.approx(1 / 2)
    assert baseline["recall_at_5_all"] == pytest.approx(3 / 5)
    assert baseline["j_at_any_all"] == pytest.approx(3 / 5)
    assert baseline["mrr_all"] == pytest.approx((1 + 0.5 + 1) / 5)
    assert baseline["mean_first_positive_rank"] == pytest.approx(4 / 3)
    assert baseline["median_first_positive_rank"] == 1

    reranker = metrics.loc["repeatedfilm_residual_mlp"]
    assert reranker["recovered_count"] == 1
    assert reranker["harmful_count"] == 1
    assert reranker["net_count"] == 0
    assert reranker["outcome_precision"] == pytest.approx(0.5)
    assert reranker["outcome_changing_precision"] == pytest.approx(2 / 3)
    assert reranker["beneficial_switch_precision"] == pytest.approx(1 / 3)
    assert reranker["switch_count"] == 3
    assert reranker["coverage_all"] == pytest.approx(3 / 5)
    assert reranker["coverage_nonempty"] == pytest.approx(3 / 4)

    outcomes = result.per_sample_outcomes
    s1 = outcomes.query(
        "method == 'repeatedfilm_residual_mlp' and sample_id == 's1'"
    ).iloc[0]
    assert s1["outcome"] == "harmful"
    assert s1["top1_original_rank_movement"] == 1
    s2 = outcomes.query(
        "method == 'repeatedfilm_residual_mlp' and sample_id == 's2'"
    ).iloc[0]
    assert s2["outcome"] == "recovered"


def test_candidate_identity_and_pool_are_invariant() -> None:
    candidates = _candidates()
    predictions = _predictions(candidates)
    predictions.loc[0, "candidate_identity_sha256"] = "changed"
    with pytest.raises(EvaluationError, match="identity"):
        evaluate_predictions(candidates, predictions, bootstrap_replicates=0)

    predictions = _predictions(candidates).iloc[:-1].copy()
    with pytest.raises(EvaluationError, match="candidate pool"):
        evaluate_predictions(candidates, predictions, bootstrap_replicates=0)


def test_q_only_and_scored_methods_are_independently_ranked() -> None:
    candidates = _candidates()
    bad_q_rank = candidates.copy()
    bad_q_rank.loc[bad_q_rank["sample_id"] == "s1", "q_raw"] = [0.1, 0.9]
    with pytest.raises(EvaluationError, match="q_raw/candidate_id"):
        evaluate_predictions(bad_q_rank, bootstrap_replicates=0)

    predictions = _predictions(candidates)
    predictions["score"] = 0.0
    predictions.loc[
        (predictions["sample_id"] == "s1")
        & (predictions["candidate_id"] == "s1_g1"),
        "score",
    ] = 1.0
    with pytest.raises(EvaluationError, match="independent score"):
        evaluate_predictions(candidates, predictions, bootstrap_replicates=0)


def test_vlm_tied_model_scores_preserve_explicit_ranking_order() -> None:
    predictions = vlm_results_to_predictions(
        [
            {
                "sample_id": "s1",
                "selected_candidate_id": "s1_g2",
                "ranking": [
                    {"candidate_id": "s1_g2", "score": 0.0},
                    {"candidate_id": "s1_g1", "score": 0.0},
                ],
            }
        ],
        method="repeatedfilm_local_vlm_visual",
        protocol="gqcnn_top5",
    )
    ordered = predictions.sort_values("rank")
    assert ordered["candidate_id"].tolist() == ["s1_g2", "s1_g1"]
    assert ordered["score"].tolist() == [2.0, 1.0]
    assert ordered["vlm_model_score"].tolist() == [0.0, 0.0]
    result = evaluate_predictions(
        _candidates().query("sample_id == 's1'"),
        predictions,
        bootstrap_replicates=0,
    )
    metric = result.per_method_metrics.query(
        "method == 'repeatedfilm_local_vlm_visual'"
    ).iloc[0]
    assert metric["harmful_count"] == 1


def test_unknown_protocol_is_rejected() -> None:
    predictions = _predictions(_candidates())
    predictions["protocol"] = "top_5_typo"
    with pytest.raises(EvaluationError, match="unknown protocol"):
        evaluate_predictions(_candidates(), predictions, bootstrap_replicates=0)


def test_protocol_b_has_an_explicit_q_top5_baseline() -> None:
    candidates = _candidates()
    predictions = _predictions(candidates)
    predictions["protocol"] = "gqcnn_top5"
    result = evaluate_predictions(candidates, predictions, bootstrap_replicates=0)
    methods = set(result.per_method_metrics["method"])
    assert methods == {
        GQCNN_TOP5_BASELINE,
        "repeatedfilm_residual_mlp",
    }
    test = result.stat_tests.iloc[0]
    assert test["baseline_method"] == GQCNN_TOP5_BASELINE


def test_prediction_labels_are_never_trusted() -> None:
    candidates = _candidates()
    predictions = _predictions(candidates)
    predictions["candidate_positive"] = False
    with pytest.raises(EvaluationError, match="candidate_positive"):
        evaluate_predictions(candidates, predictions, bootstrap_replicates=0)


def test_bare_internal_prediction_method_is_rejected() -> None:
    predictions = _predictions(_candidates())
    predictions["method"] = "q_softmask_rule"
    with pytest.raises(EvaluationError, match="bare/internal"):
        evaluate_predictions(
            _candidates(), predictions, bootstrap_replicates=0
        )


def test_supplied_baseline_must_equal_original_order() -> None:
    candidates = _candidates()
    supplied = candidates[
        [
            "sample_id",
            "candidate_id",
            "candidate_identity_sha256",
            "original_gqcnn_rank",
            "candidate_positive",
        ]
    ].rename(columns={"original_gqcnn_rank": "reranker_rank"})
    supplied["reranker_method"] = FULL_NMS_BASELINE
    supplied["protocol"] = "full_nms"
    result = evaluate_predictions(
        candidates, supplied, bootstrap_replicates=0
    )
    assert list(result.per_method_metrics["method"]) == [
        FULL_NMS_BASELINE
    ]

    supplied.loc[
        supplied["sample_id"] == "s1", "reranker_rank"
    ] = [2, 1]
    with pytest.raises(EvaluationError, match="original_gqcnn_rank"):
        evaluate_predictions(candidates, supplied, bootstrap_replicates=0)


def test_exact_mcnemar_and_holm_are_exact_and_monotone() -> None:
    test = exact_mcnemar(
        baseline_success=[True, False, False, True, False],
        method_success=[False, True, True, True, True],
    )
    assert test["recovered"] == 3
    assert test["harmful"] == 1
    assert test["discordant"] == 4
    assert test["p_value_exact_two_sided"] == pytest.approx(0.625)
    assert exact_mcnemar([True], [True])["p_value_exact_two_sided"] == 1.0

    adjusted = holm_adjust([0.01, 0.04, 0.03])
    assert adjusted == pytest.approx([0.03, 0.06, 0.06])


def test_scene_grouped_bootstrap_is_deterministic() -> None:
    candidates = _candidates()
    predictions = _predictions(candidates)
    result = evaluate_predictions(
        candidates, predictions, bootstrap_replicates=0
    )
    first = scene_grouped_bootstrap(
        result.per_sample_outcomes, replicates=200, seed=17
    )
    second = scene_grouped_bootstrap(
        result.per_sample_outcomes, replicates=200, seed=17
    )
    pd.testing.assert_frame_equal(first, second)
    assert set(first["cluster_key"]) == {"scene_sequence_id", "frame_id"}
    assert set(first["replicates"]) == {200}
    assert {
        "recall_at_5_nonempty",
        "recall_at_10_nonempty",
        "j_at_any_nonempty",
        "j_at_1_delta_nonempty",
        "net_rate_nonempty",
    } <= set(first["metric"])


def test_bundle_can_be_independently_recomputed(tmp_path: Path) -> None:
    candidates = _candidates()
    result = evaluate_predictions(
        candidates,
        _predictions(candidates),
        bootstrap_replicates=25,
        seed=4,
    )
    output = tmp_path / "results"
    manifest = write_evaluation_bundle(output, result)
    assert manifest["candidate_pool_modified"] is False
    assert {
        key: manifest[key] for key in identity_payload()
    } == identity_payload()
    assert (output / "results_bundle.json").is_file()
    assert (output / "per_method_metrics.parquet").is_file()
    assert (output / "per_sample_outcomes.parquet").is_file()
    assert (output / "per_candidate_predictions.parquet").is_file()
    assert (output / "stat_tests.parquet").is_file()
    assert (output / "bootstrap.parquet").is_file()
    assert (output / "per_method_metrics.csv").is_file()
    assert (output / "statistical_tests.json").is_file()
    assert (output / "bootstrap_intervals.json").is_file()
    assert (output / "vlm_runtime_metrics.json").is_file()
    assert (output / "commands.log").is_file()
    assert (output / "environment.json").is_file()
    assert (output / "grouped_analysis.parquet").is_file()
    assert (output / "grouped_analysis.json").is_file()
    verify_recomputed_metrics(
        output / "per_candidate_predictions.parquet",
        output / "per_sample_outcomes.parquet",
        output / "per_method_metrics.parquet",
        output / "stat_tests.parquet",
        output / "bootstrap.parquet",
    )
    grouped = pd.read_parquet(output / "grouped_analysis.parquet")
    assert {"query_type", "candidate_count_group", "q_margin_group"} <= set(
        grouped["group_dimension"]
    )


def test_vlm_runtime_metrics() -> None:
    rows = [
        {
            "sample_id": "s1",
            "selected_candidate_id": "s1_g1",
            "latency_seconds": 2.0,
            "fallback": False,
            "abstain": False,
            "cache_hit": False,
            "prompt_eval_count": 100,
            "eval_count": 20,
            "total_duration_ns": 2_500_000_000,
            "parsed_model_response": {"selected_candidate_id": "s1_g1"},
            "parser_error": None,
            "ranking": [{"candidate_id": "s1_g1"}],
        },
        {
            "sample_id": "s2",
            "selected_candidate_id": "s2_g1",
            "latency_seconds": 4.0,
            "fallback": True,
            "abstain": True,
            "cache_hit": True,
            "prompt_eval_count": 80,
            "eval_count": 10,
            "total_duration_ns": 4_500_000_000,
            "parsed_model_response": None,
            "parser_error": "invalid_candidate_id",
            "fallback_reason": "invalid_candidate_id",
            "ranking": [{"candidate_id": "s2_g1"}],
        },
        {
            "sample_id": "empty",
            "selected_candidate_id": None,
            "latency_seconds": 0.0,
            "fallback": False,
            "abstain": False,
            "cache_hit": False,
            "fallback_reason": None,
            "eligible_for_vlm": False,
            "http_call_performed": False,
            "skip_reason": "valid_empty_no_vlm_call",
            "prompt_eval_count": None,
            "eval_count": None,
            "total_duration_ns": None,
            "parsed_model_response": None,
            "parser_error": None,
            "ranking": [],
        },
    ]
    repeat = [dict(row) for row in rows]
    metrics = vlm_runtime_metrics(
        rows,
        repeat_rows=repeat,
        wall_time_seconds=10.0,
        memory_peak_mib=2048.0,
    )
    assert metrics["sample_count"] == 3
    assert metrics["called_sample_count"] == 2
    assert metrics["valid_empty_skipped_count"] == 1
    assert metrics["latency_mean_seconds"] == 3
    assert metrics["fallback_rate"] == 0.5
    assert metrics["cache_hit_rate"] == 0.5
    assert metrics["prompt_tokens_total"] == 180
    assert metrics["output_tokens_total"] == 30
    assert metrics["valid_json_rate"] == 0.5
    assert metrics["invalid_candidate_rate"] == 0.5
    assert metrics["deterministic_agreement_rate"] == 1.0
    assert metrics["memory_peak_mib"] == 2048
    assert metrics["samples_per_hour"] == 720


def test_cli_writes_a_recomputable_bundle(tmp_path: Path) -> None:
    candidates = _candidates()
    predictions = _predictions(candidates).rename(
        columns={
            "method": "reranker_method",
            "rank": "reranker_rank",
        }
    )
    candidate_path = tmp_path / "candidates.parquet"
    prediction_path = tmp_path / "predictions.parquet"
    universe_path = tmp_path / "samples.parquet"
    output = tmp_path / "evaluation"
    candidates.to_parquet(candidate_path, index=False)
    predictions.to_parquet(prediction_path, index=False)
    candidates[["sample_id", "scene_id"]].drop_duplicates().to_parquet(
        universe_path, index=False
    )
    script = (
        Path(__file__).resolve().parents[1]
        / "tools"
        / "modular_reranking"
        / "evaluate_rerankers.py"
    )
    completed = subprocess.run(
        [
            sys.executable,
            str(script),
            "--per-candidate",
            str(candidate_path),
            "--prediction",
            str(prediction_path),
            "--sample-universe",
            str(universe_path),
            "--bootstrap-replicates",
            "10000",
            "--seed",
            "3",
            "--expected-sample-count",
            "4",
            "--expected-candidate-count",
            "8",
            "--expected-nonempty-count",
            "4",
            "--expected-valid-empty-count",
            "0",
            "--expected-scene-count",
            "2",
            "--output-root",
            str(output),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    bundle = json.loads((output / "results_bundle.json").read_text())
    assert bundle["report_recomputation_verified"] is True
    assert bundle["bootstrap_replicates"] == 10000
    verify_recomputed_metrics(
        output / "per_candidate_predictions.parquet",
        output / "per_sample_outcomes.parquet",
        output / "per_method_metrics.parquet",
    )


def test_formal_evaluator_rejects_unselected_vlm_variant(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    candidate_path = tmp_path / "candidates.parquet"
    universe_path = tmp_path / "universe.parquet"
    _candidates().to_parquet(candidate_path, index=False)
    _candidates()[["sample_id", "scene_id"]].drop_duplicates().to_parquet(
        universe_path, index=False
    )
    visual_results = tmp_path / "visual.jsonl"
    metadata_results = tmp_path / "metadata.jsonl"
    visual_summary = tmp_path / "visual_summary.json"
    metadata_summary = tmp_path / "metadata_summary.json"
    for path in (
        visual_results,
        metadata_results,
        visual_summary,
        metadata_summary,
    ):
        path.write_text("{}\n")
    dummy = tmp_path / "dummy.json"
    dummy.write_text("{}\n")
    contracts = {
        "gqcnn_top5/local_vlm_visual": {
            "results": visual_results.resolve(),
            "summary": visual_summary.resolve(),
        }
    }
    monkeypatch.setattr(
        "tools.modular_reranking.evaluate_rerankers._formal_allowed_sources",
        lambda *_args, **_kwargs: (
            {"evaluation_definition": {}},
            set(),
            contracts,
        ),
    )
    args = parse_evaluator_args(
        [
            "--per-candidate",
            str(candidate_path),
            "--sample-universe",
            str(universe_path),
            "--vlm-results",
            f"local_vlm_visual_metadata={metadata_results}",
            "--vlm-summary",
            f"local_vlm_visual_metadata={metadata_summary}",
            "--expected-sample-count",
            "4",
            "--expected-candidate-count",
            "8",
            "--expected-nonempty-count",
            "4",
            "--expected-valid-empty-count",
            "0",
            "--expected-scene-count",
            "2",
            "--bootstrap-replicates",
            "10000",
            "--output-root",
            str(tmp_path / "output"),
            "--experiment-lock",
            str(dummy),
            "--formal-inference-manifest",
            str(dummy),
            "--formal-label-join-manifest",
            str(dummy),
            "--formal-vlm-manifest",
            str(dummy),
            "--formal-vlm-apply-manifest",
            str(dummy),
        ]
    )
    with pytest.raises(EvaluationError, match="method/result binding"):
        run_evaluator(args)


def test_formal_evaluator_cli_requires_exactly_one_vlm_manifest(
    tmp_path: Path,
) -> None:
    common = [
        "--per-candidate",
        str(tmp_path / "candidates.parquet"),
        "--sample-universe",
        str(tmp_path / "universe.parquet"),
        "--expected-sample-count",
        "1",
        "--expected-candidate-count",
        "1",
        "--expected-nonempty-count",
        "1",
        "--expected-valid-empty-count",
        "0",
        "--expected-scene-count",
        "1",
        "--output-root",
        str(tmp_path / "output"),
        "--experiment-lock",
        str(tmp_path / "lock.json"),
        "--formal-inference-manifest",
        str(tmp_path / "inference.json"),
        "--formal-label-join-manifest",
        str(tmp_path / "join.json"),
        "--formal-vlm-apply-manifest",
        str(tmp_path / "apply.json"),
    ]
    with pytest.raises(SystemExit):
        parse_evaluator_args(common)
    with pytest.raises(SystemExit):
        parse_evaluator_args(
            [
                *common,
                "--formal-vlm-manifest",
                str(tmp_path / "visual.json"),
                "--formal-vlm-manifest",
                str(tmp_path / "metadata.json"),
            ]
        )


def test_formal_apply_binding_uses_gt_free_inference_candidates(
    tmp_path: Path,
) -> None:
    inference_candidates = tmp_path / "test_inference_candidates.parquet"
    label_joined_candidates = tmp_path / "test_candidates_with_gt.parquet"
    paths = _required_formal_apply_paths(
        join={"inference_per_candidate": str(inference_candidates)},
        selected_contract={"manifest": tmp_path / "formal_vlm_manifest.json"},
        selection_path=tmp_path / "selection.json",
        inference_manifest_path=tmp_path / "inference_manifest.json",
        universe_path=tmp_path / "universe.parquet",
    )
    assert paths["per_candidate"] == inference_candidates.resolve()
    assert paths["per_candidate"] != label_joined_candidates.resolve()


def test_formal_vlm_attempt_chain_is_required_and_local_only(
    tmp_path: Path,
) -> None:
    lock_path = tmp_path / "lock.json"
    lock_path.write_text("{}\n")
    lock = {"manifest_content_sha256": "lock-content"}
    stage_root = tmp_path / "formal-vlm"
    attempts = stage_root / "attempts"
    attempts.mkdir(parents=True)
    locked_audit = tmp_path / "locked-audit.json"
    locked_audit.write_text("{}\n")
    session_audit = {
        "session_audit_policy_version": SESSION_AUDIT_POLICY_VERSION,
        "local_only": True,
        "remote_api_used": False,
        "machine": {"architecture": "arm64"},
        "ollama": {
            "cli_version": "0.13.5",
            "api_version": {"version": "0.13.5"},
            "executable_path": "/Applications/Ollama.app/ollama",
            "executable_sha256": "a" * 64,
            "endpoint": "http://127.0.0.1:11434",
            "server_config_path": "/tmp/server.json",
            "server_config_sha256": "b" * 64,
            "server_config": {"disable_ollama_cloud": True},
            "remote_established_connections": [],
        },
        "model": {
            "exact_name": "qwen3-vl:4b-instruct-q4_K_M",
            "manifest_sha256": "c" * 64,
            "model_layer_digest": "sha256:" + "d" * 64,
            "model_layer_bytes": 100,
            "quantization": "Q4_K_M",
            "layers": [
                {
                    "digest": "sha256:" + "d" * 64,
                    "mediaType": "application/vnd.ollama.image.model",
                    "size": 100,
                    "local_size": 100,
                    "local_sha256": "d" * 64,
                }
            ],
        },
    }
    session_audit["stable_session_contract"] = stable_session_contract(
        session_audit
    )
    stable_hash = stable_session_contract_sha256(session_audit)
    session_audit["stable_session_contract_sha256"] = stable_hash
    session_path = attempts / "session_audit_0001.json"
    session_path.write_text(json.dumps(session_audit) + "\n")
    ledger = {
        **identity_payload(),
        "schema_version": 1,
        "session_audit_policy_version": SESSION_AUDIT_POLICY_VERSION,
        "attempt_number": 1,
        "previous_attempt_sha256": None,
        "lock_path": str(lock_path),
        "lock_content_sha256": "lock-content",
        "stage": "VLM_VISUAL",
        "locked_local_audit": str(locked_audit),
        "locked_local_audit_sha256": sha256_file(locked_audit),
        "session_local_audit": str(session_path),
        "session_local_audit_sha256": sha256_file(session_path),
        "stable_session_contract_sha256": stable_hash,
        "listener_pid": 123,
        "listener_process_started": "Tue Jul 28 21:32:39 2026",
        "prefix_result_row_count": 0,
        "prefix_results_sha256": None,
    }
    ledger_path = attempts / "attempt_0001.json"
    ledger_path.write_text(json.dumps(ledger) + "\n")
    artifacts = [
        {"path": str(ledger_path), "sha256": sha256_file(ledger_path)}
    ]
    monitor_path = stage_root / "runtime_monitor.jsonl"
    monitor_path.write_text(
        json.dumps({"remote_established_connections": []}) + "\n"
    )
    results_path = stage_root / "vlm_ranking_results.jsonl"
    results_path.write_text("")
    vlm = {
        "locked_local_audit": str(locked_audit),
        "locked_local_audit_sha256": sha256_file(locked_audit),
        "stable_session_contract_sha256": stable_hash,
        "formal_attempt_ledgers": artifacts,
        "formal_attempt_chain_tip_sha256": sha256_file(ledger_path),
    }
    runtime = {
        "formal_attempt_ledgers": artifacts,
        "formal_attempt_chain_tip_sha256": sha256_file(ledger_path),
        "local_only_runtime_passed": True,
        "remote_established_connection_event_count": 0,
        "remote_established_connection_events": [],
        "monitor_attempt_count": 1,
        "monitor_sample_count": 1,
    }
    manifest_path = stage_root / "formal_vlm_manifest.json"
    _validate_formal_vlm_execution_evidence(
        vlm=vlm,
        runtime=runtime,
        results_path=results_path,
        monitor_path=monitor_path,
        manifest_path=manifest_path,
        lock_path=lock_path,
        lock=lock,
        stage="VLM_VISUAL",
    )
    runtime["remote_established_connection_event_count"] = 1
    with pytest.raises(EvaluationError, match="aggregate runtime"):
        _validate_formal_vlm_execution_evidence(
            vlm=vlm,
            runtime=runtime,
            results_path=results_path,
            monitor_path=monitor_path,
            manifest_path=manifest_path,
            lock_path=lock_path,
            lock=lock,
            stage="VLM_VISUAL",
        )


def test_formal_vlm_summary_runtime_bind_hashed_universe(
    tmp_path: Path,
) -> None:
    input_path = tmp_path / "inputs.jsonl"
    results_path = tmp_path / "results.jsonl"
    input_row = {
        "sample_id": "s1",
        "candidate_ids": [],
        "visualization_storage_mode": "on_demand_recipe",
    }
    result_row = {
        "sample_id": "s1",
        "eligible_for_vlm": False,
        "request_hash": None,
        "ranking": [],
        "input_record_sha256": _canonical_json_sha256(input_row),
    }
    input_path.write_text(json.dumps(input_row) + "\n")
    results_path.write_text(json.dumps(result_row) + "\n")
    stable_hash = "e" * 64
    vlm = {
        "input_jsonl": str(input_path),
        "input_jsonl_sha256": sha256_file(input_path),
        "results_jsonl": str(results_path),
        "results_jsonl_sha256": sha256_file(results_path),
        "sample_count": 1,
        "eligible_sample_count": 0,
        "empty_skipped_count": 1,
        "model_name": "model",
        "model_digest": "sha256:model",
        "stable_session_contract_sha256": stable_hash,
    }
    common = {
        **vlm,
        "input_split": "test",
        "formal_mode": True,
        "visualization_storage_mode": "on_demand_recipe",
        "on_demand_visual_sample_count": 0,
        "ordinary_visual_pngs_retained": 0,
    }
    summary = dict(common)
    runtime = {
        **common,
        "fresh_http_call_count": 0,
        "cache_hit_count": 0,
    }
    _validate_formal_vlm_summary_runtime(
        vlm=vlm,
        summary=summary,
        runtime=runtime,
        results_path=results_path,
    )
    runtime["input_split"] = "validation"
    with pytest.raises(EvaluationError, match="runtime binding"):
        _validate_formal_vlm_summary_runtime(
            vlm=vlm,
            summary=summary,
            runtime=runtime,
            results_path=results_path,
        )


def test_validation_primary_selector_enforces_all_gates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    method = "repeatedfilm_residual_mlp"
    outcomes = pd.DataFrame(
        [
            {
                "sample_id": f"s{index}",
                "method": method,
                "protocol": "full_nms",
                "recovered": index in {0, 1},
                "harmful": index == 2,
            }
            for index in range(100)
        ]
    )
    bootstrap = pd.DataFrame(
        [
            {
                "method": method,
                "protocol": "full_nms",
                "metric": "j_at_1_delta_all",
                "point_estimate": 0.01,
                "ci_95_lower": 0.005,
                "ci_95_upper": 0.03,
                "replicates": 10_000,
            }
        ]
    )
    outcomes_path = tmp_path / "outcomes.parquet"
    bootstrap_path = tmp_path / "bootstrap.parquet"
    outcomes.to_parquet(outcomes_path, index=False)
    bootstrap.to_parquet(bootstrap_path, index=False)
    universe_path = tmp_path / "universe.parquet"
    pd.DataFrame(
        {
            "sample_id": [f"s{index}" for index in range(100)],
            "scene_id": [f"scene-{index // 5}" for index in range(100)],
            "split": ["validation"] * 100,
        }
    ).to_parquet(universe_path, index=False)
    from src.grasping.reranking_v1.identity import sha256_file

    bundle_path = tmp_path / "results_bundle.json"
    bundle_path.write_text(
        json.dumps(
            {
                **identity_payload(),
                "report_recomputation_verified": True,
                "candidate_pool_modified": False,
                "sample_count_all": 100,
                "files": {
                    "per_sample_outcomes": {
                        "path": str(outcomes_path),
                        "sha256": sha256_file(outcomes_path),
                    },
                    "bootstrap": {
                        "path": str(bootstrap_path),
                        "sha256": sha256_file(bootstrap_path),
                    },
                },
                "provenance": {
                    "candidate_identity_invariant_enforced": True,
                    "sources": [
                        {
                            "role": "sample_universe",
                            "path": str(universe_path),
                            "sha256": sha256_file(universe_path),
                        }
                    ],
                },
            }
        )
    )
    candidate_path = tmp_path / "validation_candidates.parquet"
    candidate_path.write_bytes(b"validation-candidates")
    inference_bundle_path = tmp_path / "inference_bundle.json"
    inference_bundle_path.write_text(
        json.dumps(
            {
                **identity_payload(),
                "candidate_pool_modified": False,
                "primary_candidate_methods": [method],
            }
        )
    )
    training_path = tmp_path / "training_manifest.json"
    training_path.write_text(
        json.dumps(
            {
                **identity_payload(),
                "candidate_pool_modified": False,
                "primary_candidate_methods": [method],
                "validation_per_candidate": str(candidate_path),
                "validation_sha256": sha256_file(candidate_path),
                "validation_per_sample": str(universe_path),
                "validation_per_sample_sha256": sha256_file(universe_path),
            }
        )
    )
    runtime_path = tmp_path / "runtime_benchmark.json"
    runtime_path.write_text(
        json.dumps(
            {
                **identity_payload(),
                "sample_count": 100,
                "device_requested": "auto",
                "methods": {
                    method: {
                        "protocol": "full_nms",
                        "measurement": (
                            "conservative_complete_suite_upper_bound"
                        ),
                        "inference_seconds_per_sample": 0.1,
                    }
                },
                "inputs": {
                    "per_candidate": str(candidate_path),
                    "per_candidate_sha256": sha256_file(candidate_path),
                    "sample_universe": str(universe_path),
                    "sample_universe_sha256": sha256_file(universe_path),
                    "training_manifest": str(training_path),
                    "training_manifest_sha256": sha256_file(training_path),
                    "inference_bundle": str(inference_bundle_path),
                    "inference_bundle_sha256": sha256_file(
                        inference_bundle_path
                    ),
                },
            }
        )
    )
    source_payloads = {
        "split_audit": {"required_intersections_all_zero": True},
        "feature_allowlist": {
            "ground_truth_allowed": False,
            "features": ["q_raw"],
        },
    }
    source_artifacts = {}
    for role, payload in source_payloads.items():
        path = tmp_path / f"{role}.json"
        path.write_text(json.dumps(payload))
        source_artifacts[role] = {
            "path": str(path),
            "sha256": sha256_file(path),
        }
    source_artifacts["training_manifest"] = {
        "path": str(training_path),
        "sha256": sha256_file(training_path),
    }
    source_artifacts["runtime_benchmark"] = {
        "path": str(runtime_path),
        "sha256": sha256_file(runtime_path),
    }
    evidence_path = tmp_path / "evidence.json"
    evidence_path.write_text(
        json.dumps(
            {
                **identity_payload(),
                "maximum_inference_seconds_per_sample": 0.5,
                "formal_inference_device": "auto",
                "source_artifacts": source_artifacts,
                "candidate_methods": {
                    method: {
                        "protocol": "full_nms",
                    }
                },
            }
        )
    )
    output = tmp_path / "selection"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "select_primary_method.py",
            "--per-sample-outcomes",
            str(outcomes_path),
            "--bootstrap",
            str(bootstrap_path),
            "--evaluation-bundle",
            str(bundle_path),
            "--sample-universe",
            str(universe_path),
            "--eligibility-evidence",
            str(evidence_path),
            "--harmful-rate-limit-all-samples",
            "0.01",
            "--output-root",
            str(output),
        ],
    )
    assert select_primary_main() == 0
    selection = json.loads((output / "selection.json").read_text())
    assert selection["primary_method"] == method
    assert selection["selected_metrics"]["eligible"] is True
    assert {
        key: selection[key] for key in identity_payload()
    } == identity_payload()
