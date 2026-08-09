from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from failure_analysis.vlm_safe_rerank.locking import (
    FormalRunDenied,
    assert_formal_run_allowed,
    claim_formal_run_once,
    verify_locked_manifest,
    verify_protocol_lock,
    write_locked_manifest,
    write_protocol_lock,
)
from failure_analysis.vlm_safe_rerank.manifest import (
    ManifestError,
    assert_inference_safe_manifest,
    compute_candidate_q_identity,
    verify_data_manifest,
    verify_inference_manifest,
    write_data_manifest,
    write_inference_manifest,
)
from failure_analysis.vlm_safe_rerank.full_list_runner import run_full_list_phase
from failure_analysis.vlm_safe_rerank.runner import (
    acquire_global_api_lock,
    run_pairwise_phase,
)


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _candidate_rows(sample_count: int = 2) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for sample_index in range(sample_count):
        rows.append(
            {
                "sample_id": f"sample-{sample_index}",
                "candidates": [
                    {
                        "stable_candidate_id": f"candidate-{candidate_index}",
                        "q_raw": 0.9 - 0.1 * candidate_index,
                        "candidate_checksum": (
                            f"geometry-{sample_index}-{candidate_index}"
                        ),
                    }
                    for candidate_index in range(5)
                ],
            }
        )
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _base_artifacts(tmp_path: Path, *, denominator: int = 2) -> dict[str, Path]:
    features = tmp_path / "formal_features.jsonl"
    split = tmp_path / "formal_split.json"
    labels = tmp_path / "evaluation_labels.jsonl"
    inference = tmp_path / "INFERENCE_MANIFEST.json"
    data = tmp_path / "DATA_MANIFEST.json"
    protocol = tmp_path / "PROTOCOL_LOCK.json"
    validation = tmp_path / "validation.json"
    locked = tmp_path / "LOCKED_MANIFEST.json"
    _write_jsonl(features, _candidate_rows(denominator))
    _write_json(split, {"sample_ids": [f"sample-{i}" for i in range(denominator)]})
    _write_jsonl(
        labels,
        [{"sample_id": f"sample-{i}", "candidate_correct": "candidate-0"} for i in range(denominator)],
    )
    write_inference_manifest(
        inference,
        run_id="safe-run",
        expected_denominator=denominator,
        payload={
            "candidate_features_path": features.name,
            "split_path": split.name,
            "model_ids": ["gemini-3.6-flash"],
            "sample_count": denominator,
        },
    )
    write_data_manifest(
        data,
        run_id="safe-run",
        split_manifest=split,
        candidate_sources={"formal_test": features},
        inference_manifest=inference,
        expected_denominator=denominator,
        evaluation_sources={"labels": labels},
    )
    return {
        "features": features,
        "split": split,
        "labels": labels,
        "inference": inference,
        "data": data,
        "protocol": protocol,
        "validation": validation,
        "locked": locked,
    }


def _write_protocol(paths: dict[str, Path], *, primary: str = "flash_safe") -> dict[str, object]:
    return write_protocol_lock(
        paths["protocol"],
        run_id="safe-run",
        data_manifest=paths["data"],
        inference_manifest=paths["inference"],
        expected_denominator=2,
        protocol={"name": "safe-full-evidence", "max_output_tokens": 4096},
        primary_method=primary,
    )


def _write_validation(
    paths: dict[str, Path],
    protocol: dict[str, object],
    *,
    status: str = "GO",
    primary: str = "flash_safe",
    denominator: int = 2,
) -> None:
    _write_json(
        paths["validation"],
        {
            "validation_status": status,
            "primary_method": primary,
            "expected_denominator": denominator,
            "candidate_identity_sha256": protocol["candidate_identity_sha256"],
            "q_value_sha256": protocol["q_value_sha256"],
            "combined_identity_sha256": protocol["combined_identity_sha256"],
        },
    )


def _write_complete_chain(
    tmp_path: Path, *, status: str = "GO", primary: str = "flash_safe"
) -> dict[str, Path]:
    paths = _base_artifacts(tmp_path)
    protocol = _write_protocol(paths, primary=primary)
    _write_validation(paths, protocol, status=status, primary=primary)
    write_locked_manifest(
        paths["locked"],
        run_id="safe-run",
        protocol_lock=paths["protocol"],
        validation_result=paths["validation"],
        inference_manifest=paths["inference"],
        expected_denominator=2,
        primary_method=primary,
        validation_status=status,
    )
    return paths


def test_three_manifest_layers_are_verified_and_bound(tmp_path: Path) -> None:
    paths = _write_complete_chain(tmp_path)

    data = verify_data_manifest(paths["data"])
    protocol = verify_protocol_lock(paths["protocol"])
    locked = verify_locked_manifest(paths["locked"])

    assert data["manifest_kind"] == "vlm_safe_rerank_data_manifest"
    assert protocol["manifest_kind"] == "vlm_safe_rerank_protocol_lock"
    assert locked["manifest_kind"] == "vlm_safe_rerank_locked_manifest"
    assert locked["primary_method"] == "flash_safe"
    assert locked["expected_denominator"] == 2
    assert locked["candidate_identity_sha256"] == protocol["candidate_identity_sha256"]
    assert locked["q_value_sha256"] == protocol["q_value_sha256"]
    assert locked["inference_manifest"]["sha256"] == protocol["inference_manifest"]["sha256"]


def test_candidate_q_identity_requires_exactly_five_candidates(tmp_path: Path) -> None:
    features = tmp_path / "features.jsonl"
    rows = _candidate_rows(1)
    rows[0]["candidates"] = list(rows[0]["candidates"])[:4]
    _write_jsonl(features, rows)

    with pytest.raises(ManifestError, match="exactly five"):
        compute_candidate_q_identity(features)


def test_candidate_q_identity_is_order_independent_but_q_sensitive(tmp_path: Path) -> None:
    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"
    first_rows = _candidate_rows(2)
    second_rows = list(reversed(_candidate_rows(2)))
    for row in second_rows:
        row["candidates"] = list(reversed(row["candidates"]))
    _write_jsonl(first, first_rows)
    _write_jsonl(second, second_rows)

    first_identity = compute_candidate_q_identity(first)
    second_identity = compute_candidate_q_identity(second)
    assert first_identity == second_identity

    second_rows[0]["candidates"][0]["q_raw"] = 0.123
    _write_jsonl(second, second_rows)
    changed_identity = compute_candidate_q_identity(second)
    assert changed_identity["candidate_identity_sha256"] == first_identity["candidate_identity_sha256"]
    assert changed_identity["q_value_sha256"] != first_identity["q_value_sha256"]


def test_candidate_identity_recomputes_declared_geometry_checksum(tmp_path: Path) -> None:
    features = tmp_path / "features.jsonl"
    candidates: list[dict[str, object]] = []
    geometry_fields = (
        "row", "col", "cx", "cy", "angle_rad", "angle_deg", "width_px", "height_px", "polygon"
    )
    for index in range(5):
        candidate: dict[str, object] = {
            "stable_candidate_id": f"candidate-{index}",
            "q_raw": 0.9 - 0.1 * index,
            "q_rank": index,
            "row": index,
            "col": index + 1,
            "cx": float(index + 1),
            "cy": float(index),
            "angle_rad": 0.0,
            "angle_deg": 0.0,
            "width_px": 20.0,
            "height_px": 10.0,
            "polygon": [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
        }
        geometry = {field: candidate[field] for field in geometry_fields}
        candidate["candidate_checksum"] = hashlib.sha256(
            json.dumps(
                geometry, sort_keys=True, separators=(",", ":"), allow_nan=False
            ).encode("utf-8")
        ).hexdigest()
        candidates.append(candidate)
    _write_jsonl(features, [{"sample_id": "sample-0", "candidates": candidates}])
    compute_candidate_q_identity(features)

    candidates[0]["width_px"] = 21.0
    _write_jsonl(features, [{"sample_id": "sample-0", "candidates": candidates}])
    with pytest.raises(ManifestError, match="geometry no longer matches"):
        compute_candidate_q_identity(features)


def test_candidate_q_identity_rejects_stale_stored_q_rank(tmp_path: Path) -> None:
    features = tmp_path / "features.jsonl"
    rows = _candidate_rows(1)
    for index, candidate in enumerate(rows[0]["candidates"]):
        candidate["q_rank"] = index
    rows[0]["candidates"][0]["q_rank"] = 4
    _write_jsonl(features, rows)

    with pytest.raises(ManifestError, match="stale q_rank"):
        compute_candidate_q_identity(features)


@pytest.mark.parametrize(
    "unsafe_payload",
    [
        {"labels_path": "/tmp/labels.jsonl"},
        {"nested": {"ground_truth": [1, 2, 3]}},
        {"path": "/tmp/evaluation_only/formal.json"},
        {"inputs": [{"gt_mask": "forbidden"}]},
    ],
)
def test_inference_manifest_rejects_label_gt_and_evaluation_only_data(
    unsafe_payload: dict[str, object],
) -> None:
    with pytest.raises(ManifestError, match="forbidden"):
        assert_inference_safe_manifest(unsafe_payload)


def test_evaluation_labels_may_be_separate_from_inference_manifest(tmp_path: Path) -> None:
    paths = _base_artifacts(tmp_path)

    inference = verify_inference_manifest(paths["inference"])
    data = verify_data_manifest(paths["data"])

    assert "evaluation_sources" not in inference["inference_inputs"]
    assert "labels" in data["evaluation_sources"]


@pytest.mark.parametrize("status", ["NO-GO", "INCONCLUSIVE", "go", ""])
def test_only_exact_validation_status_go_can_create_locked_manifest(
    tmp_path: Path, status: str
) -> None:
    paths = _base_artifacts(tmp_path)
    protocol = _write_protocol(paths)
    _write_validation(paths, protocol, status=status)

    with pytest.raises(FormalRunDenied, match="exactly 'GO'"):
        write_locked_manifest(
            paths["locked"],
            run_id="safe-run",
            protocol_lock=paths["protocol"],
            validation_result=paths["validation"],
            inference_manifest=paths["inference"],
            expected_denominator=2,
        )


@pytest.mark.parametrize("primary", ["q-only", "q_only", "crog_q_only", "QOnly"])
def test_q_only_primary_cannot_create_formal_lock(tmp_path: Path, primary: str) -> None:
    paths = _base_artifacts(tmp_path)
    protocol = _write_protocol(paths, primary=primary)
    _write_validation(paths, protocol, primary=primary)

    with pytest.raises(FormalRunDenied, match="q-only"):
        write_locked_manifest(
            paths["locked"],
            run_id="safe-run",
            protocol_lock=paths["protocol"],
            validation_result=paths["validation"],
            inference_manifest=paths["inference"],
            expected_denominator=2,
        )


@pytest.mark.parametrize(
    ("cli_allow", "environment", "message"),
    [
        (False, {"ALLOW_FORMAL_API_RUN": "1"}, "explicit CLI"),
        ("1", {"ALLOW_FORMAL_API_RUN": "1"}, "explicit CLI"),
        (True, {}, "equal exactly"),
        (True, {"ALLOW_FORMAL_API_RUN": "true"}, "equal exactly"),
        (True, {"ALLOW_FORMAL_API_RUN": "01"}, "equal exactly"),
    ],
)
def test_formal_gate_requires_exact_cli_and_environment_opt_ins(
    tmp_path: Path,
    cli_allow: object,
    environment: dict[str, str],
    message: str,
) -> None:
    paths = _write_complete_chain(tmp_path)

    with pytest.raises(FormalRunDenied, match=message):
        assert_formal_run_allowed(
            paths["locked"],
            cli_allow_formal=cli_allow,  # type: ignore[arg-type]
            environ=environment,
        )


def test_formal_gate_rejects_non_p5_primary_even_with_both_opt_ins(tmp_path: Path) -> None:
    paths = _write_complete_chain(tmp_path)

    with pytest.raises(FormalRunDenied, match="only an evidence-recomputed P5"):
        assert_formal_run_allowed(
            paths["locked"],
            cli_allow_formal=True,
            environ={"ALLOW_FORMAL_API_RUN": "1"},
            expected_denominator=2,
            primary_method="flash_safe",
        )


def test_data_protocol_and_locked_layers_are_create_exclusive(tmp_path: Path) -> None:
    paths = _base_artifacts(tmp_path)
    with pytest.raises(FileExistsError):
        write_data_manifest(
            paths["data"],
            run_id="replacement",
            split_manifest=paths["split"],
            candidate_sources={"formal_test": paths["features"]},
            inference_manifest=paths["inference"],
            expected_denominator=2,
        )

    protocol = _write_protocol(paths)
    with pytest.raises(FileExistsError):
        _write_protocol(paths)

    _write_validation(paths, protocol)
    write_locked_manifest(
        paths["locked"],
        run_id="safe-run",
        protocol_lock=paths["protocol"],
        validation_result=paths["validation"],
        inference_manifest=paths["inference"],
        expected_denominator=2,
    )
    with pytest.raises(FileExistsError):
        write_locked_manifest(
            paths["locked"],
            run_id="replacement",
            protocol_lock=paths["protocol"],
            validation_result=paths["validation"],
            inference_manifest=paths["inference"],
            expected_denominator=2,
        )


def test_candidate_or_q_drift_invalidates_full_lock_chain(tmp_path: Path) -> None:
    paths = _write_complete_chain(tmp_path)
    rows = _candidate_rows(2)
    rows[0]["candidates"][0]["q_raw"] = 0.111
    _write_jsonl(paths["features"], rows)

    with pytest.raises(ManifestError, match="drift"):
        verify_locked_manifest(paths["locked"])


def test_protocol_file_drift_invalidates_locked_manifest(tmp_path: Path) -> None:
    paths = _write_complete_chain(tmp_path)
    protocol = json.loads(paths["protocol"].read_text(encoding="utf-8"))
    protocol["protocol"]["max_output_tokens"] = 1
    _write_json(paths["protocol"], protocol)

    with pytest.raises(ManifestError, match="drift|hash mismatch"):
        verify_locked_manifest(paths["locked"])


def test_validation_file_drift_invalidates_locked_manifest(tmp_path: Path) -> None:
    paths = _write_complete_chain(tmp_path)
    validation = json.loads(paths["validation"].read_text(encoding="utf-8"))
    validation["validation_status"] = "NO-GO"
    _write_json(paths["validation"], validation)

    with pytest.raises(ManifestError, match="drift"):
        verify_locked_manifest(paths["locked"])


def test_validation_and_formal_denominators_are_bound_separately(tmp_path: Path) -> None:
    paths = _base_artifacts(tmp_path)
    protocol = _write_protocol(paths)
    _write_validation(paths, protocol, denominator=1)
    locked = write_locked_manifest(
        paths["locked"],
        run_id="safe-run",
        protocol_lock=paths["protocol"],
        validation_result=paths["validation"],
        inference_manifest=paths["inference"],
        expected_denominator=2,
    )
    assert locked["expected_denominator"] == 2
    assert locked["validation_expected_denominator"] == 1
    assert verify_locked_manifest(paths["locked"])["validation_expected_denominator"] == 1


def test_primary_mismatch_cannot_be_locked(tmp_path: Path) -> None:
    paths = _base_artifacts(tmp_path)
    protocol = write_protocol_lock(
        paths["protocol"],
        run_id="safe-run",
        data_manifest=paths["data"],
        inference_manifest=paths["inference"],
        expected_denominator=2,
        protocol={"name": "safe-full-evidence"},
        eligible_primary_methods=["flash_safe", "er2_safe"],
    )
    _write_validation(paths, protocol, primary="er2_safe")

    with pytest.raises(FormalRunDenied, match="explicit primary"):
        write_locked_manifest(
            paths["locked"],
            run_id="safe-run",
            protocol_lock=paths["protocol"],
            validation_result=paths["validation"],
            inference_manifest=paths["inference"],
            expected_denominator=2,
            primary_method="flash_safe",
        )


def test_formal_claim_binds_lock_primary_denominator_inference_and_candidate_q(
    tmp_path: Path,
) -> None:
    paths = _write_complete_chain(tmp_path)
    claim_path = tmp_path / "FORMAL_RUN_CLAIM.json"

    with pytest.raises(FormalRunDenied, match="only an evidence-recomputed P5"):
        claim_formal_run_once(
            claim_path,
            paths["locked"],
            cli_allow_formal=True,
            environ={"ALLOW_FORMAL_API_RUN": "1"},
        )
    assert not claim_path.exists()


def test_formal_gate_rejects_requested_primary_or_denominator_drift(tmp_path: Path) -> None:
    paths = _write_complete_chain(tmp_path)
    environment = {"ALLOW_FORMAL_API_RUN": "1"}

    with pytest.raises(FormalRunDenied, match="denominator differs"):
        assert_formal_run_allowed(
            paths["locked"],
            cli_allow_formal=True,
            environ=environment,
            expected_denominator=3,
        )
    with pytest.raises(FormalRunDenied, match="primary method differs"):
        assert_formal_run_allowed(
            paths["locked"],
            cli_allow_formal=True,
            environ=environment,
            primary_method="er2_safe",
        )


@pytest.mark.parametrize("phase", ["formal_test", "formal-test", "FormalTest", "FORMAL"])
def test_generic_runner_cannot_bypass_formal_gate(tmp_path: Path, phase: str) -> None:
    with pytest.raises(FormalRunDenied, match="generic pairwise runner"):
        run_pairwise_phase(
            run_dir=tmp_path,
            phase=phase,
            env_file=tmp_path / "missing.env",
        )


def test_development_phase_cannot_consume_audited_formal_source(tmp_path: Path) -> None:
    feature = tmp_path / "formal_features.jsonl"
    feature.write_text('{"sample_id":"formal-sample"}\n', encoding="utf-8")
    digest = hashlib.sha256(feature.read_bytes()).hexdigest()
    audit_path = tmp_path / "audit_inventory.json"
    _write_json(
        audit_path,
        {
            "frozen_sources": {
                "formal_test": {
                    "path": str(feature.resolve()),
                    "file_sha256": digest,
                }
            }
        },
    )
    _write_json(
        tmp_path / "AUDIT_INVENTORY_IDENTITY.json",
        {"sha256": hashlib.sha256(audit_path.read_bytes()).hexdigest()},
    )
    phase_dir = tmp_path / "development"
    phase_dir.mkdir()
    manifest = phase_dir / "inference_manifest.json"
    _write_json(
        manifest,
        {"feature_file": str(feature.resolve()), "rows": []},
    )
    _write_json(
        phase_dir / "INFERENCE_MANIFEST_IDENTITY.json",
        {"sha256": hashlib.sha256(manifest.read_bytes()).hexdigest()},
    )

    with pytest.raises(FormalRunDenied, match="formal feature source"):
        run_pairwise_phase(
            run_dir=tmp_path,
            phase="development",
            env_file=tmp_path / "missing.env",
        )
    released = acquire_global_api_lock(tmp_path)
    released.close()


def test_zero_model_phase_completes_without_api_environment(tmp_path: Path) -> None:
    feature = tmp_path / "train_features.jsonl"
    feature.write_text("", encoding="utf-8")
    digest = hashlib.sha256(feature.read_bytes()).hexdigest()
    audit_path = tmp_path / "audit_inventory.json"
    _write_json(
        audit_path,
        {
            "frozen_sources": {
                "calibration": {
                    "path": str(feature.resolve()),
                    "file_sha256": digest,
                }
            }
        },
    )
    _write_json(
        tmp_path / "AUDIT_INVENTORY_IDENTITY.json",
        {"sha256": hashlib.sha256(audit_path.read_bytes()).hexdigest()},
    )
    phase_dir = tmp_path / "p5_validation"
    phase_dir.mkdir()
    manifest = phase_dir / "inference_manifest.json"
    _write_json(
        manifest,
        {"feature_file": str(feature.resolve()), "needed_models": [], "rows": []},
    )
    _write_json(
        phase_dir / "INFERENCE_MANIFEST_IDENTITY.json",
        {"sha256": hashlib.sha256(manifest.read_bytes()).hexdigest()},
    )

    summary = run_pairwise_phase(
        run_dir=tmp_path,
        phase="p5_validation",
        env_file=tmp_path / "missing.env",
        models=(),
    )

    assert summary["model_pair_variants"] == 0
    assert json.loads(
        (phase_dir / "RUNNER_STATE.json").read_text(encoding="utf-8")
    )["status"] == "completed"
    assert (phase_dir / "pairwise_responses.parquet").is_file()
    released = acquire_global_api_lock(tmp_path)
    released.close()


def test_full_list_phase_alias_cannot_bypass_formal_gate(tmp_path: Path) -> None:
    with pytest.raises(FormalRunDenied, match="generic full-list runner"):
        run_full_list_phase(
            run_dir=tmp_path,
            source_phase="development",
            output_phase="FORMAL-test",
            env_file=tmp_path / "missing.env",
        )


def test_p5_lock_requires_passed_perturbation_stability(tmp_path: Path) -> None:
    paths = _base_artifacts(tmp_path)
    protocol = _write_protocol(paths, primary="P5_er2_safe")
    _write_validation(paths, protocol, primary="P5_er2_safe")
    with pytest.raises(FormalRunDenied, match="stability"):
        write_locked_manifest(
            paths["locked"],
            run_id="safe-run",
            protocol_lock=paths["protocol"],
            validation_result=paths["validation"],
            inference_manifest=paths["inference"],
            expected_denominator=2,
            primary_method="P5_er2_safe",
            validation_status="GO",
        )

    validation = json.loads(paths["validation"].read_text(encoding="utf-8"))
    validation["stability_contract_passed"] = True
    _write_json(paths["validation"], validation)
    with pytest.raises(FormalRunDenied, match="evidence binding"):
        write_locked_manifest(
            paths["locked"],
            run_id="safe-run",
            protocol_lock=paths["protocol"],
            validation_result=paths["validation"],
            inference_manifest=paths["inference"],
            expected_denominator=2,
            primary_method="P5_er2_safe",
            validation_status="GO",
        )


def test_run_wide_provider_lock_rejects_second_phase(tmp_path: Path) -> None:
    first = acquire_global_api_lock(tmp_path)
    try:
        with pytest.raises(RuntimeError, match="another provider runner"):
            acquire_global_api_lock(tmp_path)
    finally:
        first.close()
