from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from reranking.existing_run_discovery import discover_existing_runs


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _artifact(path: Path, payload: bytes) -> dict[str, object]:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return {"path": str(path), "sha256": _sha256(path), "size_bytes": len(payload)}


def _complete_method(root: Path, method: str, *, independent: bool = True) -> dict[str, object]:
    fold_models = [
        _artifact(root / "checkpoints" / f"{method}.fold{fold}.pt", f"fold-{fold}".encode())
        for fold in range(3)
    ]
    block: dict[str, object] = {
        "status": "COMPLETE",
        "config": {"architecture": "content-declared", "hidden_dim": 17},
        "checkpoint": _artifact(root / "checkpoints" / f"{method}.final.pt", b"final"),
        "oof": {
            "status": "complete",
            "expected_folds": [0, 1, 2],
            "completed_folds": [0, 1, 2],
            "fold_count": 3,
            "fold_models": fold_models,
            "oof_predictions": _artifact(
                root / "predictions" / f"{method}.oof.jsonl", b"{}\n"
            ),
        },
    }
    if independent:
        block["independent_recomputation"] = {
            "status": "passed",
            "exact_match": True,
            "methods": [method],
        }
    return block


def _write_run(
    root: Path,
    methods: dict[str, dict[str, object]],
    *,
    status: str = "locked",
) -> Path:
    candidate = _artifact(root / "data" / "candidates.parquet", b"candidates")
    evaluator = _artifact(root / "code" / "evaluator.py", b"def evaluate(): pass\n")
    config = _write_json(
        root / "configs" / "method_config.json",
        {"methods": {method: {"family": "declared-in-content"} for method in methods}},
    )
    return _write_json(
        root / "frozen_experiment_manifest.json",
        {
            "schema_version": 1,
            "run_id": root.name,
            "status": status,
            "formal_methods": list(methods),
            "methods": methods,
            "candidate_artifact": candidate,
            "evaluator": evaluator,
            "config": {"path": str(config), "sha256": _sha256(config)},
        },
    )


def _by_method(report: dict[str, object]) -> dict[str, dict[str, object]]:
    return {
        str(item["method_id"]): item
        for item in report["methods"]  # type: ignore[index,union-attr]
    }


def test_content_declared_method_is_eligible_only_with_all_evidence(tmp_path: Path) -> None:
    run = tmp_path / "opaque-run-directory"
    method = "unseen_architecture_delta"
    _write_run(run, {method: _complete_method(run, method)})

    report = discover_existing_runs([tmp_path])
    result = _by_method(report)[method]

    assert result["eligible"] is True
    assert all(check["passed"] for check in result["checks"].values())
    assert report["eligible_methods"] == [
        {"run_id": run.name, "run_root": str(run), "method_id": method}
    ]
    assert result["configuration_sources"]


def test_v2_v3_fcer_setrank_critic_and_gate_names_are_discovered_from_content(
    tmp_path: Path,
) -> None:
    run = tmp_path / "run"
    names = (
        "v2_locked_primary",
        "v3_candidate",
        "fcer_native_variant",
        "residual_setrank_variant",
        "rgbd_critic_variant",
        "full_feature_gate_variant",
    )
    _write_run(run, {name: _complete_method(run, name) for name in names})

    discovered = _by_method(discover_existing_runs([run]))

    assert set(discovered) == set(names)
    assert all(discovered[name]["eligible"] is True for name in names)


def test_existing_files_without_completion_or_hash_contract_are_not_eligible(
    tmp_path: Path,
) -> None:
    run = tmp_path / "run"
    checkpoint = run / "checkpoint.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"present but unproven")
    _write_json(
        run / "manifest.json",
        {
            "status": "running",
            "method": "file_presence_is_not_completion",
            "checkpoint_path": str(checkpoint),
            "oof": {"status": "complete"},
        },
    )

    result = _by_method(discover_existing_runs([run]))[
        "file_presence_is_not_completion"
    ]
    codes = {gap["code"] for gap in result["gaps"]}

    assert result["eligible"] is False
    assert {
        "manifest_missing_or_incomplete",
        "complete_oof_missing",
        "checkpoint_missing_or_invalid",
        "candidate_hash_missing_or_invalid",
        "evaluator_hash_missing_or_invalid",
        "independent_recomputation_missing_or_failed",
    } <= codes
    assert any(
        item["path"] == str(checkpoint)
        for item in discover_existing_runs([run])["checkpoint_inventory"]
    )


def test_hash_drift_is_reported_with_expected_and_actual_digest(tmp_path: Path) -> None:
    run = tmp_path / "run"
    method = "hash_sensitive_method"
    _write_run(run, {method: _complete_method(run, method)})
    checkpoint = run / "checkpoints" / f"{method}.final.pt"
    checkpoint.write_bytes(b"tampered")

    result = _by_method(discover_existing_runs([run]))[method]
    issues = result["checks"]["checkpoint"]["issues"]

    assert result["eligible"] is False
    assert any(issue["issue"] == "artifact_sha256_mismatch" for issue in issues)
    mismatch = next(issue for issue in issues if issue["issue"] == "artifact_sha256_mismatch")
    assert mismatch["expected_sha256"] != mismatch["actual_sha256"]
    assert mismatch["path"] == str(checkpoint)


def test_symlinked_tree_and_symlinked_checkpoint_are_never_followed(tmp_path: Path) -> None:
    external = tmp_path / "external"
    external_method = "must_not_be_seen_through_link"
    _write_run(external, {external_method: _complete_method(external, external_method)})
    scan = tmp_path / "scan"
    scan.mkdir()
    linked_tree = scan / "linked-run"
    try:
        linked_tree.symlink_to(external, target_is_directory=True)
    except OSError as error:  # pragma: no cover - platform capability guard
        pytest.skip(f"symlinks unavailable: {error}")

    local = scan / "local-run"
    method = "symlink_checkpoint_method"
    block = _complete_method(local, method)
    real_checkpoint = local / "real.pt"
    real_checkpoint.write_bytes(b"real checkpoint")
    linked_checkpoint = local / "linked.pt"
    linked_checkpoint.symlink_to(real_checkpoint)
    block["checkpoint"] = {
        "path": str(linked_checkpoint),
        "sha256": _sha256(real_checkpoint),
    }
    _write_run(local, {method: block})

    report = discover_existing_runs([scan])
    discovered = _by_method(report)

    assert external_method not in discovered
    assert discovered[method]["eligible"] is False
    assert any(
        issue["issue"] == "artifact_symlink"
        for issue in discovered[method]["checks"]["checkpoint"]["issues"]
    )
    assert {item["path"] for item in report["scan"]["skipped_symlinks"]} >= {
        str(linked_tree),
        str(linked_checkpoint),
    }


def test_missing_independent_recomputation_is_a_structured_method_gap(
    tmp_path: Path,
) -> None:
    run = tmp_path / "run"
    method = "otherwise_complete_method"
    _write_run(run, {method: _complete_method(run, method, independent=False)})

    result = _by_method(discover_existing_runs([run]))[method]
    gap = next(
        item
        for item in result["gaps"]
        if item["code"] == "independent_recomputation_missing_or_failed"
    )

    assert result["eligible"] is False
    assert gap["evidence_issues"] == []
    assert result["checks"]["independent_recomputation"]["evidence"] == []
