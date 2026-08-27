from __future__ import annotations

import json
from pathlib import Path

import pytest

from graspnet6d import smoke
from graspnet6d.audit import sha256_file


def _write_status(directory: Path, **changes: object) -> tuple[Path, Path]:
    directory.mkdir(parents=True, exist_ok=True)
    report = directory / "4d_regression_report.md"
    report.write_text(
        "# Legacy 4-DoF regression\n\n123 tests passed with exit code zero.\n",
        encoding="utf-8",
    )
    payload: dict[str, object] = {
        "schema_version": smoke.REGRESSION_STATUS_SCHEMA_VERSION,
        "evidence_kind": smoke.REGRESSION_EVIDENCE_KIND,
        "status": "PASS",
        "exit_code": 0,
        "test_count": 123,
        "fixture_only": False,
        "placeholder": False,
        "report_path": report.name,
        "report_sha256": sha256_file(report),
    }
    payload.update(changes)
    status = directory / smoke.REGRESSION_STATUS_FILENAME
    status.write_text(json.dumps(payload), encoding="utf-8")
    return status, report


def test_regression_status_requires_verified_full_report(tmp_path: Path) -> None:
    directory = tmp_path / "regression"
    _, report = _write_status(directory)

    check = smoke.validate_4d_regression_status(directory)

    assert check.passed
    assert check.number == 12
    assert check.kind == "data_independent_regression"
    assert "123 tests" in check.evidence
    assert sha256_file(report) in check.evidence


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"schema_version": 2}, "unsupported.*schema"),
        ({"evidence_kind": "fixture"}, "not the full suite"),
        ({"status": "FAIL"}, "not PASS"),
        ({"exit_code": 1}, "not integer zero"),
        ({"exit_code": False}, "not integer zero"),
        ({"test_count": 0}, "not a positive integer"),
        ({"test_count": True}, "not a positive integer"),
        ({"fixture_only": True}, "fixture-only"),
        ({"placeholder": True}, "placeholder"),
        ({"report_path": "../report.md"}, "inside the regression directory"),
        ({"report_sha256": "not-a-hash"}, "not a lowercase SHA-256"),
    ],
)
def test_regression_status_rejects_invalid_claims(
    tmp_path: Path, changes: dict[str, object], message: str
) -> None:
    directory = tmp_path / "regression"
    _write_status(directory, **changes)

    check = smoke.validate_4d_regression_status(directory)

    assert not check.passed
    assert __import__("re").search(message, check.evidence)


def test_regression_status_rejects_mere_report_existence(tmp_path: Path) -> None:
    directory = tmp_path / "regression"
    directory.mkdir()
    (directory / "4d_regression_report.md").write_text("all passed\n", encoding="utf-8")

    check = smoke.validate_4d_regression_status(directory)

    assert not check.passed
    assert "missing regular regression status" in check.evidence


def test_regression_status_rejects_report_hash_mismatch(tmp_path: Path) -> None:
    directory = tmp_path / "regression"
    _, report = _write_status(directory)
    report.write_text("report changed after status was signed\n", encoding="utf-8")

    check = smoke.validate_4d_regression_status(directory)

    assert not check.passed
    assert "hash mismatch" in check.evidence


def test_regression_status_rejects_placeholder_report(tmp_path: Path) -> None:
    directory = tmp_path / "regression"
    status, report = _write_status(directory)
    report.write_text("placeholder\n", encoding="utf-8")
    payload = json.loads(status.read_text(encoding="utf-8"))
    payload["report_sha256"] = sha256_file(report)
    status.write_text(json.dumps(payload), encoding="utf-8")

    check = smoke.validate_4d_regression_status(directory)

    assert not check.passed
    assert "empty or a placeholder" in check.evidence


def test_run_smoke_uses_validated_regression_status(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "repo"
    _write_status(root / "artifacts/graspnet6d/regression")
    monkeypatch.setattr(smoke, "repository_root", lambda: root)
    monkeypatch.setattr(smoke, "discover_scene_ids", lambda _: [])
    monkeypatch.setattr(
        smoke,
        "_vgn_finite_check",
        lambda: smoke.Check(5, "VGN output finite", True, "test diagnostic"),
    )

    checks = smoke.run_smoke(tmp_path / "smoke-output")

    assert len(checks) == 12
    assert checks[-1].number == 12
    assert checks[-1].passed


def test_ranker_integration_threshold_cannot_be_lowered_to_smoke_size(
    tmp_path: Path,
) -> None:
    assert smoke.SMOKE_GEOMETRY_GROUPS == 8
    assert smoke.SMOKE_EVALUATOR_GROUPS == 8
    assert smoke.SMOKE_MINIMUM_NONEMPTY_CANDIDATE_GROUPS == 4
    assert smoke.RANKER_INTEGRATION_MINIMUM_GROUPS == 20
    with pytest.raises(ValueError, match="separate real-data gate.*at least 20"):
        smoke.run_real_ranker_smoke(
            tmp_path / "train.csv",
            tmp_path / "validation.csv",
            tmp_path / "test.csv",
            (tmp_path / "u1.csv", tmp_path / "u2.csv", tmp_path / "u3.csv"),
            tmp_path / "schema.yaml",
            tmp_path / "output",
            minimum_real_groups=8,
        )


def _write_candidate_manifest(run_dir: Path, rows: list[dict[str, object]]) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "candidate_manifest.jsonl").write_text(
        "".join(f"{json.dumps(row)}\n" for row in rows),
        encoding="utf-8",
    )


def _candidate_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for condition in smoke.SMOKE_GROUNDING_CONDITIONS:
        for index in range(8):
            rows.append(
                {
                    "grounding_condition": condition,
                    "group_id": f"group-{index}",
                    "candidate_count": (
                        1
                        if condition != "oracle_gt_mask" or index < 4
                        else 0
                    ),
                }
            )
    return rows


def test_real_candidate_smoke_keys_by_condition_and_counts_only_oracle(
    tmp_path: Path,
) -> None:
    rows = _candidate_rows()
    _write_candidate_manifest(tmp_path, rows)

    check = smoke._real_artifact_checks(tmp_path)[0]

    assert check.passed
    assert "grounding_condition=oracle_gt_mask" in check.evidence
    assert "oracle_indexed_groups=8" in check.evidence
    assert "oracle_non_empty_groups=4" in check.evidence
    assert "all_condition_rows=24" in check.evidence


def test_real_candidate_smoke_rejects_duplicate_within_condition(
    tmp_path: Path,
) -> None:
    rows = _candidate_rows()
    rows.append(dict(rows[0]))
    _write_candidate_manifest(tmp_path, rows)

    check = smoke._real_artifact_checks(tmp_path)[0]

    assert not check.passed
    assert "duplicate (grounding_condition, group_id)" in check.evidence


@pytest.mark.parametrize("condition", ["", "oracle", None, 7])
def test_real_candidate_smoke_rejects_malformed_condition(
    tmp_path: Path, condition: object
) -> None:
    rows = _candidate_rows()
    rows[0]["grounding_condition"] = condition
    _write_candidate_manifest(tmp_path, rows)

    check = smoke._real_artifact_checks(tmp_path)[0]

    assert not check.passed
    assert "malformed grounding_condition" in check.evidence


def test_predicted_mask_nonempty_pools_do_not_satisfy_oracle_gate(
    tmp_path: Path,
) -> None:
    rows = _candidate_rows()
    for row in rows:
        if row["grounding_condition"] == "oracle_gt_mask":
            row["candidate_count"] = 1 if row["group_id"] in {
                "group-0",
                "group-1",
                "group-2",
            } else 0
    _write_candidate_manifest(tmp_path, rows)

    check = smoke._real_artifact_checks(tmp_path)[0]

    assert not check.passed
    assert "oracle_non_empty_groups=3" in check.evidence


@pytest.mark.parametrize(
    ("grounding_condition", "passed"),
    [("oracle_gt_mask", True), ("hifics_zero_shot_mask", False)],
)
def test_evaluator_smoke_counts_only_explicit_oracle_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    grounding_condition: str,
    passed: bool,
) -> None:
    evidence_path = tmp_path / "evaluator_parity" / "evidence.json"
    evidence_path.parent.mkdir(parents=True)
    evidence_path.write_text(
        json.dumps(
            {
                "validated_group_ids": [f"group-{index}" for index in range(8)],
                "grounding_condition": grounding_condition,
            }
        ),
        encoding="utf-8",
    )

    def fake_loader(*_args: object, **_kwargs: object) -> tuple[object, dict[str, object]]:
        return object(), {
            "artifact_path": str(evidence_path),
            "verified_evidence": {"candidate_count": 8},
        }

    monkeypatch.setattr(
        "graspnet6d.stages.load_evaluator_parity_gate", fake_loader
    )

    check = smoke._real_artifact_checks(tmp_path)[2]

    assert check.passed is passed
    assert f"grounding_condition={grounding_condition!r}" in check.evidence
