from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest

from gtmask_counterfactual.io import (
    artifact_record,
    atomic_json,
    atomic_parquet,
    canonical_sha256,
)
from gtmask_counterfactual.postprocess_inputs import (
    ASSEMBLY_MANIFEST_RELATIVE_PATH,
    PostprocessInputAssemblyError,
    REQUIRED_PRODUCER_CONTRACTS,
    assemble_postprocess_inputs,
)


_TOOL_PATH = (
    Path(__file__).resolve().parents[2]
    / "tools/gtmask_counterfactual/assemble_postprocess_inputs.py"
)
_TOOL_SPEC = importlib.util.spec_from_file_location(
    "_test_assemble_postprocess_inputs", _TOOL_PATH
)
assert _TOOL_SPEC is not None and _TOOL_SPEC.loader is not None
tool = importlib.util.module_from_spec(_TOOL_SPEC)
_TOOL_SPEC.loader.exec_module(tool)


SAMPLE_IDS = ("sample-0", "sample-1")
ROUTES = ("G1", "C1", "D1")


def _hashed(path: Path, value: dict[str, object]) -> Path:
    payload = dict(value)
    payload["content_sha256"] = canonical_sha256(payload)
    return atomic_json(path, payload)


def _run(tmp_path: Path) -> Path:
    root = tmp_path / "runs/fair_gtmask_counterfactual_g1_c1_d1_synthetic"
    lock = atomic_json(
        root / "01_protocol_lock/COUNTERFACTUAL_PROTOCOL_LOCK.json",
        {"status": "LOCKED", "test_only_synthetic_contract": True},
    )
    atomic_json(
        root / "01_protocol_lock/COUNTERFACTUAL_EXECUTION.json",
        {
            "status": "RUNNING",
            "execution_count": 1,
            "protocol_lock_file_sha256": artifact_record(lock)["sha256"],
        },
    )
    atomic_json(
        root / "pipeline_status.json",
        {
            "status": "P6_D1_COUNTERFACTUAL_COMPLETE",
            "counterfactual_execution_count": 1,
        },
    )
    sample_path = atomic_parquet(
        pd.DataFrame(
            {
                "sample_id": SAMPLE_IDS,
                "scene_id": ("scene-0", "scene-1"),
                "frame_id": ("frame-0", "frame-1"),
            }
        ),
        root / "02_sample_manifest/counterfactual_manifest.parquet",
    )
    registry = atomic_parquet(
        pd.DataFrame(
            {
                "sample_id": SAMPLE_IDS,
                "gt_grasp_rectangles": (
                    "[[[0,0],[1,0],[1,1],[0,1]]]",
                    "[[[2,2],[3,2],[3,3],[2,3]]]",
                ),
            }
        ),
        root / "02_sample_manifest/gt_grasp_registry.parquet",
    )
    _hashed(
        root / "02_sample_manifest/GT_GRASP_AUTHORITY.json",
        {
            "schema_version": 1,
            "status": "COMPLETE",
            "sample_count": len(SAMPLE_IDS),
            "gt_grasp_rows_read": len(SAMPLE_IDS),
            "protocol_lock": artifact_record(lock),
            "execution_authority_mode": "prospective_execution_claim",
            "execution_claim": artifact_record(
                root / "01_protocol_lock/COUNTERFACTUAL_EXECUTION.json"
            ),
            "sample_manifest": artifact_record(sample_path),
            "registry": artifact_record(registry),
        },
    )
    return root


def _route_frames(root: Path, *, route: str, branch: str) -> Path:
    source_root = (
        root / "04_predicted_replay" / route.lower()
        if branch == "predicted"
        else root / "06_gtmask_predictions" / route.lower() / branch
    )
    candidates = pd.DataFrame(
        {
            "sample_id": [SAMPLE_IDS[0]],
            "route": [route],
            "branch": [branch],
            "candidate_id": [f"{route.lower()}-{branch}-candidate"],
            "native_rank": [1],
            "native_score": [0.75],
            "cx_px": [10.0],
            "cy_px": [20.0],
            "theta_deg": [30.0],
            # Source aliases are intentionally normalised to canonical names.
            "width_px": [12.0],
            "height_px": [6.0],
        }
    )
    samples = pd.DataFrame(
        {
            "sample_id": SAMPLE_IDS,
            "route": [route, route],
            "branch": [branch, branch],
            "candidate_count": [1, 0],
            "no_output": [False, True],
            "technical_failure": [False, False],
            "status": ["COMPLETE", "NO_OUTPUT"],
        }
    )
    candidate_path = atomic_parquet(candidates, source_root / "per_candidate.parquet")
    sample_path = atomic_parquet(samples, source_root / "per_sample.parquet")
    return _hashed(
        source_root / "manifest.json",
        {
            "schema_version": 1,
            "status": "PASS" if branch == "predicted" else "COMPLETE",
            "route": route.lower(),
            "branch": branch,
            "sample_count": len(SAMPLE_IDS),
            "candidate_count": len(candidates),
            "candidates": artifact_record(candidate_path),
            "per_sample": artifact_record(sample_path),
        },
    )


def _baseline(root: Path, predicted: dict[str, Path]) -> Path:
    return _hashed(
        root / "04_predicted_replay/BASELINE_REPLAY_MANIFEST.json",
        {
            "schema_version": 1,
            "status": "PASS",
            "sample_count": len(SAMPLE_IDS),
            "route_replays": {
                route.lower(): artifact_record(path)
                for route, path in predicted.items()
            },
            "raw_test_ground_truth_rows_read": 0,
        },
    )


def _downstream_producers(root: Path) -> None:
    covariates_path = atomic_parquet(
        pd.DataFrame(
            {
                "sample_id": SAMPLE_IDS,
                "query_type": ("name", "attribute"),
                "predicted_mask_iou": (0.4, 0.8),
                "target_area_fraction": (0.1, 0.2),
                "mask_component_count": (1, 2),
                "mask_boundary_complexity": (0.25, 0.5),
                "valid_depth_ratio": (0.9, 0.95),
                "scene_family": ("table", "floor"),
                "frame_family": ("near", "far"),
            }
        ),
        root / "03_gt_mask_registry/sample_covariates.parquet",
    )
    _hashed(
        root / "03_gt_mask_registry/SAMPLE_COVARIATES_AUTHORITY.json",
        {
            "schema_version": 1,
            "status": "COMPLETE",
            "sample_count": len(SAMPLE_IDS),
            "protocol_lock": artifact_record(
                root / "01_protocol_lock/COUNTERFACTUAL_PROTOCOL_LOCK.json"
            ),
            "execution_authority_mode": "prospective_execution_claim",
            "execution_claim": artifact_record(
                root / "01_protocol_lock/COUNTERFACTUAL_EXECUTION.json"
            ),
            "sample_manifest": artifact_record(
                root / "02_sample_manifest/counterfactual_manifest.parquet"
            ),
            "covariates": artifact_record(covariates_path),
            "gt_grasp_rows_read": 0,
        },
    )
    final_path = atomic_parquet(
        pd.DataFrame(
            [
                {
                    "sample_id": sample_id,
                    "route": route,
                    "final_correct": int(index == 0),
                }
                for route in ROUTES
                for index, sample_id in enumerate(SAMPLE_IDS)
            ]
        ),
        root / "04_predicted_replay/frozen_final_outcomes.parquet",
    )
    _hashed(
        root / "04_predicted_replay/FINAL_OUTCOMES_AUTHORITY.json",
        {
            "schema_version": 1,
            "status": "LOCKED",
            "final_outcomes": artifact_record(final_path),
            "frozen_selector_contracts": {
                route: {"system": f"{route.lower()}_synthetic"}
                for route in ROUTES
            },
        },
    )
    registry = atomic_parquet(
        pd.DataFrame({"sample_id": SAMPLE_IDS}),
        root / "04_predicted_replay/VISUAL_ASSET_REGISTRY.parquet",
    )
    _hashed(
        root / "04_predicted_replay/VISUAL_ASSET_REGISTRY.json",
        {
            "schema_version": 1,
            "status": "COMPLETE",
            "sample_count": len(SAMPLE_IDS),
            "registry": artifact_record(registry),
        },
    )


def _complete_source_dag(root: Path) -> None:
    predicted: dict[str, Path] = {}
    for route in ROUTES:
        _route_frames(root, route=route, branch="gt_oracle")
        predicted[route] = _route_frames(root, route=route, branch="predicted")
    _baseline(root, predicted)
    _downstream_producers(root)


def _unsigned(value: dict[str, object]) -> dict[str, object]:
    result = dict(value)
    result.pop("content_sha256")
    return result


def test_synthetic_production_dag_assembles_six_canonical_frames(
    tmp_path: Path,
) -> None:
    root = _run(tmp_path)
    _complete_source_dag(root)

    inputs_path, assembly_path = assemble_postprocess_inputs(
        root, expected_sample_count=len(SAMPLE_IDS)
    )

    inputs = json.loads(inputs_path.read_text(encoding="utf-8"))
    assembly = json.loads(assembly_path.read_text(encoding="utf-8"))
    assert inputs["content_sha256"] == canonical_sha256(_unsigned(inputs))
    assert assembly["content_sha256"] == canonical_sha256(_unsigned(assembly))
    assert inputs["available_routes"] == list(ROUTES)
    assert set(inputs["artifacts"]["candidates"]) == {
        f"{route}|{branch}" for route in ROUTES for branch in ("predicted", "gt_oracle")
    }
    assert set(assembly["route_source_declarations"]) == set(
        inputs["artifacts"]["candidates"]
    )
    for key, declaration_record in assembly["route_source_declarations"].items():
        declaration = json.loads(
            Path(declaration_record["path"]).read_text(encoding="utf-8")
        )
        assert declaration["content_sha256"] == canonical_sha256(_unsigned(declaration))
        assert declaration["producer_manifest"]["sha256"]
        frame = pd.read_parquet(inputs["artifacts"]["candidates"][key]["path"])
        assert "jaw_width_px" in frame
        assert "rectangle_height_px" in frame
        assert "width_px" not in frame
        assert "height_px" not in frame


def test_core_assembles_g1_c1_without_predeclared_d1_blocker(
    tmp_path: Path,
) -> None:
    root = _run(tmp_path)
    _complete_source_dag(root)
    atomic_json(
        root / "pipeline_status.json",
        {
            "status": "P5B_G1_FULL_COMPLETE",
            "counterfactual_execution_count": 1,
        },
    )
    _hashed(
        root
        / "00_audit/machine_blockers/D1_CASE_B_IRRECOVERABLE_BLOCKER.json",
        {
            "status": "UNRECOVERABLE_BLOCKER",
            "blocker_class": "IRRECOVERABLE_FROZEN_SOURCE_EVIDENCE",
            "raw_candidate_regeneration_required": True,
            "filter_only_primary_allowed": False,
            "missing_evidence": "frozen raw candidate source is absent",
            "search_paths": ["/frozen/source"],
            "stack_trace": "RuntimeError: source absent",
            "resume_command": "python -m tools.gtmask_counterfactual.run_d1_case_b predicted-replay",
        },
    )
    final_path = root / "04_predicted_replay/frozen_final_outcomes.parquet"
    final = pd.read_parquet(final_path)
    atomic_parquet(final.loc[~final["route"].eq("D1")], final_path)
    _hashed(
        root / "04_predicted_replay/FINAL_OUTCOMES_AUTHORITY.json",
        {
            "schema_version": 1,
            "status": "LOCKED",
            "final_outcomes": artifact_record(final_path),
            "frozen_selector_contracts": {
                route: {"system": f"{route.lower()}_synthetic"}
                for route in ("G1", "C1")
            },
        },
    )

    inputs_path, assembly_path = assemble_postprocess_inputs(
        root, expected_sample_count=len(SAMPLE_IDS)
    )
    inputs = json.loads(inputs_path.read_text(encoding="utf-8"))
    assembly = json.loads(assembly_path.read_text(encoding="utf-8"))
    expected_keys = {
        f"{route}|{branch}"
        for route in ("G1", "C1")
        for branch in ("predicted", "gt_oracle")
    }
    assert inputs["available_routes"] == ["G1", "C1"]
    assert set(inputs["artifacts"]["candidates"]) == expected_keys
    assert "d1_blocker" not in inputs["artifacts"]
    assert inputs["d1_secondary_status"] == "PENDING_AFTER_CORE"
    assert assembly["routes"] == ["G1", "C1"]
    assert assembly["omitted_routes"] == ["D1"]
    assert not (root / "07_candidate_tables/raw/d1").exists()


def test_missing_predicted_covariate_and_final_producers_fail_closed_but_gt_is_normalised(
    tmp_path: Path,
) -> None:
    root = _run(tmp_path)
    for route in ROUTES:
        _route_frames(root, route=route, branch="gt_oracle")
    _baseline(root, {})

    with pytest.raises(PostprocessInputAssemblyError) as captured:
        assemble_postprocess_inputs(root, expected_sample_count=len(SAMPLE_IDS))

    codes = {issue["code"] for issue in captured.value.issues}
    assert codes == {
        "PREDICTED_ROUTE_FRAME_PRODUCER_REQUIRED",
        "SAMPLE_COVARIATES_PRODUCER_REQUIRED",
        "FINAL_OUTCOMES_PRODUCER_REQUIRED",
        "VISUAL_ASSET_REGISTRY_PRODUCER_REQUIRED",
    }
    assert not (root / "07_candidate_tables/POSTPROCESS_INPUTS.json").exists()
    for route in ROUTES:
        output = root / f"07_candidate_tables/raw/{route.lower()}/gt_oracle"
        assert (output / "candidates.parquet").is_file()
        declaration = json.loads(
            (output / "source_declaration.json").read_text(encoding="utf-8")
        )
        assert declaration["producer_manifest"]["path"].endswith(
            f"06_gtmask_predictions/{route.lower()}/gt_oracle/manifest.json"
        )
    predicted_contract = REQUIRED_PRODUCER_CONTRACTS["predicted_route_frames"]
    assert predicted_contract["candidates"].startswith("04_predicted_replay/")
    assert "candidates=<exact artifact record>" in predicted_contract["manifest_fields"]


def test_exact_resume_reuses_six_frames_and_manifests(tmp_path: Path) -> None:
    root = _run(tmp_path)
    _complete_source_dag(root)
    first = assemble_postprocess_inputs(root, expected_sample_count=len(SAMPLE_IDS))
    records = {path: artifact_record(path) for path in first}

    assert (
        assemble_postprocess_inputs(
            root, resume=True, expected_sample_count=len(SAMPLE_IDS)
        )
        == first
    )
    assert {path: artifact_record(path) for path in first} == records
    with pytest.raises(FileExistsError, match="pass --resume"):
        assemble_postprocess_inputs(root, expected_sample_count=len(SAMPLE_IDS))


@pytest.mark.parametrize("target", ["producer", "normalised", "declaration"])
def test_resume_rejects_source_or_normalised_frame_tamper(
    tmp_path: Path, target: str
) -> None:
    root = _run(tmp_path)
    _complete_source_dag(root)
    assemble_postprocess_inputs(root, expected_sample_count=len(SAMPLE_IDS))
    if target == "declaration":
        path = root / "07_candidate_tables/raw/g1/gt_oracle/source_declaration.json"
        declaration = json.loads(path.read_text(encoding="utf-8"))
        declaration["candidate_count"] = 999
        atomic_json(path, declaration)
    else:
        path = (
            root / "06_gtmask_predictions/g1/gt_oracle/per_candidate.parquet"
            if target == "producer"
            else root / "07_candidate_tables/raw/g1/gt_oracle/candidates.parquet"
        )
        frame = pd.read_parquet(path)
        frame.loc[0, "cx_px"] = 999.0
        atomic_parquet(frame, path)

    with pytest.raises(PostprocessInputAssemblyError) as captured:
        assemble_postprocess_inputs(
            root, resume=True, expected_sample_count=len(SAMPLE_IDS)
        )
    assert any(
        issue.get("route") == "G1"
        and issue["code"] == "GT_ROUTE_FRAME_PRODUCER_REQUIRED"
        for issue in captured.value.issues
    )


def test_cli_exposes_only_run_dir_and_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    root = tmp_path / "runs/fair_gtmask_counterfactual_g1_c1_d1_cli"
    inputs = root / "07_candidate_tables/POSTPROCESS_INPUTS.json"
    assembly = root / ASSEMBLY_MANIFEST_RELATIVE_PATH
    monkeypatch.setattr(
        tool, "assemble_postprocess_inputs", lambda *args, **kwargs: (inputs, assembly)
    )
    assert vars(tool.parse_args(["--run-dir", str(root), "--resume"])) == {
        "run_dir": root,
        "resume": True,
    }
    assert tool.main(["--run-dir", str(root), "--resume"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result == {
        "status": "READY",
        "postprocess_inputs": str(inputs),
        "assembly_manifest": str(assembly),
    }
    with pytest.raises(SystemExit):
        tool.parse_args(["--run-dir", str(root), "--candidate-path", "/tmp/fake"])
