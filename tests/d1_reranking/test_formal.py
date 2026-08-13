from __future__ import annotations

import json
from pathlib import Path
import shutil

import pandas as pd
import pytest
from PIL import Image

from d1_reranking import formal
from unified_reranking.hashing import canonical_sha256, sha256_file


def _write_json(path: Path, value: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")
    return path


def _write_content(path: Path, value: dict[str, object]) -> Path:
    payload = dict(value)
    payload["content_sha256"] = canonical_sha256(payload)
    return _write_json(path, payload)


def _manifest(path: Path, common: Path, **extra: object) -> Path:
    return _write_content(
        path,
        {
            "schema_version": 1,
            "status": "COMPLETE",
            "candidate_test_labels_read": False,
            "sources": {"common": formal._record(common)},
            **extra,
        },
    )


def _scores(rows: list[tuple[str, str, str, float, int]]) -> pd.DataFrame:
    return pd.DataFrame(
        rows,
        columns=["source_route", "sample_id", "candidate_id", "score", "rank"],
    )


def _decisions(route: str, candidate_id: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sample_id": ["sample-0", "sample-1"],
            "selected_source_route": [route, ""],
            "selected_candidate_id": [candidate_id, ""],
        }
    )


def _synthetic_run(root: Path, *, invalid_ground_truth: bool = False) -> Path:
    common = root / "common.bin"
    common.write_bytes(b"immutable synthetic artifact")
    _write_json(
        root / "manifest.json",
        {
            "schema_version": 1,
            "status": "PRELOCK_LABEL_FREE",
            "formal_test_execution_count": 0,
            "test_label_state": "LOCKED_PREVALIDATION",
        },
    )
    _write_json(
        root / "pipeline_status.json",
        {
            "schema_version": 1,
            "status": "PRELOCK_LABEL_FREE",
            "first_incomplete_stage": "P14_FORMAL_LOCK",
            "formal_test_executed": False,
            "test_candidate_labels_read": False,
            "formal_test_execution_count": 0,
        },
    )
    source_authority_unsigned = {
        "schema_version": 1,
        "status": "COMPLETE",
        "formal_test_execution_count": 1,
        "inventory": {},
    }
    source_authority = _write_json(
        root / "source_authority.json",
        {
            **source_authority_unsigned,
            "self_sha256": canonical_sha256(source_authority_unsigned),
        },
    )
    evaluator = root / "canonical_evaluator.py"
    evaluator.write_text(
        """\
IOU_THRESHOLD = 0.25
ANGLE_THRESHOLD_DEG = 30.0

class CanonicalGrasp:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

def gt_from_corners(value):
    return value

def corners(value):
    return value

def evaluate_candidate(candidate, ground_truth):
    success = candidate.cx_px < 50.0
    return {
        "success": success,
        "pairwise": [{
            "iou": 1.0 if success else 0.0,
            "angle_error_deg": 0.0,
            "gt_index": 0,
        }] if ground_truth else [],
    }
""",
        encoding="utf-8",
    )
    formal.assemble_statistics_config(
        root,
        bootstrap_seed=20260808,
        resume=False,
    )
    statistics = root / formal.STATISTICS_CONFIG_RELATIVE_PATH
    paired = root / "paired_test.parquet"
    pd.DataFrame({"sample_id": ["sample-0", "sample-1"]}).to_parquet(
        paired, index=False
    )
    candidate_values = {
        "d1-0": (5.0, 5.0, 0.0, 10.0, 10.0, 0.9, 1),
        "d1-1": (105.0, 105.0, 0.0, 10.0, 10.0, 0.8, 2),
        "d1-2": (6.0, 5.0, 0.0, 10.0, 10.0, 0.7, 3),
        "d1-3": (106.0, 105.0, 0.0, 10.0, 10.0, 0.6, 4),
        "d1-4": (107.0, 105.0, 0.0, 10.0, 10.0, 0.5, 5),
        "d1-5": (108.0, 105.0, 0.0, 10.0, 10.0, 0.4, 6),
    }
    pool_rows = {
        "top5": ["d1-0", "d1-1", "d1-2", "d1-3", "d1-4"],
        "top10": [
            "d1-0",
            "d1-1",
            "d1-2",
            "d1-3",
            "d1-4",
            "d1-5",
        ],
        "allnms": [
            "d1-0",
            "d1-1",
            "d1-2",
            "d1-3",
            "d1-4",
            "d1-5",
        ],
    }
    pool_paths: dict[str, Path] = {}
    for pool, candidate_ids in pool_rows.items():
        path = root / f"d1_{pool}.parquet"
        rows = []
        for candidate_id in candidate_ids:
            cx, cy, theta, width, height, native_score, native_rank = candidate_values[
                candidate_id
            ]
            geometry = canonical_sha256(
                [
                    "D1",
                    "sample-0",
                    candidate_id,
                    native_rank,
                    cx,
                    cy,
                    theta,
                    width,
                    height,
                ]
            )
            rows.append(
                {
                    "route": "D1",
                    "sample_id": "sample-0",
                    "candidate_id": candidate_id,
                    "candidate_geometry_sha256": geometry,
                    "native_rank": native_rank,
                    "native_score": native_score,
                    "cx_px": cx,
                    "cy_px": cy,
                    "theta_deg": theta,
                    "width_px": width,
                    "height_px": height,
                }
            )
        pd.DataFrame(rows).to_parquet(path, index=False)
        pool_paths[pool] = path
    candidate_manifest = _write_content(
        root / "candidate_manifest.json",
        {
            "schema_version": 1,
            "status": "COMPLETE",
            "candidate_test_labels_read": False,
            "artifacts": {
                pool: formal._record(path) for pool, path in pool_paths.items()
            },
        },
    )

    ground_truth_path = root / "raw_test_ground_truth.parquet"
    ground_truth = pd.DataFrame(
        {
            "sample_id": ["sample-0", "sample-1"],
            "gt_grasp_rectangles": [
                [[[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]]],
                [[[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]]],
            ],
        }
    )
    if invalid_ground_truth:
        ground_truth = ground_truth.drop(columns=["gt_grasp_rectangles"])
    ground_truth.to_parquet(ground_truth_path, index=False)
    visual_dir = root / "visual_assets"
    visual_dir.mkdir(parents=True, exist_ok=True)
    visual_rows = []
    for index, sample_id in enumerate(("sample-0", "sample-1")):
        rgb_path = visual_dir / f"{sample_id}_rgb.png"
        gt_mask_path = visual_dir / f"{sample_id}_gt.png"
        predicted_mask_path = visual_dir / f"{sample_id}_predicted.png"
        probability_path = visual_dir / f"{sample_id}_probability.png"
        depth_path = visual_dir / f"{sample_id}_depth.png"
        Image.new("RGB", (128, 128), (40 + 20 * index, 80, 120)).save(rgb_path)
        Image.new("L", (128, 128), 255).save(gt_mask_path)
        Image.new("L", (128, 128), 0 if index == 0 else 192).save(predicted_mask_path)
        Image.new("L", (128, 128), 96 + 20 * index).save(probability_path)
        Image.new("I;16", (128, 128), 1000 + 100 * index).save(depth_path)
        visual_rows.append(
            {
                "sample_id": sample_id,
                "language_prompt": f"grasp the synthetic target {index}",
                "rgb_path": str(rgb_path.resolve()),
                "image_sha256": sha256_file(rgb_path),
                "gt_mask_path": str(gt_mask_path.resolve()),
                "gt_mask_sha256": sha256_file(gt_mask_path),
                "predicted_hifics_mask_path": str(predicted_mask_path.resolve()),
                "predicted_hifics_mask_sha256": sha256_file(predicted_mask_path),
                "hifics_probability_path": str(probability_path.resolve()),
                "hifics_probability_sha256": sha256_file(probability_path),
                "aligned_depth_path": str(depth_path.resolve()),
                "aligned_depth_sha256": sha256_file(depth_path),
                "gt_grasp_list_json": json.dumps(
                    [[[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]]]
                ),
            }
        )
    visual_ground_truth_path = root / "opaque_visual_ground_truth.parquet"
    pd.DataFrame(visual_rows).to_parquet(visual_ground_truth_path, index=False)
    source_child = root / "source_child.bin"
    source_child.write_bytes(b"synthetic immutable source child")
    closure_path = _write_content(
        root / "00_audit" / "source_closure.json",
        {
            "schema_version": 1,
            "status": "PASS",
            "canonical_inputs": {
                "test": {
                    "opaque_ground_truth": formal._record(ground_truth_path),
                    "opaque_visual_ground_truth": formal._record(
                        visual_ground_truth_path
                    ),
                }
            },
            "verified_artifacts": [formal._record(source_child)],
        },
    )
    prelock_path = _write_content(
        root / "08_lock" / "PRELOCK_READINESS.json",
        {
            "schema_version": 1,
            "status": "PASS",
            "candidate_test_labels_read": False,
            "formal_test_execution_count": 0,
            "formal_test_lock_created": False,
            "sources": {
                "common": formal._record(common),
                "source_closure": formal._record(closure_path),
                "source_run_authority": formal._record(source_authority),
            },
            "artifacts": {"declaration": formal._record(common)},
        },
    )

    component_paths: dict[str, Path] = {
        "prelock_readiness": prelock_path,
        "source_run_authority": source_authority,
        "paired_test_manifest": paired,
        "canonical_evaluator": evaluator,
        "statistics_config": statistics,
        "d1_test_candidate_manifest": candidate_manifest,
        **{f"d1_{pool}_candidates": path for pool, path in pool_paths.items()},
    }
    visual_features = root / "components" / "visual_features.parquet"
    visual_features.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "sample_id": ["sample-0"] * 5,
            "candidate_id": [f"d1-{index}" for index in range(5)],
            "calibrated_native_probability": [0.91, 0.75, 0.65, 0.55, 0.45],
        }
    ).to_parquet(visual_features, index=False)
    component_paths["d1_top5_feature_manifest"] = _write_content(
        root / "components" / "d1_top5_feature_manifest.json",
        {
            "status": "COMPLETE",
            "candidate_test_labels_read": False,
            "artifacts": {"candidate_features": formal._record(visual_features)},
        },
    )
    gate_diagnostics = root / "components" / "gate_diagnostics.parquet"
    pd.DataFrame(
        {
            "sample_id": ["sample-0", "sample-1"],
            "utility": [0.2, -0.1],
        }
    ).to_parquet(gate_diagnostics, index=False)
    component_paths["d1_primary_gate_manifest"] = _write_content(
        root / "components" / "d1_primary_gate_manifest.json",
        {
            "status": "COMPLETE",
            "candidate_test_labels_read": False,
            "artifacts": {"decisions": formal._record(gate_diagnostics)},
        },
    )
    component_paths["d1_primary_ranker_manifest"] = _write_content(
        root / "components" / "d1_primary_ranker_manifest.json",
        {
            "status": "COMPLETE",
            "candidate_test_labels_read": False,
            "selected_method": "R3",
            "artifacts": {"placeholder": formal._record(common)},
        },
    )
    for name in formal.REQUIRED_COMPONENTS.difference(component_paths):
        component_paths[name] = _manifest(root / "components" / f"{name}.json", common)

    systems = []
    for name, expected in formal.FORMAL_SYSTEMS.items():
        if name.startswith("d1_top5"):
            score_rows = [
                ("D1", "sample-0", "d1-0", 0.9, 1),
                ("D1", "sample-0", "d1-1", 0.1, 2),
                ("D1", "sample-0", "d1-2", 0.08, 3),
                ("D1", "sample-0", "d1-3", 0.06, 4),
                ("D1", "sample-0", "d1-4", 0.04, 5),
            ]
            decisions = _decisions("D1", "d1-0")
        elif name == "d1_top10_locked":
            score_rows = [
                ("D1", "sample-0", "d1-0", 0.9, 1),
                ("D1", "sample-0", "d1-1", 0.2, 2),
                ("D1", "sample-0", "d1-2", 0.1, 3),
                ("D1", "sample-0", "d1-3", 0.08, 4),
                ("D1", "sample-0", "d1-4", 0.06, 5),
                ("D1", "sample-0", "d1-5", 0.04, 6),
            ]
            decisions = _decisions("D1", "d1-0")
        elif name == "d1_allnms_locked":
            score_rows = [
                ("D1", "sample-0", "d1-0", 0.9, 1),
                ("D1", "sample-0", "d1-1", 0.3, 2),
                ("D1", "sample-0", "d1-2", 0.2, 3),
                ("D1", "sample-0", "d1-3", 0.1, 4),
                ("D1", "sample-0", "d1-4", 0.08, 5),
                ("D1", "sample-0", "d1-5", 0.06, 6),
            ]
            decisions = _decisions("D1", "d1-0")
        elif name == "three_route_crog_default_router_reference":
            score_rows = [
                ("CROG", "sample-0", "crog-0", 0.9, 1),
                ("G1", "sample-0", "g1-0", 0.8, 1),
                ("C1", "sample-0", "c1-0", 0.7, 1),
            ]
            decisions = _decisions("G1", "g1-0")
        elif name == "top15_union_reference":
            score_rows = [
                ("CROG", "sample-0", "crog-0", 0.9, 1),
                ("G1", "sample-0", "g1-0", 0.8, 2),
                ("C1", "sample-0", "c1-0", 0.7, 3),
            ]
            decisions = _decisions("G1", "g1-0")
        elif name == "four_route_crog_default_router":
            score_rows = [
                ("CROG", "sample-0", "crog-0", 0.9, 1),
                ("G1", "sample-0", "g1-0", 0.8, 1),
                ("C1", "sample-0", "c1-0", 0.7, 1),
                ("D1", "sample-0", "d1-0", 0.6, 1),
            ]
            decisions = _decisions("G1", "g1-0")
        else:
            score_rows = [
                ("CROG", "sample-0", "crog-0", 0.9, 1),
                ("G1", "sample-0", "g1-0", 0.8, 2),
                ("C1", "sample-0", "c1-0", 0.7, 3),
                ("D1", "sample-0", "d1-0", 0.6, 4),
            ]
            decisions = _decisions("G1", "g1-0")
        score_path = root / "systems" / name / "scores.parquet"
        universe_path = root / "systems" / name / "candidate_universe.parquet"
        decision_path = root / "systems" / name / "decisions.parquet"
        score_path.parent.mkdir(parents=True, exist_ok=True)
        _scores(score_rows).to_parquet(score_path, index=False)
        universe_rows = []
        for route, sample_id, candidate_id, _score, rank in score_rows:
            if route == "D1":
                cx, cy, theta, width, height, native_score, native_rank = (
                    candidate_values[candidate_id]
                )
            else:
                cx = 5.0 if route == "G1" else 105.0
                cy, theta, width, height = 5.0, 0.0, 10.0, 10.0
                native_score, native_rank = float(_score), 1
            geometry = canonical_sha256(
                [
                    route,
                    sample_id,
                    candidate_id,
                    native_rank,
                    cx,
                    cy,
                    theta,
                    width,
                    height,
                ]
            )
            universe_rows.append(
                {
                    "source_route": route,
                    "sample_id": sample_id,
                    "candidate_id": candidate_id,
                    "candidate_geometry_sha256": geometry,
                    "native_rank": native_rank,
                    "native_score": native_score,
                    "cx_px": cx,
                    "cy_px": cy,
                    "theta_deg": theta,
                    "width_px": width,
                    "height_px": height,
                }
            )
        pd.DataFrame(universe_rows).to_parquet(universe_path, index=False)
        decisions.to_parquet(decision_path, index=False)
        system_manifest = _write_content(
            root / "systems" / name / "manifest.json",
            {
                "schema_version": 1,
                "status": "COMPLETE",
                "candidate_test_labels_read": False,
                "sources": {"common": formal._record(common)},
                "artifacts": {
                    "candidate_universe": formal._record(universe_path),
                    "candidate_scores": formal._record(score_path),
                    "per_sample_decisions": formal._record(decision_path),
                },
            },
        )
        systems.append(
            {
                "name": name,
                "role": "PRIMARY" if name.startswith("d1_top5") else "SECONDARY",
                "kind": expected["kind"],
                "pool": expected["pool"],
                "routes": expected["routes"],
                "pool_component": expected["pool_component"],
                "feeds_selection": False,
                "locked_inputs": sorted(expected["inputs"]),
                "manifest": formal._record(system_manifest),
                "candidate_universe": formal._record(universe_path),
                "candidate_scores": formal._record(score_path),
                "per_sample_decisions": formal._record(decision_path),
            }
        )
    components = {name: formal._record(path) for name, path in component_paths.items()}
    sources = {
        "components": components,
        "source_closure": formal._record(closure_path),
        "raw_test_ground_truth": formal._record(ground_truth_path),
        "formal_input_manifests": {
            system["name"]: system["manifest"] for system in systems
        },
        "tools": [],
    }
    plan = _write_content(
        root / "configs" / "d1_formal_evaluation_plan.json",
        {
            "schema_version": 1,
            "status": "LOCK_READY",
            "candidate_test_labels_read": False,
            "raw_test_ground_truth_opened": False,
            "selection_feedback_allowed": False,
            "raw_test_ground_truth": formal._record(ground_truth_path),
            "ground_truth_evaluator_sha256": formal._record(evaluator)["sha256"],
            "components": components,
            "systems": systems,
            "sources": sources,
            "source_signature_sha256": canonical_sha256(sources),
        },
    )
    return plan


def _materialize_plan_writer_inputs(root: Path, plan_path: Path) -> None:
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    components = plan["components"]
    pool_records = {}
    for pool in ("top5", "top10", "allnms"):
        destination = root / "02_candidates" / f"d1_{pool}_candidates.parquet"
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(components[f"d1_{pool}_candidates"]["path"], destination)
        pool_records[pool] = formal._record(destination)
    candidate_manifest = _write_content(
        root / formal.FIXED_COMPONENT_PATHS["d1_test_candidate_manifest"],
        {
            "schema_version": 1,
            "status": "COMPLETE",
            "candidate_test_labels_read": False,
            "artifacts": pool_records,
        },
    )
    for name, relative in formal.FIXED_COMPONENT_PATHS.items():
        if name in {
            "prelock_readiness",
            "statistics_config",
            "d1_test_candidate_manifest",
        }:
            continue
        source = Path(components[name]["path"])
        destination = root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
    assert candidate_manifest.is_file()
    for system in plan["systems"]:
        destination = root / formal.FORMAL_INPUT_ROOT / system["name"]
        destination.mkdir(parents=True, exist_ok=True)
        records = {}
        for key, filename in (
            ("candidate_universe", "candidate_universe.parquet"),
            ("candidate_scores", "candidate_scores.parquet"),
            ("per_sample_decisions", "per_sample_decisions.parquet"),
        ):
            target = destination / filename
            shutil.copyfile(system[key]["path"], target)
            records[key] = formal._record(target)
        _write_content(
            destination / "manifest.json",
            {
                "schema_version": 1,
                "status": "COMPLETE",
                "candidate_test_labels_read": False,
                "sources": {"common": formal._record(root / "common.bin")},
                "artifacts": records,
            },
        )
    plan_path.unlink()


@pytest.fixture
def stable_repository_state(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        formal,
        "_repository_state",
        lambda: {
            "git_commit": "a" * 40,
            "dirty": False,
            "status_sha256": "b" * 64,
            "tracked_diff_sha256": "c" * 64,
        },
    )


def test_formal_lock_and_transaction_are_exactly_once(
    tmp_path: Path, stable_repository_state: None
) -> None:
    del stable_repository_state
    plan = _synthetic_run(tmp_path)
    lock = formal.create_formal_lock(
        tmp_path, evaluation_plan_path=plan, code_paths=(Path(formal.__file__),)
    )
    assert lock["status"] == "LOCKED"
    assert (
        formal.verify_formal_lock(tmp_path)["inventory_count"]
        == lock["inventory_count"]
    )
    assert (tmp_path / "08_lock" / formal.LOCK_DIGEST_NAME).is_file()

    result = formal.run_formal_test_once(tmp_path)
    assert result["formal_test_execution_count"] == 1
    execution = formal._load_execution(tmp_path)
    assert execution["status"] == "COMPLETE"
    assert execution["execution_count"] == 1
    bundle = pd.read_parquet(tmp_path / "09_formal_test" / formal.FORMAL_BUNDLE_NAME)
    assert set(bundle["system_name"]) == set(formal.FORMAL_SYSTEMS)
    assert bundle.groupby("system_name")["no_output"].any().all()
    assert not result["selection_feedback_used"]
    with pytest.raises(PermissionError, match="already"):
        formal.run_formal_test_once(tmp_path)


def test_label_access_denied_before_claim(
    tmp_path: Path, stable_repository_state: None
) -> None:
    del stable_repository_state
    plan = _synthetic_run(tmp_path)
    formal.create_formal_lock(tmp_path, evaluation_plan_path=plan)
    with pytest.raises(PermissionError, match="complete exclusive claim"):
        formal.assert_formal_label_access_authorized(tmp_path)
    value = json.loads(plan.read_text(encoding="utf-8"))
    ground_truth = value["raw_test_ground_truth"]
    with pytest.raises(PermissionError, match="complete exclusive claim"):
        formal._read_raw_test_ground_truth_once(
            tmp_path, Path(ground_truth["path"]), ground_truth["sha256"]
        )
    assert not (tmp_path / "09_formal_test" / formal.CLAIM_SENTINEL_NAME).exists()


def test_locked_artifact_tamper_fails_before_claim(
    tmp_path: Path, stable_repository_state: None
) -> None:
    del stable_repository_state
    plan_path = _synthetic_run(tmp_path)
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    score_path = Path(plan["systems"][0]["candidate_scores"]["path"])
    formal.create_formal_lock(tmp_path, evaluation_plan_path=plan_path)
    scores = pd.read_parquet(score_path)
    scores.loc[0, "score"] = 123.0
    scores.to_parquet(score_path, index=False)
    with pytest.raises(PermissionError, match="artifact drift"):
        formal.run_formal_test_once(tmp_path)
    assert not (tmp_path / "09_formal_test" / formal.CLAIM_SENTINEL_NAME).exists()
    assert (
        json.loads((tmp_path / "manifest.json").read_text())[
            "formal_test_execution_count"
        ]
        == 0
    )


def test_failure_after_label_open_consumes_claim_and_cannot_replay(
    tmp_path: Path, stable_repository_state: None
) -> None:
    del stable_repository_state
    plan = _synthetic_run(tmp_path, invalid_ground_truth=True)
    formal.create_formal_lock(tmp_path, evaluation_plan_path=plan)
    with pytest.raises(RuntimeError, match="ground truth misses columns"):
        formal.run_formal_test_once(tmp_path)
    execution = formal._load_execution(tmp_path)
    assert execution["status"] == "FAILED"
    assert execution["execution_count"] == 1
    assert execution["execution_consumed"] is True
    assert (
        json.loads((tmp_path / "manifest.json").read_text())[
            "formal_test_execution_count"
        ]
        == 1
    )
    with pytest.raises(PermissionError, match="already"):
        formal.run_formal_test_once(tmp_path)


def test_missing_required_system_or_component_fails_closed(
    tmp_path: Path, stable_repository_state: None
) -> None:
    del stable_repository_state
    plan_path = _synthetic_run(tmp_path)
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    plan["systems"] = [
        system for system in plan["systems"] if system["name"] != "top20_union"
    ]
    plan.pop("content_sha256")
    _write_content(plan_path, plan)
    with pytest.raises(RuntimeError, match="system inventory differs"):
        formal.create_formal_lock(tmp_path, evaluation_plan_path=plan_path)
    assert not (tmp_path / "08_lock" / formal.LOCK_NAME).exists()


def test_statistics_writer_is_immutable_and_self_hashed(tmp_path: Path) -> None:
    _write_json(
        tmp_path / "manifest.json",
        {"formal_test_execution_count": 0, "status": "PRELOCK_LABEL_FREE"},
    )
    value = formal.assemble_statistics_config(
        tmp_path, bootstrap_seed=20260808, resume=False
    )
    assert value["status"] == "LOCKED"
    assert value["formal_systems"] == sorted(formal.FORMAL_SYSTEMS)
    assert (
        formal.assemble_statistics_config(
            tmp_path, bootstrap_seed=20260808, resume=True
        )
        == value
    )
    with pytest.raises(RuntimeError, match="exists and differs"):
        formal.assemble_statistics_config(tmp_path, bootstrap_seed=7, resume=True)


def test_production_plan_writer_feeds_lock(
    tmp_path: Path, stable_repository_state: None
) -> None:
    del stable_repository_state
    manual_plan = _synthetic_run(tmp_path)
    _materialize_plan_writer_inputs(tmp_path, manual_plan)
    plan = formal.assemble_formal_evaluation_plan(
        tmp_path,
        resume=False,
        tool_paths=(Path(formal.__file__),),
    )
    assert plan["raw_test_ground_truth_opened"] is False
    assert "candidate_labels" not in plan
    lock = formal.create_formal_lock(
        tmp_path,
        evaluation_plan_path=tmp_path / formal.FORMAL_PLAN_RELATIVE_PATH,
    )
    assert lock["status"] == "LOCKED"


def test_formal_plan_resume_archives_prelock_code_drift(
    tmp_path: Path, stable_repository_state: None
) -> None:
    del stable_repository_state
    manual_plan = _synthetic_run(tmp_path)
    _materialize_plan_writer_inputs(tmp_path, manual_plan)
    tool = tmp_path / "formal_tool.py"
    tool.write_text("VERSION = 1\n", encoding="utf-8")
    first = formal.assemble_formal_evaluation_plan(
        tmp_path, resume=False, tool_paths=(tool,)
    )
    tool.write_text("VERSION = 2\n", encoding="utf-8")
    second = formal.assemble_formal_evaluation_plan(
        tmp_path, resume=True, tool_paths=(tool,)
    )
    assert second["content_sha256"] != first["content_sha256"]
    archive = (
        tmp_path
        / "configs"
        / "d1_formal_evaluation_plan_history"
        / f"{first['content_sha256']}.json"
    )
    assert json.loads(archive.read_text(encoding="utf-8")) == first
    assert formal.create_formal_lock(
        tmp_path,
        evaluation_plan_path=tmp_path / formal.FORMAL_PLAN_RELATIVE_PATH,
    )["status"] == "LOCKED"


def _bind_access_log_to_prelock_and_plan(
    root: Path, plan_path: Path, initial: bytes
) -> Path:
    access_path = root / "09_formal_test" / "test_access.log"
    access_path.parent.mkdir(parents=True, exist_ok=True)
    access_path.write_bytes(initial)
    prelock_path = root / "08_lock" / "PRELOCK_READINESS.json"
    prelock = json.loads(prelock_path.read_text(encoding="utf-8"))
    prelock.pop("content_sha256")
    access_record = formal._record(access_path)
    prelock["sources"]["access_log"] = access_record
    prelock["sources"]["k_formal_inputs"] = {
        "test_access_log": {
            "path": access_record["path"],
            "sha256": access_record["sha256"],
        }
    }
    _write_content(prelock_path, prelock)
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    plan.pop("content_sha256")
    plan["components"]["prelock_readiness"] = formal._record(prelock_path)
    _write_content(plan_path, plan)
    return access_path


def test_formal_lock_accepts_append_only_access_log_after_prelock(
    tmp_path: Path, stable_repository_state: None
) -> None:
    del stable_repository_state
    plan_path = _synthetic_run(tmp_path)
    access_path = _bind_access_log_to_prelock_and_plan(
        tmp_path,
        plan_path,
        b'{"event":"prelock_label_free_test_stage"}\n',
    )
    with access_path.open("ab") as stream:
        stream.write(b'{"event":"d1_prelock_raw_test_ground_truth_hash_only"}\n')
    assert formal.create_formal_lock(
        tmp_path, evaluation_plan_path=plan_path
    )["status"] == "LOCKED"


def test_formal_lock_rejects_rewritten_access_log_prefix(
    tmp_path: Path, stable_repository_state: None
) -> None:
    del stable_repository_state
    plan_path = _synthetic_run(tmp_path)
    access_path = _bind_access_log_to_prelock_and_plan(
        tmp_path,
        plan_path,
        b'{"event":"prelock_label_free_test_stage"}\n',
    )
    access_path.write_bytes(
        b'{"event":"forged_label_free_test_stage"}\n'
        b'{"event":"d1_prelock_raw_test_ground_truth_hash_only"}\n'
    )
    with pytest.raises(RuntimeError, match="locked prefix SHA-256 mismatch"):
        formal.create_formal_lock(tmp_path, evaluation_plan_path=plan_path)
    assert not (tmp_path / "08_lock" / formal.LOCK_NAME).exists()


def test_failed_claim_manifest_transition_is_consumed(
    tmp_path: Path,
    stable_repository_state: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    del stable_repository_state
    plan = _synthetic_run(tmp_path)
    formal.create_formal_lock(tmp_path, evaluation_plan_path=plan)
    original = formal._write_run_manifest
    calls = 0

    def fail_first(root: Path, value: object) -> None:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise OSError("synthetic manifest transition fault")
        original(root, value)  # type: ignore[arg-type]

    monkeypatch.setattr(formal, "_write_run_manifest", fail_first)
    with pytest.raises(PermissionError, match="consumed"):
        formal.run_formal_test_once(tmp_path)
    execution = formal._load_execution(tmp_path)
    assert execution["status"] == "FAILED"
    assert execution["execution_consumed"] is True
    with pytest.raises(PermissionError, match="already"):
        formal.run_formal_test_once(tmp_path)


def test_no_output_is_defined_by_empty_decision_not_score_presence(
    tmp_path: Path,
) -> None:
    plan_path = _synthetic_run(tmp_path)
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    system = dict(plan["systems"][0])
    decision_path = Path(system["per_sample_decisions"]["path"])
    decisions = pd.read_parquet(decision_path)
    decisions.loc[decisions["sample_id"].eq("sample-0"), "selected_source_route"] = ""
    decisions.loc[decisions["sample_id"].eq("sample-0"), "selected_candidate_id"] = ""
    decisions.to_parquet(decision_path, index=False)
    system["per_sample_decisions"] = formal._record(decision_path)
    scores = pd.read_parquet(system["candidate_scores"]["path"])
    outcomes = scores.loc[:, ["source_route", "sample_id", "candidate_id"]].copy()
    outcomes["candidate_success"] = True
    outcomes["best_same_gt_iou"] = 1.0
    outcomes["best_same_gt_angle_error_deg"] = 0.0
    outcomes["matched_gt_index"] = 0
    outcomes["jacquard_margin"] = 1.0
    bundle, metrics = formal._evaluate_system(
        system, outcomes, ["sample-0", "sample-1"]
    )
    sample = bundle.loc[bundle["sample_id"].eq("sample-0")]
    assert sample["row_kind"].eq("candidate").any()
    assert sample["no_output"].any()
    assert metrics["no_output_samples"] == 2
