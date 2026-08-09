from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tools.unified_reranking.prepare_test_bridge_bundle import run as prepare_bridge
from unified_reranking.hashing import canonical_sha256, sha256_file
from unified_reranking.test_bridge import validate_label_free_test_bridge_manifest


def _json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sources(root: Path) -> tuple[Path, Path, Path]:
    run = root / "run"
    historical = root / "historical"
    modular = root / "modular"
    for directory in (
        run / "01_manifests",
        run / "02_candidates",
        run / "03_features/common/g1_test",
        run / "03_features/common/c1_test",
        run / "configs",
        historical / "data",
        historical / "audit",
        modular / "manifests",
    ):
        directory.mkdir(parents=True, exist_ok=True)
    samples = pd.DataFrame(
        {
            "sample_id": ["s0", "s1", "s2"],
            "scene_id": ["z0", "z1", "z2"],
            "frame_id": ["f0", "f1", "f2"],
        }
    )
    samples.to_parquet(run / "01_manifests/paired_test.parquet", index=False)
    evaluator = run / "configs/canonical_evaluator.py"
    evaluator.write_text(
        """import numpy as np
IOU_THRESHOLD = 0.25
ANGLE_THRESHOLD_DEG = 30.0
class CanonicalGrasp:
    def __init__(self, **values): self.__dict__.update(values)
def gt_from_corners(value): return np.asarray(value, dtype=float)
def evaluate_candidate(candidate, ground_truth):
    success = candidate.theta_deg < 0.0
    return {"success": success, "pairwise": [{"gt_index": 0, "iou": 1.0 if success else 0.0, "angle_error_deg": 0.0}]}
""",
        encoding="utf-8",
    )
    inventory: dict[str, object] = {}
    for route in ("g1", "c1"):
        fair_rows = []
        for sample_id in ("s0", "s1"):
            for rank, raw_id, theta in ((1, "same", -10.0), (2, "other", 10.0)):
                geometry = [float(rank), 2.0, theta, 4.0, 2.0]
                fair_rows.append(
                    {
                        "sample_id": sample_id,
                        "candidate_id": raw_id,
                        "native_rank": rank,
                        "native_score": 1.0 / rank,
                        "cx_px": geometry[0],
                        "cy_px": geometry[1],
                        "theta_deg": geometry[2],
                        "width_px": geometry[3],
                        "height_px": geometry[4],
                        "candidate_geometry_sha256": canonical_sha256(
                            [route.upper(), sample_id, raw_id, rank, *geometry]
                        ),
                    }
                )
        fair = pd.DataFrame(fair_rows)
        fair.to_parquet(
            run / f"02_candidates/{route}_test_top5.parquet", index=False
        )
        features = fair[["sample_id", "candidate_id", "native_rank"]].copy()
        features["native_score_raw"] = fair["native_score"]
        features["p_center"] = [0.8, 1.2, 0.8, 1.2]
        features.to_parquet(
            run / f"03_features/common/{route}_test/candidate_features.parquet",
            index=False,
        )
        historical_rows = []
        for rank, raw_id, theta in ((1, "same", -12.0), (2, "other", 12.0)):
            quality = 0.9 / rank
            support = 0.75
            historical_rows.append(
                {
                    "sample_id": "s0",
                    "scene_id": "z0",
                    "split": "test",
                    "backend": route.upper(),
                    "source_candidate_id": f"source-{raw_id}",
                    "stable_candidate_id": raw_id,
                    "original_rank": rank,
                    "original_score": quality * support,
                    "center_x": float(rank) + 0.25,
                    "center_y": 2.0,
                    "angle_deg": theta,
                    "width_px": 4.0,
                    "height_px": 2.0,
                    "raw_network_quality": quality,
                    "stored_center_mask_support": support,
                    "stored_jaw_mask_support": 1.0,
                    "source_row": rank,
                    "source_column": rank,
                    "candidate_identity_sha256": canonical_sha256(
                        [route, raw_id, rank]
                    ),
                }
            )
        historical_path = (
            historical / f"data/frozen_{route}_test_top5_candidates.parquet"
        )
        historical_frame = pd.DataFrame(historical_rows)
        historical_frame.to_parquet(historical_path, index=False)
        historical_frame.to_parquet(
            historical / f"data/frozen_{route}_test_allnms_candidates.parquet",
            index=False,
        )
        inventory[f"{route.upper()}_test"] = {
            "top5_path": str(historical_path.resolve()),
            "top5_artifact_sha256": sha256_file(historical_path),
        }
    inventory_path = historical / "audit/frozen_pool_inventory.json"
    _json(inventory_path, inventory)
    formal_lock_path = historical / "08_lock/FORMAL_TEST_LOCK.json"
    _json(
        formal_lock_path,
        {
            "status": "LOCKED",
            "audit_artifacts": [
                {"path": str(inventory_path.resolve()), "sha256": sha256_file(inventory_path)}
            ],
            "candidate_artifacts": [
                {
                    "path": str(
                        (
                            historical
                            / f"data/frozen_{route}_test_allnms_candidates.parquet"
                        ).resolve()
                    ),
                    "sha256": sha256_file(
                        historical / f"data/frozen_{route}_test_allnms_candidates.parquet"
                    ),
                }
                for route in ("g1", "c1")
            ],
        },
    )
    authority_paths = [
        inventory_path,
        formal_lock_path,
        *[
            historical / f"data/frozen_{route}_test_{pool}_candidates.parquet"
            for route in ("g1", "c1")
            for pool in ("top5", "allnms")
        ],
    ]
    run_sha_path = historical / "RUN_SHA256_MANIFEST.txt"
    run_sha_path.write_text(
        "".join(
            f"{sha256_file(path)}  {path.relative_to(historical).as_posix()}\n"
            for path in sorted(authority_paths)
        ),
        encoding="utf-8",
    )
    (historical / "RUN_LOCK_SHA256.txt").write_text(
        sha256_file(run_sha_path) + "\n", encoding="utf-8"
    )
    ground_truth = modular / "manifests/test_labels.parquet"
    pd.DataFrame(
        {
            "sample_id": samples["sample_id"],
            "gt_grasp_rectangles": [
                [[[0.0, 0.0], [2.0, 0.0], [2.0, 2.0], [0.0, 2.0]]]
                for _ in range(3)
            ],
        }
    ).to_parquet(ground_truth, index=False)
    experiment_lock = {
        "schema_version": 1,
        "lock_status": "LOCKED",
        "effective": True,
        "run_id": "synthetic-modular",
        "run_dir": str(modular.resolve()),
        "lock_relative_path": "manifests/experiment_lock.json",
        "marker_relative_path": ".EXPERIMENT_LOCKED",
        "artifacts": {
            "test_labels": {
                "path": "manifests/test_labels.parquet",
                "sha256": sha256_file(ground_truth),
            }
        },
    }
    experiment_lock["manifest_content_sha256"] = canonical_sha256(experiment_lock)
    _json(modular / "manifests/experiment_lock.json", experiment_lock)
    _json(
        modular / ".EXPERIMENT_LOCKED",
        {
            "schema_version": 1,
            "lock_status": "LOCKED",
            "run_id": experiment_lock["run_id"],
            "lock_relative_path": "manifests/experiment_lock.json",
            "manifest_content_sha256": experiment_lock["manifest_content_sha256"],
        },
    )
    _json(
        modular / "FINALIZATION_COMPLETE.json",
        {
            "schema_version": 1,
            "status": "COMPLETE",
            "experiment_lock_sha256": experiment_lock["manifest_content_sha256"],
        },
    )
    formal_lock = json.loads(formal_lock_path.read_text(encoding="utf-8"))
    formal_lock["base_run"] = str(modular.resolve())
    formal_lock["source_label_artifacts"] = [
        {"path": str(ground_truth.resolve()), "sha256": sha256_file(ground_truth)}
    ]
    _json(formal_lock_path, formal_lock)
    run_sha_path.write_text(
        "".join(
            f"{sha256_file(path)}  {path.relative_to(historical).as_posix()}\n"
            for path in sorted(authority_paths)
        ),
        encoding="utf-8",
    )
    (historical / "RUN_LOCK_SHA256.txt").write_text(
        sha256_file(run_sha_path) + "\n", encoding="utf-8"
    )
    return run, historical, modular


def _prepare(root: Path) -> tuple[Path, dict[str, object]]:
    run, historical, modular = _sources(root)
    manifest = prepare_bridge(
        run_dir=run,
        historical_run=historical,
        modular_run=modular,
    )
    return run, manifest


def _resign_historical_authority(historical: Path) -> None:
    inventory_path = historical / "audit/frozen_pool_inventory.json"
    formal_lock_path = historical / "08_lock/FORMAL_TEST_LOCK.json"
    formal_lock = json.loads(formal_lock_path.read_text(encoding="utf-8"))
    formal_lock["audit_artifacts"] = [
        {"path": str(inventory_path.resolve()), "sha256": sha256_file(inventory_path)}
    ]
    _json(formal_lock_path, formal_lock)
    authority_paths = [
        inventory_path,
        formal_lock_path,
        *[
            historical / f"data/frozen_{route}_test_{pool}_candidates.parquet"
            for route in ("g1", "c1")
            for pool in ("top5", "allnms")
        ],
    ]
    run_sha_path = historical / "RUN_SHA256_MANIFEST.txt"
    run_sha_path.write_text(
        "".join(
            f"{sha256_file(path)}  {path.relative_to(historical).as_posix()}\n"
            for path in sorted(authority_paths)
        ),
        encoding="utf-8",
    )
    (historical / "RUN_LOCK_SHA256.txt").write_text(
        sha256_file(run_sha_path) + "\n", encoding="utf-8"
    )


def _resign_bundle(run: Path, frame: pd.DataFrame) -> Path:
    manifest_path = run / "11_attribution_bridge/test_bridge_input/manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    bundle_path = Path(manifest["artifacts"]["candidate_bundle"]["path"])
    frame.to_parquet(bundle_path, index=False)
    manifest["artifacts"]["candidate_bundle"]["sha256"] = sha256_file(bundle_path)
    unsigned = dict(manifest)
    unsigned.pop("content_sha256", None)
    manifest["content_sha256"] = canonical_sha256(unsigned)
    _json(manifest_path, manifest)
    return manifest_path


def test_label_free_test_bridge_qualifies_colliding_ids_and_checks_formulas(
    tmp_path: Path,
) -> None:
    run, manifest = _prepare(tmp_path)
    manifest_path = run / "11_attribution_bridge/test_bridge_input/manifest.json"
    checked, bundle, _ = validate_label_free_test_bridge_manifest(manifest_path)
    assert checked["status"] == "COMPLETE"
    assert manifest["historical_test_ground_truth_rows_read"] is False
    assert bundle["candidate_id"].str.count("::").eq(2).all()
    assert not bundle.duplicated(
        ["route", "candidate_pool_contract", "sample_id", "candidate_id"]
    ).any()
    assert set(bundle["source_candidate_id"]) == {"same", "other"}
    assert checked["pool_checks"]["g1/fair_gaussian"]["no_output_samples"] == 1
    assert checked["pool_checks"]["g1/historical_nms"]["no_output_samples"] == 2
    events = [
        json.loads(line)["event"]
        for line in (run / "09_formal_test/test_access.log").read_text().splitlines()
    ]
    assert events == ["label_free_test_bridge_input"]


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("duplicate", "duplicate"),
        ("missing", "exactly cover"),
        ("formula", "violates"),
        ("nonfinite", "finite"),
    ],
)
def test_bridge_rejects_coverage_formula_and_nonfinite_inputs(
    tmp_path: Path, mutation: str, message: str
) -> None:
    run, historical, modular = _sources(tmp_path)
    if mutation in {"duplicate", "missing", "nonfinite"}:
        path = run / "03_features/common/g1_test/candidate_features.parquet"
        frame = pd.read_parquet(path)
        if mutation == "duplicate":
            frame = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
        elif mutation == "missing":
            frame = frame.iloc[1:].copy()
        else:
            frame.loc[0, "p_center"] = np.inf
        frame.to_parquet(path, index=False)
    else:
        path = historical / "data/frozen_g1_test_top5_candidates.parquet"
        frame = pd.read_parquet(path)
        frame.loc[0, "original_score"] += 0.1
        frame.to_parquet(path, index=False)
        inventory_path = historical / "audit/frozen_pool_inventory.json"
        inventory = json.loads(inventory_path.read_text())
        inventory["G1_test"]["top5_artifact_sha256"] = sha256_file(path)
        _json(inventory_path, inventory)
        _resign_historical_authority(historical)
    with pytest.raises(ValueError, match=message):
        prepare_bridge(
            run_dir=run, historical_run=historical, modular_run=modular
        )


def test_bridge_rejects_source_backend_and_inventory_swaps(tmp_path: Path) -> None:
    run, historical, modular = _sources(tmp_path)
    path = historical / "data/frozen_g1_test_top5_candidates.parquet"
    frame = pd.read_parquet(path)
    frame["backend"] = "C1"
    frame.to_parquet(path, index=False)
    inventory_path = historical / "audit/frozen_pool_inventory.json"
    inventory = json.loads(inventory_path.read_text())
    inventory["G1_test"]["top5_artifact_sha256"] = sha256_file(path)
    _json(inventory_path, inventory)
    _resign_historical_authority(historical)
    with pytest.raises(ValueError, match="split/backend"):
        prepare_bridge(run_dir=run, historical_run=historical, modular_run=modular)

    run, historical, modular = _sources(tmp_path / "inventory")
    inventory_path = historical / "audit/frozen_pool_inventory.json"
    inventory = json.loads(inventory_path.read_text())
    inventory["G1_test"]["top5_artifact_sha256"] = "0" * 64
    _json(inventory_path, inventory)
    with pytest.raises(ValueError, match="RUN_SHA256_MANIFEST binding mismatch"):
        prepare_bridge(run_dir=run, historical_run=historical, modular_run=modular)


def test_bridge_rejects_precomputed_historical_candidate_labels(
    tmp_path: Path,
) -> None:
    run, historical, modular = _sources(tmp_path)
    path = historical / "data/frozen_g1_test_top5_candidates.parquet"
    frame = pd.read_parquet(path)
    frame["candidate_success"] = True
    frame.to_parquet(path, index=False)
    inventory_path = historical / "audit/frozen_pool_inventory.json"
    inventory = json.loads(inventory_path.read_text())
    inventory["G1_test"]["top5_artifact_sha256"] = sha256_file(path)
    _json(inventory_path, inventory)
    _resign_historical_authority(historical)
    with pytest.raises(PermissionError, match="forbidden historical-label"):
        prepare_bridge(run_dir=run, historical_run=historical, modular_run=modular)


def test_bridge_allows_label_free_candidate_geometry_iou_and_score_delta(
    tmp_path: Path,
) -> None:
    run, historical, modular = _sources(tmp_path)
    for route in ("g1", "c1"):
        path = run / f"03_features/common/{route}_test/candidate_features.parquet"
        frame = pd.read_parquet(path)
        frame["delta_to_previous"] = 0.0
        frame["max_iou_with_other_candidate"] = 0.5
        frame.to_parquet(path, index=False)
    result = prepare_bridge(
        run_dir=run, historical_run=historical, modular_run=modular
    )
    assert result["status"] == "COMPLETE"


def test_bridge_rejects_coordinated_candidate_and_inventory_drift(tmp_path: Path) -> None:
    run, historical, modular = _sources(tmp_path)
    path = historical / "data/frozen_g1_test_top5_candidates.parquet"
    frame = pd.read_parquet(path)
    frame.loc[0, "center_x"] += 1.0
    frame.to_parquet(path, index=False)
    inventory_path = historical / "audit/frozen_pool_inventory.json"
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    inventory["G1_test"]["top5_artifact_sha256"] = sha256_file(path)
    _json(inventory_path, inventory)
    with pytest.raises(ValueError, match="RUN_SHA256_MANIFEST binding mismatch"):
        prepare_bridge(run_dir=run, historical_run=historical, modular_run=modular)


def test_bridge_rejects_coordinated_ground_truth_and_modular_lock_drift(
    tmp_path: Path,
) -> None:
    run, historical, modular = _sources(tmp_path)
    ground_truth = modular / "manifests/test_labels.parquet"
    ground_truth.write_bytes(ground_truth.read_bytes() + b"coordinated-drift")
    lock_path = modular / "manifests/experiment_lock.json"
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    lock["artifacts"]["test_labels"]["sha256"] = sha256_file(ground_truth)
    lock.pop("manifest_content_sha256")
    lock["manifest_content_sha256"] = canonical_sha256(lock)
    _json(lock_path, lock)
    marker_path = modular / ".EXPERIMENT_LOCKED"
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    marker["manifest_content_sha256"] = lock["manifest_content_sha256"]
    _json(marker_path, marker)
    finalization_path = modular / "FINALIZATION_COMPLETE.json"
    finalization = json.loads(finalization_path.read_text(encoding="utf-8"))
    finalization["experiment_lock_sha256"] = lock["manifest_content_sha256"]
    _json(finalization_path, finalization)
    with pytest.raises(ValueError, match="formal lock does not bind modular Test ground truth"):
        prepare_bridge(run_dir=run, historical_run=historical, modular_run=modular)


def test_bridge_rejects_resigned_geometry_labels_and_source_tamper(
    tmp_path: Path,
) -> None:
    run, _ = _prepare(tmp_path / "geometry")
    manifest_path = run / "11_attribution_bridge/test_bridge_input/manifest.json"
    manifest = json.loads(manifest_path.read_text())
    bundle = pd.read_parquet(manifest["artifacts"]["candidate_bundle"]["path"])
    bundle.loc[0, "candidate_geometry_sha256"] = "wrong"
    manifest_path = _resign_bundle(run, bundle)
    with pytest.raises(ValueError, match="geometry binding"):
        validate_label_free_test_bridge_manifest(manifest_path)

    run, _ = _prepare(tmp_path / "labels")
    manifest_path = run / "11_attribution_bridge/test_bridge_input/manifest.json"
    manifest = json.loads(manifest_path.read_text())
    bundle = pd.read_parquet(manifest["artifacts"]["candidate_bundle"]["path"])
    bundle["candidate_success"] = True
    manifest_path = _resign_bundle(run, bundle)
    with pytest.raises(ValueError, match="schema mismatch"):
        validate_label_free_test_bridge_manifest(manifest_path)

    run, manifest = _prepare(tmp_path / "evaluator")
    evaluator = Path(manifest["sources"]["evaluator"]["path"])
    evaluator.write_text(evaluator.read_text() + "\n# tamper\n", encoding="utf-8")
    with pytest.raises(ValueError, match="evaluator hash mismatch"):
        validate_label_free_test_bridge_manifest(
            run / "11_attribution_bridge/test_bridge_input/manifest.json"
        )
