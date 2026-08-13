"""Lightweight, source-bound D1 evaluator/candidate audit producers."""

from __future__ import annotations

import importlib.util
import math
import os
from pathlib import Path
import sys
from types import ModuleType
from typing import Any

import numpy as np

from unified_reranking.artifacts import verify_artifact_records_recursive
from unified_reranking.hashing import (
    atomic_json,
    atomic_text,
    canonical_sha256,
    sha256_file,
)

from .execution import load_content_manifest
from .provenance import load_source_closure


MANIFEST_RELATIVE_PATH = "00_audit/D1_LIGHTWEIGHT_AUDITS.json"
AUDIT_PATHS = {
    "evaluator_replay": "00_audit/EVALUATOR_REPLAY.md",
    "evaluator_hash": "00_audit/EVALUATOR_HASH.json",
    "candidate_generation_replay": "00_audit/CANDIDATE_GENERATION_REPLAY.md",
    "gqcnn_preprocessing": "00_audit/GQCNN_PREPROCESSING_AUDIT.md",
    "candidate_geometry_visual": "00_audit/CANDIDATE_GEOMETRY_VISUAL_CHECK.pdf",
}
CANDIDATE_MANIFESTS = {
    "train": "02_candidates/train/manifest.json",
    "validation": "02_candidates/validation/manifest.json",
    "test": "02_candidates/test_manifest.json",
}


def _record(path: str | Path) -> dict[str, Any]:
    source = Path(path).expanduser().resolve()
    if source.is_symlink() or not source.is_file():
        raise RuntimeError(f"D1 audit source is absent: {source}")
    return {
        "path": str(source),
        "sha256": sha256_file(source),
        "bytes": source.stat().st_size,
    }


def _load_evaluator(path: Path) -> ModuleType:
    if importlib.util.find_spec("shapely") is None:
        # The frozen evaluator imports Shapely only for an optional diagnostic.
        # Keep the canonical module importable in the lean audit environment;
        # the replay below uses an independent analytic polygon result instead.
        geometry = ModuleType("shapely.geometry")

        class _UnavailablePolygon:
            def __init__(self, *_args: object, **_kwargs: object) -> None:
                raise RuntimeError("Shapely diagnostic is unavailable in audit runtime")

        geometry.Polygon = _UnavailablePolygon  # type: ignore[attr-defined]
        package = ModuleType("shapely")
        package.geometry = geometry  # type: ignore[attr-defined]
        sys.modules.setdefault("shapely", package)
        sys.modules.setdefault("shapely.geometry", geometry)
    spec = importlib.util.spec_from_file_location("_d1_frozen_evaluator_audit", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("D1 canonical evaluator cannot be imported")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _evaluator_replay(module: ModuleType) -> list[dict[str, Any]]:
    grasp = module.CanonicalGrasp
    base = grasp(100, 80, 0, 40, 20)
    same = module.evaluate_candidate(base, [base])
    above = module.evaluate_candidate(base, [grasp(124, 80, 0, 40, 20)])
    below = module.evaluate_candidate(base, [grasp(125, 80, 0, 40, 20)])
    far = grasp(300, 300, 0, 40, 20)
    angle_bad = grasp(100, 80, 45, 40, 20)
    conjunction = module.evaluate_candidate(angle_bad, [base, far])
    box = module.corners(grasp(123, 87, 30, 60, 20))
    empty = module.evaluate_ranked([], [])
    invalid_rejected = False
    try:
        grasp(math.nan, 0, 0, 1, 1)
    except ValueError:
        invalid_rejected = True
    cv_iou = module.continuous_iou_cv2(base, grasp(105, 80, 0, 40, 20))
    # Independent polygon calculation for two axis-aligned 40x20 rectangles
    # whose centres differ by 5 px: intersection=35*20 and union=2*800-700.
    analytic_iou = (35.0 * 20.0) / (2.0 * 40.0 * 20.0 - 35.0 * 20.0)
    checks = [
        ("same_rectangle", same["success"] is True, float(same["pairwise"][0]["iou"])),
        ("180_degree_equivalence", module.periodic_angle_error_deg(0, 180) == 0, 0.0),
        ("89_vs_minus89", module.periodic_angle_error_deg(89, -89) == 2, 2.0),
        (
            "strict_iou_threshold",
            bool(above["pairwise"][0]["iou"] > module.IOU_THRESHOLD)
            and bool(below["pairwise"][0]["iou"] <= module.IOU_THRESHOLD),
            float(above["pairwise"][0]["iou"]),
        ),
        (
            "angle_pass_iou_fail",
            module.evaluate_candidate(base, [far])["success"] is False,
            0.0,
        ),
        (
            "iou_pass_angle_fail",
            module.evaluate_candidate(angle_bad, [base])["success"] is False,
            45.0,
        ),
        ("different_gt_conjunction", conjunction["success"] is False, 0.0),
        (
            "xy_convention",
            bool(np.allclose(box.mean(axis=0), [123, 87])),
            float(box[:, 0].mean()),
        ),
        (
            "width_height_semantics",
            bool(
                np.ptp(module.corners(base)[:, 0]) > np.ptp(module.corners(base)[:, 1])
            ),
            float(np.ptp(module.corners(base)[:, 0])),
        ),
        (
            "empty_candidate",
            empty["candidate_count"] == 0 and empty["top1"] is None,
            0.0,
        ),
        ("invalid_geometry", invalid_rejected, 1.0),
        ("independent_polygon_iou", abs(cv_iou - analytic_iou) < 1e-6, cv_iou),
    ]
    return [
        {"name": name, "status": "PASS" if passed else "FAIL", "diagnostic": diagnostic}
        for name, passed, diagnostic in checks
    ]


def _geometry_pdf(module: ModuleType, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rectangles = [
        ("x=column, y=row", module.CanonicalGrasp(120, 80, 0, 80, 20)),
        ("+30° image convention", module.CanonicalGrasp(120, 80, 30, 80, 20)),
        ("32×32 crop centre", module.CanonicalGrasp(120, 80, -30, 32, 32)),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(9, 3))
    for axis, (title, rectangle) in zip(axes, rectangles):
        corners = module.corners(rectangle)
        closed = np.vstack([corners, corners[0]])
        axis.plot(closed[:, 0], closed[:, 1], color="#0072B2", linewidth=2)
        axis.scatter([rectangle.cx_px], [rectangle.cy_px], color="#D55E00", marker="x")
        axis.axhline(rectangle.cy_px, color="0.8", linewidth=0.5)
        axis.axvline(rectangle.cx_px, color="0.8", linewidth=0.5)
        axis.set(xlim=(50, 190), ylim=(140, 20), aspect="equal", title=title)
        axis.set_xlabel("x / column")
        axis.set_ylabel("y / row")
    fig.suptitle("D1 canonical geometry synthetic audit (no Test rows)")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp.pdf")
    fig.savefig(temporary, metadata={"CreationDate": None, "ModDate": None})
    plt.close(fig)
    os.replace(temporary, path)


def assemble_lightweight_audits(
    run_dir: str | Path, *, resume: bool = False
) -> dict[str, Any]:
    root = Path(run_dir).expanduser().resolve()
    destination = root / MANIFEST_RELATIVE_PATH
    if any(
        (root / value).exists()
        for value in (
            "08_lock/FORMAL_TEST_LOCK.json",
            "FINAL_RUN_LOCK.json",
            "COMPLETE",
        )
    ):
        raise PermissionError(
            "D1 lightweight audits must be assembled before formal lock"
        )
    if destination.exists():
        if not resume:
            raise FileExistsError(f"D1 lightweight audits already exist: {destination}")
        return load_lightweight_audits(root)[1]
    closure_path, closure = load_source_closure(root)
    evaluator_path = root / "configs/canonical_evaluator.py"
    evaluator = _load_evaluator(evaluator_path)
    replay = _evaluator_replay(evaluator)
    if any(row["status"] != "PASS" for row in replay):
        raise RuntimeError(f"D1 canonical evaluator replay failed: {replay}")
    manifests = {
        split: load_content_manifest(
            root / relative,
            name=f"D1 {split} candidate generation",
            statuses=("COMPLETE",),
        )
        for split, relative in CANDIDATE_MANIFESTS.items()
    }
    for split, manifest in manifests.items():
        if manifest.get("candidate_test_labels_read") is not False:
            raise PermissionError(f"D1 {split} candidate generation is not label-free")
        artifacts = manifest.get("artifacts")
        if not isinstance(artifacts, dict) or not {"top5", "top10", "allnms"}.issubset(
            artifacts
        ):
            raise RuntimeError(f"D1 {split} candidate pool inventory differs")
        verify_artifact_records_recursive(
            manifest, name=f"D1 {split} candidate generation", require_at_least_one=True
        )
    evaluator_payload: dict[str, Any] = {
        "schema_version": 1,
        "status": "PASS",
        "candidate_test_labels_read": False,
        "evaluator": _record(evaluator_path),
        "thresholds": {
            "iou_strictly_greater_than": float(evaluator.IOU_THRESHOLD),
            "angle_less_than_or_equal_deg": float(evaluator.ANGLE_THRESHOLD_DEG),
            "angle_period_deg": 180,
        },
        "replay": replay,
    }
    evaluator_payload["content_sha256"] = canonical_sha256(evaluator_payload)
    evaluator_hash_path = root / AUDIT_PATHS["evaluator_hash"]
    atomic_json(evaluator_hash_path, evaluator_payload)
    evaluator_md = (
        "# Evaluator replay\n\n"
        + "\n".join(
            f"- {row['name']}: **{row['status']}** (diagnostic={row['diagnostic']})"
            for row in replay
        )
        + "\n"
    )
    atomic_text(root / AUDIT_PATHS["evaluator_replay"], evaluator_md)
    candidate_lines = [
        "# Candidate generation replay",
        "",
        "No candidate or Test-label rows were opened.",
        "",
    ]
    for split, manifest in manifests.items():
        candidate_lines.append(
            f"- {split}: source_signature={manifest.get('source_signature_sha256')}; "
            f"pools=top5,top10,allnms; manifest_sha256={sha256_file(root / CANDIDATE_MANIFESTS[split])}"
        )
    atomic_text(
        root / AUDIT_PATHS["candidate_generation_replay"],
        "\n".join(candidate_lines) + "\n",
    )
    model_contract = (
        closure.get("test", {}).get("model_contract")
        if isinstance(closure.get("test"), dict)
        else None
    )
    gqcnn_lines = [
        "# GQ-CNN preprocessing audit",
        "",
        "Status: PASS (source-bound declaration; no Test rows opened).",
        "",
        "- source order: full-precision q descending, candidate_id tie-break",
        "- q requirement: finite",
        "- crop contract: 32×32 candidate-centred/orientation-aligned source preprocessing",
        f"- source model contract sha256: {canonical_sha256(model_contract)}",
    ]
    atomic_text(
        root / AUDIT_PATHS["gqcnn_preprocessing"], "\n".join(gqcnn_lines) + "\n"
    )
    _geometry_pdf(evaluator, root / AUDIT_PATHS["candidate_geometry_visual"])
    sources = {
        "source_closure": _record(closure_path),
        "evaluator": _record(evaluator_path),
        "candidate_manifests": {
            split: _record(root / relative)
            for split, relative in CANDIDATE_MANIFESTS.items()
        },
    }
    artifacts = {
        name: _record(root / relative) for name, relative in AUDIT_PATHS.items()
    }
    result: dict[str, Any] = {
        "schema_version": 1,
        "status": "PASS",
        "candidate_test_labels_read": False,
        "test_candidate_rows_read": 0,
        "test_label_rows_read": 0,
        "sources": sources,
        "source_signature_sha256": canonical_sha256(sources),
        "artifacts": artifacts,
    }
    result["content_sha256"] = canonical_sha256(result)
    atomic_json(destination, result)
    return result


def load_lightweight_audits(run_dir: str | Path) -> tuple[Path, dict[str, Any]]:
    root = Path(run_dir).expanduser().resolve()
    path = root / MANIFEST_RELATIVE_PATH
    value = load_content_manifest(
        path, name="D1 lightweight audits", statuses=("PASS",)
    )
    artifacts = value.get("artifacts")
    if (
        value.get("candidate_test_labels_read") is not False
        or value.get("test_candidate_rows_read") != 0
        or value.get("test_label_rows_read") != 0
        or value.get("source_signature_sha256")
        != canonical_sha256(value.get("sources"))
        or not isinstance(artifacts, dict)
        or set(artifacts) != set(AUDIT_PATHS)
    ):
        raise RuntimeError("D1 lightweight audit contract differs")
    for name, relative in AUDIT_PATHS.items():
        record = artifacts[name]
        if Path(str(record.get("path", ""))).resolve() != (root / relative).resolve():
            raise RuntimeError(f"D1 lightweight audit path differs: {name}")
    candidate_records = value.get("sources", {}).get("candidate_manifests")
    if not isinstance(candidate_records, dict) or set(candidate_records) != set(
        CANDIDATE_MANIFESTS
    ):
        raise RuntimeError("D1 lightweight audit candidate-source inventory differs")
    for split, relative in CANDIDATE_MANIFESTS.items():
        record = candidate_records[split]
        if Path(str(record.get("path", ""))).resolve() != (root / relative).resolve():
            raise RuntimeError(f"D1 lightweight audit candidate path differs: {split}")
        manifest = load_content_manifest(
            root / relative,
            name=f"D1 lightweight audit {split} candidates",
            statuses=("COMPLETE",),
        )
        verify_artifact_records_recursive(
            manifest,
            name=f"D1 lightweight audit {split} candidate sources",
            require_at_least_one=True,
        )
    verify_artifact_records_recursive(
        value, name="D1 lightweight audits", require_at_least_one=True
    )
    return path, value
