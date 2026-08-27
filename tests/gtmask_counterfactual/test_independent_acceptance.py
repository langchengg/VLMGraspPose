from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pandas as pd

from gtmask_counterfactual.independent import (
    NATIVE_CLASSES,
    POST_R7_CLASSES,
    canonical_corners,
    independent_recompute_from_frames,
)
from unified_reranking.hashing import atomic_json, canonical_sha256, sha256_file


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
TOOL = ROOT / "tools/gtmask_counterfactual/accept_outputs.py"


def _record(path: Path) -> dict[str, object]:
    source = path.resolve()
    return {
        "path": str(source),
        "sha256": sha256_file(source),
        "bytes": source.stat().st_size,
    }


def _self_hashed(path: Path, value: dict[str, object]) -> Path:
    payload = dict(value)
    payload["content_sha256"] = canonical_sha256(payload)
    return atomic_json(path, payload)


def _parquet(frame: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)
    return path


def _candidate(
    *, sample_id: str, route: str, branch: str, cx_px: float
) -> dict[str, object]:
    return {
        "sample_id": sample_id,
        "route": route,
        "branch": branch,
        "candidate_id": f"{route}-{branch}-{sample_id}",
        "native_rank": 1,
        "native_score": 0.5,
        "cx_px": cx_px,
        "cy_px": 60.0,
        "theta_deg": 0.0,
        "width_px": 30.0,
        "height_px": 20.0,
    }


def _branch_table(metrics: dict[str, dict[str, object]]) -> pd.DataFrame:
    rows = []
    for key, value in sorted(metrics.items()):
        route, branch = key.split("|", 1)
        rows.append(
            {
                "route": route,
                "branch": branch,
                "N": value["N"],
                "no_output": value["no_output"],
                "native_correct": value["native_j_at_1_numerator"],
                "oracle_at_5": value["oracle_at_5_numerator"],
                "oracle_all": value["oracle_all_numerator"],
                "candidate_count_mean": value["candidate_count_mean"],
                "candidate_count_median": value["candidate_count_median"],
                "candidate_count_p95": value["candidate_count_p95"],
                "native_j_at_1": value["native_j_at_1"],
                "j_at_5": value["j_at_5"],
                "mrr": value["mrr"],
                "positive_candidates_per_sample_mean": value[
                    "positive_candidates_per_sample_mean"
                ],
                "positive_candidates_per_sample_median": value[
                    "positive_candidates_per_sample_median"
                ],
                "positive_candidates_per_sample_p95": value[
                    "positive_candidates_per_sample_p95"
                ],
                "first_positive_rank_distribution_json": json.dumps(
                    value["first_positive_rank_distribution"], sort_keys=True
                ),
                "positive_candidates_per_sample_distribution_json": json.dumps(
                    value["positive_candidates_per_sample_distribution"],
                    sort_keys=True,
                ),
            }
        )
    return pd.DataFrame(rows)


def _build_saved_run(tmp_path: Path) -> Path:
    run = tmp_path / "fair_gtmask_counterfactual_g1_c1_d1_synthetic"
    sample_ids = ("s1", "s2")
    routes = ("G1", "C1")
    sample = pd.DataFrame(
        {
            "sample_id": sample_ids,
            "scene_id": ("scene-1", "scene-2"),
            "frame_id": ("frame-1", "frame-2"),
        }
    )
    grasp = canonical_corners(
        {
            "cx_px": 50.0,
            "cy_px": 60.0,
            "theta_deg": 0.0,
            "width_px": 30.0,
            "height_px": 20.0,
        }
    ).tolist()
    ground_truth = pd.DataFrame(
        {
            "sample_id": sample_ids,
            "gt_grasp_rectangles": ([grasp], [grasp]),
        }
    )
    candidate_records: dict[str, dict[str, object]] = {}
    candidate_frames = []
    for route in routes:
        for branch in ("predicted", "gt_oracle"):
            frame = pd.DataFrame(
                [
                    _candidate(
                        sample_id=sample_id,
                        route=route,
                        branch=branch,
                        cx_px=(
                            200.0
                            if branch == "predicted" and sample_id == "s2"
                            else 50.0
                        ),
                    )
                    for sample_id in sample_ids
                ]
            )
            path = _parquet(
                frame,
                run
                / "07_candidate_tables/raw"
                / route.lower()
                / branch
                / "candidates.parquet",
            )
            candidate_records[f"{route}|{branch}"] = _record(path)
            candidate_frames.append(frame)
    candidates = pd.concat(candidate_frames, ignore_index=True)
    final = pd.DataFrame(
        [
            {
                "sample_id": sample_id,
                "route": route,
                "final_correct": sample_id == "s1",
            }
            for route in routes
            for sample_id in sample_ids
        ]
    )
    recomputed = independent_recompute_from_frames(
        sample,
        candidates,
        ground_truth,
        k_by_route={route: (5,) for route in routes},
        final_outcomes=final,
        taxonomy_definitions={
            "native_classes": list(NATIVE_CLASSES),
            "post_r7_classes": list(POST_R7_CLASSES),
        },
    )

    sample_path = _parquet(
        sample, run / "02_sample_manifest/counterfactual_manifest.parquet"
    )
    ground_truth_path = _parquet(
        ground_truth, run / "02_sample_manifest/ground_truth.parquet"
    )
    final_path = _parquet(
        final, run / "04_predicted_replay/frozen_final_outcomes.parquet"
    )
    labels_path = _parquet(
        recomputed["candidate_labels"],
        run / "07_candidate_tables/per_candidate_labels.parquet",
    )
    outcomes_path = _parquet(
        recomputed["sample_outcomes"], run / "08_metrics/per_sample_outcomes.parquet"
    )
    native_path = _parquet(
        recomputed["native_taxonomy"],
        run / "09_failure_taxonomy/native_failure_taxonomy_per_sample.parquet",
    )
    post_path = _parquet(
        recomputed["post_r7_taxonomy"],
        run / "09_failure_taxonomy/post_r7_bottleneck_per_sample.parquet",
    )
    statistical_path = _parquet(
        recomputed["statistical_inputs"],
        run / "10_statistics/statistical_inputs.parquet",
    )
    inline_path = _self_hashed(
        run / "16_independent_recompute/recomputed_metrics.json",
        {"schema_version": 1, "status": "PASS"},
    )
    branch_path = run / "tables/branch_metrics.csv"
    branch_path.parent.mkdir(parents=True, exist_ok=True)
    _branch_table(recomputed["branch_metrics"]).to_csv(branch_path, index=False)
    table_bundle_path = _self_hashed(
        run / "08_metrics/TABLE_BUNDLE_MANIFEST.json",
        {
            "schema_version": 1,
            "status": "COMPLETE",
            "tables": {"branch_metrics.csv": _record(branch_path)},
        },
    )
    inputs_path = _self_hashed(
        run / "07_candidate_tables/POSTPROCESS_INPUTS.json",
        {
            "schema_version": 1,
            "status": "LOCKED",
            "artifacts": {
                "sample_manifest": _record(sample_path),
                "ground_truth": _record(ground_truth_path),
                "candidates": candidate_records,
                "final_outcomes": _record(final_path),
            },
        },
    )
    postprocess_path = _self_hashed(
        run / "08_metrics/POSTPROCESS_MANIFEST.json",
        {
            "schema_version": 1,
            "status": "PARTIAL",
            "routes": {
                "G1": "COMPLETE",
                "C1": "COMPLETE",
                "D1": "UNRECOVERABLE_BLOCKER",
            },
            "postprocess_inputs": _record(inputs_path),
            "artifacts": {
                "candidate_labels": _record(labels_path),
                "sample_outcomes": _record(outcomes_path),
                "native_taxonomy": _record(native_path),
                "post_r7_taxonomy": _record(post_path),
                "statistical_inputs": _record(statistical_path),
                "table_bundle": _record(table_bundle_path),
                "independent_recompute": _record(inline_path),
            },
        },
    )
    gallery_path = _self_hashed(
        run / "14_galleries/GALLERY_MANIFEST.json",
        {"schema_version": 1, "status": "COMPLETE"},
    )
    _self_hashed(
        run / "12_case_selection/P9_GALLERY_ACCEPTANCE.json",
        {
            "schema_version": 1,
            "status": "PASS",
            "gallery_manifest": _record(gallery_path),
            "postprocess_manifest": _record(postprocess_path),
            "eligible_count": 4,
            "selected_count": 2,
            "board_count": 2,
            "manual_qa_count": 2,
        },
    )
    atomic_json(
        run / "pipeline_status.json",
        {
            "status": "P9_GALLERIES_COMPLETE",
            "first_incomplete_stage": "P10_INDEPENDENT_RECOMPUTE_PASS",
        },
    )
    return run


def _environment() -> dict[str, str]:
    environment = os.environ.copy()
    existing = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        str(SRC) if not existing else os.pathsep.join((str(SRC), existing))
    )
    return environment


def _run_standalone(
    run: Path, *, resume: bool = False
) -> subprocess.CompletedProcess[str]:
    code = (
        "from gtmask_counterfactual.independent_acceptance import "
        "accept_independent_recompute; "
        f"print(accept_independent_recompute({str(run)!r}, "
        f"resume={resume!r}, expected_sample_count=2))"
    )
    return subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT,
        env=_environment(),
        check=False,
        capture_output=True,
        text=True,
    )


def test_standalone_p10_recomputes_and_records_real_import_audit(
    tmp_path: Path,
) -> None:
    run = _build_saved_run(tmp_path)
    completed = _run_standalone(run)
    assert completed.returncode == 0, completed.stderr
    acceptance = json.loads(
        (run / "16_independent_recompute/INDEPENDENT_VALIDATION.json").read_text()
    )
    assert acceptance["status"] == "PASS"
    assert acceptance["forbidden_modules_imported"] is False
    assert acceptance["forbidden_import_audit"]["status"] == "PASS"
    assert acceptance["forbidden_import_audit"]["observed_forbidden_modules"] == []
    assert json.loads((run / "pipeline_status.json").read_text())["status"] == (
        "P10_INDEPENDENT_RECOMPUTE_PASS"
    )

    resumed = _run_standalone(run, resume=True)
    assert resumed.returncode == 0, resumed.stderr


def test_standalone_p10_rejects_loaded_forbidden_module(tmp_path: Path) -> None:
    code = """
import sys
import types
sys.modules['gtmask_counterfactual.gallery_pipeline'] = types.ModuleType(
    'gtmask_counterfactual.gallery_pipeline'
)
from gtmask_counterfactual.independent_acceptance import accept_independent_recompute
try:
    accept_independent_recompute(sys.argv[1], expected_sample_count=2)
except RuntimeError as error:
    print(error)
    raise SystemExit(0 if 'gallery_pipeline' in str(error) else 2)
raise SystemExit(3)
"""
    completed = subprocess.run(
        [sys.executable, "-c", code, str(tmp_path)],
        cwd=ROOT,
        env=_environment(),
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert "forbidden modules at entry" in completed.stdout


def test_standalone_p10_requires_hash_bound_p9_acceptance(tmp_path: Path) -> None:
    run = _build_saved_run(tmp_path)
    acceptance_path = run / "12_case_selection/P9_GALLERY_ACCEPTANCE.json"
    acceptance = json.loads(acceptance_path.read_text())
    acceptance["selected_count"] = 3
    atomic_json(acceptance_path, acceptance)

    completed = _run_standalone(run)
    assert completed.returncode != 0
    assert "P9 gallery acceptance content hash differs" in completed.stderr
    assert not (run / "16_independent_recompute/INDEPENDENT_VALIDATION.json").exists()


def test_accept_outputs_independent_action_is_lazy_and_gallery_free(
    tmp_path: Path,
) -> None:
    code = f"""
import importlib.util
import json
import sys
spec = importlib.util.spec_from_file_location('accept_outputs_under_test', {str(TOOL)!r})
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
before = sorted(name for name in sys.modules if name.startswith('gtmask_counterfactual'))
sys.argv = ['accept_outputs.py', 'independent', '--run-dir', {str(tmp_path)!r}]
try:
    module.main()
except RuntimeError:
    pass
after = sorted(
    name for name in sys.modules
    if name in {{
        'gtmask_counterfactual.acceptance',
        'gtmask_counterfactual.gallery_pipeline',
        'gtmask_counterfactual.resource',
        'unified_reranking.gate',
    }}
)
print(json.dumps({{'before': before, 'after': after}}))
"""
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT,
        env=_environment(),
        check=True,
        capture_output=True,
        text=True,
    )
    observed = json.loads(completed.stdout)
    assert observed == {"before": [], "after": []}
