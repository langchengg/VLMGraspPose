"""Resumable command line entry point for the robustness suite."""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from . import RUN_ID
from .io import (
    atomic_json,
    read_manifest,
    record_command,
    sha256_file,
    update_manifest,
    update_progress,
)


REPOSITORY_SHA = "601fa6fb3f445d3f426d0c3ed8781539da74db46"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _run_dir(repo_root: Path, run_id: str) -> Path:
    path = repo_root / "artifacts/robustness_suite" / run_id
    if not path.is_dir():
        raise FileNotFoundError(f"robustness run does not exist: {path}")
    return path


def _source_hash_contract(repo_root: Path) -> dict[str, str]:
    return {
        "artifacts/graspnet6d/20260819_221819_graspnet6d_vgn_lambdamart/run_manifest.json": "3e00ba91042d678d2dd6ae68b52270d4a2e0f05136ab9bb74f336be7e48674ba",
        "artifacts/graspnet6d/20260819_221819_graspnet6d_vgn_lambdamart/candidate_features.parquet": "79aa6635b8c5d6fed36ef8a072fbbff2f9d9a23e07f78fa2f77ca7c351ff7a69",
        "artifacts/graspnet6d/20260819_221819_graspnet6d_vgn_lambdamart/candidate_labels.parquet": "7d4c4e9176f3dd1c9b3cd38f4b5b4751f0dfc69f33d3031ff7d4f648b65b084e",
        "runs/fair_unified_reranking_20260809_103012/FINAL_RUN_LOCK.json": "4b52eac6494e59a0f902792b794c3cca02824bf7a36bed4699b569986383f793",
        "runs/fair_unified_reranking_20260809_103012/09_formal_test/formal_test_manifest.json": "d8dbf9c84924674aef0e87c43ab7d94fd8cd029ae8a6e35a9f3d0f7dcbd2ca32",
        "runs/fair_unified_reranking_20260809_103012/09_formal_test/formal_test_realized_rankings.parquet": "c1a822b93ad211f9ae36179169de42b8327771d2f59edb64ed0792c19585206c",
        "runs/fair_unified_reranking_20260809_103012/configs/canonical_evaluator.py": "f5155590b0b8d9f0748ad463688edfe6ca595d8ab7239ef5e29a5c595aeef301",
        "runs/fair_d1_reranking_extension_20260811T145515Z/manifest.json": "c6edd361abcd2ea27acf45c82c304dae600f4f01eb4b53099ee697b84dae7ece",
        "runs/fair_d1_reranking_extension_20260811T145515Z/08_lock/FORMAL_TEST_LOCK.json": "6ebd6bfcafda5649d4aa2f37062d86d2aa26921ea2dd72f4de646d7ad51fd13a",
        "runs/fair_d1_reranking_extension_20260811T145515Z/02_candidates/d1_test_top5.parquet": "d22e4c0064b883a423a16d434d59e4d25e8e98d849d454f0dfb674df53b17971",
        "HiFi_reproduction/runs/modular_repeatedfilm_4dof_backends_v1_r0corrected_20260803_163500/manifests/test_labels.parquet": "6262b2189c02b1af239537cf814734e138fbaa6c7535fb58cf1bd65222014e70",
    }


def _extended_source_paths(repo_root: Path) -> list[Path]:
    """Return every additional small source artifact consumed by this suite."""

    relative = [
        "artifacts/graspnet6d/20260819_221819_graspnet6d_vgn_lambdamart/split_manifest.json",
        "artifacts/graspnet6d/20260819_221819_graspnet6d_vgn_lambdamart/resolved_config.yaml",
        "runs/fair_unified_reranking_20260809_103012/01_manifests/paired_test.parquet",
        "runs/fair_unified_reranking_20260809_103012/08_lock/selected_features.json",
        "runs/fair_unified_reranking_20260809_103012/08_lock/selected_hyperparameters.json",
        "runs/fair_unified_reranking_20260809_103012/09_formal_test/per_candidate_scores.parquet",
        "runs/fair_d1_reranking_extension_20260811T145515Z/09_formal_test/formal_candidate_outcomes.parquet",
        "runs/fair_d1_reranking_extension_20260811T145515Z/07_validation/primary_selection/trials/f2ed1d3e0611d7d7/manifest.json",
    ]
    primary = "runs/fair_unified_reranking_20260809_103012"
    for route in ("crog", "g1", "c1"):
        relative += [
            f"{primary}/02_candidates/{route}_test_top5.parquet",
            f"{primary}/02_candidates/{route}_test_all.parquet",
            f"{primary}/08_lock/formal_label_free/{route}_native_decisions.parquet",
            f"{primary}/08_lock/formal_label_free/{route}_ungated_decisions.parquet",
            f"{primary}/08_lock/formal_label_free/{route}_gated_decisions.parquet",
        ]
        for split in ("train", "validation", "test"):
            relative.append(
                f"{primary}/03_features/tracks/T2_matched_common/"
                f"{route}_{split}/candidate_features.parquet"
            )
            label = f"{primary}/03_features/candidate_labels_{route}_{split}_top5.parquet"
            if (repo_root / label).is_file():
                relative.append(label)
    d1 = "runs/fair_d1_reranking_extension_20260811T145515Z"
    relative += [
        f"{d1}/02_candidates/d1_test_top5.parquet",
        f"{d1}/02_candidates/d1_test_all.parquet",
        f"{d1}/08_lock/label_free_test_rankers/d1/per_sample_decisions.parquet",
        f"{d1}/08_lock/label_free_test_gates/d1/gate_test_decisions.parquet",
    ]
    for split in ("train", "validation", "test"):
        relative.append(
            f"{d1}/03_features/{split}/top5/T2_matched_common/"
            "candidate_features.parquet"
        )
        label = f"{d1}/03_features/{split}/top5/labels/candidate_labels.parquet"
        if (repo_root / label).is_file():
            relative.append(label)
    return [repo_root / path for path in relative]


def _audit_extended_sources(repo_root: Path, run_dir: Path) -> dict[str, Any]:
    """Create once, then verify, the complete small-artifact source inventory."""

    inventory_path = run_dir / "source_artifact_inventory.json"
    paths = _extended_source_paths(repo_root)
    missing = [str(path.relative_to(repo_root)) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"extended source artifacts are missing: {missing}")
    observed = {
        str(path.relative_to(repo_root)): sha256_file(path) for path in paths
    }
    if inventory_path.is_file():
        locked = json.loads(inventory_path.read_text(encoding="utf-8"))
        expected = {
            str(item["path"]): str(item["sha256"])
            for item in locked.get("files", [])
        }
        if observed != expected:
            changed = sorted(
                set(observed) ^ set(expected)
                | {
                    path
                    for path in set(observed) & set(expected)
                    if observed[path] != expected[path]
                }
            )
            raise RuntimeError(f"extended source inventory changed: {changed}")
    else:
        atomic_json(
            inventory_path,
            {
                "schema_version": 1,
                "run_id": run_dir.name,
                "selection": "all additional source artifacts consumed by 4D/6D analyses",
                "files": [
                    {"path": path, "sha256": digest}
                    for path, digest in sorted(observed.items())
                ],
            },
        )
    return {
        "status": "PASS",
        "file_count": len(observed),
        "inventory_path": str(inventory_path),
        "inventory_sha256": sha256_file(inventory_path),
    }


def audit_sources(repo_root: Path, run_dir: Path) -> dict[str, Any]:
    run_manifest = read_manifest(run_dir)
    run_local_contract = {
        "source_run_manifest.json": run_manifest["source_run_manifest_sha256"],
        "resolved_config.yaml": run_manifest["resolved_config_sha256"],
    }
    run_local_entries = []
    for filename, expected in run_local_contract.items():
        observed = sha256_file(run_dir / filename)
        run_local_entries.append(
            {
                "path": filename,
                "expected_sha256": expected,
                "observed_sha256": observed,
                "match": observed == expected,
            }
        )
        if observed != expected:
            raise RuntimeError(f"locked run-local audit input changed: {filename}")
    contract = _source_hash_contract(repo_root)
    entries: list[dict[str, Any]] = []
    failed: list[str] = []
    for relative, expected in contract.items():
        path = repo_root / relative
        observed = sha256_file(path) if path.is_file() else None
        match = observed == expected
        entries.append(
            {
                "path": relative,
                "expected_sha256": expected,
                "observed_sha256": observed,
                "exists": path.is_file(),
                "match": match,
            }
        )
        if not match:
            failed.append(relative)
    extended = _audit_extended_sources(repo_root, run_dir)
    result = {
        "status": "PASS" if not failed else "FAIL",
        "repository_sha_expected": REPOSITORY_SHA,
        "source_entries": entries,
        "failed_paths": failed,
        "extended_source_inventory": extended,
        "run_local_entries": run_local_entries,
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    atomic_json(run_dir / "source_readonly_snapshot.json", result)
    atomic_json(run_dir / "source_audit_verification.json", result)
    if failed:
        raise RuntimeError(f"source hash audit failed: {failed}")
    update_manifest(
        run_dir,
        source_artifact_inventory_sha256=extended["inventory_sha256"],
    )
    update_progress(run_dir, "audit", "COMPLETE")
    return result


def verify_preregistration(run_dir: Path) -> dict[str, Any]:
    manifest = read_manifest(run_dir)
    path = run_dir / "PRE_REGISTRATION.md"
    observed = sha256_file(path)
    expected = manifest["pre_registration_sha256"]
    result = {
        "status": "PASS" if observed == expected else "FAIL",
        "path": str(path),
        "observed_sha256": observed,
        "expected_sha256": expected,
        "verified_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    atomic_json(run_dir / "pre_registration_verification.json", result)
    if observed != expected:
        raise RuntimeError("PRE_REGISTRATION.md changed after it was locked")
    update_progress(run_dir, "preregister", "COMPLETE")
    return result


def _write_source_binding_supplement(
    repo_root: Path, run_dir: Path
) -> dict[str, Any]:
    """Bind per-route files without mutating the pre-registered source manifest."""

    runtime_status_path = run_dir / "runtime_profile/route_status.json"
    runtime_status = (
        json.loads(runtime_status_path.read_text(encoding="utf-8"))
        if runtime_status_path.is_file()
        else []
    )
    runtime_by_route = {
        str(item["route"]): item for item in runtime_status if isinstance(item, dict)
    }

    def record(relative: str) -> dict[str, Any]:
        path = repo_root / relative
        if not path.is_file():
            raise FileNotFoundError(f"source-binding artifact is missing: {relative}")
        return {
            "path": relative,
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
        }

    label = record(
        "HiFi_reproduction/runs/"
        "modular_repeatedfilm_4dof_backends_v1_r0corrected_20260803_163500/"
        "manifests/test_labels.parquet"
    )
    routes: dict[str, Any] = {}
    primary = "runs/fair_unified_reranking_20260809_103012"
    for route in ("CROG", "G1", "C1"):
        name = route.lower()
        runtime = runtime_by_route.get(route, {})
        routes[route] = {
            "run_id": "fair_unified_reranking_20260809_103012",
            "formal_status": "COMPLETE/LOCKED",
            "retrospective": False,
            "candidate_top5": record(f"{primary}/02_candidates/{name}_test_top5.parquet"),
            "candidate_all": record(f"{primary}/02_candidates/{name}_test_all.parquet"),
            "features": record(
                f"{primary}/03_features/tracks/T2_matched_common/"
                f"{name}_test/candidate_features.parquet"
            ),
            "labels": label,
            "predictions": record(
                f"{primary}/08_lock/label_free_test_rankers/"
                f"{name}/per_candidate_scores.parquet"
            ),
            "native_decisions": record(
                f"{primary}/08_lock/formal_label_free/{name}_native_decisions.parquet"
            ),
            "raw_decisions": record(
                f"{primary}/08_lock/formal_label_free/{name}_ungated_decisions.parquet"
            ),
            "gated_decisions": record(
                f"{primary}/08_lock/formal_label_free/{name}_gated_decisions.parquet"
            ),
            "formal_metrics": record(
                f"{primary}/09_formal_test/formal_test_metrics.json"
            ),
            "ranker_checkpoints": [
                {
                    "path": str(Path(path).resolve().relative_to(repo_root)),
                    "sha256": sha256_file(Path(path)),
                }
                for path in runtime.get("ranker_model_paths", [])
            ],
            "gate_checkpoint": (
                {
                    "path": str(
                        Path(runtime["gate_model_path"]).resolve().relative_to(repo_root)
                    ),
                    "sha256": sha256_file(Path(runtime["gate_model_path"])),
                }
                if runtime.get("gate_model_path")
                else None
            ),
        }
    d1_root = "runs/fair_d1_reranking_extension_20260811T145515Z"
    d1_runtime = runtime_by_route.get("D1", {})
    routes["D1"] = {
        "run_id": "fair_d1_reranking_extension_20260811T145515Z",
        "formal_status": "COMPLETE/LOCKED retrospective extension",
        "retrospective": True,
        "candidate_top5": record(f"{d1_root}/02_candidates/d1_test_top5.parquet"),
        "candidate_all": record(f"{d1_root}/02_candidates/d1_test_all.parquet"),
        "features": record(
            f"{d1_root}/03_features/test/top5/T2_matched_common/candidate_features.parquet"
        ),
        "labels": label,
        "predictions": record(
            f"{d1_root}/08_lock/label_free_test_rankers/d1/per_candidate_scores.parquet"
        ),
        "formal_outcomes": record(
            f"{d1_root}/09_formal_test/formal_candidate_outcomes.parquet"
        ),
        "formal_metrics": record(f"{d1_root}/09_formal_test/formal_test_metrics.json"),
        "ranker_checkpoints": [
            {
                "path": str(Path(path).resolve().relative_to(repo_root)),
                "sha256": sha256_file(Path(path)),
            }
            for path in d1_runtime.get("ranker_model_paths", [])
        ],
        "gate_checkpoint": (
            {
                "path": str(
                    Path(d1_runtime["gate_model_path"]).resolve().relative_to(repo_root)
                ),
                "sha256": sha256_file(Path(d1_runtime["gate_model_path"])),
            }
            if d1_runtime.get("gate_model_path")
            else None
        ),
    }
    payload = {
        "schema_version": 1,
        "run_id": run_dir.name,
        "purpose": (
            "per-route supplement; source_run_manifest.json remains byte-locked to "
            "its pre-registration hash"
        ),
        "routes": routes,
    }
    atomic_json(run_dir / "source_run_manifest_supplement.json", payload)
    return payload


def _execute_stage(
    stage: str,
    function: Callable[..., dict[str, Any]],
    repo_root: Path,
    run_dir: Path,
    *,
    resume: bool,
) -> dict[str, Any]:
    verify_preregistration(run_dir)
    audit_sources(repo_root, run_dir)
    update_progress(run_dir, stage, "RUNNING")
    try:
        result = function(repo_root, run_dir, resume=resume)
    except Exception as error:
        failure = {
            "stage": stage,
            "status": "FAILED",
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        }
        atomic_json(run_dir / f"{stage}_failure.json", failure)
        update_progress(run_dir, stage, "FAILED")
        raise
    status = str(result.get("status", "COMPLETE"))
    update_progress(run_dir, stage, status)
    atomic_json(run_dir / f"{stage}_result.json", result)
    return result


def _run_command(
    command: str,
    repo_root: Path,
    run_dir: Path,
    *,
    resume: bool,
) -> dict[str, Any]:
    if command == "audit":
        return audit_sources(repo_root, run_dir)
    if command == "preregister":
        return verify_preregistration(run_dir)
    if command == "scene-cv-6d":
        from .scene_cv_6d import run_scene_cv

        return _execute_stage(
            "scene_cv_6d", run_scene_cv, repo_root, run_dir, resume=resume
        )
    if command == "threshold-4d":
        from .four_d import run_threshold_sensitivity

        return _execute_stage(
            "threshold_4d",
            run_threshold_sensitivity,
            repo_root,
            run_dir,
            resume=resume,
        )
    if command == "duplicate-exclusion-4d":
        from .four_d import run_duplicate_exclusion

        return _execute_stage(
            "duplicate_exclusion_4d",
            run_duplicate_exclusion,
            repo_root,
            run_dir,
            resume=resume,
        )
    if command == "topk-4d":
        from .four_d import run_topk_sensitivity

        return _execute_stage(
            "topk_4d", run_topk_sensitivity, repo_root, run_dir, resume=resume
        )
    if command == "profile-runtime":
        from .runtime_profile import run_runtime_profile

        return _execute_stage(
            "runtime_profile",
            run_runtime_profile,
            repo_root,
            run_dir,
            resume=resume,
        )
    if command == "report":
        from .reporting import run_report, write_results_manifest

        verify_preregistration(run_dir)
        audit_sources(repo_root, run_dir)
        _write_source_binding_supplement(repo_root, run_dir)
        result = run_report(repo_root, run_dir)
        status = str(result["status"])
        source_manifest = json.loads(
            (run_dir / "source_run_manifest.json").read_text(encoding="utf-8")
        )
        sources = source_manifest["sources"]
        six_completion = json.loads(
            (run_dir / "6d_scene_cv/completion.json").read_text(encoding="utf-8")
        )
        runtime_completion_path = run_dir / "runtime_profile/completion.json"
        runtime_completion = (
            json.loads(runtime_completion_path.read_text(encoding="utf-8"))
            if runtime_completion_path.is_file()
            else {}
        )
        command_log_path = run_dir / "COMMAND_LOG.md"
        test_report_path = run_dir / "test_report.txt"
        six_schema_path = (
            "artifacts/graspnet6d/20260819_221819_graspnet6d_vgn_lambdamart/"
            "feature_schema.json"
        )
        if test_report_path.is_file() and "OVERALL: PASS" in test_report_path.read_text(
            encoding="utf-8", errors="replace"
        ):
            update_progress(run_dir, "tests", "COMPLETE")
        update_progress(run_dir, "report", status)
        update_manifest(
            run_dir,
            status=status,
            formal_robustness_results_emitted=(status == "COMPLETE_ROBUSTNESS_SUITE"),
            source_run_hashes={
                "graspnet6d_manifest": sources["graspnet6d"]["manifest_sha256"],
                "four_d_primary_lock": sources["four_d_primary"]["lock_sha256"],
                "four_d_primary_formal_manifest": sources["four_d_primary"][
                    "formal_manifest_sha256"
                ],
                "d1_manifest": sources["d1"]["manifest_sha256"],
                "d1_lock": sources["d1"]["lock_sha256"],
            },
            fold_hashes=six_completion["fold_audit"]["fold_manifest_sha256"],
            duplicate_map={
                "status": sources["duplicate_map"]["status"],
                "sha256": sources["duplicate_map"]["sha256"],
            },
            evaluator_sha256=sources["four_d_primary"]["evaluator_sha256"],
            feature_schema_hashes={
                "graspnet6d": six_completion["source_hashes"][six_schema_path],
                "four_d": sources["four_d_primary"]["feature_schema_sha256"],
            },
            hyperparameter_hashes={
                "graspnet6d_locked_config": sources["graspnet6d"][
                    "resolved_config_sha256"
                ],
                "four_d": sources["four_d_primary"]["hyperparameters_sha256"],
            },
            checkpoint_hashes={
                "graspnet6d_vgn": sources["graspnet6d"]["checkpoint_sha256"],
                "four_d_runtime_models": {
                    path: digest
                    for path, digest in runtime_completion.get(
                        "source_hashes", {}
                    ).items()
                    if Path(path).suffix in {".pkl", ".txt"}
                },
            },
            runtime_source_hashes=runtime_completion.get("source_hashes", {}),
            runtime_completion_signature_sha256=runtime_completion.get(
                "completion_signature_sha256"
            ),
            source_run_manifest_supplement_sha256=sha256_file(
                run_dir / "source_run_manifest_supplement.json"
            ),
            command_log_sha256=(
                sha256_file(command_log_path) if command_log_path.is_file() else None
            ),
            ended_at_utc=datetime.now(timezone.utc).isoformat(),
        )
        atomic_json(run_dir / "report_result.json", result)
        write_results_manifest(run_dir, status)
        return result
    raise ValueError(f"unknown command: {command}")


def _run_all(repo_root: Path, run_dir: Path, *, resume: bool) -> dict[str, Any]:
    results: dict[str, Any] = {}
    stages = (
        "audit",
        "preregister",
        "scene-cv-6d",
        "threshold-4d",
        "duplicate-exclusion-4d",
        "topk-4d",
        "profile-runtime",
    )
    for stage in stages:
        try:
            results[stage] = _run_command(
                stage, repo_root, run_dir, resume=resume
            )
        except Exception as error:
            results[stage] = {
                "status": "FAILED",
                "error_type": type(error).__name__,
                "error": str(error),
            }
    results["report"] = _run_command(
        "report", repo_root, run_dir, resume=resume
    )
    atomic_json(run_dir / "all_result.json", results)
    return results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m robustness_suite.cli",
        description="Post-hoc robustness analyses over immutable frozen sources.",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    for name in (
        "audit",
        "preregister",
        "scene-cv-6d",
        "threshold-4d",
        "duplicate-exclusion-4d",
        "topk-4d",
        "profile-runtime",
        "report",
        "all",
    ):
        child = commands.add_parser(name)
        child.add_argument("--run-id", default=RUN_ID)
        child.add_argument("--resume", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    repo_root = _repo_root()
    run_dir = _run_dir(repo_root, arguments.run_id)
    record_command(run_dir, [sys.executable, "-m", "robustness_suite.cli", *(argv or sys.argv[1:])])
    try:
        result = (
            _run_all(repo_root, run_dir, resume=arguments.resume)
            if arguments.command == "all"
            else _run_command(
                arguments.command,
                repo_root,
                run_dir,
                resume=arguments.resume,
            )
        )
        if arguments.command == "all":
            from .reporting import write_results_manifest

            write_results_manifest(
                run_dir, str(result["report"]["status"])
            )
    except Exception as error:
        print(f"ERROR: {type(error).__name__}: {error}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
