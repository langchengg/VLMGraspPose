#!/usr/bin/env python3
"""Run every locked predicted-mask and GT-mask-oracle formal test exactly once."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd
import pyarrow.parquet as pq


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.grasping.common.experiment_lock import verify_lock  # noqa: E402
from src.grasping.common.results import (  # noqa: E402
    assert_aggregate_matches_sample_rows,
    evaluate_prediction_records,
)
from src.grasping.common.types import Grasp4DoF, GraspPrediction  # noqa: E402
from tools.grasp4dof.recompute_reference import (  # noqa: E402
    RECOMPUTE_CONTRACT,
    _legacy_outcome_comparison,
)


METHODS = ("G0", "G1", "C0", "C1", "A0")
EXPECTED_TEST_COUNT = 7675


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def locked_test_sample_ids(
    lock: Mapping[str, Any], *, expected_count: int = EXPECTED_TEST_COUNT
) -> tuple[str, ...]:
    run = Path(str(lock.get("run_dir", ""))).resolve()
    record = lock.get("artifacts", {}).get("test_samples")
    if not isinstance(record, Mapping):
        raise ValueError("locked test-sample artifact is missing")
    path = (run / str(record.get("path", ""))).resolve()
    if (
        not path.is_file()
        or path.stat().st_size != int(record.get("bytes", -1))
        or _sha256(path) != record.get("sha256")
    ):
        raise ValueError("locked test-sample manifest drifted")
    ids = [
        str(value)
        for value in pq.read_table(path, columns=["sample_id"])
        .column("sample_id")
        .to_pylist()
    ]
    if len(ids) != expected_count or len(set(ids)) != expected_count:
        raise ValueError("locked test-sample manifest violates the identity contract")
    return tuple(ids)


def validate_r0_reference_evidence(
    directory: Path,
    *,
    lock: Mapping[str, Any],
    expected_count: int = EXPECTED_TEST_COUNT,
) -> None:
    """Bind an R0 output to both frozen retained inputs and its exact recompute."""

    evidence = _json(directory / "independent_reference_recompute.json")
    run_config = _json(directory / "run_config.json")
    metrics = _json(directory / "metrics.json")
    reference = lock.get("lineage", {}).get("reference")
    if not isinstance(reference, Mapping):
        raise ValueError("locked R0 reference lineage is missing")
    direct_hashes = reference.get("direct_inputs")
    if not isinstance(direct_hashes, Mapping) or set(direct_hashes) != {
        "per_candidate",
        "per_sample",
    }:
        raise ValueError("locked R0 direct-input lineage is incomplete")
    artifact_name = reference.get("manifest_artifact")
    manifest_artifact = lock.get("artifacts", {}).get(artifact_name)
    if not isinstance(manifest_artifact, Mapping):
        raise ValueError("locked R0 preflight artifact is missing")
    run = directory.resolve().parents[1]
    manifest_path = (run / str(manifest_artifact.get("path", ""))).resolve()
    if not manifest_path.is_file() or _sha256(manifest_path) != manifest_artifact.get(
        "sha256"
    ):
        raise ValueError("locked R0 preflight artifact drifted")
    preflight = _json(manifest_path)
    preflight_r0 = preflight.get("r0")
    if (
        not isinstance(preflight_r0, Mapping)
        or int(preflight_r0.get("sample_count", -1)) != expected_count
        or Path(str(preflight_r0.get("reference_run", ""))).resolve()
        != Path(str(reference.get("reference_run", ""))).resolve()
    ):
        raise ValueError("locked R0 preflight lineage mismatch")
    direct_records = preflight_r0.get("direct_inputs")
    if not isinstance(direct_records, Mapping) or set(direct_records) != set(
        direct_hashes
    ):
        raise ValueError("locked R0 preflight direct inputs are incomplete")
    for name, expected_sha in direct_hashes.items():
        record = direct_records[name]
        if not isinstance(record, Mapping):
            raise ValueError(f"locked R0 direct-input record is malformed: {name}")
        source_path = Path(str(record.get("path", ""))).resolve()
        if (
            record.get("sha256") != expected_sha
            or not source_path.is_file()
            or source_path.stat().st_size != int(record.get("bytes", -1))
            or _sha256(source_path) != expected_sha
        ):
            raise ValueError(f"locked R0 direct-input hash mismatch: {name}")
    if (
        evidence.get("status") != "FORMAL_RECOMPUTE_COMPLETE"
        or int(evidence.get("sample_count", -1)) != expected_count
        or int(evidence.get("saved_output_mismatch_count", -1)) != 0
        or evidence.get("metrics") != metrics
        or evidence.get("source_candidate_sha256") != direct_hashes["per_candidate"]
        or evidence.get("source_sample_sha256") != direct_hashes["per_sample"]
        or evidence.get("output_sample_sha256")
        != _sha256(directory / "per_sample_predictions.parquet")
        or evidence.get("output_candidate_sha256")
        != _sha256(directory / "per_candidate_predictions.parquet")
        or Path(str(evidence.get("source_candidate_path", ""))).resolve()
        != Path(str(direct_records["per_candidate"].get("path", ""))).resolve()
        or Path(str(evidence.get("source_sample_path", ""))).resolve()
        != Path(str(direct_records["per_sample"].get("path", ""))).resolve()
        or evidence.get("recompute_contract") != RECOMPUTE_CONTRACT
        or evidence.get("source_candidates_status") != "EXACT_REUSE"
        or evidence.get("formal_metrics_status") != "RECOMPUTED_WITH_LOCKED_EVALUATOR"
        or evidence.get("legacy_success_fields_used_for_formal_metrics") is not False
    ):
        raise ValueError("R0 independent reference evidence mismatch")
    if (
        run_config.get("reference_manifest_sha256") != manifest_artifact.get("sha256")
        or Path(str(run_config.get("reference_manifest", ""))).resolve()
        != manifest_path
        or run_config.get("source_candidate_sha256") != direct_hashes["per_candidate"]
        or run_config.get("source_sample_sha256") != direct_hashes["per_sample"]
    ):
        raise ValueError("R0 run config is not bound to locked retained inputs")

    required_source_columns = {
        "sample_id",
        "sample_index",
        "gqcnn_rank",
        "candidate_id",
        "center_u_px",
        "center_v_px",
        "angle_rad",
        "configured_width_px",
        "gqcnn_q_value",
        "source_candidate_index",
        "candidate_seed",
    }
    required_legacy_columns = {
        "sample_id",
        "sample_index",
        "mask_inference_seconds",
        "candidate_generation_time_ms",
        "gqcnn_total_time_ms",
        "raw_candidate_count",
        "failure_category",
        "top1_correct",
        "top5_correct",
        "oracle_all",
    }
    candidate_source = pd.read_parquet(
        Path(str(direct_records["per_candidate"]["path"])).resolve()
    )
    legacy = pd.read_csv(Path(str(direct_records["per_sample"]["path"])).resolve())
    if not required_source_columns.issubset(candidate_source.columns) or not (
        required_legacy_columns.issubset(legacy.columns)
    ):
        raise ValueError("R0 retained input schema mismatch")
    if legacy["sample_id"].astype(str).duplicated().any():
        raise ValueError("R0 retained per-sample IDs are duplicated")
    run_root = Path(str(lock.get("run_dir", ""))).resolve()
    samples_record = lock.get("artifacts", {}).get("test_samples")
    labels_record = lock.get("artifacts", {}).get("test_labels")
    if not isinstance(samples_record, Mapping) or not isinstance(
        labels_record, Mapping
    ):
        raise ValueError("R0 locked test manifests are missing")
    samples_path = (run_root / str(samples_record.get("path", ""))).resolve()
    labels_path = (run_root / str(labels_record.get("path", ""))).resolve()
    if (
        not samples_path.is_file()
        or samples_path.stat().st_size != int(samples_record.get("bytes", -1))
        or _sha256(samples_path) != samples_record.get("sha256")
        or not labels_path.is_file()
        or labels_path.stat().st_size != int(labels_record.get("bytes", -1))
        or _sha256(labels_path) != labels_record.get("sha256")
    ):
        raise ValueError("R0 locked test labels drifted")
    deployment = pq.read_table(samples_path).to_pylist()
    labels = pq.read_table(labels_path).to_pylist()
    labels_by_id = {str(row["sample_id"]): row for row in labels}
    expected_ids = {str(row["sample_id"]) for row in deployment}
    if (
        len(labels_by_id) != expected_count
        or set(labels_by_id) != expected_ids
        or set(legacy["sample_id"].astype(str)) != expected_ids
        or any("gt_grasp_rectangles" not in row for row in labels)
    ):
        raise ValueError("R0 retained/current sample-label coverage mismatch")
    unknown_candidate_ids = (
        set(candidate_source["sample_id"].astype(str)) - expected_ids
    )
    if unknown_candidate_ids:
        raise ValueError("R0 retained candidates contain unknown sample IDs")
    grouped = {
        str(sample_id): frame.sort_values(
            ["gqcnn_rank", "candidate_id"], kind="mergesort"
        )
        for sample_id, frame in candidate_source.groupby("sample_id", sort=False)
    }
    legacy_by_id = {str(row.sample_id): row for row in legacy.itertuples(index=False)}
    saved_samples = pq.read_table(
        directory / "per_sample_predictions.parquet"
    ).to_pylist()
    saved_candidates = pq.read_table(
        directory / "per_candidate_predictions.parquet"
    ).to_pylist()
    saved_sample_by_id = {str(row["sample_id"]): row for row in saved_samples}
    saved_candidate_by_id: dict[str, list[dict[str, Any]]] = {}
    for row in saved_candidates:
        saved_candidate_by_id.setdefault(str(row["sample_id"]), []).append(row)
    for rows in saved_candidate_by_id.values():
        rows.sort(key=lambda row: int(row["rank"]))

    def same_value(observed: Any, expected: Any) -> bool:
        if expected is None:
            return observed is None or bool(pd.isna(observed))
        if isinstance(expected, float):
            try:
                return math.isclose(
                    float(observed), expected, rel_tol=0.0, abs_tol=1e-12
                )
            except (TypeError, ValueError):
                return False
        return observed == expected

    recomputed_candidate_count = 0
    legacy_comparisons: list[tuple[str, bool, bool, bool, bool, bool, bool]] = []
    reference_root = Path(str(reference["reference_run"])).resolve()
    for row in deployment:
        sample_id = str(row["sample_id"])
        old = legacy_by_id[sample_id]
        source_frame = grouped.get(sample_id)
        pool: list[Grasp4DoF] = []
        if source_frame is not None:
            for source in source_frame.itertuples(index=False):
                pool.append(
                    Grasp4DoF(
                        center_x=float(source.center_u_px),
                        center_y=float(source.center_v_px),
                        angle_deg=math.degrees(float(source.angle_rad)),
                        width_px=float(source.configured_width_px),
                        height_px=20.0,
                        score=float(source.gqcnn_q_value),
                        candidate_id=str(source.candidate_id),
                        metadata={
                            "source_candidate_index": int(
                                source.source_candidate_index
                            ),
                            "candidate_seed": int(source.candidate_seed),
                            "source_reference_run": str(reference_root),
                        },
                    )
                )
        latency = (
            float(old.mask_inference_seconds)
            + float(old.candidate_generation_time_ms) / 1000.0
            + float(old.gqcnn_total_time_ms) / 1000.0
        )
        prediction = GraspPrediction(
            sample_id=sample_id,
            backend="dexnet_gqcnn_reference",
            conditioning_variant="repeatedfilm_predicted_mask",
            raw_candidate_count=int(old.raw_candidate_count),
            nms_candidate_count=len(pool),
            top1=pool[0] if pool else None,
            top5=tuple(pool[:5]),
            candidates=tuple(pool),
            empty_reason=None if pool else str(old.failure_category),
            runtime_seconds=latency,
            device="retained_reference_macos_cpu",
            metadata={"source_reference_run": str(reference_root)},
        )
        recomputed_sample, recomputed_candidates = evaluate_prediction_records(
            method="repeatedfilm_dexnet_gqcnn_reference",
            prediction=prediction,
            label=labels_by_id[sample_id],
        )
        legacy_comparisons.append(
            (
                sample_id,
                bool(old.top1_correct),
                bool(old.top5_correct),
                bool(old.oracle_all),
                bool(recomputed_sample["j_at_1"]),
                bool(recomputed_sample["j_at_5"]),
                bool(recomputed_sample["candidate_pool_oracle"]),
            )
        )
        observed_sample = saved_sample_by_id.get(sample_id)
        observed_candidates = saved_candidate_by_id.get(sample_id, [])
        if not isinstance(observed_sample, Mapping) or len(observed_candidates) != len(
            recomputed_candidates
        ):
            raise ValueError(f"R0 saved output coverage mismatch: {sample_id}")
        for key, expected in recomputed_sample.items():
            if key not in observed_sample or not same_value(
                observed_sample[key], expected
            ):
                raise ValueError(
                    f"R0 saved sample differs from recompute: {sample_id}/{key}"
                )
        for rank, (observed, expected) in enumerate(
            zip(observed_candidates, recomputed_candidates, strict=True), 1
        ):
            for key, expected_value in expected.items():
                if key not in observed or not same_value(observed[key], expected_value):
                    raise ValueError(
                        f"R0 saved candidate differs from recompute: {sample_id}/{rank}/{key}"
                    )
        recomputed_candidate_count += len(recomputed_candidates)
    if (
        recomputed_candidate_count != len(saved_candidates)
        or int(evidence.get("candidate_count", -1)) != recomputed_candidate_count
    ):
        raise ValueError("R0 saved candidate count differs from recompute")
    if evidence.get("legacy_outcome_comparison") != _legacy_outcome_comparison(
        legacy_comparisons
    ):
        raise ValueError(
            "R0 legacy-outcome compatibility evidence differs from recompute"
        )


def _complete(
    directory: Path,
    *,
    method_id: str,
    oracle: bool,
    lock: Mapping[str, Any],
    config: Path | None = None,
) -> bool:
    marker = directory / "COMPLETE.json"
    if not marker.is_file():
        return False
    value = _json(marker)
    if (
        value.get("schema_version") != 2
        or value.get("status") != "COMPLETE"
        or value.get("method_id") != method_id
        or value.get("split") != "test"
        or value.get("oracle") is not oracle
        or value.get("experiment_lock_sha256") != lock["manifest_content_sha256"]
    ):
        raise ValueError(f"invalid formal completion marker: {marker}")
    if int(value.get("sample_count", -1)) != EXPECTED_TEST_COUNT:
        raise ValueError(f"formal completion count mismatch: {marker}")
    names = (
        "metrics.json",
        "runtime_metrics.json",
        "memory_metrics.json",
        "run_config.json",
        "per_sample_predictions.parquet",
        "per_candidate_predictions.parquet",
    )
    if method_id == "R0":
        names = (*names, "independent_reference_recompute.json")
    if not all((directory / name).is_file() for name in names):
        raise ValueError(f"formal completion marker has missing outputs: {directory}")
    declared = value.get("artifacts")
    if not isinstance(declared, Mapping) or any(
        declared.get(name) != _sha256(directory / name) for name in names
    ):
        raise ValueError(f"formal completion artifact digest mismatch: {directory}")
    run_config = _json(directory / "run_config.json")
    locked_samples_sha = lock["artifacts"]["test_samples"]["sha256"]
    locked_labels_sha = lock["artifacts"]["test_labels"]["sha256"]
    if (
        run_config.get("method_id") != method_id
        or run_config.get("split") != "test"
        or run_config.get("oracle") is not oracle
        or run_config.get("experiment_lock_sha256") != lock["manifest_content_sha256"]
        or run_config.get("samples_manifest_sha256") != locked_samples_sha
        or run_config.get("labels_manifest_sha256") != locked_labels_sha
        or run_config.get("per_sample_sha256")
        != _sha256(directory / "per_sample_predictions.parquet")
        or run_config.get("per_candidate_sha256")
        != _sha256(directory / "per_candidate_predictions.parquet")
    ):
        raise ValueError(f"formal run provenance mismatch: {directory}")
    if config is not None and (
        run_config.get("config_sha256") != _sha256(config)
        or value.get("config_sha256") != _sha256(config)
    ):
        raise ValueError(f"formal selected-config mismatch: {directory}")
    sample_table = pq.read_table(directory / "per_sample_predictions.parquet")
    if sample_table.num_rows != EXPECTED_TEST_COUNT:
        raise ValueError(f"formal per-sample row count mismatch: {directory}")
    sample_ids = sample_table.column("sample_id").to_pylist()
    if len(set(sample_ids)) != EXPECTED_TEST_COUNT or tuple(
        str(value) for value in sample_ids
    ) != locked_test_sample_ids(lock, expected_count=EXPECTED_TEST_COUNT):
        raise ValueError(f"formal per-sample identity mismatch: {directory}")
    pq.ParquetFile(directory / "per_candidate_predictions.parquet")
    metrics = _json(directory / "metrics.json")
    try:
        assert_aggregate_matches_sample_rows(sample_table.to_pylist(), metrics)
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"formal metrics/per-sample mismatch: {directory}") from error
    if method_id == "R0":
        validate_r0_reference_evidence(
            directory, lock=lock, expected_count=EXPECTED_TEST_COUNT
        )
    return True


def _run(command: list[str], *, log: Path, command_log: Path) -> None:
    rendered = shlex.join(command)
    timestamp = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    with command_log.open("a", encoding="utf-8") as stream:
        stream.write(f"{timestamp}\t{rendered}\n")
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("x", encoding="utf-8") as stream:
        subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=True,
        )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    expected_prefix = (PROJECT_ROOT / ".venv-grasp4dof").resolve()
    if Path(sys.prefix).resolve() != expected_prefix:
        raise RuntimeError(
            f"formal inference requires isolated environment {expected_prefix}; "
            f"observed {Path(sys.prefix).resolve()}"
        )
    run = args.run_dir.expanduser().resolve()
    lock = verify_lock(run)
    selected = _json(run / "selected_configs.json")
    if set(selected) != set(METHODS):
        raise ValueError("locked selected configs must contain G0,G1,C0,C1,A0")
    command_log = run / "commands.log"
    # Preserve the venv launcher path. Resolving this symlink would execute the
    # base interpreter and silently drop the isolated site-packages/sys.prefix.
    python = sys.executable

    reference_output = run / "formal_test/R0"
    if not _complete(reference_output, method_id="R0", oracle=False, lock=lock):
        if reference_output.exists() and any(reference_output.iterdir()):
            raise RuntimeError(
                "partial R0 output requires audit; refusing to overwrite"
            )
        _run(
            [
                python,
                str(PROJECT_ROOT / "tools/grasp4dof/recompute_reference.py"),
                "--run-dir",
                str(run),
                "--output-dir",
                str(reference_output),
            ],
            log=run / "logs/formal_R0.log",
            command_log=command_log,
        )
        if not _complete(reference_output, method_id="R0", oracle=False, lock=lock):
            raise RuntimeError("R0 command returned without complete outputs")

    for oracle in (False, True):
        for method_id in METHODS:
            label = f"{method_id}-O" if oracle else method_id
            output = run / (f"oracle/{label}" if oracle else f"formal_test/{label}")
            config = Path(str(selected[method_id]["path"])).resolve()
            if _complete(
                output,
                method_id=method_id,
                oracle=oracle,
                lock=lock,
                config=config,
            ):
                continue
            if output.exists() and any(output.iterdir()):
                raise RuntimeError(
                    f"partial {label} output requires audit; refusing overwrite"
                )
            command = [
                python,
                str(PROJECT_ROOT / "tools/grasp4dof/run_method.py"),
                "--run-dir",
                str(run),
                "--split",
                "test",
                "--method",
                method_id,
                "--config",
                str(config),
                "--output-dir",
                str(output),
            ]
            if oracle:
                command.append("--oracle")
            _run(
                command,
                log=run / f"logs/formal_{label}.log",
                command_log=command_log,
            )
            if not _complete(
                output,
                method_id=method_id,
                oracle=oracle,
                lock=lock,
                config=config,
            ):
                raise RuntimeError(f"{label} returned without complete outputs")
    print(
        json.dumps(
            {
                "status": "COMPLETE",
                "lock_sha256": lock["manifest_content_sha256"],
                "predicted_methods": list(METHODS),
                "oracle_methods": [f"{item}-O" for item in METHODS],
                "reference": "R0",
                "sample_count_per_method": EXPECTED_TEST_COUNT,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
