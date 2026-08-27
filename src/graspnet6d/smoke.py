"""Truthful Go/No-Go gate: diagnostics never substitute for dataset checks."""

from __future__ import annotations

import csv
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .audit import repository_root, sha256_file
from .dataset import discover_scene_ids, validate_graspnet_structure
from .io import canonical_sha256
from .provenance import atomic_json, atomic_text
from .vgn import EXPECTED_CHECKPOINT_SHA256, benchmark_checkpoint, checkpoint_sha256, load_frozen_vgn, run_vgn


REGRESSION_STATUS_SCHEMA_VERSION = 1
REGRESSION_STATUS_FILENAME = "4d_regression_status.json"
REGRESSION_EVIDENCE_KIND = "full_4dof_regression_suite"
REAL_RANKER_SMOKE_SCHEMA = "graspnet6d_real_ranker_integration_v2"
SMOKE_GEOMETRY_GROUPS = 8
SMOKE_EVALUATOR_GROUPS = 8
SMOKE_MINIMUM_NONEMPTY_CANDIDATE_GROUPS = 4
RANKER_INTEGRATION_MINIMUM_GROUPS = 20
SMOKE_GROUNDING_CONDITIONS = (
    "oracle_gt_mask",
    "hifics_zero_shot_mask",
    "hifics_adapted_mask",
)


@dataclass(frozen=True)
class Check:
    number: int
    name: str
    passed: bool
    evidence: str
    kind: str = "formal_smoke_gate"


def analytic_diagnostic_tsdf(offset: float = 0.0) -> np.ndarray:
    """A deterministic bounded tensor for device diagnostics only, never data."""

    axis = (np.arange(40, dtype=np.float32) + 0.5) / 40.0
    x, y, z = np.meshgrid(axis, axis, axis, indexing="ij")
    signed = np.sqrt((x - 0.5 - offset) ** 2 + (y - 0.5) ** 2 + (z - 0.5) ** 2) - 0.22
    return np.clip(signed / 0.03, -1.0, 1.0)[None].astype(np.float32)


def run_device_benchmark(output_root: Path | None = None) -> dict[str, Any]:
    root = repository_root()
    output = output_root or root / "artifacts" / "graspnet6d"
    tensors = [analytic_diagnostic_tsdf(0.0), analytic_diagnostic_tsdf(0.025)]
    cpu, mps, decision = benchmark_checkpoint(tensors, warmup=1, repeats=3)
    rows = [cpu.to_record(), mps.to_record()]
    output.mkdir(parents=True, exist_ok=True)
    csv_path = output / "device_benchmark.csv"
    temporary = csv_path.with_name(f".{csv_path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, csv_path)
    text = f"""# Device decision

Formal inference device: **{decision.device.upper()}**

Reason: {decision.reason}

This benchmark used the same immutable VGN checkpoint and two deterministic
analytic 40³ diagnostic tensors on CPU and MPS. They are not GraspNet samples
and their outputs are not experiment results. A real cached-TSDF parity rerun
is still required after data preparation; CPU remains the fail-closed choice.

- CPU median latency: {cpu.median_latency_s}
- CPU p95 latency: {cpu.p95_latency_s}
- MPS median latency: {mps.median_latency_s}
- MPS p95 latency: {mps.p95_latency_s}
- MPS max / mean absolute error: {mps.max_absolute_error} / {mps.mean_absolute_error}
- MPS fallback detected: {getattr(mps, 'fallback_detected', False)}
- MPS fallback details: {getattr(mps, 'fallback_details', None)}
"""
    atomic_text(output / "device_decision.md", text)
    return {"cpu": rows[0], "mps": rows[1], "decision": asdict(decision)}


def _vgn_finite_check(tsdf_path: Path | None = None) -> Check:
    try:
        digest = checkpoint_sha256()
        if digest != EXPECTED_CHECKPOINT_SHA256:
            return Check(5, "VGN output finite", False, f"checkpoint hash mismatch: {digest}")
        model = load_frozen_vgn(device="cpu")
        if tsdf_path is None:
            tensor = analytic_diagnostic_tsdf()
            evidence = (
                "real pretrained checkpoint on deterministic diagnostic tensor; "
                "not a dataset result"
            )
            kind = "data_independent_diagnostic"
        else:
            with np.load(tsdf_path, allow_pickle=False) as archive:
                tensor = np.asarray(archive["tsdf"], dtype=np.float32)
            evidence = f"real pretrained checkpoint on cached GraspNet TSDF: {tsdf_path}"
            kind = "formal_smoke_gate"
        outputs = run_vgn(tensor, model, device="cpu")
        finite = all(
            np.isfinite(value).all()
            for value in (outputs.quality, outputs.rotation_xyzw, outputs.width_voxels)
        )
        return Check(
            5,
            "VGN output finite",
            bool(finite),
            evidence,
            kind=kind,
        )
    except Exception as error:
        return Check(5, "VGN output finite", False, f"{type(error).__name__}: {error}", kind="data_independent_diagnostic")


def validate_4d_regression_status(regression_root: Path | None = None) -> Check:
    """Validate machine-readable, non-fixture evidence for the legacy 4-DoF suite."""

    root = repository_root()
    directory = regression_root or root / "artifacts" / "graspnet6d" / "regression"
    status_path = directory / REGRESSION_STATUS_FILENAME

    def failure(evidence: str) -> Check:
        return Check(12, "Old 4-DoF regression passes", False, evidence)

    if not status_path.is_file() or status_path.is_symlink():
        return failure(f"missing regular regression status: {status_path}")
    try:
        payload = json.loads(status_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        return failure(f"invalid regression status JSON: {type(error).__name__}: {error}")
    if not isinstance(payload, dict):
        return failure("invalid regression status JSON: root must be an object")
    required = {
        "schema_version",
        "evidence_kind",
        "status",
        "exit_code",
        "test_count",
        "fixture_only",
        "placeholder",
        "report_path",
        "report_sha256",
    }
    missing = sorted(required - set(payload))
    if missing:
        return failure(f"regression status missing fields: {', '.join(missing)}")
    if payload["schema_version"] != REGRESSION_STATUS_SCHEMA_VERSION:
        return failure(f"unsupported regression status schema: {payload['schema_version']!r}")
    if payload["evidence_kind"] != REGRESSION_EVIDENCE_KIND:
        return failure(f"regression evidence is not the full suite: {payload['evidence_kind']!r}")
    if payload["status"] != "PASS":
        return failure(f"regression status is not PASS: {payload['status']!r}")
    exit_code = payload["exit_code"]
    if isinstance(exit_code, bool) or not isinstance(exit_code, int) or exit_code != 0:
        return failure(f"regression exit_code is not integer zero: {exit_code!r}")
    test_count = payload["test_count"]
    if isinstance(test_count, bool) or not isinstance(test_count, int) or test_count <= 0:
        return failure(f"regression test_count is not a positive integer: {test_count!r}")
    if payload["fixture_only"] is not False:
        return failure("fixture-only regression evidence is forbidden")
    if payload["placeholder"] is not False:
        return failure("placeholder regression evidence is forbidden")
    report_value = payload["report_path"]
    if not isinstance(report_value, str) or not report_value:
        return failure("regression report_path must be a non-empty filename")
    relative_report = Path(report_value)
    if relative_report.is_absolute() or relative_report.name != report_value:
        return failure("regression report_path must be a filename inside the regression directory")
    report_path = directory / relative_report
    if not report_path.is_file() or report_path.is_symlink():
        return failure(f"missing regular regression report: {report_path}")
    try:
        report_text = report_path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as error:
        return failure(f"unreadable regression report: {type(error).__name__}: {error}")
    normalised = report_text.strip().casefold()
    if not normalised or normalised in {"placeholder", "todo", "tbd", "fixture"}:
        return failure("regression report is empty or a placeholder")
    expected_hash = payload["report_sha256"]
    if (
        not isinstance(expected_hash, str)
        or len(expected_hash) != 64
        or any(character not in "0123456789abcdef" for character in expected_hash)
    ):
        return failure("regression report_sha256 is not a lowercase SHA-256 digest")
    actual_hash = sha256_file(report_path)
    if actual_hash != expected_hash:
        return failure(
            f"regression report hash mismatch: expected {expected_hash}, observed {actual_hash}"
        )
    return Check(
        12,
        "Old 4-DoF regression passes",
        True,
        f"validated {test_count} tests; report={report_path}; sha256={actual_hash}",
        kind="data_independent_regression",
    )


def run_real_ranker_smoke(
    training_rows_path: Path | str,
    validation_rows_path: Path | str,
    test_rows_path: Path | str,
    group_universe_paths: tuple[Path | str, Path | str, Path | str],
    feature_schema_path: Path | str,
    output_root: Path | str,
    *,
    minimum_real_groups: int = RANKER_INTEGRATION_MINIMUM_GROUPS,
    resume: bool = False,
) -> dict[str, Any]:
    """Fit/predict the real 6-DoF adapter without creating formal results.

    This is deliberately a small smoke-only fit.  Its evidence can satisfy
    Go/No-Go check 10, but is marked non-reportable and is never accepted by
    the formal analysis/report entry point.
    """

    if minimum_real_groups < RANKER_INTEGRATION_MINIMUM_GROUPS:
        raise ValueError(
            "ranker integration is a separate real-data gate and requires at least "
            f"{RANKER_INTEGRATION_MINIMUM_GROUPS} unique nonempty groups; "
            f"got minimum_real_groups={minimum_real_groups}"
        )

    import pandas as pd

    from .features import StableMissingValueImputer, load_feature_schema
    from .ranker import ValidationData, contiguous_group_sizes, fit_ranker, predict_scores

    def table(path: Path | str) -> tuple[Path, pd.DataFrame]:
        source = Path(path).expanduser().resolve()
        if not source.is_file():
            raise ValueError(f"real-ranker smoke table is absent: {source}")
        if source.suffix.lower() in {".parquet", ".pq"}:
            frame = pd.read_parquet(source)
        elif source.suffix.lower() == ".csv":
            frame = pd.read_csv(source)
        else:
            raise ValueError(f"unsupported real-ranker smoke table: {source}")
        return source, frame

    train_path, train = table(training_rows_path)
    validation_path, validation = table(validation_rows_path)
    test_path, test = table(test_rows_path)
    universes: list[pd.DataFrame] = []
    universe_sources: list[Path] = []
    for raw in group_universe_paths:
        path, frame = table(raw)
        universe_sources.append(path)
        universes.append(frame)
    schema_path = Path(feature_schema_path).expanduser().resolve()
    feature_columns = tuple(spec.name for spec in load_feature_schema(schema_path))
    required = {
        "partition",
        "scene_id",
        "group_id",
        "candidate_id",
        "relevance",
        *feature_columns,
    }
    frames = {"train": train, "validation": validation, "test": test}
    for partition, frame in frames.items():
        missing = sorted(required - set(frame.columns))
        if missing:
            raise ValueError(f"{partition} smoke rows lack columns: {missing}")
        if set(frame["partition"].astype(str)) != {partition}:
            raise ValueError(f"{partition} smoke rows contain another split")
        if frame["candidate_id"].astype(str).duplicated().any():
            raise ValueError(f"{partition} smoke rows duplicate candidate IDs")
    scene_sets = {
        name: set(frame["scene_id"].astype(str)) for name, frame in frames.items()
    }
    if any(
        scene_sets[left] & scene_sets[right]
        for left, right in (("train", "validation"), ("train", "test"), ("validation", "test"))
    ):
        raise ValueError("real-ranker smoke scenes overlap across splits")
    nonempty_groups = set().union(
        *(set(frame["group_id"].astype(str)) for frame in frames.values())
    )
    universe_groups = set().union(
        *(set(frame["group_id"].astype(str)) for frame in universes)
    )
    if not nonempty_groups.issubset(universe_groups):
        raise ValueError("real-ranker smoke rows lie outside the declared universe")
    if len(nonempty_groups) < int(minimum_real_groups):
        raise ValueError(
            f"real-ranker smoke requires {minimum_real_groups} nonempty groups, "
            f"got {len(nonempty_groups)}"
        )
    sort = ["group_id", "native_rank", "candidate_id"]
    train = train.sort_values(sort, kind="mergesort").reset_index(drop=True)
    validation = validation.sort_values(sort, kind="mergesort").reset_index(drop=True)
    test = test.sort_values(sort, kind="mergesort").reset_index(drop=True)
    imputer = StableMissingValueImputer()
    train_x = imputer.fit_transform(train[list(feature_columns)])
    validation_x = imputer.transform(validation[list(feature_columns)])
    test_x = imputer.transform(test[list(feature_columns)])
    model = fit_ranker(
        train_x.to_numpy(float),
        train["relevance"].to_numpy(),
        contiguous_group_sizes(train["group_id"], length=len(train)),
        ValidationData(
            validation_x.to_numpy(float),
            validation["relevance"].to_numpy(),
            contiguous_group_sizes(validation["group_id"], length=len(validation)),
        ),
        {
            "seed": 20260815,
            "early_stopping_rounds": 5,
            "num_leaves": 7,
            "learning_rate": 0.1,
            "n_estimators": 25,
            "min_child_samples": 1,
            "feature_fraction": 1.0,
        },
    )
    scores = predict_scores(
        model,
        test_x.to_numpy(float),
        contiguous_group_sizes(test["group_id"], length=len(test)),
        candidate_ids=test["candidate_id"].astype(str),
    )
    predictions = test[
        ["partition", "scene_id", "group_id", "candidate_id", "geometry_sha256", "native_rank", "native_score"]
    ].copy()
    predictions["smoke_rerank_score"] = scores
    output = Path(output_root).expanduser().resolve()
    prediction_path = output / "real_ranker_predictions.csv"
    evidence_path = output / "real_ranker_smoke_evidence.json"
    source_hashes = {
        str(path): sha256_file(path)
        for path in (
            train_path,
            validation_path,
            test_path,
            *universe_sources,
            schema_path,
        )
    }
    fingerprint = canonical_sha256(
        {
            "schema": REAL_RANKER_SMOKE_SCHEMA,
            "source_hashes": source_hashes,
            "minimum_real_groups": int(minimum_real_groups),
        }
    )
    if evidence_path.is_file():
        saved = json.loads(evidence_path.read_text(encoding="utf-8"))
        if not resume:
            raise ValueError(f"real-ranker smoke evidence exists: {evidence_path}")
        if (
            saved.get("input_fingerprint") != fingerprint
            or not prediction_path.is_file()
            or saved.get("predictions_sha256") != sha256_file(prediction_path)
        ):
            raise ValueError("real-ranker smoke evidence is stale")
        return saved
    _atomic_csv_frame(prediction_path, predictions)
    evidence = {
        "schema_version": REAL_RANKER_SMOKE_SCHEMA,
        "status": "PASS",
        "scope": "real_data_ranker_integration_only",
        "evidence_kind": "real_data_ranker_integration",
        "fixture_only": False,
        "formal_report_eligible": False,
        "input_fingerprint": fingerprint,
        "source_hashes": source_hashes,
        "minimum_real_groups": int(minimum_real_groups),
        "nonempty_group_count": len(nonempty_groups),
        "nonempty_group_count_by_partition": {
            name: int(frame["group_id"].astype(str).nunique())
            for name, frame in frames.items()
        },
        "universe_group_count": len(universe_groups),
        "training_rows": len(train),
        "validation_rows": len(validation),
        "test_rows": len(test),
        "test_candidate_ids_unchanged": predictions["candidate_id"].astype(str).tolist()
        == test["candidate_id"].astype(str).tolist(),
        "test_geometry_hashes_unchanged": predictions["geometry_sha256"].astype(str).tolist()
        == test["geometry_sha256"].astype(str).tolist(),
        "predictions_path": str(prediction_path),
        "predictions_sha256": sha256_file(prediction_path),
        "model": model.artifact(),
        "imputer": imputer.artifact(),
    }
    atomic_json(evidence_path, evidence)
    return evidence


def _atomic_csv_frame(path: Path, frame: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        frame.to_csv(temporary, index=False)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _first_target_record(run_dir: Path) -> dict[str, Any] | None:
    path = run_dir / "manifests" / "target_groups.jsonl"
    if not path.is_file():
        return None
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            value = json.loads(line)
            return dict(value) if isinstance(value, dict) else None
    return None


def _data_geometry_checks(run_dir: Path | None) -> tuple[list[Check], Path | None]:
    if run_dir is None:
        return (
            [
                Check(2, "RGB-D frame backprojects correctly", False, "requires official GraspNet frame data"),
                Check(3, "Target points align with instance mask", False, "requires official GraspNet frame data"),
                Check(4, "Target is inside target-centred TSDF", False, "requires a committed real TSDF"),
            ],
            None,
        )
    record = _first_target_record(run_dir)
    if record is None:
        return (
            [
                Check(2, "RGB-D frame backprojects correctly", False, "run target manifest is absent or empty"),
                Check(3, "Target points align with instance mask", False, "run target manifest is absent or empty"),
                Check(4, "Target is inside target-centred TSDF", False, "run target manifest is absent or empty"),
            ],
            None,
        )
    try:
        from PIL import Image
        from scipy.io import loadmat

        from .formal_inputs import group_artifact_slug
        from .tsdf import backproject_masked_depth, depth_to_meters

        depth = np.asarray(Image.open(Path(record["depth_path"])))
        labels = np.asarray(Image.open(Path(record["instance_label_path"])))
        metadata = loadmat(Path(record["meta_path"]))
        scale = float(np.asarray(metadata["factor_depth"]).reshape(-1)[0])
        intrinsics = np.asarray(
            np.load(Path(record["intrinsics_path"]), allow_pickle=False),
            dtype=np.float64,
        )
        mask = labels == int(record["target_instance_label"])
        meters = depth_to_meters(depth, depth_scale=scale)
        points = backproject_masked_depth(meters, mask, intrinsics)
        rows, columns = np.nonzero(mask & (meters > 0))
        projected_u = intrinsics[0, 0] * points[:, 0] / points[:, 2] + intrinsics[0, 2]
        projected_v = intrinsics[1, 1] * points[:, 1] / points[:, 2] + intrinsics[1, 2]
        maximum_error = float(
            np.max(np.hypot(projected_u - columns, projected_v - rows))
        )
        checks = [
            Check(
                2,
                "RGB-D frame backprojects correctly",
                bool(np.isfinite(maximum_error) and maximum_error < 1e-6),
                f"group={record['group_id']} points={len(points)} max_round_trip_px={maximum_error:.3g}",
            ),
            Check(
                3,
                "Target points align with instance mask",
                bool(len(points) == int(np.count_nonzero(mask & (meters > 0)))),
                f"group={record['group_id']} label={record['target_instance_label']} aligned_points={len(points)}",
            ),
        ]
        tsdf_path = (
            run_dir
            / "target_tsdf"
            / "oracle_gt_mask"
            / f"{group_artifact_slug(str(record['group_id']))}.npz"
        )
        if not tsdf_path.is_file():
            checks.append(
                Check(4, "Target is inside target-centred TSDF", False, f"missing {tsdf_path}")
            )
            return checks, None
        with np.load(tsdf_path, allow_pickle=False) as archive:
            transform = np.asarray(archive["T_local_to_camera"], dtype=np.float64)
            physical_size = float(np.asarray(archive["physical_size"]).item())
        centroid = np.median(points, axis=0)
        local = np.linalg.inv(transform)[:3, :3] @ centroid + np.linalg.inv(transform)[:3, 3]
        inside = bool(np.all(local >= -1e-8) and np.all(local <= physical_size + 1e-8))
        checks.append(
            Check(
                4,
                "Target is inside target-centred TSDF",
                inside,
                f"group={record['group_id']} target_centroid_local_m={local.tolist()} size_m={physical_size}",
            )
        )
        return checks, tsdf_path
    except Exception as error:
        evidence = f"{type(error).__name__}: {error}"
        return (
            [
                Check(2, "RGB-D frame backprojects correctly", False, evidence),
                Check(3, "Target points align with instance mask", False, evidence),
                Check(4, "Target is inside target-centred TSDF", False, evidence),
            ],
            None,
        )


def _real_artifact_checks(run_dir: Path | None) -> list[Check]:
    if run_dir is None:
        return [
            Check(
                6,
                "Eight real groups attempted and at least four VGN pools are non-empty",
                False,
                "requires eight official GraspNet TSDF groups and four non-empty pools",
            ),
            Check(
                7,
                "Eight real geometry groups pass programmatic and AI-assisted review",
                False,
                "requires hashed real coordinate audit figures and AI-assisted review",
            ),
            Check(
                8,
                "Official evaluator returns association/collision/friction for eight groups",
                False,
                "requires official scene models and validated geometry for eight groups",
            ),
            Check(9, "Formal relevance labels generated", False, "depends on check 8"),
            Check(10, "Ranker trains/predicts on 20 real groups", False, "requires a real-data smoke analysis"),
            Check(11, "Frozen candidate audit passes on real pools", False, "requires real prediction artifacts"),
        ]
    try:
        candidate_rows: list[dict[str, Any]] = []
        manifest = run_dir / "candidate_manifest.jsonl"
        if manifest.is_file():
            for line in manifest.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError("candidate manifest rows must be JSON objects")
                candidate_rows.append(row)
        candidate_keys: list[tuple[str, str]] = []
        for row in candidate_rows:
            condition = row.get("grounding_condition")
            group_id = row.get("group_id")
            candidate_count = row.get("candidate_count")
            if condition not in SMOKE_GROUNDING_CONDITIONS:
                raise ValueError(
                    "candidate manifest has a malformed grounding_condition"
                )
            if not isinstance(group_id, str) or not group_id.strip():
                raise ValueError("candidate manifest has a blank or malformed group_id")
            if (
                isinstance(candidate_count, bool)
                or not isinstance(candidate_count, int)
                or candidate_count < 0
            ):
                raise ValueError("candidate manifest has an invalid candidate_count")
            candidate_keys.append((condition, group_id))
        if len(set(candidate_keys)) != len(candidate_keys):
            raise ValueError(
                "candidate manifest has duplicate (grounding_condition, group_id) keys"
            )
        oracle_rows = [
            row
            for row in candidate_rows
            if row["grounding_condition"] == "oracle_gt_mask"
        ]
        attempted_group_ids = [str(row["group_id"]) for row in oracle_rows]
        nonempty_group_ids = {
            str(row["group_id"])
            for row in oracle_rows
            if int(row["candidate_count"]) > 0
        }
        check6 = Check(
            6,
            "Eight real groups attempted and at least four VGN pools are non-empty",
            len(attempted_group_ids) >= SMOKE_GEOMETRY_GROUPS
            and len(nonempty_group_ids) >= SMOKE_MINIMUM_NONEMPTY_CANDIDATE_GROUPS,
            (
                f"required_attempted={SMOKE_GEOMETRY_GROUPS} "
                f"required_nonempty={SMOKE_MINIMUM_NONEMPTY_CANDIDATE_GROUPS} "
                "grounding_condition=oracle_gt_mask "
                f"oracle_non_empty_groups={len(nonempty_group_ids)} "
                f"oracle_indexed_groups={len(attempted_group_ids)} "
                f"all_condition_rows={len(candidate_rows)} "
                f"manifest={manifest}"
            ),
        )
    except Exception as error:
        check6 = Check(6, "Some real samples produce non-empty candidates", False, f"{type(error).__name__}: {error}")

    try:
        from .stages import load_evaluator_geometry_contract

        contract_path = run_dir / "geometry_validation" / "evaluator_geometry_contract.json"
        _, evidence = load_evaluator_geometry_contract(contract_path, evidence_policy="formal")
        verified = evidence["verified_evidence"]
        figures = verified["audit_figure_sha256"]
        groups = verified["validated_group_ids"]
        check7 = Check(
            7,
            "Eight real geometry groups pass programmatic and AI-assisted review",
            len(groups) >= SMOKE_GEOMETRY_GROUPS
            and len(figures) >= SMOKE_GEOMETRY_GROUPS,
            (
                f"required_groups={SMOKE_GEOMETRY_GROUPS} "
                f"validated_groups={len(groups)} validated_figures={len(figures)} "
                f"contract={contract_path}"
            ),
        )
    except Exception as error:
        check7 = Check(7, "Real gripper pose visualisation is directionally validated", False, f"{type(error).__name__}: {error}")

    try:
        from .stages import load_evaluator_parity_gate

        parity_path = run_dir / "evaluator_parity" / "evaluator_parity_gate.json"
        _, evidence = load_evaluator_parity_gate(parity_path, evidence_policy="formal")
        verified = evidence["verified_evidence"]
        count = int(verified["candidate_count"])
        parity_artifact = json.loads(
            Path(evidence["artifact_path"]).read_text(encoding="utf-8")
        )
        groups = parity_artifact.get("validated_group_ids", [])
        grounding_condition = parity_artifact.get("grounding_condition")
        check8 = Check(
            8,
            "Official evaluator returns association/collision/friction for eight groups",
            isinstance(groups, list)
            and len(set(map(str, groups))) >= SMOKE_EVALUATOR_GROUPS
            and grounding_condition == "oracle_gt_mask"
            and count > 0,
            (
                f"required_groups={SMOKE_EVALUATOR_GROUPS} "
                f"grounding_condition={grounding_condition!r} "
                f"validated_groups={len(set(map(str, groups))) if isinstance(groups, list) else 0} "
                f"parity_candidates={count} gate={parity_path}"
            ),
        )
    except Exception as error:
        check8 = Check(8, "Official evaluator returns association/collision/friction", False, f"{type(error).__name__}: {error}")

    try:
        label_paths = sorted((run_dir / "official_labels").glob("*/*.json"))
        label_count = 0
        for path in label_paths:
            payload = json.loads(path.read_text(encoding="utf-8"))
            labels = payload.get("labels", [])
            for row in labels:
                if not {"associated_object_id", "collision", "friction_required", "pose_valid", "relevance"}.issubset(row):
                    raise ValueError(f"incomplete label record: {path}")
            label_count += len(labels)
        check9 = Check(
            9,
            "Formal relevance labels generated",
            label_count > 0,
            f"label_bundles={len(label_paths)} candidate_labels={label_count}",
        )
    except Exception as error:
        check9 = Check(9, "Formal relevance labels generated", False, f"{type(error).__name__}: {error}")

    try:
        smoke_evidence_path = run_dir / "smoke" / "real_ranker_smoke_evidence.json"
        if smoke_evidence_path.is_file():
            evidence = json.loads(smoke_evidence_path.read_text(encoding="utf-8"))
            prediction = Path(str(evidence.get("predictions_path", "")))
            passed = bool(
                evidence.get("schema_version") == REAL_RANKER_SMOKE_SCHEMA
                and evidence.get("status") == "PASS"
                and evidence.get("scope") == "real_data_ranker_integration_only"
                and evidence.get("evidence_kind") == "real_data_ranker_integration"
                and evidence.get("fixture_only") is False
                and evidence.get("formal_report_eligible") is False
                and int(evidence.get("minimum_real_groups", 0))
                >= RANKER_INTEGRATION_MINIMUM_GROUPS
                and int(evidence.get("nonempty_group_count", 0))
                >= RANKER_INTEGRATION_MINIMUM_GROUPS
                and evidence.get("test_candidate_ids_unchanged") is True
                and prediction.is_file()
                and evidence.get("predictions_sha256") == sha256_file(prediction)
            )
            detail = f"smoke_nonempty_groups={evidence.get('nonempty_group_count')} evidence={smoke_evidence_path}"
        else:
            analysis_manifests = sorted((run_dir / "analysis").glob("*/analysis_manifest.json"))
            group_counts: list[int] = []
            for path in analysis_manifests:
                payload = json.loads(path.read_text(encoding="utf-8"))
                if payload.get("status") != "COMPLETE":
                    raise ValueError(f"analysis is not complete: {path}")
                audit = json.loads((path.parent / "frozen_pool_audit.json").read_text(encoding="utf-8"))
                group_counts.append(int(audit["group_count"]))
                if not (path.parent / "model_selection.json").is_file() or not (path.parent / "reranked_predictions.csv").is_file():
                    raise ValueError(f"analysis lacks train/predict artifacts: {path.parent}")
            passed = bool(group_counts) and max(group_counts) >= 20
            detail = f"analysis_group_counts={group_counts}"
        check10 = Check(10, "Ranker trains/predicts on 20 real groups", passed, detail)
    except Exception as error:
        check10 = Check(10, "Ranker trains/predicts on 20 real groups", False, f"{type(error).__name__}: {error}")

    try:
        audits = sorted((run_dir / "analysis").glob("*/frozen_pool_audit.json"))
        statuses = [json.loads(path.read_text(encoding="utf-8")).get("status") for path in audits]
        if not statuses:
            smoke_evidence_path = run_dir / "smoke" / "real_ranker_smoke_evidence.json"
            evidence = json.loads(smoke_evidence_path.read_text(encoding="utf-8"))
            statuses = [
                "PASS"
                if evidence.get("test_candidate_ids_unchanged") is True
                and evidence.get("test_geometry_hashes_unchanged") is True
                and evidence.get("scope") == "real_data_ranker_integration_only"
                and evidence.get("evidence_kind") == "real_data_ranker_integration"
                and evidence.get("fixture_only") is False
                else "FAIL"
            ]
        check11 = Check(
            11,
            "Frozen candidate audit passes on real pools",
            bool(statuses) and all(status == "PASS" for status in statuses),
            f"audits={len(audits)} statuses={statuses}",
        )
    except Exception as error:
        check11 = Check(11, "Frozen candidate audit passes on real pools", False, f"{type(error).__name__}: {error}")
    return [check6, check7, check8, check9, check10, check11]


def run_smoke(
    output_root: Path | None = None, *, run_dir: Path | None = None
) -> list[Check]:
    root = repository_root()
    output = output_root or root / "artifacts" / "graspnet6d" / "smoke"
    dataset_root = root / "data_external" / "graspnet"
    scenes = discover_scene_ids(dataset_root)
    checks: list[Check] = []
    if scenes:
        try:
            report = validate_graspnet_structure(
                dataset_root, camera="kinect", scene_ids=scenes[:2],
                frame_ids=(0, 85, 170, 255), strict=False,
            )
            checks.append(Check(1, "Dataset structure correct", report.valid, f"checked scenes={scenes[:2]}; missing={len(report.missing)} invalid={len(report.invalid)}"))
        except Exception as error:
            checks.append(Check(1, "Dataset structure correct", False, f"{type(error).__name__}: {error}"))
    else:
        checks.append(Check(1, "Dataset structure correct", False, f"no scenes under {dataset_root}"))
    geometry_checks, real_tsdf = _data_geometry_checks(run_dir)
    checks.extend(geometry_checks)
    checks.append(
        _vgn_finite_check()
        if real_tsdf is None
        else _vgn_finite_check(real_tsdf)
    )
    checks.extend(_real_artifact_checks(run_dir))
    checks.append(validate_4d_regression_status())
    go = all(check.passed for check in checks)
    lines = [
        f"# Smoke test: {'GO' if go else 'NO-GO'}",
        "",
        "A data-independent diagnostic cannot satisfy a data-backed formal gate.",
        "",
        "| # | Check | Status | Evidence |",
        "|---:|---|---|---|",
    ]
    lines.extend(
        f"| {check.number} | {check.name} | {'PASS' if check.passed else 'BLOCKED'} | {check.evidence.replace('|', '/')} |"
        for check in checks
    )
    lines.extend(
        [
            "",
            "Formal paper-lite execution is forbidden until all twelve checks pass on official data.",
        ]
    )
    output.mkdir(parents=True, exist_ok=True)
    atomic_text(output / "GO_NO_GO.md", "\n".join(lines) + "\n")
    atomic_json(output / "smoke_checks.json", {"go": go, "checks": [asdict(check) for check in checks]})
    return checks


__all__ = [
    "Check",
    "REGRESSION_EVIDENCE_KIND",
    "REGRESSION_STATUS_FILENAME",
    "REGRESSION_STATUS_SCHEMA_VERSION",
    "RANKER_INTEGRATION_MINIMUM_GROUPS",
    "SMOKE_EVALUATOR_GROUPS",
    "SMOKE_GEOMETRY_GROUPS",
    "SMOKE_MINIMUM_NONEMPTY_CANDIDATE_GROUPS",
    "analytic_diagnostic_tsdf",
    "run_device_benchmark",
    "run_smoke",
    "validate_4d_regression_status",
]
