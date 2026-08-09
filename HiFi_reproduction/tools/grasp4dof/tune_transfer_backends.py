#!/usr/bin/env python3
"""Validation-only staged transfer search for the frozen G0 and C0 backends."""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import pyarrow.parquet as pq
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUN_METHOD = PROJECT_ROOT / "tools/grasp4dof/run_method.py"
FROZEN_VENV = PROJECT_ROOT / ".venv-grasp4dof"
FROZEN_PYTHON = FROZEN_VENV / "bin/python"
FROZEN_PYTHON_VERSION = (3, 11, 15)
DEFAULT_OUTPUT_NAMESPACE = "frozen_py311"
EXPECTED_VALIDATION_COUNT = 3_778
SEARCH_SCHEMA_VERSION = 1
VALIDATION_SPLIT_ALIASES = frozenset({"val", "validation"})

DEFAULT_NMS = {
    "angle_distance_deg": 15.0,
    "center_distance_px": 8.0,
    "max_output": 100,
    "rectangle_iou_threshold": 0.25,
    "width_distance_px": 10.0,
}
COMPACT_NMS = {
    "default": DEFAULT_NMS,
    "stronger": {
        "angle_distance_deg": 20.0,
        "center_distance_px": 12.0,
        "max_output": 100,
        "rectangle_iou_threshold": 0.15,
        "width_distance_px": 15.0,
    },
    "weaker": {
        "angle_distance_deg": 10.0,
        "center_distance_px": 4.0,
        "max_output": 100,
        "rectangle_iou_threshold": 0.5,
        "width_distance_px": 5.0,
    },
}
GATE_GRID = ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (1.0, 2.0))
QUALITY_GRID = (0.0, 0.05, 0.1)

RESULT_COLUMNS = (
    "method_id",
    "stage",
    "candidate_id",
    "execution_status",
    "reused_from_candidate_id",
    "selected_in_stage",
    "sample_count",
    "j_at_1",
    "j_at_5",
    "no_grasp_rate",
    "p50_latency_seconds",
    "p95_latency_seconds",
    "simplicity_score",
    "total_wall_seconds",
    "config_path",
    "config_sha256",
    "output_dir",
    "per_sample_path",
    "per_sample_sha256",
    "per_candidate_path",
    "per_candidate_sha256",
    "executor_environment_path",
    "executor_environment_sha256",
    "python_executable",
    "python_version",
    "python_prefix",
    "torch_version",
    "design_json",
)


@dataclass(frozen=True, slots=True)
class Candidate:
    method_id: str
    stage: str
    candidate_id: str
    config: Mapping[str, Any]
    design: Mapping[str, Any]


def _canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_bytes(_canonical_bytes(value))
    os.replace(temporary, path)


def _write_json_once(path: Path, value: Mapping[str, Any]) -> str:
    encoded = _canonical_bytes(value)
    digest = _sha256_bytes(encoded)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    except FileExistsError:
        if path.read_bytes() != encoded:
            raise FileExistsError(f"refusing to change immutable candidate config: {path}")
        return digest
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise
    return digest


def _contained(path: Path, root: Path, label: str) -> Path:
    resolved = path.expanduser().resolve()
    try:
        resolved.relative_to(root.expanduser().resolve())
    except ValueError as exc:
        raise ValueError(f"{label} escapes the allowed run directory: {resolved}") from exc
    return resolved


def frozen_interpreter_contract(python: Path) -> dict[str, Any]:
    """Require both this driver and every child to use the frozen 3.11 venv."""

    expected_python = FROZEN_PYTHON.absolute()
    requested_python = python.expanduser()
    if not requested_python.is_absolute():
        requested_python = (Path.cwd() / requested_python).absolute()
    if requested_python != expected_python:
        raise RuntimeError(
            f"transfer search requires {expected_python}, got {requested_python}"
        )
    if Path(sys.prefix).resolve() != FROZEN_VENV.resolve():
        raise RuntimeError(
            "the transfer-search driver itself must run inside "
            f"{FROZEN_VENV}; current sys.prefix={sys.prefix}"
        )
    probe = (
        "import json,sys,torch,pyarrow,numpy,cv2;"
        "print(json.dumps({'executable':sys.executable,'prefix':sys.prefix,"
        "'version_info':list(sys.version_info[:3]),'python_version':sys.version,"
        "'torch_version':torch.__version__,'pyarrow_version':pyarrow.__version__,"
        "'numpy_version':numpy.__version__,'opencv_version':cv2.__version__,"
        "'mps_available':torch.backends.mps.is_available()}))"
    )
    completed = subprocess.run(
        [str(expected_python), "-c", probe],
        cwd=PROJECT_ROOT,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    value = json.loads(completed.stdout)
    if Path(value["prefix"]).resolve() != FROZEN_VENV.resolve():
        raise RuntimeError(f"child interpreter has wrong prefix: {value['prefix']}")
    if tuple(value["version_info"]) != FROZEN_PYTHON_VERSION:
        raise RuntimeError(
            f"child interpreter must be Python {FROZEN_PYTHON_VERSION}, "
            f"got {tuple(value['version_info'])}"
        )
    if value.get("mps_available") is not True:
        raise RuntimeError("frozen interpreter does not expose MPS")
    value["requested_executable"] = str(expected_python)
    value["contract"] = "project_frozen_grasp4dof_python311"
    return value


def validation_manifest_contract(run_dir: Path) -> dict[str, Any]:
    """Inspect only the two validation manifests and require the complete split."""

    run = run_dir.expanduser().resolve()
    manifests = run / "manifests"
    paths = {
        "samples": manifests / "validation_samples.parquet",
        "labels": manifests / "validation_labels.parquet",
    }
    result: dict[str, Any] = {}
    for name, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"missing validation {name} manifest: {path}")
        parquet = pq.ParquetFile(path)
        rows = int(parquet.metadata.num_rows)
        if rows != EXPECTED_VALIDATION_COUNT:
            raise ValueError(
                f"validation {name} manifest must contain exactly "
                f"{EXPECTED_VALIDATION_COUNT} rows, got {rows}"
            )
        if "split" not in parquet.schema_arrow.names:
            raise ValueError(f"validation {name} manifest lacks split provenance")
        split_values = set(pq.read_table(path, columns=["split"])["split"].to_pylist())
        if not split_values or not split_values <= VALIDATION_SPLIT_ALIASES:
            raise ValueError(
                f"validation {name} manifest contains forbidden split values: "
                f"{sorted(map(str, split_values))}"
            )
        result[name] = {
            "path": str(path),
            "rows": rows,
            "sha256": _sha256_file(path),
            "observed_split_values": sorted(map(str, split_values)),
            "normalised_split": "validation",
        }
    return result


def _checkpoint_by_id(manifest: Mapping[str, Any], checkpoint_id: str) -> dict[str, Any]:
    matches = [
        item
        for item in manifest.get("checkpoints", [])
        if str(item.get("checkpoint_id")) == checkpoint_id
    ]
    if len(matches) != 1:
        raise ValueError(f"checkpoint {checkpoint_id!r} is not unique in vendor manifest")
    item = dict(matches[0])
    path = Path(str(item["path"])).expanduser().resolve()
    if not path.is_file() or _sha256_file(path) != str(item["sha256"]):
        raise ValueError(f"checkpoint file/hash mismatch for {checkpoint_id}: {path}")
    item["path"] = str(path)
    return item


def load_official_checkpoints(run_dir: Path) -> dict[str, dict[str, Any]]:
    run = run_dir.expanduser().resolve()
    gr = _load_json(run / "third_party/grconvnet_source_manifest.json")
    gg = _load_json(run / "third_party/ggcnn2_source_manifest.json")
    if gr.get("status") != "PASS" or gg.get("status") != "PASS":
        raise ValueError("official vendor audit must pass before transfer search")
    return {
        name: _checkpoint_by_id(gr, name)
        for name in ("jacquard_rgbd", "jacquard_depth", "cornell_rgbd")
    } | {"ggcnn2_cornell_state_dict": _checkpoint_by_id(gg, "state_dict")}


def _base_config(method_id: str, checkpoint: Mapping[str, Any]) -> dict[str, Any]:
    config: dict[str, Any] = {
        "center_gate_exponent": 1.0,
        "checkpoint_path": str(checkpoint["path"]),
        "checkpoint_sha256": str(checkpoint["sha256"]),
        "conditioning_variant": "dilated_crop",
        "device": "mps",
        "dilation_fraction": 0.15,
        "fixed_height_px": 20.0,
        "input_size": 224 if method_id == "G0" else 300,
        "jaw_gate_exponent": 0.0,
        "max_raw_candidates": 100,
        "min_peak_distance_px": 15 if method_id == "G0" else 20,
        "minimum_crop_side_px": 64,
        "minimum_mask_area_px": 50,
        "nms": copy.deepcopy(DEFAULT_NMS),
        "quality_threshold": 0.0,
    }
    if method_id == "G0":
        config["input_channels"] = int(checkpoint["input_channels"])
    return config


def g0_checkpoint_candidates(
    checkpoints: Mapping[str, Mapping[str, Any]],
) -> list[Candidate]:
    candidates: list[Candidate] = []
    for checkpoint_id in ("jacquard_rgbd", "jacquard_depth", "cornell_rgbd"):
        checkpoint = checkpoints[checkpoint_id]
        config = _base_config("G0", checkpoint)
        config.update(
            {
                "conditioning_variant": "dilated_crop",
                "input_size": 224,
                "dilation_fraction": 0.15,
                "quality_threshold": 0.0,
            }
        )
        candidates.append(
            Candidate(
                "G0",
                "g0_stage1_checkpoint",
                checkpoint_id,
                config,
                {
                    "checkpoint_id": checkpoint_id,
                    "dataset": checkpoint["dataset"],
                    "modalities": checkpoint["modalities"],
                    "input_channels": checkpoint["input_channels"],
                    "fixed_controls": {
                        "conditioning_variant": "dilated_crop",
                        "input_size": 224,
                        "dilation_fraction": 0.15,
                        "quality_threshold": 0.0,
                    },
                },
            )
        )
    return candidates


def conditioning_candidates(
    method_id: str, base: Mapping[str, Any], stage: str
) -> list[Candidate]:
    designs = (
        ("hard224", "hard_mask", 224, 0.15),
        ("hard300", "hard_mask", 300, 0.15),
        ("crop224_d010", "dilated_crop", 224, 0.10),
        ("crop224_d015", "dilated_crop", 224, 0.15),
        ("crop224_d020", "dilated_crop", 224, 0.20),
        ("crop300_d015", "dilated_crop", 300, 0.15),
    )
    result: list[Candidate] = []
    for name, variant, size, dilation in designs:
        config = copy.deepcopy(dict(base))
        config.update(
            {
                "conditioning_variant": variant,
                "input_size": size,
                "dilation_fraction": dilation,
            }
        )
        result.append(
            Candidate(
                method_id,
                stage,
                name,
                config,
                {
                    "conditioning_variant": variant,
                    "input_size": size,
                    "dilation_fraction": dilation,
                    "note": "dilation is inert for hard_mask and therefore fixed at 0.15",
                },
            )
        )
    return result


def gate_candidates(
    method_id: str, base: Mapping[str, Any], stage: str
) -> list[Candidate]:
    result: list[Candidate] = []
    for center, jaw in GATE_GRID:
        config = copy.deepcopy(dict(base))
        config.update({"center_gate_exponent": center, "jaw_gate_exponent": jaw})
        result.append(
            Candidate(
                method_id,
                stage,
                f"center{center:g}_jaw{jaw:g}".replace(".", "p"),
                config,
                {"center_gate_exponent": center, "jaw_gate_exponent": jaw},
            )
        )
    return result


def threshold_candidates(
    method_id: str, base: Mapping[str, Any], stage: str
) -> list[Candidate]:
    result: list[Candidate] = []
    for threshold in QUALITY_GRID:
        config = copy.deepcopy(dict(base))
        config["quality_threshold"] = threshold
        result.append(
            Candidate(
                method_id,
                stage,
                f"threshold{threshold:g}".replace(".", "p"),
                config,
                {"quality_threshold": threshold},
            )
        )
    return result


def nms_candidates(
    method_id: str, base: Mapping[str, Any], stage: str
) -> list[Candidate]:
    result: list[Candidate] = []
    for name, nms in COMPACT_NMS.items():
        config = copy.deepcopy(dict(base))
        config["nms"] = copy.deepcopy(nms)
        result.append(Candidate(method_id, stage, name, config, {"nms": nms}))
    return result


def simplicity_score(config: Mapping[str, Any]) -> int:
    """A declared final tie-breaker; lower is operationally simpler."""

    score = int(config.get("conditioning_variant") == "dilated_crop")
    score += int(int(config.get("input_size", 224)) != 224)
    score += int(float(config.get("center_gate_exponent", 0.0)) != 0.0)
    score += int(float(config.get("jaw_gate_exponent", 0.0)) != 0.0)
    score += int(float(config.get("quality_threshold", 0.0)) != 0.0)
    score += sum(
        int(config.get("nms", {}).get(key) != value)
        for key, value in DEFAULT_NMS.items()
    )
    score += int(int(config.get("input_channels", 1)) > 1)
    return score


def selection_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    """Lexicographic validation-only policy declared before running the search."""

    return (
        -round(float(row["j_at_1"]), 12),
        -round(float(row["j_at_5"]), 12),
        round(float(row["no_grasp_rate"]), 12),
        round(float(row["p50_latency_seconds"]), 6),
        round(float(row["p95_latency_seconds"]), 6),
        int(row["simplicity_score"]),
        str(row["candidate_id"]),
    )


def select_validation_winner(rows: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    if not rows:
        raise ValueError("cannot select from an empty validation stage")
    if any(int(row["sample_count"]) != EXPECTED_VALIDATION_COUNT for row in rows):
        raise ValueError("selection requires complete validation results for every candidate")
    return min(rows, key=selection_key)


def build_run_method_command(
    *,
    python: Path,
    run_method: Path,
    run_dir: Path,
    candidate: Candidate,
    config_path: Path,
    output_dir: Path,
) -> list[str]:
    """Construct the only permitted experiment command: complete validation."""

    if candidate.method_id not in {"G0", "C0"}:
        raise ValueError("transfer search only supports G0 and C0")
    return [
        str(python),
        str(run_method),
        "--run-dir",
        str(run_dir),
        "--split",
        "validation",
        "--method",
        candidate.method_id,
        "--config",
        str(config_path),
        "--output-dir",
        str(output_dir),
    ]


def _executor_environment(
    *,
    interpreter: Mapping[str, Any],
    run_method: Path,
    run_method_sha256: str,
    config_sha256: str,
) -> dict[str, Any]:
    return {
        "schema_version": SEARCH_SCHEMA_VERSION,
        "interpreter": copy.deepcopy(dict(interpreter)),
        "run_method_path": str(run_method),
        "run_method_sha256": run_method_sha256,
        "config_sha256": config_sha256,
    }


def _validate_completed_output(
    output_dir: Path,
    config_sha256: str,
    expected_environment: Mapping[str, Any],
) -> dict[str, Any]:
    required = (
        "COMPLETE.json",
        "metrics.json",
        "runtime_metrics.json",
        "run_config.json",
        "executor_environment.json",
        "per_sample_predictions.parquet",
        "per_candidate_predictions.parquet",
    )
    missing = [name for name in required if not (output_dir / name).is_file()]
    if missing:
        raise RuntimeError(f"candidate output is incomplete ({output_dir}): {missing}")
    complete = _load_json(output_dir / "COMPLETE.json")
    provenance = _load_json(output_dir / "run_config.json")
    metrics = _load_json(output_dir / "metrics.json")
    runtime = _load_json(output_dir / "runtime_metrics.json")
    environment_path = output_dir / "executor_environment.json"
    environment = _load_json(environment_path)
    if environment != dict(expected_environment):
        raise RuntimeError(
            "candidate executor environment does not match the frozen interpreter"
        )
    for payload_name, payload in (
        ("COMPLETE", complete),
        ("run_config", provenance),
        ("metrics", metrics),
        ("runtime", runtime),
    ):
        if int(payload.get("sample_count", -1)) != EXPECTED_VALIDATION_COUNT:
            raise RuntimeError(
                f"{payload_name} is not full validation: {payload.get('sample_count')}"
            )
    if provenance.get("split") != "validation":
        raise RuntimeError("candidate provenance is not validation-only")
    if provenance.get("oracle") is not False:
        raise RuntimeError("transfer search candidate unexpectedly used oracle input")
    if provenance.get("config_sha256") != config_sha256:
        raise RuntimeError("candidate config hash does not match run_method provenance")
    per_sample = output_dir / "per_sample_predictions.parquet"
    per_candidate = output_dir / "per_candidate_predictions.parquet"
    if pq.ParquetFile(per_sample).metadata.num_rows != EXPECTED_VALIDATION_COUNT:
        raise RuntimeError("per-sample Parquet is not full validation")
    per_sample_sha = _sha256_file(per_sample)
    per_candidate_sha = _sha256_file(per_candidate)
    if provenance.get("per_sample_sha256") != per_sample_sha:
        raise RuntimeError("per-sample Parquet hash drift")
    if provenance.get("per_candidate_sha256") != per_candidate_sha:
        raise RuntimeError("per-candidate Parquet hash drift")
    return {
        "metrics": metrics,
        "runtime": runtime,
        "per_sample_path": str(per_sample),
        "per_sample_sha256": per_sample_sha,
        "per_candidate_path": str(per_candidate),
        "per_candidate_sha256": per_candidate_sha,
        "executor_environment_path": str(environment_path),
        "executor_environment_sha256": _sha256_file(environment_path),
    }


def _write_results_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=RESULT_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in RESULT_COLUMNS})
    os.replace(temporary, path)


class SearchRunner:
    def __init__(
        self,
        *,
        run_dir: Path,
        python: Path,
        run_method: Path,
        validation_contract: Mapping[str, Any],
        interpreter_contract: Mapping[str, Any],
        output_namespace: str = DEFAULT_OUTPUT_NAMESPACE,
    ) -> None:
        self.run_dir = run_dir.expanduser().resolve()
        requested_python = python.expanduser()
        if not requested_python.is_absolute():
            requested_python = Path.cwd() / requested_python
        # Preserve the venv launcher path: resolving its symlink would execute the
        # Homebrew base interpreter without the frozen venv's sys.prefix.
        self.python = requested_python.absolute()
        self.run_method = run_method.expanduser().resolve()
        self.run_method_sha256 = _sha256_file(self.run_method)
        if not output_namespace or any(
            token in output_namespace for token in ("/", "\\", "..")
        ):
            raise ValueError("output namespace must be one safe path component")
        self.search_root = _contained(
            self.run_dir / "validation/transfer_search" / output_namespace,
            self.run_dir,
            "search root",
        )
        self.results_path = _contained(
            self.run_dir / "validation/transfer_search_results.csv",
            self.run_dir,
            "results table",
        )
        self.validation_contract = copy.deepcopy(dict(validation_contract))
        self.interpreter_contract = copy.deepcopy(dict(interpreter_contract))
        self.rows: list[dict[str, Any]] = []
        self.cache: dict[str, dict[str, Any]] = {}
        self.space: dict[str, Any] = {
            "schema_version": SEARCH_SCHEMA_VERSION,
            "protocol": "validation_only_full_3778",
            "output_namespace": output_namespace,
            "validation": self.validation_contract,
            "interpreter": self.interpreter_contract,
            "run_method": {
                "path": str(self.run_method),
                "sha256": self.run_method_sha256,
            },
            "selection_policy": {
                "ordered_criteria": [
                    "maximise_j_at_1",
                    "maximise_j_at_5",
                    "minimise_no_grasp_rate",
                    "minimise_p50_latency_seconds_rounded_1e-6",
                    "minimise_p95_latency_seconds_rounded_1e-6",
                    "minimise_declared_simplicity_score",
                    "candidate_id_deterministic_tiebreak",
                ],
                "forbidden_selection_sources": ["pilot", "development_smoke"],
            },
            "stages": [],
        }

    def _assert_run_method_unchanged(self, phase: str) -> None:
        current_sha256 = _sha256_file(self.run_method)
        if current_sha256 != self.run_method_sha256:
            raise RuntimeError(
                "run_method source drift detected; refusing a mixed-source "
                f"transfer search ({phase}): expected "
                f"{self.run_method_sha256}, found {current_sha256}"
            )

    def _candidate_paths(self, candidate: Candidate) -> tuple[Path, Path]:
        relative = Path(candidate.method_id) / candidate.stage / candidate.candidate_id
        config_path = _contained(
            self.search_root / "configs" / relative.with_suffix(".json"),
            self.search_root,
            "candidate config",
        )
        output_dir = _contained(
            self.search_root / "runs" / relative, self.search_root, "candidate output"
        )
        return config_path, output_dir

    def _persist_space(self) -> None:
        _atomic_json(self.search_root / "search_space.json", self.space)

    def _row_from_result(
        self,
        *,
        candidate: Candidate,
        config_path: Path,
        config_sha256: str,
        output_dir: Path,
        result: Mapping[str, Any],
        execution_status: str,
        reused_from: str = "",
    ) -> dict[str, Any]:
        metrics = result["metrics"]
        runtime = result["runtime"]
        return {
            "method_id": candidate.method_id,
            "stage": candidate.stage,
            "candidate_id": candidate.candidate_id,
            "execution_status": execution_status,
            "reused_from_candidate_id": reused_from,
            "selected_in_stage": False,
            "sample_count": int(metrics["sample_count"]),
            "j_at_1": float(metrics["j_at_1"]),
            "j_at_5": float(metrics["j_at_5"]),
            "no_grasp_rate": float(metrics["no_grasp_rate"]),
            "p50_latency_seconds": float(metrics["p50_latency_seconds"]),
            "p95_latency_seconds": float(metrics["p95_latency_seconds"]),
            "simplicity_score": simplicity_score(candidate.config),
            "total_wall_seconds": float(runtime["total_wall_seconds_including_io_and_evaluation"]),
            "config_path": str(config_path),
            "config_sha256": config_sha256,
            "output_dir": str(output_dir),
            "per_sample_path": result["per_sample_path"],
            "per_sample_sha256": result["per_sample_sha256"],
            "per_candidate_path": result["per_candidate_path"],
            "per_candidate_sha256": result["per_candidate_sha256"],
            "executor_environment_path": result["executor_environment_path"],
            "executor_environment_sha256": result[
                "executor_environment_sha256"
            ],
            "python_executable": self.interpreter_contract["executable"],
            "python_version": self.interpreter_contract["python_version"],
            "python_prefix": self.interpreter_contract["prefix"],
            "torch_version": self.interpreter_contract["torch_version"],
            "design_json": json.dumps(
                dict(candidate.design), sort_keys=True, separators=(",", ":")
            ),
        }

    def run_stage(self, candidates: Sequence[Candidate]) -> tuple[dict[str, Any], Candidate]:
        if not candidates:
            raise ValueError("stage must contain candidates")
        stage = candidates[0].stage
        method_id = candidates[0].method_id
        if any(item.stage != stage or item.method_id != method_id for item in candidates):
            raise ValueError("stage candidates must share method and stage")
        stage_record: dict[str, Any] = {
            "method_id": method_id,
            "stage": stage,
            "full_validation_required": True,
            "expected_sample_count": EXPECTED_VALIDATION_COUNT,
            "candidates": [],
        }
        self.space["stages"].append(stage_record)
        self._persist_space()

        stage_rows: list[dict[str, Any]] = []
        by_id = {candidate.candidate_id: candidate for candidate in candidates}
        for candidate in candidates:
            self._assert_run_method_unchanged(
                f"before {method_id}/{stage}/{candidate.candidate_id}"
            )
            config_path, requested_output = self._candidate_paths(candidate)
            config_sha = _write_json_once(config_path, candidate.config)
            stage_record["candidates"].append(
                {
                    "candidate_id": candidate.candidate_id,
                    "config": copy.deepcopy(dict(candidate.config)),
                    "config_path": str(config_path),
                    "config_sha256": config_sha,
                    "design": copy.deepcopy(dict(candidate.design)),
                }
            )
            self._persist_space()

            if config_sha in self.cache:
                cached = self.cache[config_sha]
                result = cached["result"]
                output_dir = Path(cached["output_dir"])
                execution_status = "CARRIED_FULL_VALIDATION"
                reused_from = str(cached["candidate_id"])
            else:
                output_dir = requested_output
                expected_environment = _executor_environment(
                    interpreter=self.interpreter_contract,
                    run_method=self.run_method,
                    run_method_sha256=self.run_method_sha256,
                    config_sha256=config_sha,
                )
                if output_dir.exists() and any(output_dir.iterdir()):
                    complete = output_dir / "COMPLETE.json"
                    if not complete.is_file():
                        raise RuntimeError(
                            "incomplete candidate output exists; refusing silent retry: "
                            f"{output_dir}"
                        )
                    execution_status = "RESUMED_COMPLETE"
                else:
                    command = build_run_method_command(
                        python=self.python,
                        run_method=self.run_method,
                        run_dir=self.run_dir,
                        candidate=candidate,
                        config_path=config_path,
                        output_dir=output_dir,
                    )
                    print(
                        json.dumps(
                            {
                                "event": "START_FULL_VALIDATION",
                                "method_id": method_id,
                                "stage": stage,
                                "candidate_id": candidate.candidate_id,
                                "expected": EXPECTED_VALIDATION_COUNT,
                                "command": command,
                            },
                            sort_keys=True,
                        ),
                        flush=True,
                    )
                    environment = dict(os.environ)
                    environment.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
                    environment.setdefault("PYTHONDONTWRITEBYTECODE", "1")
                    subprocess.run(
                        command,
                        cwd=PROJECT_ROOT,
                        env=environment,
                        check=True,
                    )
                    self._assert_run_method_unchanged(
                        f"after execution of {method_id}/{stage}/{candidate.candidate_id}"
                    )
                    _atomic_json(
                        output_dir / "executor_environment.json",
                        expected_environment,
                    )
                    execution_status = "EXECUTED_FULL_VALIDATION"
                result = _validate_completed_output(
                    output_dir, config_sha, expected_environment
                )
                self._assert_run_method_unchanged(
                    f"after validation of {method_id}/{stage}/{candidate.candidate_id}"
                )
                self.cache[config_sha] = {
                    "candidate_id": candidate.candidate_id,
                    "output_dir": str(output_dir),
                    "result": result,
                }
                reused_from = ""
            row = self._row_from_result(
                candidate=candidate,
                config_path=config_path,
                config_sha256=config_sha,
                output_dir=output_dir,
                result=result,
                execution_status=execution_status,
                reused_from=reused_from,
            )
            self.rows.append(row)
            stage_rows.append(row)
            _write_results_csv(self.results_path, self.rows)

        winner_row = dict(select_validation_winner(stage_rows))
        winner_id = str(winner_row["candidate_id"])
        for row in stage_rows:
            row["selected_in_stage"] = row["candidate_id"] == winner_id
        stage_record["selected_candidate_id"] = winner_id
        stage_record["selection_metrics"] = {
            name: winner_row[name]
            for name in (
                "j_at_1",
                "j_at_5",
                "no_grasp_rate",
                "p50_latency_seconds",
                "p95_latency_seconds",
                "simplicity_score",
            )
        }
        _write_results_csv(self.results_path, self.rows)
        self._persist_space()
        _atomic_json(
            self.search_root / "selections" / f"{stage}.json",
            {
                "schema_version": SEARCH_SCHEMA_VERSION,
                "selection_source": "complete_validation_only",
                "stage": stage,
                "method_id": method_id,
                "selected_candidate_id": winner_id,
                "selected_config_sha256": winner_row["config_sha256"],
                "selected_metrics": stage_record["selection_metrics"],
                "policy": self.space["selection_policy"],
            },
        )
        print(
            json.dumps(
                {
                    "event": "STAGE_SELECTED",
                    "method_id": method_id,
                    "stage": stage,
                    "selected_candidate_id": winner_id,
                    "metrics": stage_record["selection_metrics"],
                },
                sort_keys=True,
            ),
            flush=True,
        )
        return winner_row, by_id[winner_id]

    def write_selected(self, method_id: str, candidate: Candidate) -> Path:
        relative = (
            "grconvnet/pretrained_transfer/selected_config.json"
            if method_id == "G0"
            else "ggcnn2/pretrained_transfer/selected_config.json"
        )
        destination = _contained(self.run_dir / relative, self.run_dir, "selected config")
        _write_json_once(destination, candidate.config)
        return destination


def _run_g0(
    runner: SearchRunner, checkpoints: Mapping[str, Mapping[str, Any]]
) -> Path:
    _, checkpoint_winner = runner.run_stage(g0_checkpoint_candidates(checkpoints))
    _, conditioning_winner = runner.run_stage(
        conditioning_candidates(
            "G0", checkpoint_winner.config, "g0_stage2_conditioning"
        )
    )
    _, gate_winner = runner.run_stage(
        gate_candidates("G0", conditioning_winner.config, "g0_stage3_gate")
    )
    _, threshold_winner = runner.run_stage(
        threshold_candidates("G0", gate_winner.config, "g0_stage3_threshold")
    )
    _, nms_winner = runner.run_stage(
        nms_candidates("G0", threshold_winner.config, "g0_stage3_nms")
    )
    return runner.write_selected("G0", nms_winner)


def _run_c0(
    runner: SearchRunner, checkpoints: Mapping[str, Mapping[str, Any]]
) -> Path:
    base = _base_config("C0", checkpoints["ggcnn2_cornell_state_dict"])
    _, conditioning_winner = runner.run_stage(
        conditioning_candidates("C0", base, "c0_stage1_conditioning")
    )
    _, gate_winner = runner.run_stage(
        gate_candidates("C0", conditioning_winner.config, "c0_stage2_gate")
    )
    _, threshold_winner = runner.run_stage(
        threshold_candidates("C0", gate_winner.config, "c0_stage2_threshold")
    )
    _, nms_winner = runner.run_stage(
        nms_candidates("C0", threshold_winner.config, "c0_stage2_nms")
    )
    return runner.write_selected("C0", nms_winner)


def planned_unique_candidate_count() -> dict[str, int]:
    """Expected upper bound after carrying identical full-validation configs."""

    return {"G0": 15, "C0": 13, "total": 28}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--python", type=Path, default=FROZEN_PYTHON)
    parser.add_argument("--run-method", type=Path, default=DEFAULT_RUN_METHOD)
    parser.add_argument(
        "--output-namespace", default=DEFAULT_OUTPUT_NAMESPACE, help=argparse.SUPPRESS
    )
    parser.add_argument(
        "--methods", nargs="+", choices=("G0", "C0"), default=("G0", "C0")
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate inputs and print the declared initial search without writing",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    interpreter = frozen_interpreter_contract(args.python)
    run_dir = args.run_dir.expanduser().resolve()
    validation = validation_manifest_contract(run_dir)
    checkpoints = load_official_checkpoints(run_dir)
    if args.dry_run:
        print(
            json.dumps(
                {
                    "protocol": "validation_only_full_3778",
                    "validation": validation,
                    "methods": list(dict.fromkeys(args.methods)),
                    "output_namespace": args.output_namespace,
                    "interpreter": interpreter,
                    "planned_unique_candidates": planned_unique_candidate_count(),
                    "g0_stage1": [
                        {
                            "candidate_id": candidate.candidate_id,
                            "config": candidate.config,
                            "design": candidate.design,
                        }
                        for candidate in g0_checkpoint_candidates(checkpoints)
                    ],
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS is required for the sequential formal transfer search")
    if not args.run_method.expanduser().resolve().is_file():
        raise FileNotFoundError(args.run_method)
    runner = SearchRunner(
        run_dir=run_dir,
        python=args.python,
        run_method=args.run_method,
        validation_contract=validation,
        interpreter_contract=interpreter,
        output_namespace=args.output_namespace,
    )
    selected: dict[str, str] = {}
    for method_id in dict.fromkeys(args.methods):
        destination = (
            _run_g0(runner, checkpoints)
            if method_id == "G0"
            else _run_c0(runner, checkpoints)
        )
        selected[method_id] = str(destination)
    print(
        json.dumps(
            {
                "status": "COMPLETE",
                "protocol": "validation_only_full_3778",
                "result_table": str(runner.results_path),
                "selected_configs": selected,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
