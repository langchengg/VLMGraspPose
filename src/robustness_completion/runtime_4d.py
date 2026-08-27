"""Live, fail-closed 4-DoF deployment-runtime workers.

The worker starts from the locked RGB/depth/language manifest and executes the
actual model and decoder.  Frozen candidate, feature, score, and gate-decision
tables are opened only by :meth:`FourDRuntimeWorker.run_parity`; they are never
used to produce a timed prediction.

The implementation intentionally imports the formal source modules lazily.  In
particular, LightGBM is loaded before Torch on macOS when a re-ranker is needed,
matching the repository's audited OpenMP ordering constraint.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.util
import json
import os
import pickle
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence, TypeVar

import numpy as np
import pandas as pd

from .common import (
    PREREGISTRATION_SHA256,
    atomic_json,
    canonical_sha256,
    require_run_dir,
    sha256_file,
    verify_preregistration,
)
from .runtime_common import (
    FOUR_D_REQUIRED_STAGES,
    build_four_d_subset_manifest,
    synchronize_device,
    timed_call_ns,
)


T = TypeVar("T")
SUPPORTED_ROUTES = ("CROG", "G1", "C1", "D1")
SUPPORTED_METHODS = ("native", "raw", "gated")
SUPPORTED_MODES = ("parity", "cold", "warm-disk", "warm-preloaded")
FORMAL_SEEDS = (42, 123, 2026)
WARMUP_COUNT = 5
MEASURED_COUNT = 100
PARITY_COUNT = 20

PRIMARY_RUN_RELATIVE = Path("runs/fair_unified_reranking_20260809_103012")
FAIR_RUN_RELATIVE = Path("runs/fair_crog_hifics_g1_c1_no_rerank_20260807_091523")
BACKEND_RUN_RELATIVE = Path(
    "HiFi_reproduction/runs/"
    "modular_repeatedfilm_4dof_backends_v1_r0corrected_20260803_163500"
)
HIFI_RUN_RELATIVE = Path(
    "HiFi_reproduction/runs/hifics_ocidvlg_hierfilm_20260727_214615"
)
D1_SOURCE_RELATIVE = Path(
    "HiFi_reproduction/runs/"
    "modular_hierfilm_standard_dexnet_gqcnn_20260728_094528"
)
D1_RERANKER_RELATIVE = Path("runs/fair_d1_reranking_extension_20260811T145515Z")

EXPECTED_HASHES = {
    "paired_test": "7690f7ec4a1d13ca9e9fb01d7fbe81c1d2ae9854fc4b0da07fa4328d7002e4fd",
    "native_inference": "5bbae71830db8264494b20651ff6c62fd87210011614b159c48f11bfd4586544",
    "hifi_config": "23cf158cc653e3af2e16d635f782d6d495b120f739a7855516c314011bf234e1",
    "hifi_checkpoint": "b19a649326384ba4524295cd100b22e54cb9ea615174229fc310fbd6bc898601",
    "hifi_clip": "5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f",
    "crog_config": "132092ba6958fb8378434325400aedfa56a54503cd08536460d8a8cd6924c991",
    "crog_checkpoint": "ac1304da520fbd3f6998dea88ba1e63b39b596b775856b39e5360a29576f1ddf",
    "crog_clip": "afeb0e10f9e5a86da6080e35cf09123aca3b358a0c3e3b6c78a7b63bc04b6762",
    "g1_selected": "93f6218fb3b6c26221261da3bb08f0599d83831c4716db1db5a00f1f9e79d84d",
    "g1_checkpoint": "5fc8cdae2578a2361c80d19d44ebf0df521938cd4c00c04fedb986b1f5169aea",
    "c1_selected": "27d7c7271359cbd92de8c9fd53f9724f36d8507d2e3a7f494aa89ff92a8a3fe8",
    "c1_checkpoint": "13addaa29f1f108888946b50467731b6fb53e34d4dcea5ba9d1d741a95147bc9",
}

LINEAGE = {
    "CROG": {
        "checkpoint": EXPECTED_HASHES["crog_checkpoint"],
        "selected_config": "49f291832880b7d8ff2e277587b421dc9dd21218f4eb6c6447180c0c8a1b7703",
        "generator_config": "44136fa355b3678a1146ad16f7e8649e94fb4fc21fe77e8310c060f61caaff8a",
    },
    "G1": {
        "checkpoint": EXPECTED_HASHES["g1_checkpoint"],
        "selected_config": EXPECTED_HASHES["g1_selected"],
        "generator_config": "e03b599a0ba1f38c9680420312f1eb85b4f367eeb0b65d73142e9c30ad9b41b6",
    },
    "C1": {
        "checkpoint": EXPECTED_HASHES["c1_checkpoint"],
        "selected_config": EXPECTED_HASHES["c1_selected"],
        "generator_config": "e03b599a0ba1f38c9680420312f1eb85b4f367eeb0b65d73142e9c30ad9b41b6",
    },
}

CROG_MEAN = np.asarray([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)[:, None, None]
CROG_STD = np.asarray([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)[:, None, None]
MODEL_FEATURE_SCHEMA_SHA256 = (
    "697e2746928e6b3e75e17e21b9952004532cd2811cd2c1921a46eab1e84cbb75"
)


class RuntimeContractError(RuntimeError):
    """A fail-closed deployment or provenance contract violation."""


@dataclass(frozen=True)
class RawObservation:
    sample_id: str
    scene_id: str
    frame_id: str
    language: str
    rgb: np.ndarray
    depth_m: np.ndarray


@dataclass
class PipelineResult:
    sample_id: str
    route: str
    method: str
    candidates: pd.DataFrame
    features: pd.DataFrame
    seed_scores: np.ndarray | None
    ensemble_scores: np.ndarray | None
    native_candidate_id: str | None
    raw_candidate_id: str | None
    gated_candidate_id: str | None
    selected_candidate_id: str | None
    gate_probability_recover: float | None
    gate_probability_harm: float | None
    gate_switch: bool
    stage_rows: list[dict[str, Any]] = field(default_factory=list)
    instrumented_total_ns: int | None = None

    def selection_signature(self) -> str:
        ordered = self.candidates.sort_values(
            ["native_rank", "candidate_id"], kind="mergesort"
        )
        return canonical_sha256(
            {
                "sample_id": self.sample_id,
                "candidate_ids": ordered["candidate_id"].astype(str).tolist(),
                "candidate_geometry_sha256": ordered[
                    "candidate_geometry_sha256"
                ].astype(str).tolist(),
                "native_scores": ordered["native_score"].astype(float).tolist(),
                "native": self.native_candidate_id,
                "raw": self.raw_candidate_id,
                "gated": self.gated_candidate_id,
                "selected": self.selected_candidate_id,
            }
        )


@dataclass(frozen=True)
class _LiveGateBundle:
    model: Any
    feature_columns: tuple[str, ...]
    operating_point: Any | None
    model_path: Path


class _StageRecorder:
    def __init__(self, route: str, method: str, sample_id: str, cache_policy: str):
        self.route = route
        self.method = method
        self.sample_id = sample_id
        self.cache_policy = cache_policy
        self.mode = {
            "parity_live": "parity",
            "cold_fresh_process": "cold",
            "warm_disk": "warm-disk",
            "warm_preloaded": "warm-preloaded",
        }.get(cache_policy, cache_policy)
        self._rows: dict[str, dict[str, Any]] = {}

    def add(
        self,
        stage: str,
        elapsed_ns: int,
        *,
        status: str = "MEASURED",
        device: str = "cpu",
    ) -> None:
        if elapsed_ns < 0:
            raise AssertionError("stage time must be non-negative")
        prior = self._rows.get(stage)
        value = int(elapsed_ns) + (0 if prior is None else int(prior["elapsed_ns"]))
        self._rows[stage] = {
            "route": self.route,
            "method": self.method,
            "mode": self.mode,
            "sample_id": self.sample_id,
            "cache_policy": self.cache_policy,
            "stage": stage,
            "elapsed_ns": value,
            "elapsed_ms": value / 1e6,
            "status": status if prior is None else prior["status"],
            "deployment_stage": True,
            "device": device,
            "candidate_count": None,
        }

    def call(
        self,
        stage: str,
        function: Callable[[], T],
        *,
        device: str = "cpu",
        status: str = "MEASURED",
    ) -> T:
        elapsed, value = timed_call_ns(function, device=device)
        self.add(stage, elapsed, status=status, device=device)
        return value

    def finish(self, candidate_count: int) -> list[dict[str, Any]]:
        for stage in FOUR_D_REQUIRED_STAGES[self.route]:
            if stage not in self._rows:
                self.add(stage, 0, status="NOT_APPLICABLE_FOR_METHOD")
        for row in self._rows.values():
            row["candidate_count"] = int(candidate_count)
        return [self._rows[stage] for stage in FOUR_D_REQUIRED_STAGES[self.route]]


def _prepend_sys_path(path: Path) -> None:
    value = str(path.resolve())
    if value not in sys.path:
        sys.path.insert(0, value)


def _verified_file(path: Path, expected: str, label: str) -> Path:
    path = path.expanduser().resolve()
    if path.is_symlink() or not path.is_file() or path.stat().st_size <= 0:
        raise RuntimeContractError(f"{label} is missing or is not a regular file: {path}")
    observed = sha256_file(path)
    if observed != expected:
        raise RuntimeContractError(
            f"{label} SHA-256 mismatch: expected {expected}, observed {observed}"
        )
    return path


def _verified_record_path(record: Mapping[str, Any], label: str) -> Path:
    if not isinstance(record, Mapping):
        raise RuntimeContractError(f"{label} artifact record is missing")
    return _verified_file(Path(str(record.get("path", ""))), str(record.get("sha256", "")), label)


def _route_paths(repo: Path, route: str) -> dict[str, Path]:
    primary = repo / PRIMARY_RUN_RELATIVE
    result = {
        "paired": primary / "01_manifests/paired_test.parquet",
        "candidate": primary / f"02_candidates/{route.lower()}_test_top5.parquet",
        "feature": primary
        / f"03_features/tracks/T2_matched_common/{route.lower()}_test/candidate_features.parquet",
        "feature_manifest": primary
        / f"03_features/tracks/T2_matched_common/{route.lower()}_test/feature_manifest.json",
        "score": primary
        / f"08_lock/label_free_test_rankers/{route.lower()}/per_candidate_scores.parquet",
        "ranker_manifest": primary
        / f"08_lock/label_free_test_rankers/{route.lower()}/manifest.json",
        "gate_selection": primary / f"08_lock/gates/{route.lower()}/gate_selection.json",
        "gate_manifest": primary
        / f"08_lock/label_free_test_gates/{route.lower()}/manifest.json",
        "gate_decision": primary
        / f"08_lock/label_free_test_gates/{route.lower()}/gate_test_decisions.parquet",
        "calibration": primary / f"05_calibration/{route.lower()}_calibration_manifest.json",
    }
    return result


def _d1_status(repo: Path, *, probe_docker: bool) -> dict[str, Any]:
    source = (repo / D1_SOURCE_RELATIVE).resolve()
    reranker = (repo / D1_RERANKER_RELATIVE).resolve()
    source_manifest = source / "run_manifest.json"
    candidate_config = source / "candidates/hierfilm/run_config.json"
    scoring_config = source / "scores/hierfilm/run_config.json"
    source_payload = (
        json.loads(source_manifest.read_text(encoding="utf-8"))
        if source_manifest.is_file()
        else {}
    )
    scoring_payload = (
        json.loads(scoring_config.read_text(encoding="utf-8"))
        if scoring_config.is_file()
        else {}
    )
    records = []
    for label, path in (
        ("source_run_manifest", source_manifest),
        ("candidate_config", candidate_config),
        ("scoring_config", scoring_config),
        ("reranker_source_audit", reranker / "00_audit/D1_SOURCE_RECONCILIATION_ACTIVE.json"),
    ):
        records.append(
            {
                "label": label,
                "path": str(path.resolve()),
                "exists": path.is_file(),
                "sha256": sha256_file(path) if path.is_file() else None,
            }
        )
    docker = {
        "command": ["docker", "image", "inspect", "sha256:3d1158"],
        "executed": False,
        "return_code": None,
        "image_present": None,
        "stdout_sha256": None,
        "image_id": None,
        "architecture": None,
        "os": None,
        "size_bytes": None,
    }
    gqcnn_checkout = (repo / "HiFi_reproduction/third_party/gqcnn-official").resolve()
    # Keep the venv launcher path itself: resolving the symlink would discard
    # its virtual-environment prefix and make installed Dex-Net dependencies
    # invisible to Python.
    gqcnn_python = repo / "HiFi_reproduction/.venv-gqcnn/bin/python"
    gqcnn_probe = {
        "executed": False,
        "return_code": None,
        "python": str(gqcnn_python),
        "required_pythonpath": str(gqcnn_checkout),
        "imports_available": None,
        "stdout_sha256": None,
    }
    if probe_docker:
        process = subprocess.run(
            docker["command"], capture_output=True, text=True, check=False, timeout=30
        )
        docker.update(
            {
                "executed": True,
                "return_code": int(process.returncode),
                "image_present": process.returncode == 0,
                "stdout_sha256": hashlib.sha256(process.stdout.encode()).hexdigest(),
            }
        )
        if process.returncode == 0:
            inspected = json.loads(process.stdout)
            if len(inspected) == 1:
                docker.update(
                    {
                        "image_id": inspected[0].get("Id"),
                        "architecture": inspected[0].get("Architecture"),
                        "os": inspected[0].get("Os"),
                        "size_bytes": inspected[0].get("Size"),
                    }
                )
        environment = os.environ.copy()
        environment["PYTHONPATH"] = str(gqcnn_checkout)
        import_process = subprocess.run(
            [
                str(gqcnn_python),
                "-c",
                "import gqcnn, autolab_core, perception; print(gqcnn.__version__)",
            ],
            env=environment,
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )
        gqcnn_probe.update(
            {
                "executed": True,
                "return_code": int(import_process.returncode),
                "imports_available": import_process.returncode == 0,
                "stdout_sha256": hashlib.sha256(
                    import_process.stdout.encode()
                ).hexdigest(),
            }
        )
    return {
        "status": "NOT_EXECUTED_IMPLEMENTATION_INCOMPLETE",
        "route": "D1",
        "device": "mps+cpu+linux_amd64_cpu",
        "retrospective": True,
        "deployable_source_route": True,
        "complete_deployment": False,
        "timings": [],
        "results": [],
        "blockers": [
            "The exact D1 route crosses the grasp4dof MPS environment, the .venv-gqcnn "
            "Dex-Net sampler/filter/NMS runtime, and a pinned linux/amd64 TensorFlow-1.15 "
            "GQ-CNN Docker image. A measured one-process adapter was not implemented; "
            "cached D1 telemetry is therefore not relabelled as full deployment latency."
        ],
        "source_records": records,
        "checkpoint_and_model_hashes": {
            "hifics_checkpoint": source_payload.get("identities", {}).get(
                "checkpoint"
            ),
            "hifics_clip": source_payload.get("identities", {}).get("clip"),
            "gqcnn_model_config_sha256": scoring_payload.get("model", {}).get(
                "model_config_hash"
            ),
            "gqcnn_model_file_manifest_sha256": scoring_payload.get("model", {}).get(
                "model_file_manifest_hash"
            ),
            "gqcnn_model_file_manifest": scoring_payload.get("model", {}).get(
                "model_file_manifest", []
            ),
        },
        "docker_image_id_prefix": "sha256:3d1158",
        "docker_probe": docker,
        "gqcnn_import_probe": gqcnn_probe,
    }


def asset_preflight(
    repo: Path,
    run_dir: Path,
    route: str,
    *,
    method: str = "gated",
    verify_hashes: bool = True,
    probe_docker: bool = False,
) -> dict[str, Any]:
    """Validate immutable route assets without running a model or reading outcomes."""

    repo = repo.resolve()
    run_dir = require_run_dir(repo, run_dir)
    verify_preregistration(run_dir)
    route = route.upper()
    method = method.lower()
    if route not in SUPPORTED_ROUTES or method not in SUPPORTED_METHODS:
        raise ValueError("unsupported route or method")
    if route == "D1":
        return _d1_status(repo, probe_docker=probe_docker)
    paths = _route_paths(repo, route)
    required: list[tuple[str, Path, str | None]] = [
        ("paired_test", paths["paired"], EXPECTED_HASHES["paired_test"]),
    ]
    if route == "CROG":
        crog = repo / "crog_reproduction/CROG"
        required.extend(
            [
                (
                    "crog_config",
                    crog / "config/OCID-VLG/CROG_mac_mps_official_params_50epoch_bs8.yaml",
                    EXPECTED_HASHES["crog_config"],
                ),
                (
                    "crog_checkpoint",
                    crog
                    / "exp/OCID-VLG_multiple_mac/CROG_mac_mps_official_params_50epoch_bs8/best_jindex_model.pth",
                    EXPECTED_HASHES["crog_checkpoint"],
                ),
                ("crog_clip", crog / "exp/pretrain_clip/RN50.pt", EXPECTED_HASHES["crog_clip"]),
            ]
        )
    else:
        key = route.lower()
        selected = repo / BACKEND_RUN_RELATIVE / f"selected_configs/{route}.json"
        selected_payload = json.loads(selected.read_text(encoding="utf-8"))
        required.extend(
            [
                (f"{key}_selected", selected, EXPECTED_HASHES[f"{key}_selected"]),
                (
                    f"{key}_checkpoint",
                    Path(str(selected_payload["finetuned_checkpoint"])),
                    EXPECTED_HASHES[f"{key}_checkpoint"],
                ),
                (
                    "native_inference",
                    repo / "experiments/fair_crog_hifics_g1_c1_no_rerank/native_inference.py",
                    EXPECTED_HASHES["native_inference"],
                ),
            ]
        )
    needs_hifi = route in {"G1", "C1"} or method != "native"
    if needs_hifi:
        required.extend(
            [
                ("hifi_config", repo / HIFI_RUN_RELATIVE / "config.yaml", EXPECTED_HASHES["hifi_config"]),
                (
                    "hifi_checkpoint",
                    repo / HIFI_RUN_RELATIVE / "checkpoints/best.pth",
                    EXPECTED_HASHES["hifi_checkpoint"],
                ),
                (
                    "hifi_clip",
                    Path.home() / ".cache/clip/ViT-B-16.pt",
                    EXPECTED_HASHES["hifi_clip"],
                ),
            ]
        )
    if method != "native":
        required.extend(
            [
                ("feature_manifest", paths["feature_manifest"], None),
                ("calibration_manifest", paths["calibration"], None),
                ("ranker_manifest", paths["ranker_manifest"], None),
            ]
        )
    if method == "gated":
        required.append(("gate_selection", paths["gate_selection"], None))
    records = []
    for label, path, expected in required:
        path = path.expanduser().resolve()
        if not path.is_file() or path.is_symlink() or path.stat().st_size <= 0:
            raise RuntimeContractError(f"missing immutable asset {label}: {path}")
        observed = sha256_file(path) if verify_hashes or expected is not None else None
        if expected is not None and observed != expected:
            raise RuntimeContractError(f"{label} hash mismatch")
        records.append(
            {"label": label, "path": str(path), "sha256": observed, "bytes": path.stat().st_size}
        )
    feature_schema = None
    if method != "native":
        manifest = json.loads(paths["feature_manifest"].read_text(encoding="utf-8"))
        if (
            manifest.get("status") != "COMPLETE"
            or manifest.get("track") != "T2_matched_common"
            or int(manifest.get("model_feature_count", -1)) != 105
            or manifest.get("model_feature_schema_sha256") != MODEL_FEATURE_SCHEMA_SHA256
        ):
            raise RuntimeContractError("formal T2 feature schema is not locked and complete")
        feature_schema = list(manifest["model_feature_columns"])
    return {
        "status": "PASS",
        "route": route,
        "method": method,
        "device": "mps",
        "preregistration_sha256": PREREGISTRATION_SHA256,
        "records": records,
        "model_feature_columns": feature_schema,
        "candidate_feature_score_caches_read": False,
    }


def _load_manifest(repo: Path) -> pd.DataFrame:
    path = repo / PRIMARY_RUN_RELATIVE / "01_manifests/paired_test.parquet"
    columns = [
        "sample_id",
        "scene_id",
        "frame_id",
        "source_rgb_path",
        "source_rgb_sha256",
        "source_depth_path",
        "source_depth_sha256",
        "language",
        "language_sha256",
    ]
    frame = pd.read_parquet(path, columns=columns)
    if frame["sample_id"].astype(str).duplicated().any():
        raise RuntimeContractError("paired Test manifest contains duplicate sample IDs")
    return frame


def _verify_observation_assets(row: Mapping[str, Any]) -> None:
    _verified_file(Path(str(row["source_rgb_path"])), str(row["source_rgb_sha256"]), "source RGB")
    _verified_file(
        Path(str(row["source_depth_path"])), str(row["source_depth_sha256"]), "source depth"
    )
    observed = hashlib.sha256(str(row["language"]).encode("utf-8")).hexdigest()
    if observed != str(row["language_sha256"]):
        raise RuntimeContractError("language SHA-256 mismatch")


def _decode_observation(row: Mapping[str, Any]) -> RawObservation:
    from PIL import Image

    with Image.open(str(row["source_rgb_path"])) as image:
        rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    with Image.open(str(row["source_depth_path"])) as image:
        depth_mm = np.asarray(image)
    if depth_mm.dtype != np.uint16 or depth_mm.ndim != 2:
        raise RuntimeContractError(f"depth must be uint16 millimetres, got {depth_mm.dtype}")
    if rgb.shape[:2] != depth_mm.shape:
        raise RuntimeContractError("RGB/depth shape mismatch")
    return RawObservation(
        sample_id=str(row["sample_id"]),
        scene_id=str(row["scene_id"]),
        frame_id=str(row["frame_id"]),
        language=str(row["language"]),
        rgb=rgb,
        depth_m=depth_mm.astype(np.float32) / np.float32(1000.0),
    )


def _geometry_hash(row: Mapping[str, Any]) -> str:
    return canonical_sha256(
        [
            str(row["route"]),
            str(row["sample_id"]),
            str(row["candidate_id"]),
            int(row["native_rank"]),
            float(row["cx_px"]),
            float(row["cy_px"]),
            float(row["theta_deg"]),
            float(row["width_px"]),
            float(row["height_px"]),
        ]
    )


def _canonical_candidates(
    rows: list[dict[str, Any]], observation: RawObservation, route: str
) -> pd.DataFrame:
    columns = [
        "sample_id",
        "frame_id",
        "scene_id",
        "route",
        "split",
        "candidate_id",
        "native_rank",
        "native_score",
        "cx_px",
        "cy_px",
        "theta_deg",
        "width_px",
        "height_px",
        "valid",
        "source_checkpoint_sha256",
        "generator_config_sha256",
        "selected_config_sha256",
        "candidate_geometry_sha256",
    ]
    result_rows = []
    for item in rows:
        value = {
            "sample_id": observation.sample_id,
            "frame_id": observation.frame_id,
            "scene_id": observation.scene_id,
            "route": route,
            "split": "test",
            "candidate_id": str(item["candidate_id"]),
            "native_rank": int(item["native_rank"]),
            "native_score": float(item["native_score"]),
            "cx_px": float(item["cx_px"]),
            "cy_px": float(item["cy_px"]),
            "theta_deg": float(item["theta_deg"]),
            "width_px": float(item["width_px"]),
            "height_px": float(item["height_px"]),
            "valid": True,
            "source_checkpoint_sha256": LINEAGE[route]["checkpoint"],
            "generator_config_sha256": LINEAGE[route]["generator_config"],
            "selected_config_sha256": LINEAGE[route]["selected_config"],
        }
        value["candidate_geometry_sha256"] = _geometry_hash(value)
        result_rows.append(value)
    result = pd.DataFrame(result_rows, columns=columns)
    if len(result):
        result = result.sort_values(["native_rank", "candidate_id"], kind="mergesort").reset_index(drop=True)
        if result[["sample_id", "candidate_id"]].duplicated().any():
            raise RuntimeContractError("live decoder produced duplicate candidate IDs")
        if result["native_rank"].astype(int).tolist() != list(range(1, len(result) + 1)):
            raise RuntimeContractError("live decoder produced non-contiguous native ranks")
        if len(result) > 5:
            result = result.head(5).copy()
    return result


def _angle_error_degrees(first: float, second: float) -> float:
    return abs((float(first) - float(second) + 90.0) % 180.0 - 90.0)


def _optional_id(value: Any) -> str | None:
    if value is None or (not isinstance(value, str) and pd.isna(value)):
        return None
    result = str(value)
    return None if result in {"", "None", "nan", "<NA>"} else result


def _load_gate_bundle(repo: Path, route: str) -> _LiveGateBundle:
    from unified_reranking.gate import (
        SAFE_GATE_FEATURE_COLUMNS,
        ConservativeTransitionModel,
        GateOperatingPoint,
    )

    selection_path = _route_paths(repo, route)["gate_selection"]
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    if (
        selection.get("status") != "COMPLETE"
        or selection.get("test_access") != "NONE"
        or str(selection.get("configuration", {}).get("route", "")).upper() != route
    ):
        raise RuntimeContractError(f"{route} gate selection is not development-only COMPLETE")
    columns = tuple(selection["configuration"]["feature_columns"])
    if columns != tuple(SAFE_GATE_FEATURE_COLUMNS):
        raise RuntimeContractError(f"{route} gate feature schema differs from the safe schema")
    model_path = _verified_record_path(
        selection["artifacts"]["transition_model"], f"{route} transition gate"
    )
    with model_path.open("rb") as handle:
        model = pickle.load(handle)
    if not isinstance(model, ConservativeTransitionModel):
        raise RuntimeContractError(f"{route} transition gate has an unexpected type")
    if tuple(model.feature_names_ or ()) != columns:
        raise RuntimeContractError(f"{route} transition gate schema mismatch")
    payload = selection.get("selection", {}).get("selected_operating_point")
    point = None if payload is None else GateOperatingPoint(**payload)
    return _LiveGateBundle(model=model, feature_columns=columns, operating_point=point, model_path=model_path)


def _gate_input(features: pd.DataFrame, scores: np.ndarray, ensemble: np.ndarray) -> tuple[pd.DataFrame, str, str, int]:
    from unified_reranking.gate import SAFE_GATE_FEATURE_COLUMNS

    ordered = features.sort_values(["native_rank", "candidate_id"], kind="mergesort").reset_index(drop=True)
    ranked = ordered.assign(ensemble_score=np.asarray(ensemble, dtype=float)).sort_values(
        ["ensemble_score", "native_rank", "candidate_id"],
        ascending=[False, True, True],
        kind="mergesort",
    )
    challenger = ranked.iloc[0]
    native = ordered.iloc[0]
    challenger_id = str(challenger["candidate_id"])
    native_id = str(native["candidate_id"])
    votes = 0
    for index in range(scores.shape[1]):
        seed = ordered.assign(_score=scores[:, index]).sort_values(
            ["_score", "native_rank", "candidate_id"],
            ascending=[False, True, True],
            kind="mergesort",
        ).iloc[0]
        votes += int(str(seed["candidate_id"]) == challenger_id)

    def values(row: pd.Series) -> dict[str, float]:
        return {
            "calibrated": float(row["calibrated_native_probability"]),
            "native_score": float(row["native_score_raw"]),
            "reliability": float(row["overall_feature_reliability"]),
            "stability": float(min(row["peak_retention_rate"], row["perturbed_valid_fraction"])),
            "mask": float(row["mask_reliability"]),
        }

    n, c = values(native), values(challenger)
    exists = challenger_id != native_id
    row = {
        "ranker_score_margin": float(challenger["ensemble_score"] - ensemble[0]),
        "challenger_calibrated_probability": c["calibrated"],
        "native_calibrated_probability": n["calibrated"],
        "calibrated_probability_delta": c["calibrated"] - n["calibrated"],
        "native_score_delta": c["native_score"] - n["native_score"],
        "challenger_overall_reliability": c["reliability"],
        "overall_reliability_delta": c["reliability"] - n["reliability"],
        "challenger_perturbation_stability": c["stability"],
        "perturbation_stability_delta": c["stability"] - n["stability"],
        "challenger_mask_reliability": c["mask"],
        "mask_reliability_delta": c["mask"] - n["mask"],
        "challenger_exists_numeric": float(exists),
    }
    frame = pd.DataFrame([row], columns=list(SAFE_GATE_FEATURE_COLUMNS))
    if not np.isfinite(frame.to_numpy(float)).all():
        raise RuntimeContractError("live gate input is non-finite")
    return frame, native_id, challenger_id, votes


class FourDRuntimeWorker:
    """One route/method worker intended to live in one fresh subprocess."""

    def __init__(self, repo: Path, run_dir: Path, route: str, method: str):
        self.repo = repo.resolve()
        self.run_dir = require_run_dir(self.repo, run_dir)
        verify_preregistration(self.run_dir)
        self.route = route.upper()
        self.method = method.lower()
        if self.route not in SUPPORTED_ROUTES or self.method not in SUPPORTED_METHODS:
            raise ValueError("unsupported route/method")
        if self.route == "D1":
            raise RuntimeContractError("D1 requires the external multi-runtime adapter")
        self.paths = _route_paths(self.repo, self.route)
        self.manifest: pd.DataFrame | None = None
        self.model: Any = None
        self.model_config: Any = None
        self.formal_native: Any = None
        self.crog_cfg: Any = None
        self.crog_tokenizer: Any = None
        self.crog_detect: Any = None
        self.hifi_model: Any = None
        self.hifi_config: dict[str, Any] | None = None
        self.hifi_transform: Any = None
        self.ranker: Any = None
        self.gate: _LiveGateBundle | None = None
        self.calibrator: Any = None
        self.model_columns: tuple[str, ...] = ()
        self.source: Any = None
        self.loaded = False
        self.startup_ns: int | None = None
        self._verified_observation_ids: set[str] = set()

    def load(self) -> dict[str, Any]:
        if self.loaded:
            raise RuntimeContractError("worker models may be loaded exactly once per process")
        preflight = asset_preflight(
            self.repo, self.run_dir, self.route, method=self.method, verify_hashes=True
        )
        started = time.perf_counter_ns()
        # The formal loader requires LightGBM/OpenMP to initialise before Torch.
        if self.method != "native":
            from robustness_suite.runtime_profile import (
                RouteSource,
                _load_formal_ranker_bundle,
            )

            self.source = RouteSource(
                route=self.route,
                dimension="4D",
                device="mps",
                candidate_path=self.paths["candidate"],
                feature_path=self.paths["feature"],
                prediction_path=self.paths["score"],
            )
            self.ranker = _load_formal_ranker_bundle(self.repo, self.source)
            calibration = json.loads(self.paths["calibration"].read_text(encoding="utf-8"))
            if calibration.get("status") != "COMPLETE":
                raise RuntimeContractError("formal native-score calibration is incomplete")
            from unified_reranking.calibration import calibrator_from_serialized

            selected = str(calibration["selected_method"])
            self.calibrator = calibrator_from_serialized(
                calibration["calibrators"]["full_train"][selected]
            )
            feature_manifest = json.loads(
                self.paths["feature_manifest"].read_text(encoding="utf-8")
            )
            self.model_columns = tuple(feature_manifest["model_feature_columns"])
            if canonical_sha256(self.model_columns) != MODEL_FEATURE_SCHEMA_SHA256:
                raise RuntimeContractError("formal T2 feature schema hash mismatch")
        if self.method == "gated":
            self.gate = _load_gate_bundle(self.repo, self.route)
        if self.route == "CROG":
            self._load_crog()
            if self.method != "native":
                self._load_hifi()
        else:
            self._load_hifi()
            self._load_backend()
        self.manifest = _load_manifest(self.repo)
        synchronize_device("mps")
        self.startup_ns = time.perf_counter_ns() - started
        self.loaded = True
        return {**preflight, "startup_model_load_ns": self.startup_ns}

    def _load_crog(self) -> None:
        import logging
        import types

        import torch

        root = (self.repo / "crog_reproduction/CROG").resolve()
        _prepend_sys_path(root)
        # CROG uses loguru only for one informational line in model/__init__.py.
        # The locked inference environments no longer contain that optional
        # logger, so provide a standard-library logger without touching model
        # construction, weights, inputs, or numerical execution.
        try:
            importlib.import_module("loguru")
        except ModuleNotFoundError:
            compatibility = types.ModuleType("loguru")
            compatibility.logger = logging.getLogger("crog.runtime")
            sys.modules["loguru"] = compatibility
        config_module = importlib.import_module("utils.config")
        model_module = importlib.import_module("model")
        checkpoint_module = importlib.import_module("utils.checkpoint")
        tokenizer_module = importlib.import_module("utils.simple_tokenizer")
        grasp_module = importlib.import_module("utils.grasp_eval")
        cfg_path = root / "config/OCID-VLG/CROG_mac_mps_official_params_50epoch_bs8.yaml"
        checkpoint = root / "exp/OCID-VLG_multiple_mac/CROG_mac_mps_official_params_50epoch_bs8/best_jindex_model.pth"
        cfg = config_module.load_cfg_from_cfg_file(str(cfg_path))
        cfg.clip_pretrain = str((root / str(cfg.clip_pretrain)).resolve())
        device = torch.device("mps")
        model, _ = model_module.build_crog(cfg)
        model = model.to(device).eval()
        checkpoint_module.load_checkpoint(checkpoint, model, device, strict=True)
        self.model = model
        self.crog_cfg = cfg
        self.crog_tokenizer = tokenizer_module.SimpleTokenizer()
        self.crog_detect = grasp_module.detect_grasp_candidates

    def _load_hifi(self) -> None:
        import torch
        import yaml
        from torchvision import transforms

        hifics_root = (self.repo / "HiFi_reproduction/hifics").resolve()
        _prepend_sys_path(hifics_root)
        model_class = importlib.import_module("models.hifics").HierarchicalCLIPDensePredT
        config_path = self.repo / HIFI_RUN_RELATIVE / "config.yaml"
        checkpoint = self.repo / HIFI_RUN_RELATIVE / "checkpoints/best.pth"
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        torch.manual_seed(int(config["seed"]))
        model = model_class(
            version=config["clip_backbone"],
            extract_layers=tuple(config["projection_layers"]),
            reduce_dim=int(config["decoder_dimension"]),
            n_heads=int(config.get("decoder_heads", 4)),
            cond_layer=None,
            extended_film=True,
            hierarchical_film=True,
        )
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if payload.get("format") != "hifics_hierfilm_trainable_only_v1":
            raise RuntimeContractError("unsupported repeated-FiLM checkpoint format")
        expected = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
        trainable = payload.get("trainable_state", {})
        if set(trainable) != expected:
            raise RuntimeContractError("repeated-FiLM trainable-state keys mismatch")
        complete = model.state_dict()
        complete.update(trainable)
        model.load_state_dict(complete, strict=True)
        if int(payload.get("metadata", {}).get("global_step", -1)) != 19728:
            raise RuntimeContractError("unexpected repeated-FiLM checkpoint step")
        model.to("mps").eval()
        model.clip_model.eval()
        self.hifi_model = model
        self.hifi_config = config
        self.hifi_transform = transforms.Compose(
            [
                transforms.Resize(
                    (int(config["image_resolution"]), int(config["image_resolution"]))
                ),
                transforms.ToTensor(),
            ]
        )

    def _load_backend(self) -> None:
        path = self.repo / "experiments/fair_crog_hifics_g1_c1_no_rerank/native_inference.py"
        spec = importlib.util.spec_from_file_location("_formal_native_inference_runtime", path)
        if spec is None or spec.loader is None:
            raise RuntimeContractError("cannot import formal native inference module")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        selected_path = self.repo / BACKEND_RUN_RELATIVE / f"selected_configs/{self.route}.json"
        config, selected = module.load_config(self.route.lower(), selected_path, False)
        checkpoint = Path(str(selected["finetuned_checkpoint"])).resolve()
        model, digest, _payload = module.load_finetuned_model(
            checkpoint,
            backend="grconvnet" if self.route == "G1" else "ggcnn2",
            device="mps",
        )
        if digest != EXPECTED_HASHES[f"{self.route.lower()}_checkpoint"]:
            raise RuntimeContractError(f"{self.route} loaded checkpoint hash mismatch")
        self.formal_native = module
        self.model_config = config
        self.model = model

    def _crog_preprocess(self, observation: RawObservation) -> tuple[Any, Any, np.ndarray, tuple[int, int]]:
        import cv2
        import torch

        height, width = observation.rgb.shape[:2]
        size = int(self.crog_cfg.input_size)
        scale = min(size / height, size / width)
        new_height, new_width = height * scale, width * scale
        bias_x, bias_y = (size - new_width) / 2.0, (size - new_height) / 2.0
        source = np.asarray([[0, 0], [width, 0], [0, height]], np.float32)
        target = np.asarray(
            [[bias_x, bias_y], [new_width + bias_x, bias_y], [bias_x, new_height + bias_y]],
            np.float32,
        )
        forward = cv2.getAffineTransform(source, target)
        inverse = cv2.getAffineTransform(target, source)
        warped = cv2.warpAffine(
            observation.rgb,
            forward,
            (size, size),
            flags=cv2.INTER_CUBIC,
            borderValue=tuple((CROG_MEAN[:, 0, 0] * 255).tolist()),
        )
        normalized = (warped.transpose(2, 0, 1).astype(np.float32) / 255.0 - CROG_MEAN) / CROG_STD
        start = self.crog_tokenizer.encoder["<|startoftext|>"]
        end = self.crog_tokenizer.encoder["<|endoftext|>"]
        tokens = [start, *self.crog_tokenizer.encode(observation.language), end]
        word_length = int(self.crog_cfg.word_len)
        if len(tokens) > word_length:
            tokens = tokens[:word_length]
            tokens[-1] = end
        token_ids = torch.zeros(word_length, dtype=torch.long)
        token_ids[: len(tokens)] = torch.tensor(tokens, dtype=torch.long)
        return torch.from_numpy(normalized), token_ids, inverse, (height, width)

    def _crog_forward(self, prepared: tuple[Any, Any, np.ndarray, tuple[int, int]]) -> Any:
        import torch

        image, tokens, _inverse, _shape = prepared
        with torch.inference_mode():
            prediction, _ = self.model(
                image.unsqueeze(0).to("mps"),
                tokens.unsqueeze(0).to("mps"),
                None,
                None,
                None,
                None,
                None,
            )
        return prediction

    def _crog_decode(
        self, prediction: Any, prepared: tuple[Any, Any, np.ndarray, tuple[int, int]], observation: RawObservation
    ) -> pd.DataFrame:
        import cv2
        import torch
        import torch.nn.functional as functional

        image, _tokens, inverse, (height, width) = prepared
        instance, quality, sine, cosine, jaw_width = prediction
        values = [torch.sigmoid(instance), torch.sigmoid(quality), sine, cosine, torch.sigmoid(jaw_width)]
        if values[0].shape[-2:] != image.shape[-2:]:
            values = [
                functional.interpolate(
                    value, size=image.shape[-2:], mode="bicubic", align_corners=True
                )
                for value in values
            ]
        restored = [
            cv2.warpAffine(
                value[0, 0].float().cpu().numpy(),
                inverse,
                (width, height),
                flags=cv2.INTER_CUBIC,
            )
            for value in values
        ]
        _instance, quality_map, sine_map, cosine_map, width_map = restored
        candidates, _ = self.crog_detect(
            quality_map, sine_map, cosine_map, width_map, num_grasps=5
        )
        rows = []
        for rank, item in enumerate(sorted(candidates, key=lambda value: int(value["q_rank"])), 1):
            rows.append(
                {
                    "candidate_id": str(item["candidate_id"]),
                    "native_rank": rank,
                    "native_score": float(item["q_raw"]),
                    "cx_px": float(item["cx"]),
                    "cy_px": float(item["cy"]),
                    "theta_deg": float((float(item["angle_deg"]) + 90.0) % 180.0 - 90.0),
                    "width_px": float(item["width_px"]),
                    "height_px": float(item["height_px"]),
                }
            )
        return _canonical_candidates(rows, observation, self.route)

    def _hifi_preprocess(self, observation: RawObservation) -> Any:
        from PIL import Image

        return self.hifi_transform(Image.fromarray(observation.rgb, mode="RGB"))

    def _hifi_forward(self, tensor: Any, language: str) -> Any:
        import torch

        with torch.inference_mode():
            output = self.hifi_model(tensor.unsqueeze(0).to("mps"), [language])
        return output[0] if isinstance(output, (tuple, list)) else output

    def _hifi_postprocess(self, logits: Any, native_shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
        import cv2
        import torch
        from PIL import Image

        probability = torch.sigmoid(-logits)[0, 0].float().cpu().numpy().astype(np.float32, copy=False)
        threshold = float(self.hifi_config["validation_threshold"])
        binary = probability >= threshold
        height, width = native_shape
        native_mask = (
            np.asarray(
                Image.fromarray(binary.astype(np.uint8) * 255, mode="L").resize(
                    (width, height), resample=Image.Resampling.NEAREST
                ),
                dtype=np.uint8,
            )
            >= 128
        )
        native_probability = cv2.resize(
            probability, (width, height), interpolation=cv2.INTER_LINEAR
        ).astype(np.float32, copy=False)
        return native_probability, native_mask

    def _hifi_complete(self, observation: RawObservation) -> tuple[np.ndarray, np.ndarray]:
        tensor = self._hifi_preprocess(observation)
        logits = self._hifi_forward(tensor, observation.language)
        synchronize_device("mps")
        return self._hifi_postprocess(logits, observation.depth_m.shape)

    def _backend_preprocess(
        self, observation: RawObservation, probability: np.ndarray, mask: np.ndarray
    ) -> tuple[Any, Any]:
        import torch

        sample = self.formal_native.BackendSample(
            sample_id=observation.sample_id,
            rgb=observation.rgb,
            depth_m=observation.depth_m,
            predicted_mask=mask,
            probability_map=probability,
            mask_source="predicted",
        )
        conditioned = self.formal_native.prepare_conditioned_input(sample, self.model_config)
        model_input = conditioned.depth_chw
        if self.route == "G1":
            model_input = np.concatenate((conditioned.depth_chw, conditioned.rgb_chw), axis=0)
        tensor = torch.from_numpy(np.ascontiguousarray(model_input)).unsqueeze(0)
        return conditioned, tensor.to(device="mps", dtype=torch.float32)

    def _backend_forward(self, prepared: tuple[Any, Any]) -> Any:
        import torch

        _conditioned, tensor = prepared
        with torch.inference_mode():
            return self.model(tensor)

    def _backend_decode(
        self, outputs: Any, prepared: tuple[Any, Any], observation: RawObservation
    ) -> pd.DataFrame:
        from skimage.feature import peak_local_max

        conditioned, _tensor = prepared
        quality, cos_map, sin_map, width_map = self.formal_native.official_gaussian_post_process(
            outputs,
            expected_spatial_shape=(self.model_config.input_size, self.model_config.input_size),
        )
        decoder = self.formal_native.DECODER
        peaks = peak_local_max(
            quality,
            min_distance=int(decoder["min_peak_distance_px"]),
            threshold_abs=float(decoder["quality_threshold"]),
            num_peaks=int(decoder["max_peaks"]),
        )
        ordered = sorted(
            ((int(row), int(column)) for row, column in peaks),
            key=lambda rc: (-float(quality[rc]), rc[0], rc[1]),
        )
        rows = []
        for peak_rank, (row, column) in enumerate(ordered, 1):
            angle = float(np.degrees(0.5 * np.arctan2(sin_map[row, column], cos_map[row, column])))
            model_width = float(width_map[row, column])
            if not np.isfinite(model_width) or model_width <= 0:
                continue
            cx, cy, native_angle, native_width = conditioned.transform.model_to_native_pose(
                column, row, angle, model_width, clip=False
            )
            values = (cx, cy, native_angle, native_width, float(quality[row, column]))
            if not all(np.isfinite(values)) or native_width <= 0:
                continue
            rows.append(
                {
                    "candidate_id": f"native_peak_{peak_rank:03d}",
                    "native_rank": len(rows) + 1,
                    "native_score": float(quality[row, column]),
                    "cx_px": float(cx),
                    "cy_px": float(cy),
                    "theta_deg": float((native_angle + 90.0) % 180.0 - 90.0),
                    "width_px": float(native_width),
                    "height_px": float(native_width / 2.0),
                }
            )
            if len(rows) == 5:
                break
        return _canonical_candidates(rows, observation, self.route)

    def _features(self, candidates: pd.DataFrame, observation: RawObservation, probability: np.ndarray, mask: np.ndarray) -> pd.DataFrame:
        from unified_reranking.feature_extractors.common import extract_common_evidence
        from unified_reranking.feature_extractors.rgb import candidate_rgb_features
        from unified_reranking.feature_tracks import assemble_common_track

        common, _relations = extract_common_evidence(
            candidates,
            probability=probability,
            binary_mask=mask,
            depth_m=observation.depth_m,
        )
        rgb = candidate_rgb_features(candidates, observation.rgb)
        common = common.merge(
            rgb, on=["sample_id", "candidate_id"], how="left", validate="one_to_one"
        )
        calibrated = self.calibrator.predict(candidates["native_score"].to_numpy(float))
        calibration = candidates[["sample_id", "candidate_id"]].copy()
        calibration["calibrated_native_probability"] = calibrated
        calibration["base_logit"] = np.log(calibrated) - np.log1p(-calibrated)
        track = assemble_common_track(candidates, common, calibration)
        if track.model_columns != self.model_columns:
            raise RuntimeContractError("live T2 model feature columns differ from formal schema")
        if not np.isfinite(track.frame.loc[:, self.model_columns].to_numpy(float)).all():
            # Missingness indicators are explicit; the formal FoldPreprocessor handles
            # NaNs in raw features.  Infinities, however, are never valid.
            values = track.frame.loc[:, self.model_columns].to_numpy(float)
            if np.isinf(values).any():
                raise RuntimeContractError("live T2 feature table contains infinity")
        return track.frame.sort_values(["native_rank", "candidate_id"], kind="mergesort").reset_index(drop=True)

    def _rank(self, features: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, str]:
        from robustness_suite.runtime_profile import _ranker_scores_for_group

        scores, ensemble = _ranker_scores_for_group(self.ranker, features)
        ordered = features.assign(ensemble_score=ensemble).sort_values(
            ["ensemble_score", "native_rank", "candidate_id"],
            ascending=[False, True, True],
            kind="mergesort",
        )
        return scores, ensemble, str(ordered.iloc[0]["candidate_id"])

    def _apply_gate(
        self, features: pd.DataFrame, scores: np.ndarray, ensemble: np.ndarray
    ) -> tuple[str, float, float, bool]:
        from unified_reranking.gate import GateEvidence, gate_switch_mask

        if self.gate is None:
            raise RuntimeContractError("gate model is not loaded")
        gate_input, native, challenger, votes = _gate_input(features, scores, ensemble)
        matrix = gate_input.loc[:, self.gate.feature_columns].to_numpy(float)
        recover, harm = self.gate.model.predict_probabilities(matrix)
        point = self.gate.operating_point
        switch = False
        if point is not None:
            challenger_row = features.loc[features["candidate_id"].astype(str) == challenger].iloc[0]
            evidence = GateEvidence(
                score_margin=gate_input["ranker_score_margin"].to_numpy(),
                challenger_reliability=np.asarray(
                    [float(challenger_row["overall_feature_reliability"])]
                ),
                perturbation_stability=np.asarray(
                    [
                        float(
                            min(
                                challenger_row["peak_retention_rate"],
                                challenger_row["perturbed_valid_fraction"],
                            )
                        )
                    ]
                ),
                seed_challenger_votes=np.asarray([votes]),
                candidate_id_unchanged=np.asarray([True]),
                geometry_hash_unchanged=np.asarray([True]),
                challenger_exists=np.asarray([challenger != native]),
            )
            switch = bool(gate_switch_mask(recover, harm, evidence, point)[0])
        return (challenger if switch else native), float(recover[0]), float(harm[0]), switch

    @staticmethod
    def _execute(
        recorder: _StageRecorder | None,
        stage: str,
        function: Callable[[], T],
        *,
        device: str = "cpu",
    ) -> T:
        return function() if recorder is None else recorder.call(stage, function, device=device)

    def run_observation(
        self,
        observation: RawObservation,
        *,
        instrument: bool,
        cache_policy: str,
        input_io_ns: int = 0,
    ) -> PipelineResult:
        if not self.loaded:
            raise RuntimeContractError("worker must be loaded before inference")
        recorder = (
            _StageRecorder(self.route, self.method, observation.sample_id, cache_policy)
            if instrument
            else None
        )
        outer_start = time.perf_counter_ns() if instrument else None
        if recorder is not None:
            recorder.add(
                "input_io",
                int(input_io_ns),
                status="MEASURED" if input_io_ns else "NOT_APPLICABLE_PRELOADED",
            )
        probability = mask = None
        if self.route == "CROG":
            prepared = self._execute(
                recorder, "preprocess", lambda: self._crog_preprocess(observation)
            )
            prediction = self._execute(
                recorder, "crog_model_inference", lambda: self._crog_forward(prepared), device="mps"
            )
            candidates = self._execute(
                recorder,
                "candidate_decode",
                lambda: self._crog_decode(prediction, prepared, observation),
            )
        else:
            hifi_tensor = self._execute(
                recorder, "preprocess", lambda: self._hifi_preprocess(observation)
            )
            logits = self._execute(
                recorder,
                "hifics_inference",
                lambda: self._hifi_forward(hifi_tensor, observation.language),
                device="mps",
            )
            probability, mask = self._execute(
                recorder,
                "mask_postprocess",
                lambda: self._hifi_postprocess(logits, observation.depth_m.shape),
            )
            prepared = self._execute(
                recorder,
                "preprocess",
                lambda: self._backend_preprocess(observation, probability, mask),
            )
            outputs = self._execute(
                recorder,
                "grasp_backend_inference",
                lambda: self._backend_forward(prepared),
                device="mps",
            )
            candidates = self._execute(
                recorder,
                "candidate_decode",
                lambda: self._backend_decode(outputs, prepared, observation),
            )
        if recorder is not None:
            recorder.add(
                "nms",
                0,
                status="INTEGRATED_WITH_LOCKED_LOCAL_PEAK_CANDIDATE_DECODER",
            )
        native_id = None if candidates.empty else str(candidates.iloc[0]["candidate_id"])
        features = pd.DataFrame()
        seed_scores = ensemble = None
        raw_id = gated_id = native_id
        recover = harm = None
        switch = False
        if self.method != "native" and not candidates.empty:
            if self.route == "CROG":
                def crog_features() -> pd.DataFrame:
                    nonlocal probability, mask
                    probability, mask = self._hifi_complete(observation)
                    return self._features(candidates, observation, probability, mask)

                features = self._execute(
                    recorder,
                    "runtime_feature_extraction",
                    crog_features,
                    device="mps",
                )
            else:
                assert probability is not None and mask is not None
                features = self._execute(
                    recorder,
                    "runtime_feature_extraction",
                    lambda: self._features(candidates, observation, probability, mask),
                )
            seed_scores, ensemble, raw_id = self._execute(
                recorder, "reranker", lambda: self._rank(features)
            )
            gated_id = raw_id
            if self.method == "gated":
                gated_id, recover, harm, switch = self._execute(
                    recorder,
                    "gate",
                    lambda: self._apply_gate(features, seed_scores, ensemble),
                )
        selected = self._execute(
            recorder,
            "final_selection",
            lambda: native_id
            if self.method == "native"
            else raw_id
            if self.method == "raw"
            else gated_id,
        )
        rows = [] if recorder is None else recorder.finish(len(candidates))
        instrumented_total = None
        if outer_start is not None:
            synchronize_device("mps")
            instrumented_total = time.perf_counter_ns() - outer_start + int(input_io_ns)
        return PipelineResult(
            sample_id=observation.sample_id,
            route=self.route,
            method=self.method,
            candidates=candidates,
            features=features,
            seed_scores=seed_scores,
            ensemble_scores=ensemble,
            native_candidate_id=native_id,
            raw_candidate_id=raw_id,
            gated_candidate_id=gated_id,
            selected_candidate_id=selected,
            gate_probability_recover=recover,
            gate_probability_harm=harm,
            gate_switch=switch,
            stage_rows=rows,
            instrumented_total_ns=instrumented_total,
        )

    def run_row(
        self, row: Mapping[str, Any], *, instrument: bool, cache_policy: str
    ) -> PipelineResult:
        sample_id = str(row["sample_id"])
        if sample_id not in self._verified_observation_ids:
            _verify_observation_assets(row)
            self._verified_observation_ids.add(sample_id)
        if instrument:
            elapsed, observation = timed_call_ns(lambda: _decode_observation(row), device="cpu")
        else:
            elapsed, observation = 0, _decode_observation(row)
        return self.run_observation(
            observation,
            instrument=instrument,
            cache_policy=cache_policy,
            input_io_ns=elapsed,
        )

    def _rows_by_ids(self, sample_ids: Sequence[str]) -> list[dict[str, Any]]:
        if self.manifest is None:
            raise RuntimeContractError("manifest is not loaded")
        indexed = self.manifest.assign(sample_id=self.manifest["sample_id"].astype(str)).set_index(
            "sample_id", drop=False
        )
        missing = sorted(set(map(str, sample_ids)).difference(indexed.index))
        if missing:
            raise RuntimeContractError(f"subset misses sample IDs: {missing[:5]}")
        return [indexed.loc[str(sample_id)].to_dict() for sample_id in sample_ids]

    def run_parity(self, sample_ids: Sequence[str]) -> dict[str, Any]:
        if self.method != "gated":
            raise RuntimeContractError("parity must load the full gated route")
        if len(sample_ids) != PARITY_COUNT or len(set(sample_ids)) != PARITY_COUNT:
            raise RuntimeContractError("parity requires exactly 20 distinct locked sample IDs")
        rows = self._rows_by_ids(sample_ids)
        live = []
        for row in rows:
            live.append(self.run_row(row, instrument=False, cache_policy="parity_live"))

        # This is the only method that opens frozen candidate/feature/score/decision data.
        ids = set(map(str, sample_ids))
        expected_candidates = pd.read_parquet(self.paths["candidate"])
        expected_candidates = expected_candidates.loc[
            expected_candidates["sample_id"].astype(str).isin(ids)
        ].copy()
        expected_features = pd.read_parquet(
            self.paths["feature"], columns=["sample_id", "candidate_id", *self.model_columns]
        )
        expected_features = expected_features.loc[
            expected_features["sample_id"].astype(str).isin(ids)
        ].copy()
        expected_scores = pd.read_parquet(self.paths["score"])
        expected_scores = expected_scores.loc[
            expected_scores["sample_id"].astype(str).isin(ids)
        ].copy()
        gate_manifest = json.loads(self.paths["gate_manifest"].read_text(encoding="utf-8"))
        decision_path = _verified_record_path(
            gate_manifest["artifacts"]["decisions"], f"{self.route} parity gate decisions"
        )
        expected_gate = pd.read_parquet(decision_path)
        expected_gate = expected_gate.loc[expected_gate["sample_id"].astype(str).isin(ids)].copy()
        failures: list[dict[str, Any]] = []
        parity_rows: list[dict[str, Any]] = []
        maxima = {
            "center_px": 0.0,
            "angle_rad": 0.0,
            "width_px": 0.0,
            "height_px": 0.0,
            "native_score_abs": 0.0,
            "feature_abs": 0.0,
            "ranker_score_abs": 0.0,
            "gate_probability_abs": 0.0,
        }
        for result in live:
            sample_id = result.sample_id
            actual = result.candidates.sort_values(
                ["native_rank", "candidate_id"], kind="mergesort"
            ).reset_index(drop=True)
            expected = expected_candidates.loc[
                expected_candidates["sample_id"].astype(str) == sample_id
            ].sort_values(["native_rank", "candidate_id"], kind="mergesort").reset_index(drop=True)
            reasons = []
            if actual["candidate_id"].astype(str).tolist() != expected["candidate_id"].astype(str).tolist():
                reasons.append("candidate_identity_or_count")
            elif len(actual):
                center = np.abs(
                    actual[["cx_px", "cy_px"]].to_numpy(float)
                    - expected[["cx_px", "cy_px"]].to_numpy(float)
                ).max()
                angle = max(
                    np.radians(_angle_error_degrees(a, b))
                    for a, b in zip(actual["theta_deg"], expected["theta_deg"], strict=True)
                )
                width = np.abs(actual["width_px"].to_numpy(float) - expected["width_px"].to_numpy(float)).max()
                height = np.abs(actual["height_px"].to_numpy(float) - expected["height_px"].to_numpy(float)).max()
                native_score = np.abs(
                    actual["native_score"].to_numpy(float) - expected["native_score"].to_numpy(float)
                ).max()
                maxima["center_px"] = max(maxima["center_px"], float(center))
                maxima["angle_rad"] = max(maxima["angle_rad"], float(angle))
                maxima["width_px"] = max(maxima["width_px"], float(width))
                maxima["height_px"] = max(maxima["height_px"], float(height))
                maxima["native_score_abs"] = max(maxima["native_score_abs"], float(native_score))
                if center > 1.0 or angle > 1e-4 or width > 1.0 or height > 1.0:
                    reasons.append("candidate_geometry_tolerance")
                if not np.allclose(
                    actual["native_score"], expected["native_score"], atol=1e-5, rtol=1e-5, equal_nan=True
                ):
                    reasons.append("native_score_tolerance")
                if actual["candidate_id"].astype(str).tolist() == expected["candidate_id"].astype(str).tolist():
                    actual_features = result.features.sort_values(
                        ["native_rank", "candidate_id"], kind="mergesort"
                    )
                    reference_features = expected_features.loc[
                        expected_features["sample_id"].astype(str) == sample_id
                    ].set_index("candidate_id").loc[actual_features["candidate_id"].astype(str)].reset_index()
                    left = actual_features.loc[:, self.model_columns].to_numpy(float)
                    right = reference_features.loc[:, self.model_columns].to_numpy(float)
                    finite_diff = np.abs(left - right)
                    finite_diff = finite_diff[np.isfinite(finite_diff)]
                    maxima["feature_abs"] = max(
                        maxima["feature_abs"], float(finite_diff.max(initial=0.0))
                    )
                    if not np.allclose(left, right, atol=1e-5, rtol=1e-5, equal_nan=True):
                        reasons.append("live_T2_feature_tolerance")
                    reference_scores = expected_scores.loc[
                        expected_scores["sample_id"].astype(str) == sample_id
                    ].set_index("candidate_id").loc[actual_features["candidate_id"].astype(str)]
                    actual_scores = np.column_stack(
                        [result.seed_scores, result.ensemble_scores]
                    )
                    score_columns = [*(f"score_seed_{seed}" for seed in FORMAL_SEEDS), "ensemble_score"]
                    right_scores = reference_scores.loc[:, score_columns].to_numpy(float)
                    maxima["ranker_score_abs"] = max(
                        maxima["ranker_score_abs"],
                        float(np.abs(actual_scores - right_scores).max(initial=0.0)),
                    )
                    if not np.allclose(actual_scores, right_scores, atol=1e-5, rtol=1e-5):
                        reasons.append("live_ranker_score_tolerance")
            gate_row = expected_gate.loc[expected_gate["sample_id"].astype(str) == sample_id]
            if len(gate_row) != 1:
                reasons.append("formal_gate_denominator")
            else:
                gate_row = gate_row.iloc[0]
                expected_selected = _optional_id(gate_row["selected_candidate_id"])
                expected_native = _optional_id(gate_row["native_candidate_id"])
                expected_raw = (
                    _optional_id(gate_row["challenger_candidate_id"])
                    or expected_native
                )
                if result.native_candidate_id != expected_native:
                    reasons.append("native_top1")
                if result.raw_candidate_id != expected_raw:
                    reasons.append("raw_top1")
                if result.gated_candidate_id != expected_selected:
                    reasons.append("gated_top1")
                if result.gate_probability_recover is not None:
                    delta = max(
                        abs(result.gate_probability_recover - float(gate_row["probability_recover"])),
                        abs(result.gate_probability_harm - float(gate_row["probability_harm"])),
                    )
                    maxima["gate_probability_abs"] = max(maxima["gate_probability_abs"], delta)
                    if delta > 1e-5 + 1e-5 * max(
                        abs(float(gate_row["probability_recover"])),
                        abs(float(gate_row["probability_harm"])),
                    ):
                        reasons.append("live_gate_probability_tolerance")
            if reasons:
                failures.append({"sample_id": sample_id, "reasons": sorted(set(reasons))})
            candidate_reasons = {
                "candidate_identity_or_count",
                "candidate_geometry_tolerance",
                "native_score_tolerance",
                "native_top1",
            }
            feature_reasons = {"live_T2_feature_tolerance"}
            ranker_reasons = {"live_ranker_score_tolerance", "raw_top1"}
            gate_reasons = {
                "formal_gate_denominator",
                "gated_top1",
                "live_gate_probability_tolerance",
            }
            reason_set = set(reasons)
            candidate_ok = not bool(reason_set & candidate_reasons)
            feature_ok = candidate_ok and not bool(reason_set & feature_reasons)
            ranker_ok = feature_ok and not bool(reason_set & ranker_reasons)
            gate_ok = ranker_ok and not bool(reason_set & gate_reasons)
            parity_rows.append(
                {
                    "route": self.route,
                    "method": "gated",
                    "mode": "parity",
                    "sample_id": sample_id,
                    "device": "mps",
                    "status": "PASS" if not reasons else "FAILED_PARITY",
                    "whole_elapsed_ns": None,
                    "candidate_count": int(len(actual)),
                    "candidate_parity": candidate_ok,
                    "feature_parity": feature_ok,
                    "ranker_parity": ranker_ok,
                    "gate_parity": gate_ok,
                }
            )
        passed = not failures
        return {
            "status": "PASS" if passed else "FAILED_PARITY",
            "complete_deployment": passed,
            "route": self.route,
            "method": "gated",
            "preregistration_sha256": PREREGISTRATION_SHA256,
            "parity_sample_ids": list(map(str, sample_ids)),
            "parity_count": len(sample_ids),
            "cache_use": "parity_comparison_only",
            "tolerances": {
                "center_px": 1.0,
                "angle_rad": 1e-4,
                "width_px": 1.0,
                "height_px": 1.0,
                "numeric_atol": 1e-5,
                "numeric_rtol": 1e-5,
            },
            "maxima": maxima,
            "failure_count": len(failures),
            "failures": failures,
            "parity_rows": parity_rows,
            "candidate_parity": all(row["candidate_parity"] for row in parity_rows),
            "feature_parity": all(row["feature_parity"] for row in parity_rows),
            "ranker_parity": all(row["ranker_parity"] for row in parity_rows),
            "gate_parity": all(row["gate_parity"] for row in parity_rows),
            "timings": [],
        }


def validate_parity_proof(path: Path, route: str) -> dict[str, Any]:
    proof = json.loads(path.read_text(encoding="utf-8"))
    if (
        proof.get("status") != "PASS"
        or proof.get("complete_deployment") is not True
        or str(proof.get("route", "")).upper() != route.upper()
        or proof.get("preregistration_sha256") != PREREGISTRATION_SHA256
        or int(proof.get("parity_count", -1)) != PARITY_COUNT
    ):
        raise RuntimeContractError("a matching PASS 20-sample live parity proof is required")
    return proof


def _measurement_rows(
    worker: FourDRuntimeWorker,
    rows: Sequence[Mapping[str, Any]],
    *,
    mode: str,
    checkpoint_path: Path | None = None,
    resume: bool = False,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    preloaded: dict[str, RawObservation] = {}
    if mode == "warm-preloaded":
        for row in rows:
            _verify_observation_assets(row)
            preloaded[str(row["sample_id"])] = _decode_observation(row)

    def execute(row: Mapping[str, Any], instrument: bool) -> PipelineResult:
        if mode == "warm-preloaded":
            return worker.run_observation(
                preloaded[str(row["sample_id"])],
                instrument=instrument,
                cache_policy="warm_preloaded",
                input_io_ns=0,
            )
        return worker.run_row(row, instrument=instrument, cache_policy="warm_disk")

    if len(rows) < WARMUP_COUNT + MEASURED_COUNT:
        raise RuntimeContractError("warm profiling requires five warmups plus 100 measured rows")
    warmups = []
    for row in rows[:WARMUP_COUNT]:
        result = execute(row, False)
        warmups.append(
            {"sample_id": result.sample_id, "selection_signature": result.selection_signature()}
        )
    measured_ids = [str(row["sample_id"]) for row in rows[WARMUP_COUNT : WARMUP_COUNT + MEASURED_COUNT]]
    signature = canonical_sha256(
        {
            "preregistration_sha256": PREREGISTRATION_SHA256,
            "route": worker.route,
            "method": worker.method,
            "mode": mode,
            "measured_sample_ids": measured_ids,
        }
    )
    timings: list[dict[str, Any]] = []
    samples: list[dict[str, Any]] = []
    if checkpoint_path is not None and checkpoint_path.exists():
        if not resume:
            raise RuntimeContractError(
                f"warm timing checkpoint exists; pass --resume: {checkpoint_path}"
            )
        checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        if checkpoint.get("signature") != signature:
            raise RuntimeContractError("warm timing checkpoint signature mismatch")
        timings = list(checkpoint.get("timings", []))
        samples = list(checkpoint.get("samples", []))
        observed_ids = [str(item.get("sample_id")) for item in samples]
        if observed_ids != measured_ids[: len(observed_ids)]:
            raise RuntimeContractError("warm timing checkpoint is not the locked subset prefix")
        if {str(item.get("sample_id")) for item in timings} != set(observed_ids):
            raise RuntimeContractError("warm timing checkpoint stage/sample coverage differs")
    completed_ids = {str(item["sample_id"]) for item in samples}
    for row in rows[WARMUP_COUNT : WARMUP_COUNT + MEASURED_COUNT]:
        if str(row["sample_id"]) in completed_ids:
            continue
        synchronize_device("mps")
        start = time.perf_counter_ns()
        uninstrumented = execute(row, False)
        synchronize_device("mps")
        outer_ns = time.perf_counter_ns() - start
        instrumented = execute(row, True)
        if uninstrumented.selection_signature() != instrumented.selection_signature():
            raise RuntimeContractError(
                f"instrumented replay changed live selection: {uninstrumented.sample_id}"
            )
        stage_sum = sum(int(item["elapsed_ns"]) for item in instrumented.stage_rows)
        timings.extend(instrumented.stage_rows)
        samples.append(
            {
                "route": worker.route,
                "method": worker.method,
                "mode": mode,
                "sample_id": uninstrumented.sample_id,
                "device": "mps",
                "status": "MEASURED_FULL_DEPLOYMENT",
                "candidate_count": int(len(uninstrumented.candidates)),
                "whole_elapsed_ns": int(outer_ns),
                "uninstrumented_total_ns": int(outer_ns),
                "uninstrumented_total_ms": outer_ns / 1e6,
                "instrumented_total_ns": int(instrumented.instrumented_total_ns),
                "stage_sum_ns": int(stage_sum),
                "instrumentation_residual_ns": int(instrumented.instrumented_total_ns - stage_sum),
                "selected_candidate_id": uninstrumented.selected_candidate_id,
                "selection_signature": uninstrumented.selection_signature(),
            }
        )
        if checkpoint_path is not None:
            atomic_json(
                checkpoint_path,
                {
                    "schema_version": 1,
                    "signature": signature,
                    "status": "IN_PROGRESS",
                    "route": worker.route,
                    "method": worker.method,
                    "mode": mode,
                    "completed_count": len(samples),
                    "expected_count": MEASURED_COUNT,
                    "warmups": warmups,
                    "samples": samples,
                    "timings": timings,
                },
            )
    if checkpoint_path is not None:
        atomic_json(
            checkpoint_path,
            {
                "schema_version": 1,
                "signature": signature,
                "status": "COMPLETE" if len(samples) == MEASURED_COUNT else "IN_PROGRESS",
                "route": worker.route,
                "method": worker.method,
                "mode": mode,
                "completed_count": len(samples),
                "expected_count": MEASURED_COUNT,
                "warmups": warmups,
                "samples": samples,
                "timings": timings,
            },
        )
    return timings, [{"warmups": warmups, "samples": samples}]


def run_cli_job(args: argparse.Namespace) -> dict[str, Any]:
    repo = args.repo_root.expanduser().resolve()
    run_dir = args.run_dir.expanduser().resolve()
    route = args.route.upper()
    method = args.method.lower()
    run_dir = require_run_dir(repo, run_dir)
    verify_preregistration(run_dir)
    if route == "D1":
        return {
            **_d1_status(repo, probe_docker=True),
            "mode": args.mode,
            "method": method,
            "preregistration_sha256": PREREGISTRATION_SHA256,
        }
    subset = build_four_d_subset_manifest(repo, run_dir)
    if args.mode != "parity":
        if args.parity_proof is None:
            raise RuntimeContractError("--parity-proof is required for every timing mode")
        validate_parity_proof(args.parity_proof.expanduser().resolve(), route)
    effective_method = "gated" if args.mode == "parity" else method
    worker = FourDRuntimeWorker(repo, run_dir, route, effective_method)
    preflight = worker.load()
    if args.mode == "parity":
        result = worker.run_parity(subset["parity_4d"]["sample_ids"])
        result["startup_model_load_ns"] = worker.startup_ns
        result["asset_preflight"] = preflight
        return result
    profile_ids = list(subset["profile_4d"]["sample_ids"])
    rows = worker._rows_by_ids(profile_ids)
    if args.mode == "cold":
        result = worker.run_row(rows[0], instrument=True, cache_policy="cold_fresh_process")
        result_row = {
            "route": route,
            "method": method,
            "mode": "cold",
            "sample_id": result.sample_id,
            "device": "mps",
            "status": "MEASURED_FULL_DEPLOYMENT",
            "whole_elapsed_ns": int(result.instrumented_total_ns),
            "candidate_count": int(len(result.candidates)),
            "selected_candidate_id": result.selected_candidate_id,
        }
        return {
            "status": "COMPLETE_COLD_FULL_DEPLOYMENT",
            "complete_deployment": True,
            "route": route,
            "method": method,
            "mode": "cold",
            "device": "mps",
            "startup_model_load_ns": worker.startup_ns,
            "sample_id": result.sample_id,
            "candidate_count": int(len(result.candidates)),
            "first_full_top1_ns": result.instrumented_total_ns,
            "selected_candidate_id": result.selected_candidate_id,
            "results": [result_row],
            "timings": result.stage_rows,
            "filesystem_cache_control": "uncontrolled; fresh process, local file decode",
            "asset_preflight": preflight,
            "preregistration_sha256": PREREGISTRATION_SHA256,
        }
    # Prepend five distinct outcome-blind rows not present in the measured 100.
    all_ids = worker.manifest["sample_id"].astype(str).tolist()
    measured = set(profile_ids)
    warmup_ids = sorted(
        (sample_id for sample_id in all_ids if sample_id not in measured),
        key=lambda value: (hashlib.sha256(value.encode()).hexdigest(), value),
    )[:WARMUP_COUNT]
    if len(warmup_ids) != WARMUP_COUNT:
        raise RuntimeContractError("cannot select five distinct warmup samples")
    selected_rows = worker._rows_by_ids([*warmup_ids, *profile_ids])
    checkpoint = args.output.expanduser().resolve().with_suffix(".progress.json")
    timings, payload = _measurement_rows(
        worker,
        selected_rows,
        mode=args.mode,
        checkpoint_path=checkpoint,
        resume=bool(args.resume),
    )
    return {
        "status": "COMPLETE_WARM_FULL_DEPLOYMENT",
        "complete_deployment": True,
        "route": route,
        "method": method,
        "mode": args.mode,
        "device": "mps",
        "startup_model_load_ns": worker.startup_ns,
        "warmup_count": WARMUP_COUNT,
        "measured_count": MEASURED_COUNT,
        "warmup_sample_ids": warmup_ids,
        "measured_sample_ids": profile_ids,
        "progress_checkpoint": str(checkpoint),
        "timings": timings,
        **payload[0],
        "uncontrolled_filesystem_page_cache": True,
        "asset_preflight": preflight,
        "preregistration_sha256": PREREGISTRATION_SHA256,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--route", type=str.lower, choices=tuple(value.lower() for value in SUPPORTED_ROUTES), required=True)
    parser.add_argument("--method", choices=SUPPORTED_METHODS, required=True)
    parser.add_argument("--mode", choices=SUPPORTED_MODES, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--parity-proof", type=Path)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    output = args.output.expanduser().resolve()
    repo = args.repo_root.expanduser().resolve()
    run_dir = require_run_dir(repo, args.run_dir.expanduser().resolve())
    worker_root = (run_dir / "runtime_full" / "workers" / "4d").resolve()
    if not output.is_relative_to(worker_root) or output.suffix.lower() != ".json":
        raise SystemExit("--output must be a JSON path under runtime_full/workers/4d")
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and args.resume:
        existing = json.loads(output.read_text(encoding="utf-8"))
        if existing.get("status") in {
            "PASS",
            "COMPLETE_COLD_FULL_DEPLOYMENT",
            "COMPLETE_WARM_FULL_DEPLOYMENT",
        }:
            expected_route = args.route.upper()
            expected_method = "gated" if args.mode == "parity" else args.method
            if (
                existing.get("preregistration_sha256") != PREREGISTRATION_SHA256
                or str(existing.get("route", "")).upper() != expected_route
                or str(existing.get("method", "")).lower() != expected_method
                or str(existing.get("mode", "")) != args.mode
            ):
                raise SystemExit("existing --resume result does not match this locked job")
            return 0
    started = time.time_ns()
    try:
        result = run_cli_job(args)
        result.update(
            {
                "schema_version": 1,
                "worker": "robustness_completion.runtime_4d",
                "started_wall_time_ns": started,
                "completed_wall_time_ns": time.time_ns(),
                "command": [sys.executable, "-m", "robustness_completion.runtime_4d", *sys.argv[1:]],
            }
        )
        atomic_json(output, result)
        return 0 if result.get("status") in {
            "PASS",
            "COMPLETE_COLD_FULL_DEPLOYMENT",
            "COMPLETE_WARM_FULL_DEPLOYMENT",
        } else 2
    except BaseException as error:
        failure = {
            "schema_version": 1,
            "status": "FAILED_PARITY" if args.mode == "parity" else "FAILED_PRECONDITION",
            "complete_deployment": False,
            "route": args.route.upper(),
            "method": args.method,
            "mode": args.mode,
            "preregistration_sha256": PREREGISTRATION_SHA256,
            "blockers": [f"{type(error).__name__}: {error}"],
            "timings": [],
            "started_wall_time_ns": started,
            "completed_wall_time_ns": time.time_ns(),
            "traceback": traceback.format_exc(),
            "command": [sys.executable, "-m", "robustness_completion.runtime_4d", *sys.argv[1:]],
        }
        atomic_json(output, failure)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
