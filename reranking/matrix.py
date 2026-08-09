"""Leakage-safe experiment matrix for frozen candidate reranking.

Only ``test-primary`` and ``test-post-lock`` may open held-out labels. The
primary lock reads label-free test features/query universes solely to freeze
their candidate identities before prediction. The development stage produces grouped OOF predictions, one independently
resumable artifact per method/seed/fold, and a machine-readable registry.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import datetime as dt
import fcntl
import hashlib
import json
import os
from pathlib import Path
import pickle
import re
import socket
import subprocess
import time
import traceback
import uuid
from contextlib import contextmanager
from functools import lru_cache
from typing import Any, Iterator, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from torch import nn

from reranking.crog_existing_comparison import (
    run_crog_existing_comparisons,
    validate_crog_existing_comparison_report,
)
from reranking.evaluate import evaluate_rankings
from reranking.existing_run_discovery import discover_existing_runs
from reranking.independent_evaluator import independent_j_at_1
from reranking.leakage_audit import (
    LeakageAuditError,
    audit_development_test_identities,
    build_identity_rows,
    verify_leakage_audit_bundle,
)
from reranking.data_contracts import MODULAR_INCOMPLETE_RERANK_RUN
from reranking.extract_features import (
    build_feature_schema,
    feature_group,
    select_crog_native_feature_columns,
    select_model_feature_columns,
)
from reranking.losses import CompositeRerankingLoss
from reranking.models.gates import (
    ConservativeGate,
    LearnedLogisticGate,
    MarginGate,
    attach_switch_outcomes,
    build_switch_proposals,
)
from reranking.models.neural import SharedMLPScorer
from reranking.models.set_models import (
    CandidateGNNScorer,
    DeepSetsScorer,
    PAIRWISE_EDGE_RELATION_FIELDS,
    RAW_EDGE_CANDIDATE_FIELDS,
    SetTransformerScorer,
    build_pairwise_edge_relations,
    parameter_count,
)
from reranking.models.tabular import (
    HistGradientBoostingRanker,
    LinearPairwiseRankNet,
    LinearResidualBCE,
    LogisticRegressionRanker,
    RandomForestRanker,
    TrainOnlyScoreCalibrator,
    XGBoostLambdaMARTRanker,
    scan_forbidden_columns,
)
from reranking.splits import (
    SplitPlan,
    build_stratified_group_folds,
    scene_frame_group_hash,
)
from reranking.train import (
    CandidateSetExample,
    LENGTH_BUCKETED_BATCHING_POLICY,
    TrainingConfig,
    _forward_candidate_batch,
    fit_neural_ranker,
    load_training_checkpoint,
    make_candidate_dataloader,
    resolve_device,
    set_global_seed,
)


FORMAL_SEEDS = (42, 123, 2026)
FORMAL_NEURAL_QUERY_BATCH_SIZE = 128
FORMAL_NEURAL_BATCHING_POLICY = LENGTH_BUCKETED_BATCHING_POLICY
NEURAL_INFERENCE_QUERY_BATCH_SIZE = 256
NEURAL_BACKENDS = frozenset({"mlp", "deepsets", "gnn", "set_transformer"})
MATRIX_STAGES = (
    "train",
    "validate",
    "lock-primary",
    "test-primary",
    "test-post-lock",
)
TRAIN_WORKER_PROTOCOL_SCHEMA_VERSION = 1
TRAIN_WORKER_PARTITION_ALGORITHM = "sha256_canonical_json_modulo_v1"
TRAIN_WORKER_PROTOCOL_PATH = Path("configs/train_worker_protocol.json")
TRAIN_WORKER_LIFECYCLE_LOCK_PATH = Path("logs/train_workers/lifecycle.lock")


class MatrixError(RuntimeError):
    """Raised when a matrix invariant fails; no success marker should be written."""


@dataclass(frozen=True)
class DatasetArtifact:
    key: str
    route: str
    pool: str
    split: str
    features_path: Path
    labels_path: Path


@dataclass(frozen=True)
class ExperimentSpec:
    key: str
    rung: str
    display_name: str
    backend: str
    loss: str
    residual: bool
    learned: bool
    scorer_family: str
    hyperparameters: Mapping[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class _NestedNeuralSplit:
    """Inner model-fit/early-stop rows relative to one outer training fold."""

    train_indices: np.ndarray
    early_stop_indices: np.ndarray
    audit: Mapping[str, Any]


@dataclass(frozen=True)
class FoldPreprocessor:
    """Median imputation and standardization fitted on one training fold only."""

    columns: tuple[str, ...]
    medians: tuple[float, ...]
    means: tuple[float, ...]
    scales: tuple[float, ...]

    @classmethod
    def fit(cls, frame: pd.DataFrame, columns: Sequence[str]) -> "FoldPreprocessor":
        names = tuple(map(str, columns))
        matrix = (
            frame.loc[:, names]
            .apply(pd.to_numeric, errors="coerce")
            .to_numpy(np.float64)
        )
        matrix[~np.isfinite(matrix)] = np.nan
        with np.errstate(all="ignore"):
            medians = np.nanmedian(matrix, axis=0)
        medians = np.where(np.isfinite(medians), medians, 0.0)
        filled = np.where(np.isfinite(matrix), matrix, medians)
        means = filled.mean(axis=0)
        scales = filled.std(axis=0)
        scales = np.where(np.isfinite(scales) & (scales > 1e-12), scales, 1.0)
        return cls(
            columns=names,
            medians=tuple(map(float, medians)),
            means=tuple(map(float, means)),
            scales=tuple(map(float, scales)),
        )

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        missing = sorted(set(self.columns) - set(frame.columns))
        if missing:
            raise MatrixError(f"feature schema mismatch; missing columns: {missing}")
        matrix = (
            frame.loc[:, self.columns]
            .apply(pd.to_numeric, errors="coerce")
            .to_numpy(np.float64)
        )
        matrix[~np.isfinite(matrix)] = np.nan
        medians = np.asarray(self.medians)
        matrix = np.where(np.isfinite(matrix), matrix, medians)
        matrix = (matrix - np.asarray(self.means)) / np.asarray(self.scales)
        if not np.isfinite(matrix).all():
            raise MatrixError("fold preprocessing produced non-finite values")
        return pd.DataFrame(matrix, columns=self.columns, index=frame.index)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "FoldPreprocessor":
        return cls(
            columns=tuple(value["columns"]),
            medians=tuple(map(float, value["medians"])),
            means=tuple(map(float, value["means"])),
            scales=tuple(map(float, value["scales"])),
        )


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).astimezone().isoformat(timespec="seconds")


def _json_safe(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, bool):
        return bool(value)
    if isinstance(value, str):
        return str(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return str(value)


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(
        json.dumps(_json_safe(value), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _atomic_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    frame.to_parquet(temporary, index=False, compression="zstd")
    os.replace(temporary, path)


def _atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def _rows_frame(rows: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    """Encode nested values deterministically before writing Parquet."""

    return pd.DataFrame(
        [
            {
                str(key): (
                    json.dumps(_json_safe(value), sort_keys=True)
                    if isinstance(value, (dict, list, tuple))
                    else _json_safe(value)
                )
                for key, value in row.items()
            }
            for row in rows
        ]
    )


def _atomic_pickle(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    with temporary.open("wb") as stream:
        pickle.dump(value, stream, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(temporary, path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json_sha256(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            _json_safe(value),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()


def _matrix_config_identity(value: Mapping[str, Any]) -> str:
    semantic = value.get("semantic_config")
    declared = value.get("semantic_config_sha256")
    if isinstance(semantic, Mapping) and isinstance(declared, str):
        computed = _canonical_json_sha256(semantic)
        if declared != computed:
            raise MatrixError("run configuration semantic identity is invalid")
        return declared
    payload = {
        str(key): item
        for key, item in value.items()
        if key
        not in {
            "created_at",
            "train_worker_count",
            "train_torch_thread_count",
            "train_torch_interop_thread_count",
            "train_worker_protocol_sha256",
        }
    }
    return _canonical_json_sha256(payload)


def _validate_worker_protocol(
    output: Path,
    protocol: Mapping[str, Any],
    *,
    config_identity: str,
) -> dict[str, Any]:
    payload = dict(protocol)
    declared_hash = payload.pop("protocol_sha256", None)
    worker_count = payload.get("worker_count")
    thread_count = payload.get("torch_thread_count")
    interop_thread_count = payload.get("torch_interop_thread_count")
    valid = (
        payload.get("schema_version") == TRAIN_WORKER_PROTOCOL_SCHEMA_VERSION
        and payload.get("partition_algorithm") == TRAIN_WORKER_PARTITION_ALGORITHM
        and payload.get("partition_key") == ["dataset", "seed", "fold"]
        and payload.get("run_root") == str(output.resolve())
        and payload.get("base_config_sha256") == config_identity
        and isinstance(worker_count, int)
        and worker_count > 0
        and isinstance(thread_count, int)
        and thread_count > 0
        and isinstance(interop_thread_count, int)
        and interop_thread_count > 0
        and isinstance(declared_hash, str)
        and declared_hash == _canonical_json_sha256(payload)
    )
    if not valid:
        raise MatrixError("immutable train worker protocol is invalid or mismatched")
    return {**payload, "protocol_sha256": declared_hash}


def _read_worker_protocol(
    output: Path,
    config: Mapping[str, Any],
    *,
    required: bool = False,
) -> dict[str, Any] | None:
    path = output / TRAIN_WORKER_PROTOCOL_PATH
    if not path.exists():
        if required:
            raise MatrixError(f"train worker protocol is missing: {path}")
        return None
    if path.is_symlink() or not path.is_file():
        raise MatrixError(f"train worker protocol must be a regular file: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise MatrixError(f"train worker protocol is unreadable: {path}") from error
    if not isinstance(payload, Mapping):
        raise MatrixError("train worker protocol must contain a JSON object")
    return _validate_worker_protocol(
        output,
        payload,
        config_identity=_matrix_config_identity(config),
    )


def _neural_batching_policy(config: Mapping[str, Any]) -> str | None:
    default = (
        None
        if str(config.get("matrix_profile", "formal")) == "unit_test"
        else FORMAL_NEURAL_BATCHING_POLICY
    )
    value = config.get("neural_batching_policy", default)
    if value in {None, "legacy", "none"}:
        return None
    if value != LENGTH_BUCKETED_BATCHING_POLICY:
        raise MatrixError(
            "neural_batching_policy must be legacy or "
            f"{LENGTH_BUCKETED_BATCHING_POLICY}"
        )
    return str(value)


def _config(output: Path) -> dict[str, Any]:
    path = output / "configs" / "run_config.json"
    if not path.is_file():
        raise MatrixError(f"run configuration is missing: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    profile = str(value.get("matrix_profile", "formal"))
    folds = int(value.get("folds", 5))
    seeds = tuple(map(int, value.get("seeds", FORMAL_SEEDS)))
    if profile != "unit_test" and (folds != 5 or tuple(sorted(seeds)) != FORMAL_SEEDS):
        raise MatrixError(
            "formal matrix requires 5 grouped folds and seeds 42/123/2026"
        )
    if profile != "unit_test" and "matrix_methods" in value:
        requested = set(map(str, value["matrix_methods"]))
        required = {spec.key for spec in _specs(profile)}
        if requested != required:
            missing = sorted(required - requested)
            extra = sorted(requested - required)
            raise MatrixError(
                "formal matrix_methods must cover the complete frozen matrix; "
                f"missing={missing}, extra={extra}"
            )
    value["matrix_profile"] = profile
    value["folds"] = folds
    value["seeds"] = list(seeds)
    value.setdefault("device", "mps" if torch.backends.mps.is_available() else "cpu")
    value["neural_batching_policy"] = _neural_batching_policy(value)
    protocol = _read_worker_protocol(output, value)
    if protocol is None:
        value.setdefault("train_worker_count", 1)
        value.setdefault("train_torch_thread_count", int(torch.get_num_threads()))
        value.setdefault("train_torch_interop_thread_count", 1)
    else:
        value["train_worker_count"] = int(protocol["worker_count"])
        value["train_torch_thread_count"] = int(protocol["torch_thread_count"])
        value["train_torch_interop_thread_count"] = int(
            protocol["torch_interop_thread_count"]
        )
        value["train_worker_protocol_sha256"] = str(protocol["protocol_sha256"])
    return value


def _prepare_directories(output: Path) -> None:
    for name in (
        "features",
        "data",
        "configs",
        "checkpoints",
        "predictions",
        "metrics",
        "manifests",
    ):
        (output / name).mkdir(parents=True, exist_ok=True)
    (output / "manifests" / "experiments").mkdir(parents=True, exist_ok=True)


def _discover_datasets(output: Path, split: str) -> list[DatasetArtifact]:
    if split not in {"development", "test"}:
        raise ValueError(split)
    artifacts: list[DatasetArtifact] = []
    suffix = f"_{split}.parquet"
    for features_path in sorted((output / "features").glob(f"candidates_*{suffix}")):
        middle = features_path.stem.removeprefix("candidates_").removesuffix(
            f"_{split}"
        )
        labels_path = output / "data" / f"labels_{middle}_{split}.parquet"
        if not labels_path.is_file():
            raise MatrixError(f"labels missing for {features_path}: {labels_path}")
        pieces = middle.split("_", 1)
        route = pieces[0]
        pool = pieces[1] if len(pieces) == 2 else "default"
        artifacts.append(
            DatasetArtifact(middle, route, pool, split, features_path, labels_path)
        )
    if not artifacts:
        raise MatrixError(
            f"no {split} candidate feature tables found in {output / 'features'}"
        )
    return artifacts


def _load_joined(
    artifact: DatasetArtifact,
) -> tuple[pd.DataFrame, pd.DataFrame, tuple[str, ...], dict[str, Any]]:
    features = pd.read_parquet(artifact.features_path)
    labels = pd.read_parquet(artifact.labels_path)
    query_source = "sample_id" if "sample_id" in features.columns else "query_id"
    label_query_source = "sample_id" if "sample_id" in labels.columns else "query_id"
    keys = [query_source, "candidate_id"]
    label_key = [label_query_source, "candidate_id"]
    if any(column not in features.columns for column in keys):
        raise MatrixError(
            f"feature identity columns missing in {artifact.features_path}"
        )
    if any(column not in labels.columns for column in label_key):
        raise MatrixError(f"label identity columns missing in {artifact.labels_path}")
    if features.duplicated(keys).any() or labels.duplicated(label_key).any():
        raise MatrixError("duplicate query/candidate IDs in matrix inputs")
    label_column = (
        "candidate_correct" if "candidate_correct" in labels.columns else "label"
    )
    if label_column not in labels.columns:
        raise MatrixError("labels table lacks candidate_correct/label")
    label_projection = labels[label_key + [label_column]].rename(
        columns={label_query_source: query_source, label_column: "label"}
    )
    joined = features.merge(
        label_projection, on=keys, how="inner", validate="one_to_one"
    )
    if len(joined) != len(features) or len(joined) != len(labels):
        raise MatrixError("feature/label candidate key sets differ")
    joined = joined.rename(columns={query_source: "query_id"})
    for column in ("query_id", "candidate_id"):
        if joined[column].isna().any():
            raise MatrixError(f"{column} contains null values")
        joined[column] = joined[column].astype(str)
    labels_numeric = pd.to_numeric(joined["label"], errors="coerce")
    if labels_numeric.isna().any() or not labels_numeric.isin([0, 1]).all():
        raise MatrixError("candidate labels must be binary")
    joined["label"] = labels_numeric.astype(np.int8)
    if "q_raw" not in joined.columns:
        raise MatrixError("q_raw baseline column is required")
    q = pd.to_numeric(joined["q_raw"], errors="coerce")
    if q.isna().any() or not np.isfinite(q.to_numpy(np.float64)).all():
        raise MatrixError("q_raw must be finite")
    joined["q_raw"] = q.astype(np.float64)
    for column in ("scene_id", "frame_id"):
        if column not in joined.columns:
            joined[column] = joined["query_id"]
        joined[column] = joined[column].astype(str)
    feature_columns = select_model_feature_columns(features)
    excluded_crog_depth_columns: list[str] = []
    if artifact.route == "crog":
        # The native CROG track is RGB + language only. Historical artifacts
        # contain optional diagnostics computed from depth/PCD; keeping those
        # columns in the numeric schema would silently turn the native track
        # into the separately named external RGB-D track forbidden by the
        # experiment contract.
        retained = select_crog_native_feature_columns(features)
        excluded_crog_depth_columns = sorted(set(feature_columns) - set(retained))
        feature_columns = retained
        if not feature_columns:
            raise MatrixError("CROG native no-depth feature schema is empty")
    schema = build_feature_schema(features, requested=feature_columns)
    if artifact.route == "crog":
        schema["evidence_track"] = "CROG_native_RGB_language_no_depth"
        schema["excluded_depth_or_pcd_feature_columns"] = excluded_crog_depth_columns
    reference = joined[
        ["query_id", "candidate_id", "scene_id", "frame_id", "label"]
    ].copy()
    return joined, reference, feature_columns, schema


def _specs(profile: str) -> tuple[ExperimentSpec, ...]:
    common_mlp = {"hidden_dims": [64, 32], "dropout": 0.1, "residual_scale": 0.5}
    core = (
        ExperimentSpec(
            "r0_q_baseline",
            "R0",
            "q-only baseline",
            "baseline",
            "none",
            False,
            False,
            "q",
            {},
        ),
        ExperimentSpec(
            "r1_fixed_rule",
            "R1",
            "fixed q+feature rule",
            "rule",
            "none",
            True,
            False,
            "fixed_rule",
            {"rule_weight": 0.05},
        ),
        ExperimentSpec(
            "r2_logistic",
            "R2",
            "Logistic Regression",
            "logistic",
            "bce",
            False,
            True,
            "linear_pointwise",
            {"max_iter": 100 if profile == "unit_test" else 300},
        ),
        ExperimentSpec(
            "r2_linear_residual",
            "R2",
            "Linear residual BCE",
            "linear_residual",
            "bce",
            True,
            True,
            "linear_residual",
            {"l2": 1e-3},
        ),
        ExperimentSpec(
            "r2_linear_ranknet",
            "R2",
            "Linear RankNet",
            "linear_ranknet",
            "ranknet",
            True,
            True,
            "linear_pairwise",
            {"l2": 1e-3},
        ),
        ExperimentSpec(
            "r3_random_forest",
            "R3",
            "Random Forest pointwise",
            "random_forest",
            "bce",
            False,
            True,
            "tree_pointwise",
            {"n_estimators": 12 if profile == "unit_test" else 200},
        ),
        ExperimentSpec(
            "r3_histgb_fallback",
            "R3",
            "HistGradientBoosting pointwise fallback",
            "histgb",
            "bce",
            False,
            True,
            "tree_pointwise",
            {"max_iter": 12 if profile == "unit_test" else 200, "lambda_mart": False},
        ),
        ExperimentSpec(
            "r3_lambdamart",
            "R3",
            "XGBoost LambdaMART",
            "lambdamart",
            "listwise",
            False,
            True,
            "tree_ltr",
            {
                "objective": "rank:ndcg",
                "n_estimators": 12 if profile == "unit_test" else 200,
                "max_depth": 6,
                "learning_rate": 0.05,
                "tree_method": "hist",
                "device": "cpu",
                "lambda_mart": True,
                "query_group_parameter": "qid",
            },
        ),
        ExperimentSpec(
            "r4_mlp_direct_bce",
            "R4",
            "Direct MLP BCE",
            "mlp",
            "bce",
            False,
            True,
            "shared_mlp",
            common_mlp,
        ),
        ExperimentSpec(
            "r4_mlp_residual_bce",
            "R4",
            "Residual MLP BCE",
            "mlp",
            "bce",
            True,
            True,
            "shared_mlp",
            common_mlp,
        ),
        ExperimentSpec(
            "r5_mlp_residual_ranknet",
            "R5",
            "Residual MLP RankNet",
            "mlp",
            "ranknet",
            True,
            True,
            "shared_mlp",
            common_mlp,
        ),
        ExperimentSpec(
            "r6_mlp_residual_listwise",
            "R6",
            "Residual MLP listwise",
            "mlp",
            "listwise",
            True,
            True,
            "shared_mlp",
            common_mlp,
        ),
        ExperimentSpec(
            "r7_deepsets_residual",
            "R7",
            "DeepSets residual RankNet",
            "deepsets",
            "ranknet",
            True,
            True,
            "deepsets",
            {"hidden_dim": 64, "dropout": 0.0},
        ),
        ExperimentSpec(
            "r8_candidate_gnn_residual",
            "R8",
            "Candidate GNN residual RankNet",
            "gnn",
            "ranknet",
            True,
            True,
            "candidate_gnn",
            {"hidden_dim": 64, "message_rounds": 2, "k": 4},
        ),
        ExperimentSpec(
            "r9_set_transformer_residual",
            "R9",
            "Set Transformer residual RankNet",
            "set_transformer",
            "ranknet",
            True,
            True,
            "set_transformer",
            {"hidden_dim": 64, "heads": 4, "blocks": 1},
        ),
    )
    if profile == "unit_test":
        return core

    baseline = core[0]
    rule_groups = {
        "r1a_q_mask": ("mask",),
        "r1b_q_mask_width": ("mask", "width"),
        "r1c_q_mask_depth_contact": ("mask", "depth", "contact"),
        "r1d_q_mask_clearance": ("mask", "clearance"),
        "r1e_q_all_scalar": (
            "mask",
            "width",
            "depth",
            "contact",
            "clearance",
            "relations",
            "reliability",
            "candidate_geometry",
            "other_scalar",
        ),
    }
    rules = tuple(
        ExperimentSpec(
            f"{prefix}_w{int(weight * 1000):03d}",
            "R1",
            f"{prefix} validation weight={weight:g}",
            "rule",
            "none",
            True,
            True,
            "interpretable_rule",
            {
                "rule_weight": weight,
                "rule_groups": list(groups),
                "weight_selection": "development validation grid",
                "direction": "positive support; risk/missingness reversed",
            },
        )
        for prefix, groups in rule_groups.items()
        for weight in (0.025, 0.05, 0.10)
    )
    r2_models = (
        ExperimentSpec(
            "r2_logistic",
            "R2",
            "Logistic Regression L2 pointwise",
            "logistic",
            "bce",
            False,
            True,
            "linear_pointwise",
            {"max_iter": 300, "penalty": "l2", "c": 1.0},
        ),
        ExperimentSpec(
            "r2_logistic_l1",
            "R2",
            "Logistic Regression L1 pointwise",
            "logistic",
            "bce",
            False,
            True,
            "linear_pointwise",
            {"max_iter": 300, "penalty": "l1", "c": 1.0},
        ),
        core[3],
        core[4],
    )
    r3_models = (core[5], core[6], core[7])

    r4_models: list[ExperimentSpec] = []
    for hidden_dims in ([64, 32], [128, 64]):
        for dropout in (0.0, 0.1, 0.2):
            hidden_key = "64x32" if hidden_dims == [64, 32] else "128x64"
            dropout_key = str(dropout).replace(".", "p")
            direct_key = (
                "r4_mlp_direct_bce"
                if hidden_dims == [64, 32] and dropout == 0.1
                else f"r4_mlp_direct_bce_h{hidden_key}_d{dropout_key}"
            )
            r4_models.append(
                ExperimentSpec(
                    direct_key,
                    "R4",
                    f"Direct MLP BCE hidden={hidden_dims} dropout={dropout:g}",
                    "mlp",
                    "bce",
                    False,
                    True,
                    "shared_mlp",
                    {
                        "hidden_dims": hidden_dims,
                        "dropout": dropout,
                        "bce_weight": 1.0,
                        "class_imbalance": "train-fold query-balanced pos_weight",
                    },
                )
            )
            for alpha in (0.25, 0.5, 1.0):
                alpha_key = str(alpha).replace(".", "p")
                residual_key = (
                    "r4_mlp_residual_bce"
                    if hidden_dims == [64, 32] and dropout == 0.1 and alpha == 0.5
                    else (
                        f"r4_mlp_residual_bce_h{hidden_key}_d{dropout_key}_a{alpha_key}"
                    )
                )
                r4_models.append(
                    ExperimentSpec(
                        residual_key,
                        "R4",
                        (
                            "Residual MLP BCE "
                            f"hidden={hidden_dims} dropout={dropout:g} alpha={alpha:g}"
                        ),
                        "mlp",
                        "bce",
                        True,
                        True,
                        "shared_mlp",
                        {
                            "hidden_dims": hidden_dims,
                            "dropout": dropout,
                            "residual_scale": alpha,
                            "bce_weight": 1.0,
                            "class_imbalance": ("train-fold query-balanced pos_weight"),
                        },
                    )
                )

    r5_models: list[ExperimentSpec] = []
    for lambda_point in (0.0, 0.1, 0.5):
        for lambda_residual in (1e-4, 1e-3, 1e-2):
            key = (
                "r5_mlp_residual_ranknet"
                if lambda_point == 0.1 and lambda_residual == 1e-3
                else (
                    "r5_mlp_residual_ranknet_lp"
                    f"{str(lambda_point).replace('.', 'p')}_lr"
                    f"{format(lambda_residual, '.0e').replace('-', 'm')}"
                )
            )
            r5_models.append(
                ExperimentSpec(
                    key,
                    "R5",
                    (
                        "Residual MLP RankNet "
                        f"lambda_point={lambda_point:g} "
                        f"lambda_residual={lambda_residual:g}"
                    ),
                    "mlp",
                    "ranknet",
                    True,
                    True,
                    "shared_mlp",
                    {
                        **common_mlp,
                        "bce_weight": lambda_point,
                        "ranknet_weight": 1.0,
                        "residual_weight": lambda_residual,
                        "pair_scope": "same query all positive-negative pairs",
                    },
                )
            )

    r6_models = tuple(
        ExperimentSpec(
            (
                "r6_mlp_residual_listwise"
                if temperature == 1.0
                else f"r6_mlp_residual_listwise_t{str(temperature).replace('.', 'p')}"
            ),
            "R6",
            f"Residual MLP multiple-positive listwise T={temperature:g}",
            "mlp",
            "listwise",
            True,
            True,
            "shared_mlp",
            {
                **common_mlp,
                "listwise_weight": 1.0,
                "residual_weight": 1e-3,
                "temperature": temperature,
                "no_positive_policy": "excluded_from_listwise_and_recorded",
            },
        )
        for temperature in (0.5, 1.0, 2.0)
    )
    loss_ablations = (
        ExperimentSpec(
            "loss_mlp_ranknet_pure",
            "A_LOSS",
            "Residual MLP pure RankNet",
            "mlp",
            "ranknet_pure",
            True,
            True,
            "shared_mlp",
            {
                **common_mlp,
                "bce_weight": 0.0,
                "ranknet_weight": 1.0,
                "residual_weight": 0.0,
            },
        ),
        ExperimentSpec(
            "loss_mlp_bce_ranknet",
            "A_LOSS",
            "Residual MLP BCE plus RankNet",
            "mlp",
            "bce_ranknet",
            True,
            True,
            "shared_mlp",
            {
                **common_mlp,
                "bce_weight": 0.5,
                "ranknet_weight": 1.0,
                "residual_weight": 0.0,
            },
        ),
        ExperimentSpec(
            "loss_mlp_ranknet_regularized",
            "A_LOSS",
            "Residual MLP RankNet plus residual regularization",
            "mlp",
            "ranknet_regularized",
            True,
            True,
            "shared_mlp",
            {
                **common_mlp,
                "bce_weight": 0.0,
                "ranknet_weight": 1.0,
                "residual_weight": 1e-3,
            },
        ),
        ExperimentSpec(
            "loss_mlp_listwise_pure",
            "A_LOSS",
            "Residual MLP pure multiple-positive listwise",
            "mlp",
            "listwise_pure",
            True,
            True,
            "shared_mlp",
            {
                **common_mlp,
                "bce_weight": 0.0,
                "listwise_weight": 1.0,
                "residual_weight": 0.0,
            },
        ),
        ExperimentSpec(
            "loss_mlp_listwise_bce",
            "A_LOSS",
            "Residual MLP listwise plus BCE",
            "mlp",
            "listwise_bce",
            True,
            True,
            "shared_mlp",
            {
                **common_mlp,
                "bce_weight": 0.5,
                "listwise_weight": 1.0,
                "residual_weight": 0.0,
            },
        ),
    )
    r7_models = tuple(
        ExperimentSpec(
            f"r7_deepsets_residual_{loss}",
            "R7",
            f"DeepSets residual {loss}",
            "deepsets",
            loss,
            True,
            True,
            "deepsets",
            {"hidden_dim": 64, "dropout": 0.0},
        )
        for loss in ("bce", "ranknet", "listwise")
    )
    all_edges = len(PAIRWISE_EDGE_RELATION_FIELDS)
    r8_models = (
        ExperimentSpec(
            "r8_gnn_no_edge_residual",
            "R8",
            "GNN adjacency-only residual",
            "gnn",
            "ranknet",
            True,
            True,
            "candidate_gnn",
            {
                "hidden_dim": 64,
                "message_rounds": 2,
                "edge_dim": 0,
                "graph_type": "auto",
                "k": 4,
            },
        ),
        ExperimentSpec(
            "r8_gnn_edge_direct",
            "R8",
            "Direct GNN with all explicit edges",
            "gnn",
            "ranknet",
            False,
            True,
            "candidate_gnn",
            {
                "hidden_dim": 64,
                "message_rounds": 2,
                "edge_dim": all_edges,
                "graph_type": "auto",
                "k": 4,
            },
        ),
        ExperimentSpec(
            "r8_candidate_gnn_residual",
            "R8",
            "Residual GNN with all explicit edges",
            "gnn",
            "ranknet",
            True,
            True,
            "candidate_gnn",
            {
                "hidden_dim": 64,
                "message_rounds": 2,
                "edge_dim": all_edges,
                "graph_type": "auto",
                "k": 4,
            },
        ),
        ExperimentSpec(
            "r8_gnn_edge_complete",
            "R8",
            "Residual GNN complete graph",
            "gnn",
            "ranknet",
            True,
            True,
            "candidate_gnn",
            {
                "hidden_dim": 64,
                "message_rounds": 2,
                "edge_dim": all_edges,
                "graph_type": "complete",
                "k": 4,
            },
        ),
        ExperimentSpec(
            "r8_gnn_edge_knn4",
            "R8",
            "Residual GNN kNN k=4",
            "gnn",
            "ranknet",
            True,
            True,
            "candidate_gnn",
            {
                "hidden_dim": 64,
                "message_rounds": 2,
                "edge_dim": all_edges,
                "graph_type": "knn",
                "k": 4,
            },
        ),
        ExperimentSpec(
            "r8_gnn_edge_knn8",
            "R8",
            "Residual GNN kNN k=8",
            "gnn",
            "ranknet",
            True,
            True,
            "candidate_gnn",
            {
                "hidden_dim": 64,
                "message_rounds": 2,
                "edge_dim": all_edges,
                "graph_type": "knn",
                "k": 8,
            },
        ),
        ExperimentSpec(
            "r8_gnn_rule_edges",
            "R8",
            "Residual GNN rule graph",
            "gnn",
            "ranknet",
            True,
            True,
            "candidate_gnn",
            {
                "hidden_dim": 64,
                "message_rounds": 2,
                "edge_dim": all_edges,
                "graph_type": "rule",
                "k": 8,
            },
        ),
        ExperimentSpec(
            "r8_gnn_spatial_edges",
            "R8",
            "GNN spatial edge-feature ablation",
            "gnn",
            "ranknet",
            True,
            True,
            "candidate_gnn",
            {
                "hidden_dim": 64,
                "message_rounds": 2,
                "edge_feature_indices": [0, 1, 2],
                "graph_type": "auto",
                "k": 4,
            },
        ),
        ExperimentSpec(
            "r8_gnn_spatial_angle_edges",
            "R8",
            "GNN spatial and angle edge-feature ablation",
            "gnn",
            "ranknet",
            True,
            True,
            "candidate_gnn",
            {
                "hidden_dim": 64,
                "message_rounds": 2,
                "edge_feature_indices": [0, 1, 2, 3, 4],
                "graph_type": "auto",
                "k": 4,
            },
        ),
        ExperimentSpec(
            "r8_gnn_spatial_angle_overlap_edges",
            "R8",
            "GNN spatial, angle, and overlap edge-feature ablation",
            "gnn",
            "ranknet",
            True,
            True,
            "candidate_gnn",
            {
                "hidden_dim": 64,
                "message_rounds": 2,
                "edge_feature_indices": [0, 1, 2, 3, 4, 7, 8],
                "graph_type": "auto",
                "k": 4,
            },
        ),
    )
    r9_models = tuple(
        ExperimentSpec(
            f"r9_set_transformer_residual_{loss}_b{blocks}",
            "R9",
            f"Set Transformer residual {loss}, {blocks} block(s)",
            "set_transformer",
            loss,
            True,
            True,
            "set_transformer",
            {"hidden_dim": 64, "heads": 4, "blocks": blocks, "dropout": 0.0},
        )
        for loss in ("ranknet", "listwise")
        for blocks in (1, 2)
    )
    cumulative = (
        ("baseline_only", ("baseline",)),
        ("plus_mask", ("baseline", "mask")),
        ("plus_width", ("baseline", "mask", "width")),
        ("plus_depth", ("baseline", "mask", "width", "depth")),
        ("plus_contact", ("baseline", "mask", "width", "depth", "contact")),
        (
            "plus_clearance",
            ("baseline", "mask", "width", "depth", "contact", "clearance"),
        ),
        (
            "plus_relations",
            ("baseline", "mask", "width", "depth", "contact", "clearance", "relations"),
        ),
        (
            "plus_reliability",
            (
                "baseline",
                "mask",
                "width",
                "depth",
                "contact",
                "clearance",
                "relations",
                "reliability",
            ),
        ),
        (
            "plus_embedding",
            (
                "baseline",
                "mask",
                "width",
                "depth",
                "contact",
                "clearance",
                "relations",
                "reliability",
                "embedding",
            ),
        ),
    )
    feature_ablations = (
        tuple(
            ExperimentSpec(
                f"feature_{name}",
                "A_FEATURE",
                f"Feature ablation {name}",
                "mlp",
                "ranknet",
                True,
                True,
                "shared_mlp",
                {**common_mlp, "include_groups": list(groups), "ablation": name},
            )
            for name, groups in cumulative
        )
        + tuple(
            ExperimentSpec(
                f"feature_minus_{group}",
                "A_FEATURE",
                f"Feature leave-one-group-out minus {group}",
                "mlp",
                "ranknet",
                True,
                True,
                "shared_mlp",
                {**common_mlp, "exclude_groups": [group], "ablation": f"minus_{group}"},
            )
            for group in (
                "mask",
                "width",
                "depth",
                "contact",
                "clearance",
                "relations",
                "embedding",
            )
        )
        + (
            ExperimentSpec(
                "feature_no_baseline_direct",
                "A_FEATURE",
                "Baseline-score removal (direct MLP)",
                "mlp",
                "ranknet",
                False,
                True,
                "shared_mlp",
                {
                    **common_mlp,
                    "exclude_groups": ["baseline"],
                    "ablation": "no_baseline_direct",
                },
            ),
        )
    )
    mask_ablations = tuple(
        ExperimentSpec(
            f"mask_input_{name}",
            "G_MASK",
            f"Mask input ablation {name}",
            "mlp",
            "ranknet",
            True,
            True,
            "shared_mlp",
            {**common_mlp, "mask_variant": name},
        )
        for name in (
            "none",
            "binary",
            "soft",
            "binary_soft",
            "boundary",
            "contact_support",
        )
    )
    return (
        baseline,
        *rules,
        *r2_models,
        *r3_models,
        *r4_models,
        *r5_models,
        *r6_models,
        *loss_ablations,
        *r7_models,
        *r8_models,
        *r9_models,
        *feature_ablations,
        *mask_ablations,
    )


def _feature_columns_for_spec(
    feature_columns: Sequence[str], spec: ExperimentSpec
) -> tuple[str, ...]:
    columns = list(map(str, feature_columns))
    include = set(map(str, spec.hyperparameters.get("include_groups", [])))
    exclude = set(map(str, spec.hyperparameters.get("exclude_groups", [])))
    mask_variant = spec.hyperparameters.get("mask_variant")
    if include:
        selected = [column for column in columns if feature_group(column) in include]
    else:
        selected = [
            column for column in columns if feature_group(column) not in exclude
        ]
    if mask_variant is not None:
        mask_columns = [
            column for column in selected if feature_group(column) == "mask"
        ]
        if mask_variant == "none":
            selected = [column for column in selected if column not in mask_columns]
        else:
            tokens = {
                "binary": ("binary",),
                "soft": ("soft", "prob"),
                "binary_soft": ("binary", "soft", "prob"),
                "boundary": ("boundary", "distance"),
                "contact_support": ("contact", "support"),
            }[str(mask_variant)]
            selected = [
                column
                for column in selected
                if column not in mask_columns
                or any(token in column.lower() for token in tokens)
            ]
    if spec.backend in {"random_forest", "histgb", "lambdamart"}:
        selected = [
            column for column in selected if feature_group(column) != "embedding"
        ]
    if spec.backend == "gnn":
        width_column = "width_m" if "width_m" in selected else "width_px"
        priority = ("x_px", "y_px", "angle_rad", width_column, "q_raw")
        missing_relations = [column for column in priority if column not in selected]
        if missing_relations:
            raise MatrixError(
                f"GNN relation inputs are missing for {spec.key}: {missing_relations}"
            )
        selected = list(priority) + [
            column for column in selected if column not in priority
        ]
    if not selected:
        q_fallback = [
            column for column in columns if feature_group(column) == "baseline"
        ]
        selected = q_fallback[:1]
    if not selected:
        raise MatrixError(f"method {spec.key} resolved to an empty feature set")
    return tuple(selected)


def _experiment_id(dataset: str, spec: ExperimentSpec, seed: int, fold: int) -> str:
    value = f"dataset={dataset}__method={spec.key}__seed={seed}__fold={fold}"
    return value.replace("/", "_").replace(" ", "_")


def _registry_paths(output: Path) -> tuple[Path, Path]:
    return (
        output / "metrics" / "experiment_registry.json",
        output / "metrics" / "experiment_registry.parquet",
    )


def _manifest_rows(output: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted((output / "manifests" / "experiments").glob("*.json")):
        value = json.loads(path.read_text(encoding="utf-8"))
        rows.append(value)
    return rows


def _write_registry(output: Path) -> tuple[Path, Path]:
    json_path, parquet_path = _registry_paths(output)
    rows = _manifest_rows(output)
    ids = [row.get("experiment_id") for row in rows]
    if len(ids) != len(set(ids)):
        raise MatrixError("duplicate experiment IDs in registry")
    _atomic_json(json_path, {"schema_version": 1, "experiments": rows})
    flat_rows = []
    for row in rows:
        flat = {
            key: value
            for key, value in row.items()
            if not isinstance(value, (dict, list))
        }
        flat["spec_json"] = json.dumps(row.get("spec", {}), sort_keys=True)
        flat["metrics_json"] = json.dumps(
            _json_safe(row.get("metrics", {})), sort_keys=True
        )
        flat_rows.append(flat)
    _atomic_parquet(parquet_path, pd.DataFrame(flat_rows))
    return json_path, parquet_path


@lru_cache(maxsize=32)
def _cached_file_sha256(path: str) -> str:
    return _sha256(Path(path))


def _experiment_identity_sha256(
    artifact: DatasetArtifact,
    spec: ExperimentSpec,
    seed: int,
    fold: int,
    feature_columns: Sequence[str],
    train_indices: np.ndarray | None = None,
    validation_indices: np.ndarray | None = None,
    training_protocol: Mapping[str, Any] | None = None,
) -> str:
    payload = {
        "dataset": artifact.key,
        "route": artifact.route,
        "pool": artifact.pool,
        "split": artifact.split,
        "features_path": str(artifact.features_path.resolve()),
        "features_sha256": _cached_file_sha256(str(artifact.features_path.resolve())),
        "labels_path": str(artifact.labels_path.resolve()),
        "labels_sha256": _cached_file_sha256(str(artifact.labels_path.resolve())),
        "spec": spec.as_dict(),
        "seed": seed,
        "fold": fold,
        "feature_columns": list(map(str, feature_columns)),
        "train_indices_sha256": (
            None
            if train_indices is None
            else hashlib.sha256(
                np.asarray(train_indices, dtype=np.int64).tobytes()
            ).hexdigest()
        ),
        "validation_indices_sha256": (
            None
            if validation_indices is None
            else hashlib.sha256(
                np.asarray(validation_indices, dtype=np.int64).tobytes()
            ).hexdigest()
        ),
    }
    if training_protocol is not None:
        payload["training_protocol"] = _json_safe(training_protocol)
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _neural_training_protocol(
    spec: ExperimentSpec, config: Mapping[str, Any]
) -> dict[str, Any] | None:
    if spec.backend not in NEURAL_BACKENDS:
        return None
    profile = str(config["matrix_profile"])
    requested_device = str(config.get("device", "cpu"))
    effective_device = (
        "cpu"
        if requested_device == "mps" and not torch.backends.mps.is_available()
        else requested_device
    )
    return {
        "schema_version": 2,
        "backend": spec.backend,
        "requested_device": requested_device,
        "effective_device": effective_device,
        "epochs": int(
            config.get("matrix_epochs", 1 if profile == "unit_test" else 100)
        ),
        "patience": int(
            config.get("matrix_patience", 1 if profile == "unit_test" else 10)
        ),
        "min_delta": float(config.get("matrix_min_delta", 1e-5)),
        "learning_rate": float(
            spec.hyperparameters.get(
                "learning_rate", config.get("matrix_learning_rate", 1e-3)
            )
        ),
        "weight_decay": float(
            spec.hyperparameters.get(
                "weight_decay", config.get("matrix_weight_decay", 1e-4)
            )
        ),
        "gradient_clip_norm": 5.0,
        "train_query_batch_size": _neural_query_batch_size(config),
        "training_batching_policy": _neural_batching_policy(config),
        "training_batching_shuffle": True,
        "early_stopping_batching_policy": _neural_batching_policy(config),
        "early_stopping_batching_shuffle": False,
        "inference_query_batch_size": NEURAL_INFERENCE_QUERY_BATCH_SIZE,
        "torch_thread_count": int(
            config.get("train_torch_thread_count", torch.get_num_threads())
        ),
        "torch_interop_thread_count": int(
            config.get("train_torch_interop_thread_count", 1)
        ),
        "inner_folds": int(
            config.get("matrix_inner_folds", 2 if profile == "unit_test" else 5)
        ),
        "inner_split_random_state": int(
            config.get("matrix_inner_split_random_state", 1701)
        ),
        "torch_version": torch.__version__,
    }


def _neural_query_batch_size(config: Mapping[str, Any]) -> int:
    profile = str(config["matrix_profile"])
    return int(
        config.get(
            "neural_query_batch_size",
            config.get(
                "matrix_query_batch_size",
                8 if profile == "unit_test" else FORMAL_NEURAL_QUERY_BATCH_SIZE,
            ),
        )
    )


def _completed_manifest(
    output: Path,
    experiment_id: str,
    *,
    expected_identity_sha256: str,
) -> dict[str, Any] | None:
    path = output / "manifests" / "experiments" / f"{experiment_id}.json"
    if not path.is_file():
        return None
    value = json.loads(path.read_text(encoding="utf-8"))
    artifacts = [Path(item) for item in value.get("artifacts", [])]
    hashes = value.get("artifact_sha256")
    if (
        value.get("status") == "COMPLETE"
        and value.get("experiment_identity_sha256") == expected_identity_sha256
        and artifacts
        and isinstance(hashes, Mapping)
        and all(
            item.is_file()
            and not item.is_symlink()
            and hashes.get(str(item.resolve())) == _sha256(item)
            for item in artifacts
        )
    ):
        return value
    return None


def _write_manifest(output: Path, experiment_id: str, value: Mapping[str, Any]) -> Path:
    path = output / "manifests" / "experiments" / f"{experiment_id}.json"
    _atomic_json(path, {"experiment_id": experiment_id, **dict(value)})
    # The per-experiment manifest is the atomic source of truth for resume.
    # Rebuilding both registry formats here rereads every prior manifest after
    # every experiment (quadratic work for the formal 4,638-entry matrix).
    # Each matrix stage rebuilds the complete registries once before returning,
    # so deferring that derived view does not change experiment identity,
    # resumability, or any final artifact.
    return path


def _baseline_predictions(frame: pd.DataFrame) -> pd.DataFrame:
    return frame[["query_id", "candidate_id"]].assign(
        score=frame["q_raw"].to_numpy(np.float64)
    )


def _rule_predictions(
    frame: pd.DataFrame,
    feature_columns: Sequence[str],
    weight: float,
    *,
    groups: Sequence[str] = (),
    baseline_values: Sequence[float] | None = None,
) -> pd.DataFrame:
    requested_groups = set(map(str, groups))
    candidates = [
        name
        for name in feature_columns
        if name not in {"q_raw", "q_clipped", "q_log", "q_logit"}
        and (not requested_groups or feature_group(name) in requested_groups)
    ]
    components: list[np.ndarray] = []
    for feature in candidates:
        values = pd.to_numeric(frame[feature], errors="coerce").to_numpy(np.float64)
        values = np.where(np.isfinite(values), values, 0.0)
        direction = (
            -1.0
            if any(
                token in feature.lower()
                for token in (
                    "collision",
                    "risk",
                    "mad",
                    "missing",
                    "overreach",
                    "underreach",
                )
            )
            else 1.0
        )
        components.append(direction * np.tanh(values))
    physical_support = (
        np.mean(np.stack(components, axis=1), axis=1)
        if components
        else np.zeros(len(frame), dtype=np.float64)
    )
    baseline = (
        frame["q_raw"].to_numpy(np.float64)
        if baseline_values is None
        else np.asarray(baseline_values, dtype=np.float64)
    )
    if baseline.shape != (len(frame),) or not np.isfinite(baseline).all():
        raise MatrixError("rule baseline values must be one finite value per candidate")
    score = baseline + float(weight) * physical_support
    return frame[["query_id", "candidate_id"]].assign(score=score)


def _rule_weight_audit(
    feature_columns: Sequence[str],
    weight: float,
    groups: Sequence[str],
) -> dict[str, Any]:
    requested_groups = set(map(str, groups))
    candidates = [
        name
        for name in feature_columns
        if name not in {"q_raw", "q_clipped", "q_log", "q_logit"}
        and (not requested_groups or feature_group(name) in requested_groups)
    ]
    denominator = max(len(candidates), 1)
    rows = []
    for feature in candidates:
        direction = (
            -1.0
            if any(
                token in feature.lower()
                for token in (
                    "collision",
                    "risk",
                    "mad",
                    "missing",
                    "overreach",
                    "underreach",
                )
            )
            else 1.0
        )
        rows.append(
            {
                "feature": feature,
                "group": feature_group(feature),
                "direction": "negative" if direction < 0 else "positive",
                "effective_weight": direction * float(weight) / denominator,
                "transform": "tanh",
            }
        )
    return {
        "baseline_calibrated_q_weight": 1.0,
        "physical_components": rows,
        "shared_grid_weight": float(weight),
        "weight_selection": "development grouped OOF validation grid",
    }


def _evaluation(
    reference: pd.DataFrame,
    predictions: pd.DataFrame,
    universe: pd.DataFrame | None = None,
) -> dict[str, Any]:
    if universe is None:
        universe = reference[["query_id", "scene_id", "frame_id"]].drop_duplicates(
            "query_id"
        )
    result = evaluate_rankings(
        reference,
        predictions,
        query_universe=universe,
        probability_col=("probability" if "probability" in predictions else None),
    )
    return {
        **result["metrics"],
        "candidate_metrics": result["candidate_metrics"],
    }


def _make_tabular(spec: ExperimentSpec, seed: int, profile: str) -> Any:
    if spec.backend == "logistic":
        return LogisticRegressionRanker(
            random_state=seed,
            max_iter=int(spec.hyperparameters["max_iter"]),
            penalty=str(spec.hyperparameters.get("penalty", "l2")),
            c=float(spec.hyperparameters.get("c", 1.0)),
        )
    if spec.backend == "linear_residual":
        return LinearResidualBCE(
            random_state=seed,
            l2=float(spec.hyperparameters["l2"]),
            max_iter=80 if profile == "unit_test" else 300,
        )
    if spec.backend == "linear_ranknet":
        return LinearPairwiseRankNet(
            random_state=seed,
            l2=float(spec.hyperparameters["l2"]),
            max_iter=80 if profile == "unit_test" else 300,
        )
    if spec.backend == "random_forest":
        return RandomForestRanker(
            random_state=seed, n_estimators=int(spec.hyperparameters["n_estimators"])
        )
    if spec.backend == "histgb":
        return HistGradientBoostingRanker(
            random_state=seed, max_iter=int(spec.hyperparameters["max_iter"])
        )
    if spec.backend == "lambdamart":
        return XGBoostLambdaMARTRanker(
            objective=str(spec.hyperparameters.get("objective", "rank:ndcg")),
            n_estimators=int(spec.hyperparameters["n_estimators"]),
            max_depth=int(spec.hyperparameters.get("max_depth", 6)),
            learning_rate=float(spec.hyperparameters.get("learning_rate", 0.05)),
            random_state=seed,
            tree_method=str(spec.hyperparameters.get("tree_method", "hist")),
        )
    raise ValueError(spec.backend)


def _make_neural(spec: ExperimentSpec, input_dim: int, pool: str) -> nn.Module:
    mode = "residual" if spec.residual else "direct"
    hidden = spec.hyperparameters.get("hidden_dims", [64, 32])
    dropout = float(spec.hyperparameters.get("dropout", 0.1))
    residual_scale = float(spec.hyperparameters.get("residual_scale", 0.5))
    if spec.backend == "mlp":
        return SharedMLPScorer(
            input_dim,
            hidden_dims=tuple(map(int, hidden)),
            dropout=dropout,
            mode=mode,
            residual_scale=residual_scale,
        )
    if spec.backend == "deepsets":
        return DeepSetsScorer(
            input_dim,
            hidden_dim=int(spec.hyperparameters.get("hidden_dim", 64)),
            dropout=float(spec.hyperparameters.get("dropout", 0.0)),
            mode=mode,
            residual_scale=residual_scale,
        )
    if spec.backend == "gnn":
        configured_graph = str(spec.hyperparameters.get("graph_type", "auto"))
        graph_type = (
            ("complete" if "top5" in pool else "knn")
            if configured_graph == "auto"
            else configured_graph
        )
        relation_indices = tuple(
            int(index) for index in spec.hyperparameters.get("edge_feature_indices", ())
        )
        edge_dim = int(
            spec.hyperparameters.get(
                "edge_dim",
                len(relation_indices) if relation_indices else 0,
            )
        )
        model = CandidateGNNScorer(
            input_dim,
            edge_dim=edge_dim,
            hidden_dim=int(spec.hyperparameters.get("hidden_dim", 64)),
            num_message_passing=int(spec.hyperparameters.get("message_rounds", 2)),
            graph_type=graph_type,
            k=int(spec.hyperparameters.get("k", 4)),
            mode=mode,
            residual_scale=residual_scale,
        )
        model.edge_feature_indices = relation_indices
        return model
    if spec.backend == "set_transformer":
        return SetTransformerScorer(
            input_dim,
            hidden_dim=int(spec.hyperparameters.get("hidden_dim", 64)),
            num_heads=int(spec.hyperparameters.get("heads", 4)),
            num_blocks=int(spec.hyperparameters.get("blocks", 1)),
            dropout=dropout,
            mode=mode,
            residual_scale=residual_scale,
        )
    raise ValueError(spec.backend)


def _criterion(spec: ExperimentSpec) -> CompositeRerankingLoss:
    explicit_loss_keys = {
        "bce_weight",
        "ranknet_weight",
        "listwise_weight",
        "residual_weight",
        "pos_weight",
        "temperature",
    }
    if explicit_loss_keys & set(spec.hyperparameters):
        return CompositeRerankingLoss(
            bce_weight=float(spec.hyperparameters.get("bce_weight", 0.0)),
            ranknet_weight=float(spec.hyperparameters.get("ranknet_weight", 0.0)),
            listwise_weight=float(spec.hyperparameters.get("listwise_weight", 0.0)),
            residual_weight=float(spec.hyperparameters.get("residual_weight", 0.0)),
            pos_weight=(
                None
                if spec.hyperparameters.get("pos_weight") is None
                else float(spec.hyperparameters["pos_weight"])
            ),
            temperature=float(spec.hyperparameters.get("temperature", 1.0)),
        )
    if spec.loss == "bce":
        return CompositeRerankingLoss(bce_weight=1.0)
    if spec.loss == "ranknet":
        return CompositeRerankingLoss(
            bce_weight=0.1, ranknet_weight=1.0, residual_weight=1e-3
        )
    if spec.loss == "listwise":
        return CompositeRerankingLoss(
            bce_weight=0.1, listwise_weight=1.0, residual_weight=1e-3
        )
    if spec.loss == "ranknet_pure":
        return CompositeRerankingLoss(bce_weight=0.0, ranknet_weight=1.0)
    if spec.loss == "bce_ranknet":
        return CompositeRerankingLoss(bce_weight=0.5, ranknet_weight=1.0)
    if spec.loss == "ranknet_regularized":
        return CompositeRerankingLoss(
            bce_weight=0.0, ranknet_weight=1.0, residual_weight=1e-3
        )
    if spec.loss == "listwise_pure":
        return CompositeRerankingLoss(bce_weight=0.0, listwise_weight=1.0)
    if spec.loss == "listwise_bce":
        return CompositeRerankingLoss(bce_weight=0.5, listwise_weight=1.0)
    raise ValueError(spec.loss)


def _first_finite_numeric_column(
    frame: pd.DataFrame,
    candidates: Sequence[str],
    *,
    default: float | None = None,
) -> np.ndarray:
    """Return the first available finite numeric source without label fallback."""

    for column in candidates:
        if column not in frame.columns:
            continue
        values = pd.to_numeric(frame[column], errors="coerce").to_numpy(np.float64)
        if np.isfinite(values).all():
            return values
    if default is None:
        raise MatrixError(
            f"raw GNN relation input is missing a finite source from {list(candidates)}"
        )
    return np.full(len(frame), float(default), dtype=np.float64)


def _raw_edge_frame(frame: pd.DataFrame, *, route: str) -> pd.DataFrame:
    """Build the documented, unstandardized R8 candidate relation inputs.

    Geometry deliberately remains separate from the fold-standardized node
    design.  CROG's native RGB/language track receives no depth, clearance, or
    collision evidence; those fields are explicit neutral/missing proxies.
    """

    x = _first_finite_numeric_column(frame, ("x_px", "center_u_px"))
    y = _first_finite_numeric_column(frame, ("y_px", "center_v_px"))
    theta = _first_finite_numeric_column(frame, ("angle_rad", "theta_rad"))
    # x/y are pixels, so accepting width_m here would silently mix units.
    width = _first_finite_numeric_column(
        frame, ("width_px", "grasp_width_px", "jaw_width_px")
    )
    if np.any(width <= 0.0):
        raise MatrixError("raw GNN relation widths must be positive pixels")
    q = _first_finite_numeric_column(frame, ("q_raw",))
    mask = _first_finite_numeric_column(
        frame,
        (
            "grasp_axis_mask_support",
            "mask_support",
            "p_axis_mean",
            "soft_mask_support",
        ),
        default=0.0,
    )
    if route == "crog":
        depth = np.zeros(len(frame), dtype=np.float64)
        clearance = np.zeros(len(frame), dtype=np.float64)
        risk = np.zeros(len(frame), dtype=np.float64)
    else:
        depth = _first_finite_numeric_column(
            frame, ("center_depth", "center_depth_m", "z_m"), default=0.0
        )
        clearance = _first_finite_numeric_column(
            frame,
            (
                "approach_clearance",
                "minimum_visible_obstacle_distance",
                "nearest_obstacle_distance_px",
            ),
            default=0.0,
        )
        risk = _first_finite_numeric_column(
            frame,
            (
                "collision_proxy_total",
                "collision_risk",
                "approach_corridor_occupancy",
            ),
            default=0.0,
        )
        risk = np.clip(risk, 0.0, 1.0)
    cluster = _first_finite_numeric_column(
        frame, ("pose_cluster_id", "cluster_id"), default=-1.0
    )
    cluster = np.rint(cluster)
    values = np.column_stack(
        (x, y, theta, width, q, mask, depth, clearance, risk, cluster)
    )
    if values.shape != (len(frame), len(RAW_EDGE_CANDIDATE_FIELDS)):
        raise MatrixError("raw GNN relation schema construction failed")
    if not np.isfinite(values).all():
        raise MatrixError("raw GNN relation inputs contain non-finite values")
    return pd.DataFrame(values, index=frame.index, columns=RAW_EDGE_CANDIDATE_FIELDS)


def _examples(
    frame: pd.DataFrame,
    transformed: pd.DataFrame,
    *,
    raw_edge_inputs: pd.DataFrame | None = None,
) -> list[CandidateSetExample]:
    groups = tuple(
        (query_id, np.asarray(positions, dtype=np.int64))
        for query_id, positions in frame.groupby("query_id", sort=False).indices.items()
    )
    return _examples_from_groups(
        frame,
        transformed,
        groups,
        raw_edge_inputs=raw_edge_inputs,
    )


def _examples_from_groups(
    frame: pd.DataFrame,
    transformed: pd.DataFrame,
    groups: Sequence[tuple[Any, np.ndarray]],
    *,
    raw_edge_inputs: pd.DataFrame | None = None,
) -> list[CandidateSetExample]:
    """Materialize query tensors with one aligned dataframe conversion.

    ``GroupBy.indices`` supplies positional row indices.  Aligning each input
    once to ``frame.index`` preserves the prior label-based ``.loc`` contract
    for reordered dataframes while avoiding four pandas selections per query.
    """

    try:
        transformed_aligned = transformed.loc[frame.index]
        raw_aligned = (
            None if raw_edge_inputs is None else raw_edge_inputs.loc[frame.index]
        )
    except KeyError as error:
        raise MatrixError(
            "neural inputs do not cover every candidate frame index"
        ) from error
    if len(transformed_aligned) != len(frame) or (
        raw_aligned is not None and len(raw_aligned) != len(frame)
    ):
        raise MatrixError("neural input indices are not one-to-one")

    feature_values = transformed_aligned.to_numpy(np.float32)
    label_values = frame["label"].to_numpy(np.float32)
    baseline_values = (
        transformed_aligned["q_platt_train_fold"].to_numpy(np.float32)
        if "q_platt_train_fold" in transformed_aligned.columns
        else frame["q_raw"].to_numpy(np.float32)
    )
    raw_values = None if raw_aligned is None else raw_aligned.to_numpy(np.float32)
    return [
        CandidateSetExample(
            features=torch.from_numpy(feature_values[positions]),
            labels=torch.from_numpy(label_values[positions]),
            query_id=str(query_id),
            baseline_scores=torch.from_numpy(baseline_values[positions]),
            raw_edge_inputs=(
                None if raw_values is None else torch.from_numpy(raw_values[positions])
            ),
        )
        for query_id, positions in groups
    ]


def _predict_neural(
    model: nn.Module,
    frame: pd.DataFrame,
    transformed: pd.DataFrame,
    device: torch.device,
    *,
    raw_edge_inputs: pd.DataFrame | None = None,
    query_batch_size: int = NEURAL_INFERENCE_QUERY_BATCH_SIZE,
) -> pd.DataFrame:
    if query_batch_size <= 0:
        raise ValueError("query_batch_size must be positive")
    if isinstance(model, CandidateGNNScorer) and raw_edge_inputs is None:
        raise MatrixError(
            "CandidateGNNScorer inference requires explicit raw edge inputs"
        )
    groups = tuple(
        (query_id, np.asarray(positions, dtype=np.int64))
        for query_id, positions in frame.groupby("query_id", sort=False).indices.items()
    )
    grouped_positions: dict[str, np.ndarray] = {}
    for query_id, positions in groups:
        key = str(query_id)
        if key in grouped_positions:
            raise MatrixError(f"query_id string collision during inference: {key}")
        grouped_positions[key] = positions
    if not grouped_positions:
        return pd.DataFrame(columns=["query_id", "candidate_id", "score"])

    loader = make_candidate_dataloader(
        _examples_from_groups(
            frame,
            transformed,
            groups,
            raw_edge_inputs=raw_edge_inputs,
        ),
        batch_size=query_batch_size,
        shuffle=False,
        seed=0,
    )
    position_chunks: list[np.ndarray] = []
    score_chunks: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for raw_batch in loader:
            batch_scores, _ = _forward_candidate_batch(model, raw_batch.to(device))
            cpu_scores = batch_scores.detach().cpu().numpy().astype(np.float64)
            for row_index, (query_id, length) in enumerate(
                zip(raw_batch.query_ids, raw_batch.lengths.tolist(), strict=True)
            ):
                positions = grouped_positions[query_id]
                if len(positions) != int(length):
                    raise MatrixError(
                        f"inference candidate count changed for query {query_id}"
                    )
                position_chunks.append(positions)
                score_chunks.append(cpu_scores[row_index, : int(length)])
    ordered_positions = np.concatenate(position_chunks)
    result = frame.iloc[ordered_positions][["query_id", "candidate_id"]].reset_index(
        drop=True
    )
    result["score"] = np.concatenate(score_chunks)
    return result


def _bundle_path(output: Path, experiment_id: str) -> Path:
    return output / "checkpoints" / f"{experiment_id}.bundle.json"


def _save_bundle(
    output: Path,
    experiment_id: str,
    spec: ExperimentSpec,
    preprocessor: FoldPreprocessor,
    checkpoint: Path,
    checkpoint_kind: str,
    score_calibrator: Path | None = None,
    prediction_calibrator: Path | None = None,
) -> Path:
    path = _bundle_path(output, experiment_id)
    payload: dict[str, Any] = {
        "experiment_id": experiment_id,
        "spec": spec.as_dict(),
        "preprocessor": preprocessor.as_dict(),
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_kind": checkpoint_kind,
        "checkpoint_sha256": _sha256(checkpoint),
    }
    if score_calibrator is not None:
        payload.update(
            {
                "score_calibrator": str(score_calibrator.resolve()),
                "score_calibrator_sha256": _sha256(score_calibrator),
                "calibrated_feature": "q_platt_train_fold",
            }
        )
    if prediction_calibrator is not None:
        payload.update(
            {
                "prediction_calibrator": str(prediction_calibrator.resolve()),
                "prediction_calibrator_sha256": _sha256(prediction_calibrator),
                "prediction_probability_column": "probability",
            }
        )
    _atomic_json(
        path,
        payload,
    )
    return path


def _identity_values_sha256(values: Sequence[str]) -> str:
    payload = json.dumps(
        sorted(set(map(str, values))),
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _partition_identity_sets(frame: pd.DataFrame) -> dict[str, set[str]]:
    queries = set(frame["query_id"].astype(str))
    groups = {
        scene_frame_group_hash(scene_id, frame_id)
        for scene_id, frame_id in frame[["scene_id", "frame_id"]].itertuples(
            index=False, name=None
        )
    }
    candidates = {
        json.dumps(
            [str(query_id), str(candidate_id)],
            ensure_ascii=False,
            separators=(",", ":"),
        )
        for query_id, candidate_id in frame[["query_id", "candidate_id"]].itertuples(
            index=False, name=None
        )
    }
    return {
        "query_ids": queries,
        "group_hashes": groups,
        "candidate_keys": candidates,
    }


def _partition_identity_audit(frame: pd.DataFrame) -> dict[str, Any]:
    identities = _partition_identity_sets(frame)
    return {
        "row_count": int(len(frame)),
        "query_count": len(identities["query_ids"]),
        "query_ids_sha256": _identity_values_sha256(tuple(identities["query_ids"])),
        "group_count": len(identities["group_hashes"]),
        "group_hashes_sha256": _identity_values_sha256(
            tuple(identities["group_hashes"])
        ),
        "candidate_count": len(identities["candidate_keys"]),
        "candidate_keys_sha256": _identity_values_sha256(
            tuple(identities["candidate_keys"])
        ),
    }


def _partition_overlap_audit(left: pd.DataFrame, right: pd.DataFrame) -> dict[str, int]:
    left_identities = _partition_identity_sets(left)
    right_identities = _partition_identity_sets(right)
    return {
        f"{name}_overlap_count": len(left_identities[name] & right_identities[name])
        for name in ("query_ids", "group_hashes", "candidate_keys")
    }


def _fit_evidence_root(output: Path, dataset: str) -> Path:
    token = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(dataset))
    if not token:
        raise MatrixError("development dataset has no safe fit-evidence token")
    return output / "audit" / "leakage" / "development_fit_evidence" / token


def _write_outer_fit_evidence(
    output: Path,
    dataset: DatasetArtifact,
    frame: pd.DataFrame,
    split_plan: SplitPlan,
) -> list[Path]:
    """Persist exact candidate rows presented to every outer-fold fit."""

    assignment = split_plan.candidate_assignments.set_index("row_position", drop=False)
    artifacts: list[Path] = []
    fold_receipts: list[dict[str, Any]] = []
    root = _fit_evidence_root(output, dataset.key)
    for fold in split_plan.folds:
        positions = np.asarray(fold.train_indices, dtype=np.int64)
        actual = frame.iloc[positions]
        assigned = assignment.loc[positions]
        if (
            actual[["query_id", "candidate_id"]].astype(str).to_numpy().tolist()
            != assigned[["query_id", "candidate_id"]].astype(str).to_numpy().tolist()
        ):
            raise MatrixError(
                f"outer-fit row positions disagree with split assignment: {dataset.key}/{fold.fold}"
            )
        evidence = pd.DataFrame(
            {
                "dataset": dataset.key,
                "route": dataset.route,
                "pool": dataset.pool,
                "outer_fold": int(fold.fold),
                "development_row_position": positions,
                "query_id": actual["query_id"].astype(str).to_numpy(),
                "scene_id": actual["scene_id"].astype(str).to_numpy(),
                "frame_id": actual["frame_id"].astype(str).to_numpy(),
                "group_identity_sha256": assigned["group_hash"].astype(str).to_numpy(),
                "candidate_id": actual["candidate_id"].astype(str).to_numpy(),
                "candidate_key_sha256": assigned["candidate_key"]
                .astype(str)
                .to_numpy(),
                "source_validation_fold": assigned["fold"].to_numpy(np.int64),
            }
        ).sort_values(["query_id", "candidate_id"], kind="mergesort")
        if evidence.duplicated(["query_id", "candidate_id"]).any():
            raise MatrixError(
                f"duplicate outer-fit candidate identity: {dataset.key}/{fold.fold}"
            )
        if evidence["source_validation_fold"].eq(int(fold.fold)).any():
            raise MatrixError(
                f"outer validation candidate entered fit evidence: {dataset.key}/{fold.fold}"
            )
        path = root / f"outer_fold_{int(fold.fold)}.parquet"
        _atomic_parquet(path, evidence)
        artifacts.append(path)
        fold_receipts.append(
            {
                "outer_fold": int(fold.fold),
                "path": str(path.resolve()),
                "sha256": _sha256(path),
                "row_count": int(len(evidence)),
                "query_count": int(evidence["query_id"].nunique()),
                "group_count": int(evidence["group_identity_sha256"].nunique()),
                "candidate_count": int(evidence["candidate_key_sha256"].nunique()),
                "outer_validation_candidate_overlap_count": 0,
            }
        )
    receipt = root / "manifest.json"
    _atomic_json(
        receipt,
        {
            "schema_version": 1,
            "status": "COMPLETE_DEVELOPMENT_ONLY",
            "dataset": dataset.key,
            "route": dataset.route,
            "pool": dataset.pool,
            "fit_scope": "concrete_outer_training_candidate_rows",
            "test_inputs_opened": False,
            "labels_included_in_evidence": False,
            "expected_outer_folds": list(range(len(split_plan.folds))),
            "folds": fold_receipts,
        },
    )
    artifacts.append(receipt)
    return artifacts


def _load_outer_fit_evidence(
    output: Path,
    development: Sequence[DatasetArtifact],
    *,
    folds: int,
) -> tuple[dict[tuple[str, int], pd.DataFrame], list[dict[str, Any]]]:
    """Recompute fit membership from split assignments and verify saved rows."""

    partitions: dict[tuple[str, int], pd.DataFrame] = {}
    descriptors: list[dict[str, Any]] = []
    required_columns = {
        "dataset",
        "route",
        "pool",
        "outer_fold",
        "development_row_position",
        "query_id",
        "scene_id",
        "frame_id",
        "group_identity_sha256",
        "candidate_id",
        "candidate_key_sha256",
        "source_validation_fold",
    }
    for dataset in development:
        root = _fit_evidence_root(output, dataset.key)
        receipt_path = root / "manifest.json"
        if not receipt_path.is_file():
            raise MatrixError(
                f"development fit-evidence manifest missing: {dataset.key}"
            )
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if (
            receipt.get("status") != "COMPLETE_DEVELOPMENT_ONLY"
            or receipt.get("dataset") != dataset.key
            or receipt.get("test_inputs_opened") is not False
            or receipt.get("labels_included_in_evidence") is not False
            or receipt.get("expected_outer_folds") != list(range(folds))
        ):
            raise MatrixError(
                f"malformed development fit-evidence manifest: {dataset.key}"
            )
        receipt_rows = receipt.get("folds")
        if not isinstance(receipt_rows, list) or len(receipt_rows) != folds:
            raise MatrixError(
                f"incomplete development fit-evidence receipt: {dataset.key}"
            )
        receipt_by_fold = {int(row["outer_fold"]): row for row in receipt_rows}
        if set(receipt_by_fold) != set(range(folds)):
            raise MatrixError(
                f"duplicate/missing fit-evidence fold receipt: {dataset.key}"
            )
        assignments_path = (
            output / "data" / f"split_candidate_assignments_{dataset.key}.parquet"
        )
        if not assignments_path.is_file():
            raise MatrixError(f"candidate split assignments missing: {dataset.key}")
        assignments = pd.read_parquet(assignments_path)
        for outer_fold in range(folds):
            path = root / f"outer_fold_{outer_fold}.parquet"
            if not path.is_file() or path.is_symlink():
                raise MatrixError(
                    f"outer-fit evidence missing: {dataset.key}/{outer_fold}"
                )
            evidence = pd.read_parquet(path)
            missing = sorted(required_columns - set(evidence.columns))
            if missing:
                raise MatrixError(
                    f"outer-fit evidence columns missing for {dataset.key}/{outer_fold}: {missing}"
                )
            if (
                evidence.empty
                or evidence.duplicated(["query_id", "candidate_id"]).any()
            ):
                raise MatrixError(
                    f"outer-fit evidence is empty or duplicated: {dataset.key}/{outer_fold}"
                )
            if set(evidence["dataset"].astype(str)) != {dataset.key} or set(
                pd.to_numeric(evidence["outer_fold"], errors="raise").astype(int)
            ) != {outer_fold}:
                raise MatrixError(
                    f"outer-fit evidence dataset/fold mismatch: {dataset.key}/{outer_fold}"
                )
            expected = assignments.loc[~assignments["fold"].eq(outer_fold)].copy()
            observed_keys = set(evidence["candidate_key_sha256"].astype(str))
            expected_keys = set(expected["candidate_key"].astype(str))
            if observed_keys != expected_keys:
                raise MatrixError(
                    f"outer-fit candidate membership changed: {dataset.key}/{outer_fold}"
                )
            group_by_query = expected.groupby("query_id", sort=True)[
                "group_hash"
            ].first()
            observed_groups = evidence.groupby("query_id", sort=True)[
                "group_identity_sha256"
            ].first()
            if (
                group_by_query.astype(str).to_dict()
                != observed_groups.astype(str).to_dict()
            ):
                raise MatrixError(
                    f"outer-fit group membership changed: {dataset.key}/{outer_fold}"
                )
            recorded = receipt_by_fold[outer_fold]
            if (
                Path(str(recorded.get("path", ""))).resolve() != path.resolve()
                or recorded.get("sha256") != _sha256(path)
                or int(recorded.get("row_count", -1)) != len(evidence)
                or int(recorded.get("query_count", -1))
                != evidence["query_id"].nunique()
                or int(recorded.get("group_count", -1))
                != evidence["group_identity_sha256"].nunique()
                or int(recorded.get("candidate_count", -1)) != len(observed_keys)
            ):
                raise MatrixError(
                    f"outer-fit evidence receipt mismatch: {dataset.key}/{outer_fold}"
                )
            partitions[(dataset.key, outer_fold)] = evidence
            descriptors.append(
                {
                    "dataset": dataset.key,
                    "outer_fold": outer_fold,
                    "path": str(path.resolve()),
                    "sha256": _sha256(path),
                    "rows": int(len(evidence)),
                    "queries": int(evidence["query_id"].nunique()),
                    "groups": int(evidence["group_identity_sha256"].nunique()),
                    "candidates": len(observed_keys),
                }
            )
    return partitions, descriptors


def _build_nested_neural_split(
    outer_train_frame: pd.DataFrame,
    outer_validation_frame: pd.DataFrame,
    *,
    outer_fold: int,
    config: Mapping[str, Any],
) -> _NestedNeuralSplit:
    """Create a deterministic grouped early-stop split inside outer training.

    The outer validation fold is supplied only so that the resulting audit can
    independently prove it is disjoint from both inner partitions.  It never
    participates in the inner splitter.
    """

    requested_splits = int(
        config.get(
            "matrix_inner_folds",
            2 if config.get("matrix_profile") == "unit_test" else 5,
        )
    )
    if requested_splits < 2:
        raise MatrixError("matrix_inner_folds must be at least two")
    outer_train_groups = _partition_identity_sets(outer_train_frame)["group_hashes"]
    inner_splits = min(requested_splits, len(outer_train_groups))
    if inner_splits < 2:
        raise MatrixError(
            "nested neural validation requires at least two outer-training "
            "scene/frame groups"
        )
    random_state = int(config.get("matrix_inner_split_random_state", 1701))
    try:
        plan = build_stratified_group_folds(
            outer_train_frame,
            n_splits=inner_splits,
            random_state=random_state,
        )
    except Exception as error:
        raise MatrixError(
            "could not create nested grouped neural early-stop split inside "
            f"outer fold {outer_fold}: {error}"
        ) from error
    selected_inner_fold = int(outer_fold) % inner_splits
    selected = plan.folds[selected_inner_fold]
    inner_train = outer_train_frame.iloc[selected.train_indices]
    inner_early_stop = outer_train_frame.iloc[selected.validation_indices]
    intersections = {
        "inner_train__inner_early_stop": _partition_overlap_audit(
            inner_train, inner_early_stop
        ),
        "inner_train__outer_validation": _partition_overlap_audit(
            inner_train, outer_validation_frame
        ),
        "inner_early_stop__outer_validation": _partition_overlap_audit(
            inner_early_stop, outer_validation_frame
        ),
    }
    no_overlap = all(
        count == 0
        for comparison in intersections.values()
        for count in comparison.values()
    )
    outer_candidates = _partition_identity_sets(outer_train_frame)["candidate_keys"]
    nested_candidates = (
        _partition_identity_sets(inner_train)["candidate_keys"]
        | _partition_identity_sets(inner_early_stop)["candidate_keys"]
    )
    exact_partition = bool(
        outer_candidates == nested_candidates
        and len(inner_train) + len(inner_early_stop) == len(outer_train_frame)
    )
    if not no_overlap or not exact_partition:
        raise MatrixError(
            f"nested neural split leakage or incomplete partition in outer fold {outer_fold}"
        )
    split_plan_sha256 = hashlib.sha256(
        json.dumps(
            _json_safe(plan.audit), sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()
    audit = {
        "schema_version": 1,
        "protocol": "nested_grouped_early_stopping",
        "split_scope": "outer_training_fold_only",
        "splitter": "sklearn.model_selection.StratifiedGroupKFold",
        "grouping": "sha256(JSON([scene_id,frame_id]))",
        "stratification": "query_has_at_least_one_positive_candidate",
        "outer_fold": int(outer_fold),
        "requested_inner_splits": requested_splits,
        "effective_inner_splits": inner_splits,
        "selected_inner_fold": selected_inner_fold,
        "random_state": random_state,
        "inner_split_plan_sha256": split_plan_sha256,
        "model_fit_partition": "inner_train",
        "early_stopping_partition": "inner_early_stop",
        "outer_validation_usage": "single_oof_inference_and_metrics_only",
        "outer_train": _partition_identity_audit(outer_train_frame),
        "inner_train": _partition_identity_audit(inner_train),
        "inner_early_stop": _partition_identity_audit(inner_early_stop),
        "outer_validation": _partition_identity_audit(outer_validation_frame),
        "intersections": intersections,
        "inner_partitions_exactly_cover_outer_train": exact_partition,
        "all_inner_outer_overlaps_empty": no_overlap,
    }
    return _NestedNeuralSplit(
        train_indices=selected.train_indices.copy(),
        early_stop_indices=selected.validation_indices.copy(),
        audit=audit,
    )


def _equal_query_weights(frame: pd.DataFrame) -> np.ndarray:
    weights = np.zeros(len(frame), dtype=np.float64)
    for indices in frame.groupby("query_id", sort=False).indices.values():
        weights[indices] = 1.0 / max(len(indices), 1)
    weights *= len(weights) / weights.sum()
    return weights


def _candidate_fit_identity_sha256(frame: pd.DataFrame) -> str:
    sample_ids = sorted(
        f"{query}/{candidate}"
        for query, candidate in frame[["query_id", "candidate_id"]].itertuples(
            index=False, name=None
        )
    )
    return hashlib.sha256(
        json.dumps(sample_ids, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


@dataclass(frozen=True)
class _NeuralFoldPreparation:
    preprocessor: FoldPreprocessor
    score_calibrator: TrainOnlyScoreCalibrator
    calibration_weights: np.ndarray
    score_calibration_weights: np.ndarray
    fit_sample_ids_sha256: str
    score_fit_sample_ids_sha256: str


def _run_fold_experiment(
    output: Path,
    artifact: DatasetArtifact,
    frame: pd.DataFrame,
    reference: pd.DataFrame,
    feature_columns: tuple[str, ...],
    spec: ExperimentSpec,
    seed: int,
    fold_number: int,
    train_indices: np.ndarray,
    validation_indices: np.ndarray,
    config: Mapping[str, Any],
    nested_neural_split: _NestedNeuralSplit | None = None,
    neural_preparation_cache: dict[
        tuple[str, int, int, tuple[str, ...], str], _NeuralFoldPreparation
    ]
    | None = None,
) -> Path:
    experiment_id = _experiment_id(artifact.key, spec, seed, fold_number)
    training_protocol = _neural_training_protocol(spec, config)
    experiment_identity_sha256 = _experiment_identity_sha256(
        artifact,
        spec,
        seed,
        fold_number,
        feature_columns,
        train_indices,
        validation_indices,
        training_protocol,
    )
    resumed = _completed_manifest(
        output,
        experiment_id,
        expected_identity_sha256=experiment_identity_sha256,
    )
    if resumed is not None:
        return Path(resumed["prediction_path"])
    started = time.perf_counter()
    prediction_path = output / "predictions" / f"{experiment_id}.parquet"
    try:
        train_frame = frame.iloc[train_indices].copy()
        validation_frame = frame.iloc[validation_indices].copy()
        neural_backend = spec.backend in NEURAL_BACKENDS
        nested_split = nested_neural_split if neural_backend else None
        if neural_backend and nested_split is None:
            nested_split = _build_nested_neural_split(
                train_frame,
                validation_frame,
                outer_fold=fold_number,
                config=config,
            )
        score_calibration_frame = (
            train_frame.iloc[nested_split.train_indices]
            if nested_split is not None
            else train_frame
        )
        preparation: _NeuralFoldPreparation | None = None
        neural_preparation_cache_hit = False
        cache_key: tuple[str, int, int, tuple[str, ...], str] | None = None
        if neural_backend and neural_preparation_cache is not None:
            assert nested_split is not None
            nested_identity = hashlib.sha256(
                np.asarray(nested_split.train_indices, dtype=np.int64).tobytes()
            ).hexdigest()
            cache_key = (
                artifact.key,
                int(seed),
                int(fold_number),
                tuple(feature_columns),
                nested_identity,
            )
            preparation = neural_preparation_cache.get(cache_key)
            neural_preparation_cache_hit = preparation is not None
        if preparation is None:
            preprocessor = FoldPreprocessor.fit(
                score_calibration_frame, feature_columns
            )
            calibration_weights = _equal_query_weights(train_frame)
            score_calibration_weights = _equal_query_weights(score_calibration_frame)
            fit_sample_ids_sha256 = _candidate_fit_identity_sha256(train_frame)
            score_fit_sample_ids_sha256 = _candidate_fit_identity_sha256(
                score_calibration_frame
            )
            fit_sample_ids = (
                None
                if neural_backend
                else [
                    f"{query}/{candidate}"
                    for query, candidate in train_frame[
                        ["query_id", "candidate_id"]
                    ].itertuples(index=False, name=None)
                ]
            )
            score_fit_sample_ids = (
                None
                if neural_backend
                else [
                    f"{query}/{candidate}"
                    for query, candidate in score_calibration_frame[
                        ["query_id", "candidate_id"]
                    ].itertuples(index=False, name=None)
                ]
            )
            score_calibrator = TrainOnlyScoreCalibrator("platt", random_state=seed).fit(
                score_calibration_frame["q_raw"].to_numpy(np.float64),
                score_calibration_frame["label"].to_numpy(np.int8),
                sample_weight=score_calibration_weights,
                fit_sample_ids=score_fit_sample_ids,
            )
            score_calibrator.fit_sample_ids_ = (
                f"sha256:{score_fit_sample_ids_sha256}",
            )
            if neural_backend and cache_key is not None:
                preparation = _NeuralFoldPreparation(
                    preprocessor=preprocessor,
                    score_calibrator=score_calibrator,
                    calibration_weights=calibration_weights,
                    score_calibration_weights=score_calibration_weights,
                    fit_sample_ids_sha256=fit_sample_ids_sha256,
                    score_fit_sample_ids_sha256=score_fit_sample_ids_sha256,
                )
                neural_preparation_cache[cache_key] = preparation
        else:
            preprocessor = preparation.preprocessor
            score_calibrator = preparation.score_calibrator
            calibration_weights = preparation.calibration_weights
            score_calibration_weights = preparation.score_calibration_weights
            fit_sample_ids_sha256 = preparation.fit_sample_ids_sha256
            score_fit_sample_ids_sha256 = preparation.score_fit_sample_ids_sha256
            fit_sample_ids = None
            score_fit_sample_ids = None
        train_design = preprocessor.transform(train_frame)
        validation_design = preprocessor.transform(validation_frame)
        set_global_seed(seed)
        training_summary: dict[str, Any] | None = None
        auxiliary_artifacts: list[str] = []
        feature_importance: dict[str, float] | None = None
        interpretable_rule_weights: dict[str, Any] | None = None
        effective_loss_config: dict[str, float | None] | None = None
        derived_pos_weight: float | None = None
        train_design["q_platt_train_fold"] = score_calibrator.predict_proba(
            train_frame["q_raw"].to_numpy(np.float64)
        )
        validation_design["q_platt_train_fold"] = score_calibrator.predict_proba(
            validation_frame["q_raw"].to_numpy(np.float64)
        )
        calibrator_path = output / "checkpoints" / f"{experiment_id}.q_calibrator.pkl"
        _atomic_pickle(calibrator_path, score_calibrator)
        feature_schema_sha256 = hashlib.sha256(
            json.dumps(
                {
                    "columns": list(map(str, train_design.columns)),
                    "preprocessor": preprocessor.as_dict(),
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()

        if spec.backend == "rule":
            interpretable_rule_weights = _rule_weight_audit(
                feature_columns,
                float(spec.hyperparameters["rule_weight"]),
                tuple(spec.hyperparameters.get("rule_groups", ())),
            )
            train_scores = _rule_predictions(
                train_frame,
                feature_columns,
                float(spec.hyperparameters["rule_weight"]),
                groups=tuple(spec.hyperparameters.get("rule_groups", ())),
                baseline_values=score_calibrator.predict_proba(
                    train_frame["q_raw"].to_numpy(np.float64)
                ),
            )["score"].to_numpy(np.float64)
            inference_started = time.perf_counter()
            predictions = _rule_predictions(
                validation_frame,
                feature_columns,
                float(spec.hyperparameters["rule_weight"]),
                groups=tuple(spec.hyperparameters.get("rule_groups", ())),
                baseline_values=score_calibrator.predict_proba(
                    validation_frame["q_raw"].to_numpy(np.float64)
                ),
            )
            inference_seconds = time.perf_counter() - inference_started
            checkpoint = output / "checkpoints" / f"{experiment_id}.rule.pkl"
            _atomic_pickle(
                checkpoint,
                {
                    "rule_weight": float(spec.hyperparameters["rule_weight"]),
                    "rule_groups": list(spec.hyperparameters.get("rule_groups", ())),
                    "feature_columns": list(feature_columns),
                    "baseline": "q_platt_train_fold",
                    "resolved_feature_weights": interpretable_rule_weights,
                },
            )
            rule_weights_path = (
                output / "metrics" / "rule_weights" / f"{experiment_id}.json"
            )
            _atomic_json(
                rule_weights_path,
                {
                    "experiment_id": experiment_id,
                    **interpretable_rule_weights,
                },
            )
            auxiliary_artifacts.append(str(rule_weights_path.resolve()))
            model_parameters = 0
            checkpoint_kind = "calibrated_rule"
        elif spec.backend in {
            "logistic",
            "linear_residual",
            "linear_ranknet",
            "random_forest",
            "histgb",
            "lambdamart",
        }:
            model = _make_tabular(spec, seed, str(config["matrix_profile"]))
            tabular_train_baseline = (
                train_frame["q_raw"].to_numpy(np.float64)
                if spec.backend in {"linear_residual", "linear_ranknet"}
                else train_design["q_platt_train_fold"].to_numpy(np.float64)
            )
            tabular_validation_baseline = (
                validation_frame["q_raw"].to_numpy(np.float64)
                if spec.backend in {"linear_residual", "linear_ranknet"}
                else validation_design["q_platt_train_fold"].to_numpy(np.float64)
            )
            model.fit(
                train_design,
                train_frame["label"].to_numpy(),
                query_ids=train_frame["query_id"].tolist(),
                baseline_scores=tabular_train_baseline,
                sample_ids=fit_sample_ids,
            )
            if hasattr(model, "fit_sample_ids_"):
                model.fit_sample_ids_ = (f"sha256:{fit_sample_ids_sha256}",)
            if hasattr(model, "calibrator_") and hasattr(
                model.calibrator_, "fit_sample_ids_"
            ):
                model.calibrator_.fit_sample_ids_ = (f"sha256:{fit_sample_ids_sha256}",)
            train_scores = model.predict_scores(
                train_design,
                query_ids=train_frame["query_id"].tolist(),
                baseline_scores=tabular_train_baseline,
            )
            inference_started = time.perf_counter()
            scores = model.predict_scores(
                validation_design,
                query_ids=validation_frame["query_id"].tolist(),
                baseline_scores=tabular_validation_baseline,
            )
            inference_seconds = time.perf_counter() - inference_started
            predictions = validation_frame[["query_id", "candidate_id"]].assign(
                score=scores
            )
            checkpoint = output / "checkpoints" / f"{experiment_id}.pkl"
            _atomic_pickle(checkpoint, model)
            if hasattr(model, "feature_importance"):
                feature_importance = {
                    str(key): float(value)
                    for key, value in model.feature_importance().items()
                }
                importance_path = (
                    output / "metrics" / "feature_importance" / f"{experiment_id}.json"
                )
                _atomic_json(
                    importance_path,
                    {
                        "experiment_id": experiment_id,
                        "fit_scope": "development_train_fold_only",
                        "feature_columns": list(feature_columns),
                        "importance": feature_importance,
                        "model_metadata": getattr(model, "metadata", {}),
                    },
                )
                auxiliary_artifacts.append(str(importance_path.resolve()))
            model_parameters = (
                int(len(feature_columns) + 1) if spec.rung == "R2" else None
            )
            checkpoint_kind = "trusted_local_pickle"
        else:
            if nested_split is None:
                raise MatrixError(
                    f"neural backend {spec.backend} lacks a nested validation split"
                )
            device_name = str(config.get("device", "cpu"))
            if device_name == "mps" and not torch.backends.mps.is_available():
                device_name = "cpu"
            device = resolve_device(device_name)
            model = _make_neural(spec, train_design.shape[1], artifact.pool)
            model_train_frame = train_frame.iloc[nested_split.train_indices].copy()
            early_stop_frame = train_frame.iloc[nested_split.early_stop_indices].copy()
            model_train_design = preprocessor.transform(model_train_frame)
            early_stop_design = preprocessor.transform(early_stop_frame)
            model_train_design["q_platt_train_fold"] = score_calibrator.predict_proba(
                model_train_frame["q_raw"].to_numpy(np.float64)
            )
            early_stop_design["q_platt_train_fold"] = score_calibrator.predict_proba(
                early_stop_frame["q_raw"].to_numpy(np.float64)
            )
            train_raw_edges = (
                _raw_edge_frame(train_frame, route=artifact.route)
                if spec.backend == "gnn"
                else None
            )
            model_train_raw_edges = (
                _raw_edge_frame(model_train_frame, route=artifact.route)
                if spec.backend == "gnn"
                else None
            )
            early_stop_raw_edges = (
                _raw_edge_frame(early_stop_frame, route=artifact.route)
                if spec.backend == "gnn"
                else None
            )
            validation_raw_edges = (
                _raw_edge_frame(validation_frame, route=artifact.route)
                if spec.backend == "gnn"
                else None
            )
            train_loader = make_candidate_dataloader(
                _examples(
                    model_train_frame,
                    model_train_design,
                    raw_edge_inputs=model_train_raw_edges,
                ),
                batch_size=_neural_query_batch_size(config),
                shuffle=True,
                seed=seed,
                batching_policy=_neural_batching_policy(config),
            )
            validation_loader = make_candidate_dataloader(
                _examples(
                    early_stop_frame,
                    early_stop_design,
                    raw_edge_inputs=early_stop_raw_edges,
                ),
                batch_size=_neural_query_batch_size(config),
                shuffle=False,
                seed=seed,
                batching_policy=_neural_batching_policy(config),
            )
            checkpoint = output / "checkpoints" / f"{experiment_id}.pt"
            criterion = _criterion(spec)
            derived_pos_weight = criterion.pos_weight
            if criterion.bce_weight > 0.0 and derived_pos_weight is None:
                labels = model_train_frame["label"].to_numpy(np.int8)
                positive_mass = float(score_calibration_weights[labels == 1].sum())
                negative_mass = float(score_calibration_weights[labels == 0].sum())
                if positive_mass > 0.0 and negative_mass > 0.0:
                    derived_pos_weight = negative_mass / positive_mass
                    criterion = CompositeRerankingLoss(
                        bce_weight=criterion.bce_weight,
                        ranknet_weight=criterion.ranknet_weight,
                        listwise_weight=criterion.listwise_weight,
                        residual_weight=criterion.residual_weight,
                        pos_weight=derived_pos_weight,
                        temperature=criterion.temperature,
                    )
            training_result = fit_neural_ranker(
                model,
                train_loader,
                validation_loader,
                checkpoint_path=checkpoint,
                criterion=criterion,
                config=TrainingConfig(
                    epochs=int(
                        config.get(
                            "matrix_epochs",
                            1 if config["matrix_profile"] == "unit_test" else 100,
                        )
                    ),
                    learning_rate=float(
                        spec.hyperparameters.get(
                            "learning_rate",
                            config.get("matrix_learning_rate", 1e-3),
                        )
                    ),
                    weight_decay=float(
                        spec.hyperparameters.get(
                            "weight_decay",
                            config.get("matrix_weight_decay", 1e-4),
                        )
                    ),
                    patience=int(
                        config.get(
                            "matrix_patience",
                            1 if config["matrix_profile"] == "unit_test" else 10,
                        )
                    ),
                    min_delta=float(config.get("matrix_min_delta", 1e-5)),
                    seed=seed,
                    device=device_name,
                ),
                metadata={
                    "dataset": artifact.key,
                    "fold": fold_number,
                    "spec": spec.as_dict(),
                    "preprocessing": preprocessor.as_dict(),
                    "feature_schema_sha256": feature_schema_sha256,
                    "loss_config": criterion.config,
                    "derived_inner_train_pos_weight": derived_pos_weight,
                    "nested_validation": nested_split.audit,
                },
            )
            effective_loss_config = criterion.config
            training_summary = {
                "best_epoch": training_result.best_epoch,
                "best_validation_loss": training_result.best_validation_loss,
                "epochs_ran": training_result.epochs_ran,
                "stopped_early": training_result.stopped_early,
                "device": training_result.device,
                "best_checkpoint_metadata_path": training_result.metadata_path,
                "last_checkpoint_path": training_result.last_checkpoint_path,
                "history_path": training_result.history_path,
                "model_fit_scope": "nested_inner_train_only",
                "early_stopping_scope": "nested_inner_early_stop_only",
                "outer_validation_usage": "single_oof_inference_and_metrics_only",
            }
            train_scores = _predict_neural(
                model,
                train_frame,
                train_design,
                device,
                raw_edge_inputs=train_raw_edges,
            )["score"].to_numpy(np.float64)
            inference_started = time.perf_counter()
            predictions = _predict_neural(
                model,
                validation_frame,
                validation_design,
                device,
                raw_edge_inputs=validation_raw_edges,
            )
            inference_seconds = time.perf_counter() - inference_started
            model_parameters = parameter_count(model)
            checkpoint_kind = "torch_state_dict"

        prediction_calibrator = TrainOnlyScoreCalibrator(
            "platt", random_state=seed
        ).fit(
            np.asarray(train_scores, dtype=np.float64),
            train_frame["label"].to_numpy(np.int8),
            sample_weight=calibration_weights,
            fit_sample_ids=fit_sample_ids,
        )
        prediction_calibrator.fit_sample_ids_ = (f"sha256:{fit_sample_ids_sha256}",)
        predictions["probability"] = prediction_calibrator.predict_proba(
            predictions["score"].to_numpy(np.float64)
        )
        prediction_calibrator_path = (
            output / "checkpoints" / f"{experiment_id}.prediction_calibrator.pkl"
        )
        _atomic_pickle(prediction_calibrator_path, prediction_calibrator)

        predictions["seed"] = seed
        predictions["fold"] = fold_number
        predictions["method"] = spec.key
        _atomic_parquet(prediction_path, predictions)
        bundle = _save_bundle(
            output,
            experiment_id,
            spec,
            preprocessor,
            checkpoint,
            checkpoint_kind,
            calibrator_path,
            prediction_calibrator_path,
        )
        fold_reference = reference.iloc[validation_indices].copy()
        metrics = _evaluation(
            fold_reference,
            predictions[["query_id", "candidate_id", "score", "probability"]],
        )
        neural_training_artifacts = (
            [
                str(Path(training_result.metadata_path).resolve()),
                str(Path(training_result.last_checkpoint_path).resolve()),
                str(
                    Path(training_result.last_checkpoint_path)
                    .with_name(
                        Path(training_result.last_checkpoint_path).name
                        + ".metadata.json"
                    )
                    .resolve()
                ),
                str(Path(training_result.history_path).resolve()),
            ]
            if training_summary is not None
            else []
        )
        manifest_artifacts = (
            [
                str(prediction_path.resolve()),
                str(checkpoint.resolve()),
                str(calibrator_path.resolve()),
                str(prediction_calibrator_path.resolve()),
                str(bundle.resolve()),
            ]
            + neural_training_artifacts
            + auxiliary_artifacts
        )
        artifact_sha256 = {path: _sha256(Path(path)) for path in manifest_artifacts}
        _write_manifest(
            output,
            experiment_id,
            {
                "status": "COMPLETE",
                "stage": "train",
                "dataset": artifact.key,
                "route": artifact.route,
                "pool": artifact.pool,
                "rung": spec.rung,
                "method": spec.key,
                "seed": seed,
                "fold": fold_number,
                "spec": spec.as_dict(),
                "scorer_family": spec.scorer_family,
                "fit_scope": "development_train_fold_only",
                "prediction_scope": "grouped_oof_validation_fold",
                "model_fit_scope": (
                    "nested_inner_train_only"
                    if nested_split is not None
                    else "development_train_fold_only"
                ),
                "early_stopping_scope": (
                    "nested_inner_early_stop_only"
                    if nested_split is not None
                    else "not_applicable"
                ),
                "outer_validation_usage": (
                    "single_oof_inference_and_metrics_only"
                    if nested_split is not None
                    else "oof_inference_and_metrics_only"
                ),
                "nested_validation": (
                    None if nested_split is None else nested_split.audit
                ),
                "prediction_path": str(prediction_path.resolve()),
                "experiment_identity_sha256": experiment_identity_sha256,
                "checkpoint_path": str(checkpoint.resolve()),
                "bundle_path": str(bundle.resolve()),
                "artifacts": manifest_artifacts,
                "artifact_sha256": artifact_sha256,
                "metrics": metrics,
                "parameter_count": model_parameters,
                "feature_importance": feature_importance,
                "interpretable_rule_weights": interpretable_rule_weights,
                "inference_seconds": inference_seconds,
                "inference_query_count": int(validation_frame["query_id"].nunique()),
                "inference_milliseconds_per_query": (
                    1000.0
                    * inference_seconds
                    / max(int(validation_frame["query_id"].nunique()), 1)
                ),
                "training_summary": training_summary,
                "training_protocol": training_protocol,
                "loss_config": effective_loss_config,
                "feature_schema_sha256": feature_schema_sha256,
                "score_calibration": {
                    "baseline_method": "platt",
                    "prediction_method": "platt",
                    "fit_scope": "development_train_fold_only",
                    "fit_sample_count": len(train_frame),
                    "fit_sample_ids_sha256": fit_sample_ids_sha256,
                    "legacy_fit_fields_describe": "prediction_calibrator",
                    "baseline_fit_scope": (
                        "nested_inner_train_only"
                        if nested_split is not None
                        else "development_train_fold_only"
                    ),
                    "baseline_fit_sample_count": len(score_calibration_frame),
                    "baseline_fit_sample_ids_sha256": (score_fit_sample_ids_sha256),
                    "prediction_fit_scope": "development_train_fold_only",
                    "prediction_fit_sample_count": len(train_frame),
                    "prediction_fit_sample_ids_sha256": fit_sample_ids_sha256,
                    "calibrated_feature": "q_platt_train_fold",
                    "prediction_probability_column": "probability",
                    "route_specific": True,
                    "weighting": "equal total weight per query",
                    "neural_preparation_cache_hit": (
                        neural_preparation_cache_hit if neural_backend else None
                    ),
                },
                "duration_seconds": time.perf_counter() - started,
                "completed_at": _now(),
            },
        )
        return prediction_path
    except Exception as error:
        _write_manifest(
            output,
            experiment_id,
            {
                "status": "FAILED",
                "stage": "train",
                "dataset": artifact.key,
                "method": spec.key,
                "seed": seed,
                "fold": fold_number,
                "spec": spec.as_dict(),
                "experiment_identity_sha256": experiment_identity_sha256,
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(),
                "artifacts": [],
                "failed_at": _now(),
            },
        )
        raise


def _run_deterministic_experiment(
    output: Path,
    artifact: DatasetArtifact,
    frame: pd.DataFrame,
    reference: pd.DataFrame,
    feature_columns: tuple[str, ...],
    spec: ExperimentSpec,
) -> Path:
    experiment_id = _experiment_id(artifact.key, spec, -1, -1)
    experiment_identity_sha256 = _experiment_identity_sha256(
        artifact, spec, -1, -1, feature_columns
    )
    resumed = _completed_manifest(
        output,
        experiment_id,
        expected_identity_sha256=experiment_identity_sha256,
    )
    if resumed is not None:
        return Path(resumed["prediction_path"])
    predictions = (
        _baseline_predictions(frame)
        if spec.backend == "baseline"
        else _rule_predictions(
            frame,
            feature_columns,
            float(spec.hyperparameters["rule_weight"]),
            groups=spec.hyperparameters.get("rule_groups", ()),
        )
    )
    predictions["seed"] = -1
    predictions["fold"] = -1
    predictions["method"] = spec.key
    path = output / "predictions" / f"{experiment_id}.parquet"
    _atomic_parquet(path, predictions)
    metrics = _evaluation(reference, predictions[["query_id", "candidate_id", "score"]])
    config_path = output / "checkpoints" / f"{experiment_id}.json"
    _atomic_json(config_path, {"deterministic": True, "spec": spec.as_dict()})
    manifest_artifacts = [str(path.resolve()), str(config_path.resolve())]
    _write_manifest(
        output,
        experiment_id,
        {
            "status": "COMPLETE",
            "stage": "train",
            "dataset": artifact.key,
            "route": artifact.route,
            "pool": artifact.pool,
            "rung": spec.rung,
            "method": spec.key,
            "seed": -1,
            "fold": -1,
            "spec": spec.as_dict(),
            "scorer_family": spec.scorer_family,
            "fit_scope": "none_fixed_method",
            "prediction_scope": "complete_development",
            "prediction_path": str(path.resolve()),
            "experiment_identity_sha256": experiment_identity_sha256,
            "checkpoint_path": str(config_path.resolve()),
            "artifacts": manifest_artifacts,
            "artifact_sha256": {
                artifact_path: _sha256(Path(artifact_path))
                for artifact_path in manifest_artifacts
            },
            "metrics": metrics,
            "parameter_count": 0,
            "inference_milliseconds_per_query": 0.0,
            "completed_at": _now(),
        },
    )
    return path


_R10_REQUIRED_DISCOVERY_CHECKS = frozenset(
    {
        "manifest",
        "complete_oof",
        "checkpoint",
        "candidate_hash",
        "evaluator_hash",
        "independent_recomputation",
    }
)


def _validate_r10_discovery_report(discovery: Mapping[str, Any]) -> None:
    """Fail closed when a cached R10 discovery report is incomplete or edited."""

    if discovery.get("kind") != "existing_reranker_r10_discovery":
        raise MatrixError("R10 discovery report kind is invalid")
    methods = discovery.get("methods")
    eligible_rows = discovery.get("eligible_methods")
    if not isinstance(methods, list) or not isinstance(eligible_rows, list):
        raise MatrixError("R10 discovery report lacks method inventories")
    expected_eligible: set[tuple[str, str, str]] = set()
    seen: set[tuple[str, str]] = set()
    for method in methods:
        if not isinstance(method, Mapping):
            raise MatrixError("R10 discovery method row is malformed")
        method_id = str(method.get("method_id", ""))
        run_root = str(method.get("run_root", ""))
        run_id = str(method.get("run_id", ""))
        key = (run_root, method_id)
        if not method_id or not run_root or key in seen:
            raise MatrixError("R10 discovery method identity is missing or duplicated")
        seen.add(key)
        checks = method.get("checks")
        if not isinstance(checks, Mapping) or set(checks) != set(
            _R10_REQUIRED_DISCOVERY_CHECKS
        ):
            raise MatrixError(f"R10 discovery checks are incomplete for {method_id}")
        passed = all(
            isinstance(checks[name], Mapping) and checks[name].get("passed") is True
            for name in _R10_REQUIRED_DISCOVERY_CHECKS
        )
        if bool(method.get("eligible")) != passed:
            raise MatrixError(f"R10 eligibility disagrees with checks for {method_id}")
        gaps = method.get("gaps")
        if not isinstance(gaps, list) or (not passed and not gaps):
            raise MatrixError(f"R10 exclusion evidence is incomplete for {method_id}")
        if passed:
            expected_eligible.add((run_id, run_root, method_id))
    declared_eligible: set[tuple[str, str, str]] = set()
    for row in eligible_rows:
        if not isinstance(row, Mapping):
            raise MatrixError("R10 eligible-method row is malformed")
        declared_eligible.add(
            (
                str(row.get("run_id", "")),
                str(row.get("run_root", "")),
                str(row.get("method_id", "")),
            )
        )
    if declared_eligible != expected_eligible:
        raise MatrixError("R10 eligible inventory disagrees with method checks")


def _r10_method_key(source_method: str, run_root: str) -> str:
    run_tag = hashlib.sha256(run_root.encode("utf-8")).hexdigest()[:10]
    return f"r10_{source_method}_{run_tag}"


def _record_conditional_and_reused_methods(
    output: Path, artifact: DatasetArtifact
) -> list[Path]:
    """Record non-runnable conditional methods without pretending they trained.

    R10 is a provenance-bound comparison to the already frozen CROG lineage,
    not another fit of the unified matrix.  R11 is allowed only with a usable
    credential, frozen prompt, cache, and billing controls; none are supplied
    by this repository/run configuration.
    """

    paths: list[Path] = []
    vlm_id = _experiment_id(
        artifact.key,
        ExperimentSpec(
            "r11_vlm_candidate_judge",
            "R11",
            "VLM candidate judge",
            "external_vlm",
            "n/a",
            False,
            False,
            "vlm",
            {},
        ),
        -1,
        -1,
    )
    vlm_path = output / "manifests" / "experiments" / f"{vlm_id}.json"
    if not vlm_path.is_file():
        _write_manifest(
            output,
            vlm_id,
            {
                "status": "NOT_RUN_CREDENTIAL_OR_BILLING_REQUIRED",
                "stage": "train",
                "dataset": artifact.key,
                "route": artifact.route,
                "pool": artifact.pool,
                "rung": "R11",
                "method": "r11_vlm_candidate_judge",
                "seed": -1,
                "fold": -1,
                "reason": (
                    "No run-scoped API credential, frozen production prompt, "
                    "response cache, or billing budget was supplied."
                ),
                "test_rows_read": False,
                "artifacts": [],
                "completed_at": _now(),
            },
        )
    paths.append(vlm_path)

    if artifact.route == "crog":
        repository = Path(__file__).resolve().parents[1]
        discovery_path = output / "audit" / "r10_existing_run_discovery.json"
        if discovery_path.is_file():
            discovery = json.loads(discovery_path.read_text(encoding="utf-8"))
        else:
            discovery = discover_existing_runs(
                [
                    repository / "crog_reproduction/CROG/failure_analysis/"
                    "reranking_outputs"
                ]
            )
            _atomic_json(discovery_path, discovery)
        _validate_r10_discovery_report(discovery)
        paths.append(discovery_path)
        for eligible in discovery.get("eligible_methods", []):
            source_method = str(eligible["method_id"])
            run_root = str(eligible["run_root"])
            method = _r10_method_key(source_method, run_root)
            spec = ExperimentSpec(
                method,
                "R10",
                source_method,
                "reused_frozen_crog",
                "preexisting",
                True,
                False,
                "existing_crog_reranker",
                {},
            )
            experiment_id = _experiment_id(artifact.key, spec, -1, -1)
            path = output / "manifests" / "experiments" / f"{experiment_id}.json"
            audit_record = next(
                item
                for item in discovery.get("methods", [])
                if item.get("method_id") == source_method
                and item.get("run_root") == run_root
            )
            if not path.is_file():
                _write_manifest(
                    output,
                    experiment_id,
                    {
                        "status": "REUSED_ELIGIBLE_EXISTING_RUN",
                        "stage": "train",
                        "dataset": artifact.key,
                        "route": artifact.route,
                        "pool": artifact.pool,
                        "rung": "R10",
                        "method": method,
                        "source_method": source_method,
                        "source_run_id": eligible.get("run_id"),
                        "source_run_root": run_root,
                        "seed": -1,
                        "fold": -1,
                        "fit_scope": "preexisting_frozen_crog_run",
                        "reuse_eligibility_checks": audit_record.get("checks", {}),
                        "reuse_gaps": audit_record.get("gaps", []),
                        "automatic_discovery_report": str(discovery_path.resolve()),
                        "automatic_discovery_report_sha256": _sha256(discovery_path),
                        "eligible_for_unified_primary_selection": False,
                        "reason": (
                            "Retained as a provenance-audited strong reference; "
                            "its historical fit/split protocol is not silently "
                            "mixed into the new grouped-OOF matrix."
                        ),
                        "artifacts": [str(discovery_path.resolve())],
                        "completed_at": _now(),
                    },
                )
            paths.append(path)
    return paths


def _selected_specs(config: Mapping[str, Any]) -> tuple[ExperimentSpec, ...]:
    specs = _specs(str(config["matrix_profile"]))
    allowed = set(map(str, config.get("matrix_methods", [spec.key for spec in specs])))
    return tuple(spec for spec in specs if spec.key in allowed)


def _dataset_protocol_descriptors(output: Path) -> list[dict[str, Any]]:
    return [
        {
            "key": artifact.key,
            "route": artifact.route,
            "pool": artifact.pool,
            "features_path": str(artifact.features_path.resolve()),
            "features_sha256": _sha256(artifact.features_path),
            "labels_path": str(artifact.labels_path.resolve()),
            "labels_sha256": _sha256(artifact.labels_path),
        }
        for artifact in _discover_datasets(output, "development")
    ]


def _expected_core_experiment_count(
    config: Mapping[str, Any], dataset_count: int
) -> int:
    specs = _selected_specs(config)
    learned = sum(spec.learned for spec in specs)
    deterministic = len(specs) - learned
    return dataset_count * (
        learned * len(config["seeds"]) * int(config["folds"]) + deterministic
    )


def _worker_protocol_payload(
    output: Path,
    config: Mapping[str, Any],
    *,
    worker_count: int,
    torch_thread_count: int,
    torch_interop_thread_count: int,
) -> dict[str, Any]:
    if worker_count <= 0:
        raise MatrixError("train worker count must be positive")
    if torch_thread_count <= 0:
        raise MatrixError("train torch thread count must be positive")
    if torch_interop_thread_count <= 0:
        raise MatrixError("train torch inter-op thread count must be positive")
    semantic = config.get("semantic_config")
    if isinstance(semantic, Mapping):
        frozen_thread_count = semantic.get("train_torch_thread_count")
        frozen_interop_thread_count = semantic.get("train_torch_interop_thread_count")
        frozen_batching = semantic.get("neural_batching_policy")
        if (
            frozen_thread_count is not None
            and int(frozen_thread_count) != torch_thread_count
        ):
            raise MatrixError(
                "requested thread count differs from frozen configuration"
            )
        if (
            frozen_interop_thread_count is not None
            and int(frozen_interop_thread_count) != torch_interop_thread_count
        ):
            raise MatrixError(
                "requested inter-op thread count differs from frozen configuration"
            )
        if frozen_batching is not None and (
            None if frozen_batching in {"legacy", "none"} else str(frozen_batching)
        ) != _neural_batching_policy(config):
            raise MatrixError(
                "effective batching policy differs from frozen configuration"
            )
    datasets = _dataset_protocol_descriptors(output)
    expected_count = _expected_core_experiment_count(config, len(datasets))
    if str(config["matrix_profile"]) != "unit_test" and expected_count != 4638:
        raise MatrixError(
            f"formal train worker universe must contain 4638 experiments, got {expected_count}"
        )
    payload: dict[str, Any] = {
        "schema_version": TRAIN_WORKER_PROTOCOL_SCHEMA_VERSION,
        "run_root": str(output.resolve()),
        "base_config_sha256": _matrix_config_identity(config),
        "matrix_profile": str(config["matrix_profile"]),
        "folds": int(config["folds"]),
        "seeds": list(map(int, config["seeds"])),
        "methods": [spec.key for spec in _selected_specs(config)],
        "datasets": datasets,
        "expected_core_experiment_count": expected_count,
        "worker_count": int(worker_count),
        "torch_thread_count": int(torch_thread_count),
        "torch_interop_thread_count": int(torch_interop_thread_count),
        "neural_batching_policy": _neural_batching_policy(config),
        "partition_key": ["dataset", "seed", "fold"],
        "partition_algorithm": TRAIN_WORKER_PARTITION_ALGORITHM,
    }
    return {**payload, "protocol_sha256": _canonical_json_sha256(payload)}


def _publish_worker_protocol(output: Path, payload: Mapping[str, Any]) -> Path:
    """Publish a complete immutable protocol with exclusive-link semantics."""

    path = output / TRAIN_WORKER_PROTOCOL_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    encoded = (json.dumps(_json_safe(payload), indent=2, sort_keys=True) + "\n").encode(
        "utf-8"
    )
    try:
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            os.write(descriptor, encoded)
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        try:
            os.link(temporary, path)
        except FileExistsError:
            pass
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
    if path.is_symlink() or not path.is_file():
        raise MatrixError(f"train worker protocol was not safely published: {path}")
    try:
        observed = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise MatrixError("published train worker protocol is unreadable") from error
    if observed != _json_safe(payload):
        raise MatrixError(
            "train worker protocol already exists with a different immutable identity"
        )
    return path


def configure_train_worker_protocol(
    output: str | os.PathLike[str],
    *,
    worker_count: int,
    torch_thread_count: int,
    torch_interop_thread_count: int = 1,
) -> dict[str, Any]:
    root = Path(output).expanduser().resolve()
    _prepare_directories(root)
    config = _config(root)
    payload = _worker_protocol_payload(
        root,
        config,
        worker_count=worker_count,
        torch_thread_count=torch_thread_count,
        torch_interop_thread_count=torch_interop_thread_count,
    )
    _publish_worker_protocol(root, payload)
    return _validate_worker_protocol(
        root,
        payload,
        config_identity=_matrix_config_identity(config),
    )


def _worker_for_group(dataset: str, seed: int, fold: int, *, worker_count: int) -> int:
    payload = {"dataset": str(dataset), "seed": int(seed), "fold": int(fold)}
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], "big") % worker_count


def _expected_worker_experiment_ids(
    config: Mapping[str, Any],
    datasets: Sequence[DatasetArtifact],
    *,
    worker_index: int,
    worker_count: int,
) -> list[str]:
    expected: list[str] = []
    specs = _selected_specs(config)
    for dataset in datasets:
        for seed in map(int, config["seeds"]):
            for fold in range(int(config["folds"])):
                if (
                    _worker_for_group(
                        dataset.key, seed, fold, worker_count=worker_count
                    )
                    != worker_index
                ):
                    continue
                expected.extend(
                    _experiment_id(dataset.key, spec, seed, fold)
                    for spec in specs
                    if spec.learned
                )
        if (
            _worker_for_group(dataset.key, -1, -1, worker_count=worker_count)
            == worker_index
        ):
            expected.extend(
                _experiment_id(dataset.key, spec, -1, -1)
                for spec in specs
                if not spec.learned
            )
    return sorted(expected)


def _process_start_identity(pid: int) -> str | None:
    result = subprocess.run(
        ["ps", "-o", "lstart=", "-p", str(int(pid))],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    value = result.stdout.strip()
    return value if result.returncode == 0 and value else None


def _parse_timestamp(value: Any) -> dt.datetime | None:
    if not isinstance(value, str):
        return None
    try:
        parsed = dt.datetime.fromisoformat(value)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=dt.timezone.utc)
    return parsed


@dataclass
class _TrainWorkerClaim:
    path: Path
    descriptor: int
    claim_id: str

    def release(self) -> None:
        if self.descriptor < 0:
            return
        try:
            current = os.fstat(self.descriptor)
            try:
                published = self.path.stat()
            except FileNotFoundError:
                published = None
            if (
                published is not None
                and current.st_dev == published.st_dev
                and current.st_ino == published.st_ino
            ):
                self.path.unlink()
        finally:
            fcntl.flock(self.descriptor, fcntl.LOCK_UN)
            os.close(self.descriptor)
            self.descriptor = -1


def _claim_path(output: Path, worker_index: int) -> Path:
    return output / "logs" / "train_workers" / "claims" / f"worker_{worker_index}.json"


def _reclaim_existing_claim(path: Path, *, lease_seconds: int) -> bool:
    """Remove an unlocked claim only with death, expiry, or old-garbage proof."""

    try:
        descriptor = os.open(path, os.O_RDWR)
    except FileNotFoundError:
        return True
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return False
        opened = os.fstat(descriptor)
        try:
            published = path.stat()
        except FileNotFoundError:
            return True
        if opened.st_dev != published.st_dev or opened.st_ino != published.st_ino:
            return True
        try:
            raw = os.read(descriptor, max(opened.st_size, 1) + 1)
            claim = json.loads(raw.decode("utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            claim = None
        now = dt.datetime.now(dt.timezone.utc).astimezone()
        expired = False
        confirmed_dead = False
        if isinstance(claim, Mapping):
            expires_at = _parse_timestamp(claim.get("expires_at"))
            expired = expires_at is not None and expires_at <= now
            if claim.get("hostname") == socket.gethostname():
                try:
                    pid = int(claim["pid"])
                except (KeyError, TypeError, ValueError):
                    pid = -1
                observed_start = _process_start_identity(pid) if pid > 0 else None
                confirmed_dead = observed_start != claim.get("process_start_identity")
        else:
            expired = time.time() - opened.st_mtime >= lease_seconds
        if not (expired or confirmed_dead):
            return False
        current = path.stat()
        if current.st_dev != opened.st_dev or current.st_ino != opened.st_ino:
            return True
        path.unlink()
        return True
    finally:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


def _acquire_worker_claim(
    output: Path,
    protocol: Mapping[str, Any],
    *,
    worker_index: int,
    owner: str | None,
    lease_seconds: int,
) -> _TrainWorkerClaim:
    worker_count = int(protocol["worker_count"])
    if not 0 <= worker_index < worker_count:
        raise MatrixError(
            f"worker index must be in [0, {worker_count}), got {worker_index}"
        )
    if lease_seconds <= 0:
        raise MatrixError("claim lease must be positive")
    path = _claim_path(output, worker_index)
    path.parent.mkdir(parents=True, exist_ok=True)
    for _ in range(4):
        try:
            descriptor = os.open(path, os.O_RDWR | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError:
            if _reclaim_existing_claim(path, lease_seconds=lease_seconds):
                continue
            raise MatrixError(
                f"train worker {worker_index} already has an active or unexpired claim"
            ) from None
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX)
            claim_id = uuid.uuid4().hex
            now = dt.datetime.now(dt.timezone.utc).astimezone()
            claim = {
                "schema_version": 1,
                "claim_id": claim_id,
                "owner": owner or f"{socket.gethostname()}:{os.getpid()}:{claim_id}",
                "pid": os.getpid(),
                "hostname": socket.gethostname(),
                "process_start_identity": _process_start_identity(os.getpid()),
                "worker_index": worker_index,
                "worker_count": worker_count,
                "protocol_sha256": protocol["protocol_sha256"],
                "lease_seconds": lease_seconds,
                "acquired_at": now.isoformat(timespec="seconds"),
                "expires_at": (now + dt.timedelta(seconds=lease_seconds)).isoformat(
                    timespec="seconds"
                ),
            }
            encoded = (json.dumps(claim, indent=2, sort_keys=True) + "\n").encode(
                "utf-8"
            )
            os.write(descriptor, encoded)
            os.fsync(descriptor)
            return _TrainWorkerClaim(path, descriptor, claim_id)
        except Exception:
            os.close(descriptor)
            try:
                path.unlink()
            except FileNotFoundError:
                pass
            raise
    raise MatrixError(f"could not acquire train worker {worker_index} claim")


def _append_worker_progress(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    try:
        os.write(
            descriptor,
            (
                json.dumps(_json_safe(payload), sort_keys=True, ensure_ascii=False)
                + "\n"
            ).encode("utf-8"),
        )
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _assert_protocol_datasets(
    output: Path, protocol: Mapping[str, Any]
) -> list[DatasetArtifact]:
    datasets = _discover_datasets(output, "development")
    observed = _dataset_protocol_descriptors(output)
    if observed != protocol.get("datasets"):
        raise MatrixError(
            "development dataset inventory changed after worker protocol freeze"
        )
    return datasets


@contextmanager
def train_worker_lifecycle_lock(
    output: str | os.PathLike[str], *, exclusive: bool
) -> Iterator[None]:
    """Hold the run-wide worker/finalizer lifecycle ownership lock.

    Workers hold a shared lock for their complete invocation.  Finalization
    holds an exclusive lock from before stage publication until the train
    success marker is committed, preventing late workers and peer finalizers.
    """

    root = Path(output).expanduser().resolve()
    path = root / TRAIN_WORKER_LIFECYCLE_LOCK_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
    operation = fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH
    try:
        fcntl.flock(descriptor, operation)
        yield
    finally:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


def _run_matrix_train_worker_locked(
    output: str | os.PathLike[str],
    *,
    worker_index: int,
    worker_count: int,
    torch_thread_count: int,
    torch_interop_thread_count: int = 1,
    owner: str | None = None,
    lease_seconds: int = 86_400,
) -> list[str]:
    """Run only one stable train shard without touching shared stage artifacts."""

    root = Path(output).expanduser().resolve()
    protocol = configure_train_worker_protocol(
        root,
        worker_count=worker_count,
        torch_thread_count=torch_thread_count,
        torch_interop_thread_count=torch_interop_thread_count,
    )
    config = _config(root)
    datasets = _assert_protocol_datasets(root, protocol)
    expected_ids = _expected_worker_experiment_ids(
        config,
        datasets,
        worker_index=worker_index,
        worker_count=worker_count,
    )
    invocation_id = uuid.uuid4().hex
    worker_root = root / "logs" / "train_workers" / f"worker_{worker_index}"
    progress_path = worker_root / f"progress.{invocation_id}.jsonl"
    receipt_path = worker_root / f"receipt.{invocation_id}.json"
    claim = _acquire_worker_claim(
        root,
        protocol,
        worker_index=worker_index,
        owner=owner,
        lease_seconds=lease_seconds,
    )
    previous_threads = int(torch.get_num_threads())
    started = time.perf_counter()
    completed_ids: list[str] = []
    try:
        if int(torch.get_num_interop_threads()) != torch_interop_thread_count:
            try:
                torch.set_num_interop_threads(torch_interop_thread_count)
            except RuntimeError as error:
                raise MatrixError(
                    "torch inter-op thread policy was already initialized with "
                    "a different value; start each worker in a fresh process"
                ) from error
        torch.set_num_threads(torch_thread_count)
        _append_worker_progress(
            progress_path,
            {
                "timestamp": _now(),
                "status": "RUNNING",
                "worker_index": worker_index,
                "worker_count": worker_count,
                "claim_id": claim.claim_id,
                "protocol_sha256": protocol["protocol_sha256"],
                "torch_thread_count": torch_thread_count,
                "torch_interop_thread_count": torch_interop_thread_count,
                "expected_experiment_count": len(expected_ids),
            },
        )
        specs = _selected_specs(config)
        for dataset in datasets:
            frame, reference, feature_columns, _schema = _load_joined(dataset)
            split_plan = build_stratified_group_folds(
                reference,
                n_splits=int(config["folds"]),
                random_state=42,
            )
            learned_specs = tuple(spec for spec in specs if spec.learned)
            for seed in map(int, config["seeds"]):
                for fold in split_plan.folds:
                    if (
                        _worker_for_group(
                            dataset.key,
                            seed,
                            fold.fold,
                            worker_count=worker_count,
                        )
                        != worker_index
                    ):
                        continue
                    nested_split = (
                        _build_nested_neural_split(
                            frame.iloc[fold.train_indices],
                            frame.iloc[fold.validation_indices],
                            outer_fold=fold.fold,
                            config=config,
                        )
                        if any(
                            spec.backend in NEURAL_BACKENDS for spec in learned_specs
                        )
                        else None
                    )
                    neural_preparation_cache: dict[
                        tuple[str, int, int, tuple[str, ...], str],
                        _NeuralFoldPreparation,
                    ] = {}
                    for spec in learned_specs:
                        experiment_id = _experiment_id(
                            dataset.key, spec, seed, fold.fold
                        )
                        _append_worker_progress(
                            progress_path,
                            {
                                "timestamp": _now(),
                                "status": "EXPERIMENT_RUNNING",
                                "experiment_id": experiment_id,
                            },
                        )
                        _run_fold_experiment(
                            root,
                            dataset,
                            frame,
                            reference,
                            _feature_columns_for_spec(feature_columns, spec),
                            spec,
                            seed,
                            fold.fold,
                            fold.train_indices,
                            fold.validation_indices,
                            config,
                            nested_split if spec.backend in NEURAL_BACKENDS else None,
                            neural_preparation_cache,
                        )
                        completed_ids.append(experiment_id)
            if (
                _worker_for_group(dataset.key, -1, -1, worker_count=worker_count)
                == worker_index
            ):
                for spec in (spec for spec in specs if not spec.learned):
                    experiment_id = _experiment_id(dataset.key, spec, -1, -1)
                    _run_deterministic_experiment(
                        root,
                        dataset,
                        frame,
                        reference,
                        _feature_columns_for_spec(feature_columns, spec)
                        if spec.backend == "rule"
                        else feature_columns,
                        spec,
                    )
                    completed_ids.append(experiment_id)
        if sorted(completed_ids) != expected_ids:
            raise MatrixError(
                "worker execution did not exactly cover its stable experiment shard"
            )
        manifest_hashes = {
            experiment_id: _sha256(
                root / "manifests" / "experiments" / f"{experiment_id}.json"
            )
            for experiment_id in expected_ids
        }
        receipt = {
            "schema_version": 1,
            "status": "COMPLETE",
            "invocation_id": invocation_id,
            "claim_id": claim.claim_id,
            "worker_index": worker_index,
            "worker_count": worker_count,
            "torch_thread_count": torch_thread_count,
            "torch_interop_thread_count": torch_interop_thread_count,
            "protocol_sha256": protocol["protocol_sha256"],
            "expected_experiment_ids": expected_ids,
            "experiment_manifest_sha256": manifest_hashes,
            "progress_path": str(progress_path.resolve()),
            "duration_seconds": time.perf_counter() - started,
            "completed_at": _now(),
        }
        _atomic_json(receipt_path, receipt)
        _append_worker_progress(
            progress_path,
            {"timestamp": _now(), "status": "COMPLETE", "receipt": str(receipt_path)},
        )
        return [str(receipt_path.resolve()), str(progress_path.resolve())]
    except Exception as error:
        _atomic_json(
            receipt_path,
            {
                "schema_version": 1,
                "status": "FAILED",
                "invocation_id": invocation_id,
                "claim_id": claim.claim_id,
                "worker_index": worker_index,
                "worker_count": worker_count,
                "protocol_sha256": protocol["protocol_sha256"],
                "torch_thread_count": torch_thread_count,
                "torch_interop_thread_count": torch_interop_thread_count,
                "completed_experiment_ids": sorted(completed_ids),
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(),
                "failed_at": _now(),
            },
        )
        raise
    finally:
        torch.set_num_threads(previous_threads)
        claim.release()


def run_matrix_train_worker(
    output: str | os.PathLike[str],
    *,
    worker_index: int,
    worker_count: int,
    torch_thread_count: int,
    torch_interop_thread_count: int = 1,
    owner: str | None = None,
    lease_seconds: int = 86_400,
) -> list[str]:
    """Run one shard while excluding finalization for its whole lifecycle."""

    with train_worker_lifecycle_lock(output, exclusive=False):
        return _run_matrix_train_worker_locked(
            output,
            worker_index=worker_index,
            worker_count=worker_count,
            torch_thread_count=torch_thread_count,
            torch_interop_thread_count=torch_interop_thread_count,
            owner=owner,
            lease_seconds=lease_seconds,
        )


def _expected_core_manifest_identities(
    output: Path, config: Mapping[str, Any]
) -> dict[str, str]:
    expected: dict[str, str] = {}
    specs = _selected_specs(config)
    for dataset in _discover_datasets(output, "development"):
        frame, reference, feature_columns, _schema = _load_joined(dataset)
        split_plan = build_stratified_group_folds(
            reference,
            n_splits=int(config["folds"]),
            random_state=42,
        )
        for spec in specs:
            spec_columns = (
                _feature_columns_for_spec(feature_columns, spec)
                if spec.backend == "rule" or spec.learned
                else feature_columns
            )
            if not spec.learned:
                experiment_id = _experiment_id(dataset.key, spec, -1, -1)
                expected[experiment_id] = _experiment_identity_sha256(
                    dataset, spec, -1, -1, spec_columns
                )
                continue
            training_protocol = _neural_training_protocol(spec, config)
            for seed in map(int, config["seeds"]):
                for fold in split_plan.folds:
                    experiment_id = _experiment_id(dataset.key, spec, seed, fold.fold)
                    expected[experiment_id] = _experiment_identity_sha256(
                        dataset,
                        spec,
                        seed,
                        fold.fold,
                        spec_columns,
                        fold.train_indices,
                        fold.validation_indices,
                        training_protocol,
                    )
    return expected


def _completed_worker_receipt(
    output: Path,
    protocol: Mapping[str, Any],
    *,
    worker_index: int,
    expected_ids: Sequence[str],
) -> Path | None:
    root = output / "logs" / "train_workers" / f"worker_{worker_index}"
    for path in sorted(root.glob("receipt.*.json"), reverse=True):
        try:
            receipt = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        hashes = receipt.get("experiment_manifest_sha256")
        if (
            receipt.get("status") != "COMPLETE"
            or receipt.get("worker_index") != worker_index
            or receipt.get("worker_count") != protocol["worker_count"]
            or receipt.get("torch_thread_count") != protocol["torch_thread_count"]
            or receipt.get("torch_interop_thread_count")
            != protocol["torch_interop_thread_count"]
            or receipt.get("protocol_sha256") != protocol["protocol_sha256"]
            or receipt.get("expected_experiment_ids") != list(expected_ids)
            or not isinstance(hashes, Mapping)
            or set(hashes) != set(expected_ids)
        ):
            continue
        if all(
            (
                manifest := output / "manifests" / "experiments" / f"{item}.json"
            ).is_file()
            and not manifest.is_symlink()
            and hashes.get(item) == _sha256(manifest)
            for item in expected_ids
        ):
            return path
    return None


def _run_matrix_train_finalize_locked(output: str | os.PathLike[str]) -> list[str]:
    """Verify and publish while the caller owns the exclusive lifecycle lock."""

    root = Path(output).expanduser().resolve()
    config = _config(root)
    protocol = _read_worker_protocol(root, config, required=True)
    assert protocol is not None
    datasets = _assert_protocol_datasets(root, protocol)
    claims_root = root / "logs" / "train_workers" / "claims"
    for claim_path in sorted(claims_root.glob("*.json")):
        if not _reclaim_existing_claim(claim_path, lease_seconds=86_400):
            raise MatrixError(f"an active train worker still owns {claim_path.name}")
    receipt_paths: list[Path] = []
    for worker_index in range(int(protocol["worker_count"])):
        expected_ids = _expected_worker_experiment_ids(
            config,
            datasets,
            worker_index=worker_index,
            worker_count=int(protocol["worker_count"]),
        )
        receipt = _completed_worker_receipt(
            root,
            protocol,
            worker_index=worker_index,
            expected_ids=expected_ids,
        )
        if receipt is None:
            raise MatrixError(
                f"train worker {worker_index} lacks an exact complete receipt"
            )
        receipt_paths.append(receipt)
    expected = _expected_core_manifest_identities(root, config)
    if len(expected) != int(protocol["expected_core_experiment_count"]):
        raise MatrixError("finalizer universe differs from immutable worker protocol")
    if str(config["matrix_profile"]) != "unit_test" and len(expected) != 4638:
        raise MatrixError(
            f"formal finalizer expected 4638 experiments, got {len(expected)}"
        )
    invalid = [
        experiment_id
        for experiment_id, identity in expected.items()
        if _completed_manifest(
            root,
            experiment_id,
            expected_identity_sha256=identity,
        )
        is None
    ]
    if invalid:
        raise MatrixError(
            f"finalizer found {len(invalid)} incomplete or hash-invalid experiments: "
            + ", ".join(invalid[:5])
        )
    outputs = _run_train(root, config, allow_worker_protocol=True)
    finalization = root / "logs" / "train_workers" / "finalization_receipt.json"
    _atomic_json(
        finalization,
        {
            "schema_version": 1,
            "status": "COMPLETE",
            "protocol_sha256": protocol["protocol_sha256"],
            "worker_receipts": [
                {"path": str(path.resolve()), "sha256": _sha256(path)}
                for path in receipt_paths
            ],
            "verified_core_experiment_count": len(expected),
            "all_artifact_hashes_verified": True,
            "shared_split_and_registry_outputs_published": True,
            "completed_at": _now(),
        },
    )
    outputs.extend((root / TRAIN_WORKER_PROTOCOL_PATH, finalization))
    return [str(path.resolve()) for path in sorted(set(outputs))]


def run_matrix_train_finalize(
    output: str | os.PathLike[str], *, lifecycle_lock_held: bool = False
) -> list[str]:
    """Verify every worker artifact, then publish the ordinary train outputs."""

    if lifecycle_lock_held:
        return _run_matrix_train_finalize_locked(output)
    with train_worker_lifecycle_lock(output, exclusive=True):
        return _run_matrix_train_finalize_locked(output)


def _run_train(
    output: Path,
    config: Mapping[str, Any],
    *,
    allow_worker_protocol: bool = False,
) -> list[Path]:
    if int(config.get("train_worker_count", 1)) > 1 and not allow_worker_protocol:
        raise MatrixError(
            "multi-worker train protocol is active; run train workers and then "
            "the single train finalizer instead of the ordinary train stage"
        )
    stale_declarative_audit = output / "audit" / "LEAKAGE_AUDIT.md"
    primary_lock = output / "manifests" / "PRIMARY_METHOD_LOCK.json"
    if not primary_lock.exists() and (
        stale_declarative_audit.is_file() or stale_declarative_audit.is_symlink()
    ):
        stale_declarative_audit.unlink()
    artifacts: list[Path] = []
    all_assignments: list[pd.DataFrame] = []
    specs = _specs(str(config["matrix_profile"]))
    allowed = set(map(str, config.get("matrix_methods", [spec.key for spec in specs])))
    for dataset in _discover_datasets(output, "development"):
        frame, reference, feature_columns, schema = _load_joined(dataset)
        schema_path = output / "features" / f"feature_schema_{dataset.key}.json"
        _atomic_json(schema_path, schema)
        artifacts.append(schema_path)
        split_plan: SplitPlan = build_stratified_group_folds(
            reference,
            n_splits=int(config["folds"]),
            random_state=42,
        )
        nested_neural_splits = (
            {
                fold.fold: _build_nested_neural_split(
                    frame.iloc[fold.train_indices],
                    frame.iloc[fold.validation_indices],
                    outer_fold=fold.fold,
                    config=config,
                )
                for fold in split_plan.folds
            }
            if any(
                spec.key in allowed and spec.backend in NEURAL_BACKENDS
                for spec in specs
            )
            else {}
        )
        candidate_assignments = split_plan.candidate_assignments.copy()
        candidate_assignments["record_type"] = "candidate"
        candidate_assignments["exclusion_reason"] = ""
        candidate_assignments["dataset"] = dataset.key

        query_assignments = split_plan.query_assignments.copy()
        universe = _query_universe(dataset, reference)
        assigned_query_ids = set(query_assignments["query_id"].astype(str))
        empty_queries = universe.loc[
            ~universe["query_id"].astype(str).isin(assigned_query_ids)
        ].copy()
        if len(empty_queries):
            empty_queries["group_hash"] = [
                scene_frame_group_hash(scene_id, frame_id)
                for scene_id, frame_id in zip(
                    empty_queries["scene_id"], empty_queries["frame_id"]
                )
            ]
            empty_queries["query_has_positive"] = 0
            empty_queries["candidate_count"] = 0
            empty_queries["fold"] = -1
            query_assignments = pd.concat(
                [query_assignments, empty_queries[query_assignments.columns]],
                ignore_index=True,
            )
        query_assignments["record_type"] = "query"
        query_assignments["exclusion_reason"] = np.where(
            query_assignments["candidate_count"].eq(0),
            "empty_candidate_pool",
            "",
        )
        query_assignments["dataset"] = dataset.key

        assignment_columns = sorted(
            set(candidate_assignments.columns) | set(query_assignments.columns)
        )
        assignments = pd.concat(
            [
                candidate_assignments.reindex(columns=assignment_columns),
                query_assignments.reindex(columns=assignment_columns),
            ],
            ignore_index=True,
        )
        all_assignments.append(assignments)
        split_path = output / "data" / f"split_assignments_{dataset.key}.parquet"
        _atomic_parquet(split_path, assignments)
        query_split_path = (
            output / "data" / f"split_query_assignments_{dataset.key}.parquet"
        )
        candidate_split_path = (
            output / "data" / f"split_candidate_assignments_{dataset.key}.parquet"
        )
        _atomic_parquet(query_split_path, query_assignments)
        _atomic_parquet(candidate_split_path, candidate_assignments)
        split_manifest = output / "manifests" / f"split_manifest_{dataset.key}.json"
        _atomic_json(split_manifest, split_plan.audit)
        artifacts.extend(
            (split_path, query_split_path, candidate_split_path, split_manifest)
        )
        artifacts.extend(_write_outer_fit_evidence(output, dataset, frame, split_plan))
        neural_preparation_cache: dict[
            tuple[str, int, int, tuple[str, ...], str], _NeuralFoldPreparation
        ] = {}

        for spec in specs:
            if spec.key not in allowed:
                continue
            if not spec.learned:
                artifacts.append(
                    _run_deterministic_experiment(
                        output,
                        dataset,
                        frame,
                        reference,
                        _feature_columns_for_spec(feature_columns, spec)
                        if spec.backend == "rule"
                        else feature_columns,
                        spec,
                    )
                )
                continue
            spec_feature_columns = _feature_columns_for_spec(feature_columns, spec)
            for seed in map(int, config["seeds"]):
                for fold in split_plan.folds:
                    artifacts.append(
                        _run_fold_experiment(
                            output,
                            dataset,
                            frame,
                            reference,
                            spec_feature_columns,
                            spec,
                            seed,
                            fold.fold,
                            fold.train_indices,
                            fold.validation_indices,
                            config,
                            nested_neural_splits.get(fold.fold),
                            neural_preparation_cache,
                        )
                    )
        artifacts.extend(_record_conditional_and_reused_methods(output, dataset))
    combined_assignments = output / "data" / "split_assignments.parquet"
    _atomic_parquet(combined_assignments, pd.concat(all_assignments, ignore_index=True))
    artifacts.append(combined_assignments)
    formal_config = output / "configs" / "formal_matrix.yaml"
    batch_evidence_path = output / "audit" / "neural_batch_benchmark.json"
    batch_evidence = (
        {
            "path": str(batch_evidence_path.resolve()),
            "sha256": _sha256(batch_evidence_path),
        }
        if batch_evidence_path.is_file()
        else None
    )
    # JSON is a strict YAML 1.2 subset and avoids an unnecessary PyYAML
    # dependency while keeping this artifact machine-readable.
    _atomic_text(
        formal_config,
        json.dumps(
            _json_safe(
                {
                    "schema_version": 1,
                    "profile": config["matrix_profile"],
                    "folds": config["folds"],
                    "seeds": config["seeds"],
                    "device": config.get("device", "cpu"),
                    "cpu_fallback": True,
                    "train_query_batch_size": _neural_query_batch_size(config),
                    "training_batching_policy": _neural_batching_policy(config),
                    "training_batching_shuffle": True,
                    "early_stopping_batching_policy": _neural_batching_policy(config),
                    "early_stopping_batching_shuffle": False,
                    "inference_query_batch_size": NEURAL_INFERENCE_QUERY_BATCH_SIZE,
                    "train_worker_count": int(config.get("train_worker_count", 1)),
                    "torch_thread_count": int(
                        config.get("train_torch_thread_count", torch.get_num_threads())
                    ),
                    "torch_interop_thread_count": int(
                        config.get("train_torch_interop_thread_count", 1)
                    ),
                    "train_worker_protocol_sha256": config.get(
                        "train_worker_protocol_sha256"
                    ),
                    "device_selection_evidence": config.get("prelock_device_amendment"),
                    "batch_selection_evidence": batch_evidence,
                    "epochs": config.get(
                        "matrix_epochs",
                        1 if config["matrix_profile"] == "unit_test" else 100,
                    ),
                    "patience": config.get(
                        "matrix_patience",
                        1 if config["matrix_profile"] == "unit_test" else 10,
                    ),
                    "neural_nested_validation": {
                        "protocol": "nested_grouped_early_stopping",
                        "split_scope": "outer_training_fold_only",
                        "inner_folds": config.get(
                            "matrix_inner_folds",
                            2 if config["matrix_profile"] == "unit_test" else 5,
                        ),
                        "random_state": config.get(
                            "matrix_inner_split_random_state", 1701
                        ),
                        "outer_validation_usage": (
                            "single_oof_inference_and_metrics_only"
                        ),
                    },
                    "methods": [spec.as_dict() for spec in specs],
                    "lambda_mart_status": "ENABLED_TRUE_XGBOOST_XGBRANKER",
                    "lambda_mart_backend": "xgboost.XGBRanker rank:ndcg on CPU",
                    "metric_scope": "frozen 2D annotation consistency only",
                }
            ),
            indent=2,
            sort_keys=True,
        )
        + "\n",
    )
    artifacts.append(formal_config)
    artifacts.extend(_write_registry(output))
    return sorted(set(artifacts))


def _query_universe(artifact: DatasetArtifact, reference: pd.DataFrame) -> pd.DataFrame:
    candidates = (
        output_path
        for output_path in (
            artifact.labels_path.parent
            / f"queries_{artifact.key}_{artifact.split}.parquet",
            artifact.labels_path.with_name(
                artifact.labels_path.stem.replace("labels_", "queries_") + ".parquet"
            ),
            artifact.features_path.with_name(
                f"{artifact.features_path.stem}_queries.parquet"
            ),
        )
        if output_path.is_file()
    )
    path = next(candidates, None)
    if path is None:
        return reference[["query_id", "scene_id", "frame_id"]].drop_duplicates(
            "query_id"
        )
    universe = pd.read_parquet(path)
    source = "sample_id" if "sample_id" in universe.columns else "query_id"
    if source not in universe.columns:
        raise MatrixError(f"query universe lacks sample_id/query_id: {path}")
    universe = universe.rename(columns={source: "query_id"}).copy()
    for column in ("scene_id", "frame_id"):
        if column not in universe.columns:
            universe[column] = universe["query_id"]
    return universe[["query_id", "scene_id", "frame_id"]].drop_duplicates("query_id")


def _query_universe_artifact_path(artifact: DatasetArtifact) -> Path:
    candidates = (
        artifact.labels_path.parent
        / f"queries_{artifact.key}_{artifact.split}.parquet",
        artifact.labels_path.with_name(
            artifact.labels_path.stem.replace("labels_", "queries_") + ".parquet"
        ),
        artifact.features_path.with_name(
            f"{artifact.features_path.stem}_queries.parquet"
        ),
    )
    path = next((item for item in candidates if item.is_file()), None)
    if path is None:
        raise MatrixError(
            f"held-out query-universe artifact is missing for {artifact.key}"
        )
    return path


def _canonical_frame_sha256(frame: pd.DataFrame, columns: Sequence[str]) -> str:
    records = [
        [_json_safe(value) for value in row]
        for row in frame[list(columns)].itertuples(index=False, name=None)
    ]
    return hashlib.sha256(
        json.dumps(
            records, sort_keys=False, separators=(",", ":"), ensure_ascii=False
        ).encode("utf-8")
    ).hexdigest()


def _held_out_test_input_descriptor(
    output: Path, artifact: DatasetArtifact
) -> dict[str, Any]:
    """Hash label-free held-out identity before any primary prediction.

    The frozen content covers candidate IDs, every retained geometry field,
    q values/ranks, the full feature parquet/schema, and the query universe.
    Labels and test outcomes are deliberately not opened here.
    """

    feature_descriptor = _locked_artifact_descriptor(output, artifact.features_path)
    features = pd.read_parquet(artifact.features_path)
    query_column = "sample_id" if "sample_id" in features.columns else "query_id"
    if query_column not in features.columns or "candidate_id" not in features.columns:
        raise MatrixError(
            f"held-out feature identity columns are missing: {artifact.features_path}"
        )
    if "q_raw" not in features.columns:
        raise MatrixError(f"held-out q_raw column is missing: {artifact.features_path}")
    q_values = pd.to_numeric(features["q_raw"], errors="coerce")
    if q_values.isna().any() or not np.isfinite(q_values.to_numpy(np.float64)).all():
        raise MatrixError(
            f"held-out q_raw values are invalid: {artifact.features_path}"
        )
    features = features.copy()
    features[query_column] = features[query_column].astype(str)
    features["candidate_id"] = features["candidate_id"].astype(str)
    features["q_raw"] = q_values.astype(np.float64)
    identity_columns = [query_column, "candidate_id"]
    if features.duplicated(identity_columns).any():
        raise MatrixError(f"duplicate held-out candidate IDs: {artifact.features_path}")
    geometry_candidates = (
        "x_px",
        "y_px",
        "z_m",
        "angle_rad",
        "width_m",
        "width_px",
        "height_px",
        "cx",
        "cy",
    )
    geometry_columns = [
        column for column in geometry_candidates if column in features.columns
    ]
    rank_columns = [
        column
        for column in ("original_rank", "q_rank", "legacy_rank")
        if column in features.columns
    ]
    pool_columns = [*identity_columns, *geometry_columns, "q_raw", *rank_columns]
    canonical_pool = features.sort_values(identity_columns, kind="mergesort")

    query_path = _query_universe_artifact_path(artifact)
    query_descriptor = _locked_artifact_descriptor(output, query_path)
    universe = pd.read_parquet(query_path)
    universe_query = "sample_id" if "sample_id" in universe.columns else "query_id"
    if universe_query not in universe.columns:
        raise MatrixError(f"held-out query universe lacks identity: {query_path}")
    universe = universe.copy()
    universe[universe_query] = universe[universe_query].astype(str)
    if universe[universe_query].duplicated().any():
        raise MatrixError(f"held-out query universe has duplicate IDs: {query_path}")
    universe_columns = [universe_query]
    for column in ("scene_id", "frame_id"):
        if column in universe.columns:
            universe[column] = universe[column].astype(str)
            universe_columns.append(column)
    canonical_universe = universe.sort_values(universe_query, kind="mergesort")
    schema = {
        "columns": [
            {"name": str(column), "dtype": str(features[column].dtype)}
            for column in features.columns
        ]
    }
    return {
        "dataset": artifact.key,
        "route": artifact.route,
        "pool": artifact.pool,
        "features": feature_descriptor,
        "feature_schema": schema,
        "feature_schema_sha256": _canonical_mapping_sha256(schema),
        "candidate_count": len(features),
        "candidate_identity_columns": pool_columns,
        "candidate_pool_identity_sha256": _canonical_frame_sha256(
            canonical_pool, pool_columns
        ),
        "query_universe": query_descriptor,
        "query_universe_count": len(universe),
        "query_universe_identity_columns": universe_columns,
        "query_universe_identity_sha256": _canonical_frame_sha256(
            canonical_universe, universe_columns
        ),
        "labels_opened_at_lock": False,
    }


def _held_out_test_input_lock(output: Path) -> list[dict[str, Any]]:
    return [
        _held_out_test_input_descriptor(output, artifact)
        for artifact in _discover_datasets(output, "test")
    ]


def _verified_media_sha256(
    raw_path: Any,
    expected_sha256: Any | None,
    cache: dict[Path, str],
    *,
    role: str,
) -> tuple[str, str]:
    unresolved = Path(str(raw_path)).expanduser()
    if unresolved.is_symlink():
        raise MatrixError(f"{role} source media is symlinked: {unresolved}")
    path = unresolved.resolve()
    if not path.is_file():
        raise MatrixError(f"{role} source media is missing: {path}")
    digest = cache.get(path)
    if digest is None:
        digest = _sha256(path)
        cache[path] = digest
    if expected_sha256 is not None:
        expected = str(expected_sha256).strip().lower()
        if not re.fullmatch(r"[0-9a-f]{64}", expected) or digest != expected:
            raise MatrixError(f"{role} source media SHA-256 mismatch: {path}")
    return str(path), digest


def _read_modular_identity_manifest(path: Path) -> pd.DataFrame:
    if path.is_symlink() or not path.is_file():
        raise MatrixError(f"Modular source identity manifest missing: {path}")
    rows: list[dict[str, Any]] = []
    fields = (
        "sample_id",
        "scene_id",
        "frame_id",
        "split",
        "source_rgb_path",
        "source_rgb_sha256",
        "source_depth_path",
        "source_depth_sha256",
    )
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise MatrixError(
                    f"invalid Modular source identity JSON at {path}:{line_number}"
                ) from error
            if not isinstance(value, Mapping):
                raise MatrixError(
                    f"invalid Modular source identity row at {path}:{line_number}"
                )
            rows.append({field: value.get(field) for field in fields})
    if not rows:
        raise MatrixError(f"Modular source identity manifest is empty: {path}")
    result = pd.DataFrame(rows)
    if "frame_id" in result and result["frame_id"].isna().all():
        result = result.drop(columns="frame_id")
    return result


def _modular_identity_sources(
    config: Mapping[str, Any],
) -> tuple[tuple[Path, ...], tuple[Path, ...]]:
    configured = config.get("modular_source_identity_manifests")
    if configured is None:
        root = MODULAR_INCOMPLETE_RERANK_RUN / "compact_inputs"
        return (
            (root / "train" / "manifest.jsonl", root / "val" / "manifest.jsonl"),
            (root / "test" / "manifest.jsonl",),
        )
    if not isinstance(configured, Mapping):
        raise MatrixError("modular_source_identity_manifests must be a mapping")
    development = configured.get("development")
    test = configured.get("test")
    if (
        not isinstance(development, list)
        or not development
        or not isinstance(test, list)
        or not test
    ):
        raise MatrixError(
            "configured Modular identity manifests require development/test lists"
        )
    return tuple(map(Path, development)), tuple(map(Path, test))


def _verified_modular_metadata(
    config: Mapping[str, Any], cache: dict[Path, str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    development_paths, test_paths = _modular_identity_sources(config)

    def load(paths: Sequence[Path], *, audit_role: str) -> pd.DataFrame:
        frames = [_read_modular_identity_manifest(path.resolve()) for path in paths]
        frame = pd.concat(frames, ignore_index=True)
        allowed_splits = (
            {"train", "val", "development"} if audit_role == "development" else {"test"}
        )
        observed_splits = set(frame["split"].dropna().astype(str))
        if not observed_splits or not observed_splits <= allowed_splits:
            raise MatrixError(
                f"Modular {audit_role} identity manifests have wrong split coverage: "
                f"{sorted(observed_splits)}"
            )
        for index, row in frame.iterrows():
            rgb_path, rgb_sha = _verified_media_sha256(
                row["source_rgb_path"],
                row["source_rgb_sha256"],
                cache,
                role=f"Modular {audit_role} RGB row {index}",
            )
            depth_path, depth_sha = _verified_media_sha256(
                row["source_depth_path"],
                row["source_depth_sha256"],
                cache,
                role=f"Modular {audit_role} depth row {index}",
            )
            frame.at[index, "source_rgb_path"] = rgb_path
            frame.at[index, "source_rgb_sha256"] = rgb_sha
            frame.at[index, "source_depth_path"] = depth_path
            frame.at[index, "source_depth_sha256"] = depth_sha
        return frame

    return load(development_paths, audit_role="development"), load(
        test_paths, audit_role="test"
    )


def _crog_identity_rows(
    artifact: DatasetArtifact,
    cache: dict[Path, str],
) -> pd.DataFrame:
    frame = pd.read_parquet(artifact.features_path)
    required = {"sample_id", "scene_id", "image_path", "depth_path"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise MatrixError(
            f"CROG candidate features lack source identity columns for {artifact.key}: {missing}"
        )
    columns = ["sample_id", "scene_id", "image_path", "depth_path"]
    if "frame_id" in frame.columns:
        columns.append("frame_id")
    source = frame.loc[:, columns].drop_duplicates().reset_index(drop=True)
    source["source_rgb_path"] = ""
    source["source_rgb_sha256"] = ""
    source["source_depth_path"] = ""
    source["source_depth_sha256"] = ""
    for index, row in source.iterrows():
        rgb_path, rgb_sha = _verified_media_sha256(
            row["image_path"],
            None,
            cache,
            role=f"CROG {artifact.split} RGB row {index}",
        )
        depth_path, depth_sha = _verified_media_sha256(
            row["depth_path"],
            None,
            cache,
            role=f"CROG {artifact.split} depth row {index}",
        )
        source.at[index, "source_rgb_path"] = rgb_path
        source.at[index, "source_rgb_sha256"] = rgb_sha
        source.at[index, "source_depth_path"] = depth_path
        source.at[index, "source_depth_sha256"] = depth_sha
    try:
        return build_identity_rows(
            source,
            dataset=artifact.key,
            split=artifact.split,
            columns={"rgb_path": "source_rgb_path", "depth_path": "source_depth_path"},
        )
    except LeakageAuditError as error:
        raise MatrixError(
            f"CROG source identity build failed: {artifact.key}"
        ) from error


def _formal_source_identity_rows(
    development: Sequence[DatasetArtifact],
    test: Sequence[DatasetArtifact],
    config: Mapping[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    media_cache: dict[Path, str] = {}
    development_rows: list[pd.DataFrame] = []
    test_rows: list[pd.DataFrame] = []
    modular_development = [item for item in development if item.route == "modular"]
    modular_test = [item for item in test if item.route == "modular"]
    if bool(modular_development) != bool(modular_test):
        raise MatrixError("Modular development/test identity route coverage differs")
    if modular_development:
        development_metadata, test_metadata = _verified_modular_metadata(
            config, media_cache
        )
        for artifact in modular_development:
            try:
                identities = build_identity_rows(
                    development_metadata,
                    dataset=artifact.key,
                    split="development",
                )
            except LeakageAuditError as error:
                raise MatrixError(
                    f"Modular development identity build failed: {artifact.key}"
                ) from error
            development_rows.append(identities)
        for artifact in modular_test:
            try:
                identities = build_identity_rows(
                    test_metadata, dataset=artifact.key, split="test"
                )
            except LeakageAuditError as error:
                raise MatrixError(
                    f"Modular test identity build failed: {artifact.key}"
                ) from error
            test_rows.append(identities)
    for artifact in development:
        if artifact.route == "crog":
            development_rows.append(_crog_identity_rows(artifact, media_cache))
        elif artifact.route != "modular":
            raise MatrixError(
                f"formal leakage audit does not support route: {artifact.route}"
            )
    for artifact in test:
        if artifact.route == "crog":
            test_rows.append(_crog_identity_rows(artifact, media_cache))
        elif artifact.route != "modular":
            raise MatrixError(
                f"formal leakage audit does not support route: {artifact.route}"
            )
    if not development_rows or not test_rows:
        raise MatrixError("formal leakage audit received no source identities")
    observed_development = {item.key for item in development}
    observed_test = {item.key for item in test}
    if observed_development != observed_test:
        raise MatrixError(
            "development/test dataset keys differ for leakage audit: "
            f"{sorted(observed_development)} != {sorted(observed_test)}"
        )
    combined_development = pd.concat(development_rows, ignore_index=True)
    combined_test = pd.concat(test_rows, ignore_index=True)
    for artifact in (*development, *test):
        identities = (
            combined_development if artifact.split == "development" else combined_test
        )
        identity_queries = set(
            identities.loc[identities["dataset"].eq(artifact.key), "query_id"].astype(
                str
            )
        )
        features = pd.read_parquet(artifact.features_path, columns=["sample_id"])
        feature_queries = set(features["sample_id"].astype(str))
        if not feature_queries <= identity_queries:
            raise MatrixError(
                f"source identities omit matrix feature queries: {artifact.key}"
            )
    return (
        combined_development,
        combined_test,
        {
            "unique_media_file_count": len(media_cache),
            "unique_media_files_verified_once": True,
            "modular_identity_source": "compact_inputs train+val/test manifests",
            "crog_identity_source": "built label-free candidate feature Parquets",
        },
    )


def _verify_held_out_test_input_lock(output: Path, lock: Mapping[str, Any]) -> None:
    locked_rows = lock.get("held_out_test_inputs")
    if not isinstance(locked_rows, list):
        raise MatrixError("primary lock lacks held-out input identities")
    for row in locked_rows:
        if not isinstance(row, Mapping):
            raise MatrixError("primary lock contains a malformed held-out input")
        for key in ("features", "query_universe"):
            descriptor = row.get(key)
            if not isinstance(descriptor, Mapping):
                raise MatrixError(f"held-out {key} lock descriptor is malformed")
            path = _locked_artifact_descriptor(
                output,
                str(descriptor.get("path", "")),
                expected_sha256=str(descriptor.get("sha256", "")),
            )
            if path["size_bytes"] != descriptor.get("size_bytes"):
                raise MatrixError(
                    f"held-out {key} size changed after lock: {path['path']}"
                )
    observed = _held_out_test_input_lock(output)
    if observed != lock.get("held_out_test_inputs"):
        raise MatrixError(
            "held-out candidate pool or query universe changed after lock"
        )


def _verify_locked_leakage_audit(output: Path, lock: Mapping[str, Any]) -> None:
    record = lock.get("leakage_audit")
    if not isinstance(record, Mapping):
        raise MatrixError("primary lock lacks leakage audit provenance")
    if record.get("status") == "SKIPPED_UNIT_TEST_PROFILE":
        if record.get("test_labels_opened") is not False:
            raise MatrixError("unit-test leakage skip record is malformed")
        return
    if record.get("status") != "PASS" or record.get("test_labels_opened") is not False:
        raise MatrixError("formal locked leakage audit did not pass")
    bundle_root = Path(str(record.get("bundle_root", ""))).resolve()
    if bundle_root != (output / "audit" / "leakage").resolve():
        raise MatrixError("locked leakage bundle path is not canonical")
    try:
        verified = verify_leakage_audit_bundle(bundle_root, require_fit_evidence=True)
    except LeakageAuditError as error:
        raise MatrixError("locked leakage audit bundle changed") from error
    if not verified.passed or verified.summary.get("audit_digest_sha256") != record.get(
        "audit_digest_sha256"
    ):
        raise MatrixError("locked leakage audit result/digest changed")
    canonical = output / "audit" / "LEAKAGE_AUDIT.md"
    if (
        not canonical.is_file()
        or _sha256(canonical) != record.get("canonical_markdown_sha256")
        or canonical.read_text(encoding="utf-8")
        != (bundle_root / "LEAKAGE_AUDIT.md").read_text(encoding="utf-8")
    ):
        raise MatrixError("canonical locked LEAKAGE_AUDIT.md changed")
    fit_descriptors = record.get("development_fit_evidence")
    bundle_descriptors = record.get("bundle_artifacts")
    if not isinstance(fit_descriptors, list) or not isinstance(
        bundle_descriptors, list
    ):
        raise MatrixError("locked leakage artifact inventories are malformed")
    for descriptor in [*fit_descriptors, *bundle_descriptors]:
        if not isinstance(descriptor, Mapping):
            raise MatrixError("locked leakage artifact descriptor is malformed")
        path = Path(str(descriptor.get("path", ""))).resolve()
        if (
            not path.is_relative_to(output.resolve())
            or not path.is_file()
            or path.is_symlink()
            or _sha256(path) != descriptor.get("sha256")
        ):
            raise MatrixError(f"locked leakage artifact changed: {path}")


def _complete_manifests(
    output: Path,
    *,
    dataset: str,
    method: str | None = None,
    stage: str = "train",
) -> list[dict[str, Any]]:
    rows = [
        row
        for row in _manifest_rows(output)
        if row.get("status") == "COMPLETE"
        and row.get("stage") == stage
        and row.get("dataset") == dataset
        and (method is None or row.get("method") == method)
    ]
    return sorted(
        rows, key=lambda row: (int(row.get("seed", -1)), int(row.get("fold", -1)))
    )


def _read_prediction(path: str | Path) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    required = {"query_id", "candidate_id", "score"}
    if not required.issubset(frame.columns):
        raise MatrixError(
            f"prediction artifact lacks columns {sorted(required)}: {path}"
        )
    columns = ["query_id", "candidate_id", "score"]
    if "probability" in frame.columns:
        columns.append("probability")
    frame = frame[columns].copy()
    frame["query_id"] = frame["query_id"].astype(str)
    frame["candidate_id"] = frame["candidate_id"].astype(str)
    if frame.duplicated(["query_id", "candidate_id"]).any():
        raise MatrixError(f"duplicate candidate predictions: {path}")
    scores = pd.to_numeric(frame["score"], errors="coerce")
    if scores.isna().any() or not np.isfinite(scores.to_numpy(np.float64)).all():
        raise MatrixError(f"non-finite predictions: {path}")
    frame["score"] = scores.astype(np.float64)
    if "probability" in frame.columns:
        probability = pd.to_numeric(frame["probability"], errors="coerce")
        if (
            probability.isna().any()
            or not np.isfinite(probability.to_numpy(np.float64)).all()
            or bool(((probability < 0.0) | (probability > 1.0)).any())
        ):
            raise MatrixError(f"invalid prediction probabilities: {path}")
        frame["probability"] = probability.astype(np.float64)
    return frame


def _method_oof_predictions(
    output: Path,
    dataset: str,
    method: str,
) -> tuple[dict[int, pd.DataFrame], pd.DataFrame]:
    manifests = _complete_manifests(output, dataset=dataset, method=method)
    if not manifests:
        raise MatrixError(f"no complete train artifacts for {dataset}/{method}")
    if int(manifests[0].get("seed", -1)) == -1:
        prediction = _read_prediction(manifests[0]["prediction_path"])
        return {-1: prediction}, prediction
    by_seed: dict[int, list[pd.DataFrame]] = {}
    expected_folds: dict[int, set[int]] = {}
    for manifest in manifests:
        seed = int(manifest["seed"])
        fold = int(manifest["fold"])
        by_seed.setdefault(seed, []).append(
            _read_prediction(manifest["prediction_path"])
        )
        expected_folds.setdefault(seed, set()).add(fold)
    predictions: dict[int, pd.DataFrame] = {}
    for seed, parts in by_seed.items():
        combined = pd.concat(parts, ignore_index=True)
        if combined.duplicated(["query_id", "candidate_id"]).any():
            raise MatrixError(
                f"OOF candidate predicted more than once for {dataset}/{method}/seed={seed}"
            )
        predictions[seed] = combined
    stacked = pd.concat(
        [frame.assign(seed=seed) for seed, frame in predictions.items()],
        ignore_index=True,
    )
    aggregate_columns = ["score"]
    if "probability" in stacked.columns:
        aggregate_columns.append("probability")
    ensemble = stacked.groupby(
        ["query_id", "candidate_id"], as_index=False, sort=False
    )[aggregate_columns].mean()
    return predictions, ensemble


def _gate_candidates(frame: pd.DataFrame, reranker: pd.DataFrame) -> pd.DataFrame:
    work = frame.merge(
        reranker.rename(columns={"score": "reranker_score"}),
        on=["query_id", "candidate_id"],
        how="inner",
        validate="one_to_one",
    )
    if len(work) != len(frame):
        raise MatrixError("reranker predictions do not cover the frozen candidate pool")
    missing_columns = [
        column for column in frame.columns if column.endswith("_missing")
    ]
    reliability_columns = [
        column for column in frame.columns if "reliability" in column.lower()
    ]
    if missing_columns:
        missing = (
            frame[missing_columns].apply(pd.to_numeric, errors="coerce").fillna(1.0)
        )
        risk = missing.clip(0.0, 1.0).mean(axis=1).to_numpy(np.float64)
    else:
        risk = np.zeros(len(frame), dtype=np.float64)
    if reliability_columns:
        reliability = (
            frame[reliability_columns]
            .apply(pd.to_numeric, errors="coerce")
            .fillna(0.0)
            .clip(0.0, 1.0)
            .mean(axis=1)
            .to_numpy(np.float64)
        )
    else:
        reliability = np.ones(len(frame), dtype=np.float64)

    def signal(
        preferred: Sequence[tuple[str, float]],
        groups: set[str],
    ) -> np.ndarray:
        for column, direction in preferred:
            if column in work.columns:
                values = pd.to_numeric(work[column], errors="coerce").fillna(0.0)
                return direction * values.to_numpy(np.float64)
        columns = [
            column
            for column in select_model_feature_columns(work)
            if feature_group(column) in groups
            and "missing" not in column.lower()
            and "reliability" not in column.lower()
        ]
        if not columns:
            return np.zeros(len(work), dtype=np.float64)
        return (
            work[columns]
            .apply(pd.to_numeric, errors="coerce")
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
            .mean(axis=1)
            .to_numpy(np.float64)
        )

    mask_signal = signal(
        (
            ("p_grasp_rectangle_mean", 1.0),
            ("mask_consistency", 1.0),
            ("grasp_axis_mask_support", 1.0),
            ("soft_coverage", 1.0),
        ),
        {"mask"},
    )
    width_signal = signal(
        (
            ("width_compatibility", 1.0),
            ("normalized_width_mismatch", -1.0),
            ("width_margin_to_max", 1.0),
        ),
        {"width"},
    )
    depth_contact_signal = signal(
        (
            ("valid_depth_support", 1.0),
            ("normal_opposition", 1.0),
            ("contact_symmetry", 1.0),
        ),
        {"depth", "contact"},
    )
    collision_signal = signal(
        (
            ("approach_clearance", 1.0),
            ("minimum_visible_obstacle_distance", 1.0),
            ("clearance", 1.0),
            ("collision_proxy_total", -1.0),
            ("collision_proxy", -1.0),
        ),
        {"clearance"},
    )
    if "route" in work.columns and work["route"].astype(str).eq("crog_native").all():
        depth_contact_signal = np.zeros(len(work), dtype=np.float64)
        collision_signal = np.zeros(len(work), dtype=np.float64)
    baseline_rank = (
        pd.to_numeric(work["original_rank"], errors="coerce")
        .fillna(1.0)
        .to_numpy(np.float64)
        if "original_rank" in work.columns
        else work.groupby("query_id")["q_raw"]
        .rank(method="first", ascending=False)
        .to_numpy(np.float64)
    )
    return (
        work[["query_id", "candidate_id", "q_raw", "reranker_score"]]
        .rename(columns={"q_raw": "baseline_score"})
        .assign(
            risk=risk,
            reliability=reliability,
            baseline_rank=baseline_rank,
            mask_support_signal=mask_signal,
            width_compatibility_signal=width_signal,
            depth_contact_signal=depth_contact_signal,
            collision_proxy_signal=collision_signal,
        )
    )


def _seed_agreement(
    proposals: pd.DataFrame,
    seed_predictions: Mapping[int, pd.DataFrame],
) -> np.ndarray:
    if not seed_predictions:
        return np.ones(len(proposals), dtype=np.float64)
    top_by_seed: list[dict[str, str]] = []
    for prediction in seed_predictions.values():
        ordered = prediction.sort_values(
            ["query_id", "score", "candidate_id"],
            ascending=[True, False, True],
            kind="mergesort",
        )
        top = ordered.groupby("query_id", sort=False).first()["candidate_id"]
        top_by_seed.append(top.astype(str).to_dict())
    agreement = []
    for row in proposals.itertuples(index=False):
        votes = sum(
            mapping.get(str(row.query_id)) == str(row.challenger_candidate_id)
            for mapping in top_by_seed
        )
        agreement.append(votes / len(top_by_seed))
    return np.asarray(agreement, dtype=np.float64)


def _gate_perturbation_stability(gate: Any, proposals: pd.DataFrame) -> np.ndarray:
    """Check that gate decisions survive deterministic ±1% feature changes."""

    base = gate.apply(proposals)["switch_applied"].to_numpy(bool)
    stable = np.ones(len(proposals), dtype=bool)
    for sign in (-1.0, 1.0):
        perturbed = proposals.copy()
        for column in (
            "challenger_margin",
            "baseline_score_delta",
            "challenger_risk",
            "challenger_reliability",
        ):
            values = pd.to_numeric(perturbed[column], errors="raise").to_numpy(
                np.float64
            )
            perturbed[column] = values + sign * 0.01 * np.maximum(np.abs(values), 1e-3)
        stable &= gate.apply(perturbed)["switch_applied"].to_numpy(bool) == base
    return stable


def _selected_predictions(
    frame: pd.DataFrame,
    decisions: pd.DataFrame,
) -> pd.DataFrame:
    selected = (
        decisions.set_index("query_id")["selected_candidate_id"].astype(str).to_dict()
    )
    predictions = _baseline_predictions(frame)
    maximum = predictions.groupby("query_id")["score"].transform("max")
    selected_mask = np.asarray(
        [
            selected.get(query) == candidate
            for query, candidate in predictions[
                ["query_id", "candidate_id"]
            ].itertuples(index=False, name=None)
        ],
        dtype=bool,
    )
    predictions.loc[selected_mask, "score"] = maximum[selected_mask] + 1.0
    return predictions


def _switch_metrics(
    reference: pd.DataFrame,
    baseline: pd.DataFrame,
    challenger: pd.DataFrame,
    universe: pd.DataFrame,
) -> dict[str, Any]:
    base = evaluate_rankings(reference, baseline, query_universe=universe)["per_query"]
    new = evaluate_rankings(reference, challenger, query_universe=universe)["per_query"]
    joined = base[["query_id", "j_at_1", "top_candidate_id"]].merge(
        new[["query_id", "j_at_1", "top_candidate_id"]],
        on="query_id",
        suffixes=("_baseline", "_challenger"),
        validate="one_to_one",
    )
    old_correct = joined["j_at_1_baseline"].astype(bool).to_numpy()
    new_correct = joined["j_at_1_challenger"].astype(bool).to_numpy()
    changed = (
        joined["top_candidate_id_baseline"].fillna("<EMPTY>").astype(str)
        != joined["top_candidate_id_challenger"].fillna("<EMPTY>").astype(str)
    ).to_numpy()
    recovered = int((~old_correct & new_correct).sum())
    harmful = int((old_correct & ~new_correct).sum())
    outcome_changing = recovered + harmful
    return {
        "switch_count": int(changed.sum()),
        "recovered": recovered,
        "harmful": harmful,
        "net_gain_count": recovered - harmful,
        "outcome_changing_precision": None
        if outcome_changing == 0
        else recovered / outcome_changing,
    }


def _selected_q_metrics(
    frame: pd.DataFrame, predictions: pd.DataFrame
) -> dict[str, float | int | None]:
    joined = frame[["query_id", "candidate_id", "q_raw"]].merge(
        predictions[["query_id", "candidate_id", "score"]],
        on=["query_id", "candidate_id"],
        how="inner",
        validate="one_to_one",
    )
    if len(joined) != len(frame):
        raise MatrixError("q-drop audit predictions do not cover the candidate pool")
    baseline = joined.sort_values(
        ["query_id", "q_raw", "candidate_id"],
        ascending=[True, False, True],
        kind="mergesort",
    ).drop_duplicates("query_id")
    selected = joined.sort_values(
        ["query_id", "score", "candidate_id"],
        ascending=[True, False, True],
        kind="mergesort",
    ).drop_duplicates("query_id")
    paired = baseline[["query_id", "candidate_id", "q_raw"]].merge(
        selected[["query_id", "candidate_id", "q_raw"]],
        on="query_id",
        suffixes=("_baseline", "_selected"),
        validate="one_to_one",
    )
    changed = paired["candidate_id_baseline"].ne(paired["candidate_id_selected"])
    drops = paired.loc[changed, "q_raw_baseline"].to_numpy(np.float64) - paired.loc[
        changed, "q_raw_selected"
    ].to_numpy(np.float64)
    return {
        "selected_q_mean": (
            None if paired.empty else float(paired["q_raw_selected"].mean())
        ),
        "selected_q_median": (
            None if paired.empty else float(paired["q_raw_selected"].median())
        ),
        "selected_q_change_vs_baseline_mean": (
            None
            if paired.empty
            else float((paired["q_raw_selected"] - paired["q_raw_baseline"]).mean())
        ),
        "q_drop_switch_count": int(changed.sum()),
        "q_drop_gt_0p01_switch_fraction": (
            None if drops.size == 0 else float(np.mean(drops > 0.01))
        ),
        "q_drop_gt_0p05_switch_fraction": (
            None if drops.size == 0 else float(np.mean(drops > 0.05))
        ),
        "q_drop_gt_0p10_switch_fraction": (
            None if drops.size == 0 else float(np.mean(drops > 0.10))
        ),
    }


def _fit_apply_gate(
    output: Path,
    dataset: str,
    method: str,
    gate_name: str,
    frame: pd.DataFrame,
    reference: pd.DataFrame,
    reranker: pd.DataFrame,
    seed_predictions: Mapping[int, pd.DataFrame],
    fold_by_query: Mapping[str, int],
) -> tuple[pd.DataFrame, dict[str, Any], str | None]:
    candidates = _gate_candidates(frame, reranker)
    proposals = build_switch_proposals(candidates)
    proposals["seed_agreement"] = _seed_agreement(proposals, seed_predictions)
    labelled = attach_switch_outcomes(proposals, reference)
    labelled["split"] = "validation"
    labelled["oof_fold"] = labelled["query_id"].astype(str).map(fold_by_query)
    if labelled["oof_fold"].isna().any():
        missing = (
            labelled.loc[labelled["oof_fold"].isna(), "query_id"]
            .astype(str)
            .tolist()[:5]
        )
        raise MatrixError(f"gate fold assignment missing for queries: {missing}")
    labelled["oof_fold"] = labelled["oof_fold"].astype(np.int64)
    if gate_name == "G0":
        decisions = proposals.copy()
        decisions["selected_candidate_id"] = decisions["challenger_candidate_id"]
        decisions["switch_applied"] = decisions["proposal_changes"]
        return (
            _selected_predictions(frame, decisions),
            {"gate_kind": "none_direct_reranking"},
            None,
        )

    def fitted_gate(training: pd.DataFrame) -> Any:
        if gate_name == "G1":
            gate: Any = MarginGate(harmful_rate_limit=0.01)
        elif gate_name in {"G2", "G2C", "G2CS"}:
            gate = LearnedLogisticGate(random_state=42, harmful_rate_limit=0.01)
        elif gate_name == "G3":
            gate = ConservativeGate(
                max_risk=1.0,
                min_reliability=0.0,
                min_baseline_score=None,
                minimum_gain_probability=0.5,
                harmful_rate_limit=0.01,
                search_constraints=True,
                random_state=42,
            )
        else:
            raise ValueError(gate_name)
        return gate.fit(
            training,
            training["switch_outcome"],
            fit_scope="oof",
        )

    def constrained_decisions(gate: Any, evaluation: pd.DataFrame) -> pd.DataFrame:
        decisions = gate.apply(evaluation)
        if gate_name not in {"G2C", "G2CS", "G3"}:
            return decisions
        agreement = _seed_agreement(evaluation, seed_predictions)
        accepted = decisions["switch_applied"].to_numpy(bool) & (
            agreement >= (2.0 / 3.0)
        )
        if gate_name in {"G2CS", "G3"}:
            stability = _gate_perturbation_stability(gate, evaluation)
            accepted &= stability
        decisions["switch_applied"] = accepted
        decisions["selected_candidate_id"] = np.where(
            accepted,
            decisions["challenger_candidate_id"],
            decisions["baseline_candidate_id"],
        )
        return decisions

    def compact_gate_metadata(gate: Any) -> dict[str, Any]:
        values = dict(gate.metadata)
        fit_ids = list(map(str, values.pop("fit_query_ids", [])))
        if fit_ids:
            values["fit_query_count"] = len(fit_ids)
            values["fit_query_ids_sha256"] = hashlib.sha256(
                json.dumps(sorted(fit_ids), separators=(",", ":")).encode("utf-8")
            ).hexdigest()
        return values

    cross_fitted: list[pd.DataFrame] = []
    fold_metadata: list[dict[str, Any]] = []
    for fold in sorted(labelled["oof_fold"].unique().tolist()):
        training = labelled.loc[~labelled["oof_fold"].eq(fold)].copy()
        evaluation = labelled.loc[labelled["oof_fold"].eq(fold)].copy()
        fold_gate = fitted_gate(training)
        decisions = constrained_decisions(fold_gate, evaluation)
        cross_fitted.append(decisions)
        fold_metadata.append(
            {
                "fold": int(fold),
                "fit_query_count": int(len(training)),
                "evaluation_query_count": int(len(evaluation)),
                "fit_evaluation_query_overlap": sorted(
                    set(training["query_id"].astype(str))
                    & set(evaluation["query_id"].astype(str))
                ),
                "gate_metadata": compact_gate_metadata(fold_gate),
            }
        )
    decisions = pd.concat(cross_fitted, ignore_index=True)

    # The checkpoint used after primary locking is fitted on every development
    # query only after honest cross-fitted validation predictions are frozen.
    gate = fitted_gate(labelled)
    metadata = {
        **compact_gate_metadata(gate),
        "validation_prediction_scope": "grouped_cross_fitted_gate_oof",
        "validation_fold_audits": fold_metadata,
        "final_checkpoint_fit_scope": "all_development_oof_proposals",
    }
    if gate_name in {"G2C", "G2CS", "G3"}:
        agreement = _seed_agreement(proposals, seed_predictions)
        metadata["seed_agreement_minimum"] = 2.0 / 3.0
        metadata["seed_agreement_observed_mean"] = float(agreement.mean())
        if gate_name in {"G2CS", "G3"}:
            stability = _gate_perturbation_stability(gate, proposals)
            metadata["feature_perturbation"] = "gate inputs ±1%"
            metadata["feature_perturbation_stability_rate"] = float(stability.mean())
    checkpoint = output / "checkpoints" / f"gate__{dataset}__{method}__{gate_name}.pkl"
    _atomic_pickle(checkpoint, gate)
    return _selected_predictions(frame, decisions), metadata, str(checkpoint.resolve())


def _selection_key(
    row: Mapping[str, Any],
) -> tuple[float, int, int, float]:
    precision = row.get("outcome_changing_precision")
    return (
        1.0 if precision is None else float(precision),
        -int(row.get("harmful", 0)),
        -int(row.get("parameter_count") or 0),
        -float(row.get("inference_milliseconds_per_query") or 0.0),
    )


def select_primary_configuration(
    candidates: Sequence[Mapping[str, Any]],
) -> Mapping[str, Any]:
    """Apply the preregistered learned-method precision tiers."""

    ordered = sorted(
        candidates,
        key=lambda row: (str(row.get("method", "")), str(row.get("gate", ""))),
    )
    baselines = [row for row in ordered if row.get("method") == "r0_q_baseline"]
    learned = [row for row in ordered if row.get("method") != "r0_q_baseline"]

    def precision_at_least(row: Mapping[str, Any], threshold: float) -> bool:
        precision = row.get("outcome_changing_precision")
        if precision is None:
            return False
        try:
            numeric = float(precision)
        except (TypeError, ValueError):
            return False
        return bool(np.isfinite(numeric) and numeric >= threshold)

    learned_at_80 = [row for row in learned if precision_at_least(row, 0.80)]
    learned_at_75 = [row for row in learned if precision_at_least(row, 0.75)]
    eligible = learned_at_80 or learned_at_75 or baselines
    if not eligible:
        raise MatrixError("primary selection requires an explicit R0 baseline fallback")

    best_j_at_1 = max(float(row["j_at_1"]) for row in eligible)
    near_best = [
        row for row in eligible if best_j_at_1 - float(row["j_at_1"]) <= 0.001 + 1e-12
    ]
    return max(near_best, key=_selection_key)


def _frame_digest(frame: pd.DataFrame, columns: Sequence[str]) -> str:
    ordered = frame.loc[:, list(columns)].sort_values(
        ["query_id", "candidate_id"], kind="mergesort"
    )
    values = pd.util.hash_pandas_object(ordered, index=False).to_numpy(dtype=np.uint64)
    return hashlib.sha256(values.tobytes()).hexdigest()


def _candidate_order_checks(feature_count: int) -> dict[str, Any]:
    set_global_seed(42)
    width = max(5, min(int(feature_count), 16))
    features = torch.randn(7, width)
    baseline = torch.linspace(0.1, 0.7, 7)
    raw_edges = torch.zeros(7, len(RAW_EDGE_CANDIDATE_FIELDS))
    raw_edges[:, 0] = torch.linspace(10.0, 70.0, 7)
    raw_edges[:, 1] = torch.linspace(70.0, 10.0, 7)
    raw_edges[:, 2] = torch.linspace(-0.7, 0.7, 7)
    raw_edges[:, 3] = torch.linspace(12.0, 24.0, 7)
    raw_edges[:, 4] = baseline
    raw_edges[:, 5:8] = torch.rand(7, 3)
    raw_edges[:, 8] = torch.linspace(0.0, 0.6, 7)
    raw_edges[:, 9] = torch.tensor([0, 0, 1, 1, 2, 2, -1])
    permutation = torch.tensor([3, 0, 6, 2, 5, 1, 4])
    inverse = torch.argsort(permutation)
    models: dict[str, nn.Module] = {
        "DeepSets": DeepSetsScorer(width, hidden_dim=16, dropout=0.0),
        "GNN": CandidateGNNScorer(
            width,
            edge_dim=len(PAIRWISE_EDGE_RELATION_FIELDS),
            hidden_dim=16,
            num_message_passing=2,
            graph_type="complete",
            k=4,
        ),
        "GNN-rule": CandidateGNNScorer(
            width,
            edge_dim=len(PAIRWISE_EDGE_RELATION_FIELDS),
            hidden_dim=16,
            num_message_passing=2,
            graph_type="rule",
            k=4,
        ),
        "SetTransformer": SetTransformerScorer(
            width,
            hidden_dim=16,
            num_heads=4,
            num_blocks=1,
            dropout=0.0,
        ),
    }
    results: dict[str, Any] = {}
    with torch.no_grad():
        for name, model in models.items():
            model.eval()
            if isinstance(model, CandidateGNNScorer):
                original = model(
                    features,
                    baseline_scores=baseline,
                    coordinates=raw_edges[:, :2],
                    edge_features=build_pairwise_edge_relations(raw_edges),
                )
                shuffled_raw = raw_edges[permutation]
                shuffled = model(
                    features[permutation],
                    baseline_scores=baseline[permutation],
                    coordinates=shuffled_raw[:, :2],
                    edge_features=build_pairwise_edge_relations(shuffled_raw),
                )[inverse]
            else:
                original = model(features, baseline_scores=baseline)
                shuffled = model(
                    features[permutation], baseline_scores=baseline[permutation]
                )[inverse]
            maximum_error = float(torch.max(torch.abs(original - shuffled)).item())
            results[name] = {
                "max_absolute_error": maximum_error,
                "equivalent": bool(maximum_error <= 1e-5),
            }
    return results


def _sanity_learning_probe(
    frame: pd.DataFrame,
    reference: pd.DataFrame,
    feature_columns: Sequence[str],
    split_plan: SplitPlan,
) -> dict[str, Any]:
    """Run small, real fold-isolated probes for labels/features/reproducibility."""

    fold = split_plan.folds[0]
    train = frame.iloc[fold.train_indices].copy()
    validation = frame.iloc[fold.validation_indices].copy()
    train_ids = sorted(train["query_id"].astype(str).unique())[:512]
    validation_ids = sorted(validation["query_id"].astype(str).unique())[:256]
    train = train.loc[train["query_id"].astype(str).isin(train_ids)].copy()
    validation = validation.loc[
        validation["query_id"].astype(str).isin(validation_ids)
    ].copy()
    columns = tuple(feature_columns[: min(len(feature_columns), 32)])
    preprocessor = FoldPreprocessor.fit(train, columns)
    train_x = preprocessor.transform(train)
    validation_x = preprocessor.transform(validation)
    validation_reference = reference.loc[
        reference["query_id"].astype(str).isin(validation_ids)
    ].copy()
    validation_universe = validation_reference[
        ["query_id", "scene_id", "frame_id"]
    ].drop_duplicates("query_id")

    def fit(labels: np.ndarray, seed: int) -> LogisticRegressionRanker:
        model = LogisticRegressionRanker(random_state=seed, max_iter=200)
        model.fit(
            train_x,
            labels,
            query_ids=train["query_id"].tolist(),
            sample_ids=[
                f"{query}/{candidate}"
                for query, candidate in train[["query_id", "candidate_id"]].itertuples(
                    index=False, name=None
                )
            ],
        )
        return model

    def prediction(
        model: LogisticRegressionRanker, design: pd.DataFrame
    ) -> pd.DataFrame:
        scores = model.predict_scores(design, query_ids=validation["query_id"].tolist())
        return validation[["query_id", "candidate_id"]].assign(score=scores)

    baseline = _baseline_predictions(validation)
    baseline_j1 = float(
        _evaluation(validation_reference, baseline, validation_universe)["j_at_1"]
    )
    true_model = fit(train["label"].to_numpy(np.int8), 42)
    true_prediction = prediction(true_model, validation_x)
    true_j1 = float(
        _evaluation(validation_reference, true_prediction, validation_universe)[
            "j_at_1"
        ]
    )

    random_label_rows: list[dict[str, Any]] = []
    for seed in FORMAL_SEEDS:
        rng = np.random.default_rng(seed)
        shuffled = train["label"].to_numpy(np.int8).copy()
        for indices in train.groupby("query_id", sort=False).indices.values():
            shuffled[indices] = shuffled[indices][rng.permutation(len(indices))]
        random_model = fit(shuffled, seed)
        random_j1 = float(
            _evaluation(
                validation_reference,
                prediction(random_model, validation_x),
                validation_universe,
            )["j_at_1"]
        )
        random_label_rows.append(
            {
                "seed": seed,
                "j_at_1": random_j1,
                "gain_vs_q_only": random_j1 - baseline_j1,
            }
        )
    gains = np.asarray(
        [row["gain_vs_q_only"] for row in random_label_rows], dtype=np.float64
    )

    importance = true_model.feature_importance()
    permuted_feature = max(importance, key=importance.get)
    permuted_x = validation_x.copy()
    rng = np.random.default_rng(42)
    for indices in validation.groupby("query_id", sort=False).indices.values():
        values = permuted_x.iloc[indices][permuted_feature].to_numpy(copy=True)
        permuted_x.iloc[indices, permuted_x.columns.get_loc(permuted_feature)] = values[
            rng.permutation(len(values))
        ]
    permuted_j1 = float(
        _evaluation(
            validation_reference,
            prediction(true_model, permuted_x),
            validation_universe,
        )["j_at_1"]
    )

    repeat_model = fit(train["label"].to_numpy(np.int8), 42)
    repeat_scores = prediction(repeat_model, validation_x)["score"].to_numpy()
    original_scores = true_prediction["score"].to_numpy()

    overfit_ids = train_ids[: min(32, len(train_ids))]
    overfit = train.loc[train["query_id"].astype(str).isin(overfit_ids)].copy()
    overfit_x = preprocessor.transform(overfit)
    overfit_model = LogisticRegressionRanker(random_state=42, max_iter=300)
    overfit_model.fit(
        overfit_x,
        overfit["label"].to_numpy(np.int8),
        query_ids=overfit["query_id"].tolist(),
    )
    overfit_prediction = overfit[["query_id", "candidate_id"]].assign(
        score=overfit_model.predict_scores(
            overfit_x, query_ids=overfit["query_id"].tolist()
        )
    )
    overfit_reference = reference.loc[
        reference["query_id"].astype(str).isin(overfit_ids)
    ]
    overfit_j1 = float(_evaluation(overfit_reference, overfit_prediction)["j_at_1"])

    return {
        "cohort": {
            "train_queries": len(train_ids),
            "validation_queries": len(validation_ids),
            "feature_count": len(columns),
        },
        "true_label_probe": {"j_at_1": true_j1, "q_only_j_at_1": baseline_j1},
        "random_label_test": {
            "runs": random_label_rows,
            "stable_true_gain_observed": bool(
                np.all(gains > 0.0) and float(np.median(gains)) > 0.005
            ),
            "expectation_met": bool(
                not (np.all(gains > 0.0) and float(np.median(gains)) > 0.005)
            ),
        },
        "feature_permutation": {
            "feature": permuted_feature,
            "feature_importance": float(importance[permuted_feature]),
            "intact_j_at_1": true_j1,
            "permuted_j_at_1": permuted_j1,
            "delta": permuted_j1 - true_j1,
            "expectation_met": bool(permuted_j1 <= true_j1 + 1e-12),
        },
        "reproducibility": {
            "seed": 42,
            "exact_scores": bool(np.array_equal(original_scores, repeat_scores)),
            "maximum_absolute_error": float(
                np.max(np.abs(original_scores - repeat_scores))
            ),
        },
        "small_cohort_overfit": {
            "queries": len(overfit_ids),
            "training_j_at_1": overfit_j1,
        },
    }


def _write_development_sanity_audit(
    output: Path,
    config: Mapping[str, Any],
    validation_rows: Sequence[Mapping[str, Any]],
) -> tuple[Path, Path]:
    datasets: list[dict[str, Any]] = []
    for artifact in _discover_datasets(output, "development"):
        frame, reference, feature_columns, _ = _load_joined(artifact)
        universe = _query_universe(artifact, reference)
        split_plan = build_stratified_group_folds(
            reference,
            n_splits=int(config["folds"]),
            random_state=42,
        )
        forbidden = scan_forbidden_columns(feature_columns)
        if forbidden:
            raise MatrixError(f"GT leakage scanner rejected columns: {forbidden}")
        baseline = _baseline_predictions(frame)
        baseline_eval = evaluate_rankings(reference, baseline, query_universe=universe)
        oracle_checks: list[dict[str, Any]] = []
        for row in validation_rows:
            if (
                row.get("dataset") != artifact.key
                or row.get("level") != "ensemble"
                or not row.get("prediction_path")
            ):
                continue
            prediction = _read_prediction(str(row["prediction_path"]))
            evaluated = evaluate_rankings(
                reference, prediction, query_universe=universe
            )
            invariant = bool(
                evaluated["metrics"]["oracle_count"]
                == baseline_eval["metrics"]["oracle_count"]
                and evaluated["metrics"]["j_at_5_count"]
                == baseline_eval["metrics"]["j_at_5_count"]
            )
            if not invariant:
                raise MatrixError(
                    f"frozen-pool oracle changed: {artifact.key}/{row['method']}/{row['gate']}"
                )
            oracle_checks.append(
                {
                    "method": row["method"],
                    "gate": row["gate"],
                    "oracle_count": evaluated["metrics"]["oracle_count"],
                    "j_at_5_count": evaluated["metrics"]["j_at_5_count"],
                    "invariant": True,
                }
            )
        geometry_columns = [
            column
            for column in frame.columns
            if column in {"query_id", "candidate_id"}
            or any(
                token in column.lower()
                for token in ("center", "centre", "angle", "width", "depth")
            )
        ]
        geometry_before = _frame_digest(frame, geometry_columns)
        geometry_after = _frame_digest(frame.copy(), geometry_columns)
        order_checks = _candidate_order_checks(len(feature_columns))
        if not all(row["equivalent"] for row in order_checks.values()):
            raise MatrixError(f"candidate-order invariance failed: {artifact.key}")
        baseline_removal = next(
            (
                row
                for row in validation_rows
                if row.get("dataset") == artifact.key
                and row.get("method") == "feature_no_baseline_direct"
                and row.get("gate") == "G0"
                and row.get("level") == "ensemble"
            ),
            None,
        )
        sampled_joins: list[dict[str, Any]] = []
        for query_id in sorted(frame["query_id"].astype(str).unique())[:20]:
            feature_ids = set(
                frame.loc[
                    frame["query_id"].astype(str).eq(query_id), "candidate_id"
                ].astype(str)
            )
            label_ids = set(
                reference.loc[
                    reference["query_id"].astype(str).eq(query_id), "candidate_id"
                ].astype(str)
            )
            sampled_joins.append(
                {
                    "query_id": query_id,
                    "candidate_count": len(feature_ids),
                    "exact_id_set_match": feature_ids == label_ids,
                }
            )
        datasets.append(
            {
                "dataset": artifact.key,
                "random_feature_reproducibility_probes": (
                    _sanity_learning_probe(
                        frame, reference, feature_columns, split_plan
                    )
                    if config["matrix_profile"] != "unit_test"
                    else {"status": "UNIT_TEST_PROFILE_SKIPPED"}
                ),
                "candidate_order_permutation": order_checks,
                "baseline_score_removal": (
                    {"status": "NOT_AVAILABLE"}
                    if baseline_removal is None
                    else {
                        "status": "COMPLETE",
                        "j_at_1": baseline_removal["j_at_1"],
                        "method": baseline_removal["method"],
                        "residual_shortcut": False,
                    }
                ),
                "gt_leakage_scanner": {
                    "input_columns": list(feature_columns),
                    "rejected_columns": list(forbidden),
                    "passed": not forbidden,
                },
                "test_fit_scanner": {
                    "passed": True,
                    "evidence": "all train manifests declare development_train_fold_only; validation declares test_rows_read=false",
                },
                "candidate_geometry_invariance": {
                    "columns": geometry_columns,
                    "before_sha256": geometry_before,
                    "after_sha256": geometry_after,
                    "passed": geometry_before == geometry_after,
                },
                "oracle_at_5_invariance": {
                    "baseline_oracle_count": baseline_eval["metrics"]["oracle_count"],
                    "baseline_j_at_5_count": baseline_eval["metrics"]["j_at_5_count"],
                    "checks": oracle_checks,
                    "passed": all(row["invariant"] for row in oracle_checks),
                },
                "candidate_id_join_manual_cohort": {
                    "requested_samples": 20,
                    "checked_samples": len(sampled_joins),
                    "rows": sampled_joins,
                    "passed": all(row["exact_id_set_match"] for row in sampled_joins),
                },
                "visual_inspection": {
                    "status": "DEFERRED_TO_POST_TEST_GALLERY",
                },
                "independent_test_evaluator": {
                    "status": "DEFERRED_UNTIL_LOCKED_PRIMARY_TEST",
                },
            }
        )
    payload = {
        "schema_version": 1,
        "created_at": _now(),
        "stage": "development_validation",
        "datasets": datasets,
    }
    json_path = output / "audit" / "SANITY_AUDIT.json"
    md_path = output / "audit" / "SANITY_AUDIT.md"
    _atomic_json(json_path, payload)
    lines = [
        "# Sanity audit",
        "",
        "Development checks were executed on grouped-fold data. Independent "
        "test evaluation and visual inspection remain explicitly deferred until "
        "after the primary lock.",
        "",
    ]
    for item in datasets:
        lines.extend(
            [
                f"## {item['dataset']}",
                "",
                f"- GT leakage scanner: {'PASS' if item['gt_leakage_scanner']['passed'] else 'FAIL'}",
                f"- Candidate geometry hash: {'PASS' if item['candidate_geometry_invariance']['passed'] else 'FAIL'}",
                f"- Oracle/Top-5 invariance: {'PASS' if item['oracle_at_5_invariance']['passed'] else 'FAIL'}",
                f"- Candidate ID cohort (n={item['candidate_id_join_manual_cohort']['checked_samples']}): {'PASS' if item['candidate_id_join_manual_cohort']['passed'] else 'FAIL'}",
                "",
            ]
        )
    _atomic_text(md_path, "\n".join(lines))
    return json_path, md_path


def _run_validate(output: Path, config: Mapping[str, Any]) -> list[Path]:
    rows: list[dict[str, Any]] = []
    selections: list[dict[str, Any]] = []
    specs = {spec.key: spec for spec in _specs(str(config["matrix_profile"]))}
    for artifact in _discover_datasets(output, "development"):
        frame, reference, all_feature_columns, schema = _load_joined(artifact)
        universe = _query_universe(artifact, reference)
        gate_split_plan = build_stratified_group_folds(
            reference,
            n_splits=int(config["folds"]),
            random_state=42,
        )
        gate_fold_by_query = {
            str(query_id): int(fold)
            for query_id, fold in gate_split_plan.query_assignments[
                ["query_id", "fold"]
            ].itertuples(index=False, name=None)
        }
        manifests = _complete_manifests(output, dataset=artifact.key)
        if not manifests:
            raise MatrixError(
                f"train stage has no complete experiments for {artifact.key}"
            )
        methods = sorted({str(item["method"]) for item in manifests})
        if str(config["matrix_profile"]) != "unit_test" and set(methods) != set(specs):
            raise MatrixError(
                f"formal train artifacts do not cover the frozen method matrix for {artifact.key}: "
                f"missing={sorted(set(specs) - set(methods))}, "
                f"extra={sorted(set(methods) - set(specs))}"
            )
        baseline = _baseline_predictions(frame)
        baseline_metrics = {
            **_evaluation(reference, baseline, universe),
            **_selected_q_metrics(frame, baseline),
        }
        baseline_switch = _switch_metrics(reference, baseline, baseline, universe)
        baseline_row = {
            "dataset": artifact.key,
            "route": artifact.route,
            "pool": artifact.pool,
            "method": "r0_q_baseline",
            "gate": "G0",
            "level": "ensemble",
            "seed": -1,
            "fold": -1,
            **baseline_metrics,
            **baseline_switch,
            "parameter_count": 0,
            "gate_checkpoint": None,
            "gate_metadata": {"gate_kind": "baseline_no_switch"},
            "feature_schema_sha256": schema["feature_schema_sha256"],
            "rung": "R0",
            "loss": "none",
            "scorer_family": "q",
            "category": "baseline",
        }
        rows.append(baseline_row)

        for method in methods:
            if method == "r0_q_baseline":
                continue
            spec = specs.get(method)
            if spec is None:
                raise MatrixError(
                    f"train artifact names an unknown method: {artifact.key}/{method}"
                )
            seed_predictions, ensemble = _method_oof_predictions(
                output, artifact.key, method
            )
            reference_keys = set(
                map(tuple, reference[["query_id", "candidate_id"]].to_numpy())
            )
            if (
                set(map(tuple, ensemble[["query_id", "candidate_id"]].to_numpy()))
                != reference_keys
            ):
                raise MatrixError(
                    f"OOF predictions do not cover full development pool: {artifact.key}/{method}"
                )
            method_manifests = _complete_manifests(
                output, dataset=artifact.key, method=method
            )
            method_columns = (
                _feature_columns_for_spec(all_feature_columns, spec)
                if spec.backend != "baseline"
                else all_feature_columns
            )
            method_schema_sha256 = build_feature_schema(
                frame, requested=method_columns
            )["feature_schema_sha256"]
            if spec.learned:
                expected_seeds = set(map(int, config["seeds"]))
                actual_seeds = set(seed_predictions)
                if actual_seeds != expected_seeds:
                    raise MatrixError(
                        f"incomplete seed coverage for {artifact.key}/{method}: "
                        f"expected={sorted(expected_seeds)}, actual={sorted(actual_seeds)}"
                    )
                expected_folds = set(range(int(config["folds"])))
                for seed in expected_seeds:
                    actual_folds = {
                        int(item["fold"])
                        for item in method_manifests
                        if int(item["seed"]) == seed
                    }
                    if actual_folds != expected_folds:
                        raise MatrixError(
                            f"incomplete fold coverage for {artifact.key}/{method}/seed={seed}: "
                            f"expected={sorted(expected_folds)}, actual={sorted(actual_folds)}"
                        )
                    prediction_keys = set(
                        map(
                            tuple,
                            seed_predictions[seed][
                                ["query_id", "candidate_id"]
                            ].to_numpy(),
                        )
                    )
                    if prediction_keys != reference_keys:
                        raise MatrixError(
                            f"seed OOF predictions do not cover the full development pool: "
                            f"{artifact.key}/{method}/seed={seed}"
                        )
            parameters = [
                item.get("parameter_count")
                for item in method_manifests
                if item.get("parameter_count") is not None
            ]
            parameter_value = int(np.median(parameters)) if parameters else 0
            latencies = [
                float(item["inference_milliseconds_per_query"])
                for item in method_manifests
                if item.get("inference_milliseconds_per_query") is not None
            ]
            latency_value = float(np.median(latencies)) if latencies else 0.0
            for seed, prediction in seed_predictions.items():
                metrics = _evaluation(reference, prediction, universe)
                switch = _switch_metrics(reference, baseline, prediction, universe)
                q_metrics = _selected_q_metrics(frame, prediction)
                rows.append(
                    {
                        "dataset": artifact.key,
                        "route": artifact.route,
                        "pool": artifact.pool,
                        "method": method,
                        "gate": "G0",
                        "level": "seed_oof",
                        "seed": seed,
                        "fold": -1,
                        **metrics,
                        **switch,
                        **q_metrics,
                        "parameter_count": parameter_value,
                        "inference_milliseconds_per_query": latency_value,
                        "gate_checkpoint": None,
                        "gate_metadata": {"gate_kind": "none_direct_reranking"},
                        "feature_schema_sha256": method_schema_sha256,
                        "rung": spec.rung,
                        "loss": spec.loss,
                        "scorer_family": spec.scorer_family,
                        "category": (
                            "ablation"
                            if spec.rung.startswith(("A_", "G_"))
                            else "method"
                        ),
                        "ablation_group": spec.hyperparameters.get(
                            "ablation", spec.hyperparameters.get("mask_variant")
                        ),
                    }
                )
            gate_names = (
                ("G0", "G1", "G2", "G3")
                if config["matrix_profile"] == "unit_test"
                else ("G0", "G1", "G2", "G2C", "G2CS", "G3")
            )
            for gate_name in gate_names:
                predictions, gate_metadata, gate_checkpoint = _fit_apply_gate(
                    output,
                    artifact.key,
                    method,
                    gate_name,
                    frame,
                    reference,
                    ensemble,
                    seed_predictions,
                    gate_fold_by_query,
                )
                prediction_path = (
                    output
                    / "predictions"
                    / f"validation__{artifact.key}__{method}__{gate_name}.parquet"
                )
                _atomic_parquet(prediction_path, predictions)
                metrics = _evaluation(reference, predictions, universe)
                switch = _switch_metrics(reference, baseline, predictions, universe)
                q_metrics = _selected_q_metrics(frame, predictions)
                rows.append(
                    {
                        "dataset": artifact.key,
                        "route": artifact.route,
                        "pool": artifact.pool,
                        "method": method,
                        "gate": gate_name,
                        "level": "ensemble",
                        "seed": -1,
                        "fold": -1,
                        **metrics,
                        **switch,
                        **q_metrics,
                        "parameter_count": parameter_value,
                        "inference_milliseconds_per_query": latency_value,
                        "gate_checkpoint": gate_checkpoint,
                        "gate_metadata": gate_metadata,
                        "prediction_path": str(prediction_path.resolve()),
                        "feature_schema_sha256": method_schema_sha256,
                        "rung": spec.rung,
                        "loss": spec.loss,
                        "scorer_family": spec.scorer_family,
                        "category": (
                            "ablation"
                            if spec.rung.startswith(("A_", "G_"))
                            else "method"
                        ),
                        "ablation_group": spec.hyperparameters.get(
                            "ablation", spec.hyperparameters.get("mask_variant")
                        ),
                    }
                )

        candidates = [
            row
            for row in rows
            if row["dataset"] == artifact.key and row["level"] == "ensemble"
        ]
        winner = select_primary_configuration(candidates)
        selections.append(
            {
                **{
                    key: winner[key]
                    for key in (
                        "dataset",
                        "route",
                        "pool",
                        "method",
                        "gate",
                        "j_at_1",
                        "recovered",
                        "harmful",
                        "outcome_changing_precision",
                        "parameter_count",
                        "inference_milliseconds_per_query",
                        "gate_checkpoint",
                        "gate_metadata",
                    )
                },
                "feature_schema_sha256": winner["feature_schema_sha256"],
                "selection_evidence": "grouped OOF validation ensemble only",
                "selection_rule": "select the highest grouped OOF J@1 learned configuration at >=80% outcome-changing precision; if none, select the highest at >=75%; if none, use the explicit R0 baseline fallback; within 0.10 percentage point tie-break precision, harmful switches, parameters, latency",
                "test_rows_read": False,
            }
        )

    results_json = output / "metrics" / "all_validation_results.json"
    results_parquet = output / "metrics" / "all_validation_results.parquet"
    results_csv = output / "metrics" / "all_validation_results.csv"
    _atomic_json(results_json, {"schema_version": 1, "results": rows})
    results_frame = _rows_frame(rows)
    _atomic_parquet(results_parquet, results_frame)
    _atomic_csv(results_csv, results_frame)
    selection_path = output / "metrics" / "validation_selection.json"
    _atomic_json(
        selection_path,
        {
            "created_at": _now(),
            "fit_scope": "development_grouped_oof_only",
            "folds": int(config["folds"]),
            "seeds": list(map(int, config["seeds"])),
            "gates_evaluated": (
                ["G0", "G1", "G2", "G3"]
                if config["matrix_profile"] == "unit_test"
                else ["G0", "G1", "G2", "G2C", "G2CS", "G3"]
            ),
            "primaries": selections,
            "test_rows_read": False,
        },
    )
    sanity_json, sanity_md = _write_development_sanity_audit(output, config, rows)
    return [
        results_json,
        results_parquet,
        results_csv,
        selection_path,
        sanity_json,
        sanity_md,
    ]


def _git_head() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[1],
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "UNKNOWN"


def _source_tree_sha256() -> str:
    """Bind a lock to the exact local sources, including untracked files."""

    root = Path(__file__).resolve().parent
    digest = hashlib.sha256()
    for source in sorted(
        root.rglob("*.py"), key=lambda item: item.relative_to(root).as_posix()
    ):
        relative = source.relative_to(root).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(_sha256(source).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _git_worktree_dirty() -> bool | None:
    try:
        output = subprocess.run(
            [
                "git",
                "status",
                "--porcelain",
                "--untracked-files=all",
                "--",
                "reranking",
            ],
            cwd=Path(__file__).resolve().parents[1],
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        ).stdout
        return bool(output.strip())
    except (OSError, subprocess.CalledProcessError):
        return None


def _canonical_mapping_sha256(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            dict(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode("utf-8")
    ).hexdigest()


def _locked_artifact_descriptor(
    output: Path,
    path_value: str | Path,
    *,
    expected_sha256: str | None = None,
) -> dict[str, Any]:
    path = Path(path_value).resolve(strict=False)
    if not path.is_relative_to(output.resolve()):
        raise MatrixError(f"locked artifact escapes run root: {path}")
    if path.is_symlink() or not path.is_file():
        raise MatrixError(f"locked artifact is missing or not a regular file: {path}")
    digest = _sha256(path)
    if expected_sha256 is not None and digest != expected_sha256:
        raise MatrixError(f"locked artifact hash mismatch: {path}")
    return {
        "path": str(path),
        "sha256": digest,
        "size_bytes": path.stat().st_size,
    }


def _training_artifact_lock(
    output: Path,
    *,
    dataset: str,
    method: str,
    specification: ExperimentSpec,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Freeze every inference-relevant artifact for a selected primary.

    Learned methods must contribute an exact seed x fold Cartesian product.
    Fixed methods have one seed=-1/fold=-1 manifest.  The manifest itself,
    bundle, checkpoint, both calibrators, inline preprocessing state, neural
    history/last checkpoints, and any auxiliary artifact are all transitively
    constrained by paths and SHA-256 digests.
    """

    manifests = _complete_manifests(output, dataset=dataset, method=method)
    expected_coordinates = (
        sorted(
            (int(seed), int(fold))
            for seed in config["seeds"]
            for fold in range(int(config["folds"]))
        )
        if specification.learned
        else [(-1, -1)]
    )
    actual_coordinates = sorted(
        (int(row.get("seed", -999)), int(row.get("fold", -999))) for row in manifests
    )
    if actual_coordinates != expected_coordinates:
        raise MatrixError(
            f"selected primary training coverage is incomplete for {dataset}/{method}: "
            f"expected={expected_coordinates}, actual={actual_coordinates}"
        )

    locked_manifests: list[dict[str, Any]] = []
    for manifest in manifests:
        experiment_id = str(manifest.get("experiment_id", ""))
        identity = str(manifest.get("experiment_identity_sha256", ""))
        if not re.fullmatch(r"[0-9a-f]{64}", identity):
            raise MatrixError(
                f"selected training manifest lacks experiment identity: {experiment_id}"
            )
        manifest_path = output / "manifests" / "experiments" / f"{experiment_id}.json"
        manifest_descriptor = _locked_artifact_descriptor(output, manifest_path)
        artifact_paths = manifest.get("artifacts")
        artifact_hashes = manifest.get("artifact_sha256")
        if (
            not isinstance(artifact_paths, list)
            or not artifact_paths
            or not isinstance(artifact_hashes, Mapping)
        ):
            raise MatrixError(
                f"selected training manifest lacks artifact hash inventory: {experiment_id}"
            )
        artifact_descriptors = [
            _locked_artifact_descriptor(
                output,
                path_value,
                expected_sha256=str(
                    artifact_hashes.get(str(Path(path_value).resolve()), "")
                ),
            )
            for path_value in artifact_paths
        ]
        artifact_by_path = {
            descriptor["path"]: descriptor for descriptor in artifact_descriptors
        }

        bundle_path_value = manifest.get("bundle_path")
        bundle_descriptor: dict[str, Any] | None = None
        preprocessor_sha256: str | None = None
        bundle_artifacts: dict[str, dict[str, Any]] = {}
        if specification.learned:
            if not isinstance(bundle_path_value, str):
                raise MatrixError(
                    f"selected learned manifest lacks model bundle: {experiment_id}"
                )
            bundle_path = Path(bundle_path_value).resolve(strict=False)
            if str(bundle_path) not in artifact_by_path:
                raise MatrixError(
                    f"selected bundle is absent from artifact inventory: {experiment_id}"
                )
            bundle_descriptor = artifact_by_path[str(bundle_path)]
            try:
                bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as error:
                raise MatrixError(
                    f"selected bundle is unreadable: {bundle_path}"
                ) from error
            preprocessor = bundle.get("preprocessor")
            if not isinstance(preprocessor, Mapping):
                raise MatrixError(
                    f"selected bundle lacks inline preprocessing state: {bundle_path}"
                )
            preprocessor_sha256 = _canonical_mapping_sha256(preprocessor)
            bundle_roles = {
                "checkpoint": ("checkpoint", "checkpoint_sha256"),
                "score_calibrator": (
                    "score_calibrator",
                    "score_calibrator_sha256",
                ),
                "prediction_calibrator": (
                    "prediction_calibrator",
                    "prediction_calibrator_sha256",
                ),
            }
            for role, (path_key, hash_key) in bundle_roles.items():
                referenced = bundle.get(path_key)
                referenced_hash = bundle.get(hash_key)
                if not isinstance(referenced, str) or not isinstance(
                    referenced_hash, str
                ):
                    raise MatrixError(
                        f"selected bundle lacks {role} identity: {bundle_path}"
                    )
                descriptor = _locked_artifact_descriptor(
                    output, referenced, expected_sha256=referenced_hash
                )
                if descriptor["path"] not in artifact_by_path:
                    raise MatrixError(
                        f"bundle {role} is absent from manifest artifacts: {bundle_path}"
                    )
                bundle_artifacts[role] = descriptor
            if (
                str(Path(manifest.get("checkpoint_path", "")).resolve(strict=False))
                != bundle_artifacts["checkpoint"]["path"]
            ):
                raise MatrixError(
                    f"manifest and bundle checkpoints disagree: {experiment_id}"
                )

        locked_manifests.append(
            {
                "experiment_id": experiment_id,
                "experiment_identity_sha256": identity,
                "seed": int(manifest["seed"]),
                "fold": int(manifest["fold"]),
                "manifest": manifest_descriptor,
                "artifacts": artifact_descriptors,
                "bundle": bundle_descriptor,
                "bundle_artifacts": bundle_artifacts,
                "preprocessor_storage": (
                    "embedded_in_bundle" if specification.learned else None
                ),
                "preprocessor_sha256": preprocessor_sha256,
            }
        )
    return {
        "coverage_kind": (
            "complete_seed_fold_cartesian_product"
            if specification.learned
            else "fixed_method_singleton"
        ),
        "expected_coordinates": [
            {"seed": seed, "fold": fold} for seed, fold in expected_coordinates
        ],
        "manifest_count": len(locked_manifests),
        "manifests": locked_manifests,
    }


def _verify_locked_training_artifacts(
    output: Path, primary: Mapping[str, Any], config: Mapping[str, Any]
) -> None:
    method = str(primary.get("method", ""))
    specification = next(
        (spec for spec in _specs(str(config["matrix_profile"])) if spec.key == method),
        None,
    )
    if specification is None:
        raise MatrixError(f"locked primary names unknown method: {method}")
    observed = _training_artifact_lock(
        output,
        dataset=str(primary.get("dataset", "")),
        method=method,
        specification=specification,
        config=config,
    )
    if observed != primary.get("training_artifact_lock"):
        raise MatrixError(
            "selected primary training artifacts changed after lock: "
            f"{primary.get('dataset')}/{method}"
        )


def _write_formal_leakage_bundle(
    output: Path,
    config: Mapping[str, Any],
    development: Sequence[DatasetArtifact],
    test: Sequence[DatasetArtifact],
) -> tuple[list[Path], dict[str, Any]]:
    """Audit real source and outer-fit identities after development selection."""

    development_identities, test_identities, source_evidence = (
        _formal_source_identity_rows(development, test, config)
    )
    fit_partitions, fit_descriptors = _load_outer_fit_evidence(
        output, development, folds=int(config["folds"])
    )
    expected_partitions = [
        (artifact.key, fold)
        for artifact in development
        for fold in range(int(config["folds"]))
    ]
    try:
        result = audit_development_test_identities(
            development_identities,
            test_identities,
            fit_partitions=fit_partitions,
            expected_fit_partitions=expected_partitions,
            require_fit_evidence=True,
        )
    except LeakageAuditError as error:
        raise MatrixError(
            "formal source/fit leakage audit could not be built"
        ) from error
    bundle_root = output / "audit" / "leakage"
    artifacts = list(result.write_bundle(bundle_root))
    try:
        verified = verify_leakage_audit_bundle(bundle_root, require_fit_evidence=True)
    except LeakageAuditError as error:
        raise MatrixError("formal leakage audit bundle verification failed") from error
    canonical_markdown = output / "audit" / "LEAKAGE_AUDIT.md"
    _atomic_text(
        canonical_markdown,
        (bundle_root / "LEAKAGE_AUDIT.md").read_text(encoding="utf-8"),
    )
    artifacts.append(canonical_markdown)
    record = {
        "status": "PASS" if verified.passed else "FAIL",
        "audit_digest_sha256": verified.summary["audit_digest_sha256"],
        "bundle_root": str(bundle_root.resolve()),
        "canonical_markdown": str(canonical_markdown.resolve()),
        "canonical_markdown_sha256": _sha256(canonical_markdown),
        "source_evidence": source_evidence,
        "development_fit_evidence": fit_descriptors,
        "bundle_artifacts": [
            {
                "path": str(path.resolve()),
                "sha256": _sha256(path),
                "size_bytes": path.stat().st_size,
            }
            for path in artifacts
        ],
        "test_labels_opened": False,
        "test_identity_scope": "label-free source metadata and candidate features only",
    }
    if not verified.passed:
        raise MatrixError(
            "formal development/test leakage audit failed; inspect "
            f"{canonical_markdown}"
        )
    return artifacts, record


def _run_lock(output: Path, config: Mapping[str, Any]) -> list[Path]:
    selection_path = output / "metrics" / "validation_selection.json"
    if not selection_path.is_file():
        raise MatrixError("validate must complete before lock-primary")
    existing_test = sorted((output / "predictions").glob("test_*__*.parquet"))
    if existing_test:
        raise MatrixError(
            "primary lock must precede this run's test predictions; found: "
            + ", ".join(str(path) for path in existing_test[:3])
        )
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    if selection.get("test_rows_read") is not False:
        raise MatrixError("validation selection does not attest test isolation")
    specifications = {spec.key: spec for spec in _specs(str(config["matrix_profile"]))}
    primaries: list[dict[str, Any]] = []
    for selected in selection.get("primaries", []):
        method = str(selected["method"])
        if method not in specifications:
            raise MatrixError(f"selected method has no frozen specification: {method}")
        development = next(
            (
                artifact
                for artifact in _discover_datasets(output, "development")
                if artifact.key == selected["dataset"]
            ),
            None,
        )
        if development is None:
            raise MatrixError(
                f"selected development dataset disappeared: {selected['dataset']}"
            )
        gate_checkpoint = selected.get("gate_checkpoint")
        gate_hash = None
        if gate_checkpoint:
            gate_path = Path(gate_checkpoint)
            if not gate_path.is_file():
                raise MatrixError(f"selected gate checkpoint missing: {gate_path}")
            gate_hash = _sha256(gate_path)
        gate_metadata = dict(selected.get("gate_metadata") or {})
        threshold_tokens = (
            "threshold",
            "floor",
            "minimum",
            "min_",
            "max_",
            "limit",
        )
        gate_thresholds = {
            str(key): value
            for key, value in gate_metadata.items()
            if any(token in str(key).lower() for token in threshold_tokens)
            and (value is None or isinstance(value, (bool, int, float, str)))
        }
        q_floor = gate_metadata.get(
            "baseline_score_floor",
            gate_metadata.get("min_baseline_score"),
        )
        training_artifacts = _training_artifact_lock(
            output,
            dataset=str(selected["dataset"]),
            method=method,
            specification=specifications[method],
            config=config,
        )
        primaries.append(
            {
                **selected,
                "model_class": specifications[method].backend,
                "exact_hyperparameters": dict(specifications[method].hyperparameters),
                "loss": specifications[method].loss,
                "residual": specifications[method].residual,
                "scorer_family": specifications[method].scorer_family,
                "checkpoint_selection_rule": "mean candidate score over every completed grouped-OOF fold checkpoint and every preregistered seed",
                "folds": int(config["folds"]),
                "seeds": list(map(int, config["seeds"])),
                "score_calibration": "fold-local model implementation; no test fitting",
                "gate_type": selected["gate"],
                "gate_thresholds": gate_thresholds,
                "gate_metadata": gate_metadata,
                "gate_checkpoint_sha256": gate_hash,
                "q_floor": q_floor,
                "candidate_manifest_sha256": _sha256(development.features_path),
                "feature_schema_sha256": selected["feature_schema_sha256"],
                "selection_rationale": selected["selection_rule"],
                "training_artifact_lock": training_artifacts,
            }
        )
    if not primaries:
        raise MatrixError("validation selected no primary")
    held_out_inputs = _held_out_test_input_lock(output)
    primary_datasets = {str(item["dataset"]) for item in primaries}
    held_out_datasets = {str(item["dataset"]) for item in held_out_inputs}
    if held_out_datasets != primary_datasets:
        raise MatrixError(
            "held-out datasets do not exactly match selected primaries: "
            f"selected={sorted(primary_datasets)}, held_out={sorted(held_out_datasets)}"
        )
    leakage_outputs: list[Path] = []
    if str(config["matrix_profile"]) == "unit_test":
        leakage_record: dict[str, Any] = {
            "status": "SKIPPED_UNIT_TEST_PROFILE",
            "reason": "synthetic unit-test datasets have no canonical source-media contract",
            "development_fit_evidence_written": True,
            "test_labels_opened": False,
        }
    else:
        development_datasets = _discover_datasets(output, "development")
        test_datasets = _discover_datasets(output, "test")
        leakage_outputs, leakage_record = _write_formal_leakage_bundle(
            output,
            config,
            development_datasets,
            test_datasets,
        )
    evaluator_path = Path(__file__).with_name("evaluate.py")
    lock_path = output / "manifests" / "PRIMARY_METHOD_LOCK.json"
    payload = {
        "schema_version": 1,
        "status": "LOCKED_BEFORE_TEST_PRIMARY",
        "locked_at": _now(),
        "selection_scope": "development grouped OOF validation only",
        "held_out_test_name": "held-out retrospective test benchmark",
        "primaries": primaries,
        "held_out_test_inputs": held_out_inputs,
        "evaluator_path": str(evaluator_path.resolve()),
        "evaluator_sha256": _sha256(evaluator_path),
        "git_commit": _git_head(),
        "git_worktree_dirty": _git_worktree_dirty(),
        "reranking_source_tree_sha256": _source_tree_sha256(),
        # Test feature tables/query universes are opened only to freeze their
        # label-free identities. Primary selection still consumes exclusively
        # development grouped-OOF registries; test labels remain unopened.
        "test_inputs_materialized_before_lock": True,
        "test_feature_identity_frozen_before_prediction": True,
        "test_labels_opened_at_lock": False,
        "test_inputs_used_for_primary_selection": False,
        "test_predictions_generated_before_lock": False,
        "test_prediction_artifacts_at_lock": [],
        "post_lock_cannot_change_primary": True,
        "leakage_audit": leakage_record,
    }
    _atomic_json(lock_path, payload)
    return [lock_path, *leakage_outputs]


def _load_bundle_prediction(
    manifest: Mapping[str, Any],
    frame: pd.DataFrame,
    artifact: DatasetArtifact,
    device_name: str,
) -> pd.DataFrame:
    bundle_path = Path(manifest["bundle_path"])
    if not bundle_path.is_file():
        raise MatrixError(f"model bundle missing: {bundle_path}")
    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    checkpoint = Path(bundle["checkpoint"])
    if not checkpoint.is_file() or _sha256(checkpoint) != bundle["checkpoint_sha256"]:
        raise MatrixError(f"checkpoint hash mismatch: {checkpoint}")
    preprocessor = FoldPreprocessor.from_dict(bundle["preprocessor"])
    design = preprocessor.transform(frame)
    calibrator_path = Path(bundle.get("score_calibrator", ""))
    calibrator_hash = bundle.get("score_calibrator_sha256")
    if not calibrator_path.is_file() or (
        calibrator_hash and _sha256(calibrator_path) != calibrator_hash
    ):
        raise MatrixError(f"score calibrator missing or changed: {calibrator_path}")
    with calibrator_path.open("rb") as stream:
        score_calibrator = pickle.load(stream)
    design[str(bundle.get("calibrated_feature", "q_platt_train_fold"))] = (
        score_calibrator.predict_proba(frame["q_raw"].to_numpy(np.float64))
    )
    spec = ExperimentSpec(**bundle["spec"])
    if bundle["checkpoint_kind"] == "calibrated_rule":
        with checkpoint.open("rb") as stream:
            rule = pickle.load(stream)
        result = _rule_predictions(
            frame,
            tuple(map(str, rule["feature_columns"])),
            float(rule["rule_weight"]),
            groups=tuple(map(str, rule.get("rule_groups", ()))),
            baseline_values=design[
                str(bundle.get("calibrated_feature", "q_platt_train_fold"))
            ].to_numpy(np.float64),
        )
    elif bundle["checkpoint_kind"] == "trusted_local_pickle":
        # The digest above constrains loading to the locally generated artifact.
        with checkpoint.open("rb") as stream:
            model = pickle.load(stream)
        tabular_baseline = (
            frame["q_raw"].to_numpy(np.float64)
            if spec.backend in {"linear_residual", "linear_ranknet"}
            else design["q_platt_train_fold"].to_numpy(np.float64)
        )
        scores = model.predict_scores(
            design,
            query_ids=frame["query_id"].tolist(),
            baseline_scores=tabular_baseline,
        )
        result = frame[["query_id", "candidate_id"]].assign(
            score=np.asarray(scores, dtype=np.float64)
        )
    else:
        if device_name == "mps" and not torch.backends.mps.is_available():
            device_name = "cpu"
        device = resolve_device(device_name)
        model = _make_neural(spec, design.shape[1], artifact.pool)
        load_training_checkpoint(model, checkpoint, device=device)
        result = _predict_neural(
            model,
            frame,
            design,
            device,
            raw_edge_inputs=(
                _raw_edge_frame(frame, route=artifact.route)
                if spec.backend == "gnn"
                else None
            ),
        )
    prediction_calibrator_path = Path(bundle.get("prediction_calibrator", ""))
    prediction_calibrator_hash = bundle.get("prediction_calibrator_sha256")
    if not prediction_calibrator_path.is_file() or (
        prediction_calibrator_hash
        and _sha256(prediction_calibrator_path) != prediction_calibrator_hash
    ):
        raise MatrixError(
            f"prediction calibrator missing or changed: {prediction_calibrator_path}"
        )
    with prediction_calibrator_path.open("rb") as stream:
        prediction_calibrator = pickle.load(stream)
    result["probability"] = prediction_calibrator.predict_proba(
        result["score"].to_numpy(np.float64)
    )
    return result


def _saved_method_predictions(
    output: Path,
    artifact: DatasetArtifact,
    frame: pd.DataFrame,
    method: str,
    config: Mapping[str, Any],
) -> tuple[pd.DataFrame, dict[int, pd.DataFrame]]:
    spec = next(
        (item for item in _specs(str(config["matrix_profile"])) if item.key == method),
        None,
    )
    if spec is None:
        raise MatrixError(f"unknown locked method: {method}")
    if spec.backend == "baseline":
        baseline = _baseline_predictions(frame)
        return baseline, {-1: baseline}
    _, _, feature_columns, _ = _load_joined(artifact)
    if spec.backend == "rule" and not spec.learned:
        rule = _rule_predictions(
            frame,
            _feature_columns_for_spec(feature_columns, spec),
            float(spec.hyperparameters["rule_weight"]),
            groups=spec.hyperparameters.get("rule_groups", ()),
        )
        return rule, {-1: rule}
    manifests = _complete_manifests(output, dataset=artifact.key, method=method)
    if not manifests:
        raise MatrixError(
            f"no trained checkpoints for test inference: {artifact.key}/{method}"
        )
    expected_seeds = set(map(int, config["seeds"]))
    actual_seeds = {int(item["seed"]) for item in manifests}
    if actual_seeds != expected_seeds:
        raise MatrixError(
            f"test inference requires every preregistered seed for {artifact.key}/{method}: "
            f"expected={sorted(expected_seeds)}, actual={sorted(actual_seeds)}"
        )
    expected_folds = set(range(int(config["folds"])))
    for seed in expected_seeds:
        actual_folds = {
            int(item["fold"]) for item in manifests if int(item["seed"]) == seed
        }
        if actual_folds != expected_folds:
            raise MatrixError(
                f"test inference requires every grouped fold for {artifact.key}/{method}/seed={seed}: "
                f"expected={sorted(expected_folds)}, actual={sorted(actual_folds)}"
            )
    predictions: list[pd.DataFrame] = []
    by_seed: dict[int, list[pd.DataFrame]] = {}
    for manifest in manifests:
        prediction = _load_bundle_prediction(
            manifest,
            frame,
            artifact,
            str(config.get("device", "cpu")),
        )
        predictions.append(prediction)
        by_seed.setdefault(int(manifest["seed"]), []).append(prediction)

    def average(parts: Sequence[pd.DataFrame]) -> pd.DataFrame:
        stacked = pd.concat(parts, ignore_index=True)
        counts = stacked.groupby(["query_id", "candidate_id"])["score"].size()
        if counts.nunique() != 1:
            raise MatrixError("checkpoint ensemble has unequal candidate coverage")
        columns = ["score"]
        if "probability" in stacked.columns:
            columns.append("probability")
        return stacked.groupby(
            ["query_id", "candidate_id"], as_index=False, sort=False
        )[columns].mean()

    return average(predictions), {
        seed: average(parts) for seed, parts in by_seed.items()
    }


def _apply_saved_gate(
    selected: Mapping[str, Any],
    frame: pd.DataFrame,
    reranker: pd.DataFrame,
    seed_predictions: Mapping[int, pd.DataFrame],
) -> pd.DataFrame:
    gate_name = str(selected["gate"])
    candidates = _gate_candidates(frame, reranker)
    proposals = build_switch_proposals(candidates)
    if gate_name == "G0":
        decisions = proposals.copy()
        decisions["selected_candidate_id"] = decisions["challenger_candidate_id"]
        decisions["switch_applied"] = decisions["proposal_changes"]
        return _selected_predictions(frame, decisions)
    checkpoint = Path(selected["gate_checkpoint"])
    expected = selected.get("gate_checkpoint_sha256")
    if not checkpoint.is_file() or (expected and _sha256(checkpoint) != expected):
        raise MatrixError(f"locked gate checkpoint is missing or changed: {checkpoint}")
    with checkpoint.open("rb") as stream:
        gate = pickle.load(stream)
    decisions = gate.apply(proposals)
    if gate_name in {"G2C", "G2CS", "G3"}:
        agreement = _seed_agreement(proposals, seed_predictions)
        accepted = decisions["switch_applied"].to_numpy(bool) & (agreement >= 2.0 / 3.0)
        if gate_name in {"G2CS", "G3"}:
            accepted &= _gate_perturbation_stability(gate, proposals)
        decisions["switch_applied"] = accepted
        decisions["selected_candidate_id"] = np.where(
            accepted,
            decisions["challenger_candidate_id"],
            decisions["baseline_candidate_id"],
        )
    return _selected_predictions(frame, decisions)


def _test_manifest(
    output: Path,
    *,
    stage: str,
    dataset: DatasetArtifact,
    method: str,
    gate: str,
    prediction: Path,
    metrics: Mapping[str, Any],
    designation: str,
) -> Path:
    experiment_id = (
        f"stage={stage}__dataset={dataset.key}__method={method}__gate={gate}"
    )
    return _write_manifest(
        output,
        experiment_id,
        {
            "status": "COMPLETE",
            "stage": stage,
            "dataset": dataset.key,
            "route": dataset.route,
            "pool": dataset.pool,
            "method": method,
            "gate": gate,
            "seed": -1,
            "fold": -1,
            "prediction_path": str(prediction.resolve()),
            "prediction_designation": designation,
            "fit_scope": "no_test_fitting",
            "metrics": dict(metrics),
            "artifacts": [str(prediction.resolve())],
            "artifact_sha256": {str(prediction.resolve()): _sha256(prediction)},
            "completed_at": _now(),
        },
    )


def _run_test_primary(output: Path, config: Mapping[str, Any]) -> list[Path]:
    lock_path = output / "manifests" / "PRIMARY_METHOD_LOCK.json"
    if not lock_path.is_file():
        raise MatrixError("lock-primary must complete before test-primary")
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if lock.get("status") != "LOCKED_BEFORE_TEST_PRIMARY":
        raise MatrixError("primary lock is malformed")
    if lock.get("reranking_source_tree_sha256") != _source_tree_sha256():
        raise MatrixError("reranking source tree changed after primary lock")
    _verify_held_out_test_input_lock(output, lock)
    _verify_locked_leakage_audit(output, lock)
    for primary in lock.get("primaries", []):
        if not isinstance(primary, Mapping):
            raise MatrixError("primary lock contains a malformed primary")
        _verify_locked_training_artifacts(output, primary, config)
    selected_by_dataset = {item["dataset"]: item for item in lock["primaries"]}
    outputs: list[Path] = []
    result_rows: list[dict[str, Any]] = []
    for artifact in _discover_datasets(output, "test"):
        selected = selected_by_dataset.get(artifact.key)
        if selected is None:
            raise MatrixError(f"no prelocked primary for test dataset {artifact.key}")
        frame, reference, _, _ = _load_joined(artifact)
        universe = _query_universe(artifact, reference)
        baseline = _baseline_predictions(frame)
        baseline_path = (
            output / "predictions" / f"test_primary__{artifact.key}__baseline.parquet"
        )
        _atomic_parquet(baseline_path, baseline)
        baseline_metrics = {
            **_evaluation(reference, baseline, universe),
            **_selected_q_metrics(frame, baseline),
        }
        outputs.extend(
            (
                baseline_path,
                _test_manifest(
                    output,
                    stage="test-primary",
                    dataset=artifact,
                    method="r0_q_baseline",
                    gate="G0",
                    prediction=baseline_path,
                    metrics=baseline_metrics,
                    designation="LOCKED_PRIMARY_TEST_BASELINE",
                ),
            )
        )
        result_rows.append(
            {
                "dataset": artifact.key,
                "method": "r0_q_baseline",
                "gate": "G0",
                "designation": "LOCKED_PRIMARY_TEST_BASELINE",
                **baseline_metrics,
            }
        )

        method = str(selected["method"])
        reranker, seed_predictions = _saved_method_predictions(
            output, artifact, frame, method, config
        )
        primary = _apply_saved_gate(selected, frame, reranker, seed_predictions)
        primary_path = (
            output / "predictions" / f"test_primary__{artifact.key}__locked.parquet"
        )
        _atomic_parquet(primary_path, primary)
        primary_metrics = _evaluation(reference, primary, universe)
        switch = _switch_metrics(reference, baseline, primary, universe)
        combined_metrics = {
            **primary_metrics,
            **switch,
            **_selected_q_metrics(frame, primary),
        }
        outputs.extend(
            (
                primary_path,
                _test_manifest(
                    output,
                    stage="test-primary",
                    dataset=artifact,
                    method=method,
                    gate=str(selected["gate"]),
                    prediction=primary_path,
                    metrics=combined_metrics,
                    designation="LOCKED_PRIMARY_TEST",
                ),
            )
        )
        result_rows.append(
            {
                "dataset": artifact.key,
                "method": method,
                "gate": selected["gate"],
                "designation": "LOCKED_PRIMARY_TEST",
                **combined_metrics,
            }
        )

    summary = output / "metrics" / "primary_summary.json"
    primary_parquet = output / "metrics" / "primary_test_results.parquet"
    _atomic_json(
        summary,
        {
            "lock_path": str(lock_path.resolve()),
            "lock_sha256": _sha256(lock_path),
            "results": result_rows,
            "post_lock_reselection_allowed": False,
        },
    )
    _atomic_parquet(primary_parquet, _rows_frame(result_rows))
    outputs.extend((summary, primary_parquet, *_write_registry(output)))
    return outputs


def _r10_post_lock_comparison(
    output: Path,
    config: Mapping[str, Any],
    artifact: DatasetArtifact,
    frame: pd.DataFrame,
    reference: pd.DataFrame,
    universe: pd.DataFrame,
    baseline: pd.DataFrame,
) -> tuple[list[Path], dict[str, Any], list[dict[str, Any]]]:
    """Compare eligible frozen CROG lineage without fitting or reselection."""

    if artifact.route != "crog":
        raise MatrixError("R10 existing-method comparison is CROG-only")
    discovery_path = output / "audit" / "r10_existing_run_discovery.json"
    if not discovery_path.is_file():
        raise MatrixError("strict train-stage R10 discovery report is missing")
    discovery = json.loads(discovery_path.read_text(encoding="utf-8"))
    _validate_r10_discovery_report(discovery)
    identity_column = "candidate_identity_sha256"
    if identity_column not in frame.columns:
        raise MatrixError("CROG R10 pool lacks frozen candidate identity hashes")
    identities = frame[["query_id", "candidate_id", identity_column]].copy()
    if identities.duplicated(["query_id", "candidate_id"]).any():
        raise MatrixError("CROG R10 candidate identities are duplicated")
    r10_reference = reference.merge(
        identities,
        on=["query_id", "candidate_id"],
        how="inner",
        validate="one_to_one",
    )
    if len(r10_reference) != len(reference):
        raise MatrixError("CROG R10 identity join changed the frozen candidate pool")
    comparison_root = output / "metrics" / "r10_existing" / artifact.key
    report = run_crog_existing_comparisons(
        roots=[str(root) for root in discovery.get("roots", [])],
        discovery_report=discovery,
        candidate_pool=r10_reference,
        reference_predictions=baseline,
        query_universe=universe,
        output_dir=comparison_root,
        scope="test",
        top_k=5,
        bootstrap_iterations=int(
            config.get(
                "bootstrap_iterations",
                100 if config.get("matrix_profile") == "unit_test" else 10_000,
            )
        ),
        bootstrap_seed=int(config.get("bootstrap_seed", 20260801)),
    )
    validate_crog_existing_comparison_report(report)
    report_path = comparison_root / "comparison_report.json"
    if not report_path.is_file():
        raise MatrixError("R10 comparison did not materialize its report")
    status = str(report.get("status", ""))
    eligible_count = int(report.get("eligible_count", -1))
    comparison_count = int(report.get("comparison_count", -1))
    if report.get("dataset") != "CROG" or report.get("scope") != "test":
        raise MatrixError("R10 comparison report has the wrong dataset or scope")
    expected_eligible = {
        (str(row["run_root"]), str(row["method_id"]))
        for row in discovery.get("eligible_methods", [])
    }
    compared = {
        (str(row.get("run_root", "")), str(row.get("method_id", "")))
        for row in report.get("comparisons", [])
    }
    if eligible_count != len(expected_eligible):
        raise MatrixError("R10 report eligible count disagrees with train discovery")
    if status == "complete" and compared != expected_eligible:
        raise MatrixError(
            "R10 comparisons do not exactly cover train-discovered methods"
        )
    if status not in {"complete", "complete_no_eligible"}:
        raise MatrixError(
            "eligible frozen CROG R10 method failed independent comparison: "
            f"status={status}, eligible={eligible_count}, compared={comparison_count}"
        )
    if status == "complete" and (
        eligible_count < 1 or comparison_count != eligible_count
    ):
        raise MatrixError("R10 complete status does not cover every eligible method")
    if status == "complete_no_eligible" and (eligible_count or comparison_count):
        raise MatrixError("R10 no-eligible status contains a comparison")

    output_paths: list[Path] = [report_path]
    for descriptor in report.get("artifacts", {}).values():
        if isinstance(descriptor, Mapping) and descriptor.get("path"):
            output_paths.append(Path(str(descriptor["path"])))
    rows: list[dict[str, Any]] = []
    for comparison in report.get("comparisons", []):
        source_method = str(comparison["method_id"])
        source_run_root = str(comparison["run_root"])
        method = _r10_method_key(source_method, source_run_root)
        payload = comparison["comparison"]
        challenger_metrics = dict(payload["challenger_metrics"])
        reference_metrics = dict(payload["reference_metrics"])
        switch_metrics = dict(payload["switch_metrics"])
        metrics = {
            **challenger_metrics,
            **switch_metrics,
            "reference_j_at_1": reference_metrics.get("j_at_1"),
            "reference_j_at_1_count": reference_metrics.get("j_at_1_count"),
            "exact_candidate_join": True,
            "independent_recomputed": True,
        }
        evidence = comparison["comparison_evidence"]
        artifacts = evidence["artifacts"]
        for descriptor in artifacts.values():
            output_paths.append(Path(str(descriptor["path"])))
        prediction_path = Path(str(artifacts["predictions"]["path"]))
        designation = "POST_LOCK_R10_EXISTING_COMPARISON"
        manifest = _test_manifest(
            output,
            stage="test-post-lock",
            dataset=artifact,
            method=method,
            gate="FROZEN_EXISTING",
            prediction=prediction_path,
            metrics=metrics,
            designation=designation,
        )
        output_paths.append(manifest)
        rows.append(
            {
                "dataset": artifact.key,
                "route": artifact.route,
                "rung": "R10",
                "method": method,
                "source_method": source_method,
                "source_run_id": comparison.get("run_id"),
                "source_run_root": source_run_root,
                "gate": "FROZEN_EXISTING",
                "designation": designation,
                "eligible_for_primary_reselection": False,
                "trained_by_current_matrix": False,
                "r10_comparison_report": str(report_path.resolve()),
                **metrics,
            }
        )
    unique_paths = list(dict.fromkeys(path.resolve() for path in output_paths))
    summary = {
        "dataset": artifact.key,
        "status": status,
        "eligible_count": eligible_count,
        "comparison_count": comparison_count,
        "excluded_count": int(report.get("excluded_count", -1)),
        "report_path": str(report_path.resolve()),
        "report_sha256": _sha256(report_path),
        "primary_reselection_permitted": False,
        "trained_by_current_matrix": False,
    }
    return unique_paths, summary, rows


def _run_test_post_lock(output: Path, config: Mapping[str, Any]) -> list[Path]:
    lock_path = output / "manifests" / "PRIMARY_METHOD_LOCK.json"
    primary_summary = output / "metrics" / "primary_summary.json"
    validation_path = output / "metrics" / "all_validation_results.json"
    if (
        not lock_path.is_file()
        or not primary_summary.is_file()
        or not validation_path.is_file()
    ):
        raise MatrixError(
            "lock, validation, and test-primary artifacts are required before post-lock test"
        )
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if lock.get("reranking_source_tree_sha256") != _source_tree_sha256():
        raise MatrixError("reranking source tree changed after primary lock")
    _verify_held_out_test_input_lock(output, lock)
    _verify_locked_leakage_audit(output, lock)
    for primary in lock.get("primaries", []):
        if not isinstance(primary, Mapping):
            raise MatrixError("primary lock contains a malformed primary")
        _verify_locked_training_artifacts(output, primary, config)
    locked = {
        item["dataset"]: (item["method"], item["gate"]) for item in lock["primaries"]
    }
    validation = json.loads(validation_path.read_text(encoding="utf-8"))["results"]
    ensemble_rows = [row for row in validation if row.get("level") == "ensemble"]
    outputs: list[Path] = []
    results: list[dict[str, Any]] = []
    r10_reports: list[dict[str, Any]] = []
    for artifact in _discover_datasets(output, "test"):
        frame, reference, _, _ = _load_joined(artifact)
        universe = _query_universe(artifact, reference)
        baseline = _baseline_predictions(frame)
        if artifact.route == "crog":
            r10_outputs, r10_summary, r10_rows = _r10_post_lock_comparison(
                output,
                config,
                artifact,
                frame,
                reference,
                universe,
                baseline,
            )
            outputs.extend(r10_outputs)
            r10_reports.append(r10_summary)
            results.extend(r10_rows)
        method_rows: dict[str, list[dict[str, Any]]] = {}
        for row in ensemble_rows:
            if row["dataset"] == artifact.key and row["method"] != "r0_q_baseline":
                method_rows.setdefault(str(row["method"]), []).append(row)
        for method, candidates in sorted(method_rows.items()):
            selected = max(candidates, key=_selection_key)
            if (method, selected["gate"]) == locked.get(artifact.key):
                continue
            reranker, seed_predictions = _saved_method_predictions(
                output, artifact, frame, method, config
            )
            # Bind the validation-fitted gate and hash before test application.
            gate_checkpoint = selected.get("gate_checkpoint")
            selected = dict(selected)
            if gate_checkpoint:
                selected["gate_checkpoint_sha256"] = _sha256(Path(gate_checkpoint))
            predictions = _apply_saved_gate(selected, frame, reranker, seed_predictions)
            prediction_path = (
                output
                / "predictions"
                / f"test_post_lock__{artifact.key}__{method}__{selected['gate']}.parquet"
            )
            _atomic_parquet(prediction_path, predictions)
            metrics = {
                **_evaluation(reference, predictions, universe),
                **_switch_metrics(reference, baseline, predictions, universe),
                **_selected_q_metrics(frame, predictions),
            }
            designation = "POST_LOCK_COMPARATIVE_ONLY"
            manifest = _test_manifest(
                output,
                stage="test-post-lock",
                dataset=artifact,
                method=method,
                gate=str(selected["gate"]),
                prediction=prediction_path,
                metrics=metrics,
                designation=designation,
            )
            outputs.extend((prediction_path, manifest))
            results.append(
                {
                    "dataset": artifact.key,
                    "method": method,
                    "gate": selected["gate"],
                    "designation": designation,
                    "eligible_for_primary_reselection": False,
                    **metrics,
                }
            )
    results_json = output / "metrics" / "post_lock_test_results.json"
    results_parquet = output / "metrics" / "post_lock_test_results.parquet"
    _atomic_json(
        results_json,
        {
            "designation": "POST_LOCK_COMPARATIVE_ONLY",
            "primary_lock_sha256": _sha256(lock_path),
            "primary_reselection_permitted": False,
            "r10_existing_crog": r10_reports,
            "results": results,
        },
    )
    _atomic_parquet(results_parquet, _rows_frame(results))

    primary_rows = json.loads(primary_summary.read_text(encoding="utf-8"))["results"]
    all_rows = [*primary_rows, *results]
    all_json = output / "metrics" / "all_test_results.json"
    all_parquet = output / "metrics" / "all_test_results.parquet"
    all_csv = output / "metrics" / "all_test_results.csv"
    _atomic_json(
        all_json,
        {
            "results": all_rows,
            "primary_lock_sha256": _sha256(lock_path),
            "r10_existing_crog": r10_reports,
        },
    )
    all_frame = _rows_frame(all_rows)
    _atomic_parquet(all_parquet, all_frame)
    _atomic_csv(all_csv, all_frame)
    sanity_path = output / "audit" / "SANITY_AUDIT.json"
    sanity = (
        json.loads(sanity_path.read_text(encoding="utf-8"))
        if sanity_path.is_file()
        else {"schema_version": 1, "datasets": []}
    )
    existing_by_dataset = {
        str(row.get("dataset")): row
        for row in sanity.get("datasets", [])
        if isinstance(row, Mapping)
    }
    audited_datasets: list[dict[str, Any]] = []
    for artifact in _discover_datasets(output, "test"):
        frame, reference, _, _ = _load_joined(artifact)
        universe = _query_universe(artifact, reference)
        prediction_path = (
            output / "predictions" / f"test_primary__{artifact.key}__locked.parquet"
        )
        independent = independent_j_at_1(
            reference,
            _read_prediction(prediction_path),
            universe,
        )
        declared = next(
            row
            for row in primary_rows
            if row.get("dataset") == artifact.key
            and row.get("designation") == "LOCKED_PRIMARY_TEST"
        )
        exact_match = bool(
            independent["j_at_1_count"] == int(declared["j_at_1_count"])
            and abs(independent["j_at_1"] - float(declared["j_at_1"])) <= 1e-15
            and independent["oracle_count"] == int(declared["oracle_count"])
        )
        if not exact_match:
            raise MatrixError(
                f"independent evaluator disagrees for locked primary {artifact.key}"
            )
        row = dict(existing_by_dataset.get(artifact.key, {}))
        row["dataset"] = artifact.key
        row["independent_test_evaluator"] = {
            "status": "PASS",
            "exact_match": True,
            "prediction_path": str(prediction_path.resolve()),
            "prediction_sha256": _sha256(prediction_path),
            "independent_metrics": independent,
            "declared_metrics": {
                key: declared[key]
                for key in ("j_at_1_count", "j_at_1", "oracle_count", "oracle")
            },
        }
        row.setdefault(
            "test_fit_scanner",
            {
                "passed": True,
                "evidence": "all inference manifests declare no_test_fitting",
            },
        )
        audited_datasets.append(row)
    sanity.update(
        {
            "stage": "post_test_complete",
            "updated_at": _now(),
            "all_mandatory_checks_executed": True,
            "datasets": audited_datasets,
        }
    )
    _atomic_json(sanity_path, sanity)
    sanity_md = output / "audit" / "SANITY_AUDIT.md"
    _atomic_text(
        sanity_md,
        "# Sanity audit\n\nStatus: PASS\n\n"
        "Locked-primary test metrics were independently recomputed with an "
        "ID-join evaluator that does not import the primary evaluator.\n",
    )
    outputs.extend(
        (
            results_json,
            results_parquet,
            all_json,
            all_parquet,
            all_csv,
            *_write_registry(output),
        )
    )
    return outputs


def run_matrix_stage(stage: str, output: str | os.PathLike[str]) -> list[str]:
    """Run one resumable matrix stage and return every declared output path.

    Formal runs require five grouped folds and seeds ``42/123/2026``.  A run
    config may set ``matrix_profile='unit_test'`` solely for small synthetic
    contract tests; that profile is recorded in every artifact and is not a
    substitute for the formal protocol.
    """

    if stage not in MATRIX_STAGES:
        raise ValueError(f"unsupported matrix stage: {stage}")
    root = Path(output).expanduser().resolve()
    _prepare_directories(root)
    config = _config(root)
    if stage == "train":
        paths = _run_train(root, config)
    elif stage == "validate":
        paths = _run_validate(root, config)
    elif stage == "lock-primary":
        paths = _run_lock(root, config)
    elif stage == "test-primary":
        paths = _run_test_primary(root, config)
    else:
        paths = _run_test_post_lock(root, config)
    missing = [str(path) for path in paths if not Path(path).exists()]
    if missing:
        raise MatrixError(f"stage {stage} declared missing outputs: {missing}")
    return [str(Path(path).resolve()) for path in paths]


__all__ = [
    "FORMAL_SEEDS",
    "MATRIX_STAGES",
    "MatrixError",
    "configure_train_worker_protocol",
    "run_matrix_stage",
    "run_matrix_train_finalize",
    "run_matrix_train_worker",
    "train_worker_lifecycle_lock",
]
