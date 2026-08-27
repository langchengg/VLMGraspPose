"""Leakage-safe post-feature analysis for frozen GraspNet 6-DoF pools.

This module starts *after* candidate feature extraction and official per-candidate
labelling.  It deliberately owns no dataset, VGN, or evaluator logic.  Its
inputs are content-addressed files, and every comparison retains the exact same
candidate IDs and geometry hashes.  Fixture-scoped runs are useful for tests,
but are explicitly barred from producing paper reports or plots.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import math
from pathlib import Path
import re
from typing import Any, Literal, Mapping, Sequence

import numpy as np
import pandas as pd

from .features import (
    StableMissingValueImputer,
    assert_no_gt_leakage,
    load_feature_schema,
    select_feature_columns,
)
from .io import atomic_json, atomic_text, canonical_sha256, sha256_file
from .metrics import (
    derive_graded_relevance,
    evaluate_target_rankings,
    mcnemar_exact,
    paired_intervention_outcomes,
    paired_metric_deltas,
    scene_cluster_bootstrap,
)
from .ranker import (
    FORMAL_SEEDS,
    GradedLightGBMLambdaRank,
    ValidationData,
    contiguous_group_sizes,
)

INPUT_SCHEMA = "graspnet6d_post_feature_input_v1"
ANALYSIS_SCHEMA = "graspnet6d_post_feature_analysis_v1"
STATE_SCHEMA = "graspnet6d_post_feature_state_v1"
FORMAL_SCOPE = "formal_real_data"
FIXTURE_SCOPE = "fixture_only"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")

_PARTITIONS = ("train", "validation", "test")
_BASE_ROW_COLUMNS = {
    "partition",
    "scene_id",
    "group_id",
    "candidate_id",
    "geometry_sha256",
    "native_rank",
    "native_score",
    "collision",
    "pose_valid",
    "friction_required",
    "relevance",
}
_TARGET_ID_COLUMNS = {"target_object_id", "associated_object_id"}
_ALLOWED_TUNABLES = {
    "num_leaves",
    "learning_rate",
    "n_estimators",
    "min_child_samples",
    "feature_fraction",
}
_EXPECTED_GROUNDING = (
    "oracle_gt_mask",
    "hifics_zero_shot_mask",
    "hifics_adapted_mask",
)
_OWNED_OUTPUT_NAMES = {
    "analysis_manifest.json",
    "analysis_state.json",
    "resolved_analysis_config.json",
    "native_predictions.csv",
    "b1_predictions.csv",
    "reranked_predictions.csv",
    "gated_predictions.csv",
    "oracle_predictions.csv",
    "metrics.csv",
    "metrics.json",
    "per_group_metrics.csv",
    "paired_outcomes.csv",
    "bootstrap_results.csv",
    "significance_tests.json",
    "ablation_results.csv",
    "failure_taxonomy.csv",
    "failure_summary.json",
    "frozen_pool_audit.json",
    "model_selection.json",
}


class AnalysisInputError(ValueError):
    """An input is stale, incomplete, malformed, or leakage-prone."""


class SplitLeakageError(AnalysisInputError):
    """A scene, group, candidate, or declared row partition crosses splits."""


class FrozenPoolError(AnalysisInputError):
    """A prediction no longer represents the exact frozen candidate pool."""


class FormalReportRefused(RuntimeError):
    """Formal prose/plots were requested for incomplete or fixture evidence."""


@dataclass(frozen=True)
class InputFile:
    path: Path
    sha256: str


@dataclass(frozen=True)
class PartitionInput:
    rows: InputFile
    group_universe: InputFile


@dataclass(frozen=True)
class AnalysisInputManifest:
    source_path: Path
    manifest_sha256: str
    run_id: str
    status: Literal["COMPLETE", "BLOCKED", "FAILED"]
    scope: Literal["formal_real_data", "fixture_only"]
    fixture_only: bool
    feature_schema: InputFile
    partitions: Mapping[str, PartitionInput]
    provenance: Mapping[str, str]


def _default_config_grid() -> tuple[Mapping[str, Any], ...]:
    # Four deliberately small, preregistrable corners of the checked-in search
    # space.  Callers can provide another bounded list, but never a test-driven
    # callback or arbitrary model option.
    return (
        {
            "num_leaves": 15,
            "learning_rate": 0.03,
            "n_estimators": 200,
            "min_child_samples": 10,
            "feature_fraction": 0.8,
        },
        {
            "num_leaves": 15,
            "learning_rate": 0.05,
            "n_estimators": 500,
            "min_child_samples": 20,
            "feature_fraction": 1.0,
        },
        {
            "num_leaves": 31,
            "learning_rate": 0.03,
            "n_estimators": 500,
            "min_child_samples": 20,
            "feature_fraction": 0.8,
        },
        {
            "num_leaves": 63,
            "learning_rate": 0.05,
            "n_estimators": 200,
            "min_child_samples": 10,
            "feature_fraction": 1.0,
        },
    )


@dataclass(frozen=True)
class GateOperatingPointSpec:
    """Dependency-light serialization of the existing gate operating point."""

    lambda_harm: float
    utility_threshold: float
    score_margin_threshold: float
    reliability_threshold: float
    stability_threshold: float
    minimum_seed_votes: int = 2


def _default_gate_points() -> tuple[GateOperatingPointSpec, ...]:
    return tuple(
        GateOperatingPointSpec(
            lambda_harm=lambda_harm,
            utility_threshold=0.0,
            score_margin_threshold=0.0,
            reliability_threshold=0.5,
            stability_threshold=2.0 / 3.0,
            minimum_seed_votes=2,
        )
        for lambda_harm in (1.0, 2.0, 4.0)
    )


@dataclass(frozen=True)
class AnalysisConfig:
    """All post-feature choices fixed before validation/test evaluation."""

    config_grid: tuple[Mapping[str, Any], ...] = field(
        default_factory=_default_config_grid
    )
    seeds: tuple[int, ...] = FORMAL_SEEDS
    primary_seed: int = FORMAL_SEEDS[0]
    early_stopping_rounds: int = 50
    max_k: int = 50
    bootstrap_iterations: int = 10_000
    bootstrap_seed: int = FORMAL_SEEDS[0]
    all_missing_fill_values: Mapping[str, float] = field(default_factory=dict)
    b1_center_probability_min: float = 0.5
    b1_min_closing_fraction: float = 0.0
    b1_min_points_between_fingers: float = 0.0
    grounding_target_coverage_min: float = 0.5
    tsdf_valid_voxel_fraction_min: float = 0.01
    gate_operating_points: tuple[GateOperatingPointSpec, ...] = field(
        default_factory=_default_gate_points
    )
    attempt_gate_fit: bool = True

    def validated(self, *, scope: str) -> "AnalysisConfig":
        if tuple(self.seeds) != FORMAL_SEEDS or self.primary_seed != FORMAL_SEEDS[0]:
            raise AnalysisInputError(
                f"training seeds are locked to {FORMAL_SEEDS} with primary {FORMAL_SEEDS[0]}"
            )
        if not self.config_grid or len(self.config_grid) > 48:
            raise AnalysisInputError("validation config_grid must contain 1..48 trials")
        if scope == FORMAL_SCOPE and len(self.config_grid) < 2:
            raise AnalysisInputError(
                "formal validation selection requires at least two trials"
            )
        normalised: list[dict[str, Any]] = []
        for trial in self.config_grid:
            if not isinstance(trial, Mapping):
                raise AnalysisInputError("every validation trial must be a mapping")
            unknown = sorted(set(trial).difference(_ALLOWED_TUNABLES))
            missing = sorted(_ALLOWED_TUNABLES.difference(trial))
            if unknown or missing:
                raise AnalysisInputError(
                    f"ranker trial keys differ from the locked small search: "
                    f"missing={missing}, unknown={unknown}"
                )
            candidate = {str(key): trial[key] for key in sorted(trial)}
            if int(candidate["num_leaves"]) <= 1:
                raise AnalysisInputError("num_leaves must exceed one")
            if (
                int(candidate["n_estimators"]) <= 0
                or int(candidate["min_child_samples"]) <= 0
            ):
                raise AnalysisInputError(
                    "tree counts and min_child_samples must be positive"
                )
            for name in ("learning_rate", "feature_fraction"):
                value = float(candidate[name])
                if (
                    not math.isfinite(value)
                    or value <= 0.0
                    or (name == "feature_fraction" and value > 1.0)
                ):
                    raise AnalysisInputError(f"invalid {name} in validation trial")
            normalised.append(candidate)
        hashes = [canonical_sha256(item) for item in normalised]
        if len(hashes) != len(set(hashes)):
            raise AnalysisInputError("validation config_grid contains duplicate trials")
        if self.early_stopping_rounds <= 0 or self.max_k < 50:
            raise AnalysisInputError(
                "early stopping must be positive and max_k at least 50"
            )
        if self.bootstrap_iterations <= 0:
            raise AnalysisInputError("bootstrap_iterations must be positive")
        if scope == FORMAL_SCOPE and self.bootstrap_iterations != 10_000:
            raise AnalysisInputError(
                "formal analysis requires exactly 10,000 bootstrap replicates"
            )
        numeric = (
            self.b1_center_probability_min,
            self.b1_min_closing_fraction,
            self.b1_min_points_between_fingers,
            self.grounding_target_coverage_min,
            self.tsdf_valid_voxel_fraction_min,
        )
        if not np.isfinite(np.asarray(numeric, dtype=float)).all():
            raise AnalysisInputError("analysis thresholds must be finite")
        if not 0.0 <= self.b1_center_probability_min <= 1.0:
            raise AnalysisInputError("B1 center threshold must lie in [0,1]")
        if not 0.0 <= self.grounding_target_coverage_min <= 1.0:
            raise AnalysisInputError("grounding coverage threshold must lie in [0,1]")
        if self.tsdf_valid_voxel_fraction_min < 0.0:
            raise AnalysisInputError("TSDF valid-voxel threshold must be non-negative")
        if not self.gate_operating_points:
            raise AnalysisInputError(
                "at least one validation-only gate operating point is required"
            )
        for point in self.gate_operating_points:
            if not isinstance(point, GateOperatingPointSpec):
                raise AnalysisInputError(
                    "gate operating points must use GateOperatingPointSpec"
                )
            values = np.asarray(
                [
                    point.lambda_harm,
                    point.utility_threshold,
                    point.score_margin_threshold,
                    point.reliability_threshold,
                    point.stability_threshold,
                ],
                dtype=float,
            )
            if not np.isfinite(values).all():
                raise AnalysisInputError("gate operating-point values must be finite")
            if point.lambda_harm not in {1.0, 2.0, 4.0}:
                raise AnalysisInputError("gate lambda_harm must be 1, 2, or 4")
            if (
                not 0.0 <= point.reliability_threshold <= 1.0
                or not 0.0 <= point.stability_threshold <= 1.0
            ):
                raise AnalysisInputError(
                    "gate probability thresholds must lie in [0,1]"
                )
            if point.minimum_seed_votes != 2:
                raise AnalysisInputError(
                    "gate consensus is locked to at least 2-of-3 seeds"
                )
        return self

    def artifact(self) -> dict[str, Any]:
        return {
            "config_grid": [dict(item) for item in self.config_grid],
            "seeds": list(self.seeds),
            "primary_seed": self.primary_seed,
            "early_stopping_rounds": self.early_stopping_rounds,
            "max_k": self.max_k,
            "bootstrap_iterations": self.bootstrap_iterations,
            "bootstrap_seed": self.bootstrap_seed,
            "all_missing_fill_values": dict(self.all_missing_fill_values),
            "b1": {
                "center_probability_min": self.b1_center_probability_min,
                "min_closing_fraction_exclusive": self.b1_min_closing_fraction,
                "min_points_between_fingers_exclusive": self.b1_min_points_between_fingers,
            },
            "failure_thresholds": {
                "grounding_target_coverage_min": self.grounding_target_coverage_min,
                "tsdf_valid_voxel_fraction_min": self.tsdf_valid_voxel_fraction_min,
            },
            "gate_operating_points": [
                asdict(item) for item in self.gate_operating_points
            ],
            "attempt_gate_fit": self.attempt_gate_fit,
        }


@dataclass(frozen=True)
class AnalysisResult:
    output_dir: Path
    input_manifest: AnalysisInputManifest
    analysis_fingerprint: str
    output_manifest: Mapping[str, Any]
    resumed: bool


def _strict_keys(
    value: Mapping[str, Any], expected: set[str], description: str
) -> None:
    observed = set(value)
    missing = sorted(expected - observed)
    unknown = sorted(observed - expected)
    if missing or unknown:
        raise AnalysisInputError(
            f"{description} fields differ from schema: missing={missing}, unknown={unknown}"
        )


def _digest(value: Any, description: str) -> str:
    text = str(value)
    if _SHA256.fullmatch(text) is None:
        raise AnalysisInputError(f"{description} must be a lowercase SHA-256 digest")
    return text


def _input_file(raw: Mapping[str, Any], *, base: Path, description: str) -> InputFile:
    _strict_keys(raw, {"path", "sha256"}, description)
    path = Path(str(raw["path"])).expanduser()
    if not path.is_absolute():
        path = base / path
    path = path.resolve()
    expected = _digest(raw["sha256"], f"{description}.sha256")
    try:
        observed = sha256_file(path)
    except (OSError, ValueError) as error:
        raise AnalysisInputError(f"invalid {description}: {path}: {error}") from error
    if observed != expected:
        raise AnalysisInputError(
            f"stale {description}: expected {expected}, observed {observed}: {path}"
        )
    return InputFile(path=path, sha256=expected)


def load_analysis_input_manifest(path: str | Path) -> AnalysisInputManifest:
    """Load a strict, content-addressed post-feature input manifest."""

    source = Path(path).expanduser().resolve()
    try:
        manifest_hash = sha256_file(source)
        raw = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError) as error:
        raise AnalysisInputError(
            f"cannot load analysis input manifest {source}: {error}"
        ) from error
    if not isinstance(raw, Mapping):
        raise AnalysisInputError("analysis input manifest must be a JSON object")
    _strict_keys(
        raw,
        {
            "schema_version",
            "run_id",
            "status",
            "scope",
            "fixture_only",
            "feature_schema",
            "partitions",
            "provenance",
        },
        "analysis input manifest",
    )
    if raw["schema_version"] != INPUT_SCHEMA:
        raise AnalysisInputError(f"unsupported input schema {raw['schema_version']!r}")
    run_id = str(raw["run_id"]).strip()
    if not run_id:
        raise AnalysisInputError("run_id must be non-empty")
    status = str(raw["status"])
    if status not in {"COMPLETE", "BLOCKED", "FAILED"}:
        raise AnalysisInputError("input status must be COMPLETE, BLOCKED, or FAILED")
    scope = str(raw["scope"])
    if scope not in {FORMAL_SCOPE, FIXTURE_SCOPE}:
        raise AnalysisInputError(f"scope must be {FORMAL_SCOPE!r} or {FIXTURE_SCOPE!r}")
    if not isinstance(raw["fixture_only"], bool):
        raise AnalysisInputError("fixture_only must be boolean")
    fixture_only = bool(raw["fixture_only"])
    if (scope == FIXTURE_SCOPE) != fixture_only:
        raise AnalysisInputError("scope and fixture_only disagree")
    if scope == FORMAL_SCOPE and re.search(
        r"fixture|synthetic|dummy|unit[-_]?test|smoke", run_id, re.I
    ):
        raise AnalysisInputError(
            "a fixture-labelled run_id cannot claim formal_real_data scope"
        )
    base = source.parent
    if not isinstance(raw["feature_schema"], Mapping):
        raise AnalysisInputError(
            "feature_schema must be a content-addressed file object"
        )
    feature_schema = _input_file(
        raw["feature_schema"], base=base, description="feature schema"
    )
    partitions_raw = raw["partitions"]
    if not isinstance(partitions_raw, Mapping):
        raise AnalysisInputError("partitions must be an object")
    _strict_keys(partitions_raw, set(_PARTITIONS), "partitions")
    partitions: dict[str, PartitionInput] = {}
    for name in _PARTITIONS:
        entry = partitions_raw[name]
        if not isinstance(entry, Mapping):
            raise AnalysisInputError(f"partition {name!r} must be an object")
        _strict_keys(entry, {"rows", "group_universe"}, f"partition {name}")
        if not isinstance(entry["rows"], Mapping) or not isinstance(
            entry["group_universe"], Mapping
        ):
            raise AnalysisInputError(f"partition {name!r} file entries must be objects")
        partitions[name] = PartitionInput(
            rows=_input_file(entry["rows"], base=base, description=f"{name} rows"),
            group_universe=_input_file(
                entry["group_universe"],
                base=base,
                description=f"{name} group universe",
            ),
        )
    provenance = raw["provenance"]
    if not isinstance(provenance, Mapping) or not provenance:
        raise AnalysisInputError(
            "provenance must contain named upstream SHA-256 values"
        )
    provenance_hashes = {
        str(name): _digest(value, f"provenance.{name}")
        for name, value in provenance.items()
    }
    if any(not name.strip() for name in provenance_hashes):
        raise AnalysisInputError("provenance names must be non-empty")
    return AnalysisInputManifest(
        source_path=source,
        manifest_sha256=manifest_hash,
        run_id=run_id,
        status=status,  # type: ignore[arg-type]
        scope=scope,  # type: ignore[arg-type]
        fixture_only=fixture_only,
        feature_schema=feature_schema,
        partitions=partitions,
        provenance=provenance_hashes,
    )


def _read_table(source: InputFile, description: str) -> pd.DataFrame:
    suffix = source.path.suffix.lower()
    try:
        if suffix == ".csv":
            frame = pd.read_csv(source.path)
        elif suffix == ".jsonl":
            frame = pd.read_json(source.path, lines=True)
        elif suffix in {".parquet", ".pq"}:
            frame = pd.read_parquet(source.path)
        else:
            raise AnalysisInputError(f"unsupported {description} format {suffix!r}")
    except (OSError, ValueError) as error:
        raise AnalysisInputError(
            f"cannot read {description} {source.path}: {error}"
        ) from error
    if frame.empty:
        raise AnalysisInputError(
            f"{description} is empty; placeholder inputs are forbidden"
        )
    return frame


def _truth(values: pd.Series, *, name: str) -> np.ndarray:
    if values.isna().any():
        raise AnalysisInputError(f"{name} contains null values")
    if pd.api.types.is_bool_dtype(values):
        return values.to_numpy(bool)
    lowered = values.astype(str).str.strip().str.lower()
    if not lowered.isin({"true", "false", "1", "0"}).all():
        raise AnalysisInputError(f"{name} must contain booleans")
    return lowered.isin({"true", "1"}).to_numpy(bool)


def _validate_partition_rows(
    frame: pd.DataFrame,
    universe: pd.DataFrame,
    *,
    name: str,
    feature_columns: Sequence[str],
    fixture_only: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    required = _BASE_ROW_COLUMNS | set(feature_columns)
    if not (_TARGET_ID_COLUMNS <= set(frame.columns)) and "target_match" not in frame:
        required |= _TARGET_ID_COLUMNS
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise AnalysisInputError(f"{name} labeled rows lack columns: {missing}")
    if (
        frame[list(required)]
        .drop(columns=list(feature_columns), errors="ignore")
        .isna()
        .any()
        .any()
    ):
        raise AnalysisInputError(
            f"{name} labeled row metadata/supervision contains null values"
        )
    declared = frame["partition"].astype(str).str.lower()
    if not declared.eq(name).all():
        observed = sorted(declared.unique().tolist())
        raise SplitLeakageError(
            f"{name} file contains rows declared as {observed}; test rows may never enter train"
        )
    for column in ("scene_id", "group_id", "candidate_id"):
        if frame[column].astype(str).str.strip().eq("").any():
            raise AnalysisInputError(f"{name}.{column} contains empty identifiers")
    if frame["candidate_id"].astype(str).duplicated().any():
        raise AnalysisInputError(f"{name} candidate IDs must be globally unique")
    if frame.duplicated(["group_id", "candidate_id"]).any():
        raise AnalysisInputError(f"{name} contains duplicate group/candidate rows")
    geometry = frame["geometry_sha256"].astype(str)
    if not geometry.map(lambda value: _SHA256.fullmatch(value) is not None).all():
        raise AnalysisInputError(f"{name} geometry_sha256 values are invalid")
    ranks = pd.to_numeric(frame["native_rank"], errors="coerce").to_numpy(float)
    scores = pd.to_numeric(frame["native_score"], errors="coerce").to_numpy(float)
    if (
        not np.isfinite(ranks).all()
        or not np.equal(ranks, np.floor(ranks)).all()
        or np.any(ranks <= 0)
        or not np.isfinite(scores).all()
    ):
        raise AnalysisInputError(f"{name} native ranks/scores are invalid")
    work = frame.copy()
    work["native_rank"] = ranks.astype(np.int64)
    work["native_score"] = scores
    if work.duplicated(["group_id", "native_rank"]).any():
        raise AnalysisInputError(
            f"{name} native ranks must be unique within each group"
        )
    if "pre_nms_native_rank" in work:
        pre_nms_ranks = pd.to_numeric(
            work["pre_nms_native_rank"], errors="coerce"
        ).to_numpy(float)
        if (
            not np.isfinite(pre_nms_ranks).all()
            or not np.equal(pre_nms_ranks, np.floor(pre_nms_ranks)).all()
            or np.any(pre_nms_ranks <= 0)
        ):
            raise AnalysisInputError(
                f"{name}.pre_nms_native_rank must contain finite positive integers"
            )
        work["pre_nms_native_rank"] = pre_nms_ranks.astype(np.int64)
        if not np.array_equal(
            work["pre_nms_native_rank"].to_numpy(), work["native_rank"].to_numpy()
        ):
            raise AnalysisInputError(
                f"{name}.pre_nms_native_rank disagrees with the frozen native rank"
            )
        if not fixture_only and np.any(pre_nms_ranks > 100):
            raise AnalysisInputError(
                f"{name}.pre_nms_native_rank exceeds the formal K=100 source pool"
            )
    elif not fixture_only:
        raise AnalysisInputError(
            f"{name} formal rows lack the mandatory pre_nms_native_rank A7 contract"
        )
    for _group, part in work.groupby("group_id", sort=False):
        if part["scene_id"].astype(str).nunique() != 1:
            raise AnalysisInputError(f"{name} group crosses scenes")
    _truth(work["collision"], name=f"{name}.collision")
    _truth(work["pose_valid"], name=f"{name}.pose_valid")
    friction = pd.to_numeric(work["friction_required"], errors="coerce").to_numpy(float)
    if not np.isfinite(friction).all():
        raise AnalysisInputError(f"{name}.friction_required must be finite")
    reported = pd.to_numeric(work["relevance"], errors="coerce").to_numpy(float)
    derived = derive_graded_relevance(work)
    if (
        not np.isfinite(reported).all()
        or not np.equal(reported, np.floor(reported)).all()
        or not np.array_equal(reported.astype(np.int32), derived)
    ):
        raise AnalysisInputError(
            f"{name} relevance disagrees with official raw outcomes"
        )
    work["relevance"] = derived
    if "is_fixture" in work:
        marked = _truth(work["is_fixture"], name=f"{name}.is_fixture")
        if bool(marked.any()) != fixture_only or (fixture_only and not marked.all()):
            raise AnalysisInputError(
                f"{name} fixture markers disagree with input scope"
            )
    if "target_match" in work:
        _truth(work["target_match"], name=f"{name}.target_match")
    if "grounding_condition" in work:
        grounding = work["grounding_condition"].astype(str)
        if (
            work["grounding_condition"].isna().any()
            or not grounding.isin(_EXPECTED_GROUNDING).all()
        ):
            raise AnalysisInputError(f"{name} has an unsupported grounding_condition")
    if "tsdf_view_condition" in work:
        views = work["tsdf_view_condition"].astype(str)
        if (
            work["tsdf_view_condition"].isna().any()
            or not views.isin({"single_view", "five_view"}).all()
        ):
            raise AnalysisInputError(f"{name} has an unsupported tsdf_view_condition")

    universe_required = {"partition", "group_id", "scene_id"}
    missing_universe = sorted(universe_required.difference(universe.columns))
    if missing_universe:
        raise AnalysisInputError(
            f"{name} group universe lacks columns: {missing_universe}"
        )
    universe_work = universe.copy()
    if universe_work[list(universe_required)].isna().any().any():
        raise AnalysisInputError(f"{name} group universe contains null identifiers")
    if not universe_work["partition"].astype(str).str.lower().eq(name).all():
        raise SplitLeakageError(f"{name} group universe declares another partition")
    if universe_work["group_id"].astype(str).duplicated().any():
        raise AnalysisInputError(f"{name} group universe IDs must be unique")
    terminal_columns = {"generation_status", "grounding_failure_reason"}
    present_terminal_columns = terminal_columns.intersection(universe_work.columns)
    if present_terminal_columns and present_terminal_columns != terminal_columns:
        raise AnalysisInputError(
            f"{name} group universe has an incomplete generation-status contract"
        )
    if present_terminal_columns:
        generation = universe_work["generation_status"]
        if generation.isna().any():
            raise AnalysisInputError(
                f"{name} group universe contains null generation_status"
            )
        generation_text = generation.astype(str)
        allowed_generation = {
            "completed_vgn_inference",
            "skipped_grounding_failure",
        }
        if not generation_text.isin(allowed_generation).all():
            raise AnalysisInputError(
                f"{name} group universe contains an unsupported generation_status"
            )
        reasons = universe_work["grounding_failure_reason"]
        skipped = generation_text.eq("skipped_grounding_failure")
        allowed_reasons = {
            "empty_predicted_mask",
            "no_valid_predicted_mask_depth",
        }
        if skipped.any() and (
            reasons.loc[skipped].isna().any()
            or not reasons.loc[skipped].astype(str).isin(allowed_reasons).all()
        ):
            raise AnalysisInputError(
                f"{name} grounding terminals contain an invalid failure reason"
            )
        completed_reason = reasons.loc[~skipped]
        if (
            completed_reason.notna() & completed_reason.astype(str).str.strip().ne("")
        ).any():
            raise AnalysisInputError(
                f"{name} completed VGN groups must not claim grounding failure"
            )
        if "grounding_condition" in universe_work and skipped.any():
            skipped_conditions = universe_work.loc[
                skipped, "grounding_condition"
            ].astype(str)
            if skipped_conditions.eq("oracle_gt_mask").any():
                raise AnalysisInputError(
                    f"{name} oracle groups cannot claim predicted grounding terminals"
                )
    if "is_fixture" in universe_work:
        universe_marked = _truth(
            universe_work["is_fixture"], name=f"{name}.universe.is_fixture"
        )
        if bool(universe_marked.any()) != fixture_only or (
            fixture_only and not universe_marked.all()
        ):
            raise AnalysisInputError(
                f"{name} group-universe fixture markers disagree with input scope"
            )
    universe_work["group_id"] = universe_work["group_id"].astype(str)
    universe_work["scene_id"] = universe_work["scene_id"].astype(str)
    row_scene = (
        work.assign(
            group_id=work["group_id"].astype(str), scene_id=work["scene_id"].astype(str)
        )
        .groupby("group_id", sort=False)["scene_id"]
        .first()
    )
    universe_scene = universe_work.set_index("group_id")["scene_id"]
    extra = sorted(set(row_scene.index).difference(universe_scene.index))
    if extra:
        raise AnalysisInputError(
            f"{name} rows contain groups outside the universe: {extra[:5]}"
        )
    mismatched = row_scene[row_scene.ne(universe_scene.reindex(row_scene.index))]
    if len(mismatched):
        raise AnalysisInputError(f"{name} row scenes disagree with group universe")
    for condition_column in ("grounding_condition", "tsdf_view_condition"):
        if condition_column in work and condition_column in universe_work:
            row_condition = work.groupby("group_id", sort=False)[condition_column].agg(
                lambda values: tuple(sorted(set(map(str, values))))
            )
            if row_condition.map(len).ne(1).any():
                raise AnalysisInputError(
                    f"{name} groups cross {condition_column} settings"
                )
            row_condition = row_condition.map(lambda values: values[0])
            universe_condition = universe_work.set_index("group_id")[
                condition_column
            ].astype(str)
            if row_condition.ne(universe_condition.reindex(row_condition.index)).any():
                raise AnalysisInputError(
                    f"{name} row {condition_column} disagrees with group universe"
                )
    return work.sort_values(
        ["group_id", "native_rank", "candidate_id"], kind="mergesort"
    ).reset_index(drop=True), universe_work.reset_index(drop=True)


def validate_split_disjointness(
    rows: Mapping[str, pd.DataFrame], universes: Mapping[str, pd.DataFrame]
) -> None:
    """Reject any scene/group/candidate overlap before model fitting."""

    for left_index, left in enumerate(_PARTITIONS):
        for right in _PARTITIONS[left_index + 1 :]:
            for column in ("scene_id", "group_id", "candidate_id"):
                left_values = set(rows[left][column].astype(str))
                right_values = set(rows[right][column].astype(str))
                overlap = sorted(left_values & right_values)
                if overlap:
                    raise SplitLeakageError(
                        f"{column} overlap between {left} and {right}: {overlap[:5]}"
                    )
            for column in ("scene_id", "group_id"):
                left_values = set(universes[left][column].astype(str))
                right_values = set(universes[right][column].astype(str))
                overlap = sorted(left_values & right_values)
                if overlap:
                    raise SplitLeakageError(
                        f"group-universe {column} overlap between {left} and {right}: "
                        f"{overlap[:5]}"
                    )


def hard_target_support_scores(
    rows: pd.DataFrame, config: AnalysisConfig
) -> pd.DataFrame:
    """Attach the preregistered deterministic B1 score without using labels."""

    required = {
        "group_id",
        "candidate_id",
        "native_score",
        "center_mask_probability",
        "target_point_fraction_inside_closing_volume",
        "target_points_between_fingers",
    }
    missing = sorted(required.difference(rows.columns))
    if missing:
        raise AnalysisInputError(f"B1 target-support inputs are missing: {missing}")
    work = rows.copy().reset_index(drop=True)
    numeric_columns = sorted(required - {"group_id", "candidate_id"})
    numeric = work[numeric_columns].apply(pd.to_numeric, errors="coerce")
    nonnumeric = work[numeric_columns].notna() & numeric.isna()
    if nonnumeric.any().any():
        raise AnalysisInputError("B1 target-support inputs must be numeric or missing")
    values = numeric.to_numpy(float)
    if np.isinf(values).any() or not np.isfinite(numeric["native_score"]).all():
        raise AnalysisInputError(
            "B1 native scores must be finite and support cannot be infinite"
        )
    eligible = (
        numeric["center_mask_probability"].ge(config.b1_center_probability_min)
        & numeric["target_point_fraction_inside_closing_volume"].gt(
            config.b1_min_closing_fraction
        )
        & numeric["target_points_between_fingers"].gt(
            config.b1_min_points_between_fingers
        )
    )
    work["b1_eligible"] = eligible.to_numpy(bool)
    scores = np.empty(len(work), dtype=float)
    for _group_id, indexes in work.groupby("group_id", sort=False).groups.items():
        positions = np.asarray(list(indexes), dtype=np.int64)
        native = work.loc[positions, "native_score"].to_numpy(float)
        bonus = float(np.ptp(native) + 1.0)
        scores[positions] = native + bonus * work.loc[
            positions, "b1_eligible"
        ].to_numpy(float)
    work["b1_score"] = scores
    return work


def assert_frozen_prediction_pool(
    reference: pd.DataFrame, challenger: pd.DataFrame, *, system: str
) -> None:
    """Require exact candidate membership and immutable geometry per group."""

    required = {"group_id", "candidate_id", "geometry_sha256"}
    for name, frame in (("reference", reference), (system, challenger)):
        missing = sorted(required.difference(frame.columns))
        if missing:
            raise FrozenPoolError(f"{name} prediction lacks columns: {missing}")
        if frame.duplicated(["group_id", "candidate_id"]).any():
            raise FrozenPoolError(f"{name} prediction contains duplicate candidates")
    left = (
        reference[list(required)]
        .astype(str)
        .sort_values(["group_id", "candidate_id"], kind="mergesort")
        .reset_index(drop=True)
    )
    right = (
        challenger[list(required)]
        .astype(str)
        .sort_values(["group_id", "candidate_id"], kind="mergesort")
        .reset_index(drop=True)
    )
    if not left[["group_id", "candidate_id"]].equals(
        right[["group_id", "candidate_id"]]
    ):
        raise FrozenPoolError(f"{system} changed frozen candidate membership")
    if not left["geometry_sha256"].equals(right["geometry_sha256"]):
        raise FrozenPoolError(f"{system} changed frozen candidate geometry")


def _frozen_pool_audit(
    reference: pd.DataFrame,
    systems: Mapping[str, pd.DataFrame],
    universe: pd.DataFrame,
) -> dict[str, Any]:
    groups: list[dict[str, Any]] = []
    universe_ids = sorted(set(universe["group_id"].astype(str)))
    extra = sorted(set(reference["group_id"].astype(str)).difference(universe_ids))
    if extra:
        raise FrozenPoolError(
            f"native candidates lie outside the group universe: {extra[:5]}"
        )
    for group_id in universe_ids:
        part = reference.loc[reference["group_id"].astype(str).eq(group_id)]
        records = sorted(
            part[["candidate_id", "geometry_sha256"]]
            .astype(str)
            .to_dict(orient="records"),
            key=lambda row: row["candidate_id"],
        )
        groups.append(
            {
                "group_id": str(group_id),
                "candidate_count": len(records),
                "candidate_geometry_fingerprint": canonical_sha256(records),
            }
        )
    fingerprints: dict[str, str] = {}
    for system, frame in systems.items():
        assert_frozen_prediction_pool(reference, frame, system=system)
        records = sorted(
            frame[["group_id", "candidate_id", "geometry_sha256"]]
            .astype(str)
            .to_dict(orient="records"),
            key=lambda row: (row["group_id"], row["candidate_id"]),
        )
        fingerprints[system] = canonical_sha256(records)
    if len(set(fingerprints.values())) != 1:
        raise FrozenPoolError("system pool fingerprints unexpectedly differ")
    return {
        "schema_version": "graspnet6d_frozen_pool_audit_v1",
        "status": "PASS",
        "assertion": "identical candidate IDs and geometry SHA-256 per group",
        "candidate_count": len(reference),
        "group_count": len(universe_ids),
        "non_empty_group_count": reference["group_id"].astype(str).nunique(),
        "empty_group_count": len(universe_ids)
        - reference["group_id"].astype(str).nunique(),
        "system_fingerprints": fingerprints,
        "groups": groups,
    }


@dataclass
class _VariantFit:
    variant: str
    feature_columns: tuple[str, ...]
    selected_config: Mapping[str, Any]
    selected_config_sha256: str
    validation_trials: list[dict[str, Any]]
    imputer_artifact: Mapping[str, Any]
    validation_predictions: pd.DataFrame
    test_predictions: pd.DataFrame
    models: tuple[GradedLightGBMLambdaRank, ...]


def _validation_universe(universe: pd.DataFrame) -> pd.DataFrame:
    return universe[["group_id", "scene_id"]].copy()


def _attach_seed_scores(
    frame: pd.DataFrame,
    matrices: Sequence[np.ndarray],
    *,
    seeds: Sequence[int],
    prefix: str,
) -> pd.DataFrame:
    if len(matrices) != len(seeds):
        raise AssertionError("one score vector is required per locked seed")
    result = frame.copy()
    score_columns: list[str] = []
    for seed, raw in zip(seeds, matrices, strict=True):
        score = np.asarray(raw, dtype=float)
        if score.shape != (len(frame),) or not np.isfinite(score).all():
            raise RuntimeError(f"ranker seed {seed} returned invalid scores")
        column = f"{prefix}_seed_{seed}"
        result[column] = score
        score_columns.append(column)
    result[prefix] = result[score_columns].mean(axis=1)
    result[f"{prefix}_std"] = result[score_columns].std(axis=1, ddof=0)
    return result


def _fit_variant(
    variant: str,
    *,
    schema: Any,
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    validation_universe: pd.DataFrame,
    config: AnalysisConfig,
) -> _VariantFit:
    """Select only on validation, then touch Test exactly once per final seed."""

    columns = select_feature_columns(schema, variant)
    for name, frame in (("train", train), ("validation", validation), ("test", test)):
        missing = sorted(set(columns).difference(frame.columns))
        if missing:
            raise AnalysisInputError(f"{variant} {name} rows lack features: {missing}")
    assert_no_gt_leakage(columns)
    imputer = StableMissingValueImputer(
        all_missing_fill_values=config.all_missing_fill_values
    )
    train_x = imputer.fit_transform(train[list(columns)])
    validation_x = imputer.transform(validation[list(columns)])
    train_groups = contiguous_group_sizes(train["group_id"], length=len(train))
    validation_groups = contiguous_group_sizes(
        validation["group_id"], length=len(validation)
    )
    models_by_config: dict[str, tuple[GradedLightGBMLambdaRank, ...]] = {}
    trials: list[dict[str, Any]] = []
    for raw_trial in config.config_grid:
        trial = dict(raw_trial)
        trial_sha = canonical_sha256(trial)
        seed_metrics: list[float] = []
        fitted: list[GradedLightGBMLambdaRank] = []
        for seed in config.seeds:
            model = GradedLightGBMLambdaRank(
                seed=seed,
                early_stopping_rounds=config.early_stopping_rounds,
                **trial,
            ).fit_grouped(
                train_x.to_numpy(float),
                train["relevance"].to_numpy(np.int32),
                train_groups,
                validation=ValidationData(
                    validation_x.to_numpy(float),
                    validation["relevance"].to_numpy(np.int32),
                    validation_groups,
                ),
            )
            score = model.predict(
                validation_x.to_numpy(float),
                candidate_ids=validation["candidate_id"].astype(str).tolist(),
            )
            scored = validation.copy()
            scored["_validation_score"] = score
            metrics, _ = evaluate_target_rankings(
                scored,
                _validation_universe(validation_universe),
                score_column="_validation_score",
                max_k=config.max_k,
            )
            seed_metrics.append(float(metrics["ndcg_at_10"]))
            fitted.append(model)
        models_by_config[trial_sha] = tuple(fitted)
        trials.append(
            {
                "config_sha256": trial_sha,
                "config": trial,
                "selection_partition": "validation_only",
                "selection_metric": "mean_ndcg_at_10_across_locked_seeds",
                "seed_values": {
                    str(seed): value
                    for seed, value in zip(config.seeds, seed_metrics, strict=True)
                },
                "mean_ndcg_at_10": float(np.mean(seed_metrics)),
                "std_ndcg_at_10": float(np.std(seed_metrics, ddof=0)),
            }
        )
    # Canonical hash is the final tie-break, independent of trial declaration order.
    selected_trial = sorted(
        trials,
        key=lambda item: (-float(item["mean_ndcg_at_10"]), str(item["config_sha256"])),
    )[0]
    selected_sha = str(selected_trial["config_sha256"])
    selected_models = models_by_config[selected_sha]
    validation_scores = [
        model.predict(
            validation_x.to_numpy(float),
            candidate_ids=validation["candidate_id"].astype(str).tolist(),
        )
        for model in selected_models
    ]
    validation_prediction = _attach_seed_scores(
        validation,
        validation_scores,
        seeds=config.seeds,
        prefix="raw_rerank_score",
    )

    # No Test matrix, score, outcome, or metric is touched until the validation
    # operating point has been frozen above.
    test_x = imputer.transform(test[list(columns)])
    test_scores = [
        model.predict(
            test_x.to_numpy(float),
            candidate_ids=test["candidate_id"].astype(str).tolist(),
        )
        for model in selected_models
    ]
    test_prediction = _attach_seed_scores(
        test,
        test_scores,
        seeds=config.seeds,
        prefix="raw_rerank_score",
    )
    return _VariantFit(
        variant=variant,
        feature_columns=columns,
        selected_config=dict(selected_trial["config"]),
        selected_config_sha256=selected_sha,
        validation_trials=trials,
        imputer_artifact=imputer.artifact(),
        validation_predictions=validation_prediction,
        test_predictions=test_prediction,
        models=selected_models,
    )


def _ranked_top(part: pd.DataFrame, score_column: str) -> pd.Series:
    ordered = part.sort_values(
        [score_column, "native_rank", "candidate_id"],
        ascending=[False, True, True],
        kind="mergesort",
    )
    return ordered.iloc[0]


def _gate_group_evidence(
    rows: pd.DataFrame,
    *,
    config: AnalysisConfig,
    include_outcomes: bool,
) -> pd.DataFrame:
    """Build label-free gate covariates; outcomes are optional and explicit."""

    seed_columns = [f"raw_rerank_score_seed_{seed}" for seed in config.seeds]
    required = {
        "group_id",
        "scene_id",
        "candidate_id",
        "geometry_sha256",
        "native_rank",
        "native_score",
        "raw_rerank_score",
        *seed_columns,
    }
    missing = sorted(required.difference(rows.columns))
    if missing:
        raise AnalysisInputError(f"gate evidence rows lack columns: {missing}")
    records: list[dict[str, Any]] = []
    for group_id, part in rows.groupby("group_id", sort=True):
        native = _ranked_top(part, "native_score")
        challenger = _ranked_top(part, "raw_rerank_score")
        ordered = part.sort_values(
            ["raw_rerank_score", "native_rank", "candidate_id"],
            ascending=[False, True, True],
            kind="mergesort",
        )
        first = float(ordered.iloc[0]["raw_rerank_score"])
        second = (
            first if len(ordered) == 1 else float(ordered.iloc[1]["raw_rerank_score"])
        )
        seed_top_ids = [
            str(_ranked_top(part, column)["candidate_id"]) for column in seed_columns
        ]
        challenger_id = str(challenger["candidate_id"])
        votes = int(sum(candidate_id == challenger_id for candidate_id in seed_top_ids))
        margin = max(0.0, first - second)
        reliability = float(1.0 / (1.0 + math.exp(-float(np.clip(margin, -40, 40)))))
        record: dict[str, Any] = {
            "group_id": str(group_id),
            "scene_id": str(part.iloc[0]["scene_id"]),
            "native_candidate_id": str(native["candidate_id"]),
            "native_geometry_sha256": str(native["geometry_sha256"]),
            "challenger_candidate_id": challenger_id,
            "challenger_geometry_sha256": str(challenger["geometry_sha256"]),
            "ranker_score_margin": margin,
            "native_score_delta": float(
                challenger["native_score"] - native["native_score"]
            ),
            "challenger_exists_numeric": float(
                challenger_id != str(native["candidate_id"])
            ),
            "challenger_reliability": reliability,
            "perturbation_stability": float(votes / len(seed_columns)),
            "seed_challenger_votes": votes,
            "candidate_id_unchanged": bool(
                challenger_id in set(part["candidate_id"].astype(str))
            ),
            "geometry_hash_unchanged": bool(
                str(challenger["geometry_sha256"])
                == str(
                    part.set_index(part["candidate_id"].astype(str)).loc[
                        challenger_id, "geometry_sha256"
                    ]
                )
            ),
            "challenger_exists": bool(challenger_id != str(native["candidate_id"])),
        }
        if include_outcomes:
            evaluated = part.copy()
            _, native_per_group = evaluate_target_rankings(
                evaluated,
                pd.DataFrame(
                    {"group_id": [str(group_id)], "scene_id": [record["scene_id"]]}
                ),
                score_column="native_score",
                max_k=config.max_k,
            )
            _, challenger_per_group = evaluate_target_rankings(
                evaluated,
                pd.DataFrame(
                    {"group_id": [str(group_id)], "scene_id": [record["scene_id"]]}
                ),
                score_column="raw_rerank_score",
                max_k=config.max_k,
            )
            record["native_correct"] = bool(
                native_per_group.iloc[0]["top1_success_mu_1.2"]
            )
            record["challenger_correct"] = bool(
                challenger_per_group.iloc[0]["top1_success_mu_1.2"]
            )
        records.append(record)
    return pd.DataFrame(records)


def _scene_fold_map(scene_ids: Sequence[Any]) -> dict[str, str]:
    scenes = sorted(set(map(str, scene_ids)))
    if len(scenes) < 2:
        raise AnalysisInputError("gate OOF training requires at least two Train scenes")
    fold_count = min(3, len(scenes))
    return {scene: f"fold-{index % fold_count}" for index, scene in enumerate(scenes)}


def _oof_ranker_predictions(
    train: pd.DataFrame,
    *,
    feature_columns: Sequence[str],
    selected_config: Mapping[str, Any],
    config: AnalysisConfig,
) -> tuple[pd.DataFrame, Mapping[str, str]]:
    """Generate scene-held-out Train predictions without early-stop leakage."""

    scene_to_fold = _scene_fold_map(train["scene_id"])
    output = train.copy()
    seed_columns = [f"raw_rerank_score_seed_{seed}" for seed in config.seeds]
    for column in seed_columns:
        output[column] = np.nan
    for fold in sorted(set(scene_to_fold.values())):
        validation_scenes = {
            scene for scene, assigned in scene_to_fold.items() if assigned == fold
        }
        fit_mask = ~train["scene_id"].astype(str).isin(validation_scenes)
        holdout_mask = ~fit_mask
        fit_rows = (
            train.loc[fit_mask]
            .copy()
            .sort_values(["group_id", "native_rank", "candidate_id"], kind="mergesort")
        )
        held_rows = (
            train.loc[holdout_mask]
            .copy()
            .sort_values(["group_id", "native_rank", "candidate_id"], kind="mergesort")
        )
        if fit_rows.empty or held_rows.empty:
            raise AnalysisInputError("gate OOF fold contains an empty row partition")
        imputer = StableMissingValueImputer(
            all_missing_fill_values=config.all_missing_fill_values
        )
        fit_x = imputer.fit_transform(fit_rows[list(feature_columns)])
        held_x = imputer.transform(held_rows[list(feature_columns)])
        group_sizes = contiguous_group_sizes(fit_rows["group_id"], length=len(fit_rows))
        for seed, column in zip(config.seeds, seed_columns, strict=True):
            # No held-out labels are supplied as a validation set.  This is a
            # true OOF prediction, not an early-stopped in-fold prediction.
            model = GradedLightGBMLambdaRank(
                seed=seed,
                early_stopping_rounds=config.early_stopping_rounds,
                **dict(selected_config),
            ).fit_grouped(
                fit_x.to_numpy(float),
                fit_rows["relevance"].to_numpy(np.int32),
                group_sizes,
                validation=None,
            )
            scores = model.predict(
                held_x.to_numpy(float),
                candidate_ids=held_rows["candidate_id"].astype(str).tolist(),
            )
            output.loc[held_rows.index, column] = scores
    if output[seed_columns].isna().any().any():
        raise RuntimeError("gate OOF predictions do not cover every Train candidate")
    output["raw_rerank_score"] = output[seed_columns].mean(axis=1)
    output["raw_rerank_score_std"] = output[seed_columns].std(axis=1, ddof=0)
    return output, scene_to_fold


def _gate_feature_matrix(groups: pd.DataFrame) -> tuple[np.ndarray, tuple[str, ...]]:
    columns = (
        "ranker_score_margin",
        "native_score_delta",
        "challenger_exists_numeric",
    )
    matrix = groups[list(columns)].to_numpy(float)
    if not np.isfinite(matrix).all():
        raise AnalysisInputError("gate covariates must be finite")
    return matrix, columns


def _gate_evidence(groups: pd.DataFrame) -> Any:
    # Lazy import keeps metric-only and fixture runs usable when the legacy
    # gate's optional statsmodels dependency is not installed. A formal gate
    # attempt records that missing dependency and fails closed to native.
    from unified_reranking.gate import GateEvidence

    return GateEvidence(
        score_margin=groups["ranker_score_margin"].to_numpy(float),
        challenger_reliability=groups["challenger_reliability"].to_numpy(float),
        perturbation_stability=groups["perturbation_stability"].to_numpy(float),
        seed_challenger_votes=groups["seed_challenger_votes"].to_numpy(int),
        candidate_id_unchanged=groups["candidate_id_unchanged"].to_numpy(bool),
        geometry_hash_unchanged=groups["geometry_hash_unchanged"].to_numpy(bool),
        challenger_exists=groups["challenger_exists"].to_numpy(bool),
    )


def _fit_and_apply_gate(
    train: pd.DataFrame,
    validation_predictions: pd.DataFrame,
    test_predictions: pd.DataFrame,
    *,
    all_feature_fit: _VariantFit,
    config: AnalysisConfig,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Fit only from Train OOF and select only from Validation; else native."""

    test = test_predictions.copy()
    test_group_ids = sorted(set(test["group_id"].astype(str)))
    fail_closed = np.zeros(len(test_group_ids), dtype=bool)
    test_groups: pd.DataFrame | None = None
    artifact: dict[str, Any] = {
        "schema_version": "graspnet6d_expected_gain_gate_v1",
        "mathematics_source": "src/unified_reranking/gate.py",
        "old_4d_models_or_thresholds_reused": False,
        "transition_training_scope": "train_oof_only",
        "operating_point_selection_scope": "validation_only",
        "test_outcomes_used_for_fit_or_selection": False,
    }
    if not config.attempt_gate_fit:
        artifact.update(
            {
                "status": "FAIL_CLOSED_NATIVE",
                "reason_code": "GATE_FIT_DISABLED",
                "reason": "gate fitting was explicitly disabled before this fixture run",
            }
        )
        switches = fail_closed
    else:
        try:
            from unified_reranking.gate import (
                ConservativeTransitionModel,
                GateOperatingPoint,
                OOFTransitionData,
                gate_switch_mask,
                select_gate_operating_point,
            )

            operating_points = tuple(
                GateOperatingPoint(**asdict(point))
                if isinstance(point, GateOperatingPointSpec)
                else point
                for point in config.gate_operating_points
            )
            oof_rows, scene_to_fold = _oof_ranker_predictions(
                train,
                feature_columns=all_feature_fit.feature_columns,
                selected_config=all_feature_fit.selected_config,
                config=config,
            )
            oof_groups = _gate_group_evidence(
                oof_rows, config=config, include_outcomes=True
            )
            oof_matrix, gate_columns = _gate_feature_matrix(oof_groups)
            transition = ConservativeTransitionModel(seed=config.primary_seed).fit(
                OOFTransitionData(
                    features=oof_matrix,
                    feature_names=gate_columns,
                    native_correct=oof_groups["native_correct"].to_numpy(bool),
                    challenger_correct=oof_groups["challenger_correct"].to_numpy(bool),
                    scene_ids=oof_groups["scene_id"].astype(str).to_numpy(),
                    oof_fold_ids=oof_groups["scene_id"]
                    .astype(str)
                    .map(scene_to_fold)
                    .to_numpy(),
                    prediction_source="train_oof",
                )
            )
            validation_groups = _gate_group_evidence(
                validation_predictions, config=config, include_outcomes=True
            )
            validation_matrix, validation_columns = _gate_feature_matrix(
                validation_groups
            )
            if validation_columns != gate_columns:
                raise AssertionError(
                    "gate feature schema drifted between Train and Validation"
                )
            p_recover, p_harm = transition.predict_probabilities(validation_matrix)
            selection = select_gate_operating_point(
                p_recover,
                p_harm,
                _gate_evidence(validation_groups),
                validation_groups["native_correct"].to_numpy(bool),
                validation_groups["challenger_correct"].to_numpy(bool),
                validation_groups["scene_id"].astype(str).to_numpy(),
                operating_points,
                bootstrap_iterations=config.bootstrap_iterations,
                bootstrap_seed=config.bootstrap_seed,
            )
            artifact.update(
                {
                    "transition_model": transition.artifact(),
                    "train_oof_group_count": len(oof_groups),
                    "train_oof_evidence_sha256": canonical_sha256(
                        oof_groups.sort_values("group_id").to_dict(orient="records")
                    ),
                    "scene_fold_assignment_sha256": canonical_sha256(scene_to_fold),
                    "validation_group_count": len(validation_groups),
                    "validation_evidence_sha256": canonical_sha256(
                        validation_groups.sort_values("group_id").to_dict(
                            orient="records"
                        )
                    ),
                    "selection": selection.artifact(),
                }
            )
            if selection.status != "GO" or selection.selected_operating_point is None:
                artifact.update(
                    {
                        "status": "FAIL_CLOSED_NATIVE",
                        "reason_code": "VALIDATION_NO_GO",
                        "reason": "no validation operating point had a positive bootstrap lower bound",
                    }
                )
                switches = fail_closed
            else:
                # Test covariates are constructed only after Train OOF fitting
                # and Validation operating-point selection are both frozen.
                test_groups = _gate_group_evidence(
                    test_predictions, config=config, include_outcomes=False
                )
                test_matrix, test_columns = _gate_feature_matrix(test_groups)
                if test_columns != gate_columns:
                    raise AssertionError("gate feature schema drifted on Test")
                test_recover, test_harm = transition.predict_probabilities(test_matrix)
                switches = gate_switch_mask(
                    test_recover,
                    test_harm,
                    _gate_evidence(test_groups),
                    selection.selected_operating_point,
                )
                artifact.update(
                    {
                        "status": "GO",
                        "reason_code": None,
                        "reason": None,
                        "test_covariate_sha256": canonical_sha256(
                            test_groups.drop(
                                columns=[
                                    column
                                    for column in (
                                        "native_correct",
                                        "challenger_correct",
                                    )
                                    if column in test_groups
                                ]
                            )
                            .sort_values("group_id")
                            .to_dict(orient="records")
                        ),
                    }
                )
        except Exception as error:
            # Gate infeasibility (not enough transition classes, fold imbalance,
            # or calibration failure) is an expected conservative outcome.  It
            # is recorded, never silently converted to a claimed gated gain.
            artifact.update(
                {
                    "status": "FAIL_CLOSED_NATIVE",
                    "reason_code": "NEW_6D_GATE_EVIDENCE_NOT_FITTABLE",
                    "reason": f"{type(error).__name__}: {error}",
                }
            )
            switches = fail_closed
    selected_group_ids = (
        test_group_ids
        if test_groups is None
        else test_groups["group_id"].astype(str).tolist()
    )
    if len(switches) != len(selected_group_ids):
        raise AssertionError(
            "gate returned one switch decision per group, not candidate"
        )
    switch_by_group = dict(zip(selected_group_ids, switches.astype(bool), strict=True))
    test["gate_switched"] = (
        test["group_id"].astype(str).map(switch_by_group).astype(bool)
    )
    test["gated_score"] = np.where(
        test["gate_switched"], test["raw_rerank_score"], test["native_score"]
    )
    artifact["test_switch_count"] = int(sum(switch_by_group.values()))
    artifact["test_group_count"] = len(selected_group_ids)
    artifact["test_switch_rate"] = float(np.mean(list(switch_by_group.values())))
    return test, artifact


def _system_evaluation(
    rows: pd.DataFrame,
    universe: pd.DataFrame,
    *,
    score_column: str,
    max_k: int,
) -> tuple[dict[str, Any], pd.DataFrame]:
    return evaluate_target_rankings(
        rows,
        _validation_universe(universe),
        score_column=score_column,
        max_k=max_k,
    )


def _metric_row(
    *,
    scope: str,
    system: str,
    score_column: str,
    seed: int | str | None,
    metrics: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "analysis_scope": scope,
        "partition": "test",
        "system": system,
        "score_column": score_column,
        "seed": seed,
        **dict(metrics),
    }


def _evaluate_systems(
    native: pd.DataFrame,
    b1: pd.DataFrame,
    raw: pd.DataFrame,
    gated: pd.DataFrame,
    oracle: pd.DataFrame,
    universe: pd.DataFrame,
    *,
    scope: str,
    config: AnalysisConfig,
) -> tuple[
    list[dict[str, Any]],
    dict[str, pd.DataFrame],
    dict[str, Any],
    pd.DataFrame,
    pd.DataFrame,
    dict[str, Any],
]:
    specifications = {
        "B0_NATIVE": (native, "native_score", None),
        "B1_HARD_TARGET_SUPPORT": (b1, "b1_score", None),
        "R0_RAW": (raw, "raw_rerank_score", "ensemble_mean"),
        "R1_GATED": (gated, "gated_score", "ensemble_gate"),
        "O_ORACLE": (oracle, "oracle_score", None),
    }
    metric_rows: list[dict[str, Any]] = []
    per_group: dict[str, pd.DataFrame] = {}
    metric_maps: dict[str, dict[str, Any]] = {}
    for system, (frame, score_column, seed) in specifications.items():
        metrics, groups = _system_evaluation(
            frame, universe, score_column=score_column, max_k=config.max_k
        )
        metric_maps[system] = metrics
        per_group[system] = groups
        metric_rows.append(
            _metric_row(
                scope=scope,
                system=system,
                score_column=score_column,
                seed=seed,
                metrics=metrics,
            )
        )
    seed_metric_maps: dict[str, Mapping[str, Any]] = {}
    for seed in config.seeds:
        column = f"raw_rerank_score_seed_{seed}"
        metrics, _ = _system_evaluation(
            raw, universe, score_column=column, max_k=config.max_k
        )
        seed_metric_maps[str(seed)] = metrics
        metric_rows.append(
            _metric_row(
                scope=scope,
                system="R0_RAW_SEED",
                score_column=column,
                seed=seed,
                metrics=metrics,
            )
        )
    selected_seed_metrics = (
        "target_p_at_1_mu_0.4",
        "target_p_at_1_mu_0.8",
        "target_p_at_1_mu_1.2",
        "target_graspnet_style_ap_mean_mu_0.2_to_1.2",
        "ndcg_at_10",
        "mrr",
    )
    seed_summary: dict[str, Any] = {
        "seeds": list(config.seeds),
        "primary_seed": config.primary_seed,
        "metrics": {},
    }
    for metric in selected_seed_metrics:
        values = np.asarray(
            [float(seed_metric_maps[str(seed)][metric]) for seed in config.seeds],
            dtype=float,
        )
        seed_summary["metrics"][metric] = {
            "mean": float(values.mean()),
            "std_population": float(values.std(ddof=0)),
            "primary_seed_value": float(
                seed_metric_maps[str(config.primary_seed)][metric]
            ),
            "minimum": float(values.min()),
            "maximum": float(values.max()),
        }

    paired_frames: list[pd.DataFrame] = []
    bootstrap_frames: list[pd.DataFrame] = []
    paired_summaries: dict[str, Any] = {}
    significance: dict[str, Any] = {}
    native_groups = per_group["B0_NATIVE"]
    for challenger in ("B1_HARD_TARGET_SUPPORT", "R0_RAW", "R1_GATED"):
        summary, paired = paired_intervention_outcomes(
            native_groups, per_group[challenger], mu=1.2
        )
        paired.insert(0, "analysis_scope", scope)
        paired.insert(1, "reference", "B0_NATIVE")
        paired.insert(2, "challenger", challenger)
        paired_frames.append(paired)
        paired_summaries[challenger] = summary
        reference_correct = native_groups["top1_success_mu_1.2"].to_numpy(bool)
        challenger_correct = per_group[challenger]["top1_success_mu_1.2"].to_numpy(bool)
        significance[challenger] = mcnemar_exact(reference_correct, challenger_correct)
        deltas = paired_metric_deltas(native_groups, per_group[challenger], mu=1.2)
        bootstrap = scene_cluster_bootstrap(
            deltas,
            delta_columns=(
                "delta_p_at_1",
                "delta_ap",
                "delta_mrr",
                "net_recovered_rate",
            ),
            iterations=config.bootstrap_iterations,
            seed=config.bootstrap_seed,
        )
        bootstrap.insert(0, "analysis_scope", scope)
        bootstrap.insert(1, "reference", "B0_NATIVE")
        bootstrap.insert(2, "challenger", challenger)
        bootstrap_frames.append(bootstrap)
    paired_table = pd.concat(paired_frames, ignore_index=True)
    bootstrap_table = pd.concat(bootstrap_frames, ignore_index=True)
    metric_payload = {
        "schema_version": "graspnet6d_metrics_v1",
        "analysis_scope": scope,
        "metric_source": "raw official per-candidate association/collision/friction rows",
        "systems": metric_maps,
        "raw_seed_metrics": seed_metric_maps,
        "raw_seed_summary": seed_summary,
        "paired_outcomes": paired_summaries,
    }
    significance_payload = {
        "schema_version": "graspnet6d_significance_v1",
        "analysis_scope": scope,
        "success_threshold_mu": 1.2,
        "tests": significance,
    }
    return (
        metric_rows,
        per_group,
        metric_payload,
        paired_table,
        bootstrap_table,
        significance_payload,
    )


def _append_ablation_metrics(
    records: list[dict[str, Any]],
    *,
    scope: str,
    ablation: str,
    setting: str,
    status: str,
    metrics: Mapping[str, Any] | None,
    reason: str | None = None,
    selected_config_sha256: str | None = None,
    actual_candidate_count: int | None = None,
) -> None:
    base = {
        "analysis_scope": scope,
        "ablation": ablation,
        "setting": setting,
        "status": status,
        "reason": reason,
        "selected_config_sha256": selected_config_sha256,
        "actual_candidate_count": actual_candidate_count,
    }
    if metrics is None:
        # An absent experiment gets a status record, not a fabricated numeric
        # value, sentinel zero, or "not-run" string in a metric field.
        records.append({**base, "metric_name": None, "metric_value": None})
        return
    for name, value in metrics.items():
        records.append({**base, "metric_name": str(name), "metric_value": value})


def _run_ablations(
    *,
    schema: Any,
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    universes: Mapping[str, pd.DataFrame],
    all_feature_fit: _VariantFit,
    base_metrics: Mapping[str, Mapping[str, Any]],
    gated_rows: pd.DataFrame,
    gate_artifact: Mapping[str, Any],
    scope: str,
    config: AnalysisConfig,
) -> tuple[pd.DataFrame, dict[str, _VariantFit]]:
    records: list[dict[str, Any]] = []
    fits: dict[str, _VariantFit] = {"A2": all_feature_fit}
    _append_ablation_metrics(
        records,
        scope=scope,
        ablation="A0",
        setting="native_score",
        status="EXECUTED",
        metrics=base_metrics["B0_NATIVE"],
        actual_candidate_count=len(test),
    )
    variants = {
        "A1": "q_only",
        "A2": "all",
        "A3": "drop_semantic",
        "A4": "drop_collision",
        "A5": "drop_local_geometry",
        "A6": "drop_orientation",
    }
    for ablation, variant in variants.items():
        if ablation != "A2":
            try:
                expected_columns = select_feature_columns(schema, variant)
            except ValueError as error:
                _append_ablation_metrics(
                    records,
                    scope=scope,
                    ablation=ablation,
                    setting=variant,
                    status="UNEXECUTED_INPUT_MISSING",
                    metrics=None,
                    reason=f"{type(error).__name__}: {error}",
                )
                continue
            missing_by_partition = {
                name: sorted(set(expected_columns).difference(frame.columns))
                for name, frame in (
                    ("train", train),
                    ("validation", validation),
                    ("test", test),
                )
                if set(expected_columns).difference(frame.columns)
            }
            if missing_by_partition:
                _append_ablation_metrics(
                    records,
                    scope=scope,
                    ablation=ablation,
                    setting=variant,
                    status="UNEXECUTED_INPUT_MISSING",
                    metrics=None,
                    reason=f"missing feature inputs: {missing_by_partition}",
                )
                continue
        fit = (
            all_feature_fit
            if ablation == "A2"
            else _fit_variant(
                variant,
                schema=schema,
                train=train,
                validation=validation,
                test=test,
                validation_universe=universes["validation"],
                config=config,
            )
        )
        fits[ablation] = fit
        metrics, _ = _system_evaluation(
            fit.test_predictions,
            universes["test"],
            score_column="raw_rerank_score",
            max_k=config.max_k,
        )
        _append_ablation_metrics(
            records,
            scope=scope,
            ablation=ablation,
            setting=variant,
            status="EXECUTED",
            metrics=metrics,
            selected_config_sha256=fit.selected_config_sha256,
            actual_candidate_count=len(test),
        )

    if "pre_nms_native_rank" not in test:
        for k in (20, 50, 100):
            _append_ablation_metrics(
                records,
                scope=scope,
                ablation="A7",
                setting=f"top_k_{k}",
                status="UNEXECUTED_INPUT_MISSING",
                metrics=None,
                reason="pre_nms_native_rank is absent; post-NMS rows cannot prove a shared pre-NMS pool",
            )
    else:
        pre_rank = pd.to_numeric(test["pre_nms_native_rank"], errors="coerce").to_numpy(
            float
        )
        if (
            not np.isfinite(pre_rank).all()
            or not np.equal(pre_rank, np.floor(pre_rank)).all()
            or np.any(pre_rank <= 0)
        ):
            raise AnalysisInputError(
                "pre_nms_native_rank must contain finite positive integers"
            )
        for k in (20, 50, 100):
            prediction_pre_rank = pd.to_numeric(
                all_feature_fit.test_predictions["pre_nms_native_rank"],
                errors="coerce",
            ).to_numpy(float)
            selected = all_feature_fit.test_predictions.loc[
                prediction_pre_rank <= k
            ].copy()
            if scope == FORMAL_SCOPE and k == 100 and len(selected) != len(test):
                raise AnalysisInputError(
                    "formal A7 K=100 does not reproduce the complete frozen comparison pool"
                )
            metrics, _ = _system_evaluation(
                selected,
                universes["test"],
                score_column="raw_rerank_score",
                max_k=config.max_k,
            )
            _append_ablation_metrics(
                records,
                scope=scope,
                ablation="A7",
                setting=f"top_k_{k}",
                status="EXECUTED",
                metrics=metrics,
                selected_config_sha256=all_feature_fit.selected_config_sha256,
                actual_candidate_count=len(selected),
            )

    for condition in _EXPECTED_GROUNDING:
        if (
            "grounding_condition" not in test
            or "grounding_condition" not in universes["test"]
        ):
            _append_ablation_metrics(
                records,
                scope=scope,
                ablation="A8",
                setting=condition,
                status="UNEXECUTED_INPUT_MISSING",
                metrics=None,
                reason="grounding_condition is absent from labeled rows or group universe",
            )
            continue
        selected_rows = all_feature_fit.test_predictions.loc[
            test["grounding_condition"].astype(str).eq(condition)
        ].copy()
        selected_universe = (
            universes["test"]
            .loc[universes["test"]["grounding_condition"].astype(str).eq(condition)]
            .copy()
        )
        if selected_universe.empty:
            _append_ablation_metrics(
                records,
                scope=scope,
                ablation="A8",
                setting=condition,
                status="UNEXECUTED_INPUT_MISSING",
                metrics=None,
                reason=f"no Test groups exist for {condition}",
            )
        else:
            metrics, _ = _system_evaluation(
                selected_rows,
                selected_universe,
                score_column="raw_rerank_score",
                max_k=config.max_k,
            )
            _append_ablation_metrics(
                records,
                scope=scope,
                ablation="A8",
                setting=condition,
                status="EXECUTED",
                metrics=metrics,
                selected_config_sha256=all_feature_fit.selected_config_sha256,
                actual_candidate_count=len(selected_rows),
            )

    for view in ("single_view", "five_view"):
        if (
            "tsdf_view_condition" not in test
            or "tsdf_view_condition" not in universes["test"]
        ):
            status = (
                "UNEXECUTED_RESOURCE_DEPENDENT"
                if view == "five_view"
                else "UNEXECUTED_INPUT_MISSING"
            )
            reason = (
                "five-view TSDF artifacts were not supplied; no metric row was fabricated"
                if view == "five_view"
                else "tsdf_view_condition is absent from labeled rows or group universe"
            )
            _append_ablation_metrics(
                records,
                scope=scope,
                ablation="A9",
                setting=view,
                status=status,
                metrics=None,
                reason=reason,
            )
            continue
        selected_rows = all_feature_fit.test_predictions.loc[
            test["tsdf_view_condition"].astype(str).eq(view)
        ].copy()
        selected_universe = (
            universes["test"]
            .loc[universes["test"]["tsdf_view_condition"].astype(str).eq(view)]
            .copy()
        )
        if selected_universe.empty:
            _append_ablation_metrics(
                records,
                scope=scope,
                ablation="A9",
                setting=view,
                status=(
                    "UNEXECUTED_RESOURCE_DEPENDENT"
                    if view == "five_view"
                    else "UNEXECUTED_INPUT_MISSING"
                ),
                metrics=None,
                reason=f"no {view} Test artifacts were supplied",
            )
        else:
            metrics, _ = _system_evaluation(
                selected_rows,
                selected_universe,
                score_column="raw_rerank_score",
                max_k=config.max_k,
            )
            _append_ablation_metrics(
                records,
                scope=scope,
                ablation="A9",
                setting=view,
                status="EXECUTED",
                metrics=metrics,
                selected_config_sha256=all_feature_fit.selected_config_sha256,
                actual_candidate_count=len(selected_rows),
            )

    _append_ablation_metrics(
        records,
        scope=scope,
        ablation="A10",
        setting="raw_reranker",
        status="EXECUTED",
        metrics=base_metrics["R0_RAW"],
        selected_config_sha256=all_feature_fit.selected_config_sha256,
        actual_candidate_count=len(test),
    )
    gated_metrics, _ = _system_evaluation(
        gated_rows,
        universes["test"],
        score_column="gated_score",
        max_k=config.max_k,
    )
    _append_ablation_metrics(
        records,
        scope=scope,
        ablation="A10",
        setting="expected_gain_gate",
        status=(
            "EXECUTED" if gate_artifact["status"] == "GO" else "EXECUTED_FAIL_CLOSED"
        ),
        metrics=gated_metrics,
        reason=(
            None
            if gate_artifact["status"] == "GO"
            else str(gate_artifact.get("reason"))
        ),
        selected_config_sha256=all_feature_fit.selected_config_sha256,
        actual_candidate_count=len(test),
    )
    return pd.DataFrame(records), fits


def _group_optional_value(
    rows: pd.DataFrame, universe_row: pd.Series, column: str
) -> Any | None:
    if column in universe_row.index and pd.notna(universe_row[column]):
        return universe_row[column]
    if column in rows and len(rows):
        values = rows[column].dropna()
        if len(values):
            return values.iloc[0]
    return None


def _optional_bool(value: Any | None) -> bool | None:
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"true", "1"}:
        return True
    if text in {"false", "0"}:
        return False
    raise AnalysisInputError(
        f"failure-taxonomy boolean diagnostic is invalid: {value!r}"
    )


def _failure_taxonomy(
    raw_rows: pd.DataFrame,
    universe: pd.DataFrame,
    native_groups: pd.DataFrame,
    reranked_groups: pd.DataFrame,
    *,
    scope: str,
    config: AnalysisConfig,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    native = native_groups.set_index("group_id")
    reranked = reranked_groups.set_index("group_id")
    records: list[dict[str, Any]] = []
    for universe_row in universe.reset_index(drop=True).itertuples(index=False):
        universe_series = pd.Series(universe_row._asdict())
        group_id = str(universe_series["group_id"])
        scene_id = str(universe_series["scene_id"])
        part = raw_rows.loc[raw_rows["group_id"].astype(str).eq(group_id)]
        native_row = native.loc[group_id]
        reranked_row = reranked.loc[group_id]
        native_success = bool(native_row["top1_success_mu_1.2"])
        reranked_success = bool(reranked_row["top1_success_mu_1.2"])
        oracle_50 = bool(native_row["oracle_at_50_mu_1.2"])
        grounding = _group_optional_value(part, universe_series, "grounding_condition")
        generation_status = _group_optional_value(
            part, universe_series, "generation_status"
        )
        grounding_failure_reason = _group_optional_value(
            part, universe_series, "grounding_failure_reason"
        )
        mask_empty = _optional_bool(
            _group_optional_value(part, universe_series, "mask_empty")
        )
        wrong_mask = _optional_bool(
            _group_optional_value(part, universe_series, "mask_mainly_wrong_object")
        )
        valid_target_depth = _optional_bool(
            _group_optional_value(part, universe_series, "mask_has_valid_target_depth")
        )
        coverage_raw = _group_optional_value(
            part, universe_series, "grounding_target_coverage"
        )
        coverage = None if coverage_raw is None else float(coverage_raw)
        workspace_invalid = _optional_bool(
            _group_optional_value(part, universe_series, "workspace_invalid")
        )
        transform_invalid = _optional_bool(
            _group_optional_value(part, universe_series, "transform_invalid")
        )
        target_in_volume = _optional_bool(
            _group_optional_value(part, universe_series, "target_in_tsdf_volume")
        )
        voxel_raw = _group_optional_value(
            part, universe_series, "tsdf_valid_voxel_fraction"
        )
        voxel_fraction = None if voxel_raw is None else float(voxel_raw)
        visible_geometry_insufficient = _optional_bool(
            _group_optional_value(
                part, universe_series, "visible_geometry_insufficient"
            )
        )
        collisions = (
            np.zeros(0, dtype=bool)
            if part.empty
            else _truth(part["collision"], name=f"{group_id}.collision")
        )
        all_collision = bool(len(collisions) and collisions.all())

        predicted = grounding is not None and str(grounding) != "oracle_gt_mask"
        if predicted and generation_status == "skipped_grounding_failure":
            category, reason = (
                "F1_GROUNDING_FAILURE",
                "predicted grounding terminal: " + str(grounding_failure_reason),
            )
        elif predicted and (
            mask_empty is True
            or wrong_mask is True
            or valid_target_depth is False
            or (
                coverage is not None and coverage < config.grounding_target_coverage_min
            )
        ):
            category, reason = (
                "F1_GROUNDING_FAILURE",
                "predicted-mask grounding gate failed",
            )
        elif (
            workspace_invalid is True
            or transform_invalid is True
            or target_in_volume is False
            or (
                voxel_fraction is not None
                and voxel_fraction < config.tsdf_valid_voxel_fraction_min
            )
        ):
            category, reason = (
                "F2_TSDF_WORKSPACE_FAILURE",
                "TSDF/workspace acceptance gate failed",
            )
        elif all_collision or visible_geometry_insufficient is True:
            category, reason = (
                "F7_UNRECOVERABLE_COLLISION_GEOMETRY_FAILURE",
                "all candidates collide or visible geometry is explicitly insufficient",
            )
        elif part.empty or not oracle_50:
            category, reason = (
                "F3_CANDIDATE_GENERATION_FAILURE",
                "empty pool or no valid target grasp within the frozen top-50 pool",
            )
        elif native_success and not reranked_success:
            category, reason = (
                "F6_RERANKING_HARM",
                "native Top-1 succeeded but reranked Top-1 failed",
            )
        elif not native_success and reranked_success:
            category, reason = (
                "F5_RERANKING_RECOVERY",
                "native Top-1 failed but reranked Top-1 succeeded",
            )
        elif not native_success:
            category, reason = (
                "F4_NATIVE_ORDERING_FAILURE",
                "a valid target candidate exists but native and reranked Top-1 both failed",
            )
        else:
            category, reason = (
                "S0_UNCHANGED_SUCCESS",
                "native and reranked Top-1 both succeeded",
            )
        records.append(
            {
                "analysis_scope": scope,
                "group_id": group_id,
                "scene_id": scene_id,
                "category": category,
                "reason": reason,
                "candidate_count": len(part),
                "oracle_at_50": oracle_50,
                "native_top1_success_mu_1.2": native_success,
                "reranked_top1_success_mu_1.2": reranked_success,
                "all_candidates_collision": all_collision,
            }
        )
    frame = pd.DataFrame(records)
    counts = frame["category"].value_counts().sort_index()
    summary = {
        "schema_version": "graspnet6d_failure_taxonomy_v1",
        "analysis_scope": scope,
        "success_threshold_mu": 1.2,
        "exclusive": True,
        "group_count": len(frame),
        "counts": {str(key): int(value) for key, value in counts.items()},
        "thresholds": {
            "grounding_target_coverage_min": config.grounding_target_coverage_min,
            "tsdf_valid_voxel_fraction_min": config.tsdf_valid_voxel_fraction_min,
        },
    }
    return frame, summary


def _atomic_csv(path: Path, frame: pd.DataFrame) -> Path:
    return atomic_text(
        path,
        frame.to_csv(index=False, lineterminator="\n", float_format="%.17g"),
    )


def _resume_result(
    output_dir: Path,
    input_manifest: AnalysisInputManifest,
    fingerprint: str,
    *,
    resume: bool,
) -> AnalysisResult | None:
    state_path = output_dir / "analysis_state.json"
    if not state_path.exists():
        return None
    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise AnalysisInputError(
            f"cannot read prior analysis state: {error}"
        ) from error
    if not isinstance(state, Mapping) or state.get("schema_version") != STATE_SCHEMA:
        raise AnalysisInputError("prior analysis state has an unsupported schema")
    if state.get("analysis_fingerprint") != fingerprint:
        raise AnalysisInputError(
            "output directory belongs to different inputs/config; use another run directory"
        )
    if not resume:
        raise AnalysisInputError(
            f"analysis state already exists with status {state.get('status')!r}; pass resume=True"
        )
    if state.get("status") != "COMPLETE":
        return None
    manifest_path = output_dir / "analysis_manifest.json"
    if not manifest_path.is_file() or state.get(
        "output_manifest_sha256"
    ) != sha256_file(manifest_path):
        raise AnalysisInputError("completed analysis manifest is missing or stale")
    try:
        output_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise AnalysisInputError(
            f"cannot read completed output manifest: {error}"
        ) from error
    if output_manifest.get("analysis_fingerprint") != fingerprint:
        raise AnalysisInputError("completed output manifest fingerprint mismatch")
    outputs = output_manifest.get("outputs")
    if not isinstance(outputs, Mapping) or not outputs:
        raise AnalysisInputError("completed output manifest contains no output hashes")
    for relative, expected in outputs.items():
        path = output_dir / str(relative)
        if sha256_file(path) != _digest(expected, f"output {relative}"):
            raise AnalysisInputError(
                f"completed output is missing or stale: {relative}"
            )
    return AnalysisResult(
        output_dir=output_dir,
        input_manifest=input_manifest,
        analysis_fingerprint=fingerprint,
        output_manifest=output_manifest,
        resumed=True,
    )


def run_post_feature_analysis(
    input_manifest_path: str | Path,
    output_dir: str | Path,
    *,
    config: AnalysisConfig | None = None,
    resume: bool = False,
) -> AnalysisResult:
    """Run the full frozen-pool analysis from content-addressed labeled rows.

    Fixture inputs execute real code and may emit fixture-scoped diagnostics,
    but :func:`write_formal_analysis_report` will refuse them.  A BLOCKED or
    FAILED upstream run cannot enter model selection at all.
    """

    inputs = load_analysis_input_manifest(input_manifest_path)
    if inputs.status != "COMPLETE":
        raise AnalysisInputError(
            f"post-feature analysis requires upstream status COMPLETE, got {inputs.status}"
        )
    resolved_config = (config or AnalysisConfig()).validated(scope=inputs.scope)
    configuration = resolved_config.artifact()
    fingerprint = canonical_sha256(
        {
            "analysis_schema": ANALYSIS_SCHEMA,
            "input_manifest_sha256": inputs.manifest_sha256,
            "feature_schema_sha256": inputs.feature_schema.sha256,
            "partitions": {
                name: {
                    "rows_sha256": inputs.partitions[name].rows.sha256,
                    "group_universe_sha256": inputs.partitions[
                        name
                    ].group_universe.sha256,
                }
                for name in _PARTITIONS
            },
            "provenance": dict(inputs.provenance),
            "config": configuration,
        }
    )
    root = Path(output_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    if not (root / "analysis_state.json").exists():
        conflicts = sorted(
            name for name in _OWNED_OUTPUT_NAMES if (root / name).exists()
        )
        if conflicts:
            raise AnalysisInputError(
                "analysis-owned outputs exist without a resumable state: "
                f"{conflicts}; use another output directory"
            )
    resumed = _resume_result(root, inputs, fingerprint, resume=resume)
    if resumed is not None:
        return resumed
    atomic_json(
        root / "analysis_state.json",
        {
            "schema_version": STATE_SCHEMA,
            "status": "RUNNING",
            "analysis_fingerprint": fingerprint,
            "input_manifest_sha256": inputs.manifest_sha256,
        },
    )
    try:
        schema = load_feature_schema(inputs.feature_schema.path)
        all_feature_columns = tuple(spec.name for spec in schema)
        rows: dict[str, pd.DataFrame] = {}
        universes: dict[str, pd.DataFrame] = {}
        for name in _PARTITIONS:
            partition = inputs.partitions[name]
            raw_rows = _read_table(partition.rows, f"{name} labeled feature rows")
            raw_universe = _read_table(
                partition.group_universe, f"{name} group universe"
            )
            rows[name], universes[name] = _validate_partition_rows(
                raw_rows,
                raw_universe,
                name=name,
                feature_columns=all_feature_columns,
                fixture_only=inputs.fixture_only,
            )
        validate_split_disjointness(rows, universes)

        all_fit = _fit_variant(
            "all",
            schema=schema,
            train=rows["train"],
            validation=rows["validation"],
            test=rows["test"],
            validation_universe=universes["validation"],
            config=resolved_config,
        )
        native = rows["test"].copy()
        raw = all_fit.test_predictions.copy()
        b1 = hard_target_support_scores(native, resolved_config)
        oracle = native.copy()
        oracle["oracle_score"] = oracle["relevance"].astype(float)
        gated, gate_artifact = _fit_and_apply_gate(
            rows["train"],
            all_fit.validation_predictions,
            raw,
            all_feature_fit=all_fit,
            config=resolved_config,
        )
        for frame in (native, b1, raw, gated, oracle):
            frame.insert(0, "analysis_scope", inputs.scope)
        pool_audit = _frozen_pool_audit(
            native,
            {
                "B0_NATIVE": native,
                "B1_HARD_TARGET_SUPPORT": b1,
                "R0_RAW": raw,
                "R1_GATED": gated,
                "O_ORACLE": oracle,
            },
            universes["test"],
        )
        pool_audit["analysis_scope"] = inputs.scope
        pool_audit["input_manifest_sha256"] = inputs.manifest_sha256

        (
            metric_rows,
            per_group,
            metrics_payload,
            paired_table,
            bootstrap_table,
            significance_payload,
        ) = _evaluate_systems(
            native,
            b1,
            raw,
            gated,
            oracle,
            universes["test"],
            scope=inputs.scope,
            config=resolved_config,
        )
        metrics_payload["gate"] = gate_artifact
        metrics_payload["input_manifest_sha256"] = inputs.manifest_sha256
        ablations, variant_fits = _run_ablations(
            schema=schema,
            train=rows["train"],
            validation=rows["validation"],
            test=rows["test"],
            universes=universes,
            all_feature_fit=all_fit,
            base_metrics=metrics_payload["systems"],
            gated_rows=gated,
            gate_artifact=gate_artifact,
            scope=inputs.scope,
            config=resolved_config,
        )
        failure_table, failure_summary = _failure_taxonomy(
            raw,
            universes["test"],
            per_group["B0_NATIVE"],
            per_group["R0_RAW"],
            scope=inputs.scope,
            config=resolved_config,
        )

        model_selection = {
            "schema_version": "graspnet6d_validation_model_selection_v1",
            "analysis_scope": inputs.scope,
            "selection_partition": "validation_only",
            "test_access_during_selection": False,
            "seeds": list(resolved_config.seeds),
            "primary_seed": resolved_config.primary_seed,
            "variants": {
                ablation: {
                    "variant": fit.variant,
                    "feature_columns": list(fit.feature_columns),
                    "selected_config": dict(fit.selected_config),
                    "selected_config_sha256": fit.selected_config_sha256,
                    "validation_trials": fit.validation_trials,
                    "imputer": dict(fit.imputer_artifact),
                    "rankers": [model.artifact() for model in fit.models],
                }
                for ablation, fit in variant_fits.items()
            },
        }
        per_group_table = pd.concat(
            [
                frame.assign(analysis_scope=inputs.scope, system=system)
                for system, frame in per_group.items()
            ],
            ignore_index=True,
        )

        artifacts: dict[str, Path] = {}

        def csv(name: str, frame: pd.DataFrame) -> None:
            artifacts[name] = _atomic_csv(root / name, frame)

        def json_artifact(name: str, payload: Any) -> None:
            artifacts[name] = atomic_json(root / name, payload)

        csv("native_predictions.csv", native)
        csv("b1_predictions.csv", b1)
        csv("reranked_predictions.csv", raw)
        csv("gated_predictions.csv", gated)
        csv("oracle_predictions.csv", oracle)
        csv("metrics.csv", pd.DataFrame(metric_rows))
        csv("per_group_metrics.csv", per_group_table)
        csv("paired_outcomes.csv", paired_table)
        csv("bootstrap_results.csv", bootstrap_table)
        csv("ablation_results.csv", ablations)
        csv("failure_taxonomy.csv", failure_table)
        json_artifact("metrics.json", metrics_payload)
        json_artifact("significance_tests.json", significance_payload)
        json_artifact("failure_summary.json", failure_summary)
        json_artifact("frozen_pool_audit.json", pool_audit)
        json_artifact("model_selection.json", model_selection)
        json_artifact(
            "resolved_analysis_config.json",
            {
                "schema_version": ANALYSIS_SCHEMA,
                "analysis_scope": inputs.scope,
                "config": configuration,
                "config_sha256": canonical_sha256(configuration),
            },
        )
        output_hashes = {
            str(path.relative_to(root)): sha256_file(path)
            for path in sorted(artifacts.values())
        }
        output_manifest = {
            "schema_version": ANALYSIS_SCHEMA,
            "status": "COMPLETE",
            "analysis_scope": inputs.scope,
            "fixture_only": inputs.fixture_only,
            "run_id": inputs.run_id,
            "analysis_fingerprint": fingerprint,
            "input_manifest_path": str(inputs.source_path),
            "input_manifest_sha256": inputs.manifest_sha256,
            "feature_schema_sha256": inputs.feature_schema.sha256,
            "upstream_provenance": dict(inputs.provenance),
            "split_guards": {
                "scene_disjoint": True,
                "group_disjoint": True,
                "candidate_disjoint": True,
                "declared_partition_checked": True,
                "test_used_for_training_or_selection": False,
            },
            "frozen_pool_status": "PASS",
            "outputs": output_hashes,
            "formal_report_eligible": inputs.scope == FORMAL_SCOPE
            and not inputs.fixture_only,
        }
        output_manifest_path = atomic_json(
            root / "analysis_manifest.json", output_manifest
        )
        atomic_json(
            root / "analysis_state.json",
            {
                "schema_version": STATE_SCHEMA,
                "status": "COMPLETE",
                "analysis_fingerprint": fingerprint,
                "input_manifest_sha256": inputs.manifest_sha256,
                "output_manifest_sha256": sha256_file(output_manifest_path),
            },
        )
        return AnalysisResult(
            output_dir=root,
            input_manifest=inputs,
            analysis_fingerprint=fingerprint,
            output_manifest=output_manifest,
            resumed=False,
        )
    except Exception as error:
        atomic_json(
            root / "analysis_state.json",
            {
                "schema_version": STATE_SCHEMA,
                "status": "FAILED",
                "analysis_fingerprint": fingerprint,
                "input_manifest_sha256": inputs.manifest_sha256,
                "failure_type": type(error).__name__,
                "failure_message": str(error),
            },
        )
        raise


def write_formal_analysis_report(result: AnalysisResult) -> Mapping[str, str]:
    """Write factual RESULTS/plots only for COMPLETE, real, non-fixture evidence."""

    manifest = result.output_manifest
    inputs = result.input_manifest
    if (
        manifest.get("status") != "COMPLETE"
        or inputs.status != "COMPLETE"
        or inputs.scope != FORMAL_SCOPE
        or inputs.fixture_only
        or manifest.get("analysis_scope") != FORMAL_SCOPE
        or manifest.get("formal_report_eligible") is not True
    ):
        raise FormalReportRefused(
            "formal report/plots require a COMPLETE formal_real_data input manifest; "
            "fixture and blocked evidence is diagnostic only"
        )
    output_dir = result.output_dir
    metrics_path = output_dir / "metrics.csv"
    analysis_manifest_path = output_dir / "analysis_manifest.json"
    try:
        state = json.loads(
            (output_dir / "analysis_state.json").read_text(encoding="utf-8")
        )
        analysis_manifest_hash = sha256_file(analysis_manifest_path)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        raise FormalReportRefused(
            f"formal analysis state cannot be verified: {error}"
        ) from error
    if (
        state.get("status") != "COMPLETE"
        or state.get("analysis_fingerprint") != result.analysis_fingerprint
        or state.get("output_manifest_sha256") != analysis_manifest_hash
    ):
        raise FormalReportRefused(
            "formal analysis state/manifest is incomplete or stale"
        )
    expected_metrics_hash = manifest.get("outputs", {}).get("metrics.csv")
    if (
        _SHA256.fullmatch(str(expected_metrics_hash)) is None
        or sha256_file(metrics_path) != expected_metrics_hash
    ):
        raise FormalReportRefused("formal metrics are missing or stale")
    metrics = pd.read_csv(metrics_path)
    if metrics.empty or metrics["analysis_scope"].astype(str).ne(FORMAL_SCOPE).any():
        raise FormalReportRefused("metrics are empty or contain non-formal scope rows")
    raw_seed = metrics.loc[metrics["system"].eq("R0_RAW_SEED")].copy()
    if set(pd.to_numeric(raw_seed["seed"]).astype(int)) != set(FORMAL_SEEDS):
        raise FormalReportRefused(
            "formal metrics do not contain all three locked ranker seeds"
        )
    native_row = metrics.loc[metrics["system"].eq("B0_NATIVE")]
    if len(native_row) != 1:
        raise FormalReportRefused(
            "formal metrics lack exactly one native reference row"
        )
    native_value = float(native_row.iloc[0]["target_p_at_1_mu_1.2"])
    plot_rows: list[dict[str, Any]] = []
    for seed in FORMAL_SEEDS:
        plot_rows.append(
            {
                "condition": "all_test_groups",
                "system": "B0 native",
                "seed": seed,
                "value": native_value,
                "status": "COMPLETE",
            }
        )
        row = raw_seed.loc[pd.to_numeric(raw_seed["seed"]).astype(int).eq(seed)]
        plot_rows.append(
            {
                "condition": "all_test_groups",
                "system": "R0 raw reranker",
                "seed": seed,
                "value": float(row.iloc[0]["target_p_at_1_mu_1.2"]),
                "status": "COMPLETE",
            }
        )
    from .reporting import plot_system_comparison

    plot_paths = plot_system_comparison(
        pd.DataFrame(plot_rows),
        output_dir / "figures" / "target_p_at_1_mu_1.2_by_seed",
        ylabel="Target P@1 (mu=1.2)",
        title="Frozen 6-DoF candidate pool",
    )
    ensemble = metrics.loc[metrics["system"].isin(["B0_NATIVE", "R0_RAW", "R1_GATED"])]
    keyed = ensemble.set_index("system")
    results_text = f"""# Results

Evidence scope: `{FORMAL_SCOPE}`. All values below were read from
`metrics.csv`, which was recomputed from content-addressed raw evaluator rows.

| system | Target P@1 (mu=0.4) | Target P@1 (mu=0.8) | Target P@1 (mu=1.2) | target-specific GraspNet-style mean AP |
|---|---:|---:|---:|---:|
| B0 native | {float(keyed.loc["B0_NATIVE", "target_p_at_1_mu_0.4"]):.6f} | {float(keyed.loc["B0_NATIVE", "target_p_at_1_mu_0.8"]):.6f} | {float(keyed.loc["B0_NATIVE", "target_p_at_1_mu_1.2"]):.6f} | {float(keyed.loc["B0_NATIVE", "target_graspnet_style_ap_mean_mu_0.2_to_1.2"]):.6f} |
| R0 raw reranker | {float(keyed.loc["R0_RAW", "target_p_at_1_mu_0.4"]):.6f} | {float(keyed.loc["R0_RAW", "target_p_at_1_mu_0.8"]):.6f} | {float(keyed.loc["R0_RAW", "target_p_at_1_mu_1.2"]):.6f} | {float(keyed.loc["R0_RAW", "target_graspnet_style_ap_mean_mu_0.2_to_1.2"]):.6f} |
| R1 gated reranker | {float(keyed.loc["R1_GATED", "target_p_at_1_mu_0.4"]):.6f} | {float(keyed.loc["R1_GATED", "target_p_at_1_mu_0.8"]):.6f} | {float(keyed.loc["R1_GATED", "target_p_at_1_mu_1.2"]):.6f} | {float(keyed.loc["R1_GATED", "target_graspnet_style_ap_mean_mu_0.2_to_1.2"]):.6f} |

The paired intervals and exact tests are in `bootstrap_results.csv` and
`significance_tests.json`. The frozen-pool proof is in
`frozen_pool_audit.json`. No official unconditional GraspNet leaderboard AP is
claimed here.
"""
    results_path = atomic_text(output_dir / "RESULTS.md", results_text)
    report_outputs = {
        "RESULTS.md": sha256_file(results_path),
        str(plot_paths["pdf"].relative_to(output_dir)): sha256_file(plot_paths["pdf"]),
        str(plot_paths["png"].relative_to(output_dir)): sha256_file(plot_paths["png"]),
    }
    report_manifest = {
        "schema_version": "graspnet6d_formal_analysis_report_v1",
        "status": "COMPLETE",
        "analysis_scope": FORMAL_SCOPE,
        "analysis_fingerprint": result.analysis_fingerprint,
        "analysis_manifest_sha256": sha256_file(analysis_manifest_path),
        "outputs": report_outputs,
    }
    report_manifest_path = atomic_json(
        output_dir / "formal_report_manifest.json", report_manifest
    )
    return {
        **report_outputs,
        "formal_report_manifest.json": sha256_file(report_manifest_path),
    }


__all__ = [
    "ANALYSIS_SCHEMA",
    "FIXTURE_SCOPE",
    "FORMAL_SCOPE",
    "INPUT_SCHEMA",
    "AnalysisConfig",
    "AnalysisInputError",
    "AnalysisInputManifest",
    "AnalysisResult",
    "FormalReportRefused",
    "FrozenPoolError",
    "GateOperatingPointSpec",
    "InputFile",
    "PartitionInput",
    "SplitLeakageError",
    "assert_frozen_prediction_pool",
    "hard_target_support_scores",
    "load_analysis_input_manifest",
    "run_post_feature_analysis",
    "validate_split_disjointness",
    "write_formal_analysis_report",
]
