"""Post-hoc five-fold scene-grouped CV over the frozen 6-DoF source run.

The implementation deliberately starts from the source run's aggregate raw
feature table and official candidate-label table.  The already assembled
analysis input is used only to recover and cross-check the immutable geometry
SHA-256 attached to each candidate.  No candidate generation, VGN inference,
feature extraction, or official geometry evaluation occurs here.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
from typing import Any, Mapping, Sequence
import warnings

import numpy as np
import pandas as pd

from graspnet6d.features import StableMissingValueImputer, assert_no_gt_leakage
from graspnet6d.io import canonical_sha256, sha256_file
from graspnet6d.metrics import (
    evaluate_target_rankings,
    mcnemar_exact,
    paired_intervention_outcomes,
    paired_metric_deltas,
    scene_cluster_bootstrap,
)
from graspnet6d.ranker import (
    FORMAL_SEEDS,
    LABEL_GAIN,
    GradedLightGBMLambdaRank,
    ValidationData,
    contiguous_group_sizes,
)


SOURCE_RUN_ID = "20260819_221819_graspnet6d_vgn_lambdamart"
PRIMARY_CONDITION = "oracle_gt_mask"
SPLIT_SEED = 20260815
BOOTSTRAP_ITERATIONS = 10_000
BOOTSTRAP_SEED = 20260815
EXPECTED_SCENES = tuple(f"scene_{index:04d}" for index in range(60, 90))
EXPECTED_HYPERPARAMETERS: dict[str, Any] = {
    "num_leaves": 15,
    "learning_rate": 0.03,
    "n_estimators": 200,
    "min_child_samples": 10,
    "feature_fraction": 0.8,
}
EXPECTED_EARLY_STOPPING_ROUNDS = 50

_IDENTITY_COLUMNS = ("group_id", "candidate_id")
_GEOMETRY_COLUMNS = ("geometry_sha256", "gripper_width_m")
_LABEL_COLUMNS = (
    "target_object_id",
    "associated_object_id",
    "target_match",
    "collision",
    "pose_valid",
    "friction_required",
    "relevance",
)


@dataclass(frozen=True)
class SourceData:
    rows: pd.DataFrame
    universe: pd.DataFrame
    feature_columns: tuple[str, ...]
    hyperparameters: Mapping[str, Any]
    early_stopping_rounds: int
    source_hashes: Mapping[str, str]
    source_paths: tuple[Path, ...]


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"cannot serialise {type(value).__name__}")


def _atomic_bytes(path: Path, payload: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise
    return path


def _atomic_json(path: Path, payload: Any) -> Path:
    rendered = json.dumps(
        payload,
        indent=2,
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
        default=_json_default,
    )
    return _atomic_bytes(path, (rendered + "\n").encode("utf-8"))


def _atomic_csv(path: Path, frame: pd.DataFrame) -> Path:
    return _atomic_bytes(path, frame.to_csv(index=False).encode("utf-8"))


def _atomic_parquet(path: Path, frame: pd.DataFrame) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".parquet", dir=path.parent
    )
    os.close(descriptor)
    try:
        frame.to_parquet(temporary, index=False)
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise
    return path


def _assert_isolated_run_dir(repo_root: Path, run_dir: Path) -> tuple[Path, Path]:
    repo = repo_root.expanduser().resolve()
    output = run_dir.expanduser().resolve()
    allowed = (repo / "artifacts" / "robustness_suite").resolve()
    source = (repo / "artifacts" / "graspnet6d" / SOURCE_RUN_ID).resolve()
    if output == allowed or allowed not in output.parents:
        raise ValueError("6-DoF CV output must be a child robustness-suite run directory")
    if output == source or source in output.parents or output in source.parents:
        raise ValueError("robustness output must not overlap the locked source run")
    preregistration = output / "PRE_REGISTRATION.md"
    if not preregistration.is_file():
        raise FileNotFoundError(
            "PRE_REGISTRATION.md must already exist before scene-CV execution"
        )
    return repo, output


def make_scene_folds(
    scene_ids: Sequence[Any], *, split_seed: int = SPLIT_SEED
) -> list[dict[str, Any]]:
    """Create the pre-registered deterministic 19/5/6 scene folds."""

    scenes = sorted({str(scene) for scene in scene_ids})
    if len(scenes) != 30 or any(not scene for scene in scenes):
        raise ValueError("scene-CV requires exactly 30 unique non-empty scenes")
    shuffled = np.asarray(scenes, dtype=object)
    np.random.default_rng(int(split_seed)).shuffle(shuffled)
    folds: list[dict[str, Any]] = []
    for fold_index in range(5):
        test = tuple(map(str, shuffled[fold_index * 6 : (fold_index + 1) * 6]))
        remaining = np.asarray(sorted(set(scenes).difference(test)), dtype=object)
        np.random.default_rng(int(split_seed) + fold_index).shuffle(remaining)
        validation = tuple(map(str, remaining[:5]))
        train = tuple(map(str, remaining[5:]))
        folds.append(
            {
                "fold": fold_index,
                "split_seed": int(split_seed),
                "validation_seed": int(split_seed) + fold_index,
                "train_scenes": list(train),
                "validation_scenes": list(validation),
                "test_scenes": list(test),
            }
        )
    validate_scene_folds(folds, scenes)
    return folds


def validate_scene_folds(
    folds: Sequence[Mapping[str, Any]], expected_scenes: Sequence[Any]
) -> None:
    expected = {str(scene) for scene in expected_scenes}
    if len(folds) != 5 or len(expected) != 30:
        raise ValueError("fold audit requires five folds and 30 expected scenes")
    test_occurrences: list[str] = []
    for index, fold in enumerate(folds):
        train = set(map(str, fold["train_scenes"]))
        validation = set(map(str, fold["validation_scenes"]))
        test = set(map(str, fold["test_scenes"]))
        if (len(train), len(validation), len(test)) != (19, 5, 6):
            raise ValueError(f"fold {index} does not have a 19/5/6 split")
        if train & validation or train & test or validation & test:
            raise ValueError(f"fold {index} scene partitions overlap")
        if train | validation | test != expected:
            raise ValueError(f"fold {index} does not cover the locked scene universe")
        test_occurrences.extend(test)
    if set(test_occurrences) != expected or len(test_occurrences) != len(expected):
        raise ValueError("every scene must appear exactly once as outer test")


def assign_group_splits(
    universe: pd.DataFrame, fold: Mapping[str, Any]
) -> pd.DataFrame:
    """Attach one outer-fold split to every complete target group."""

    required = {"group_id", "scene_id"}
    missing = sorted(required.difference(universe.columns))
    if missing:
        raise ValueError(f"group universe lacks columns: {missing}")
    assignments: dict[str, str] = {}
    for partition in ("train", "validation", "test"):
        for scene in fold[f"{partition}_scenes"]:
            scene_id = str(scene)
            if scene_id in assignments:
                raise ValueError(f"scene {scene_id} appears in multiple partitions")
            assignments[scene_id] = partition
    result = universe[["group_id", "scene_id"]].copy()
    result["group_id"] = result["group_id"].astype(str)
    result["scene_id"] = result["scene_id"].astype(str)
    result["outer_partition"] = result["scene_id"].map(assignments)
    if result["outer_partition"].isna().any():
        raise ValueError("a group scene is absent from its fold manifest")
    if result["group_id"].duplicated().any():
        raise ValueError("group universe contains duplicate group IDs")
    return result


def fit_fold_preprocessor(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    feature_columns: Sequence[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Fit median preprocessing only on the supplied outer-train rows."""

    columns = assert_no_gt_leakage(feature_columns)
    imputer = StableMissingValueImputer()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", pd.errors.PerformanceWarning)
        train_matrix = imputer.fit_transform(train[list(columns)])
        validation_matrix = imputer.transform(validation[list(columns)])
        test_matrix = imputer.transform(test[list(columns)])
    artifact = imputer.artifact()
    if artifact.get("fit_scope") != "training_only":
        raise AssertionError("imputer did not retain its train-only contract")
    return train_matrix, validation_matrix, test_matrix, artifact


def assert_frozen_candidate_pool(
    source: pd.DataFrame, prediction: pd.DataFrame, *, system: str
) -> dict[str, Any]:
    """Require byte-identical IDs/hashes and unchanged candidate width/order."""

    required = {*_IDENTITY_COLUMNS, *_GEOMETRY_COLUMNS}
    for name, frame in (("source", source), (system, prediction)):
        missing = sorted(required.difference(frame.columns))
        if missing:
            raise ValueError(f"{name} candidate rows lack frozen fields: {missing}")
        if frame.duplicated(list(_IDENTITY_COLUMNS)).any():
            raise ValueError(f"{name} contains duplicate candidate IDs within a group")
    source_keys = source[list(_IDENTITY_COLUMNS)].astype(str).to_numpy()
    prediction_keys = prediction[list(_IDENTITY_COLUMNS)].astype(str).to_numpy()
    if source_keys.shape != prediction_keys.shape or not np.array_equal(
        source_keys, prediction_keys
    ):
        raise ValueError(f"{system} changed candidate count, membership, or row order")
    source_geometry = source["geometry_sha256"].astype(str).to_numpy()
    predicted_geometry = prediction["geometry_sha256"].astype(str).to_numpy()
    if not np.array_equal(source_geometry, predicted_geometry):
        raise ValueError(f"{system} changed frozen candidate geometry")
    source_width = pd.to_numeric(source["gripper_width_m"], errors="coerce").to_numpy(float)
    predicted_width = pd.to_numeric(
        prediction["gripper_width_m"], errors="coerce"
    ).to_numpy(float)
    if not np.array_equal(source_width, predicted_width, equal_nan=True):
        raise ValueError(f"{system} changed frozen candidate width")
    for column in ("native_rank", "native_score"):
        if column in source or column in prediction:
            if column not in source or column not in prediction:
                raise ValueError(f"{system} dropped frozen {column}")
            before = pd.to_numeric(source[column], errors="coerce").to_numpy(float)
            after = pd.to_numeric(prediction[column], errors="coerce").to_numpy(float)
            if not np.array_equal(before, after, equal_nan=True):
                raise ValueError(f"{system} changed frozen {column}")
    return {
        "system": system,
        "status": "PASS",
        "candidate_count": int(len(source)),
        "candidate_identity_geometry_fingerprint": canonical_sha256(
            source[[*_IDENTITY_COLUMNS, *_GEOMETRY_COLUMNS]]
            .fillna("NaN")
            .to_dict(orient="records")
        ),
    }


def _source_path(repo: Path) -> Path:
    return repo / "artifacts" / "graspnet6d" / SOURCE_RUN_ID


def _resolve_manifest_path(manifest_path: Path, declared: str) -> Path:
    path = Path(declared).expanduser()
    return path.resolve() if path.is_absolute() else (manifest_path.parent / path).resolve()


def _load_source(repo: Path) -> SourceData:
    root = _source_path(repo)
    run_manifest_path = root / "run_manifest.json"
    run_manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
    if run_manifest.get("run_id") != SOURCE_RUN_ID:
        raise ValueError("locked 6-DoF run ID does not match its manifest")
    if run_manifest.get("formal_results_emitted") is not True:
        raise ValueError("6-DoF source is not a formal emitted result")
    if run_manifest.get("completion_status") != "COMPLETE_TRAIN3_SUBSET":
        raise ValueError("6-DoF source completion status is not locked COMPLETE")

    feature_path = root / "candidate_features.parquet"
    label_path = root / "candidate_labels.parquet"
    selection_path = root / "analysis" / PRIMARY_CONDITION / "model_selection.json"
    analysis_manifest_path = root / "analysis" / PRIMARY_CONDITION / "analysis_manifest.json"
    schema_path = root / "feature_schema.json"
    essential = (
        run_manifest_path,
        feature_path,
        label_path,
        selection_path,
        analysis_manifest_path,
        schema_path,
    )
    missing = [str(path) for path in essential if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"locked 6-DoF source files are missing: {missing}")

    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    primary = selection.get("variants", {}).get("A2")
    if not isinstance(primary, Mapping) or primary.get("variant") != "all":
        raise ValueError("source model selection lacks the formal all-feature A2 ranker")
    feature_columns = tuple(map(str, primary.get("feature_columns", ())))
    assert_no_gt_leakage(feature_columns)
    hyperparameters = dict(primary.get("selected_config", {}))
    if hyperparameters != EXPECTED_HYPERPARAMETERS:
        raise ValueError(
            "source LambdaMART hyperparameters drifted from the pre-registration: "
            f"{hyperparameters}"
        )
    ranker_artifacts = primary.get("rankers", [])
    if len(ranker_artifacts) != len(FORMAL_SEEDS):
        raise ValueError("source does not contain all three locked ranker seeds")
    for seed, artifact in zip(FORMAL_SEEDS, ranker_artifacts, strict=True):
        parameters = artifact.get("parameters", {})
        if (
            artifact.get("seed") != seed
            or artifact.get("early_stopping_rounds") != EXPECTED_EARLY_STOPPING_ROUNDS
            or parameters.get("objective") != "lambdarank"
            or parameters.get("device_type") != "cpu"
            or parameters.get("label_gain") != list(LABEL_GAIN)
            or any(parameters.get(key) != value for key, value in hyperparameters.items())
        ):
            raise ValueError("source primary LambdaMART contract has changed")

    features = pd.read_parquet(feature_path)
    features = features.loc[
        features["condition"].astype(str).eq(PRIMARY_CONDITION)
    ].copy()
    if tuple(features.columns[:5]) != (
        "group_id",
        "candidate_id",
        "scene_id",
        "split",
        "condition",
    ):
        raise ValueError("raw candidate feature identity schema changed")
    if sorted(set(feature_columns).difference(features.columns)):
        raise ValueError("raw candidate feature table lacks the locked schema")

    labels = pd.read_parquet(label_path)
    labels = labels.loc[
        labels["grounding_condition"].astype(str).eq(PRIMARY_CONDITION)
    ].copy()
    universe = labels[
        [
            "group_id",
            "scene_id",
            "candidate_count",
            "candidate_pool_fingerprint",
            "candidate_bundle_sha256",
        ]
    ].drop_duplicates("group_id", keep="first")
    universe["group_id"] = universe["group_id"].astype(str)
    universe["scene_id"] = universe["scene_id"].astype(str)
    if len(universe) != 1_440 or set(universe["scene_id"]) != set(EXPECTED_SCENES):
        raise ValueError("oracle group universe is not the locked 1,440 groups/30 scenes")
    candidates = labels.loc[labels["record_kind"].astype(str).eq("candidate_label")].copy()
    if len(candidates) != len(features):
        raise ValueError("raw feature and official-label candidate counts differ")
    label_fields = [
        *_IDENTITY_COLUMNS,
        *_LABEL_COLUMNS,
        "candidate_pool_fingerprint",
        "candidate_bundle_sha256",
    ]
    rows = features.merge(
        candidates[label_fields],
        on=list(_IDENTITY_COLUMNS),
        how="inner",
        validate="one_to_one",
    )
    if len(rows) != len(features):
        raise ValueError("raw feature/label join changed frozen candidate membership")

    analysis_manifest = json.loads(analysis_manifest_path.read_text(encoding="utf-8"))
    input_manifest_path = _resolve_manifest_path(
        analysis_manifest_path, str(analysis_manifest["input_manifest_path"])
    )
    if sha256_file(input_manifest_path) != analysis_manifest["input_manifest_sha256"]:
        raise ValueError("formal oracle analysis-input manifest hash mismatch")
    input_manifest = json.loads(input_manifest_path.read_text(encoding="utf-8"))
    assembled_parts: list[pd.DataFrame] = []
    input_files: list[Path] = [input_manifest_path]
    universe_parts: list[pd.DataFrame] = []
    for partition in ("train", "validation", "test"):
        declared = input_manifest["partitions"][partition]
        row_path = _resolve_manifest_path(input_manifest_path, declared["rows"]["path"])
        group_path = _resolve_manifest_path(
            input_manifest_path, declared["group_universe"]["path"]
        )
        if sha256_file(row_path) != declared["rows"]["sha256"]:
            raise ValueError(f"formal {partition} assembled-row hash mismatch")
        if sha256_file(group_path) != declared["group_universe"]["sha256"]:
            raise ValueError(f"formal {partition} group-universe hash mismatch")
        assembled_parts.append(
            pd.read_parquet(row_path)[
                ["group_id", "candidate_id", "geometry_sha256", "pre_nms_native_rank"]
            ]
        )
        universe_parts.append(pd.read_parquet(group_path)[["group_id", "scene_id"]])
        input_files.extend((row_path, group_path))
    assembled = pd.concat(assembled_parts, ignore_index=True)
    assembled_universe = pd.concat(universe_parts, ignore_index=True)
    if set(map(tuple, assembled_universe.astype(str).to_numpy())) != set(
        map(tuple, universe[["group_id", "scene_id"]].astype(str).to_numpy())
    ):
        raise ValueError("aggregate label universe differs from formal assembled universe")
    rows = rows.merge(
        assembled,
        on=list(_IDENTITY_COLUMNS),
        how="inner",
        validate="one_to_one",
    )
    if len(rows) != len(features):
        raise ValueError("geometry join changed frozen candidate membership")
    if not rows["geometry_sha256"].astype(str).str.fullmatch(r"[0-9a-f]{64}").all():
        raise ValueError("candidate geometry SHA-256 values are invalid")
    if rows["scene_id"].astype(str).ne(
        rows["group_id"].astype(str).map(universe.set_index("group_id")["scene_id"])
    ).any():
        raise ValueError("candidate groups do not inherit the universe scene")
    observed_counts = rows.groupby("group_id", sort=False).size()
    expected_counts = pd.to_numeric(
        universe.set_index("group_id")["candidate_count"], errors="raise"
    ).astype(int)
    expected_nonempty = expected_counts[expected_counts.gt(0)]
    if not observed_counts.sort_index().equals(expected_nonempty.sort_index()):
        raise ValueError("candidate counts differ from frozen per-group counts")

    rows = rows.sort_values(
        ["group_id", "native_rank", "candidate_id"], kind="mergesort"
    ).reset_index(drop=True)
    rows["relevance"] = pd.to_numeric(rows["relevance"], errors="raise").astype(np.int32)
    if rows.duplicated(list(_IDENTITY_COLUMNS)).any():
        raise ValueError("source contains duplicate candidate IDs")

    source_paths = tuple(dict.fromkeys([*essential, *input_files]))
    hashes = {
        str(path.relative_to(repo)): sha256_file(path) for path in source_paths
    }
    return SourceData(
        rows=rows,
        universe=universe.reset_index(drop=True),
        feature_columns=feature_columns,
        hyperparameters=hyperparameters,
        early_stopping_rounds=EXPECTED_EARLY_STOPPING_ROUNDS,
        source_hashes=hashes,
        source_paths=source_paths,
    )


def _verify_source_unchanged(repo: Path, source: SourceData) -> None:
    observed = {
        str(path.relative_to(repo)): sha256_file(path) for path in source.source_paths
    }
    if observed != dict(source.source_hashes):
        raise RuntimeError("a locked 6-DoF source artifact changed during scene-CV")


def _fold_manifest(
    fold: Mapping[str, Any], source: SourceData
) -> dict[str, Any]:
    assignments = assign_group_splits(source.universe, fold)
    rows = source.rows.merge(
        assignments[["group_id", "outer_partition"]],
        on="group_id",
        validate="many_to_one",
    )
    group_counts = assignments["outer_partition"].value_counts().to_dict()
    candidate_counts = rows["outer_partition"].value_counts().to_dict()
    return {
        "schema_version": "robustness_6d_scene_fold_v1",
        "analysis_nature": "post-hoc robustness and sensitivity analysis",
        **dict(fold),
        "condition": PRIMARY_CONDITION,
        "training_seeds": list(FORMAL_SEEDS),
        "group_counts": {
            partition: int(group_counts.get(partition, 0))
            for partition in ("train", "validation", "test")
        },
        "candidate_counts": {
            partition: int(candidate_counts.get(partition, 0))
            for partition in ("train", "validation", "test")
        },
        "feature_schema_sha256": canonical_sha256(list(source.feature_columns)),
        "hyperparameters_sha256": canonical_sha256(dict(source.hyperparameters)),
        "candidate_source_sha256": source.source_hashes[
            f"artifacts/graspnet6d/{SOURCE_RUN_ID}/candidate_features.parquet"
        ],
        "label_source_sha256": source.source_hashes[
            f"artifacts/graspnet6d/{SOURCE_RUN_ID}/candidate_labels.parquet"
        ],
    }


def _lock_fold_manifests(
    output: Path, folds: Sequence[Mapping[str, Any]], source: SourceData
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    folder = output / "6d_scene_cv" / "folds"
    locked: list[dict[str, Any]] = []
    fold_hashes: dict[str, str] = {}
    for fold in folds:
        payload = _fold_manifest(fold, source)
        path = folder / f"fold_{int(fold['fold'])}.json"
        if path.exists():
            observed = json.loads(path.read_text(encoding="utf-8"))
            if observed != payload:
                raise ValueError(f"existing locked fold manifest differs: {path}")
        else:
            _atomic_json(path, payload)
        digest = sha256_file(path)
        fold_hashes[path.name] = digest
        locked.append(payload)
    audit = {
        "schema_version": "robustness_6d_fold_audit_v1",
        "status": "LOCKED_PRE_MODEL",
        "split_rule": "seeded metadata-only scene shuffle",
        "split_seed": SPLIT_SEED,
        "scene_count": 30,
        "fold_count": 5,
        "outer_test_union": sorted(EXPECTED_SCENES),
        "each_scene_appears_once_in_outer_test": True,
        "train_validation_test_disjoint": True,
        "groups_inherit_scene_split": True,
        "fold_manifest_sha256": fold_hashes,
    }
    audit_path = output / "6d_scene_cv" / "fold_audit.json"
    if audit_path.exists():
        if json.loads(audit_path.read_text(encoding="utf-8")) != audit:
            raise ValueError("existing fold audit differs from the pre-model audit")
    else:
        _atomic_json(audit_path, audit)
    return locked, audit


def _fold_rows(
    source: SourceData, fold: Mapping[str, Any]
) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    assignments = assign_group_splits(source.universe, fold)
    assigned_rows = source.rows.merge(
        assignments[["group_id", "outer_partition"]],
        on="group_id",
        validate="many_to_one",
    )
    rows: dict[str, pd.DataFrame] = {}
    universes: dict[str, pd.DataFrame] = {}
    for partition in ("train", "validation", "test"):
        rows[partition] = (
            assigned_rows.loc[assigned_rows["outer_partition"].eq(partition)]
            .drop(columns="outer_partition")
            .sort_values(["group_id", "native_rank", "candidate_id"], kind="mergesort")
        )
        universes[partition] = assignments.loc[
            assignments["outer_partition"].eq(partition), ["group_id", "scene_id"]
        ].reset_index(drop=True)
        if set(rows[partition]["scene_id"].astype(str)).difference(
            set(map(str, fold[f"{partition}_scenes"]))
        ):
            raise ValueError(f"{partition} candidate rows violate scene inheritance")
    return rows, universes


def _attach_seed_scores(
    rows: pd.DataFrame, scores: Sequence[np.ndarray]
) -> pd.DataFrame:
    result = rows.copy()
    columns: list[str] = []
    for seed, values in zip(FORMAL_SEEDS, scores, strict=True):
        array = np.asarray(values, dtype=float)
        if array.shape != (len(result),) or not np.isfinite(array).all():
            raise RuntimeError(f"seed {seed} produced invalid predictions")
        column = f"raw_rerank_score_seed_{seed}"
        result[column] = array
        columns.append(column)
    result["raw_rerank_score"] = result[columns].mean(axis=1)
    result["raw_rerank_score_std"] = result[columns].std(axis=1, ddof=0)
    return result


def _train_fold(
    fold: Mapping[str, Any], source: SourceData
) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows, _ = _fold_rows(source, fold)
    train, validation, test = rows["train"], rows["validation"], rows["test"]
    train_x, validation_x, test_x, imputer_artifact = fit_fold_preprocessor(
        train, validation, test, source.feature_columns
    )
    train_groups = contiguous_group_sizes(train["group_id"], length=len(train))
    validation_groups = contiguous_group_sizes(
        validation["group_id"], length=len(validation)
    )
    models: list[GradedLightGBMLambdaRank] = []
    validation_scores: list[np.ndarray] = []
    test_scores: list[np.ndarray] = []
    for seed in FORMAL_SEEDS:
        model = GradedLightGBMLambdaRank(
            seed=seed,
            early_stopping_rounds=source.early_stopping_rounds,
            **dict(source.hyperparameters),
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
        validation_scores.append(
            model.predict(
                validation_x.to_numpy(float),
                candidate_ids=validation["candidate_id"].astype(str).tolist(),
            )
        )
        test_scores.append(
            model.predict(
                test_x.to_numpy(float),
                candidate_ids=test["candidate_id"].astype(str).tolist(),
            )
        )
        models.append(model)
    validation_predictions = _attach_seed_scores(validation, validation_scores)
    predictions = _attach_seed_scores(test, test_scores)

    # Reuse the source implementation of the expected-gain gate.  It creates
    # scene-held-out OOF Train predictions, calibrates on outer Validation, and
    # catches non-fittable evidence by returning FAIL_CLOSED_NATIVE.
    from graspnet6d.experiment_analysis import AnalysisConfig, _fit_and_apply_gate

    gate_config = AnalysisConfig(
        config_grid=(dict(source.hyperparameters),),
        seeds=FORMAL_SEEDS,
        primary_seed=FORMAL_SEEDS[0],
        early_stopping_rounds=source.early_stopping_rounds,
        max_k=50,
        bootstrap_iterations=BOOTSTRAP_ITERATIONS,
        bootstrap_seed=BOOTSTRAP_SEED,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", pd.errors.PerformanceWarning)
        predictions, gate_artifact = _fit_and_apply_gate(
            train,
            validation_predictions,
            predictions,
            all_feature_fit=SimpleNamespace(
                feature_columns=source.feature_columns,
                selected_config=source.hyperparameters,
            ),
            config=gate_config,
        )
    predictions["oracle_score"] = predictions["relevance"].astype(float)
    predictions["fold"] = int(fold["fold"])
    predictions["outer_partition"] = "test"
    frozen_audits = [
        assert_frozen_candidate_pool(test, predictions, system=system)
        for system in ("raw_lambdamart", "expected_gain_gate", "oracle")
    ]
    metadata = {
        "schema_version": "robustness_6d_fold_fit_v1",
        "status": "COMPLETE",
        "fold": int(fold["fold"]),
        "condition": PRIMARY_CONDITION,
        "feature_columns": list(source.feature_columns),
        "feature_schema_sha256": canonical_sha256(list(source.feature_columns)),
        "hyperparameters": dict(source.hyperparameters),
        "hyperparameters_sha256": canonical_sha256(dict(source.hyperparameters)),
        "early_stopping_rounds": source.early_stopping_rounds,
        "seeds": list(FORMAL_SEEDS),
        "imputer": imputer_artifact,
        "rankers": [model.artifact() for model in models],
        "gate": gate_artifact,
        "frozen_pool_assertions": frozen_audits,
    }
    return predictions, metadata


def _metric_numerator(metrics: Mapping[str, Any], key: str) -> int:
    return int(round(float(metrics[key]) * int(metrics["group_count"])))


def _system_metrics(
    rows: pd.DataFrame, universe: pd.DataFrame, score_column: str
) -> tuple[dict[str, Any], pd.DataFrame]:
    return evaluate_target_rankings(
        rows,
        universe[["group_id", "scene_id"]],
        score_column=score_column,
        max_k=50,
    )


def _normalised_metric_row(
    *,
    scope: str,
    fold: int | None,
    system: str,
    score_column: str,
    seed: int | str | None,
    metrics: Mapping[str, Any],
) -> dict[str, Any]:
    count = int(metrics["group_count"])
    return {
        "scope": scope,
        "fold": fold,
        "system": system,
        "score_column": score_column,
        "seed": seed,
        "n_scenes": None,
        "n_groups": count,
        "n_candidates": int(metrics["candidate_count"]),
        "non_empty_pool_n": _metric_numerator(metrics, "non_empty_pool_rate"),
        "non_empty_pool_rate": metrics["non_empty_pool_rate"],
        "p_at_1_mu_0.4_n": _metric_numerator(metrics, "target_p_at_1_mu_0.4"),
        "p_at_1_mu_0.4": metrics["target_p_at_1_mu_0.4"],
        "p_at_1_mu_0.8_n": _metric_numerator(metrics, "target_p_at_1_mu_0.8"),
        "p_at_1_mu_0.8": metrics["target_p_at_1_mu_0.8"],
        "p_at_1_mu_1.2_n": _metric_numerator(metrics, "target_p_at_1_mu_1.2"),
        "p_at_1_mu_1.2": metrics["target_p_at_1_mu_1.2"],
        "target_ap_mu_1.2": metrics["target_graspnet_style_ap_mu_1.2"],
        "target_ap_mean_mu_0.2_to_1.2": metrics[
            "target_graspnet_style_ap_mean_mu_0.2_to_1.2"
        ],
        "ndcg_at_1": metrics["ndcg_at_1"],
        "ndcg_at_5": metrics["ndcg_at_5"],
        "ndcg_at_10": metrics["ndcg_at_10"],
        "mrr": metrics["mrr"],
        "mean_first_valid_rank": metrics["mean_first_valid_target_rank"],
        "oracle_at_5_n": _metric_numerator(metrics, "oracle_at_5"),
        "oracle_at_5": metrics["oracle_at_5"],
        "oracle_at_10_n": _metric_numerator(metrics, "oracle_at_10"),
        "oracle_at_10": metrics["oracle_at_10"],
        "oracle_at_20_n": _metric_numerator(metrics, "oracle_at_20"),
        "oracle_at_20": metrics["oracle_at_20"],
        "oracle_at_50_n": _metric_numerator(metrics, "oracle_at_50"),
        "oracle_at_50": metrics["oracle_at_50"],
    }


def _evaluate_fold(
    fold: Mapping[str, Any], source: SourceData, predictions: pd.DataFrame, gate: Mapping[str, Any]
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    _, universes = _fold_rows(source, fold)
    universe = universes["test"]
    systems = {
        "native": "native_score",
        "raw_lambdamart": "raw_rerank_score",
        "expected_gain_gated": "gated_score",
        "oracle": "oracle_score",
    }
    metrics_by_system: dict[str, dict[str, Any]] = {}
    groups_by_system: dict[str, pd.DataFrame] = {}
    normalised: list[dict[str, Any]] = []
    for system, score in systems.items():
        metrics, per_group = _system_metrics(predictions, universe, score)
        metrics_by_system[system] = metrics
        groups_by_system[system] = per_group
        row = _normalised_metric_row(
            scope="fold_ensemble",
            fold=int(fold["fold"]),
            system=system,
            score_column=score,
            seed="ensemble_mean" if system == "raw_lambdamart" else None,
            metrics=metrics,
        )
        row["n_scenes"] = 6
        normalised.append(row)
    paired_summary, _ = paired_intervention_outcomes(
        groups_by_system["native"], groups_by_system["raw_lambdamart"]
    )
    gated_summary, _ = paired_intervention_outcomes(
        groups_by_system["native"], groups_by_system["expected_gain_gated"]
    )
    native = metrics_by_system["native"]
    raw = metrics_by_system["raw_lambdamart"]
    fold_result = {
        "fold": int(fold["fold"]),
        "train_scenes": "|".join(map(str, fold["train_scenes"])),
        "validation_scenes": "|".join(map(str, fold["validation_scenes"])),
        "test_scenes": "|".join(map(str, fold["test_scenes"])),
        "n_train_scenes": 19,
        "n_validation_scenes": 5,
        "n_test_scenes": 6,
        "n_groups": int(native["group_count"]),
        "n_candidates": int(native["candidate_count"]),
        "non_empty_pool_n": _metric_numerator(native, "non_empty_pool_rate"),
        "non_empty_pool_rate": native["non_empty_pool_rate"],
        "native_p_at_1_mu_1.2_n": _metric_numerator(native, "target_p_at_1_mu_1.2"),
        "native_p_at_1_mu_1.2": native["target_p_at_1_mu_1.2"],
        "raw_p_at_1_mu_1.2_n": _metric_numerator(raw, "target_p_at_1_mu_1.2"),
        "raw_p_at_1_mu_1.2": raw["target_p_at_1_mu_1.2"],
        "delta_p_at_1_mu_1.2": paired_summary["delta_p_at_1"],
        "delta_p_at_1_mu_1.2_pp": 100.0 * float(paired_summary["delta_p_at_1"]),
        "recovered": paired_summary["recovered"],
        "harmful": paired_summary["harmful"],
        "net_recovered": paired_summary["net_recovered"],
        "outcome_changing_precision": paired_summary["outcome_changing_precision"],
        "oracle_at_5_n": _metric_numerator(native, "oracle_at_5"),
        "oracle_at_5": native["oracle_at_5"],
        "oracle_at_10_n": _metric_numerator(native, "oracle_at_10"),
        "oracle_at_10": native["oracle_at_10"],
        "oracle_at_20_n": _metric_numerator(native, "oracle_at_20"),
        "oracle_at_20": native["oracle_at_20"],
        "oracle_at_50_n": _metric_numerator(native, "oracle_at_50"),
        "oracle_at_50": native["oracle_at_50"],
        "gate_status": gate.get("status"),
        "gate_switch_count": gate.get("test_switch_count", 0),
        "gated_delta_p_at_1_mu_1.2_pp": 100.0 * float(gated_summary["delta_p_at_1"]),
    }

    seed_rows: list[dict[str, Any]] = []
    for seed in FORMAL_SEEDS:
        score = f"raw_rerank_score_seed_{seed}"
        metrics, groups = _system_metrics(predictions, universe, score)
        paired, _ = paired_intervention_outcomes(groups_by_system["native"], groups)
        row = _normalised_metric_row(
            scope="fold_seed",
            fold=int(fold["fold"]),
            system="raw_lambdamart_seed",
            score_column=score,
            seed=seed,
            metrics=metrics,
        )
        row.update(
            {
                "n_scenes": 6,
                "delta_p_at_1_mu_1.2": paired["delta_p_at_1"],
                "delta_p_at_1_mu_1.2_pp": 100.0 * float(paired["delta_p_at_1"]),
                "recovered": paired["recovered"],
                "harmful": paired["harmful"],
                "net_recovered": paired["net_recovered"],
                "outcome_changing_precision": paired["outcome_changing_precision"],
            }
        )
        seed_rows.append(row)
    return fold_result, normalised, seed_rows


def _pooled_outputs(
    source: SourceData,
    folds: Sequence[Mapping[str, Any]],
    predictions: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any], pd.DataFrame, dict[str, Any]]:
    fold_by_scene = {
        str(scene): int(fold["fold"])
        for fold in folds
        for scene in fold["test_scenes"]
    }
    universe = source.universe[["group_id", "scene_id"]].copy()
    universe["fold"] = universe["scene_id"].astype(str).map(fold_by_scene)
    if universe["fold"].isna().any() or len(universe) != 1_440:
        raise ValueError("pooled OOF universe does not cover all 1,440 groups")
    if len(predictions) != len(source.rows):
        raise ValueError("pooled OOF predictions do not cover every frozen candidate")
    source_sorted = source.rows.sort_values(
        ["group_id", "native_rank", "candidate_id"], kind="mergesort"
    ).reset_index(drop=True)
    prediction_sorted = predictions.sort_values(
        ["group_id", "native_rank", "candidate_id"], kind="mergesort"
    ).reset_index(drop=True)
    assert_frozen_candidate_pool(source_sorted, prediction_sorted, system="pooled_oof")

    systems = {
        "native": "native_score",
        "raw_lambdamart": "raw_rerank_score",
        "expected_gain_gated": "gated_score",
        "oracle": "oracle_score",
    }
    metric_rows: list[dict[str, Any]] = []
    groups: dict[str, pd.DataFrame] = {}
    metrics: dict[str, dict[str, Any]] = {}
    for system, score in systems.items():
        system_metrics, per_group = _system_metrics(prediction_sorted, universe, score)
        groups[system] = per_group
        metrics[system] = system_metrics
        row = _normalised_metric_row(
            scope="pooled_oof",
            fold=None,
            system=system,
            score_column=score,
            seed="ensemble_mean" if system == "raw_lambdamart" else None,
            metrics=system_metrics,
        )
        row["n_scenes"] = 30
        metric_rows.append(row)
    seed_rows: list[dict[str, Any]] = []
    for seed in FORMAL_SEEDS:
        score = f"raw_rerank_score_seed_{seed}"
        seed_metrics, seed_groups = _system_metrics(prediction_sorted, universe, score)
        paired, _ = paired_intervention_outcomes(groups["native"], seed_groups)
        row = _normalised_metric_row(
            scope="pooled_oof_seed",
            fold=None,
            system="raw_lambdamart_seed",
            score_column=score,
            seed=seed,
            metrics=seed_metrics,
        )
        row.update(
            {
                "n_scenes": 30,
                "delta_p_at_1_mu_1.2": paired["delta_p_at_1"],
                "delta_p_at_1_mu_1.2_pp": 100.0 * float(paired["delta_p_at_1"]),
                "recovered": paired["recovered"],
                "harmful": paired["harmful"],
                "net_recovered": paired["net_recovered"],
                "outcome_changing_precision": paired["outcome_changing_precision"],
            }
        )
        seed_rows.append(row)

    paired_summary, paired = paired_intervention_outcomes(
        groups["native"], groups["raw_lambdamart"]
    )
    deltas = paired_metric_deltas(groups["native"], groups["raw_lambdamart"])
    bootstrap = scene_cluster_bootstrap(
        deltas,
        delta_columns=("delta_p_at_1", "delta_ap", "delta_mrr", "net_recovered_rate"),
        iterations=BOOTSTRAP_ITERATIONS,
        seed=BOOTSTRAP_SEED,
        confidence=0.95,
    )
    significance = {
        "schema_version": "robustness_6d_scene_cv_significance_v1",
        "analysis_nature": "post-hoc robustness and sensitivity analysis",
        "primary_metric": "P@1 at mu <= 1.2",
        "primary_uncertainty": "scene-cluster percentile bootstrap",
        "bootstrap": bootstrap.loc[bootstrap["metric"].eq("delta_p_at_1")]
        .iloc[0]
        .to_dict(),
        "mcnemar_supportive": mcnemar_exact(
            paired["top1_success_mu_1.2_reference"],
            paired["top1_success_mu_1.2_challenger"],
        ),
        "paired_outcomes": paired_summary,
    }
    paired = paired.merge(
        universe[["group_id", "fold"]], on="group_id", validate="one_to_one"
    )
    scene = (
        paired.groupby("scene_id", sort=True)
        .agg(
            fold=("fold", "first"),
            n_groups=("group_id", "size"),
            native_success_n=("top1_success_mu_1.2_reference", "sum"),
            raw_success_n=("top1_success_mu_1.2_challenger", "sum"),
            recovered=("recovered", "sum"),
            harmful=("harmful", "sum"),
            net_recovered=("net_recovered", "sum"),
        )
        .reset_index()
    )
    scene["native_p_at_1_mu_1.2"] = scene["native_success_n"] / scene["n_groups"]
    scene["raw_p_at_1_mu_1.2"] = scene["raw_success_n"] / scene["n_groups"]
    scene["delta_p_at_1_mu_1.2"] = (
        scene["raw_p_at_1_mu_1.2"] - scene["native_p_at_1_mu_1.2"]
    )
    scene["delta_p_at_1_mu_1.2_pp"] = 100.0 * scene["delta_p_at_1_mu_1.2"]
    native_oracle = groups["native"][["scene_id", "group_id", "oracle_at_50_mu_1.2"]]
    oracle_scene = (
        native_oracle.groupby("scene_id", sort=True)["oracle_at_50_mu_1.2"]
        .agg([("oracle_at_50_n", "sum"), ("oracle_at_50", "mean")])
        .reset_index()
    )
    scene = scene.merge(oracle_scene, on="scene_id", validate="one_to_one")
    return (
        pd.DataFrame(metric_rows),
        pd.DataFrame(seed_rows),
        paired,
        significance,
        scene,
        {"metrics": metrics, "paired": paired_summary},
    )


def run_scene_cv(
    repo_root: Path, run_dir: Path, *, resume: bool = True
) -> dict[str, Any]:
    """Execute/resume the complete primary oracle-mask five-fold 6-DoF CV."""

    repo, output = _assert_isolated_run_dir(Path(repo_root), Path(run_dir))
    source = _load_source(repo)
    folds = make_scene_folds(source.universe["scene_id"], split_seed=SPLIT_SEED)
    locked_folds, fold_audit = _lock_fold_manifests(output, folds, source)
    cv_dir = output / "6d_scene_cv"
    completion_path = cv_dir / "completion.json"
    preregistration_sha256 = sha256_file(output / "PRE_REGISTRATION.md")
    if resume and completion_path.is_file():
        completion = json.loads(completion_path.read_text(encoding="utf-8"))
        required_outputs = [cv_dir / name for name in completion.get("outputs", {})]
        if (
            completion.get("status") == "COMPLETE_6D_SCENE_CV"
            and completion.get("source_hashes") == dict(source.source_hashes)
            and completion.get("preregistration_sha256") == preregistration_sha256
            and all(path.is_file() for path in required_outputs)
            and all(sha256_file(path) == completion["outputs"][path.name] for path in required_outputs)
        ):
            _verify_source_unchanged(repo, source)
            return completion

    fold_results: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    seed_rows: list[dict[str, Any]] = []
    prediction_parts: list[pd.DataFrame] = []
    gate_artifacts: dict[str, Any] = {}
    fold_output_hashes: dict[str, Any] = {}
    for fold in locked_folds:
        index = int(fold["fold"])
        manifest_path = cv_dir / "folds" / f"fold_{index}.json"
        prediction_path = cv_dir / "folds" / f"fold_{index}_predictions.parquet"
        metadata_path = cv_dir / "folds" / f"fold_{index}_fit.json"
        prediction: pd.DataFrame
        metadata: dict[str, Any]
        if resume and prediction_path.is_file() and metadata_path.is_file():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            prediction = pd.read_parquet(prediction_path)
            _, universes = _fold_rows(source, fold)
            expected_test = source.rows.loc[
                source.rows["scene_id"].astype(str).isin(fold["test_scenes"])
            ].copy()
            assert_frozen_candidate_pool(expected_test, prediction, system=f"fold_{index}_resume")
            if (
                metadata.get("status") != "COMPLETE"
                or metadata.get("fold_manifest_sha256") != sha256_file(manifest_path)
                or metadata.get("prediction_sha256") != sha256_file(prediction_path)
                or metadata.get("n_test_groups") != len(universes["test"])
            ):
                raise ValueError(f"fold {index} resume metadata is stale")
        else:
            prediction, metadata = _train_fold(fold, source)
            _atomic_parquet(prediction_path, prediction)
            metadata.update(
                {
                    "fold_manifest_sha256": sha256_file(manifest_path),
                    "prediction_sha256": sha256_file(prediction_path),
                    "n_test_groups": int(fold["group_counts"]["test"]),
                    "source_hashes": dict(source.source_hashes),
                }
            )
            _atomic_json(metadata_path, metadata)
        gate_artifacts[str(index)] = metadata["gate"]
        result, fold_metrics, fold_seed_rows = _evaluate_fold(
            fold, source, prediction, metadata["gate"]
        )
        fold_results.append(result)
        metric_rows.extend(fold_metrics)
        seed_rows.extend(fold_seed_rows)
        prediction_parts.append(prediction)
        fold_output_hashes[str(index)] = {
            "predictions": sha256_file(prediction_path),
            "fit": sha256_file(metadata_path),
        }

    oof_predictions = pd.concat(prediction_parts, ignore_index=True)
    (
        pooled_metrics,
        pooled_seed_rows,
        paired_outcomes,
        significance,
        scene_effects,
        pooled_payload,
    ) = _pooled_outputs(source, locked_folds, oof_predictions)
    seed_rows.extend(pooled_seed_rows.to_dict(orient="records"))
    oof_metrics = pd.concat([pd.DataFrame(metric_rows), pooled_metrics], ignore_index=True)
    bootstrap = scene_cluster_bootstrap(
        paired_metric_deltas(
            _system_metrics(oof_predictions, source.universe, "native_score")[1],
            _system_metrics(oof_predictions, source.universe, "raw_rerank_score")[1],
        ),
        delta_columns=("delta_p_at_1", "delta_ap", "delta_mrr", "net_recovered_rate"),
        iterations=BOOTSTRAP_ITERATIONS,
        seed=BOOTSTRAP_SEED,
        confidence=0.95,
    )

    fold_frame = pd.DataFrame(fold_results).sort_values("fold").reset_index(drop=True)
    seed_frame = pd.DataFrame(seed_rows).sort_values(
        ["scope", "seed", "fold"], na_position="last", kind="mergesort"
    )
    fold_summary = {
        "schema_version": "robustness_6d_fold_summary_v1",
        "fold_count": 5,
        "positive_fold_count": int((fold_frame["delta_p_at_1_mu_1.2"] > 0).sum()),
        "zero_fold_count": int((fold_frame["delta_p_at_1_mu_1.2"] == 0).sum()),
        "negative_fold_count": int((fold_frame["delta_p_at_1_mu_1.2"] < 0).sum()),
        "mean_delta_p_at_1_mu_1.2": float(fold_frame["delta_p_at_1_mu_1.2"].mean()),
        "sd_delta_p_at_1_mu_1.2": float(fold_frame["delta_p_at_1_mu_1.2"].std(ddof=1)),
        "mean_native_p_at_1_mu_1.2": float(fold_frame["native_p_at_1_mu_1.2"].mean()),
        "sd_native_p_at_1_mu_1.2": float(fold_frame["native_p_at_1_mu_1.2"].std(ddof=1)),
        "mean_raw_p_at_1_mu_1.2": float(fold_frame["raw_p_at_1_mu_1.2"].mean()),
        "sd_raw_p_at_1_mu_1.2": float(fold_frame["raw_p_at_1_mu_1.2"].std(ddof=1)),
        "pooled_oof": pooled_payload,
        "gate_by_fold": gate_artifacts,
    }

    outputs: dict[str, Path] = {}
    outputs["fold_results.csv"] = _atomic_csv(cv_dir / "fold_results.csv", fold_frame)
    outputs["seed_results.csv"] = _atomic_csv(cv_dir / "seed_results.csv", seed_frame)
    outputs["oof_predictions.parquet"] = _atomic_parquet(
        cv_dir / "oof_predictions.parquet", oof_predictions
    )
    outputs["oof_metrics.csv"] = _atomic_csv(cv_dir / "oof_metrics.csv", oof_metrics)
    outputs["bootstrap_results.csv"] = _atomic_csv(
        cv_dir / "bootstrap_results.csv", bootstrap
    )
    outputs["significance_tests.json"] = _atomic_json(
        cv_dir / "significance_tests.json", significance
    )
    outputs["scene_level_effects.csv"] = _atomic_csv(
        cv_dir / "scene_level_effects.csv", scene_effects
    )
    outputs["paired_outcomes.parquet"] = _atomic_parquet(
        cv_dir / "paired_outcomes.parquet", paired_outcomes
    )
    outputs["fold_summary.json"] = _atomic_json(
        cv_dir / "fold_summary.json", fold_summary
    )
    _verify_source_unchanged(repo, source)
    completion = {
        "schema_version": "robustness_6d_scene_cv_completion_v1",
        "status": "COMPLETE_6D_SCENE_CV",
        "analysis_nature": "post-hoc robustness and sensitivity analysis",
        "source_run_id": SOURCE_RUN_ID,
        "condition": PRIMARY_CONDITION,
        "fold_count": 5,
        "scene_count": 30,
        "group_count": 1_440,
        "candidate_count": int(len(source.rows)),
        "fold_audit": fold_audit,
        "fold_output_hashes": fold_output_hashes,
        "source_hashes": dict(source.source_hashes),
        "preregistration_sha256": preregistration_sha256,
        "outputs": {name: sha256_file(path) for name, path in outputs.items()},
    }
    _atomic_json(completion_path, completion)
    return completion


__all__ = [
    "BOOTSTRAP_ITERATIONS",
    "EXPECTED_EARLY_STOPPING_ROUNDS",
    "EXPECTED_HYPERPARAMETERS",
    "EXPECTED_SCENES",
    "PRIMARY_CONDITION",
    "SOURCE_RUN_ID",
    "SPLIT_SEED",
    "assert_frozen_candidate_pool",
    "assign_group_splits",
    "fit_fold_preprocessor",
    "make_scene_folds",
    "run_scene_cv",
    "validate_scene_folds",
]
