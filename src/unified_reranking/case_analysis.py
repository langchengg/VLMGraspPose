"""Read-only post-formal case and mechanism analysis primitives.

This module deliberately has no writer that accepts the formal source run as an
output.  Callers provide an independent derived directory and use the source
only through hash-verified reads.
"""

from __future__ import annotations

import hashlib
import json
import pickletools
from collections import defaultdict
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# Keep LightGBM's native runtime ahead of PyTorch's OpenMP runtime on macOS.
try:
    import lightgbm as _lightgbm
except (
    ModuleNotFoundError
):  # pragma: no cover - explicit runtime preflight catches this
    _lightgbm = None

from .ablation import FEATURE_FAMILY_ORDER, feature_family
from .datasets import FoldPreprocessor, build_inference_query_arrays


ROUTES = ("crog", "g1", "c1")
EXPECTED_FINAL_LOCK_SHA256 = (
    "4b52eac6494e59a0f902792b794c3cca02824bf7a36bed4699b569986383f793"
)
EXPECTED = {
    "crog": {
        "native": 6848,
        "ungated": 7087,
        "gated": 7089,
        "recovered": 278,
        "harmful": 37,
    },
    "g1": {
        "native": 3647,
        "ungated": 4339,
        "gated": 4347,
        "recovered": 729,
        "harmful": 29,
    },
    "c1": {
        "native": 3363,
        "ungated": 4318,
        "gated": 4317,
        "recovered": 1006,
        "harmful": 52,
    },
}
GALLERY_QUOTAS = {
    "recovered": 12,
    "harmful": 12,
    "wrong_to_wrong_solvable": 8,
    "no_positive_pool": 8,
    "gate_prevented_harmful": 8,
    "gate_missed_recoverable": 8,
    "correct_to_correct_switch": 6,
}
MECHANISM_BY_FAMILY = {
    "native_calibration": "M1 native-score recalibration",
    "soft_target_support": "M2 soft target support",
    "jaw_geometry": "M3/M4 jaw geometry and contact alignment",
    "angle_agreement": "M4 orientation agreement",
    "depth_contact": "M5 depth/contact evidence",
    "collision_proxy": "M6 collision-proxy evidence",
    "backend_consensus": "M8 cross-backend consensus",
    "reliability_context": "M7 relational/reliability context",
}


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode()
    ).hexdigest()


def artifact_record(path: str | Path) -> dict[str, Any]:
    resolved = Path(path).resolve()
    return {
        "path": str(resolved),
        "sha256": sha256_file(resolved),
        "bytes": resolved.stat().st_size,
    }


def load_json(
    path: str | Path, *, expected_sha256: str | None = None
) -> dict[str, Any]:
    resolved = Path(path).resolve()
    if expected_sha256 is not None and sha256_file(resolved) != expected_sha256:
        raise RuntimeError(f"SHA-256 mismatch: {resolved}")
    value = json.loads(resolved.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {resolved}")
    return value


def verify_record(record: Mapping[str, Any], *, name: str) -> Path:
    path = Path(str(record.get("path", ""))).resolve()
    expected = str(record.get("sha256", ""))
    if len(expected) != 64 or not path.is_file() or sha256_file(path) != expected:
        raise RuntimeError(f"{name} artifact record is invalid")
    return path


def critical_source_snapshot(source_run: str | Path) -> dict[str, Any]:
    """Hash the formal authorities plus a non-content metadata inventory.

    The formal inventory already binds every immutable byte.  Rehashing all
    100+ GiB twice would add no independent semantic check, so this snapshot
    combines exact authority hashes with an exact relative-path/size/mtime
    inventory to detect any source-run write during analysis.
    """

    source = Path(source_run).resolve()
    authorities = [
        "FINAL_RUN_LOCK.json",
        "manifest.json",
        "COMPLETE",
        "08_lock/FORMAL_TEST_LOCK.json",
        "08_lock/formal_evaluation_plan.json",
        "08_lock/candidate_test_label_manifest.json",
        "09_formal_test/FORMAL_TEST_EXECUTION.json",
        "09_formal_test/formal_test_manifest.json",
        "09_formal_test/formal_test_metrics.json",
        "09_formal_test/formal_test_statistics.json",
    ]
    records = {name: artifact_record(source / name) for name in authorities}
    metadata_rows = []
    for path in sorted(source.rglob("*")):
        if path.is_file():
            stat = path.stat()
            metadata_rows.append(
                (str(path.relative_to(source)), stat.st_size, stat.st_mtime_ns)
            )
    final_lock = load_json(source / "FINAL_RUN_LOCK.json")
    if records["FINAL_RUN_LOCK.json"]["sha256"] != EXPECTED_FINAL_LOCK_SHA256:
        raise RuntimeError(
            "source FINAL_RUN_LOCK SHA-256 differs from the declared authority"
        )
    if (
        final_lock.get("status") != "COMPLETE"
        or int(final_lock.get("formal_test_execution_count", -1)) != 1
    ):
        raise RuntimeError("source final lock is not one-shot COMPLETE")
    if int(final_lock.get("inventory_count", -1)) != len(
        final_lock.get("inventory", [])
    ):
        raise RuntimeError("source final-lock inventory count is inconsistent")
    return {
        "source_run": str(source),
        "critical_artifacts": records,
        "metadata_inventory_count": len(metadata_rows),
        "metadata_inventory_sha256": canonical_sha256(metadata_rows),
        "final_lock_inventory_count": int(final_lock["inventory_count"]),
        "formal_test_execution_count": 1,
    }


def source_index(source_run: str | Path) -> pd.DataFrame:
    source = Path(source_run).resolve()
    rows: list[dict[str, Any]] = []
    names = [
        "FINAL_RUN_LOCK.json",
        "manifest.json",
        "COMPLETE",
        "08_lock/FORMAL_TEST_LOCK.json",
        "08_lock/formal_evaluation_plan.json",
        "08_lock/candidate_test_label_manifest.json",
        "09_formal_test/formal_test_manifest.json",
        "09_formal_test/FORMAL_TEST_EXECUTION.json",
        "09_formal_test/formal_test_metrics.json",
        "09_formal_test/formal_test_statistics.json",
        "09_formal_test/per_candidate_scores.parquet",
        "09_formal_test/per_sample_decisions.parquet",
        "09_formal_test/formal_test_realized_rankings.parquet",
        "09_formal_test/formal_test_outcomes_wide.parquet",
        "15_independent_recompute/INDEPENDENT_RECOMPUTE.json",
        "15_independent_recompute/independent_per_sample.parquet",
        "13_failure_galleries/failure_taxonomy_per_sample.parquet",
        "01_manifests/paired_test.parquet",
        "07_validation/selected_primary_ungated.json",
        "07_validation/ablations/feature_ablation_manifest.json",
        "07_validation/ablations/cumulative_feature_ablation.csv",
        "07_validation/ablations/leave_one_family_out_ablation.csv",
        "08_lock/selected_features.json",
        "08_lock/selected_hyperparameters.json",
        "08_lock/gate_thresholds.json",
        "tables/TABLE_GENERATION_MANIFEST.json",
        "14_reports/FINAL_REPORT_EN.md",
    ]
    for route in ROUTES:
        names.extend(
            [
                f"02_candidates/{route}_test_top5.parquet",
                f"02_candidates/{route}_test_all.parquet",
                f"03_features/tracks/T2_matched_common/{route}_test/candidate_features.parquet",
                f"08_lock/label_free_test_rankers/{route}/manifest.json",
                f"08_lock/label_free_test_rankers/{route}/per_candidate_scores.parquet",
                f"08_lock/label_free_test_rankers/{route}/per_sample_decisions.parquet",
                f"08_lock/label_free_test_gates/{route}/manifest.json",
                f"08_lock/label_free_test_gates/{route}/gate_test_inputs.parquet",
                f"08_lock/label_free_test_gates/{route}/gate_test_decisions.parquet",
            ]
        )
    for name in names:
        path = source / name
        if not path.is_file():
            raise FileNotFoundError(path)
        rows.append({"role": name, **artifact_record(path)})
    selection = load_json(source / "07_validation/selected_primary_ungated.json")
    for route in ROUTES:
        for role in ("validation_manifest", "oof_manifest"):
            path = Path(selection["selections"][route][role])
            rows.append({"role": f"selected/{route}/{role}", **artifact_record(path)})
        ranker = load_json(
            source / f"08_lock/label_free_test_rankers/{route}/manifest.json"
        )
        for seed, record in ranker["sources"]["applications"].items():
            app_path = verify_record(record, name=f"{route}/{seed} application")
            app = load_json(app_path)
            cell_path = verify_record(
                app["sources"]["cell_manifest"], name=f"{route}/{seed} cell"
            )
            cell = load_json(cell_path)
            rows.append(
                {
                    "role": f"selected/{route}/seed_{seed}/cell_manifest",
                    **artifact_record(cell_path),
                }
            )
            rows.append(
                {
                    "role": f"selected/{route}/seed_{seed}/model",
                    **artifact_record(
                        verify_record(cell["artifacts"]["model"], name="model")
                    ),
                }
            )
    return pd.DataFrame(rows)


def _route_labels(source: Path) -> pd.DataFrame:
    manifest = load_json(source / "08_lock/candidate_test_label_manifest.json")
    label_path = Path(str(manifest["candidate_labels_path"]))
    if sha256_file(label_path) != str(manifest["candidate_labels_sha256"]):
        raise RuntimeError(
            "candidate diagnostic label table differs from locked manifest"
        )
    frame = pd.read_parquet(label_path)
    normalization = manifest.get("normalization", {})
    variant_column = str(normalization.get("variant_column", "variant"))
    route_column = str(normalization.get("route_column", "method"))
    include_variants = list(map(str, normalization.get("include_variants", ())))
    if variant_column not in frame or route_column not in frame or not include_variants:
        raise RuntimeError("locked label normalization contract is incomplete")
    frame = frame[frame[variant_column].astype(str).isin(include_variants)].copy()
    route_map = {
        "CROG": "crog",
        "G1": "g1",
        "C1": "c1",
        "crog": "crog",
        "g1": "g1",
        "c1": "c1",
    }
    frame["route"] = frame[route_column].map(route_map)
    frame = frame[frame["route"].isin(ROUTES)].copy()
    frame["sample_id"] = frame["sample_id"].astype(str)
    frame["candidate_id"] = frame["candidate_id"].astype(str)
    keep = [
        "route",
        "sample_id",
        "candidate_id",
        "candidate_success",
        "diagnostic_gt_index",
        "diagnostic_iou",
        "diagnostic_angle_error_deg",
    ]
    if frame[keep[:3]].duplicated().any():
        raise RuntimeError("canonical diagnostic labels duplicate route/candidate keys")
    return frame[keep]


def _gate_reasons(
    inputs: pd.DataFrame, decisions: pd.DataFrame, point: Mapping[str, Any]
) -> pd.DataFrame:
    frame = inputs.merge(
        decisions[
            [
                "sample_id",
                "probability_recover",
                "probability_harm",
                "utility",
                "switch",
            ]
        ],
        on="sample_id",
        how="left",
        validate="one_to_one",
    )
    checks = {
        "utility": frame["utility"] > float(point["utility_threshold"]),
        "score_margin": frame["score_margin"] > float(point["score_margin_threshold"]),
        "reliability": frame["challenger_reliability"]
        > float(point["reliability_threshold"]),
        "stability": frame["perturbation_stability"]
        >= float(point["stability_threshold"]),
        "seed_votes": frame["seed_challenger_votes"]
        >= int(point["minimum_seed_votes"]),
        "candidate_immutable": frame["candidate_id_unchanged"].astype(bool),
        "geometry_immutable": frame["geometry_hash_unchanged"].astype(bool),
        "challenger_exists": frame["challenger_exists"].astype(bool),
    }
    for name, values in checks.items():
        frame[f"gate_check_{name}"] = values
    computed = np.logical_and.reduce(
        [values.to_numpy(bool) for values in checks.values()]
    )
    if not np.array_equal(computed, frame["switch"].to_numpy(bool)):
        raise RuntimeError("gate decision differs from locked threshold conjunction")
    reason_order = list(checks)
    frame["gate_decision_reason"] = [
        "accepted_all_conditions"
        if accepted
        else "rejected_"
        + "+".join(
            name
            for name in reason_order
            if not bool(frame.iloc[index][f"gate_check_{name}"])
        )
        for index, accepted in enumerate(computed)
    ]
    return frame


def build_decision_chain(
    source_run: str | Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    source = Path(source_run).resolve()
    paired = pd.read_parquet(source / "01_manifests/paired_test.parquet")
    if len(paired) != 7675 or paired["sample_id"].astype(str).nunique() != 7675:
        raise RuntimeError("paired Test denominator is not 7,675 unique samples")
    paired["sample_id"] = paired["sample_id"].astype(str)
    taxonomy = pd.read_parquet(
        source / "13_failure_galleries/failure_taxonomy_per_sample.parquet"
    )
    taxonomy["sample_id"] = taxonomy["sample_id"].astype(str)
    diagnostics = _route_labels(source)
    thresholds = load_json(source / "08_lock/gate_thresholds.json")["routes"]
    sample_frames: list[pd.DataFrame] = []
    candidate_frames: list[pd.DataFrame] = []
    gate_frames: list[pd.DataFrame] = []
    for route in ROUTES:
        candidates = pd.read_parquet(
            source / f"02_candidates/{route}_test_top5.parquet"
        )
        candidates["sample_id"] = candidates["sample_id"].astype(str)
        candidates["candidate_id"] = candidates["candidate_id"].astype(str)
        scores = pd.read_parquet(
            source
            / f"08_lock/label_free_test_rankers/{route}/per_candidate_scores.parquet"
        )
        scores[["sample_id", "candidate_id"]] = scores[
            ["sample_id", "candidate_id"]
        ].astype(str)
        ranker_decisions = pd.read_parquet(
            source
            / f"08_lock/label_free_test_rankers/{route}/per_sample_decisions.parquet"
        )
        gate_inputs = pd.read_parquet(
            source / f"08_lock/label_free_test_gates/{route}/gate_test_inputs.parquet"
        )
        gate_decisions = pd.read_parquet(
            source
            / f"08_lock/label_free_test_gates/{route}/gate_test_decisions.parquet"
        )
        for frame in (ranker_decisions, gate_inputs, gate_decisions):
            frame["sample_id"] = frame["sample_id"].astype(str)
        candidate = candidates.merge(
            scores,
            on=["sample_id", "candidate_id", "native_rank"],
            validate="one_to_one",
        )
        candidate["route"] = route
        candidate = candidate.merge(
            diagnostics[diagnostics["route"] == route],
            on=["route", "sample_id", "candidate_id"],
            how="left",
            validate="one_to_one",
        )
        if candidate["candidate_success"].isna().any():
            raise RuntimeError(f"{route} diagnostic labels do not exactly cover Top-5")
        candidate["improved_rank"] = (
            candidate.groupby("sample_id")["ensemble_score"]
            .rank(method="first", ascending=False)
            .astype(int)
        )
        # Restore exact stable tie-breaking rather than relying on frame order.
        candidate = candidate.sort_values(
            ["sample_id", "ensemble_score", "native_rank", "candidate_id"],
            ascending=[True, False, True, True],
            kind="mergesort",
        )
        candidate["improved_rank"] = candidate.groupby("sample_id").cumcount() + 1
        top = candidate[candidate["improved_rank"] == 1][
            ["sample_id", "candidate_id"]
        ].rename(columns={"candidate_id": "computed_ungated_candidate_id"})
        if (
            not top.merge(
                ranker_decisions[["sample_id", "selected_candidate_id"]],
                on="sample_id",
                validate="one_to_one",
            )
            .eval("computed_ungated_candidate_id == selected_candidate_id")
            .all()
        ):
            raise RuntimeError(
                f"{route} exact reranking differs from locked ungated Top-1"
            )
        gate = _gate_reasons(
            gate_inputs, gate_decisions, thresholds[route]["operating_point"]
        )
        gate.insert(0, "route", route)
        gate_frames.append(gate)
        tax = taxonomy[taxonomy["route"] == route].copy()
        sample = paired.merge(
            tax,
            on=["sample_id", "scene_id", "frame_id"],
            how="left",
            validate="one_to_one",
        )
        sample["route"] = route
        sample = sample.merge(
            ranker_decisions[
                [
                    "sample_id",
                    "ensemble_score_margin",
                    "seed_challenger_votes",
                    "challenger_exists",
                ]
            ],
            on="sample_id",
            how="left",
            validate="one_to_one",
        )
        sample = sample.merge(
            gate[
                [
                    "sample_id",
                    "probability_recover",
                    "probability_harm",
                    "utility",
                    "switch",
                    "score_margin",
                    "challenger_reliability",
                    "perturbation_stability",
                    "gate_decision_reason",
                    *[
                        f"gate_check_{name}"
                        for name in (
                            "utility",
                            "score_margin",
                            "reliability",
                            "stability",
                            "seed_votes",
                            "candidate_immutable",
                            "geometry_immutable",
                            "challenger_exists",
                        )
                    ],
                ]
            ],
            on="sample_id",
            how="left",
            validate="one_to_one",
        )
        sample["outcome"] = np.select(
            [
                (~sample["native_correct"]) & sample["gated_correct"],
                sample["native_correct"] & (~sample["gated_correct"]),
                sample["native_correct"] & sample["gated_correct"],
            ],
            ["recovered", "harmful", "correct_retained"],
            default="wrong_retained",
        )
        sample["analysis_category"] = np.select(
            [
                (~sample["native_correct"]) & sample["gated_correct"],
                sample["native_correct"] & (~sample["gated_correct"]),
                (~sample["native_correct"])
                & (~sample["gated_correct"])
                & sample["top5_positive"]
                & (~sample["E9"]),
                sample["E1"],
                sample["native_correct"]
                & (~sample["ungated_correct"])
                & sample["gated_correct"],
                (~sample["native_correct"])
                & sample["ungated_correct"]
                & (~sample["gated_correct"]),
                sample["native_correct"]
                & sample["gated_correct"]
                & (sample["native_candidate_id"] != sample["gated_candidate_id"]),
            ],
            [
                "recovered",
                "harmful",
                "wrong_to_wrong_solvable",
                "no_positive_pool",
                "gate_prevented_harmful",
                "gate_missed_recoverable",
                "correct_to_correct_switch",
            ],
            default="other",
        )
        candidate_frames.append(candidate)
        sample_frames.append(sample)
    samples = pd.concat(sample_frames, ignore_index=True)
    candidates = pd.concat(candidate_frames, ignore_index=True)
    gates = pd.concat(gate_frames, ignore_index=True)
    if len(samples) != 23025 or samples[["route", "sample_id"]].duplicated().any():
        raise RuntimeError("decision chain is not the exact 7,675 x 3 route universe")
    return samples, candidates, gates


def reconcile_formal(source_run: str | Path, samples: pd.DataFrame) -> pd.DataFrame:
    source = Path(source_run).resolve()
    metrics = load_json(source / "09_formal_test/formal_test_metrics.json")["systems"]
    rows = []
    mismatches = []
    for route in ROUTES:
        frame = samples[samples["route"] == route]
        values = {
            "native": int(frame["native_correct"].sum()),
            "ungated": int(frame["ungated_correct"].sum()),
            "gated": int(frame["gated_correct"].sum()),
            "recovered": int(
                ((~frame["native_correct"]) & frame["gated_correct"]).sum()
            ),
            "harmful": int((frame["native_correct"] & (~frame["gated_correct"])).sum()),
        }
        for key, expected in EXPECTED[route].items():
            if values[key] != expected:
                mismatches.append((route, key, values[key], expected))
        if values["native"] != int(metrics[f"{route}_native"]["j_at_1_numerator"]):
            mismatches.append(
                (
                    route,
                    "formal_native",
                    values["native"],
                    metrics[f"{route}_native"]["j_at_1_numerator"],
                )
            )
        if values["gated"] != int(
            metrics[f"{route}_gated_primary"]["j_at_1_numerator"]
        ):
            mismatches.append(
                (
                    route,
                    "formal_gated",
                    values["gated"],
                    metrics[f"{route}_gated_primary"]["j_at_1_numerator"],
                )
            )
        oracle = int(frame["top5_positive"].sum())
        headroom = oracle / len(frame) - values["native"] / len(frame)
        delta = values["gated"] / len(frame) - values["native"] / len(frame)
        rows.append(
            {
                "route": route,
                "sample_count": len(frame),
                **values,
                "native_j_at_1": values["native"] / len(frame),
                "ungated_j_at_1": values["ungated"] / len(frame),
                "gated_j_at_1": values["gated"] / len(frame),
                "delta_percentage_points": delta * 100,
                "oracle_at_5": oracle / len(frame),
                "headroom_percentage_points": headroom * 100,
                "headroom_recovery_at_5": delta / headroom,
                "net": values["recovered"] - values["harmful"],
                "irreparable": int((frame["E0"] | frame["E1"] | frame["E2"]).sum()),
            }
        )
    if mismatches:
        raise RuntimeError("SOURCE_RESULT_MISMATCH " + repr(mismatches))
    return pd.DataFrame(rows)


class NativeBooster:
    """Opcode-only safe restoration of the native model string."""

    def __init__(self, path: str | Path) -> None:
        strings = [
            argument
            for _opcode, argument, _position in pickletools.genops(
                Path(path).read_bytes()
            )
            if isinstance(argument, str) and argument.startswith("tree\nversion=")
        ]
        if len(strings) != 1:
            raise RuntimeError(
                "locked model must contain exactly one native LightGBM string"
            )
        if _lightgbm is None:
            raise ModuleNotFoundError("LightGBM is required for contribution replay")
        self.booster = _lightgbm.Booster(model_str=strings[0])

    def predict(self, matrix: np.ndarray, *, pred_contrib: bool = False) -> np.ndarray:
        return np.asarray(
            self.booster.predict(
                np.asarray(matrix, dtype=np.float64), pred_contrib=pred_contrib
            )
        )


def _flat_matrix(arrays: Any) -> np.ndarray:
    return arrays.features.numpy()[~arrays.padding_mask.numpy()]


def replay_contributions(
    source_run: str | Path, candidates: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Replay all three frozen seed models and average exact tree contributions."""

    source = Path(source_run).resolve()
    contribution_parts: list[pd.DataFrame] = []
    importance_rows: list[dict[str, Any]] = []
    replay_rows: list[dict[str, Any]] = []
    for route in ROUTES:
        feature_path = (
            source
            / f"03_features/tracks/T2_matched_common/{route}_test/candidate_features.parquet"
        )
        features = (
            pd.read_parquet(feature_path)
            .sort_values(["sample_id", "native_rank", "candidate_id"], kind="mergesort")
            .reset_index(drop=True)
        )
        ranker_manifest = load_json(
            source / f"08_lock/label_free_test_rankers/{route}/manifest.json"
        )
        locked_scores = pd.read_parquet(
            ranker_manifest["artifacts"]["predictions"]["path"]
        )
        locked_scores = locked_scores.sort_values(
            ["sample_id", "native_rank", "candidate_id"], kind="mergesort"
        ).reset_index(drop=True)
        seed_contribs = []
        seed_scores = []
        feature_names: tuple[str, ...] | None = None
        for seed in (42, 123, 2026):
            app_record = ranker_manifest["sources"]["applications"][str(seed)]
            app_path = verify_record(app_record, name=f"{route}/{seed} application")
            app = load_json(app_path)
            cell_path = verify_record(
                app["sources"]["cell_manifest"], name=f"{route}/{seed} cell"
            )
            cell = load_json(cell_path)
            model_path = verify_record(
                cell["artifacts"]["model"], name=f"{route}/{seed} model"
            )
            pre = FoldPreprocessor.from_artifact(cell["preprocessor"])
            arrays = build_inference_query_arrays(features, preprocessor=pre)
            matrix = _flat_matrix(arrays)
            model = NativeBooster(model_path)
            scores = model.predict(matrix)
            contrib = model.predict(matrix, pred_contrib=True)
            if contrib.shape != (len(matrix), matrix.shape[1] + 1):
                raise RuntimeError("LightGBM contribution matrix has an invalid shape")
            if not np.allclose(contrib.sum(axis=1), scores, atol=1e-8, rtol=1e-8):
                raise RuntimeError("LightGBM contribution additivity failed")
            expected = locked_scores[f"score_seed_{seed}"].to_numpy(float)
            max_error = float(np.max(np.abs(scores - expected)))
            if max_error > 1e-8:
                raise RuntimeError(
                    f"{route}/{seed} replay differs from locked scores: {max_error}"
                )
            names = tuple(pre.columns)
            if feature_names is None:
                feature_names = names
            elif feature_names != names:
                raise RuntimeError("seed feature schemas differ")
            seed_scores.append(scores)
            seed_contribs.append(contrib)
            gain = model.booster.feature_importance(importance_type="gain")
            split = model.booster.feature_importance(importance_type="split")
            for name, gain_value, split_value in zip(names, gain, split, strict=True):
                importance_rows.append(
                    {
                        "route": route,
                        "seed": seed,
                        "feature": name,
                        "family": feature_family(name),
                        "gain": float(gain_value),
                        "split": int(split_value),
                    }
                )
        assert feature_names is not None
        ensemble_score = np.mean(seed_scores, axis=0)
        if not np.allclose(
            ensemble_score,
            locked_scores["ensemble_score"].to_numpy(float),
            atol=1e-8,
            rtol=1e-8,
        ):
            raise RuntimeError(f"{route} ensemble replay differs from locked scores")
        ensemble_contrib = np.mean(seed_contribs, axis=0)
        rows = locked_scores[["sample_id", "candidate_id", "native_rank"]].copy()
        for index, name in enumerate(feature_names):
            rows[f"contrib::{name}"] = ensemble_contrib[:, index]
        rows["expected_value"] = ensemble_contrib[:, -1]
        rows["replayed_score"] = ensemble_score
        rows.insert(0, "route", route)
        contribution_parts.append(rows)
        replay_rows.append(
            {
                "route": route,
                "candidate_rows": len(rows),
                "feature_count": len(feature_names),
                "max_score_error": float(
                    np.max(
                        np.abs(
                            ensemble_score
                            - locked_scores["ensemble_score"].to_numpy(float)
                        )
                    )
                ),
                "max_additivity_error": float(
                    np.max(np.abs(ensemble_contrib.sum(axis=1) - ensemble_score))
                ),
                "seed_count": 3,
            }
        )
    return (
        pd.concat(contribution_parts, ignore_index=True),
        pd.DataFrame(importance_rows),
        pd.DataFrame(replay_rows),
    )


def pair_contribution_analysis(
    samples: pd.DataFrame, contributions: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    contribution_columns = [
        column for column in contributions if column.startswith("contrib::")
    ]
    lookup = contributions.set_index(["route", "sample_id", "candidate_id"])
    pair_rows: list[dict[str, Any]] = []
    long_rows: list[dict[str, Any]] = []
    for row in samples.itertuples(index=False):
        native_key = (row.route, row.sample_id, str(row.native_candidate_id))
        challenger_key = (row.route, row.sample_id, str(row.ungated_candidate_id))
        if native_key not in lookup.index or challenger_key not in lookup.index:
            if int(row.candidate_count_top5) != 0:
                raise RuntimeError(
                    "decision candidate is missing from contribution table"
                )
            pair_rows.append(
                {
                    "route": row.route,
                    "sample_id": row.sample_id,
                    "outcome": row.outcome,
                    "native_candidate_id": row.native_candidate_id,
                    "challenger_candidate_id": row.ungated_candidate_id,
                    "final_candidate_id": row.gated_candidate_id,
                    "score_margin": float("nan"),
                    "expected_value_delta": float("nan"),
                    "dominant_family": "not_applicable_no_output",
                    "dominant_family_delta": float("nan"),
                    "dominant_abs_share": float("nan"),
                    "mechanism": "not_applicable_no_output",
                    **{
                        f"family_delta::{family}": float("nan")
                        for family in FEATURE_FAMILY_ORDER
                    },
                }
            )
            continue
        native = lookup.loc[native_key]
        challenger = lookup.loc[challenger_key]
        family_delta: dict[str, float] = defaultdict(float)
        for column in contribution_columns:
            feature = column.split("::", 1)[1]
            delta = float(challenger[column] - native[column])
            family = feature_family(feature)
            family_delta[family] += delta
            if row.ungated_candidate_id != row.native_candidate_id:
                long_rows.append(
                    {
                        "route": row.route,
                        "sample_id": row.sample_id,
                        "outcome": row.outcome,
                        "feature": feature,
                        "family": family,
                        "challenger_minus_native": delta,
                    }
                )
        magnitude = sum(abs(value) for value in family_delta.values())
        ordered = sorted(
            family_delta.items(), key=lambda item: (-abs(item[1]), item[0])
        )
        dominant_family, dominant_delta = ordered[0]
        share = abs(dominant_delta) / magnitude if magnitude else 0.0
        mechanism = (
            MECHANISM_BY_FAMILY[dominant_family]
            if share >= 0.35
            else "mixed_no_single_dominant"
        )
        pair_rows.append(
            {
                "route": row.route,
                "sample_id": row.sample_id,
                "outcome": row.outcome,
                "native_candidate_id": row.native_candidate_id,
                "challenger_candidate_id": row.ungated_candidate_id,
                "final_candidate_id": row.gated_candidate_id,
                "score_margin": row.score_margin,
                "expected_value_delta": float(
                    challenger["expected_value"] - native["expected_value"]
                ),
                "dominant_family": dominant_family,
                "dominant_family_delta": dominant_delta,
                "dominant_abs_share": share,
                "mechanism": mechanism,
                **{
                    f"family_delta::{family}": family_delta.get(family, 0.0)
                    for family in FEATURE_FAMILY_ORDER
                },
            }
        )
    return pd.DataFrame(pair_rows), pd.DataFrame(long_rows)


def enrich_samples(
    samples: pd.DataFrame, candidates: pd.DataFrame, pairs: pd.DataFrame
) -> pd.DataFrame:
    candidate_index = candidates.set_index(["route", "sample_id", "candidate_id"])
    output = samples.merge(
        pairs,
        on=["route", "sample_id", "outcome", "native_candidate_id"],
        how="left",
        validate="one_to_one",
        suffixes=("", "_pair"),
    )
    for prefix, column in (
        ("native", "native_candidate_id"),
        ("ungated", "ungated_candidate_id"),
        ("final", "gated_candidate_id"),
    ):
        keys = list(
            zip(
                output["route"],
                output["sample_id"],
                output[column].astype(str),
                strict=True,
            )
        )
        matched = candidate_index.reindex(
            pd.MultiIndex.from_tuples(keys, names=candidate_index.index.names)
        )
        for value in (
            "native_rank",
            "native_score",
            "ensemble_score",
            "candidate_success",
            "diagnostic_iou",
            "diagnostic_angle_error_deg",
            "cx_px",
            "cy_px",
            "theta_deg",
            "width_px",
            "height_px",
        ):
            if value in matched:
                output[f"{prefix}_{value}"] = matched[value].to_numpy()
    return output


def deterministic_case_selection(samples: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for route in ROUTES:
        route_frame = samples[samples["route"] == route]
        for category, quota in GALLERY_QUOTAS.items():
            eligible = route_frame[route_frame["analysis_category"] == category].copy()
            eligible["selection_sha256"] = [
                hashlib.sha256(f"{route}|{category}|{sample_id}".encode()).hexdigest()
                for sample_id in eligible["sample_id"]
            ]
            chosen = eligible.sort_values("selection_sha256").head(
                min(quota, len(eligible))
            )
            for order, row in enumerate(chosen.itertuples(index=False), start=1):
                rows.append(
                    {
                        "route": route,
                        "category": category,
                        "selection_order": order,
                        "sample_id": row.sample_id,
                        "scene_id": row.scene_id,
                        "frame_id": row.frame_id,
                        "selection_sha256": row.selection_sha256,
                        "mechanism": getattr(row, "mechanism", "not_applicable"),
                    }
                )
    result = pd.DataFrame(rows)
    if result.duplicated(["route", "category", "sample_id"]).any():
        raise RuntimeError("case selection is not unique")
    return result


def global_tables(
    samples: pd.DataFrame, candidates: pd.DataFrame, pairs: pd.DataFrame
) -> dict[str, pd.DataFrame]:
    outcome = (
        samples.groupby(["route", "outcome"], observed=True)
        .size()
        .rename("count")
        .reset_index()
    )
    ranks = (
        candidates.groupby(["route", "native_rank"], observed=True)["candidate_success"]
        .agg(candidate_rows="size", positive_count="sum", positive_rate="mean")
        .reset_index()
    )
    gate = (
        samples.groupby(["route", "gate_decision_reason"], observed=True)
        .size()
        .rename("count")
        .reset_index()
    )
    mechanism = (
        pairs[pairs["native_candidate_id"] != pairs["challenger_candidate_id"]]
        .groupby(["route", "outcome", "mechanism"], observed=True)
        .size()
        .rename("count")
        .reset_index()
    )
    iou_angle = []
    for route in ROUTES:
        frame = samples[samples["route"] == route]
        for outcome_name in (
            "recovered",
            "harmful",
            "correct_retained",
            "wrong_retained",
        ):
            selected = frame[frame["outcome"] == outcome_name]
            iou_angle.append(
                {
                    "route": route,
                    "outcome": outcome_name,
                    "count": len(selected),
                    "native_iou_mean": selected["native_diagnostic_iou"].mean(),
                    "final_iou_mean": selected["final_diagnostic_iou"].mean(),
                    "native_angle_error_mean": selected[
                        "native_diagnostic_angle_error_deg"
                    ].mean(),
                    "final_angle_error_mean": selected[
                        "final_diagnostic_angle_error_deg"
                    ].mean(),
                }
            )
    return {
        "outcome_counts": outcome,
        "native_rank_positive_profile": ranks,
        "gate_rejection_reasons": gate,
        "dominant_mechanisms": mechanism,
        "iou_angle_by_outcome": pd.DataFrame(iou_angle),
    }


def build_inventory(root: str | Path, *, exclude: Iterable[str] = ()) -> pd.DataFrame:
    base = Path(root).resolve()
    excluded = set(exclude)
    rows = []
    for path in sorted(base.rglob("*")):
        if path.is_file() and str(path.relative_to(base)) not in excluded:
            rows.append(
                {
                    "relative_path": str(path.relative_to(base)),
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    return pd.DataFrame(rows)


__all__ = [
    "EXPECTED_FINAL_LOCK_SHA256",
    "GALLERY_QUOTAS",
    "MECHANISM_BY_FAMILY",
    "ROUTES",
    "artifact_record",
    "build_decision_chain",
    "build_inventory",
    "canonical_sha256",
    "critical_source_snapshot",
    "deterministic_case_selection",
    "enrich_samples",
    "global_tables",
    "load_json",
    "pair_contribution_analysis",
    "reconcile_formal",
    "replay_contributions",
    "sha256_file",
    "source_index",
    "verify_record",
]
