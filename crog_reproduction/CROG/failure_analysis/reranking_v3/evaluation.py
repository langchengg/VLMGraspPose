from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np

from failure_analysis.reranking_v2.schema import read_jsonl

from .calibration import risk_coverage_curve
from .metrics import paired_switch_metrics, ranking_metrics, strip_metric_arrays
from .reporting import write_csv
from .schema import artifact_identity, atomic_write_json, stable_sample_id
from .statistics import holm_adjust, paired_cluster_statistics


def _unique_records(
    path: str | Path, *, key, artifact_name: str,
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for record in read_jsonl(path):
        sample_id = str(key(record))
        if sample_id in result:
            raise ValueError(f"duplicate {artifact_name} sample ID: {sample_id}")
        result[sample_id] = record
    return result


def _require_exact_cohort(
    observed: set[str], expected: set[str], *, artifact_name: str,
) -> None:
    missing = sorted(expected - observed)
    extra = sorted(observed - expected)
    if missing or extra:
        raise ValueError(
            f"{artifact_name} cohort differs from the frozen evaluation cohort: "
            f"missing={missing[:5]}, extra={extra[:5]}"
        )


def _cohort(
    *, features_path: str | Path, label_path: str | Path, allowed_ids: set[str] | None,
) -> tuple[list[str], dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    features = _unique_records(
        features_path,
        key=lambda value: stable_sample_id(str(value["split"]), value["sample_id"]),
        artifact_name="feature",
    )
    labels = _unique_records(
        label_path, key=lambda value: value["sample_id"], artifact_name="label",
    )
    expected = set(features) if allowed_ids is None else set(map(str, allowed_ids))
    if not expected:
        raise ValueError("evaluation cohort is empty")
    _require_exact_cohort(set(features), expected, artifact_name="feature")
    _require_exact_cohort(set(labels), expected, artifact_name="label")
    ids = sorted(expected)
    return ids, features, labels


def _candidate_ids(feature: Mapping[str, Any]) -> list[str]:
    values = [str(value["candidate_id"]) for value in feature["candidates"]]
    if len(values) != 5 or len(set(values)) != 5: raise ValueError("frozen candidate identity is invalid")
    return values


def _candidate_identity(feature: Mapping[str, Any]) -> tuple[list[str], dict[str, str]]:
    candidate_ids = _candidate_ids(feature)
    checksums = {
        str(value["candidate_id"]): str(value["candidate_checksum"])
        for value in feature["candidates"]
    }
    if len(checksums) != 5:
        raise ValueError("frozen candidate checksum identity is invalid")
    return candidate_ids, checksums


def _validated_probabilities(
    record: Mapping[str, Any], *, candidate_ids: list[str], sample_id: str,
) -> np.ndarray | None:
    raw = record.get("candidate_correctness_probabilities")
    if raw is None:
        return None
    values = np.asarray(raw, dtype=np.float64)
    if (
        values.shape != (5,)
        or not np.isfinite(values).all()
        or np.any(values < 0.0)
        or np.any(values > 1.0)
    ):
        raise ValueError(f"invalid candidate probabilities: {sample_id}")
    probability_ids = record.get("candidate_probability_ids")
    if probability_ids is None:
        # Historical V2 artifacts define this vector in the frozen candidate
        # order even though candidate_order is the ranking order.
        return values
    observed_ids = list(map(str, probability_ids))
    if len(observed_ids) != 5 or len(set(observed_ids)) != 5 or set(observed_ids) != set(candidate_ids):
        raise ValueError(f"candidate probability identity differs: {sample_id}")
    by_id = dict(zip(observed_ids, values, strict=True))
    return np.asarray([by_id[value] for value in candidate_ids], dtype=np.float64)


def _method_arrays(
    *, ids: list[str], features: Mapping[str, dict[str, Any]], prediction_path: str | Path | None,
) -> tuple[np.ndarray, np.ndarray]:
    predictions = None if prediction_path is None else _unique_records(
        prediction_path, key=lambda value: value["sample_id"], artifact_name="prediction",
    )
    if predictions is not None:
        _require_exact_cohort(set(predictions), set(ids), artifact_name="prediction")
    rankings = np.empty((len(ids), 5), np.int64); probabilities = np.empty((len(ids), 5), np.float64)
    for row, sample_id in enumerate(ids):
        candidate_ids = _candidate_ids(features[sample_id]); index = {value: i for i, value in enumerate(candidate_ids)}
        if predictions is None:
            order = candidate_ids
            probability = [float(value.get("q_probability", value["q_raw"])) for value in features[sample_id]["candidates"]]
        else:
            record = predictions[sample_id]; order = list(map(str, record["candidate_order"])); probability = _validated_probabilities(record, candidate_ids=candidate_ids, sample_id=sample_id)
            if probability is None: probability = [float(value.get("q_probability", value["q_raw"])) for value in features[sample_id]["candidates"]]
        if len(order) != 5 or set(order) != set(candidate_ids): raise ValueError(f"method changed candidate pool: {sample_id}")
        rankings[row] = [index[value] for value in order]
        probability_values = np.asarray(probability, dtype=np.float64)
        if (
            probability_values.shape != (5,)
            or not np.isfinite(probability_values).all()
            or np.any(probability_values < 0.0)
            or np.any(probability_values > 1.0)
        ):
            raise ValueError(f"invalid candidate probabilities: {sample_id}")
        probabilities[row] = probability_values
    return rankings, probabilities


def _labels(
    ids: list[str], labels: Mapping[str, dict[str, Any]],
    features: Mapping[str, dict[str, Any]],
) -> np.ndarray:
    result = []
    for sample_id in ids:
        candidate_ids, checksums = _candidate_identity(features[sample_id])
        values = labels[sample_id]["candidate_labels"]
        by_id: dict[str, Mapping[str, Any]] = {}
        for value in values:
            candidate_id = str(value["candidate_id"])
            if candidate_id in by_id:
                raise ValueError(f"duplicate label candidate ID: {sample_id}/{candidate_id}")
            by_id[candidate_id] = value
        if len(by_id) != 5 or set(by_id) != set(candidate_ids):
            raise ValueError(f"label candidate IDs differ: {sample_id}")
        for candidate_id in candidate_ids:
            if str(by_id[candidate_id].get("candidate_checksum")) != checksums[candidate_id]:
                raise ValueError(f"label candidate checksum differs: {sample_id}/{candidate_id}")
        result.append([float(by_id[value]["candidate_correct"]) for value in candidate_ids])
    return np.asarray(result, dtype=np.float64)


def evaluate_method_suite(
    *,
    features_path: str | Path,
    corrected_labels_path: str | Path,
    legacy_labels_path: str | Path,
    methods: Mapping[str, str | Path | None],
    output_dir: str | Path,
    allowed_ids: set[str] | None = None,
    iterations: int = 10_000,
    seed: int = 20260801,
) -> dict[str, Any]:
    """Explicit evaluation-stage label join for corrected and legacy tracks."""
    output = Path(output_dir)
    if output.exists(): raise FileExistsError(output)
    if "q_only" not in methods or methods["q_only"] is not None: raise ValueError("q_only must be the feature-order baseline")
    if "v2_locked_primary" not in methods: raise ValueError("V2 locked baseline is required")
    corrected_ids, features, corrected_records = _cohort(features_path=features_path, label_path=corrected_labels_path, allowed_ids=allowed_ids)
    legacy_ids, _, legacy_records = _cohort(features_path=features_path, label_path=legacy_labels_path, allowed_ids=allowed_ids)
    if corrected_ids != legacy_ids: raise ValueError("corrected/legacy cohorts differ")
    ids = corrected_ids
    method_arrays = {name: _method_arrays(ids=ids, features=features, prediction_path=path) for name, path in methods.items()}
    output.mkdir(parents=True)
    results: dict[str, Any] = {}; result_rows = []; statistical_records = []; calibration_payload = {"reliability": {}, "risk_coverage": {}}
    for track, label_records in (("corrected_scientific", corrected_records), ("legacy_official_compatibility", legacy_records)):
        y = _labels(ids, label_records, features); frame_ids = np.asarray([label_records[value]["frame_id"] for value in ids]); scene_ids = np.asarray([label_records[value]["sequence_id"] for value in ids])
        track_results: dict[str, Any] = {}; correct_masks: dict[str, np.ndarray] = {}; rankings_by_name = {name:value[0] for name,value in method_arrays.items()}
        for name, (ranking, probability) in method_arrays.items():
            metric = ranking_metrics(y, ranking, probabilities=probability)
            correct_masks[name] = metric["top_correct"]
            vs_q = paired_switch_metrics(y, rankings_by_name["q_only"], ranking)
            vs_v2 = paired_switch_metrics(y, rankings_by_name["v2_locked_primary"], ranking)
            record = strip_metric_arrays(metric) | {"vs_q_only":strip_metric_arrays(vs_q),"vs_v2_locked_primary":strip_metric_arrays(vs_v2)}
            track_results[name] = record
            result_rows.append({"track":track,"method":name,**{key:value for key,value in strip_metric_arrays(metric).items() if key!="reliability"},"delta_vs_q":vs_q["delta_j_at_1"],"delta_vs_v2":vs_v2["delta_j_at_1"],"recovered_vs_q":vs_q["recovered"],"harmful_vs_q":vs_q["harmful"],"recovered_vs_v2":vs_v2["recovered"],"harmful_vs_v2":vs_v2["harmful"],"neutral_switch_vs_v2":vs_v2["neutral_switch"],"switch_coverage_vs_v2":vs_v2["switch_coverage"],"outcome_changing_precision_vs_v2":vs_v2["outcome_changing_precision"],"headroom_recovered_vs_v2":vs_v2["headroom_recovered"]})
            if track == "corrected_scientific":
                calibration_payload["reliability"][name] = metric["reliability"]
                confidence = probability[np.arange(len(ids)), ranking[:,0]]
                calibration_payload["risk_coverage"][name] = risk_coverage_curve(metric["top_correct"], confidence)
        raw_p: dict[str,float] = {}; pending=[]
        for name in methods:
            if name == "q_only": continue
            for reference in ("q_only","v2_locked_primary"):
                if name == reference: continue
                key=f"{track}:{name}:vs:{reference}"; stats=paired_cluster_statistics(correct_masks[reference],correct_masks[name],frame_ids=frame_ids,scene_ids=scene_ids,iterations=iterations,seed=seed)
                raw_p[key]=stats["mcnemar"]["raw_p"]; pending.append((key,name,reference,stats))
        adjusted=holm_adjust(raw_p)
        for key,name,reference,stats in pending:
            record={"track":track,"method":name,"reference":reference,"effect":stats["mcnemar"]["effect"],"recovered":stats["mcnemar"]["recovered"],"harmful":stats["mcnemar"]["harmful"],"raw_p":stats["mcnemar"]["raw_p"],"holm_adjusted_p":adjusted[key],"frame_ci_lower":stats["frame_bootstrap"]["ci95"][0],"frame_ci_upper":stats["frame_bootstrap"]["ci95"][1],"scene_ci_lower":stats["scene_bootstrap"]["ci95"][0],"scene_ci_upper":stats["scene_bootstrap"]["ci95"][1],"sample_count":len(ids),"frame_group_count":stats["frame_bootstrap"]["group_count"],"scene_group_count":stats["scene_bootstrap"]["group_count"],"bootstrap_iterations":iterations}
            statistical_records.append(record)
        results[track] = track_results
        oracle_values={value["oracle_correct"] for value in track_results.values()}
        if len(oracle_values)!=1: raise AssertionError("reranking changed Oracle@5")
    write_csv(output/"results.csv",result_rows); write_csv(output/"pairwise_statistics.csv",statistical_records)
    atomic_write_json(output/"calibration_curves.json",calibration_payload)
    summary={"schema_version":"3.0.0","kind":"v3_dual_track_evaluation","status":"complete","sample_count":len(ids),"methods":list(methods),"tracks":results,"oracle_unchanged":True,"statistics":statistical_records,"bootstrap_iterations":int(iterations),"seed":int(seed),"labels_read":[str(Path(corrected_labels_path).resolve()),str(Path(legacy_labels_path).resolve())],"inputs":{"features":artifact_identity(features_path),"predictions":{name:None if path is None else artifact_identity(path) for name,path in methods.items()}}}
    atomic_write_json(output/"summary.json",summary); return summary
