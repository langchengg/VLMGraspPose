from utils.grasp_metrics import CORRECTED_EVALUATOR_VERSION, evaluate_candidate

from .schema import SCHEMA_VERSION


def candidate_evaluation(
    candidate,
    gt_grasps,
    iou_threshold=0.25,
    *,
    evaluator_version=CORRECTED_EVALUATOR_VERSION,
):
    return evaluate_candidate(
        candidate,
        gt_grasps,
        iou_threshold=float(iou_threshold),
        evaluator_version=evaluator_version,
    )


def candidate_is_valid(
    candidate,
    gt_grasps,
    iou_threshold=0.25,
    *,
    evaluator_version=CORRECTED_EVALUATOR_VERSION,
):
    if gt_grasps is None or len(gt_grasps) == 0:
        return False
    return bool(
        candidate_evaluation(
            candidate,
            gt_grasps,
            iou_threshold,
            evaluator_version=evaluator_version,
        )["candidate_success"]
    )


def build_label_record(
    feature_record,
    gt_grasps,
    *,
    old_record=None,
    iou_threshold=0.25,
    evaluator_version=CORRECTED_EVALUATOR_VERSION,
):
    candidate_labels = []
    for candidate in feature_record.get("candidates", []):
        evaluation = candidate_evaluation(
            candidate,
            gt_grasps,
            iou_threshold,
            evaluator_version=evaluator_version,
        )
        candidate_labels.append({
            "candidate_id": candidate["candidate_id"],
            "candidate_checksum": candidate["candidate_checksum"],
            "candidate_valid": bool(evaluation["candidate_success"]),
            "best_gt": evaluation["best_gt"],
            "failure_mode": evaluation["failure_mode"],
            "pairwise": evaluation["pairwise"],
        })
    validities = [item["candidate_valid"] for item in candidate_labels]
    record = {
        "schema_version": SCHEMA_VERSION,
        "evaluator_version": evaluator_version,
        "sample_id": feature_record["sample_id"],
        "candidate_labels": candidate_labels,
        "gt_grasp_count": int(len(gt_grasps) if gt_grasps is not None else 0),
    }
    if old_record is not None:
        record["regression_reference"] = {
            "old_j1_success": bool(old_record.get("j1_success", False)),
            "old_jany_success": bool(old_record.get("jany_success", False)),
            "recomputed_original_top1_success": bool(validities[0]) if validities else False,
            "recomputed_oracle_success": bool(any(validities)),
        }
    return record


def label_map(label_record):
    return {
        item["candidate_id"]: bool(item["candidate_valid"])
        for item in label_record.get("candidate_labels", [])
    }


def validate_label_candidate_join(feature_record, label_record):
    feature = {
        item["candidate_id"]: item["candidate_checksum"]
        for item in feature_record.get("candidates", [])
    }
    labels = {
        item["candidate_id"]: item["candidate_checksum"]
        for item in label_record.get("candidate_labels", [])
    }
    if feature != labels:
        raise ValueError(f"candidate label join mismatch for sample {feature_record['sample_id']}")


def regression_mismatches(label_records):
    mismatches = []
    for record in label_records:
        reference = record.get("regression_reference")
        if not reference:
            continue
        if (
            reference["old_j1_success"] != reference["recomputed_original_top1_success"]
            or reference["old_jany_success"] != reference["recomputed_oracle_success"]
        ):
            mismatches.append(record["sample_id"])
    return mismatches
