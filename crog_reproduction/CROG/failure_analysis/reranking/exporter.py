import argparse
import json
import math
import platform
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import skimage
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

import utils.config as config
from failure_analysis.failure_utils import (
    GRASP_IOU_THRESHOLD,
    MASK_THRESHOLD,
    bbox_from_mask,
    mask_to_rle,
    nearest_gt_errors,
    point_inside_mask,
    to_jsonable,
)
from model import build_crog
from utils.checkpoint import load_checkpoint
from utils.dataset import OCIDVLGDataset
from utils.device import get_device, move_to_device
from utils.grasp_eval import (
    calculate_jacquard_index,
    detect_grasp_candidates,
    detect_grasps,
)
from utils.grasp_metrics import (
    CORRECTED_EVALUATOR_VERSION,
    binary_mask_iou,
    load_raw_binary_target_mask,
)

from .feature_extraction import (
    DEFAULT_FEATURE_CONFIG,
    DEFAULT_GRIPPER_CONFIG,
    extract_features_for_candidates,
    pcd_path_from_image,
)
from .labels import build_label_record, regression_mismatches
from .schema import (
    COMMIT_FILENAME,
    FEATURES_FILENAME,
    LABELS_FILENAME,
    METADATA_FILENAME,
    PREDICTIONS_FILENAME,
    SCHEMA_VERSION,
    append_mode,
    assert_no_forbidden_feature_keys,
    load_metadata,
    make_metadata,
    prepare_output_dir,
    read_jsonl,
    recover_committed_jsonl_prefix,
    sha256_file,
    validate_resume_metadata,
    write_jsonl_record,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = "config/OCID-VLG/CROG_mac_mps_official_params_50epoch_bs8.yaml"
DEFAULT_CHECKPOINT = (
    "exp/OCID-VLG_multiple_mac/"
    "CROG_mac_mps_official_params_50epoch_bs8/best_jindex_model.pth"
)
DEFAULT_TEST_REGRESSION_REFERENCE = "failure_analysis/predictions/test_predictions.jsonl"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Export frozen CROG Top-K inference features, physically separate labels, "
            "and a backward-compatible combined prediction view."
        )
    )
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument(
        "--output",
        help="New output directory. Default: failure_analysis/reranking_outputs/<split>_<timestamp>.",
    )
    parser.add_argument("--limit", "--max-samples", dest="limit", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", choices=("cpu", "mps"), default="mps")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--save-dense-maps", action="store_true")
    parser.add_argument("--include-mask-rle", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--feature-config", help="Optional JSON override for feature extraction.")
    parser.add_argument("--gripper-config", help="Optional JSON virtual-gripper configuration.")
    parser.add_argument("--calibration", help="Train-only width/depth calibration JSON.")
    parser.add_argument(
        "--regression-reference",
        help=(
            "Immutable independent prediction JSONL. For test, defaults to "
            "failure_analysis/predictions/test_predictions.jsonl when present."
        ),
    )
    parser.add_argument(
        "--allow-regression-mismatch",
        action="store_true",
        help="Keep outputs but do not fail if independent old/new J@1 or J@Any differs.",
    )
    return parser.parse_args(argv)


def _resolve_path(path):
    path = Path(path)
    return path if path.is_absolute() else (REPO_ROOT / path).resolve()


def _load_json(path, default):
    if path is None:
        return dict(default)
    return {**default, **json.loads(_resolve_path(path).read_text(encoding="utf-8"))}


def _load_training_calibration(path):
    if path is None:
        return None, None
    resolved = _resolve_path(path)
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    provenance = payload.get("provenance")
    calibration = payload.get("calibration")
    if not isinstance(provenance, dict) or provenance.get("source_split") != "train":
        raise ValueError(
            "--calibration must be a provenance-bearing JSON object with "
            "provenance.source_split='train'"
        )
    if not isinstance(calibration, dict):
        raise ValueError("--calibration JSON must contain a calibration object")
    provenance = {
        **provenance,
        "path": str(resolved),
        "file_sha256": sha256_file(resolved),
    }
    return calibration, provenance


def _load_regression_reference(path, split):
    if path is None and split == "test":
        default = _resolve_path(DEFAULT_TEST_REGRESSION_REFERENCE)
        path = default if default.exists() else None
    if path is None:
        return None, None
    resolved = _resolve_path(path)
    records = {}
    for record in read_jsonl(resolved):
        sample_id = str(record["sample_id"])
        if sample_id in records:
            raise ValueError(f"duplicate regression-reference sample_id: {sample_id}")
        records[sample_id] = {
            "sample_id": record["sample_id"],
            "j1_success": bool(record.get("j1_success", False)),
            "jany_success": bool(record.get("jany_success", False)),
        }
    identity = {
        "path": str(resolved),
        "file_sha256": sha256_file(resolved),
        "record_count": len(records),
    }
    return records, identity


def _resolve_device(requested):
    if requested == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("--device mps requested but MPS is unavailable")
    return torch.device(requested) if requested else get_device(prefer_mps=True)


def _restore_original(value, inverse, width, height):
    return cv2.warpAffine(value, inverse, (width, height), flags=cv2.INTER_CUBIC)


def _safe_jaccard(preds, targets):
    if not preds or targets is None or len(targets) == 0:
        return 0
    copied = np.asarray(targets, dtype=np.float32).copy()
    return int(calculate_jacquard_index(list(preds), copied, GRASP_IOU_THRESHOLD))


def _confidence_at_grasp(quality_mask, grasps):
    if not grasps:
        return math.nan
    x, y = int(round(float(grasps[0][0]))), int(round(float(grasps[0][1])))
    if y < 0 or y >= quality_mask.shape[0] or x < 0 or x >= quality_mask.shape[1]:
        return math.nan
    return float(quality_mask[y, x])


def _scene_instance_count(mask_path):
    mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
    if mask is None:
        return None
    return int(np.sum(np.unique(mask) > 0))


def _runtime_metadata(device):
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "opencv": cv2.__version__,
        "scikit_image": skimage.__version__,
        "device": str(device),
        "mps_available": bool(torch.backends.mps.is_available()),
    }


def _legacy_row(
    *,
    sent_id,
    sample_index,
    split,
    version,
    data,
    index,
    root,
    base_dataset,
    gt_grasps,
    target_mask,
    pred_mask,
    pred_qua,
    top1,
    top5,
    candidates,
    include_mask_rle,
):
    mask_iou = binary_mask_iou(pred_mask, target_mask)
    center_error, angle_error, width_error = nearest_gt_errors(top1, gt_grasps)
    predicted_center = top1[0][:2] if top1 else None
    mask_path = root / base_dataset.mask_paths[sample_index]
    depth_path = root / base_dataset.depth_paths[sample_index]
    row = {
        "sample_id": sent_id,
        "sample_index": sample_index,
        "split": split,
        "version": version,
        "image_path": str(Path(data["img_path"][index]).resolve()),
        "depth_path": str(depth_path.resolve()),
        "mask_path": str(mask_path.resolve()),
        "scene_id": data["scene_id"][index],
        "language_instruction": data["sentence"][index],
        "target_name": data["target"][index],
        "target_idx": int(data["target_idx"][index]),
        "obj_id": int(base_dataset.objIDs[sample_index]),
        "bbox_xyxy": [int(value) for value in data["bbox"][index]],
        "bbox_area": int(
            (data["bbox"][index][2] - data["bbox"][index][0])
            * (data["bbox"][index][3] - data["bbox"][index][1])
        ),
        "scene_instance_count": _scene_instance_count(mask_path),
        "gt_grasps": gt_grasps.tolist(),
        "gt_grasp_count": int(len(gt_grasps)),
        "gt_mask_area": int(np.sum(target_mask)),
        "gt_mask_bbox": bbox_from_mask(target_mask),
        "predicted_mask_area": int(np.sum(pred_mask)),
        "predicted_mask_bbox": bbox_from_mask(pred_mask),
        "predicted_grasps_top1": top1,
        "predicted_grasps_top5": top5,
        "predicted_confidence": _confidence_at_grasp(pred_qua, top1),
        "predicted_center_in_gt_mask": point_inside_mask(predicted_center, target_mask),
        "predicted_center_in_pred_mask": point_inside_mask(predicted_center, pred_mask),
        "mask_iou": mask_iou,
        "pr50_success": bool(mask_iou > 0.5),
        "pr60_success": bool(mask_iou > 0.6),
        "pr70_success": bool(mask_iou > 0.7),
        "pr80_success": bool(mask_iou > 0.8),
        "pr90_success": bool(mask_iou > 0.9),
        "j1_success": bool(_safe_jaccard(top1, gt_grasps)),
        "jany_success": bool(_safe_jaccard(top5, gt_grasps)),
        "grasp_center_error": center_error,
        "grasp_angle_error": angle_error,
        "grasp_width_error": width_error,
        "schema_version": SCHEMA_VERSION,
        "evaluator_version": CORRECTED_EVALUATOR_VERSION,
        "candidates": candidates,
    }
    if include_mask_rle:
        row["predicted_mask_rle"] = mask_to_rle(pred_mask)
    return row


@torch.no_grad()
def export_predictions(cli):
    if cli.resume and cli.overwrite:
        raise ValueError("--resume and --overwrite are mutually exclusive")
    cfg = config.load_cfg_from_cfg_file(str(_resolve_path(cli.config)))
    device = _resolve_device(cli.device or cfg.device)
    root = (REPO_ROOT / cfg.root_path).resolve()
    checkpoint = _resolve_path(cli.checkpoint)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = _resolve_path(
        cli.output or f"failure_analysis/reranking_outputs/{cli.split}_{timestamp}"
    )
    feature_config = _load_json(cli.feature_config, DEFAULT_FEATURE_CONFIG)
    gripper_config = _load_json(cli.gripper_config, DEFAULT_GRIPPER_CONFIG)
    calibration, calibration_provenance = _load_training_calibration(cli.calibration)
    regression_reference, regression_reference_identity = _load_regression_reference(
        cli.regression_reference, cli.split
    )
    generation = feature_config.get("candidate_generation", {})
    frozen_generation = {"k": 5, "peak_threshold": 0.4, "min_distance": 2}
    if generation != frozen_generation:
        raise ValueError(
            "candidate generation is frozen and must remain exactly "
            f"{frozen_generation}; received {generation}"
        )
    if float(feature_config.get("mask_threshold", MASK_THRESHOLD)) != MASK_THRESHOLD:
        raise ValueError(f"predicted mask threshold is frozen at {MASK_THRESHOLD}")
    checkpoint_hash = sha256_file(checkpoint)

    dataset = OCIDVLGDataset(
        root_dir=str(root),
        input_size=cfg.input_size,
        word_length=cfg.word_len,
        split=cli.split,
        version=cfg.version,
    )
    base_dataset = dataset
    selected_indices = list(range(min(cli.limit, len(dataset)))) if cli.limit is not None else list(range(len(dataset)))
    effective_batch_size = int(cli.batch_size or cfg.batch_size_val)
    metadata = make_metadata(
        repo_root=REPO_ROOT,
        checkpoint_path=checkpoint,
        checkpoint_sha256=checkpoint_hash,
        dataset_root=root,
        split=cli.split,
        sample_count=len(selected_indices),
        feature_config=feature_config,
        gripper_config=gripper_config,
        runtime=_runtime_metadata(device),
        config_sha256=sha256_file(_resolve_path(cli.config)),
        output_config={
            "include_mask_rle": bool(cli.include_mask_rle),
            "save_dense_maps": bool(cli.save_dense_maps),
            "regression_reference": regression_reference_identity,
            "batch_size": effective_batch_size,
            "workers": int(cli.workers),
        },
        calibration=calibration,
        calibration_provenance=calibration_provenance,
    )
    metadata.update(
        {
            "config_path": str(_resolve_path(cli.config)),
            "dataset_version": cfg.version,
            "mask_representation": "sigmoid probability; binary uses strict probability > 0.35",
            "coordinate_system": "x=column, y=row; peak coordinates are (row,column)",
            "postprocessing_provenance": {
                "quality_sigmoid": True,
                "quality_multiplied_by_predicted_mask": False,
                "inverse_transform_before_candidate_generation": True,
                "resize_mode": "torch bicubic align_corners=True then cv2 INTER_CUBIC inverse affine",
                "angle": "0.5*atan2(sin2theta,cos2theta), radians internally, degrees in grasp",
                "width": "sigmoid width map * 100 px; fixed rectangle height 20 px",
                "peak_local_max": feature_config["candidate_generation"],
            },
            "dense_maps_saved": bool(cli.save_dense_maps),
            "stores_predicted_mask_rle": bool(cli.include_mask_rle),
        }
    )
    prepare_output_dir(output_dir, resume=cli.resume, overwrite=cli.overwrite)
    metadata_path = output_dir / METADATA_FILENAME
    if cli.resume:
        existing_metadata = load_metadata(metadata_path)
        validate_resume_metadata(existing_metadata, metadata)
        completed_order = recover_committed_jsonl_prefix(
            output_dir,
            (FEATURES_FILENAME, LABELS_FILENAME, PREDICTIONS_FILENAME),
        )
        completed = set(completed_order)
        metadata = existing_metadata
    else:
        completed = set()
        metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    selected_indices = [
        index
        for index in selected_indices
        if str(base_dataset.get_sent_from_index(index)) not in completed
    ]
    if not selected_indices:
        mismatch_ids = metadata.get("regression_mismatch_sample_ids", [])
        if mismatch_ids and not cli.allow_regression_mismatch:
            raise AssertionError(
                f"independent baseline regression mismatches: {len(mismatch_ids)}"
            )
        return {
            "output_dir": output_dir,
            "features": output_dir / FEATURES_FILENAME,
            "labels": output_dir / LABELS_FILENAME,
            "predictions": output_dir / PREDICTIONS_FILENAME,
            "metadata": metadata_path,
            "newly_written": 0,
            "total_written": len(completed),
        }
    dataset = Subset(base_dataset, selected_indices)
    loader = DataLoader(
        dataset,
        batch_size=effective_batch_size,
        shuffle=False,
        num_workers=int(cli.workers),
        pin_memory=False,
        collate_fn=OCIDVLGDataset.collate_fn,
    )

    model, _ = build_crog(cfg)
    model = model.to(device).eval()
    load_checkpoint(str(checkpoint), model, device)
    mode = append_mode(resume=cli.resume, overwrite=cli.overwrite)
    features_path = output_dir / FEATURES_FILENAME
    labels_path = output_dir / LABELS_FILENAME
    predictions_path = output_dir / PREDICTIONS_FILENAME
    commit_path = output_dir / COMMIT_FILENAME
    q_rank_mismatches = int(metadata.get("q_rank_mismatch_sample_count", 0))
    regression_mismatch_ids = list(metadata.get("regression_mismatch_sample_ids", []))
    if cli.resume:
        q_rank_mismatches = sum(
            any(
                candidate["legacy_rank"] != candidate["q_rank"]
                for candidate in record.get("candidates", [])
            )
            for record in read_jsonl(output_dir / FEATURES_FILENAME)
        )
        regression_mismatch_ids = regression_mismatches(
            read_jsonl(output_dir / LABELS_FILENAME)
        )
    newly_written = 0
    with (
        features_path.open(mode, encoding="utf-8") as feature_handle,
        labels_path.open(mode, encoding="utf-8") as label_handle,
        predictions_path.open(mode, encoding="utf-8") as prediction_handle,
        commit_path.open(mode, encoding="utf-8") as commit_handle,
    ):
        for data in tqdm(loader, desc="Exporting frozen CROG candidates", ncols=100):
            image = data["img"]
            values = move_to_device(
                (
                    image,
                    data["word_vec"],
                    data["mask"],
                    data["grasp_masks"]["qua"],
                    data["grasp_masks"]["sin"],
                    data["grasp_masks"]["cos"],
                    data["grasp_masks"]["wid"],
                ),
                device,
            )
            image, text, ins_mask, grasp_qua, grasp_sin, grasp_cos, grasp_wid = values
            pred, target = model(
                image,
                text,
                ins_mask.unsqueeze(1),
                grasp_qua.unsqueeze(1),
                grasp_sin.unsqueeze(1),
                grasp_cos.unsqueeze(1),
                grasp_wid.unsqueeze(1),
            )
            ins_pred, qua_pred, sin_pred, cos_pred, wid_pred = pred
            ins_pred = torch.sigmoid(ins_pred)
            qua_pred = torch.sigmoid(qua_pred)
            wid_pred = torch.sigmoid(wid_pred)
            if ins_pred.shape[-2:] != image.shape[-2:]:
                kwargs = {"size": image.shape[-2:], "mode": "bicubic", "align_corners": True}
                ins_pred = F.interpolate(ins_pred, **kwargs).squeeze(1)
                qua_pred = F.interpolate(qua_pred, **kwargs).squeeze(1)
                sin_pred = F.interpolate(sin_pred, **kwargs).squeeze(1)
                cos_pred = F.interpolate(cos_pred, **kwargs).squeeze(1)
                wid_pred = F.interpolate(wid_pred, **kwargs).squeeze(1)
            else:
                ins_pred = ins_pred.squeeze(1)
                qua_pred = qua_pred.squeeze(1)
                sin_pred = sin_pred.squeeze(1)
                cos_pred = cos_pred.squeeze(1)
                wid_pred = wid_pred.squeeze(1)
            for index in range(ins_pred.shape[0]):
                sent_id = int(data["sent_id"][index])
                sample_index = int(base_dataset.get_index_from_sent(sent_id))
                inverse = data["inverse"][index]
                height, width = [int(value) for value in data["ori_size"][index]]
                gt_grasps = np.asarray(data["grasps"][index], dtype=np.float32)
                pred_mask_probability = _restore_original(
                    ins_pred[index].cpu().numpy(), inverse, width, height
                )
                pred_mask = pred_mask_probability > float(feature_config["mask_threshold"])
                pred_qua = _restore_original(qua_pred[index].cpu().numpy(), inverse, width, height)
                pred_sin = _restore_original(sin_pred[index].cpu().numpy(), inverse, width, height)
                pred_cos = _restore_original(cos_pred[index].cpu().numpy(), inverse, width, height)
                pred_wid = _restore_original(wid_pred[index].cpu().numpy(), inverse, width, height)
                mask_path = root / base_dataset.mask_paths[sample_index]
                object_id = int(base_dataset.objIDs[sample_index])
                target_mask = load_raw_binary_target_mask(mask_path, object_id)

                candidates, _ = detect_grasp_candidates(
                    pred_qua,
                    pred_sin,
                    pred_cos,
                    pred_wid,
                    int(feature_config["candidate_generation"]["k"]),
                )
                top1, _ = detect_grasps(pred_qua, pred_sin, pred_cos, pred_wid, 1)
                top5 = [candidate["legacy_grasp"] for candidate in candidates]
                if bool(top1) != bool(top5) or (top1 and top1[0] != top5[0]):
                    raise AssertionError(f"legacy top-1 is not the prefix of frozen Top-K: {sent_id}")
                q_rank_mismatches += int(
                    any(candidate["legacy_rank"] != candidate["q_rank"] for candidate in candidates)
                )
                depth_m = data["depth"][index].cpu().numpy().astype(np.float32)
                candidates = extract_features_for_candidates(
                    candidates,
                    mask_probability=pred_mask_probability,
                    quality=pred_qua,
                    sin_map=pred_sin,
                    cos_map=pred_cos,
                    depth_m=depth_m,
                    feature_config=feature_config,
                    calibration=calibration,
                )
                image_path = Path(data["img_path"][index]).resolve()
                depth_path = (root / base_dataset.depth_paths[sample_index]).resolve()
                pcd_path = pcd_path_from_image(image_path)
                feature_record = {
                    "schema_version": SCHEMA_VERSION,
                    "sample_id": sent_id,
                    "sample_index": sample_index,
                    "split": cli.split,
                    "scene_id": data["scene_id"][index],
                    "image_path": str(image_path),
                    "depth_path": str(depth_path),
                    "pcd_path": str(pcd_path) if pcd_path else None,
                    "pcd_available": bool(pcd_path and pcd_path.exists()),
                    "language_instruction": data["sentence"][index],
                    "predicted_mask_area": int(pred_mask.sum()),
                    "predicted_mask_bbox": bbox_from_mask(pred_mask),
                    "candidates": candidates,
                    "provenance": metadata["postprocessing_provenance"],
                }
                if cli.include_mask_rle:
                    feature_record["predicted_mask_rle"] = mask_to_rle(pred_mask)
                assert_no_forbidden_feature_keys(feature_record)
                old_record = _legacy_row(
                    sent_id=sent_id,
                    sample_index=sample_index,
                    split=cli.split,
                    version=cfg.version,
                    data=data,
                    index=index,
                    root=root,
                    base_dataset=base_dataset,
                    gt_grasps=gt_grasps,
                    target_mask=target_mask,
                    pred_mask=pred_mask,
                    pred_qua=pred_qua,
                    top1=top1,
                    top5=top5,
                    candidates=candidates,
                    include_mask_rle=cli.include_mask_rle,
                )
                regression_old_record = None
                if regression_reference is not None:
                    regression_old_record = regression_reference.get(str(sent_id))
                    if regression_old_record is None:
                        raise ValueError(
                            f"independent regression reference missing sample_id {sent_id}"
                        )
                label_record = build_label_record(
                    feature_record,
                    gt_grasps,
                    old_record=regression_old_record,
                    iou_threshold=GRASP_IOU_THRESHOLD,
                )
                reference = label_record.get("regression_reference")
                if reference and (
                    reference["old_j1_success"]
                    != reference["recomputed_original_top1_success"]
                    or reference["old_jany_success"]
                    != reference["recomputed_oracle_success"]
                ):
                    regression_mismatch_ids.append(sent_id)
                write_jsonl_record(feature_handle, to_jsonable(feature_record))
                write_jsonl_record(label_handle, to_jsonable(label_record))
                write_jsonl_record(prediction_handle, to_jsonable(old_record))
                if cli.save_dense_maps:
                    dense_dir = output_dir / "dense_maps"
                    dense_dir.mkdir(parents=True, exist_ok=True)
                    np.savez_compressed(
                        dense_dir / f"sample_{sent_id}.npz",
                        mask_probability=pred_mask_probability.astype(np.float16),
                        quality=pred_qua.astype(np.float16),
                        sin2theta=pred_sin.astype(np.float16),
                        cos2theta=pred_cos.astype(np.float16),
                        width=pred_wid.astype(np.float16),
                    )
                write_jsonl_record(commit_handle, {"sample_id": sent_id})
                newly_written += 1

    total_written = len(completed) + newly_written
    metadata["sample_count"] = total_written
    metadata["q_rank_mismatch_sample_count"] = int(q_rank_mismatches)
    metadata["regression_reference"] = regression_reference_identity
    metadata["regression_mismatch_sample_ids"] = regression_mismatch_ids
    metadata["regression_mismatch_sample_count"] = len(regression_mismatch_ids)
    metadata["regression_verified"] = bool(
        regression_reference_identity is not None and not regression_mismatch_ids
    )
    metadata["features_path"] = str(features_path)
    metadata["labels_path"] = str(labels_path)
    metadata["predictions_path"] = str(predictions_path)
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if regression_mismatch_ids and not cli.allow_regression_mismatch:
        raise AssertionError(
            "independent baseline regression mismatches: "
            f"{len(regression_mismatch_ids)}; see metadata.json"
        )
    return {
        "output_dir": output_dir,
        "features": features_path,
        "labels": labels_path,
        "predictions": predictions_path,
        "metadata": metadata_path,
        "newly_written": newly_written,
        "total_written": total_written,
    }


def main(argv=None):
    result = export_predictions(parse_args(argv))
    print(json.dumps({key: str(value) for key, value in result.items()}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
