#!/usr/bin/env python3
import argparse
import json
import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import utils.config as config
from model import build_crog
from utils.checkpoint import load_checkpoint
from utils.dataset import OCIDVLGDataset
from utils.device import get_device, move_to_device
from utils.grasp_eval import calculate_jacquard_index, detect_grasps
from utils.grasp_metrics import (
    CORRECTED_EVALUATOR_VERSION,
    binary_mask_iou,
    load_raw_binary_target_mask,
)

from failure_utils import (
    GRASP_IOU_THRESHOLD,
    MASK_THRESHOLD,
    bbox_from_mask,
    ensure_dir,
    mask_to_rle,
    nearest_gt_errors,
    point_inside_mask,
    to_jsonable,
)


DEFAULT_CONFIG = "config/OCID-VLG/CROG_mac_mps_official_params_50epoch_bs8.yaml"
DEFAULT_CHECKPOINT = (
    "exp/OCID-VLG_multiple_mac/"
    "CROG_mac_mps_official_params_50epoch_bs8/best_jindex_model.pth"
)
DEFAULT_OUTPUT = "failure_analysis/predictions/test_predictions.jsonl"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export per-sample CROG predictions for failure analysis."
    )
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--device", help="Override config device, for example cpu or mps.")
    parser.add_argument("--include-mask-rle", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def resolve_device(requested):
    if requested in (None, "auto"):
        return get_device(prefer_mps=True)
    return torch.device(requested)


def restore_original(value, inverse, width, height):
    return cv2.warpAffine(value, inverse, (width, height), flags=cv2.INTER_CUBIC)


def safe_jaccard(preds, targets):
    if not preds or targets is None or len(targets) == 0:
        return 0
    return int(calculate_jacquard_index(list(preds), np.asarray(targets, dtype=np.float32).copy(), GRASP_IOU_THRESHOLD))


def confidence_at_grasp(quality_mask, grasps):
    if not grasps:
        return math.nan
    x, y = int(round(float(grasps[0][0]))), int(round(float(grasps[0][1])))
    if y < 0 or y >= quality_mask.shape[0] or x < 0 or x >= quality_mask.shape[1]:
        return math.nan
    return float(quality_mask[y, x])


def scene_instance_count(mask_path):
    mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
    if mask is None:
        return None
    values = np.unique(mask)
    return int(np.sum(values > 0))


@torch.no_grad()
def main():
    cli = parse_args()
    cfg = config.load_cfg_from_cfg_file(cli.config)
    device = resolve_device(cli.device or cfg.device)
    root = (REPO_ROOT / cfg.root_path).resolve()
    checkpoint = (REPO_ROOT / cli.checkpoint).resolve()
    output_path = (REPO_ROOT / cli.output).resolve()

    dataset = OCIDVLGDataset(
        root_dir=str(root),
        input_size=cfg.input_size,
        word_length=cfg.word_len,
        split=cli.split,
        version=cfg.version,
    )
    base_dataset = dataset
    if cli.max_samples is not None:
        dataset = Subset(dataset, range(min(cli.max_samples, len(dataset))))

    loader = DataLoader(
        dataset,
        batch_size=cli.batch_size or cfg.batch_size_val,
        shuffle=False,
        num_workers=cfg.workers_val if cli.workers is None else cli.workers,
        pin_memory=device.type == "cuda",
        collate_fn=OCIDVLGDataset.collate_fn,
    )

    model, _ = build_crog(cfg)
    model = model.to(device).eval()
    load_checkpoint(str(checkpoint), model, device)

    ensure_dir(output_path.parent)
    meta_path = output_path.with_suffix(".meta.json")
    metadata = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "repo_root": str(REPO_ROOT),
        "config": str((REPO_ROOT / cli.config).resolve()),
        "checkpoint": str(checkpoint),
        "split": cli.split,
        "version": cfg.version,
        "dataset_root": str(root),
        "dataset_samples": len(base_dataset),
        "exported_samples_limit": cli.max_samples,
        "batch_size": cli.batch_size or cfg.batch_size_val,
        "device": str(device),
        "mask_threshold": MASK_THRESHOLD,
        "grasp_iou_threshold": GRASP_IOU_THRESHOLD,
        "stores_predicted_mask_rle": bool(cli.include_mask_rle),
    }
    meta_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    with open(output_path, "w", encoding="utf-8") as handle:
        for data in tqdm(loader, desc="Exporting predictions", ncols=100):
            image = data["img"]
            text = data["word_vec"]
            ins_mask = data["mask"]
            grasp_qua_mask = data["grasp_masks"]["qua"]
            grasp_sin_mask = data["grasp_masks"]["sin"]
            grasp_cos_mask = data["grasp_masks"]["cos"]
            grasp_wid_mask = data["grasp_masks"]["wid"]

            image, text, ins_mask, grasp_qua_mask, grasp_sin_mask, grasp_cos_mask, grasp_wid_mask = move_to_device(
                (image, text, ins_mask, grasp_qua_mask, grasp_sin_mask, grasp_cos_mask, grasp_wid_mask),
                device,
            )
            pred, target = model(
                image,
                text,
                ins_mask.unsqueeze(1),
                grasp_qua_mask.unsqueeze(1),
                grasp_sin_mask.unsqueeze(1),
                grasp_cos_mask.unsqueeze(1),
                grasp_wid_mask.unsqueeze(1),
            )

            ins_pred, qua_pred, sin_pred, cos_pred, wid_pred = pred

            ins_pred = torch.sigmoid(ins_pred)
            qua_pred = torch.sigmoid(qua_pred)
            wid_pred = torch.sigmoid(wid_pred)
            if ins_pred.shape[-2:] != image.shape[-2:]:
                resize_kwargs = {
                    "size": image.shape[-2:],
                    "mode": "bicubic",
                    "align_corners": True,
                }
                ins_pred = F.interpolate(ins_pred, **resize_kwargs).squeeze(1)
                qua_pred = F.interpolate(qua_pred, **resize_kwargs).squeeze(1)
                sin_pred = F.interpolate(sin_pred, **resize_kwargs).squeeze(1)
                cos_pred = F.interpolate(cos_pred, **resize_kwargs).squeeze(1)
                wid_pred = F.interpolate(wid_pred, **resize_kwargs).squeeze(1)

            for idx in range(ins_pred.shape[0]):
                sent_id = int(data["sent_id"][idx])
                sample_index = int(base_dataset.get_index_from_sent(sent_id))
                inverse = data["inverse"][idx]
                height, width = [int(v) for v in data["ori_size"][idx]]
                gt_grasps = np.asarray(data["grasps"][idx], dtype=np.float32)

                pred_mask_score = restore_original(ins_pred[idx].cpu().numpy(), inverse, width, height)
                pred_mask = pred_mask_score > MASK_THRESHOLD
                pred_qua = restore_original(qua_pred[idx].cpu().numpy(), inverse, width, height)
                pred_sin = restore_original(sin_pred[idx].cpu().numpy(), inverse, width, height)
                pred_cos = restore_original(cos_pred[idx].cpu().numpy(), inverse, width, height)
                pred_wid = restore_original(wid_pred[idx].cpu().numpy(), inverse, width, height)

                mask_path = root / base_dataset.mask_paths[sample_index]
                object_id = int(base_dataset.objIDs[sample_index])
                target_mask = load_raw_binary_target_mask(mask_path, object_id)
                mask_iou = binary_mask_iou(pred_mask, target_mask)

                top1, _ = detect_grasps(pred_qua, pred_sin, pred_cos, pred_wid, 1)
                top5, _ = detect_grasps(pred_qua, pred_sin, pred_cos, pred_wid, 5)
                center_error, angle_error, width_error = nearest_gt_errors(top1, gt_grasps)
                predicted_center = top1[0][:2] if top1 else None

                depth_path = root / base_dataset.depth_paths[sample_index]
                row = {
                    "sample_id": sent_id,
                    "evaluator_version": CORRECTED_EVALUATOR_VERSION,
                    "sample_index": sample_index,
                    "split": cli.split,
                    "version": cfg.version,
                    "image_path": str(Path(data["img_path"][idx]).resolve()),
                    "depth_path": str(depth_path.resolve()),
                    "mask_path": str(mask_path.resolve()),
                    "scene_id": data["scene_id"][idx],
                    "language_instruction": data["sentence"][idx],
                    "target_name": data["target"][idx],
                    "target_idx": int(data["target_idx"][idx]),
                    "obj_id": object_id,
                    "bbox_xyxy": [int(v) for v in data["bbox"][idx]],
                    "bbox_area": int((data["bbox"][idx][2] - data["bbox"][idx][0]) * (data["bbox"][idx][3] - data["bbox"][idx][1])),
                    "scene_instance_count": scene_instance_count(mask_path),
                    "gt_grasps": gt_grasps.tolist(),
                    "gt_grasp_count": int(len(gt_grasps)),
                    "gt_mask_area": int(np.sum(target_mask)),
                    "gt_mask_bbox": bbox_from_mask(target_mask),
                    "predicted_mask_area": int(np.sum(pred_mask)),
                    "predicted_mask_bbox": bbox_from_mask(pred_mask),
                    "predicted_grasps_top1": top1,
                    "predicted_grasps_top5": top5,
                    "predicted_confidence": confidence_at_grasp(pred_qua, top1),
                    "predicted_center_in_gt_mask": point_inside_mask(predicted_center, target_mask),
                    "predicted_center_in_pred_mask": point_inside_mask(predicted_center, pred_mask),
                    "mask_iou": mask_iou,
                    "pr50_success": bool(mask_iou > 0.5),
                    "pr60_success": bool(mask_iou > 0.6),
                    "pr70_success": bool(mask_iou > 0.7),
                    "pr80_success": bool(mask_iou > 0.8),
                    "pr90_success": bool(mask_iou > 0.9),
                    "j1_success": bool(safe_jaccard(top1, gt_grasps)),
                    "jany_success": bool(safe_jaccard(top5, gt_grasps)),
                    "grasp_center_error": center_error,
                    "grasp_angle_error": angle_error,
                    "grasp_width_error": width_error,
                }
                if cli.include_mask_rle:
                    row["predicted_mask_rle"] = mask_to_rle(pred_mask)
                handle.write(json.dumps(to_jsonable(row), sort_keys=True) + "\n")


if __name__ == "__main__":
    from failure_analysis.reranking.exporter import main as reranking_main

    reranking_main()
