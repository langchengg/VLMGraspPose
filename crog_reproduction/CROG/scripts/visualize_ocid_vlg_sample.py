#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

import utils.config as config
from model import build_crog
from utils.checkpoint import load_checkpoint
from utils.dataset import OCIDVLGDataset
from utils.device import get_device, move_to_device
from utils.grasp_eval import detect_grasps


def draw_grasps(image, grasps, color=(255, 0, 0)):
    drawn = image.copy()
    for rect in grasps:
        center_x, center_y, width, height, theta = rect[:5]
        box = cv2.boxPoints(((center_x, center_y), (width, height), -(theta + 180)))
        box = np.asarray(box, dtype=np.intp)
        cv2.polylines(drawn, [box], True, color, 2)
    return drawn


def overlay_mask(image, mask, color=(255, 0, 0)):
    result = image.copy().astype(np.float32)
    mask = mask.astype(bool)
    result[mask] = result[mask] * 0.45 + np.asarray(color) * 0.55
    return result.astype(np.uint8)


@torch.no_grad()
def predict_sample(model, sample, device):
    batch = OCIDVLGDataset.collate_fn([sample])
    image, text, mask, qua, sin, cos, wid = move_to_device(
        (
            batch["img"], batch["word_vec"], batch["mask"],
            batch["grasp_masks"]["qua"], batch["grasp_masks"]["sin"],
            batch["grasp_masks"]["cos"], batch["grasp_masks"]["wid"],
        ),
        device,
    )
    pred, _ = model(
        image, text, mask.unsqueeze(1), qua.unsqueeze(1), sin.unsqueeze(1),
        cos.unsqueeze(1), wid.unsqueeze(1),
    )
    ins, qua_pred, sin_pred, cos_pred, wid_pred = pred
    input_size = image.shape[-2:]
    tensors = [ins, qua_pred, sin_pred, cos_pred, wid_pred]
    tensors = [
        F.interpolate(value, input_size, mode="bicubic", align_corners=True)
        if value.shape[-2:] != input_size else value
        for value in tensors
    ]
    ins, qua_pred, sin_pred, cos_pred, wid_pred = tensors
    ins = torch.sigmoid(ins)[0, 0].cpu().numpy()
    qua_pred = torch.sigmoid(qua_pred)[0, 0].cpu().numpy()
    sin_pred = sin_pred[0, 0].cpu().numpy()
    cos_pred = cos_pred[0, 0].cpu().numpy()
    wid_pred = torch.sigmoid(wid_pred)[0, 0].cpu().numpy()

    height, width = sample["ori_size"]
    inverse = sample["inverse"]
    restore = lambda value: cv2.warpAffine(value, inverse, (width, height), flags=cv2.INTER_CUBIC)
    ins = restore(ins) > 0.35
    qua_pred, sin_pred, cos_pred, wid_pred = map(
        restore, (qua_pred, sin_pred, cos_pred, wid_pred)
    )
    grasps, _ = detect_grasps(qua_pred, sin_pred, cos_pred, wid_pred, 1)
    return ins, grasps


def main():
    parser = argparse.ArgumentParser(description="Visualize OCID-VLG samples")
    parser.add_argument("--config", required=True)
    parser.add_argument("--num_samples", type=int, default=10)
    parser.add_argument("--split", choices=("train", "val", "test"), default="val")
    parser.add_argument("--checkpoint")
    parser.add_argument("--output_dir", default="outputs/mac_debug_visualizations")
    args = parser.parse_args()
    cfg = config.load_cfg_from_cfg_file(args.config)
    root = Path(cfg.root_path).expanduser().resolve()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = OCIDVLGDataset(
        root_dir=str(root), input_size=cfg.input_size, word_length=cfg.word_len,
        split=args.split, version=cfg.version,
    )
    model = None
    device = None
    if args.checkpoint:
        device = get_device(prefer_mps=True) if cfg.device == "auto" else torch.device(cfg.device)
        model, _ = build_crog(cfg)
        model = model.to(device).eval()
        load_checkpoint(args.checkpoint, model, device)

    for index in range(min(args.num_samples, len(dataset))):
        sample = dataset[index]
        rgb = cv2.cvtColor(cv2.imread(sample["img_path"]), cv2.COLOR_BGR2RGB)
        mask_path = root / dataset.mask_paths[index]
        full_mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
        gt_mask = full_mask == sample["objID"]
        gt_grasps = sample["grasp_rects"]
        depth = sample.get("depth")

        columns = 6 if model is not None else 4
        fig, axes = plt.subplots(1, columns, figsize=(5 * columns, 5))
        axes[0].imshow(rgb)
        axes[0].set_title("RGB")
        axes[1].imshow(overlay_mask(rgb, gt_mask))
        axes[1].set_title("Ground-truth mask")
        axes[2].imshow(draw_grasps(rgb, gt_grasps))
        axes[2].set_title("Ground-truth grasps")
        axes[3].imshow(depth, cmap="gray")
        axes[3].set_title("Depth")

        if model is not None:
            pred_mask, pred_grasps = predict_sample(model, sample, device)
            axes[4].imshow(overlay_mask(rgb, pred_mask, color=(0, 255, 0)))
            axes[4].set_title("Predicted mask")
            axes[5].imshow(draw_grasps(rgb, pred_grasps, color=(0, 255, 0)))
            axes[5].set_title("Predicted grasp")

        for axis in axes:
            axis.axis("off")
        fig.suptitle(sample["sentence"], fontsize=16)
        fig.tight_layout()
        save_path = output_dir / f"{args.split}_sample_{index}.png"
        fig.savefig(save_path, dpi=140)
        plt.close(fig)
        print(f"saved: {save_path}")


if __name__ == "__main__":
    main()
