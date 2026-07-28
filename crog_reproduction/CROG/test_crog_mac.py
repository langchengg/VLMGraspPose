import argparse
import os

import cv2
import torch
from loguru import logger
from torch.utils.data import DataLoader, Subset

import utils.config as config
from engine.crog_engine import inference_with_grasp
from model import build_crog
from utils.checkpoint import load_checkpoint
from utils.dataset import OCIDVLGDataset
from utils.device import get_device
from utils.misc import setup_logger


cv2.setNumThreads(0)


def parse_args():
    parser = argparse.ArgumentParser(description="Single-device CROG evaluation")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", choices=("train", "val", "test"), default="val")
    cli = parser.parse_args()
    args = config.load_cfg_from_cfg_file(cli.config)
    args.checkpoint = cli.checkpoint
    args.eval_split = cli.split
    return args


def resolve_device(requested):
    if requested in (None, "auto"):
        return get_device(prefer_mps=True)
    return torch.device(requested)


@logger.catch(reraise=True)
def main():
    args = parse_args()
    args.device = resolve_device(args.device)
    args.output_dir = os.path.join(args.output_folder, args.exp_name)
    os.makedirs(args.output_dir, exist_ok=True)
    setup_logger(args.output_dir, distributed_rank=0, filename="test_mac.log", mode="a")

    dataset = OCIDVLGDataset(
        root_dir=args.root_path,
        input_size=args.input_size,
        word_length=args.word_len,
        split=args.eval_split,
        version=args.version,
    )
    if args.eval_split == "val" and args.max_val_samples is not None:
        dataset = Subset(dataset, range(min(args.max_val_samples, len(dataset))))
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size_val,
        shuffle=False,
        num_workers=args.workers_val,
        pin_memory=args.device.type == "cuda",
        collate_fn=OCIDVLGDataset.collate_fn,
    )

    model, _ = build_crog(args)
    model = model.to(args.device)
    load_checkpoint(args.checkpoint, model, args.device)
    logger.info("Evaluating {} on {}", args.checkpoint, args.device)
    inference_with_grasp(loader, model, args)


if __name__ == "__main__":
    main()
