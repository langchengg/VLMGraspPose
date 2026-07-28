import argparse
import datetime
import json
import os
import time
from functools import partial

import cv2
import torch
import torch.cuda.amp as amp
import torch.utils.data as data
from loguru import logger
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import Subset

import utils.config as config
from engine.crog_engine import train_with_grasp, validate_with_grasp, validate_without_grasp
from model import build_crog
from utils.checkpoint import copy_checkpoint_atomic, load_checkpoint, save_checkpoint
from utils.dataset import OCIDVLGDataset
from utils.device import empty_cache, get_device, record_memory_sample
from utils.misc import init_random_seed, set_random_seed, setup_logger, worker_init_fn


cv2.setNumThreads(0)


def get_parser():
    parser = argparse.ArgumentParser(description="Single-device CROG training")
    parser.add_argument("--config", required=True, help="YAML config file")
    parser.add_argument("--opts", default=None, nargs=argparse.REMAINDER)
    cli_args = parser.parse_args()
    cfg = config.load_cfg_from_cfg_file(cli_args.config)
    if cli_args.opts is not None:
        cfg = config.merge_cfg_from_list(cfg, cli_args.opts)
    return cfg


def _resolve_device(requested):
    if requested in (None, "auto"):
        return get_device(prefer_mps=True)
    device = torch.device(requested)
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is not available")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device


def _limit_dataset(dataset, maximum):
    if maximum is None:
        return dataset
    maximum = min(int(maximum), len(dataset))
    return Subset(dataset, range(maximum))


def _build_dataset(args, split):
    dataset = OCIDVLGDataset(
        root_dir=args.root_path,
        input_size=args.input_size,
        word_length=args.word_len,
        split=split,
        version=args.version,
    )
    maximum = args.max_train_samples if split == "train" else args.max_val_samples
    return _limit_dataset(dataset, maximum)


def _timing_summary(train_seconds, validation_seconds, checkpoint_seconds,
                    total_seconds, train_iterations):
    return {
        "train_seconds": float(train_seconds),
        "validation_seconds": float(validation_seconds),
        "checkpoint_seconds": float(checkpoint_seconds),
        "total_seconds": float(total_seconds),
        "train_iterations": int(train_iterations),
        "average_seconds_per_iteration": (
            float(train_seconds) / train_iterations if train_iterations else 0.0
        ),
    }


def _memory_summary(samples):
    if not samples:
        return {}
    allocated = [sample["allocated_bytes"] for sample in samples]
    driver = [sample["driver_bytes"] for sample in samples]
    return {
        "sample_count": len(samples),
        "allocated_start_bytes": allocated[0],
        "allocated_peak_bytes": max(allocated),
        "allocated_end_bytes": allocated[-1],
        "driver_start_bytes": driver[0],
        "driver_peak_bytes": max(driver),
        "driver_end_bytes": driver[-1],
        "samples": samples,
    }


def _mid_epoch_checkpoint_interval(args):
    return max(0, int(getattr(args, "checkpoint_interval", 0) or 0))


def _validation_interval(args):
    return max(1, int(getattr(args, "validation_interval", 1) or 1))


def _should_validate_epoch(epoch, args):
    interval = _validation_interval(args)
    return epoch % interval == 0 or epoch == args.epochs


def _make_mid_epoch_checkpoint_callback(
    args,
    model,
    optimizer,
    scheduler,
    scaler,
    best_iou,
    best_j_index,
):
    interval = _mid_epoch_checkpoint_interval(args)
    if interval <= 0:
        return None

    next_checkpoint_step = {"value": interval}
    checkpoint_path = os.path.join(args.output_dir, "mid_epoch_model.pth")

    def checkpoint_callback(epoch, iteration, total_batches):
        if iteration >= total_batches or iteration < next_checkpoint_step["value"]:
            return
        save_checkpoint(
            checkpoint_path,
            model,
            optimizer,
            scheduler,
            scaler,
            epoch=epoch - 1,
            epoch_in_progress=epoch,
            iteration=iteration,
            total_batches=total_batches,
            mid_epoch=True,
            best_iou=best_iou,
            best_j_index=best_j_index,
        )
        logger.info(
            "Saved mid-epoch checkpoint {} at epoch {} iteration {}/{}",
            checkpoint_path,
            epoch,
            iteration,
            total_batches,
        )
        while next_checkpoint_step["value"] <= iteration:
            next_checkpoint_step["value"] += interval

    return checkpoint_callback


@logger.catch(reraise=True)
def main():
    run_started = time.perf_counter()
    args = get_parser()
    args.device = _resolve_device(args.device)
    args.output_dir = os.path.join(args.output_folder, args.exp_name)
    os.makedirs(args.output_dir, exist_ok=True)
    setup_logger(args.output_dir, distributed_rank=0, filename="train_mac.log", mode="a")

    args.manual_seed = init_random_seed(args.manual_seed, device=args.device)
    set_random_seed(args.manual_seed, deterministic=False)
    logger.info("Device: {}", args.device)
    logger.info(args)

    if not os.path.isfile(args.clip_pretrain):
        raise FileNotFoundError(
            f"CLIP checkpoint not found: {args.clip_pretrain}. See README_MAC.md."
        )

    model, param_list = build_crog(args)
    model = model.to(args.device)
    optimizer = torch.optim.Adam(
        param_list, lr=args.base_lr, weight_decay=args.weight_decay
    )
    scheduler = MultiStepLR(
        optimizer, milestones=args.milestones, gamma=args.lr_decay
    )
    scaler = amp.GradScaler() if args.device.type == "cuda" else None

    train_data = _build_dataset(args, "train")
    val_data = _build_dataset(args, "val")
    pin_memory = args.device.type == "cuda"
    init_fn = partial(
        worker_init_fn,
        num_workers=args.workers,
        rank=0,
        seed=args.manual_seed,
    )
    train_loader = data.DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=pin_memory,
        worker_init_fn=init_fn if args.workers > 0 else None,
        drop_last=False,
        collate_fn=OCIDVLGDataset.collate_fn,
    )
    val_loader = data.DataLoader(
        val_data,
        batch_size=args.batch_size_val,
        shuffle=False,
        num_workers=args.workers_val,
        pin_memory=pin_memory,
        drop_last=False,
        collate_fn=OCIDVLGDataset.collate_fn,
    )

    best_iou = 0.0
    best_j_index = 0.0
    last_iou = None
    last_prec = {}
    last_j_index = [0.0, 0.0]
    if args.resume:
        if not os.path.isfile(args.resume):
            raise FileNotFoundError(f"Resume checkpoint not found: {args.resume}")
        checkpoint = load_checkpoint(
            args.resume, model, args.device, optimizer, scheduler, scaler
        )
        args.start_epoch = checkpoint.get("epoch", args.start_epoch)
        best_iou = checkpoint.get("best_iou", 0.0)
        best_j_index = checkpoint.get("best_j_index", 0.0)
        last_iou = checkpoint.get("cur_iou")
        last_prec = checkpoint.get("prec", {})
        last_j_index = checkpoint.get("j_index", [0.0, 0.0])
        logger.info("Resumed {} at epoch {}", args.resume, args.start_epoch)
        if checkpoint.get("mid_epoch"):
            logger.warning(
                "Resume checkpoint is mid-epoch epoch={} iteration={}/{}. "
                "Training will restart that epoch from the beginning with the saved model state.",
                checkpoint.get("epoch_in_progress"),
                checkpoint.get("iteration"),
                checkpoint.get("total_batches"),
            )

    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        epoch_log = epoch + 1
        epoch_started = time.perf_counter()
        args.memory_samples = []
        args.loss_finite = True
        args.loss_min = float("inf")
        args.loss_max = float("-inf")
        record_memory_sample(args.memory_samples, args.device, "epoch_start", 0)

        train_started = time.perf_counter()
        args.mid_epoch_checkpoint_callback = _make_mid_epoch_checkpoint_callback(
            args, model, optimizer, scheduler, scaler, best_iou, best_j_index
        )
        train_with_grasp(
            train_loader, model, optimizer, scheduler, scaler, epoch_log, args
        )
        args.mid_epoch_checkpoint_callback = None
        train_seconds = time.perf_counter() - train_started

        validation_started = time.perf_counter()
        validated_this_epoch = _should_validate_epoch(epoch_log, args)
        if validated_this_epoch and args.use_grasp_masks:
            last_iou, last_prec, last_j_index = validate_with_grasp(
                val_loader, model, epoch_log, args
            )
        elif validated_this_epoch:
            last_iou, last_prec, last_j_index = validate_without_grasp(
                val_loader, model, epoch_log, args
            )
        else:
            logger.info(
                "Skipping validation at epoch {}/{}; validation_interval={}",
                epoch_log,
                args.epochs,
                _validation_interval(args),
            )
        validation_seconds = time.perf_counter() - validation_started
        record_memory_sample(
            args.memory_samples,
            args.device,
            "validation_end" if validated_this_epoch else "validation_skipped",
            len(val_loader) if validated_this_epoch else 0,
        )

        if validated_this_epoch:
            best_iou = max(best_iou, last_iou)
            best_j_index = max(best_j_index, last_j_index[0])
        last_path = os.path.join(args.output_dir, "last_model.pth")
        checkpoint_started = time.perf_counter()
        save_checkpoint(
            last_path,
            model,
            optimizer,
            scheduler,
            scaler,
            epoch=epoch_log,
            cur_iou=last_iou,
            best_iou=best_iou,
            best_j_index=best_j_index,
            prec=last_prec,
            j_index=last_j_index,
            validated=validated_this_epoch,
            validation_interval=_validation_interval(args),
        )
        if validated_this_epoch and last_iou >= best_iou:
            copy_checkpoint_atomic(
                last_path, os.path.join(args.output_dir, "best_iou_model.pth")
            )
        if validated_this_epoch and last_j_index[0] >= best_j_index:
            copy_checkpoint_atomic(
                last_path, os.path.join(args.output_dir, "best_jindex_model.pth")
            )
        checkpoint_seconds = time.perf_counter() - checkpoint_started
        record_memory_sample(
            args.memory_samples, args.device, "checkpoint_end", len(train_loader)
        )

        epoch_seconds = time.perf_counter() - epoch_started
        timing = _timing_summary(
            train_seconds,
            validation_seconds,
            checkpoint_seconds,
            epoch_seconds,
            len(train_loader),
        )
        timing["checkpoint_size_bytes"] = os.path.getsize(last_path)
        timing["loss_finite"] = bool(args.loss_finite)
        timing["loss_min"] = args.loss_min
        timing["loss_max"] = args.loss_max
        timing["estimated_50_epoch_seconds"] = epoch_seconds * 50
        timing["memory"] = _memory_summary(args.memory_samples)
        timing_path = os.path.join(
            args.output_dir, "timing_epoch_{:03d}.json".format(epoch_log)
        )
        with open(timing_path, "w") as timing_file:
            json.dump(timing, timing_file, indent=2)
        logger.info("TIMING_SUMMARY {}", json.dumps(timing))

        scheduler.step()
        empty_cache(args.device)

    elapsed = datetime.timedelta(seconds=int(time.time() - start_time))
    logger.info("Run wall time seconds={:.3f}", time.perf_counter() - run_started)
    logger.info("Best IoU={} Best J@1={} Training time={}", best_iou, best_j_index, elapsed)


if __name__ == "__main__":
    main()
