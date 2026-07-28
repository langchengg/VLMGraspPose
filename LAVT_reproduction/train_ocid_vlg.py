"""Train the official LAVT architecture on an audited OCID-VLG manifest."""

from __future__ import annotations

import json
import math
import os
import shutil
import sys
import time
from argparse import Namespace
from datetime import datetime
from pathlib import Path
from typing import Any

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch
import yaml

from args import get_parser
from ocid_vlg.checkpoint import load_checkpoint, save_checkpoint
from ocid_vlg.device import resolve_device
from ocid_vlg.engine import (
    build_dataset,
    build_loader,
    build_model,
    configure_logging,
    environment_snapshot,
    evaluate_loader,
    official_parameter_groups,
    save_json,
    save_yaml,
    set_seed,
    sha256_file,
    train_one_epoch,
)
from ocid_vlg.losses import build_loss


def parse_args() -> Namespace:
    parser = get_parser()
    preliminary, _ = parser.parse_known_args()
    if preliminary.config:
        config_path = Path(preliminary.config).expanduser().resolve()
        values = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        if not isinstance(values, dict):
            parser.error(f"config must contain a YAML mapping: {config_path}")
        aliases = {
            "batch_size": "batch_size",
            "weight_decay": "weight_decay",
            "num_workers": "num_workers",
            "bert": "ck_bert",
        }
        known_dests = {action.dest for action in parser._actions}
        defaults: dict[str, Any] = {}
        unknown: list[str] = []
        for key, value in values.items():
            destination = aliases.get(key, key)
            if destination in known_dests:
                defaults[destination] = value
            else:
                unknown.append(key)
        if unknown:
            parser.error(
                f"unknown config keys in {config_path}: {', '.join(sorted(unknown))}"
            )
        parser.set_defaults(**defaults)
    args = parser.parse_args()
    if args.num_workers is None:
        args.num_workers = args.workers
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = (
        "1" if args.pytorch_enable_mps_fallback else "0"
    )
    return args


def _validate_args(args: Namespace) -> None:
    required = {
        "--ocid_root": args.ocid_root,
        "--ocid_api_root": args.ocid_api_root,
        "--train_manifest": args.train_manifest,
        "--val_manifest": args.val_manifest,
        "--pretrained_swin_weights": args.pretrained_swin_weights,
        "--ck_bert": args.ck_bert,
    }
    if args.resume:
        required.pop("--pretrained_swin_weights")
    missing = [name for name, value in required.items() if not value]
    if missing:
        raise ValueError(f"missing required arguments: {', '.join(missing)}")
    if args.model != "lavt_one":
        raise ValueError("OCID-VLG primary reproduction requires --model lavt_one")
    if args.optimizer != "AdamW" or args.scheduler != "polynomial":
        raise ValueError("only the audited AdamW + polynomial schedule is supported")
    if args.polynomial_power <= 0:
        raise ValueError("--polynomial_power must be positive")
    if args.batch_size < 1 or args.grad_accum_steps < 1:
        raise ValueError("batch size and grad accumulation must be positive")
    if args.stop_after_epochs is not None and args.stop_after_epochs < 1:
        raise ValueError("--stop_after_epochs must be positive")
    actual_effective = args.batch_size * args.grad_accum_steps
    if (
        args.effective_batch_size is not None
        and args.effective_batch_size != actual_effective
    ):
        raise ValueError(
            f"effective batch mismatch: declared {args.effective_batch_size}, "
            f"actual {actual_effective}"
        )


def _run_directory(args: Namespace) -> Path:
    if args.resolved_run_dir:
        return Path(args.resolved_run_dir).expanduser().resolve()
    if args.resume:
        resume = Path(args.resume).expanduser().resolve()
        if resume.parent.name == "checkpoints":
            return resume.parent.parent
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return (Path(args.output_root) / f"{timestamp}_{args.run_name}").resolve()


def _append_jsonl(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False) + "\n")


def main() -> int:
    args = parse_args()
    _validate_args(args)
    device = resolve_device(args.device)
    set_seed(args.seed)
    run_dir = _run_directory(args)
    checkpoints_dir = run_dir / "checkpoints"
    for directory in (
        run_dir,
        checkpoints_dir,
        run_dir / "predictions",
        run_dir / "figures",
        run_dir / "reports",
    ):
        directory.mkdir(parents=True, exist_ok=True)
    logger = configure_logging(run_dir / "train.log")
    logger.info("run_dir=%s device=%s", run_dir, device)
    logger.info(
        "PYTORCH_ENABLE_MPS_FALLBACK=%s",
        os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK"),
    )
    previous_training_time_seconds = 0.0
    previous_status_path = run_dir / "RUN_STATUS.json"
    if args.resume and previous_status_path.is_file():
        previous_status = json.loads(
            previous_status_path.read_text(encoding="utf-8")
        )
        previous_training_time_seconds = float(
            previous_status.get("total_training_time_seconds", 0.0)
        )

    resolved_config = vars(args).copy()
    resolved_config["device_resolved"] = str(device)
    resolved_config["effective_batch_size_actual"] = (
        args.batch_size * args.grad_accum_steps
    )
    resolved_config["run_dir"] = str(run_dir)
    resolved_config["dataset_version"] = args.ocid_version
    if args.pretrained_swin_weights and Path(args.pretrained_swin_weights).is_file():
        weight_path = Path(args.pretrained_swin_weights).resolve()
        resolved_config["pretrained_swin"] = {
            "path": str(weight_path),
            "size_bytes": weight_path.stat().st_size,
            "sha256": sha256_file(weight_path),
        }
    if Path(args.ck_bert).is_dir():
        bert_path = Path(args.ck_bert).resolve()
        resolved_config["bert_files"] = {
            file.name: {
                "size_bytes": file.stat().st_size,
                "sha256": sha256_file(file),
            }
            for file in bert_path.iterdir()
            if file.is_file()
        }
    resolved_config_path = run_dir / "config_resolved.yaml"
    resume_config_path: Path | None = None
    if args.resume and resolved_config_path.is_file():
        initial_config = (
            yaml.safe_load(resolved_config_path.read_text(encoding="utf-8")) or {}
        )
        if not isinstance(initial_config, dict):
            raise ValueError(f"invalid existing resolved config: {resolved_config_path}")
        resolved_config = {**initial_config, **resolved_config}
        resume_config_path = run_dir / (
            "config_resume_invocation_"
            f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.yaml"
        )
        save_yaml(resume_config_path, resolved_config)
    else:
        save_yaml(resolved_config_path, resolved_config)
    save_json(run_dir / "environment.json", environment_snapshot(device))
    audit_source = Path("outputs/dataset_audit.json")
    if audit_source.is_file():
        shutil.copy2(audit_source, run_dir / "dataset_audit.json")

    train_dataset = build_dataset(
        args,
        "train",
        args.train_manifest,
        args.limit_train_samples,
    )
    val_dataset = build_dataset(
        args,
        args.validation_split,
        args.val_manifest,
        args.limit_val_samples,
    )
    train_loader = build_loader(
        train_dataset,
        batch_size=args.batch_size,
        train=True,
        num_workers=args.num_workers,
        device=device,
    )
    val_loader = build_loader(
        val_dataset,
        batch_size=1,
        train=False,
        num_workers=args.num_workers,
        device=device,
    )
    logger.info("train_samples=%d val_samples=%d", len(train_dataset), len(val_dataset))

    model = build_model(
        args, device, initialize_backbone=not bool(args.resume)
    )
    load_audit = getattr(model.backbone, "pretrained_load_audit", None)
    if load_audit:
        save_json(run_dir / "pretrained_swin_load_audit.json", load_audit)
        resolved_config["pretrained_swin_load_audit"] = load_audit
        save_yaml(resolved_config_path, resolved_config)

    optimizer = torch.optim.AdamW(
        official_parameter_groups(model),
        lr=args.lr,
        weight_decay=args.weight_decay,
        amsgrad=args.amsgrad,
    )
    steps_per_epoch = math.ceil(len(train_loader) / args.grad_accum_steps)
    total_steps = max(1, steps_per_epoch * args.epochs)
    resolved_config["scheduler_steps_per_epoch"] = steps_per_epoch
    resolved_config["scheduler_total_steps"] = total_steps
    if resume_config_path is not None:
        save_yaml(resume_config_path, resolved_config)
    else:
        save_yaml(resolved_config_path, resolved_config)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: max(0.0, 1.0 - step / total_steps) ** args.polynomial_power,
    )
    criterion = build_loss(args.loss)
    start_epoch = 0
    best = {"mean_iou": -1.0, "overall_iou": -1.0}
    if args.resume:
        payload = load_checkpoint(
            args.resume,
            model,
            optimizer,
            scheduler,
            device=device,
            strict=True,
            restore_rng=True,
        )
        checkpoint_config = payload.get("config") or {}
        resume_invariants = (
            "model",
            "swin_type",
            "window12",
            "img_size",
            "max_tokens",
            "optimizer",
            "lr",
            "weight_decay",
            "scheduler",
            "polynomial_power",
            "epochs",
            "batch_size",
            "grad_accum_steps",
            "effective_batch_size",
            "train_manifest",
            "limit_train_samples",
            "scheduler_steps_per_epoch",
            "scheduler_total_steps",
            "seed",
        )
        drift = {
            key: {
                "checkpoint": checkpoint_config[key],
                "current": resolved_config[key],
            }
            for key in resume_invariants
            if key in checkpoint_config
            and checkpoint_config[key] != resolved_config.get(key)
        }
        if drift:
            raise ValueError(
                "resume would change optimizer/scheduler-defining configuration: "
                + json.dumps(drift, sort_keys=True)
            )
        start_epoch = int(payload["next_epoch"])
        best.update(payload.get("best_metrics") or payload.get("best") or {})
        logger.info("resumed checkpoint=%s next_epoch=%d", args.resume, start_epoch)

    run_started = time.perf_counter()
    metrics_path = run_dir / "metrics_history.jsonl"
    invocation_end_epoch = (
        min(args.epochs, start_epoch + args.stop_after_epochs)
        if args.stop_after_epochs is not None
        else args.epochs
    )
    for epoch in range(start_epoch, invocation_end_epoch):
        train_result = train_one_epoch(
            model,
            train_loader,
            optimizer,
            scheduler,
            criterion,
            device,
            epoch=epoch,
            grad_accum_steps=args.grad_accum_steps,
            print_freq=args.print_freq,
        )
        metrics_model, metrics_original, _ = evaluate_loader(
            model,
            val_loader,
            device,
            prediction_policy=args.prediction_policy,
            threshold=args.threshold,
            collect_arrays=False,
        )
        history = {
            **train_result,
            "validation_model_resolution": metrics_model,
            "validation_original_resolution": metrics_original,
            "elapsed_total_seconds": time.perf_counter() - run_started,
        }
        _append_jsonl(metrics_path, history)
        checkpoint_extra = {
            "completed_epochs": epoch + 1,
            "train_loss": train_result["train_loss"],
            "validation_model_resolution": metrics_model,
            "validation_original_resolution": metrics_original,
        }
        if metrics_original["mean_iou"] > best["mean_iou"]:
            best["mean_iou"] = metrics_original["mean_iou"]
            best["mean_iou_epoch"] = epoch
            save_checkpoint(
                checkpoints_dir / "checkpoint_best_miou.pth",
                model,
                optimizer,
                scheduler,
                epoch=epoch,
                best_metrics=best,
                config=resolved_config,
                seed=args.seed,
                device=device,
                extra=checkpoint_extra,
            )
        if metrics_original["overall_iou"] > best["overall_iou"]:
            best["overall_iou"] = metrics_original["overall_iou"]
            best["overall_iou_epoch"] = epoch
            save_checkpoint(
                checkpoints_dir / "checkpoint_best_overall_iou.pth",
                model,
                optimizer,
                scheduler,
                epoch=epoch,
                best_metrics=best,
                config=resolved_config,
                seed=args.seed,
                device=device,
                extra=checkpoint_extra,
            )
        save_checkpoint(
            checkpoints_dir / "checkpoint_last.pth",
            model,
            optimizer,
            scheduler,
            epoch=epoch,
            best_metrics=best,
            config=resolved_config,
            seed=args.seed,
            device=device,
            extra=checkpoint_extra,
        )
        logger.info(
            "epoch=%d train_loss=%.6f val_miou=%.6f val_oiou=%.6f",
            epoch,
            train_result["train_loss"],
            metrics_original["mean_iou"],
            metrics_original["overall_iou"],
        )

    completed_epochs = max(start_epoch, invocation_end_epoch)
    limited = any(
        value is not None
        for value in (
            args.limit_train_samples,
            args.limit_val_samples,
            args.limit_test_samples,
        )
    )
    training_horizon_complete = completed_epochs >= args.epochs
    status = (
        "SUCCESS_SMOKE"
        if limited and training_horizon_complete
        else "PARTIAL_TRAINING"
        if not training_horizon_complete
        else "TRAINING_COMPLETE_PENDING_FULL_TEST"
    )
    invocation_training_time_seconds = time.perf_counter() - run_started
    save_json(
        run_dir / "RUN_STATUS.json",
        {
            "status": status,
            "completed_epochs": completed_epochs,
            "last_invocation_training_time_seconds": invocation_training_time_seconds,
            "total_training_time_seconds": (
                previous_training_time_seconds + invocation_training_time_seconds
            ),
            "best": best,
            "checkpoint_last": str(
                (checkpoints_dir / "checkpoint_last.pth").resolve()
            ),
        },
    )
    print(str(run_dir))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"{type(error).__name__}: {error}", file=sys.stderr)
        raise
