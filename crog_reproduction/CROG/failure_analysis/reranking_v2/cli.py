from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch

from failure_analysis.reranking.exporter import main as export_v1_compatible

from .artifacts import atomic_savez_compressed
from .calibration import fit_temperature_from_artifacts
from .enhanced_data import flatten_candidate_arrays, load_enhanced_arrays
from .evaluation import (
    combine_method_evaluations,
    evaluate_method,
)
from .extract import extract_enhanced_features
from .gallery import build_failure_gallery
from .inference import (
    export_npz_rankings,
    predict_primary_ensemble,
    predict_scalar_gate_ensemble,
)
from .independent_evaluator import assert_matches_primary, recompute_counts
from .labels import build_dual_labels
from .models.latent_residual import (
    predict_latent_residual_arrays,
    train_latent_residual_arrays,
)
from .models.rgbd_critic import (
    predict_critic_arrays,
    train_critic_arrays,
)
from .models.setrank import (
    predict_setrank_arrays,
    train_setrank_arrays,
)
from .models.uncertainty import stability_statistics
from .models.vlm_reviewer import prepare_vlm_dry_run
from .oof import (
    build_token_ablation_artifacts,
    train_oof_base_models,
    train_oof_setrank_and_gate,
)
from .protocol import claim_test_once, lock_experiment, split_ids, verify_lock
from .reporting import assemble_results_bundle, build_report_artifact
from .schema import atomic_write_json, read_jsonl
from .schema import sha256_file
from .splits import build_split_manifest
from .stability_runner import run_stability_extraction
from .training import _atomic_torch_save, train_scalar_gate_experiment
from .datasets import load_inference_features, load_joined


REPO_ROOT = Path(__file__).resolve().parents[2]


def _common(parser):
    parser.add_argument("--device", choices=("auto", "mps", "cpu"), default="auto")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")


def _parser():
    parser = argparse.ArgumentParser(
        description="CROG Re-ranking V2 leakage-resistant experiment CLI"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    audit = subparsers.add_parser("audit")
    audit.add_argument("--dataset-root", default="../OCID-VLG")
    audit.add_argument("--output", required=True)
    audit.add_argument("--no-content-hash", action="store_true")
    _common(audit)

    export = subparsers.add_parser("export-dev")
    export.add_argument("--split", choices=("train", "val"), required=True)
    export.add_argument("--output", required=True)
    export.add_argument("--batch-size", type=int, default=16)
    _common(export)

    labels = subparsers.add_parser("build-labels")
    labels.add_argument("--features", required=True)
    labels.add_argument("--predictions", required=True)
    labels.add_argument("--output", required=True)
    _common(labels)

    for name in ("extract-crops", "extract-latents"):
        extract = subparsers.add_parser(name)
        extract.add_argument("--split", choices=("train", "val", "test"), required=True)
        extract.add_argument("--frozen-features", required=True)
        extract.add_argument("--split-manifest", required=True)
        extract.add_argument("--output", required=True)
        extract.add_argument("--batch-size", type=int, default=8)
        extract.add_argument("--crop-size", type=int, default=32)
        extract.add_argument("--roi-size", type=int, default=5)
        _common(extract)

    critic = subparsers.add_parser("train-critic")
    _training_data_args(critic, enhanced=True)
    critic.add_argument(
        "--ablation",
        choices=("rgb", "rgb_mask_q", "rgb_depth", "all", "all_no_template"),
        default="all",
    )
    critic.add_argument("--output", required=True)
    critic.add_argument("--epochs", type=int, default=20)
    _common(critic)

    latent = subparsers.add_parser("train-latent")
    _training_data_args(latent, enhanced=True)
    latent.add_argument(
        "--feature-set", choices=("scalar", "pre_decoder", "post_decoder"), default="post_decoder"
    )
    latent.add_argument("--alpha", type=float, default=1.0)
    latent.add_argument("--output", required=True)
    latent.add_argument("--epochs", type=int, default=25)
    _common(latent)

    setrank = subparsers.add_parser("train-setrank")
    _training_data_args(setrank, enhanced=True)
    setrank.add_argument("--tokens-npz")
    setrank.add_argument("--alpha", type=float, default=1.0)
    setrank.add_argument("--output", required=True)
    setrank.add_argument("--epochs", type=int, default=25)
    _common(setrank)

    oof_base = subparsers.add_parser("train-oof-base")
    _training_data_args(oof_base, enhanced=True)
    oof_base.add_argument("--output", required=True)
    oof_base.add_argument("--critic-epochs", type=int, default=10)
    oof_base.add_argument("--latent-epochs", type=int, default=12)
    _common(oof_base)

    oof_primary = subparsers.add_parser("train-oof-primary")
    _training_data_args(oof_primary, enhanced=True)
    oof_primary.add_argument("--base-oof", required=True)
    oof_primary.add_argument("--base-validation", required=True)
    oof_primary.add_argument("--output", required=True)
    oof_primary.add_argument("--setrank-epochs", type=int, default=12)
    oof_primary.add_argument("--gate-epochs", type=int, default=25)
    _common(oof_primary)

    token_ablations = subparsers.add_parser("build-token-ablations")
    _training_data_args(token_ablations, enhanced=True)
    token_ablations.add_argument("--base-oof", required=True)
    token_ablations.add_argument("--base-validation", required=True)
    token_ablations.add_argument("--output", required=True)
    _common(token_ablations)

    gate = subparsers.add_parser("train-gate")
    gate.add_argument("--train-features", required=True)
    gate.add_argument("--train-legacy-labels", required=True)
    gate.add_argument("--validation-features", required=True)
    gate.add_argument("--validation-legacy-labels", required=True)
    gate.add_argument("--validation-corrected-labels", required=True)
    gate.add_argument("--split-manifest", required=True)
    gate.add_argument("--output", required=True)
    gate.add_argument("--epochs", type=int, default=40)
    _common(gate)

    stability = subparsers.add_parser("run-stability")
    stability.add_argument("--scores", required=True, help="NPY [samples,K,perturbations]")
    stability.add_argument("--kappa", type=float, required=True)
    stability.add_argument("--output", required=True)
    _common(stability)

    extract_stability = subparsers.add_parser("extract-stability")
    extract_stability.add_argument(
        "--split", choices=("train", "val", "test"), required=True
    )
    extract_stability.add_argument("--frozen-features", required=True)
    extract_stability.add_argument("--split-manifest", required=True)
    extract_stability.add_argument(
        "--critic-models", nargs="+", required=True
    )
    extract_stability.add_argument("--crop-size", type=int, default=32)
    extract_stability.add_argument("--kappa", type=float, default=1.0)
    extract_stability.add_argument("--batch-size", type=int, default=8)
    extract_stability.add_argument("--output", required=True)
    _common(extract_stability)

    prepare_vlm = subparsers.add_parser("prepare-vlm")
    prepare_vlm.add_argument("--request-manifest")
    prepare_vlm.add_argument("--features")
    prepare_vlm.add_argument("--max-panels", type=int, default=5)
    prepare_vlm.add_argument("--output", required=True)
    _common(prepare_vlm)

    run_vlm = subparsers.add_parser("run-vlm")
    run_vlm.add_argument("--cache-dir", required=True)
    run_vlm.add_argument("--replay", action="store_true")
    run_vlm.add_argument("--output", required=True)
    _common(run_vlm)

    for name in ("evaluate-validation", "evaluate-test"):
        evaluate = subparsers.add_parser(name)
        evaluate.add_argument("--summary-input")
        evaluate.add_argument("--method")
        evaluate.add_argument("--features")
        evaluate.add_argument("--legacy-labels")
        evaluate.add_argument("--corrected-labels")
        evaluate.add_argument("--predictions")
        evaluate.add_argument("--expected-legacy-oracle", type=float)
        evaluate.add_argument("--bootstrap-iterations", type=int, default=10_000)
        evaluate.add_argument("--output", required=True)
        _common(evaluate)

    combine = subparsers.add_parser("combine-evaluations")
    combine.add_argument("--summaries", nargs="+", required=True)
    combine.add_argument("--output", required=True)
    _common(combine)

    export_rankings = subparsers.add_parser("export-rankings")
    export_rankings.add_argument("--features", required=True)
    export_rankings.add_argument("--scores-npz", required=True)
    export_rankings.add_argument("--score-key", default="scores")
    export_rankings.add_argument("--probability-key")
    export_rankings.add_argument("--method", required=True)
    export_rankings.add_argument("--output", required=True)
    _common(export_rankings)

    calibrate = subparsers.add_parser("fit-calibration")
    calibrate.add_argument("--predictions", required=True)
    calibrate.add_argument("--labels", required=True)
    calibrate.add_argument("--output", required=True)
    _common(calibrate)

    predict = subparsers.add_parser("predict-primary")
    predict.add_argument("--features", required=True)
    predict.add_argument("--enhanced", required=True)
    predict.add_argument("--critic-models", nargs="+", required=True)
    predict.add_argument("--latent-models", nargs="+", required=True)
    predict.add_argument("--setrank-models", nargs="+", required=True)
    predict.add_argument("--gate-models", nargs="+", required=True)
    predict.add_argument("--policy", required=True)
    predict.add_argument("--split-manifest")
    predict.add_argument(
        "--partition",
        choices=("train", "calibration", "validation", "test"),
    )
    predict.add_argument("--stability")
    predict.add_argument("--alpha", type=float, default=1.0)
    predict.add_argument("--kappa", type=float, default=1.0)
    predict.add_argument("--required-consensus", type=int, default=2)
    predict.add_argument(
        "--candidate-probability-temperature", type=float, default=1.0
    )
    predict.add_argument("--output", required=True)
    _common(predict)

    lock = subparsers.add_parser("lock-experiment")
    lock.add_argument("--spec", required=True)
    lock.add_argument("--output", required=True)
    _common(lock)

    run_test = subparsers.add_parser("run-test")
    run_test.add_argument("--lock", required=True)
    run_test.add_argument("--test-run-dir", required=True)
    run_test.add_argument("--frozen-features", required=True)
    run_test.add_argument("--frozen-predictions", required=True)
    run_test.add_argument("--split-manifest", required=True)
    run_test.add_argument("--batch-size", type=int, default=8)
    _common(run_test)

    report = subparsers.add_parser("build-report")
    report.add_argument("--results", required=True)
    report.add_argument("--output", required=True)
    _common(report)

    assemble_report = subparsers.add_parser("assemble-report")
    assemble_report.add_argument("--validation-comparison", required=True)
    assemble_report.add_argument("--test-comparison", required=True)
    assemble_report.add_argument("--primary-test-summary", required=True)
    assemble_report.add_argument("--primary-method", required=True)
    assemble_report.add_argument("--calibration")
    assemble_report.add_argument("--vlm-status", default="blocked")
    assemble_report.add_argument("--output", required=True)
    _common(assemble_report)

    gallery = subparsers.add_parser("build-gallery")
    gallery.add_argument("--features", required=True)
    gallery.add_argument("--legacy-labels", required=True)
    gallery.add_argument("--raw-predictions", required=True)
    gallery.add_argument("--reranker-predictions", required=True)
    gallery.add_argument("--per-group", type=int, default=5)
    gallery.add_argument("--output", required=True)
    _common(gallery)
    return parser


def _training_data_args(parser, *, enhanced):
    parser.add_argument("--train-features", required=True)
    parser.add_argument("--train-labels", required=True)
    parser.add_argument("--validation-features", required=True)
    parser.add_argument("--validation-labels", required=True)
    parser.add_argument("--split-manifest", required=True)
    if enhanced:
        parser.add_argument("--train-enhanced", required=True)
        parser.add_argument("--validation-enhanced", required=True)


def _development_samples(args):
    train = load_joined(
        args.train_features,
        args.train_labels,
        allowed_sample_ids=split_ids(args.split_manifest, "train"),
    )
    validation = load_joined(args.validation_features, args.validation_labels)
    if args.max_samples:
        train = train[: args.max_samples]
        validation = validation[: args.max_samples]
    return train, validation


def _run_critic(args):
    output = Path(args.output)
    if (
        args.resume
        and output.exists()
        and output.with_suffix(".validation.npz").exists()
    ):
        return {
            "model": str(output),
            "validation_predictions": str(
                output.with_suffix(".validation.npz")
            ),
            "resumed_complete": True,
        }
    train, validation = _development_samples(args)
    train_arrays = load_enhanced_arrays(args.train_enhanced, train)
    validation_arrays = load_enhanced_arrays(args.validation_enhanced, validation)
    train_flat = flatten_candidate_arrays(train_arrays)
    validation_flat = flatten_candidate_arrays(validation_arrays)
    channel_sets = {
        "rgb": (0, 1, 2),
        "rgb_mask_q": (0, 1, 2, 5, 6),
        "rgb_depth": (0, 1, 2, 3, 4),
        "all": tuple(range(14)),
        "all_no_template": tuple(range(10)),
    }
    if args.dry_run:
        return {"train": len(train), "validation": len(validation), "channels": channel_sets[args.ablation]}
    artifact = (
        torch.load(output, map_location="cpu", weights_only=False)
        if args.resume and output.exists()
        else train_critic_arrays(
            train_flat["crops"],
            train_flat["labels"],
            train_flat["sample_index"],
            train_flat["q"],
            validation_flat["crops"],
            validation_flat["labels"],
            validation_flat["sample_index"],
            validation_flat["q"],
            channels=channel_sets[args.ablation],
            seed=args.seed,
            device=args.device,
            epochs=args.epochs,
        )
    )
    if not output.exists():
        _atomic_torch_save(artifact, output)
    scores, embeddings = predict_critic_arrays(
        artifact, validation_flat["crops"], device=args.device
    )
    atomic_savez_compressed(
        output.with_suffix(".validation.npz"),
        scores=scores.reshape(len(validation), 5),
        embeddings=embeddings.reshape(len(validation), 5, -1),
        sample_ids=validation_arrays["sample_ids"],
    )
    return {"model": str(output), "train": len(train), "validation": len(validation)}


def _run_latent(args):
    output = Path(args.output)
    if (
        args.resume
        and output.exists()
        and output.with_suffix(".validation.npz").exists()
    ):
        return {
            "model": str(output),
            "validation_predictions": str(
                output.with_suffix(".validation.npz")
            ),
            "resumed_complete": True,
        }
    train, validation = _development_samples(args)
    train_arrays = load_enhanced_arrays(args.train_enhanced, train, include_crops=False)
    validation_arrays = load_enhanced_arrays(
        args.validation_enhanced, validation, include_crops=False
    )
    name = {
        "scalar": "scalar",
        "pre_decoder": "latent_pre",
        "post_decoder": "latent_post",
    }[args.feature_set]
    if args.dry_run:
        return {"feature_set": name, "train": len(train), "validation": len(validation)}
    artifact = (
        torch.load(output, map_location="cpu", weights_only=False)
        if args.resume and output.exists()
        else train_latent_residual_arrays(
            train_arrays[name],
            train_arrays["q"],
            train_arrays["labels"],
            validation_arrays[name],
            validation_arrays["q"],
            validation_arrays["labels"],
            seed=args.seed,
            device=args.device,
            epochs=args.epochs,
        )
    )
    if not output.exists():
        _atomic_torch_save(artifact, output)
    scores, residuals = predict_latent_residual_arrays(
        artifact,
        validation_arrays[name],
        validation_arrays["q"],
        alpha=args.alpha,
        device=args.device,
    )
    atomic_savez_compressed(
        output.with_suffix(".validation.npz"),
        scores=scores,
        residuals=residuals,
        sample_ids=validation_arrays["sample_ids"],
    )
    return {"model": str(output), "train": len(train), "validation": len(validation)}


def _run_setrank(args):
    output = Path(args.output)
    if (
        args.resume
        and output.exists()
        and output.with_suffix(".validation.npz").exists()
    ):
        return {
            "model": str(output),
            "validation_predictions": str(
                output.with_suffix(".validation.npz")
            ),
            "resumed_complete": True,
        }
    train, validation = _development_samples(args)
    train_arrays = load_enhanced_arrays(args.train_enhanced, train, include_crops=False)
    validation_arrays = load_enhanced_arrays(
        args.validation_enhanced, validation, include_crops=False
    )
    if args.tokens_npz:
        with np.load(args.tokens_npz) as payload:
            train_tokens = payload["train_tokens"]
            validation_tokens = payload["validation_tokens"]
    else:
        train_tokens = train_arrays["scalar"]
        validation_tokens = validation_arrays["scalar"]
    if args.dry_run:
        return {"token_dim": train_tokens.shape[-1], "train": len(train)}
    artifact = (
        torch.load(output, map_location="cpu", weights_only=False)
        if args.resume and output.exists()
        else train_setrank_arrays(
            train_tokens,
            train_arrays["q"],
            train_arrays["labels"],
            validation_tokens,
            validation_arrays["q"],
            validation_arrays["labels"],
            seed=args.seed,
            device=args.device,
            epochs=args.epochs,
        )
    )
    if not output.exists():
        _atomic_torch_save(artifact, output)
    scores, probabilities, residuals = predict_setrank_arrays(
        artifact,
        validation_tokens,
        validation_arrays["q"],
        alpha=args.alpha,
        device=args.device,
    )
    atomic_savez_compressed(
        output.with_suffix(".validation.npz"),
        scores=scores,
        probabilities=probabilities,
        residuals=residuals,
        sample_ids=validation_arrays["sample_ids"],
    )
    return {"model": str(output), "train": len(train), "validation": len(validation)}


def main(argv=None):
    args = _parser().parse_args(argv)
    command = args.command
    if command == "audit":
        if args.dry_run:
            result = {"dataset_root": str(Path(args.dataset_root).resolve())}
        else:
            result = build_split_manifest(
                args.dataset_root,
                args.output,
                hash_content=not args.no_content_hash,
            )["audit"]
    elif command == "export-dev":
        forwarded = [
            "--split",
            args.split,
            "--output",
            args.output,
            "--batch-size",
            str(args.batch_size),
            "--workers",
            str(args.num_workers),
            "--device",
            "mps" if args.device == "auto" else args.device,
        ]
        if args.max_samples:
            forwarded += ["--limit", str(args.max_samples)]
        if args.resume:
            forwarded.append("--resume")
        if args.dry_run:
            result = {"forwarded_arguments": forwarded}
        else:
            export_v1_compatible(forwarded)
            return
    elif command == "build-labels":
        result = (
            {"features": args.features, "predictions": args.predictions}
            if args.dry_run
            else {
                key: str(value)
                for key, value in build_dual_labels(
                    args.features,
                    args.predictions,
                    args.output,
                    resume=args.resume,
                ).items()
            }
        )
    elif command in ("extract-crops", "extract-latents"):
        if args.dry_run:
            result = {"shared_extraction": True, "split": args.split}
        else:
            result = extract_enhanced_features(
                split=args.split,
                frozen_features_path=args.frozen_features,
                output_dir=args.output,
                split_manifest_path=args.split_manifest,
                device=args.device,
                batch_size=args.batch_size,
                workers=args.num_workers,
                crop_size=args.crop_size,
                roi_size=args.roi_size,
                max_samples=args.max_samples,
                resume=args.resume,
                seed=args.seed,
            )
    elif command == "train-critic":
        result = _run_critic(args)
    elif command == "train-latent":
        result = _run_latent(args)
    elif command == "train-setrank":
        result = _run_setrank(args)
    elif command == "train-oof-base":
        completed_summary = Path(args.output) / "summary.json"
        if args.resume and completed_summary.exists():
            result = json.loads(
                completed_summary.read_text(encoding="utf-8")
            )
        else:
            train, validation = _development_samples(args)
            if args.dry_run:
                result = {
                    "train": len(train),
                    "validation": len(validation),
                    "folds": 3,
                    "seeds": [31, 37, 43],
                }
            else:
                result = train_oof_base_models(
                    train_samples=train,
                    validation_samples=validation,
                    train_arrays=load_enhanced_arrays(
                        args.train_enhanced, train
                    ),
                    validation_arrays=load_enhanced_arrays(
                        args.validation_enhanced, validation
                    ),
                    split_manifest=args.split_manifest,
                    output_dir=args.output,
                    seeds=(31, 37, 43),
                    device=args.device,
                    critic_epochs=args.critic_epochs,
                    latent_epochs=args.latent_epochs,
                    resume=args.resume,
                )
    elif command == "train-oof-primary":
        completed_summary = Path(args.output) / "summary.json"
        if args.resume and completed_summary.exists():
            result = json.loads(
                completed_summary.read_text(encoding="utf-8")
            )
        else:
            train, validation = _development_samples(args)
            if args.dry_run:
                result = {
                    "train": len(train),
                    "validation": len(validation),
                    "folds": 3,
                    "seeds": [31, 37, 43],
                }
            else:
                result = train_oof_setrank_and_gate(
                    train_samples=train,
                    validation_samples=validation,
                    train_arrays=load_enhanced_arrays(
                        args.train_enhanced,
                        train,
                        include_crops=False,
                    ),
                    validation_arrays=load_enhanced_arrays(
                        args.validation_enhanced,
                        validation,
                        include_crops=False,
                    ),
                    base_oof_path=args.base_oof,
                    base_validation_path=args.base_validation,
                    split_manifest=args.split_manifest,
                    output_dir=args.output,
                    seeds=(31, 37, 43),
                    device=args.device,
                    setrank_epochs=args.setrank_epochs,
                    gate_epochs=args.gate_epochs,
                    resume=args.resume,
                )
    elif command == "build-token-ablations":
        completed_summary = Path(args.output) / "summary.json"
        if args.resume and completed_summary.exists():
            result = json.loads(
                completed_summary.read_text(encoding="utf-8")
            )
        else:
            train, validation = _development_samples(args)
            if args.dry_run:
                result = {
                    "train": len(train),
                    "validation": len(validation),
                    "variants": [
                        "scalar",
                        "scalar_critic",
                        "scalar_latent",
                        "scalar_critic_latent",
                    ],
                }
            else:
                result = build_token_ablation_artifacts(
                    train_arrays=load_enhanced_arrays(
                        args.train_enhanced,
                        train,
                        include_crops=False,
                    ),
                    validation_arrays=load_enhanced_arrays(
                        args.validation_enhanced,
                        validation,
                        include_crops=False,
                    ),
                    base_oof_path=args.base_oof,
                    base_validation_path=args.base_validation,
                    output_dir=args.output,
                    resume=args.resume,
                )
    elif command == "train-gate":
        completed_summary = Path(args.output) / "validation_summary.json"
        if args.resume and completed_summary.exists():
            result = json.loads(
                completed_summary.read_text(encoding="utf-8")
            )
        else:
            result = (
                {"output": args.output}
                if args.dry_run
                else train_scalar_gate_experiment(
                    train_features_path=args.train_features,
                    train_legacy_labels_path=args.train_legacy_labels,
                    validation_features_path=args.validation_features,
                    validation_legacy_labels_path=args.validation_legacy_labels,
                    validation_corrected_labels_path=args.validation_corrected_labels,
                    split_manifest_path=args.split_manifest,
                    output_dir=args.output,
                    seed=args.seed,
                    device=args.device,
                    epochs=args.epochs,
                    resume=args.resume,
                )
            )
    elif command == "run-stability":
        scores = np.load(args.scores)
        stats = [
            [
                stability_statistics(scores[sample, candidate], args.kappa)
                for candidate in range(scores.shape[1])
            ]
            for sample in range(scores.shape[0])
        ]
        atomic_write_json(args.output, {"kappa": args.kappa, "statistics": stats})
        result = {"samples": scores.shape[0], "output": args.output}
    elif command == "extract-stability":
        result = (
            {
                "split": args.split,
                "critic_models": len(args.critic_models),
                "dry_run": True,
            }
            if args.dry_run
            else run_stability_extraction(
                split=args.split,
                frozen_features_path=args.frozen_features,
                split_manifest_path=args.split_manifest,
                critic_model_paths=args.critic_models,
                output_dir=args.output,
                device=args.device,
                batch_size=args.batch_size,
                workers=args.num_workers,
                crop_size=args.crop_size,
                kappa=args.kappa,
                max_samples=args.max_samples,
                resume=args.resume,
                seed=args.seed,
            )
        )
    elif command == "prepare-vlm":
        if args.features:
            result = (
                {
                    "features": args.features,
                    "max_panels": args.max_panels,
                    "dry_run": True,
                }
                if args.dry_run
                else prepare_vlm_dry_run(
                    features_path=args.features,
                    output_dir=args.output,
                    max_samples=args.max_panels,
                    seed=args.seed,
                )
            )
        elif args.request_manifest:
            requests = list(read_jsonl(args.request_manifest))
            forbidden = (
                "gt",
                "label",
                "success",
                "iou",
                "angle_error",
                "oracle",
            )
            for request in requests:
                for key in request:
                    if any(token in key.lower() for token in forbidden):
                        raise ValueError(
                            f"VLM request contains forbidden field: {key}"
                        )
            atomic_write_json(
                args.output,
                {
                    "status": "prepared",
                    "request_count": len(requests),
                    "live_calls": False,
                },
            )
            result = {
                "request_count": len(requests),
                "output": args.output,
            }
        else:
            raise ValueError(
                "prepare-vlm requires --features or --request-manifest"
            )
    elif command == "run-vlm":
        cache = Path(args.cache_dir)
        files = sorted(cache.glob("*.json"))
        if not files:
            result = {
                "status": "blocked",
                "reason": "no provider credentials and no replay cache",
                "fallback": "q_only",
            }
        else:
            result = {"status": "replayed", "response_count": len(files)}
        atomic_write_json(args.output, result)
    elif command in ("evaluate-validation", "evaluate-test"):
        if args.summary_input:
            payload = json.loads(
                Path(args.summary_input).read_text(encoding="utf-8")
            )
            atomic_write_json(args.output, payload)
            result = {
                "compatibility_copy": args.output,
                "source": args.summary_input,
            }
        else:
            required = {
                "--method": args.method,
                "--features": args.features,
                "--legacy-labels": args.legacy_labels,
                "--corrected-labels": args.corrected_labels,
            }
            missing = [name for name, value in required.items() if not value]
            if missing:
                raise ValueError(
                    "full evaluation requires " + ", ".join(missing)
                )
            result = (
                {
                    "method": args.method,
                    "features": args.features,
                    "dry_run": True,
                }
                if args.dry_run
                else evaluate_method(
                    method=args.method,
                    features_path=args.features,
                    legacy_labels_path=args.legacy_labels,
                    corrected_labels_path=args.corrected_labels,
                    prediction_path=args.predictions,
                    output_dir=args.output,
                    expected_legacy_oracle=args.expected_legacy_oracle,
                    bootstrap_iterations=args.bootstrap_iterations,
                    bootstrap_seed=args.seed,
                )
            )
    elif command == "combine-evaluations":
        result = (
            {"summaries": args.summaries, "output": args.output}
            if args.dry_run
            else combine_method_evaluations(args.summaries, args.output)
        )
    elif command == "export-rankings":
        samples = load_inference_features(args.features)
        if args.max_samples:
            samples = samples[: args.max_samples]
        result = (
            {
                "method": args.method,
                "samples": len(samples),
                "scores_npz": args.scores_npz,
                "dry_run": True,
            }
            if args.dry_run
            else export_npz_rankings(
                samples=samples,
                npz_path=args.scores_npz,
                output_path=args.output,
                method=args.method,
                score_key=args.score_key,
                probability_key=args.probability_key,
            )
        )
    elif command == "fit-calibration":
        result = (
            {
                "predictions": args.predictions,
                "labels": args.labels,
                "dry_run": True,
            }
            if args.dry_run
            else fit_temperature_from_artifacts(
                predictions_path=args.predictions,
                labels_path=args.labels,
                output_path=args.output,
            )
        )
    elif command == "predict-primary":
        samples = load_inference_features(args.features)
        if bool(args.partition) != bool(args.split_manifest):
            raise ValueError(
                "--partition and --split-manifest must be provided together"
            )
        if args.partition:
            allowed = split_ids(args.split_manifest, args.partition)
            samples = [
                sample for sample in samples
                if sample.sample_id in allowed
            ]
        if args.max_samples:
            samples = samples[: args.max_samples]
        policy = json.loads(Path(args.policy).read_text(encoding="utf-8"))
        if "selected" in policy:
            policy = policy["selected"]
        completed_summary = Path(args.output) / "summary.json"
        if args.resume and completed_summary.exists():
            result = json.loads(
                completed_summary.read_text(encoding="utf-8")
            )
        elif args.dry_run:
            result = {
                "samples": len(samples),
                "models": {
                    "critic": len(args.critic_models),
                    "latent": len(args.latent_models),
                    "setrank": len(args.setrank_models),
                    "gate": len(args.gate_models),
                },
            }
        else:
            arrays = load_enhanced_arrays(
                args.enhanced,
                samples,
                include_labels=False,
            )
            result = predict_primary_ensemble(
                samples=samples,
                arrays=arrays,
                critic_models=args.critic_models,
                latent_models=args.latent_models,
                setrank_models=args.setrank_models,
                gate_models=args.gate_models,
                policy=policy,
                output_dir=args.output,
                device=args.device,
                alpha=args.alpha,
                uncertainty_kappa=args.kappa,
                required_consensus=args.required_consensus,
                candidate_probability_temperature=(
                    args.candidate_probability_temperature
                ),
                stability_path=args.stability,
            )
    elif command == "lock-experiment":
        spec = json.loads(Path(args.spec).read_text(encoding="utf-8"))
        result = lock_experiment(args.output, repo_root=REPO_ROOT, **spec)
    elif command == "run-test":
        locked = verify_lock(args.lock, repo_root=REPO_ROOT)
        if sha256_file(args.split_manifest) != locked["split_manifest"]["sha256"]:
            raise ValueError(
                "formal test split manifest differs from the frozen lock"
            )
        if args.dry_run:
            print(
                json.dumps(
                    {
                        "status": "dry_run",
                        "lock_verified": True,
                        "primary_method": locked["primary_method"],
                        "test_run_dir": args.test_run_dir,
                        "formal_claim_created": False,
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
            return
        if args.max_samples is not None:
            raise ValueError(
                "formal run-test requires the complete locked test cohort"
            )
        claim_test_once(
            args.test_run_dir, args.lock, resume=args.resume
        )
        test_root = Path(args.test_run_dir)
        primary = locked["configs"]["primary"]
        expected_test_inputs = {
            "frozen_features_sha256": args.frozen_features,
            "frozen_predictions_sha256": args.frozen_predictions,
        }
        for field, path in expected_test_inputs.items():
            expected_hash = primary.get(field)
            if expected_hash is None:
                raise ValueError(
                    f"frozen primary config lacks required {field}"
                )
            if sha256_file(path) != expected_hash:
                raise ValueError(
                    f"formal test input differs from locked {field}"
                )
        locked_device = str(primary.get("device", ""))
        if not locked_device:
            raise ValueError("frozen primary config lacks required device")
        if args.device != locked_device:
            raise ValueError(
                f"formal test device {args.device!r} != locked {locked_device!r}"
            )
        formal_seed = int(
            primary.get("inference_seed", locked["seeds"][0])
        )
        if args.seed != formal_seed:
            raise ValueError(
                f"formal test seed {args.seed} != locked seed {formal_seed}"
            )
        locked_batch_size = int(primary.get("batch_size", args.batch_size))
        if args.batch_size != locked_batch_size:
            raise ValueError(
                "formal test batch size differs from frozen manifest"
            )
        locked_workers = int(
            primary.get("num_workers", args.num_workers)
        )
        if args.num_workers != locked_workers:
            raise ValueError(
                "formal test worker count differs from frozen manifest"
            )
        enhanced = extract_enhanced_features(
            split="test",
            frozen_features_path=args.frozen_features,
            output_dir=test_root / "enhanced_test",
            split_manifest_path=args.split_manifest,
            device=args.device,
            batch_size=args.batch_size,
            workers=args.num_workers,
            crop_size=primary.get("crop_size", 32),
            roi_size=primary.get("roi_size", 5),
            checkpoint_path=locked["checkpoint"]["path"],
            max_samples=args.max_samples,
            resume=args.resume,
            seed=args.seed,
        )
        stability = run_stability_extraction(
            split="test",
            frozen_features_path=args.frozen_features,
            split_manifest_path=args.split_manifest,
            critic_model_paths=primary["critic_models"],
            output_dir=test_root / "stability_test",
            device=args.device,
            batch_size=args.batch_size,
            workers=args.num_workers,
            crop_size=primary.get("crop_size", 32),
            kappa=primary["uncertainty_kappa"],
            checkpoint_path=locked["checkpoint"]["path"],
            max_samples=args.max_samples,
            resume=args.resume,
            seed=args.seed,
        )
        samples = load_inference_features(args.frozen_features)
        if args.max_samples:
            samples = samples[: args.max_samples]
        arrays = load_enhanced_arrays(
            test_root / "enhanced_test",
            samples,
            include_labels=False,
        )
        prediction_summary = (
            test_root / "primary_predictions" / "summary.json"
        )
        predictions = (
            json.loads(prediction_summary.read_text(encoding="utf-8"))
            if args.resume and prediction_summary.exists()
            else predict_primary_ensemble(
                samples=samples,
                arrays=arrays,
                critic_models=primary["critic_models"],
                latent_models=primary["latent_models"],
                setrank_models=primary["setrank_models"],
                gate_models=primary["gate_models"],
                policy=primary["policy"],
                output_dir=test_root / "primary_predictions",
                device=args.device,
                alpha=primary["alpha"],
                uncertainty_kappa=primary["uncertainty_kappa"],
                required_consensus=primary["required_consensus"],
                candidate_probability_temperature=primary.get(
                    "candidate_probability_temperature", 1.0
                ),
                stability_path=(
                    test_root / "stability_test" / "stability.npz"
                ),
            )
        )
        scalar_gate = None
        scalar_config = locked["configs"].get("scalar_gate")
        if scalar_config:
            scalar_summary_path = (
                test_root / "scalar_gate_predictions" / "summary.json"
            )
            if args.resume and scalar_summary_path.exists():
                scalar_gate = json.loads(
                    scalar_summary_path.read_text(encoding="utf-8")
                )
            else:
                scalar_prediction_dir = (
                    test_root / "scalar_gate_predictions"
                )
                scalar_prediction_dir.mkdir(
                    parents=True, exist_ok=False
                )
                scalar_gate = predict_scalar_gate_ensemble(
                    samples=samples,
                    gate_models=scalar_config["gate_models"],
                    policy=scalar_config["policy"],
                    output_path=(
                        scalar_prediction_dir / "predictions.jsonl"
                    ),
                    device=args.device,
                )
                atomic_write_json(scalar_summary_path, scalar_gate)
        labels_dir = test_root / "labels"
        labels = {
            track: labels_dir / track / "labels.jsonl"
            for track in ("legacy_official", "corrected")
        }
        if not (
            args.resume
            and all(path.exists() for path in labels.values())
        ):
            labels = build_dual_labels(
                args.frozen_features,
                args.frozen_predictions,
                labels_dir,
                resume=args.resume,
            )
        method_predictions: dict[str, str | None] = {
            "q_only": None,
            **predictions["component_prediction_paths"],
            locked["primary_method"]: predictions["prediction_path"],
        }
        if scalar_gate:
            method_predictions["scalar_gate"] = scalar_gate[
                "prediction_path"
            ]
        evaluation_summaries = []
        method_evaluations = {}
        for method, prediction_path in method_predictions.items():
            evaluation_dir = test_root / "evaluations" / method
            evaluation_summary = evaluation_dir / "summary.json"
            method_evaluations[method] = (
                json.loads(
                    evaluation_summary.read_text(encoding="utf-8")
                )
                if args.resume and evaluation_summary.exists()
                else evaluate_method(
                    method=method,
                    features_path=args.frozen_features,
                    legacy_labels_path=labels["legacy_official"],
                    corrected_labels_path=labels["corrected"],
                    prediction_path=prediction_path,
                    output_dir=evaluation_dir,
                    expected_legacy_oracle=0.9087272522395628,
                    bootstrap_iterations=10_000,
                    bootstrap_seed=args.seed,
                )
            )
            evaluation_summaries.append(evaluation_summary)
        comparison_path = test_root / "comparison" / "method_results.json"
        comparison = (
            json.loads(comparison_path.read_text(encoding="utf-8"))
            if args.resume and comparison_path.exists()
            else combine_method_evaluations(
                evaluation_summaries,
                test_root / "comparison",
            )
        )
        evaluation = method_evaluations[locked["primary_method"]]
        independent_evaluations = {}
        for method, prediction_path in method_predictions.items():
            independent = recompute_counts(
                features=args.frozen_features,
                labels=labels["legacy_official"],
                predictions=prediction_path,
            )
            assert_matches_primary(
                method_evaluations[method]["legacy_official"],
                independent,
            )
            independent_evaluations[method] = independent
            atomic_write_json(
                test_root
                / "evaluations"
                / method
                / "independent_summary.json",
                independent,
            )
        independent = independent_evaluations[
            locked["primary_method"]
        ]
        result = {
            "enhanced": enhanced,
            "stability": stability,
            "predictions": predictions,
            "scalar_gate_predictions": scalar_gate,
            "evaluations": method_evaluations,
            "comparison": comparison,
            "primary_evaluation": evaluation,
            "independent_evaluations": independent_evaluations,
            "independent_primary_evaluation": independent,
        }
        atomic_write_json(
            test_root / "TEST_RUN_COMPLETE.json", result
        )
    elif command == "build-report":
        result = (
            {
                "report_source": args.results,
                "output": args.output,
                "dry_run": True,
            }
            if args.dry_run
            else build_report_artifact(
                json.loads(
                    Path(args.results).read_text(encoding="utf-8")
                ),
                args.output,
            )
        )
    elif command == "assemble-report":
        result = (
            {
                "validation_comparison": args.validation_comparison,
                "test_comparison": args.test_comparison,
                "dry_run": True,
            }
            if args.dry_run
            else assemble_results_bundle(
                validation_comparison=args.validation_comparison,
                test_comparison=args.test_comparison,
                primary_test_summary=args.primary_test_summary,
                primary_method=args.primary_method,
                output_path=args.output,
                vlm_status=args.vlm_status,
                calibration=args.calibration,
            )
        )
    elif command == "build-gallery":
        result = (
            {
                "features": args.features,
                "per_group": args.per_group,
                "dry_run": True,
            }
            if args.dry_run
            else build_failure_gallery(
                features_path=args.features,
                legacy_labels_path=args.legacy_labels,
                raw_predictions_path=args.raw_predictions,
                reranker_predictions_path=args.reranker_predictions,
                output_dir=args.output,
                per_group=args.per_group,
            )
        )
    else:
        raise AssertionError(command)
    print(json.dumps(result, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
