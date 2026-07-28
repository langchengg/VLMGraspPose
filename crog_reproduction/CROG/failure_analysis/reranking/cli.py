import argparse
import json
import sys
from pathlib import Path

from . import exporter
from .evaluate import evaluate_paths, write_evaluation
from .labels import build_label_record, regression_mismatches
from .rankers import RANKER_NAMES
from .schema import (
    canonical_json,
    completed_sample_ids,
    file_identity,
    make_run_manifest,
    read_jsonl,
    validate_run_manifest,
)
from .train_mlp import load_mlp_scorer, save_artifact, train_mlp


REPO_ROOT = Path(__file__).resolve().parents[2]


def _optional_file_identity(path):
    return file_identity(path) if path else None


def _load_provenance_weights(path):
    resolved = Path(path).resolve()
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    provenance = payload.get("provenance")
    weights = payload.get("weights")
    if not isinstance(provenance, dict) or provenance.get("source_split") != "val":
        raise ValueError(
            "--tuned-weights must declare provenance.source_split='val'"
        )
    if not isinstance(weights, dict):
        raise ValueError("--tuned-weights JSON must contain a weights object")
    return weights


def _top_level_help():
    parser = argparse.ArgumentParser(description="Frozen-candidate CROG post-hoc reranking")
    parser.add_argument(
        "command",
        nargs="?",
        choices=("build-features", "build-labels", "evaluate", "train-mlp"),
    )
    parser.print_help()


def _build_labels_parser():
    parser = argparse.ArgumentParser(
        prog="python -m failure_analysis.reranking.cli build-labels",
        description="Build physically separate official-evaluator labels from features and combined predictions.",
    )
    parser.add_argument("--features", required=True)
    parser.add_argument("--predictions", required=True, help="Combined prediction JSONL containing GT grasps.")
    parser.add_argument(
        "--regression-reference",
        help="Optional independent immutable prediction JSONL for old J@1/J@Any checks.",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--seed", type=int, default=17)
    return parser


def _build_labels(argv):
    args = _build_labels_parser().parse_args(argv)
    if args.resume and args.overwrite:
        raise ValueError("--resume and --overwrite are mutually exclusive")
    output = Path(args.output)
    run_manifest = make_run_manifest(
        "build-labels",
        REPO_ROOT,
        {
            "features": file_identity(args.features),
            "predictions": file_identity(args.predictions),
            "regression_reference": _optional_file_identity(args.regression_reference),
            "limit": args.limit,
            "device": args.device,
            "seed": args.seed,
        },
    )
    manifest_path = output.with_suffix(output.suffix + ".metadata.json")
    if args.resume and output.exists():
        if not manifest_path.exists():
            raise ValueError("resume refused: labels run metadata is missing")
        validate_run_manifest(
            json.loads(manifest_path.read_text(encoding="utf-8")), run_manifest
        )
    if output.exists() and not (args.resume or args.overwrite):
        raise FileExistsError("labels output exists; use --resume or explicit --overwrite")
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(run_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    predictions = {str(record["sample_id"]): record for record in read_jsonl(args.predictions)}
    regression_reference = (
        {
            str(record["sample_id"]): record
            for record in read_jsonl(args.regression_reference)
        }
        if args.regression_reference
        else None
    )
    completed = completed_sample_ids(output) if args.resume else set()
    mode = "a" if args.resume else "w" if args.overwrite else "x"
    records = []
    with output.open(mode, encoding="utf-8") as handle:
        for feature in read_jsonl(args.features, limit=args.limit):
            sample_id = str(feature["sample_id"])
            if sample_id in completed:
                continue
            prediction = predictions.get(sample_id)
            if prediction is None:
                raise ValueError(f"missing combined prediction/GT for sample {sample_id}")
            if regression_reference is not None and sample_id not in regression_reference:
                raise ValueError(f"missing independent regression reference for sample {sample_id}")
            record = build_label_record(
                feature,
                prediction.get("gt_grasps", []),
                old_record=(
                    regression_reference.get(sample_id)
                    if regression_reference is not None
                    else None
                ),
            )
            handle.write(canonical_json(record) + "\n")
            records.append(record)
    mismatches = regression_mismatches(records)
    result = {
        "output": str(output),
        "new_records": len(records),
        "regression_mismatch_sample_ids": mismatches,
    }
    print(json.dumps(result, indent=2, sort_keys=True))


def _evaluate_parser():
    parser = argparse.ArgumentParser(
        prog="python -m failure_analysis.reranking.cli evaluate",
        description="Evaluate a ranker without changing the frozen candidate set.",
    )
    parser.add_argument("--features", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--ranker", choices=RANKER_NAMES, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    parser.add_argument("--mlp-model")
    parser.add_argument("--tuned-weights", help="Validation-only non-negative weight JSON.")
    parser.add_argument("--bootstrap-iterations", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def _evaluate(argv):
    args = _evaluate_parser().parse_args(argv)
    if args.resume and args.overwrite:
        raise ValueError("--resume and --overwrite are mutually exclusive")
    output = Path(args.output)
    summary_path = output / "summary.json"
    run_manifest = make_run_manifest(
        "evaluate",
        REPO_ROOT,
        {
            "features": file_identity(args.features),
            "labels": file_identity(args.labels),
            "ranker": args.ranker,
            "limit": args.limit,
            "device": args.device,
            "mlp_model": _optional_file_identity(args.mlp_model),
            "tuned_weights": _optional_file_identity(args.tuned_weights),
            "bootstrap_iterations": args.bootstrap_iterations,
            "seed": args.seed,
        },
    )
    if args.resume and summary_path.exists():
        manifest_path = output / "run_manifest.json"
        required = [summary_path, output / "per_sample.jsonl", output / "case_index.json", manifest_path]
        if not all(path.exists() for path in required):
            raise ValueError("resume refused: evaluation output is incomplete")
        validate_run_manifest(
            json.loads(manifest_path.read_text(encoding="utf-8")), run_manifest
        )
        print(summary_path.read_text(encoding="utf-8"), end="")
        return
    scorer = None
    if args.ranker == "mlp":
        if not args.mlp_model:
            raise ValueError("--ranker mlp requires --mlp-model")
        scorer = load_mlp_scorer(args.mlp_model, device=args.device)
    tuned_weights = None
    if args.ranker == "rule_val_tuned":
        if not args.tuned_weights:
            raise ValueError("rule_val_tuned requires --tuned-weights from validation")
        tuned_weights = _load_provenance_weights(args.tuned_weights)
    summary, outcomes, categories = evaluate_paths(
        args.features,
        args.labels,
        ranker=args.ranker,
        limit=args.limit,
        tuned_weights=tuned_weights,
        mlp_scorer=scorer,
        seed=args.seed,
        bootstrap_iterations=args.bootstrap_iterations,
    )
    write_evaluation(
        output,
        summary,
        outcomes,
        categories,
        overwrite=args.overwrite,
        run_manifest=run_manifest,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


def _train_parser():
    parser = argparse.ArgumentParser(
        prog="python -m failure_analysis.reranking.cli train-mlp",
        description="Train the small allowlisted post-hoc MLP ranker; official test is rejected.",
    )
    parser.add_argument("--features", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--validation-features")
    parser.add_argument("--validation-labels")
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def _train(argv):
    args = _train_parser().parse_args(argv)
    if args.resume and args.overwrite:
        raise ValueError("--resume and --overwrite are mutually exclusive")
    output = Path(args.output)
    run_manifest = make_run_manifest(
        "train-mlp",
        REPO_ROOT,
        {
            "features": file_identity(args.features),
            "labels": file_identity(args.labels),
            "validation_features": _optional_file_identity(args.validation_features),
            "validation_labels": _optional_file_identity(args.validation_labels),
            "limit": args.limit,
            "device": args.device,
            "epochs": args.epochs,
            "patience": args.patience,
            "learning_rate": args.learning_rate,
            "validation_fraction": args.validation_fraction,
            "seed": args.seed,
        },
    )
    if args.resume and output.exists():
        manifest_path = output.with_suffix(".manifest.json")
        if not manifest_path.exists():
            raise ValueError("resume refused: MLP run manifest is missing")
        existing = json.loads(manifest_path.read_text(encoding="utf-8")).get("run")
        validate_run_manifest(existing, run_manifest)
        print(json.dumps({"output": str(output), "resumed": True}, indent=2))
        return
    features = list(read_jsonl(args.features, limit=args.limit))
    sample_ids = {str(record["sample_id"]) for record in features}
    labels = [record for record in read_jsonl(args.labels) if str(record["sample_id"]) in sample_ids]
    validation_features = (
        list(read_jsonl(args.validation_features, limit=args.limit))
        if args.validation_features
        else None
    )
    validation_labels = None
    if validation_features is not None:
        if not args.validation_labels:
            raise ValueError("--validation-features requires --validation-labels")
        validation_ids = {str(record["sample_id"]) for record in validation_features}
        validation_labels = [
            record
            for record in read_jsonl(args.validation_labels)
            if str(record["sample_id"]) in validation_ids
        ]
    artifact = train_mlp(
        features,
        labels,
        validation_feature_records=validation_features,
        validation_label_records=validation_labels,
        seed=args.seed,
        validation_fraction=args.validation_fraction,
        epochs=args.epochs,
        patience=args.patience,
        learning_rate=args.learning_rate,
        device=args.device,
    )
    artifact["run_manifest"] = run_manifest
    model_path, manifest_path = save_artifact(artifact, output, overwrite=args.overwrite)
    print(
        json.dumps(
            {
                "model": str(model_path),
                "manifest": str(manifest_path),
                "epochs_run": len(artifact["history"]),
                "calibration": artifact["calibration"],
            },
            indent=2,
            sort_keys=True,
        )
    )


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help"):
        _top_level_help()
        return
    command, rest = argv[0], argv[1:]
    if command == "build-features":
        exporter.main(rest)
    elif command == "build-labels":
        _build_labels(rest)
    elif command == "evaluate":
        _evaluate(rest)
    elif command == "train-mlp":
        _train(rest)
    else:
        raise SystemExit(f"unknown command: {command}")


if __name__ == "__main__":
    main()
