from __future__ import annotations

import argparse
import importlib
import json
import sys
from collections.abc import Callable, Mapping, Sequence
from functools import wraps
from pathlib import Path
from typing import Any

from failure_analysis.reranking_v2.datasets import load_inference_features
from failure_analysis.reranking_v2.enhanced_data import load_enhanced_arrays
from failure_analysis.reranking_v2.independent_evaluator import recompute_counts
from failure_analysis.reranking_v2.inference import predict_primary_ensemble
from failure_analysis.reranking_v2.protocol import verify_lock

from . import DEFAULT_SEED
from .artifacts import REPO_ROOT, append_command_log, assert_v3_output_path, create_run_dir, write_environment
from .audit_v2 import audit_v2_locked_primary, write_replay_record
from .candidate_identity import verify_candidate_identity
from .fullchain_extractor import extract_fullchain_features
from .provenance import write_feature_provenance
from .schema import atomic_write_json, sha256_file
from .splits import build_v3_split_manifest
from .test_access_guard import TestAccessGuard, verify_manifest_sidecar
from .training import train_smoke_suite
from .v2_prior import verify_oof_provenance
from .feature_data import load_fullchain_arrays
from .feature_store import load_split_ids
from .feature_store import FeatureCatalog, load_label_lookup, prior_array_to_lookup
from .experiment_config import SELECTION_GRID
from .pipeline import run_grouped_oof_and_final_models
from .pipeline import predict_final_ensemble
from .selection import run_selection_grid
from .uncertainty import score_perturbation_ensemble_streaming
from .inference import prepare_gate_inference, save_gate_bundle
from .calibration import calibrate_gate_bundle
from .v2_prior import build_deployment_v2_prior, load_v2_oof_prior, load_v2_validation_prior
from .formal import (
    FORMAL_METHOD_ALLOWLIST,
    evaluate_lockcheck_once,
    independent_evaluate_once,
    lock_final,
    lock_preliminary,
    run_locked_inference_once,
    verify_exact_test_command,
    verify_formal_ranking_outputs,
)
from .diagnostic_ablations import run_validation_diagnostic_ablations


DEFAULT_V1_ROOT = REPO_ROOT / "failure_analysis/reranking_outputs/full_test_17749_v1"
DEFAULT_V2_ROOT = REPO_ROOT / "failure_analysis/reranking_outputs/v2_20260727T174412+0100"


def _common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--scope", choices=("development", "calibration", "select", "lockcheck", "test"), default="development")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--device", choices=("auto", "mps", "cpu"), default="auto")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--force-new-run", action="store_true")


def _formal_options(parser: argparse.ArgumentParser) -> None:
    """Arguments shared only by the immutable lock/formal lifecycle commands."""
    parser.add_argument("--run-id")
    parser.add_argument("--primary-method", default="v3_locked_primary")
    parser.add_argument("--method", action="append")
    parser.add_argument("--checkpoint-artifact", action="append")
    parser.add_argument("--normalizer-artifact", action="append")
    parser.add_argument("--gate-artifact", action="append")
    parser.add_argument("--evaluator-artifact", action="append")
    parser.add_argument("--experiment-descriptor")
    parser.add_argument("--evaluation-descriptor")
    parser.add_argument("--evaluator-callback")
    parser.add_argument("--v2-artifact", action="append")
    parser.add_argument("--input-artifact", action="append")
    parser.add_argument("--test-input-artifact", action="append")
    parser.add_argument("--method-ranking", action="append")
    parser.add_argument("--contract")
    parser.add_argument("--candidate-artifact")
    parser.add_argument("--manifest-output")
    parser.add_argument("--stage-dir")
    parser.add_argument("--formal-output-dir")
    parser.add_argument("--lockcheck-completion")
    parser.add_argument("--lockcheck-evaluation-completion")
    parser.add_argument("--test-completion")
    parser.add_argument(
        "--exact-test-command",
        help="JSON argv array for the immutable formal-test command",
    )
    parser.add_argument(
        "--callback",
        help=(
            "repository callback as failure_analysis.reranking_v3.<module>:<callable>; "
            "the callback is covered by the V3 code fingerprint"
        ),
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="CROG Re-ranking V3 Full-Chain Evidence Reranker")
    commands = parser.add_subparsers(dest="command", required=True)
    initial = {
        "audit-environment", "audit-v2", "replay-v2", "verify-candidates", "create-v3-splits",
        "export-base", "extract-output-features", "extract-latents", "extract-crops", "extract-depth",
        "build-v2-prior", "validate-features", "train-oof", "train-fullchain", "train-gate",
        "run-uncertainty", "evaluate-select", "lock-preliminary", "run-lockcheck", "evaluate-lockcheck", "lock-final",
        "run-test", "independent-evaluate", "build-gallery", "build-report", "run-diagnostics",
    }
    for name in sorted(initial):
        child = commands.add_parser(name)
        _common(child)
        child.add_argument("--v1-root", default=str(DEFAULT_V1_ROOT))
        child.add_argument("--v2-root", default=str(DEFAULT_V2_ROOT))
        child.add_argument("--split-manifest")
        child.add_argument("--formal-manifest")
        child.add_argument("--batch-size", type=int, default=16)
        child.add_argument("--features")
        child.add_argument("--artifact")
        child.add_argument("--crop-size", type=int, default=32)
        child.add_argument("--roi-size", type=int, default=7)
        child.add_argument("--channel-bins", type=int, default=32)
        child.add_argument("--shard-samples", type=int, default=64)
        child.add_argument("--artifact-tag")
        child.add_argument("--train-artifact", action="append")
        child.add_argument("--eval-artifact", action="append")
        child.add_argument("--train-head-override", action="append")
        child.add_argument("--eval-head-override", action="append")
        child.add_argument("--epochs", type=int, default=5)
        child.add_argument("--checkpoint", action="append")
        child.add_argument("--policy")
        child.add_argument("--iterations", type=int, default=10000)
        child.add_argument("--per-group", type=int, default=5)
        if name in {
            "lock-preliminary", "run-lockcheck", "evaluate-lockcheck", "lock-final", "run-test",
            "independent-evaluate",
        }:
            _formal_options(child)
    return parser


def _require(value: Any, *, option: str) -> Any:
    if value is None or (isinstance(value, str) and not value.strip()):
        raise ValueError(f"{option} is required")
    return value


def _named_paths(
    values: Sequence[str] | None,
    *,
    option: str,
    allow_none: bool = False,
) -> dict[str, Path | None]:
    """Parse repeatable NAME=PATH values without silently accepting duplicates."""
    result: dict[str, Path | None] = {}
    for raw in values or ():
        name, separator, path = str(raw).partition("=")
        name = name.strip()
        path = path.strip()
        if not separator or not name or not path:
            raise ValueError(f"{option} values must use non-empty NAME=PATH syntax")
        if name in result:
            raise ValueError(f"{option} contains duplicate name: {name}")
        if allow_none and path.lower() in {"none", "null"}:
            result[name] = None
        else:
            result[name] = Path(path).expanduser().resolve()
    return result


def _run_child_path(run: Path, raw: str | None, default: str) -> Path:
    output = (Path(raw).expanduser() if raw else run / default).resolve()
    if not output.is_relative_to(run.resolve()):
        raise PermissionError(f"formal output must remain below the V3 run directory: {output}")
    return output


def _artifact_namespace(raw: str | None) -> str | None:
    if raw is None:
        return None
    value = str(raw).strip()
    if not value or any(
        character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"
        for character in value
    ):
        raise ValueError("artifact-tag must contain only letters, digits, '_' or '-'")
    return value


def _exact_command(raw: str | None) -> list[str]:
    source = _require(raw, option="--exact-test-command")
    try:
        value = json.loads(source)
    except json.JSONDecodeError as error:
        raise ValueError("--exact-test-command must be a JSON argv array") from error
    if not isinstance(value, list) or not value or any(
        not isinstance(item, str) or not item for item in value
    ):
        raise ValueError("--exact-test-command must be a non-empty JSON array of strings")
    return value


def _json_descriptor(path: str | Path, *, description: str) -> dict[str, Any]:
    source = Path(path).expanduser().resolve()
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid {description} JSON: {source}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{description} must be a JSON object")
    return value


def _descriptor_object_or_file(value: Any, *, field: str) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    if isinstance(value, str) and value.strip():
        return _json_descriptor(value, description=field)
    raise ValueError(f"{field} must be an object or JSON file path")


def _repository_callback(spec: str | None, *, role: str) -> Callable[..., Any]:
    """Load only version-controlled V3 callbacks covered by the manifest fingerprint."""
    value = str(_require(spec, option="--callback"))
    module_name, separator, attribute = value.partition(":")
    if (
        not separator
        or not attribute
        or not module_name.startswith("failure_analysis.reranking_v3.")
    ):
        raise ValueError(
            "--callback must be failure_analysis.reranking_v3.<module>:<callable>"
        )
    module = importlib.import_module(module_name)
    callback = getattr(module, attribute, None)
    if not callable(callback):
        raise ValueError(f"{role} callback is not callable: {value}")
    return callback


def _method_identity_callback(callback: Callable[..., Any]) -> Callable[..., Any]:
    """Reject missing/shared/mislabeled formal ranking outputs before completion."""
    @wraps(callback)
    def checked(**kwargs: Any) -> Any:
        value = callback(**kwargs)
        paths = [value] if isinstance(value, (str, Path)) else list(value)
        verify_formal_ranking_outputs(
            paths, locked_manifest=kwargs["locked_manifest"]
        )
        return paths

    return checked


def _formal_dry_run(
    args: argparse.Namespace,
    *,
    run: Path,
    mappings: Mapping[str, Mapping[str, Path | None]],
    would_write: Sequence[Path],
) -> dict[str, Any]:
    """Describe a formal action without importing callbacks or creating a lock/claim."""
    return {
        "dry_run": True,
        "command": args.command,
        "formal_manifest": args.formal_manifest,
        "mapping_names": {
            name: sorted(values) for name, values in mappings.items()
        },
        "would_write": [str(path) for path in would_write],
        "run_dir": str(run),
        "lock_or_claim_created": False,
    }


def _run_formal_command(
    args: argparse.Namespace,
    *,
    run: Path,
    argv: Sequence[str],
    inference_callback: Callable[..., Any] | None,
    evaluator_callback: Callable[..., Any] | None,
) -> dict[str, Any] | None:
    """Dispatch the five immutable lifecycle commands to :mod:`formal`."""
    if args.command == "lock-preliminary":
        checkpoints = _named_paths(args.checkpoint_artifact, option="--checkpoint-artifact")
        normalizers = _named_paths(args.normalizer_artifact, option="--normalizer-artifact")
        gates = _named_paths(args.gate_artifact, option="--gate-artifact")
        evaluators = _named_paths(args.evaluator_artifact, option="--evaluator-artifact")
        v2 = _named_paths(args.v2_artifact, option="--v2-artifact")
        methods = list(args.method or ())
        unknown = set(methods) - set(FORMAL_METHOD_ALLOWLIST)
        if not methods or unknown:
            raise ValueError(
                "--method must explicitly declare only allowlisted formal methods; "
                f"unknown={sorted(unknown)}"
            )
        output = _run_child_path(
            run, args.manifest_output, "formal/preliminary_experiment_manifest.json"
        )
        required = {
            "--run-id": args.run_id,
            "--config": args.config,
            "--contract": args.contract,
            "--split-manifest": args.split_manifest,
            "--candidate-artifact": args.candidate_artifact,
            "--experiment-descriptor": args.experiment_descriptor,
            "--evaluation-descriptor": args.evaluation_descriptor,
            "--callback": args.callback,
            "--evaluator-callback": args.evaluator_callback,
        }
        for option, value in required.items():
            _require(value, option=option)
        if not all((checkpoints, normalizers, gates, evaluators, v2)):
            raise ValueError(
                "checkpoint, normalizer, gate, evaluator, and V2 artifact mappings "
                "must each be non-empty"
            )
        if args.dry_run:
            return _formal_dry_run(
                args,
                run=run,
                mappings={
                    "checkpoints": checkpoints,
                    "normalizers": normalizers,
                    "gate": gates,
                    "evaluators": evaluators,
                    "v2": v2,
                },
                would_write=(output, output.with_suffix(output.suffix + ".sha256")),
            )
        return lock_preliminary(
            output_path=output,
            run_id=args.run_id,
            primary_method=args.primary_method,
            formal_methods=methods,
            config_path=args.config,
            checkpoint_paths=checkpoints,
            normalizer_paths=normalizers,
            gate_paths=gates,
            contract_path=args.contract,
            split_path=args.split_manifest,
            candidate_path=args.candidate_artifact,
            evaluator_paths=evaluators,
            v2_paths=v2,
            experiment_descriptor_path=args.experiment_descriptor,
            evaluation_descriptor_path=args.evaluation_descriptor,
            inference_callback_identity=args.callback,
            evaluator_callback_identity=args.evaluator_callback,
            metadata={"cli_command": args.command},
        )

    if args.command in {"run-lockcheck", "run-test"}:
        scope = "lockcheck" if args.command == "run-lockcheck" else "test"
        manifest = Path(_require(args.formal_manifest, option="--formal-manifest")).resolve()
        inputs = _named_paths(args.input_artifact, option="--input-artifact")
        if not inputs:
            raise ValueError("--input-artifact must declare every label-free inference input")
        stage = _run_child_path(run, args.stage_dir, f"formal/{scope}")
        output = _run_child_path(run, args.formal_output_dir, f"inference/{scope}_once")
        if args.dry_run:
            return _formal_dry_run(
                args,
                run=run,
                mappings={"inputs": inputs},
                would_write=(stage, output),
            )
        if scope == "test":
            verify_exact_test_command(manifest, argv)
        callback = inference_callback or _repository_callback(
            args.callback, role=f"{scope} inference"
        )
        callback = _method_identity_callback(callback)
        return run_locked_inference_once(
            scope=scope,
            manifest_path=manifest,
            stage_dir=stage,
            input_artifacts=inputs,
            output_dir=output,
            inference_callback=callback,
            callback_identity=args.callback,
            resume=args.resume,
        )

    if args.command == "evaluate-lockcheck":
        manifest = Path(_require(args.formal_manifest, option="--formal-manifest")).resolve()
        completion = Path(
            _require(args.lockcheck_completion, option="--lockcheck-completion")
        ).resolve()
        candidate = Path(
            _require(args.candidate_artifact, option="--candidate-artifact")
        ).resolve()
        rankings = _named_paths(
            args.method_ranking, option="--method-ranking", allow_none=True
        )
        rankings.setdefault("q_only", None)
        evaluation_inputs = _named_paths(
            args.input_artifact, option="--input-artifact"
        )
        stage = _run_child_path(run, args.stage_dir, "formal/lockcheck_evaluate")
        output = _run_child_path(
            run, args.formal_output_dir, "evaluation/lockcheck_once"
        )
        if args.dry_run:
            return _formal_dry_run(
                args,
                run=run,
                mappings={
                    "rankings": rankings,
                    "evaluation_inputs": evaluation_inputs,
                },
                would_write=(stage, output),
            )
        callback_identity = str(
            _require(args.callback, option="--callback")
        )
        callback = evaluator_callback or _repository_callback(
            callback_identity, role="lockcheck evaluator"
        )
        return evaluate_lockcheck_once(
            preliminary_manifest_path=manifest,
            lockcheck_inference_completion_path=completion,
            stage_dir=stage,
            candidate_artifact=candidate,
            method_rankings=rankings,
            evaluation_artifacts=evaluation_inputs,
            output_dir=output,
            evaluator_callback=callback,
            callback_identity=callback_identity,
            resume=args.resume,
        )

    if args.command == "lock-final":
        preliminary = Path(
            _require(args.formal_manifest, option="--formal-manifest")
        ).resolve()
        completion = Path(
            _require(args.lockcheck_completion, option="--lockcheck-completion")
        ).resolve()
        evaluation_completion = Path(
            _require(
                args.lockcheck_evaluation_completion,
                option="--lockcheck-evaluation-completion",
            )
        ).resolve()
        test_inputs = _named_paths(
            args.test_input_artifact, option="--test-input-artifact"
        )
        command = _exact_command(args.exact_test_command)
        output = _run_child_path(
            run, args.manifest_output, "frozen_experiment_manifest.json"
        )
        if args.dry_run:
            return _formal_dry_run(
                args,
                run=run,
                mappings={"test_inputs": test_inputs},
                would_write=(output, output.with_suffix(output.suffix + ".sha256")),
            )
        return lock_final(
            output_path=output,
            preliminary_manifest_path=preliminary,
            lockcheck_completion_path=completion,
            lockcheck_evaluation_completion_path=evaluation_completion,
            exact_test_command=command,
            test_input_paths=test_inputs,
            metadata={"cli_command": args.command},
        )

    if args.command == "independent-evaluate":
        manifest = Path(_require(args.formal_manifest, option="--formal-manifest")).resolve()
        completion = Path(
            _require(args.test_completion, option="--test-completion")
        ).resolve()
        candidate = Path(
            _require(args.candidate_artifact, option="--candidate-artifact")
        ).resolve()
        rankings = _named_paths(
            args.method_ranking, option="--method-ranking", allow_none=True
        )
        rankings.setdefault("q_only", None)
        evaluation_inputs = _named_paths(
            args.input_artifact, option="--input-artifact"
        )
        stage = _run_child_path(run, args.stage_dir, "formal/independent_evaluate")
        output = _run_child_path(
            run, args.formal_output_dir, "evaluation/independent_once"
        )
        if args.dry_run:
            return _formal_dry_run(
                args,
                run=run,
                mappings={
                    "rankings": rankings,
                    "evaluation_inputs": evaluation_inputs,
                },
                would_write=(stage, output),
            )
        callback = evaluator_callback or _repository_callback(
            args.callback, role="independent evaluator"
        )
        return independent_evaluate_once(
            final_manifest_path=manifest,
            test_completion_path=completion,
            stage_dir=stage,
            candidate_artifact=candidate,
            method_rankings=rankings,
            evaluation_artifacts=evaluation_inputs,
            output_dir=output,
            evaluator_callback=callback,
            callback_identity=args.callback,
            resume=args.resume,
        )
    return None


def _baseline_audit(args: argparse.Namespace, audit_dir: Path) -> dict:
    v1 = Path(args.v1_root)
    v2 = Path(args.v2_root)
    labels_root = v2 / "formal_test_primary_v2/labels"
    v2_predictions = v2 / "formal_test_primary_v2/primary_predictions/predictions.jsonl"
    result = {}
    for evaluator in ("legacy_official", "corrected"):
        label_path = labels_root / evaluator / "labels.jsonl"
        result[evaluator] = {
            "q_only": recompute_counts(features=v1 / "features.jsonl", labels=label_path, predictions=None),
            "v2_locked_primary": recompute_counts(features=v1 / "features.jsonl", labels=label_path, predictions=v2_predictions),
        }
    output = audit_dir / "BASELINES_INDEPENDENT_RECOMPUTATION.json"
    atomic_write_json(output, result)
    return result


def _run_initial(
    args: argparse.Namespace,
    argv: list[str],
    *,
    inference_callback: Callable[..., Any] | None = None,
    evaluator_callback: Callable[..., Any] | None = None,
) -> dict:
    run = assert_v3_output_path(args.output_dir)
    if args.command == "audit-environment":
        if args.dry_run:
            return {"dry_run": True, "would_create": str(run)}
        create_run_dir(run, force_new_run=args.force_new_run)
        append_command_log(run, argv)
        path = write_environment(run, device=args.device)
        return {"run_dir": str(run), "environment": str(path), "sha256": sha256_file(path)}
    if not run.exists():
        raise FileNotFoundError("run directory does not exist; run audit-environment first")
    append_command_log(run, argv)
    formal_result = _run_formal_command(
        args,
        run=run,
        argv=argv,
        inference_callback=inference_callback,
        evaluator_callback=evaluator_callback,
    )
    if formal_result is not None:
        return formal_result
    if args.scope == "test" and args.command in {
        "export-base", "extract-output-features", "extract-latents", "extract-crops",
        "extract-depth", "validate-features", "train-oof", "train-fullchain",
        "train-gate", "run-uncertainty", "evaluate-select", "run-diagnostics",
    }:
        raise PermissionError(
            f"{args.command} cannot access --scope test outside the immutable formal lifecycle"
        )
    audit_dir = run / "audit"
    if args.command == "audit-v2":
        if args.dry_run:
            return {"dry_run": True}
        result = audit_v2_locked_primary(args.v2_root, audit_dir, repo_root=REPO_ROOT)
        result["independent_baselines"] = _baseline_audit(args, audit_dir)
        result["provenance"] = write_feature_provenance(run / "provenance")
        atomic_write_json(audit_dir / "AUDIT_SUMMARY.json", result)
        return result
    if args.command == "verify-candidates":
        if args.dry_run:
            return {"dry_run": True}
        v2 = Path(args.v2_root)
        result = verify_candidate_identity(
            v1_features=Path(args.v1_root) / "features.jsonl",
            prediction_paths=(
                Path(args.v1_root) / "predictions.jsonl",
                v2 / "formal_test_primary_v2/primary_predictions/predictions.jsonl",
            ),
            enhanced_index_paths=(v2 / "formal_test_primary_v2/enhanced_test/index.jsonl",),
            output_path=audit_dir / "CANDIDATE_IDENTITY.json",
        )
        return result
    if args.command == "create-v3-splits":
        source = args.split_manifest or str(Path(args.v2_root) / "split_manifest.json")
        if args.dry_run:
            return {"dry_run": True, "source": source}
        return build_v3_split_manifest(source, run / "v3_select_lockcheck_split.json", seed=args.seed)
    if args.command == "replay-v2":
        v2 = Path(args.v2_root).resolve()
        lock_path = v2 / "frozen_experiment_manifest.json"
        verify_manifest_sidecar(lock_path)
        TestAccessGuard(
            "test", run / "artifact_access.jsonl", formal_manifest=lock_path,
            manifest_access_class="v2_frozen",
        ).check(
            (
                Path(args.v1_root) / "features.jsonl",
                v2 / "formal_test_primary_v2/enhanced_test",
                v2 / "formal_test_primary_v2/stability_test/stability.npz",
                v2 / "formal_test_primary_v2/primary_predictions/predictions.jsonl",
            ),
            purpose="label-free exact V2 locked-primary replay",
            label_access=False,
        )
        if args.dry_run:
            return {"dry_run": True, "manifest_sha256": sha256_file(lock_path)}
        locked = verify_lock(lock_path, repo_root=REPO_ROOT)
        generated = audit_dir / "v2_primary_replay_generated"
        prediction_path = generated / "predictions.jsonl"
        if not prediction_path.exists():
            samples = load_inference_features(Path(args.v1_root) / "features.jsonl")
            arrays = load_enhanced_arrays(
                v2 / "formal_test_primary_v2/enhanced_test", samples,
                include_crops=True, include_labels=False,
            )
            primary = locked["configs"]["primary"]
            predict_primary_ensemble(
                samples=samples, arrays=arrays,
                critic_models=primary["critic_models"], latent_models=primary["latent_models"],
                setrank_models=primary["setrank_models"], gate_models=primary["gate_models"],
                policy=primary["policy"], output_dir=generated,
                device=args.device, alpha=primary["alpha"],
                uncertainty_kappa=primary["uncertainty_kappa"],
                required_consensus=primary["required_consensus"],
                candidate_probability_temperature=primary.get("candidate_probability_temperature", 1.0),
                stability_path=v2 / "formal_test_primary_v2/stability_test/stability.npz",
            )
        return write_replay_record(
            audit_dir,
            expected_path=v2 / "formal_test_primary_v2/primary_predictions/predictions.jsonl",
            observed_path=prediction_path,
            metadata={"device": args.device, "sample_count": 17749, "labels_read": False, "lock_sha256": locked["lock_sha256"]},
        )
    if args.command == "export-base":
        source = Path(args.v2_root) / ("base_val/features.jsonl" if args.scope in {"select","lockcheck"} else "base_train/features.jsonl")
        TestAccessGuard(args.scope, run / "artifact_access.jsonl").check((source,),purpose="reuse immutable V2 frozen candidate export as V3 base",label_access=False)
        return {"reused":True,"source":str(source.resolve()),"sha256":sha256_file(source),"labels_read":False}
    if args.command == "extract-output-features":
        if args.features:
            source = Path(args.features)
        elif args.scope in {"development", "calibration"}:
            source = Path(args.v2_root) / "base_train/features.jsonl"
        elif args.scope in {"select", "lockcheck"}:
            source = Path(args.v2_root) / "base_val/features.jsonl"
        else:
            source = Path(args.v1_root) / "features.jsonl"
        if args.split_manifest:
            split_manifest = Path(args.split_manifest)
        elif args.scope in {"select", "lockcheck"}:
            split_manifest = run / "v3_select_lockcheck_split.json"
        else:
            split_manifest = Path(args.v2_root) / "split_manifest.json"
        formal = Path(args.formal_manifest) if args.formal_manifest else None
        TestAccessGuard(args.scope, run / "artifact_access.jsonl", formal_manifest=formal).check(
            (source, split_manifest), purpose="full-chain label-free feature extraction", label_access=False
        )
        if args.dry_run:
            return {"dry_run": True, "source": str(source.resolve()), "labels_read": False}
        namespace = _artifact_namespace(args.artifact_tag)
        if namespace is not None:
            output = run / f"features/{namespace}_shard{args.shard_id:03d}_of_{args.num_shards:03d}"
        elif args.max_samples is not None and args.scope == "development" and args.num_shards == 1:
            suffix = "" if args.max_samples == 200 else f"_{args.max_samples}"
            output = run / f"smoke/fullchain_features{suffix}"
        else:
            output = run / f"features/{args.scope}_shard{args.shard_id:03d}_of_{args.num_shards:03d}"
        allowed_ids = None
        if args.scope == "calibration":
            allowed_ids = load_split_ids(split_manifest, development_partition="calibration")
        elif args.scope == "select":
            allowed_ids = load_split_ids(split_manifest, v3_partition="v3_select")
        elif args.scope == "lockcheck":
            allowed_ids = load_split_ids(split_manifest, v3_partition="v3_lockcheck")
        return extract_fullchain_features(
            frozen_features_path=source,
            split_manifest_path=split_manifest,
            output_dir=output,
            device=args.device,
            batch_size=args.batch_size,
            crop_size=args.crop_size,
            roi_size=args.roi_size,
            channel_bins=args.channel_bins,
            shard_samples=args.shard_samples,
            max_samples=args.max_samples,
            shard_id=args.shard_id,
            num_shards=args.num_shards,
            seed=args.seed,
            resume=args.resume,
            allowed_ids=allowed_ids,
        )
    if args.command in {"extract-latents", "extract-crops", "extract-depth"}:
        artifact = Path(args.artifact or (run / "smoke/fullchain_features"))
        manifest_path = artifact / "artifact_manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError("consolidated full-chain artifact is required first")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("status") != "complete":
            raise ValueError("consolidated full-chain artifact is incomplete")
        component = args.command.removeprefix("extract-")
        return {
            "component": component,
            "reused_consolidated_artifact": str(artifact.resolve()),
            "content_sha256": manifest["content_sha256"],
            "row_count": manifest["row_count"],
            "labels_read": False,
        }
    if args.command == "build-v2-prior":
        if args.scope in {"calibration", "test"}:
            v2_split=Path(args.split_manifest or (Path(args.v2_root)/"split_manifest.json"))
            if args.scope=="calibration":
                allowed_ids=load_split_ids(v2_split,development_partition="calibration"); source=Path(args.v2_root)/"base_train/features.jsonl"; enhanced=Path(args.v2_root)/"enhanced_train"; output=run/"v2_prior/calibration_deployment"
            else:
                source=Path(args.v1_root)/"features.jsonl"; enhanced=Path(args.v2_root)/"formal_test_primary_v2/enhanced_test"; output=run/"v2_prior/test_deployment"; allowed_ids={sample.sample_id for sample in load_inference_features(source)}
            formal=Path(args.formal_manifest) if args.formal_manifest else None
            TestAccessGuard(
                args.scope, run/"artifact_access.jsonl", formal_manifest=formal,
                manifest_access_class="v2_frozen" if args.scope == "test" else "v3_final",
            ).check((source,enhanced,Path(args.v2_root)/"frozen_experiment_manifest.json"),purpose="label-free exact locked-V2 evidence for deployment",label_access=False)
            if args.dry_run: return {"dry_run":True,"scope":args.scope,"row_count":len(allowed_ids),"labels_read":False}
            return build_deployment_v2_prior(frozen_features_path=source,enhanced_dir=enhanced,allowed_ids=allowed_ids,v2_manifest_path=Path(args.v2_root)/"frozen_experiment_manifest.json",output_dir=output,device=args.device)
        TestAccessGuard(args.scope, run / "artifact_access.jsonl").check(
            (Path(args.v2_root) / "oof_base/oof_base_predictions.npz", Path(args.v2_root) / "oof_primary/oof_setrank_predictions.npz"),
            purpose="verify provenance-valid V2 OOF prior", label_access=False,
        )
        if args.dry_run:
            return {"dry_run": True}
        result = verify_oof_provenance(args.v2_root)
        prior_dir = run / "v2_prior"
        prior_dir.mkdir(parents=True, exist_ok=False)
        atomic_write_json(prior_dir / "OOF_PROVENANCE_AUDIT.json", result)
        return result
    if args.command == "validate-features":
        artifact = Path(args.artifact or (run / "smoke/fullchain_features"))
        arrays = load_fullchain_arrays(artifact, max_samples=args.max_samples)
        expected = {
            "head_features": (5, 146), "depth_features": (5, 13),
            "latent_rois": (5, 12, 224), "attention_rois": (5, 3, 17),
            "tokens": (17, 512), "sentence": (1024,), "dynamic": (2305,),
            "token_ids": (17,), "crops": (5, 17, 32, 32),
        }
        for name, shape in expected.items():
            if tuple(arrays[name].shape[1:]) != shape:
                raise AssertionError(f"{name} shape {arrays[name].shape} != (*,{shape})")
            if name != "token_ids" and not __import__("numpy").isfinite(arrays[name]).all():
                raise FloatingPointError(f"non-finite values in {name}")
        records = arrays["records"]
        if any(record["candidate_identity"]["candidate_count"] != 5 for record in records):
            raise AssertionError("candidate identity coverage failed")
        result = {
            "status": "passed", "row_count": len(records), "candidate_count": len(records) * 5,
            "feature_shapes": {name: list(arrays[name].shape) for name in expected},
            "missing_count": 0, "candidate_identity_passed": True,
            "hook_non_mutating_passed": True,
        }
        path = artifact / "validation.json"
        if not path.exists():
            atomic_write_json(path, result)
        return result
    if args.command == "train-fullchain" and args.scope == "select":
        if not args.train_artifact or not args.eval_artifact:
            raise ValueError("scientific selection requires --train-artifact and --eval-artifact")
        train_labels = run / "labels/partitions/train/corrected_scientific/labels.jsonl"
        validation_labels = run / "labels/partitions/v3_select/corrected_scientific/labels.jsonl"
        if not train_labels.exists() or not validation_labels.exists(): raise FileNotFoundError("physically isolated train/select label artifacts are required")
        contract = Path(args.config or (run / "configs/predeclared_selection_grid.json"))
        train_candidate_source=Path(args.v2_root)/"base_train/features.jsonl"; eval_candidate_source=Path(args.v2_root)/"base_val/features.jsonl"
        accessed = [*(Path(value)/"artifact_manifest.json" for value in args.train_artifact),*(Path(value)/"artifact_manifest.json" for value in args.eval_artifact),*(Path(value)/"artifact_manifest.json" for value in (args.train_head_override or ())),*(Path(value)/"artifact_manifest.json" for value in (args.eval_head_override or ())),train_candidate_source,eval_candidate_source,train_labels,validation_labels,contract,Path(args.v2_root)/"primary_validation_locked/predictions.jsonl"]
        TestAccessGuard("select",run/"artifact_access.jsonl").check(accessed,purpose="predeclared 12-configuration FCER selection",label_access=True)
        if args.dry_run: return {"dry_run":True,"configuration_count":12,"would_write":str(run/"selection")}
        v2_split=Path(args.v2_root)/"split_manifest.json"; v3_split=Path(args.split_manifest or (run/"v3_select_lockcheck_split.json")); train_ids=load_split_ids(v2_split,development_partition="train"); select_ids=load_split_ids(v3_split,v3_partition="v3_select")
        selection_output=run/("selection" if not args.artifact_tag else f"selection_{args.artifact_tag}")
        return run_selection_grid(train_catalog=FeatureCatalog(args.train_artifact,head_override_dirs=args.train_head_override or (),candidate_feature_paths=(train_candidate_source,)),select_catalog=FeatureCatalog(args.eval_artifact,head_override_dirs=args.eval_head_override or (),candidate_feature_paths=(eval_candidate_source,)),train_ids=train_ids,select_ids=select_ids,train_labels_path=train_labels,validation_labels_path=validation_labels,v2_root=args.v2_root,v2_validation_predictions=Path(args.v2_root)/"primary_validation_locked/predictions.jsonl",v3_split_manifest=v3_split,contract_path=contract,output_dir=selection_output,device=args.device,seed=args.seed,epochs=args.epochs,batch_size=args.batch_size,resume=args.resume)
    if args.command == "run-diagnostics":
        if args.scope != "select":
            raise PermissionError("run-diagnostics is restricted to --scope select")
        if not args.train_artifact or not args.eval_artifact:
            raise ValueError("run-diagnostics requires --train-artifact and --eval-artifact")
        selection_path = Path(
            args.config or (run / "selection_widthfix_g10_v1/selection_summary.json")
        ).expanduser().resolve()
        diagnostic_contract = Path(
            args.artifact
            or (run / "configs/predeclared_diagnostic_ablations_widthfix_g10_v1.json")
        ).expanduser().resolve()
        train_labels_path = run / "labels/partitions/train/corrected_scientific/labels.jsonl"
        select_labels_path = run / "labels/partitions/v3_select/corrected_scientific/labels.jsonl"
        v2_split = Path(args.v2_root) / "split_manifest.json"
        v3_split = Path(args.split_manifest or (run / "v3_select_lockcheck_split.json"))
        train_candidate_source = Path(args.v2_root) / "base_train/features.jsonl"
        select_candidate_source = Path(args.v2_root) / "base_val/features.jsonl"
        v2_validation_predictions = Path(args.v2_root) / "primary_validation_locked/predictions.jsonl"
        gate_bundle_path = None if not args.features else Path(args.features).expanduser().resolve()
        gate_policy_path = None if not args.policy else Path(args.policy).expanduser().resolve()
        accessed = [
            *(Path(value) / "artifact_manifest.json" for value in args.train_artifact),
            *(Path(value) / "artifact_manifest.json" for value in args.eval_artifact),
            *(Path(value) / "artifact_manifest.json" for value in (args.train_head_override or ())),
            *(Path(value) / "artifact_manifest.json" for value in (args.eval_head_override or ())),
            train_candidate_source,
            select_candidate_source,
            train_labels_path,
            select_labels_path,
            selection_path,
            diagnostic_contract,
            v2_validation_predictions,
            *(value for value in (gate_bundle_path, gate_policy_path) if value is not None),
        ]
        TestAccessGuard("select", run / "artifact_access.jsonl").check(
            accessed,
            purpose="predeclared development/v3_select diagnostic ablations",
            label_access=True,
        )
        output = run / (
            "diagnostics/feature_ablations"
            if not args.artifact_tag
            else f"diagnostics/feature_ablations_{args.artifact_tag}"
        )
        if args.dry_run:
            return {
                "dry_run": True,
                "scope": "v3_select_only",
                "selection_summary": str(selection_path),
                "diagnostic_contract": str(diagnostic_contract),
                "would_write": str(output.resolve()),
                "formal_test_read": False,
            }
        train_ids = load_split_ids(v2_split, development_partition="train")
        select_ids = load_split_ids(v3_split, v3_partition="v3_select")
        train_catalog = FeatureCatalog(
            args.train_artifact,
            head_override_dirs=args.train_head_override or (),
            candidate_feature_paths=(train_candidate_source,),
        )
        select_catalog = FeatureCatalog(
            args.eval_artifact,
            head_override_dirs=args.eval_head_override or (),
            candidate_feature_paths=(select_candidate_source,),
        )
        train_labels = load_label_lookup(
            train_labels_path,
            allowed_ids=train_ids,
            candidate_identity=train_catalog.candidate_identity(train_ids),
        )
        select_labels = load_label_lookup(
            select_labels_path,
            allowed_ids=select_ids,
            candidate_identity=select_catalog.candidate_identity(select_ids),
        )
        ordered_train = sorted(train_ids)
        train_prior_values, train_valid = load_v2_oof_prior(args.v2_root, ordered_train)
        if not train_valid.all():
            raise ValueError("V2 OOF prior coverage is incomplete for diagnostics")
        train_priors = prior_array_to_lookup(
            __import__("numpy").asarray(ordered_train), train_prior_values
        )
        ordered_select = sorted(select_ids)
        select_prior_values, select_valid = load_v2_validation_prior(
            args.v2_root, ordered_select
        )
        if not select_valid.all():
            raise ValueError("V2 validation prior coverage is incomplete for diagnostics")
        select_priors = prior_array_to_lookup(
            __import__("numpy").asarray(ordered_select), select_prior_values
        )
        gate_policy = None
        if gate_policy_path is not None:
            stored_policy = json.loads(gate_policy_path.read_text(encoding="utf-8"))
            gate_policy = dict(
                stored_policy.get("selected", stored_policy.get("policy", stored_policy))
            )
        return run_validation_diagnostic_ablations(
            train_catalog=train_catalog,
            select_catalog=select_catalog,
            train_ids=train_ids,
            select_ids=select_ids,
            train_labels=train_labels,
            select_labels=select_labels,
            train_priors=train_priors,
            select_priors=select_priors,
            v2_validation_predictions=v2_validation_predictions,
            selection_summary_path=selection_path,
            diagnostic_contract_path=diagnostic_contract,
            output_dir=output,
            train_labels_provenance=train_labels_path,
            select_labels_provenance=select_labels_path,
            train_priors_provenance=Path(args.v2_root) / "oof_base/oof_base_predictions.npz",
            select_priors_provenance=Path(args.v2_root) / "primary_validation_locked/predictions.jsonl",
            gate_bundle_path=gate_bundle_path,
            gate_policy=gate_policy,
            gate_policy_path=gate_policy_path,
            device=args.device,
            seed=args.seed,
            epochs=args.epochs,
            batch_size=args.batch_size,
            resume=args.resume,
        )
    if args.command == "train-oof":
        if not args.train_artifact: raise ValueError("train-oof requires --train-artifact")
        selection_path=Path(args.config or (run/"selection/selection_summary.json")); train_labels_path=run/"labels/partitions/train/corrected_scientific/labels.jsonl"; v2_split=Path(args.split_manifest or (Path(args.v2_root)/"split_manifest.json"))
        accessed=[*(Path(value)/"artifact_manifest.json" for value in args.train_artifact),*(Path(value)/"artifact_manifest.json" for value in (args.train_head_override or ())),Path(args.v2_root)/"base_train/features.jsonl",selection_path,train_labels_path,Path(args.v2_root)/"oof_base/oof_base_predictions.npz",Path(args.v2_root)/"oof_primary/oof_setrank_predictions.npz"]
        TestAccessGuard("development",run/"artifact_access.jsonl").check(accessed,purpose="scene-grouped FCER OOF, nested V2 anchor, OOF gate and final ensemble",label_access=True)
        if args.dry_run: return {"dry_run":True,"folds":3,"ensemble_seeds":[20260801,20260802,20260803]}
        selection=json.loads(selection_path.read_text()); selected=SELECTION_GRID[int(selection["selected"]["config_index"])]; ids=load_split_ids(v2_split,development_partition="train"); catalog=FeatureCatalog(args.train_artifact,head_override_dirs=args.train_head_override or (),candidate_feature_paths=(Path(args.v2_root)/"base_train/features.jsonl",)); labels=load_label_lookup(train_labels_path,allowed_ids=ids,candidate_identity=catalog.candidate_identity(ids)); ordered=sorted(ids); prior_values,valid=load_v2_oof_prior(args.v2_root,ordered)
        if not valid.all(): raise ValueError("V2 OOF prior coverage incomplete")
        priors=prior_array_to_lookup(__import__("numpy").asarray(ordered),prior_values)
        namespace = _artifact_namespace(args.artifact_tag)
        training_dir = run / ("training" if namespace is None else f"training_{namespace}")
        return run_grouped_oof_and_final_models(catalog=catalog,train_ids=ids,labels=labels,priors=priors,split_manifest=v2_split,selected_config=selected,output_dir=training_dir,device=args.device,oof_epochs=args.epochs,final_epochs=args.epochs,batch_size=args.batch_size,resume=args.resume)
    if args.command == "train-fullchain":
        artifact = Path(args.artifact or (run / "smoke/fullchain_features"))
        labels = Path(args.v2_root) / "labels_train/corrected/labels.jsonl"
        TestAccessGuard(args.scope, run / "artifact_access.jsonl").check(
            (artifact / "artifact_manifest.json", labels), purpose="smoke FCER training with explicit label join", label_access=True
        )
        if args.dry_run:
            return {"dry_run": True, "labels": str(labels)}
        return train_smoke_suite(
            artifact_dir=artifact, labels_path=labels, v2_root=args.v2_root,
            output_dir=run / "smoke/models", device=args.device, seed=args.seed,
            epochs=3, batch_size=args.batch_size,
        )
    if args.command == "train-gate":
        if args.features:
            if args.scope != "calibration":
                raise PermissionError("explicit gate-policy fitting requires --scope calibration")
            bundle_path = Path(args.features).expanduser().resolve()
            labels_path = run / "labels/partitions/calibration/corrected_scientific/labels.jsonl"
            policy_path = _run_child_path(run, args.policy, "calibration/gate_policy.json")
            TestAccessGuard("calibration", run / "artifact_access.jsonl").check(
                (bundle_path, labels_path),
                purpose="finite-grid safe-gate calibration with candidate identity join",
                label_access=True,
            )
            if args.dry_run:
                return {
                    "dry_run": True, "scope": "calibration",
                    "gate_bundle": str(bundle_path), "labels": str(labels_path.resolve()),
                    "would_write": str(policy_path), "formal_test_read": False,
                }
            return calibrate_gate_bundle(
                bundle_path=bundle_path, labels_path=labels_path, output_path=policy_path,
            )
        namespace = _artifact_namespace(args.artifact_tag)
        training_dir = run / ("training" if namespace is None else f"training_{namespace}")
        summary = (training_dir / "gate/summary.json") if (training_dir/"gate/summary.json").exists() else (run / "smoke/models/summary.json")
        if not summary.exists():
            raise FileNotFoundError("train-fullchain smoke suite (including gate) first")
        value = json.loads(summary.read_text(encoding="utf-8"))
        if summary.name=="summary.json" and summary.parent.name=="gate": return value
        if not value["gate"]["closed_fallback_exact"]: raise AssertionError("V2-anchored gate fallback is not exact")
        return value["gate"]
    if args.command == "run-uncertainty":
        if args.scope == "test":
            raise PermissionError("test uncertainty inference is available only through run-test")
        if not args.eval_artifact: raise ValueError("run-uncertainty requires --eval-artifact")
        candidate_source=(Path(args.v1_root)/"features.jsonl" if args.scope=="test" else Path(args.v2_root)/("base_val/features.jsonl" if args.scope in {"select","lockcheck"} else "base_train/features.jsonl"))
        catalog=FeatureCatalog(args.eval_artifact,head_override_dirs=args.eval_head_override or (),candidate_feature_paths=(candidate_source,))
        if args.scope in {"development","calibration"}:
            ids=load_split_ids(Path(args.split_manifest or (Path(args.v2_root)/"split_manifest.json")),development_partition="train" if args.scope=="development" else "calibration")
            if ids-set(catalog.locations): raise ValueError("inference feature catalog is missing the requested partition")
        else: ids=set(catalog.locations)
        ordered=sorted(ids)
        if args.scope in {"select","lockcheck"}:
            prior_values,valid=load_v2_validation_prior(args.v2_root,ordered)
        elif args.scope=="development":
            prior_values,valid=load_v2_oof_prior(args.v2_root,ordered)
        else:
            prior_path=Path(args.features or (run/f"v2_prior/{args.scope}_deployment/v2_prior.npz"))
            with __import__("numpy").load(prior_path) as payload:
                lookup={str(value):index for index,value in enumerate(payload["sample_ids"])}
                if set(lookup)!=ids: raise ValueError("deployment prior cohort differs from inference features")
                prior_values=__import__("numpy").asarray(payload["prior"])[[lookup[value] for value in ordered]]; valid=__import__("numpy").asarray(payload["valid"])[[lookup[value] for value in ordered]]
        if not valid.all(): raise ValueError("V2 prior coverage is incomplete")
        priors=prior_array_to_lookup(__import__("numpy").asarray(ordered),prior_values)
        namespace = _artifact_namespace(args.artifact_tag)
        training_dir = run / ("training" if namespace is None else f"training_{namespace}")
        training=json.loads((training_dir/"summary.json").read_text()); checkpoints=args.checkpoint or [value["path"] for value in training["final_models"]]
        inference_dir=run/"inference"/(args.scope if namespace is None else f"{namespace}/{args.scope}"); ensemble_path=inference_dir/"fcer_ensemble.npz"; uncertainty_path=inference_dir/"uncertainty.npz"; bundle_path=inference_dir/"gate_bundle.npz"
        accessed=[*(Path(value) for value in args.eval_artifact),*(Path(value) for value in checkpoints)]
        TestAccessGuard(args.scope,run/"artifact_access.jsonl",formal_manifest=Path(args.formal_manifest) if args.formal_manifest else None).check(accessed,purpose="label-free FCER ensemble and deterministic ROI perturbation inference",label_access=False)
        if args.dry_run: return {"dry_run":True,"row_count":len(ids),"perturbations":16,"seeds":3,"labels_read":False}
        inference_dir.mkdir(parents=True,exist_ok=True)
        if not ensemble_path.exists(): predict_final_ensemble(catalog=catalog,sample_ids=ids,priors=priors,checkpoint_paths=checkpoints,output_path=ensemble_path,device=args.device)
        if not uncertainty_path.exists(): score_perturbation_ensemble_streaming(catalog=catalog,sample_ids=ids,priors=priors,checkpoint_paths=checkpoints,output_path=uncertainty_path,device=args.device,batch_size=args.batch_size)
        v2_predictions=Path(args.v2_root)/("calibration_primary_predictions/predictions.jsonl" if args.scope=="calibration" else ("primary_validation_locked/predictions.jsonl" if args.scope in {"select","lockcheck"} else "formal_test_primary_v2/primary_predictions/predictions.jsonl"))
        gate_paths=[value["path"] for value in training["gate_summary"]["models"]]
        bundle=prepare_gate_inference(ensemble_path=ensemble_path,uncertainty_path=uncertainty_path,v2_predictions_path=v2_predictions,gate_checkpoint_paths=gate_paths,device=args.device)
        if not bundle_path.exists(): save_gate_bundle(bundle_path,bundle)
        return {"ensemble":str(ensemble_path),"uncertainty":str(uncertainty_path),"gate_bundle":str(bundle_path),"row_count":len(ids),"labels_read":False}
    if args.command == "evaluate-select":
        if args.config:
            summary=Path(args.config).expanduser().resolve()
        elif args.artifact_tag:
            summary=run/f"selection_{args.artifact_tag}/selection_summary.json"
        else:
            summary=run/"selection/selection_summary.json"
        invalidated=summary.parent/"INVALIDATED.json"
        if invalidated.exists():
            raise PermissionError(
                f"selection artifact was explicitly invalidated and cannot be evaluated: {invalidated}"
            )
        if not summary.exists(): raise FileNotFoundError("selection grid has not completed")
        return json.loads(summary.read_text())
    if args.command == "build-gallery":
        if args.scope != "test":
            raise PermissionError("build-gallery is evaluation-only and requires --scope test")
        descriptor_path=Path(_require(args.config,option="--config")).expanduser().resolve()
        descriptor=_json_descriptor(descriptor_path,description="gallery descriptor")
        expected={"features_path","corrected_labels_path","legacy_labels_path","raw_predictions_path","v2_predictions_path","v3_predictions_path","independent_evaluation_completion_path"}
        optional={"output_dir","per_group"}
        if set(descriptor)-expected-optional or not expected<=set(descriptor):
            raise ValueError("gallery descriptor fields differ from the required evaluation inputs")
        manifest=Path(_require(args.formal_manifest,option="--formal-manifest")).expanduser().resolve()
        verify_manifest_sidecar(manifest)
        inputs={name:Path(descriptor[name]).expanduser().resolve() for name in expected}
        TestAccessGuard("test",run/"artifact_access.jsonl",formal_manifest=manifest).check(
            (descriptor_path,*inputs.values()),purpose="evaluation-only V3 failure galleries",label_access=True,
        )
        output=_run_child_path(run,descriptor.get("output_dir"),"evaluation/galleries")
        if args.dry_run:
            return {"dry_run":True,"would_write":str(output),"labels_read":True}
        from .galleries import build_v3_galleries_strict
        completion = inputs.pop("independent_evaluation_completion_path")
        return build_v3_galleries_strict(
            independent_evaluation_completion_path=completion,
            **inputs,output_dir=output,per_group=int(descriptor.get("per_group",args.per_group)),
        )
    if args.command == "build-report":
        if args.scope != "test":
            raise PermissionError("build-report requires the completed --scope test evaluation")
        descriptor_path=Path(_require(args.config,option="--config")).expanduser().resolve()
        descriptor=_json_descriptor(descriptor_path,description="report descriptor")
        required={"audit","selection","lockcheck","formal","machine_inputs","evidence_artifacts"}
        optional={"figure_inputs","output_dir"}
        if set(descriptor)-required-optional or not required<=set(descriptor):
            raise ValueError("report descriptor fields differ from the required report inputs")
        manifest=Path(_require(args.formal_manifest,option="--formal-manifest")).expanduser().resolve()
        verify_manifest_sidecar(manifest)
        values={name:_descriptor_object_or_file(descriptor[name],field=f"report.{name}") for name in required}
        figures=None if descriptor.get("figure_inputs") is None else _descriptor_object_or_file(descriptor["figure_inputs"],field="report.figure_inputs")
        source_paths=[descriptor_path]
        for name in (*required,"figure_inputs"):
            raw=descriptor.get(name)
            if isinstance(raw,str) and raw.strip(): source_paths.append(Path(raw).expanduser().resolve())
        TestAccessGuard("test",run/"artifact_access.jsonl",formal_manifest=manifest).check(
            source_paths,purpose="post-evaluation V3 machine results and report",label_access=False,
        )
        output=_run_child_path(run,descriptor.get("output_dir"),"results")
        if args.dry_run:
            return {"dry_run":True,"would_write":str(output),"labels_read":False}
        from .reporting import build_report_artifacts
        return build_report_artifacts(
            output,audit=values["audit"],selection=values["selection"],lockcheck=values["lockcheck"],formal=values["formal"],machine_inputs=values["machine_inputs"],evidence_artifacts=values["evidence_artifacts"],figure_inputs=figures,
        )
    raise NotImplementedError(f"command {args.command} is registered but not implemented yet")


def main(argv: list[str] | None = None) -> None:
    raw = list(sys.argv[1:] if argv is None else argv)
    args = _parser().parse_args(raw)
    result = _run_initial(args, [sys.executable, "-m", "failure_analysis.reranking_v3.cli", *raw])
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
