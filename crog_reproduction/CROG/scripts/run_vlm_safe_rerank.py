#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from failure_analysis.vlm_safe_rerank.locking import assert_formal_run_allowed
from failure_analysis.vlm_safe_rerank.local_experiment import calibrate_local_only, validate_local_only
from failure_analysis.vlm_safe_rerank.api_calibration import (
    API_CALIBRATION_CONFIRMATION_PHASE,
    API_CALIBRATION_PHASE,
    calibrate_api_safe_gate,
)
from failure_analysis.vlm_safe_rerank.renderer import PerturbationVariant
from failure_analysis.vlm_safe_rerank.critic_evaluation import evaluate_diagnostic_phase
from failure_analysis.vlm_safe_rerank.runner import (
    prepare_cohort,
    prepare_confirmation_phase,
    prepare_perturbation_phase,
    run_pairwise_phase,
)
from failure_analysis.vlm_safe_rerank.full_list_runner import (
    evaluate_full_list_phase,
    run_full_list_phase,
)
from failure_analysis.vlm_safe_rerank.reporting import generate_reports
from failure_analysis.vlm_safe_rerank.finalization import finalize_safe_rerank
from failure_analysis.vlm_safe_rerank.p5_validation import (
    evaluate_p5_validation,
    prepare_p5_validation_inference,
)


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description="Protected q-only + advisory Gemini reranking experiment")
    sub = root.add_subparsers(dest="command", required=True)
    for name in ("prepare-smoke", "prepare-diagnostic", "prepare-calibration"):
        command = sub.add_parser(name)
        command.add_argument("--run-dir", required=True)
        command.add_argument("--seed", type=int, default=20260803)
        command.add_argument("--per-cohort", type=int, default=30)
    for name in (
        "smoke", "diagnostic", "diagnostic-expanded", "diagnostic-perturbations", "p3-diagnostic",
        "api-calibration", "diagnostic-confirmation", "api-calibration-confirmation",
        "p5-validation", "p5-validation-confirmation",
    ):
        command = sub.add_parser(name)
        command.add_argument("--run-dir", required=True)
        command.add_argument("--env-file", default=".env")
        command.add_argument("--model", action="append", choices=["gemini-robotics-er-2-preview", "gemini-3.6-flash"])
        command.add_argument("--protocol", default="P4", choices=["P3", "P4", "P5"])
        command.add_argument("--max-samples", type=int)
        command.add_argument("--max-api-cost-usd", type=float)
        command.add_argument("--concurrency", type=int, default=1)
        command.add_argument("--resume", action="store_true")
        command.add_argument("--replay-only", action="store_true")
        command.add_argument("--seed", type=int, default=20260803)
        command.add_argument("--all-perturbations", action="store_true")
        command.add_argument("--variant", action="append", choices=[variant.value for variant in PerturbationVariant])
        command.add_argument("--max-transport-retries", type=int, default=3)
        command.add_argument("--circuit-breaker-terminal-failures", type=int, default=5)
    formal = sub.add_parser("formal-test")
    formal.add_argument("--run-dir", required=True)
    formal.add_argument("--locked-manifest", required=True)
    formal.add_argument("--allow-formal", action="store_true")
    p2 = sub.add_parser("p2-full-list-smoke")
    p2.add_argument("--run-dir", required=True)
    p2.add_argument("--env-file", default=".env")
    p2.add_argument("--model", action="append", choices=["gemini-robotics-er-2-preview", "gemini-3.6-flash"])
    p2.add_argument("--max-samples", type=int)
    p2.add_argument("--max-transport-retries", type=int, default=1)
    p2_eval = sub.add_parser("p2-evaluate")
    p2_eval.add_argument("--run-dir", required=True)
    prepare_p5 = sub.add_parser("prepare-p5-validation")
    prepare_p5.add_argument("--run-dir", required=True)
    p5_eval = sub.add_parser("p5-evaluate")
    p5_eval.add_argument("--run-dir", required=True)
    confirmation = sub.add_parser("prepare-confirmation")
    confirmation.add_argument("--run-dir", required=True)
    confirmation.add_argument("--source-phase", required=True)
    confirmation.add_argument("--destination-phase", required=True)
    perturbations = sub.add_parser("prepare-perturbations")
    perturbations.add_argument("--run-dir", required=True)
    perturbations.add_argument("--source-phase", required=True)
    perturbations.add_argument("--destination-phase", required=True)
    diagnostic_eval = sub.add_parser("evaluate-diagnostic")
    diagnostic_eval.add_argument("--run-dir", required=True)
    diagnostic_eval.add_argument("--phase", required=True)
    diagnostic_eval.add_argument("--perturbation-phase")
    # Named placeholders keep the requested unified CLI stable while their
    # implementations live in evaluator-only modules.
    for name in ("audit", "replay-existing", "build-cohorts", "calibrate", "calibrate-api", "validate", "finalize", "report"):
        command = sub.add_parser(name)
        command.add_argument("--run-dir", required=True)
    return root


def main() -> int:
    args = parser().parse_args()
    if args.command == "prepare-smoke":
        result = prepare_cohort(run_dir=args.run_dir, phase="smoke", all_challengers=False, seed=args.seed)
    elif args.command == "prepare-diagnostic":
        result = prepare_cohort(
            run_dir=args.run_dir, phase="diagnostic", all_challengers=True, seed=args.seed,
            per_cohort={name: args.per_cohort for name in ("protected_correct", "recoverable_error", "unrecoverable_error")},
        )
    elif args.command == "prepare-calibration":
        result = prepare_cohort(
            run_dir=args.run_dir, phase=API_CALIBRATION_PHASE, all_challengers=False, seed=args.seed,
            partition="calibration",
            per_cohort={name: args.per_cohort for name in ("protected_correct", "recoverable_error", "unrecoverable_error")},
        )
    elif args.command in (
        "smoke", "diagnostic", "diagnostic-expanded", "diagnostic-perturbations", "p3-diagnostic",
        "api-calibration", "diagnostic-confirmation", "api-calibration-confirmation",
        "p5-validation", "p5-validation-confirmation",
    ):
        if args.max_api_cost_usd is not None:
            raise SystemExit("set GEMINI_MAX_SPEND_USD in the mode-0600 env file; CLI budget overrides are forbidden")
        if args.concurrency != 1:
            raise SystemExit("current safe runner is deliberately serialized; concurrency must be 1")
        variants = (
            tuple(PerturbationVariant) if args.all_perturbations
            else tuple(PerturbationVariant(value) for value in args.variant)
            if args.variant else (PerturbationVariant.ORIGINAL,)
        )
        if args.command == "diagnostic-perturbations" and not args.variant and not args.all_perturbations:
            variants = tuple(
                variant for variant in PerturbationVariant
                if variant is not PerturbationVariant.ORIGINAL
            )
        if args.command == "diagnostic-perturbations":
            variants = tuple(
                variant for variant in variants
                if variant is not PerturbationVariant.ORIGINAL
            )
        if args.command.endswith("confirmation"):
            if args.variant and set(args.variant) != {PerturbationVariant.PANEL_SWAP.value}:
                raise SystemExit("confirmation phases require only --variant panel_swap")
            variants = (PerturbationVariant.PANEL_SWAP,)
        phase = args.command.replace("-", "_")
        if args.command == "api-calibration":
            phase = API_CALIBRATION_PHASE
        elif args.command == "api-calibration-confirmation":
            phase = API_CALIBRATION_CONFIRMATION_PHASE
        requested_models = args.model
        frozen_manifest_path = Path(args.run_dir) / phase / "inference_manifest.json"
        if requested_models is None and frozen_manifest_path.is_file():
            frozen_manifest = json.loads(frozen_manifest_path.read_text(encoding="utf-8"))
            if "needed_models" in frozen_manifest:
                requested_models = list(frozen_manifest["needed_models"])
        result = run_pairwise_phase(
            run_dir=args.run_dir, phase=phase, env_file=args.env_file,
            models=(
                requested_models
                if requested_models is not None
                else ("gemini-robotics-er-2-preview", "gemini-3.6-flash")
            ),
            protocol=args.protocol, variants=variants, max_pairs=args.max_samples,
            max_transport_retries=args.max_transport_retries,
            circuit_breaker_terminal_failures=args.circuit_breaker_terminal_failures,
            replay_only=args.replay_only,
        )
    elif args.command == "formal-test":
        assert_formal_run_allowed(
            args.locked_manifest,
            cli_allow_formal=args.allow_formal,
        )
        raise SystemExit(
            "formal gate verified, but no provider request was started and no one-time claim was consumed; "
            "this run has no integrated formal provider runner"
        )
    elif args.command == "prepare-confirmation":
        result = prepare_confirmation_phase(
            run_dir=args.run_dir, source_phase=args.source_phase,
            destination_phase=args.destination_phase,
        )
    elif args.command == "prepare-perturbations":
        result = prepare_perturbation_phase(
            run_dir=args.run_dir,
            source_phase=args.source_phase,
            destination_phase=args.destination_phase,
        )
    elif args.command == "evaluate-diagnostic":
        result = evaluate_diagnostic_phase(
            args.run_dir,
            args.phase,
            perturbation_phase=args.perturbation_phase,
        )
    elif args.command == "p2-full-list-smoke":
        result = run_full_list_phase(
            run_dir=args.run_dir, env_file=args.env_file,
            models=args.model or ("gemini-robotics-er-2-preview", "gemini-3.6-flash"),
            max_samples=args.max_samples,
            max_transport_retries=args.max_transport_retries,
        )
    elif args.command == "p2-evaluate":
        result = evaluate_full_list_phase(args.run_dir)
    elif args.command == "prepare-p5-validation":
        result = prepare_p5_validation_inference(args.run_dir)
    elif args.command == "p5-evaluate":
        result = evaluate_p5_validation(args.run_dir)
    elif args.command == "calibrate":
        result = calibrate_local_only(args.run_dir)
    elif args.command == "calibrate-api":
        result = calibrate_api_safe_gate(args.run_dir)
    elif args.command == "validate":
        result = validate_local_only(args.run_dir)
    elif args.command == "report":
        result = generate_reports(args.run_dir)
    elif args.command == "finalize":
        result = finalize_safe_rerank(args.run_dir)
    else:
        raise SystemExit(f"{args.command} is not wired yet; no state was changed")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
