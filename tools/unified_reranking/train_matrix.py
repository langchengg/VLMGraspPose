"""Plan and execute the resumable matched-budget Validation/OOF matrix."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from unified_reranking.hashing import atomic_json, canonical_sha256, sha256_file
from unified_reranking.rules import resolve_rule_features


ROUTES = ("crog", "g1", "c1")
TRACKS = ("T1_native", "T2_matched_common", "T3_tri_backend")
SEEDS = (42, 123, 2026)
RULES = (
    "soft_target_support",
    "jaw_support",
    "angle_agreement",
    "depth_contact",
    "collision_proxy",
    "sign_constrained_linear",
)

# Equal four-trial shortlist for every neural loss.  It spans every level in
# the declared lr/wd/alpha grids without creating an unmanageable Cartesian
# explosion before Official Validation selection.
NEURAL_TRIALS = (
    {"learning_rate": 1e-4, "weight_decay": 0.0, "alpha": 0.25},
    {"learning_rate": 1e-4, "weight_decay": 1e-4, "alpha": 0.5},
    {"learning_rate": 3e-4, "weight_decay": 0.0, "alpha": 1.0},
    {"learning_rate": 3e-4, "weight_decay": 1e-4, "alpha": 0.5},
)
TEMPERATURES = (0.5, 1.0, 2.0, 1.0)
BETAS = (0.5, 1.0, 2.0, 1.0)
TREE_TRIALS = (
    {"num_leaves": 7, "tree_learning_rate": 0.03, "n_estimators": 100},
    {"num_leaves": 15, "tree_learning_rate": 0.05, "n_estimators": 200},
    {"num_leaves": 31, "tree_learning_rate": 0.03, "n_estimators": 200},
    {"num_leaves": 63, "tree_learning_rate": 0.05, "n_estimators": 400},
)


@dataclass(frozen=True)
class Job:
    identifier: str
    command: tuple[str, ...]

    def serialize(self) -> dict[str, object]:
        return {"identifier": self.identifier, "command": list(self.command)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--phase", required=True, choices=("screen", "selected", "encoder"))
    parser.add_argument("--routes", nargs="+", choices=ROUTES, default=list(ROUTES))
    parser.add_argument("--tracks", nargs="+", choices=TRACKS, default=list(TRACKS))
    parser.add_argument("--selection-json", type=Path)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--max-parallel", type=int, default=1)
    parser.add_argument("--allow-active-legacy-worker", action="store_true")
    return parser.parse_args()


def _modes(*, selected: bool) -> Iterable[tuple[str, int | None]]:
    if selected:
        for fold in range(5):
            yield "oof", fold
    yield "validation", None


def _cell_command(
    run_dir: Path,
    route: str,
    track: str,
    *,
    encoder: str,
    loss: str,
    seed: int,
    mode: str,
    fold: int | None,
    parameters: dict[str, object],
) -> tuple[str, ...]:
    command = [
        sys.executable,
        "-m",
        "tools.unified_reranking.train_matrix_cell",
        "--run-dir",
        str(run_dir),
        "--route",
        route,
        "--track",
        track,
        "--encoder",
        encoder,
        "--loss",
        loss,
        "--seed",
        str(seed),
        "--mode",
        mode,
    ]
    if fold is not None:
        command.extend(("--fold", str(fold)))
    for key, value in parameters.items():
        command.extend((f"--{key.replace('_', '-')}", str(value)))
    return tuple(command)


def _rule_command(
    run_dir: Path,
    route: str,
    track: str,
    *,
    method: str,
    alpha: float,
    seed: int,
    mode: str,
    fold: int | None,
) -> tuple[str, ...]:
    command = [
        sys.executable,
        "-m",
        "tools.unified_reranking.run_interpretable_rule",
        "--run-dir",
        str(run_dir),
        "--route",
        route,
        "--track",
        track,
        "--method",
        method,
        "--alpha",
        str(alpha),
        "--seed",
        str(seed),
        "--mode",
        mode,
    ]
    if fold is not None:
        command.extend(("--fold", str(fold)))
    return tuple(command)


def _screen_jobs(run_dir: Path, routes: Iterable[str], tracks: Iterable[str]) -> list[Job]:
    jobs: list[Job] = []
    for route in routes:
        for track in tracks:
            feature_manifest_path = (
                run_dir
                / "03_features"
                / "tracks"
                / track
                / f"{route}_validation"
                / "feature_manifest.json"
            )
            if not feature_manifest_path.is_file():
                raise FileNotFoundError(f"feature track is not ready: {feature_manifest_path}")
            feature_manifest = json.loads(feature_manifest_path.read_text(encoding="utf-8"))
            available_families = {
                item.family
                for item in resolve_rule_features(feature_manifest["model_feature_columns"])
            }
            methods = tuple(method for method in RULES if method in available_families) + (
                "sign_constrained_linear",
            )
            for method in methods:
                for alpha in (0.25, 0.5, 1.0):
                    command = _rule_command(
                        run_dir, route, track, method=method, alpha=alpha, seed=42, mode="validation", fold=None
                    )
                    jobs.append(Job(canonical_sha256(command)[:16], command))
            for encoder, losses in (("linear", ("bce", "ranknet")), ("mlp", ("bce", "ranknet", "listwise", "jacquard_margin_ranknet"))):
                for loss in losses:
                    for index, trial in enumerate(NEURAL_TRIALS):
                        parameters = dict(trial)
                        if loss == "listwise":
                            parameters["temperature"] = TEMPERATURES[index]
                        if loss == "jacquard_margin_ranknet":
                            parameters["beta"] = BETAS[index]
                        command = _cell_command(
                            run_dir,
                            route,
                            track,
                            encoder=encoder,
                            loss=loss,
                            seed=42,
                            mode="validation",
                            fold=None,
                            parameters=parameters,
                        )
                        jobs.append(Job(canonical_sha256(command)[:16], command))
            for trial in TREE_TRIALS:
                command = _cell_command(
                    run_dir,
                    route,
                    track,
                    encoder="lambdamart",
                    loss="lambdarank",
                    seed=42,
                    mode="validation",
                    fold=None,
                    parameters=trial,
                )
                jobs.append(Job(canonical_sha256(command)[:16], command))
    return jobs


def _load_selection(path: Path | None, phase: str) -> dict[str, dict[str, object]]:
    if path is None:
        raise ValueError(f"{phase} phase requires --selection-json")
    value = json.loads(path.read_text(encoding="utf-8"))
    selections = value.get("selections", value)
    if not isinstance(selections, dict):
        raise ValueError("selection JSON must contain a selections object")
    return selections


def _selected_jobs(
    run_dir: Path,
    routes: Iterable[str],
    tracks: Iterable[str],
    selection: dict[str, dict[str, object]],
    *,
    encoder_phase: bool,
) -> list[Job]:
    jobs: list[Job] = []
    for route in routes:
        for track in tracks:
            key = f"{route}/{track}"
            if key not in selection:
                raise ValueError(f"selection JSON misses {key}")
            raw_choices = selection[key]
            choices = raw_choices if isinstance(raw_choices, list) else [raw_choices]
            if encoder_phase and len(choices) != 1:
                raise ValueError(f"encoder phase requires one locked loss choice for {key}")
            for raw_choice in choices:
                chosen = dict(raw_choice)
                loss = str(chosen["loss"])
                parameters = dict(chosen.get("parameters", {}))
                encoders = ("mlp", "deepsets", "set_transformer", "gnn") if encoder_phase else (str(chosen["encoder"]),)
                for encoder in encoders:
                    blocks = (1, 2) if encoder == "set_transformer" and encoder_phase else (int(parameters.get("num_attention_blocks", 2)),)
                    for block_count in blocks:
                        local = dict(parameters)
                        local["num_attention_blocks"] = block_count
                        for seed in SEEDS:
                            for mode, fold in _modes(selected=True):
                                command = _cell_command(
                                    run_dir,
                                    route,
                                    track,
                                    encoder=encoder,
                                    loss=loss,
                                    seed=seed,
                                    mode=mode,
                                    fold=fold,
                                    parameters=local,
                                )
                                jobs.append(Job(canonical_sha256(command)[:16], command))
    return jobs


def build_jobs(args: argparse.Namespace) -> list[Job]:
    run_dir = args.run_dir.resolve()
    if args.phase == "screen":
        return _screen_jobs(run_dir, args.routes, args.tracks)
    selection = _load_selection(args.selection_json, args.phase)
    return _selected_jobs(
        run_dir,
        args.routes,
        args.tracks,
        selection,
        encoder_phase=args.phase == "encoder",
    )


def _legacy_worker_active() -> bool:
    result = subprocess.run(
        ("pgrep", "-f", "reranking.run_experiment_matrix"),
        check=False,
        capture_output=True,
        text=True,
    )
    return result.returncode == 0 and bool(result.stdout.strip())


def _run_job(job: Job, cwd: Path, environment: dict[str, str]) -> dict[str, object]:
    result = subprocess.run(job.command, cwd=cwd, env=environment, capture_output=True, text=True)
    output_manifest: dict[str, str] | None = None
    if result.returncode == 0:
        for line in reversed(result.stdout.splitlines()):
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            manifest_value = payload.get("manifest") if isinstance(payload, dict) else None
            expected = payload.get("manifest_sha256") if isinstance(payload, dict) else None
            if isinstance(manifest_value, str) and isinstance(expected, str):
                manifest_path = Path(manifest_value).resolve()
                if not manifest_path.is_file() or sha256_file(manifest_path) != expected:
                    raise RuntimeError(
                        f"matrix job returned an invalid output manifest: {job.identifier}"
                    )
                output_manifest = {"path": str(manifest_path), "sha256": expected}
                break
        if output_manifest is None:
            raise RuntimeError(
                f"matrix job did not declare its output manifest: {job.identifier}"
            )
    return {
        "identifier": job.identifier,
        "returncode": result.returncode,
        "stdout_tail": result.stdout[-4000:],
        "stderr_tail": result.stderr[-4000:],
        "output_manifest": output_manifest,
    }


def main() -> int:
    args = parse_args()
    if args.max_parallel <= 0:
        raise ValueError("max parallelism must be positive")
    jobs = build_jobs(args)
    run_dir = args.run_dir.resolve()
    selection_record = None
    if args.phase != "screen":
        if args.selection_json is None:
            raise ValueError(f"{args.phase} phase requires --selection-json")
        selection_path = args.selection_json.resolve()
        selection_record = {
            "path": str(selection_path),
            "sha256": sha256_file(selection_path),
        }
    plan = {
        "status": "PLANNED",
        "phase": args.phase,
        "routes": list(args.routes),
        "tracks": list(args.tracks),
        "job_count": len(jobs),
        "matched_neural_trial_count": len(NEURAL_TRIALS),
        "matched_tree_trial_count": len(TREE_TRIALS),
        "selection": selection_record,
        "planner_tool": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256_file(Path(__file__)),
        },
        "jobs": [job.serialize() for job in jobs],
    }
    plan_path = run_dir / "05_models" / "matrix_plans" / f"{args.phase}_{canonical_sha256(plan)[:16]}.json"
    atomic_json(plan_path, plan)
    print(json.dumps({"plan": str(plan_path), "jobs": len(jobs), "execute": args.execute}, sort_keys=True), flush=True)
    if not args.execute:
        return 0
    if _legacy_worker_active() and not args.allow_active_legacy_worker:
        raise RuntimeError("active legacy reranking worker detected; refusing concurrent ranker training")
    environment = os.environ.copy()
    environment["PYTHONPATH"] = f"{SRC}:{environment.get('PYTHONPATH', '')}".rstrip(":")
    results: list[dict[str, object]] = []
    with ThreadPoolExecutor(max_workers=args.max_parallel) as executor:
        future_to_job = {
            executor.submit(_run_job, job, ROOT, environment): job for job in jobs
        }
        for future in as_completed(future_to_job):
            result = future.result()
            results.append(result)
            print(json.dumps(result, sort_keys=True), flush=True)
            if result["returncode"] != 0:
                for pending in future_to_job:
                    pending.cancel()
                execution_payload = {
                    "status": "FAILED",
                    "phase": args.phase,
                    "plan": {
                        "path": str(plan_path.resolve()),
                        "sha256": sha256_file(plan_path),
                    },
                    "selection": selection_record,
                    "results": sorted(results, key=lambda item: str(item["identifier"])),
                }
                atomic_json(plan_path.with_name(f"{plan_path.stem}_execution.json"), execution_payload)
                atomic_json(
                    plan_path.parent / f"{args.phase}_latest_execution.json",
                    execution_payload,
                )
                return 1
    ordered_results = sorted(results, key=lambda item: str(item["identifier"]))
    output_manifests = [item["output_manifest"] for item in ordered_results]
    if len(output_manifests) != len(jobs) or any(
        not isinstance(record, dict) for record in output_manifests
    ):
        raise RuntimeError("matrix execution did not bind every output cell manifest")
    if len({str(record["path"]) for record in output_manifests}) != len(
        output_manifests
    ):
        raise RuntimeError("matrix execution produced duplicate cell manifests")
    execution_payload = {
        "status": "COMPLETE",
        "phase": args.phase,
        "plan": {
            "path": str(plan_path.resolve()),
            "sha256": sha256_file(plan_path),
        },
        "selection": selection_record,
        "job_count": len(jobs),
        "results": ordered_results,
        "output_manifests": output_manifests,
    }
    atomic_json(plan_path.with_name(f"{plan_path.stem}_execution.json"), execution_payload)
    atomic_json(
        plan_path.parent / f"{args.phase}_latest_execution.json",
        execution_payload,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
