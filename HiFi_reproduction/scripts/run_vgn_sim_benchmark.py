#!/usr/bin/env python3
"""Protocol-faithful VGN PyBullet clutter-removal benchmark.

The runner reuses the pinned VGN ``ClutterRemovalSim`` and the tested local
network/post-processing adapter without modifying ``third_party/vgn``.  It is
fail-closed: missing official assets, a dependency/version mismatch, or an
unsupported recording request produces a structured blocker with null
simulation metrics rather than a fabricated success result.
"""

from __future__ import annotations

import argparse
import contextlib
import importlib.metadata
import importlib.util
import json
import logging
import os
import platform
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import numpy as np

from src.grasping.vgn_adapter import (
    OFFICIAL_QUALITY_THRESHOLD,
    OFFICIAL_VGN_COMMIT,
    OFFICIAL_VGN_REPOSITORY,
    checkpoint_sha256,
    ensure_official_vgn_path,
    load_official_network,
    predict_official,
    resolve_device_info,
    run_official_postprocessing,
    sort_candidates_by_quality,
    tsdf_grid_from_volume,
)
from src.grasping.vgn_pipeline import atomic_write_csv, atomic_write_json


LOGGER = logging.getLogger("vgn_sim_benchmark")
OFFICIAL_PYBULLET_VERSION = "2.7.9"
OFFICIAL_CHECKPOINT_SHA256 = (
    "ba3391d0805e9c9b178cd18106866313cee808ff2b654f689663e92a814cec4b"
)
POLICIES = ("official_sim_random", "highest_vgn_quality")


@dataclass(frozen=True)
class Scenario:
    name: str
    scene: str
    object_set: str
    num_objects: int
    official_rounds: int
    required_object_urdfs: int


OFFICIAL_SCENARIOS = (
    Scenario("blocks_5", "pile", "blocks", 5, 200, 5),
    Scenario("pile_5", "pile", "pile/test", 5, 200, 40),
    Scenario("packed_5", "packed", "packed/test", 5, 200, 16),
    Scenario("blocks_10", "pile", "blocks", 10, 100, 5),
    Scenario("pile_10", "pile", "pile/test", 10, 100, 40),
)

TRIAL_FIELDS = (
    "trial_id",
    "policy",
    "scenario",
    "scene_seed",
    "round_index",
    "attempt_index",
    "initial_object_count",
    "object_count_before",
    "object_count_after",
    "candidate_count",
    "selected_vgn_quality",
    "selected_width_m",
    "selected_position_xyz_m",
    "integration_time_s",
    "planning_time_s",
    "execution_time_s",
    "official_label",
    "retained_object_success",
    "failure_category",
    "score_source",
)

ROUND_FIELDS = (
    "round_id",
    "policy",
    "scenario",
    "scene_seed",
    "round_index",
    "requested_object_count",
    "initial_object_count",
    "initial_scene_objects_json",
    "attempt_count",
    "successful_execution_count",
    "terminal_reason",
    "round_wall_time_s",
)


def official_retention_success(
    contact_count: int,
    opening_width_m: float,
    *,
    max_opening_width_m: float = 0.08,
) -> bool:
    """Return the exact success predicate used by upstream VGN simulation."""

    return bool(
        int(contact_count) > 0
        and float(opening_width_m) > 0.1 * float(max_opening_width_m)
    )


def _blocker(code: str, message: str, **details: Any) -> dict[str, Any]:
    return {"code": code, "message": message, "details": details}


def _git_commit(root: Path) -> str | None:
    try:
        return subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def simulation_preflight(
    *,
    vgn_root: Path | str,
    weights: Path | str,
    require_video: bool = False,
) -> dict[str, Any]:
    """Inspect all prerequisites without importing ROS or running physics."""

    root = Path(vgn_root).expanduser().resolve()
    checkpoint = Path(weights).expanduser().resolve()
    blockers: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    checks: dict[str, Any] = {}

    marker = root / "src" / "vgn" / "simulation.py"
    checks["vgn_checkout"] = str(root)
    if not marker.is_file():
        blockers.append(
            _blocker(
                "missing_official_vgn_checkout",
                "Pinned official VGN checkout is missing.",
                expected=str(marker),
                repository=OFFICIAL_VGN_REPOSITORY,
                commit=OFFICIAL_VGN_COMMIT,
            )
        )
    commit = _git_commit(root) if marker.is_file() else None
    checks["vgn_commit"] = commit
    if marker.is_file() and commit != OFFICIAL_VGN_COMMIT:
        blockers.append(
            _blocker(
                "vgn_commit_mismatch",
                "Official VGN checkout is not at the pinned CoRL 2020 commit.",
                found=commit,
                required=OFFICIAL_VGN_COMMIT,
            )
        )

    checks["checkpoint_path"] = str(checkpoint)
    if not checkpoint.is_file():
        blockers.append(
            _blocker(
                "missing_vgn_checkpoint",
                "Official pretrained VGN checkpoint is missing.",
                expected=str(checkpoint),
            )
        )
        checkpoint_hash = None
    else:
        checkpoint_hash = checkpoint_sha256(checkpoint)
        if checkpoint_hash != OFFICIAL_CHECKPOINT_SHA256:
            blockers.append(
                _blocker(
                    "vgn_checkpoint_hash_mismatch",
                    "Checkpoint does not match the frozen official data bundle.",
                    found=checkpoint_hash,
                    required=OFFICIAL_CHECKPOINT_SHA256,
                )
            )
    checks["checkpoint_sha256"] = checkpoint_hash

    asset_files = (
        root / "data" / "urdfs" / "panda" / "hand.urdf",
        root / "data" / "urdfs" / "setup" / "plane.urdf",
        root / "data" / "urdfs" / "setup" / "box.urdf",
    )
    missing_assets = [str(path) for path in asset_files if not path.is_file()]
    object_counts: dict[str, int] = {}
    for scenario in OFFICIAL_SCENARIOS:
        object_root = root / "data" / "urdfs" / scenario.object_set
        count = len(list(object_root.glob("*.urdf"))) if object_root.is_dir() else 0
        object_counts[scenario.object_set] = count
        if count < scenario.required_object_urdfs:
            missing_assets.append(
                f"{object_root} (found {count}, require >= {scenario.required_object_urdfs} URDFs)"
            )
    checks["object_urdf_counts"] = object_counts
    if missing_assets:
        blockers.append(
            _blocker(
                "missing_official_simulation_assets",
                "Official VGN URDF assets are incomplete; obtain the data bundle linked by the official README.",
                missing=sorted(set(missing_assets)),
            )
        )

    pybullet_spec = importlib.util.find_spec("pybullet")
    if pybullet_spec is None:
        blockers.append(
            _blocker(
                "missing_pybullet",
                "PyBullet is not installed; strict protocol requires the upstream pinned version.",
                required_version=OFFICIAL_PYBULLET_VERSION,
            )
        )
        pybullet_version = None
    else:
        try:
            pybullet_version = importlib.metadata.version("pybullet")
        except importlib.metadata.PackageNotFoundError:
            pybullet_version = None
        if pybullet_version != OFFICIAL_PYBULLET_VERSION:
            blockers.append(
                _blocker(
                    "pybullet_version_mismatch",
                    "PyBullet physics version differs from the pinned upstream requirement.",
                    found=pybullet_version,
                    required=OFFICIAL_PYBULLET_VERSION,
                )
            )
        else:
            try:
                import pybullet

                if not pybullet.isNumpyEnabled():
                    blockers.append(
                        _blocker(
                            "pybullet_numpy_disabled",
                            "PyBullet must be compiled with NumPy support.",
                        )
                    )
            except Exception as error:  # pragma: no cover - depends on binary loader
                blockers.append(
                    _blocker(
                        "pybullet_import_failed",
                        "PyBullet is installed but cannot be imported.",
                        error=f"{type(error).__name__}: {error}",
                    )
                )
    checks["pybullet_version"] = pybullet_version

    for package in ("torch", "open3d", "scipy"):
        try:
            checks[f"{package}_version"] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            checks[f"{package}_version"] = None
            blockers.append(
                _blocker(
                    f"missing_{package}", f"Required simulation dependency {package} is missing."
                )
            )

    if int(np.__version__.split(".", maxsplit=1)[0]) >= 2:
        warnings.append(
            _blocker(
                "numpy2_tsdf_compatibility_adapter",
                "NumPy 2.x detected; the tested scalar-safe TSDF grid adapter will be used.",
                numpy=np.__version__,
            )
        )
    checks["numpy_version"] = np.__version__
    checks["python_version"] = platform.python_version()
    checks["platform"] = platform.platform()

    if require_video:
        blockers.append(
            _blocker(
                "phase_video_capture_not_implemented",
                "Phase-labelled before/pregrasp/closure/lift/final video capture is not implemented; refusing to claim recorded protocol videos.",
            )
        )

    return {
        "status": "ok" if not blockers else "blocked",
        "protocol": "official_vgn_corl2020_pybullet_clutter_removal",
        "repository_url": OFFICIAL_VGN_REPOSITORY,
        "required_commit": OFFICIAL_VGN_COMMIT,
        "quality_threshold": OFFICIAL_QUALITY_THRESHOLD,
        "checks": checks,
        "warnings": warnings,
        "blockers": blockers,
    }


def build_blocked_aggregate(
    preflight: Mapping[str, Any], config: Mapping[str, Any]
) -> dict[str, Any]:
    """Create a null-valued simulation result for a blocked run."""

    return {
        "status": "blocked",
        "metric_scope": "pybullet_simulated_physical_execution",
        "protocol": "official_vgn_corl2020_pybullet_clutter_removal",
        "simulated_grasp_success_rate": None,
        "simulated_percent_cleared": None,
        "planning_time_s": None,
        "no_grasp_rate": None,
        "collision_failure_rate": None,
        "slip_failure_rate": None,
        "timeout_rate": None,
        "real_robot_metrics_not_computed_here": True,
        "reason": "simulation preflight failed",
        "blockers": list(preflight.get("blockers", [])),
        "config": dict(config),
    }


@contextlib.contextmanager
def _working_directory(path: Path) -> Iterator[None]:
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def _snapshot_scene(sim: Any) -> list[dict[str, Any]]:
    bodies = list(sim.world.bodies.values())
    # The table is inserted first and is the sole non-object body after reset.
    objects = bodies[1:]
    return [
        {
            "uid": int(body.uid),
            "name": str(body.name),
            "pose": body.get_pose().to_dict(),
        }
        for body in objects
    ]


def _scenario_seed(base_seed: int, scenario_index: int) -> int:
    return int(base_seed) + 100_003 * int(scenario_index)


def _make_grasp(candidate: Any) -> Any:
    from vgn.grasp import Grasp
    from vgn.utils.transform import Rotation, Transform

    return Grasp(
        Transform(
            Rotation.from_quat(candidate.quaternion_task_xyzw),
            np.asarray(candidate.position_task_m, dtype=np.float64).copy(),
        ),
        float(candidate.width_m),
    )


def _select_execution_candidate(
    candidates: Sequence[Any], policy: str, rng: np.random.RandomState
) -> Any:
    if policy == "highest_vgn_quality":
        return sort_candidates_by_quality(candidates)[0]
    if policy == "official_sim_random":
        return candidates[int(rng.permutation(len(candidates))[0])]
    raise ValueError(f"unsupported simulation selection policy: {policy}")


def _aggregate_completed(
    trials: Sequence[Mapping[str, Any]],
    rounds: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    attempts = len(trials)
    successes = sum(int(bool(row["retained_object_success"])) for row in trials)
    initial_objects = sum(int(row["initial_object_count"]) for row in rounds)
    no_grasp_rounds = sum(row["terminal_reason"] == "no_grasp" for row in rounds)
    planning = [float(row["planning_time_s"]) for row in trials]

    by_policy_scenario: list[dict[str, Any]] = []
    for policy in config["selection_policies"]:
        for scenario in OFFICIAL_SCENARIOS:
            selected_trials = [
                row
                for row in trials
                if row["policy"] == policy and row["scenario"] == scenario.name
            ]
            selected_rounds = [
                row
                for row in rounds
                if row["policy"] == policy and row["scenario"] == scenario.name
            ]
            denominator = len(selected_trials)
            cleared_denominator = sum(
                int(row["initial_object_count"]) for row in selected_rounds
            )
            selected_successes = sum(
                int(bool(row["retained_object_success"])) for row in selected_trials
            )
            by_policy_scenario.append(
                {
                    "policy": policy,
                    "scenario": scenario.name,
                    "round_count": len(selected_rounds),
                    "attempted_executions": denominator,
                    "successful_executions": selected_successes,
                    "initial_objects": cleared_denominator,
                    "simulated_grasp_success_rate": (
                        selected_successes / denominator if denominator else None
                    ),
                    "simulated_percent_cleared": (
                        selected_successes / cleared_denominator
                        if cleared_denominator
                        else None
                    ),
                }
            )

    return {
        "status": "completed",
        "metric_scope": "pybullet_simulated_physical_execution",
        "protocol": "official_vgn_corl2020_pybullet_clutter_removal",
        "attempted_executions": attempts,
        "successful_executions": successes,
        "initial_objects": initial_objects,
        "simulated_grasp_success_rate": successes / attempts if attempts else None,
        "simulated_percent_cleared": (
            successes / initial_objects if initial_objects else None
        ),
        "planning_time_s": {
            "mean": float(np.mean(planning)) if planning else None,
            "median": float(np.median(planning)) if planning else None,
        },
        "no_grasp_rate": no_grasp_rounds / len(rounds) if rounds else None,
        "collision_failure_rate": None,
        "slip_failure_rate": None,
        "timeout_rate": None,
        "failure_taxonomy_reason": (
            "Pinned upstream execute_grasp returns only SUCCESS/FAILURE; "
            "collision, slip and timeout are not inferred from that binary label."
        ),
        "real_robot_metrics_not_computed_here": True,
        "by_policy_scenario": by_policy_scenario,
        "config": dict(config),
    }


def run_benchmark(config: Mapping[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Execute all configured scenarios after a successful strict preflight."""

    root = Path(config["vgn_root"])
    ensure_official_vgn_path(root)
    device_info = resolve_device_info(str(config["device"]), logger=LOGGER)
    net = load_official_network(
        config["weights"],
        device=device_info.resolved,
        vgn_root=root,
        logger=LOGGER,
    )

    from vgn.grasp import Label
    from vgn.simulation import ClutterRemovalSim

    trial_rows: list[dict[str, Any]] = []
    round_rows: list[dict[str, Any]] = []

    with _working_directory(root):
        for policy in config["selection_policies"]:
            for scenario_index, scenario in enumerate(OFFICIAL_SCENARIOS):
                rounds_to_run = (
                    scenario.official_rounds
                    if config["official_paper_protocol"]
                    else int(config["rounds_per_scenario"])
                )
                scene_seed = _scenario_seed(int(config["seed"]), scenario_index)
                candidate_rng = np.random.RandomState(scene_seed + 50_000_003)
                sim = ClutterRemovalSim(
                    scenario.scene,
                    scenario.object_set,
                    gui=bool(config["sim_gui"]),
                    seed=scene_seed,
                )
                try:
                    for round_index in range(rounds_to_run):
                        round_start = time.perf_counter()
                        sim.reset(scenario.num_objects)
                        initial_count = int(sim.num_objects)
                        initial_scene = _snapshot_scene(sim)
                        attempt_index = 0
                        successful_count = 0
                        consecutive_failures = 1
                        last_label: Any = None
                        terminal_reason = "unknown"

                        while (
                            sim.num_objects > 0
                            and consecutive_failures < 2
                        ):
                            tsdf, point_cloud, integration_time = sim.acquire_tsdf(n=6)
                            if point_cloud.is_empty():
                                terminal_reason = "empty_point_cloud"
                                break

                            tsdf_grid = tsdf_grid_from_volume(tsdf)
                            planning_start = time.perf_counter()
                            prediction = predict_official(
                                tsdf_grid, net, device_info.resolved, logger=LOGGER
                            )
                            post = run_official_postprocessing(
                                tsdf_grid,
                                prediction.qual_vol,
                                prediction.rot_vol,
                                prediction.width_vol,
                                voxel_size_m=float(tsdf.voxel_size),
                            )
                            candidates = list(post.candidates)
                            if not candidates:
                                terminal_reason = "no_grasp"
                                break

                            candidate = _select_execution_candidate(
                                candidates, policy, candidate_rng
                            )
                            grasp = _make_grasp(candidate)
                            # Upstream times the complete planner, including
                            # random candidate ordering and metric conversion.
                            planning_time = time.perf_counter() - planning_start
                            before = int(sim.num_objects)
                            execution_start = time.perf_counter()
                            label, measured_width = sim.execute_grasp(
                                grasp, allow_contact=True
                            )
                            execution_time = time.perf_counter() - execution_start
                            after = int(sim.num_objects)
                            retained = bool(label == Label.SUCCESS)
                            if retained:
                                successful_count += 1

                            trial_rows.append(
                                {
                                    "trial_id": (
                                        f"{policy}/{scenario.name}/"
                                        f"r{round_index:04d}/a{attempt_index:02d}"
                                    ),
                                    "policy": policy,
                                    "scenario": scenario.name,
                                    "scene_seed": scene_seed,
                                    "round_index": round_index,
                                    "attempt_index": attempt_index,
                                    "initial_object_count": initial_count,
                                    "object_count_before": before,
                                    "object_count_after": after,
                                    "candidate_count": len(candidates),
                                    "selected_vgn_quality": float(candidate.vgn_quality),
                                    "selected_width_m": float(candidate.width_m),
                                    "selected_position_xyz_m": json.dumps(
                                        candidate.position_task_m.tolist(), separators=(",", ":")
                                    ),
                                    "integration_time_s": float(integration_time),
                                    "planning_time_s": planning_time,
                                    "execution_time_s": execution_time,
                                    "official_label": int(label),
                                    "retained_object_success": retained,
                                    "failure_category": (
                                        "" if retained else "official_unclassified_failure"
                                    ),
                                    "score_source": "official_vgn_processed_quality",
                                    "measured_gripper_width_m": float(measured_width),
                                }
                            )
                            attempt_index += 1

                            if last_label == Label.FAILURE and label == Label.FAILURE:
                                consecutive_failures += 1
                            else:
                                consecutive_failures = 1
                            last_label = label

                        if sim.num_objects == 0:
                            terminal_reason = "cleared"
                        elif consecutive_failures >= 2:
                            terminal_reason = "two_consecutive_failures"

                        round_rows.append(
                            {
                                "round_id": f"{policy}/{scenario.name}/r{round_index:04d}",
                                "policy": policy,
                                "scenario": scenario.name,
                                "scene_seed": scene_seed,
                                "round_index": round_index,
                                "requested_object_count": scenario.num_objects,
                                "initial_object_count": initial_count,
                                "initial_scene_objects_json": json.dumps(
                                    initial_scene, separators=(",", ":"), sort_keys=True
                                ),
                                "attempt_count": attempt_index,
                                "successful_execution_count": successful_count,
                                "terminal_reason": terminal_reason,
                                "round_wall_time_s": time.perf_counter() - round_start,
                            }
                        )
                finally:
                    sim.world.close()
    return trial_rows, round_rows


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vgn-root", type=Path, default=Path("third_party/vgn"))
    parser.add_argument(
        "--weights",
        type=Path,
        default=Path("third_party/vgn/data/models/vgn_conv.pth"),
    )
    parser.add_argument(
        "--selection-policy",
        nargs="+",
        choices=POLICIES,
        default=list(POLICIES),
    )
    parser.add_argument("--rounds-per-scenario", type=int)
    parser.add_argument("--official-paper-protocol", action="store_true")
    parser.add_argument("--quality-threshold", type=float, default=0.9)
    parser.add_argument("--record-video", action="store_true")
    parser.add_argument("--record-representative-videos", action="store_true")
    parser.add_argument("--sim-gui", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--output", type=Path, default=Path("outputs/vgn_sim_benchmark"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-level", default="INFO")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper()),
        format="%(asctime)s %(levelname)s %(message)s",
    )
    if float(args.quality_threshold) != OFFICIAL_QUALITY_THRESHOLD:
        raise SystemExit("quality threshold is locked to the official value 0.90")
    if args.rounds_per_scenario is not None and args.rounds_per_scenario <= 0:
        raise SystemExit("--rounds-per-scenario must be positive")
    if (
        not args.preflight_only
        and not args.official_paper_protocol
        and args.rounds_per_scenario is None
    ):
        raise SystemExit(
            "Choose --official-paper-protocol or an explicit --rounds-per-scenario; "
            "there is no implicit expensive run."
        )
    if args.official_paper_protocol and args.rounds_per_scenario is not None:
        raise SystemExit(
            "--official-paper-protocol and --rounds-per-scenario are mutually exclusive"
        )
    if len(set(args.selection_policy)) != len(args.selection_policy):
        raise SystemExit("duplicate --selection-policy values are not allowed")

    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    config = {
        "vgn_root": str(args.vgn_root.expanduser().resolve()),
        "weights": str(args.weights.expanduser().resolve()),
        "selection_policies": list(args.selection_policy),
        "official_paper_protocol": bool(args.official_paper_protocol),
        "rounds_per_scenario": args.rounds_per_scenario,
        "quality_threshold": float(args.quality_threshold),
        "seed": int(args.seed),
        "device": args.device,
        "sim_gui": bool(args.sim_gui),
        "record_video": bool(args.record_video),
        "record_representative_videos": bool(args.record_representative_videos),
        "scenarios": [asdict(scenario) for scenario in OFFICIAL_SCENARIOS],
    }
    atomic_write_json(output / "run_config.json", config)
    preflight = simulation_preflight(
        vgn_root=config["vgn_root"],
        weights=config["weights"],
        require_video=bool(args.record_video or args.record_representative_videos),
    )
    atomic_write_json(output / "preflight.json", preflight)
    if preflight["status"] != "ok":
        aggregate = build_blocked_aggregate(preflight, config)
        atomic_write_json(output / "aggregate.json", aggregate)
        LOGGER.error("Simulation blocked: %s", [b["code"] for b in preflight["blockers"]])
        return 2
    if args.preflight_only:
        atomic_write_json(
            output / "aggregate.json",
            {
                "status": "preflight_ok_not_executed",
                "metric_scope": "pybullet_simulated_physical_execution",
                "simulated_grasp_success_rate": None,
                "simulated_percent_cleared": None,
                "reason": "preflight-only mode; no physics executions were attempted",
                "real_robot_metrics_not_computed_here": True,
                "config": config,
            },
        )
        return 0

    try:
        trials, rounds = run_benchmark(config)
    except Exception as error:
        atomic_write_json(
            output / "aggregate.json",
            {
                "status": "failed",
                "metric_scope": "pybullet_simulated_physical_execution",
                "simulated_grasp_success_rate": None,
                "simulated_percent_cleared": None,
                "reason": f"{type(error).__name__}: {error}",
                "real_robot_metrics_not_computed_here": True,
                "config": config,
            },
        )
        LOGGER.exception("Simulation execution failed")
        return 1

    atomic_write_csv(output / "trials.csv", trials, TRIAL_FIELDS)
    atomic_write_csv(output / "rounds.csv", rounds, ROUND_FIELDS)
    atomic_write_json(output / "aggregate.json", _aggregate_completed(trials, rounds, config))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
