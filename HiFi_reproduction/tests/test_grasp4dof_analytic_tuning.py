from __future__ import annotations

import json
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from src.grasping.backends import AnalyticGraspConfig
from tools.grasp4dof.tune_analytic import (
    AnalyticTuner,
    canonical_config_hash,
    coordinate_trial_upper_bound,
    selection_key,
)


FAKE_RUNNER = r'''#!/usr/bin/env python3
import argparse
import hashlib
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


parser = argparse.ArgumentParser()
parser.add_argument("--run-dir", type=Path, required=True)
parser.add_argument("--split", required=True)
parser.add_argument("--method", required=True)
parser.add_argument("--config", type=Path, required=True)
parser.add_argument("--output-dir", type=Path, required=True)
parser.add_argument("--sample-list", type=Path)
args = parser.parse_args()
assert args.split == "validation"
assert args.method == "A0"
manifests = args.run_dir / "manifests"
validation = manifests / "validation_samples.parquet"
labels = manifests / "validation_labels.parquet"
all_ids = pq.read_table(validation, columns=["sample_id"]).column("sample_id").to_pylist()
if args.sample_list:
    ids = [row["sample_id"] for row in json.loads(args.sample_list.read_text())["samples"]]
else:
    ids = all_ids
config = json.loads(args.config.read_text())
preferred = {
    "component_policy": "largest",
    "opening_radius_px": 0,
    "closing_radius_px": 1,
    "contour_spacing_px": 3.0,
    "antipodal_alignment_min": 0.7,
    "min_axis_mask_support": 0.7,
    "max_axis_depth_jump_m": 0.03,
}
quality = sum(config[key] == value for key, value in preferred.items())
j1 = 0.10 + quality * 0.01
args.output_dir.mkdir(parents=True)
pq.write_table(pa.table({"sample_id": ids}), args.output_dir / "per_sample_predictions.parquet")
pq.write_table(
    pa.table({"sample_id": ids, "candidate_id": [f"candidate-{i}" for i in range(len(ids))]}),
    args.output_dir / "per_candidate_predictions.parquet",
)
(args.output_dir / "metrics.json").write_text(json.dumps({
    "j_at_1": j1,
    "j_at_5": j1 + 0.05,
    "recall_at_5": j1 + 0.06,
    "no_grasp_rate": 0.02,
    "p50_latency_seconds": 0.01,
    "p95_latency_seconds": 0.02,
}))
(args.output_dir / "runtime_metrics.json").write_text(json.dumps({
    "total_wall_seconds_including_io_and_evaluation": 0.1,
}))
(args.output_dir / "run_config.json").write_text(json.dumps({
    "method": "repeatedfilm_mask_depth_analytic",
    "split": "validation",
    "sample_count": len(ids),
    "config_sha256": sha(args.config),
    "samples_manifest_sha256": sha(validation),
    "labels_manifest_sha256": sha(labels),
}))
(args.output_dir / "COMPLETE.json").write_text(json.dumps({
    "status": "COMPLETE", "sample_count": len(ids),
}))
counter = args.run_dir / "fake_runner_invocations.txt"
with counter.open("a") as stream:
    stream.write(f"{args.split},{'pilot' if args.sample_list else 'full'}\n")
'''


def _write_run(tmp_path: Path) -> tuple[Path, Path]:
    run_dir = tmp_path / "run"
    manifests = run_dir / "manifests"
    manifests.mkdir(parents=True)
    validation_ids = [f"validation-{index:03d}" for index in range(12)]
    pq.write_table(
        pa.table(
            {
                "sample_id": validation_ids,
                "split": ["val"] * len(validation_ids),
            }
        ),
        manifests / "validation_samples.parquet",
    )
    pq.write_table(
        pa.table({"sample_id": validation_ids, "label": [1] * len(validation_ids)}),
        manifests / "validation_labels.parquet",
    )
    pilot_ids = validation_ids[:5]
    (manifests / "pilot_100.json").write_text(
        json.dumps(
            {
                "sample_count": len(pilot_ids),
                "samples": [{"sample_id": sample_id} for sample_id in pilot_ids],
            }
        )
    )
    # A malformed test manifest is a canary: tuning succeeds only if it never
    # tries to inspect test data.
    (manifests / "test_samples.parquet").write_bytes(b"must-not-be-read")
    runner = tmp_path / "fake_run_method.py"
    runner.write_text(FAKE_RUNNER)
    return run_dir, runner


def test_coordinate_search_bound_is_not_cartesian() -> None:
    from dataclasses import asdict

    config = asdict(AnalyticGraspConfig())
    space = AnalyticGraspConfig.validation_search_space()
    bound = coordinate_trial_upper_bound(config, space)
    cartesian = 1
    for values in space.values():
        cartesian *= len(values)
    assert bound == 9
    assert bound < cartesian


def test_selection_order_uses_j1_j5_no_grasp_runtime_then_simplicity() -> None:
    base = {
        "corrected_j_at_1": 0.2,
        "corrected_j_at_5": 0.3,
        "no_grasp_rate": 0.1,
        "p95_latency_seconds": 0.4,
        "complexity": 1,
        "config_sha256": "a",
    }
    assert selection_key({**base, "corrected_j_at_1": 0.21}) < selection_key(base)
    assert selection_key({**base, "corrected_j_at_5": 0.31}) < selection_key(base)
    assert selection_key({**base, "no_grasp_rate": 0.09}) < selection_key(base)
    assert selection_key({**base, "p95_latency_seconds": 0.39}) < selection_key(base)
    assert selection_key({**base, "complexity": 0}) < selection_key(base)


def test_tuner_runs_validation_only_preserves_outputs_and_resumes(tmp_path: Path) -> None:
    run_dir, runner = _write_run(tmp_path)
    venv_python = Path(pa.__file__).resolve().parents[4] / "bin" / "python"
    tuner = AnalyticTuner(
        run_dir=run_dir,
        runner=runner,
        python=venv_python,
        pilot_manifest=run_dir / "manifests" / "pilot_100.json",
        max_full_configs=3,
        full_workers=3,
    )
    selected = tuner.run()

    search_space = json.loads(
        (run_dir / "analytic" / "validation_search_space.json").read_text()
    )
    selected_config = json.loads(
        (run_dir / "analytic" / "selected_config.json").read_text()
    )
    results = (run_dir / "analytic" / "validation_search_results.csv").read_text()
    invocations = (run_dir / "fake_runner_invocations.txt").read_text().splitlines()
    assert search_space["split"] == "validation"
    assert search_space["pilot_trial_upper_bound"] == 9
    assert selected["config_sha256"] == canonical_config_hash(selected_config)
    assert selected_config["component_policy"] == "largest"
    assert selected_config["max_axis_depth_jump_m"] == 0.03
    assert len([line for line in invocations if line.endswith(",pilot")]) == 9
    assert len([line for line in invocations if line.endswith(",full")]) == 3
    assert set(line.split(",")[0] for line in invocations) == {"validation"}
    assert "corrected_j_at_1" in results

    outputs = list((run_dir / "validation" / "analytic_search").glob("*/*/COMPLETE.json"))
    assert len(outputs) == 12
    for complete in outputs:
        output_dir = complete.parent
        assert (output_dir / "per_sample_predictions.parquet").is_file()
        assert (output_dir / "per_candidate_predictions.parquet").is_file()

    # A second invocation validates and reuses COMPLETE outputs instead of
    # mutating them or launching the runner again.
    rerun = AnalyticTuner(
        run_dir=run_dir,
        runner=runner,
        python=venv_python,
        pilot_manifest=run_dir / "manifests" / "pilot_100.json",
        max_full_configs=3,
        full_workers=3,
    )
    rerun.run()
    assert (run_dir / "fake_runner_invocations.txt").read_text().splitlines() == invocations
    assert (run_dir / "manifests" / "test_samples.parquet").read_bytes() == b"must-not-be-read"


def test_rejects_pilot_ids_outside_validation(tmp_path: Path) -> None:
    run_dir, runner = _write_run(tmp_path)
    pilot = run_dir / "manifests" / "pilot_100.json"
    pilot.write_text(
        json.dumps({"sample_count": 1, "samples": [{"sample_id": "test-only"}]})
    )
    try:
        AnalyticTuner(
            run_dir=run_dir,
            runner=runner,
            python=Path(sys.executable),
            pilot_manifest=pilot,
            max_full_configs=3,
            full_workers=1,
        )
    except ValueError as error:
        assert "outside validation" in str(error)
    else:
        raise AssertionError("out-of-validation pilot ID was accepted")
