# Real-robot evaluation protocol

## Status and safety boundary

Physical execution is disabled in this repository. There is no ROS/MoveIt
integration, reviewed robot driver, base-to-camera transform, workspace
collision model, or emergency-stop integration. `DryRunExecutor` validates an
immutable request but cannot send motion commands. The scripts in this document
prepare and record experiments; they do not control a robot.

A hardware executor must be implemented and reviewed separately for the exact
robot, gripper, controller, calibration and workspace. No command in this
repository should be connected to a physical robot merely by replacing a host
name or topic.

Until manually verified physical trial logs exist, the required result is:

```json
{
  "real_robot_grasp_success_rate": null,
  "reason": "no physical robot execution logs"
}
```

Offline VGN quality, candidate coverage and PyBullet success are separate
measurements and may never fill this field.

## Preregistered trial design

Prepare at least 50 trials; 100 are recommended. The candidate table must
explicitly provide:

- `sample_id` and one language `instruction`;
- object category and geometric shape;
- clutter level;
- predicted-mask IoU;
- official processed VGN quality;
- grasp approach direction;
- seen/unseen status.

The deterministic selector round-robins across the declared categorical strata
and covers quality/IoU ranges within each stratum:

```bash
python -m scripts.prepare_real_robot_trials \
  --input outputs/robot_trial_candidates.csv \
  --output outputs/real_robot_trials \
  --count 100 \
  --success-definition lift_10cm_hold_3s \
  --seed 42
```

Every preregistered trial permits exactly one query, one grounding result, one
top-1 grasp generation and one physical execution. A failed pose must not be
modified and counted as a retry of the same trial.

## Fixed success definition

Choose one definition before execution and retain it for the entire manifest:

1. `lift_10cm_hold_3s`: the **target object** is lifted at least 0.10 m above
   its support and held for at least 3 seconds; or
2. `placed_in_bin`: the **target object** is deposited in the predefined bin.

`wrong_object=true` is a failure under either definition. The record command
checks that the manually reported success agrees with the preregistered rule.

## Required pre-execution review

Before introducing a hardware executor, a qualified operator must verify and
sign off all of the following outside this repository:

- robot and gripper model/limits, payload, firmware and controller mode;
- calibrated camera-to-base transform and its uncertainty;
- grasp-frame convention and gripper width limits;
- table, bin, self-collision and workspace collision geometry;
- reachable pregrasp, grasp and retreat trajectory through IK and planning;
- speed/force limits, protective stop and emergency stop;
- exclusion zone, operator position and recovery procedure;
- immutable trial ID, instruction, mask and selected top-1 pose.

Dry-run validation example:

```python
from src.robot import DryRunExecutor, ExecutionRequest

result = DryRunExecutor().execute(request)
assert result.physical_execution_attempted is False
assert result.physical_success is None
```

Passing a dry run is not authorization to move hardware.

## Trial evidence and manual record

Each physical trial directory must contain:

```text
before_rgb.png
predicted_mask.png
top1_overlay.png
pregrasp.png
closure.png
lift.png
after.png
trial.mp4
```

Record all labels explicitly: `grounding_correct`, `target_contact`,
`object_lifted`, `held_for_3s`, `placed_in_bin`, `collision`, `wrong_object`,
`slip`, `planning_failure`, and `execution_failure`. The annotator identity and
whether labels came from a human, sensor, or both are mandatory. Recording
requires the literal confirmation token
`I_CONFIRM_PHYSICAL_EXECUTION_OCCURRED`; this is an evidence gate, not a robot
motion gate.

```bash
python -m scripts.record_real_robot_trial \
  --trials-manifest outputs/real_robot_trials/trials_manifest.jsonl \
  --trial-id robot_trial_0000 \
  --output outputs/real_robot_trials \
  --artifacts-dir /reviewed/evidence/robot_trial_0000 \
  --annotator OPERATOR_ID \
  --label-source human_and_sensor \
  --confirm-physical-execution I_CONFIRM_PHYSICAL_EXECUTION_OCCURRED \
  --success \
  --grounding-correct --target-contact --object-lifted --held-for-3s \
  --no-placed-in-bin --no-collision --no-wrong-object --no-slip \
  --no-planning-failure --no-execution-failure
```

Trial records are write-once. Correcting an annotation requires a documented
review procedure; rerunning this command will not overwrite a record.

## Metrics

Summarize only the physical records:

```bash
python -m scripts.summarize_real_robot_trials \
  --trials-root outputs/real_robot_trials \
  --trials-manifest outputs/real_robot_trials/trials_manifest.jsonl \
  --output outputs/real_robot_trials/summary
```

The summary reports numerator, denominator, rate and Wilson 95% interval for:

- `real_robot_grasp_success_rate`: successful target-object physical
  executions divided by attempted physical trials;
- `end_to_end_real_success_rate`: correct grounding and successful target
  grasp divided by attempted language trials;
- `conditional_grasp_success_given_correct_grounding`: successful target
  grasps divided by grounding-correct attempted trials.

Preregistered but unexecuted rows are reported separately and never counted as
physical attempts. If no genuine physical record exists, all real-robot rates
remain null. PyBullet results remain named `simulated_grasp_success_rate` and
`simulated_percent_cleared`.

## Provenance

The related simulation protocol is the official VGN CoRL 2020 clutter-removal
implementation at commit
`d7af0622433f52ae88ebe81533f12b46b33e951a`:

- https://github.com/ethz-asl/vgn/tree/d7af0622433f52ae88ebe81533f12b46b33e951a
- https://proceedings.mlr.press/v155/breyer21a/breyer21a.pdf

That protocol is useful complementary evidence, but it is not a substitute for
the physical procedure above.

