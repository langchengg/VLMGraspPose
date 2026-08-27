# GraspNet–VGN coordinate contract

Status: implemented algebra; **real-data gripper conversion validation is still
required before formal candidate evaluation**.

## Notation

All geometry uses SI metres and column vectors. `T_A_B` maps coordinates from
frame B into frame A:

```text
p_A = T_A_B @ [p_B.x, p_B.y, p_B.z, 1]
T_A_C = T_A_B @ T_B_C
R_A_B = T_A_B[:3, :3]
t_A_B = T_A_B[:3, 3]
```

A grasp pose stores the gripper-frame basis vectors as the **columns** of its
rotation matrix. Homogeneous transforms must have bottom row `[0, 0, 0, 1]`;
rotations must satisfy `R.T @ R = I` and `det(R) = +1` within tolerance.

## Frames

| Symbol | Definition | Axes / handedness |
|---|---|---|
| I | image pixels `(u,v)` | `u` right, `v` down; not a metric 3-D frame |
| C | depth camera at the selected view | OpenCV pinhole: x right, y down, z forward; right-handed |
| C0 | the scene's first camera | defined by GraspNet `camera_poses.npy` |
| T | table-aligned scene frame | obtained exactly as the official API composes `cam0_wrt_table @ camera_pose`; right-handed |
| L | target-local VGN workspace | axes parallel to T; origin is the deterministic target-centred 0.30 m cube origin |
| GV | VGN gripper | +z approach/finger direction, +y jaw-closing axis, +x completes the right-handed basis |
| GG | GraspNet gripper | +x approach, +y jaw-closing axis, +z gripper height axis |

The official API calls each `camera_poses[frame]` the camera pose relative to
the first frame and applies `cam0_wrt_table @ camera_pose` before transforming
camera points to the table-aligned frame. Consequently this route defines:

```text
T_T_C = cam0_wrt_table @ camera_poses[frame]
T_C_T = inverse(T_T_C)
```

This direction is tested by round trip and, once data exists, by reprojection
and official API parity. File names alone are not used to infer direction.

## Projection

For positive depth `z` and intrinsics `(fx, fy, cx, cy)`:

```text
u = fx * x / z + cx
v = fy * y / z + cy
x = (u - cx) * z / fx
y = (v - cy) * z / fy
```

Depth is converted to metres using each annotation's `factor_depth`. A fixed
fallback depth scale is prohibited. Data-backed coordinate acceptance requires
median sampled reprojection error below 1 px and p95 below 2 px.

## Target-local workspace

The target mask is used to recover target depth points and choose a translation
for L. It does not rescale the object and does not delete table or neighbouring
geometry. `R_T_L = I`; the 0.30 m cube preserves VGN's 7.5 mm voxel size at
resolution 40 and its 0.03 m truncation distance. Single-view TSDF integration
uses the complete depth image cropped only by the local physical volume.

## VGN and GraspNet grippers

The source-supported axis hypothesis is:

```text
R_C_GG = R_C_GV @ R_GV_GG
R_GV_GG columns = [GV +z, GV +y, GV -x]
R_GV_GG = [[ 0, 0,-1],
            [ 0, 1, 0],
            [ 1, 0, 0]]
```

This is a proper rotation (`det=+1`) and maps VGN approach/closing to GraspNet
approach/closing. It is deliberately marked a **hypothesis**, not an accepted
formal conversion, until ten real scenes pass projected gripper visualisation
and the frozen-candidate adapter agrees with the official scene evaluator.

GraspNet's 17-value grasp row is:

```text
[score, width, height, depth, R(9 row-major), translation(3), object_id]
```

Collision geometry and the Dex-Net contact centre use `depth`; VGN produces a
pose and jaw width but no GraspNet `height` or `depth`. Therefore both values
are explicit, frozen candidate fields. They may not be silently set to zero or
borrowed from an unrelated gripper.  The checked-in proposal uses GraspNet's
fixed collision height `0.02 m` and VGN's physical finger depth `0.05 m`.
Those values and the axis matrix above are hypotheses to render and review,
not self-validating facts: formal candidate generation remains blocked until
the train/validation real-data gallery and evaluator-parity gates accept the
complete translation, axis, height, and depth semantics.

## Required acceptance checks

The unit tests cover projection, inverse transforms, SO(3), deterministic
serialization, and workspace bounds. Formal acceptance additionally requires:

1. real GraspNet depth/mask/point reprojection thresholds;
2. camera–table–camera and local–table–local round trips on real files;
3. target centroid inside the fixed workspace;
4. candidate centre projects into the expected image region;
5. ten audit figures with camera/table axes and gripper approach arrows;
6. per-candidate association, collision, friction, and validity parity with an
   official `eval_scene` example without changing frozen candidate membership.

Sources: [graspnetAPI](https://github.com/graspnet/graspnetAPI),
[VGN](https://github.com/ethz-asl/vgn).
