# Failure Taxonomy for CROG-Style Language-Guided Grasping

## 1. grounding_failure

The model attends to the wrong object or wrong image region.

Symptoms:
- predicted mask has low IoU with the target mask
- predicted grasp is outside the target mask
- predicted heatmap activates on a distractor object

## 2. localization_failure

The model identifies the right object approximately but places the grasp centre incorrectly.

Symptoms:
- mask IoU may be acceptable
- grasp centre is far from valid grasp annotations
- grasp point is on an object edge or background

## 3. orientation_failure

The predicted grasp centre is near the target object, but the gripper angle is wrong.

Symptoms:
- centre error is small
- angle error is too large
- J@1 fails due to orientation mismatch

## 4. width_failure

The grasp location and orientation are plausible, but the predicted gripper opening is wrong.

Symptoms:
- width is too small or too large compared with labelled grasps
- predicted grasp would not enclose the object properly

## 5. language_ambiguity_failure

The instruction contains spatial, colour, size, or attribute ambiguity.

Symptoms:
- multiple similar objects are present
- prompt uses cues such as left, right, behind, front, middle, small, large, colour, or relative position
- the model selects a plausible but wrong instance

## 6. clutter_occlusion_failure

The target is partially occluded or close to other objects.

Symptoms:
- high scene clutter
- nearby objects overlap the target region
- predicted grasp visually collides with neighbouring objects

## 7. top1_ranking_failure

The top-1 prediction fails, but another top-k grasp candidate succeeds or appears more plausible.

Symptoms:
- J@1 fails but J@Any succeeds
- the final selected grasp is not the semantically or geometrically best candidate
- this category directly motivates semantic-geometric re-ranking

## 8. dataset_or_metric_edge_case

The failure may come from annotation ambiguity or strict metric behaviour.

Symptoms:
- predicted grasp appears visually reasonable
- metric marks it as failure
- multiple valid grasps may exist but only a subset are annotated

## Notes

This taxonomy is diagnostic. It should be used to structure dissertation discussion, not to claim that each failure can be automatically corrected by re-ranking.
