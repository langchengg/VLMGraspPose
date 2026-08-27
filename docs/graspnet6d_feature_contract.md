# Runtime-only 6-DoF feature contract

The machine-readable source of truth is
`configs/graspnet6d/feature_schema_6d_v1.json`. Features are computed for every
member of a frozen VGN pool and may change scores/order only; candidate IDs,
translation, rotation, width, GraspNet height/depth, and membership are immutable.

## Allowed sources

- RGB and depth available at inference time;
- the selected grounding mask/probability for the named condition;
- the scene point cloud, camera intrinsics, and table transform;
- frozen VGN score, rank, pose, width, and voxel index;
- deterministic runtime geometric proxies.

In `oracle_gt_mask`, mask-derived features are allowed only because the condition
is explicitly an oracle-grounding counterfactual. In predicted-mask tracks the
same feature code consumes the predicted probability/mask. Probability-only GT
fields are represented as documented constants or missing values with indicators,
not as hidden condition labels.

## Prohibited sources

The model feature matrix must never contain target/distractor object ID, GT object
pose or mesh, official collision label, friction score, relevance, target success,
split outcome, or evaluator association. Scene/group/candidate identifiers are
retained only as non-feature keys. Column validation is deny-by-default against
these supervision and identity tokens.

## Feature families

The schema groups runtime values into native confidence, 2-D target support,
3-D target support, physical width, continuous rotation-6D and relative
orientation, collision-risk proxies, local geometry, and grounding quality.
Metric 3-D distances are metres and angles radians. Projected boundary and
centroid distances are divided by the image diagonal (dimensionless); occupancy,
coverage, support, entropy and ratios are dimensionless. Division uses a
configured epsilon and adds a missing/invalid indicator rather than dropping
rows.

Official collision and runtime collision-risk are distinct: occupancy/table
features are estimates from the observed scene and are model inputs; official
collision is evaluator-only supervision/measurement.

## Missingness and preprocessing

Median imputation and missing-indicator selection are fit on training scenes
only. Validation and test call transform only. An all-missing training column is
an error unless its schema explicitly defines a constant for an oracle condition.
Groups remain contiguous for LightGBM, and the sum of group sizes must equal the
number of rows exactly.

## Ablations

`A1` uses only native confidence; `A2` all runtime families; `A3` drops target
support and grounding quality; `A4` drops collision risk; `A5` drops local
geometry; `A6` retains the raw continuous rotation-6D but drops relative
orientation. Ablation selectors operate from this schema so a result cannot be
hand-assembled with an accidental label column.
