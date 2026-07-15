# HiFi-CS failure-case analysis

This analysis covers the 20 lowest-IoU formal test examples. All 20 have IoU 0.0. Categories were assigned only after inspecting each original RGB image, exact query, original-resolution GT, probability map, binary prediction, and overlay. Low IoU alone was not used to infer a semantic cause.

## Primary categories

| Category | Count | Percentage | Average IoU |
|---|---:|---:|---:|
| spatial_relation_misunderstanding | 19 | 95% | 0.000 |
| wrong_object_instance | 1 | 5% | 0.000 |

The dominant visible pattern is selection of a plausible object of the requested class but the wrong relational instance. Examples include left/right cereal boxes, mugs relative to balls or shampoo, nearest/rear food bags, and soda cans relative to balls or fruit. The single non-relational query, `banana`, selects a different elongated object.

Secondary labels are conservative. Most spatial failures are also `wrong_object_instance`. Two predictions (`q0000347...` and `q0000935...`) visibly contain multiple disconnected distractor regions and are marked `over_segmentation` secondarily. No colour, shape, ambiguity, perspective, or annotation-edge label was assigned without direct visual evidence.

## Downstream AnyGrasp risks

- **Wrong-object grasps:** all 20 predicted masks are disjoint from the GT. A target-cropped point cloud would therefore direct AnyGrasp to a distractor rather than the referred object.
- **Over-segmentation:** the two multi-region predictions can admit points from several objects, allowing a high-scoring grasp on the wrong instance even if part of the desired area were present.
- **Incomplete target point clouds:** fragmented predictions such as the small-ball and relation-heavy marker cases can leave too few coherent object points. This can suppress valid grasps or bias pose/width estimation.
- **Clutter sensitivity:** many failures occur in scenes with several same-category or geometrically similar instances. The mask is often locally crisp, but its semantic ownership is wrong; downstream geometric quality cannot repair that upstream identity error.

The detailed evidence table is `runs/hifics_ocidvlg_20260711_112921/evaluation/failure_cases.csv`. This report does not claim that these 20 tied zero-IoU cases represent the frequency of categories over every failure in the 7,675-sample test set.
