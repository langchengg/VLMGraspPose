# AnyGrasp verified subset report

- Geometry/data-contract status: **DONE**
- Selected samples: 20
- Geometry-verified samples: 20
- Human visual target-correspondence pass: 13
- Human visual target-correspondence fail/unsafe: 7
- AnyGrasp inference ran: false

## Group counts

- highest_iou: 5
- nearest_median: 5
- lowest_iou: 5
- diverse_clutter_spatial: 5

## Selected samples and geometric diagnostics

| Group | Sample ID | IoU | Fit p95 px | Projection p95 px | Target valid points | Valid fraction | Clutter instances |
|---|---|---:|---:|---:|---:|---:|---:|
| highest_iou | `q0017427_f1ae5b4a8a6fa0fd` | 0.970635 | 0.4980951359734599 | 0.4980951359734079 | 28266 | 0.999399 | 7 |
| highest_iou | `q0013877_3118bbdd8962642e` | 0.970120 | 0.5932915638691392 | 0.5932915638691392 | 18759 | 1.000000 | 14 |
| highest_iou | `q0017392_40a230ec469fb66f` | 0.969522 | 0.4980951359734599 | 0.4980951359734079 | 28272 | 0.999399 | 7 |
| highest_iou | `q0017405_293b32adb8d63846` | 0.969456 | 0.4980951359734599 | 0.4980951359734079 | 28324 | 0.999435 | 7 |
| highest_iou | `q0013837_5ad8fa6192623b45` | 0.968588 | 0.5932915638691392 | 0.5932915638691392 | 18793 | 1.000000 | 14 |
| nearest_median | `q0005827_36618721c3cfb831` | 0.862946 | 0.6047151813205266 | 0.6047151813205266 | 10746 | 0.959464 | 20 |
| nearest_median | `q0006653_a291fe46e71d0766` | 0.862963 | 0.741438075953281 | 0.741438075953281 | 2045 | 1.000000 | 9 |
| nearest_median | `q0004968_ad4915d78a4cec34` | 0.862968 | 0.6397031645064424 | 0.6397031645064424 | 5906 | 1.000000 | 14 |
| nearest_median | `q0015435_5c57f98ec5c51500` | 0.862971 | 0.6102795445000864 | 0.6102795445001623 | 2262 | 0.958069 | 7 |
| nearest_median | `q0017618_dd176e8af2caddfd` | 0.862972 | 0.5549260458158383 | 0.5549260458158383 | 3693 | 0.904039 | 2 |
| lowest_iou | `q0000184_cd814c91120a1dd7` | 0.000000 | 0.64594330651104 | 0.6459433065110071 | 6469 | 0.972343 | 16 |
| lowest_iou | `q0000239_691644f17e011e50` | 0.000000 | 0.64594330651104 | 0.6459433065110071 | 9019 | 0.984177 | 16 |
| lowest_iou | `q0000253_e9c86f8bf16b81fc` | 0.000000 | 0.6201450244687833 | 0.6201450244687833 | 2666 | 0.997754 | 20 |
| lowest_iou | `q0000292_4c3fb641aa6d19f5` | 0.000000 | 0.6201450244687833 | 0.6201450244687833 | 2450 | 0.995935 | 20 |
| lowest_iou | `q0000327_9cb7ae108a822c78` | 0.000000 | 0.6201450244687833 | 0.6201450244687833 | 2384 | 1.000000 | 20 |
| diverse_clutter_spatial | `q0001442_617a5f0c3991136c` | 0.863913 | 0.6740442948672025 | 0.6740442948672037 | 2058 | 1.000000 | 20 |
| diverse_clutter_spatial | `q0006183_26b145d5bba6728a` | 0.238060 | 0.6360039723139504 | 0.6360039723139504 | 2598 | 0.999615 | 20 |
| diverse_clutter_spatial | `q0009634_f6698b510aa44cc5` | 0.346005 | 0.5198426689202597 | 0.5198426689202597 | 5482 | 0.954387 | 20 |
| diverse_clutter_spatial | `q0000805_51738df666e3ac4d` | 0.944918 | 0.6641050364192815 | 0.6641050364192815 | 10836 | 1.000000 | 19 |
| diverse_clutter_spatial | `q0002095_602fbe177b9c09d8` | 0.933371 | 0.636661116761566 | 0.636661116761566 | 8940 | 1.000000 | 19 |

## Human visual target-correspondence audit

This audit inspected the RGB, query, ground truth, probability map, binary prediction, and GT/prediction overlay for every selected sample. A failed semantic check does not invalidate the file format or RGB-depth geometry; it means the predicted-mask bundle is unsafe to treat as the requested object for grasping.

| Group | Sample ID | Visual result | Evidence |
|---|---|---|---|
| highest_iou | `q0017427_f1ae5b4a8a6fa0fd` | pass | Prediction closely follows the queried cereal box. |
| highest_iou | `q0013877_3118bbdd8962642e` | pass | Prediction closely follows the queried keyboard. |
| highest_iou | `q0017392_40a230ec469fb66f` | pass | Prediction closely follows the green/red cereal box. |
| highest_iou | `q0017405_293b32adb8d63846` | pass | Prediction closely follows the queried Chocos box. |
| highest_iou | `q0013837_5ad8fa6192623b45` | pass | Prediction closely follows the queried keyboard. |
| nearest_median | `q0005827_36618721c3cfb831` | pass | Prediction substantially overlaps the Mega Pack cereal target. |
| nearest_median | `q0006653_a291fe46e71d0766` | pass | Prediction substantially overlaps the marker specified relative to the apple. |
| nearest_median | `q0004968_ad4915d78a4cec34` | pass | Prediction substantially overlaps the queried rice bag. |
| nearest_median | `q0015435_5c57f98ec5c51500` | pass | Prediction substantially overlaps the food can left of the towel. |
| nearest_median | `q0017618_dd176e8af2caddfd` | pass | Prediction substantially overlaps the queried noodles package. |
| lowest_iou | `q0000184_cd814c91120a1dd7` | fail | A different cereal-box instance is selected; prediction and GT are disjoint. |
| lowest_iou | `q0000239_691644f17e011e50` | fail | The other tissue package is selected rather than the relationally specified instance. |
| lowest_iou | `q0000253_e9c86f8bf16b81fc` | fail | The right-ball target is missed and a different object is selected. |
| lowest_iou | `q0000292_4c3fb641aa6d19f5` | fail | A distractor mug is selected instead of the mug behind the shampoo. |
| lowest_iou | `q0000327_9cb7ae108a822c78` | fail | A distractor mug is selected instead of the rear-right referent. |
| diverse_clutter_spatial | `q0001442_617a5f0c3991136c` | pass | Prediction substantially overlaps the right shampoo product. |
| diverse_clutter_spatial | `q0006183_26b145d5bba6728a` | fail | Prediction is fragmented across multiple objects and only partly covers the queried apple. |
| diverse_clutter_spatial | `q0009634_f6698b510aa44cc5` | fail | Prediction includes the queried soft-drink can and an adjacent distractor can, creating a wrong-object grasp risk. |
| diverse_clutter_spatial | `q0000805_51738df666e3ac4d` | pass | Prediction closely follows the right food-box product. |
| diverse_clutter_spatial | `q0002095_602fbe177b9c09d8` | pass | Prediction closely follows the left towel. |

## Caveats

- The automated tool verifies geometry and bundle integrity; the separate human audit above records semantic target correspondence.
- Sparse zero/invalid depth pixels inside a predicted mask are reported through the valid-point fraction; a sample is blocked only when it has no reconstructable target point.
- Factory calibration and camera/robot extrinsics are unavailable; stored intrinsics are effective pinhole fits derived from organized PCDs.
- Full-DoF AnyGrasp generation remains blocked until the licensed SDK, checkpoint, compatible CUDA runtime, and required extrinsic frame mapping are available.
- No AnyGrasp inference was run and no full-DoF grasp poses were generated.
