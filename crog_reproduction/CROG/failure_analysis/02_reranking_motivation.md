# Re-ranking Motivation from Failure Cases

This analysis is diagnostic, not an improvement claim. It uses the reproduced CROG outputs to identify where a semantic-geometric re-ranking module is most defensible.

## What Better Training Alone May Not Solve

- Dataset or metric edge cases can remain ambiguous even when the predicted grasp is visually plausible.
- Top-1 ranking failures are selection errors: the exported top-k heatmap peaks include an accepted grasp, but the first selected grasp fails.
- Clutter and occlusion can require explicit clearance and collision-sensitive features rather than only stronger mask supervision.

## Ranking-Related Evidence

- Samples analysed: 17749
- J@1 fails while J@Any succeeds: 1361 (7.67%)
- Selected ranking-motivation cases: 10

## Score-Term Mapping

| failure type | relevance to re-ranking | possible score terms |
|---|---|---|
| grounding_failure | May help only when the correct region remains in the candidate set. | M(g_i), Sem(g_i) |
| localization_failure | Candidate centres can be favoured by target-mask or target-cloud proximity. | M(g_i), distance-to-mask, Q(g_i) |
| orientation_failure | Needs candidate-level orientation quality, not just mask quality. | geometric orientation features, Q(g_i) |
| width_failure | Needs compatibility between predicted gripper opening and object extent. | width compatibility, detector confidence |
| clutter_occlusion_failure | Clearance and collision checks are directly relevant. | Clear(g_i), Coll(g_i) |
| top1_ranking_failure | Direct target for re-ranking when a correct top-k grasp exists. | Q(g_i), M(g_i), Sem(g_i), Clear(g_i), Coll(g_i) |
| dataset_or_metric_edge_case | Should be treated as a limitation rather than promised as solvable. | analysis flag, not an optimisation target |

## Observed Failure Counts

- top1_ranking_failure: 1361
- grounding_failure: 631
- width_failure: 598
- orientation_failure: 200
- dataset_or_metric_edge_case: 185
- localization_failure: 116
- clutter_occlusion_failure: 15

## Conclusion

The strongest motivation for semantic-geometric re-ranking is the subset where J@Any succeeds but J@1 fails. In those samples, the model has produced at least one acceptable grasp candidate, so changing the final selection criterion could plausibly improve the selected grasp without changing the CROG architecture.
