# Selective SAM 3 visual-grounding audit

## Authoritative local baseline

- Pipeline: five-stage repeated-FiLM HiFi-CS (`extended_film=true`, `hierarchical_film=true`).
- Checkpoint SHA-256: `b19a649326384ba4524295cd100b22e54cb9ea615174229fc310fbd6bc898601`.
- Frozen unique-test manifest SHA-256: `915e002bf31f044419db7140bc1145b8fcc45f9a6b35259637d923c6d4610409`.
- Formal denominator: **7,675 referring expressions from 325 RGB frames**.
- Independently recomputed mIoU: **0.807413711517**.
- P@50=6997/7675 (91.166124%), P@60=6818/7675 (88.833876%), P@70=6475/7675 (84.364821%), P@80=5655/7675 (73.680782%), P@90=3363/7675 (43.817590%).

The audit rebuilt every 352x352 binary prediction from the stored probability
map with `probability >= 0.5`, verified its stored model-resolution mask,
verified nearest-neighbour mapping to the 480x640 native coarse mask, and then
recomputed float32 per-sample IoU from the frozen processed GT mask. The result
matches the authoritative baseline exactly.

## Paper/local protocol comparison

The paper values are retained only as a **paper numeric reference**. The local
protocol is the official OCID-VLG `unique` train/validation/test split
(26,295/3,778/7,675 expressions) with an explicit scene-disjoint validation
set and a 352x352 Standard evaluator. The paper describes a 70/30 split, while
the released preparation script consumes pre-existing `unique` train/test
files and does not publish enough information to prove that this local split is
the exact paper split. Exact paper reproduction is therefore not claimed.

## Recovery required to equal the paper numeric reference

- Total IoU sum required for mIoU: **577.054764**.
- P@50: needs 117 additional threshold crossings.
- P@60: needs 253 additional threshold crossings.
- P@70: needs 550 additional threshold crossings.
- P@80: needs 1,229 additional threshold crossings.
- P@90: needs 3,024 additional threshold crossings.

## Leakage boundary

This baseline audit may read GT. Prompt generation, trigger inference,
candidate selection, and formal mask writing are implemented in separate
prediction-only modules. Formal test GT may be loaded only after output masks
are frozen and checksummed.
