# CROG Gemini Evidence V1 — Audit

## Baseline reproduction

Phase 0 independently sorts every test candidate by full-precision `q_raw` descending with `candidate_id` as the stable tie-break. It does not trust row order as the only ranking implementation.

| Quantity | Count | Rate |
|---|---:|---:|
| Test expressions | 17,749 | — |
| Frozen candidates | 88,745 | exactly 5/sample |
| Legacy q-only | 14,768 | 83.2046876% |
| Legacy Oracle@5 | 16,129 | 90.8727252% |
| Corrected q-only | 15,840 | 89.2444645% |
| Corrected Oracle@5 | 16,746 | 94.3489774% |

The canonical frozen test identity stream SHA-256 is `34e773b06119619bd45771f8e000c317ecc768d51a132ef6dc7b40e5f3faa36c`. There is one exact test q tie: `multiple:test:00017622`, candidates 3/4, `q_raw=0.7320538759231567`; the candidate-ID tie-break agrees with the frozen order.

## Split audit

The scene/group-separated manifest contains development 53,431, calibration 9,790, validation 8,669, and formal test 17,749 expressions. Pairwise overlap is zero for stable sample ID, frame, scene, RGB SHA-256, depth SHA-256, and combined RGB-D SHA-256. Sequence overlap is reported by the legacy split policy but is not used to claim sequence-disjoint official splits.

## Source and leakage audit

Full-resolution CROG prediction maps did not previously exist (`dense_maps_saved=false`), so they were regenerated from the frozen checkpoint. Existing V2 32×32 aligned crops and latent/critic/SetRank/gate outputs are not Gemini inputs. The strict forward path passes only image and tokenized text.

For the development smoke cohort, 10 boards and 50 candidate-evidence rows were exported. Frozen-forward identity maximum difference is 0.0. Each board is 2200×2200, uses one fixed layout and deterministic A–E mapping, and includes no GT/evaluation panel. Serialized metadata and request manifests pass the recursive forbidden-field scan.

The referring expression is XML-escaped inside `<referring_expression>` and treated as untrusted data; it cannot close the tag and inject a replacement system instruction.

## Immutable V2 audit

Before and after implementation, the complete 13 GiB V2 tree digest is:

`2c0791359d252af73987d180ae1852ac86d3200ebfe2ac93d74a18ad4581481c`

The old V2 frozen lock also verifies against its checkpoint, split, evaluator, code fingerprint, and model artifacts. The new experiment does not modify `failure_analysis/reranking/`, `failure_analysis/reranking_v2/`, `model/`, `utils/`, the old result roots, checkpoint, or prior V2 documentation.

## API and capability audit

- Python: 3.12.12; MPS available.
- Installed/pinned SDK: `google-genai==2.16.0`; `pip check` passes.
- Exact Flash model: `gemini-3.6-flash`, stable and publicly documented.
- Exact ER2 model: `gemini-robotics-er-2-preview`, enumerated by the official v1beta Interactions schema; public model card, pricing, and project access were not published at audit time, so access/capabilities remain an empirical smoke question.
- Interactions default storage is overridden with `store=False`.
- Live smoke credentials were loaded from the ignored, mode-600 `.env` into process environment only. The key value and budget value are absent from source, config, manifests, cache, logs, and reports.
- Resolved smoke responses: 20 (same 10 development samples × two exact models). There were 26 paid attempts: six development-time token truncations and 20 valid responses. The truncated attempts remain in the audit cache.
- A 512-token ceiling left only four visible tokens after medium thinking; 1536 also truncated, and two Flash cases truncated at 3072. Targeted 4096 retries completed both, so 4096 is the development default pending pilot confirmation.

The key originally pasted in conversation was not copied into any command or artifact. Live work used a locally supplied environment value; secret scans and SQLite schema inspection found no persisted key or request headers.

## Scientific scope

J@1 measures consistency with annotated OCID-VLG 2D grasp rectangles. It is not physical grasp success, force closure, collision-free reachability, lifting success, a 6-DoF pose, or trajectory success.

This is a post-hoc benchmark extension on a previously used test split. The Gemini prompt, renderer, thresholds and primary method must nevertheless be frozen using development, calibration and validation data before any Gemini formal-test run.
