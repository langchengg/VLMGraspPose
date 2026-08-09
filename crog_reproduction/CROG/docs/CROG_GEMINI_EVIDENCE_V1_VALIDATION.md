# CROG Gemini Evidence V1 — Validation Status

## Completed offline validation

- Phase 0 exact baseline, candidate count/order, tie-break, split and hash audits: passed.
- Frozen no-GT CROG replay for 10 development samples: passed.
- Candidate identity after replay: maximum difference 0.0.
- M/Q/W probability bounds after interpolation handling: `[0,1]`.
- Board dimensions/layout/layer audit: passed, 2200×2200.
- Deterministic A–E mapping and reverse mapping: passed.
- Strict response schema, fallbacks, retry/no-retry, redaction, SQLite idempotency, budget guard, Safe/Consensus and lock rules: passed.
- Legacy/Corrected independent baseline recomputation: passed.
- Whole old V2 tree digest and frozen lock: unchanged/passed.

Test commands and results:

```text
.venv/bin/python -m pytest -q tests/test_crog_gemini_evidence_v1.py
60 passed in 37.06s

.venv/bin/python -m pytest -q tests/test_reranking_candidates.py tests/test_corrected_evaluator.py tests/test_reranking_v2.py
57 passed, 1 non-failing PyTorch warning in 4.44s

.venv/bin/python -m pip check
No broken requirements found.
```

## Invalidated artifact disclosure

Visual inspection found small negative candidate mask/jaw values in the first 10-sample export. Cause: sigmoid probabilities were bicubic-upsampled, which can overshoot the closed probability interval. That artifact is marked `INVALIDATED.json` and is not a request source. The exporter now clips M/Q/W probabilities to `[0,1]`; all 50 replacement candidate rows pass bounds checks. Raw logits remain unmodified.

## API smoke status

Phase B access smoke is complete for the same 10 development samples and both exact model IDs. ER2 produced 10/10 valid responses at a 3072-token ceiling. Flash produced 8/10 at 3072; the other two were JSON truncations caused by 2402/2428 thought tokens and were recovered with targeted 4096-token requests. The resolved cohort is therefore 20/20 valid with zero model abstentions and no technical fallback in the final per-sample decisions.

Across configuration diagnosis there were 26 paid attempts: 20 valid and six deliberately retained truncation failures. Selection agreement was 9/10. On this coverage-selected, non-representative smoke cohort, q-only was 8/10; ER2 Direct was 7/10 (Recovered/Harmful/Net = 0/1/-1), while Flash Direct was 8/10 (0/0/0), identically under Legacy and Corrected labels. These are diagnostics, not validation estimates and not a basis for primary selection.

## Not yet scientifically valid to run

The following phases are deliberately not represented as completed:

- 100-sample pilot and 20×3 repeatability study;
- 500–1,000 development input ablation;
- calibration threshold search;
- full validation and primary selection;
- Standard/Batch parity (Batch Interactions is not available; any Batch experiment requires a separately specified `generateContent` request);
- immutable experiment lock;
- one-time formal test;
- McNemar/Holm/10,000-draw clustered intervals on Gemini outcomes;
- outcome-driven gallery.

No valid frozen Gemini manifest was written. `lock_status.json` explains why. This prevents formal-test execution or test-based primary selection before the missing prerequisites exist.

## Resume gate

Before the 100-sample pilot, rerun a small fixed-config check with the now-selected 4096 ceiling, then record the actual environment budget without persisting its value. A numeric `GEMINI_MAX_SPEND_USD` remains mandatory for pilot, calibration, validation, or formal work. No primary or threshold may be chosen from this smoke cohort.

<!-- GEMINI-OFFLINE-FINALIZER:START -->
## Generated offline finalization status

Generated: 2026-08-01T14:38:21Z

### Summary

The persisted run is currently `blocked_credentials` with publication scope `partial`. No incomplete phase is presented as a formal result.

### Execution status

- pilot: pending
- stability: pending
- ablation: pending
- calibration: pending
- validation: pending
- formal_test: pending
- lock: pending

### Baseline integrity

- Samples: 17749
- Candidate rows: 88745
- Exactly five candidates per sample: 17749
- Legacy q-only successes: 14768
- Legacy Oracle@5 successes: 16129
- Corrected q-only successes: 15840
- Corrected Oracle@5 successes: 16746
- Candidate identity: 34e773b06119619bd45771f8e000c317ecc768d51a132ef6dc7b40e5f3faa36c
- Old V2 tree: 2c0791359d252af73987d180ae1852ac86d3200ebfe2ac93d74a18ad4581481c

### Method results

| method | legacy_j1 | legacy_delta_pp | legacy_recovered | legacy_harmful | legacy_net | corrected_j1 | corrected_delta_pp | corrected_recovered | corrected_harmful | corrected_net | switch_coverage | outcome_precision |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| crog_q_only | 0.8320468758803313 | 0.0 | 0 | 0 | 0 | 0.8924446447687193 | 0.0 | 0 | 0 | 0 | 0.0 |  |
| gemini_robotics_er2_crog_evidence_direct |  |  |  |  |  |  |  |  |  |  |  |  |
| gemini_robotics_er2_crog_evidence_safe |  |  |  |  |  |  |  |  |  |  |  |  |
| gemini_3_6_flash_crog_evidence_direct |  |  |  |  |  |  |  |  |  |  |  |  |
| gemini_3_6_flash_crog_evidence_safe |  |  |  |  |  |  |  |  |  |  |  |  |
| gemini_dual_consensus_safe |  |  |  |  |  |  |  |  |  |  |  |  |
| locked_gemini_primary |  |  |  |  |  |  |  |  |  |  |  |  |

### Independent recomputation

Formal publication gate passed: False.

### Cost and runtime

Published ER2 token prices were available from the official Gemini pricing table; the reported ER2 amount is a usage-based estimate rather than an independently verified provider invoice, and the experiment retained a conservative configurable per-request budget reserve.

Runtime and token totals are derived only from persisted cache records. They do not imply provider billing finality.

### Verified assumptions

- Secret scan: passed
- Duplicate successful request hashes: 0
- Candidate identity audit: passed
- Manifest/request arithmetic: passed
- Formal test used for selection: no; formal publication requires the frozen validation lock.

### Remaining risks

- The test split is not pristine.
- ER2 is a preview model and the provider may update or withdraw it.
- API outputs are not completely deterministic.
- CROG predicted M/Q/angle/W evidence may be wrong.
- RGB alone cannot verify real contact.
- J@1 is not physical grasp success.
- API cost, quota, and provider data-handling remain operational risks.
<!-- GEMINI-OFFLINE-FINALIZER:END -->
