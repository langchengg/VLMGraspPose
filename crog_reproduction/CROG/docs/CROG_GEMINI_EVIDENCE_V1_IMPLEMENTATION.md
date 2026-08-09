# CROG Gemini Evidence V1 — Implementation

## Summary

This experiment reranks exactly five frozen CROG 4-DoF candidates. It does not generate, translate, rotate, resize, merge, or refine any candidate. The implemented path is:

`RGB + referring expression → frozen CROG → frozen Top-5 + predicted M/Q/sin(2θ)/cos(2θ)/W → one 2200×2200 evidence board → one Gemini ranking response → Direct/Safe/Consensus Top-1`.

The implementation lives in `failure_analysis/gemini_crog_evidence_v1/`; the pre-existing `failure_analysis/reranking/` and `failure_analysis/reranking_v2/` trees remain unchanged.

## Data actually used for reranking

Request-side data are limited to RGB, the referring expression, CROG-predicted M, Q, sin(2θ), cos(2θ), W, frozen candidate geometry, original q/rank, and deterministic candidate-local evidence. Depth is excluded from the primary protocol.

`program`, `answer`, `target`, `target_idx`, `box`, `grasps`, `objID`, every GT dense map, GT grasp rectangles, evaluator errors, candidate correctness, Oracle, Recovered/Harmful, and every old V2 reranker score/selection are forbidden. Machine-readable contracts are in:

- `failure_analysis/gemini_crog_evidence_v1/inference_input_allowlist.json`
- `failure_analysis/gemini_crog_evidence_v1/forbidden_inference_fields.json`
- `failure_analysis/gemini_crog_evidence_v1/evaluation_only_fields.json`

## Frozen CROG replay

The exporter restores full-resolution predicted maps from the frozen checkpoint and official config. It replays original groups of 16 samples on MPS, matching the prior frozen export batch geometry. CROG is called as `model(image, text)` in evaluation mode; GT tensors are never arguments to the forward call. Candidate coordinates and order are rechecked against the frozen artifact after every forward pass.

M/Q/W are sigmoid outputs before bicubic upsampling. Because cubic interpolation can overshoot slightly, probability semantics are restored by clipping candidate evidence statistics to `[0,1]`; raw logits remain separately available. The first un-clipped 10-sample artifact is explicitly marked invalid and is not used.

Candidate evidence includes Q patches/prominence/entropy, predicted-mask rectangle/axis/jaw/boundary support, 180°-periodic angle concentration and consistency, predicted-width consistency, and Top-5 candidate relations. No unlabelled 512/1024-dimensional latent vectors are sent to Gemini.

## Composite renderer

Each request uses one deterministic 2200×2200 PNG with:

- original RGB, predicted-mask contour, and all five frozen rectangles;
- predicted M and binary contour;
- predicted Q and frozen peak centres;
- axial angle hue with magnitude-dependent brightness and candidate arrows;
- decoded width on a fixed 0–100 px scale;
- five equal-scale RGB candidate cards with compact predicted evidence.

A SHA-256-derived A–E permutation is keyed by sample ID and seed 47. Candidate colours depend only on A–E. The same stored mapping is used for both models. Layer manifests explicitly state that no evaluation overlay is present.

## Structured response and policies

Pydantic rejects extra fields, missing/duplicate A–E IDs, new coordinates, free text, unknown reason codes, non-finite/out-of-range scores, inconsistent selected/ranking IDs, and increasing score order. Technical failures and model abstentions are counted separately and both fall back to q-only.

Direct, Safe, and dual-model consensus are derived from one response per model/sample. Safe switches require validation-locked confidence, margin, and selected-overall thresholds. Calibration maximizes `Recovered − Harmful` subject to Legacy Harmful ≤ 1.0 percentage point; if positive net gain is unavailable, Safe becomes q-only.

## Gemini API contract

The project pins `google-genai==2.16.0` and uses the v1beta Interactions API with exact model IDs `gemini-robotics-er-2-preview` and `gemini-3.6-flash`. Each call is independent, `store=False`, `background=False`, `stream=False`, has no previous interaction and no tools, uses inline PNG data at locked resolution, model-default temperature, `thinking_level=medium`, no thought summaries, and the current text/JSON `response_format` schema. The first live access smoke showed that a 512-token generation ceiling left only four visible tokens after about 490 hidden thought tokens. A 1536-token retry also truncated, and two Flash cases with roughly 2400 thought tokens truncated at 3072. Targeted 4096-token retries completed those responses with 3315 and 3460 combined thought/output tokens. The development default is therefore 4096; every change creates a distinct request hash and all failed responses remain cached for audit.

The SDK reads only `GEMINI_API_KEY`. Budget and concurrency are read only from `GEMINI_MAX_SPEND_USD` and `GEMINI_MAX_CONCURRENCY`. The SQLite cache stores hashes, final structured output and usage, but no headers, key, GT, correctness, or evaluator result. Retry is limited to 429/5xx/timeout/reset, at most five retries; 400/401/403/404 are permanent.

Official sources checked before implementation:

- https://ai.google.dev/api/interactions-api
- https://ai.google.dev/gemini-api/docs/interactions-overview
- https://ai.google.dev/gemini-api/docs/structured-output
- https://ai.google.dev/gemini-api/docs/media-resolution
- https://ai.google.dev/gemini-api/docs/thinking
- https://ai.google.dev/gemini-api/docs/batch-api
- https://ai.google.dev/gemini-api/docs/pricing
- https://pypi.org/project/google-genai/2.16.0/

Batch is not silently substituted: Google currently documents Batch for `generateContent`, not Batch Interactions. Standard Interactions is therefore the implemented transport unless a later, separately tested parity protocol locks a Batch `generateContent` request.

## Reproduction commands

Use `.venv/bin/python -m ...` because the relocated `.venv/bin/pytest` and `.venv/bin/pip` shebangs point to an obsolete path.

```bash
.venv/bin/python -m failure_analysis.gemini_crog_evidence_v1.cli phase0 --output <run>/phase0
.venv/bin/python -m failure_analysis.gemini_crog_evidence_v1.cli select-smoke --output <run>/smoke_selection --count 10
.venv/bin/python -m failure_analysis.gemini_crog_evidence_v1.cli export-evidence --split train --selected-local-ids <ids.json> --frozen-features <train_features.jsonl> --output <run>/smoke_evidence --device mps --keep-dense-maps
.venv/bin/python -m failure_analysis.gemini_crog_evidence_v1.cli run-smoke --request-manifest <requests.jsonl> --evidence-schema <evidence_schema.json> --output <run>/smoke_api
```

The lock command is dry-run only until calibration and full validation have produced a validation-selected primary method.
