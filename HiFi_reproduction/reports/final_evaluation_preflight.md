# Final HiFi-CS evaluation preflight

- Historical preflight timestamp: `2026-07-11 17:43:49 BST`
- Formal command: `./scripts/evaluate_ocidvlg_mps.sh runs/hifics_ocidvlg_20260711_112921 runs/hifics_ocidvlg_20260711_112921/checkpoints/final.pth`
- Run ID: `hifics_ocidvlg_20260711_112921`
- Checkpoint: `runs/hifics_ocidvlg_20260711_112921/checkpoints/final.pth`
- Checkpoint SHA-256: `436a54ecc159a36664f55f762463c54fc9b082f44205cee8020bed59fb5280d0`
- Checkpoint load: passed with `torch.load(..., map_location="cpu", weights_only=False)`
- Checkpoint optimizer update: `20000`
- Git commit recorded by the run: `4be6b3be7ce79fae481fb51616adfa2b803f07a0`
- Current HiFi-CS repository commit: `4be6b3be7ce79fae481fb51616adfa2b803f07a0`
- Run config: `runs/hifics_ocidvlg_20260711_112921/config.json`
- Evaluation config: `hifics/experiments/config_macos_ocidvlg.yaml`
- Checkpoint/run/current critical config comparison: passed; no mismatch in model, backbone version, FiLM mode, extract layers, reduction dimension, image size, mask convention, batch size, update horizon, device, or AMP setting
- Test manifest: `runs/hifics_ocidvlg_20260711_112921/ocid_vlg_test.json`
- Evaluation manifest resolved by the script: `hifics/datasets/ocidvlg_final_dataset/test/ocid_vlg_test.json`
- Test sample count: `7675`
- Device: Apple MPS (`torch.backends.mps.is_available() == True`)
- Precision: FP32 (`amp: false`); no autocast is used by the evaluation loop
- CPU fallback: disabled; `PYTORCH_ENABLE_MPS_FALLBACK` is unset
- Model mode: `model.eval()`
- Gradient mode: `torch.inference_mode()`
- Test loader: `shuffle=False`, `drop_last=False`, deterministic manifest order
- Configured sigmoid threshold: `0.85`
- Canonical foreground threshold with `invert_mask: true`: `0.15`
- Image size: `352 x 352`
- IoU implementation: threshold sigmoid logits into a canonical foreground boolean mask, compute per-sample intersection divided by union, and define two empty masks as IoU 1.0
- P@X implementation: percentage of samples with per-sample IoU strictly greater than X (`IoU > X`), matching the checked-in repository implementation; this convention will not be changed after evaluation
- Evaluator source SHA-256: `b8bb3f51f0959b4f5ea31f3f938c980a9eaf6698d071f8b477787481c948ac5d`
- Metric-test source SHA-256: `176bbd9b9b032f8d1c31f472d55e1306de52e94e9dfc4d662a4f2e761cea1bcd`
- Active training/evaluation/export processes at preflight: none
- Active screen sessions at preflight: none
- Blockers: none. The reporting-only evaluator extension was completed before the formal run. It persists minimum/maximum IoU, median batch-normalized per-sample inference time, complete evaluation-loop wall-clock runtime, and machine-readable threshold conventions without changing predictions, thresholds, masks, sample order, IoU, or P@X semantics.

No provenance check failed. The reporting extension passed 15 focused metric tests. A two-sample MPS smoke was run interactively but its post-extension artifact was not retained, so the durable evidence for proceeding was the focused suite plus the provenance checks above. This is a historical preflight; the formal evaluation subsequently completed.
