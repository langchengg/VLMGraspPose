# macOS/MPS code changes

All changes are confined to the cloned HiFi-CS checkout or its surrounding `HiFi_reproduction` orchestration directory. The model architecture in `models/hifics.py` is unchanged.

| File / function | Released behavior | macOS behavior | Reason | Model semantics | Reproducibility impact |
|---|---|---|---|---|---|
| `general_utils.py::resolve_device` | no shared resolver; CUDA defaults | explicit `auto/cuda/mps/cpu`, CUDA then MPS then CPU | prevent silent CPU on Apple Silicon | unchanged | device is now recorded; explicit accelerator requests fail instead of falling back |
| `general_utils.py::seed_everything` | no seeding | seeds Python, NumPy, PyTorch and MPS | repeatable sampling | unchanged | MPS is not claimed bitwise deterministic |
| checkpoint helpers | weight-only saves | atomic model/optimizer/scheduler/RNG/metadata checkpoints plus validated legacy loading | exact optimizer-update resume | unchanged | materially improves resumability; legacy files load only when every trainable decoder tensor is present |
| `datasets/dataloader.py::get_data_loader` | always `shuffle=True`; implicit worker flags | split-aware shuffle and configurable macOS-safe flags | preserve evaluation order/all samples | unchanged | evaluation result order is stable; aggregate metrics should be unchanged except where dropped/shuffled assumptions mattered |
| `training.py` | CUDA-or-CPU, CUDA AMP, no resume/accumulation/metadata | shared device selection, FP32 MPS, correct gradient accumulation, configuration/manifest-validated resume with RNG and exact shuffled-loader position, full-state checkpoints, source snapshot | Apple Silicon execution and durable long runs | architecture/loss/optimizer/scheduler unchanged | backend floating-point order differs; physical/effective batch and dirty source state are recorded |
| `score.py` | unused sigmoid then raw-logit threshold; shuffled test | probability threshold, canonical foreground masks, ordered complete test, IoU distribution, P@50-90, timing and hashes | correct documented metric protocol | training unchanged; evaluation semantics corrected | local metrics are not directly interchangeable with numbers produced by the released bug |
| `datasets/prepare_ocidvlg.py` | hard-coded paths, destructive rewrite, broken output-relative JSON paths, no validation | path CLI, skip-existing, explicit overwrite, validation-only, raw/processed validation, correct repo-relative paths | safe reuse and valid manifests | same 352 x 352 RGB/depth/binary-mask content | output paths now load correctly; source manifests remain official `unique` |
| Mac YAML files | one Linux/CUDA-oriented YAML | separate smoke/full/aligned-eval configurations | preserve upstream file and record deviations | released repository model settings preserved | AMP/workers/device/seed differ and are documented |
| `tools/mac_smoke_test.py` | absent | full one-batch MPS gate | block full training on incompatibility | none | records real device/fallback/counts |
| `tools/benchmark_mps_batch.py` | absent | physical batches 1,2,4,8,16, three updates each | evidence-based batch selection | none | selected batch 16 matches repository batch |
| `tools/export_ocidvlg_predictions.py` | absent | deterministic best/typical/failure panels | qualitative audit | none | uses recorded checkpoint/manifest/threshold |
| `HiFi_reproduction/scripts/*.sh` | absent | reproducible bootstrap/prepare/validate/smoke/benchmark/train/evaluate/export/status/resume commands | reliable local operation | none | every script activates the isolated venv and fails non-zero on errors |

## CPU fallback

`PYTORCH_ENABLE_MPS_FALLBACK` remains disabled. No unsupported MPS operator was observed in the model forward/backward, optimizer step, or checkpoint smoke path. No whole-model CPU move was introduced.
