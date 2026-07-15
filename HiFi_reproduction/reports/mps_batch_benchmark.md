# MPS batch benchmark

All tests used the released ViT-B/16 HiFi-CS model at 352 x 352, float32 MPS, one optimizer step per iteration, and three iterations per physical batch. CPU fallback was disabled.

| Batch | Result | Mean iteration | Max observed current allocation | Max observed driver allocation | Mean loss |
|---:|---|---:|---:|---:|---:|
| 1 | pass | 0.151 s | 0.680 GB | 1.669 GB | 0.7023 |
| 2 | pass | 0.118 s | 0.711 GB | 1.686 GB | 0.6962 |
| 4 | pass | 0.267 s | 0.772 GB | 2.785 GB | 0.6944 |
| 8 | pass | 0.321 s | 0.780 GB | 3.954 GB | 0.6940 |
| 16 | pass | 0.583 s | 0.843 GB | 6.144 GB | 0.6940 |

Selected physical batch: **16**. Selected accumulation steps: **1**. Effective batch: **16**. Batch 16 was the largest required candidate and its maximum synchronized observation remained below PyTorch's approximately 17.76 GiB recommended MPS working set measured during the environment audit. PyTorch MPS exposes current/driver allocation but no CUDA-like peak-memory reset/query, so these are observations after synchronized steps, not proven intra-step peaks. Timing is not monotonic because MPS compilation/caching and unified-memory allocation affect such short runs; stability across the required iterations and available headroom determine the selection.
