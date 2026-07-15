# Configuration deviations

| Parameter | Paper | Repository | macOS run | Reason / expected effect |
|---|---|---|---|---|
| Device | RTX 5000 | CUDA-or-CPU | MPS FP32 | Platform-only; floating-point order may differ |
| AMP | not fully specified | CUDA AMP enabled | disabled | CUDA AMP is not valid on MPS; avoids unverified FP16 numerics |
| Workers | not specified | DataLoader default | 0 | avoids macOS multiprocessing deadlocks; throughput may decrease |
| Pin memory | not specified | default false | false | pinned CUDA transfer is not useful for MPS |
| Seed | not specified | absent | 42 | makes Python/NumPy/PyTorch sampling repeatable where supported |
| Physical batch | not specified | 16 | 16 | benchmarked for three forward/backward iterations with finite loss; maximum observed driver allocation 6.14 GB |
| Effective batch | not specified | 16 | 16 | preserve repository batch semantics as closely as practical |
| Optimizer | Adam | AdamW | AdamW | preserves released repository, not paper claim |
| FiLM | before every decoder block | `extended_film: False` | false | preserves released repository; prevents an undocumented architecture change |
| Split | stated 70/30 | `unique` train/test (70/10/20 source, val omitted) | same repository split | preserves released code; makes paper comparison approximate |
