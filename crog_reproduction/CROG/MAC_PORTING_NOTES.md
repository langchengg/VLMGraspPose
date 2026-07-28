# CROG Mac/MPS porting notes

Audit baseline: official `HilbertXu/CROG` commit `1eeee85de1fe6bffdc66c9ed9a622028ea04578e`.

Audit command:

```bash
rg -n --hidden -g '!.git/**' -g '!.venv/**' \
  -e '\.cuda\(' -e 'torch\.cuda' -e 'DistributedDataParallel' \
  -e 'torch\.distributed' -e '\bdist\.' -e 'DistributedSampler' \
  -e 'SyncBatchNorm' -e 'amp\.autocast' -e 'GradScaler' \
  -e 'pin_memory\s*=\s*True' -e 'map_location.*cuda' \
  -e 'device_ids' -e 'set_device' -e 'nccl' .
```

## Occurrence classification

| File and lines | Occurrence | Decision | Reason |
|---|---|---|---|
| `train_crog.py:17-18` | CUDA AMP and distributed imports | Left unchanged | Official CUDA/DDP entrypoint is preserved byte-for-byte. |
| `train_crog.py:67,87` | CUDA device count and `set_device` | Left unchanged | Required by the official multi-GPU launcher; Mac uses `train_crog_mac.py`. |
| `train_crog.py:96,109,245,274` | Process-group initialization, barrier, and rank checks | Left unchanged | Official DDP coordination remains available. |
| `train_crog.py:114` | `SyncBatchNorm` | Left unchanged | Official multi-GPU behavior; Mac configs set `sync_bn: false`. |
| `train_crog.py:125` | CUDA `GradScaler` | Left unchanged | Mac entrypoint creates it only for a CUDA device. |
| `train_crog.py:154-155` | `.cuda()`, `DistributedDataParallel`, and `device_ids` | Left unchanged | Official CUDA wrapping is intentionally retained. |
| `train_crog.py:182,184` | `DistributedSampler` | Left unchanged | Mac entrypoint uses ordinary shuffled/unshuffled DataLoaders. |
| `train_crog.py:189,198` | `pin_memory=True` | Left unchanged | Mac entrypoint enables pinned memory only for CUDA. |
| `train_crog.py:209` | CUDA checkpoint `map_location` | Left unchanged | Mac checkpoint helper always loads through CPU and normalizes DDP prefixes. |
| `train_crog.py:222,271` | CUDA cache clearing | Left unchanged | Mac entrypoint uses `utils.device.empty_cache`. |
| `test_crog.py:65,70` | Pinned memory and `DataParallel(...).cuda()` | Left unchanged | Official evaluation path is retained; `test_crog_mac.py` is device-neutral. |
| `test_diff_refer_types.py:60,93` | DataParallel CUDA and pinned memory | Left unchanged | Pre-existing experimental script; it also imports nonexistent `engine.engine`. |
| `engine/crog_engine.py:9-10` | CUDA AMP and distributed imports | Wrapped | CUDA autocast is selected only when a CUDA scaler is supplied; distributed helpers first check initialization. |
| `engine/crog_engine.py:27-29` | `dist.all_reduce/get_world_size` | Wrapped | DDP keeps mean reductions; single-device runs return local tensors. |
| `engine/crog_engine.py:118` | `amp.autocast` | Wrapped | Used only when `scaler is not None`; MPS/CPU use `nullcontext` and FP32. |
| `engine/crog_engine.py:165` | Commented rank check | Left as comment | No runtime effect. |
| `engine/crog_engine.py` former train/validation/inference `.cuda()` calls | Changed | All CROG tensors now pass through `move_to_device` using the selected/model device. |
| `utils/misc.py:12,31` | Distributed import and seed broadcast | Wrapped by existing `world_size` behavior | Single-device seed initialization returns before broadcast. |
| `utils/misc.py:40-41` | CUDA seeding | Wrapped | Runs only when CUDA is available. |
| `utils/misc.py:51,53,58,60` | Distributed gather | Wrapped | Returns the input unchanged unless a process group is initialized. |
| `train_crog_mac.py:10,45,95` | CUDA AMP import, availability check, scaler | CUDA-only condition | Allows the new entrypoint to run on one CUDA GPU while disabling AMP on MPS/CPU. |
| `utils/device.py:11,38` | CUDA availability/cache calls | Backend dispatch | Used only for device selection or when `device.type == "cuda"`. |
| `scripts/check_mps.py:28` | CUDA availability report | Left unchanged | Read-only diagnostics. |
| `train_ssg.py:17-18,67,87,96,109,114,125,128-129,147,149,154,163,174,187,204,215,220,235,238` | CUDA AMP/DDP/SyncBN/samplers/cache/rank logic | Left unchanged | SSG is a separate method and outside the CROG Mac port. |
| `engine/ssg_engine.py:7-8,50-59,73-78,97,132-141,195-196` | CUDA transfers and distributed reductions | Left unchanged | SSG is outside scope. |
| `tools/latency.py:41,44,51-52,57,63,66` | CUDA-only latency and memory benchmark | Left unchanged | Tool measures CUDA-specific behavior and is not called by the Mac path. |
| `config/OCID-VLG/*.yaml` original files, lines `51` or `53` | `dist_backend: nccl` | Left unchanged | Original CUDA/DDP configurations remain intact. |
| `config/OCID-Grasp/ssg_r50.yaml:68` | `dist_backend: nccl` | Left unchanged | SSG/CUDA configuration is out of scope. |
| `environment.yml:59` | NVIDIA NCCL package | Left unchanged | Official Linux/CUDA environment remains reproducible; Mac uses `requirements_mac.txt`. |
| `README.md:31` | DDP-only support statement | Left unchanged | It accurately describes the official entrypoint; `README_MAC.md` documents the added path. |
| `utils/box_utils.py:185`, `utils/grasp_eval.py:156` | Commented `.cuda()` examples | Left as comments | No runtime effect. |
| `tests/test_mac_device.py`, `tests/test_mac_source_compatibility.py` | CUDA strings in tests | Left unchanged | Mocks and static assertions verify dispatch/absence of direct engine transfers. |

No direct `.cuda()` call remains in `engine/crog_engine.py`.

## Additional compatibility changes

- `utils/device.py` provides MPS-first auto selection, recursive tensor movement, and backend-specific cache clearing.
- `utils/checkpoint.py` loads official `module.*` checkpoints into an unwrapped model and saves canonical unprefixed state dictionaries. Checkpoints are written through a same-directory `.tmp` file and `os.replace()` so an interrupted write does not corrupt the previous target. `best_iou_model.pth` and `best_jindex_model.pth` copies also use the same atomic replacement pattern.
- Gradient accumulation divides by the actual accumulation-window size, including the final incomplete window, and supports `scaler=None`.
- For singleton micro-batches, only `BatchNorm1d` modules use running statistics; their affine weights remain trainable. CROG has one such layer in the text projection.
- `utils/grasp_eval.py` replaces removed `np.float`/`np.int0` aliases with `float`/`np.intp`. Metric thresholds and geometry are unchanged.
- The first MPS debug validation failed at `torch.from_numpy(iou_list).to(image.device)` because NumPy produced `float64`, which MPS does not support. All three CROG metric-conversion sites now cast to `float32` through `_as_metric_tensor`; the eight-sample training and validation rerun then completed without CPU fallback.
- PyTorch emits a forward-compatibility warning because the official `torch.meshgrid` call omits `indexing`. The operation currently runs on MPS and is left unchanged to avoid an unnecessary model-code edit.
- Modern PyArrow no longer supports the repository's legacy `pa.serialize`/`pa.deserialize` LMDB helpers. The OCID-VLG path used by CROG does not call them; legacy LMDB conversion remains unsupported on the Mac environment.
- No global MPS CPU fallback is enabled. Any future unsupported operation must be isolated, reproduced, and documented before adding a local fallback.
- `train_crog_mac.py` records per-phase timing, finite-loss bounds, checkpoint size, and sampled accelerator allocator statistics in `timing_epoch_NNN.json`. This is reporting-only and does not change the model, loss, data, optimizer, or metrics.
- `train_crog_mac.py` supports optional rolling mid-epoch recovery checkpoints with `checkpoint_interval`. The full Mac config sets `checkpoint_interval: 1000`, producing `mid_epoch_model.pth` only after an optimizer step. One-epoch timing configs set `checkpoint_interval: 0` so historical timing commands remain comparable. Mid-epoch recovery restores model/optimizer/scheduler/scaler state but restarts the interrupted epoch from the beginning rather than resuming the exact shuffled mini-batch position.

## Full-dataset one-epoch timing run (2026-06-19)

Command:

```bash
python train_crog_mac.py --config config/OCID-VLG/CROG_mac_mps_full_1epoch.yaml
```

- Config: input 320, batch 1, validation batch 1, accumulation 8, workers 0, all 63,221 training and 8,669 validation samples, one epoch, MPS FP32.
- Training: 9,859.457 seconds for 63,221 iterations (0.155952 seconds/iteration).
- Validation: 765.126 seconds. Checkpoint write/copies: 3.864 seconds. Total epoch: 10,628.448 seconds (2:57:08.448).
- Loss stayed finite for every checked mini-batch (observed range 0.007693 to 21.968525).
- After allocator warm-up, sampled MPS allocated memory stayed between 2,082,791,936 and 2,082,894,592 bytes; driver memory stayed between 3,974,561,792 and 3,985,047,552 bytes. This is stable rather than monotonic growth.
- `last_model.pth` is 1,766,242,325 bytes (1.645 GiB).
- Measured validation metrics were RIS IoU 25.28%, Precision@50/60/70/80/90 15.80/8.89/5.17/2.38/0.32%, RGS J@1 7.24%, and RGS J@Any (logged as J@5) 12.99%. These are one-epoch timing-run observations, not reproduction claims.
- Linear 50-epoch projection including validation/checkpointing each epoch: 531,422 seconds, 147.62 hours, or 6.15 continuous days. No 50-epoch process was started.
- An earlier attached-terminal attempt reached iteration 25,000 and was terminated when its tool session was interrupted. It produced no checkpoint and is excluded from all timing above. The completed rerun used the same config in a detached `screen` session.

## Batch-size 2 CPU-resource timing run (2026-06-25)

Command:

```bash
caffeinate -dimsu /usr/bin/time -l .venv/bin/python train_crog_mac.py --config config/OCID-VLG/CROG_mac_mps_full_1epoch_bs2_cpu_resource.yaml
```

- Config: based on `CROG_mac_mps_full_1epoch.yaml`; the training hyperparameter change is `batch_size: 2`. The `exp_name` suffix isolates outputs so the previous batch-size 1 artifacts are not overwritten. Validation batch remains 1, accumulation remains 8, workers remain 0, input remains 320, and no sample subset is used.
- Effective training batch is 16 because `batch_size: 2` with `accumulation_steps: 8` differs from the batch-size 1 run's effective batch of 8; one-epoch metrics are therefore not a quality comparison.
- Training: 7,017.675 seconds for 31,611 iterations (0.222001 seconds/iteration).
- Validation: 758.300 seconds. Checkpoint write/copies: 3.962 seconds. Script-measured total epoch: 7,779.956 seconds (2:09:39.956). `/usr/bin/time -l` measured 7,786.98 real seconds including process setup/teardown.
- CPU from `/usr/bin/time -l`: 3,548.49 user seconds, 862.51 system seconds, average CPU utilization 56.65% of one core over wall time.
- CPU memory from `/usr/bin/time -l`: maximum resident set size 2,385,084,416 bytes (2.221 GiB), peak memory footprint 7,743,754,272 bytes (7.212 GiB). A 60-second `ps` sampler collected 128 samples from 2026-06-25T14:57:22+01:00 to 2026-06-25T17:04:24+01:00; sampled CPU averaged 55.31%, peaked at 81.50%, and sampled RSS peaked at 1,391,376 KiB. The `/usr/bin/time -l` max RSS is the authoritative peak.
- Loss stayed finite for every checked mini-batch (observed range 0.017109 to 20.858719).
- MPS peak allocated memory was 2,185,443,840 bytes and peak driver memory was 5,325,979,648 bytes. After warm-up, allocated memory stayed essentially flat around 2.185 GB; driver memory stayed around 5.16 GB during training and ended around 5.315 GB after validation/checkpointing.
- `last_model.pth` is 1,766,242,325 bytes (1.645 GiB).
- Measured validation metrics were RIS IoU 1.80%, Precision@50/60/70/80/90 0.69/0.24/0.00/0.00/0.00%, RGS J@1 0.09%, and RGS J@Any (logged as J@5) 0.35%. These are one-epoch timing-run observations, not reproduction claims.
- Linear 50-epoch projection including validation/checkpointing each epoch: 388,998 seconds, 108.05 hours, or 4.50 continuous days. No 50-epoch process was started.
- Exit code: 0.

## Official-parameter one-epoch attempt on Mac/MPS (2026-07-02)

Command:

```bash
caffeinate -dimsu /usr/bin/time -l .venv/bin/python train_crog_mac.py --config config/OCID-VLG/CROG_mac_mps_official_params_1epoch.yaml
```

- Config intent: match the official `config/OCID-VLG/crog_multiple_r50.yaml` training hyperparameters as closely as possible while setting `epochs: 1` and using the Mac single-device entrypoint.
- Official hyperparameters used: `input_size: 416`, `batch_size: 24`, `batch_size_val: 24`, `workers: 2`, `workers_val: 2`, `base_lr: 0.0001`, `lr_decay: 0.1`, `lr_multi: 0.1`, `weight_decay: 0.0`, `milestones: [35]`, `use_pretrained_clip: true`, `use_contrastive: true`, and `use_grasp_masks: true`.
- Mac/runtime differences: `root_path` points to the local dataset at `../OCID-VLG`; `resume` is `null` because `weights/best_jindex_model.pth` is not present locally and a resumed checkpoint with epoch greater than 1 would not perform a one-epoch training run; `sync_bn` is `false` because this is not DDP; `accumulation_steps` is `1` so the effective batch remains 24.
- Result: the run started on MPS and completed the first logged training step, but did not reach step 100 after more than 20 minutes. It was manually interrupted to avoid a long, non-practical Mac run. Exit code: 130.
- First logged MPS memory sample: allocated 3,920.4 MB and driver memory 22,502.5 MB at step 1/2,635. This is far above the batch-size 1 and batch-size 2 Mac runs and is the main evidence that the official batch/input setting is not practical on this Mac.
- `/usr/bin/time -l` at interruption: 1,307.13 real seconds, 26.56 user seconds, 346.49 system seconds, average CPU utilization 28.54% of one core, maximum resident set size 1,580,875,776 bytes (1.472 GiB), and peak memory footprint 30,035,185,000 bytes (27.97 GiB).
- A 30-second `ps` sampler collected 43 samples from 2026-07-02T17:00:44+01:00 to 2026-07-02T17:21:45+01:00; sampled CPU averaged 27.97% and peaked at 87.10%. The sampled RSS does not capture MPS/driver memory pressure; use the MPS and `/usr/bin/time -l` values above for resource conclusions.
- No checkpoint was produced because CROG saves at epoch boundaries and the run was interrupted during the first epoch.
- Recommendation from this attempt: do not run the official `batch_size: 24`, `input_size: 416` setting directly on this Apple Silicon/MPS machine. Use smaller Mac-specific batches, or move official-parameter training to an NVIDIA CUDA server.

## Official-parameter one-epoch run with training batch size 8 (2026-07-02)

Command:

```bash
caffeinate -dimsu /usr/bin/time -l .venv/bin/python train_crog_mac.py --config config/OCID-VLG/CROG_mac_mps_official_params_1epoch_bs8.yaml
```

- Config intent: use the official `config/OCID-VLG/crog_multiple_r50.yaml` settings, but set `epochs: 1` and `batch_size: 8` as requested.
- Official hyperparameters kept: `input_size: 416`, `batch_size_val: 24`, `workers: 2`, `workers_val: 2`, `base_lr: 0.0001`, `lr_decay: 0.1`, `lr_multi: 0.1`, `weight_decay: 0.0`, `milestones: [35]`, `manual_seed: 0`, `use_pretrained_clip: true`, `use_contrastive: true`, and `use_grasp_masks: true`.
- Mac/runtime differences: `root_path` points to the local dataset at `../OCID-VLG`; `resume` is `null` because `weights/best_jindex_model.pth` is not present locally and a resumed checkpoint with epoch greater than 1 would not perform a one-epoch training run; `sync_bn` is `false` because this is not DDP; `accumulation_steps` is `1`, so the effective training batch is exactly 8.
- Training completed successfully on MPS: 7,880.780 seconds for 7,903 iterations, or 0.997188 seconds/iteration. The iteration count matches full-dataset training: 63,221 samples divided by batch size 8 rounds to 7,903 batches.
- Validation completed successfully with the official validation batch size 24: 786.806 seconds for 362 validation batches. Checkpoint write/copies took 5.332 seconds. Script-measured total epoch time was 8,672.929 seconds (2:24:32.929); `/usr/bin/time -l` measured 8,681.50 real seconds including process setup/teardown.
- CPU from `/usr/bin/time -l`: 2,727.40 user seconds, 1,289.97 system seconds, average CPU utilization 46.27% of one core over wall time. A 30-second `ps` sampler collected 289 samples from 2026-07-02T17:27:23+01:00 to 2026-07-02T19:51:29+01:00; sampled CPU averaged 20.38% and peaked at 89.40%. The sampled RSS does not capture MPS/driver memory pressure.
- CPU memory from `/usr/bin/time -l`: maximum resident set size 1,640,546,304 bytes (1.528 GiB), and peak memory footprint 18,242,735,896 bytes (16.99 GiB).
- Loss stayed finite for every checked mini-batch (observed range 0.012186 to 20.452501).
- MPS peak allocated memory was 2,886,900,992 bytes (2.688 GiB). MPS peak driver memory was 16,350,920,704 bytes (15.23 GiB), and driver memory ended at 16,118,136,832 bytes (15.01 GiB). Training allocation was stable after warm-up, but validation/checkpointing raised driver memory substantially versus the batch-size 1 and batch-size 2 Mac configs.
- `last_model.pth`, `best_iou_model.pth`, and `best_jindex_model.pth` were produced. Each checkpoint is 1,766,242,325 bytes (1.645 GiB). Exit code: 0.
- Measured validation metrics were RIS IoU 50.60%, Precision@50/60/70/80/90 48.83/38.06/27.51/16.60/1.15%, RGS J@1 29.43%, and RGS J@Any (logged as J@5) 37.76%. These are one-epoch timing-run observations, not reproduction claims.
- Linear 50-epoch projection including validation/checkpointing each epoch: 433,646 seconds, 120.46 hours, or 5.02 continuous days. No 50-epoch process was started.
- Practical recommendation from this run: `batch_size: 8`, `input_size: 416` is trainable on this Apple Silicon/MPS machine for one full epoch, but it leaves much less MPS driver-memory headroom than the Mac-specific 320px configs. For exact official 50-epoch training or controlled reproduction, an NVIDIA CUDA server remains the safer option.

## Official-parameter 50-epoch run with training batch size 8 (started 2026-07-02)

Command:

```bash
caffeinate -dimsu /usr/bin/time -l .venv/bin/python train_crog_mac.py --config config/OCID-VLG/CROG_mac_mps_official_params_50epoch_bs8.yaml
```

- Config intent: use the official `config/OCID-VLG/crog_multiple_r50.yaml` settings for a full 50-epoch Mac/MPS run, but set `batch_size: 8` as requested.
- Official hyperparameters kept: `input_size: 416`, `epochs: 50`, `batch_size_val: 24`, `workers: 2`, `workers_val: 2`, `base_lr: 0.0001`, `lr_decay: 0.1`, `lr_multi: 0.1`, `weight_decay: 0.0`, `milestones: [35]`, `manual_seed: 0`, `use_pretrained_clip: true`, `use_contrastive: true`, and `use_grasp_masks: true`.
- Mac/runtime differences: `root_path` points to the local dataset at `../OCID-VLG`; `resume` is `null` because this is a fresh 50-epoch run from the CLIP RN50 initialization; `sync_bn` is `false` because this is not DDP; `accumulation_steps` is `1`, so the effective training batch is exactly 8.
- Safety settings: `checkpoint_interval: 1000` writes a rolling `mid_epoch_model.pth` after optimizer steps, and all checkpoint writes use same-directory temp files plus atomic replacement.
- Launch method: detached `screen` session `crog_bs8_50epoch` with `caffeinate`; `/usr/bin/time -l` output goes to `exp/OCID-VLG_multiple_mac/CROG_mac_mps_official_params_50epoch_bs8/official_params_50epoch_bs8_time_l_console.log`.
- Resource sampler: detached `screen` session `crog_bs8_50epoch_ps`, sampling Python PID `68193` every 60 seconds to `exp/OCID-VLG_multiple_mac/CROG_mac_mps_official_params_50epoch_bs8/cpu_ps_samples_pid68193.txt`.
- Preflight: no existing CROG training screen was active; CLIP RN50 checkpoint and local OCID-VLG dataset existed; filesystem free space was about 16 GiB, enough for rolling checkpoints but tight. No old artifacts were deleted.
- First health check: the run reached epoch 1 iteration 200/7903 with finite loss and no startup error. First MPS sample was allocated 2,742.2 MB and driver memory 8,734.5 MB at step 1/7903.
- Independent monitoring added after launch: `scripts/monitor_crog_training_diary.py` reads only existing run artifacts (`*console.log`, `timing_epoch_*.json`, `cpu_ps_samples_pid*.txt`, and checkpoint metadata) and writes `TRAINING_DIARY.md` / `TRAINING_DIARY.json` with atomic replacement. It does not import torch, does not open checkpoint contents, and does not signal the training process except optional `os.kill(pid, 0)` liveness checking.

## Mac validation interval change (2026-07-04)

- Added `validation_interval` support to `train_crog_mac.py`. The default is `1`, preserving the prior behavior of validating after every epoch.
- Set `config/OCID-VLG/CROG_mac_mps_official_params_50epoch_bs8.yaml` to `validation_interval: 5`, so the 50-epoch Mac/MPS run validates only on epochs 5, 10, 15, ..., 50. The final epoch is always validated even if it is not divisible by the interval.
- Non-validation epochs still write `last_model.pth` with atomic replacement and preserve the latest known validation metrics in checkpoint metadata. `best_iou_model.pth` and `best_jindex_model.pth` are only updated on epochs that actually run validation.
- This change affects only the Mac single-device entrypoint. The official CUDA/DDP `train_crog.py` path remains unchanged.

## Files added or changed

Added: Mac train/test entrypoints, seven Mac configs (debug, full-training, full-dataset one-epoch timing, batch-size 2 CPU-resource timing, official-parameter one-epoch attempt, official-parameter batch-size 8 one-epoch run, and official-parameter batch-size 8 50-epoch run), device/checkpoint utilities, three inspection/visualization scripts, Mac requirements/README, focused tests, and `REPRODUCTION_LOG.md`.

Changed shared files: `engine/crog_engine.py`, `utils/misc.py`, `utils/grasp_eval.py`, `utils/checkpoint.py`, and `.gitignore`. Timing/finite-loss/MPS-memory reporting was added to the Mac path only; checkpoint writing now uses atomic replacement for Mac checkpoint safety.

Unchanged by verification: `train_crog.py` matches `origin/main` blob `102d03bf538bbd0246279b6d9229d76bd62a3657`.
