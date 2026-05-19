# Full VLM Run Status

## Dataset Scale

OCID-VLG referring-expression samples:

| Refer split | Split | Samples |
|---|---:|---:|
| multiple | train | 63,221 |
| multiple | val | 8,669 |
| multiple | test | 17,749 |
| novel-classes | train | 26,247 |
| novel-classes | val | 2,916 |
| novel-classes | test | 8,585 |
| novel-instances | train | 22,423 |
| novel-instances | val | 2,491 |
| novel-instances | test | 12,834 |
| unique | train | 26,295 |
| unique | val | 3,778 |
| unique | test | 7,675 |

Total: **202,883 language-conditioned samples**.

## Full Run Command

The full VLM command is now supported with resume:

```bash
python scripts/run_dataset.py \
  --dataset ocid_vlg \
  --dataset-root data/OCID-VLG \
  --refer-split all \
  --split all \
  --target-source vlm \
  --vlm-backend florence2 \
  --scorer rule_based \
  --output-root outputs/vlm_all \
  --top-k 5 \
  --resume
```

## Verified VLM Batch

To verify the all-split entry point without starting a multi-day CPU job, this command was run:

```bash
python scripts/run_dataset.py \
  --dataset ocid_vlg \
  --dataset-root data/OCID-VLG \
  --refer-split all \
  --split all \
  --target-source vlm \
  --vlm-backend florence2 \
  --scorer rule_based \
  --output-root outputs/vlm_all_smoke \
  --top-k 5 \
  --max-samples 3 \
  --overwrite
```

Result:

```text
processed=3 failures=0
```

Proxy evaluation was also generated:

```bash
python scripts/evaluate_outputs.py --output-root outputs/vlm_all_smoke --mode proxy
```

Generated reports:

- `outputs/vlm_all_smoke/metrics_by_dataset.csv`
- `outputs/vlm_all_smoke/metrics_by_split.csv`
- `outputs/vlm_all_smoke/metrics_by_scene.csv`
- `outputs/vlm_all_smoke/metrics_by_target_source.csv`
- `outputs/vlm_all_smoke/metrics_by_scorer.csv`
- `outputs/vlm_all_smoke/runtime_report.csv`
- `outputs/vlm_all_smoke/failure_cases.csv`

## Runtime Estimate

The 3-sample VLM batch ran at about **3.67 seconds/sample** after model load. At that rate, the full 202,883-sample OCID-VLG VLM run would take roughly **207 hours** on this CPU setup. Use `--resume` for any full run.

## Example Output

Example generated from `outputs/vlm_all_smoke`:

```json
{
  "command": "Pick the left marker",
  "target_label": "Pick the left marker",
  "target_source": "vlm",
  "best_grasp": {
    "position": [-0.020978419542398194, 0.31269808610234984, 1.0459148299137728],
    "orientation_quaternion": [0.19803620479287998, 0.6788090022909482, -0.19803620479287998, 0.6788090022909482],
    "approach_vector": [0.0, 0.0, -1.0],
    "closing_direction": [0.518278981034755, 0.8126512952567967, -0.26642967202439544],
    "gripper_width": 0.1,
    "grasp_type": "top_down",
    "final_score": 0.7291910123336802
  },
  "top_k": [
    {"rank": 1, "score": 0.7291910123336802},
    {"rank": 2, "score": 0.6669640159521941},
    {"rank": 3, "score": 0.5473872357610997}
  ]
}
```

## Warning

Florence-2 VLM mode is executable, but grounding quality still needs evaluation. On the earlier single-sample smoke test, Florence-2 predicted a bbox different from the OCID-VLG GT bbox. Do not treat full VLM results as final quality metrics until grounding IoU is measured.
