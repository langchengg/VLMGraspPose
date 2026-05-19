# VLMGraspPose

Active project:

```text
target_aware_vlm_grasping/
```

This repository was merged into one active, Mac-compatible OCID-VLG / OCID-Grasp pipeline. Older root-level code and the previous `target_aware_graspnet` implementation were archived under:

```text
legacy/
```

Start here:

```bash
cd target_aware_vlm_grasping
python -m pytest tests
python scripts/run_one_sample.py --dataset ocid_vlg --dataset-root ../data/raw/OCID-VLG --index 0 --target-source oracle --scorer rule_based --output-root outputs/debug --top-k 5 --overwrite
```
