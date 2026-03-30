"""
==========================================================================
 VLMGraspPose — Complete Architecture Pipeline for Google Colab
==========================================================================
 Copy-paste ALL of this into ONE Colab cell and run.
 Requires: Colab Pro ($10/mo) + T4 GPU runtime (for VLM experiments)

 Architecture Flow:
   ┌─ Branch A: Target Grounding (GT / VLM / Phrase Grounding) ──┐
   │                                                              ├→ Association → Re-Ranking → Selection
   └─ Branch B: Grasp Proposal (Antipodal Geometric Sampler) ────┘

 Experiment Matrix (automatically runs all combinations):
   ┌─────────────┬───────────┬───────────────┬────────────┐
   │  Grounder   │  Scorer   │  Features     │   Split    │
   ├─────────────┼───────────┼───────────────┼────────────┤
   │  gt         │  rule     │  core (5-dim) │  test_seen │
   │  gt         │  logistic │  core (5-dim) │  test_seen │
   │  gt         │  mlp      │  core (5-dim) │  test_seen │
   │  gt         │  mlp      │  ext  (9-dim) │  test_seen │
   │  vlm        │  rule     │  core (5-dim) │  test_seen │  ← needs GPU
   │  vlm        │  mlp      │  core (5-dim) │  test_seen │  ← needs GPU
   │  phrase     │  rule     │  core (5-dim) │  test_seen │  ← needs GPU
   └─────────────┴───────────┴───────────────┴────────────┘

 Estimated time: ~5-7 hours (including data download)
==========================================================================
"""

# ╔═══════════════════════════════════════════════════════════════════╗
# ║  PHASE 0 — ENVIRONMENT SETUP                                     ║
# ╚═══════════════════════════════════════════════════════════════════╝

import os, json, glob, shutil

# 0a. Mount Google Drive for persistence
from google.colab import drive
drive.mount('/content/drive')

WORK_DIR = '/content/drive/MyDrive/VLMGraspPose'
os.makedirs(WORK_DIR, exist_ok=True)

# 0b. Clone project code (first run only)
REPO_URL = 'https://github.com/langchengg/VLMGraspPose.git'  # ← your repo URL

if not os.path.exists(f'{WORK_DIR}/config.py'):
    print("=" * 60)
    print("First run: cloning project code")
    print("=" * 60)
    os.system(f'git clone {REPO_URL} {WORK_DIR}')
else:
    print("[OK] Project code exists, pulling latest...")
    os.system(f'cd {WORK_DIR} && git pull --ff-only 2>/dev/null || true')

os.chdir(WORK_DIR)
print(f"Working directory: {os.getcwd()}")

# 0c. Install dependencies
print("\n[SETUP] Installing dependencies...")
os.system('pip install -q numpy scipy scikit-learn torch Pillow open3d tqdm '
          'matplotlib opencv-python-headless huggingface_hub transformers '
          'einops timm gdown')

# 0d. Enable train split in config
config_path = f'{WORK_DIR}/config.py'
with open(config_path, 'r') as f:
    config_content = f.read()

if '# "train"' in config_content:
    config_content = config_content.replace(
        '    # "train": PROJECT_ROOT / "train",',
        '    "train": PROJECT_ROOT / "train",'
    )
    with open(config_path, 'w') as f:
        f.write(config_content)
    print("[OK] Enabled train split in config.py")

# 0e. Check GPU availability
import torch
HAS_GPU = torch.cuda.is_available()
if HAS_GPU:
    print(f"[GPU] {torch.cuda.get_device_name(0)} "
          f"({torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB VRAM)")
else:
    print("[WARN] No GPU — VLM experiments (Branch A: vlm/phrase) will be skipped")


# ╔═══════════════════════════════════════════════════════════════════╗
# ║  PHASE 1 — INPUT DATA (Architecture Block 1)                     ║
# ║  Download: Text queries + RGB images + Depth maps + Point Clouds  ║
# ╚═══════════════════════════════════════════════════════════════════╝

print("\n" + "═" * 60)
print("  BLOCK 1 — INPUT: Download Scene Data")
print("═" * 60)

for split, size in [('test_seen', '~7 GB'), ('train', '~30 GB')]:
    split_dir = f'{WORK_DIR}/{split}'
    if os.path.exists(split_dir) and len([d for d in os.listdir(split_dir) if d.startswith('scene_')]) > 0:
        n = len([d for d in os.listdir(split_dir) if d.startswith('scene_')])
        print(f"  ✓ {split}: {n} scenes (already downloaded)")
    else:
        print(f"\n  >>> Downloading {split} ({size})...")
        flag = '--test-seen' if split == 'test_seen' else '--train'
        os.system(f'cd {WORK_DIR} && python scripts/download_data.py {flag}')

# Download Florence-2 weights if GPU available
if HAS_GPU:
    florence_dir = f'{WORK_DIR}/models/florence-2-base'
    if not os.path.exists(florence_dir) or not os.listdir(florence_dir):
        print("\n  >>> Downloading Florence-2 model weights (~450 MB)...")
        os.system(f'cd {WORK_DIR} && python scripts/download_weights.py --florence2')
    else:
        print(f"  ✓ Florence-2 weights ready")

# Preprocess JSONL indices
print("\n  Preprocessing data indices...")
for split in ['test_seen', 'train']:
    jsonl = f'{WORK_DIR}/processed/{split}.jsonl'
    if not os.path.exists(jsonl):
        os.system(f'cd {WORK_DIR} && python -m data.preprocess --split {split}')
    else:
        print(f"  ✓ {split}.jsonl exists")


# ╔═══════════════════════════════════════════════════════════════════╗
# ║  PHASE 2 — TRAINING                                              ║
# ║  Branch B (Grasp Proposal) + Block 4 (Features) on TRAIN split    ║
# ║  Then train Re-Ranking scorers (Block 5)                          ║
# ╚═══════════════════════════════════════════════════════════════════╝

print("\n" + "═" * 60)
print("  PHASE 2 — TRAINING: Generate features + Train scorers")
print("═" * 60)

# 2a. Generate training features (GT grounding + rule scorer on train split)
#     This runs Blocks 1-4 on the train split (~2-3 hours)
train_result = f'{WORK_DIR}/results/pipeline_summary_train_rule.json'
if not os.path.exists(train_result):
    print("\n  >>> Running pipeline on TRAIN split (~2-3 hours)...")
    print("      Block 2: GT Grounding → Block 3: Grasp Proposal → Block 4: Features")
    os.system(f'cd {WORK_DIR} && python -m experiments.run_pipeline '
              f'--split train --grounder gt --scorer rule')
else:
    print("  ✓ Training features already generated")

# 2b. Train Logistic Regression scorer (Block 5)
print("\n  >>> Training Logistic Regression scorer (Block 5)...")
os.system(f'cd {WORK_DIR} && python -m experiments.train_ranker --mode pseudo --scorer logistic')

# 2c. Train MLP scorer (Block 5)
print("\n  >>> Training MLP scorer (Block 5)...")
os.system(f'cd {WORK_DIR} && python -m experiments.train_ranker --mode pseudo --scorer mlp')


# ╔═══════════════════════════════════════════════════════════════════╗
# ║  PHASE 3 — EVALUATION: Full Architecture on TEST_SEEN             ║
# ║  Run all experiment combinations through Blocks 2→3→4→5→6        ║
# ╚═══════════════════════════════════════════════════════════════════╝

print("\n" + "═" * 60)
print("  PHASE 3 — EVALUATION: Full Architecture Experiments")
print("═" * 60)

# Define experiment matrix
experiments = [
    # (grounder,  scorer,     extended, needs_gpu, description)
    ("gt",       "rule",     False,    False,  "GT + Rule (baseline)"),
    ("gt",       "logistic", False,    False,  "GT + Logistic"),
    ("gt",       "mlp",      False,    False,  "GT + MLP"),
    ("gt",       "mlp",      True,     False,  "GT + MLP + Extended Features"),
    ("vlm",      "rule",     False,    True,   "VLM (open-vocab) + Rule"),
    ("vlm",      "mlp",      False,    True,   "VLM (open-vocab) + MLP"),
    ("phrase",   "rule",     False,    True,   "VLM (phrase grounding) + Rule"),
]

completed = []
skipped = []

for grounder, scorer, extended, needs_gpu, desc in experiments:
    print(f"\n  ┌─ Experiment: {desc}")

    if needs_gpu and not HAS_GPU:
        print(f"  └─ [SKIP] Requires GPU")
        skipped.append(desc)
        continue

    # Build command
    cmd = (f'cd {WORK_DIR} && python -m experiments.run_pipeline '
           f'--split test_seen --grounder {grounder} --scorer {scorer}')
    if extended:
        cmd += ' --extended'

    print(f"  │  Branch A: {grounder:<8} Branch B: antipodal")
    print(f"  │  Features: {'ext 9-dim' if extended else 'core 5-dim':<12} Re-Ranking: {scorer}")

    os.system(cmd)

    completed.append(desc)
    print(f"  └─ [DONE] ✓")

print(f"\n  Completed: {len(completed)} experiments")
if skipped:
    print(f"  Skipped (no GPU): {len(skipped)} experiments")


# ╔═══════════════════════════════════════════════════════════════════╗
# ║  PHASE 4 — BLOCK 7: EVALUATION METRICS                           ║
# ║  Compute Hit@K, position error, angular error for all experiments ║
# ╚═══════════════════════════════════════════════════════════════════╝

print("\n" + "═" * 60)
print("═" * 60)
print("  ★  BLOCK 7 — EVALUATION: Metric Comparison  ★")
print("═" * 60)
print("═" * 60)

# 4a. Compare all GT-mode scorers
os.system(f'cd {WORK_DIR} && python -m experiments.eval --compare')

# 4b. Evaluate VLM results separately
for vlm_result in sorted(glob.glob(f'{WORK_DIR}/results/pipeline_summary_test_seen_*_vlm.json')) + \
                  sorted(glob.glob(f'{WORK_DIR}/results/pipeline_summary_test_seen_*_phrase.json')):
    print(f"\n  >>> Evaluating: {os.path.basename(vlm_result)}")
    os.system(f'cd {WORK_DIR} && python -m experiments.eval --results {vlm_result}')

# 4c. GT comparison (position/angular error)
print("\n  >>> Computing Grasp vs GT deviation metrics...")
for scorer in ['rule', 'mlp']:
    os.system(f'cd {WORK_DIR} && python -m vis.compare_gt --scorer {scorer} --max-samples 100')


# ╔═══════════════════════════════════════════════════════════════════╗
# ║  PHASE 5 — VISUALIZATION: Generate Paper Figures                  ║
# ║  2D overlay, 3D point cloud, GT comparison panels                 ║
# ╚═══════════════════════════════════════════════════════════════════╝

print("\n" + "═" * 60)
print("  PHASE 5 — VISUALIZATION: Paper Figures")
print("═" * 60)

# Find the best samples for visualization
results_file = f'{WORK_DIR}/results/pipeline_summary_test_seen_rule.json'
sample_ids_to_vis = []
if os.path.exists(results_file):
    with open(results_file) as f:
        data = json.load(f)
    results_sorted = sorted(
        data.get('results', []),
        key=lambda r: r['selections'][0]['final_score'] if r.get('selections') else 0,
        reverse=True
    )
    sample_ids_to_vis = [r['sample_id'] for r in results_sorted[:6]]
    print(f"  Selected {len(sample_ids_to_vis)} samples for visualization")

for sid in sample_ids_to_vis:
    # 2D: RGB + bbox + gripper overlay
    os.system(f'cd {WORK_DIR} && python -m vis.vis_2d --sample {sid} --scorer rule')
    os.system(f'cd {WORK_DIR} && python -m vis.vis_2d --sample {sid} --scorer mlp')
    # 3D: Point cloud + gripper skeleton
    os.system(f'cd {WORK_DIR} && python -m vis.vis_3d --sample {sid} --scorer rule')
    # GT comparison panel
    os.system(f'cd {WORK_DIR} && python -m vis.compare_gt --sample {sid} --scorer rule --draw')
    os.system(f'cd {WORK_DIR} && python -m vis.compare_gt --sample {sid} --scorer mlp --draw')

# Display in Colab
from IPython.display import display, Image as IPImage

for pattern, label in [('*_2d.png', '2D Overlay'), ('*_3d.png', '3D Point Cloud'),
                        ('*_compare.png', 'GT Comparison')]:
    files = sorted(glob.glob(f'{WORK_DIR}/vis_output/{pattern}'))[:3]
    for vf in files:
        print(f"\n📸 [{label}] {os.path.basename(vf)}")
        display(IPImage(filename=vf, width=900))


# ╔═══════════════════════════════════════════════════════════════════╗
# ║  PHASE 6 — FINAL REPORT                                          ║
# ║  Print all paper-ready results                                    ║
# ╚═══════════════════════════════════════════════════════════════════╝

print("\n")
print("█" * 60)
print("█                                                          █")
print("█     ★  COMPLETE — Paper Results Summary  ★              █")
print("█                                                          █")
print("█" * 60)

# Table 1: Scorer + Grounder comparison
metrics_files = sorted(glob.glob(f'{WORK_DIR}/results/*.metrics.json'))
if metrics_files:
    print("\n📊 Table 1: Architecture Configuration Comparison")
    print("─" * 75)
    print(f"{'Grounder':<10} {'Scorer':<10} {'Hit@1':<10} {'Hit@5':<10} "
          f"{'AvgScore':<12} {'Latency':<10}")
    print("─" * 75)
    for mf in metrics_files:
        with open(mf) as f:
            m = json.load(f)
        print(f"{m.get('grounder','gt'):<10} "
              f"{m.get('scorer','?'):<10} "
              f"{m.get('target_hit_at_1',0):<10.4f} "
              f"{m.get('target_hit_at_5',0):<10.4f} "
              f"{m.get('avg_top1_score',0):<12.4f} "
              f"{m.get('avg_latency',0):<10.3f}s")
    print("─" * 75)

# Table 2: GT deviation
compare_reports = sorted(glob.glob(f'{WORK_DIR}/vis_output/comparison_report_*.json'))
if compare_reports:
    print("\n📊 Table 2: Grasp Pose vs Ground Truth")
    print("─" * 75)
    for cr in compare_reports:
        with open(cr) as f:
            s = json.load(f)
        scorer = s.get('scorer', '?')
        pe = s.get('position_error_cm', {})
        ae = s.get('angular_error_deg', {})
        wr = s.get('width_ratio', {})
        print(f"\n  Scorer: {scorer}")
        print(f"    Position Error:    {pe.get('mean',0):.2f} ± {pe.get('std',0):.2f} cm")
        print(f"    Angular Error:     {ae.get('mean',0):.1f} ± {ae.get('std',0):.1f}°")
        print(f"    Target Hit (mask): {s.get('target_hit_rate_mask',0)*100:.1f}%")
        print(f"    Target Hit (bbox): {s.get('target_hit_rate_bbox',0)*100:.1f}%")
        print(f"    Width Ratio:       {wr.get('mean',0):.2f} ± {wr.get('std',0):.2f}")

# File summary
print(f"\n\n📁 All outputs saved to Google Drive:")
print(f"  Results:     {WORK_DIR}/results/")
print(f"  Figures:     {WORK_DIR}/vis_output/")
print(f"  Models:      {WORK_DIR}/models/")
print(f"  Features:    {WORK_DIR}/features/")

results_count = len(glob.glob(f'{WORK_DIR}/results/*.json'))
vis_count = len(glob.glob(f'{WORK_DIR}/vis_output/*.png'))
print(f"\n  Total: {results_count} result files, {vis_count} figure files")

print("\n" + "█" * 60)
print("  ✅ All done! Results are permanently saved in Google Drive.")
print("  📝 Ready for paper writing.")
print("█" * 60)
